#!/usr/bin/env python3
"""Build a propagation-state-gated local field and replay the complete V15 chain."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

import analyze_e86_surface_endpoint_v11_blend as e86
import analyze_round2_canonical_500_versions as canonical
import analyze_v7_complex_carrier_stacking as stacking
import analyze_v7_stacked_lifecycle_overlay as lifecycle_overlay
import build_e193_independent_v11_base as independent
import build_v7_complex_carrier_bank as carrier_bank
from carrier_calibration import estimate_carrier
from common.official_metrics import point_metric_statistics_torch
from complex_baselines import batched_predict_many_torch, query_site_neighbors
from project_paths import PATHS
from round2.data import load_positions
import run_v6_regime_component_lifecycle as lifecycle
import run_v7_complex_carrier_exact_projection as v7_projection
from run_foundation_benchmark import load_or_compute_power
from run_rbf_spectral_field import FOLD_CACHE, OOF_META, statistics
from run_v9_joint_sionna_migration import Candidate as V9Candidate
from run_v9_joint_sionna_migration import apply_candidate as apply_v9_candidate
from run_v9_joint_sionna_migration import point_metrics
import train_e190_official_sionna_path_calibrator as e190
import train_e193_propagation_memory as e193
import train_e195_directional_probe_calibrator as e195
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution


VERSION = "Round2-E199-propagation-state-gated-local-field"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e199_propagation_gated_field"
FOLD_DIR = ARTIFACT_DIR / "folds"
STATIC_DIR = ARTIFACT_DIR / "static"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e199_propagation_gated_field"
LOG_DIR = ANALYSIS_DIR / "logs"
PREPARE_REPORT = ANALYSIS_DIR / "prepare.json"
FULL_REPORT = ANALYSIS_DIR / "fixed500_report.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
PAIR_NEIGHBORS = STATIC_DIR / "pair_neighbor_ids.npy"
PAIR_DISTANCES = STATIC_DIR / "pair_distances_f16.npy"
PAIR_TARGETS = STATIC_DIR / "pair_targets_f16.npy"
PAIR_CARRIER = STATIC_DIR / "pair_carrier_parameters.npz"
PAIR_TARGET_DONE = STATIC_DIR / "pair_targets_done.npy"

ORIGINAL_BANK = independent.CARRIER_BANK
GATED_BANK = ARTIFACT_DIR / "oof_gated_carrier_bank.npy"
COMBINED_BANK = ARTIFACT_DIR / "oof_combined_carrier_bank.npy"
GRAM_CACHE = ARTIFACT_DIR / "oof_port_gram.npz"
STACK_CHANNEL = ARTIFACT_DIR / "oof_stacked_complex_channel.npy"
PROJECTED_CHANNEL = ARTIFACT_DIR / "oof_stacked_v4_projection.npy"
SCALED_CHANNEL = ARTIFACT_DIR / "oof_stacked_v4_projection_nested_scaled.npy"
CURRENT_AD = ARTIFACT_DIR / "oof_stacked_current_ad_f16.npy"
V7_CHANNEL = ARTIFACT_DIR / "oof_v7_channel.npy"
V9_CHANNEL = ARTIFACT_DIR / "oof_v9_channel.npy"
V9_PDP = ARTIFACT_DIR / "oof_v9_pdp_f16.npy"
V10_ALPHA = ARTIFACT_DIR / "oof_v10_alpha_f16.npy"
V10_CHANNEL = ARTIFACT_DIR / "oof_v10_channel.npy"
V11_GATE_FEATURES = ARTIFACT_DIR / "oof_v11_gate_features.npy"
V11_GATE = ARTIFACT_DIR / "oof_v11_gate_f16.npy"
V11_CHANNEL = ARTIFACT_DIR / "oof_v11_channel.npy"
FINAL_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
FIXED_ENDPOINT = ARTIFACT_DIR / "fixed500_strict_e190_endpoint.npz"

POINTS = 4000
CANDIDATE_POOL = 64
DONORS = 32
STATE_DIM = 963 * 2
STATE_HIDDEN = 96
PAIR_HIDDEN = 192
PAIR_BATCH_QUERIES = 96
PAIR_TARGET_BATCH = 32
TRAIN_EPOCHS = 24
VALIDATION_EPOCHS = (8, 16, 24)
GATE_STRENGTH = 6.0
GATE_LIMIT = 2.0
V15_SCORE = 0.6863750036074728
TARGET_SCORE = 0.700000
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    outer: int
    validation_fold: int
    epochs: int = TRAIN_EPOCHS
    pair_batch_queries: int = PAIR_BATCH_QUERIES
    learning_rate: float = 8e-4
    weight_decay: float = 2e-4
    listwise_weight: float = 0.25
    residual_weight: float = 0.01
    random_state: int = 20260813199


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "fold", "aggregate"), default="prepare"
    )
    parser.add_argument("--outer", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fold_paths(outer: int) -> dict[str, Path]:
    return {
        "checkpoint": FOLD_DIR / f"outer{outer}_model.pt",
        "gated_bank": FOLD_DIR / f"outer{outer}_gated_carriers.npy",
        "query": FOLD_DIR / f"outer{outer}_query.npz",
        "report": ANALYSIS_DIR / f"outer{outer}_report.json",
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def digest_rows(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_channel(path: Path, count: int = 500) -> np.memmap:
    value = np.load(path, mmap_mode="r")
    expected = (count, 256, 4, 192)
    if value.shape != expected or value.dtype != np.complex64:
        raise ValueError(f"invalid channel {PATHS.relative(path)}: {value.shape}/{value.dtype}")
    for start in range(0, count, 8):
        block = np.asarray(value[start : start + 8])
        if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
            raise ValueError(f"non-finite channel {PATHS.relative(path)}")
    return value


def canonical_split(outer: int, validation_fold: int) -> dict[str, np.ndarray]:
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    outer_rows = folds[f"fold{outer}_query"].astype(np.int64)
    validation_rows = folds[f"fold{validation_fold}_query"].astype(np.int64)
    if not np.array_equal(outer_rows, query_ids[fold_id == outer]):
        raise RuntimeError(f"E199 canonical outer{outer} identity mismatch")
    if not np.array_equal(validation_rows, query_ids[fold_id == validation_fold]):
        raise RuntimeError(f"E199 canonical validation{validation_fold} identity mismatch")
    outer_fit = folds[f"fold{outer}_fit"].astype(np.int64)
    validation_fit = folds[f"fold{validation_fold}_fit"].astype(np.int64)
    screen_fit = np.intersect1d(outer_fit, validation_fit, assume_unique=True)
    leakage = {
        "outer_in_screen_fit": int(np.intersect1d(outer_rows, screen_fit).size),
        "validation_in_screen_fit": int(
            np.intersect1d(validation_rows, screen_fit).size
        ),
        "outer_in_refit": int(np.intersect1d(outer_rows, outer_fit).size),
    }
    if any(leakage.values()):
        raise RuntimeError(f"E199 fold leakage: {leakage}")
    return {
        "screen_fit": screen_fit.astype(np.int64),
        "validation": validation_rows,
        "refit": outer_fit,
        "outer": outer_rows,
        "leakage": leakage,
    }


def build_neighbor_pool(
    positions: np.ndarray,
    sites: np.ndarray,
    full_fold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor = np.empty((POINTS, CANDIDATE_POOL), dtype=np.int32)
    distance = np.empty((POINTS, CANDIDATE_POOL), dtype=np.float32)
    all_rows = np.arange(POINTS, dtype=np.int64)
    for site_value in np.unique(sites):
        for fold_value in np.unique(full_fold):
            query = all_rows[(sites == site_value) & (full_fold == fold_value)]
            pool = all_rows[(sites == site_value) & (full_fold != fold_value)]
            tree = cKDTree(positions[pool, :2])
            local_distance, local_id = tree.query(
                positions[query, :2], k=CANDIDATE_POOL, workers=-1
            )
            neighbor[query] = pool[local_id]
            distance[query] = local_distance.astype(np.float32)
    if np.any(neighbor == np.arange(POINTS, dtype=np.int32)[:, None]):
        raise RuntimeError("E199 training neighbor pool contains query identity")
    return neighbor, distance


def fit_pair_carriers(
    train_channel: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    full_fold: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    beta = np.zeros(POINTS, dtype=np.float32)
    gamma = np.zeros(POINTS, dtype=np.float32)
    records: dict[str, Any] = {}
    all_rows = np.arange(POINTS, dtype=np.int64)
    for fold_value in np.unique(full_fold):
        fit = all_rows[full_fold != fold_value]
        values = estimate_carrier(
            train_channel,
            positions,
            sites,
            bs_positions,
            fit,
            point_power,
            max_pairs_per_site=500,
            seed=20_260_813 + int(fold_value) * 101,
        )
        records[str(int(fold_value))] = {
            str(site): carrier.to_dict() for site, carrier in values.items()
        }
        for site_value, carrier in values.items():
            rows = (full_fold == fold_value) & (sites == site_value)
            beta[rows] = carrier.beta
            gamma[rows] = carrier.gamma
    return beta, gamma, records


@torch.inference_mode()
def pair_targets_block(
    train_channel: torch.Tensor,
    query_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    device = train_channel.device
    query = torch.as_tensor(query_ids, dtype=torch.long, device=device)
    donor_id = torch.as_tensor(neighbor_ids, dtype=torch.long, device=device)
    donor = train_channel[donor_id]
    truth = train_channel[query]
    site = sites[query_ids]
    query_range = np.linalg.norm(
        positions[query_ids] - bs_positions[site], axis=1
    )
    donor_range = np.linalg.norm(
        positions[neighbor_ids] - bs_positions[site, None, :], axis=2
    )
    subcarrier = np.linspace(-0.5, 0.5, 192, dtype=np.float32)
    local_beta = beta[query_ids, None] + gamma[query_ids, None] * subcarrier[None]
    phase = np.exp(
        1j * (donor_range - query_range[:, None])[:, :, None] * local_beta[:, None]
    ).astype(np.complex64)
    aligned = donor * torch.as_tensor(phase, device=device)[:, :, None, None]
    batch, count = aligned.shape[:2]
    flat_prediction = aligned.reshape(batch * count, 256, 4, 192)
    flat_truth = truth[:, None].expand(-1, count, -1, -1, -1).reshape_as(
        flat_prediction
    )
    values = point_metric_statistics_torch(flat_prediction, flat_truth)
    nmse = values.squared_error / np.maximum(values.truth_energy, TINY)
    target = np.stack(
        (values.pas, values.pdp, 1.0 / (1.0 + nmse)), axis=-1
    ).reshape(batch, count, 3)
    return target.astype(np.float32)


def prepare_static(device: torch.device, force: bool) -> dict[str, Any]:
    started = time.time()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if PREPARE_REPORT.is_file() and not force:
        ready = all(
            path.is_file()
            for path in (
                e193.TRAIN_BASE_EMBEDDING,
                e193.TRAIN_SCATTER_EMBEDDING,
                ORIGINAL_BANK,
            )
        )
        if ready:
            return json.loads(PREPARE_REPORT.read_text(encoding="utf-8"))

    e193.prepare_static_inputs(force=False)
    base = np.load(e193.TRAIN_BASE_EMBEDDING, mmap_mode="r")
    scatter = np.load(e193.TRAIN_SCATTER_EMBEDDING, mmap_mode="r")
    bank = np.load(ORIGINAL_BANK, mmap_mode="r")
    if base.shape != (POINTS, 963) or scatter.shape != (POINTS, 963):
        raise RuntimeError(
            f"invalid E199 propagation states {base.shape}/{scatter.shape}"
        )
    if bank.shape != (len(carrier_bank.carrier_configs()), 500, 256, 4, 192):
        raise RuntimeError(f"invalid E199 original carrier bank {bank.shape}")
    if not np.isfinite(np.asarray(base)).all() or not np.isfinite(
        np.asarray(scatter)
    ).all():
        raise RuntimeError("non-finite E199 propagation state")
    report = {
        "experiment": VERSION,
        "status": "prepare_complete",
        "data_lineage": {
            "official_train_h": PATHS.relative(TRAIN_CHANNEL),
            "external_train_state": [
                PATHS.relative(e193.TRAIN_BASE_EMBEDDING),
                PATHS.relative(e193.TRAIN_SCATTER_EMBEDDING),
            ],
            "hidden_test_h_used": False,
        },
        "static_inputs": {
            "base_state_shape": list(base.shape),
            "scatter_state_shape": list(scatter.shape),
            "original_carrier_bank_shape": list(bank.shape),
            "pair_targets": "computed independently inside each canonical fold",
        },
        "performance": {
            "device": torch.cuda.get_device_name(device),
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "phase_seconds": {"static_validation": float(time.time() - started)},
            "measured_bottleneck": "static_validation",
            "wall_seconds": float(time.time() - started),
        },
    }
    atomic_json(PREPARE_REPORT, report)
    return report


class PairUtilityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, STATE_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(STATE_HIDDEN),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(3, 48),
            nn.GELU(),
            nn.Linear(48, 3),
        )
        pair_dim = STATE_HIDDEN * 4 + 9
        self.state_head = nn.Sequential(
            nn.Linear(pair_dim, PAIR_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(PAIR_HIDDEN),
            nn.Linear(PAIR_HIDDEN, 96),
            nn.GELU(),
            nn.Linear(96, 3),
        )

    def forward(
        self,
        states: torch.Tensor,
        query_ids: torch.Tensor,
        donor_ids: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.state_encoder(states[query_ids])
        donor = self.state_encoder(states[donor_ids])
        query_expanded = query[:, None].expand_as(donor)
        pair = torch.cat(
            (
                query_expanded,
                donor,
                torch.abs(query_expanded - donor),
                query_expanded * donor,
                geometry,
            ),
            dim=-1,
        )
        distance = self.distance_head(geometry[..., :3])
        residual = 0.35 * torch.tanh(self.state_head(pair))
        prediction = torch.sigmoid(distance + residual)
        return prediction, residual


def state_tensor(device: torch.device) -> torch.Tensor:
    base = np.asarray(np.load(e193.TRAIN_BASE_EMBEDDING, mmap_mode="r"), dtype=np.float32)
    scatter = np.asarray(
        np.load(e193.TRAIN_SCATTER_EMBEDDING, mmap_mode="r"), dtype=np.float32
    )
    value = np.concatenate((base, scatter), axis=1)
    if value.shape != (POINTS, STATE_DIM):
        raise RuntimeError(f"invalid E199 state tensor {value.shape}")
    return torch.as_tensor(value, device=device)


def pair_geometry(
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    base_embedding: np.ndarray,
    scatter_embedding: np.ndarray,
) -> np.ndarray:
    query_position = positions[query_ids]
    donor_position = positions[donor_ids]
    delta = donor_position[..., :2] - query_position[:, None, :2]
    distance = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(distance[..., None], 1e-6)
    query_bs = bs_positions[sites[query_ids]]
    query_range = np.linalg.norm(query_position - query_bs, axis=1)
    donor_range = np.linalg.norm(donor_position - query_bs[:, None], axis=2)
    base_similarity = np.sum(
        base_embedding[query_ids, None] * base_embedding[donor_ids], axis=-1
    )
    scatter_similarity = np.sum(
        scatter_embedding[query_ids, None] * scatter_embedding[donor_ids], axis=-1
    )
    return np.stack(
        (
            np.log1p(distance) / 4.0,
            np.minimum(distance, 40.0) / 20.0,
            np.minimum(np.abs(donor_range - query_range[:, None]), 40.0) / 20.0,
            direction[..., 0],
            direction[..., 1],
            np.sin(np.arctan2(delta[..., 1], delta[..., 0])),
            np.cos(np.arctan2(delta[..., 1], delta[..., 0])),
            base_similarity,
            scatter_similarity,
        ),
        axis=-1,
    ).astype(np.float32)


def select_safe_training_pairs(
    query_rows: np.ndarray,
    allowed_rows: np.ndarray,
    pool_neighbor: np.ndarray,
    pool_distance: np.ndarray,
    pool_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed = np.zeros(POINTS, dtype=bool)
    allowed[allowed_rows] = True
    neighbor = np.empty((len(query_rows), DONORS), dtype=np.int32)
    distance = np.empty((len(query_rows), DONORS), dtype=np.float32)
    target = np.empty((len(query_rows), DONORS, 3), dtype=np.float32)
    for local, query in enumerate(query_rows):
        valid = np.flatnonzero(allowed[pool_neighbor[query]])
        if len(valid) < DONORS:
            raise RuntimeError(
                f"E199 query {query} has only {len(valid)} safe training donors"
            )
        chosen = valid[:DONORS]
        neighbor[local] = pool_neighbor[query, chosen]
        distance[local] = pool_distance[query, chosen]
        target[local] = pool_target[query, chosen]
    if not np.all(allowed[neighbor]):
        raise RuntimeError("E199 safe pair selection leaked held donor labels")
    return neighbor, distance, target


def query_neighbors(
    query_rows: np.ndarray,
    allowed_rows: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distance, neighbor = query_site_neighbors(
        positions,
        sites,
        allowed_rows,
        query_rows,
        max_k=DONORS,
    )
    return neighbor.astype(np.int32), distance.astype(np.float32)


def query_neighbors_without_identity(
    query_rows: np.ndarray,
    allowed_rows: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distance, neighbor = query_site_neighbors(
        positions,
        sites,
        allowed_rows,
        query_rows,
        max_k=DONORS + 1,
    )
    output_neighbor = np.empty((len(query_rows), DONORS), dtype=np.int32)
    output_distance = np.empty((len(query_rows), DONORS), dtype=np.float32)
    for local, query in enumerate(query_rows):
        valid = neighbor[local] != query
        selected = np.flatnonzero(valid)[:DONORS]
        if len(selected) != DONORS:
            raise RuntimeError(f"E199 query {query} has insufficient strict donors")
        output_neighbor[local] = neighbor[local, selected]
        output_distance[local] = distance[local, selected]
    allowed = np.zeros(POINTS, dtype=bool)
    allowed[allowed_rows] = True
    if not np.all(allowed[output_neighbor]):
        raise RuntimeError("E199 strict donor selection escaped the allowed fold")
    return output_neighbor, output_distance


def compute_targets_for_queries(
    train_channel: torch.Tensor,
    query_rows: np.ndarray,
    neighbor: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    carrier: dict[int, Any],
) -> np.ndarray:
    beta = np.zeros(POINTS, dtype=np.float32)
    gamma = np.zeros(POINTS, dtype=np.float32)
    for site_value, value in carrier.items():
        rows = sites == int(site_value)
        beta[rows] = value.beta
        gamma[rows] = value.gamma
    output = np.empty((len(query_rows), DONORS, 3), dtype=np.float32)
    for start in range(0, len(query_rows), PAIR_TARGET_BATCH):
        end = min(start + PAIR_TARGET_BATCH, len(query_rows))
        output[start:end] = pair_targets_block(
            train_channel,
            query_rows[start:end],
            neighbor[start:end],
            positions,
            sites,
            bs_positions,
            beta,
            gamma,
        )
    return output


def model_loss(
    prediction: torch.Tensor,
    residual: torch.Tensor,
    target: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    regression = F.smooth_l1_loss(prediction, target, beta=0.05)
    weights = torch.tensor((0.4, 0.4, 0.2), device=prediction.device)
    predicted_score = torch.sum(prediction * weights, dim=-1)
    target_score = torch.sum(target * weights, dim=-1)
    target_rank = torch.softmax(target_score / 0.08, dim=1)
    listwise = torch.sum(
        -target_rank * torch.log_softmax(predicted_score / 0.08, dim=1), dim=1
    ).mean()
    residual_penalty = residual.square().mean()
    loss = (
        regression
        + config.listwise_weight * listwise
        + config.residual_weight * residual_penalty
    )
    return loss, {
        "loss": float(loss.detach().item()),
        "regression": float(regression.detach().item()),
        "listwise": float(listwise.detach().item()),
        "residual_rms": float(torch.sqrt(residual_penalty).detach().item()),
    }


def evaluate_pair_model(
    model: PairUtilityModel,
    states: torch.Tensor,
    query_rows: np.ndarray,
    neighbor: np.ndarray,
    geometry: np.ndarray,
    target: np.ndarray,
    config: Config,
) -> dict[str, float]:
    model.eval()
    values: list[dict[str, float]] = []
    with torch.inference_mode():
        for start in range(0, len(query_rows), config.pair_batch_queries):
            end = min(start + config.pair_batch_queries, len(query_rows))
            query = torch.as_tensor(
                query_rows[start:end], dtype=torch.long, device=states.device
            )
            donor = torch.as_tensor(
                neighbor[start:end], dtype=torch.long, device=states.device
            )
            local_geometry = torch.as_tensor(
                geometry[start:end], device=states.device
            )
            local_target = torch.as_tensor(target[start:end], device=states.device)
            prediction, residual = model(states, query, donor, local_geometry)
            _, item = model_loss(prediction, residual, local_target, config)
            values.append(item)
    return {
        name: float(np.mean([value[name] for value in values])) for name in values[0]
    }


def train_pair_model(
    states: torch.Tensor,
    query_rows: np.ndarray,
    neighbor: np.ndarray,
    geometry: np.ndarray,
    target: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    config: Config,
    epochs: int,
    seed: int,
) -> tuple[PairUtilityModel, int, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PairUtilityModel().to(states.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(query_rows))
        train_values: list[dict[str, float]] = []
        for start in range(0, len(order), config.pair_batch_queries):
            local = order[start : start + config.pair_batch_queries]
            query = torch.as_tensor(
                query_rows[local], dtype=torch.long, device=states.device
            )
            donor = torch.as_tensor(
                neighbor[local], dtype=torch.long, device=states.device
            )
            local_geometry = torch.as_tensor(geometry[local], device=states.device)
            local_target = torch.as_tensor(target[local], device=states.device)
            prediction, residual = model(states, query, donor, local_geometry)
            loss, item = model_loss(prediction, residual, local_target, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite E199 pair model gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_values.append(item)
        scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "train": {
                name: float(np.mean([value[name] for value in train_values]))
                for name in train_values[0]
            },
        }
        if validation is not None and epoch in VALIDATION_EPOCHS:
            validation_value = evaluate_pair_model(
                model, states, *validation, config
            )
            record["validation"] = validation_value
            if validation_value["loss"] < best_loss:
                best_loss = validation_value["loss"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch in VALIDATION_EPOCHS or epoch == epochs:
            message = (
                f"E199 epoch={epoch}/{epochs} train={record['train']['loss']:.6f}"
            )
            if "validation" in record:
                message += f" validation={record['validation']['loss']:.6f}"
            print(message, flush=True)
    if validation is not None:
        if best_state is None or best_epoch == 0:
            raise RuntimeError("E199 screen selected no model")
        model.load_state_dict(best_state)
        return model, best_epoch, history
    return model, epochs, history


@torch.inference_mode()
def predict_gate(
    model: PairUtilityModel,
    states: torch.Tensor,
    query_rows: np.ndarray,
    neighbor: np.ndarray,
    geometry: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    gate = np.empty((len(query_rows), 3, DONORS), dtype=np.float32)
    predicted = np.empty((len(query_rows), DONORS, 3), dtype=np.float32)
    for start in range(0, len(query_rows), batch_size):
        end = min(start + batch_size, len(query_rows))
        query = torch.as_tensor(
            query_rows[start:end], dtype=torch.long, device=states.device
        )
        donor = torch.as_tensor(
            neighbor[start:end], dtype=torch.long, device=states.device
        )
        local_geometry = torch.as_tensor(geometry[start:end], device=states.device)
        output, residual = model(states, query, donor, local_geometry)
        compatibility = residual.permute(0, 2, 1)
        compatibility -= compatibility.mean(dim=2, keepdim=True)
        local_gate = torch.exp(
            torch.clamp(GATE_STRENGTH * compatibility, -GATE_LIMIT, GATE_LIMIT)
        )
        gate[start:end] = local_gate.cpu().numpy().astype(np.float32)
        predicted[start:end] = output.cpu().numpy().astype(np.float32)
    return gate, predicted


def generate_gated_carriers(
    model: PairUtilityModel,
    states: torch.Tensor,
    query_rows: np.ndarray,
    allowed_rows: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    train_channel_np: np.ndarray,
    train_channel_gpu: torch.Tensor,
    point_power: np.ndarray,
    device: torch.device,
    output_path: Path,
    outer: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    neighbor, distance = query_neighbors(
        query_rows, allowed_rows, positions, sites
    )
    base_embedding = np.asarray(
        np.load(e193.TRAIN_BASE_EMBEDDING, mmap_mode="r"), dtype=np.float32
    )
    scatter_embedding = np.asarray(
        np.load(e193.TRAIN_SCATTER_EMBEDDING, mmap_mode="r"), dtype=np.float32
    )
    geometry = pair_geometry(
        query_rows,
        neighbor,
        positions,
        sites,
        bs_positions,
        base_embedding,
        scatter_embedding,
    )
    gate, predicted = predict_gate(
        model, states, query_rows, neighbor, geometry, 128
    )
    carrier = estimate_carrier(
        train_channel_np,
        positions,
        sites,
        bs_positions,
        allowed_rows,
        point_power,
        max_pairs_per_site=500,
        seed=20_260_805 + outer * 101,
    )
    templates = carrier_bank.carrier_configs()
    configs_by_site = {
        int(site): [
            type(config)(
                **{
                    **config.to_dict(),
                    "beta": value.beta,
                    "gamma": value.gamma,
                }
            )
            for config in templates
        ]
        for site, value in carrier.items()
    }
    partial = output_path.with_name(output_path.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(3 * len(templates), len(query_rows), 256, 4, 192),
    )
    for start in range(0, len(query_rows), 8):
        end = min(start + 8, len(query_rows))
        for gate_index in range(3):
            prediction, _, _ = batched_predict_many_torch(
                train_channel_gpu,
                neighbor[start:end],
                positions[neighbor[start:end]],
                positions[query_rows[start:end]],
                distance[start:end],
                bs_positions,
                sites[query_rows[start:end]],
                configs_by_site,
                point_power,
                donor_gate=gate[start:end, gate_index],
            )
            destination = slice(
                gate_index * len(templates), (gate_index + 1) * len(templates)
            )
            output[destination, start:end] = (
                prediction.permute(1, 0, 2, 3, 4).cpu().numpy()
            )
        output.flush()
    del output
    os.replace(partial, output_path)
    return (
        {
            "path": PATHS.relative(output_path),
            "shape": [3 * len(templates), len(query_rows), 256, 4, 192],
            "gate_names": ["pas", "pdp", "complex"],
            "gate_mean": np.mean(gate, axis=(0, 2)).tolist(),
            "gate_std": np.std(gate, axis=(0, 2)).tolist(),
            "gate_p05": np.quantile(gate, 0.05, axis=(0, 2)).tolist(),
            "gate_p95": np.quantile(gate, 0.95, axis=(0, 2)).tolist(),
            "predicted_utility_mean": np.mean(predicted, axis=(0, 1)).tolist(),
            "carrier": {str(site): value.to_dict() for site, value in carrier.items()},
        },
        neighbor,
        distance,
        gate,
    )


def run_fold(outer: int, device: torch.device, force: bool) -> dict[str, Any]:
    started = time.time()
    paths = fold_paths(outer)
    if paths["report"].is_file() and paths["gated_bank"].is_file() and not force:
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        if report.get("status") == "fold_complete":
            return report
    config = Config(outer=outer, validation_fold=(outer + 1) % 5)
    split = canonical_split(outer, config.validation_fold)
    setup, positions, _, sites, _ = load_positions()
    positions = np.asarray(positions, dtype=np.float64)
    sites = np.asarray(sites, dtype=np.int8)
    bs_positions = np.asarray(setup.bs_positions, dtype=np.float64)
    train_np = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = load_or_compute_power(train_np, 16)
    train_gpu = torch.as_tensor(
        np.asarray(train_np), dtype=torch.complex64, device=device
    )
    states = state_tensor(device)
    base_embedding = np.asarray(
        np.load(e193.TRAIN_BASE_EMBEDDING, mmap_mode="r"), dtype=np.float32
    )
    scatter_embedding = np.asarray(
        np.load(e193.TRAIN_SCATTER_EMBEDDING, mmap_mode="r"), dtype=np.float32
    )
    phase: dict[str, float] = {}
    step = time.time()
    screen_neighbor, _ = query_neighbors_without_identity(
        split["screen_fit"],
        split["screen_fit"],
        positions,
        sites,
    )
    screen_geometry = pair_geometry(
        split["screen_fit"],
        screen_neighbor,
        positions,
        sites,
        bs_positions,
        base_embedding,
        scatter_embedding,
    )
    validation_neighbor, _ = query_neighbors(
        split["validation"], split["screen_fit"], positions, sites
    )
    validation_geometry = pair_geometry(
        split["validation"],
        validation_neighbor,
        positions,
        sites,
        bs_positions,
        base_embedding,
        scatter_embedding,
    )
    validation_carrier = estimate_carrier(
        train_np,
        positions,
        sites,
        bs_positions,
        split["screen_fit"],
        point_power,
        max_pairs_per_site=500,
        seed=20_260_813 + outer * 101,
    )
    validation_target = compute_targets_for_queries(
        train_gpu,
        split["validation"],
        validation_neighbor,
        positions,
        sites,
        bs_positions,
        validation_carrier,
    )
    screen_target = compute_targets_for_queries(
        train_gpu,
        split["screen_fit"],
        screen_neighbor,
        positions,
        sites,
        bs_positions,
        validation_carrier,
    )
    phase["screen_pair_data"] = float(time.time() - step)

    step = time.time()
    screen_model, best_epoch, screen_history = train_pair_model(
        states,
        split["screen_fit"],
        screen_neighbor,
        screen_geometry,
        screen_target,
        (
            split["validation"],
            validation_neighbor,
            validation_geometry,
            validation_target,
        ),
        config,
        config.epochs,
        config.random_state + outer,
    )
    phase["screen_training"] = float(time.time() - step)
    del screen_model, screen_neighbor, screen_geometry, screen_target
    if device.type == "cuda":
        torch.cuda.empty_cache()

    step = time.time()
    refit_neighbor, _ = query_neighbors_without_identity(
        split["refit"],
        split["refit"],
        positions,
        sites,
    )
    refit_geometry = pair_geometry(
        split["refit"],
        refit_neighbor,
        positions,
        sites,
        bs_positions,
        base_embedding,
        scatter_embedding,
    )
    refit_carrier = estimate_carrier(
        train_np,
        positions,
        sites,
        bs_positions,
        split["refit"],
        point_power,
        max_pairs_per_site=500,
        seed=20_260_913 + outer * 101,
    )
    refit_target = compute_targets_for_queries(
        train_gpu,
        split["refit"],
        refit_neighbor,
        positions,
        sites,
        bs_positions,
        refit_carrier,
    )
    model, _, refit_history = train_pair_model(
        states,
        split["refit"],
        refit_neighbor,
        refit_geometry,
        refit_target,
        None,
        config,
        best_epoch,
        config.random_state + 100_000 + outer,
    )
    phase["refit_training"] = float(time.time() - step)

    step = time.time()
    carrier_report, neighbor, distance, gate = generate_gated_carriers(
        model,
        states,
        split["outer"],
        split["refit"],
        positions,
        sites,
        bs_positions,
        train_np,
        train_gpu,
        point_power,
        device,
        paths["gated_bank"],
        outer,
    )
    atomic_save_npz(
        paths["query"],
        query_ids=split["outer"].astype(np.int32),
        neighbor_ids=neighbor,
        distances=distance.astype(np.float16),
        gate=gate.astype(np.float16),
    )
    atomic_torch_save(
        paths["checkpoint"],
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "best_epoch": best_epoch,
            "outer": outer,
            "split_digest": digest_rows(split["refit"], split["outer"]),
        },
    )
    phase["gated_carriers"] = float(time.time() - step)
    peak = float(torch.cuda.max_memory_allocated(device) / 2**30)
    report = {
        "experiment": VERSION,
        "status": "fold_complete",
        "outer": outer,
        "configuration": asdict(config),
        "split": {
            "screen_fit": int(len(split["screen_fit"])),
            "validation": int(len(split["validation"])),
            "refit": int(len(split["refit"])),
            "outer": int(len(split["outer"])),
            "zero_label_overlap": split["leakage"],
        },
        "selection": {
            "best_epoch": best_epoch,
            "screen_history": screen_history,
            "refit_history": refit_history,
        },
        "gated_carriers": carrier_report,
        "performance": {
            "device": torch.cuda.get_device_name(device),
            "peak_torch_memory_gib": peak,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "phase_seconds": phase,
            "measured_bottleneck": max(phase, key=phase.get),
            "wall_seconds": float(time.time() - started),
        },
    }
    atomic_json(paths["report"], report)
    return report


def open_combined_bank_parts() -> tuple[np.memmap, dict[int, np.memmap], np.ndarray]:
    configs = carrier_bank.carrier_configs()
    original = np.load(ORIGINAL_BANK, mmap_mode="r")
    if original.shape != (len(configs), 500, 256, 4, 192):
        raise RuntimeError(f"invalid original carrier bank {original.shape}")
    metadata = np.load(OOF_META, allow_pickle=False)
    fold_id = metadata["fold_id"].astype(np.int8)
    local_index = np.empty(500, dtype=np.int32)
    gated: dict[int, np.memmap] = {}
    for outer in range(5):
        paths = fold_paths(outer)
        rows = np.flatnonzero(fold_id == outer)
        query = np.load(paths["query"], allow_pickle=False)["query_ids"].astype(np.int64)
        expected_query = metadata["query_ids"].astype(np.int64)[rows]
        if not np.array_equal(query, expected_query):
            raise RuntimeError(f"E199 outer{outer} carrier identity mismatch")
        value = np.load(paths["gated_bank"], mmap_mode="r")
        if value.shape != (3 * len(configs), len(rows), 256, 4, 192):
            raise RuntimeError(f"invalid E199 outer{outer} gated bank {value.shape}")
        gated[outer] = value
        local_index[rows] = np.arange(len(rows), dtype=np.int32)
    return original, gated, local_index


def combined_bank_row(
    original: np.memmap,
    gated: dict[int, np.memmap],
    local_index: np.ndarray,
    fold_id: np.ndarray,
    row: int,
) -> np.ndarray:
    outer = int(fold_id[row])
    return np.concatenate(
        (
            np.asarray(original[:, row], dtype=np.complex64),
            np.asarray(gated[outer][:, int(local_index[row])], dtype=np.complex64),
        ),
        axis=0,
    )


def compute_combined_gram(device: torch.device, force: bool) -> dict[str, np.ndarray]:
    if GRAM_CACHE.is_file() and not force:
        cached = np.load(GRAM_CACHE, allow_pickle=False)
        required = {"gram", "cross", "energy", "query_ids", "fold_id"}
        if required.issubset(cached.files):
            return {key: cached[key] for key in cached.files}
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    original, gated, local_index = open_combined_bank_parts()
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    carriers = 4 * len(carrier_bank.carrier_configs())
    gram = np.empty((500, 8, carriers, carriers), dtype=np.complex64)
    cross = np.empty((500, 8, carriers), dtype=np.complex64)
    energy = np.empty((500, 8), dtype=np.float64)
    completed = 0
    for outer in range(5):
        rows = np.flatnonzero(fold_id == outer)
        original_fold = np.asarray(original[:, rows], dtype=np.complex64)
        gated_fold = np.asarray(gated[outer], dtype=np.complex64)
        if gated_fold.shape[1] != len(rows):
            raise RuntimeError(f"E199 outer{outer} batch identity mismatch")
        for start in range(0, len(rows), 4):
            end = min(start + 4, len(rows))
            local_rows = rows[start:end]
            candidate_np = np.concatenate(
                (original_fold[:, start:end], gated_fold[:, start:end]), axis=0
            ).transpose(1, 0, 2, 3, 4)
            candidate = torch.as_tensor(candidate_np, device=device).reshape(
                end - start, carriers, 2, 16, 8, 4, 192
            )
            truth = torch.as_tensor(
                np.asarray(truth_all[query_ids[local_rows]], dtype=np.complex64),
                device=device,
            ).reshape(end - start, 2, 16, 8, 4, 192)
            with torch.inference_mode():
                candidate = candidate.permute(0, 2, 5, 1, 3, 4, 6).reshape(
                    end - start, 8, carriers, -1
                )
                truth = truth.permute(0, 1, 4, 2, 3, 5).reshape(
                    end - start, 8, -1
                )
                gram[local_rows] = torch.einsum(
                    "bgcd,bged->bgce", candidate.conj(), candidate
                ).cpu().numpy()
                cross[local_rows] = torch.einsum(
                    "bgcd,bgd->bgc", candidate.conj(), truth
                ).cpu().numpy()
                energy[local_rows] = (
                    torch.sum(truth.abs().square(), dim=2, dtype=torch.float64)
                    .cpu()
                    .numpy()
                )
            completed += end - start
            if completed % 20 < end - start or completed == len(query_ids):
                print(f"E199 carrier Gram {completed}/{len(query_ids)}", flush=True)
        del original_fold, gated_fold
    atomic_save_npz(
        GRAM_CACHE,
        gram=gram,
        cross=cross,
        energy=energy,
        query_ids=query_ids,
        fold_id=fold_id,
    )
    return {
        "gram": gram,
        "cross": cross,
        "energy": energy,
        "query_ids": query_ids,
        "fold_id": fold_id,
    }


def combine_row(carriers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    count = carriers.shape[0]
    candidate = (
        carriers.reshape(count, 2, 16, 8, 4, 192)
        .transpose(1, 4, 0, 2, 3, 5)
        .reshape(8, count, 16, 8, 192)
    )
    combined = np.einsum("gcvhd,gc->gvhd", candidate, weights, optimize=True)
    return (
        combined.reshape(2, 4, 16, 8, 192)
        .transpose(0, 2, 3, 1, 4)
        .reshape(256, 4, 192)
        .astype(np.complex64)
    )


def build_combined_oof_stack(
    cache: dict[str, np.ndarray],
    config: stacking.StackConfig,
    site: np.ndarray,
    regime: np.ndarray,
) -> None:
    metadata = np.load(OOF_META, allow_pickle=False)
    fold_id = metadata["fold_id"].astype(np.int8)
    labels = stacking._scope_labels(config, site, regime)
    original, gated, local_index = open_combined_bank_parts()
    partial = STACK_CHANNEL.with_name(STACK_CHANNEL.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=(500, 256, 4, 192)
    )
    for outer in range(5):
        fit = np.flatnonzero(fold_id != outer)
        held = np.flatnonzero(fold_id == outer)
        weights = stacking.fit_weights(cache, fit, labels, config)
        row_weight = v7_projection.row_weights(labels, held, weights, config)
        for local, row in enumerate(held):
            carriers = combined_bank_row(
                original, gated, local_index, fold_id, int(row)
            )
            output[row] = combine_row(carriers, row_weight[local])
        output.flush()
        print(f"E199 stacked OOF fold={outer} count={len(held)}", flush=True)
    del output
    os.replace(partial, STACK_CHANNEL)


def build_complete_base(device: torch.device, force: bool) -> np.memmap:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    setup, positions, _, train_site, _ = load_positions()
    del setup, positions
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(lifecycle.REGIME_CACHE, allow_pickle=False)[
        "oof_regime"
    ].astype(np.int8)
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")

    cache = compute_combined_gram(device, force)
    selection = json.loads(v7_projection.STACK_SELECTION.read_text(encoding="utf-8"))
    config = v7_projection.stack_config_from_record(
        selection["posterior_best"]["config"]
    )
    build_combined_oof_stack(cache, config, site, regime)

    original_stack = v7_projection.OOF_STACK
    original_projected = v7_projection.OOF_PROJECTED
    v7_projection.OOF_STACK = STACK_CHANNEL
    v7_projection.OOF_PROJECTED = PROJECTED_CHANNEL
    try:
        v7_projection.project_v4_policy(device, 8, query_ids, site)
    finally:
        v7_projection.OOF_STACK = original_stack
        v7_projection.OOF_PROJECTED = original_projected
    projected_score = v7_projection.score_channel(
        np.load(PROJECTED_CHANNEL, mmap_mode="r"),
        train_channel,
        query_ids,
        site,
        fold_id,
        8,
    )

    original_scaled = independent.SCALED_CHANNEL
    independent.SCALED_CHANNEL = SCALED_CHANNEL
    try:
        original_projected_path = independent.PROJECTED_CHANNEL
        independent.PROJECTED_CHANNEL = PROJECTED_CHANNEL
        try:
            independent.build_scaled_projection(projected_score, fold_id, site, force)
        finally:
            independent.PROJECTED_CHANNEL = original_projected_path
    finally:
        independent.SCALED_CHANNEL = original_scaled

    original_paths = {
        "SCALED_CHANNEL": independent.SCALED_CHANNEL,
        "CURRENT_AD": independent.CURRENT_AD,
        "V7_CHANNEL": independent.V7_CHANNEL,
    }
    independent.SCALED_CHANNEL = SCALED_CHANNEL
    independent.CURRENT_AD = CURRENT_AD
    independent.V7_CHANNEL = V7_CHANNEL
    try:
        independent.build_v7_lifecycle(device, 8, fold_id, site, regime, force)
    finally:
        for name, value in original_paths.items():
            setattr(independent, name, value)

    original_v9 = independent.V9_CHANNEL
    original_v7 = independent.V7_CHANNEL
    independent.V7_CHANNEL = V7_CHANNEL
    independent.V9_CHANNEL = V9_CHANNEL
    try:
        independent.build_v9(device, 8, force)
    finally:
        independent.V7_CHANNEL = original_v7
        independent.V9_CHANNEL = original_v9

    original_paths = {
        "V9_CHANNEL": independent.V9_CHANNEL,
        "V9_PDP": independent.V9_PDP,
        "V10_ALPHA": independent.V10_ALPHA,
        "V10_CHANNEL": independent.V10_CHANNEL,
        "V11_GATE_FEATURES": independent.V11_GATE_FEATURES,
        "V11_GATE": independent.V11_GATE,
        "V11_CHANNEL": independent.V11_CHANNEL,
    }
    independent.V9_CHANNEL = V9_CHANNEL
    independent.V9_PDP = V9_PDP
    independent.V10_ALPHA = V10_ALPHA
    independent.V10_CHANNEL = V10_CHANNEL
    independent.V11_GATE_FEATURES = V11_GATE_FEATURES
    independent.V11_GATE = V11_GATE
    independent.V11_CHANNEL = V11_CHANNEL
    try:
        independent.build_v10_v11(device, 8, query_ids, force)
    finally:
        for name, value in original_paths.items():
            setattr(independent, name, value)
    return validate_channel(V11_CHANNEL)


def rebuild_fixed_endpoint(device: torch.device, force: bool) -> dict[str, np.ndarray]:
    if FIXED_ENDPOINT.is_file() and not force:
        cached = np.load(FIXED_ENDPOINT, allow_pickle=False)
        return {name: cached[name] for name in ("mass", "coordinate", "port")}
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    mass = np.zeros((500, e85.PATHS_PER_POINT), dtype=np.float32)
    coordinate = np.zeros((500, e85.PATHS_PER_POINT, 3), dtype=np.float32)
    port = np.zeros(
        (500, e85.PATHS_PER_POINT, e85.MP, e85.N_UE), dtype=np.float32
    )
    store = e195.ResidentStore(device)
    mapping_data = e190.e98.MappingData(device)
    base_coordinate = np.asarray(
        np.load(e85.BASE_COORDINATE, mmap_mode="r"), dtype=np.float32
    )
    for outer in range(5):
        local_rows = np.flatnonzero(fold_id == outer)
        rows = query_ids[local_rows]
        checkpoint = torch.load(
            e195.fold_paths(outer)["refit_base"],
            map_location=device,
            weights_only=False,
        )
        mapping = np.load(e195.fold_paths(outer)["mapping"], allow_pickle=False)
        corrected = e195.candidate_coordinates(
            mapping["refit_latent"].astype(np.float32),
            device,
            mapping_data,
            base_coordinate,
        )
        store.coordinate = torch.as_tensor(corrected, device=device)
        config = e190.Config(**checkpoint["config"])
        model = e195.make_fresh_base_model(
            store, config, device, config.random_state + 100_000 + outer
        )
        model.load_state_dict(checkpoint["state_dict"])
        local_mass, local_coordinate, local_port = e195.endpoint_arrays(
            model,
            store,
            rows,
            torch.as_tensor(checkpoint["feature_mean"], device=device),
            torch.as_tensor(checkpoint["feature_std"], device=device),
            64,
            include_probe=False,
        )
        mass[local_rows] = local_mass
        coordinate[local_rows] = local_coordinate
        port[local_rows] = local_port
        del model, checkpoint, mapping, corrected
        torch.cuda.empty_cache()
    atomic_save_npz(
        FIXED_ENDPOINT,
        query_ids=query_ids.astype(np.int32),
        mass=mass,
        coordinate=coordinate,
        port=port,
    )
    return {"mass": mass, "coordinate": coordinate, "port": port}


def project_endpoint(
    base: torch.Tensor,
    endpoint_pas: np.ndarray,
    endpoint_pdp: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    spec = e86.Candidate(
        endpoint="e199_propagation_gated_field",
        mode="sqrt",
        pas_beta=0.20,
        pdp_beta=0.20,
        order="pas_pdp",
        cycles=4,
    )
    return e86.make_candidate(
        base,
        pas_distribution(base),
        pdp_distribution(base),
        torch.as_tensor(endpoint_pas, device=device),
        torch.as_tensor(endpoint_pdp, device=device),
        spec,
    )


def aggregate(device: torch.device, force: bool) -> dict[str, Any]:
    started = time.time()
    phase: dict[str, float] = {}
    step = time.time()
    base = build_complete_base(device, force)
    phase["complete_base"] = float(time.time() - step)
    step = time.time()
    endpoint = rebuild_fixed_endpoint(device, force)
    endpoint_pas, endpoint_pdp = e190.render_endpoint(
        endpoint["mass"], endpoint["coordinate"], endpoint["port"], device
    )
    independent_base = torch.as_tensor(
        np.array(np.load(independent.V11_CHANNEL, mmap_mode="r"), copy=True),
        device=device,
    )
    baseline_replay = project_endpoint(
        independent_base, endpoint_pas, endpoint_pdp, device
    )
    phase["endpoint_rebuild_identity"] = float(time.time() - step)

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    truth_np = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_np[query_ids], dtype=np.complex64), device=device
    )
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    baseline_metric = point_metrics(
        baseline_replay, truth, truth_pas, truth_pdp
    )
    baseline = statistics(*baseline_metric)
    step = time.time()
    candidate = project_endpoint(
        torch.as_tensor(np.array(base, copy=True), device=device),
        endpoint_pas,
        endpoint_pdp,
        device,
    )
    candidate_metric = point_metrics(candidate, truth, truth_pas, truth_pdp)
    selected = statistics(*candidate_metric)
    output = candidate.cpu().numpy().astype(np.complex64)
    atomic_save_npy(FINAL_CHANNEL, output)
    phase["projection_and_scoring"] = float(time.time() - step)

    pas, pdp, error, energy = candidate_metric
    base_pas, base_pdp, base_error, base_energy = baseline_metric
    v15_report = json.loads(e195.V15_REPORT.read_text(encoding="utf-8"))
    folds = []
    for outer in range(5):
        mask = fold_id == outer
        selected_fold = statistics(
            pas[mask], pdp[mask], error[mask], energy[mask]
        )
        baseline_fold = statistics(
            base_pas[mask], base_pdp[mask], base_error[mask], base_energy[mask]
        )
        v15_fold = next(
            row["selected"]
            for row in v15_report["folds"]
            if int(row["outer"]) == outer
        )
        folds.append(
            {
                "outer": outer,
                "selected": selected_fold,
                "strict_baseline": baseline_fold,
                "delta_vs_strict": e190.metric_delta(selected_fold, baseline_fold),
                "v15": v15_fold,
                "delta_vs_v15": e190.metric_delta(selected_fold, v15_fold),
            }
        )
    delta = e190.metric_delta(selected, baseline)
    report = {
        "experiment": VERSION,
        "status": "complete",
        "selected": selected,
        "baseline": baseline,
        "delta": delta,
        "target_delta": float(selected["score"] - TARGET_SCORE),
        "folds": folds,
        "gate": {
            "beats_v15": bool(float(selected["score"]) > V15_SCORE),
            "reaches_070": bool(float(selected["score"]) >= TARGET_SCORE),
            "positive_folds": int(
                sum(float(row["delta_vs_v15"]["score"]) > 0.0 for row in folds)
            ),
            "minimum_fold_delta": float(
                min(float(row["delta_vs_v15"]["score"]) for row in folds)
            ),
        },
        "identity": {
            "strict_e190_endpoint_source": "E195 canonical-strict fold endpoints",
            "v15_score_reference": V15_SCORE,
            "strict_baseline_score": float(baseline["score"]),
        },
        "data_lineage": {
            "official_train_h": PATHS.relative(TRAIN_CHANNEL),
            "external_propagation_state": [
                PATHS.relative(e193.TRAIN_BASE_EMBEDDING),
                PATHS.relative(e193.TRAIN_SCATTER_EMBEDDING),
            ],
            "historical_v15_channel_used_as_candidate_input": False,
            "historical_v15_channel_read": False,
            "e190_endpoint_source": "canonical-strict official-H-trained parameters",
            "hidden_official_test_h_used": False,
        },
        "output": {
            "path": PATHS.relative(FINAL_CHANNEL),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output.real).all() and np.isfinite(output.imag).all()),
        },
        "performance": {
            "device": torch.cuda.get_device_name(device),
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "phase_seconds": phase,
            "measured_bottleneck": max(phase, key=phase.get),
            "wall_seconds": float(time.time() - started),
        },
    }
    atomic_json(FULL_REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E199 requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")
    try:
        if args.stage == "prepare":
            report = prepare_static(device, args.force)
        elif args.stage == "fold":
            report = run_fold(args.outer, device, args.force)
        else:
            report = aggregate(device, args.force)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
