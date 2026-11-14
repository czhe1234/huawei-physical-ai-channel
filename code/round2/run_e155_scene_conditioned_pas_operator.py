#!/usr/bin/env python3
"""Train a scene-conditioned masked-hole PAS correction operator for V7."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from torch import nn

import analyze_e138_v12_guided_official_correction_student as e138
import analyze_e153_v7_v12_peak_actions as e153
import train_e88_dense_path_residual_student as e88
import train_surface_interaction_energy_model as e85
from analyze_e86_surface_endpoint_v11_blend import endpoint_target
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, solve_linear_rbf_weights, statistics
from run_v4_masked_hole_neural_process import CrossBlock
from run_v9_joint_sionna_migration import point_metrics
from train_offline_sionna_geometry_student import position_features


VERSION = "Round2-E155-scene-conditioned-pas-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "scene_conditioned_pas_operator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_scene_conditioned_pas_operator"
REPORT = ANALYSIS_DIR / "scene_conditioned_pas_operator.json"
METRICS = ARTIFACT_DIR / "scene_conditioned_pas_operator_metrics.npz"
V7_PAS_CACHE = ARTIFACT_DIR / "v7_compact_pas_f32.npy"

TRAIN_CHANNEL = e153.TRAIN_CHANNEL
V7_CHANNEL = e153.V7_CHANNEL
TEACHER_PAS = e138.TEACHER_PAS
OFFICIAL_PAS = e85.TARGET_PAS

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PAS_SIZE = MP * MH * MV * N_UE
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    neighbors: int = 24
    candidate_neighbors: int = 192
    guard_radii_m: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0)
    rbf_smoothing: float = 10.0
    context_donors_per_query: int = 8
    maximum_context_points: int = 192
    hole_sizes: tuple[int, ...] = (24, 40, 64, 80)
    spectrum_rank: int = 64
    correction_rank: int = 32
    hidden: int = 192
    heads: int = 6
    self_layers: int = 2
    cross_layers: int = 2
    dropout: float = 0.05
    coefficient_bound: float = 3.0
    steps: int = 600
    validation_interval: int = 50
    learning_rate: float = 1.5e-4
    weight_decay: float = 3e-5
    coefficient_weight: float = 0.08
    action_regularization: float = 0.003
    project_beta: float = 0.20
    project_cycles: int = 4
    seed: int = 260_810_155


@dataclass(frozen=True)
class ContextTable:
    query_ids: np.ndarray
    donor_ids: np.ndarray
    weights: np.ndarray
    distances: np.ndarray
    anchors_path: Path

    def anchors(self) -> np.ndarray:
        return np.load(self.anchors_path, mmap_mode="r")


@dataclass(frozen=True)
class SpectralPCA:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray


class SceneStore:
    def __init__(self) -> None:
        geometry_paths = e88.geometry_paths()
        setup, positions, _, site, _ = load_positions()
        normalization = np.load(geometry_paths["normalization"], allow_pickle=False)
        raw_position = position_features(positions, site, setup.bs_positions)
        raw_scalar = np.asarray(
            np.load(geometry_paths["train_scalar"], mmap_mode="r"), dtype=np.float32
        )
        self.position = np.clip(
            (raw_position - normalization["position_mean"])
            / normalization["position_std"],
            -8.0,
            8.0,
        ).astype(np.float32)
        self.scalar = np.clip(
            (raw_scalar - normalization["scalar_mean"])
            / normalization["scalar_std"],
            -8.0,
            8.0,
        ).astype(np.float32)
        self.polar = np.load(geometry_paths["train_polar"], mmap_mode="r")
        self.corridor = np.load(geometry_paths["train_corridor"], mmap_mode="r")
        self.positions = positions.astype(np.float32)
        self.site = site.astype(np.int8)

    def query_batch(
        self, ids: np.ndarray, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {
            "position": torch.as_tensor(self.position[ids], device=device),
            "scalar": torch.as_tensor(self.scalar[ids], device=device),
            "polar": torch.as_tensor(
                np.asarray(self.polar[ids], dtype=np.float32), device=device
            ),
            "corridor": torch.as_tensor(
                np.asarray(self.corridor[ids], dtype=np.float32), device=device
            ),
            "site": torch.as_tensor(self.site[ids].astype(np.int64), device=device),
        }

    def context_batch(
        self, ids: np.ndarray, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {
            "position": torch.as_tensor(self.position[ids], device=device),
            "scalar": torch.as_tensor(self.scalar[ids], device=device),
            "site": torch.as_tensor(self.site[ids].astype(np.int64), device=device),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--folds", default="all")
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument(
        "--validation-interval", type=int, default=Config.validation_interval
    )
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--force-context", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_probability(
    value: np.ndarray, axes: tuple[int, ...]
) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = output.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        output,
        np.maximum(total, TINY),
        out=np.zeros_like(output),
        where=total > 0.0,
    )


def normalize_amplitude_torch(value: torch.Tensor) -> torch.Tensor:
    shaped = torch.relu(value).reshape(-1, MP, MH, MV, N_UE)
    norm = torch.sqrt(
        shaped.square().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-20)
    )
    return (shaped / norm).reshape(len(value), PAS_SIZE)


def endpoint_cosine_torch(
    prediction_amplitude: torch.Tensor, truth_amplitude: torch.Tensor
) -> torch.Tensor:
    prediction = prediction_amplitude.reshape(-1, MP, MH, MV, N_UE).square()
    truth = truth_amplitude.reshape(-1, MP, MH, MV, N_UE).square()
    prediction = prediction.permute(0, 4, 1, 2, 3).reshape(-1, N_UE, MP * MH * MV)
    truth = truth.permute(0, 4, 1, 2, 3).reshape(-1, N_UE, MP * MH * MV)
    numerator = (prediction * truth).sum(dim=-1)
    denominator = torch.sqrt(
        prediction.square().sum(dim=-1) * truth.square().sum(dim=-1)
    )
    return numerator / torch.where(
        denominator == 0.0, torch.full_like(denominator, 1e-12), denominator
    )


def endpoint_cosine_numpy(
    prediction: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    left = np.asarray(prediction, dtype=np.float64).reshape(
        -1, MP, MH, MV, N_UE
    )
    right = np.asarray(truth, dtype=np.float64).reshape(
        -1, MP, MH, MV, N_UE
    )
    left = left.transpose(0, 4, 1, 2, 3)
    right = right.transpose(0, 4, 1, 2, 3)
    left = left.reshape(len(left), N_UE, -1)
    right = right.reshape(len(right), N_UE, -1)
    numerator = np.sum(left * right, axis=-1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1, dtype=np.float64)
        * np.sum(right * right, axis=-1, dtype=np.float64)
    )
    return np.divide(
        numerator,
        np.where(denominator == 0.0, 1e-12, denominator),
    ).mean(axis=1)


def load_pas_arrays(
    device: torch.device, metric_batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    official = normalize_probability(
        np.asarray(np.load(OFFICIAL_PAS, mmap_mode="r"), dtype=np.float32),
        (1, 2, 3),
    )
    if V7_PAS_CACHE.is_file():
        v7 = np.asarray(np.load(V7_PAS_CACHE, mmap_mode="r"), dtype=np.float32)
    else:
        v7, _, _ = e138.compact_channel_endpoints(
            np.load(V7_CHANNEL, mmap_mode="r"), device, metric_batch_size
        )
        np.save(V7_PAS_CACHE, v7.astype(np.float32))
    teacher_raw = np.load(TEACHER_PAS, mmap_mode="r")
    if teacher_raw.shape == (500, MP, MH, MV, N_UE, S_SC):
        teacher = np.asarray(teacher_raw, dtype=np.float32).mean(axis=-1)
    elif teacher_raw.shape == (500, MP, MH, MV, N_UE):
        teacher = np.asarray(teacher_raw, dtype=np.float32)
    else:
        raise ValueError(f"unexpected teacher PAS shape: {teacher_raw.shape}")
    teacher = normalize_probability(teacher, (1, 2, 3))
    return official, v7, teacher


def fit_spectral_pca(
    amplitude: np.ndarray, fit_ids: np.ndarray, rank: int, seed: int
) -> SpectralPCA:
    model = PCA(
        n_components=rank,
        svd_solver="randomized",
        random_state=seed % (2**32),
    ).fit(amplitude[fit_ids])
    scale = np.sqrt(np.maximum(model.explained_variance_, 1e-8)).astype(np.float32)
    coefficients = (
        model.transform(amplitude).astype(np.float32) / scale[None]
    )
    return SpectralPCA(
        mean=model.mean_.astype(np.float32),
        components=model.components_.astype(np.float32),
        scale=scale,
        coefficients=coefficients,
    )


def fit_correction_basis(
    delta: np.ndarray, rank: int
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(delta, dtype=np.float64).reshape(len(delta), -1)
    _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, len(vectors))
    total = float(np.sum(singular * singular, dtype=np.float64))
    explained = float(
        np.sum(singular[:keep] * singular[:keep], dtype=np.float64)
        / max(total, TINY)
    )
    return vectors[:keep].astype(np.float32), explained


def context_paths(outer: int, inner: int, role: str) -> tuple[Path, Path]:
    prefix = ARTIFACT_DIR / f"outer{outer}_inner{inner}_{role}"
    return prefix.with_name(prefix.name + "_context.npz"), prefix.with_name(
        prefix.name + "_anchors_f16.npy"
    )


def build_context_table(
    outer: int,
    inner: int,
    role: str,
    query_ids: np.ndarray,
    donor_pool: np.ndarray,
    store: SceneStore,
    truth_amplitude: np.ndarray,
    config: Config,
    force: bool,
) -> ContextTable:
    cache_path, anchors_path = context_paths(outer, inner, role)
    if not force and cache_path.is_file() and anchors_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            np.array_equal(cached["query_ids"], query_ids)
            and np.array_equal(cached["guard_radii_m"], config.guard_radii_m)
            and int(cached["neighbors"]) == config.neighbors
        ):
            return ContextTable(
                query_ids=query_ids,
                donor_ids=cached["donor_ids"].astype(np.int64),
                weights=cached["weights"].astype(np.float32),
                distances=cached["distances"].astype(np.float32),
                anchors_path=anchors_path,
            )

    guards = np.asarray(config.guard_radii_m, dtype=np.float32)
    shape = (len(query_ids), len(guards), config.neighbors)
    donor_ids = np.empty(shape, dtype=np.int32)
    weights = np.empty(shape, dtype=np.float32)
    distances = np.empty(shape, dtype=np.float32)
    for site_value in np.unique(store.site[query_ids]):
        rows = np.flatnonzero(store.site[query_ids] == site_value)
        bank = donor_pool[store.site[donor_pool] == site_value]
        count = min(config.candidate_neighbors, len(bank))
        candidate_distance, local = cKDTree(store.positions[bank, :2]).query(
            store.positions[query_ids[rows], :2], k=count, workers=-1
        )
        candidate_distance = np.asarray(candidate_distance).reshape(len(rows), -1)
        candidate_ids = bank[np.asarray(local).reshape(len(rows), -1)]
        for local_row, output_row in enumerate(rows):
            query_id = int(query_ids[output_row])
            candidates = candidate_ids[local_row]
            physical_distance = candidate_distance[local_row]
            scene_distance = np.sqrt(
                np.mean(
                    (
                        store.scalar[candidates]
                        - store.scalar[query_id][None]
                    )
                    ** 2,
                    axis=1,
                    dtype=np.float64,
                )
            )
            cost = physical_distance * (1.0 + 0.12 * np.clip(scene_distance, 0.0, 4.0))
            for guard_index, guard in enumerate(guards):
                keep = (candidates != query_id) & (physical_distance >= guard)
                order = np.flatnonzero(keep)[np.argsort(cost[keep])]
                selected = order[: config.neighbors]
                if len(selected) != config.neighbors:
                    raise RuntimeError(
                        f"query {query_id} guard {guard:g} has {len(selected)} donors"
                    )
                current_ids = candidates[selected]
                current_distance = physical_distance[selected]
                donor_ids[output_row, guard_index] = current_ids
                distances[output_row, guard_index] = current_distance
                weights[output_row, guard_index] = solve_linear_rbf_weights(
                    store.positions[current_ids, :2],
                    store.positions[query_id, :2],
                    config.rbf_smoothing,
                )
            if (local_row + 1) % 400 == 0 or local_row + 1 == len(rows):
                print(
                    f"E155 context outer={outer} role={role} site={int(site_value)} "
                    f"{local_row + 1}/{len(rows)}",
                    flush=True,
                )

    np.savez_compressed(
        cache_path,
        query_ids=query_ids.astype(np.int32),
        donor_ids=donor_ids,
        weights=weights,
        distances=distances,
        guard_radii_m=guards,
        neighbors=np.int32(config.neighbors),
    )
    anchor_memmap = np.lib.format.open_memmap(
        anchors_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(query_ids), len(guards), MP, MH, MV, N_UE),
    )
    flat_truth = truth_amplitude.reshape(len(truth_amplitude), PAS_SIZE)
    for start in range(0, len(query_ids), 64):
        stop = min(start + 64, len(query_ids))
        selected = flat_truth[donor_ids[start:stop]]
        anchor = np.einsum(
            "qgk,qgkd->qgd", weights[start:stop], selected, optimize=True
        )
        anchor = np.maximum(anchor, 0.0).reshape(
            stop - start, len(guards), MP, MH, MV, N_UE
        )
        norm = np.sqrt(
            np.sum(anchor * anchor, axis=(2, 3, 4), keepdims=True, dtype=np.float64)
        )
        anchor = np.divide(
            anchor,
            np.maximum(norm, 1e-20),
            out=np.zeros_like(anchor),
            where=norm > 0.0,
        )
        anchor_memmap[start:stop] = anchor.astype(np.float16)
    anchor_memmap.flush()
    return ContextTable(query_ids, donor_ids, weights, distances, anchors_path)


class EpisodeBuilder:
    def __init__(
        self,
        table: ContextTable,
        positions: np.ndarray,
        site: np.ndarray,
        hole_sizes: tuple[int, ...],
        seed: int,
    ) -> None:
        self.table = table
        self.positions = positions
        self.site = site
        self.hole_sizes = hole_sizes
        self.rng = np.random.default_rng(seed)
        self.site_rows = {
            int(value): np.flatnonzero(site[table.query_ids] == value)
            for value in np.unique(site[table.query_ids])
        }
        self.site_trees = {
            value: cKDTree(positions[table.query_ids[rows], :2])
            for value, rows in self.site_rows.items()
        }

    def sample(self) -> np.ndarray:
        site_value = int(self.rng.choice(list(self.site_rows)))
        rows = self.site_rows[site_value]
        size = min(int(self.rng.choice(self.hole_sizes)), len(rows))
        center = int(self.rng.integers(0, len(rows)))
        local = np.asarray(
            self.site_trees[site_value].query(
                self.positions[self.table.query_ids[rows[center]], :2], k=size
            )[1]
        ).reshape(-1)
        return rows[local]


def context_ids_for_rows(
    table: ContextTable,
    rows: np.ndarray,
    positions: np.ndarray,
    maximum: int,
    donors_per_query: int,
) -> np.ndarray:
    query_ids = table.query_ids[rows]
    candidates = np.unique(table.donor_ids[rows, :, :donors_per_query])
    candidates = candidates[~np.isin(candidates, query_ids)]
    if len(candidates) <= maximum:
        return candidates.astype(np.int64)
    distance = cKDTree(positions[query_ids, :2]).query(
        positions[candidates, :2], k=1, workers=-1
    )[0]
    return candidates[np.argsort(distance)[:maximum]].astype(np.int64)


def conv_encoder(output: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(4, 48, 3, padding=1),
        nn.GroupNorm(8, 48),
        nn.SiLU(),
        nn.Conv2d(48, 64, 3, stride=(1, 2), padding=1),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv2d(64, 96, 3, stride=(2, 2), padding=1),
        nn.GroupNorm(8, 96),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d((2, 4)),
        nn.Flatten(),
        nn.Linear(96 * 8, output),
        nn.SiLU(),
    )


class SceneConditionedPASOperator(nn.Module):
    def __init__(
        self,
        position_size: int,
        scalar_size: int,
        spectrum_rank: int,
        correction_rank: int,
        guard_count: int,
        config: Config,
    ) -> None:
        super().__init__()
        branch = config.hidden // 3
        self.config = config
        self.query_scalar = nn.Sequential(
            nn.Linear(position_size + scalar_size, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, branch),
            nn.SiLU(),
        )
        self.polar = conv_encoder(branch)
        self.corridor = conv_encoder(branch)
        self.scene_fusion = nn.Sequential(
            nn.Linear(branch * 3, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.main_spectrum = nn.Sequential(
            nn.Linear(spectrum_rank, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.guard_embedding = nn.Parameter(
            torch.randn(guard_count, config.hidden) / math.sqrt(config.hidden)
        )
        self.alternative_spectrum = nn.Sequential(
            nn.Linear(spectrum_rank, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        alternative_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=config.hidden * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.alternative_encoder = nn.TransformerEncoder(
            alternative_layer, num_layers=1, enable_nested_tensor=False
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(config.hidden * 3, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.context_scene = nn.Sequential(
            nn.Linear(position_size + scalar_size, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.context_spectrum = nn.Sequential(
            nn.Linear(spectrum_rank, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(config.hidden * 2, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        query_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=config.hidden * 3,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_self = nn.TransformerEncoder(
            query_layer,
            num_layers=config.self_layers,
            enable_nested_tensor=False,
        )
        self.cross = nn.ModuleList(
            [
                CrossBlock(config.hidden, config.heads, config.dropout)
                for _ in range(config.cross_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, correction_rank),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        query: dict[str, torch.Tensor],
        main_coefficients: torch.Tensor,
        alternative_coefficients: torch.Tensor,
        context: dict[str, torch.Tensor],
        context_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        scene = self.scene_fusion(
            torch.cat(
                (
                    self.query_scalar(
                        torch.cat((query["position"], query["scalar"]), dim=1)
                    ),
                    self.polar(query["polar"]),
                    self.corridor(query["corridor"]),
                ),
                dim=1,
            )
        )
        alternative = self.alternative_spectrum(alternative_coefficients)
        alternative = alternative + self.guard_embedding[None]
        alternative = self.alternative_encoder(alternative).mean(dim=1)
        token = self.query_fusion(
            torch.cat(
                (scene, self.main_spectrum(main_coefficients), alternative), dim=1
            )
        )[None]
        token = self.query_self(token)
        context_token = self.context_fusion(
            torch.cat(
                (
                    self.context_scene(
                        torch.cat((context["position"], context["scalar"]), dim=1)
                    ),
                    self.context_spectrum(context_coefficients),
                ),
                dim=1,
            )
        )[None]
        for block in self.cross:
            token = block(token, context_token)
        return self.config.coefficient_bound * torch.tanh(self.output(token[0]))


def encode_spectrum_torch(
    amplitude: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return ((amplitude - mean) @ components.T) / scale


def reconstruct_amplitude_torch(
    anchor: torch.Tensor,
    standardized_coefficients: torch.Tensor,
    basis: torch.Tensor,
    coefficient_scale: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    correction = standardized_coefficients * coefficient_scale
    return normalize_amplitude_torch(anchor + float(alpha) * (correction @ basis))


def run_model_rows(
    model: SceneConditionedPASOperator,
    rows: np.ndarray,
    table: ContextTable,
    store: SceneStore,
    spectral: SpectralPCA,
    main_amplitude: np.ndarray | torch.Tensor,
    device: torch.device,
    config: Config,
    alternative_amplitude: torch.Tensor | None = None,
) -> torch.Tensor:
    query_ids = table.query_ids[rows]
    context_ids = context_ids_for_rows(
        table,
        rows,
        store.positions,
        config.maximum_context_points,
        config.context_donors_per_query,
    )
    if alternative_amplitude is None:
        alternative = torch.as_tensor(
            np.asarray(table.anchors()[rows], dtype=np.float32).reshape(
                len(rows), len(config.guard_radii_m), PAS_SIZE
            ),
            device=device,
        )
    else:
        alternative = alternative_amplitude.reshape(
            len(rows), len(config.guard_radii_m), PAS_SIZE
        ).to(device)
    if isinstance(main_amplitude, torch.Tensor):
        main = main_amplitude.reshape(len(rows), PAS_SIZE).to(device)
    else:
        main = torch.as_tensor(main_amplitude, device=device)
    mean = torch.as_tensor(spectral.mean, device=device)
    components = torch.as_tensor(spectral.components, device=device)
    scale = torch.as_tensor(spectral.scale, device=device)
    main_coefficients = encode_spectrum_torch(main, mean, components, scale)
    alternative_coefficients = encode_spectrum_torch(
        alternative, mean, components, scale
    )
    context_coefficients = torch.as_tensor(
        spectral.coefficients[context_ids], device=device
    )
    return model(
        store.query_batch(query_ids, device),
        main_coefficients,
        alternative_coefficients,
        store.context_batch(context_ids, device),
        context_coefficients,
    )


def dynamic_episode_anchors(
    table: ContextTable,
    rows: np.ndarray,
    truth_amplitude: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    query_ids = table.query_ids[rows]
    donor_ids = table.donor_ids[rows]
    weights = np.array(table.weights[rows], dtype=np.float32, copy=True)
    overlap = np.isin(donor_ids, query_ids)
    weights[overlap] = 0.0
    total = weights.sum(axis=-1, keepdims=True)
    bad = np.abs(total[..., 0]) < 0.05
    weights = np.divide(
        weights,
        np.where(np.abs(total) < 1e-8, 1.0, total),
        out=np.zeros_like(weights),
    )
    if np.any(bad):
        inverse = 1.0 / np.maximum(table.distances[rows], 1e-3) ** 2
        inverse[overlap] = 0.0
        inverse /= np.maximum(inverse.sum(axis=-1, keepdims=True), 1e-8)
        weights[bad] = inverse[bad]
    donor_ids_t = torch.as_tensor(donor_ids, dtype=torch.long, device=device)
    weights_t = torch.as_tensor(weights, device=device)
    selected = truth_amplitude[donor_ids_t].reshape(
        len(rows), donor_ids.shape[1], donor_ids.shape[2], PAS_SIZE
    )
    anchor = torch.einsum("qgk,qgkd->qgd", weights_t, selected)
    return normalize_amplitude_torch(
        anchor.reshape(len(rows) * donor_ids.shape[1], PAS_SIZE)
    ).reshape(len(rows), donor_ids.shape[1], PAS_SIZE)


@torch.inference_mode()
def predict_table(
    model: SceneConditionedPASOperator,
    table: ContextTable,
    store: SceneStore,
    spectral: SpectralPCA,
    main_amplitude: np.ndarray,
    device: torch.device,
    config: Config,
) -> np.ndarray:
    model.eval()
    output = np.empty(
        (len(table.query_ids), config.correction_rank), dtype=np.float32
    )
    for site_value in np.unique(store.site[table.query_ids]):
        rows = np.flatnonzero(store.site[table.query_ids] == site_value)
        output[rows] = (
            run_model_rows(
                model,
                rows,
                table,
                store,
                spectral,
                main_amplitude[rows],
                device,
                config,
            )
            .float()
            .cpu()
            .numpy()
        )
    return output


def reconstruct_amplitude_numpy(
    anchor: np.ndarray,
    standardized_coefficients: np.ndarray,
    basis: np.ndarray,
    coefficient_scale: np.ndarray,
    alpha: float,
) -> np.ndarray:
    correction = standardized_coefficients * coefficient_scale[None]
    amplitude = np.maximum(
        anchor.reshape(len(anchor), -1) + float(alpha) * correction @ basis, 0.0
    ).reshape(len(anchor), MP, MH, MV, N_UE)
    norm = np.sqrt(
        np.sum(amplitude * amplitude, axis=(1, 2, 3), keepdims=True, dtype=np.float64)
    )
    return np.divide(
        amplitude,
        np.maximum(norm, 1e-20),
        out=np.zeros_like(amplitude),
        where=norm > 0.0,
    )


def validation_summary(
    prediction: np.ndarray,
    anchor: np.ndarray,
    truth_probability: np.ndarray,
    basis: np.ndarray,
    coefficient_scale: np.ndarray,
) -> dict[str, object]:
    alpha_grid = (0.0, 0.25, 0.50, 0.75, 1.0)
    rows = []
    baseline = endpoint_cosine_numpy(np.square(anchor), truth_probability)
    for alpha in alpha_grid:
        amplitude = reconstruct_amplitude_numpy(
            anchor, prediction, basis, coefficient_scale, alpha
        )
        cosine = endpoint_cosine_numpy(np.square(amplitude), truth_probability)
        rows.append(
            {
                "alpha": float(alpha),
                "pas": float(np.mean(cosine)),
                "delta": float(np.mean(cosine - baseline)),
                "positive_fraction": float(np.mean(cosine >= baseline)),
            }
        )
    best = max(rows, key=lambda item: item["pas"])
    return {
        "baseline_pas": float(np.mean(baseline)),
        "rows": rows,
        "best": best,
    }


def train_fold(
    outer: int,
    config: Config,
    store: SceneStore,
    official_probability: np.ndarray,
    truth_amplitude: np.ndarray,
    v7_probability: np.ndarray,
    teacher_probability: np.ndarray,
    fixed_query_ids: np.ndarray,
    fixed_fold_id: np.ndarray,
    device: torch.device,
    force_context: bool,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    started = time.time()
    inner = (outer + 1) % 5
    set_seed(config.seed + outer * 1009)
    outer_rows = np.flatnonzero(fixed_fold_id == outer)
    inner_rows = np.flatnonzero(fixed_fold_id == inner)
    excluded_train = np.concatenate(
        (fixed_query_ids[outer_rows], fixed_query_ids[inner_rows])
    )
    all_ids = np.arange(len(official_probability), dtype=np.int64)
    nonzero = official_probability.reshape(len(official_probability), -1).sum(axis=1) > 0
    train_pool = all_ids[~np.isin(all_ids, excluded_train) & nonzero]
    inner_query_ids = fixed_query_ids[inner_rows]
    outer_query_ids = fixed_query_ids[outer_rows]
    outer_donor_pool = all_ids[
        ~np.isin(all_ids, outer_query_ids) & nonzero
    ]

    train_table = build_context_table(
        outer,
        inner,
        "train",
        train_pool,
        train_pool,
        store,
        truth_amplitude,
        config,
        force_context,
    )
    inner_table = build_context_table(
        outer,
        inner,
        "inner",
        inner_query_ids,
        train_pool,
        store,
        truth_amplitude,
        config,
        force_context,
    )
    outer_table = build_context_table(
        outer,
        inner,
        "outer",
        outer_query_ids,
        outer_donor_pool,
        store,
        truth_amplitude,
        config,
        force_context,
    )

    spectral = fit_spectral_pca(
        truth_amplitude.reshape(len(truth_amplitude), PAS_SIZE),
        train_pool,
        config.spectrum_rank,
        config.seed + outer,
    )
    basis_rows = np.flatnonzero(
        (fixed_fold_id != outer) & (fixed_fold_id != inner)
    )
    teacher_delta = np.sqrt(np.maximum(teacher_probability[basis_rows], 0.0)) - np.sqrt(
        np.maximum(v7_probability[basis_rows], 0.0)
    )
    basis, basis_explained = fit_correction_basis(
        teacher_delta, config.correction_rank
    )
    training_anchor = np.asarray(
        train_table.anchors()[:, 0], dtype=np.float32
    ).reshape(len(train_pool), PAS_SIZE)
    training_delta = truth_amplitude[train_pool].reshape(len(train_pool), PAS_SIZE) - training_anchor
    training_coefficients = training_delta @ basis.T
    coefficient_scale = np.std(training_coefficients, axis=0, dtype=np.float64).astype(
        np.float32
    )
    coefficient_scale = np.maximum(coefficient_scale, 2e-3)

    model = SceneConditionedPASOperator(
        store.position.shape[1],
        store.scalar.shape[1],
        config.spectrum_rank,
        config.correction_rank,
        len(config.guard_radii_m),
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    basis_t = torch.as_tensor(basis, device=device)
    coefficient_scale_t = torch.as_tensor(coefficient_scale, device=device)
    truth_amplitude_t = torch.as_tensor(truth_amplitude, device=device)
    builder = EpisodeBuilder(
        train_table,
        store.positions,
        store.site,
        config.hole_sizes,
        config.seed + outer * 2003,
    )
    inner_anchor = np.asarray(
        inner_table.anchors()[:, 0], dtype=np.float32
    ).reshape(len(inner_query_ids), PAS_SIZE)
    inner_truth = official_probability[inner_query_ids]
    initial_prediction = predict_table(
        model,
        inner_table,
        store,
        spectral,
        inner_anchor,
        device,
        config,
    )
    initial_validation = validation_summary(
        initial_prediction,
        inner_anchor,
        inner_truth,
        basis,
        coefficient_scale,
    )
    best_score = float(initial_validation["best"]["pas"])
    best_alpha = float(initial_validation["best"]["alpha"])
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = [
        {"step": 0, "validation": initial_validation}
    ]
    print(
        f"E155 outer={outer} step=0 inner_pas={best_score:.6f} "
        f"alpha={best_alpha:.2f}",
        flush=True,
    )

    rolling: list[tuple[float, float, float, float]] = []
    guard_choices = np.arange(min(2, len(config.guard_radii_m)))
    for step in range(1, config.steps + 1):
        model.train()
        rows = builder.sample()
        main_guard = int(builder.rng.choice(guard_choices))
        episode_anchors = dynamic_episode_anchors(
            train_table, rows, truth_amplitude_t, device
        )
        main_anchor = episode_anchors[:, main_guard]
        prediction = run_model_rows(
            model,
            rows,
            train_table,
            store,
            spectral,
            main_anchor,
            device,
            config,
            alternative_amplitude=episode_anchors,
        )
        query_ids = train_table.query_ids[rows]
        truth = truth_amplitude_t[query_ids].reshape(len(rows), PAS_SIZE)
        target_coefficients = ((truth - main_anchor) @ basis_t.T) / coefficient_scale_t
        predicted_amplitude = reconstruct_amplitude_torch(
            main_anchor,
            prediction,
            basis_t,
            coefficient_scale_t,
            1.0,
        )
        cosine = endpoint_cosine_torch(predicted_amplitude, truth).mean()
        coefficient_loss = torch.nn.functional.smooth_l1_loss(
            prediction, target_coefficients
        )
        action = prediction.square().mean()
        loss = (
            1.0
            - cosine
            + config.coefficient_weight * coefficient_loss
            + config.action_regularization * action
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()
        rolling.append(
            (
                float(loss.detach().cpu()),
                float(cosine.detach().cpu()),
                float(coefficient_loss.detach().cpu()),
                float(action.detach().sqrt().cpu()),
            )
        )
        if step % config.validation_interval == 0 or step == config.steps:
            prediction_inner = predict_table(
                model,
                inner_table,
                store,
                spectral,
                inner_anchor,
                device,
                config,
            )
            validation = validation_summary(
                prediction_inner,
                inner_anchor,
                inner_truth,
                basis,
                coefficient_scale,
            )
            train_summary = {
                "loss": float(np.mean([item[0] for item in rolling])),
                "pas": float(np.mean([item[1] for item in rolling])),
                "coefficient_loss": float(np.mean([item[2] for item in rolling])),
                "action_rms": float(np.mean([item[3] for item in rolling])),
            }
            history.append(
                {"step": step, "train": train_summary, "validation": validation}
            )
            current = float(validation["best"]["pas"])
            if current > best_score:
                best_score = current
                best_alpha = float(validation["best"]["alpha"])
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E155 outer={outer} step={step}/{config.steps} "
                f"train_pas={train_summary['pas']:.6f} "
                f"inner_pas={current:.6f} alpha={validation['best']['alpha']:.2f} "
                f"best={best_score:.6f}@{best_step}",
                flush=True,
            )
            rolling.clear()

    model.load_state_dict(best_state)
    outer_main = np.sqrt(np.maximum(v7_probability[outer_rows], 0.0)).reshape(
        len(outer_rows), PAS_SIZE
    )
    outer_prediction = predict_table(
        model,
        outer_table,
        store,
        spectral,
        outer_main,
        device,
        config,
    )
    outer_target_amplitude = reconstruct_amplitude_numpy(
        outer_main,
        outer_prediction,
        basis,
        coefficient_scale,
        best_alpha,
    )
    outer_target = np.square(outer_target_amplitude)
    outer_truth = official_probability[outer_query_ids]
    endpoint_baseline = endpoint_cosine_numpy(
        np.square(outer_main.reshape(-1, MP, MH, MV, N_UE)), outer_truth
    )
    endpoint_prediction = endpoint_cosine_numpy(outer_target, outer_truth)

    oracle_coefficients = (
        truth_amplitude[outer_query_ids].reshape(len(outer_query_ids), PAS_SIZE)
        - outer_main
    ) @ basis.T
    oracle_standardized = oracle_coefficients / coefficient_scale[None]
    oracle_target = reconstruct_amplitude_numpy(
        outer_main,
        oracle_standardized,
        basis,
        coefficient_scale,
        1.0,
    ) ** 2

    checkpoint = ARTIFACT_DIR / f"outer{outer}_scene_pas_operator.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": asdict(config),
            "outer": outer,
            "inner": inner,
            "best_step": best_step,
            "best_alpha": best_alpha,
            "spectral_mean": spectral.mean,
            "spectral_components": spectral.components,
            "spectral_scale": spectral.scale,
            "correction_basis": basis,
            "coefficient_scale": coefficient_scale,
        },
        checkpoint,
    )
    record = {
        "outer": outer,
        "inner": inner,
        "train_count": int(len(train_pool)),
        "inner_count": int(len(inner_query_ids)),
        "outer_count": int(len(outer_query_ids)),
        "teacher_basis_explained_energy": basis_explained,
        "best_step": best_step,
        "best_alpha": best_alpha,
        "best_inner_pas": best_score,
        "outer_endpoint": {
            "baseline_pas": float(np.mean(endpoint_baseline)),
            "prediction_pas": float(np.mean(endpoint_prediction)),
            "delta": float(np.mean(endpoint_prediction - endpoint_baseline)),
            "positive_fraction": float(
                np.mean(endpoint_prediction >= endpoint_baseline)
            ),
        },
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
        "seconds": float(time.time() - started),
    }
    return record, outer_target.astype(np.float32), oracle_target.astype(np.float32)


def evaluate_complete(
    endpoint_pas: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_all = np.load(V7_CHANNEL, mmap_mode="r")
    arrays = tuple(np.empty(len(query_ids), dtype=np.float64) for _ in range(4))
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(base_all[start:stop], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:stop]], dtype=np.complex64, copy=True),
            device=device,
        )
        compact = torch.as_tensor(endpoint_pas[start:stop], device=device)
        full_endpoint = compact[..., None].expand(-1, -1, -1, -1, -1, S_SC)
        with torch.inference_mode():
            base_pas = pas_distribution(base)
            desired_pas = endpoint_target(
                base_pas,
                full_endpoint,
                config.project_beta,
                "sqrt",
                (1, 2, 3),
            )
            reference_power = base.abs().square().sum(dim=(1, 2, 3))
            prediction = base
            for _ in range(config.project_cycles):
                prediction = exact_pas_projection(prediction, desired_pas)
                prediction = preserve_point_power(prediction, reference_power)
            values = point_metrics(
                prediction,
                truth,
                pas_distribution(truth),
                pdp_distribution(truth),
            )
        for destination, source in zip(arrays, values, strict=True):
            destination[start:stop] = source
        if stop % 100 == 0 or stop == len(query_ids):
            print(f"E155 complete score {stop}/{len(query_ids)}", flush=True)
    return arrays


def summarize_complete(
    values: tuple[np.ndarray, ...],
    baseline: tuple[np.ndarray, ...],
    fold_id: np.ndarray,
) -> dict[str, object]:
    overall = statistics(*values)
    base_overall = statistics(*baseline)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        current = statistics(*values, mask)
        base = statistics(*baseline, mask)
        folds.append(
            {
                "fold": fold,
                "metrics": current,
                "baseline": base,
                "delta": float(current["score"] - base["score"]),
            }
        )
    return {
        "metrics": overall,
        "baseline": base_overall,
        "delta_to_v7": float(overall["score"] - base_overall["score"]),
        "positive_folds": int(sum(item["delta"] > 0.0 for item in folds)),
        "worst_fold_delta": float(min(item["delta"] for item in folds)),
        "folds": folds,
    }


def finite_payload(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_payload(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_payload(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(
        steps=args.steps,
        validation_interval=args.validation_interval,
    )
    device = torch.device(args.device)
    if args.folds == "all":
        selected_folds = list(range(5))
    else:
        selected_folds = [int(value) for value in args.folds.split(",")]
    if any(value not in range(5) for value in selected_folds):
        raise ValueError(selected_folds)

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    store = SceneStore()
    official_probability, v7_probability, teacher_probability = load_pas_arrays(
        device, args.metric_batch_size
    )
    truth_amplitude = np.sqrt(np.maximum(official_probability, 0.0)).astype(
        np.float32
    )
    predicted_pas = np.array(v7_probability, dtype=np.float32, copy=True)
    oracle_pas = np.array(v7_probability, dtype=np.float32, copy=True)
    records = []
    for outer in selected_folds:
        record, outer_target, outer_oracle = train_fold(
            outer,
            config,
            store,
            official_probability,
            truth_amplitude,
            v7_probability,
            teacher_probability,
            query_ids,
            fold_id,
            device,
            args.force_context,
        )
        rows = np.flatnonzero(fold_id == outer)
        predicted_pas[rows] = outer_target
        oracle_pas[rows] = outer_oracle
        records.append(record)

    payload: dict[str, object] = {
        "experiment": VERSION,
        "status": "partial_screen" if len(selected_folds) < 5 else "strict_screen_complete",
        "official_inputs_only_at_inference": True,
        "test_sionna_used": False,
        "v12_role": "outer-fit PAS correction basis only",
        "primary_hypothesis": (
            "Whole-hole query relations, observed boundary PAS, and structured map "
            "tensors can predict low-rank V7 PAS corrections without explicit paths."
        ),
        "device": str(device),
        "selected_folds": selected_folds,
        "config": asdict(config),
        "fold_training": records,
    }

    if len(selected_folds) == 5:
        truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
        baseline_values = e138.baseline_metrics(
            V7_CHANNEL, truth_all, query_ids, device, args.metric_batch_size
        )
        prediction_values = evaluate_complete(
            predicted_pas,
            truth_all,
            query_ids,
            device,
            args.metric_batch_size,
            config,
        )
        oracle_values = evaluate_complete(
            oracle_pas,
            truth_all,
            query_ids,
            device,
            args.metric_batch_size,
            config,
        )
        result = summarize_complete(prediction_values, baseline_values, fold_id)
        oracle = summarize_complete(oracle_values, baseline_values, fold_id)
        passed = bool(
            result["delta_to_v7"] >= 0.005
            and result["positive_folds"] == 5
        )
        payload.update(
            {
                "complete_result": result,
                "teacher_basis_oracle": oracle,
                "gate": {
                    "minimum_score_gain": 0.005,
                    "minimum_positive_folds": 5,
                    "actual_score_gain": result["delta_to_v7"],
                    "actual_positive_folds": result["positive_folds"],
                    "passed": passed,
                },
                "decision": {
                    "passed": passed,
                    "expand_pdp_nmse": passed,
                    "generate_candidate": False,
                },
            }
        )
        np.savez_compressed(
            METRICS,
            query_ids=query_ids.astype(np.int32),
            fold_id=fold_id,
            predicted_pas=predicted_pas.astype(np.float16),
            oracle_pas=oracle_pas.astype(np.float16),
            prediction_pas=prediction_values[0],
            prediction_pdp=prediction_values[1],
            prediction_error=prediction_values[2],
            prediction_energy=prediction_values[3],
            baseline_pas=baseline_values[0],
            baseline_pdp=baseline_values[1],
            baseline_error=baseline_values[2],
            baseline_energy=baseline_values[3],
            oracle_metric_pas=oracle_values[0],
            oracle_metric_pdp=oracle_values[1],
            oracle_metric_error=oracle_values[2],
            oracle_metric_energy=oracle_values[3],
        )
        payload["outputs"] = {
            "report": PATHS.relative(REPORT),
            "metrics": PATHS.relative(METRICS),
        }
    payload["seconds"] = float(time.time() - started)
    if not finite_payload(payload):
        raise ValueError("non-finite E155 report")
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
