#!/usr/bin/env python3
"""Official-only 2.5D barrier-aware neural spectrum-bank attention."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 274
VERSION = "v274_planar_barrier_attention"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v274_planar_attention_targets.npz"
OUTPUT = OUT_DIR / "v274_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
FOOTPRINT_CACHE = PATHS.cache_dir(272) / "v272_extruded_footprint.npz"
SIGNATURE_CACHE = PATHS.cache_dir(273) / "v273_planar_barrier_features.npz"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (274, 1274, 2274, 3274, 4274)
OUTER_COUNT = 320
INNER_COUNT = 240
K_NEIGHBORS = 32
EPOCHS = 80
EVAL_EVERY = 5
BATCH_SIZE = 64
MIN_OFFICIAL_DELTA = 5e-4
FINAL_SEEDS = (27401, 27431, 27461)
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


@dataclass
class RawSet:
    query_ids: np.ndarray
    neighbor_ids: np.ndarray
    edge: np.ndarray
    base_weights: np.ndarray


@dataclass
class ModelSet:
    query_ids: np.ndarray
    query_node: np.ndarray
    neighbor_ids: np.ndarray
    edge: np.ndarray
    base_weights: np.ndarray


@dataclass
class Standardizer:
    node_mean: np.ndarray
    node_std: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray


def build_node_features(
    all_pos: np.ndarray,
    signatures: np.ndarray,
    blocker_count: np.ndarray,
    scene_features: np.ndarray,
) -> np.ndarray:
    delta = all_pos[:, :2] - np.asarray([50.0, 0.0], dtype=np.float64)[None, :]
    distance = np.linalg.norm(delta, axis=1)
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    base = np.stack(
        [
            all_pos[:, 0] / 400.0,
            all_pos[:, 1] / 350.0,
            distance / 450.0,
            np.log1p(distance) / 7.0,
            np.sin(angle),
            np.cos(angle),
            blocker_count.astype(np.float64) / 4.0,
            (blocker_count == 0).astype(np.float64),
        ],
        axis=1,
    )
    return np.nan_to_num(
        np.concatenate(
            [base, signatures.astype(np.float32), scene_features.astype(np.float32)],
            axis=1,
        ),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


def build_raw_set(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    all_pos: np.ndarray,
    signatures: np.ndarray,
    blocker_count: np.ndarray,
    footprint: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
) -> RawSet:
    pair = v273.pair_features(
        query_ids,
        bank_ids,
        all_pos,
        signatures,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    selected_columns = np.zeros((len(query_ids), K_NEIGHBORS), dtype=np.int64)
    for row, query_id in enumerate(query_ids):
        candidates = np.flatnonzero(pair.indices[row] != query_id)
        if len(candidates) < K_NEIGHBORS:
            raise RuntimeError("insufficient non-self planar neighbors")
        selected_columns[row] = candidates[:K_NEIGHBORS]

    def take(array: np.ndarray) -> np.ndarray:
        return np.take_along_axis(array, selected_columns, axis=1)

    neighbor_ids = take(pair.indices)
    distance = take(pair.distance)
    segment_barrier = take(pair.segment_barrier)
    segment_height = take(pair.segment_height)
    signature_distance = take(pair.signature_distance)
    delta = all_pos[neighbor_ids, :2] - all_pos[query_ids, None, :2]
    q_block = blocker_count[query_ids, None]
    n_block = blocker_count[neighbor_ids]
    q_angle = np.arctan2(
        all_pos[query_ids, 1], all_pos[query_ids, 0] - 50.0
    )[:, None]
    n_angle = np.arctan2(
        all_pos[neighbor_ids, 1], all_pos[neighbor_ids, 0] - 50.0
    )
    angle_delta = np.arctan2(np.sin(n_angle - q_angle), np.cos(n_angle - q_angle))
    edge = np.stack(
        [
            delta[..., 0] / 50.0,
            delta[..., 1] / 50.0,
            distance / 40.0,
            np.log1p(distance) / 5.0,
            segment_barrier / 4.0,
            segment_height / 30.0,
            signature_distance,
            np.broadcast_to(q_block, n_block.shape) / 4.0,
            n_block / 4.0,
            np.abs(n_block - q_block) / 4.0,
            (signature_distance == 0.0).astype(np.float32),
            np.sin(angle_delta),
            np.cos(angle_delta),
        ],
        axis=2,
    ).astype(np.float32)
    base_weights = 1.0 / np.maximum(distance, 0.05) ** 2
    base_weights /= np.maximum(base_weights.sum(axis=1, keepdims=True), 1e-20)
    return RawSet(
        query_ids=query_ids.astype(np.int64),
        neighbor_ids=neighbor_ids.astype(np.int64),
        edge=np.nan_to_num(edge, nan=0.0, posinf=8.0, neginf=-8.0),
        base_weights=base_weights.astype(np.float32),
    )


def fit_standardizer(
    node_raw: np.ndarray,
    bank_ids: np.ndarray,
    train_set: RawSet,
) -> Standardizer:
    node_mean = node_raw[bank_ids].mean(axis=0).astype(np.float32)
    node_std = np.maximum(node_raw[bank_ids].std(axis=0), 1e-5).astype(np.float32)
    flat_edge = train_set.edge.reshape(-1, train_set.edge.shape[-1])
    edge_mean = flat_edge.mean(axis=0).astype(np.float32)
    edge_std = np.maximum(flat_edge.std(axis=0), 1e-5).astype(np.float32)
    return Standardizer(node_mean, node_std, edge_mean, edge_std)


def apply_standardizer(
    raw: RawSet,
    node_raw: np.ndarray,
    stats: Standardizer,
) -> ModelSet:
    node = np.clip((node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0)
    edge = np.clip((raw.edge - stats.edge_mean) / stats.edge_std, -8.0, 8.0)
    return ModelSet(
        query_ids=raw.query_ids,
        query_node=node[raw.query_ids].astype(np.float32),
        neighbor_ids=raw.neighbor_ids,
        edge=edge.astype(np.float32),
        base_weights=raw.base_weights,
    )


def base_prediction(
    data: ModelSet,
    sh: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weight_sh = data.base_weights[..., None, None]
    weight_d = data.base_weights[..., None]
    pred_sh = np.sum(sh[data.neighbor_ids] * weight_sh, axis=1, dtype=np.float64)
    pred_d = np.sum(d[data.neighbor_ids] * weight_d, axis=1, dtype=np.float64)
    return normalize(pred_sh), normalize(pred_d)


class PlanarBarrierAttention(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, sh_dim: int, d_dim: int):
        super().__init__()
        latent = 96
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(node_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 96), nn.SiLU(), nn.LayerNorm(96), nn.Linear(96, 48)
        )
        self.sh_encoder = nn.Sequential(
            nn.Linear(sh_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, 48)
        )
        self.d_encoder = nn.Sequential(
            nn.Linear(d_dim, 160), nn.SiLU(), nn.LayerNorm(160), nn.Linear(160, 48)
        )
        trunk_dim = latent * 3 + 48 * 3
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, 320),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(320, 160),
            nn.SiLU(),
        )
        self.sh_score = nn.Linear(160, 1)
        self.d_score = nn.Linear(160, 1)
        self.sh_scale = nn.Parameter(torch.tensor(-0.2, dtype=torch.float32))
        self.d_scale = nn.Parameter(torch.tensor(-0.2, dtype=torch.float32))

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        edge: torch.Tensor,
        neighbor_sh: torch.Tensor,
        neighbor_d: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_encoder(query_node).unsqueeze(1).expand(-1, K_NEIGHBORS, -1)
        n = self.neighbor_encoder(neighbor_node)
        e = self.edge_encoder(edge)
        sh_feature = self.sh_encoder(neighbor_sh.flatten(2))
        d_feature = self.d_encoder(neighbor_d)
        hidden = self.trunk(torch.cat([q, n, q * n, e, sh_feature, d_feature], dim=2))
        base_log = torch.log(base_weights.clamp_min(1e-12))
        sh_scale = 1.75 * torch.sigmoid(self.sh_scale)
        d_scale = 1.75 * torch.sigmoid(self.d_scale)
        sh_logits = base_log + sh_scale * torch.tanh(self.sh_score(hidden).squeeze(2))
        d_logits = base_log + d_scale * torch.tanh(self.d_score(hidden).squeeze(2))
        return torch.softmax(sh_logits, dim=1), torch.softmax(d_logits, dim=1)


def model_loss(
    pred_sh: torch.Tensor,
    pred_d: torch.Tensor,
    truth_sh: torch.Tensor,
    truth_d: torch.Tensor,
    sh_weight: torch.Tensor,
    d_weight: torch.Tensor,
    base_weight: torch.Tensor,
) -> torch.Tensor:
    sh_loss = 1.0 - F.cosine_similarity(
        pred_sh.flatten(1), truth_sh.flatten(1), dim=1
    ).mean()
    d_loss = 1.0 - F.cosine_similarity(pred_d, truth_d, dim=1).mean()
    sh_kl = torch.sum(
        sh_weight
        * (torch.log(sh_weight.clamp_min(1e-12)) - torch.log(base_weight.clamp_min(1e-12))),
        dim=1,
    ).mean()
    d_kl = torch.sum(
        d_weight
        * (torch.log(d_weight.clamp_min(1e-12)) - torch.log(base_weight.clamp_min(1e-12))),
        dim=1,
    ).mean()
    return 0.52 * sh_loss + 0.48 * d_loss + 0.006 * (sh_kl + d_kl)


@torch.no_grad()
def predict_model(
    model: PlanarBarrierAttention,
    data: ModelSet,
    node_raw: np.ndarray,
    stats: Standardizer,
    sh: np.ndarray,
    d: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    node = np.clip((node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0).astype(np.float32)
    pred_sh = np.empty((len(data.query_ids),) + sh.shape[1:], dtype=np.float32)
    pred_d = np.empty((len(data.query_ids), d.shape[1]), dtype=np.float32)
    all_sh_weight = np.empty_like(data.base_weights)
    all_d_weight = np.empty_like(data.base_weights)
    for start in range(0, len(data.query_ids), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.query_ids))
        neighbor_ids = data.neighbor_ids[start:end]
        sh_weight, d_weight = model(
            torch.from_numpy(data.query_node[start:end]).to(device),
            torch.from_numpy(node[neighbor_ids]).to(device),
            torch.from_numpy(data.edge[start:end]).to(device),
            torch.from_numpy(sh[neighbor_ids]).to(device),
            torch.from_numpy(d[neighbor_ids]).to(device),
            torch.from_numpy(data.base_weights[start:end]).to(device),
        )
        sh_neighbors = torch.from_numpy(sh[neighbor_ids]).to(device)
        d_neighbors = torch.from_numpy(d[neighbor_ids]).to(device)
        block_sh = torch.sum(sh_weight[..., None, None] * sh_neighbors, dim=1)
        block_d = torch.sum(d_weight[..., None] * d_neighbors, dim=1)
        block_sh /= block_sh.flatten(1).sum(dim=1)[:, None, None].clamp_min(1e-30)
        block_d /= block_d.sum(dim=1, keepdim=True).clamp_min(1e-30)
        pred_sh[start:end] = block_sh.cpu().numpy()
        pred_d[start:end] = block_d.cpu().numpy()
        all_sh_weight[start:end] = sh_weight.cpu().numpy()
        all_d_weight[start:end] = d_weight.cpu().numpy()
    return pred_sh, pred_d, all_sh_weight, all_d_weight


def train_model(
    train_data: ModelSet,
    node_raw: np.ndarray,
    stats: Standardizer,
    sh: np.ndarray,
    d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: ModelSet | None,
) -> tuple[PlanarBarrierAttention, dict[str, object]]:
    set_seed(seed)
    model = PlanarBarrierAttention(
        node_raw.shape[1],
        train_data.edge.shape[2],
        int(np.prod(sh.shape[1:])),
        int(np.prod(d.shape[1:])),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    node = np.clip((node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    baseline_metrics = None
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    if validation is not None:
        base_sh, base_d = base_prediction(validation, sh, d)
        baseline_metrics = metrics(
            base_sh, base_d, sh[validation.query_ids], d[validation.query_ids]
        )
        best_metric = baseline_metrics["weighted"]
        best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data.query_ids))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            neighbor_ids = train_data.neighbor_ids[rows]
            base_weight = torch.from_numpy(train_data.base_weights[rows]).to(device)
            sh_weight, d_weight = model(
                torch.from_numpy(train_data.query_node[rows]).to(device),
                torch.from_numpy(node[neighbor_ids]).to(device),
                torch.from_numpy(train_data.edge[rows]).to(device),
                torch.from_numpy(sh[neighbor_ids]).to(device),
                torch.from_numpy(d[neighbor_ids]).to(device),
                base_weight,
            )
            sh_neighbors = torch.from_numpy(sh[neighbor_ids]).to(device)
            d_neighbors = torch.from_numpy(d[neighbor_ids]).to(device)
            pred_sh = torch.sum(sh_weight[..., None, None] * sh_neighbors, dim=1)
            pred_d = torch.sum(d_weight[..., None] * d_neighbors, dim=1)
            truth_sh = torch.from_numpy(sh[train_data.query_ids[rows]]).to(device)
            truth_d = torch.from_numpy(d[train_data.query_ids[rows]]).to(device)
            loss = model_loss(
                pred_sh,
                pred_d,
                truth_sh,
                truth_d,
                sh_weight,
                d_weight,
                base_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
            }
            if validation is not None:
                pred_sh, pred_d, _, _ = predict_model(
                    model, validation, node_raw, stats, sh, d, device
                )
                validation_metrics = metrics(
                    pred_sh,
                    pred_d,
                    sh[validation.query_ids],
                    d[validation.query_ids],
                )
                item["validation"] = validation_metrics
                item["baseline"] = baseline_metrics
                if validation_metrics["weighted"] > best_metric:
                    best_metric = validation_metrics["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"v274 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "baseline": baseline_metrics,
        "history": history,
    }


def load_geometry() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    all_pos = np.vstack([train_pos, test_pos])
    scene = np.load(SCENE_CACHE)
    signature_data = np.load(SIGNATURE_CACHE)
    signatures = np.vstack(
        [signature_data["train_signature"], signature_data["test_signature"]]
    ).astype(bool)
    blocker_count = np.concatenate(
        [signature_data["train_blocker_count"], signature_data["test_blocker_count"]]
    ).astype(np.int16)
    scene_features = np.vstack(
        [scene["train_features"], scene["test_features"]]
    ).astype(np.float32)
    footprint = np.load(FOOTPRINT_CACHE)["footprint"].astype(bool)
    return (
        all_pos,
        signatures,
        blocker_count,
        scene_features,
        footprint,
        scene["grid_z_max"].astype(np.float32),
        float(scene["x_min"]),
        float(scene["y_min"]),
        float(scene["resolution"]),
    )


def extract_current_marginals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = np.load(path, mmap_mode="r")
    first_sh, first_d = v247.extract_current(
        np.asarray(source[:1], dtype=np.complex64)
    )
    sh = np.empty((len(source),) + first_sh.shape[1:], dtype=np.float32)
    d = np.empty((len(source),) + first_d.shape[1:], dtype=np.float32)
    sh[:1] = first_sh
    d[:1] = first_d
    for start in range(1, len(source), 5):
        end = min(start + 5, len(source))
        sh[start:end], d[start:end] = v247.extract_current(
            np.asarray(source[start:end], dtype=np.complex64)
        )
    return sh, d


def choose_residual_alpha(
    base: np.ndarray,
    model: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        blended = normalize(base + alpha * (model - base))
        rows.append(
            {
                "alpha": float(alpha),
                "cosine": float(np.mean(row_cos(blended, truth))),
            }
        )
    best = max(rows, key=lambda row: row["cosine"])
    return float(best["alpha"]), rows


def stats_to_dict(stats: Standardizer) -> dict[str, np.ndarray]:
    return {
        "node_mean": stats.node_mean,
        "node_std": stats.node_std,
        "edge_mean": stats.edge_mean,
        "edge_std": stats.edge_std,
    }


def main() -> None:
    for path in (
        BASE,
        TARGET_CACHE,
        SCENE_CACHE,
        FOOTPRINT_CACHE,
        SIGNATURE_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = np.load(TARGET_CACHE)
    sh = targets["sh"].astype(np.float32)
    d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    (
        all_pos,
        signatures,
        blocker_count,
        scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = load_geometry()
    node_raw = build_node_features(all_pos, signatures, blocker_count, scene_features)

    outer_rows: list[dict[str, object]] = []
    outer_base_sh: list[np.ndarray] = []
    outer_base_d: list[np.ndarray] = []
    outer_model_sh: list[np.ndarray] = []
    outer_model_d: list[np.ndarray] = []
    outer_truth_sh: list[np.ndarray] = []
    outer_truth_d: list[np.ndarray] = []
    selected_epochs: list[int] = []

    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit,
            train_pos[:, :2],
            test_pos[:, :2],
            INNER_COUNT,
            seed + 10_000,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        raw_inner_train = build_raw_set(
            inner_fit,
            inner_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        raw_inner_val = build_raw_set(
            inner_ids,
            inner_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        inner_stats = fit_standardizer(node_raw, inner_fit, raw_inner_train)
        inner_train = apply_standardizer(raw_inner_train, node_raw, inner_stats)
        inner_val = apply_standardizer(raw_inner_val, node_raw, inner_stats)
        _inner_model, inner_report = train_model(
            inner_train,
            node_raw,
            inner_stats,
            sh,
            d,
            seed,
            EPOCHS,
            device,
            inner_val,
        )
        selected_epoch = int(inner_report["best_epoch"])
        selected_epochs.append(selected_epoch)

        raw_outer_train = build_raw_set(
            outer_fit,
            outer_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        raw_outer_val = build_raw_set(
            outer_ids,
            outer_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        outer_stats = fit_standardizer(node_raw, outer_fit, raw_outer_train)
        outer_train = apply_standardizer(raw_outer_train, node_raw, outer_stats)
        outer_val = apply_standardizer(raw_outer_val, node_raw, outer_stats)
        base_sh, base_d = base_prediction(outer_val, sh, d)
        baseline = metrics(base_sh, base_d, sh[outer_ids], d[outer_ids])
        if selected_epoch == 0:
            model_sh, model_d = base_sh, base_d
            outer_training = {"seed": seed, "best_epoch": 0, "history": []}
            sh_weight_shift = np.zeros(len(outer_ids), dtype=np.float32)
            d_weight_shift = np.zeros(len(outer_ids), dtype=np.float32)
        else:
            outer_model, outer_training = train_model(
                outer_train,
                node_raw,
                outer_stats,
                sh,
                d,
                seed + 50_000,
                selected_epoch,
                device,
                None,
            )
            model_sh, model_d, sh_weight, d_weight = predict_model(
                outer_model,
                outer_val,
                node_raw,
                outer_stats,
                sh,
                d,
                device,
            )
            sh_weight_shift = np.sum(
                np.abs(sh_weight - outer_val.base_weights), axis=1
            )
            d_weight_shift = np.sum(
                np.abs(d_weight - outer_val.base_weights), axis=1
            )
        model_metric = metrics(model_sh, model_d, sh[outer_ids], d[outer_ids])
        delta = {
            key: model_metric[key] - baseline[key]
            for key in ("sh", "d", "weighted")
        }
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "inner_training": inner_report,
                "outer_training": outer_training,
                "baseline": baseline,
                "model": model_metric,
                "delta": delta,
                "sh_weight_l1": summarize(sh_weight_shift),
                "d_weight_l1": summarize(d_weight_shift),
            }
        )
        outer_base_sh.append(base_sh)
        outer_base_d.append(base_d)
        outer_model_sh.append(model_sh)
        outer_model_d.append(model_d)
        outer_truth_sh.append(sh[outer_ids])
        outer_truth_d.append(d[outer_ids])
        print(
            f"v274 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
            f"baseline={baseline['weighted']:.6f} model={model_metric['weighted']:.6f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )

    fold_delta = np.asarray(
        [row["delta"]["weighted"] for row in outer_rows], dtype=np.float64
    )
    fold_sh_delta = np.asarray(
        [row["delta"]["sh"] for row in outer_rows], dtype=np.float64
    )
    fold_d_delta = np.asarray(
        [row["delta"]["d"] for row in outer_rows], dtype=np.float64
    )
    official_selected = bool(
        np.mean(fold_delta) >= MIN_OFFICIAL_DELTA
        and np.sum(fold_delta > 0.0) >= 3
        and np.mean(fold_sh_delta) > 0.0
        and np.mean(fold_d_delta) > 0.0
    )
    cv_base_sh = np.concatenate(outer_base_sh)
    cv_base_d = np.concatenate(outer_base_d)
    cv_model_sh = np.concatenate(outer_model_sh)
    cv_model_d = np.concatenate(outer_model_d)
    cv_truth_sh = np.concatenate(outer_truth_sh)
    cv_truth_d = np.concatenate(outer_truth_d)
    alpha_sh, alpha_sh_grid = choose_residual_alpha(
        cv_base_sh, cv_model_sh, cv_truth_sh
    )
    alpha_d, alpha_d_grid = choose_residual_alpha(
        cv_base_d, cv_model_d, cv_truth_d
    )

    final_training: list[dict[str, object]] = []
    local_score = None
    projection_diagnostics = None
    if official_selected:
        final_epoch = max(int(np.median([epoch for epoch in selected_epochs if epoch > 0])), 1)
        test_ids = np.arange(len(train_pos), len(all_pos), dtype=np.int64)
        raw_full_train = build_raw_set(
            valid_ids,
            valid_ids,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        raw_test = build_raw_set(
            test_ids,
            valid_ids,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        full_stats = fit_standardizer(node_raw, valid_ids, raw_full_train)
        full_train = apply_standardizer(raw_full_train, node_raw, full_stats)
        test_set = apply_standardizer(raw_test, node_raw, full_stats)
        test_base_sh, test_base_d = base_prediction(test_set, sh, d)
        test_model_sh_parts = []
        test_model_d_parts = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                full_train,
                node_raw,
                full_stats,
                sh,
                d,
                seed,
                final_epoch,
                device,
                None,
            )
            model_sh, model_d, _, _ = predict_model(
                model, test_set, node_raw, full_stats, sh, d, device
            )
            test_model_sh_parts.append(model_sh)
            test_model_d_parts.append(model_d)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                    "standardizer": stats_to_dict(full_stats),
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
        test_model_sh = normalize(np.mean(test_model_sh_parts, axis=0))
        test_model_d = normalize(np.mean(test_model_d_parts, axis=0))
        current_sh, current_d = extract_current_marginals(BASE)
        target_sh = normalize(
            current_sh + alpha_sh * (test_model_sh - test_base_sh)
        )
        target_d = normalize(current_d + alpha_d * (test_model_d - test_base_d))
        np.savez_compressed(
            TARGETS_OUT,
            base_sh=test_base_sh,
            base_d=test_base_d,
            model_sh=test_model_sh,
            model_d=test_model_d,
            target_sh=target_sh,
            target_d=target_d,
            alpha_sh=np.asarray(alpha_sh, dtype=np.float32),
            alpha_d=np.asarray(alpha_d, dtype=np.float32),
        )
        projection_diagnostics = v247.project(target_sh, target_d, OUTPUT)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "local_gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "A neural attention kernel can use the verified 2.5D barrier signal softly, "
            "preserving useful cross-wall multipath neighbors while suppressing false mixing."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP spectrum bank",
                "official train/test XY coordinates",
                "official point-cloud wall/component/height features",
                "strict v239 official-only carrier",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "hidden labels",
                "RT MIMO H or path coefficients",
                "local replay truth for training, epoch, or alpha selection",
            ],
            "local_truth_usage": (
                "one final gate score after nested official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "neighbors": K_NEIGHBORS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "mean_delta": {
                "sh": float(np.mean(fold_sh_delta)),
                "d": float(np.mean(fold_d_delta)),
                "weighted": float(np.mean(fold_delta)),
            },
            "weighted_wins": int(np.sum(fold_delta > 0.0)),
            "selected_epochs": selected_epochs,
            "selected": official_selected,
            "residual_alpha_sh": alpha_sh,
            "residual_alpha_d": alpha_d,
            "alpha_sh_grid": alpha_sh_grid,
            "alpha_d_grid": alpha_d_grid,
        },
        "final_training": final_training,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in projection_diagnostics.items()}
            if projection_diagnostics is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
