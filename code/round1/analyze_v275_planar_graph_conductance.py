#!/usr/bin/env python3
"""Learn 2.5D graph conductance for transductive planar spectrum diffusion."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
import analyze_v274_planar_barrier_attention as v274
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 275
VERSION = "v275_planar_graph_conductance"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v275_graph_targets.npz"
OUTPUT = OUT_DIR / "v275_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (275, 1275, 2275, 3275, 4275)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 60
EVAL_EVERY = 5
BATCH_SIZE = 80
MIXES = (0.30, 0.55, 0.75)
ITERATIONS = (2, 4, 8)
MIN_OFFICIAL_DELTA = 5e-4
FINAL_SEEDS = (27501, 27531, 27561)
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


class GraphConductance(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        latent = 96
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256, latent)
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(node_dim, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256, latent)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 96), nn.SiLU(), nn.LayerNorm(96), nn.Linear(96, 64)
        )
        self.trunk = nn.Sequential(
            nn.Linear(latent * 3 + 64, 320),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(320, 160),
            nn.SiLU(),
        )
        self.sh_score = nn.Linear(160, 1)
        self.d_score = nn.Linear(160, 1)
        self.sh_scale = nn.Parameter(torch.tensor(-0.15, dtype=torch.float32))
        self.d_scale = nn.Parameter(torch.tensor(-0.15, dtype=torch.float32))

    def forward(
        self,
        query: torch.Tensor,
        neighbor: torch.Tensor,
        edge: torch.Tensor,
        base_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_encoder(query).unsqueeze(1).expand(-1, neighbor.shape[1], -1)
        n = self.neighbor_encoder(neighbor)
        e = self.edge_encoder(edge)
        hidden = self.trunk(torch.cat([q, n, q * n, e], dim=2))
        base_log = torch.log(base_weight.clamp_min(1e-12))
        sh_scale = 2.0 * torch.sigmoid(self.sh_scale)
        d_scale = 2.0 * torch.sigmoid(self.d_scale)
        sh_logit = base_log + sh_scale * torch.tanh(self.sh_score(hidden).squeeze(2))
        d_logit = base_log + d_scale * torch.tanh(self.d_score(hidden).squeeze(2))
        return torch.softmax(sh_logit, dim=1), torch.softmax(d_logit, dim=1)


def direct_prediction(
    data: v274.ModelSet,
    sh: np.ndarray,
    d: np.ndarray,
    sh_weight: np.ndarray,
    d_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_sh = np.sum(
        sh[data.neighbor_ids] * sh_weight[..., None, None], axis=1, dtype=np.float64
    )
    pred_d = np.sum(
        d[data.neighbor_ids] * d_weight[..., None], axis=1, dtype=np.float64
    )
    return normalize(pred_sh), normalize(pred_d)


def oracle_weights(
    query_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    base_weight: torch.Tensor,
    sh: np.ndarray,
    d: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_sh = torch.from_numpy(sh[query_ids]).to(device).flatten(1)
    neighbor_sh = torch.from_numpy(sh[neighbor_ids]).to(device).flatten(2)
    query_d = torch.from_numpy(d[query_ids]).to(device)
    neighbor_d = torch.from_numpy(d[neighbor_ids]).to(device)
    sh_similarity = F.cosine_similarity(query_sh[:, None, :], neighbor_sh, dim=2)
    d_similarity = F.cosine_similarity(query_d[:, None, :], neighbor_d, dim=2)
    base_log = torch.log(base_weight.clamp_min(1e-12))
    return (
        torch.softmax(base_log + 6.0 * sh_similarity, dim=1),
        torch.softmax(base_log + 6.0 * d_similarity, dim=1),
    )


@torch.no_grad()
def predict_weights(
    model: GraphConductance,
    data: v274.ModelSet,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    node = np.clip(
        (node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0
    ).astype(np.float32)
    sh_weight = np.empty_like(data.base_weights)
    d_weight = np.empty_like(data.base_weights)
    for start in range(0, len(data.query_ids), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.query_ids))
        neighbor_ids = data.neighbor_ids[start:end]
        block_sh, block_d = model(
            torch.from_numpy(data.query_node[start:end]).to(device),
            torch.from_numpy(node[neighbor_ids]).to(device),
            torch.from_numpy(data.edge[start:end]).to(device),
            torch.from_numpy(data.base_weights[start:end]).to(device),
        )
        sh_weight[start:end] = block_sh.cpu().numpy()
        d_weight[start:end] = block_d.cpu().numpy()
    return sh_weight, d_weight


def train_model(
    train_data: v274.ModelSet,
    validation: v274.ModelSet | None,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    sh: np.ndarray,
    d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[GraphConductance, dict[str, object]]:
    set_seed(seed)
    model = GraphConductance(node_raw.shape[1], train_data.edge.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    node = np.clip(
        (node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0
    ).astype(np.float32)
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    validation_base = None
    if validation is not None:
        validation_base = metrics(
            *direct_prediction(
                validation,
                sh,
                d,
                validation.base_weights,
                validation.base_weights,
            ),
            sh[validation.query_ids],
            d[validation.query_ids],
        )
        best_metric = validation_base["weighted"]
        best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data.query_ids))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            query_ids = train_data.query_ids[rows]
            neighbor_ids = train_data.neighbor_ids[rows]
            base_weight = torch.from_numpy(train_data.base_weights[rows]).to(device)
            pred_sh_weight, pred_d_weight = model(
                torch.from_numpy(train_data.query_node[rows]).to(device),
                torch.from_numpy(node[neighbor_ids]).to(device),
                torch.from_numpy(train_data.edge[rows]).to(device),
                base_weight,
            )
            target_sh_weight, target_d_weight = oracle_weights(
                query_ids, neighbor_ids, base_weight, sh, d, device
            )
            sh_kl = torch.sum(
                target_sh_weight
                * (
                    torch.log(target_sh_weight.clamp_min(1e-12))
                    - torch.log(pred_sh_weight.clamp_min(1e-12))
                ),
                dim=1,
            ).mean()
            d_kl = torch.sum(
                target_d_weight
                * (
                    torch.log(target_d_weight.clamp_min(1e-12))
                    - torch.log(pred_d_weight.clamp_min(1e-12))
                ),
                dim=1,
            ).mean()
            loss = 0.52 * sh_kl + 0.48 * d_kl
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
                val_sh_weight, val_d_weight = predict_weights(
                    model, validation, node_raw, stats, device
                )
                pred_sh, pred_d = direct_prediction(
                    validation, sh, d, val_sh_weight, val_d_weight
                )
                validation_metric = metrics(
                    pred_sh,
                    pred_d,
                    sh[validation.query_ids],
                    d[validation.query_ids],
                )
                item["validation"] = validation_metric
                item["baseline"] = validation_base
                if validation_metric["weighted"] > best_metric:
                    best_metric = validation_metric["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"v275 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "baseline": validation_base,
        "history": history,
    }


def graph_matrix(
    data: v274.ModelSet,
    weight: np.ndarray,
    node_count: int,
) -> sparse.csr_matrix:
    if not np.array_equal(data.query_ids, np.arange(node_count)):
        raise ValueError("graph query ids must cover every node in order")
    rows = np.repeat(np.arange(node_count, dtype=np.int64), weight.shape[1])
    cols = data.neighbor_ids.reshape(-1)
    matrix = sparse.coo_matrix(
        (weight.reshape(-1), (rows, cols)), shape=(node_count, node_count)
    ).tocsr()
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    return sparse.diags(1.0 / np.maximum(row_sum, 1e-20)) @ matrix


def direct_anchors(
    known_ids: np.ndarray,
    all_pos: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_ids = np.arange(len(all_pos), dtype=np.int64)
    unknown_ids = np.setdiff1d(all_ids, known_ids)
    anchor_sh, anchor_d = v254.euclidean_predict(
        all_pos[known_ids, :2],
        all_pos[unknown_ids, :2],
        sh[known_ids],
        d[known_ids],
        12,
        2.0,
    )
    return unknown_ids, anchor_sh, anchor_d


def propagate(
    known_ids: np.ndarray,
    all_pos: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
    matrix_sh: sparse.csr_matrix,
    matrix_d: sparse.csr_matrix,
    mix: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = len(all_pos)
    unknown_ids, anchor_sh, anchor_d = direct_anchors(known_ids, all_pos, sh, d)
    field_sh = np.zeros((node_count,) + sh.shape[1:], dtype=np.float32)
    field_d = np.zeros((node_count,) + d.shape[1:], dtype=np.float32)
    field_sh[known_ids] = sh[known_ids]
    field_d[known_ids] = d[known_ids]
    field_sh[unknown_ids] = anchor_sh
    field_d[unknown_ids] = anchor_d
    flat_sh = field_sh.reshape(node_count, -1)
    flat_d = field_d.reshape(node_count, -1)
    anchor_sh_flat = anchor_sh.reshape(len(unknown_ids), -1)
    anchor_d_flat = anchor_d.reshape(len(unknown_ids), -1)
    for _ in range(iterations):
        next_sh = matrix_sh @ flat_sh
        next_d = matrix_d @ flat_d
        flat_sh[unknown_ids] = (
            mix * anchor_sh_flat + (1.0 - mix) * next_sh[unknown_ids]
        )
        flat_d[unknown_ids] = (
            mix * anchor_d_flat + (1.0 - mix) * next_d[unknown_ids]
        )
        flat_sh[known_ids] = sh[known_ids].reshape(len(known_ids), -1)
        flat_d[known_ids] = d[known_ids].reshape(len(known_ids), -1)
    return (
        normalize(flat_sh.reshape(field_sh.shape)),
        normalize(flat_d.reshape(field_d.shape)),
    )


def select_propagation(
    known_ids: np.ndarray,
    validation_ids: np.ndarray,
    all_pos: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
    matrix_sh: sparse.csr_matrix,
    matrix_d: sparse.csr_matrix,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for mix in MIXES:
        for iterations in ITERATIONS:
            pred_sh, pred_d = propagate(
                known_ids,
                all_pos,
                sh,
                d,
                matrix_sh,
                matrix_d,
                mix,
                iterations,
            )
            score = metrics(
                pred_sh[validation_ids],
                pred_d[validation_ids],
                sh[validation_ids],
                d[validation_ids],
            )
            rows.append({"mix": mix, "iterations": iterations, "metrics": score})
    return max(rows, key=lambda row: row["metrics"]["weighted"]), rows


def choose_residual_alpha(
    base: np.ndarray, model: np.ndarray, truth: np.ndarray
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


def main() -> None:
    for path in (
        BASE,
        TARGET_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
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
    ) = v274.load_geometry()
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    all_ids = np.arange(len(all_pos), dtype=np.int64)
    raw_graph = v274.build_raw_set(
        all_ids,
        all_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )

    outer_rows: list[dict[str, object]] = []
    outer_base_sh: list[np.ndarray] = []
    outer_base_d: list[np.ndarray] = []
    outer_graph_sh: list[np.ndarray] = []
    outer_graph_d: list[np.ndarray] = []
    outer_truth_sh: list[np.ndarray] = []
    outer_truth_d: list[np.ndarray] = []
    selected_epochs: list[int] = []
    selected_mixes: list[float] = []
    selected_iterations: list[int] = []

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

        raw_inner_train = v274.build_raw_set(
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
        raw_inner_val = v274.build_raw_set(
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
        inner_stats = v274.fit_standardizer(node_raw, inner_fit, raw_inner_train)
        inner_train = v274.apply_standardizer(raw_inner_train, node_raw, inner_stats)
        inner_val = v274.apply_standardizer(raw_inner_val, node_raw, inner_stats)
        inner_model, inner_training = train_model(
            inner_train,
            inner_val,
            node_raw,
            inner_stats,
            sh,
            d,
            seed,
            EPOCHS,
            device,
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)
        inner_graph = v274.apply_standardizer(raw_graph, node_raw, inner_stats)
        if selected_epoch == 0:
            inner_sh_weight = inner_graph.base_weights
            inner_d_weight = inner_graph.base_weights
        else:
            inner_sh_weight, inner_d_weight = predict_weights(
                inner_model, inner_graph, node_raw, inner_stats, device
            )
        inner_matrix_sh = graph_matrix(inner_graph, inner_sh_weight, len(all_pos))
        inner_matrix_d = graph_matrix(inner_graph, inner_d_weight, len(all_pos))
        selected_propagation, propagation_grid = select_propagation(
            inner_fit,
            inner_ids,
            all_pos,
            sh,
            d,
            inner_matrix_sh,
            inner_matrix_d,
        )
        mix = float(selected_propagation["mix"])
        iterations = int(selected_propagation["iterations"])
        selected_mixes.append(mix)
        selected_iterations.append(iterations)

        raw_outer_train = v274.build_raw_set(
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
        outer_stats = v274.fit_standardizer(node_raw, outer_fit, raw_outer_train)
        outer_train = v274.apply_standardizer(raw_outer_train, node_raw, outer_stats)
        outer_graph = v274.apply_standardizer(raw_graph, node_raw, outer_stats)
        if selected_epoch == 0:
            outer_model = None
            outer_training = {"seed": seed + 50_000, "best_epoch": 0, "history": []}
            outer_sh_weight = outer_graph.base_weights
            outer_d_weight = outer_graph.base_weights
        else:
            outer_model, outer_training = train_model(
                outer_train,
                None,
                node_raw,
                outer_stats,
                sh,
                d,
                seed + 50_000,
                selected_epoch,
                device,
            )
            outer_sh_weight, outer_d_weight = predict_weights(
                outer_model, outer_graph, node_raw, outer_stats, device
            )
        outer_matrix_sh = graph_matrix(outer_graph, outer_sh_weight, len(all_pos))
        outer_matrix_d = graph_matrix(outer_graph, outer_d_weight, len(all_pos))
        graph_sh, graph_d = propagate(
            outer_fit,
            all_pos,
            sh,
            d,
            outer_matrix_sh,
            outer_matrix_d,
            mix,
            iterations,
        )
        _, base_sh_all, base_d_all = direct_anchors(outer_fit, all_pos, sh, d)
        unknown_ids = np.setdiff1d(all_ids, outer_fit)
        base_sh = np.zeros((len(all_pos),) + sh.shape[1:], dtype=np.float32)
        base_d = np.zeros((len(all_pos),) + d.shape[1:], dtype=np.float32)
        base_sh[outer_fit] = sh[outer_fit]
        base_d[outer_fit] = d[outer_fit]
        base_sh[unknown_ids] = base_sh_all
        base_d[unknown_ids] = base_d_all
        baseline = metrics(
            base_sh[outer_ids], base_d[outer_ids], sh[outer_ids], d[outer_ids]
        )
        graph_metric = metrics(
            graph_sh[outer_ids], graph_d[outer_ids], sh[outer_ids], d[outer_ids]
        )
        delta = {
            key: graph_metric[key] - baseline[key]
            for key in ("sh", "d", "weighted")
        }
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "selected_mix": mix,
                "selected_iterations": iterations,
                "inner_training": inner_training,
                "inner_propagation_grid": propagation_grid,
                "outer_training": outer_training,
                "baseline": baseline,
                "graph": graph_metric,
                "delta": delta,
            }
        )
        outer_base_sh.append(base_sh[outer_ids])
        outer_base_d.append(base_d[outer_ids])
        outer_graph_sh.append(graph_sh[outer_ids])
        outer_graph_d.append(graph_d[outer_ids])
        outer_truth_sh.append(sh[outer_ids])
        outer_truth_d.append(d[outer_ids])
        print(
            f"v275 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
            f"mix={mix:.2f} it={iterations} baseline={baseline['weighted']:.6f} "
            f"graph={graph_metric['weighted']:.6f} delta={delta['weighted']:+.6f}",
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
    cv_graph_sh = np.concatenate(outer_graph_sh)
    cv_graph_d = np.concatenate(outer_graph_d)
    cv_truth_sh = np.concatenate(outer_truth_sh)
    cv_truth_d = np.concatenate(outer_truth_d)
    alpha_sh, alpha_sh_grid = choose_residual_alpha(
        cv_base_sh, cv_graph_sh, cv_truth_sh
    )
    alpha_d, alpha_d_grid = choose_residual_alpha(
        cv_base_d, cv_graph_d, cv_truth_d
    )

    final_training: list[dict[str, object]] = []
    local_score = None
    projection_diagnostics = None
    if official_selected:
        positive_epochs = [epoch for epoch in selected_epochs if epoch > 0]
        final_epoch = max(int(np.median(positive_epochs)), 1)
        final_mix = float(np.median(selected_mixes))
        final_iterations = int(np.median(selected_iterations))
        raw_full_train = v274.build_raw_set(
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
        full_stats = v274.fit_standardizer(node_raw, valid_ids, raw_full_train)
        full_train = v274.apply_standardizer(raw_full_train, node_raw, full_stats)
        full_graph = v274.apply_standardizer(raw_graph, node_raw, full_stats)
        graph_sh_parts = []
        graph_d_parts = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                full_train,
                None,
                node_raw,
                full_stats,
                sh,
                d,
                seed,
                final_epoch,
                device,
            )
            sh_weight, d_weight = predict_weights(
                model, full_graph, node_raw, full_stats, device
            )
            matrix_sh = graph_matrix(full_graph, sh_weight, len(all_pos))
            matrix_d = graph_matrix(full_graph, d_weight, len(all_pos))
            graph_sh, graph_d = propagate(
                valid_ids,
                all_pos,
                sh,
                d,
                matrix_sh,
                matrix_d,
                final_mix,
                final_iterations,
            )
            graph_sh_parts.append(graph_sh[len(train_pos) :])
            graph_d_parts.append(graph_d[len(train_pos) :])
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "mix": final_mix,
                    "iterations": final_iterations,
                    "model_state": model.state_dict(),
                    "standardizer": v274.stats_to_dict(full_stats),
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
        test_graph_sh = normalize(np.mean(graph_sh_parts, axis=0))
        test_graph_d = normalize(np.mean(graph_d_parts, axis=0))
        test_ids = np.arange(len(train_pos), len(all_pos), dtype=np.int64)
        _unknown, anchor_sh, anchor_d = direct_anchors(valid_ids, all_pos, sh, d)
        unknown_ids = np.setdiff1d(all_ids, valid_ids)
        location = {int(node): row for row, node in enumerate(unknown_ids)}
        test_rows = np.asarray([location[int(node)] for node in test_ids], dtype=np.int64)
        test_base_sh = anchor_sh[test_rows]
        test_base_d = anchor_d[test_rows]
        current_sh, current_d = v274.extract_current_marginals(BASE)
        target_sh = normalize(
            current_sh + alpha_sh * (test_graph_sh - test_base_sh)
        )
        target_d = normalize(current_d + alpha_d * (test_graph_d - test_base_d))
        np.savez_compressed(
            TARGETS_OUT,
            base_sh=test_base_sh,
            base_d=test_base_d,
            graph_sh=test_graph_sh,
            graph_d=test_graph_d,
            target_sh=target_sh,
            target_d=target_d,
            alpha_sh=np.asarray(alpha_sh, dtype=np.float32),
            alpha_d=np.asarray(alpha_d, dtype=np.float32),
            mix=np.asarray(final_mix, dtype=np.float32),
            iterations=np.asarray(final_iterations, dtype=np.int32),
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
            "Official test coordinates are intermediate nodes of the same planar field, so "
            "learned barrier-aware conductance and label-clamped multi-hop diffusion can add "
            "information beyond direct train-to-test attention."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP labels",
                "official train/test XY coordinates",
                "official point-cloud wall/component/height features",
                "strict v239 official-only carrier",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "hidden labels",
                "RT MIMO H or path coefficients",
                "local replay truth for training or selection",
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
            "epochs": EPOCHS,
            "mixes": list(MIXES),
            "iterations": list(ITERATIONS),
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
            "selected_mixes": selected_mixes,
            "selected_iterations": selected_iterations,
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
