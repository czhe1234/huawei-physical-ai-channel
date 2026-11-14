#!/usr/bin/env python3
"""v234 analysis: constrained official-train support-bank attention.

The model cannot synthesize a free residual.  For every query it only assigns
weights to nearby released training supports, using official coordinates,
point-cloud scene/geodesic features, and point-cloud multibounce path tokens.
This tests whether the missing propagation mode is mainly a neighbor-selection
problem while keeping generation inside the observed official support manifold.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v233_official_joint_ad_manifold as v233
import generate_v121_support_template_manifold as v121
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import MH, MV, S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 234
VERSION = "v234_official_neighbor_support_attention"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v234_official_neighbor_support_attention.json"
ACTION_OUT = OUT_DIR / "v234_neighbor_attention_actions.npz"
TARGET_OUT = OUT_DIR / "v234_test_attention_target.npy"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

K_NEIGHBORS = 64
VAL_COUNT = 320
TRAIN_EPOCHS = 140
BATCH_SIZE = 28
SEEDS = (23401, 23419, 23441)
FLOOR = 1e-10
GATE_SCORE = 0.645604
REGIMES = ("clear", "light", "medium", "heavy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compact_path_features(prefix: str, pos: np.ndarray, indices: np.ndarray | None) -> np.ndarray:
    numeric, path_type, valid = v215.make_path_tokens(prefix, pos, indices)
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(numeric * valid[..., None], axis=1) / count
    masked = np.where(valid[..., None], numeric, -1e4)
    maximum = np.max(masked, axis=1)
    maximum[maximum < -1e3] = 0.0
    type_count = np.stack([(path_type == value).sum(axis=1) for value in range(4)], axis=1).astype(np.float32)
    type_count /= np.maximum(type_count.sum(axis=1, keepdims=True), 1.0)
    return np.concatenate(
        [mean.astype(np.float32), maximum.astype(np.float32), valid.mean(axis=1, keepdims=True).astype(np.float32), type_count],
        axis=1,
    ).astype(np.float32)


def node_features(
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
    support: np.ndarray,
    log_power: np.ndarray,
) -> np.ndarray:
    scene = np.load(v215.SCENE_CACHE)[f"{prefix}_features"].astype(np.float32)
    geodesic = np.load(v215.GEODESIC_CACHE)[f"{prefix}_features"].astype(np.float32)
    if indices is not None:
        scene = scene[indices]
        geodesic = geodesic[indices]
    regimes = regime_of(blocked)
    onehot = np.stack([(regimes.astype(str) == value).astype(np.float32) for value in REGIMES], axis=1)
    support_sig = v121.support_signature(np.asarray(support, dtype=np.float32), FLOOR)
    path = compact_path_features(prefix, pos, indices)
    return np.nan_to_num(
        np.concatenate(
            [
                v215.position_features(pos),
                blocked[:, None].astype(np.float32) / 160.0,
                onehot,
                scene,
                geodesic,
                path,
                support_sig,
                log_power[:, None].astype(np.float32),
            ],
            axis=1,
        ),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


def candidate_neighbors(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    query_ids: np.ndarray | None,
    bank_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = v218.effective_distance(query_pos, query_blocked, bank_pos, bank_blocked)
    if query_ids is not None and bank_ids is not None:
        dist[query_ids[:, None] == bank_ids[None, :]] = np.inf
    k = min(K_NEIGHBORS, len(bank_pos) - (1 if query_ids is not None else 0))
    idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
    dsel = np.take_along_axis(dist, idx, axis=1)
    order = np.argsort(dsel, axis=1)
    idx = np.take_along_axis(idx, order, axis=1).astype(np.int32)
    dsel = np.take_along_axis(dsel, order, axis=1).astype(np.float32)
    base = 1.0 / np.maximum(dsel, 0.35) ** 2
    same = regime_of(query_blocked)[:, None] == regime_of(bank_blocked)[idx]
    base *= np.where(same, 1.22, 0.70).astype(np.float32)
    base /= np.maximum(base.sum(axis=1, keepdims=True), TINY)
    return idx, dsel, base.astype(np.float32)


def edge_features(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    idx: np.ndarray,
    dsel: np.ndarray,
) -> np.ndarray:
    neighbor_pos = bank_pos[idx]
    delta = neighbor_pos - query_pos[:, None, :]
    euclidean = np.linalg.norm(delta, axis=2)
    q_bs = query_pos - np.asarray([50.0, 0.0, 25.0], dtype=np.float32)[None, :]
    n_bs = neighbor_pos - np.asarray([50.0, 0.0, 25.0], dtype=np.float32)[None, None, :]
    q_az = np.arctan2(q_bs[:, 1], q_bs[:, 0])[:, None]
    n_az = np.arctan2(n_bs[..., 1], n_bs[..., 0])
    daz = np.arctan2(np.sin(n_az - q_az), np.cos(n_az - q_az))
    qreg = regime_of(query_blocked)
    nreg = regime_of(bank_blocked)[idx]
    query_blocked_edge = np.broadcast_to(query_blocked[:, None], idx.shape)
    return np.nan_to_num(
        np.stack(
            [
                delta[..., 0] / 100.0,
                delta[..., 1] / 100.0,
                delta[..., 2] / 30.0,
                euclidean / 40.0,
                np.log1p(euclidean) / 5.0,
                dsel / 40.0,
                np.sin(daz),
                np.cos(daz),
                query_blocked_edge / 160.0,
                bank_blocked[idx] / 160.0,
                np.abs(bank_blocked[idx] - query_blocked_edge) / 160.0,
                (qreg[:, None] == nreg).astype(np.float32),
            ],
            axis=2,
        ),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


@dataclass
class QuerySet:
    query_features: np.ndarray
    neighbor_idx: np.ndarray
    edge_features: np.ndarray
    base_weights: np.ndarray


def weighted_average_support(bank: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.empty((len(idx), MH, MV, S_SC), dtype=np.float32)
    for start in range(0, len(idx), 8):
        end = min(start + 8, len(idx))
        block = np.zeros((end - start, MH, MV, S_SC), dtype=np.float32)
        for rank in range(idx.shape[1]):
            block += weights[start:end, rank, None, None, None] * bank[idx[start:end, rank]]
        out[start:end] = v233.normalize_support(block)
    return out


def standardize_nodes_and_edges(
    bank_features: np.ndarray,
    query_raw: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[QuerySet], dict[str, np.ndarray]]:
    node_mean = bank_features.mean(axis=0).astype(np.float32)
    node_std = np.maximum(bank_features.std(axis=0), 1e-5).astype(np.float32)
    all_edges = np.concatenate([value[2].reshape(-1, value[2].shape[2]) for value in query_raw], axis=0)
    edge_mean = all_edges.mean(axis=0).astype(np.float32)
    edge_std = np.maximum(all_edges.std(axis=0), 1e-5).astype(np.float32)
    bank = np.clip((bank_features - node_mean) / node_std, -8.0, 8.0).astype(np.float32)
    sets = []
    for features, idx, edges, base in query_raw:
        sets.append(
            QuerySet(
                np.clip((features - node_mean) / node_std, -8.0, 8.0).astype(np.float32),
                idx.astype(np.int32),
                np.clip((edges - edge_mean) / edge_std, -8.0, 8.0).astype(np.float32),
                base.astype(np.float32),
            )
        )
    return bank, sets, {"node_mean": node_mean, "node_std": node_std, "edge_mean": edge_mean, "edge_std": edge_std}


class NeighborAttention(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        latent = 160
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 384), nn.SiLU(), nn.LayerNorm(384), nn.Dropout(0.05), nn.Linear(384, latent)
        )
        self.bank_encoder = nn.Sequential(
            nn.Linear(node_dim, 384), nn.SiLU(), nn.LayerNorm(384), nn.Dropout(0.05), nn.Linear(384, latent)
        )
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, 96), nn.SiLU(), nn.LayerNorm(96), nn.Linear(96, 64))
        self.score = nn.Sequential(
            nn.Linear(latent * 3 + 64, 384),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(384, 192),
            nn.SiLU(),
            nn.Linear(192, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

    def forward(
        self,
        query: torch.Tensor,
        bank_neighbors: torch.Tensor,
        edges: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_encoder(query).unsqueeze(1).expand(-1, bank_neighbors.shape[1], -1)
        n = self.bank_encoder(bank_neighbors)
        e = self.edge_encoder(edges)
        raw = self.score(torch.cat([q, n, q * n, e], dim=2)).squeeze(2)
        scale = 1.50 * torch.sigmoid(self.residual_scale)
        logits = torch.log(base_weights.clamp_min(1e-12)) + scale * torch.tanh(raw)
        return torch.softmax(logits, dim=1)


def support_loss(pred: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    p = pred.reshape(-1, MH, MV, S_SC)
    t = truth.reshape(-1, MH, MV, S_SC)
    joint = 1.0 - F.cosine_similarity(pred, truth, dim=1).mean()
    beam = 1.0 - F.cosine_similarity(p.sum(dim=3).reshape(len(p), -1), t.sum(dim=3).reshape(len(t), -1), dim=1).mean()
    delay = 1.0 - F.cosine_similarity(p.sum(dim=(1, 2)), t.sum(dim=(1, 2)), dim=1).mean()
    kl = torch.sum(weights * (torch.log(weights.clamp_min(1e-12)) - torch.log(base.clamp_min(1e-12))), dim=1).mean()
    return 0.42 * joint + 0.25 * beam + 0.31 * delay + 0.02 * kl


@torch.no_grad()
def predict_attention(
    model: NeighborAttention,
    query: QuerySet,
    bank_features: np.ndarray,
    bank_support: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    bank_feat_t = torch.from_numpy(bank_features).to(device)
    bank_support_t = torch.from_numpy(np.asarray(bank_support, dtype=np.float32).reshape(len(bank_support), -1)).to(device)
    out = np.empty((len(query.query_features), MH, MV, S_SC), dtype=np.float32)
    all_weights = np.empty_like(query.base_weights, dtype=np.float32)
    for start in range(0, len(out), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(out))
        idx = torch.from_numpy(query.neighbor_idx[start:end]).to(device)
        weights = model(
            torch.from_numpy(query.query_features[start:end]).to(device),
            bank_feat_t[idx],
            torch.from_numpy(query.edge_features[start:end]).to(device),
            torch.from_numpy(query.base_weights[start:end]).to(device),
        )
        pred = torch.sum(weights[..., None] * bank_support_t[idx], dim=1)
        pred /= pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        out[start:end] = pred.cpu().numpy().reshape(end - start, MH, MV, S_SC)
        all_weights[start:end] = weights.cpu().numpy()
    return out, all_weights


def base_prediction(query: QuerySet, bank_support: np.ndarray) -> np.ndarray:
    return weighted_average_support(np.asarray(bank_support, dtype=np.float32), query.neighbor_idx, query.base_weights)


def train_attention(
    train_query: QuerySet,
    train_target: np.ndarray,
    bank_features: np.ndarray,
    bank_support: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[QuerySet, np.ndarray] | None,
) -> tuple[NeighborAttention, dict[str, object]]:
    set_seed(seed)
    model = NeighborAttention(bank_features.shape[1], train_query.edge_features.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.0e-4, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    bank_feat_t = torch.from_numpy(bank_features).to(device)
    bank_support_t = torch.from_numpy(np.asarray(bank_support, dtype=np.float32).reshape(len(bank_support), -1)).to(device)
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    validation_base_metrics = None
    if validation is not None:
        validation_base_metrics = v233.support_metrics(base_prediction(validation[0], bank_support), validation[1])
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_query.query_features))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            idx = torch.from_numpy(train_query.neighbor_idx[ids]).to(device)
            base = torch.from_numpy(train_query.base_weights[ids]).to(device)
            weights = model(
                torch.from_numpy(train_query.query_features[ids]).to(device),
                bank_feat_t[idx],
                torch.from_numpy(train_query.edge_features[ids]).to(device),
                base,
            )
            pred = torch.sum(weights[..., None] * bank_support_t[idx], dim=1)
            pred /= pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
            truth = torch.from_numpy(np.asarray(train_target[ids], dtype=np.float32).reshape(len(ids), -1)).to(device)
            loss = support_loss(pred, truth, weights, base)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if validation is not None:
                val_query, val_truth = validation
                val_pred, _ = predict_attention(model, val_query, bank_features, bank_support, device)
                metrics = v233.support_metrics(val_pred, val_truth)
                item["val_model"] = metrics
                item["val_base"] = validation_base_metrics
                if metrics["weighted"] > best_metric:
                    best_metric = metrics["weighted"]
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v234 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def make_query_set(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    query_support: np.ndarray,
    query_log_power: np.ndarray,
    prefix: str,
    indices: np.ndarray | None,
    query_ids: np.ndarray | None,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    bank_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = node_features(prefix, indices, query_pos, query_blocked, query_support, query_log_power)
    idx, dsel, base = candidate_neighbors(query_pos, query_blocked, bank_pos, bank_blocked, query_ids, bank_ids)
    edges = edge_features(query_pos, query_blocked, bank_pos, bank_blocked, idx, dsel)
    return features, idx, edges, base


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support, valid, log_power = v233.load_normalized_train_support()
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v233.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    matched = choose_val_indices(train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", VERSION_NUMBER)
    val_ids = np.intersect1d(matched, valid_ids)
    train_ids = np.setdiff1d(valid_ids, val_ids)

    idx_train, _, w_train = candidate_neighbors(
        train_pos[train_ids], train_blocked[train_ids], train_pos[train_ids], train_blocked[train_ids], train_ids, train_ids
    )
    idx_val, _, w_val = candidate_neighbors(
        train_pos[val_ids], train_blocked[val_ids], train_pos[train_ids], train_blocked[train_ids], None, None
    )
    split_bank = np.asarray(support[train_ids], dtype=np.float32)
    train_anchor = weighted_average_support(split_bank, idx_train, w_train)
    val_anchor = weighted_average_support(split_bank, idx_val, w_val)
    train_anchor_power = np.sum(log_power[train_ids][idx_train] * w_train, axis=1).astype(np.float32)
    val_anchor_power = np.sum(log_power[train_ids][idx_val] * w_val, axis=1).astype(np.float32)

    bank_raw = node_features("train", train_ids, train_pos[train_ids], train_blocked[train_ids], split_bank, log_power[train_ids])
    train_raw = make_query_set(
        train_pos[train_ids], train_blocked[train_ids], train_anchor, train_anchor_power, "train", train_ids,
        train_ids, train_pos[train_ids], train_blocked[train_ids], train_ids,
    )
    val_raw = make_query_set(
        train_pos[val_ids], train_blocked[val_ids], val_anchor, val_anchor_power, "train", val_ids,
        None, train_pos[train_ids], train_blocked[train_ids], None,
    )
    bank_features, [train_query, val_query], split_stats = standardize_nodes_and_edges(bank_raw, [train_raw, val_raw])
    validation_model, validation_report = train_attention(
        train_query,
        np.asarray(support[train_ids], dtype=np.float32),
        bank_features,
        split_bank,
        SEEDS[0],
        TRAIN_EPOCHS,
        device,
        (val_query, np.asarray(support[val_ids], dtype=np.float32)),
    )
    val_prediction, val_weights = predict_attention(validation_model, val_query, bank_features, split_bank, device)
    val_target = np.asarray(support[val_ids], dtype=np.float32)
    val_alpha, calibration_report = v233.calibrate_alpha(val_anchor, val_prediction, val_target, regime_of(train_blocked[val_ids]))
    validation_metrics = {
        "base": v233.support_metrics(val_anchor, val_target),
        "model": v233.support_metrics(val_prediction, val_target),
    }
    best_epoch = max(20, int(validation_report["best_epoch"]))
    del validation_model, val_prediction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_bank = np.asarray(support[valid_ids], dtype=np.float32)
    idx_final, _, w_final = candidate_neighbors(
        train_pos[valid_ids], train_blocked[valid_ids], train_pos[valid_ids], train_blocked[valid_ids], valid_ids, valid_ids
    )
    idx_test, _, w_test = candidate_neighbors(
        test_pos, test_blocked, train_pos[valid_ids], train_blocked[valid_ids], None, None
    )
    final_anchor = weighted_average_support(final_bank, idx_final, w_final)
    test_anchor = weighted_average_support(final_bank, idx_test, w_test)
    final_anchor_power = np.sum(log_power[valid_ids][idx_final] * w_final, axis=1).astype(np.float32)
    test_anchor_power = np.sum(log_power[valid_ids][idx_test] * w_test, axis=1).astype(np.float32)
    final_bank_raw = node_features("train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], final_bank, log_power[valid_ids])
    final_raw = make_query_set(
        train_pos[valid_ids], train_blocked[valid_ids], final_anchor, final_anchor_power, "train", valid_ids,
        valid_ids, train_pos[valid_ids], train_blocked[valid_ids], valid_ids,
    )
    test_raw = make_query_set(
        test_pos, test_blocked, test_anchor, test_anchor_power, "test", None,
        None, train_pos[valid_ids], train_blocked[valid_ids], None,
    )
    final_bank_features, [final_query, test_query], final_stats = standardize_nodes_and_edges(final_bank_raw, [final_raw, test_raw])
    seed_targets = []
    seed_weights = []
    final_training = []
    for seed in SEEDS:
        model, info = train_attention(
            final_query,
            final_bank,
            final_bank_features,
            final_bank,
            seed,
            best_epoch,
            device,
            None,
        )
        torch.save(
            {"state_dict": model.state_dict(), "feature_stats": final_stats, "seed": seed, "epochs": best_epoch},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        target, weights = predict_attention(model, test_query, final_bank_features, final_bank, device)
        seed_targets.append(target)
        seed_weights.append(weights)
        final_training.append(info)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_target = v233.normalize_support(np.mean(np.stack(seed_targets), axis=0))
    np.save(TARGET_OUT, test_target)
    weights_mean = np.mean(np.stack(seed_weights), axis=0).astype(np.float32)
    entropy = -np.sum(weights_mean * np.log(np.maximum(weights_mean, 1e-12)), axis=1) / math.log(weights_mean.shape[1])
    effective_count = 1.0 / np.maximum(np.sum(weights_mean**2, axis=1), TINY)
    current_support = v233.extract_base_support(test_pos)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    confidence = np.clip((0.92 - entropy) / 0.34, 0.0, 1.0).astype(np.float32)
    confidence *= np.clip((row_cos := v233.row_cos(test_target, test_anchor)) - 0.72, 0.0, 0.24) / 0.24
    variants = [
        ("linear_calibrated", np.clip(alpha_base * (0.45 + 0.55 * confidence), 0.0, 0.90), "linear"),
        ("log_calibrated", np.clip(alpha_base * (0.38 + 0.62 * confidence), 0.0, 0.82), "log"),
        ("log_aggressive", np.clip(1.35 * alpha_base * (0.44 + 0.56 * confidence), 0.0, 1.02), "log"),
        (
            "heavy_neighbor_overdrive",
            np.asarray([{"clear": 0.0, "light": 0.08, "medium": 0.42, "heavy": 0.64}[str(r)] for r in regimes], dtype=np.float32)
            * (0.50 + 0.50 * confidence),
            "log",
        ),
    ]
    results = []
    for name, alpha, mode in variants:
        output, diagnostics = v233.apply_target(
            name,
            test_target,
            alpha.astype(np.float32),
            test_pos,
            mode,
            output_dir=OUT_DIR,
            version_prefix="v234",
        )
        results.append(
            {"name": name, "mode": mode, "output": PATHS.relative(output), "score": v208.score_prediction(output), "diagnostics": diagnostics}
        )
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        weights=weights_mean,
        entropy=entropy.astype(np.float32),
        effective_count=effective_count.astype(np.float32),
        confidence=confidence.astype(np.float32),
        target_vs_anchor=row_cos.astype(np.float32),
        target_vs_v226=v233.row_cos(test_target, np.asarray(current_support)),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "The missing propagation mode is identifiable by learning which real official train supports should be combined, without free residual synthesis.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H support bank",
                "official train/test coordinates",
                "official point-cloud scene/geodesic features",
                "official point-cloud multibounce path candidates",
                "v226 strict official-data carrier",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth",
                "hidden online labels",
                "raw RT H",
            ],
        },
        "training": {
            "device": str(device),
            "valid_train_count": int(len(valid_ids)),
            "split_train_count": int(len(train_ids)),
            "validation_count": int(len(val_ids)),
            "neighbors": K_NEIGHBORS,
            "best_epoch": best_epoch,
            "validation": validation_report,
            "validation_metrics": validation_metrics,
            "calibration": calibration_report,
            "alpha_by_regime": val_alpha,
            "final_seeds": list(SEEDS),
            "final_training": final_training,
            "node_dim": int(final_bank_features.shape[1]),
            "edge_dim": int(final_query.edge_features.shape[2]),
        },
        "test_diagnostics": {
            "weight_entropy": summarize(entropy),
            "effective_count": summarize(effective_count),
            "confidence": summarize(confidence),
            "target_vs_anchor": summarize(row_cos),
            "target_vs_v226": summarize(v233.row_cos(test_target, np.asarray(current_support))),
            "anchor_vs_v226": summarize(v233.row_cos(test_anchor, np.asarray(current_support))),
        },
        "variants": results,
        "local_gate": {
            "required_score_traceback_leaked": GATE_SCORE,
            "best_variant": best["name"],
            "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
            "gap_to_gate": best["score"]["score_traceback_leaked"] - GATE_SCORE,
        },
        "outputs": {"target": PATHS.relative(TARGET_OUT), "actions": PATHS.relative(ACTION_OUT), "report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
