#!/usr/bin/env python3
"""Generate v113: overdrive MoE path support target.

v112 reached 0.497283 online, confirming that the kernel-MoE/support-only
branch is still alive.  v113 deliberately makes the same idea more aggressive:
it starts from v112, treats v112 as the current support state, uses the
v112->v111->v109 trajectory as a momentum signal, and lets the MoE gate make a
larger medium/heavy support move.

The experts are deterministic support targets:

1. strict path-token train-kNN target;
2. broad path-token train-kNN target;
3. spatial/geometry train-kNN target;
4. test-test smoothing target from the current v112 support;
5. verified momentum target from v109 -> v111 -> v112;
6. current v112 target as the conservative fallback.

The gate is trained with pure NumPy on training leave-one-out support
prediction: for each training point, the script builds analogous expert
targets from the other training points and learns which expert combination is
closest to the true support.  At test time the learned gate only chooses
between support targets; it never predicts the complex channel directly.

The output keeps the base complex phase and every point's total power.
Only the collapsed joint AD power support P(angle_h, angle_v, delay) is nudged.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import BS_POS, MH, MV, S_SC, summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")
MAT_COUNT = 6
TYPE_COUNT = 4


def normalize_support(x: np.ndarray, floor: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    y = y / np.maximum(y.sum(axis=(1, 2, 3), keepdims=True), TINY)
    if floor > 0:
        y = np.maximum(y, floor).astype(np.float32)
        y /= np.maximum(y.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return y.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1).astype(np.float64)
    bb = b.reshape(b.shape[0], -1).astype(np.float64)
    num = np.sum(aa * bb, axis=1)
    den = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1))
    return num / np.maximum(den, 1e-30)


def path_token_embedding(path_npz: np.lib.npyio.NpzFile, prefix: str, pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    valid = path_npz[f"{prefix}_valid"].astype(np.float32)
    ptype = np.asarray(path_npz[f"{prefix}_path_type"], dtype=np.int32)
    mat = np.asarray(path_npz[f"{prefix}_material_id"], dtype=np.int32)
    delay = np.nan_to_num(np.asarray(path_npz[f"{prefix}_delay_bin"], dtype=np.float32), nan=99.0, posinf=99.0, neginf=99.0)
    score = np.nan_to_num(np.asarray(path_npz[f"{prefix}_score"], dtype=np.float32), nan=-8.0, posinf=4.0, neginf=-8.0)
    length = np.nan_to_num(np.asarray(path_npz[f"{prefix}_path_length"], dtype=np.float32), nan=0.0, posinf=900.0, neginf=0.0)
    excess = np.nan_to_num(np.asarray(path_npz[f"{prefix}_excess_length"], dtype=np.float32), nan=0.0, posinf=900.0, neginf=0.0)
    edge = np.nan_to_num(np.asarray(path_npz[f"{prefix}_edge_margin"], dtype=np.float32), nan=0.0, posinf=0.35, neginf=0.0)
    inc = np.nan_to_num(np.asarray(path_npz[f"{prefix}_incidence_cos"], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    tx = np.nan_to_num(np.asarray(path_npz[f"{prefix}_tx_dir"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rx = np.nan_to_num(np.asarray(path_npz[f"{prefix}_rx_dir"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    n, p = valid.shape
    type_oh = np.zeros((n, p, TYPE_COUNT), dtype=np.float32)
    type_idx = np.clip(ptype + 1, 0, TYPE_COUNT - 1)
    type_oh[np.arange(n)[:, None], np.arange(p)[None, :], type_idx] = 1.0
    mat_oh = np.zeros((n, p, MAT_COUNT), dtype=np.float32)
    mat_idx = np.clip(mat + 1, 0, MAT_COUNT - 1)
    mat_oh[np.arange(n)[:, None], np.arange(p)[None, :], mat_idx] = 1.0

    token = np.concatenate(
        [
            valid[..., None],
            type_oh,
            mat_oh,
            (delay / 24.0)[..., None],
            (np.clip(score, -8.0, 4.0) / 4.0)[..., None],
            (length / 240.0)[..., None],
            (np.log1p(np.maximum(excess, 0.0)) / 7.0)[..., None],
            (edge / 0.35)[..., None],
            inc[..., None],
            tx,
            rx,
        ],
        axis=2,
    ).reshape(n, -1)

    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arctan2(rel[:, 2], np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6))
    reg = regime_of(blocked)
    reg_oh = np.stack([(reg == r).astype(np.float32) for r in REGIMES], axis=1)
    geom = np.column_stack(
        [
            (pos[:, 0] - 120.0) / 120.0,
            (pos[:, 1] + 80.0) / 120.0,
            pos[:, 2] / 30.0,
            dist / 250.0,
            np.log1p(dist) / 6.0,
            np.sin(az),
            np.cos(az),
            np.sin(el),
            np.cos(el),
            blocked / 160.0,
            np.log1p(np.maximum(blocked, 0.0)) / 6.0,
        ]
    ).astype(np.float32)
    return np.nan_to_num(np.concatenate([geom, reg_oh, token], axis=1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_x = np.concatenate([train, test], axis=0)
    med = np.median(all_x, axis=0, keepdims=True).astype(np.float32)
    iqr = (np.quantile(all_x, 0.75, axis=0, keepdims=True) - np.quantile(all_x, 0.25, axis=0, keepdims=True)).astype(np.float32)
    scale = np.maximum(iqr, 1e-3)
    return (
        np.clip((train - med) / scale, -8.0, 8.0).astype(np.float32),
        np.clip((test - med) / scale, -8.0, 8.0).astype(np.float32),
    )


def pairwise_knn(query: np.ndarray, bank: np.ndarray, k: int, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    qn = np.sum(query * query, axis=1, keepdims=True)
    bn = np.sum(bank * bank, axis=1)[None, :]
    d = qn + bn - 2.0 * (query @ bank.T)
    d = np.maximum(d, 0.0).astype(np.float32)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    ids = np.argpartition(d, min(k, bank.shape[0] - 1), axis=1)[:, :k]
    vals = np.take_along_axis(d, ids, axis=1)
    order = np.argsort(vals, axis=1)
    ids = np.take_along_axis(ids, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return ids.astype(np.int64), vals.astype(np.float32)


def weights_from_dist(d: np.ndarray, p: float, floor: float) -> np.ndarray:
    w = 1.0 / np.maximum(np.sqrt(np.maximum(d, 0.0)), floor) ** p
    w = w / np.maximum(w.sum(axis=1, keepdims=True), TINY)
    return w.astype(np.float32)


def spatial_embedding(pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arctan2(rel[:, 2], np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6))
    reg = regime_of(blocked)
    reg_oh = np.stack([(reg == r).astype(np.float32) for r in REGIMES], axis=1)
    feat = np.column_stack(
        [
            (pos[:, 0] - 120.0) / 120.0,
            (pos[:, 1] + 80.0) / 120.0,
            pos[:, 2] / 30.0,
            dist / 250.0,
            np.log1p(dist) / 6.0,
            np.sin(az),
            np.cos(az),
            np.sin(el),
            np.cos(el),
            blocked / 160.0,
            np.log1p(np.maximum(blocked, 0.0)) / 6.0,
        ]
    ).astype(np.float32)
    return np.nan_to_num(np.concatenate([feat, reg_oh], axis=1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def softmax_np(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = x.astype(np.float64) / max(float(temp), 1e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)).astype(np.float32)


def entropy01(w: np.ndarray) -> np.ndarray:
    return (-(w * np.log(np.maximum(w, 1e-12))).sum(axis=1) / math.log(w.shape[1])).astype(np.float32)


def kernel_conf(w: np.ndarray, max_ref: float) -> np.ndarray:
    ent = entropy01(w)
    maxw = w.max(axis=1)
    return (
        np.clip((maxw - 1.0 / w.shape[1]) / max(max_ref, 1e-6), 0.0, 1.0)
        * np.clip((0.94 - ent) / 0.60, 0.0, 1.0)
    ).astype(np.float32)


def gate_features(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    candidates: list[np.ndarray],
    current: np.ndarray,
    confs: list[np.ndarray],
    entropies: list[np.ndarray],
) -> np.ndarray:
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arctan2(rel[:, 2], np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6))
    reg_oh = np.stack([(regimes == r).astype(np.float32) for r in REGIMES], axis=1)
    base = np.column_stack(
        [
            (pos[:, 0] - 120.0) / 120.0,
            (pos[:, 1] + 80.0) / 120.0,
            dist / 250.0,
            np.log1p(dist) / 6.0,
            np.sin(az),
            np.cos(az),
            np.sin(el),
            np.cos(el),
            blocked / 160.0,
            np.log1p(np.maximum(blocked, 0.0)) / 6.0,
        ]
    ).astype(np.float32)
    cos_cur = [row_cos(c, current).astype(np.float32) for c in candidates[:-1]]
    pair_cos = [
        row_cos(candidates[0], candidates[1]).astype(np.float32),
        row_cos(candidates[0], candidates[2]).astype(np.float32),
        row_cos(candidates[1], candidates[2]).astype(np.float32),
        row_cos(candidates[3], current).astype(np.float32),
        row_cos(candidates[4], current).astype(np.float32),
    ]
    agreement = np.minimum.reduce([np.asarray(x) for x in pair_cos[:3]]).astype(np.float32)
    divergence = (1.0 - np.minimum.reduce([np.asarray(x) for x in cos_cur[:4]])).astype(np.float32)
    feat = np.column_stack(
        [
            base,
            reg_oh,
            *confs,
            *entropies,
            *cos_cur,
            *pair_cos,
            agreement,
            divergence,
        ]
    )
    return np.nan_to_num(feat, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def fit_moe_gate(
    train_feat: np.ndarray,
    test_feat: np.ndarray,
    expert_cos: np.ndarray,
    label_temp: float,
    pred_temp: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target = softmax_np(expert_cos, temp=label_temp)
    # Add a little smoothing so the linear gate does not collapse to a single
    # expert on noisy LOO labels.
    target = (0.94 * target + 0.06 / target.shape[1]).astype(np.float32)
    med = np.median(train_feat, axis=0, keepdims=True).astype(np.float32)
    iqr = (np.quantile(train_feat, 0.75, axis=0, keepdims=True) - np.quantile(train_feat, 0.25, axis=0, keepdims=True)).astype(np.float32)
    scale = np.maximum(iqr, 1e-3)
    xtr = np.clip((train_feat - med) / scale, -8.0, 8.0)
    xte = np.clip((test_feat - med) / scale, -8.0, 8.0)
    xtr = np.concatenate([xtr, np.ones((xtr.shape[0], 1), dtype=np.float32)], axis=1).astype(np.float64)
    xte = np.concatenate([xte, np.ones((xte.shape[0], 1), dtype=np.float32)], axis=1).astype(np.float64)
    y = np.log(np.maximum(target, 1e-4)).astype(np.float64)
    xtx = xtr.T @ xtr
    xtx.flat[:: xtx.shape[0] + 1] += float(ridge)
    w = np.linalg.solve(xtx, xtr.T @ y)
    train_prob = softmax_np((xtr @ w).astype(np.float32), temp=pred_temp)
    test_prob = softmax_np((xte @ w).astype(np.float32), temp=pred_temp)
    pred_expert = np.argmax(train_prob, axis=1)
    oracle_expert = np.argmax(expert_cos, axis=1)
    report = {
        "oracle_expert_fraction": {str(i): float(np.mean(oracle_expert == i)) for i in range(expert_cos.shape[1])},
        "train_pred_expert_fraction": {str(i): float(np.mean(pred_expert == i)) for i in range(expert_cos.shape[1])},
        "test_pred_expert_fraction": {str(i): float(np.mean(np.argmax(test_prob, axis=1) == i)) for i in range(expert_cos.shape[1])},
        "train_gate_top1_match": float(np.mean(pred_expert == oracle_expert)),
        "train_target_entropy": summarize(entropy01(target)),
        "train_pred_entropy": summarize(entropy01(train_prob)),
        "test_pred_entropy": summarize(entropy01(test_prob)),
    }
    return train_prob.astype(np.float32), test_prob.astype(np.float32), report


def log_average(bank: np.ndarray, idx: np.ndarray, weights: np.ndarray, floor: float, batch: int) -> np.ndarray:
    out = np.empty((idx.shape[0], MH, MV, S_SC), dtype=np.float32)
    for s in range(0, idx.shape[0], batch):
        e = min(s + batch, idx.shape[0])
        vals = np.log(np.maximum(bank[idx[s:e]], floor)).astype(np.float32)
        mix = np.sum(weights[s:e, :, None, None, None] * vals, axis=1)
        pred = np.exp(mix).astype(np.float32)
        pred /= np.maximum(pred.sum(axis=(1, 2, 3), keepdims=True), TINY)
        out[s:e] = pred
        print(f"log-average {e}/{idx.shape[0]}", flush=True)
    return out


def log_blend(targets: list[np.ndarray], coeffs: np.ndarray, floor: float, batch: int) -> np.ndarray:
    n = targets[0].shape[0]
    out = np.empty_like(targets[0], dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        acc = np.zeros((e - s, MH, MV, S_SC), dtype=np.float32)
        for j, tgt in enumerate(targets):
            acc += coeffs[s:e, j, None, None, None] * np.log(np.maximum(tgt[s:e], floor)).astype(np.float32)
        pred = np.exp(acc).astype(np.float32)
        pred /= np.maximum(pred.sum(axis=(1, 2, 3), keepdims=True), TINY)
        out[s:e] = pred
    return out


def signed_row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1).astype(np.float64)
    bb = b.reshape(b.shape[0], -1).astype(np.float64)
    num = np.sum(aa * bb, axis=1)
    den = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1))
    return num / np.maximum(den, 1e-30)


def momentum_support(
    current: np.ndarray,
    prev: np.ndarray,
    prev2: np.ndarray,
    regimes: np.ndarray,
    agreement: np.ndarray,
    divergence: np.ndarray,
    floor: float,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrapolate only when the last two verified support moves align."""
    n = current.shape[0]
    log_cur = np.log(np.maximum(current, floor)).astype(np.float32)
    log_prev = np.log(np.maximum(prev, floor)).astype(np.float32)
    log_prev2 = np.log(np.maximum(prev2, floor)).astype(np.float32)
    d1 = log_cur - log_prev
    d0 = log_prev - log_prev2
    trend = np.clip((signed_row_cos(d1, d0) + 0.10) / 0.85, 0.0, 1.0).astype(np.float32)
    reg_strength = {"clear": 0.00, "light": 0.030, "medium": 0.72, "heavy": 1.06}
    base = np.array([reg_strength[str(r)] for r in regimes], dtype=np.float32)
    gate = base * trend * np.clip(agreement, 0.0, 1.0) * np.clip(divergence, 0.0, 1.0)
    gate = np.clip(gate, 0.0, 1.08).astype(np.float32)

    out = np.empty_like(current, dtype=np.float32)
    # Clip component-wise momentum so a few tiny bins do not explode.
    direction = np.clip(1.48 * d1 + 0.82 * d0, -3.10, 3.10).astype(np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        mix = log_cur[s:e] + gate[s:e, None, None, None] * direction[s:e]
        pred = np.exp(mix).astype(np.float32)
        pred /= np.maximum(pred.sum(axis=(1, 2, 3), keepdims=True), TINY)
        out[s:e] = pred
    return out, trend, gate


def train_mix_support(truth: np.ndarray, v98like: np.ndarray, blocked: np.ndarray, floor: float, batch: int) -> np.ndarray:
    reg = regime_of(blocked)
    truth_weight = {"clear": 0.30, "light": 0.24, "medium": 0.18, "heavy": 0.14}
    a = np.array([truth_weight[str(r)] for r in reg], dtype=np.float32)
    out = np.empty_like(v98like, dtype=np.float32)
    for s in range(0, len(v98like), batch):
        e = min(s + batch, len(v98like))
        aa = a[s:e, None, None, None]
        mix = aa * np.log(np.maximum(truth[s:e], floor)).astype(np.float32)
        mix += (1.0 - aa) * np.log(np.maximum(v98like[s:e], floor)).astype(np.float32)
        pred = np.exp(mix).astype(np.float32)
        pred /= np.maximum(pred.sum(axis=(1, 2, 3), keepdims=True), TINY)
        out[s:e] = pred
    return out


def blend_coefficients(
    regimes: np.ndarray,
    conf_train: np.ndarray,
    conf_test: np.ndarray,
    agreement: np.ndarray,
    divergence: np.ndarray,
    anchor_agreement: np.ndarray,
) -> np.ndarray:
    # Columns: current-best target, test-test smoothed target, manual train
    # kernel target, momentum target, v108 anchor, v107 anchor, v106 anchor,
    # v105 anchor, v104 anchor, v102 anchor.  There is intentionally no v98
    # learned-attention target column.
    base = {
        "clear": np.array([0.9660, 0.0120, 0.0100, 0.0000, 0.0060, 0.0030, 0.0015, 0.0008, 0.0005, 0.0002], dtype=np.float32),
        "light": np.array([0.9180, 0.0280, 0.0260, 0.0040, 0.0080, 0.0050, 0.0030, 0.0020, 0.0012, 0.0008], dtype=np.float32),
        "medium": np.array([0.1500, 0.2800, 0.3300, 0.1900, 0.0180, 0.0120, 0.0080, 0.0060, 0.0040, 0.0020], dtype=np.float32),
        "heavy": np.array([0.0950, 0.3150, 0.3650, 0.1800, 0.0180, 0.0110, 0.0070, 0.0045, 0.0028, 0.0017], dtype=np.float32),
    }
    coeff = np.stack([base[str(r)] for r in regimes], axis=0)
    # v108 is high risk, but the extra movement is mostly controlled by the
    # explicit momentum target.  Low-confidence points fall back toward current.
    overdrive = np.clip(agreement, 0.0, 1.0) * np.clip(divergence, 0.0, 1.0)
    q = (
        0.58 * overdrive
        + 0.14 * np.clip(agreement, 0.0, 1.0)
        + 0.08 * np.clip(anchor_agreement, 0.0, 1.0)
        + 0.10 * np.clip(conf_train, 0.0, 1.0)
        + 0.10 * np.clip(conf_test, 0.0, 1.0)
    ).astype(np.float32)
    c = (0.44 + 0.56 * q).astype(np.float32)
    nonbase = coeff[:, 1:].sum(axis=1)
    coeff[:, 0] += nonbase * (1.0 - c)
    coeff[:, 1:] *= c[:, None]
    coeff /= np.maximum(coeff.sum(axis=1, keepdims=True), TINY)
    return coeff.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--path-candidates", default=ASSETS + "/sionna_path_candidates.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v112/v112_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=str(PATHS.cache_dir(113) / "v113_Round1_Test_Channel.npy"))
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-current-support", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--v109-anchor", default=ASSETS + "/cache_v111/v111_manual_kernel_path_support_target.npy")
    ap.add_argument("--v108-anchor", default=ASSETS + "/cache_v109/v109_accelerated_momentum_path_support_target.npy")
    ap.add_argument("--v107-anchor", default=ASSETS + "/cache_v107/v107_recursive_overdrive_path_support_target.npy")
    ap.add_argument("--v106-anchor", default=ASSETS + "/cache_v106/v106_agreement_overdrive_path_support_target.npy")
    ap.add_argument("--v105-anchor", default=ASSETS + "/cache_v105/v105_selective_ultra_path_support_target.npy")
    ap.add_argument("--v104-anchor", default=ASSETS + "/cache_v104/v104_aggressive_transductive_path_support_target.npy")
    ap.add_argument("--v102-anchor", default=ASSETS + "/cache_v102/v102_v98like_support_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v113/v113_overdrive_moe_path_support_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v113/v113_overdrive_moe_path_support_weights.npy")
    ap.add_argument("--moe-out", default=ASSETS + "/cache_v113/v113_overdrive_moe_path_support_moe_weights.npy")
    ap.add_argument("--report", default=ASSETS + "/v113_overdrive_moe_path_support_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v113_overdrive_moe_path_support_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v113_overdrive_moe_path_support.zip"))
    ap.add_argument("--k-strict", type=int, default=8)
    ap.add_argument("--k-broad", type=int, default=42)
    ap.add_argument("--k-spatial", type=int, default=28)
    ap.add_argument("--k-test", type=int, default=8)
    ap.add_argument("--strict-power", type=float, default=4.10)
    ap.add_argument("--broad-power", type=float, default=2.20)
    ap.add_argument("--spatial-power", type=float, default=2.65)
    ap.add_argument("--test-power", type=float, default=3.35)
    ap.add_argument("--strict-floor", type=float, default=0.30)
    ap.add_argument("--broad-floor", type=float, default=0.78)
    ap.add_argument("--spatial-floor", type=float, default=0.60)
    ap.add_argument("--test-floor", type=float, default=0.42)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--eta", default="clear:0.0008,light:0.0060,medium:0.900,heavy:1.320")
    ap.add_argument("--min-gate", type=float, default=0.26)
    ap.add_argument("--max-gate", type=float, default=3.25)
    ap.add_argument("--conf-center", type=float, default=0.19)
    ap.add_argument("--conf-width", type=float, default=0.13)
    ap.add_argument("--entropy-low", type=float, default=0.14)
    ap.add_argument("--entropy-high", type=float, default=0.88)
    ap.add_argument("--gain-clip", default="0.32,4.10")
    ap.add_argument("--top-gain-clip", default="0.48,3.05")
    ap.add_argument("--top-components", type=int, default=3072)
    ap.add_argument("--moe-label-temp", type=float, default=0.026)
    ap.add_argument("--moe-pred-temp", type=float, default=0.62)
    ap.add_argument("--moe-ridge", type=float, default=9.0)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()
    args.target_floor = float(args.floor)

    data = Path(args.data)
    pos_train = np.load(data / "Round1_Train_Pos.npy")
    pos_test = np.load(data / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    path_npz = np.load(args.path_candidates)

    print("load supports", flush=True)
    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    train_current = normalize_support(np.load(args.train_current_support, mmap_mode="r"), float(args.floor))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    v109_anchor = normalize_support(np.load(args.v109_anchor), float(args.floor))
    v108_anchor = normalize_support(np.load(args.v108_anchor), float(args.floor))
    v107_anchor = normalize_support(np.load(args.v107_anchor), float(args.floor))
    v106_anchor = normalize_support(np.load(args.v106_anchor), float(args.floor))
    v105_anchor = normalize_support(np.load(args.v105_anchor), float(args.floor))
    v104_anchor = normalize_support(np.load(args.v104_anchor), float(args.floor))
    v102_anchor = normalize_support(np.load(args.v102_anchor), float(args.floor))

    print("build embeddings", flush=True)
    path_train, path_test = standardize(
        path_token_embedding(path_npz, "train", pos_train, blocked_train),
        path_token_embedding(path_npz, "test", pos_test, blocked_test),
    )
    spatial_train, spatial_test = standardize(
        spatial_embedding(pos_train, blocked_train),
        spatial_embedding(pos_test, blocked_test),
    )

    print("knn: path strict/broad, spatial, test smooth", flush=True)
    idx_strict_tr, dist_strict_tr = pairwise_knn(path_train, path_train, int(args.k_strict), exclude_self=True)
    idx_broad_tr, dist_broad_tr = pairwise_knn(path_train, path_train, int(args.k_broad), exclude_self=True)
    idx_spatial_tr, dist_spatial_tr = pairwise_knn(spatial_train, spatial_train, int(args.k_spatial), exclude_self=True)
    idx_smooth_tr, dist_smooth_tr = pairwise_knn(path_train, path_train, int(args.k_test), exclude_self=True)

    idx_strict_te, dist_strict_te = pairwise_knn(path_test, path_train, int(args.k_strict), exclude_self=False)
    idx_broad_te, dist_broad_te = pairwise_knn(path_test, path_train, int(args.k_broad), exclude_self=False)
    idx_spatial_te, dist_spatial_te = pairwise_knn(spatial_test, spatial_train, int(args.k_spatial), exclude_self=False)
    idx_test_te, dist_test_te = pairwise_knn(path_test, path_test, int(args.k_test) + 1, exclude_self=True)
    idx_test_te = idx_test_te[:, : int(args.k_test)]
    dist_test_te = dist_test_te[:, : int(args.k_test)]

    w_strict_tr = weights_from_dist(dist_strict_tr, float(args.strict_power), float(args.strict_floor))
    w_broad_tr = weights_from_dist(dist_broad_tr, float(args.broad_power), float(args.broad_floor))
    w_spatial_tr = weights_from_dist(dist_spatial_tr, float(args.spatial_power), float(args.spatial_floor))
    w_smooth_tr = weights_from_dist(dist_smooth_tr, float(args.test_power), float(args.test_floor))

    w_strict_te = weights_from_dist(dist_strict_te, float(args.strict_power), float(args.strict_floor))
    w_broad_te = weights_from_dist(dist_broad_te, float(args.broad_power), float(args.broad_floor))
    w_spatial_te = weights_from_dist(dist_spatial_te, float(args.spatial_power), float(args.spatial_floor))
    w_test_te = weights_from_dist(dist_test_te, float(args.test_power), float(args.test_floor))

    print("build train LOO expert targets", flush=True)
    tr_strict = log_average(truth, idx_strict_tr, w_strict_tr, float(args.floor), int(args.batch))
    tr_broad = log_average(truth, idx_broad_tr, w_broad_tr, float(args.floor), int(args.batch))
    tr_spatial = log_average(truth, idx_spatial_tr, w_spatial_tr, float(args.floor), int(args.batch))
    tr_smooth = log_average(train_current, idx_smooth_tr, w_smooth_tr, float(args.floor), int(args.batch))
    tr_current = train_current
    tr_momentum = train_current

    print("build test expert targets", flush=True)
    te_strict = log_average(truth, idx_strict_te, w_strict_te, float(args.floor), int(args.batch))
    te_broad = log_average(truth, idx_broad_te, w_broad_te, float(args.floor), int(args.batch))
    te_spatial = log_average(truth, idx_spatial_te, w_spatial_te, float(args.floor), int(args.batch))
    te_smooth = log_average(current, idx_test_te, w_test_te, float(args.floor), int(args.batch))

    agreement = np.clip((np.minimum(row_cos(te_strict, te_broad), row_cos(te_strict, te_spatial)) - 0.68) / 0.20, 0.0, 1.0)
    divergence = np.clip((1.0 - np.minimum.reduce([row_cos(te_strict, current), row_cos(te_broad, current), row_cos(te_spatial, current)])) / 0.14, 0.0, 1.0)
    te_momentum, momentum_trend, momentum_gate = momentum_support(
        current, v109_anchor, v108_anchor, regimes_test, agreement, divergence, float(args.floor), int(args.batch)
    )

    train_candidates = [tr_strict, tr_broad, tr_spatial, tr_smooth, tr_momentum, tr_current]
    test_candidates = [te_strict, te_broad, te_spatial, te_smooth, te_momentum, current]
    expert_names = ["path_strict", "path_broad", "spatial", "smooth_current", "momentum", "current"]
    expert_cos = np.stack([row_cos(c, truth) for c in train_candidates], axis=1).astype(np.float32)

    train_confs = [kernel_conf(w_strict_tr, 0.32), kernel_conf(w_broad_tr, 0.24), kernel_conf(w_spatial_tr, 0.26), kernel_conf(w_smooth_tr, 0.40)]
    test_confs = [kernel_conf(w_strict_te, 0.32), kernel_conf(w_broad_te, 0.24), kernel_conf(w_spatial_te, 0.26), kernel_conf(w_test_te, 0.40)]
    train_ents = [entropy01(w_strict_tr), entropy01(w_broad_tr), entropy01(w_spatial_tr), entropy01(w_smooth_tr)]
    test_ents = [entropy01(w_strict_te), entropy01(w_broad_te), entropy01(w_spatial_te), entropy01(w_test_te)]
    train_feat = gate_features(pos_train, blocked_train, regimes_train, train_candidates, tr_current, train_confs, train_ents)
    test_feat = gate_features(pos_test, blocked_test, regimes_test, test_candidates, current, test_confs, test_ents)
    train_prob, test_prob, moe_report = fit_moe_gate(
        train_feat,
        test_feat,
        expert_cos,
        float(args.moe_label_temp),
        float(args.moe_pred_temp),
        float(args.moe_ridge),
    )
    moe_target = log_blend(test_candidates, test_prob, float(args.floor), int(args.batch))

    moe_conf = np.clip(1.0 - test_prob[:, -1], 0.0, 1.0)
    max_prob = test_prob.max(axis=1)
    shape_conf = np.clip((row_cos(moe_target, current) - 0.90) / 0.08, 0.0, 1.0)
    reg_strength = {"clear": 0.045, "light": 0.120, "medium": 0.780, "heavy": 0.960}
    strength = np.array([reg_strength[str(r)] for r in regimes_test], dtype=np.float32)
    strength *= (0.55 + 0.45 * np.clip((max_prob - 0.25) / 0.45, 0.0, 1.0)).astype(np.float32)
    strength *= (0.68 + 0.32 * moe_conf).astype(np.float32)
    strength *= (0.54 + 0.46 * shape_conf).astype(np.float32)
    strength = np.clip(strength, 0.0, 0.970).astype(np.float32)
    final_coeff = np.zeros((len(current), 4), dtype=np.float32)
    final_coeff[:, 0] = 1.0 - strength
    final_coeff[:, 1] = 0.955 * strength
    final_coeff[:, 2] = 0.030 * strength
    final_coeff[:, 3] = 0.015 * strength
    target = log_blend([current, moe_target, v109_anchor, v108_anchor], final_coeff, float(args.floor), int(args.batch))

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target)
    np.save(args.weights_out, w_strict_te.astype(np.float32))
    np.save(args.moe_out, test_prob.astype(np.float32))

    proj = projection(target, w_strict_te.astype(np.float32), regimes_test, pos_test, args)
    payload = {
        "method": "v113_overdrive_moe_path_support",
        "base": args.base,
        "operation": "overdrive MoE gate over support targets on top of v112; phase/scalar/total power preserved",
        "args": vars(args),
        "experts": expert_names,
        "moe": moe_report,
        "train_expert_cos": {name: summarize(expert_cos[:, i]) for i, name in enumerate(expert_names)},
        "test_moe_weight": {name: summarize(test_prob[:, i]) for i, name in enumerate(expert_names)},
        "graph": {
            "strict_weight_max": summarize(w_strict_te.max(axis=1)),
            "broad_weight_max": summarize(w_broad_te.max(axis=1)),
            "spatial_weight_max": summarize(w_spatial_te.max(axis=1)),
            "test_weight_max": summarize(w_test_te.max(axis=1)),
            "strict_entropy": summarize(entropy01(w_strict_te)),
            "broad_entropy": summarize(entropy01(w_broad_te)),
            "spatial_entropy": summarize(entropy01(w_spatial_te)),
            "test_entropy": summarize(entropy01(w_test_te)),
            "momentum_trend": summarize(momentum_trend),
            "momentum_gate": summarize(momentum_gate),
            "moe_strength": summarize(strength),
            "moe_conf_noncurrent": summarize(moe_conf),
            "moe_shape_conf": summarize(shape_conf),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "moe_vs_current": summarize(row_cos(moe_target, current)),
            "target_vs_moe": summarize(row_cos(target, moe_target)),
            "target_vs_v109": summarize(row_cos(target, v109_anchor)),
            "target_vs_v108": summarize(row_cos(target, v108_anchor)),
            "strict_vs_current": summarize(row_cos(te_strict, current)),
            "broad_vs_current": summarize(row_cos(te_broad, current)),
            "spatial_vs_current": summarize(row_cos(te_spatial, current)),
            "smooth_vs_current": summarize(row_cos(te_smooth, current)),
            "momentum_vs_current": summarize(row_cos(te_momentum, current)),
            "strict_broad_agreement": summarize(row_cos(te_strict, te_broad)),
            "strict_spatial_agreement": summarize(row_cos(te_strict, te_spatial)),
            "broad_spatial_agreement": summarize(row_cos(te_broad, te_spatial)),
        },
        "projection": proj,
        "outputs": {"submission_npy": args.output, "target": args.target_out, "weights": args.weights_out, "moe": args.moe_out},
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    def write_zip(path: str) -> None:
        files = [
            (args.output, "Round1_Test_Channel.npy"),
            (__file__, "generate_v113_overdrive_moe_path_support.py"),
            (ASSETS + "/generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
            (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for src, arc in files:
                zf.write(src, arc)
        print(f"wrote {path}", flush=True)

    write_zip(args.zip_out)
    if args.candidate_zip:
        write_zip(args.candidate_zip)


if __name__ == "__main__":
    main()
