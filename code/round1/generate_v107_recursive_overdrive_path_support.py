#!/usr/bin/env python3
"""Generate v107: recursive agreement-overdrive path-token support projection.

v106 crossed 0.49 online, proving that the agreement-overdrive support branch
still has headroom.  v107 recursively applies the same idea on top of v106:

1. base is the current best v106 output;
2. the v106 support target is the current anchor;
3. v105/v104/v102/v98 support targets are kept as history anchors;
4. test-test smoothed support and train path-token support are pushed harder in
   medium/heavy points when they agree and diverge from v106;
5. clear/light stay almost frozen.

The output still keeps the base complex phase and every point's total power.
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
    # Columns: current-best target, test-test smoothed target, train path-token
    # target, v98 sharp target, v105 anchor, v104 anchor, v102 anchor.
    base = {
        "clear": np.array([0.93, 0.025, 0.025, 0.006, 0.009, 0.004, 0.001], dtype=np.float32),
        "light": np.array([0.84, 0.060, 0.060, 0.012, 0.018, 0.007, 0.003], dtype=np.float32),
        "medium": np.array([0.22, 0.370, 0.350, 0.040, 0.010, 0.007, 0.003], dtype=np.float32),
        "heavy": np.array([0.12, 0.440, 0.390, 0.040, 0.006, 0.003, 0.001], dtype=np.float32),
    }
    coeff = np.stack([base[str(r)] for r in regimes], axis=0)
    # v107 is deliberately high risk: when new train/test targets agree and
    # v106 still differs from them, let new information dominate medium/heavy.
    overdrive = np.clip(agreement, 0.0, 1.0) * np.clip(divergence, 0.0, 1.0)
    q = (
        0.62 * overdrive
        + 0.12 * np.clip(agreement, 0.0, 1.0)
        + 0.08 * np.clip(anchor_agreement, 0.0, 1.0)
        + 0.09 * np.clip(conf_train, 0.0, 1.0)
        + 0.09 * np.clip(conf_test, 0.0, 1.0)
    ).astype(np.float32)
    c = (0.46 + 0.54 * q).astype(np.float32)
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
    ap.add_argument("--base", default=ASSETS + "/cache_v106/v106_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=str(PATHS.cache_dir(107) / "v107_Round1_Test_Channel.npy"))
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--v98like-train-support", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v106/v106_agreement_overdrive_path_support_target.npy")
    ap.add_argument("--v105-anchor", default=ASSETS + "/cache_v105/v105_selective_ultra_path_support_target.npy")
    ap.add_argument("--v104-anchor", default=ASSETS + "/cache_v104/v104_aggressive_transductive_path_support_target.npy")
    ap.add_argument("--v102-anchor", default=ASSETS + "/cache_v102/v102_v98like_support_target.npy")
    ap.add_argument("--v98-target", default=ASSETS + "/cache_v98/v98_path_ensemble_graph_attention_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v107/v107_recursive_overdrive_path_support_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v107/v107_recursive_overdrive_path_weights.npy")
    ap.add_argument("--report", default=ASSETS + "/v107_recursive_overdrive_path_support_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v107_recursive_overdrive_path_support_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v107_recursive_overdrive_path_support.zip"))
    ap.add_argument("--k-train", type=int, default=24)
    ap.add_argument("--k-test", type=int, default=10)
    ap.add_argument("--dist-power", type=float, default=2.2)
    ap.add_argument("--dist-floor", type=float, default=0.55)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--eta", default="clear:0.002,light:0.012,medium:0.500,heavy:0.840")
    ap.add_argument("--min-gate", type=float, default=0.36)
    ap.add_argument("--max-gate", type=float, default=2.45)
    ap.add_argument("--conf-center", type=float, default=0.25)
    ap.add_argument("--conf-width", type=float, default=0.18)
    ap.add_argument("--entropy-low", type=float, default=0.18)
    ap.add_argument("--entropy-high", type=float, default=0.88)
    ap.add_argument("--gain-clip", default="0.44,2.85")
    ap.add_argument("--top-gain-clip", default="0.60,2.05")
    ap.add_argument("--top-components", type=int, default=896)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()
    # The shared v98 projection helper expects this field name.
    args.target_floor = float(args.floor)

    data = Path(args.data)
    pos_train = np.load(data / "Round1_Train_Pos.npy")
    pos_test = np.load(data / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_test)
    path_npz = np.load(args.path_candidates)

    print("load supports", flush=True)
    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    v98like_train = normalize_support(np.load(args.v98like_train_support, mmap_mode="r"), float(args.floor))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    v105_anchor = normalize_support(np.load(args.v105_anchor), float(args.floor))
    v104_anchor = normalize_support(np.load(args.v104_anchor), float(args.floor))
    v102_anchor = normalize_support(np.load(args.v102_anchor), float(args.floor))
    v98 = normalize_support(np.load(args.v98_target), float(args.floor))
    train_mix = train_mix_support(truth, v98like_train, blocked_train, float(args.floor), int(args.batch))

    print("build path-token embeddings", flush=True)
    emb_train, emb_test = standardize(
        path_token_embedding(path_npz, "train", pos_train, blocked_train),
        path_token_embedding(path_npz, "test", pos_test, blocked_test),
    )
    idx_tr, dist_tr = pairwise_knn(emb_test, emb_train, int(args.k_train), exclude_self=False)
    idx_tt, dist_tt = pairwise_knn(emb_test, emb_test, int(args.k_test) + 1, exclude_self=True)
    idx_tt = idx_tt[:, : int(args.k_test)]
    dist_tt = dist_tt[:, : int(args.k_test)]
    w_tr = weights_from_dist(dist_tr, float(args.dist_power), float(args.dist_floor))
    w_tt = weights_from_dist(dist_tt, float(args.dist_power), float(args.dist_floor))

    train_target = log_average(train_mix, idx_tr, w_tr, float(args.floor), int(args.batch))
    test_smooth = log_average(current, idx_tt, w_tt, float(args.floor), int(args.batch))

    entropy_tr = -(w_tr * np.log(np.maximum(w_tr, 1e-12))).sum(axis=1) / math.log(w_tr.shape[1])
    entropy_tt = -(w_tt * np.log(np.maximum(w_tt, 1e-12))).sum(axis=1) / math.log(w_tt.shape[1])
    conf_tr = np.clip((w_tr.max(axis=1) - 1.0 / w_tr.shape[1]) / 0.32, 0.0, 1.0) * np.clip((0.92 - entropy_tr) / 0.55, 0.0, 1.0)
    conf_tt = np.clip((w_tt.max(axis=1) - 1.0 / w_tt.shape[1]) / 0.40, 0.0, 1.0) * np.clip((0.92 - entropy_tt) / 0.55, 0.0, 1.0)
    train_test_cos = row_cos(train_target, test_smooth)
    train_current_cos = row_cos(train_target, current)
    test_current_cos = row_cos(test_smooth, current)
    train_v105_cos = row_cos(train_target, v105_anchor)
    test_v105_cos = row_cos(test_smooth, v105_anchor)
    train_v104_cos = row_cos(train_target, v104_anchor)
    test_v104_cos = row_cos(test_smooth, v104_anchor)
    train_v102_cos = row_cos(train_target, v102_anchor)
    test_v102_cos = row_cos(test_smooth, v102_anchor)
    agreement = np.clip((train_test_cos - 0.68) / 0.22, 0.0, 1.0).astype(np.float32)
    divergence = np.clip((1.0 - np.minimum(train_current_cos, test_current_cos)) / 0.14, 0.0, 1.0).astype(np.float32)
    anchor_agreement = np.clip((0.5 * (train_v105_cos + test_v105_cos) - 0.52) / 0.35, 0.0, 1.0)
    anchor_agreement *= np.clip((0.5 * (train_v104_cos + test_v104_cos) - 0.44) / 0.38, 0.0, 1.0)
    anchor_agreement *= np.clip((0.5 * (train_v102_cos + test_v102_cos) - 0.32) / 0.46, 0.0, 1.0)
    anchor_agreement = anchor_agreement.astype(np.float32)
    coeff = blend_coefficients(regimes, conf_tr, conf_tt, agreement, divergence, anchor_agreement)
    target = log_blend([current, test_smooth, train_target, v98, v105_anchor, v104_anchor, v102_anchor], coeff, float(args.floor), int(args.batch))

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target)
    # Projection confidence uses train graph weights; it is also what the report
    # exposes as the support-neighbor confidence.
    np.save(args.weights_out, w_tr.astype(np.float32))

    proj = projection(target, w_tr.astype(np.float32), regimes, pos_test, args)
    payload = {
        "method": "v107_recursive_overdrive_path_support",
        "base": args.base,
        "operation": "recursive support-only projection on top of v106 using stronger train/test path-token overdrive",
        "args": vars(args),
        "graph": {
            "train_weight_max": summarize(w_tr.max(axis=1)),
            "test_weight_max": summarize(w_tt.max(axis=1)),
            "train_entropy": summarize(entropy_tr),
            "test_entropy": summarize(entropy_tt),
            "conf_train": summarize(conf_tr),
            "conf_test": summarize(conf_tt),
            "train_test_agreement_raw": summarize(train_test_cos),
            "train_current_cos": summarize(train_current_cos),
            "test_current_cos": summarize(test_current_cos),
            "train_v105_anchor_cos": summarize(train_v105_cos),
            "test_v105_anchor_cos": summarize(test_v105_cos),
            "train_v104_anchor_cos": summarize(train_v104_cos),
            "test_v104_anchor_cos": summarize(test_v104_cos),
            "train_v102_anchor_cos": summarize(train_v102_cos),
            "test_v102_anchor_cos": summarize(test_v102_cos),
            "agreement_gate": summarize(agreement),
            "divergence_gate": summarize(divergence),
            "anchor_agreement_gate": summarize(anchor_agreement),
            "overdrive_gate": summarize(agreement * divergence),
            "coeff_current": summarize(coeff[:, 0]),
            "coeff_test_smooth": summarize(coeff[:, 1]),
            "coeff_train": summarize(coeff[:, 2]),
            "coeff_v98": summarize(coeff[:, 3]),
            "coeff_v105_anchor": summarize(coeff[:, 4]),
            "coeff_v104_anchor": summarize(coeff[:, 5]),
            "coeff_v102_anchor": summarize(coeff[:, 6]),
        },
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "target_vs_v105_anchor": summarize(row_cos(target, v105_anchor)),
            "target_vs_v104_anchor": summarize(row_cos(target, v104_anchor)),
            "target_vs_v102_anchor": summarize(row_cos(target, v102_anchor)),
            "target_vs_v98": summarize(row_cos(target, v98)),
            "current_vs_v105_anchor": summarize(row_cos(current, v105_anchor)),
            "current_vs_v104_anchor": summarize(row_cos(current, v104_anchor)),
            "current_vs_v102_anchor": summarize(row_cos(current, v102_anchor)),
            "train_target_vs_current": summarize(row_cos(train_target, current)),
            "test_smooth_vs_current": summarize(row_cos(test_smooth, current)),
        },
        "projection": proj,
        "outputs": {"submission_npy": args.output, "target": args.target_out, "weights": args.weights_out},
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    def write_zip(path: str) -> None:
        files = [
            (args.output, "Round1_Test_Channel.npy"),
            (__file__, "generate_v107_recursive_overdrive_path_support.py"),
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
