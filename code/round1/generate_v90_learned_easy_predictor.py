#!/usr/bin/env python3
"""Generate v90: learned easy-target predictor + v87 projection.

The online-positive v87 result showed that non-coherent easy targets help when
projected on top of the hard v83 phase/component baseline.  v90 keeps the v87
projection mechanism but replaces the raw geometry-gated kNN target with a
learned residual predictor.

The model does NOT predict complex channels.  It predicts only corrections to:
  - normalized PAS/beam shape,
  - normalized PDP/delay shape,
  - BS polarization energy split,
  - UE antenna energy split,
  - top4/top16 AD energy fraction.

Training labels are derived from the 2000 training channels in PHV AD domain.
Features include coordinates, map/Sionna geometry features, path summaries, and
the leave-one-out kNN easy-target prediction.  A validation split chooses a
conservative residual shrink per target block before final training on all data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import PATHS
from powerfield_common import ASSETS, TINY, geom_effective_dist, regime_of
from generate_v87_easy_field_projection import (
    MH,
    MV,
    MP,
    N_ANT,
    S_SC,
    BS_POS,
    ad_transform,
    apply_ratio,
    beta_subcarriers,
    build_easy_cache,
    inverse_ad,
    marginal_stats_from_power,
    row_cos,
    summarize,
    top_masks_from_collapsed,
)


def require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.preprocessing import RobustScaler
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "v90 needs scikit-learn. Run it through code/tools/run_in_env.sh."
        ) from exc
    return ExtraTreesRegressor, RobustScaler


def mean_pair_cos(v: np.ndarray) -> float:
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + TINY)
    g = vn @ vn.T
    iu = np.triu_indices(len(v), 1)
    return float(g[iu].mean()) if len(iu[0]) else 1.0


def shape_entropy(x: np.ndarray) -> np.ndarray:
    x = x / np.maximum(x.sum(axis=1, keepdims=True), TINY)
    return (-np.sum(x * np.log(x + TINY), axis=1)).astype(np.float32)


def predict_knn_easy(
    bank: dict[str, np.ndarray],
    pos_train: np.ndarray,
    blocked_train: np.ndarray,
    pos_query: np.ndarray,
    blocked_query: np.ndarray,
    k: int,
    p: float,
    exclude_self: bool = False,
) -> dict[str, np.ndarray]:
    n = len(pos_query)
    out = {
        "total": np.empty(n, dtype=np.float32),
        "beam": np.empty((n, MH * MV), dtype=np.float32),
        "delay": np.empty((n, S_SC), dtype=np.float32),
        "pol": np.empty((n, MP), dtype=np.float32),
        "ue": np.empty((n, N_ANT), dtype=np.float32),
        "top4": np.empty(n, dtype=np.float32),
        "top16": np.empty(n, dtype=np.float32),
        "agree_beam": np.empty(n, dtype=np.float32),
        "agree_delay": np.empty(n, dtype=np.float32),
        "eff_nn": np.empty(n, dtype=np.float32),
        "eff_k": np.empty(n, dtype=np.float32),
        "std_log_total": np.empty(n, dtype=np.float32),
    }
    for i in range(n):
        eff = geom_effective_dist(pos_train, blocked_train, pos_query[i, :2], blocked_query[i]).copy()
        eff[bank["total"] <= 0] = np.inf
        if exclude_self:
            eff[i] = np.inf
        idx = np.argpartition(eff, k)[:k]
        idx = idx[np.argsort(eff[idx])]
        w = 1.0 / np.maximum(eff[idx], 0.3) ** p
        w = w / np.sum(w)
        out["total"][i] = float(np.exp(np.sum(w * bank["log_total"][idx])))
        for key in ["beam", "delay", "pol", "ue"]:
            val = np.sum(w.reshape((-1,) + (1,) * (bank[key].ndim - 1)) * bank[key][idx], axis=0)
            out[key][i] = (val / np.maximum(np.sum(val), TINY)).astype(np.float32)
        out["top4"][i] = float(np.sum(w * bank["top4"][idx]))
        out["top16"][i] = float(np.sum(w * bank["top16"][idx]))
        out["agree_beam"][i] = mean_pair_cos(bank["beam"][idx])
        out["agree_delay"][i] = mean_pair_cos(bank["delay"][idx])
        out["eff_nn"][i] = float(eff[idx[0]])
        out["eff_k"][i] = float(eff[idx[-1]])
        out["std_log_total"][i] = float(np.std(bank["log_total"][idx]))
    return out


def path_summary(path_file: str, prefix: str) -> np.ndarray:
    paths = np.load(path_file)
    valid = paths[f"{prefix}_valid"].astype(bool)
    ptype = paths[f"{prefix}_path_type"].astype(np.int8)
    delay = paths[f"{prefix}_delay_bin"].astype(np.float32)
    score = paths[f"{prefix}_score"].astype(np.float32)
    excess = paths[f"{prefix}_excess_length"].astype(np.float32)
    inc = paths[f"{prefix}_incidence_cos"].astype(np.float32)
    los_clear = paths[f"{prefix}_los_clear"].astype(np.float32)
    hit = paths[f"{prefix}_los_hit_count"].astype(np.float32)
    n = valid.shape[0]

    def safe_stat(arr: np.ndarray, mask: np.ndarray, fn: str, fill: float = 0.0) -> np.ndarray:
        out = np.full(n, fill, dtype=np.float32)
        for i in range(n):
            vals = arr[i, mask[i]]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if fn == "max":
                out[i] = np.max(vals)
            elif fn == "min":
                out[i] = np.min(vals)
            elif fn == "mean":
                out[i] = np.mean(vals)
            elif fn == "std":
                out[i] = np.std(vals)
        return out

    ref = valid & (ptype == 1)
    los = valid & (ptype == 0)
    feats = [
        valid.sum(axis=1).astype(np.float32),
        ref.sum(axis=1).astype(np.float32),
        los.sum(axis=1).astype(np.float32),
        los_clear,
        hit,
        safe_stat(score, valid, "max"),
        safe_stat(score, ref, "max"),
        safe_stat(delay, valid, "min"),
        safe_stat(delay, valid, "mean"),
        safe_stat(delay, ref, "mean"),
        safe_stat(excess, ref, "min"),
        safe_stat(excess, ref, "mean"),
        safe_stat(inc, ref, "max"),
        safe_stat(inc, ref, "mean"),
    ]
    return np.stack(feats, axis=1).astype(np.float32)


def base_features(
    pos: np.ndarray,
    blocked: np.ndarray,
    knn: dict[str, np.ndarray],
    map_feat: np.ndarray,
    sionna_feat: np.ndarray,
    path_feat: np.ndarray,
) -> np.ndarray:
    d_bs = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    az = np.arctan2(pos[:, 1] - BS_POS[1], pos[:, 0] - BS_POS[0])
    x = (pos[:, 0] - 150.0) / 100.0
    y = (pos[:, 1] + 75.0) / 100.0
    ff = [x, y, d_bs / 100.0, np.log(d_bs + 1e-6) - 4.0, np.sin(az), np.cos(az)]
    for freq in (1.0, 2.0, 4.0, 8.0, 16.0):
        for v in (x, y):
            ff.append(np.sin(np.pi * freq * v))
            ff.append(np.cos(np.pi * freq * v))
    ff = np.stack(ff, axis=1).astype(np.float32)
    regimes = regime_of(blocked)
    regime_oh = np.stack([
        (regimes == "clear"),
        (regimes == "light"),
        (regimes == "medium"),
        (regimes == "heavy"),
    ], axis=1).astype(np.float32)
    bfeat = np.stack([
        blocked / 80.0,
        np.log1p(np.maximum(blocked, 0.0)),
        knn["agree_beam"],
        knn["agree_delay"],
        knn["eff_nn"] / 5.0,
        knn["eff_k"] / 10.0,
        knn["std_log_total"],
        np.log(knn["total"] + TINY),
        knn["top4"],
        knn["top16"],
        shape_entropy(knn["beam"]),
        shape_entropy(knn["delay"]),
    ], axis=1).astype(np.float32)
    return np.concatenate([ff, regime_oh, bfeat, map_feat, sionna_feat, path_feat], axis=1).astype(np.float32)


def robust_scale(train_x: np.ndarray, test_x: np.ndarray):
    _ExtraTrees, RobustScaler = require_sklearn()
    scaler = RobustScaler(quantile_range=(10, 90))
    return scaler.fit_transform(train_x).astype(np.float32), scaler.transform(test_x).astype(np.float32), scaler


def normalize_shape(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 1e-30)
    return (x / np.maximum(x.sum(axis=1, keepdims=True), TINY)).astype(np.float32)


def apply_log_residual(base: np.ndarray, pred_resid: np.ndarray, shrink: float, clip: float) -> np.ndarray:
    corr = np.exp(np.clip(shrink * pred_resid, -clip, clip)).astype(np.float32)
    return normalize_shape(base * corr)


def cosine_mean(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(row_cos(pred, true)))


def choose_shrink(base: np.ndarray, true: np.ndarray, pred_resid: np.ndarray, grid: list[float], clip: float) -> tuple[float, dict[str, float]]:
    scores = {}
    for s in grid:
        pred = apply_log_residual(base, pred_resid, s, clip)
        scores[str(s)] = cosine_mean(pred, true)
    best = max(scores, key=scores.get)
    return float(best), scores


def choose_shrink_scalar(base: np.ndarray, true: np.ndarray, pred_delta: np.ndarray, grid: list[float], lo: float, hi: float) -> tuple[float, dict[str, float]]:
    scores = {}
    for s in grid:
        pred = np.clip(base + s * pred_delta, lo, hi)
        scores[str(s)] = -float(np.mean(np.abs(pred - true)))
    best = max(scores, key=scores.get)
    return float(best), scores


def make_model(seed: int, n_estimators: int, min_leaf: int, max_depth: int | None):
    ExtraTrees, _RobustScaler = require_sklearn()
    return ExtraTrees(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_features=0.75,
        random_state=seed,
        n_jobs=-1,
        bootstrap=False,
    )


def train_predict_block(
    x: np.ndarray,
    x_test: np.ndarray,
    base_train: np.ndarray,
    true_train: np.ndarray,
    base_test: np.ndarray,
    val_mask: np.ndarray,
    name: str,
    seed: int,
    n_estimators: int,
    min_leaf: int,
    max_depth: int | None,
    clip: float,
) -> tuple[np.ndarray, dict[str, object]]:
    y = (np.log(true_train + TINY) - np.log(base_train + TINY)).astype(np.float32)
    tr = ~val_mask
    model = make_model(seed, n_estimators, min_leaf, max_depth)
    model.fit(x[tr], y[tr])
    val_res = model.predict(x[val_mask]).astype(np.float32)
    shrink, scores = choose_shrink(
        base_train[val_mask], true_train[val_mask], val_res,
        [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
        clip,
    )
    base_score = scores["0.0"]
    best_score = scores[str(shrink)]
    final = make_model(seed + 13, n_estimators, min_leaf, max_depth)
    final.fit(x, y)
    test_res = final.predict(x_test).astype(np.float32)
    pred_test = apply_log_residual(base_test, test_res, shrink, clip)
    report = {
        "name": name,
        "shrink": shrink,
        "val_base_cos_mean": base_score,
        "val_best_cos_mean": best_score,
        "val_scores": scores,
    }
    return pred_test, report


def train_predict_scalar(
    x: np.ndarray,
    x_test: np.ndarray,
    base_train: np.ndarray,
    true_train: np.ndarray,
    base_test: np.ndarray,
    val_mask: np.ndarray,
    name: str,
    seed: int,
    n_estimators: int,
    min_leaf: int,
    max_depth: int | None,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, dict[str, object]]:
    y = (true_train - base_train).reshape(-1, 1).astype(np.float32)
    tr = ~val_mask
    model = make_model(seed, n_estimators, min_leaf, max_depth)
    model.fit(x[tr], y[tr])
    val_delta = model.predict(x[val_mask]).reshape(-1).astype(np.float32)
    shrink, scores = choose_shrink_scalar(
        base_train[val_mask], true_train[val_mask], val_delta,
        [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
        lo, hi,
    )
    final = make_model(seed + 17, n_estimators, min_leaf, max_depth)
    final.fit(x, y)
    test_delta = final.predict(x_test).reshape(-1).astype(np.float32)
    pred_test = np.clip(base_test + shrink * test_delta, lo, hi).astype(np.float32)
    report = {
        "name": name,
        "shrink": shrink,
        "val_base_mae": -scores["0.0"],
        "val_best_mae": -scores[str(shrink)],
        "val_scores_neg_mae": scores,
    }
    return pred_test, report


def project_to_channel(
    targets: dict[str, np.ndarray],
    base_path: str,
    output_path: str,
    pos_test: np.ndarray,
    beta_s: np.ndarray,
    batch: int,
    report_stats: dict[str, list[float]],
) -> None:
    base = np.load(base_path, mmap_mode="r")
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.complex64, shape=base.shape)
    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    for start in range(0, 500, batch):
        end = min(start + batch, 500)
        n = end - start
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        p0 = (np.abs(ad0) ** 2).astype(np.float32)
        p = p0.copy()

        target = {
            "beam": targets["beam"][start:end].reshape(n, MH, MV).astype(np.float32),
            "delay": targets["delay"][start:end].astype(np.float32),
            "pol": targets["pol"][start:end].astype(np.float32),
            "ue": targets["ue"][start:end].astype(np.float32),
        }
        regimes = targets["regime"][start:end]
        eta_beam = np.array([0.0 if r in ("clear", "light") else (0.16 if r == "medium" else 0.14) for r in regimes], dtype=np.float32)
        eta_delay = np.array([0.0 if r in ("clear", "light") else (0.10 if r == "medium" else 0.06) for r in regimes], dtype=np.float32)
        eta_pol = np.array([0.0 if r in ("clear", "light") else 0.08 for r in regimes], dtype=np.float32)
        eta_ue = np.array([0.0 if r in ("clear", "light") else 0.08 for r in regimes], dtype=np.float32)
        eta_top = np.array([0.0 if r in ("clear", "light") else (0.12 if r == "medium" else 0.10) for r in regimes], dtype=np.float32)

        for _ in range(2):
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["beam"], cur["beam"], (eta_beam / 2)[:, None, None], (2, 3), 1e-8 / (MH * MV))
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["delay"], cur["delay"], (eta_delay / 2)[:, None], (5,), 1e-8 / S_SC)
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["pol"], cur["pol"], (eta_pol / 2)[:, None], (1,), 1e-8 / MP)
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["ue"], cur["ue"], (eta_ue / 2)[:, None], (4,), 1e-8 / N_ANT)

        p0_coll = p0.sum(axis=(1, 4))
        p_coll = p.sum(axis=(1, 4))
        mask4, mask16 = top_masks_from_collapsed(p0_coll, 4, 16)
        flat_coll = p_coll.reshape(n, -1)
        flat_mask4 = mask4.reshape(n, -1)
        cur_total = np.maximum(flat_coll.sum(axis=1), TINY)
        cur4 = np.sum(flat_coll * flat_mask4, axis=1) / cur_total
        desired4 = cur4 * np.maximum(targets["top4"][start:end] / np.maximum(cur4, TINY), TINY) ** eta_top
        desired4 = np.clip(desired4, cur4 - 0.018, cur4 + 0.018)
        desired4 = np.clip(desired4, 1e-4, 0.98)
        top_scale = desired4 / np.maximum(cur4, TINY)
        rest_scale = (1.0 - desired4) / np.maximum(1.0 - cur4, TINY)
        scale_hvt = np.where(mask4, top_scale[:, None, None, None], rest_scale[:, None, None, None])
        p *= scale_hvt[:, None, :, :, None, :].astype(np.float32)

        p0_total = np.maximum(p0.sum(axis=(1, 2, 3, 4, 5)), TINY)
        p_total = np.maximum(p.sum(axis=(1, 2, 3, 4, 5)), TINY)
        p *= (p0_total / p_total)[:, None, None, None, None, None].astype(np.float32)

        gain = np.sqrt(p / np.maximum(p0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.75, 1.25)
        mask16_full = mask16[:, None, :, :, None, :]
        mask4_full = mask4[:, None, :, :, None, :]
        gain = np.where(mask16_full, np.clip(gain, 0.92, 1.12), gain)
        gain = np.where(mask4_full, np.clip(gain, 0.97, 1.05), gain)

        ad1 = (ad0 * gain).astype(np.complex64)
        h1 = inverse_ad(ad1).reshape(n, 256, 4, 192)
        h1 = h1 * np.exp(-1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        out[start:end] = h1.astype(np.complex64)
        out.flush()

        p1 = (np.abs(ad1) ** 2).astype(np.float32)
        m0 = marginal_stats_from_power(p0)
        m1 = marginal_stats_from_power(p1)
        p1_coll = p1.sum(axis=(1, 4))
        flat0 = p0_coll.reshape(n, -1)
        flat1 = p1_coll.reshape(n, -1)
        total0 = np.maximum(flat0.sum(axis=1), TINY)
        total1 = np.maximum(flat1.sum(axis=1), TINY)
        top4_0 = np.sum(flat0 * flat_mask4, axis=1) / total0
        top4_1 = np.sum(flat1 * flat_mask4, axis=1) / total1
        report_stats["power_ratio"].extend((m1["total"] / m0["total"]).tolist())
        report_stats["pdp_cos_to_base"].extend(row_cos(m1["delay"], m0["delay"]).tolist())
        report_stats["pas_cos_to_base"].extend(row_cos(m1["beam"], m0["beam"]).tolist())
        report_stats["top4_frac_delta"].extend((top4_1 - top4_0).tolist())
        print(f"generated {end}/500", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=ASSETS + "/cache_v83/v83_Round1_Test_Channel.npy")
    parser.add_argument("--output", default=str(PATHS.cache_dir(90) / "v90_Round1_Test_Channel.npy"))
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    parser.add_argument("--easy-cache", default=ASSETS + "/train_easy_marginals_v87_g010.npz")
    parser.add_argument("--map-features", default=ASSETS + "/map_features.npz")
    parser.add_argument("--sionna-features", default=ASSETS + "/sionna_etoile_features.npz")
    parser.add_argument("--sionna-paths", default=ASSETS + "/sionna_path_candidates.npz")
    parser.add_argument("--report", default=ASSETS + "/v90_learned_easy_predictor_generate.json")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--power", type=float, default=3.0)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--min-leaf", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    pos_train = np.load(Path(args.data) / "Round1_Train_Pos.npy")
    pos_test = np.load(Path(args.data) / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes_test = regime_of(blocked_test)
    beta_s = beta_subcarriers(args.beta, args.gamma)
    bank = build_easy_cache(Path(args.easy_cache), Path(args.data), beta_s, args.batch)

    print("building LOO/test kNN easy predictions ...", flush=True)
    knn_train = predict_knn_easy(bank, pos_train, blocked_train, pos_train, blocked_train, args.k, args.power, exclude_self=True)
    knn_test = predict_knn_easy(bank, pos_train, blocked_train, pos_test, blocked_test, args.k, args.power, exclude_self=False)

    map_npz = np.load(args.map_features)
    sionna_npz = np.load(args.sionna_features)
    path_train = path_summary(args.sionna_paths, "train")
    path_test = path_summary(args.sionna_paths, "test")
    x_train_raw = base_features(
        pos_train,
        blocked_train,
        knn_train,
        map_npz["train_features"].astype(np.float32),
        sionna_npz["train_features"].astype(np.float32),
        path_train,
    )
    x_test_raw = base_features(
        pos_test,
        blocked_test,
        knn_test,
        map_npz["test_features"].astype(np.float32),
        sionna_npz["test_features"].astype(np.float32),
        path_test,
    )
    x_train, x_test, _scaler = robust_scale(x_train_raw, x_test_raw)

    val_mask = (np.arange(len(pos_train)) % 5) == 0
    reports = []
    print("training learned residual predictors ...", flush=True)
    pred_beam, rep = train_predict_block(
        x_train, x_test, knn_train["beam"], bank["beam"], knn_test["beam"], val_mask,
        "beam", 9001, args.n_estimators, args.min_leaf, args.max_depth, clip=1.0,
    )
    reports.append(rep)
    pred_delay, rep = train_predict_block(
        x_train, x_test, knn_train["delay"], bank["delay"], knn_test["delay"], val_mask,
        "delay", 9002, args.n_estimators, args.min_leaf, args.max_depth, clip=1.0,
    )
    reports.append(rep)
    pred_pol, rep = train_predict_block(
        x_train, x_test, knn_train["pol"], bank["pol"], knn_test["pol"], val_mask,
        "pol", 9003, max(120, args.n_estimators // 2), args.min_leaf, args.max_depth, clip=0.6,
    )
    reports.append(rep)
    pred_ue, rep = train_predict_block(
        x_train, x_test, knn_train["ue"], bank["ue"], knn_test["ue"], val_mask,
        "ue", 9004, max(120, args.n_estimators // 2), args.min_leaf, args.max_depth, clip=0.6,
    )
    reports.append(rep)
    pred_top4, rep = train_predict_scalar(
        x_train, x_test, knn_train["top4"], bank["top4"], knn_test["top4"], val_mask,
        "top4", 9005, max(120, args.n_estimators // 2), args.min_leaf, args.max_depth, 0.0, 0.98,
    )
    reports.append(rep)
    pred_top16, rep = train_predict_scalar(
        x_train, x_test, knn_train["top16"], bank["top16"], knn_test["top16"], val_mask,
        "top16", 9006, max(120, args.n_estimators // 2), args.min_leaf, args.max_depth, 0.0, 0.995,
    )
    reports.append(rep)

    targets = {
        "beam": pred_beam,
        "delay": pred_delay,
        "pol": pred_pol,
        "ue": pred_ue,
        "top4": pred_top4,
        "top16": pred_top16,
        "regime": regimes_test,
    }
    gen_stats = {"power_ratio": [], "pdp_cos_to_base": [], "pas_cos_to_base": [], "top4_frac_delta": []}
    print("projecting learned easy targets to channel ...", flush=True)
    project_to_channel(targets, args.base, args.output, pos_test, beta_s, args.batch, gen_stats)

    payload = {
        "method": "v90_learned_easy_predictor",
        "base": args.base,
        "operation": "ExtraTrees learned residual correction to kNN easy targets + v87 projection; phase frozen",
        "model": {
            "type": "ExtraTreesRegressor",
            "n_estimators": int(args.n_estimators),
            "min_leaf": int(args.min_leaf),
            "max_depth": int(args.max_depth),
            "validation": "idx % 5 == 0",
        },
        "block_reports": reports,
        "overall": {k: summarize(v) for k, v in gen_stats.items()},
        "target_summary": {
            "knn_test_agree_beam": summarize(knn_test["agree_beam"]),
            "knn_test_agree_delay": summarize(knn_test["agree_delay"]),
            "learned_top4": summarize(pred_top4),
            "learned_top16": summarize(pred_top16),
        },
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
