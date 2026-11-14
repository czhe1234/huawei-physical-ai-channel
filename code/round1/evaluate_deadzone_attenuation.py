#!/usr/bin/env python3
from project_paths import PATHS
"""Evaluate geometry/power based dead-zone attenuation on top of v36.

The goal is to test whether very-low-power / inside-building points can be
identified from geometry and nearby training power, without changing the
confirmed v36 phase structure.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_regime_candidates import classify_train_points
from evaluate_regime_rho_combos import rho_for_los
from evaluate_s1style_geom_blend import (
    build_height_map,
    geometry_cache,
    predict_phase_idw,
    select_geom_neighbors,
    train_log_power,
)
from generate_geometric_steering_blend import unit_vectors
from build_map_features import BS_POS


def parse_list(text, cast=float):
    return [cast(x) for x in text.split(",") if x.strip()]


def train_power(train_h, batch=64):
    out = np.empty(train_h.shape[0], dtype=np.float64)
    for start in range(0, train_h.shape[0], batch):
        end = min(start + batch, train_h.shape[0])
        block = np.asarray(train_h[start:end])
        out[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return out


def sample_height_features(points, data_dir, radii=(0, 1, 2, 3, 5)):
    hmap, origin, cell = build_height_map(Path(data_dir) / "Round1_Map.ply", cell=1.0, z_min=1.0)
    feats = []
    ij = np.floor((points[:, :2] - origin[None, :]) / cell).astype(np.int64)
    for r in radii:
        grid = hmap if r == 0 else maximum_filter(hmap, size=2 * r + 1, mode="constant", cval=0.0)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < grid.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < grid.shape[1])
        val = np.zeros(len(points), dtype=np.float32)
        val[ok] = grid[ij[ok, 0], ij[ok, 1]]
        feats.append(val)
    return np.stack(feats, axis=1).astype(np.float32)


def make_features(pos, map_features, blocked, height_features):
    d = pos - BS_POS[None, :]
    r = np.linalg.norm(d, axis=1)
    theta = np.arctan2(d[:, 1], d[:, 0])
    base = np.column_stack([
        pos[:, 0],
        pos[:, 1],
        r,
        np.sin(theta),
        np.cos(theta),
        blocked.astype(np.float32),
        np.log1p(blocked.astype(np.float32)),
    ]).astype(np.float32)
    return np.concatenate([base, map_features.astype(np.float32), height_features], axis=1)


def robust_scale(train_x, *arrays):
    med = np.median(train_x, axis=0)
    q25, q75 = np.percentile(train_x, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return [(a - med[None, :]) / scale[None, :] for a in arrays]


def weighted_knn_scores(train_x, train_labels, query_x, k, exclude_index=None, power=2.0):
    tree = cKDTree(train_x)
    kk = min(k + (1 if exclude_index is not None else 0), len(train_x))
    dist, idx = tree.query(query_x, k=kk, workers=-1)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    scores = np.empty(len(query_x), dtype=np.float64)
    for i in range(len(query_x)):
        ii = idx[i]
        dd = dist[i]
        if exclude_index is not None:
            keep = ii != int(exclude_index[i])
            ii = ii[keep]
            dd = dd[keep]
        ii = ii[:k]
        dd = dd[:k]
        w = 1.0 / np.maximum(dd, 1e-6) ** power
        w = w / np.sum(w)
        scores[i] = float(np.dot(w, train_labels[ii]))
    return scores


def predict_v36(vi, pos, h, tree, blocked, logpw, dist_all, u_all, labels, args):
    d, idx = tree.query(pos[vi, :2], k=args.k + 1, workers=-1)
    keep = idx != vi
    idx = idx[keep][: args.k]
    d = d[keep][: args.k]
    neigh = np.asarray(h[idx])
    rel_xy = pos[idx, :2] - pos[vi, :2]
    coeff = affine_coefficients(rel_xy, d, args.power, args.ridge)
    v9 = predict_affine(
        neigh,
        coeff,
        dist_all[idx] - dist_all[vi],
        u_all,
        idx,
        vi,
        args.beta,
        args.gamma,
        args.alpha,
        args.steering_scale,
    )

    geom_cfg = {
        "k": int(args.geom_k),
        "los_penalty": float(args.los_penalty),
        "occ_penalty": float(args.occ_penalty),
        "r_penalty": float(args.r_penalty),
        "theta_penalty": float(args.theta_penalty),
        "occ_scale": float(args.occ_scale),
        "r_scale": float(args.r_scale),
        "theta_scale_deg": float(args.theta_scale_deg),
    }
    gidx, _raw_d, eff_d = select_geom_neighbors(pos, blocked, pos[vi], blocked[vi], vi, geom_cfg)
    w = idw_weights(eff_d, args.power)
    geom = predict_phase_idw(
        np.asarray(h[gidx]),
        w,
        dist_all[gidx] - dist_all[vi],
        args.beta,
        args.gamma,
        logpw,
        gidx,
    )
    rho = rho_for_los(labels[vi]["los"], (args.clear_rho, args.light_rho, args.medium_rho, args.heavy_rho))
    return ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--height-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/deadzone_attenuation_eval.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--thresholds", default="1e-10,1e-8,1e-6")
    parser.add_argument("--score-ks", default="8,16,32,64")
    parser.add_argument("--score-floors", default="0.0,0.10,0.25,0.50")
    parser.add_argument("--lambdas", default="0.5,1.0,2.0,4.0,8.0")
    parser.add_argument("--feature-power", type=float, default=2.0)
    parser.add_argument("--heavy-only", action="store_true", default=True)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--geom-k", type=int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--clear-rho", type=float, default=0.0)
    parser.add_argument("--light-rho", type=float, default=0.0)
    parser.add_argument("--medium-rho", type=float, default=1.0)
    parser.add_argument("--heavy-rho", type=float, default=4.0)
    parser.add_argument("--los-penalty", type=float, default=1.0)
    parser.add_argument("--occ-penalty", type=float, default=0.5)
    parser.add_argument("--r-penalty", type=float, default=0.0)
    parser.add_argument("--theta-penalty", type=float, default=0.0)
    parser.add_argument("--occ-scale", type=float, default=80.0)
    parser.add_argument("--r-scale", type=float, default=80.0)
    parser.add_argument("--theta-scale-deg", type=float, default=30.0)
    args = parser.parse_args()

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(data / "Round1_Test_Pos.npy").astype(np.float64)
    h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    pwr = train_power(h)
    logp = np.log10(np.maximum(pwr, 1e-30))

    map_payload = np.load(args.map_features)
    train_map = map_payload["train_features"].astype(np.float32)
    test_map = map_payload["test_features"].astype(np.float32)
    train_blocked, _test_blocked = geometry_cache(args.data, args.height_cache, False)
    train_hfeat = sample_height_features(pos, args.data)
    test_hfeat = sample_height_features(test_pos, args.data)
    train_x = make_features(pos, train_map, train_blocked, train_hfeat)
    test_x = make_features(test_pos, test_map, _test_blocked, test_hfeat)
    train_xs, test_xs = robust_scale(train_x, train_x, test_x)

    feature_indices = [0, 1, 2, 3, 5, 9, 13, 17, 21, 25, 28, 32, 33, 34, 35, 39, 40, 41]
    labels = classify_train_points(pos, train_map, train_blocked, feature_indices)

    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    qx = train_xs[val_idx]
    thresholds = parse_list(args.thresholds, float)
    score_ks = parse_list(args.score_ks, int)
    floors = parse_list(args.score_floors, float)
    lambdas = parse_list(args.lambdas, float)

    score_cache = {}
    for th in thresholds:
        y = (pwr <= th).astype(np.float64)
        for k_score in score_ks:
            score_cache[(th, k_score)] = weighted_knn_scores(
                train_xs,
                y,
                qx,
                k_score,
                exclude_index=val_idx,
                power=args.feature_power,
            )

    tree = cKDTree(pos[:, :2])
    logpw = train_log_power(h)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    names = ["v36"]
    configs = []
    for th in thresholds:
        for k_score in score_ks:
            for floor in floors:
                for lam in lambdas:
                    name = f"att_th{th:g}_k{k_score}_f{floor:g}_l{lam:g}"
                    names.append(name)
                    configs.append((name, th, k_score, floor, lam))

    stats = {name: {"num": 0.0, "den": 0.0, "count": 0} for name in names}
    bucket_stats = {}
    atten_sums = {name: 0.0 for name in names if name != "v36"}
    atten_counts = {name: 0 for name in names if name != "v36"}
    score_summaries = {}

    for row, vi in enumerate(val_idx):
        vi = int(vi)
        truth = np.asarray(h[vi])
        pred = predict_v36(vi, pos, h, tree, train_blocked, logpw, dist_all, u_all, labels, args)
        den = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        num = float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
        stats["v36"]["num"] += num
        stats["v36"]["den"] += den
        stats["v36"]["count"] += 1
        bucket = labels[vi]["los"]
        bucket_stats.setdefault(bucket, {name: {"num": 0.0, "den": 0.0, "count": 0} for name in names})
        bucket_stats[bucket]["v36"]["num"] += num
        bucket_stats[bucket]["v36"]["den"] += den
        bucket_stats[bucket]["v36"]["count"] += 1

        for name, th, k_score, floor, lam in configs:
            raw = float(score_cache[(th, k_score)][row])
            score = max(0.0, (raw - floor) / max(1.0 - floor, 1e-6))
            if args.heavy_only and bucket != "heavy":
                score = 0.0
            gain = float(np.exp(-lam * score))
            pred2 = (pred * np.float32(gain)).astype(np.complex64)
            num2 = float(np.sum(np.abs(pred2 - truth) ** 2, dtype=np.float64))
            stats[name]["num"] += num2
            stats[name]["den"] += den
            stats[name]["count"] += 1
            bucket_stats[bucket][name]["num"] += num2
            bucket_stats[bucket][name]["den"] += den
            bucket_stats[bucket][name]["count"] += 1
            atten_sums[name] += 1.0 - gain
            atten_counts[name] += int(score > 0.0)

        if (row + 1) % 50 == 0 or row + 1 == len(val_idx):
            print(f"evaluated {row + 1}/{len(val_idx)}", flush=True)

    results = []
    base_nmse = stats["v36"]["num"] / max(stats["v36"]["den"], 1e-30)
    for name, s in stats.items():
        nmse = s["num"] / max(s["den"], 1e-30)
        rec = {
            "name": name,
            "count": int(s["count"]),
            "global_nmse": float(nmse),
            "delta_nmse_vs_v36": float(nmse - base_nmse),
        }
        if name != "v36":
            rec["atten_mean"] = float(atten_sums[name] / max(s["count"], 1))
            rec["atten_nonzero_count"] = int(atten_counts[name])
        results.append(rec)
    results.sort(key=lambda r: r["global_nmse"])

    bucket_results = {}
    for bucket, rows in bucket_stats.items():
        base = rows["v36"]["num"] / max(rows["v36"]["den"], 1e-30)
        out = []
        for name, s in rows.items():
            nmse = s["num"] / max(s["den"], 1e-30)
            out.append({
                "name": name,
                "count": int(s["count"]),
                "global_nmse": float(nmse),
                "delta_nmse_vs_v36": float(nmse - base),
            })
        out.sort(key=lambda r: r["global_nmse"])
        bucket_results[bucket] = out[:20]

    # Score distribution for the test set, used to judge whether a candidate
    # would affect many hidden-test points.
    test_score_summary = {}
    for th in thresholds:
        y = (pwr <= th).astype(np.float64)
        for k_score in score_ks:
            sc = weighted_knn_scores(train_xs, y, test_xs, k_score, None, power=args.feature_power)
            test_score_summary[f"th{th:g}_k{k_score}"] = {
                "q": [float(x) for x in np.quantile(sc, [0, 0.5, 0.9, 0.95, 0.98, 0.99, 1.0])],
                "mean": float(np.mean(sc)),
                "gt_0p1": int(np.sum(sc > 0.1)),
                "gt_0p25": int(np.sum(sc > 0.25)),
                "gt_0p5": int(np.sum(sc > 0.5)),
            }

    payload = {
        "args": vars(args),
        "train_power_quantiles": [float(x) for x in np.quantile(pwr, [0, .001, .005, .01, .02, .05, .1, .2, .5])],
        "results": results[:50],
        "bucket_results": bucket_results,
        "test_score_summary": test_score_summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"out": args.out, "top10": results[:10], "test_score_summary": test_score_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
