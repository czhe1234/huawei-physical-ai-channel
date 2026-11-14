#!/usr/bin/env python3
from project_paths import PATHS
"""Evaluate adaptive-heavy rho rules against fixed heavy-rho references."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from analyze_test_regimes import BS_POS
from evaluate_ad_coeff_methods import eval_one
from evaluate_local_ad_paths import idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_regime_candidates import classify_train_points
from evaluate_s1style_geom_blend import geometry_cache, predict_phase_idw, select_geom_neighbors, train_log_power
from generate_adaptive_heavy_rho import DEFAULT_FEATURE_INDICES, heavy_rho
from generate_geometric_steering_blend import unit_vectors
from spectrum_projection import MetricAccumulator


def add_metric(stats, accs, name, pred, truth):
    num, den = eval_one(truth[None], pred[None])
    stats[name]["num"] += float(num)
    stats[name]["den"] += float(den)
    stats[name]["count"] += 1
    accs[name].add(pred[None], truth[None])


def finalize(stats, accs):
    rows = []
    for name, s in stats.items():
        nmse = s["num"] / max(s["den"], 1e-30)
        rows.append({
            "name": name,
            "count": int(s["count"]),
            "global_nmse_direct": float(nmse),
            **accs[name].result(),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def build_train_labels(pos, map_features, blocked, feature_indices):
    labels = classify_train_points(pos, map_features, blocked, feature_indices)
    for i, label in enumerate(labels):
        label["blocked"] = float(blocked[i])
        label["bs_dist"] = float(np.linalg.norm(pos[i] - BS_POS))
    return labels


def fixed_rho(label, heavy):
    if label["los"] == "clear":
        return 0.0, "clear"
    if label["los"] == "light":
        return 0.0, "light"
    if label["los"] == "medium":
        return 1.0, "medium"
    return float(heavy), f"heavy_{heavy:g}"


def adaptive_rho(label, args):
    if label["los"] == "clear":
        return 0.0, "clear"
    if label["los"] == "light":
        return 0.0, "light"
    if label["los"] == "medium":
        return 1.0, "medium"
    return heavy_rho(label, args)


def predict_v9_geom(vi, pos, h, tree, blocked, logpw, dist_all, u_all, args):
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
    return v9, geom


def evaluate_set(set_name, val_idx, pos, h, labels, blocked, logpw, dist_all, u_all, args):
    refs = [(f"fixed_h{h:g}", h) for h in args.fixed_heavy_values]
    names = [name for name, _ in refs] + ["adaptive_v44"]
    stats = {n: {"num": 0.0, "den": 0.0, "count": 0} for n in names}
    accs = {n: MetricAccumulator("sumpol") for n in names}
    bucket_stats = {}
    bucket_accs = {}
    rho_counts = {}
    tree = cKDTree(pos[:, :2])

    for n, vi in enumerate(val_idx):
        vi = int(vi)
        truth = np.asarray(h[vi])
        v9, geom = predict_v9_geom(vi, pos, h, tree, blocked, logpw, dist_all, u_all, args)
        preds = {}
        for name, heavy in refs:
            rho, _label = fixed_rho(labels[vi], heavy)
            preds[name] = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
        rho, rho_label = adaptive_rho(labels[vi], args)
        preds["adaptive_v44"] = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
        rho_counts[rho_label] = rho_counts.get(rho_label, 0) + 1

        for name, pred in preds.items():
            add_metric(stats, accs, name, pred, truth)

        bucket_keys = [
            f"los:{labels[vi]['los']}",
            f"density:{labels[vi]['density']}",
            f"novelty:{labels[vi]['novelty']}",
            f"boundary:{labels[vi]['boundary']}",
            f"adaptive_rho:{rho_label}",
        ]
        for key in bucket_keys:
            if key not in bucket_stats:
                bucket_stats[key] = {m: {"num": 0.0, "den": 0.0, "count": 0} for m in names}
                bucket_accs[key] = {m: MetricAccumulator("sumpol") for m in names}
            for name, pred in preds.items():
                add_metric(bucket_stats[key], bucket_accs[key], name, pred, truth)

        if (n + 1) % 50 == 0 or n + 1 == len(val_idx):
            print(f"{set_name}: {n + 1}/{len(val_idx)}", flush=True)

    return {
        "name": set_name,
        "count": int(len(val_idx)),
        "results": finalize(stats, accs),
        "rho_counts": rho_counts,
        "bucket_results": {key: finalize(bucket_stats[key], bucket_accs[key]) for key in sorted(bucket_stats)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--height-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--val-sets", default=str(PATHS.artifacts / "shared_features/test_regime_analysis/test_matched_val_sets.npz"))
    parser.add_argument("--sets", default="feature_matched,stress_feature_boundary,heavy_blocked")
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/adaptive_heavy_rho_eval_v44.json"))
    parser.add_argument("--feature-indices", default=DEFAULT_FEATURE_INDICES)
    parser.add_argument("--fixed-heavy-values", type=float, nargs="+", default=[3.0, 3.75, 4.0, 4.5, 5.0])
    parser.add_argument("--heavy-low-rho", type=float, default=3.25)
    parser.add_argument("--heavy-mid-rho", type=float, default=3.75)
    parser.add_argument("--heavy-high-rho", type=float, default=4.25)
    parser.add_argument("--heavy-risk-blocked", type=float, default=115.0)
    parser.add_argument("--heavy-strong-blocked", type=float, default=145.0)
    parser.add_argument("--heavy-risk-feature-nn1", type=float, default=2.1)
    parser.add_argument("--heavy-strong-feature-nn1", type=float, default=1.7)
    parser.add_argument("--heavy-strong-nn1", type=float, default=3.1)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--geom-k", type=int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--los-penalty", type=float, default=1.0)
    parser.add_argument("--occ-penalty", type=float, default=0.5)
    parser.add_argument("--r-penalty", type=float, default=0.0)
    parser.add_argument("--theta-penalty", type=float, default=0.0)
    parser.add_argument("--occ-scale", type=float, default=80.0)
    parser.add_argument("--r-scale", type=float, default=80.0)
    parser.add_argument("--theta-scale-deg", type=float, default=30.0)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy")).astype(np.float64)
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    map_payload = np.load(args.map_features)
    train_map = map_payload["train_features"].astype(np.float32)
    blocked, _test_blocked = geometry_cache(args.data, args.height_cache, False)
    feature_indices = [int(x) for x in args.feature_indices.split(",") if x.strip()]
    labels = build_train_labels(pos, train_map, blocked, feature_indices)
    logpw = train_log_power(h)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    val_payload = np.load(args.val_sets)
    results = []
    for set_name in [x.strip() for x in args.sets.split(",") if x.strip()]:
        results.append(evaluate_set(
            set_name,
            val_payload[set_name].astype(np.int64),
            pos,
            h,
            labels,
            blocked,
            logpw,
            dist_all,
            u_all,
            args,
        ))

    payload = {"args": vars(args), "sets": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({
        "out": args.out,
        "sets": [{"name": r["name"], "rho_counts": r["rho_counts"], "top": r["results"][:5]} for r in results],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
