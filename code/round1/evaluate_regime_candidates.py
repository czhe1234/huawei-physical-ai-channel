#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from analyze_test_regimes import (
    BS_POS,
    boundary_label,
    density_label,
    los_label,
    make_feature_matrix,
    novelty_label,
    qtiles,
    robust_standardize,
)
from evaluate_ad_coeff_methods import eval_one
from evaluate_local_ad_paths import idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_s1style_geom_blend import (
    geometry_cache,
    predict_phase_idw,
    select_geom_neighbors,
    train_log_power,
)
from generate_geometric_steering_blend import unit_vectors
from spectrum_projection import MetricAccumulator


def point_power(h, idx):
    return float(np.sum(np.abs(np.asarray(h[idx])) ** 2, dtype=np.float64))


def add_metric(stats, accs, name, pred, truth):
    num, den = eval_one(truth[None], pred[None])
    stats[name]["num"] += float(num)
    stats[name]["den"] += float(den)
    stats[name]["count"] += 1
    accs[name].add(pred[None], truth[None])


def finalize(stats, accs):
    out = []
    for name, s in stats.items():
        nmse = s["num"] / max(s["den"], 1e-30)
        out.append({
            "name": name,
            "count": int(s["count"]),
            "global_nmse_direct": float(nmse),
            **accs[name].result(),
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def classify_train_points(pos, map_features, blocked, feature_indices):
    tree_xy = cKDTree(pos[:, :2])
    knn_dist, knn_idx_raw = tree_xy.query(pos[:, :2], k=33, workers=-1)
    knn_dist = knn_dist[:, 1:33]
    knn_idx = knn_idx_raw[:, 1:33]

    feat = make_feature_matrix(pos, map_features, blocked, knn_dist, feature_indices)
    feat_z, _ = robust_standardize(feat, feat)
    tree_feat = cKDTree(feat_z)
    feat_dist_raw, _ = tree_feat.query(feat_z, k=33, workers=-1)
    feat_dist = feat_dist_raw[:, 1:33]

    nn_q = qtiles(knn_dist[:, 0])
    feat_q = qtiles(feat_dist[:, 0])
    labels = []
    for i in range(len(pos)):
        b = los_label(blocked[i])
        d = density_label(knn_dist[i, 0], nn_q["p25"], nn_q["p75"], nn_q["p90"])
        n = novelty_label(feat_dist[i, 0], feat_q["p50"], feat_q["p75"], feat_q["p90"])
        boundary, clear_frac, block_std = boundary_label(blocked[knn_idx[i, :16]])
        labels.append({
            "los": b,
            "density": d,
            "novelty": n,
            "boundary": boundary,
            "local_clear_frac": clear_frac,
            "local_block_std": block_std,
            "nn1": float(knn_dist[i, 0]),
            "feature_nn1": float(feat_dist[i, 0]),
        })
    return labels


def candidate_names(rhos):
    names = [
        "v9_g010",
        "pll_g015",
        "geom_soft_pw_k16",
    ]
    names.extend([f"blend_rho{rho:g}" for rho in rhos])
    names.extend([
        "rule_stable_familiar_blend030",
        "rule_heavy_stable_blend030",
        "rule_heavy_notnovel_blend030",
        "rule_nonstress_blend030",
        "rule_clear_dense_blend010",
    ])
    return names


def build_predictions(vi, pos, h, tree_neighbors, train_blocked, logpw, dist_all, u_all, args, rhos):
    max_k = args.k
    d, idx = tree_neighbors.query(pos[vi, :2], k=max_k + 1, workers=-1)
    keep = idx != vi
    idx = idx[keep][:max_k]
    d = d[keep][:max_k]
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
        0.10,
        1.0,
        args.steering_scale,
    )
    pll_g015 = predict_affine(
        neigh,
        coeff,
        dist_all[idx] - dist_all[vi],
        u_all,
        idx,
        vi,
        args.beta,
        0.15,
        1.0,
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
    gidx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, pos[vi], train_blocked[vi], vi, geom_cfg)
    w = idw_weights(eff_d, args.power)
    geom = predict_phase_idw(
        np.asarray(h[gidx]),
        w,
        dist_all[gidx] - dist_all[vi],
        args.beta,
        0.10,
        logpw,
        gidx,
    )

    preds = {
        "v9_g010": v9,
        "pll_g015": pll_g015,
        "geom_soft_pw_k16": geom,
    }
    for rho in rhos:
        preds[f"blend_rho{rho:g}"] = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
    return preds


def rule_prediction(label, preds):
    los = label["los"]
    density = label["density"]
    novelty = label["novelty"]
    boundary = label["boundary"]
    stress = (
        novelty in {"novel", "very_novel"}
        or density in {"sparse", "very_sparse"}
        or boundary != "stable"
    )
    out = {}
    out["rule_stable_familiar_blend030"] = preds["blend_rho0.3"] if (
        boundary == "stable" and novelty in {"familiar", "normal"}
    ) else preds["v9_g010"]
    out["rule_heavy_stable_blend030"] = preds["blend_rho0.3"] if (
        los == "heavy" and boundary == "stable"
    ) else preds["v9_g010"]
    out["rule_heavy_notnovel_blend030"] = preds["blend_rho0.3"] if (
        los == "heavy" and novelty in {"familiar", "normal"} and boundary == "stable"
    ) else preds["v9_g010"]
    out["rule_nonstress_blend030"] = preds["blend_rho0.3"] if not stress else preds["v9_g010"]
    out["rule_clear_dense_blend010"] = preds["blend_rho0.1"] if (
        los == "clear" and density == "dense" and boundary == "stable"
    ) else preds["v9_g010"]
    return out


def evaluate_set(name, val_idx, pos, h, labels, train_blocked, logpw, dist_all, u_all, args, rhos):
    tree = cKDTree(pos[:, :2])
    names = candidate_names(rhos)
    stats = {n: {"num": 0.0, "den": 0.0, "count": 0} for n in names}
    accs = {n: MetricAccumulator("sumpol") for n in names}
    bucket_stats = {}
    bucket_accs = {}
    power_rows = []

    for n, vi in enumerate(val_idx):
        vi = int(vi)
        truth = np.asarray(h[vi])
        preds = build_predictions(vi, pos, h, tree, train_blocked, logpw, dist_all, u_all, args, rhos)
        preds.update(rule_prediction(labels[vi], preds))
        for method, pred in preds.items():
            add_metric(stats, accs, method, pred, truth)

        bucket_keys = [
            ("los", labels[vi]["los"]),
            ("density", labels[vi]["density"]),
            ("novelty", labels[vi]["novelty"]),
            ("boundary", labels[vi]["boundary"]),
        ]
        for bucket_type, bucket_value in bucket_keys:
            key = f"{bucket_type}:{bucket_value}"
            if key not in bucket_stats:
                bucket_stats[key] = {m: {"num": 0.0, "den": 0.0, "count": 0} for m in names}
                bucket_accs[key] = {m: MetricAccumulator("sumpol") for m in names}
            for method, pred in preds.items():
                add_metric(bucket_stats[key], bucket_accs[key], method, pred, truth)

        den = point_power(h, vi)
        row = {"idx": vi, "truth_power": den, **labels[vi]}
        for method, pred in preds.items():
            num = float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
            row[f"nmse_{method}"] = num / max(den, 1e-30)
        power_rows.append(row)
        if (n + 1) % 50 == 0 or n + 1 == len(val_idx):
            print(f"{name}: {n + 1}/{len(val_idx)}", flush=True)

    bucket_results = {}
    for key in bucket_stats:
        bucket_results[key] = finalize(bucket_stats[key], bucket_accs[key])

    per_point_summary = []
    for row in power_rows:
        base = row["nmse_v9_g010"]
        best_method = min(names, key=lambda m: row[f"nmse_{m}"])
        per_point_summary.append({
            "idx": int(row["idx"]),
            "truth_power": float(row["truth_power"]),
            "los": row["los"],
            "density": row["density"],
            "novelty": row["novelty"],
            "boundary": row["boundary"],
            "v9_nmse": float(base),
            "best_method": best_method,
            "best_nmse": float(row[f"nmse_{best_method}"]),
            "best_gain": float(base - row[f"nmse_{best_method}"]),
        })

    return {
        "name": name,
        "count": int(len(val_idx)),
        "results": finalize(stats, accs),
        "bucket_results": bucket_results,
        "per_point_top_gains": sorted(per_point_summary, key=lambda r: r["best_gain"], reverse=True)[:50],
        "per_point_worst_losses_vs_v9": sorted(
            per_point_summary,
            key=lambda r: r["best_gain"],
        )[:50],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--height-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--val-sets", default=str(PATHS.artifacts / "shared_features/test_regime_analysis/test_matched_val_sets.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/regime_candidate_eval.json"))
    parser.add_argument("--feature-indices", default="0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41")
    parser.add_argument("--sets", default="feature_matched,cluster_matched,stress_feature_boundary,clear_los,heavy_blocked")
    parser.add_argument("--rhos", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--geom-k", type=int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
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
    train_blocked, _ = geometry_cache(args.data, args.height_cache, False)
    logpw = train_log_power(h)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    feature_indices = [int(x) for x in args.feature_indices.split(",") if x.strip()]
    labels = classify_train_points(pos, train_map, train_blocked, feature_indices)
    val_payload = np.load(args.val_sets)
    rhos = [float(x) for x in args.rhos.split(",") if x.strip()]
    set_names = [x.strip() for x in args.sets.split(",") if x.strip()]

    all_results = []
    for set_name in set_names:
        val_idx = val_payload[set_name].astype(np.int64)
        all_results.append(evaluate_set(set_name, val_idx, pos, h, labels, train_blocked, logpw, dist_all, u_all, args, rhos))

    payload = {
        "args": vars(args),
        "candidate_names": candidate_names(rhos),
        "sets": all_results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({
        "out": args.out,
        "sets": [
            {
                "name": s["name"],
                "count": s["count"],
                "top5": s["results"][:5],
            }
            for s in all_results
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
