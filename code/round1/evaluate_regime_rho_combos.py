#!/usr/bin/env python3
from project_paths import PATHS
"""Evaluate blocked-regime-specific rho combinations for v9 + geometry blend."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from analyze_test_regimes import BS_POS, make_feature_matrix, robust_standardize
from evaluate_ad_coeff_methods import eval_one
from evaluate_local_ad_paths import idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_regime_candidates import classify_train_points
from evaluate_s1style_geom_blend import geometry_cache, predict_phase_idw, select_geom_neighbors, train_log_power
from generate_geometric_steering_blend import unit_vectors
from spectrum_projection import MetricAccumulator


def parse_combos(text):
    combos = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(",")]
        if len(parts) != 5:
            raise ValueError(f"combo must be name,clear,light,medium,heavy: {item}")
        name = parts[0]
        vals = tuple(float(x) for x in parts[1:])
        combos.append((name, vals))
    return combos


def combo_default():
    return ";".join([
        "v9,0.00,0.00,0.00,0.00",
        "v12,0.30,0.30,0.30,0.30",
        "v15,0.10,0.30,0.30,0.30",
        "c005_l030_m030_h030,0.05,0.30,0.30,0.30",
        "c000_l030_m030_h030,0.00,0.30,0.30,0.30",
        "c010_l030_m035_h035,0.10,0.30,0.35,0.35",
        "c005_l030_m035_h035,0.05,0.30,0.35,0.35",
        "c010_l030_m030_h040,0.10,0.30,0.30,0.40",
        "c005_l030_m030_h040,0.05,0.30,0.30,0.40",
        "c010_l035_m035_h040,0.10,0.35,0.35,0.40",
        "c005_l035_m035_h040,0.05,0.35,0.35,0.40",
        "c010_l030_m040_h050,0.10,0.30,0.40,0.50",
        "c005_l030_m040_h050,0.05,0.30,0.40,0.50",
    ])


def rho_for_los(los, vals):
    clear, light, medium, heavy = vals
    if los == "clear":
        return clear
    if los == "light":
        return light
    if los == "medium":
        return medium
    return heavy


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
        row = {
            "name": name,
            "count": int(s["count"]),
            "global_nmse_direct": float(nmse),
            **accs[name].result(),
        }
        out.append(row)
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def predict_v9_and_geom(vi, pos, h, tree, blocked, logpw, dist_all, u_all, args):
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


def evaluate_set(set_name, val_idx, pos, h, labels, blocked, logpw, dist_all, u_all, combos, args):
    tree = cKDTree(pos[:, :2])
    names = [name for name, _vals in combos]
    stats = {n: {"num": 0.0, "den": 0.0, "count": 0} for n in names}
    accs = {n: MetricAccumulator("sumpol") for n in names}
    bucket_stats = {}
    bucket_accs = {}

    for n, vi in enumerate(val_idx):
        vi = int(vi)
        truth = np.asarray(h[vi])
        v9, geom = predict_v9_and_geom(vi, pos, h, tree, blocked, logpw, dist_all, u_all, args)
        los = labels[vi]["los"]
        for name, vals in combos:
            rho = rho_for_los(los, vals)
            pred = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
            add_metric(stats, accs, name, pred, truth)

            bkey = f"los:{los}"
            if bkey not in bucket_stats:
                bucket_stats[bkey] = {m: {"num": 0.0, "den": 0.0, "count": 0} for m in names}
                bucket_accs[bkey] = {m: MetricAccumulator("sumpol") for m in names}
            add_metric(bucket_stats[bkey], bucket_accs[bkey], name, pred, truth)

        if (n + 1) % 50 == 0 or n + 1 == len(val_idx):
            print(f"{set_name}: {n + 1}/{len(val_idx)}", flush=True)

    return {
        "name": set_name,
        "count": int(len(val_idx)),
        "results": finalize(stats, accs),
        "bucket_results": {key: finalize(bucket_stats[key], bucket_accs[key]) for key in bucket_stats},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--height-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--val-sets", default=str(PATHS.artifacts / "shared_features/test_regime_analysis/test_matched_val_sets.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/regime_rho_combo_eval.json"))
    parser.add_argument("--feature-indices", default="0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41")
    parser.add_argument("--sets", default="feature_matched,cluster_matched,stress_feature_boundary,clear_los,heavy_blocked")
    parser.add_argument("--combos", default=combo_default())
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

    combos = parse_combos(args.combos)
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy")).astype(np.float64)
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    map_payload = np.load(args.map_features)
    train_map = map_payload["train_features"].astype(np.float32)
    blocked, _ = geometry_cache(args.data, args.height_cache, False)
    logpw = train_log_power(h)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    feature_indices = [int(x) for x in args.feature_indices.split(",") if x.strip()]

    tree_xy = cKDTree(pos[:, :2])
    knn_dist, _knn_idx_raw = tree_xy.query(pos[:, :2], k=33, workers=-1)
    _feat = make_feature_matrix(pos, train_map, blocked, knn_dist[:, 1:33], feature_indices)
    # classify_train_points recomputes the same features and keeps label logic centralized.
    labels = classify_train_points(pos, train_map, blocked, feature_indices)

    val_payload = np.load(args.val_sets)
    set_names = [x.strip() for x in args.sets.split(",") if x.strip()]
    results = []
    for set_name in set_names:
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
            combos,
            args,
        ))

    payload = {
        "args": vars(args),
        "combos": [{"name": name, "clear": vals[0], "light": vals[1], "medium": vals[2], "heavy": vals[3]} for name, vals in combos],
        "sets": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({
        "out": args.out,
        "sets": [{"name": r["name"], "top5": r["results"][:5]} for r in results],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
