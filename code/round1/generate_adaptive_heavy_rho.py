#!/usr/bin/env python3
from project_paths import PATHS
"""Generate adaptive-heavy regime geometry blend.

This keeps the confirmed v36 structure but splits the heavy blocked test
points into low/mid/high rho groups using geometry-aware test features.
"""

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
    make_feature_matrix,
    novelty_label,
    qtiles,
    robust_standardize,
)
from evaluate_local_ad_paths import idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_s1style_geom_blend import geometry_cache, predict_phase_idw, select_geom_neighbors, train_log_power
from generate_geometric_steering_blend import unit_vectors


DEFAULT_FEATURE_INDICES = "0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41"


def validate_shapes(pos, test_pos, train_h):
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"bad train position shape {pos.shape}")
    if test_pos.ndim != 2 or test_pos.shape[1] != 3:
        raise ValueError(f"bad test position shape {test_pos.shape}")
    if train_h.ndim != 4 or train_h.shape[1:] != (256, 4, 192):
        raise ValueError(f"bad channel shape {train_h.shape}")


def los_label(blocked):
    if blocked <= 0:
        return "clear"
    if blocked <= 25:
        return "light"
    if blocked <= 100:
        return "medium"
    return "heavy"


def build_test_labels(train_pos, test_pos, train_map, test_map, train_blocked, test_blocked, feature_indices):
    tree_xy = cKDTree(train_pos[:, :2])
    test_knn_dist, test_knn_idx = tree_xy.query(test_pos[:, :2], k=32, workers=-1)
    train_knn_dist, train_knn_idx_raw = tree_xy.query(train_pos[:, :2], k=33, workers=-1)
    train_knn_dist = train_knn_dist[:, 1:33]

    train_feat = make_feature_matrix(train_pos, train_map, train_blocked, train_knn_dist, feature_indices)
    test_feat = make_feature_matrix(test_pos, test_map, test_blocked, test_knn_dist, feature_indices)
    train_feat_z, test_feat_z = robust_standardize(train_feat, test_feat)
    feat_tree = cKDTree(train_feat_z)
    test_feat_dist, _ = feat_tree.query(test_feat_z, k=32, workers=-1)
    train_feat_dist_raw, _ = feat_tree.query(train_feat_z, k=33, workers=-1)
    train_feat_dist = train_feat_dist_raw[:, 1:33]

    nn_q = qtiles(train_knn_dist[:, 0])
    feat_q = qtiles(train_feat_dist[:, 0])
    labels = []
    for i in range(len(test_pos)):
        boundary, clear_frac, block_std = boundary_label(train_blocked[test_knn_idx[i, :16]])
        vec = test_pos[i] - BS_POS
        labels.append({
            "los": los_label(test_blocked[i]),
            "density": density_label(test_knn_dist[i, 0], nn_q["p25"], nn_q["p75"], nn_q["p90"]),
            "novelty": novelty_label(test_feat_dist[i, 0], feat_q["p50"], feat_q["p75"], feat_q["p90"]),
            "boundary": boundary,
            "local_clear_frac": float(clear_frac),
            "local_block_std": float(block_std),
            "nn1": float(test_knn_dist[i, 0]),
            "feature_nn1": float(test_feat_dist[i, 0]),
            "blocked": float(test_blocked[i]),
            "bs_dist": float(np.linalg.norm(vec)),
        })
    return labels


def heavy_rho(label, args):
    risky = (
        label["novelty"] == "very_novel"
        or label["density"] in {"sparse", "very_sparse"}
        or label["blocked"] <= args.heavy_risk_blocked
        or label["feature_nn1"] > args.heavy_risk_feature_nn1
    )
    strong = (
        label["blocked"] >= args.heavy_strong_blocked
        and label["novelty"] in {"familiar", "normal"}
        and label["density"] in {"dense", "normal"}
        and label["feature_nn1"] <= args.heavy_strong_feature_nn1
        and label["nn1"] <= args.heavy_strong_nn1
    )
    if risky:
        return float(args.heavy_low_rho), "heavy_low"
    if strong:
        return float(args.heavy_high_rho), "heavy_high"
    return float(args.heavy_mid_rho), "heavy_mid"


def rho_for_label(label, args):
    los = label["los"]
    if los == "clear":
        return float(args.clear_rho), "clear"
    if los == "light":
        return float(args.light_rho), "light"
    if los == "medium":
        return float(args.medium_rho), "medium"
    return heavy_rho(label, args)


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    validate_shapes(pos, test_pos, train_h)

    train_blocked, test_blocked = geometry_cache(args.data, args.cache, args.force_geometry)
    map_payload = np.load(args.map_features)
    train_map = map_payload["train_features"].astype(np.float32)
    test_map = map_payload["test_features"].astype(np.float32)
    feature_indices = [int(x) for x in args.feature_indices.split(",") if x.strip()]
    labels = build_test_labels(pos, test_pos, train_map, test_map, train_blocked, test_blocked, feature_indices)
    logpw = train_log_power(train_h)

    max_k = max(args.k, args.geom_k)
    tree = cKDTree(pos[:, :2])
    distances, neighbor_idx = tree.query(test_pos[:, :2], k=max_k, workers=-1)
    if max_k == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]

    pos_all = np.concatenate([pos, test_pos], axis=0)
    dist_all = np.linalg.norm(pos_all - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos_all)
    test_offset = len(pos)

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

    out = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), train_h.shape[1], train_h.shape[2], train_h.shape[3]),
    )
    branch_stats = {
        "clear": 0,
        "light": 0,
        "medium": 0,
        "heavy_low": 0,
        "heavy_mid": 0,
        "heavy_high": 0,
        "rho_sum": 0.0,
    }
    for start in range(0, len(test_pos), args.batch):
        end = min(start + args.batch, len(test_pos))
        for i in range(start, end):
            idx = neighbor_idx[i, : args.k]
            rel_xy = pos[idx, :2] - test_pos[i, :2]
            coeff = affine_coefficients(rel_xy, distances[i, : args.k], args.power, args.ridge)
            neigh = np.asarray(train_h[idx])
            qi = test_offset + i
            v9 = predict_affine(
                neigh,
                coeff,
                dist_all[idx] - dist_all[qi],
                u_all,
                idx,
                qi,
                args.beta,
                args.gamma,
                args.alpha,
                args.steering_scale,
            )

            gidx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, test_pos[i], test_blocked[i], None, geom_cfg)
            weights = idw_weights(eff_d, args.power)
            geom = predict_phase_idw(
                np.asarray(train_h[gidx]),
                weights,
                dist_all[gidx] - dist_all[qi],
                args.beta,
                args.gamma,
                logpw if args.geom_power_sep else None,
                gidx if args.geom_power_sep else None,
            )
            rho, label = rho_for_label(labels[i], args)
            out[i] = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
            branch_stats[label] += 1
            branch_stats["rho_sum"] += rho

        block = np.asarray(out[start:end])
        if not (np.isfinite(block.real).all() and np.isfinite(block.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)

    branch_stats["rho_mean"] = branch_stats["rho_sum"] / max(len(test_pos), 1)
    branch_stats["heavy_rho_mean"] = (
        branch_stats["rho_sum"] - args.medium_rho * branch_stats["medium"]
    ) / max(branch_stats["heavy_low"] + branch_stats["heavy_mid"] + branch_stats["heavy_high"], 1)
    return {
        "method": "adaptive_heavy_rho",
        "base": "v9_physical_local_linear",
        "rule": "clear/light/medium fixed; heavy split by blocked, density, novelty and local feature distance",
        "clear_rho": float(args.clear_rho),
        "light_rho": float(args.light_rho),
        "medium_rho": float(args.medium_rho),
        "heavy_low_rho": float(args.heavy_low_rho),
        "heavy_mid_rho": float(args.heavy_mid_rho),
        "heavy_high_rho": float(args.heavy_high_rho),
        "heavy_risk_blocked": float(args.heavy_risk_blocked),
        "heavy_strong_blocked": float(args.heavy_strong_blocked),
        "heavy_risk_feature_nn1": float(args.heavy_risk_feature_nn1),
        "heavy_strong_feature_nn1": float(args.heavy_strong_feature_nn1),
        "heavy_strong_nn1": float(args.heavy_strong_nn1),
        "k": int(args.k),
        "geom_k": int(args.geom_k),
        "geom_power_sep": bool(args.geom_power_sep),
        "power": float(args.power),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "ridge": float(args.ridge),
        "branch_stats": branch_stats,
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/adaptive_heavy_rho_generate.json"))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--feature-indices", default=DEFAULT_FEATURE_INDICES)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--clear-rho", type=float, default=0.0)
    parser.add_argument("--light-rho", type=float, default=0.0)
    parser.add_argument("--medium-rho", type=float, default=1.0)
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
    parser.add_argument("--geom-power-sep", action="store_true", default=True)
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
    result = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
