#!/usr/bin/env python3
from project_paths import PATHS
"""Generate v46: v36 plus heavy-only dead-zone attenuation.

This keeps the online-confirmed v36 phase/geometry structure, then applies a
scalar gain to heavy blocked test points whose geometry-feature nearest
neighbors in the training set have very low channel power.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from build_map_features import BS_POS
from evaluate_local_ad_paths import idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_s1style_geom_blend import (
    build_height_map,
    geometry_cache,
    predict_phase_idw,
    select_geom_neighbors,
    train_log_power,
)
from generate_geometric_steering_blend import unit_vectors
from generate_regime_s1style_geom_blend import rho_for_blocked, validate_shapes


def train_power(train_h, batch=64):
    out = np.empty(train_h.shape[0], dtype=np.float64)
    for start in range(0, train_h.shape[0], batch):
        end = min(start + batch, train_h.shape[0])
        block = np.asarray(train_h[start:end])
        out[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return out


def sample_height_features(points, data_dir, radii=(0, 1, 2, 3, 5)):
    hmap, origin, cell = build_height_map(Path(data_dir) / "Round1_Map.ply", cell=1.0, z_min=1.0)
    ij = np.floor((points[:, :2] - origin[None, :]) / cell).astype(np.int64)
    feats = []
    for radius in radii:
        grid = hmap if radius == 0 else maximum_filter(hmap, size=2 * radius + 1, mode="constant", cval=0.0)
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


def robust_scale(train_x, query_x):
    med = np.median(train_x, axis=0)
    q25, q75 = np.percentile(train_x, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return (train_x - med[None, :]) / scale[None, :], (query_x - med[None, :]) / scale[None, :]


def weighted_knn_scores(train_x, train_labels, query_x, k, power):
    tree = cKDTree(train_x)
    kk = min(int(k), len(train_x))
    dist, idx = tree.query(query_x, k=kk, workers=-1)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    scores = np.empty(len(query_x), dtype=np.float64)
    for i in range(len(query_x)):
        w = 1.0 / np.maximum(dist[i], 1e-6) ** power
        w = w / np.sum(w)
        scores[i] = float(np.dot(w, train_labels[idx[i]]))
    return scores


def build_dead_scores(args, pos, test_pos, train_h, train_blocked, test_blocked):
    payload = np.load(args.map_features)
    train_map = payload["train_features"].astype(np.float32)
    test_map = payload["test_features"].astype(np.float32)
    train_hfeat = sample_height_features(pos, args.data)
    test_hfeat = sample_height_features(test_pos, args.data)
    train_x = make_features(pos, train_map, train_blocked, train_hfeat)
    test_x = make_features(test_pos, test_map, test_blocked, test_hfeat)
    train_xs, test_xs = robust_scale(train_x, test_x)

    pwr = train_power(train_h)
    labels = (pwr <= args.dead_threshold).astype(np.float64)
    score = weighted_knn_scores(train_xs, labels, test_xs, args.dead_k, args.feature_power)
    clipped = np.maximum(0.0, (score - args.dead_floor) / max(1.0 - args.dead_floor, 1e-6))
    gains = np.exp(-args.dead_lambda * clipped).astype(np.float32)
    if args.heavy_only:
        gains = np.where(test_blocked > 100, gains, 1.0).astype(np.float32)
    return score.astype(np.float32), gains.astype(np.float32), pwr


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    validate_shapes(pos, test_pos, train_h)

    train_blocked, test_blocked = geometry_cache(args.data, args.cache, args.force_geometry)
    dead_score, gains, train_pwr = build_dead_scores(args, pos, test_pos, train_h, train_blocked, test_blocked)
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
        "heavy": 0,
        "rho_sum": 0.0,
        "gain_sum": 0.0,
        "attenuated_count": 0,
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
            rho, label = rho_for_blocked(
                int(test_blocked[i]),
                args.clear_rho,
                args.light_rho,
                args.medium_rho,
                args.heavy_rho,
            )
            pred = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
            pred = (pred * gains[i]).astype(np.complex64)
            out[i] = pred
            branch_stats[label] += 1
            branch_stats["rho_sum"] += rho
            branch_stats["gain_sum"] += float(gains[i])
            branch_stats["attenuated_count"] += int(gains[i] < 0.999)

        block = np.asarray(out[start:end])
        if not (np.isfinite(block.real).all() and np.isfinite(block.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)

    branch_stats["rho_mean"] = branch_stats["rho_sum"] / max(len(test_pos), 1)
    branch_stats["gain_mean"] = branch_stats["gain_sum"] / max(len(test_pos), 1)
    return {
        "method": "deadzone_attenuated_regime_s1style_geom_blend",
        "base": "v36_regime_s1style_geom_blend",
        "rule": "v36 prediction multiplied by heavy-only low-power-neighbor dead-zone gain",
        "clear_rho": float(args.clear_rho),
        "light_rho": float(args.light_rho),
        "medium_rho": float(args.medium_rho),
        "heavy_rho": float(args.heavy_rho),
        "dead_threshold": float(args.dead_threshold),
        "dead_k": int(args.dead_k),
        "dead_floor": float(args.dead_floor),
        "dead_lambda": float(args.dead_lambda),
        "feature_power": float(args.feature_power),
        "heavy_only": bool(args.heavy_only),
        "dead_score_quantiles": [float(x) for x in np.quantile(dead_score, [0, 0.5, 0.9, 0.95, 0.98, 0.99, 1.0])],
        "gain_quantiles": [float(x) for x in np.quantile(gains, [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0])],
        "train_power_quantiles": [float(x) for x in np.quantile(train_pwr, [0, .001, .005, .01, .02, .05, .1, .2, .5])],
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
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/deadzone_attenuation_generate.json"))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--clear-rho", type=float, default=0.0)
    parser.add_argument("--light-rho", type=float, default=0.0)
    parser.add_argument("--medium-rho", type=float, default=1.0)
    parser.add_argument("--heavy-rho", type=float, default=4.0)
    parser.add_argument("--dead-threshold", type=float, default=1e-6)
    parser.add_argument("--dead-k", type=int, default=64)
    parser.add_argument("--dead-floor", type=float, default=0.0)
    parser.add_argument("--dead-lambda", type=float, default=8.0)
    parser.add_argument("--feature-power", type=float, default=2.0)
    parser.add_argument("--heavy-only", action="store_true", default=True)
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
