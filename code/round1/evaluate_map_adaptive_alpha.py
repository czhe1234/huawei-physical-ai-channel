#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS, idw_weights, parse_floats
from generate_geometric_steering_blend import steering_phase, unit_vectors
from spectrum_projection import MetricAccumulator, exact_pas_scores, exact_pdp_scores


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def full_predict(neigh_h, weights, radial, steer_phase=None):
    coeff = weights.astype(np.float32) * radial.astype(np.complex64)
    if steer_phase is None:
        return np.sum(coeff[:, None, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)
    return np.sum(coeff[:, None, None, None] * steer_phase[:, :, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)


def blend(base, steer, alpha):
    return ((1.0 - alpha) * base + alpha * steer).astype(np.complex64)


def eval_one(pred, truth):
    diff = pred - truth
    return (
        float(np.sum(np.abs(diff) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def make_split(pos, test_pos, mode, val_count, seed, k):
    if mode == "random":
        train_idx, val_idx = build_split(len(pos), val_count, seed)
        tree = cKDTree(pos[train_idx, :2])
        distances, local = tree.query(pos[val_idx, :2], k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            local = local[:, None]
        neighbor_idx = train_idx[local]
        return train_idx, val_idx, distances, neighbor_idx

    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], val_count, mode, seed)
    tree = cKDTree(pos[:, :2])
    distances_all, local_all = tree.query(pos[val_idx, :2], k=k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), k), dtype=np.int64)
    distances = np.empty((len(val_idx), k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:k]
        distances[i] = distances_all[i, keep][:k]
    return np.arange(len(pos)), val_idx, distances, neighbor_idx


def feature_rule(features, names, rule_name, threshold):
    if rule_name == "los_min_ge":
        return features[:, names.index("los_min_dist")] >= threshold
    if rule_name == "los_frac1_le":
        return features[:, names.index("los_frac_r1p0")] <= threshold
    if rule_name == "los_frac15_le":
        return features[:, names.index("los_frac_r1p5")] <= threshold
    if rule_name == "rx_wall2_ge":
        return features[:, names.index("rx_wall_dist2")] >= threshold
    raise ValueError(rule_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/map_adaptive_alpha.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--mode", choices=["random", "matched-test"], default="random")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--fixed-alphas", type=parse_floats, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--rule-alphas", type=parse_floats, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    args = parser.parse_args()

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    feature_data = np.load(args.features, allow_pickle=True)
    train_features = feature_data["train_features"]
    feature_names = [str(x) for x in feature_data["feature_names"]]

    _, val_idx, distances, neighbor_idx = make_split(pos, test_pos, args.mode, args.val_count, args.seed, args.k)
    val_features = train_features[val_idx]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    rules = []
    for threshold in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        rules.append(("los_min_ge", threshold))
    for threshold in [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]:
        rules.append(("los_frac1_le", threshold))
        rules.append(("los_frac15_le", threshold))
    for threshold in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        rules.append(("rx_wall2_ge", threshold))

    candidates = []
    for alpha in args.fixed_alphas:
        candidates.append(("fixed", "none", 0.0, alpha, alpha))
    for rule_name, threshold in rules:
        for clear_alpha in args.rule_alphas:
            for blocked_alpha in args.rule_alphas:
                if abs(clear_alpha - blocked_alpha) < 1e-12:
                    continue
                candidates.append(("rule", rule_name, threshold, clear_alpha, blocked_alpha))

    alpha_values = sorted(set(args.fixed_alphas + args.rule_alphas))
    alpha_to_col = {alpha: i for i, alpha in enumerate(alpha_values)}
    nums = np.zeros((len(val_idx), len(alpha_values)), dtype=np.float64)
    dens = np.zeros((len(val_idx), len(alpha_values)), dtype=np.float64)
    pas = np.zeros((len(val_idx), len(alpha_values)), dtype=np.float64)
    pdp = np.zeros((len(val_idx), len(alpha_values)), dtype=np.float64)
    clear_cache = {(name, thr): feature_rule(val_features, feature_names, name, thr) for name, thr in rules}

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        idx = neighbor_idx[ii]
        neigh = np.asarray(h[idx])
        weights = idw_weights(distances[ii], args.power)
        radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
        steer_phase = steering_phase(u_all, idx, vi, args.steering_scale)
        base = full_predict(neigh, weights, radial)
        steer = full_predict(neigh, weights, radial, steer_phase)
        for alpha in alpha_values:
            pred = blend(base, steer, alpha)
            col = alpha_to_col[alpha]
            num, den = eval_one(pred, truth)
            nums[ii, col] = num
            dens[ii, col] = den
            pas[ii, col] = exact_pas_scores(pred, truth, "sumpol")[0]
            pdp[ii, col] = exact_pdp_scores(pred, truth)[0]

        if (ii + 1) % 20 == 0 or ii + 1 == len(val_idx):
            print(f"evaluated {ii + 1}/{len(val_idx)}", flush=True)

    results = []
    for candidate in candidates:
        kind, rule_name, threshold, clear_alpha, blocked_alpha = candidate
        if kind == "fixed":
            cols = np.full(len(val_idx), alpha_to_col[clear_alpha], dtype=np.int64)
            clear_count = 0
        else:
            clear_mask = clear_cache[(rule_name, threshold)]
            clear_count = int(np.sum(clear_mask))
            cols = np.where(clear_mask, alpha_to_col[clear_alpha], alpha_to_col[blocked_alpha])
        row_ids = np.arange(len(val_idx))
        num = float(np.sum(nums[row_ids, cols]))
        den = float(np.sum(dens[row_ids, cols]))
        nmse = num / max(den, 1e-30)
        mean_pas = float(np.mean(pas[row_ids, cols]))
        mean_pdp = float(np.mean(pdp[row_ids, cols]))
        inv_nmse = float(1.0 / (1.0 + nmse))
        score = 0.4 * mean_pas + 0.4 * mean_pdp + 0.2 * inv_nmse
        results.append({
            "kind": kind,
            "rule": rule_name,
            "threshold": float(threshold),
            "clear_alpha": float(clear_alpha),
            "blocked_alpha": float(blocked_alpha),
            "clear_count": clear_count,
            "global_nmse_direct": float(nmse),
            "score": score,
            "pas": mean_pas,
            "pdp": mean_pdp,
            "global_nmse": float(nmse),
            "inv_nmse": inv_nmse,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    payload = {
        "mode": args.mode,
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
