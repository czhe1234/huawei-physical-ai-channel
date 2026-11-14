#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_frequency_dependent_phase import predict_freq
from evaluate_local_ad_paths import BS_POS, idw_weights, parse_floats, parse_ints
from generate_geometric_steering_blend import unit_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/frequency_phase_nmse_fast.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--val-mode", choices=["random", "matched-test"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k-values", type=parse_ints, default="16,24,32")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gammas", type=parse_floats, default="-0.2,-0.15,-0.13,-0.11,-0.08,-0.05,0,0.05,0.08,0.11,0.13,0.15,0.2")
    parser.add_argument("--alphas", type=parse_floats, default="0.5,0.75,1.0")
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    max_k = max(args.k_values)
    tree = cKDTree(pos[:, :2])
    d_all, local_all = tree.query(pos[val_idx, :2], k=max_k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), max_k), dtype=np.int64)
    distances = np.empty((len(val_idx), max_k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:max_k]
        distances[i] = d_all[i, keep][:max_k]

    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    candidates = [(k, g, a) for k in args.k_values for g in args.gammas for a in args.alphas]
    nums = np.zeros(len(candidates), dtype=np.float64)
    dens = np.zeros(len(candidates), dtype=np.float64)

    for row, vi in enumerate(val_idx):
        truth = np.asarray(h[vi])
        den = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        idx_full = neighbor_idx[row]
        neigh_full = np.asarray(h[idx_full])
        for ci, (k, gamma, alpha) in enumerate(candidates):
            idx = idx_full[:k]
            neigh = neigh_full[:k]
            weights = idw_weights(distances[row, :k], args.power)
            dist_delta = dist_all[idx] - dist_all[vi]
            pred = predict_freq(
                neigh,
                weights,
                dist_delta,
                u_all,
                idx,
                vi,
                args.beta,
                gamma,
                alpha,
                args.steering_scale,
            )
            nums[ci] += float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
            dens[ci] += den
        if (row + 1) % 10 == 0 or row + 1 == len(val_idx):
            print(f"evaluated {row + 1}/{len(val_idx)}", flush=True)

    results = []
    for (k, gamma, alpha), num, den in zip(candidates, nums, dens):
        nmse = float(num / max(den, 1e-30))
        results.append({
            "k": int(k),
            "gamma": float(gamma),
            "alpha": float(alpha),
            "global_nmse": nmse,
            "inv_nmse": float(1.0 / (1.0 + nmse)),
        })
    results.sort(key=lambda r: r["global_nmse"])
    payload = {
        "seed": int(args.seed),
        "val_mode": args.val_mode,
        "val_count": int(len(val_idx)),
        "beta": float(args.beta),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:40]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
