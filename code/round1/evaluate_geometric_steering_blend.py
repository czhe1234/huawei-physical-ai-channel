#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_geometric_steering import BS_POS, idw_weights, predict_config, unit_vectors
from spectrum_projection import MetricAccumulator


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x]


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/geometric_steering_blend.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--mode", choices=["random", "matched-test"], default="random")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--alphas", type=parse_floats, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--batch", type=int, default=5)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    if args.mode == "random":
        train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
        tree = cKDTree(pos[train_idx, :2])
        distances, local = tree.query(pos[val_idx, :2], k=args.k, workers=-1)
        if args.k == 1:
            distances = distances[:, None]
            local = local[:, None]
        neighbor_idx = train_idx[local]
    else:
        val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, "matched-test", args.seed)
        tree = cKDTree(pos[:, :2])
        distances_all, local_all = tree.query(pos[val_idx, :2], k=args.k + 1, workers=-1)
        neighbor_idx = np.empty((len(val_idx), args.k), dtype=np.int64)
        distances = np.empty((len(val_idx), args.k), dtype=np.float64)
        for i, vi in enumerate(val_idx):
            keep = local_all[i] != vi
            neighbor_idx[i] = local_all[i, keep][:args.k]
            distances[i] = distances_all[i, keep][:args.k]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    cfg_base = {"name": "baseline", "order": "phv", "h_axis": "none", "v_axis": "none", "bs_scale": 0.0, "ue_scale": 0.0}
    cfg_geo = {"name": "bs_steering", "order": "phv", "h_axis": "uy", "v_axis": "nuz", "bs_scale": 4.71238898038, "ue_scale": 0.0}
    accs = {alpha: MetricAccumulator("sumpol") for alpha in args.alphas}

    for start in range(0, len(val_idx), args.batch):
        end = min(start + args.batch, len(val_idx))
        truth = np.asarray(h[val_idx[start:end]])
        preds = {alpha: np.empty_like(truth) for alpha in args.alphas}
        for j, row in enumerate(range(start, end)):
            vi = val_idx[row]
            idx = neighbor_idx[row]
            neigh = np.asarray(h[idx])
            weights = idw_weights(distances[row], args.power)
            radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
            base = predict_config(neigh, weights, radial, u_all, idx, vi, cfg_base)
            geo = predict_config(neigh, weights, radial, u_all, idx, vi, cfg_geo)
            for alpha in args.alphas:
                preds[alpha][j] = ((1.0 - alpha) * base + alpha * geo).astype(np.complex64)
        for alpha in args.alphas:
            accs[alpha].add(preds[alpha], truth)
        print(f"evaluated {end}/{len(val_idx)}", flush=True)

    results = []
    for alpha in args.alphas:
        results.append({
            "alpha": float(alpha),
            "mode": args.mode,
            "seed": int(args.seed),
            "k": int(args.k),
            "power": float(args.power),
            "beta": float(args.beta),
            **accs[alpha].result(),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    payload = {"seed": int(args.seed), "val_count": int(args.val_count), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
