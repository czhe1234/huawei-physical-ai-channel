#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_geometric_steering import (
    BS_POS,
    axis_value,
    bs_indices,
    build_split,
    idw_weights,
    parse_floats,
    parse_ints,
    predict_config,
    unit_vectors,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/geometric_steering_param_grid.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="12,16,20,24")
    parser.add_argument("--powers", type=parse_floats, default="1.5,2.0,2.5")
    parser.add_argument("--betas", type=parse_floats, default="140.0,140.3,140.6")
    parser.add_argument("--order", default="phv")
    parser.add_argument("--h-axis", default="uy")
    parser.add_argument("--v-axis", default="nuz")
    parser.add_argument("--bs-scale", type=float, default=4.71238898038)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    max_k = max(args.k_values)
    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=max_k, workers=-1)
    if max_k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    cfg = {
        "name": f"bs_{args.order}_h{args.h_axis}_v{args.v_axis}_s{args.bs_scale:g}",
        "order": args.order,
        "h_axis": args.h_axis,
        "v_axis": args.v_axis,
        "bs_scale": float(args.bs_scale),
        "ue_scale": 0.0,
    }
    candidates = [(k, p, b) for k in args.k_values for p in args.powers for b in args.betas]
    nums = np.zeros(len(candidates), dtype=np.float64)
    dens = np.zeros(len(candidates), dtype=np.float64)

    for c, vi in enumerate(val_idx):
        idx_full = neighbor_idx[c]
        neigh_full = np.asarray(h[idx_full])
        truth = np.asarray(h[vi])
        den = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        for j, (k, power, beta) in enumerate(candidates):
            idx = idx_full[:k]
            neigh = neigh_full[:k]
            weights = idw_weights(distances[c, :k], power)
            radial = np.exp(1j * beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
            pred = predict_config(neigh, weights, radial, u_all, idx, vi, cfg)
            diff = pred - truth
            nums[j] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
            dens[j] += den
        if (c + 1) % 10 == 0 or c + 1 == len(val_idx):
            print(f"evaluated {c + 1}/{len(val_idx)}", flush=True)

    results = []
    for (k, power, beta), num, den in zip(candidates, nums, dens):
        nmse = float(num / max(den, 1e-30))
        results.append({
            **cfg,
            "k": int(k),
            "power": float(power),
            "beta": float(beta),
            "global_nmse": nmse,
            "inv_nmse": float(1.0 / (1.0 + nmse)),
        })
    results.sort(key=lambda r: r["global_nmse"])
    payload = {"seed": int(args.seed), "val_count": int(args.val_count), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
