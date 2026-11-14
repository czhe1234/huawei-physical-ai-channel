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


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x]


def build_random_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def eval_candidate(h, pos, val_idx, neighbor_idx, distances, cfg, k, power, beta):
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    num = 0.0
    den = 0.0
    for c, vi in enumerate(val_idx):
        idx = neighbor_idx[c, :k]
        neigh = np.asarray(h[idx])
        truth = np.asarray(h[vi])
        weights = idw_weights(distances[c, :k], power)
        radial = np.exp(1j * beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
        pred = predict_config(neigh, weights, radial, u_all, idx, vi, cfg)
        diff = pred - truth
        num += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        den += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
    nmse = num / max(den, 1e-30)
    return {"global_nmse": float(nmse), "inv_nmse": float(1.0 / (1.0 + nmse))}


def make_neighbors(pos_xy, train_idx, val_idx, max_k, mode):
    if mode == "random":
        tree = cKDTree(pos_xy[train_idx])
        distances, local = tree.query(pos_xy[val_idx], k=max_k, workers=-1)
        if max_k == 1:
            distances = distances[:, None]
            local = local[:, None]
        return train_idx[local], distances

    tree = cKDTree(pos_xy)
    distances_all, local_all = tree.query(pos_xy[val_idx], k=max_k + 1, workers=-1)
    if max_k + 1 == 1:
        distances_all = distances_all[:, None]
        local_all = local_all[:, None]
    neighbor_idx = np.empty((len(val_idx), max_k), dtype=np.int64)
    distances = np.empty((len(val_idx), max_k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:max_k]
        distances[i] = distances_all[i, keep][:max_k]
    return neighbor_idx, distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/geometric_steering_robust.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seeds", type=parse_ints, default="20260703,20260704,20260705")
    parser.add_argument("--modes", nargs="+", choices=["random", "matched-test"], default=["random", "matched-test"])
    parser.add_argument("--k-values", type=parse_ints, default="16,24")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--h-axis", default="uy")
    parser.add_argument("--v-axis", default="nuz")
    parser.add_argument("--bs-scale", type=float, default=4.71238898038)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    max_k = max(args.k_values)
    cfgs = [
        {"name": "baseline", "order": "phv", "h_axis": "none", "v_axis": "none", "bs_scale": 0.0, "ue_scale": 0.0},
        {"name": "bs_steering", "order": "phv", "h_axis": args.h_axis, "v_axis": args.v_axis, "bs_scale": float(args.bs_scale), "ue_scale": 0.0},
    ]

    rows = []
    for mode in args.modes:
        seeds = args.seeds if mode == "random" else [args.seeds[0]]
        for seed in seeds:
            if mode == "random":
                train_idx, val_idx = build_random_split(len(pos), args.val_count, seed)
            else:
                val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, "matched-test", seed)
                train_idx = None
            neighbor_idx, distances = make_neighbors(pos[:, :2], train_idx, val_idx, max_k, mode)
            for k in args.k_values:
                for cfg in cfgs:
                    metrics = eval_candidate(h, pos, val_idx, neighbor_idx, distances, cfg, k, args.power, args.beta)
                    row = {
                        "mode": mode,
                        "seed": int(seed),
                        "val_count": int(len(val_idx)),
                        "candidate": cfg["name"],
                        "k": int(k),
                        "power": float(args.power),
                        "beta": float(args.beta),
                        "h_axis": cfg["h_axis"],
                        "v_axis": cfg["v_axis"],
                        "bs_scale": float(cfg["bs_scale"]),
                        **metrics,
                    }
                    rows.append(row)
                    print(json.dumps(row, indent=2), flush=True)
    rows.sort(key=lambda r: (r["mode"], r["seed"], r["global_nmse"]))
    summary = {}
    for candidate in sorted(set(r["candidate"] for r in rows)):
        for k in args.k_values:
            vals = [r["global_nmse"] for r in rows if r["candidate"] == candidate and r["k"] == k]
            summary[f"{candidate}_k{k}"] = {
                "mean_nmse": float(np.mean(vals)),
                "max_nmse": float(np.max(vals)),
                "min_nmse": float(np.min(vals)),
                "count": int(len(vals)),
            }
    payload = {"rows": rows, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print("SUMMARY")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
