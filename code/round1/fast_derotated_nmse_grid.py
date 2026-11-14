#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x]


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def weights_for(distances, k, power):
    d = distances[:, :k].astype(np.float64)
    zero = d[:, 0] < 1e-12
    w = 1.0 / np.maximum(d, 1e-6) ** power
    w /= np.sum(w, axis=1, keepdims=True)
    if np.any(zero):
        w[zero] = 0.0
        w[zero, 0] = 1.0
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/fast_derotated_nmse_grid.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument("--k-values", type=parse_ints, default="4,6,8,12,16,20,24,32,48,64")
    parser.add_argument("--powers", type=parse_floats, default="1.0,1.5,2.0,2.5,3.0,4.0")
    parser.add_argument("--betas", type=parse_floats, default="138.5,139.0,139.5,140.0,140.3,140.6,141.0,141.5")
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=args.max_k, workers=-1)
    if args.max_k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)

    n_val = len(val_idx)
    gram = np.empty((n_val, args.max_k, args.max_k), dtype=np.complex64)
    cross = np.empty((n_val, args.max_k), dtype=np.complex64)
    truth_norm = np.empty(n_val, dtype=np.float64)

    for i, vi in enumerate(val_idx):
        truth = np.asarray(h[vi]).reshape(-1)
        neigh = np.asarray(h[neighbor_idx[i]]).reshape(args.max_k, -1)
        truth_norm[i] = np.sum(np.abs(truth) ** 2, dtype=np.float64)
        cross[i] = np.sum(neigh * np.conj(truth)[None, :], axis=1)
        gram[i] = neigh @ np.conj(neigh.T)
        if (i + 1) % 25 == 0 or i + 1 == n_val:
            print(f"cached {i + 1}/{n_val}", flush=True)

    den = float(np.sum(truth_norm))
    results = []
    query_dist = dist_all[val_idx]
    neighbor_dist = dist_all[neighbor_idx]

    for k in args.k_values:
        if k > args.max_k:
            continue
        g = gram[:, :k, :k]
        c = cross[:, :k]
        for power in args.powers:
            w = weights_for(distances, k, power)
            for beta in args.betas:
                phase = np.exp(1j * beta * (neighbor_dist[:, :k] - query_dist[:, None]))
                a = w * phase
                pred_norm = np.einsum("bi,bij,bj->b", a, g, np.conj(a), optimize=True).real
                cr = np.einsum("bi,bi->b", a, c, optimize=True).real
                nmse = float((np.sum(pred_norm - 2.0 * cr + truth_norm)) / den)
                results.append({
                    "k": int(k),
                    "power": float(power),
                    "beta": float(beta),
                    "global_nmse": nmse,
                    "inv_nmse": float(1.0 / (1.0 + nmse)),
                })
    results.sort(key=lambda row: row["global_nmse"])
    payload = {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
