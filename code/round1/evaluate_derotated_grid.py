#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from spectrum_projection import MetricAccumulator


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x]


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def idw_matrix(distances, k, power):
    d = np.asarray(distances[:, :k], dtype=np.float64)
    zero = d[:, 0] < 1e-12
    w = 1.0 / np.maximum(d, 1e-6) ** power
    w /= np.sum(w, axis=1, keepdims=True)
    if np.any(zero):
        w[zero] = 0.0
        w[zero, 0] = 1.0
    return w.astype(np.float32)


def predict_batch(neigh, neighbor_dist, query_dist, distances, k, power, beta):
    weights = idw_matrix(distances, k, power)
    phase = np.exp(1j * beta * (neighbor_dist[:, :k] - query_dist[:, None])).astype(np.complex64)
    coeff = weights * phase
    return np.sum(coeff[:, :, None, None, None] * neigh[:, :k], axis=1).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/derotated_grid.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="8,12,16,20,24")
    parser.add_argument("--powers", type=parse_floats, default="1.5,2.0,2.5")
    parser.add_argument("--betas", type=parse_floats, default="139.6,140.0,140.3,140.6,141.0")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--pas-mode", choices=["sumpol", "fullpol"], default="sumpol")
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

    candidates = []
    for k in args.k_values:
        for power in args.powers:
            for beta in args.betas:
                candidates.append((k, power, beta))

    accs = {f"k{k}_p{power:g}_b{beta:g}": MetricAccumulator(args.pas_mode) for k, power, beta in candidates}

    for start in range(0, len(val_idx), args.batch):
        end = min(start + args.batch, len(val_idx))
        truth = np.asarray(h[val_idx[start:end]])
        neigh = np.asarray([np.asarray(h[row]) for row in neighbor_idx[start:end]])
        neighbor_dist = dist_all[neighbor_idx[start:end]]
        query_dist = dist_all[val_idx[start:end]]
        d_batch = distances[start:end]
        for k, power, beta in candidates:
            name = f"k{k}_p{power:g}_b{beta:g}"
            pred = predict_batch(neigh, neighbor_dist, query_dist, d_batch, k, power, beta)
            accs[name].add(pred, truth)
        print(f"evaluated {end}/{len(val_idx)}", flush=True)

    results = []
    for k, power, beta in candidates:
        name = f"k{k}_p{power:g}_b{beta:g}"
        results.append({
            "k": int(k),
            "power": float(power),
            "beta": float(beta),
            "pas_mode": args.pas_mode,
            **accs[name].result(),
        })
    results.sort(key=lambda row: row["score"], reverse=True)
    payload = {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:20]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
