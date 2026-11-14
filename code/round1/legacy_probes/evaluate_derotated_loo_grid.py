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


def choose_val_indices(pos_xy, test_xy, count, mode, seed):
    if mode == "all":
        return np.arange(len(pos_xy), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if mode == "random":
        return np.sort(rng.choice(len(pos_xy), size=count, replace=False))
    if mode != "matched-test":
        raise ValueError(f"unknown val mode {mode}")

    train_tree = cKDTree(pos_xy)
    train_nn, _ = train_tree.query(pos_xy, k=2, workers=-1)
    train_nn = train_nn[:, 1]
    test_nn, _ = train_tree.query(test_xy, k=1, workers=-1)
    targets = np.quantile(test_nn, (np.arange(count) + 0.5) / count)
    order = np.argsort(train_nn)
    sorted_nn = train_nn[order]
    used = np.zeros(len(pos_xy), dtype=bool)
    chosen = []
    for target in targets:
        center = int(np.searchsorted(sorted_nn, target))
        best = None
        best_delta = np.inf
        lo = max(0, center - 64)
        hi = min(len(order), center + 65)
        for j in range(lo, hi):
            idx = int(order[j])
            if used[idx]:
                continue
            delta = abs(float(train_nn[idx]) - float(target))
            if delta < best_delta:
                best = idx
                best_delta = delta
        if best is None:
            for idx in order[np.argsort(np.abs(sorted_nn - target))]:
                idx = int(idx)
                if not used[idx]:
                    best = idx
                    break
        used[best] = True
        chosen.append(best)
    return np.sort(np.asarray(chosen, dtype=np.int64))


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
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/derotated_loo_grid.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--val-mode", choices=["matched-test", "random", "all"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="8,10,12,14,16,20")
    parser.add_argument("--powers", type=parse_floats, default="1.5,1.8,2.0,2.2,2.5")
    parser.add_argument("--betas", type=parse_floats, default="140.0,140.2,140.3,140.4,140.6")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--pas-mode", choices=["sumpol", "fullpol"], default="sumpol")
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    max_k = max(args.k_values)

    tree = cKDTree(pos[:, :2])
    distances, local = tree.query(pos[val_idx, :2], k=max_k + 1, workers=-1)
    if max_k + 1 == 1:
        distances = distances[:, None]
        local = local[:, None]

    neighbor_idx = np.empty((len(val_idx), max_k), dtype=np.int64)
    neighbor_dist_xy = np.empty((len(val_idx), max_k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local[i] != vi
        neighbor_idx[i] = local[i, keep][:max_k]
        neighbor_dist_xy[i] = distances[i, keep][:max_k]

    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    candidates = [(k, power, beta) for k in args.k_values for power in args.powers for beta in args.betas]
    accs = {f"k{k}_p{power:g}_b{beta:g}": MetricAccumulator(args.pas_mode) for k, power, beta in candidates}

    for start in range(0, len(val_idx), args.batch):
        end = min(start + args.batch, len(val_idx))
        truth = np.asarray(h[val_idx[start:end]])
        neigh = np.asarray([np.asarray(h[row]) for row in neighbor_idx[start:end]])
        neighbor_dist = dist_all[neighbor_idx[start:end]]
        query_dist = dist_all[val_idx[start:end]]
        d_batch = neighbor_dist_xy[start:end]
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
        "val_count": int(len(val_idx)),
        "val_mode": args.val_mode,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:20]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
