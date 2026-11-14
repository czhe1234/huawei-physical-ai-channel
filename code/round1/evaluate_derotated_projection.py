#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from spectrum_projection import MetricAccumulator, project_pas_sumpol, project_pdp
from evaluate_derotated_loo_grid import choose_val_indices


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def parse_methods(text):
    return [x for x in str(text).split(",") if x]


def idw_weights(distances, power):
    d = np.asarray(distances, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w.astype(np.float32)
    w = 1.0 / np.maximum(d, 1e-6) ** power
    w /= np.sum(w)
    return w.astype(np.float32)


def derotated_base(neigh, distances, train_dist, query_dist, beta, power):
    weights = idw_weights(distances, power)
    phase = np.exp(1j * beta * (train_dist - query_dist)).astype(np.complex64)
    pred = np.sum(weights[:, None, None, None] * phase[:, None, None, None] * neigh, axis=0)
    return pred.astype(np.complex64), weights


def apply_method(base, neigh, weights, method, pass_count):
    if method == "base":
        return base
    if not method.startswith("base+"):
        raise ValueError(f"unknown method {method}")
    steps = method.split("+")[1:]
    pred = base.astype(np.complex64, copy=True)
    for _ in range(pass_count):
        for step in steps:
            if step == "pdp":
                pred = project_pdp(pred, neigh, weights)
            elif step == "passum":
                pred = project_pas_sumpol(pred, neigh, weights)
            else:
                raise ValueError(f"unknown step {step}")
    return pred.astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/derotated_projection.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--val-mode", choices=["matched-test", "random", "all"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--power", type=float, default=2.4)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--methods", type=parse_methods, default="base,base+pdp,base+passum,base+pdp+passum,base+passum+pdp")
    parser.add_argument("--pass-count", type=int, default=1)
    parser.add_argument("--pas-mode", choices=["sumpol", "fullpol"], default="sumpol")
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    tree = cKDTree(pos[:, :2])
    distances, local = tree.query(pos[val_idx, :2], k=args.k + 1, workers=-1)
    if args.k + 1 == 1:
        distances = distances[:, None]
        local = local[:, None]

    neighbor_idx = np.empty((len(val_idx), args.k), dtype=np.int64)
    neighbor_dist_xy = np.empty((len(val_idx), args.k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local[i] != vi
        neighbor_idx[i] = local[i, keep][:args.k]
        neighbor_dist_xy[i] = distances[i, keep][:args.k]

    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    accs = {method: MetricAccumulator(args.pas_mode) for method in args.methods}

    for i, vi in enumerate(val_idx):
        truth = np.asarray(h[vi])[None]
        neigh = np.asarray(h[neighbor_idx[i]])
        base, weights = derotated_base(
            neigh,
            neighbor_dist_xy[i],
            dist_all[neighbor_idx[i]],
            dist_all[vi],
            args.beta,
            args.power,
        )
        for method in args.methods:
            pred = apply_method(base, neigh, weights, method, args.pass_count)
            accs[method].add(pred[None], truth)
        if (i + 1) % 25 == 0 or i + 1 == len(val_idx):
            print(f"evaluated {i + 1}/{len(val_idx)}", flush=True)

    results = []
    for method in args.methods:
        results.append({
            "method": method,
            "k": int(args.k),
            "power": float(args.power),
            "beta": float(args.beta),
            "pass_count": int(args.pass_count),
            "pas_mode": args.pas_mode,
            **accs[method].result(),
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
    for row in results:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
