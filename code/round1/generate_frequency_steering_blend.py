#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_frequency_dependent_phase import predict_freq
from evaluate_local_ad_paths import BS_POS, idw_weights
from generate_geometric_steering_blend import unit_vectors


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    dist_train = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)
    pos_all = np.concatenate([pos, test_pos], axis=0)
    u_all = unit_vectors(pos_all)
    dist_all = np.concatenate([dist_train, dist_test])
    test_offset = len(pos)

    tree = cKDTree(pos[:, :2])
    distances, neighbor_idx = tree.query(test_pos[:, :2], k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]

    out = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), train_h.shape[1], train_h.shape[2], train_h.shape[3]),
    )
    for start in range(0, len(test_pos), args.batch):
        end = min(start + args.batch, len(test_pos))
        for i in range(start, end):
            idx = neighbor_idx[i]
            neigh = np.asarray(train_h[idx])
            weights = idw_weights(distances[i], args.power)
            dist_delta = dist_all[idx] - dist_all[test_offset + i]
            out[i] = predict_freq(
                neigh,
                weights,
                dist_delta,
                u_all,
                idx,
                test_offset + i,
                args.beta,
                args.gamma,
                args.alpha,
                args.steering_scale,
            )
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)
    return {
        "method": "frequency_steering_blend",
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "steering_scale": float(args.steering_scale),
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/frequency_steering_generate.json"))
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--batch", type=int, default=5)
    args = parser.parse_args()
    result = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
