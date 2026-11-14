#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from generate_geometric_steering_blend import (
    BS_POS,
    idw_weights,
    steering_phase,
    unit_vectors,
)


def load_test_alpha(features_path, rule, threshold, clear_alpha, blocked_alpha):
    data = np.load(features_path, allow_pickle=True)
    features = data["test_features"]
    names = [str(x) for x in data["feature_names"]]
    if rule == "los_min_ge":
        clear = features[:, names.index("los_min_dist")] >= threshold
    elif rule == "los_frac1_le":
        clear = features[:, names.index("los_frac_r1p0")] <= threshold
    elif rule == "los_frac15_le":
        clear = features[:, names.index("los_frac_r1p5")] <= threshold
    else:
        raise ValueError(rule)
    alpha = np.where(clear, clear_alpha, blocked_alpha).astype(np.float32)
    return alpha, clear


def predict_one(train_h, indices, distances, dist_all, u_all, query_idx, k, power, beta, alpha, steering_scale):
    indices = indices[:k]
    distances = distances[:k]
    neigh = np.asarray(train_h[indices])
    weights = idw_weights(distances, power)
    radial = np.exp(1j * beta * (dist_all[indices] - dist_all[query_idx])).astype(np.complex64)
    coeff = weights.astype(np.float32) * radial
    baseline = np.sum(coeff[:, None, None, None] * neigh, axis=0)
    phase = steering_phase(u_all, indices, query_idx, steering_scale)
    steered = np.sum(coeff[:, None, None, None] * phase[:, :, None, None] * neigh, axis=0)
    return ((1.0 - alpha) * baseline + alpha * steered).astype(np.complex64)


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    alpha, clear = load_test_alpha(args.features, args.rule, args.threshold, args.clear_alpha, args.blocked_alpha)

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
            out[i] = predict_one(
                train_h,
                neighbor_idx[i],
                distances[i],
                dist_all,
                u_all,
                test_offset + i,
                args.k,
                args.power,
                args.beta,
                float(alpha[i]),
                args.steering_scale,
            )
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)

    return {
        "method": "map_los_adaptive_alpha",
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "rule": args.rule,
        "threshold": float(args.threshold),
        "clear_alpha": float(args.clear_alpha),
        "blocked_alpha": float(args.blocked_alpha),
        "clear_count": int(np.sum(clear)),
        "blocked_count": int(len(clear) - np.sum(clear)),
        "features": args.features,
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/map_adaptive_alpha_generate.json"))
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--rule", default="los_min_ge")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--clear-alpha", type=float, default=1.0)
    parser.add_argument("--blocked-alpha", type=float, default=0.25)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()
    result = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
