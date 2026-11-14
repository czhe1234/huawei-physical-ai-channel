#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def idw_weights(distances, power):
    d = np.asarray(distances, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w
    w = 1.0 / np.maximum(d, 1e-6) ** power
    return w / np.sum(w)


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def predict_one(train_h, indices, distances, train_dist, query_dist, beta, power):
    neigh = np.asarray(train_h[indices])
    weights = idw_weights(distances, power).astype(np.float32)
    phase = np.exp(1j * beta * (train_dist[indices] - query_dist)).astype(np.complex64)
    return np.sum(weights[:, None, None, None] * phase[:, None, None, None] * neigh, axis=0).astype(np.complex64)


def global_nmse(pred, truth):
    return (
        float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def evaluate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    xy_train = pos[train_idx, :2]
    xy_val = pos[val_idx, :2]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)

    tree = cKDTree(xy_train)
    distances, local_idx = tree.query(xy_val, k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        local_idx = local_idx[:, None]
    neighbor_idx = train_idx[local_idx]

    num = 0.0
    den = 0.0
    for start in range(0, len(val_idx), args.batch):
        end = min(start + args.batch, len(val_idx))
        pred = np.empty((end - start, 256, 4, 192), dtype=np.complex64)
        for j, i in enumerate(range(start, end)):
            pred[j] = predict_one(
                train_h,
                neighbor_idx[i],
                distances[i],
                dist_all,
                dist_all[val_idx[i]],
                args.beta,
                args.power,
            )
        truth = np.asarray(train_h[val_idx[start:end]])
        a, b = global_nmse(pred, truth)
        num += a
        den += b
        print(f"validated {end}/{len(val_idx)}", flush=True)
    result = {
        "method": "derotated_full_idw",
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "global_nmse": float(num / max(den, 1e-30)),
        "inv_nmse": float(1.0 / (1.0 + num / max(den, 1e-30))),
    }
    return result


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    dist_train = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)

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
                dist_train,
                dist_test[i],
                args.beta,
                args.power,
            )
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)
    return {
        "method": "derotated_full_idw",
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/derotated_idw_result.json"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--power", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    result = generate(args) if args.generate else evaluate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
