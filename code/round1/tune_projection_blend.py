#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from spectrum_projection import (
    MetricAccumulator,
    idw_weights,
    predict_projected,
)


def parse_float_list(text):
    return [float(x) for x in str(text).split(",") if x]


def parse_int_list(text):
    return [int(x) for x in str(text).split(",") if x]


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def weighted_sum(weights, values):
    return np.tensordot(weights.astype(np.float32), values, axes=(0, 0)).astype(np.complex64)


def power_scale(pred, neigh, distances, power, mode):
    if mode == "none":
        return pred
    pred_power = float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
    if pred_power <= 1e-30:
        return pred
    neigh_power = np.sum(np.abs(neigh) ** 2, axis=(1, 2, 3), dtype=np.float64)
    weights = idw_weights(distances, power)
    if mode == "linear":
        target_power = float(np.dot(weights, neigh_power))
    elif mode == "log":
        target_power = float(np.exp(np.dot(weights, np.log(np.maximum(neigh_power, 1e-30)))))
    else:
        raise ValueError(mode)
    if target_power <= 1e-30:
        return np.zeros_like(pred)
    return (pred * np.float32(np.sqrt(target_power / pred_power))).astype(np.complex64)


def evaluate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))[:, :2]
    channels = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    tree = cKDTree(pos[train_idx])
    results = []
    for k in args.k_values:
        distances, local_idx = tree.query(pos[val_idx], k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            local_idx = local_idx[:, None]
        neighbor_idx = train_idx[local_idx]
        for power in args.powers:
            accs = {}
            for alpha in args.alphas:
                for scale in args.power_scales:
                    name = f"blend_a{alpha:g}_{scale}"
                    accs[name] = MetricAccumulator(args.pas_mode)
            for start in range(0, len(val_idx), args.eval_batch):
                end = min(start + args.eval_batch, len(val_idx))
                truth = np.asarray(channels[val_idx[start:end]])
                preds = {name: np.empty_like(truth) for name in accs}
                for j in range(end - start):
                    neigh = np.asarray(channels[neighbor_idx[start + j]])
                    w = idw_weights(distances[start + j], power)
                    base = weighted_sum(w, neigh)
                    proj = predict_projected(neigh, distances[start + j], args.projection, power, args.pass_count)
                    for alpha in args.alphas:
                        blended = ((1.0 - alpha) * base + alpha * proj).astype(np.complex64)
                        for scale in args.power_scales:
                            name = f"blend_a{alpha:g}_{scale}"
                            preds[name][j] = power_scale(blended, neigh, distances[start + j], power, scale)
                for name, acc in accs.items():
                    acc.add(preds[name], truth)
            for name, acc in accs.items():
                alpha = float(name.split("_")[1][1:])
                scale = name.split("_", 2)[2]
                row = {
                    "k": int(k),
                    "power": float(power),
                    "projection": args.projection,
                    "pass_count": int(args.pass_count),
                    "alpha": alpha,
                    "power_scale": scale,
                    "pas_mode": args.pas_mode,
                    **acc.result(),
                }
                results.append(row)
                print(json.dumps(row, indent=2), flush=True)
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"seed": args.seed, "val_count": args.val_count, "results": results}


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))[:, :2]
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))[:, :2]
    channels = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    tree = cKDTree(pos)
    distances, neighbor_idx = tree.query(test_pos, k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]
    out = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), channels.shape[1], channels.shape[2], channels.shape[3]),
    )
    for i in range(len(test_pos)):
        neigh = np.asarray(channels[neighbor_idx[i]])
        w = idw_weights(distances[i], args.power)
        base = weighted_sum(w, neigh)
        proj = predict_projected(neigh, distances[i], args.projection, args.power, args.pass_count)
        pred = ((1.0 - args.alpha) * base + args.alpha * proj).astype(np.complex64)
        out[i] = power_scale(pred, neigh, distances[i], args.power, args.power_scale)
        if (i + 1) % 25 == 0 or i + 1 == len(test_pos):
            out.flush()
            print(f"generated {i + 1}/{len(test_pos)}", flush=True)
    return {
        "k": int(args.k),
        "power": float(args.power),
        "projection": args.projection,
        "pass_count": int(args.pass_count),
        "alpha": float(args.alpha),
        "power_scale": args.power_scale,
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/projection_blend_results.json"))
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_int_list, default="2,3")
    parser.add_argument("--powers", type=parse_float_list, default="2.0,3.0,4.0")
    parser.add_argument("--projection", default="idw+passum+pdp")
    parser.add_argument("--pass-count", type=int, default=1)
    parser.add_argument("--alphas", type=parse_float_list, default="0,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--power-scales", type=lambda s: [x for x in str(s).split(",") if x], default="none,linear,log")
    parser.add_argument("--pas-mode", choices=["sumpol", "fullpol"], default="sumpol")
    parser.add_argument("--eval-batch", type=int, default=10)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--power-scale", choices=["none", "linear", "log"], default="none")
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    args = parser.parse_args()

    if args.generate:
        result = generate(args)
    else:
        result = evaluate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
