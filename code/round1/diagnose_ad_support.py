#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_local_ad_paths import (
    BS_POS,
    ad_transform,
    flat_components,
    idw_weights,
    inverse_ad,
    parse_ints,
    select_components,
    top_components,
)


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def sparse_from_truth(truth_ad, component_ids):
    out = np.zeros_like(truth_ad)
    hh, vv, ss = flat_components(component_ids)
    for h_idx, v_idx, s_idx in zip(hh, vv, ss):
        out[:, :, h_idx, v_idx, :, s_idx] = truth_ad[:, :, h_idx, v_idx, :, s_idx]
    return out


def nmse(pred, truth):
    num = float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
    den = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
    return num / max(den, 1e-30)


def overlap(a, b):
    if len(a) == 0:
        return 0.0
    return len(set(a.tolist()) & set(b.tolist())) / len(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/ad_support_diagnostic.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--component-counts", type=parse_ints, default="128,512,1024,4096")
    parser.add_argument("--per-neighbor", type=int, default=256)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]

    accum = {
        c: {
            "true_oracle": [],
            "nearest_support_oracle": [],
            "weighted_support_oracle": [],
            "union_support_oracle": [],
            "nearest_overlap": [],
            "weighted_overlap": [],
            "union_overlap": [],
        }
        for c in args.component_counts
    }

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        truth_ad = ad_transform(truth)
        truth_power = np.sum(np.abs(truth_ad) ** 2, axis=(1, 4), dtype=np.float64)[0]
        idx = neighbor_idx[ii]
        neigh_ad = ad_transform(np.asarray(h[idx]))
        neigh_power = np.sum(np.abs(neigh_ad) ** 2, axis=(1, 4), dtype=np.float64)
        w = idw_weights(distances[ii], args.power)

        for c in args.component_counts:
            true_ids = top_components(truth_power, c)
            nearest_ids = top_components(neigh_power[0], c)
            weighted_ids = select_components(neigh_power, w, "weighted", args.per_neighbor, c)
            union_ids = select_components(neigh_power, w, "union", args.per_neighbor, c)

            true_pred = inverse_ad(sparse_from_truth(truth_ad, true_ids))
            nearest_pred = inverse_ad(sparse_from_truth(truth_ad, nearest_ids))
            weighted_pred = inverse_ad(sparse_from_truth(truth_ad, weighted_ids))
            union_pred = inverse_ad(sparse_from_truth(truth_ad, union_ids))

            accum[c]["true_oracle"].append(nmse(true_pred, truth))
            accum[c]["nearest_support_oracle"].append(nmse(nearest_pred, truth))
            accum[c]["weighted_support_oracle"].append(nmse(weighted_pred, truth))
            accum[c]["union_support_oracle"].append(nmse(union_pred, truth))
            accum[c]["nearest_overlap"].append(overlap(true_ids, nearest_ids))
            accum[c]["weighted_overlap"].append(overlap(true_ids, weighted_ids))
            accum[c]["union_overlap"].append(overlap(true_ids, union_ids))
        if (ii + 1) % 10 == 0 or ii + 1 == len(val_idx):
            print(f"diagnosed {ii + 1}/{len(val_idx)}", flush=True)

    results = []
    for c, vals in accum.items():
        row = {"component_count": int(c)}
        for key, arr in vals.items():
            arr = np.asarray(arr, dtype=np.float64)
            row[key + "_mean"] = float(np.mean(arr))
            row[key + "_median"] = float(np.median(arr))
        results.append(row)
    payload = {"seed": int(args.seed), "val_count": int(args.val_count), "k": int(args.k), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
