#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree


def idw_weights(distances, power=2.0):
    distances = np.asarray(distances, dtype=np.float64)
    if distances[0] < 1e-12:
        weights = np.zeros_like(distances)
        weights[0] = 1.0
        return weights
    weights = 1.0 / np.maximum(distances, 1e-6) ** power
    return weights / np.sum(weights)


def main():
    parser = argparse.ArgumentParser(description="Generate Round1_Test_Channel.npy from Round1 training data.")
    parser.add_argument("--data", default=str(PATHS.round1), help="Directory containing Round1_*.npy files")
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"), help="Output .npy path")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest training positions")
    parser.add_argument("--power", type=float, default=2.0, help="Inverse-distance weighting power")
    args = parser.parse_args()

    setup_path = os.path.join(args.data, "Round1_Setup.json")
    with open(setup_path) as f:
        setup = json.load(f)

    train_pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))[:, :2]
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))[:, :2]
    train_channel = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")

    expected_tail = (setup["M"], setup["N"], setup["S"])
    if train_channel.shape[1:] != expected_tail:
        raise ValueError(f"Unexpected train channel shape {train_channel.shape}; expected tail {expected_tail}")

    tree = cKDTree(train_pos)
    try:
        distances, indices = tree.query(test_pos, k=args.k, workers=-1)
    except TypeError:
        distances, indices = tree.query(test_pos, k=args.k)

    if args.k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    output_shape = (test_pos.shape[0],) + train_channel.shape[1:]
    pred = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=output_shape)

    for i in range(test_pos.shape[0]):
        neigh = np.asarray(train_channel[indices[i]])
        weights = idw_weights(distances[i], args.power).astype(np.float32)
        pred[i] = np.tensordot(weights, neigh, axes=(0, 0)).astype(np.complex64)
        if (i + 1) % 50 == 0 or i + 1 == test_pos.shape[0]:
            print(f"generated {i + 1}/{test_pos.shape[0]}", flush=True)

    pred.flush()
    print(f"saved {args.output} shape={output_shape} dtype=complex64")


if __name__ == "__main__":
    main()
