#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)
MH, MV, MP = 16, 8, 2


def idw_weights(distances, power):
    d = np.asarray(distances, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w.astype(np.float32)
    w = 1.0 / np.maximum(d, 1e-6) ** power
    w /= np.sum(w)
    return w.astype(np.float32)


def unit_vectors(pos):
    vec = pos - BS_POS[None, :]
    return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-30)


def bs_phv_indices():
    arr = np.empty((MP, MH, MV, 2), dtype=np.float64)
    for p in range(MP):
        for h in range(MH):
            for v in range(MV):
                arr[p, h, v] = (h - (MH - 1) / 2.0, v - (MV - 1) / 2.0)
    return arr.reshape(-1, 2)


def steering_phase(u_all, train_idx, query_idx, scale):
    ant = bs_phv_indices()
    h_coord = ant[:, 0]
    v_coord = ant[:, 1]
    h_val = u_all[:, 1]       # uy
    v_val = -u_all[:, 2]      # nuz
    delta_h = h_val[query_idx] - h_val[train_idx]
    delta_v = v_val[query_idx] - v_val[train_idx]
    phase_arg = scale * (
        delta_h[:, None] * h_coord[None, :] + delta_v[:, None] * v_coord[None, :]
    )
    return np.exp(1j * phase_arg).astype(np.complex64)


def predict_one(train_h, indices, distances, pos, dist_all, u_all, query_idx, k, power, beta, alpha, steering_scale):
    indices = indices[:k]
    distances = distances[:k]
    neigh = np.asarray(train_h[indices])
    weights = idw_weights(distances, power)
    radial = np.exp(1j * beta * (dist_all[indices] - dist_all[query_idx])).astype(np.complex64)
    coeff = weights * radial
    baseline = np.sum(coeff[:, None, None, None] * neigh, axis=0)
    phase = steering_phase(u_all, indices, query_idx, steering_scale)
    steered = np.sum(coeff[:, None, None, None] * phase[:, :, None, None] * neigh, axis=0)
    return ((1.0 - alpha) * baseline + alpha * steered).astype(np.complex64)


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    dist_train = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)
    pos_all = np.concatenate([pos, test_pos], axis=0)
    u_all = unit_vectors(pos_all)

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
    dist_all = np.concatenate([dist_train, dist_test])
    test_offset = len(pos)
    for start in range(0, len(test_pos), args.batch):
        end = min(start + args.batch, len(test_pos))
        for i in range(start, end):
            out[i] = predict_one(
                train_h,
                neighbor_idx[i],
                distances[i],
                pos,
                dist_all,
                u_all,
                test_offset + i,
                args.k,
                args.power,
                args.beta,
                args.alpha,
                args.steering_scale,
            )
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)
    return {
        "method": "geometric_steering_blend",
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "alpha": float(args.alpha),
        "steering_order": "phv",
        "h_axis": "uy",
        "v_axis": "nuz",
        "steering_scale": float(args.steering_scale),
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/geometric_steering_generate.json"))
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--alpha", type=float, default=0.5)
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
