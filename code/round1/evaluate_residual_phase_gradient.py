#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x]


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def idw_weights(distances, power):
    d = np.asarray(distances, dtype=np.float64)
    if d[0] < 1e-12:
        w = np.zeros_like(d)
        w[0] = 1.0
        return w
    w = 1.0 / np.maximum(d, 1e-6) ** power
    return w / np.sum(w)


def candidate_gradients(magnitudes, angles):
    grads = [[0.0, 0.0]]
    labels = ["zero"]
    for mag in magnitudes:
        if mag == 0:
            continue
        for theta in np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False):
            grads.append([mag * np.cos(theta), mag * np.sin(theta)])
            labels.append(f"g{mag:g}_{theta:.4f}")
    return np.asarray(grads, dtype=np.float64), labels


def choose_gradient(z_flat, xy, grads, mode):
    k = z_flat.shape[0]
    pairs_i, pairs_j = np.triu_indices(k, 1)
    delta = xy[pairs_i] - xy[pairs_j]
    corr = np.einsum("ij,ij->i", z_flat[pairs_i], np.conj(z_flat[pairs_j]), optimize=True)
    norm = np.linalg.norm(z_flat[pairs_i], axis=1) * np.linalg.norm(z_flat[pairs_j], axis=1)
    good = norm > 1e-30
    if not np.any(good):
        return 0
    delta = delta[good]
    corr = corr[good] / norm[good]
    weight = np.abs(corr)
    if mode == "unweighted":
        weight = np.ones_like(weight)
    elif mode == "coherence2":
        weight = weight * weight
    phase = np.exp(1j * (delta @ grads.T))
    score = np.real((weight[:, None] * corr[:, None] * phase).sum(axis=0))
    return int(np.argmax(score))


def predict_one(train_h, indices, distances, pos, dist_all, query_idx, grads, grad_idx, beta, power):
    neigh = np.asarray(train_h[indices])
    weights = idw_weights(distances, power).astype(np.float32)
    radial = np.exp(1j * beta * (dist_all[indices] - dist_all[query_idx])).astype(np.complex64)
    g = grads[grad_idx]
    residual = np.exp(1j * ((pos[indices, :2] - pos[query_idx, :2]) @ g)).astype(np.complex64)
    coeff = weights * radial * residual
    return np.sum(coeff[:, None, None, None] * neigh, axis=0).astype(np.complex64)


def nmse_num_den(pred, truth):
    diff = pred - truth
    return (
        float(np.sum(np.abs(diff) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def evaluate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    max_k = max(args.k_values + [args.fit_k])
    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=max_k, workers=-1)
    if max_k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    grads, labels = candidate_gradients(args.magnitudes, args.angles)

    rows = []
    for k in args.k_values:
        for power in args.powers:
            for mode in args.modes:
                num = 0.0
                den = 0.0
                chosen = np.zeros(len(labels), dtype=np.int64)
                for c, vi in enumerate(val_idx):
                    fit_idx = neighbor_idx[c, :args.fit_k]
                    fit_h = np.asarray(h[fit_idx]).reshape(args.fit_k, -1)
                    z = fit_h * np.exp(1j * args.beta * dist_all[fit_idx])[:, None].astype(np.complex64)
                    if mode == "baseline":
                        grad_idx = 0
                    else:
                        grad_idx = choose_gradient(z, pos[fit_idx, :2], grads, mode)
                    chosen[grad_idx] += 1
                    pred = predict_one(
                        h,
                        neighbor_idx[c, :k],
                        distances[c, :k],
                        pos,
                        dist_all,
                        vi,
                        grads,
                        grad_idx,
                        args.beta,
                        power,
                    )
                    truth = np.asarray(h[vi])
                    a, b = nmse_num_den(pred, truth)
                    num += a
                    den += b
                    if (c + 1) % 25 == 0 or c + 1 == len(val_idx):
                        print(f"mode={mode} k={k} p={power:g} {c + 1}/{len(val_idx)}", flush=True)
                nmse = num / max(den, 1e-30)
                top = sorted([(labels[i], int(v)) for i, v in enumerate(chosen) if v], key=lambda x: -x[1])[:10]
                row = {
                    "mode": mode,
                    "k": int(k),
                    "fit_k": int(args.fit_k),
                    "power": float(power),
                    "beta": float(args.beta),
                    "global_nmse": float(nmse),
                    "inv_nmse": float(1.0 / (1.0 + nmse)),
                    "top_gradients": top,
                }
                rows.append(row)
                print(json.dumps(row, indent=2), flush=True)
    rows.sort(key=lambda r: r["global_nmse"])
    return {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "magnitudes": args.magnitudes,
        "angles": int(args.angles),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/residual_phase_gradient.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="16,20")
    parser.add_argument("--fit-k", type=int, default=24)
    parser.add_argument("--powers", type=parse_floats, default="2.0")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--magnitudes", type=parse_floats, default="10,20,40,60,80")
    parser.add_argument("--angles", type=int, default=48)
    parser.add_argument("--modes", choices=["baseline", "weighted", "unweighted", "coherence2"], nargs="+", default=["baseline", "weighted", "coherence2"])
    args = parser.parse_args()
    result = evaluate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("TOP")
    for row in result["results"][:20]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
