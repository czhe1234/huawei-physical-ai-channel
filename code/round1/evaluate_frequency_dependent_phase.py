#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_local_ad_paths import BS_POS, S_SC, idw_weights, parse_floats, parse_ints
from generate_geometric_steering_blend import bs_phv_indices, unit_vectors
from spectrum_projection import MetricAccumulator


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def eval_one(truth, pred):
    return (
        float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def steering_phase_freq(u_all, train_idx, query_idx, steering_scale, beta_ratio):
    ant = bs_phv_indices()
    h_coord = ant[:, 0]
    v_coord = ant[:, 1]
    h_val = u_all[:, 1]
    v_val = -u_all[:, 2]
    delta_h = h_val[query_idx] - h_val[train_idx]
    delta_v = v_val[query_idx] - v_val[train_idx]
    spatial = delta_h[:, None] * h_coord[None, :] + delta_v[:, None] * v_coord[None, :]
    phase_arg = steering_scale * spatial[:, :, None] * beta_ratio[None, None, :]
    return np.exp(1j * phase_arg).astype(np.complex64)


def predict_freq(neigh_h, weights, dist_delta, u_all, idx, vi, beta0, gamma, alpha, steering_scale):
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    beta_s = (beta0 + gamma * sub).astype(np.float32)
    beta_ratio = (beta_s / beta0).astype(np.float32)
    radial = np.exp(1j * dist_delta[:, None] * beta_s[None, :]).astype(np.complex64)
    coeff = weights.astype(np.float32)[:, None] * radial
    baseline = np.sum(coeff[:, None, None, :] * neigh_h, axis=0)
    steer = steering_phase_freq(u_all, idx, vi, steering_scale, beta_ratio)
    steered = np.sum(coeff[:, None, None, :] * steer[:, :, None, :] * neigh_h, axis=0)
    return ((1.0 - alpha) * baseline + alpha * steered).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/frequency_dependent_phase.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="8,16,24")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gammas", type=parse_floats, default="-80,-40,-20,-10,0,10,20,40,80")
    parser.add_argument("--alphas", type=parse_floats, default="0,0.5,1.0")
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    max_k = max(args.k_values)
    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=max_k, workers=-1)
    if max_k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    candidates = []
    for k in args.k_values:
        for gamma in args.gammas:
            for alpha in args.alphas:
                candidates.append((k, gamma, alpha))
    stats = {str(c): {"num": 0.0, "den": 0.0} for c in candidates}
    accs = {str(c): MetricAccumulator("sumpol") for c in candidates}

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        for k in args.k_values:
            idx = neighbor_idx[ii, :k]
            neigh = np.asarray(h[idx])
            weights = idw_weights(distances[ii, :k], args.power)
            dist_delta = dist_all[idx] - dist_all[vi]
            for gamma in args.gammas:
                for alpha in args.alphas:
                    pred = predict_freq(
                        neigh,
                        weights,
                        dist_delta,
                        u_all,
                        idx,
                        vi,
                        args.beta,
                        gamma,
                        alpha,
                        args.steering_scale,
                    )[None]
                    key = str((k, gamma, alpha))
                    num, den = eval_one(truth, pred)
                    stats[key]["num"] += num
                    stats[key]["den"] += den
                    accs[key].add(pred, truth)
        if (ii + 1) % 10 == 0 or ii + 1 == len(val_idx):
            print(f"evaluated {ii + 1}/{len(val_idx)}", flush=True)

    results = []
    for c in candidates:
        key = str(c)
        results.append({
            "k": int(c[0]),
            "gamma": float(c[1]),
            "alpha": float(c[2]),
            "global_nmse_direct": float(stats[key]["num"] / max(stats[key]["den"], 1e-30)),
            **accs[key].result(),
        })
    results.sort(key=lambda r: r["global_nmse"])
    payload = {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "beta": float(args.beta),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
