#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS, idw_weights, parse_floats, parse_ints
from evaluate_frequency_dependent_phase import S_SC
from generate_geometric_steering_blend import bs_phv_indices, unit_vectors
from spectrum_projection import MetricAccumulator


def affine_coefficients(rel_xy, distances, power, ridge):
    w = idw_weights(distances, power).astype(np.float64)
    x = np.column_stack([np.ones(len(rel_xy)), rel_xy.astype(np.float64)])
    xw = x * w[:, None]
    xtwx = x.T @ xw
    scale = max(float(np.trace(xtwx)) / xtwx.shape[0], 1e-12)
    reg = np.eye(3, dtype=np.float64) * (ridge * scale)
    reg[0, 0] = 0.0
    try:
        inv = np.linalg.inv(xtwx + reg)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(xtwx + reg)
    # Intercept at query position: e0^T (X^T W X)^-1 X^T W.
    return (inv[0] @ xw.T).astype(np.float32)


def steering_phase_query(u_all, train_idx, query_idx, steering_scale):
    ant = bs_phv_indices()
    h_coord = ant[:, 0]
    v_coord = ant[:, 1]
    h_val = u_all[:, 1]
    v_val = -u_all[:, 2]
    delta_h = h_val[query_idx] - h_val[train_idx]
    delta_v = v_val[query_idx] - v_val[train_idx]
    phase_arg = steering_scale * (
        delta_h[:, None] * h_coord[None, :] + delta_v[:, None] * v_coord[None, :]
    )
    return np.exp(1j * phase_arg).astype(np.complex64)


def predict_affine(neigh, coeff, dist_delta, u_all, idx, qi, beta, gamma, alpha, steering_scale):
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    beta_s = (beta + gamma * sub).astype(np.float32)
    radial = np.exp(1j * dist_delta[:, None] * beta_s[None, :]).astype(np.complex64)
    base_terms = radial[:, None, None, :] * neigh
    base = np.sum(coeff[:, None, None, None] * base_terms, axis=0)
    steer = steering_phase_query(u_all, idx, qi, steering_scale)
    geo = np.sum(coeff[:, None, None, None] * steer[:, :, None, None] * base_terms, axis=0)
    return ((1.0 - alpha) * base + alpha * geo).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/physical_local_linear.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k-values", type=parse_ints, default="8,12,16,24,32")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gammas", type=parse_floats, default="0,0.15")
    parser.add_argument("--alphas", type=parse_floats, default="0.5,0.75,1.0")
    parser.add_argument("--ridges", type=parse_floats, default="0,0.001,0.01,0.1")
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--batch", type=int, default=5)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    max_k = max(args.k_values)
    tree = cKDTree(pos[:, :2])
    d_all, local_all = tree.query(pos[val_idx, :2], k=max_k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), max_k), dtype=np.int64)
    distances = np.empty((len(val_idx), max_k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:max_k]
        distances[i] = d_all[i, keep][:max_k]

    pos_all = pos
    dist_all = np.linalg.norm(pos_all - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos_all)
    candidates = [(k, gamma, alpha, ridge) for k in args.k_values for gamma in args.gammas for alpha in args.alphas for ridge in args.ridges]
    accs = {str(c): MetricAccumulator("sumpol") for c in candidates}
    nums = {str(c): 0.0 for c in candidates}
    dens = {str(c): 0.0 for c in candidates}

    for row, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        den = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        idx_full = neighbor_idx[row]
        neigh_full = np.asarray(h[idx_full])
        for k in args.k_values:
            idx = idx_full[:k]
            neigh = neigh_full[:k]
            rel_xy = pos[idx, :2] - pos[vi, :2]
            dist_delta = dist_all[idx] - dist_all[vi]
            for ridge in args.ridges:
                coeff = affine_coefficients(rel_xy, distances[row, :k], args.power, ridge)
                for gamma in args.gammas:
                    for alpha in args.alphas:
                        pred = predict_affine(
                            neigh,
                            coeff,
                            dist_delta,
                            u_all,
                            idx,
                            vi,
                            args.beta,
                            gamma,
                            alpha,
                            args.steering_scale,
                        )[None]
                        key = str((k, gamma, alpha, ridge))
                        nums[key] += float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
                        dens[key] += den
                        accs[key].add(pred, truth)
        if (row + 1) % 10 == 0 or row + 1 == len(val_idx):
            print(f"evaluated {row + 1}/{len(val_idx)}", flush=True)

    results = []
    for c in candidates:
        key = str(c)
        nmse = nums[key] / max(dens[key], 1e-30)
        results.append({
            "k": int(c[0]),
            "gamma": float(c[1]),
            "alpha": float(c[2]),
            "ridge": float(c[3]),
            "global_nmse_direct": float(nmse),
            **accs[key].result(),
        })
    results.sort(key=lambda r: r["global_nmse_direct"])
    payload = {
        "seed": int(args.seed),
        "val_count": int(len(val_idx)),
        "val_mode": args.val_mode,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:40]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
