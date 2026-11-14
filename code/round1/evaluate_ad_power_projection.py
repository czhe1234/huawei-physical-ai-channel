#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_local_ad_paths import BS_POS, ad_transform, idw_weights, inverse_ad, parse_floats, parse_ints
from generate_geometric_steering_blend import steering_phase, unit_vectors
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


def full_predict(neigh_h, weights, radial, steer_phase=None):
    coeff = weights.astype(np.float32) * radial.astype(np.complex64)
    if steer_phase is None:
        return np.sum(coeff[:, None, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)
    return np.sum(coeff[:, None, None, None] * steer_phase[:, :, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)


def project_ad_power(base, target_power, strength):
    base_ad = ad_transform(base)
    base_power = np.sum(np.abs(base_ad) ** 2, axis=(1, 4), keepdims=True, dtype=np.float64)
    scale = np.sqrt(np.maximum(target_power[:, None, :, :, None, :], 0.0) / np.maximum(base_power, 1e-30))
    projected = inverse_ad((base_ad * scale).astype(np.complex64))
    return ((1.0 - strength) * base + strength * projected).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/ad_power_projection.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="8,16,24")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--strengths", type=parse_floats, default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--base-alphas", type=parse_floats, default="0,0.5,1.0")
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
        for alpha in args.base_alphas:
            for strength in args.strengths:
                candidates.append((k, alpha, strength))
    stats = {str(c): {"num": 0.0, "den": 0.0} for c in candidates}
    accs = {str(c): MetricAccumulator("sumpol") for c in candidates}

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        for k in args.k_values:
            idx = neighbor_idx[ii, :k]
            neigh = np.asarray(h[idx])
            weights = idw_weights(distances[ii, :k], args.power)
            radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
            steer = steering_phase(u_all, idx, vi, args.steering_scale)
            derot = full_predict(neigh, weights, radial)
            steered = full_predict(neigh, weights, radial, steer)

            corrected = neigh * radial[:, None, None, None] * steer[:, :, None, None]
            corrected_ad = ad_transform(corrected)
            corrected_power = np.sum(np.abs(corrected_ad) ** 2, axis=(1, 4), dtype=np.float64)
            target_power = np.tensordot(weights.astype(np.float32), corrected_power, axes=(0, 0))[None]

            for alpha in args.base_alphas:
                base = ((1.0 - alpha) * derot + alpha * steered).astype(np.complex64)
                for strength in args.strengths:
                    pred = project_ad_power(base, target_power, strength)
                    key = str((k, alpha, strength))
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
            "base_alpha": float(c[1]),
            "projection_strength": float(c[2]),
            "global_nmse_direct": float(stats[key]["num"] / max(stats[key]["den"], 1e-30)),
            **accs[key].result(),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    payload = {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
