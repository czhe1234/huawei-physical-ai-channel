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
    parse_words,
    predict_sparse_ad,
    select_components,
)
from generate_geometric_steering_blend import steering_phase, unit_vectors
from spectrum_projection import MetricAccumulator


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def full_predict(neigh_h, weights, radial, steer_phase=None):
    x = coeff = weights.astype(np.float32) * radial.astype(np.complex64)
    if steer_phase is None:
        return np.sum(x[:, None, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)
    return np.sum(x[:, None, None, None] * steer_phase[:, :, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)


def eval_one(truth, pred):
    diff = pred - truth
    return (
        float(np.sum(np.abs(diff) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/steered_ad_paths.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="8,16,24")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--component-counts", type=parse_ints, default="128,512,1024")
    parser.add_argument("--per-neighbor", type=int, default=128)
    parser.add_argument("--modes", type=parse_words, default="weighted,union")
    parser.add_argument("--align-options", type=parse_words, default="no")
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
        candidates.extend([
            ("full_derot", k, 0, "all", False),
            ("full_steer", k, 0, "all", False),
            ("full_blend05", k, 0, "all", False),
        ])
        for comp_count in args.component_counts:
            for mode in args.modes:
                for align_text in args.align_options:
                    candidates.append(("steered_ad", k, comp_count, mode, align_text == "yes"))
                    candidates.append(("blend05_ad", k, comp_count, mode, align_text == "yes"))

    stats = {str(c): {"num": 0.0, "den": 0.0} for c in candidates}
    accs = {str(c): MetricAccumulator("sumpol") for c in candidates}

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        for k in args.k_values:
            idx = neighbor_idx[ii, :k]
            neigh_h = np.asarray(h[idx])
            w = idw_weights(distances[ii, :k], args.power)
            radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
            steer = steering_phase(u_all, idx, vi, args.steering_scale)
            full_derot = full_predict(neigh_h, w, radial)
            full_steer = full_predict(neigh_h, w, radial, steer)
            full_blend = (0.5 * full_derot + 0.5 * full_steer).astype(np.complex64)
            full_map = {
                "full_derot": full_derot,
                "full_steer": full_steer,
                "full_blend05": full_blend,
            }
            for name, pred in full_map.items():
                key = str((name, k, 0, "all", False))
                num, den = eval_one(truth, pred)
                stats[key]["num"] += num
                stats[key]["den"] += den
                accs[key].add(pred, truth)

            corrected = neigh_h * radial[:, None, None, None] * steer[:, :, None, None]
            corrected_ad = ad_transform(corrected)
            corrected_power = np.sum(np.abs(corrected_ad) ** 2, axis=(1, 4), dtype=np.float64)
            derot_ad = ad_transform(neigh_h * radial[:, None, None, None])
            derot_power = np.sum(np.abs(derot_ad) ** 2, axis=(1, 4), dtype=np.float64)

            for comp_count in args.component_counts:
                for mode in args.modes:
                    component_ids = select_components(corrected_power, w, mode, args.per_neighbor, comp_count)
                    blend_component_ids = select_components(0.5 * corrected_power + 0.5 * derot_power, w, mode, args.per_neighbor, comp_count)
                    for align_text in args.align_options:
                        align = align_text == "yes"
                        pred_ad = predict_sparse_ad(corrected_ad, w, component_ids, align)
                        pred = inverse_ad(pred_ad)
                        key = str(("steered_ad", k, comp_count, mode, align))
                        num, den = eval_one(truth, pred)
                        stats[key]["num"] += num
                        stats[key]["den"] += den
                        accs[key].add(pred, truth)

                        pred_ad_s = predict_sparse_ad(corrected_ad, w, blend_component_ids, align)
                        pred_s = inverse_ad(pred_ad_s)
                        pred_ad_d = predict_sparse_ad(derot_ad, w, blend_component_ids, align)
                        pred_d = inverse_ad(pred_ad_d)
                        pred_b = (0.5 * pred_s + 0.5 * pred_d).astype(np.complex64)
                        key = str(("blend05_ad", k, comp_count, mode, align))
                        num, den = eval_one(truth, pred_b)
                        stats[key]["num"] += num
                        stats[key]["den"] += den
                        accs[key].add(pred_b, truth)
        if (ii + 1) % 10 == 0 or ii + 1 == len(val_idx):
            print(f"evaluated {ii + 1}/{len(val_idx)}", flush=True)

    results = []
    for c in candidates:
        key = str(c)
        nmse = stats[key]["num"] / max(stats[key]["den"], 1e-30)
        results.append({
            "candidate": c[0],
            "k": int(c[1]),
            "component_count": int(c[2]),
            "mode": c[3],
            "align": bool(c[4]),
            "global_nmse_direct": float(nmse),
            **accs[key].result(),
        })
    results.sort(key=lambda r: r["global_nmse"])
    payload = {"seed": int(args.seed), "val_count": int(args.val_count), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:30]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
