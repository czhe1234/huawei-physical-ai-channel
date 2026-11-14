#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_ad_coeff_methods import (
    aligned_iter_idw,
    aligned_nearest_idw,
    build_split,
    complex_idw,
    complex_scalar_oracle,
    eval_one,
    full_predict,
    idw_weights,
    magnitude_idw_phase_idw,
    magnitude_idw_phase_nearest,
    magnitude_oracle,
    phase_oracle,
)
from evaluate_local_ad_paths import BS_POS, MP, MH, MV, parse_ints, parse_words
from generate_geometric_steering_blend import steering_phase, unit_vectors
from spectrum_projection import MetricAccumulator


NV, NP, S_SC = 2, 2, 192
TOTAL_COMPONENTS = MH * MV * NV * S_SC


def ad3_transform(h, n_order):
    cir = np.fft.ifft(h, axis=-1)
    if n_order == "vp":
        arr = cir.reshape(h.shape[0], MP, MH, MV, NV, NP, S_SC)
    elif n_order == "pv":
        arr = cir.reshape(h.shape[0], MP, MH, MV, NP, NV, S_SC).transpose(0, 1, 2, 3, 5, 4, 6)
    else:
        raise ValueError(n_order)
    return np.fft.fftn(arr, axes=(2, 3, 4)).astype(np.complex64)


def inverse_ad3(ad, n_order):
    arr = np.fft.ifftn(ad, axes=(2, 3, 4))
    if n_order == "vp":
        cir = arr.reshape(ad.shape[0], MP * MH * MV, NV * NP, S_SC)
    elif n_order == "pv":
        cir = arr.transpose(0, 1, 2, 3, 5, 4, 6).reshape(ad.shape[0], MP * MH * MV, NP * NV, S_SC)
    else:
        raise ValueError(n_order)
    return np.fft.fft(cir, axis=-1).astype(np.complex64)


def flat_components3(component_ids):
    h = component_ids // (MV * NV * S_SC)
    rem = component_ids % (MV * NV * S_SC)
    v = rem // (NV * S_SC)
    rem = rem % (NV * S_SC)
    uv = rem // S_SC
    s = rem % S_SC
    return h.astype(np.int64), v.astype(np.int64), uv.astype(np.int64), s.astype(np.int64)


def top_components_flat(power, count):
    flat = power.reshape(-1)
    count = min(count, flat.size)
    ids = np.argpartition(flat, -count)[-count:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    return ids.astype(np.int64)


def select_components3(neigh_power, weights, mode, per_neighbor, total_count):
    if mode == "weighted":
        score = np.tensordot(weights.astype(np.float32), neigh_power, axes=(0, 0))
        return top_components_flat(score, total_count)
    if mode == "nearest":
        return top_components_flat(neigh_power[0], total_count)
    if mode == "union":
        ids = []
        for j in range(neigh_power.shape[0]):
            ids.extend(top_components_flat(neigh_power[j], per_neighbor).tolist())
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if ids.size > total_count:
            score = np.tensordot(weights.astype(np.float32), neigh_power, axes=(0, 0)).reshape(-1)
            ids = ids[np.argsort(score[ids])[::-1][:total_count]]
        return ids.astype(np.int64)
    raise ValueError(mode)


def ad3_to_flat(ad):
    return ad.transpose(0, 2, 3, 4, 6, 1, 5).reshape(ad.shape[0], TOTAL_COMPONENTS, MP, NP)


def flat_to_ad3(flat):
    return flat.reshape(flat.shape[0], MH, MV, NV, S_SC, MP, NP).transpose(0, 5, 1, 2, 3, 6, 4)


def gather_coeff3(ad, component_ids):
    return ad3_to_flat(ad)[:, component_ids]


def scatter_coeff3(component_ids, coeff):
    flat = np.zeros((1, TOTAL_COMPONENTS, MP, NP), dtype=np.complex64)
    flat[0, component_ids] = coeff.astype(np.complex64)
    return flat_to_ad3(flat)


def make_sparse_pred(component_ids, coeff_pred, n_order):
    return inverse_ad3(scatter_coeff3(component_ids, coeff_pred), n_order)


def add_result(stats, accs, key, pred, truth):
    num, den = eval_one(truth, pred)
    stats[key]["num"] += num
    stats[key]["den"] += den
    accs[key].add(pred, truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/ue_angle_delay.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="8,24")
    parser.add_argument("--component-counts", type=parse_ints, default="512,1024,4096")
    parser.add_argument("--n-orders", type=parse_words, default="vp,pv")
    parser.add_argument("--modes", type=parse_words, default="weighted")
    parser.add_argument("--methods", type=parse_words, default="complex_idw,mag_idw_phase_idw,aligned_iter")
    parser.add_argument("--per-neighbor", type=int, default=256)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
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

    method_fns = {
        "complex_idw": complex_idw,
        "nearest": lambda coeff, weights: coeff[0].astype(np.complex64),
        "mag_idw_phase_nearest": magnitude_idw_phase_nearest,
        "mag_idw_phase_idw": magnitude_idw_phase_idw,
        "aligned_nearest": aligned_nearest_idw,
        "aligned_iter": aligned_iter_idw,
    }

    candidates = []
    for k in args.k_values:
        candidates.extend([
            ("full_derot", k, 0, "full", "none", False),
            ("full_steer", k, 0, "full", "none", False),
            ("full_blend05", k, 0, "full", "none", False),
        ])
        for n_order in args.n_orders:
            for comp_count in args.component_counts:
                for mode in args.modes:
                    for method in args.methods:
                        candidates.append((method, k, comp_count, mode, n_order, False))
                    for oracle_name in [
                        "support_truth_oracle",
                        "complex_idw_phase_oracle",
                        "complex_idw_magnitude_oracle",
                        "complex_idw_scalar_oracle",
                    ]:
                        candidates.append((oracle_name, k, comp_count, mode, n_order, True))

    stats = {str(c): {"num": 0.0, "den": 0.0} for c in candidates}
    accs = {str(c): MetricAccumulator("sumpol") for c in candidates}

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        truth_ad_by_order = {n_order: ad3_transform(truth, n_order) for n_order in args.n_orders}
        truth_flat_by_order = {n_order: ad3_to_flat(truth_ad_by_order[n_order]) for n_order in args.n_orders}

        for k in args.k_values:
            idx = neighbor_idx[ii, :k]
            neigh_h = np.asarray(h[idx])
            w = idw_weights(distances[ii, :k], args.power)
            radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
            steer = steering_phase(u_all, idx, vi, args.steering_scale)
            full_derot = full_predict(neigh_h, w, radial)
            full_steer = full_predict(neigh_h, w, radial, steer)
            full_blend = (0.5 * full_derot + 0.5 * full_steer).astype(np.complex64)
            add_result(stats, accs, str(("full_derot", k, 0, "full", "none", False)), full_derot, truth)
            add_result(stats, accs, str(("full_steer", k, 0, "full", "none", False)), full_steer, truth)
            add_result(stats, accs, str(("full_blend05", k, 0, "full", "none", False)), full_blend, truth)

            corrected = neigh_h * radial[:, None, None, None] * steer[:, :, None, None]
            for n_order in args.n_orders:
                corrected_ad = ad3_transform(corrected, n_order)
                corrected_power = np.sum(np.abs(corrected_ad) ** 2, axis=(1, 5), dtype=np.float64)
                truth_flat = truth_flat_by_order[n_order]
                for comp_count in args.component_counts:
                    for mode in args.modes:
                        component_ids = select_components3(corrected_power, w, mode, args.per_neighbor, comp_count)
                        coeff = gather_coeff3(corrected_ad, component_ids)
                        truth_coeff = truth_flat[0, component_ids]

                        complex_pred = None
                        for method in args.methods:
                            pred_coeff = method_fns[method](coeff, w)
                            if method == "complex_idw":
                                complex_pred = pred_coeff
                            pred = make_sparse_pred(component_ids, pred_coeff, n_order)
                            key = str((method, k, comp_count, mode, n_order, False))
                            add_result(stats, accs, key, pred, truth)

                        if complex_pred is None:
                            complex_pred = complex_idw(coeff, w)
                        oracle_map = {
                            "support_truth_oracle": truth_coeff,
                            "complex_idw_phase_oracle": phase_oracle(complex_pred, truth_coeff),
                            "complex_idw_magnitude_oracle": magnitude_oracle(complex_pred, truth_coeff),
                            "complex_idw_scalar_oracle": complex_scalar_oracle(complex_pred, truth_coeff),
                        }
                        for oracle_name, oracle_coeff in oracle_map.items():
                            pred = make_sparse_pred(component_ids, oracle_coeff, n_order)
                            key = str((oracle_name, k, comp_count, mode, n_order, True))
                            add_result(stats, accs, key, pred, truth)

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
            "n_order": c[4],
            "oracle": bool(c[5]),
            "global_nmse_direct": float(nmse),
            **accs[key].result(),
        })
    results.sort(key=lambda r: r["global_nmse"])
    payload = {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "power": float(args.power),
        "beta": float(args.beta),
        "steering_scale": float(args.steering_scale),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:50]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
