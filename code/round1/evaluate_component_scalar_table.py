#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_ad_coeff_methods import complex_idw, component_scalar, eval_one, full_predict
from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS, idw_weights, parse_ints
from evaluate_ue_angle_delay import (
    TOTAL_COMPONENTS,
    ad3_to_flat,
    ad3_transform,
    flat_to_ad3,
    gather_coeff3,
    inverse_ad3,
    scatter_coeff3,
    select_components3,
)
from generate_geometric_steering_blend import steering_phase, unit_vectors
from spectrum_projection import MetricAccumulator


def make_neighbors(pos, query_idx, train_pool, k):
    tree = cKDTree(pos[train_pool, :2])
    d, local = tree.query(pos[query_idx, :2], k=k + 1, workers=-1)
    if query_idx.ndim == 0:
        d = d[None, :]
        local = local[None, :]
    out_idx = np.empty((len(query_idx), k), dtype=np.int64)
    out_d = np.empty((len(query_idx), k), dtype=np.float64)
    for i, qi in enumerate(query_idx):
        cand = train_pool[local[i]]
        keep = cand != qi
        out_idx[i] = cand[keep][:k]
        out_d[i] = d[i, keep][:k]
    return out_d, out_idx


def scalar_for_query(h, pos, u_all, dist_all, qi, neigh_idx, distances, component_count, n_order, beta, steering_scale, power):
    neigh_h = np.asarray(h[neigh_idx])
    weights = idw_weights(distances, power)
    radial = np.exp(1j * beta * (dist_all[neigh_idx] - dist_all[qi])).astype(np.complex64)
    steer = steering_phase(u_all, neigh_idx, qi, steering_scale)
    corrected = neigh_h * radial[:, None, None, None] * steer[:, :, None, None]
    corrected_ad = ad3_transform(corrected, n_order)
    corrected_power = np.sum(np.abs(corrected_ad) ** 2, axis=(1, 5), dtype=np.float64)
    component_ids = select_components3(corrected_power, weights, "weighted", 256, component_count)
    coeff = gather_coeff3(corrected_ad, component_ids)
    pred_coeff = complex_idw(coeff, weights)
    truth = np.asarray(h[qi:qi + 1])
    truth_flat = ad3_to_flat(ad3_transform(truth, n_order))
    truth_coeff = truth_flat[0, component_ids]
    scalar = component_scalar(pred_coeff, truth_coeff)
    strength = np.sum(np.abs(pred_coeff.reshape(pred_coeff.shape[0], -1)) ** 2, axis=1).astype(np.float64)
    return component_ids, scalar, strength


def build_scalar_table(h, pos, train_idx, sample_count, seed, k, component_count, n_order, beta, steering_scale, power):
    rng = np.random.default_rng(seed)
    sample = np.array(train_idx, copy=True)
    if sample_count > 0 and sample_count < len(sample):
        sample = rng.choice(sample, size=sample_count, replace=False)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    d, neigh = make_neighbors(pos, sample, np.asarray(train_idx, dtype=np.int64), k)
    num = np.zeros(TOTAL_COMPONENTS, dtype=np.complex128)
    den = np.zeros(TOTAL_COMPONENTS, dtype=np.float64)
    count = np.zeros(TOTAL_COMPONENTS, dtype=np.int32)
    for i, qi in enumerate(sample):
        ids, scalar, strength = scalar_for_query(
            h, pos, u_all, dist_all, int(qi), neigh[i], d[i], component_count, n_order, beta, steering_scale, power
        )
        w = np.maximum(strength, 1e-12)
        num[ids] += scalar.astype(np.complex128) * w
        den[ids] += w
        count[ids] += 1
        if (i + 1) % 50 == 0 or i + 1 == len(sample):
            print(f"scalar table {i + 1}/{len(sample)}", flush=True)
    table = np.ones(TOTAL_COMPONENTS, dtype=np.complex64)
    good = den > 0
    table[good] = (num[good] / den[good]).astype(np.complex64)
    mag = np.abs(table)
    too_big = mag > 2.0
    table[too_big] *= (2.0 / mag[too_big]).astype(np.float32)
    return table, count


def predict_query(h, pos, u_all, dist_all, qi, neigh_idx, distances, table, component_count, n_order, beta, steering_scale, power):
    neigh_h = np.asarray(h[neigh_idx])
    weights = idw_weights(distances, power)
    radial = np.exp(1j * beta * (dist_all[neigh_idx] - dist_all[qi])).astype(np.complex64)
    steer = steering_phase(u_all, neigh_idx, qi, steering_scale)
    corrected = neigh_h * radial[:, None, None, None] * steer[:, :, None, None]
    corrected_ad = ad3_transform(corrected, n_order)
    corrected_power = np.sum(np.abs(corrected_ad) ** 2, axis=(1, 5), dtype=np.float64)
    component_ids = select_components3(corrected_power, weights, "weighted", 256, component_count)
    coeff = gather_coeff3(corrected_ad, component_ids)
    pred_coeff = complex_idw(coeff, weights)
    pred_coeff = (pred_coeff * table[component_ids][:, None, None]).astype(np.complex64)
    pred = inverse_ad3(scatter_coeff3(component_ids, pred_coeff), n_order)
    full = full_predict(neigh_h, weights, radial, steer)
    return pred, full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/component_scalar_table.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--table-samples", type=int, default=1000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--component-counts", type=parse_ints, default="256,512,1024")
    parser.add_argument("--n-order", default="vp")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    train_idx = np.setdiff1d(np.arange(len(pos), dtype=np.int64), val_idx, assume_unique=False)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    d_val, neigh_val = make_neighbors(pos, val_idx, train_idx, args.k)

    results = []
    for component_count in args.component_counts:
        table, count = build_scalar_table(
            h,
            pos,
            train_idx,
            args.table_samples,
            args.seed + component_count,
            args.k,
            component_count,
            args.n_order,
            args.beta,
            args.steering_scale,
            args.power,
        )
        acc_table = MetricAccumulator("sumpol")
        acc_full = MetricAccumulator("sumpol")
        num_table = den_table = num_full = den_full = 0.0
        for i, qi in enumerate(val_idx):
            pred, full = predict_query(
                h,
                pos,
                u_all,
                dist_all,
                int(qi),
                neigh_val[i],
                d_val[i],
                table,
                component_count,
                args.n_order,
                args.beta,
                args.steering_scale,
                args.power,
            )
            truth = np.asarray(h[qi:qi + 1])
            nt, dt = eval_one(truth, pred)
            nf, df = eval_one(truth, full)
            num_table += nt
            den_table += dt
            num_full += nf
            den_full += df
            acc_table.add(pred, truth)
            acc_full.add(full, truth)
            if (i + 1) % 20 == 0 or i + 1 == len(val_idx):
                print(f"evaluated comp={component_count} {i + 1}/{len(val_idx)}", flush=True)
        results.append({
            "method": "component_scalar_table",
            "component_count": int(component_count),
            "table_nonzero": int(np.sum(count > 0)),
            "global_nmse_direct": float(num_table / max(den_table, 1e-30)),
            **acc_table.result(),
        })
        results.append({
            "method": "full_steer_reference",
            "component_count": int(component_count),
            "global_nmse_direct": float(num_full / max(den_full, 1e-30)),
            **acc_full.result(),
        })
    results.sort(key=lambda r: r["global_nmse_direct"])
    payload = {
        "seed": int(args.seed),
        "val_count": int(len(val_idx)),
        "val_mode": args.val_mode,
        "k": int(args.k),
        "n_order": args.n_order,
        "table_samples": int(args.table_samples),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
