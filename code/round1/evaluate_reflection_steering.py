#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS, idw_weights, parse_floats
from generate_geometric_steering_blend import bs_phv_indices, steering_phase, unit_vectors
from spectrum_projection import MetricAccumulator


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def full_predict(neigh_h, weights, radial, phase=None):
    coeff = weights.astype(np.float32) * radial.astype(np.complex64)
    if phase is None:
        return np.sum(coeff[:, None, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)
    return np.sum(coeff[:, None, None, None] * phase[:, :, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)


def eval_one(pred, truth):
    return (
        float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def make_split(pos, test_pos, mode, val_count, seed, k):
    if mode == "random":
        train_idx, val_idx = build_split(len(pos), val_count, seed)
        tree = cKDTree(pos[train_idx, :2])
        distances, local = tree.query(pos[val_idx, :2], k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            local = local[:, None]
        return val_idx, distances, train_idx[local]
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], val_count, mode, seed)
    tree = cKDTree(pos[:, :2])
    distances_all, local_all = tree.query(pos[val_idx, :2], k=k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), k), dtype=np.int64)
    distances = np.empty((len(val_idx), k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:k]
        distances[i] = distances_all[i, keep][:k]
    return val_idx, distances, neighbor_idx


def phase_from_units(u_neighbors, u_query, scale):
    ant = bs_phv_indices()
    h_coord = ant[:, 0]
    v_coord = ant[:, 1]
    delta_h = u_query[1] - u_neighbors[:, 1]
    delta_v = (-u_query[2]) - (-u_neighbors[:, 2])
    phase_arg = scale * (
        delta_h[:, None] * h_coord[None, :] + delta_v[:, None] * v_coord[None, :]
    )
    return np.exp(1j * phase_arg).astype(np.complex64)


def reflection_lengths_and_units(points, plane_point, plane_normal):
    n = np.asarray(plane_normal, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        n = n / norm
    p0 = np.asarray(plane_point, dtype=np.float64)
    mirror_bs = BS_POS - 2.0 * np.dot(BS_POS - p0, n) * n
    points = np.asarray(points, dtype=np.float64)
    vec = points - mirror_bs[None, :]
    lengths = np.linalg.norm(vec, axis=1)
    denom = vec @ n
    numer = np.dot(p0 - mirror_bs, n)
    t = np.divide(numer, denom, out=np.full_like(denom, np.nan), where=np.abs(denom) > 1e-9)
    refl = mirror_bs[None, :] + t[:, None] * vec
    out_vec = refl - BS_POS[None, :]
    out_norm = np.linalg.norm(out_vec, axis=1, keepdims=True)
    units = out_vec / np.maximum(out_norm, 1e-9)
    valid = np.isfinite(t) & (t > 0.0) & (t < 1.0) & (out_norm[:, 0] > 1e-9)
    return lengths.astype(np.float64), units.astype(np.float64), valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/reflection_steering.json"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--mode", choices=["random", "matched-test"], default="random")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--rhos", type=parse_floats, default="0,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    feature_data = np.load(args.features, allow_pickle=True)
    train_features = feature_data["train_features"]
    names = [str(x) for x in feature_data["feature_names"]]

    val_idx, distances, neighbor_idx = make_split(pos, test_pos, args.mode, args.val_count, args.seed, args.k)
    val_features = train_features[val_idx]
    clear = val_features[:, names.index("los_min_dist")] >= 0.5
    plane_points = val_features[:, [
        names.index("rx_wall_x2"),
        names.index("rx_wall_y2"),
        names.index("rx_wall_z2"),
    ]]
    plane_normals = val_features[:, [
        names.index("rx_wall_nx2"),
        names.index("rx_wall_ny2"),
        names.index("rx_wall_nz2"),
    ]]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    candidates = [
        ("direct_a0", 0.0),
        ("direct_a025", 0.25),
        ("direct_a05", 0.5),
        ("direct_a1", 1.0),
        ("reflect", 1.0),
        ("map_alpha", 0.0),
    ]
    for rho in args.rhos:
        candidates.append((f"blocked_reflect_rho{rho:g}", rho))
        candidates.append((f"all_reflect_rho{rho:g}", rho))

    accs = {str(c): MetricAccumulator("sumpol") for c in candidates}
    stats = {str(c): {"num": 0.0, "den": 0.0, "valid": 0} for c in candidates}

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        idx = neighbor_idx[ii]
        neigh = np.asarray(h[idx])
        weights = idw_weights(distances[ii], args.power)
        radial_direct = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
        direct_phase = steering_phase(u_all, idx, vi, args.steering_scale)
        direct0 = full_predict(neigh, weights, radial_direct)
        direct1 = full_predict(neigh, weights, radial_direct, direct_phase)
        direct025 = (0.75 * direct0 + 0.25 * direct1).astype(np.complex64)
        direct05 = (0.5 * direct0 + 0.5 * direct1).astype(np.complex64)
        map_alpha = direct1 if clear[ii] else direct025

        all_points = np.concatenate([pos[idx], pos[vi:vi + 1]], axis=0)
        lengths, units, valid = reflection_lengths_and_units(all_points, plane_points[ii], plane_normals[ii])
        radial_ref = np.exp(1j * args.beta * (lengths[:-1] - lengths[-1])).astype(np.complex64)
        ref_phase = phase_from_units(units[:-1], units[-1], args.steering_scale)
        reflect = full_predict(neigh, weights, radial_ref, ref_phase)

        pred_map = {
            ("direct_a0", 0.0): direct0,
            ("direct_a025", 0.25): direct025,
            ("direct_a05", 0.5): direct05,
            ("direct_a1", 1.0): direct1,
            ("reflect", 1.0): reflect,
            ("map_alpha", 0.0): map_alpha,
        }
        for rho in args.rhos:
            pred_map[(f"blocked_reflect_rho{rho:g}", rho)] = (
                direct1 if clear[ii] else ((1.0 - rho) * direct025 + rho * reflect).astype(np.complex64)
            )
            pred_map[(f"all_reflect_rho{rho:g}", rho)] = ((1.0 - rho) * direct05 + rho * reflect).astype(np.complex64)

        for c, pred in pred_map.items():
            key = str(c)
            num, den = eval_one(pred, truth)
            stats[key]["num"] += num
            stats[key]["den"] += den
            stats[key]["valid"] += int(np.all(valid))
            accs[key].add(pred, truth)

        if (ii + 1) % 20 == 0 or ii + 1 == len(val_idx):
            print(f"evaluated {ii + 1}/{len(val_idx)}", flush=True)

    results = []
    for c in candidates:
        key = str(c)
        results.append({
            "candidate": c[0],
            "rho": float(c[1]),
            "clear_count": int(np.sum(clear)),
            "valid_reflection_count": int(stats[key]["valid"]),
            "global_nmse_direct": float(stats[key]["num"] / max(stats[key]["den"], 1e-30)),
            **accs[key].result(),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    payload = {
        "mode": args.mode,
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
