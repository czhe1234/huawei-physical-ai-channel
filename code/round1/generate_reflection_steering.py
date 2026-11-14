#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from generate_geometric_steering_blend import (
    BS_POS,
    bs_phv_indices,
    idw_weights,
    steering_phase,
    unit_vectors,
)


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
    units = out_vec / np.maximum(np.linalg.norm(out_vec, axis=1, keepdims=True), 1e-9)
    return lengths.astype(np.float64), units.astype(np.float64)


def full_predict(neigh, weights, radial, phase=None):
    coeff = weights.astype(np.float32) * radial.astype(np.complex64)
    if phase is None:
        return np.sum(coeff[:, None, None, None] * neigh, axis=0).astype(np.complex64)
    return np.sum(coeff[:, None, None, None] * phase[:, :, None, None] * neigh, axis=0).astype(np.complex64)


def load_geometry(features_path):
    data = np.load(features_path, allow_pickle=True)
    features = data["test_features"]
    names = [str(x) for x in data["feature_names"]]
    clear = features[:, names.index("los_min_dist")] >= 0.5
    plane_points = features[:, [
        names.index("rx_wall_x2"),
        names.index("rx_wall_y2"),
        names.index("rx_wall_z2"),
    ]]
    plane_normals = features[:, [
        names.index("rx_wall_nx2"),
        names.index("rx_wall_ny2"),
        names.index("rx_wall_nz2"),
    ]]
    return clear, plane_points, plane_normals


def predict_one(train_h, pos, idx, distances, test_pos, query_test_i, clear, plane_point, plane_normal, dist_all, u_all, args):
    neigh = np.asarray(train_h[idx])
    weights = idw_weights(distances, args.power)
    query_global_idx = len(pos) + query_test_i
    radial_direct = np.exp(1j * args.beta * (dist_all[idx] - dist_all[query_global_idx])).astype(np.complex64)
    direct_phase = steering_phase(u_all, idx, query_global_idx, args.steering_scale)
    direct0 = full_predict(neigh, weights, radial_direct)
    direct1 = full_predict(neigh, weights, radial_direct, direct_phase)
    if clear:
        return direct1
    direct025 = (0.75 * direct0 + 0.25 * direct1).astype(np.complex64)

    all_points = np.concatenate([pos[idx], test_pos[query_test_i:query_test_i + 1]], axis=0)
    lengths, units = reflection_lengths_and_units(all_points, plane_point, plane_normal)
    radial_ref = np.exp(1j * args.beta * (lengths[:-1] - lengths[-1])).astype(np.complex64)
    ref_phase = phase_from_units(units[:-1], units[-1], args.steering_scale)
    reflect = full_predict(neigh, weights, radial_ref, ref_phase)
    return ((1.0 - args.reflection_rho) * direct025 + args.reflection_rho * reflect).astype(np.complex64)


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    clear, plane_points, plane_normals = load_geometry(args.features)

    pos_all = np.concatenate([pos, test_pos], axis=0)
    dist_all = np.linalg.norm(pos_all - BS_POS[None, :], axis=1)
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
    for start in range(0, len(test_pos), args.batch):
        end = min(start + args.batch, len(test_pos))
        for i in range(start, end):
            out[i] = predict_one(
                train_h,
                pos,
                neighbor_idx[i],
                distances[i],
                test_pos,
                i,
                bool(clear[i]),
                plane_points[i],
                plane_normals[i],
                dist_all,
                u_all,
                args,
            )
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)

    return {
        "method": "map_los_nearest_wall_reflection",
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "los_rule": "los_min_dist >= 0.5",
        "clear_action": "direct_alpha_1.0",
        "blocked_action": "0.75 * direct_alpha_0.25 + rho * reflection",
        "reflection_rho": float(args.reflection_rho),
        "clear_count": int(np.sum(clear)),
        "blocked_count": int(len(clear) - np.sum(clear)),
        "features": args.features,
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/reflection_steering_generate.json"))
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--reflection-rho", type=float, default=0.25)
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
