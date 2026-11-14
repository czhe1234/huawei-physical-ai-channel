#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_feature_matched_candidates import choose_feature_matched_indices, make_query_features
from evaluate_geometric_steering import BS_POS, predict_config, unit_vectors
from evaluate_local_ad_paths import idw_weights
from generate_geometric_steering_blend import steering_phase
from spectrum_projection import MetricAccumulator


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x]


def load_planes(path, max_planes):
    payload = json.load(open(path))
    planes = payload["planes"][:max_planes]
    n = np.asarray([p["normal"] for p in planes], dtype=np.float64)
    c = np.asarray([p["c"] for p in planes], dtype=np.float64)
    t_min = np.asarray([p["t_min"] for p in planes], dtype=np.float64)
    t_max = np.asarray([p["t_max"] for p in planes], dtype=np.float64)
    z_min = np.asarray([p["z_min"] for p in planes], dtype=np.float64)
    z_max = np.asarray([p["z_max"] for p in planes], dtype=np.float64)
    counts = np.asarray([p["count"] for p in planes], dtype=np.float64)
    tangent = np.stack([-n[:, 1], n[:, 0]], axis=1)
    return {
        "normal": n,
        "c": c,
        "tangent": tangent,
        "t_min": t_min,
        "t_max": t_max,
        "z_min": z_min,
        "z_max": z_max,
        "counts": counts,
    }


def select_specular_planes(points, planes, top_r, margin_t, margin_z):
    pxy = points[:, :2].astype(np.float64)
    pz = points[:, 2].astype(np.float64)
    n = planes["normal"]
    c = planes["c"]
    tangent = planes["tangent"]
    bs_xy = BS_POS[:2].astype(np.float64)
    bs_z = float(BS_POS[2])
    bs_dot = n @ bs_xy
    mirror_xy = bs_xy[None, :] - 2.0 * (bs_dot - c)[:, None] * n
    denom = np.sum((pxy[:, None, :] - mirror_xy[None, :, :]) * n[None, :, :], axis=2)
    num = c - np.sum(mirror_xy * n, axis=1)
    tau = num[None, :] / np.where(np.abs(denom) > 1e-12, denom, np.nan)
    refl_xy = mirror_xy[None, :, :] + tau[:, :, None] * (pxy[:, None, :] - mirror_xy[None, :, :])
    refl_z = bs_z + tau * (pz[:, None] - bs_z)
    t_coord = np.einsum("pjk,jk->pj", refl_xy, tangent)
    path = np.linalg.norm(refl_xy - bs_xy[None, None, :], axis=2) + np.sqrt(
        np.sum((refl_xy - pxy[:, None, :]) ** 2, axis=2) + (refl_z - pz[:, None]) ** 2
    )
    valid = (
        np.isfinite(tau)
        & (tau > 0.0)
        & (tau < 1.0)
        & (t_coord >= planes["t_min"][None, :] - margin_t)
        & (t_coord <= planes["t_max"][None, :] + margin_t)
        & (refl_z >= planes["z_min"][None, :] - margin_z)
        & (refl_z <= planes["z_max"][None, :] + margin_z)
    )
    count_score = np.sqrt(planes["counts"] / max(float(np.max(planes["counts"])), 1.0))
    score = count_score[None, :] / np.maximum(path, 1e-6)
    score[~valid] = -np.inf
    top_r = min(top_r, score.shape[1])
    idx = np.argpartition(-score, kth=np.arange(top_r), axis=1)[:, :top_r]
    vals = np.take_along_axis(score, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return idx.astype(np.int64), vals


def specular_path_for_plane(points, plane):
    p = points.astype(np.float64)
    n = plane["normal"].astype(np.float64)
    c = float(plane["c"])
    bs = BS_POS.astype(np.float64)
    bs_dot = float(np.dot(n[:2], bs[:2]))
    mirror_xy = bs[:2] - 2.0 * (bs_dot - c) * n[:2]
    denom = (p[:, :2] - mirror_xy[None, :]) @ n[:2]
    num = c - float(np.dot(mirror_xy, n[:2]))
    tau = num / np.where(np.abs(denom) > 1e-12, denom, np.nan)
    refl_xy = mirror_xy[None, :] + tau[:, None] * (p[:, :2] - mirror_xy[None, :])
    refl_z = bs[2] + tau * (p[:, 2] - bs[2])
    refl = np.column_stack([refl_xy, refl_z])
    first = np.linalg.norm(refl - bs[None, :], axis=1)
    second = np.linalg.norm(p - refl, axis=1)
    length = first + second
    unit = (refl - bs[None, :]) / np.maximum(first[:, None], 1e-30)
    bad = ~np.isfinite(length)
    if np.any(bad):
        # Fall back to the infinite-plane virtual-source path length.
        mirror = np.array([mirror_xy[0], mirror_xy[1], bs[2]], dtype=np.float64)
        length[bad] = np.linalg.norm(p[bad] - mirror[None, :], axis=1)
        unit[bad] = (p[bad] - bs[None, :]) / np.maximum(np.linalg.norm(p[bad] - bs[None, :], axis=1, keepdims=True), 1e-30)
    return length, unit


def make_split(args, pos, test_pos):
    if args.val_mode == "feature-matched":
        map_feature_names = [
            "los_min_dist",
            "los_p05_dist",
            "los_p10_dist",
            "los_mean_inv_dist",
            "los_frac_r0p5",
            "los_frac_r0p75",
            "los_frac_r1p0",
            "los_frac_r1p5",
            "los_frac_r2p0",
            "los_frac_r3p0",
            "rx_wall_dist3",
            "rx_wall_dist2",
            "rx_wall_nx2",
            "rx_wall_ny2",
            "rx_wall_nz2",
        ]
        train_feat, test_feat, _ = make_query_features(args.data, args.features, map_feature_names)
        return choose_feature_matched_indices(train_feat, test_feat, args.val_count, args.seed, 128)
    return choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--planes", default=str(PATHS.analysis / "results/json/wall_planes.json"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/specular_reflection_500.json"))
    parser.add_argument("--val-mode", choices=["random", "matched-test", "feature-matched"], default="feature-matched")
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--max-planes", type=int, default=200)
    parser.add_argument("--top-r", type=int, default=3)
    parser.add_argument("--rhos", type=parse_floats, default="0,0.1,0.2,0.3")
    parser.add_argument("--margin-t", type=float, default=4.0)
    parser.add_argument("--margin-z", type=float, default=5.0)
    parser.add_argument("--batch", type=int, default=5)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    val_idx = make_split(args, pos, test_pos)
    planes = load_planes(args.planes, args.max_planes)
    plane_idx, plane_scores = select_specular_planes(pos[val_idx], planes, args.top_r, args.margin_t, args.margin_z)

    tree = cKDTree(pos[:, :2])
    distances_all, local_all = tree.query(pos[val_idx, :2], k=args.k + 1, workers=-1)
    neighbor_idx = np.empty((len(val_idx), args.k), dtype=np.int64)
    distances = np.empty((len(val_idx), args.k), dtype=np.float64)
    for i, vi in enumerate(val_idx):
        keep = local_all[i] != vi
        neighbor_idx[i] = local_all[i, keep][:args.k]
        distances[i] = distances_all[i, keep][:args.k]

    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    cfg_base = {"name": "baseline", "order": "phv", "h_axis": "none", "v_axis": "none", "bs_scale": 0.0, "ue_scale": 0.0}
    cfg_geo = {"name": "bs_steering", "order": "phv", "h_axis": "uy", "v_axis": "nuz", "bs_scale": args.steering_scale, "ue_scale": 0.0}
    accs = {rho: MetricAccumulator("sumpol") for rho in args.rhos}

    plane_dicts = []
    for j in range(len(planes["c"])):
        plane_dicts.append({
            "normal": np.array([planes["normal"][j, 0], planes["normal"][j, 1], 0.0], dtype=np.float64),
            "c": float(planes["c"][j]),
        })

    valid_count = int(np.sum(np.isfinite(plane_scores[:, 0]) & (plane_scores[:, 0] > -np.inf)))
    for start in range(0, len(val_idx), args.batch):
        end = min(start + args.batch, len(val_idx))
        truth = np.asarray(h[val_idx[start:end]])
        preds = {rho: np.empty_like(truth) for rho in args.rhos}
        for j, row in enumerate(range(start, end)):
            vi = val_idx[row]
            idx = neighbor_idx[row]
            neigh = np.asarray(h[idx])
            weights = idw_weights(distances[row], args.power)
            radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
            base = predict_config(neigh, weights, radial, u_all, idx, vi, cfg_base)
            geo = predict_config(neigh, weights, radial, u_all, idx, vi, cfg_geo)
            direct = (0.5 * base + 0.5 * geo).astype(np.complex64)

            refl_acc = np.zeros_like(direct, dtype=np.complex64)
            refl_weight = 0.0
            for rr in range(args.top_r):
                score = plane_scores[row, rr]
                if not np.isfinite(score) or score <= -np.inf:
                    continue
                pl = plane_dicts[int(plane_idx[row, rr])]
                points = np.concatenate([pos[idx], pos[vi:vi + 1]], axis=0)
                lengths, units = specular_path_for_plane(points, pl)
                radial_ref = np.exp(1j * args.beta * (lengths[:-1] - lengths[-1])).astype(np.complex64)
                phase_ref = steering_phase(units, np.arange(len(idx)), len(idx), args.steering_scale)
                coeff = weights * radial_ref
                reflect = np.sum(coeff[:, None, None, None] * phase_ref[:, :, None, None] * neigh, axis=0).astype(np.complex64)
                w = max(float(score), 0.0)
                refl_acc += np.float32(w) * reflect
                refl_weight += w
            if refl_weight > 0.0:
                reflect = (refl_acc / np.float32(refl_weight)).astype(np.complex64)
            else:
                reflect = direct
            for rho in args.rhos:
                preds[rho][j] = ((1.0 - rho) * direct + rho * reflect).astype(np.complex64)
        for rho in args.rhos:
            accs[rho].add(preds[rho], truth)
        print(f"evaluated {end}/{len(val_idx)}", flush=True)

    results = []
    for rho in args.rhos:
        results.append({
            "rho": float(rho),
            "val_mode": args.val_mode,
            "k": int(args.k),
            "top_r": int(args.top_r),
            "max_planes": int(args.max_planes),
            **accs[rho].result(),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    payload = {
        "seed": int(args.seed),
        "val_count": int(len(val_idx)),
        "valid_first_plane_count": valid_count,
        "mean_first_plane_score": float(np.mean(np.maximum(plane_scores[:, 0], 0.0))),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
