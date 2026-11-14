#!/usr/bin/env python3
from project_paths import PATHS
"""Evaluate path-candidate-aware geometry branch blends.

This keeps the confirmed v36 regime structure, but replaces or mixes the
height-map geometry branch with a branch whose neighbor selection also matches
specular path candidates from the point cloud.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import idw_weights, parse_floats
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_s1style_geom_blend import geometry_cache, predict_phase_idw, select_geom_neighbors, train_log_power
from generate_geometric_steering_blend import unit_vectors
from path_candidate_geom_common import load_path_cache, path_signature, rho_for_blocked, select_path_geom_neighbors
from spectrum_projection import MetricAccumulator


BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def add_metric(stats, accs, name, pred, truth):
    if name not in stats:
        stats[name] = [0.0, 0.0, 0]
        accs[name] = MetricAccumulator("sumpol")
    stats[name][0] += float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
    stats[name][1] += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
    stats[name][2] += 1
    accs[name].add(pred[None], truth[None])


def finalize(stats, accs):
    rows = []
    for name, (num, den, count) in stats.items():
        rows.append({
            "name": name,
            "count": int(count),
            "global_nmse_direct": float(num / max(den, 1e-30)),
            **accs[name].result(),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/path_candidate_geom_blend_eval.json"))
    parser.add_argument("--height-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--geom-k", type=int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--clear-rho", type=float, default=0.0)
    parser.add_argument("--light-rho", type=float, default=0.0)
    parser.add_argument("--medium-rho", type=float, default=1.0)
    parser.add_argument("--heavy-rho", type=float, default=4.0)
    parser.add_argument("--branch-mixes", type=parse_floats, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--heavy-only", action="store_true", default=True)
    parser.add_argument("--path-top", type=int, default=3)
    parser.add_argument("--path-plane-penalty", type=float, default=0.40)
    parser.add_argument("--path-delay-penalty", type=float, default=0.25)
    parser.add_argument("--path-excess-penalty", type=float, default=0.05)
    parser.add_argument("--path-score-penalty", type=float, default=0.05)
    parser.add_argument("--path-delay-scale", type=float, default=1.0)
    parser.add_argument("--path-excess-scale", type=float, default=25.0)
    parser.add_argument("--path-score-scale", type=float, default=1.0)
    parser.add_argument("--los-penalty", type=float, default=1.0)
    parser.add_argument("--occ-penalty", type=float, default=0.5)
    parser.add_argument("--r-penalty", type=float, default=0.0)
    parser.add_argument("--theta-penalty", type=float, default=0.0)
    parser.add_argument("--occ-scale", type=float, default=80.0)
    parser.add_argument("--r-scale", type=float, default=80.0)
    parser.add_argument("--theta-scale-deg", type=float, default=30.0)
    args = parser.parse_args()

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    train_h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    train_blocked, _ = geometry_cache(args.data, args.height_cache, False)
    logpw = train_log_power(train_h)
    paths = load_path_cache(args.paths)
    train_paths = path_signature(paths, "train")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)

    tree = cKDTree(pos[:, :2])
    distances, neighbor_idx = tree.query(pos[val_idx, :2], k=args.k + 1, workers=-1)
    pos_all = pos
    dist_all = np.linalg.norm(pos_all - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos_all)

    base_cfg = {
        "k": int(args.geom_k),
        "los_penalty": float(args.los_penalty),
        "occ_penalty": float(args.occ_penalty),
        "r_penalty": float(args.r_penalty),
        "theta_penalty": float(args.theta_penalty),
        "occ_scale": float(args.occ_scale),
        "r_scale": float(args.r_scale),
        "theta_scale_deg": float(args.theta_scale_deg),
    }
    path_cfg = dict(base_cfg)
    path_cfg.update({
        "path_top": int(args.path_top),
        "path_plane_penalty": float(args.path_plane_penalty),
        "path_delay_penalty": float(args.path_delay_penalty),
        "path_excess_penalty": float(args.path_excess_penalty),
        "path_score_penalty": float(args.path_score_penalty),
        "path_delay_scale": float(args.path_delay_scale),
        "path_excess_scale": float(args.path_excess_scale),
        "path_score_scale": float(args.path_score_scale),
    })

    stats = {}
    accs = {}
    branch_counts = {"clear": 0, "light": 0, "medium": 0, "heavy": 0}
    for row, vi in enumerate(val_idx):
        truth = np.asarray(train_h[vi])
        keep = neighbor_idx[row] != vi
        idx = neighbor_idx[row, keep][: args.k]
        d = distances[row, keep][: args.k]
        neigh = np.asarray(train_h[idx])
        coeff = affine_coefficients(pos[idx, :2] - pos[vi, :2], d, args.power, args.ridge)
        v9 = predict_affine(
            neigh,
            coeff,
            dist_all[idx] - dist_all[vi],
            u_all,
            idx,
            int(vi),
            args.beta,
            args.gamma,
            args.alpha,
            args.steering_scale,
        )

        old_idx, _old_raw, old_eff = select_geom_neighbors(
            pos, train_blocked, pos[vi], train_blocked[vi], int(vi), base_cfg
        )
        old_w = idw_weights(old_eff, args.power)
        old_geom = predict_phase_idw(
            np.asarray(train_h[old_idx]),
            old_w,
            dist_all[old_idx] - dist_all[vi],
            args.beta,
            args.gamma,
            logpw,
            old_idx,
        )

        path_idx, _path_raw, path_eff = select_path_geom_neighbors(
            pos,
            train_blocked,
            train_paths,
            pos[vi],
            train_blocked[vi],
            {k: v[int(vi)] for k, v in train_paths.items()},
            int(vi),
            path_cfg,
        )
        path_w = idw_weights(path_eff, args.power)
        path_geom = predict_phase_idw(
            np.asarray(train_h[path_idx]),
            path_w,
            dist_all[path_idx] - dist_all[vi],
            args.beta,
            args.gamma,
            logpw,
            path_idx,
        )

        rho, label = rho_for_blocked(
            int(train_blocked[vi]), args.clear_rho, args.light_rho, args.medium_rho, args.heavy_rho
        )
        branch_counts[label] += 1
        old_pred = ((1.0 - rho) * v9 + rho * old_geom).astype(np.complex64)
        add_metric(stats, accs, "old_v36_rule", old_pred, truth)

        for mix in args.branch_mixes:
            use_mix = float(mix) if (label == "heavy" or not args.heavy_only) else 0.0
            geom = ((1.0 - use_mix) * old_geom + use_mix * path_geom).astype(np.complex64)
            pred = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
            add_metric(stats, accs, f"pathmix_{mix:g}", pred, truth)
            add_metric(stats, accs, f"{label}_pathmix_{mix:g}", pred, truth)

        if (row + 1) % 50 == 0 or row + 1 == len(val_idx):
            print(f"evaluated {row + 1}/{len(val_idx)}", flush=True)

    payload = {
        "args": vars(args),
        "branch_counts": branch_counts,
        "results": finalize(stats, accs),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"out": args.out, "top": payload["results"][:12], "branch_counts": branch_counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
