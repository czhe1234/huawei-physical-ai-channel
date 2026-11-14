#!/usr/bin/env python3
from project_paths import PATHS
"""Generate path-candidate-aware regime geometry blend.

This is a v45-style candidate. It keeps the online-confirmed v36 structure:
physical-local-linear base plus regime-dependent geometry extrapolation. The
only change is the geometry branch: neighbor selection also matches specular
path candidates from the point-cloud-derived path cache.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_map_features import BS_POS
from evaluate_local_ad_paths import idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_s1style_geom_blend import geometry_cache, predict_phase_idw, select_geom_neighbors, train_log_power
from generate_geometric_steering_blend import unit_vectors
from path_candidate_geom_common import load_path_cache, path_signature, rho_for_blocked, select_path_geom_neighbors


def validate_shapes(pos, test_pos, train_h):
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"bad train position shape {pos.shape}")
    if test_pos.ndim != 2 or test_pos.shape[1] != 3:
        raise ValueError(f"bad test position shape {test_pos.shape}")
    if train_h.ndim != 4 or train_h.shape[1:] != (256, 4, 192):
        raise ValueError(f"bad channel shape {train_h.shape}")


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    validate_shapes(pos, test_pos, train_h)

    train_blocked, test_blocked = geometry_cache(args.data, args.cache, args.force_geometry)
    logpw = train_log_power(train_h)
    paths = load_path_cache(args.paths)
    train_paths = path_signature(paths, "train")
    test_paths = path_signature(paths, "test")

    max_k = max(args.k, args.geom_k)
    tree = cKDTree(pos[:, :2])
    distances, neighbor_idx = tree.query(test_pos[:, :2], k=max_k, workers=-1)
    if max_k == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]

    pos_all = np.concatenate([pos, test_pos], axis=0)
    dist_all = np.linalg.norm(pos_all - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos_all)
    test_offset = len(pos)

    old_cfg = {
        "k": int(args.geom_k),
        "los_penalty": float(args.los_penalty),
        "occ_penalty": float(args.occ_penalty),
        "r_penalty": float(args.r_penalty),
        "theta_penalty": float(args.theta_penalty),
        "occ_scale": float(args.occ_scale),
        "r_scale": float(args.r_scale),
        "theta_scale_deg": float(args.theta_scale_deg),
    }
    path_cfg = dict(old_cfg)
    path_cfg.update({
        "path_top": int(args.path_top),
        "path_plane_penalty": float(args.path_plane_penalty),
        "path_delay_penalty": float(args.path_delay_penalty),
        "path_excess_penalty": float(args.path_excess_penalty),
        "path_score_penalty": float(args.path_score_penalty),
        "path_delay_scale": float(args.path_delay_scale),
        "path_excess_scale": float(args.path_excess_scale),
        "path_score_scale": float(args.path_score_scale),
        "two_stage": bool(args.path_two_stage),
    })

    out = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), train_h.shape[1], train_h.shape[2], train_h.shape[3]),
    )
    branch_stats = {
        "clear": 0,
        "light": 0,
        "medium": 0,
        "heavy": 0,
        "rho_sum": 0.0,
        "pathmix_sum": 0.0,
    }
    for start in range(0, len(test_pos), args.batch):
        end = min(start + args.batch, len(test_pos))
        for i in range(start, end):
            idx = neighbor_idx[i, : args.k]
            rel_xy = pos[idx, :2] - test_pos[i, :2]
            coeff = affine_coefficients(rel_xy, distances[i, : args.k], args.power, args.ridge)
            neigh = np.asarray(train_h[idx])
            qi = test_offset + i
            v9 = predict_affine(
                neigh,
                coeff,
                dist_all[idx] - dist_all[qi],
                u_all,
                idx,
                qi,
                args.beta,
                args.gamma,
                args.alpha,
                args.steering_scale,
            )

            old_idx, _raw_d, old_eff = select_geom_neighbors(
                pos, train_blocked, test_pos[i], test_blocked[i], None, old_cfg
            )
            old_w = idw_weights(old_eff, args.power)
            old_geom = predict_phase_idw(
                np.asarray(train_h[old_idx]),
                old_w,
                dist_all[old_idx] - dist_all[qi],
                args.beta,
                args.gamma,
                logpw if args.geom_power_sep else None,
                old_idx if args.geom_power_sep else None,
            )

            path_idx, _path_raw, path_eff = select_path_geom_neighbors(
                pos,
                train_blocked,
                train_paths,
                test_pos[i],
                test_blocked[i],
                {k: v[i] for k, v in test_paths.items()},
                None,
                path_cfg,
            )
            path_w = idw_weights(path_eff, args.power)
            path_geom = predict_phase_idw(
                np.asarray(train_h[path_idx]),
                path_w,
                dist_all[path_idx] - dist_all[qi],
                args.beta,
                args.gamma,
                logpw if args.geom_power_sep else None,
                path_idx if args.geom_power_sep else None,
            )

            rho, label = rho_for_blocked(
                int(test_blocked[i]), args.clear_rho, args.light_rho, args.medium_rho, args.heavy_rho
            )
            pathmix = args.pathmix if (label == "heavy" or not args.heavy_only) else 0.0
            geom = ((1.0 - pathmix) * old_geom + pathmix * path_geom).astype(np.complex64)
            out[i] = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
            branch_stats[label] += 1
            branch_stats["rho_sum"] += rho
            branch_stats["pathmix_sum"] += pathmix

        block = np.asarray(out[start:end])
        if not (np.isfinite(block.real).all() and np.isfinite(block.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)

    branch_stats["rho_mean"] = branch_stats["rho_sum"] / max(len(test_pos), 1)
    branch_stats["pathmix_mean"] = branch_stats["pathmix_sum"] / max(len(test_pos), 1)
    return {
        "method": "path_candidate_geom_blend",
        "base": "v36_regime_s1style_geom_blend",
        "rule": "v36 rho by blocked count; geometry branch uses specular path-candidate-aware neighbor selection",
        "clear_rho": float(args.clear_rho),
        "light_rho": float(args.light_rho),
        "medium_rho": float(args.medium_rho),
        "heavy_rho": float(args.heavy_rho),
        "pathmix": float(args.pathmix),
        "heavy_only": bool(args.heavy_only),
        "path_top": int(args.path_top),
        "path_plane_penalty": float(args.path_plane_penalty),
        "path_delay_penalty": float(args.path_delay_penalty),
        "path_excess_penalty": float(args.path_excess_penalty),
        "path_score_penalty": float(args.path_score_penalty),
        "k": int(args.k),
        "geom_k": int(args.geom_k),
        "geom_power_sep": bool(args.geom_power_sep),
        "power": float(args.power),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "ridge": float(args.ridge),
        "branch_stats": branch_stats,
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/path_candidate_geom_blend_generate.json"))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--clear-rho", type=float, default=0.0)
    parser.add_argument("--light-rho", type=float, default=0.0)
    parser.add_argument("--medium-rho", type=float, default=1.0)
    parser.add_argument("--heavy-rho", type=float, default=4.0)
    parser.add_argument("--pathmix", type=float, default=1.0)
    parser.add_argument("--heavy-only", action="store_true", default=True)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--geom-k", type=int, default=16)
    parser.add_argument("--geom-power-sep", action="store_true", default=True)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--los-penalty", type=float, default=1.0)
    parser.add_argument("--occ-penalty", type=float, default=0.5)
    parser.add_argument("--r-penalty", type=float, default=0.0)
    parser.add_argument("--theta-penalty", type=float, default=0.0)
    parser.add_argument("--occ-scale", type=float, default=80.0)
    parser.add_argument("--r-scale", type=float, default=80.0)
    parser.add_argument("--theta-scale-deg", type=float, default=30.0)
    parser.add_argument("--path-top", type=int, default=3)
    parser.add_argument("--path-plane-penalty", type=float, default=0.80)
    parser.add_argument("--path-delay-penalty", type=float, default=0.50)
    parser.add_argument("--path-excess-penalty", type=float, default=0.10)
    parser.add_argument("--path-score-penalty", type=float, default=0.05)
    parser.add_argument("--path-delay-scale", type=float, default=1.0)
    parser.add_argument("--path-excess-scale", type=float, default=25.0)
    parser.add_argument("--path-score-scale", type=float, default=1.0)
    parser.add_argument("--path-two-stage", action="store_true", default=False)
    args = parser.parse_args()

    result = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
