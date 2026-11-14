#!/usr/bin/env python3
from project_paths import PATHS
"""Generate a simple test-regime-aware v9 + s1-style geometry blend.

v15 keeps the confirmed v12 idea but avoids applying the same geometry blend
strength to clear-LoS points. Local regime analysis showed rho=0.30 helps
blocked/stress regimes, while clear-LoS points prefer v9 or a small rho.
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


def validate_shapes(pos, test_pos, train_h):
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"bad train position shape {pos.shape}")
    if test_pos.ndim != 2 or test_pos.shape[1] != 3:
        raise ValueError(f"bad test position shape {test_pos.shape}")
    if train_h.ndim != 4 or train_h.shape[1:] != (256, 4, 192):
        raise ValueError(f"bad channel shape {train_h.shape}")


def rho_for_blocked(blocked, clear_rho, light_rho, medium_rho, heavy_rho):
    if blocked <= 0:
        return float(clear_rho), "clear"
    if blocked <= 25:
        return float(light_rho), "light"
    if blocked <= 100:
        return float(medium_rho), "medium"
    return float(heavy_rho), "heavy"


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    validate_shapes(pos, test_pos, train_h)

    train_blocked, test_blocked = geometry_cache(args.data, args.cache, args.force_geometry)
    logpw = train_log_power(train_h)

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

    geom_cfg = {
        "k": int(args.geom_k),
        "los_penalty": float(args.los_penalty),
        "occ_penalty": float(args.occ_penalty),
        "r_penalty": float(args.r_penalty),
        "theta_penalty": float(args.theta_penalty),
        "occ_scale": float(args.occ_scale),
        "r_scale": float(args.r_scale),
        "theta_scale_deg": float(args.theta_scale_deg),
    }

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

            gidx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, test_pos[i], test_blocked[i], None, geom_cfg)
            weights = idw_weights(eff_d, args.power)
            geom = predict_phase_idw(
                np.asarray(train_h[gidx]),
                weights,
                dist_all[gidx] - dist_all[qi],
                args.beta,
                args.gamma,
                logpw if args.geom_power_sep else None,
                gidx if args.geom_power_sep else None,
            )
            rho, label = rho_for_blocked(
                int(test_blocked[i]),
                args.clear_rho,
                args.light_rho,
                args.medium_rho,
                args.heavy_rho,
            )
            pred = ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)
            out[i] = pred
            branch_stats[label] += 1
            branch_stats["rho_sum"] += rho

        block = np.asarray(out[start:end])
        if not (np.isfinite(block.real).all() and np.isfinite(block.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)

    branch_stats["rho_mean"] = branch_stats["rho_sum"] / max(len(test_pos), 1)
    return {
        "method": "regime_s1style_geom_blend",
        "base": "v9_physical_local_linear",
        "rule": "rho by height-map blocked count",
        "clear_rho": float(args.clear_rho),
        "light_rho": float(args.light_rho),
        "medium_rho": float(args.medium_rho),
        "heavy_rho": float(args.heavy_rho),
        "k": int(args.k),
        "geom_k": int(args.geom_k),
        "geom_power_sep": bool(args.geom_power_sep),
        "power": float(args.power),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "ridge": float(args.ridge),
        "los_penalty": float(args.los_penalty),
        "occ_penalty": float(args.occ_penalty),
        "r_penalty": float(args.r_penalty),
        "theta_penalty": float(args.theta_penalty),
        "branch_stats": branch_stats,
        "output": args.output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/regime_s1style_geom_blend_generate.json"))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--clear-rho", type=float, default=0.10)
    parser.add_argument("--light-rho", type=float, default=0.30)
    parser.add_argument("--medium-rho", type=float, default=0.30)
    parser.add_argument("--heavy-rho", type=float, default=0.30)
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
    args = parser.parse_args()
    result = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
