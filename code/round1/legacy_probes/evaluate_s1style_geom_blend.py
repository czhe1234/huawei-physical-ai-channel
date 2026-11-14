#!/usr/bin/env python3
from project_paths import PATHS
"""Evaluate s1-style height-map geometry gating as a blend on top of v9.

This does not use test labels. It builds a 2.5D height map from Round1_Map.ply,
computes LoS/occlusion features, evaluates geometry-gated phase-aligned IDW on
a training holdout, and checks whether a small blend with the v9 physical local
linear baseline improves local metrics.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_map_features import BS_POS, read_ply_memmap
from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_frequency_dependent_phase import S_SC
from evaluate_local_ad_paths import idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from generate_geometric_steering_blend import unit_vectors
from spectrum_projection import MetricAccumulator


def build_height_map(ply_path, cell=2.0, z_min=1.0):
    vertices = read_ply_memmap(ply_path)
    x = np.asarray(vertices["x"])
    y = np.asarray(vertices["y"])
    z = np.asarray(vertices["z"])
    mask = z >= z_min
    xy = np.stack([x[mask], y[mask]], axis=1).astype(np.float64)
    zz = z[mask].astype(np.float32)
    origin = xy.min(axis=0)
    ij = np.floor((xy - origin[None, :]) / cell).astype(np.int64)
    width = int(ij[:, 0].max()) + 1
    height = int(ij[:, 1].max()) + 1
    hmap = np.zeros((width, height), dtype=np.float32)
    np.maximum.at(hmap.reshape(-1), ij[:, 0] * height + ij[:, 1], zz)
    return hmap, origin.astype(np.float64), float(cell)


def occlusion_counts(pos, hmap, origin, cell, samples=300, skip_near=4.0, clear=1.0):
    pts = np.asarray(pos, dtype=np.float64)
    width, height = hmap.shape
    blocked = np.zeros(len(pts), dtype=np.int32)
    t_all = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    for i, p in enumerate(pts):
        d_xy = np.linalg.norm(p[:2] - BS_POS[:2])
        if d_xy < 1e-6:
            continue
        seg = BS_POS[None, :] * (1.0 - t_all[:, None]) + p[None, :] * t_all[:, None]
        keep = (t_all * d_xy > skip_near) & (t_all * d_xy < d_xy - skip_near)
        seg = seg[keep]
        if len(seg) == 0:
            continue
        ij = np.floor((seg[:, :2] - origin[None, :]) / cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < width) & (ij[:, 1] >= 0) & (ij[:, 1] < height)
        ij = ij[ok]
        ray_z = seg[ok, 2]
        if len(ij) == 0:
            continue
        blocked[i] = int(np.sum(hmap[ij[:, 0], ij[:, 1]] > ray_z + clear))
        if (i + 1) % 500 == 0:
            print(f"occlusion {i + 1}/{len(pts)}", flush=True)
    return blocked


def geometry_cache(data_dir, cache_path, force=False):
    cache = Path(cache_path)
    if cache.exists() and not force:
        payload = np.load(cache)
        return payload["train_blocked"], payload["test_blocked"]
    data = Path(data_dir)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    hmap, origin, cell = build_height_map(data / "Round1_Map.ply")
    all_pos = np.concatenate([train_pos, test_pos], axis=0)
    blocked = occlusion_counts(all_pos, hmap, origin, cell)
    train_blocked = blocked[: len(train_pos)]
    test_blocked = blocked[len(train_pos) :]
    np.savez_compressed(cache, train_blocked=train_blocked, test_blocked=test_blocked)
    return train_blocked, test_blocked


def train_log_power(train_h, batch=64):
    out = np.empty(train_h.shape[0], dtype=np.float64)
    for start in range(0, train_h.shape[0], batch):
        end = min(start + batch, train_h.shape[0])
        block = np.asarray(train_h[start:end])
        out[start:end] = np.log(np.maximum(np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64), 1e-30))
    return out


def angle_abs_diff(theta, q_theta):
    return np.abs(np.angle(np.exp(1j * (theta - q_theta))))


def select_geom_neighbors(pos, train_blocked, q_pos, q_blocked, q_index, cfg):
    d_xy = np.linalg.norm(pos[:, :2] - q_pos[:2], axis=1)
    if q_index is not None:
        d_xy[q_index] = np.inf
    los = train_blocked == 0
    q_los = q_blocked == 0
    d = pos - BS_POS[None, :]
    train_r = np.linalg.norm(d, axis=1)
    train_theta = np.arctan2(d[:, 1], d[:, 0])
    q_d = q_pos - BS_POS
    q_r = np.linalg.norm(q_d)
    q_theta = np.arctan2(q_d[1], q_d[0])
    occ_diff = np.abs(train_blocked.astype(np.float64) - float(q_blocked)) / max(cfg["occ_scale"], 1e-6)
    r_diff = np.abs(train_r - q_r) / max(cfg["r_scale"], 1e-6)
    theta_diff = angle_abs_diff(train_theta, q_theta) / np.deg2rad(max(cfg["theta_scale_deg"], 1e-6))
    eff = d_xy * (
        1.0
        + cfg["los_penalty"] * (los != q_los).astype(np.float64)
        + cfg["occ_penalty"] * occ_diff
        + cfg["r_penalty"] * r_diff
        + cfg["theta_penalty"] * theta_diff
    )
    if cfg.get("two_stage", False):
        mask = (los == q_los) & (np.abs(train_blocked - q_blocked) <= cfg.get("two_stage_occ_window", 50.0))
        if np.sum(mask) >= cfg["k"]:
            eff = np.where(mask, eff, np.inf)
    k = min(cfg["k"], len(pos))
    idx = np.argpartition(eff, k - 1)[:k]
    idx = idx[np.argsort(eff[idx])]
    return idx, d_xy[idx], eff[idx]


def predict_phase_idw(neigh, weights, dist_delta, beta, gamma, log_power=None, logpw_idx=None):
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    beta_s = (beta + gamma * sub).astype(np.float32)
    radial = np.exp(1j * dist_delta[:, None] * beta_s[None, :]).astype(np.complex64)
    pred = np.sum(weights[:, None, None, None] * radial[:, None, None, :] * neigh, axis=0).astype(np.complex64)
    if log_power is not None and logpw_idx is not None:
        cur = float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        if cur > 1e-30:
            target = float(np.exp(np.dot(weights.astype(np.float64), log_power[logpw_idx])))
            pred = (pred * np.float32(np.sqrt(target / cur))).astype(np.complex64)
    return pred


def add_metric(accs, nums, dens, name, pred, truth):
    nums[name] += float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
    dens[name] += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
    accs[name].add(pred[None], truth[None])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/s1style_geom_blend_val200.json"))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--rhos", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--force-geometry", action="store_true")
    args = parser.parse_args()

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    train_h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    train_blocked, _ = geometry_cache(args.data, args.cache, args.force_geometry)
    logpw = train_log_power(train_h)

    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    max_k = 24
    tree = cKDTree(pos[:, :2])
    d_all, idx_all = tree.query(pos[val_idx, :2], k=max_k + 1, workers=-1)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    rhos = [float(x) for x in args.rhos.split(",") if x]
    geom_cfgs = [
        {"name": "phase_k16", "k": 16, "geom": False, "power_sep": False},
        {"name": "phase_pw_k16", "k": 16, "geom": False, "power_sep": True},
        {"name": "geom_soft_k16_L1_O05", "k": 16, "geom": True, "power_sep": False, "los_penalty": 1.0, "occ_penalty": 0.5, "r_penalty": 0.0, "theta_penalty": 0.0, "occ_scale": 80.0, "r_scale": 80.0, "theta_scale_deg": 30.0},
        {"name": "geom_soft_pw_k16_L1_O05", "k": 16, "geom": True, "power_sep": True, "los_penalty": 1.0, "occ_penalty": 0.5, "r_penalty": 0.0, "theta_penalty": 0.0, "occ_scale": 80.0, "r_scale": 80.0, "theta_scale_deg": 30.0},
        {"name": "geom_soft_k24_L1_O05_R02_T02", "k": 24, "geom": True, "power_sep": False, "los_penalty": 1.0, "occ_penalty": 0.5, "r_penalty": 0.2, "theta_penalty": 0.2, "occ_scale": 80.0, "r_scale": 80.0, "theta_scale_deg": 30.0},
        {"name": "geom_soft_pw_k24_L1_O05_R02_T02", "k": 24, "geom": True, "power_sep": True, "los_penalty": 1.0, "occ_penalty": 0.5, "r_penalty": 0.2, "theta_penalty": 0.2, "occ_scale": 80.0, "r_scale": 80.0, "theta_scale_deg": 30.0},
        {"name": "geom_two_stage_pw_k16", "k": 16, "geom": True, "power_sep": True, "two_stage": True, "two_stage_occ_window": 50.0, "los_penalty": 1.0, "occ_penalty": 0.5, "r_penalty": 0.0, "theta_penalty": 0.0, "occ_scale": 80.0, "r_scale": 80.0, "theta_scale_deg": 30.0},
    ]

    names = ["v9_affine"]
    for cfg in geom_cfgs:
        names.append(cfg["name"])
        for rho in rhos:
            names.append(f"blend_{cfg['name']}_rho{rho:g}")
    accs = {name: MetricAccumulator("sumpol") for name in names}
    nums = {name: 0.0 for name in names}
    dens = {name: 0.0 for name in names}

    for row, vi in enumerate(val_idx):
        truth = np.asarray(train_h[vi])
        keep = idx_all[row] != vi
        idx = idx_all[row, keep][:max_k]
        d_raw = d_all[row, keep][:max_k]
        rel_xy = pos[idx, :2] - pos[vi, :2]
        coeff = affine_coefficients(rel_xy, d_raw, args.power, args.ridge)
        neigh = np.asarray(train_h[idx])
        v9 = predict_affine(
            neigh,
            coeff,
            dist_all[idx] - dist_all[vi],
            u_all,
            idx,
            vi,
            args.beta,
            args.gamma,
            1.0,
            args.steering_scale,
        )
        add_metric(accs, nums, dens, "v9_affine", v9, truth)

        preds = {}
        for cfg in geom_cfgs:
            if cfg["geom"]:
                gidx, raw_d, eff_d = select_geom_neighbors(pos, train_blocked, pos[vi], train_blocked[vi], vi, cfg)
                d_for_w = eff_d
            else:
                gidx = idx[: cfg["k"]]
                raw_d = d_raw[: cfg["k"]]
                d_for_w = raw_d
            w = idw_weights(d_for_w, args.power)
            gneigh = np.asarray(train_h[gidx])
            pred = predict_phase_idw(
                gneigh,
                w,
                dist_all[gidx] - dist_all[vi],
                args.beta,
                args.gamma,
                logpw if cfg["power_sep"] else None,
                gidx if cfg["power_sep"] else None,
            )
            preds[cfg["name"]] = pred
            add_metric(accs, nums, dens, cfg["name"], pred, truth)
            for rho in rhos:
                blended = ((1.0 - rho) * v9 + rho * pred).astype(np.complex64)
                add_metric(accs, nums, dens, f"blend_{cfg['name']}_rho{rho:g}", blended, truth)

        if (row + 1) % 20 == 0 or row + 1 == len(val_idx):
            print(f"evaluated {row + 1}/{len(val_idx)}", flush=True)

    results = []
    for name in names:
        nmse = nums[name] / max(dens[name], 1e-30)
        row = {"name": name, "global_nmse_direct": float(nmse), **accs[name].result()}
        results.append(row)
    results.sort(key=lambda x: x["score"], reverse=True)
    payload = {
        "args": vars(args),
        "val_indices": val_idx.astype(int).tolist(),
        "train_los_frac": float(np.mean(train_blocked == 0)),
        "val_los_frac": float(np.mean(train_blocked[val_idx] == 0)),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
