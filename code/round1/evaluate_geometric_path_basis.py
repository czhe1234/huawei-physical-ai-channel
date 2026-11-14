#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_wall_planes import canonical_normals
from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_frequency_dependent_phase import S_SC
from evaluate_local_ad_paths import BS_POS, idw_weights, parse_floats
from evaluate_specular_reflection import load_planes
from generate_geometric_steering_blend import bs_phv_indices
from spectrum_projection import MetricAccumulator


MH, MV, MP, N_ANT = 16, 8, 2, 4


def load_plane_subset(path, max_planes):
    raw = load_planes(path, max_planes)
    return raw


def path_params(pos, planes, margin_t, margin_z):
    p = pos.astype(np.float64)
    npos = len(p)
    l_count = len(planes["c"]) + 1
    length = np.zeros((npos, l_count), dtype=np.float64)
    unit = np.zeros((npos, l_count, 3), dtype=np.float64)
    valid = np.zeros((npos, l_count), dtype=bool)

    vec = p - BS_POS[None, :]
    d = np.linalg.norm(vec, axis=1)
    length[:, 0] = d
    unit[:, 0] = vec / np.maximum(d[:, None], 1e-30)
    valid[:, 0] = True

    pxy = p[:, :2]
    pz = p[:, 2]
    bs_xy = BS_POS[:2]
    bs_z = float(BS_POS[2])
    for j in range(len(planes["c"])):
        n = planes["normal"][j]
        c = float(planes["c"][j])
        tangent = planes["tangent"][j]
        mirror_xy = bs_xy - 2.0 * (float(np.dot(n, bs_xy)) - c) * n
        denom = (pxy - mirror_xy[None, :]) @ n
        num = c - float(np.dot(mirror_xy, n))
        tau = num / np.where(np.abs(denom) > 1e-12, denom, np.nan)
        refl_xy = mirror_xy[None, :] + tau[:, None] * (pxy - mirror_xy[None, :])
        refl_z = bs_z + tau * (pz - bs_z)
        t_coord = refl_xy @ tangent
        ok = (
            np.isfinite(tau)
            & (tau > 0.0)
            & (tau < 1.0)
            & (t_coord >= planes["t_min"][j] - margin_t)
            & (t_coord <= planes["t_max"][j] + margin_t)
            & (refl_z >= planes["z_min"][j] - margin_z)
            & (refl_z <= planes["z_max"][j] + margin_z)
        )
        refl = np.column_stack([refl_xy, refl_z])
        first_vec = refl - BS_POS[None, :]
        first = np.linalg.norm(first_vec, axis=1)
        second = np.linalg.norm(p - refl, axis=1)
        length[:, j + 1] = first + second
        unit[:, j + 1] = first_vec / np.maximum(first[:, None], 1e-30)
        valid[:, j + 1] = ok
    return length, unit, valid


def path_basis(length, unit, beta, gamma, steering_scale):
    ant = bs_phv_indices()
    h_coord = ant[:, 0]
    v_coord = ant[:, 1]
    u_h = unit[..., 1]
    u_v = -unit[..., 2]
    spatial = steering_scale * (u_h[..., None] * h_coord[None, :] + u_v[..., None] * v_coord[None, :])
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float64)
    freq = -length[..., None] * (beta + gamma * sub[None, :])
    phase = spatial[..., :, None] + freq[..., None, :]
    return np.exp(1j * phase).astype(np.complex64)


def project_coefficients(h, length, unit, valid, beta, gamma, steering_scale, batch):
    npos, l_count = length.shape
    coeff = np.zeros((npos, l_count, N_ANT), dtype=np.complex64)
    norm = float(MP * MH * MV * S_SC)
    for start in range(0, npos, batch):
        end = min(start + batch, npos)
        hh = np.asarray(h[start:end])
        for l in range(l_count):
            b = path_basis(length[start:end, l], unit[start:end, l], beta, gamma, steering_scale)
            # b: B,M,S. Project each UE antenna independently.
            c = np.sum(np.conj(b[:, :, None, :]) * hh, axis=(1, 3)) / norm
            c[~valid[start:end, l]] = 0.0
            coeff[start:end, l] = c.astype(np.complex64)
        print(f"projected {end}/{npos}", flush=True)
    return coeff


def make_prediction(query_pos_idx, neighbor_idx, distances, coeff, train_path_length, train_path_unit, train_valid,
                    query_length, query_unit, query_valid, beta, gamma, steering_scale, power, top_paths):
    weights = idw_weights(distances, power)
    l_count = query_length.shape[0]
    path_scores = np.zeros(l_count, dtype=np.float64)
    pred_coeff = np.zeros((l_count, N_ANT), dtype=np.complex64)
    for l in range(l_count):
        if not query_valid[l]:
            continue
        mask = train_valid[neighbor_idx, l]
        if not np.any(mask):
            continue
        w = weights[mask]
        w = w / max(float(np.sum(w)), 1e-30)
        vals = coeff[neighbor_idx[mask], l]
        pred_coeff[l] = np.tensordot(w.astype(np.float32), vals, axes=(0, 0)).astype(np.complex64)
        path_scores[l] = float(np.sum(np.abs(pred_coeff[l]) ** 2))
    if top_paths > 0 and top_paths < l_count:
        keep = np.argpartition(path_scores, -top_paths)[-top_paths:]
        mask_keep = np.zeros(l_count, dtype=bool)
        mask_keep[keep] = True
    else:
        mask_keep = path_scores > 0
    pred = np.zeros((MP * MH * MV, N_ANT, S_SC), dtype=np.complex64)
    for l in np.flatnonzero(mask_keep & query_valid):
        b = path_basis(query_length[l:l + 1], query_unit[l:l + 1], beta, gamma, steering_scale)[0]
        pred += b[:, None, :] * pred_coeff[l][None, :, None]
    return pred.astype(np.complex64)


def full_steering_blend(neigh_h, weights, train_dist, query_dist, train_unit, query_unit, beta, gamma, alpha, steering_scale):
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float64)
    beta_s = beta + gamma * sub
    radial = np.exp(1j * (train_dist[:, None] - query_dist) * beta_s[None, :]).astype(np.complex64)
    coeff = weights[:, None].astype(np.float32) * radial
    base = np.sum(coeff[:, None, None, :] * neigh_h, axis=0)
    ant = bs_phv_indices()
    h_coord = ant[:, 0]
    v_coord = ant[:, 1]
    dh = query_unit[1] - train_unit[:, 1]
    dv = -query_unit[2] - (-train_unit[:, 2])
    phase = steering_scale * (dh[:, None] * h_coord[None, :] + dv[:, None] * v_coord[None, :])
    steer = np.exp(1j * phase).astype(np.complex64)
    geo = np.sum(coeff[:, None, None, :] * steer[:, :, None, None] * neigh_h, axis=0)
    return ((1.0 - alpha) * base + alpha * geo).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--planes", default=str(PATHS.analysis / "results/json/wall_planes.json"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/geometric_path_basis.json"))
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--max-planes", type=int, default=40)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--top-paths", type=int, default=8)
    parser.add_argument("--rhos", type=parse_floats, default="0,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--margin-t", type=float, default=4.0)
    parser.add_argument("--margin-z", type=float, default=5.0)
    parser.add_argument("--project-batch", type=int, default=25)
    args = parser.parse_args()

    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    planes = load_plane_subset(args.planes, args.max_planes)
    length, unit, valid = path_params(pos, planes, args.margin_t, args.margin_z)
    coeff = project_coefficients(h, length, unit, valid, args.beta, args.gamma, args.steering_scale, args.project_batch)

    tree = cKDTree(pos[:, :2])
    d_all, local_all = tree.query(pos[val_idx, :2], k=args.k + 1, workers=-1)
    train_dist = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    train_unit = (pos - BS_POS[None, :]) / np.maximum(train_dist[:, None], 1e-30)
    accs = {rho: MetricAccumulator("sumpol") for rho in args.rhos}
    nums = {rho: 0.0 for rho in args.rhos}
    dens = {rho: 0.0 for rho in args.rhos}

    for row, vi in enumerate(val_idx):
        keep = local_all[row] != vi
        idx = local_all[row, keep][:args.k]
        dist = d_all[row, keep][:args.k]
        weights = idw_weights(dist, args.power)
        truth = np.asarray(h[vi])
        den = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        direct = full_steering_blend(
            np.asarray(h[idx]),
            weights,
            train_dist[idx],
            train_dist[vi],
            train_unit[idx],
            train_unit[vi],
            args.beta,
            args.gamma,
            0.75,
            args.steering_scale,
        )
        path_pred = make_prediction(
            vi,
            idx,
            dist,
            coeff,
            length,
            unit,
            valid,
            length[vi],
            unit[vi],
            valid[vi],
            args.beta,
            args.gamma,
            args.steering_scale,
            args.power,
            args.top_paths,
        )
        for rho in args.rhos:
            pred = ((1.0 - rho) * direct + rho * path_pred).astype(np.complex64)
            nums[rho] += float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
            dens[rho] += den
            accs[rho].add(pred[None], truth[None])
        if (row + 1) % 10 == 0 or row + 1 == len(val_idx):
            print(f"evaluated {row + 1}/{len(val_idx)}", flush=True)

    results = []
    for rho in args.rhos:
        nmse = nums[rho] / max(dens[rho], 1e-30)
        results.append({
            "rho": float(rho),
            "max_planes": int(args.max_planes),
            "top_paths": int(args.top_paths),
            "global_nmse_direct": float(nmse),
            **accs[rho].result(),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    payload = {
        "seed": int(args.seed),
        "val_mode": args.val_mode,
        "val_count": int(len(val_idx)),
        "gamma": float(args.gamma),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
