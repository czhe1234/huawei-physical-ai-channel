#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_local_ad_paths import (
    BS_POS,
    MP,
    MH,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    flat_components,
    idw_weights,
    inverse_ad,
    parse_floats,
    parse_ints,
    parse_words,
    top_components,
)
from generate_geometric_steering_blend import bs_phv_indices, steering_phase, unit_vectors
from spectrum_projection import MetricAccumulator


TOTAL_COMPONENTS = MH * MV * S_SC


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def eval_one(truth, pred):
    return (
        float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def steering_to_reference(pos, scale):
    u = unit_vectors(pos)
    ant = bs_phv_indices()
    h_coord = ant[:, 0]
    v_coord = ant[:, 1]
    h_val = u[:, 1]
    v_val = -u[:, 2]
    phase_arg = -scale * (h_val[:, None] * h_coord[None, :] + v_val[:, None] * v_coord[None, :])
    return np.exp(1j * phase_arg).astype(np.complex64)


def to_representation(h, pos, rep, beta, steering_scale):
    if rep == "raw":
        return h
    dist = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    out = h * np.exp(1j * beta * dist)[:, None, None, None].astype(np.complex64)
    if rep == "derot":
        return out.astype(np.complex64)
    if rep == "canonical":
        phase = steering_to_reference(pos, steering_scale)
        return (out * phase[:, :, None, None]).astype(np.complex64)
    raise ValueError(rep)


def from_representation(h_rep, pos, rep, beta, steering_scale):
    if rep == "raw":
        return h_rep.astype(np.complex64)
    dist = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    out = h_rep * np.exp(-1j * beta * dist)[:, None, None, None].astype(np.complex64)
    if rep == "derot":
        return out.astype(np.complex64)
    if rep == "canonical":
        phase = steering_to_reference(pos, steering_scale)
        return (out * np.conj(phase)[:, :, None, None]).astype(np.complex64)
    raise ValueError(rep)


def ad_to_flat(ad):
    return ad.transpose(0, 2, 3, 5, 1, 4).reshape(ad.shape[0], TOTAL_COMPONENTS, MP, N_ANT)


def flat_to_ad(flat):
    return flat.reshape(flat.shape[0], MH, MV, S_SC, MP, N_ANT).transpose(0, 4, 1, 2, 5, 3)


def scatter_flat(component_ids, coeff, batch_size):
    flat = np.zeros((batch_size, TOTAL_COMPONENTS, MP, N_ANT), dtype=np.complex64)
    flat[:, component_ids] = coeff.reshape(batch_size, component_ids.size, MP, N_ANT).astype(np.complex64)
    return flat_to_ad(flat)


def gather_flat(ad, component_ids):
    return ad_to_flat(ad)[:, component_ids].reshape(ad.shape[0], -1)


def global_power(channels, pos, train_idx, rep, beta, steering_scale, batch):
    power = None
    for start in range(0, len(train_idx), batch):
        ids = train_idx[start:start + batch]
        h_rep = to_representation(np.asarray(channels[ids]), pos[ids], rep, beta, steering_scale)
        ad = ad_transform(h_rep)
        p = np.sum(np.abs(ad) ** 2, axis=(0, 1, 4), dtype=np.float64)
        power = p if power is None else power + p
        print(f"power {rep} {min(start + batch, len(train_idx))}/{len(train_idx)}", flush=True)
    return power


def make_features(pos, freqs, angle_count, include_radial):
    xy = pos[:, :2].astype(np.float64)
    center = np.array([150.0, -50.0], dtype=np.float64)
    scale = np.array([100.0, 150.0], dtype=np.float64)
    z = (xy - center[None, :]) / scale[None, :]
    feats = [
        np.ones((len(pos), 1), dtype=np.float64),
        z,
        z[:, :1] ** 2,
        z[:, 1:2] ** 2,
        z[:, :1] * z[:, 1:2],
    ]
    if include_radial:
        vec = pos - BS_POS[None, :]
        dist = np.linalg.norm(vec, axis=1, keepdims=True)
        theta = np.arctan2(vec[:, 1:2], vec[:, 0:1])
        feats.extend([
            dist / 250.0,
            np.sin(theta),
            np.cos(theta),
        ])
    for freq in freqs:
        for angle in np.linspace(0.0, np.pi, angle_count, endpoint=False):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            proj = freq * (xy @ direction[:, None])
            feats.append(np.sin(proj))
            feats.append(np.cos(proj))
    x = np.concatenate(feats, axis=1)
    return x.astype(np.float64)


def fit_ridge(x, y, ridge):
    xtx = x.T @ x
    reg = ridge * max(float(np.trace(xtx)) / xtx.shape[0], 1e-12)
    xtx = xtx + reg * np.eye(xtx.shape[0], dtype=np.float64)
    rhs = x.T @ y
    return np.linalg.solve(xtx, rhs)


def full_predict(neigh_h, weights, radial, steer_phase=None):
    coeff = weights.astype(np.float32) * radial.astype(np.complex64)
    if steer_phase is None:
        return np.sum(coeff[:, None, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)
    return np.sum(coeff[:, None, None, None] * steer_phase[:, :, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)


def add_result(acc, nums, name, pred, truth):
    num, den = eval_one(truth, pred)
    nums[name]["num"] += num
    nums[name]["den"] += den
    acc[name].add(pred, truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/global_ad_rff.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--representations", type=parse_words, default="canonical")
    parser.add_argument("--component-counts", type=parse_ints, default="512,1024")
    parser.add_argument("--freqs", type=parse_floats, default="0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28")
    parser.add_argument("--angle-count", type=int, default=12)
    parser.add_argument("--ridges", type=parse_floats, default="0.001,0.01,0.1,1.0")
    parser.add_argument("--include-radial", action="store_true")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    channels = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    x_train = make_features(pos[train_idx], args.freqs, args.angle_count, args.include_radial)
    x_val = make_features(pos[val_idx], args.freqs, args.angle_count, args.include_radial)

    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=args.k, workers=-1)
    if args.k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    results = []
    for rep in args.representations:
        power = global_power(channels, pos, train_idx, rep, args.beta, args.steering_scale, args.batch)
        max_components = max(args.component_counts)
        max_ids = top_components(power, max_components)

        y_parts = []
        for start in range(0, len(train_idx), args.batch):
            ids = train_idx[start:start + args.batch]
            h_rep = to_representation(np.asarray(channels[ids]), pos[ids], rep, args.beta, args.steering_scale)
            ad = ad_transform(h_rep)
            y_parts.append(gather_flat(ad, max_ids))
            print(f"coeff {rep} {min(start + args.batch, len(train_idx))}/{len(train_idx)}", flush=True)
        y_all = np.concatenate(y_parts, axis=0).astype(np.complex64)

        for comp_count in args.component_counts:
            component_ids = max_ids[:comp_count]
            y = y_all[:, :comp_count * MP * N_ANT].astype(np.complex128)
            for ridge in args.ridges:
                coef = fit_ridge(x_train, y, ridge)
                accs = {
                    "rff": MetricAccumulator("sumpol"),
                    "support_truth_oracle": MetricAccumulator("sumpol"),
                    "full_derot": MetricAccumulator("sumpol"),
                    "full_steer": MetricAccumulator("sumpol"),
                    "full_blend05": MetricAccumulator("sumpol"),
                }
                nums = {name: {"num": 0.0, "den": 0.0} for name in accs}
                for start in range(0, len(val_idx), args.batch):
                    end = min(start + args.batch, len(val_idx))
                    ids = val_idx[start:end]
                    truth = np.asarray(channels[ids])
                    coeff_pred = (x_val[start:end] @ coef).astype(np.complex64)
                    ad_pred = scatter_flat(component_ids, coeff_pred, end - start)
                    pred_rep = inverse_ad(ad_pred)
                    pred = from_representation(pred_rep, pos[ids], rep, args.beta, args.steering_scale)
                    add_result(accs, nums, "rff", pred, truth)

                    truth_rep = to_representation(truth, pos[ids], rep, args.beta, args.steering_scale)
                    truth_ad = ad_transform(truth_rep)
                    truth_coeff = gather_flat(truth_ad, component_ids)
                    oracle_ad = scatter_flat(component_ids, truth_coeff, end - start)
                    oracle = from_representation(inverse_ad(oracle_ad), pos[ids], rep, args.beta, args.steering_scale)
                    add_result(accs, nums, "support_truth_oracle", oracle, truth)

                    for local_i, vi in enumerate(ids):
                        idx = neighbor_idx[start + local_i]
                        neigh = np.asarray(channels[idx])
                        w = idw_weights(distances[start + local_i], args.power)
                        radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
                        steer = steering_phase(u_all, idx, vi, args.steering_scale)
                        derot = full_predict(neigh, w, radial)
                        steer_pred = full_predict(neigh, w, radial, steer)
                        blend = (0.5 * derot + 0.5 * steer_pred).astype(np.complex64)
                        add_result(accs, nums, "full_derot", derot, truth[local_i:local_i + 1])
                        add_result(accs, nums, "full_steer", steer_pred, truth[local_i:local_i + 1])
                        add_result(accs, nums, "full_blend05", blend, truth[local_i:local_i + 1])
                    print(
                        f"eval rep={rep} K={comp_count} ridge={ridge:g} {end}/{len(val_idx)}",
                        flush=True,
                    )

                for name, acc in accs.items():
                    row = {
                        "candidate": name,
                        "representation": rep,
                        "component_count": int(comp_count),
                        "ridge": float(ridge),
                        "features": int(x_train.shape[1]),
                        "global_nmse_direct": float(nums[name]["num"] / max(nums[name]["den"], 1e-30)),
                        **acc.result(),
                    }
                    results.append(row)
                    print(json.dumps(row, indent=2), flush=True)

    results.sort(key=lambda r: r["global_nmse"])
    payload = {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "freqs": [float(x) for x in args.freqs],
        "angle_count": int(args.angle_count),
        "include_radial": bool(args.include_radial),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
