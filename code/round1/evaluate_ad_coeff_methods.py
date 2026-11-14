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
    select_components,
)
from generate_geometric_steering_blend import steering_phase, unit_vectors
from spectrum_projection import MetricAccumulator


TOTAL_COMPONENTS = MH * MV * S_SC


def build_split(n, val_count, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def eval_one(truth, pred):
    diff = pred - truth
    return (
        float(np.sum(np.abs(diff) ** 2, dtype=np.float64)),
        float(np.sum(np.abs(truth) ** 2, dtype=np.float64)),
    )


def full_predict(neigh_h, weights, radial, steer_phase=None):
    coeff = weights.astype(np.float32) * radial.astype(np.complex64)
    if steer_phase is None:
        return np.sum(coeff[:, None, None, None] * neigh_h, axis=0, keepdims=True).astype(np.complex64)
    return np.sum(
        coeff[:, None, None, None] * steer_phase[:, :, None, None] * neigh_h,
        axis=0,
        keepdims=True,
    ).astype(np.complex64)


def ad_to_flat(ad):
    return ad.transpose(0, 2, 3, 5, 1, 4).reshape(ad.shape[0], TOTAL_COMPONENTS, MP, N_ANT)


def flat_to_ad(flat):
    return flat.reshape(flat.shape[0], MH, MV, S_SC, MP, N_ANT).transpose(0, 4, 1, 2, 5, 3)


def gather_coeff(ad, component_ids):
    flat = ad_to_flat(ad)
    return flat[:, component_ids]


def scatter_coeff(component_ids, coeff):
    flat = np.zeros((1, TOTAL_COMPONENTS, MP, N_ANT), dtype=np.complex64)
    flat[0, component_ids] = coeff.astype(np.complex64)
    return flat_to_ad(flat)


def unit_complex(x):
    mag = np.abs(x)
    out = np.ones_like(x, dtype=np.complex64)
    good = mag > 1e-30
    out[good] = (x[good] / mag[good]).astype(np.complex64)
    return out


def complex_idw(coeff, weights):
    return np.tensordot(weights.astype(np.float32), coeff, axes=(0, 0)).astype(np.complex64)


def nearest_coeff(coeff, weights):
    del weights
    return coeff[0].astype(np.complex64)


def magnitude_idw_phase_nearest(coeff, weights):
    amp = np.tensordot(weights.astype(np.float32), np.abs(coeff), axes=(0, 0))
    return (amp * unit_complex(coeff[0])).astype(np.complex64)


def magnitude_idw_phase_idw(coeff, weights):
    pred = complex_idw(coeff, weights)
    amp = np.tensordot(weights.astype(np.float32), np.abs(coeff), axes=(0, 0))
    return (amp * unit_complex(pred)).astype(np.complex64)


def align_to_reference(coeff, weights, reference):
    del weights
    out = coeff.copy()
    ref = reference.reshape(reference.shape[0], -1)
    cur = coeff.reshape(coeff.shape[0], coeff.shape[1], -1)
    for j in range(1, coeff.shape[0]):
        corr = np.sum(cur[j] * np.conj(ref), axis=1)
        factor = np.ones_like(corr, dtype=np.complex64)
        good = np.abs(corr) > 1e-30
        # cur * conj(cur/ref phase) rotates each component vector toward the reference.
        factor[good] = np.conj(corr[good] / np.abs(corr[good])).astype(np.complex64)
        out[j] *= factor[:, None, None]
    return out


def aligned_nearest_idw(coeff, weights):
    aligned = align_to_reference(coeff, weights, coeff[0])
    return complex_idw(aligned, weights)


def aligned_iter_idw(coeff, weights, steps=2):
    pred = complex_idw(coeff, weights)
    aligned = coeff
    for _ in range(steps):
        aligned = align_to_reference(coeff, weights, pred)
        pred = complex_idw(aligned, weights)
    return pred.astype(np.complex64)


def local_linear_predict(coeff, rel_pos, weights, ridge):
    k = coeff.shape[0]
    y = coeff.reshape(k, -1).astype(np.complex128)
    x = np.concatenate([np.ones((k, 1), dtype=np.float64), rel_pos.astype(np.float64)], axis=1)
    sw = np.sqrt(weights.astype(np.float64))[:, None]
    xw = x * sw
    yw = y * sw
    xtx = xw.T @ xw
    scale = max(float(np.trace(xtx)) / xtx.shape[0], 1e-12)
    reg = np.eye(xtx.shape[0], dtype=np.float64) * (ridge * scale)
    reg[0, 0] = 0.0
    rhs = xw.T @ yw
    try:
        coef = np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx + reg) @ rhs
    return coef[0].reshape(coeff.shape[1], coeff.shape[2], coeff.shape[3]).astype(np.complex64)


def local_linear_complex(coeff, weights, rel_pos, ridge):
    return local_linear_predict(coeff, rel_pos, weights, ridge)


def local_linear_phase(coeff, weights, rel_pos, ridge):
    amp = np.tensordot(weights.astype(np.float32), np.abs(coeff), axes=(0, 0))
    phase = local_linear_predict(unit_complex(coeff), rel_pos, weights, ridge)
    return (amp * unit_complex(phase)).astype(np.complex64)


def build_gradients(magnitudes, angle_count):
    grads = [(0.0, 0.0)]
    for mag in magnitudes:
        if abs(mag) < 1e-12:
            continue
        for angle in np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False):
            grads.append((mag * np.cos(angle), mag * np.sin(angle)))
    return np.asarray(grads, dtype=np.float32)


def gradient_search_predict(coeff, weights, rel_pos, gradients, batch_components):
    rel_xy = rel_pos[:, :2].astype(np.float32)
    phase = np.exp(-1j * (gradients @ rel_xy.T)).astype(np.complex64)
    mix = phase * weights.astype(np.float32)[None, :]
    flat = coeff.reshape(coeff.shape[0], coeff.shape[1], -1)
    out = np.empty((coeff.shape[1], flat.shape[2]), dtype=np.complex64)
    for start in range(0, coeff.shape[1], batch_components):
        end = min(start + batch_components, coeff.shape[1])
        pred = np.einsum("gk,kld->gld", mix, flat[:, start:end], optimize=True)
        score = np.sum(np.abs(pred) ** 2, axis=2)
        best = np.argmax(score, axis=0)
        out[start:end] = pred[best, np.arange(end - start)]
    return out.reshape(coeff.shape[1], coeff.shape[2], coeff.shape[3]).astype(np.complex64)


def gradient_search_oracle(coeff, weights, rel_pos, gradients, batch_components, truth):
    rel_xy = rel_pos[:, :2].astype(np.float32)
    phase = np.exp(-1j * (gradients @ rel_xy.T)).astype(np.complex64)
    mix = phase * weights.astype(np.float32)[None, :]
    flat = coeff.reshape(coeff.shape[0], coeff.shape[1], -1)
    truth_flat = truth.reshape(truth.shape[0], -1)
    out = np.empty((coeff.shape[1], flat.shape[2]), dtype=np.complex64)
    for start in range(0, coeff.shape[1], batch_components):
        end = min(start + batch_components, coeff.shape[1])
        pred = np.einsum("gk,kld->gld", mix, flat[:, start:end], optimize=True)
        err = np.sum(np.abs(pred - truth_flat[None, start:end]) ** 2, axis=2)
        best = np.argmin(err, axis=0)
        out[start:end] = pred[best, np.arange(end - start)]
    return out.reshape(coeff.shape[1], coeff.shape[2], coeff.shape[3]).astype(np.complex64)


def phase_oracle(pred, truth):
    return (np.abs(pred) * unit_complex(truth)).astype(np.complex64)


def magnitude_oracle(pred, truth):
    return (np.abs(truth) * unit_complex(pred)).astype(np.complex64)


def common_phase_oracle(pred, truth):
    p = pred.reshape(pred.shape[0], -1)
    t = truth.reshape(truth.shape[0], -1)
    corr = np.sum(t * np.conj(p), axis=1)
    factor = np.ones_like(corr, dtype=np.complex64)
    good = np.abs(corr) > 1e-30
    factor[good] = (corr[good] / np.abs(corr[good])).astype(np.complex64)
    return (pred * factor[:, None, None]).astype(np.complex64)


def complex_scalar_oracle(pred, truth):
    p = pred.reshape(pred.shape[0], -1)
    t = truth.reshape(truth.shape[0], -1)
    den = np.sum(np.abs(p) ** 2, axis=1)
    num = np.sum(t * np.conj(p), axis=1)
    factor = np.zeros_like(num, dtype=np.complex64)
    good = den > 1e-30
    factor[good] = (num[good] / den[good]).astype(np.complex64)
    return (pred * factor[:, None, None]).astype(np.complex64)


def component_scalar(pred, target):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    den = np.sum(np.abs(p) ** 2, axis=1)
    num = np.sum(t * np.conj(p), axis=1)
    factor = np.ones_like(num, dtype=np.complex64)
    good = den > 1e-30
    factor[good] = (num[good] / den[good]).astype(np.complex64)
    return factor


def loo_scalar_predict(coeff, weights, rel_pos, power, mode, clip_abs):
    base = complex_idw(coeff, weights)
    if coeff.shape[0] < 3:
        return base
    scalars = np.empty((coeff.shape[0], coeff.shape[1]), dtype=np.complex64)
    all_ids = np.arange(coeff.shape[0])
    for j in range(coeff.shape[0]):
        mask = all_ids != j
        d = np.linalg.norm(rel_pos[mask] - rel_pos[j], axis=1)
        wj = idw_weights(d, power)
        pred_j = complex_idw(coeff[mask], wj)
        scalars[j] = component_scalar(pred_j, coeff[j])

    if mode == "nearest":
        scalar = scalars[0]
    elif mode == "idw":
        scalar = np.tensordot(weights.astype(np.float32), scalars, axes=(0, 0)).astype(np.complex64)
    elif mode == "phase_idw":
        scalar = unit_complex(np.tensordot(weights.astype(np.float32), scalars, axes=(0, 0)).astype(np.complex64))
    elif mode == "phase_nearest":
        scalar = unit_complex(scalars[0])
    else:
        raise ValueError(mode)

    if clip_abs > 0.0 and mode in {"nearest", "idw"}:
        amp = np.clip(np.abs(scalar), 1.0 / clip_abs, clip_abs)
        scalar = (amp * unit_complex(scalar)).astype(np.complex64)
    return (base * scalar[:, None, None]).astype(np.complex64)


def make_sparse_pred(component_ids, coeff_pred):
    return inverse_ad(scatter_coeff(component_ids, coeff_pred))


def add_result(stats, accs, key, pred, truth):
    num, den = eval_one(truth, pred)
    stats[key]["num"] += num
    stats[key]["den"] += den
    accs[key].add(pred, truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/ad_coeff_methods.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--k-values", type=parse_ints, default="8,16,24")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--component-counts", type=parse_ints, default="512,1024,4096")
    parser.add_argument("--mode", default="weighted")
    parser.add_argument("--per-neighbor", type=int, default=256)
    parser.add_argument("--linear-ridges", type=parse_floats, default="0.0,0.001,0.01")
    parser.add_argument("--gradient-magnitudes", type=parse_floats, default="0,40,80,120,160,200,240,280")
    parser.add_argument("--gradient-angles", type=int, default=32)
    parser.add_argument("--gradient-batch-components", type=int, default=512)
    parser.add_argument(
        "--methods",
        type=parse_words,
        default="complex_idw,nearest,mag_idw_phase_nearest,mag_idw_phase_idw,aligned_nearest,aligned_iter,linear_complex,linear_phase",
    )
    args = parser.parse_args()
    gradients = build_gradients(args.gradient_magnitudes, args.gradient_angles)

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    train_idx, val_idx = build_split(len(pos), args.val_count, args.seed)
    max_k = max(args.k_values)
    tree = cKDTree(pos[train_idx, :2])
    distances, local = tree.query(pos[val_idx, :2], k=max_k, workers=-1)
    if max_k == 1:
        distances = distances[:, None]
        local = local[:, None]
    neighbor_idx = train_idx[local]
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)

    candidates = []
    for k in args.k_values:
        candidates.extend([
            ("full_derot", k, 0, "full", False, 0.0),
            ("full_steer", k, 0, "full", False, 0.0),
            ("full_blend05", k, 0, "full", False, 0.0),
        ])
        for comp_count in args.component_counts:
            for method in args.methods:
                if method.startswith("linear_"):
                    for ridge in args.linear_ridges:
                        candidates.append((method, k, comp_count, args.mode, False, ridge))
                else:
                    candidates.append((method, k, comp_count, args.mode, False, 0.0))
            for oracle_name in [
                "support_truth_oracle",
                "gradient_search_oracle",
                "complex_idw_phase_oracle",
                "complex_idw_magnitude_oracle",
                "complex_idw_common_phase_oracle",
                "complex_idw_scalar_oracle",
            ]:
                candidates.append((oracle_name, k, comp_count, args.mode, True, 0.0))

    stats = {str(c): {"num": 0.0, "den": 0.0} for c in candidates}
    accs = {str(c): MetricAccumulator("sumpol") for c in candidates}

    method_fns = {
        "complex_idw": lambda coeff, w, rel, ridge: complex_idw(coeff, w),
        "nearest": lambda coeff, w, rel, ridge: nearest_coeff(coeff, w),
        "mag_idw_phase_nearest": lambda coeff, w, rel, ridge: magnitude_idw_phase_nearest(coeff, w),
        "mag_idw_phase_idw": lambda coeff, w, rel, ridge: magnitude_idw_phase_idw(coeff, w),
        "aligned_nearest": lambda coeff, w, rel, ridge: aligned_nearest_idw(coeff, w),
        "aligned_iter": lambda coeff, w, rel, ridge: aligned_iter_idw(coeff, w),
        "linear_complex": lambda coeff, w, rel, ridge: local_linear_complex(coeff, w, rel, ridge),
        "linear_phase": lambda coeff, w, rel, ridge: local_linear_phase(coeff, w, rel, ridge),
        "gradient_search": lambda coeff, w, rel, ridge: gradient_search_predict(
            coeff,
            w,
            rel,
            gradients,
            args.gradient_batch_components,
        ),
        "loo_scalar_idw": lambda coeff, w, rel, ridge: loo_scalar_predict(
            coeff,
            w,
            rel,
            args.power,
            "idw",
            4.0,
        ),
        "loo_scalar_nearest": lambda coeff, w, rel, ridge: loo_scalar_predict(
            coeff,
            w,
            rel,
            args.power,
            "nearest",
            4.0,
        ),
        "loo_scalar_phase_idw": lambda coeff, w, rel, ridge: loo_scalar_predict(
            coeff,
            w,
            rel,
            args.power,
            "phase_idw",
            0.0,
        ),
        "loo_scalar_phase_nearest": lambda coeff, w, rel, ridge: loo_scalar_predict(
            coeff,
            w,
            rel,
            args.power,
            "phase_nearest",
            0.0,
        ),
    }

    for ii, vi in enumerate(val_idx):
        truth = np.asarray(h[vi:vi + 1])
        truth_ad = ad_transform(truth)
        truth_flat = ad_to_flat(truth_ad)

        for k in args.k_values:
            idx = neighbor_idx[ii, :k]
            neigh_h = np.asarray(h[idx])
            w = idw_weights(distances[ii, :k], args.power)
            radial = np.exp(1j * args.beta * (dist_all[idx] - dist_all[vi])).astype(np.complex64)
            steer = steering_phase(u_all, idx, vi, args.steering_scale)

            full_derot = full_predict(neigh_h, w, radial)
            full_steer = full_predict(neigh_h, w, radial, steer)
            full_blend = (0.5 * full_derot + 0.5 * full_steer).astype(np.complex64)
            add_result(stats, accs, str(("full_derot", k, 0, "full", False, 0.0)), full_derot, truth)
            add_result(stats, accs, str(("full_steer", k, 0, "full", False, 0.0)), full_steer, truth)
            add_result(stats, accs, str(("full_blend05", k, 0, "full", False, 0.0)), full_blend, truth)

            corrected = neigh_h * radial[:, None, None, None] * steer[:, :, None, None]
            corrected_ad = ad_transform(corrected)
            corrected_power = np.sum(np.abs(corrected_ad) ** 2, axis=(1, 4), dtype=np.float64)
            rel_pos = pos[idx] - pos[vi]

            for comp_count in args.component_counts:
                component_ids = select_components(corrected_power, w, args.mode, args.per_neighbor, comp_count)
                coeff = gather_coeff(corrected_ad, component_ids)
                truth_coeff = truth_flat[0, component_ids]

                pred_cache = {}
                for method in args.methods:
                    if method.startswith("linear_"):
                        for ridge in args.linear_ridges:
                            pred_coeff = method_fns[method](coeff, w, rel_pos, ridge)
                            pred_cache[(method, ridge)] = pred_coeff
                            pred = make_sparse_pred(component_ids, pred_coeff)
                            key = str((method, k, comp_count, args.mode, False, ridge))
                            add_result(stats, accs, key, pred, truth)
                    else:
                        pred_coeff = method_fns[method](coeff, w, rel_pos, 0.0)
                        pred_cache[(method, 0.0)] = pred_coeff
                        pred = make_sparse_pred(component_ids, pred_coeff)
                        key = str((method, k, comp_count, args.mode, False, 0.0))
                        add_result(stats, accs, key, pred, truth)

                complex_pred = pred_cache[("complex_idw", 0.0)]
                oracle_map = {
                    "support_truth_oracle": truth_coeff,
                    "gradient_search_oracle": gradient_search_oracle(
                        coeff,
                        w,
                        rel_pos,
                        gradients,
                        args.gradient_batch_components,
                        truth_coeff,
                    ),
                    "complex_idw_phase_oracle": phase_oracle(complex_pred, truth_coeff),
                    "complex_idw_magnitude_oracle": magnitude_oracle(complex_pred, truth_coeff),
                    "complex_idw_common_phase_oracle": common_phase_oracle(complex_pred, truth_coeff),
                    "complex_idw_scalar_oracle": complex_scalar_oracle(complex_pred, truth_coeff),
                }
                for oracle_name, oracle_coeff in oracle_map.items():
                    pred = make_sparse_pred(component_ids, oracle_coeff)
                    key = str((oracle_name, k, comp_count, args.mode, True, 0.0))
                    add_result(stats, accs, key, pred, truth)

        if (ii + 1) % 10 == 0 or ii + 1 == len(val_idx):
            print(f"evaluated {ii + 1}/{len(val_idx)}", flush=True)

    results = []
    for c in candidates:
        key = str(c)
        nmse = stats[key]["num"] / max(stats[key]["den"], 1e-30)
        results.append({
            "candidate": c[0],
            "k": int(c[1]),
            "component_count": int(c[2]),
            "mode": c[3],
            "oracle": bool(c[4]),
            "ridge": float(c[5]),
            "global_nmse_direct": float(nmse),
            **accs[key].result(),
        })
    results.sort(key=lambda r: r["global_nmse"])
    payload = {
        "seed": int(args.seed),
        "val_count": int(args.val_count),
        "power": float(args.power),
        "beta": float(args.beta),
        "steering_scale": float(args.steering_scale),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    for row in results[:50]:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
