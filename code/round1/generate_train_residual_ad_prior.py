#!/usr/bin/env python3
"""Generate v58: train-derived angle-delay residual prior.

This candidate removes the external ``s1 - v9`` residual.  It estimates the
missing correction from the training set itself:

    residual_train[i] = H_train[i] - v36_LOO[i]

The residuals are carrier/frequency de-rotated, transformed to BS angle-delay
space, and cached.  For each test point we select geometrically similar
training residuals, choose dominant residual components by weighted component
energy, interpolate only those complex coefficients, rotate back, and add the
predicted residual to the v36 base.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_map_features import BS_POS
from evaluate_frequency_dependent_phase import S_SC
from evaluate_local_ad_paths import MH, MP, MV, N_ANT, ad_transform, idw_weights, inverse_ad
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_s1style_geom_blend import geometry_cache, predict_phase_idw, select_geom_neighbors, train_log_power
from generate_geometric_steering_blend import unit_vectors
from generate_regime_s1style_geom_blend import rho_for_blocked, validate_shapes


AD_SHAPE = (MP, MH, MV, N_ANT, S_SC)
COMP_SHAPE = (MH, MV, S_SC)


def regime_label(blocked: int) -> str:
    if blocked <= 0:
        return "clear"
    if blocked <= 25:
        return "light"
    if blocked <= 100:
        return "medium"
    return "heavy"


def beta_subcarriers(beta: float, gamma: float) -> np.ndarray:
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    return (float(beta) + float(gamma) * sub).astype(np.float32)


def radial_phase(distance: float, beta_s: np.ndarray, sign: float) -> np.ndarray:
    return np.exp(1j * sign * float(distance) * beta_s).astype(np.complex64)


def flat_to_hvs(component_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(component_ids, dtype=np.int64)
    h = ids // (MV * S_SC)
    rem = ids % (MV * S_SC)
    v = rem // S_SC
    s = rem % S_SC
    return h.astype(np.int64), v.astype(np.int64), s.astype(np.int64)


def top_components(score: np.ndarray, count: int) -> np.ndarray:
    flat = np.asarray(score).reshape(-1)
    k = min(int(count), flat.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    ids = np.argpartition(flat, -k)[-k:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    return ids.astype(np.int64)


def power_preserve(delta: np.ndarray, reference_power: float) -> np.ndarray:
    cur = float(np.sum(np.abs(delta) ** 2, dtype=np.float64))
    if cur <= 1e-30 or reference_power <= 1e-30:
        return delta
    return (delta * np.float32(np.sqrt(reference_power / cur))).astype(np.complex64)


def make_geom_cfg(args, k: int | None = None) -> dict:
    return {
        "k": int(k if k is not None else args.geom_k),
        "los_penalty": float(args.los_penalty),
        "occ_penalty": float(args.occ_penalty),
        "r_penalty": float(args.r_penalty),
        "theta_penalty": float(args.theta_penalty),
        "occ_scale": float(args.occ_scale),
        "r_scale": float(args.r_scale),
        "theta_scale_deg": float(args.theta_scale_deg),
    }


def v36_rho(blocked: int, args) -> tuple[float, str]:
    return rho_for_blocked(
        int(blocked),
        args.base_clear_rho,
        args.base_light_rho,
        args.base_medium_rho,
        args.base_heavy_rho,
    )


def predict_v36_loo(
    vi: int,
    pos: np.ndarray,
    train_h,
    tree: cKDTree,
    train_blocked: np.ndarray,
    logpw: np.ndarray,
    dist_all: np.ndarray,
    u_all: np.ndarray,
    args,
    geom_cfg: dict,
) -> np.ndarray:
    d_raw, idx_raw = tree.query(pos[vi, :2], k=args.k + 1, workers=-1)
    idx_raw = np.atleast_1d(idx_raw)
    d_raw = np.atleast_1d(d_raw)
    keep = idx_raw != vi
    idx = idx_raw[keep][: args.k]
    distances = d_raw[keep][: args.k]
    rel_xy = pos[idx, :2] - pos[vi, :2]
    coeff = affine_coefficients(rel_xy, distances, args.power, args.ridge)
    v9 = predict_affine(
        np.asarray(train_h[idx]),
        coeff,
        dist_all[idx] - dist_all[vi],
        u_all,
        idx,
        vi,
        args.beta,
        args.gamma,
        args.alpha,
        args.steering_scale,
    )

    gidx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, pos[vi], train_blocked[vi], vi, geom_cfg)
    weights = idw_weights(eff_d, args.power)
    geom = predict_phase_idw(
        np.asarray(train_h[gidx]),
        weights,
        dist_all[gidx] - dist_all[vi],
        args.beta,
        args.gamma,
        logpw if args.geom_power_sep else None,
        gidx if args.geom_power_sep else None,
    )
    rho, _ = v36_rho(int(train_blocked[vi]), args)
    return ((1.0 - rho) * v9 + rho * geom).astype(np.complex64)


def build_residual_cache(args, pos, train_h, train_blocked, logpw, dist_all, u_all) -> dict:
    cache_ad = Path(args.residual_ad_cache)
    cache_power = Path(args.residual_power_cache)
    meta_path = Path(args.residual_meta)
    if cache_ad.exists() and cache_power.exists() and meta_path.exists() and not args.force_residual:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    cache_ad.parent.mkdir(parents=True, exist_ok=True)
    cache_power.parent.mkdir(parents=True, exist_ok=True)
    tree = cKDTree(pos[:, :2])
    geom_cfg = make_geom_cfg(args, args.geom_k)
    beta_s = beta_subcarriers(args.beta, args.gamma)

    res_ad = np.lib.format.open_memmap(
        cache_ad,
        mode="w+",
        dtype=np.complex64,
        shape=(len(pos),) + AD_SHAPE,
    )
    res_power = np.lib.format.open_memmap(
        cache_power,
        mode="w+",
        dtype=np.float32,
        shape=(len(pos),) + COMP_SHAPE,
    )
    stats = {
        "n_train": int(len(pos)),
        "residual_power_sum": 0.0,
        "truth_power_sum": 0.0,
        "base_power_sum": 0.0,
        "regime_counts": {"clear": 0, "light": 0, "medium": 0, "heavy": 0},
        "method": "v36_loo_true_residual_derotated_ad",
    }

    for i in range(len(pos)):
        truth = np.asarray(train_h[i], dtype=np.complex64)
        base = predict_v36_loo(i, pos, train_h, tree, train_blocked, logpw, dist_all, u_all, args, geom_cfg)
        residual = (truth - base).astype(np.complex64)
        phase = radial_phase(dist_all[i], beta_s, +1.0)
        residual_derot = (residual * phase[None, None, :]).astype(np.complex64)
        ad = ad_transform(residual_derot[None])[0]
        res_ad[i] = ad
        res_power[i] = np.sum(np.abs(ad) ** 2, axis=(0, 3), dtype=np.float64).astype(np.float32)

        label = regime_label(int(train_blocked[i]))
        stats["regime_counts"][label] += 1
        stats["residual_power_sum"] += float(np.sum(np.abs(residual) ** 2, dtype=np.float64))
        stats["truth_power_sum"] += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        stats["base_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))

        if (i + 1) % args.cache_flush == 0 or i + 1 == len(pos):
            res_ad.flush()
            res_power.flush()
            print(f"cached train residual {i + 1}/{len(pos)}", flush=True)

    stats["residual_to_truth_power"] = stats["residual_power_sum"] / max(stats["truth_power_sum"], 1e-30)
    stats["base_to_truth_power"] = stats["base_power_sum"] / max(stats["truth_power_sum"], 1e-30)
    payload = {
        "cache_ad": str(cache_ad),
        "cache_power": str(cache_power),
        "shape_ad": [int(x) for x in res_ad.shape],
        "shape_power": [int(x) for x in res_power.shape],
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "base_rhos": {
            "clear": float(args.base_clear_rho),
            "light": float(args.base_light_rho),
            "medium": float(args.base_medium_rho),
            "heavy": float(args.base_heavy_rho),
        },
        "stats": stats,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def gain_for_regime(blocked: int, args) -> tuple[float, str]:
    label = regime_label(int(blocked))
    gains = {
        "clear": float(args.clear_gain),
        "light": float(args.light_gain),
        "medium": float(args.medium_gain),
        "heavy": float(args.heavy_gain),
    }
    return gains[label], label


def predict_residual_from_cache(
    q_pos: np.ndarray,
    q_blocked: int,
    pos: np.ndarray,
    train_blocked: np.ndarray,
    res_ad,
    res_power,
    dist_q: float,
    beta_s: np.ndarray,
    args,
) -> tuple[np.ndarray, dict]:
    geom_cfg = make_geom_cfg(args, args.residual_k)
    idx, _raw_d, eff_d = select_geom_neighbors(pos, train_blocked, q_pos, q_blocked, None, geom_cfg)
    weights = idw_weights(eff_d, args.residual_power)
    neigh_power = np.asarray(res_power[idx])
    score = np.tensordot(weights.astype(np.float32), neigh_power, axes=(0, 0))
    comp_ids = top_components(score, args.components)
    hs, vs, ss = flat_to_hvs(comp_ids)

    pred_ad = np.zeros((1,) + AD_SHAPE, dtype=np.complex64)
    selected_score_sum = float(np.sum(score.reshape(-1)[comp_ids], dtype=np.float64)) if comp_ids.size else 0.0
    total_score_sum = float(np.sum(score, dtype=np.float64))
    for h, v, s in zip(hs, vs, ss):
        coeff = np.zeros((MP, N_ANT), dtype=np.complex64)
        for w, ridx in zip(weights, idx):
            coeff += np.float32(w) * np.asarray(res_ad[int(ridx), :, int(h), int(v), :, int(s)])
        pred_ad[0, :, int(h), int(v), :, int(s)] = coeff

    residual = inverse_ad(pred_ad)[0]
    phase = radial_phase(dist_q, beta_s, -1.0)
    residual = (residual * phase[None, None, :]).astype(np.complex64)
    return residual, {
        "neighbor_idx": idx,
        "weights": weights,
        "selected_component_energy_frac": selected_score_sum / max(total_score_sum, 1e-30),
        "selected_components": int(comp_ids.size),
    }


def generate(args) -> dict:
    data = Path(args.data)
    pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    train_h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    validate_shapes(pos, test_pos, train_h)

    v36 = np.load(args.v36, mmap_mode="r")
    if v36.shape != (len(test_pos), 256, 4, 192):
        raise ValueError(f"bad v36 shape {v36.shape}")

    train_blocked, test_blocked = geometry_cache(args.data, args.cache, args.force_geometry)
    logpw = train_log_power(train_h)
    dist_train = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_train = unit_vectors(pos)
    cache_meta = build_residual_cache(args, pos, train_h, train_blocked, logpw, dist_train, u_train)

    res_ad = np.load(args.residual_ad_cache, mmap_mode="r")
    res_power = np.load(args.residual_power_cache, mmap_mode="r")
    beta_s = beta_subcarriers(args.beta, args.gamma)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=v36.shape)
    stats = {
        "regime_counts": {"clear": 0, "light": 0, "medium": 0, "heavy": 0},
        "gain_sum": 0.0,
        "base_power_sum": 0.0,
        "residual_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "selected_component_energy_frac_sum": 0.0,
    }

    for start in range(0, len(test_pos), args.batch):
        end = min(start + args.batch, len(test_pos))
        for i in range(start, end):
            base = np.asarray(v36[i], dtype=np.complex64)
            residual, meta = predict_residual_from_cache(
                test_pos[i],
                int(test_blocked[i]),
                pos,
                train_blocked,
                res_ad,
                res_power,
                float(dist_test[i]),
                beta_s,
                args,
            )
            if args.preserve_train_residual_power:
                neigh_idx = meta["neighbor_idx"]
                weights = meta["weights"]
                target_power = float(np.dot(weights.astype(np.float64), np.sum(np.asarray(res_power[neigh_idx]), axis=(1, 2, 3), dtype=np.float64)))
                residual = power_preserve(residual, target_power)

            gain, label = gain_for_regime(int(test_blocked[i]), args)
            pred = (base + np.float32(gain) * residual).astype(np.complex64)
            if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
                raise FloatingPointError(f"non-finite prediction at {i}")
            out[i] = pred

            stats["regime_counts"][label] += 1
            stats["gain_sum"] += float(gain)
            stats["base_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
            stats["residual_power_sum"] += float(np.sum(np.abs(residual) ** 2, dtype=np.float64))
            stats["pred_power_sum"] += float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
            stats["selected_component_energy_frac_sum"] += float(meta["selected_component_energy_frac"])

        out.flush()
        block = np.asarray(out[start:end])
        if not (np.isfinite(block.real).all() and np.isfinite(block.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        print(f"generated {end}/{len(test_pos)}", flush=True)

    stats["gain_mean"] = stats["gain_sum"] / max(len(test_pos), 1)
    stats["residual_to_base_power"] = stats["residual_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_to_base_power"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["selected_component_energy_frac_mean"] = stats["selected_component_energy_frac_sum"] / max(len(test_pos), 1)

    payload = {
        "method": "train_residual_ad_prior",
        "version": str(args.version),
        "formula": "v36 + gain(regime) * inverse_ad(weighted top-component AD residual prior from H_train - v36_LOO)",
        "uses_external_s1": False,
        "v36": args.v36,
        "output": args.output,
        "residual_cache": cache_meta,
        "prediction": {
            "components": int(args.components),
            "residual_k": int(args.residual_k),
            "residual_power": float(args.residual_power),
            "preserve_train_residual_power": bool(args.preserve_train_residual_power),
            "gains": {
                "clear": float(args.clear_gain),
                "light": float(args.light_gain),
                "medium": float(args.medium_gain),
                "heavy": float(args.heavy_gain),
            },
        },
        "stats": stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(58) / "train_residual_ad_prior_v58_generate.json"))
    parser.add_argument("--version", default="v58")
    parser.add_argument("--residual-ad-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad.npy"))
    parser.add_argument("--residual-power-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad_power.npy"))
    parser.add_argument("--residual-meta", default=str(PATHS.analysis / "results/json/train_v36_loo_residual_derot_ad_meta.json"))
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-residual", action="store_true")
    parser.add_argument("--cache-flush", type=int, default=25)
    parser.add_argument("--batch", type=int, default=5)

    # v36 base parameters.
    parser.add_argument("--base-clear-rho", type=float, default=0.0)
    parser.add_argument("--base-light-rho", type=float, default=0.0)
    parser.add_argument("--base-medium-rho", type=float, default=1.0)
    parser.add_argument("--base-heavy-rho", type=float, default=4.0)
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

    # Residual prior prediction parameters.
    parser.add_argument("--residual-k", type=int, default=32)
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--clear-gain", type=float, default=0.25)
    parser.add_argument("--light-gain", type=float, default=0.25)
    parser.add_argument("--medium-gain", type=float, default=0.75)
    parser.add_argument("--heavy-gain", type=float, default=1.50)
    parser.add_argument("--preserve-train-residual-power", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = generate(args)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
