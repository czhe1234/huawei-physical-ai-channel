#!/usr/bin/env python3
"""Generate v60: local AD component scalar operator without external s1.

This is a Track-B / general-method candidate.  It does not use the external
``s1`` output.  It uses only training channels, v36 LOO residual cache,
geometry-neighbor selection, and H's angle-delay component structure.

For each training point:

    residual_ad = AD((H_train - v36_LOO) * carrier_derotation)
    base_ad     = AD(H_train * carrier_derotation) - residual_ad

For each test point and strong base AD component, it estimates a local complex
operator from nearby training points:

    residual_component ~= scalar(component, local geometry) * base_component

and applies that operator to the test v36 component.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from build_map_features import BS_POS
from evaluate_local_ad_paths import MP, MH, MV, N_ANT, S_SC, ad_transform, idw_weights, inverse_ad
from evaluate_s1style_geom_blend import geometry_cache, select_geom_neighbors
from generate_train_residual_ad_prior import beta_subcarriers, radial_phase


def regime_label(blocked: int) -> str:
    if blocked <= 0:
        return "clear"
    if blocked <= 25:
        return "light"
    if blocked <= 100:
        return "medium"
    return "heavy"


def top_components(power: np.ndarray, count: int) -> np.ndarray:
    flat = np.asarray(power).reshape(-1)
    k = min(int(count), flat.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    ids = np.argpartition(flat, -k)[-k:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    return ids.astype(np.int64)


def flat_to_hvs(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    h = ids // (MV * S_SC)
    rem = ids % (MV * S_SC)
    v = rem // S_SC
    s = rem % S_SC
    return h.astype(np.int64), v.astype(np.int64), s.astype(np.int64)


def make_geom_cfg(args) -> dict:
    return {
        "k": int(args.residual_k),
        "los_penalty": float(args.los_penalty),
        "occ_penalty": float(args.occ_penalty),
        "r_penalty": float(args.r_penalty),
        "theta_penalty": float(args.theta_penalty),
        "occ_scale": float(args.occ_scale),
        "r_scale": float(args.r_scale),
        "theta_scale_deg": float(args.theta_scale_deg),
    }


def build_base_ad_cache(args, train_h, residual_ad, dist_train, beta_s) -> dict:
    cache_path = Path(args.base_ad_cache)
    meta_path = Path(args.base_ad_meta)
    expected_shape = (train_h.shape[0], MP, MH, MV, N_ANT, S_SC)
    if cache_path.exists() and meta_path.exists() and not args.force_base_cache:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    base_ad = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.complex64,
        shape=expected_shape,
    )
    stats = {
        "n_train": int(train_h.shape[0]),
        "base_ad_power_sum": 0.0,
        "residual_ad_power_sum": 0.0,
        "truth_ad_power_sum": 0.0,
    }
    for i in range(train_h.shape[0]):
        truth = np.asarray(train_h[i], dtype=np.complex64)
        phase = radial_phase(float(dist_train[i]), beta_s, +1.0)
        truth_derot = (truth * phase[None, None, :]).astype(np.complex64)
        truth_ad = ad_transform(truth_derot[None])[0]
        res = np.asarray(residual_ad[i], dtype=np.complex64)
        base = (truth_ad - res).astype(np.complex64)
        base_ad[i] = base
        stats["base_ad_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        stats["residual_ad_power_sum"] += float(np.sum(np.abs(res) ** 2, dtype=np.float64))
        stats["truth_ad_power_sum"] += float(np.sum(np.abs(truth_ad) ** 2, dtype=np.float64))
        if (i + 1) % args.cache_flush == 0 or i + 1 == train_h.shape[0]:
            base_ad.flush()
            print(f"cached base AD {i + 1}/{train_h.shape[0]}", flush=True)

    stats["base_to_truth_ad_power"] = stats["base_ad_power_sum"] / max(stats["truth_ad_power_sum"], 1e-30)
    stats["residual_to_truth_ad_power"] = stats["residual_ad_power_sum"] / max(stats["truth_ad_power_sum"], 1e-30)
    payload = {
        "cache": str(cache_path),
        "shape": [int(x) for x in expected_shape],
        "beta": float(args.beta),
        "gamma": float(args.gamma),
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


def estimate_scalar(neigh_base: np.ndarray, neigh_res: np.ndarray, weights: np.ndarray, scalar_ridge: float, scalar_cap: float) -> complex:
    # neigh_* shape: (K, MP, N_ANT)
    w = weights.astype(np.float64)
    prod = np.sum(np.conj(neigh_base) * neigh_res, axis=(1, 2))
    pow_base = np.sum(np.abs(neigh_base) ** 2, axis=(1, 2), dtype=np.float64)
    num = np.sum(w * prod)
    den = float(np.sum(w * pow_base))
    ridge = float(scalar_ridge) * max(float(np.sum(w * pow_base)), 1e-30)
    if den <= 1e-30:
        return 0.0 + 0.0j
    scalar = num / (den + ridge)
    mag = abs(scalar)
    if mag > scalar_cap > 0:
        scalar *= scalar_cap / mag
    return scalar


def constrain_scalar(scalar: complex, args) -> complex:
    if not args.positive_real_only:
        return scalar
    real = max(float(np.real(scalar)), 0.0)
    imag = float(args.imag_scale) * float(np.imag(scalar)) if real > 0.0 else 0.0
    out = complex(real, imag)
    mag = abs(out)
    if mag > args.scalar_cap > 0:
        out *= args.scalar_cap / mag
    return out


def generate(args) -> dict:
    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    train_h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    v36 = np.load(args.v36, mmap_mode="r")
    if v36.shape != (len(test_pos), 256, 4, 192):
        raise ValueError(f"bad v36 shape {v36.shape}")

    train_blocked, test_blocked = geometry_cache(args.data, args.cache, False)
    residual_ad = np.load(args.residual_ad_cache, mmap_mode="r")
    beta_s = beta_subcarriers(args.beta, args.gamma)
    dist_train = np.linalg.norm(train_pos - BS_POS[None, :], axis=1)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)
    base_meta = build_base_ad_cache(args, train_h, residual_ad, dist_train, beta_s)
    base_ad_cache = np.load(args.base_ad_cache, mmap_mode="r")

    geom_cfg = make_geom_cfg(args)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=v36.shape)
    stats = {
        "regime_counts": {"clear": 0, "light": 0, "medium": 0, "heavy": 0},
        "gain_sum": 0.0,
        "base_power_sum": 0.0,
        "corr_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "scalar_abs_sum": 0.0,
        "scalar_count": 0,
        "scalar_clipped": 0,
    }

    for i in range(len(test_pos)):
        base_h = np.asarray(v36[i], dtype=np.complex64)
        phase = radial_phase(float(dist_test[i]), beta_s, +1.0)
        base_derot = (base_h * phase[None, None, :]).astype(np.complex64)
        base_ad = ad_transform(base_derot[None])
        comp_power = np.sum(np.abs(base_ad[0]) ** 2, axis=(0, 3), dtype=np.float64)
        comp_ids = top_components(comp_power, args.components)
        hs, vs, ss = flat_to_hvs(comp_ids)

        idx, _raw_d, eff_d = select_geom_neighbors(
            train_pos,
            train_blocked,
            test_pos[i],
            int(test_blocked[i]),
            None,
            geom_cfg,
        )
        weights = idw_weights(eff_d, args.residual_power)
        corr_ad = np.zeros_like(base_ad)
        gain, label = gain_for_regime(int(test_blocked[i]), args)

        for h, v, s in zip(hs, vs, ss):
            neigh_base = np.asarray(base_ad_cache[idx, :, int(h), int(v), :, int(s)])
            neigh_res = np.asarray(residual_ad[idx, :, int(h), int(v), :, int(s)])
            scalar = estimate_scalar(neigh_base, neigh_res, weights, args.scalar_ridge, args.scalar_cap)
            scalar = constrain_scalar(scalar, args)
            if abs(scalar) >= args.scalar_cap - 1e-6:
                stats["scalar_clipped"] += 1
            stats["scalar_abs_sum"] += float(abs(scalar))
            stats["scalar_count"] += 1
            corr_ad[0, :, int(h), int(v), :, int(s)] = np.complex64(gain * scalar) * base_ad[0, :, int(h), int(v), :, int(s)]

        corr_derot = inverse_ad(corr_ad)[0]
        if args.max_corr_power_ratio > 0:
            base_power = float(np.sum(np.abs(base_h) ** 2, dtype=np.float64))
            corr_power = float(np.sum(np.abs(corr_derot) ** 2, dtype=np.float64))
            max_corr = float(args.max_corr_power_ratio) * max(base_power, 1e-30)
            if corr_power > max_corr:
                corr_derot *= np.float32(np.sqrt(max_corr / corr_power))
        phase_back = radial_phase(float(dist_test[i]), beta_s, -1.0)
        corr = (corr_derot * phase_back[None, None, :]).astype(np.complex64)
        pred = (base_h + corr).astype(np.complex64)
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite prediction at {i}")
        out[i] = pred
        if (i + 1) % args.batch == 0 or i + 1 == len(test_pos):
            out.flush()
            print(f"generated {i + 1}/{len(test_pos)}", flush=True)

        stats["regime_counts"][label] += 1
        stats["gain_sum"] += float(gain)
        stats["base_power_sum"] += float(np.sum(np.abs(base_h) ** 2, dtype=np.float64))
        stats["corr_power_sum"] += float(np.sum(np.abs(corr) ** 2, dtype=np.float64))
        stats["pred_power_sum"] += float(np.sum(np.abs(pred) ** 2, dtype=np.float64))

    stats["gain_mean"] = stats["gain_sum"] / max(len(test_pos), 1)
    stats["corr_to_base_power"] = stats["corr_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_to_base_power"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["scalar_abs_mean"] = stats["scalar_abs_sum"] / max(stats["scalar_count"], 1)

    payload = {
        "method": "local_ad_scalar_operator",
        "version": "v60",
        "uses_external_s1": False,
        "formula": "v36 + gain(regime) * local_scalar(component, geometry) * AD(v36 component)",
        "v36": args.v36,
        "residual_cache": args.residual_ad_cache,
        "base_ad_cache": base_meta,
        "params": {
            "components": int(args.components),
            "residual_k": int(args.residual_k),
            "residual_power": float(args.residual_power),
            "scalar_ridge": float(args.scalar_ridge),
            "scalar_cap": float(args.scalar_cap),
            "positive_real_only": bool(args.positive_real_only),
            "imag_scale": float(args.imag_scale),
            "max_corr_power_ratio": float(args.max_corr_power_ratio),
            "gains": {
                "clear": float(args.clear_gain),
                "light": float(args.light_gain),
                "medium": float(args.medium_gain),
                "heavy": float(args.heavy_gain),
            },
        },
        "stats": stats,
        "output": args.output,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--residual-ad-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad.npy"))
    parser.add_argument("--base-ad-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_base_derot_ad.npy"))
    parser.add_argument("--base-ad-meta", default=str(PATHS.analysis / "results/json/train_v36_loo_base_derot_ad_meta.json"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(60) / "local_ad_scalar_operator_v60_generate.json"))
    parser.add_argument("--force-base-cache", action="store_true")
    parser.add_argument("--cache-flush", type=int, default=25)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--components", type=int, default=48)
    parser.add_argument("--residual-k", type=int, default=48)
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--scalar-ridge", type=float, default=0.5)
    parser.add_argument("--scalar-cap", type=float, default=0.35)
    parser.add_argument("--positive-real-only", action="store_true")
    parser.add_argument("--imag-scale", type=float, default=0.25)
    parser.add_argument("--max-corr-power-ratio", type=float, default=0.06)
    parser.add_argument("--clear-gain", type=float, default=0.10)
    parser.add_argument("--light-gain", type=float, default=0.10)
    parser.add_argument("--medium-gain", type=float, default=0.50)
    parser.add_argument("--heavy-gain", type=float, default=1.00)
    parser.add_argument("--los-penalty", type=float, default=1.0)
    parser.add_argument("--occ-penalty", type=float, default=0.5)
    parser.add_argument("--r-penalty", type=float, default=0.0)
    parser.add_argument("--theta-penalty", type=float, default=0.0)
    parser.add_argument("--occ-scale", type=float, default=80.0)
    parser.add_argument("--r-scale", type=float, default=80.0)
    parser.add_argument("--theta-scale-deg", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = generate(args)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
