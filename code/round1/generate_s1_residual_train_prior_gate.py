#!/usr/bin/env python3
"""Generate v59: v55 residual with train-prior calibrated component gates.

v55 proved that ``s1 - v9`` contains useful test-point residual structure, but
only a small number of angle-delay components should receive strong AD-shift
calibration.  v58 proved that train LOO true residuals contain signal, but are
not accurate enough to replace ``s1 - v9`` directly.

This candidate keeps the v55 residual source and uses the v58 train residual
prior only as a confidence signal:

* top strong components are shifted more if the train prior agrees with the
  shifted residual direction;
* top strong components are shifted less if the prior conflicts;
* a few extra non-top4 components can receive medium shift only when supported
  by the train prior.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_local_ad_paths import ad_transform, inverse_ad
from evaluate_s1style_geom_blend import geometry_cache
from generate_s1_residual_regime_fusion import regime_label
from generate_train_residual_ad_prior import beta_subcarriers, predict_residual_from_cache
from build_map_features import BS_POS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--v9", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v9.npy"))
    parser.add_argument("--s1", default=str(PATHS.artifacts / "external/tmp_blend_inspect/s1.npy"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--residual-ad-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad.npy"))
    parser.add_argument("--residual-power-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad_power.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(59) / "s1_residual_train_prior_gate_v59_generate.json"))

    parser.add_argument("--clear-lambda", type=float, default=1.0)
    parser.add_argument("--light-lambda", type=float, default=1.0)
    parser.add_argument("--medium-lambda", type=float, default=1.0)
    parser.add_argument("--heavy-lambda", type=float, default=4.0)
    parser.add_argument("--clear-phase-deg", type=float, default=0.0)
    parser.add_argument("--light-phase-deg", type=float, default=10.0)
    parser.add_argument("--medium-phase-deg", type=float, default=3.0)
    parser.add_argument("--heavy-phase-deg", type=float, default=2.0)

    parser.add_argument("--top-components", type=int, default=4)
    parser.add_argument("--extra-components", type=int, default=12)
    parser.add_argument("--weak-rho", type=float, default=0.08)
    parser.add_argument("--top-rho", type=float, default=0.75)
    parser.add_argument("--top-boost", type=float, default=0.15)
    parser.add_argument("--top-conflict-penalty", type=float, default=0.25)
    parser.add_argument("--top-min-rho", type=float, default=0.45)
    parser.add_argument("--top-max-rho", type=float, default=0.92)
    parser.add_argument("--extra-rho", type=float, default=0.32)
    parser.add_argument("--extra-boost", type=float, default=0.12)
    parser.add_argument("--extra-align-thresh", type=float, default=0.12)
    parser.add_argument("--prior-power-frac", type=float, default=0.02)
    parser.add_argument("--delay-shift", type=int, default=-1)
    parser.add_argument("--h-shift", type=int, default=-1)
    parser.add_argument("--v-shift", type=int, default=1)
    parser.add_argument("--preserve-delta-power", action="store_true")

    # Train residual prior prediction.  These arguments are consumed by
    # generate_train_residual_ad_prior.predict_residual_from_cache.
    parser.add_argument("--residual-k", type=int, default=32)
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--los-penalty", type=float, default=1.0)
    parser.add_argument("--occ-penalty", type=float, default=0.5)
    parser.add_argument("--r-penalty", type=float, default=0.0)
    parser.add_argument("--theta-penalty", type=float, default=0.0)
    parser.add_argument("--occ-scale", type=float, default=80.0)
    parser.add_argument("--r-scale", type=float, default=80.0)
    parser.add_argument("--theta-scale-deg", type=float, default=30.0)
    parser.add_argument("--batch", type=int, default=2)
    return parser.parse_args()


def top_mask(power: np.ndarray, count: int) -> np.ndarray:
    flat = power.reshape(power.shape[0], -1)
    k = min(int(count), flat.shape[1])
    mask_flat = np.zeros_like(flat, dtype=bool)
    if k <= 0:
        return mask_flat.reshape(power.shape)
    ids = np.argpartition(flat, -k, axis=1)[:, -k:]
    rows = np.arange(flat.shape[0])[:, None]
    mask_flat[rows, ids] = True
    return mask_flat.reshape(power.shape)


def ranked_mask(power: np.ndarray, start_rank: int, end_rank: int) -> np.ndarray:
    flat = power.reshape(power.shape[0], -1)
    n = flat.shape[1]
    lo = max(int(start_rank), 0)
    hi = min(int(end_rank), n)
    mask_flat = np.zeros_like(flat, dtype=bool)
    if hi <= lo:
        return mask_flat.reshape(power.shape)
    order_count = hi
    ids = np.argpartition(flat, -order_count, axis=1)[:, -order_count:]
    rows = np.arange(flat.shape[0])
    for r, row_ids in zip(rows, ids):
        row_ids = row_ids[np.argsort(flat[r, row_ids])[::-1]]
        mask_flat[r, row_ids[lo:hi]] = True
    return mask_flat.reshape(power.shape)


def component_coherence(shifted_ad: np.ndarray, prior_ad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shifted_power = np.sum(np.abs(shifted_ad) ** 2, axis=(1, 4), dtype=np.float64)
    prior_power = np.sum(np.abs(prior_ad) ** 2, axis=(1, 4), dtype=np.float64)
    dot = np.sum(np.conj(shifted_ad) * prior_ad, axis=(1, 4))
    denom = np.sqrt(np.maximum(shifted_power * prior_power, 1e-30))
    coh = np.real(dot) / denom
    return np.clip(coh, -1.0, 1.0).astype(np.float32), shifted_power, prior_power


def main():
    args = parse_args()

    v9 = np.load(args.v9, mmap_mode="r")
    s1 = np.load(args.s1, mmap_mode="r")
    v36 = np.load(args.v36, mmap_mode="r")
    if v9.shape != s1.shape or v9.shape != v36.shape:
        raise ValueError(f"shape mismatch: v9={v9.shape}, s1={s1.shape}, v36={v36.shape}")
    if v9.shape != (500, 256, 4, 192):
        raise ValueError(f"unexpected output shape {v9.shape}")

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    train_blocked, test_blocked = geometry_cache(args.data, args.cache, False)
    res_ad = np.load(args.residual_ad_cache, mmap_mode="r")
    res_power = np.load(args.residual_power_cache, mmap_mode="r")
    beta_s = beta_subcarriers(args.beta, args.gamma)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)

    lambdas = {
        "clear": float(args.clear_lambda),
        "light": float(args.light_lambda),
        "medium": float(args.medium_lambda),
        "heavy": float(args.heavy_lambda),
    }
    phase_deg = {
        "clear": float(args.clear_phase_deg),
        "light": float(args.light_phase_deg),
        "medium": float(args.medium_phase_deg),
        "heavy": float(args.heavy_phase_deg),
    }
    phase = {k: np.exp(1j * np.deg2rad(v)).astype(np.complex64) for k, v in phase_deg.items()}

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=v36.shape)
    stats = {
        "clear": 0,
        "light": 0,
        "medium": 0,
        "heavy": 0,
        "lambda_sum": 0.0,
        "base_power_sum": 0.0,
        "raw_delta_power_sum": 0.0,
        "mixed_delta_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "top_rho_sum": 0.0,
        "top_count": 0,
        "extra_promoted_count": 0,
        "extra_rho_sum": 0.0,
        "energy_weighted_rho_sum": 0.0,
        "total_ad_energy_sum": 0.0,
        "top_coh_sum": 0.0,
        "top_coh_count": 0,
    }

    for start in range(0, len(test_blocked), args.batch):
        end = min(start + args.batch, len(test_blocked))
        base = np.asarray(v36[start:end], dtype=np.complex64)
        raw_delta = np.asarray(s1[start:end], dtype=np.complex64) - np.asarray(v9[start:end], dtype=np.complex64)

        lam = np.empty(end - start, dtype=np.float32)
        ph = np.empty(end - start, dtype=np.complex64)
        for j, blocked in enumerate(test_blocked[start:end]):
            label = regime_label(int(blocked))
            lam[j] = lambdas[label]
            ph[j] = phase[label]
            stats[label] += 1
            stats["lambda_sum"] += float(lam[j])

        phased_delta = raw_delta * ph[:, None, None, None]
        ad_delta = ad_transform(phased_delta)
        shifted_ad = np.roll(
            ad_delta,
            shift=(args.h_shift, args.v_shift, args.delay_shift),
            axis=(2, 3, 5),
        )

        prior = np.empty_like(phased_delta)
        for local_i, global_i in enumerate(range(start, end)):
            prior[local_i], _meta = predict_residual_from_cache(
                test_pos[global_i],
                int(test_blocked[global_i]),
                train_pos,
                train_blocked,
                res_ad,
                res_power,
                float(dist_test[global_i]),
                beta_s,
                args,
            )
        prior_ad = ad_transform(prior)
        coh, shifted_power, prior_power = component_coherence(shifted_ad, prior_ad)

        strong = top_mask(shifted_power, args.top_components)
        extra = ranked_mask(shifted_power, args.top_components, args.extra_components)
        prior_threshold = (
            args.prior_power_frac
            * np.max(prior_power.reshape(prior_power.shape[0], -1), axis=1)[:, None, None, None]
        )
        prior_supported = prior_power >= np.maximum(prior_threshold, 1e-30)

        rho_hvs = np.full_like(coh, float(args.weak_rho), dtype=np.float32)
        top_rho = (
            float(args.top_rho)
            + float(args.top_boost) * np.maximum(coh, 0.0)
            - float(args.top_conflict_penalty) * np.maximum(-coh, 0.0)
        )
        top_rho = np.clip(top_rho, float(args.top_min_rho), float(args.top_max_rho)).astype(np.float32)
        rho_hvs = np.where(strong, top_rho, rho_hvs)

        extra_supported = extra & prior_supported & (coh >= float(args.extra_align_thresh))
        extra_rho = np.clip(
            float(args.extra_rho) + float(args.extra_boost) * np.maximum(coh, 0.0),
            float(args.weak_rho),
            float(args.top_rho),
        ).astype(np.float32)
        rho_hvs = np.where(extra_supported, extra_rho, rho_hvs)

        rho = rho_hvs[:, None, :, :, None, :]
        mixed_ad = ((1.0 - rho) * ad_delta + rho * shifted_ad).astype(np.complex64)
        mixed_delta = inverse_ad(mixed_ad)

        if args.preserve_delta_power:
            raw_power = np.sum(np.abs(phased_delta) ** 2, axis=(1, 2, 3), dtype=np.float64)
            mixed_power = np.sum(np.abs(mixed_delta) ** 2, axis=(1, 2, 3), dtype=np.float64)
            scale = np.sqrt(raw_power / np.maximum(mixed_power, 1e-30)).astype(np.float32)
            mixed_delta *= scale[:, None, None, None]

        pred = (base + lam[:, None, None, None] * mixed_delta).astype(np.complex64)
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out[start:end] = pred
        out.flush()

        total_energy = np.sum(shifted_power, dtype=np.float64)
        stats["base_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        stats["raw_delta_power_sum"] += float(np.sum(np.abs(raw_delta) ** 2, dtype=np.float64))
        stats["mixed_delta_power_sum"] += float(np.sum(np.abs(mixed_delta) ** 2, dtype=np.float64))
        stats["pred_power_sum"] += float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        stats["total_ad_energy_sum"] += float(total_energy)
        stats["energy_weighted_rho_sum"] += float(np.sum(shifted_power * rho_hvs, dtype=np.float64))
        stats["top_rho_sum"] += float(np.sum(rho_hvs[strong], dtype=np.float64))
        stats["top_count"] += int(np.sum(strong))
        stats["extra_promoted_count"] += int(np.sum(extra_supported))
        stats["extra_rho_sum"] += float(np.sum(rho_hvs[extra_supported], dtype=np.float64))
        stats["top_coh_sum"] += float(np.sum(coh[strong], dtype=np.float64))
        stats["top_coh_count"] += int(np.sum(strong))
        print(f"generated {end}/{len(test_blocked)}", flush=True)

    stats["lambda_mean"] = stats["lambda_sum"] / max(len(test_blocked), 1)
    stats["raw_delta_power_ratio"] = stats["raw_delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["mixed_delta_power_ratio"] = stats["mixed_delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_power_ratio"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["energy_weighted_rho"] = stats["energy_weighted_rho_sum"] / max(stats["total_ad_energy_sum"], 1e-30)
    stats["top_rho_mean"] = stats["top_rho_sum"] / max(stats["top_count"], 1)
    stats["extra_rho_mean"] = stats["extra_rho_sum"] / max(stats["extra_promoted_count"], 1)
    stats["top_coh_mean"] = stats["top_coh_sum"] / max(stats["top_coh_count"], 1)

    payload = {
        "method": "s1_residual_train_prior_gate",
        "version": "v59",
        "formula": "v36 + lambda(regime) * inverse_ad(train-prior-calibrated component gate over s1-v9 residual)",
        "v9": args.v9,
        "s1": args.s1,
        "v36": args.v36,
        "train_residual_cache": {
            "ad": args.residual_ad_cache,
            "power": args.residual_power_cache,
            "prior_components": int(args.components),
            "residual_k": int(args.residual_k),
        },
        "lambdas": lambdas,
        "phase_deg": phase_deg,
        "gate": {
            "top_components": int(args.top_components),
            "extra_components": int(args.extra_components),
            "weak_rho": float(args.weak_rho),
            "top_rho": float(args.top_rho),
            "top_boost": float(args.top_boost),
            "top_conflict_penalty": float(args.top_conflict_penalty),
            "top_min_rho": float(args.top_min_rho),
            "top_max_rho": float(args.top_max_rho),
            "extra_rho": float(args.extra_rho),
            "extra_boost": float(args.extra_boost),
            "extra_align_thresh": float(args.extra_align_thresh),
            "prior_power_frac": float(args.prior_power_frac),
        },
        "shift": {
            "delay": int(args.delay_shift),
            "h": int(args.h_shift),
            "v": int(args.v_shift),
            "preserve_delta_power": bool(args.preserve_delta_power),
        },
        "stats": stats,
        "output": args.output,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
