#!/usr/bin/env python3
"""Generate component-gated inverse-RT AD-shift residual candidates.

v53 applied the calibrated AD shift uniformly to 25% of the residual and
improved online.  v54 applied it to 100% of the residual and dropped slightly.
This generator uses the internal structure of H: it moves only the dominant
angle-delay residual components strongly, while keeping weak/diffuse
components closer to the original residual.
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--v9", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v9.npy"))
    parser.add_argument("--s1", default=str(PATHS.artifacts / "external/tmp_blend_inspect/s1.npy"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/s1_residual_component_gate_generate.json"))
    parser.add_argument("--clear-lambda", type=float, default=1.0)
    parser.add_argument("--light-lambda", type=float, default=1.0)
    parser.add_argument("--medium-lambda", type=float, default=1.0)
    parser.add_argument("--heavy-lambda", type=float, default=4.0)
    parser.add_argument("--clear-phase-deg", type=float, default=0.0)
    parser.add_argument("--light-phase-deg", type=float, default=10.0)
    parser.add_argument("--medium-phase-deg", type=float, default=3.0)
    parser.add_argument("--heavy-phase-deg", type=float, default=2.0)
    parser.add_argument("--strong-rho", type=float, default=0.75)
    parser.add_argument("--weak-rho", type=float, default=0.10)
    parser.add_argument("--top-components", type=int, default=4)
    parser.add_argument("--delay-shift", type=int, default=-1)
    parser.add_argument("--h-shift", type=int, default=-1)
    parser.add_argument("--v-shift", type=int, default=1)
    parser.add_argument("--preserve-delta-power", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    return parser.parse_args()


def top_component_mask(power, top_components):
    flat = power.reshape(power.shape[0], -1)
    k = min(int(top_components), flat.shape[1])
    ids = np.argpartition(flat, -k, axis=1)[:, -k:]
    mask_flat = np.zeros_like(flat, dtype=bool)
    rows = np.arange(flat.shape[0])[:, None]
    mask_flat[rows, ids] = True
    return mask_flat.reshape(power.shape)


def main():
    args = parse_args()

    v9 = np.load(args.v9, mmap_mode="r")
    s1 = np.load(args.s1, mmap_mode="r")
    v36 = np.load(args.v36, mmap_mode="r")
    if v9.shape != s1.shape or v9.shape != v36.shape:
        raise ValueError(f"shape mismatch: v9={v9.shape}, s1={s1.shape}, v36={v36.shape}")
    if v9.shape != (500, 256, 4, 192):
        raise ValueError(f"unexpected output shape {v9.shape}")

    _train_blocked, test_blocked = geometry_cache(args.data, args.cache, False)
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
        "strong_energy_sum": 0.0,
        "total_ad_energy_sum": 0.0,
        "rho_energy_sum": 0.0,
        "raw_shift_corr_re": 0.0,
        "raw_shift_corr_im": 0.0,
    }

    for start in range(0, len(test_blocked), args.batch):
        end = min(start + args.batch, len(test_blocked))
        base = np.asarray(v36[start:end], dtype=np.complex64)
        raw_delta = np.asarray(s1[start:end], dtype=np.complex64) - np.asarray(v9[start:end], dtype=np.complex64)

        lam = np.empty((end - start,), dtype=np.float32)
        ph = np.empty((end - start,), dtype=np.complex64)
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

        # Component power is measured after shifting, because the output support
        # is the calibrated AD coordinate system.
        component_power = np.sum(np.abs(shifted_ad) ** 2, axis=(1, 4), dtype=np.float64)
        strong_mask = top_component_mask(component_power, args.top_components)
        rho_hvs = np.where(strong_mask, args.strong_rho, args.weak_rho).astype(np.float32)
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

        total_energy = np.sum(component_power, axis=(1, 2, 3), dtype=np.float64)
        strong_energy = np.sum(np.where(strong_mask, component_power, 0.0), axis=(1, 2, 3), dtype=np.float64)
        rho_energy = np.sum(component_power * rho_hvs.astype(np.float64), axis=(1, 2, 3), dtype=np.float64)
        stats["base_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        stats["raw_delta_power_sum"] += float(np.sum(np.abs(raw_delta) ** 2, dtype=np.float64))
        stats["mixed_delta_power_sum"] += float(np.sum(np.abs(mixed_delta) ** 2, dtype=np.float64))
        stats["pred_power_sum"] += float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        stats["strong_energy_sum"] += float(np.sum(strong_energy, dtype=np.float64))
        stats["total_ad_energy_sum"] += float(np.sum(total_energy, dtype=np.float64))
        stats["rho_energy_sum"] += float(np.sum(rho_energy, dtype=np.float64))
        corr = np.vdot(phased_delta, inverse_ad(shifted_ad))
        stats["raw_shift_corr_re"] += float(corr.real)
        stats["raw_shift_corr_im"] += float(corr.imag)
        print(f"generated {end}/{len(test_blocked)}", flush=True)

    stats["lambda_mean"] = stats["lambda_sum"] / max(len(test_blocked), 1)
    stats["raw_delta_power_ratio"] = stats["raw_delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["mixed_delta_power_ratio"] = stats["mixed_delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_power_ratio"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["strong_energy_frac"] = stats["strong_energy_sum"] / max(stats["total_ad_energy_sum"], 1e-30)
    stats["energy_weighted_rho"] = stats["rho_energy_sum"] / max(stats["total_ad_energy_sum"], 1e-30)
    corr = complex(stats["raw_shift_corr_re"], stats["raw_shift_corr_im"])
    stats["raw_shift_corr_phase_deg"] = float(np.angle(corr, deg=True)) if abs(corr) > 0 else None

    payload = {
        "method": "s1_residual_component_gate",
        "formula": "v36 + lambda(regime) * inverse_ad((1-rho(component))*AD(residual) + rho(component)*ADShift(AD(residual)))",
        "v9": args.v9,
        "s1": args.s1,
        "v36": args.v36,
        "lambdas": lambdas,
        "phase_deg": phase_deg,
        "component_gate": {
            "top_components": int(args.top_components),
            "strong_rho": float(args.strong_rho),
            "weak_rho": float(args.weak_rho),
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
