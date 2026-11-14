#!/usr/bin/env python3
"""Generate v55 + Sionna exact-path component gate.

This keeps the confirmed v55 mechanism intact:

    v36 + lambda(regime) * inverse_ad(component-gated ADShift(s1 - v9))

The only change is a narrow, geometry-derived adjustment of the top component
rho.  For each AD residual component, we compare its delay bin with the exact
Sionna mesh path candidates (LOS + one-bounce specular). Components whose delay
matches a candidate path are shifted slightly more; poorly matched components
are shifted slightly less.  This uses the continuous mesh scene as a Physical AI
feature/gate, not as a ray-traced channel synthesizer.
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
    parser.add_argument("--sionna-paths", default=str(PATHS.artifacts / "shared_features/sionna_path_candidates.npz"))
    parser.add_argument("--v9", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v9.npy"))
    parser.add_argument("--s1", default=str(PATHS.artifacts / "external/tmp_blend_inspect/s1.npy"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/sionna_path_component_gate_generate.json"))

    parser.add_argument("--clear-lambda", type=float, default=1.0)
    parser.add_argument("--light-lambda", type=float, default=1.0)
    parser.add_argument("--medium-lambda", type=float, default=1.0)
    parser.add_argument("--heavy-lambda", type=float, default=4.0)
    parser.add_argument("--clear-phase-deg", type=float, default=0.0)
    parser.add_argument("--light-phase-deg", type=float, default=10.0)
    parser.add_argument("--medium-phase-deg", type=float, default=3.0)
    parser.add_argument("--heavy-phase-deg", type=float, default=2.0)

    parser.add_argument("--top-components", type=int, default=4)
    parser.add_argument("--top-rho", type=float, default=0.78)
    parser.add_argument("--weak-rho", type=float, default=0.10)
    parser.add_argument("--path-sigma", type=float, default=0.75)
    parser.add_argument("--path-center", type=float, default=0.35)
    parser.add_argument("--path-gain", type=float, default=0.10)
    parser.add_argument("--ref-adv-gain", type=float, default=0.0)
    parser.add_argument("--ref-adv-power", type=float, default=1.0)
    parser.add_argument("--hit-gain", type=float, default=0.015)
    parser.add_argument("--top-min-rho", type=float, default=0.68)
    parser.add_argument("--top-max-rho", type=float, default=0.88)
    parser.add_argument("--delay-shift", type=int, default=-1)
    parser.add_argument("--h-shift", type=int, default=-1)
    parser.add_argument("--v-shift", type=int, default=1)
    parser.add_argument("--preserve-delta-power", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    return parser.parse_args()


def top_component_mask(power: np.ndarray, top_components: int) -> np.ndarray:
    flat = power.reshape(power.shape[0], -1)
    k = min(int(top_components), flat.shape[1])
    ids = np.argpartition(flat, -k, axis=1)[:, -k:]
    mask_flat = np.zeros_like(flat, dtype=bool)
    rows = np.arange(flat.shape[0])[:, None]
    mask_flat[rows, ids] = True
    return mask_flat.reshape(power.shape)


def path_delay_confidence(path_delay: np.ndarray, path_valid: np.ndarray, sigma: float, s_count: int) -> np.ndarray:
    """Return per-delay confidence from the closest valid path candidate."""

    delay_grid = np.arange(s_count, dtype=np.float32)[None, :]
    conf = np.zeros((path_delay.shape[0], s_count), dtype=np.float32)
    for i in range(path_delay.shape[0]):
        valid = path_valid[i] & np.isfinite(path_delay[i])
        if not np.any(valid):
            continue
        err = np.min(np.abs(delay_grid - path_delay[i, valid].astype(np.float32)[:, None]), axis=0)
        conf[i] = np.exp(-0.5 * (err / max(float(sigma), 1e-6)) ** 2).astype(np.float32)
    return conf


def path_type_delay_confidence(
    path_delay: np.ndarray,
    path_valid: np.ndarray,
    path_type: np.ndarray,
    type_id: int,
    sigma: float,
    s_count: int,
) -> np.ndarray:
    return path_delay_confidence(path_delay, path_valid & (path_type == int(type_id)), sigma, s_count)


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
    paths = np.load(args.sionna_paths)
    path_delay = paths["test_delay_bin"].astype(np.float32)
    path_valid = paths["test_valid"].astype(bool)
    path_type = paths["test_path_type"].astype(np.int16)
    los_hit_count = paths["test_los_hit_count"].astype(np.float32)
    if path_delay.shape[0] != v9.shape[0]:
        raise ValueError(f"path/test mismatch: paths={path_delay.shape}, channel={v9.shape}")

    path_conf = path_delay_confidence(path_delay, path_valid, args.path_sigma, v9.shape[-1])
    los_conf = path_type_delay_confidence(path_delay, path_valid, path_type, 0, args.path_sigma, v9.shape[-1])
    ref_conf = path_type_delay_confidence(path_delay, path_valid, path_type, 1, args.path_sigma, v9.shape[-1])
    hit_norm = np.clip((los_hit_count - 4.0) / 10.0, 0.0, 1.0).astype(np.float32)
    ref_gate = np.clip((np.asarray(test_blocked, dtype=np.float32) - 25.0) / 100.0, 0.0, 1.0) ** float(args.ref_adv_power)

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
        "top_rho_sum": 0.0,
        "top_count": 0,
        "top_conf_sum": 0.0,
        "top_hit_sum": 0.0,
        "top_rho_min": float("inf"),
        "top_rho_max": float("-inf"),
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

        component_power = np.sum(np.abs(shifted_ad) ** 2, axis=(1, 4), dtype=np.float64)
        strong_mask = top_component_mask(component_power, args.top_components)
        rho_hvs = np.full(component_power.shape, float(args.weak_rho), dtype=np.float32)

        conf_s = path_conf[start:end]  # B x S
        conf_hvs = np.broadcast_to(conf_s[:, None, None, :], component_power.shape).astype(np.float32)
        los_hvs = np.broadcast_to(los_conf[start:end, None, None, :], component_power.shape).astype(np.float32)
        ref_hvs = np.broadcast_to(ref_conf[start:end, None, None, :], component_power.shape).astype(np.float32)
        ref_adv = np.maximum(ref_hvs - los_hvs, 0.0)
        ref_term = ref_gate[start:end, None, None, None]
        hit_term = hit_norm[start:end, None, None, None]
        top_rho = (
            float(args.top_rho)
            + float(args.path_gain) * (conf_hvs - float(args.path_center))
            + float(args.ref_adv_gain) * ref_adv * ref_term
            + float(args.hit_gain) * hit_term
        )
        top_rho = np.clip(top_rho, float(args.top_min_rho), float(args.top_max_rho)).astype(np.float32)
        rho_hvs = np.where(strong_mask, top_rho, rho_hvs)

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
        top_vals = rho_hvs[strong_mask]
        conf_vals = conf_hvs[strong_mask]
        ref_adv_vals = ref_adv[strong_mask]
        hit_vals = np.broadcast_to(hit_term, component_power.shape)[strong_mask]
        stats["base_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        stats["raw_delta_power_sum"] += float(np.sum(np.abs(raw_delta) ** 2, dtype=np.float64))
        stats["mixed_delta_power_sum"] += float(np.sum(np.abs(mixed_delta) ** 2, dtype=np.float64))
        stats["pred_power_sum"] += float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        stats["strong_energy_sum"] += float(np.sum(strong_energy, dtype=np.float64))
        stats["total_ad_energy_sum"] += float(np.sum(total_energy, dtype=np.float64))
        stats["rho_energy_sum"] += float(np.sum(rho_energy, dtype=np.float64))
        stats["top_rho_sum"] += float(np.sum(top_vals, dtype=np.float64))
        stats["top_count"] += int(top_vals.size)
        stats["top_conf_sum"] += float(np.sum(conf_vals, dtype=np.float64))
        stats.setdefault("top_ref_adv_sum", 0.0)
        stats["top_ref_adv_sum"] += float(np.sum(ref_adv_vals, dtype=np.float64))
        stats["top_hit_sum"] += float(np.sum(hit_vals, dtype=np.float64))
        stats["top_rho_min"] = min(stats["top_rho_min"], float(np.min(top_vals)))
        stats["top_rho_max"] = max(stats["top_rho_max"], float(np.max(top_vals)))
        print(f"generated {end}/{len(test_blocked)}", flush=True)

    stats["lambda_mean"] = stats["lambda_sum"] / max(len(test_blocked), 1)
    stats["raw_delta_power_ratio"] = stats["raw_delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["mixed_delta_power_ratio"] = stats["mixed_delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_power_ratio"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["strong_energy_frac"] = stats["strong_energy_sum"] / max(stats["total_ad_energy_sum"], 1e-30)
    stats["energy_weighted_rho"] = stats["rho_energy_sum"] / max(stats["total_ad_energy_sum"], 1e-30)
    stats["top_rho_mean"] = stats["top_rho_sum"] / max(stats["top_count"], 1)
    stats["top_conf_mean"] = stats["top_conf_sum"] / max(stats["top_count"], 1)
    stats["top_ref_adv_mean"] = stats.get("top_ref_adv_sum", 0.0) / max(stats["top_count"], 1)
    stats["top_hit_mean"] = stats["top_hit_sum"] / max(stats["top_count"], 1)

    payload = {
        "method": "sionna_path_component_gate",
        "base": "v55_s1_residual_component_gate",
        "formula": "v36 + lambda(regime) * inverse_ad(Sionna-path-adjusted component gate over ADShift(s1-v9 residual))",
        "v9": args.v9,
        "s1": args.s1,
        "v36": args.v36,
        "sionna_paths": args.sionna_paths,
        "lambdas": lambdas,
        "phase_deg": phase_deg,
        "gate": {
            "top_components": int(args.top_components),
            "top_rho": float(args.top_rho),
            "weak_rho": float(args.weak_rho),
            "path_sigma": float(args.path_sigma),
            "path_center": float(args.path_center),
            "path_gain": float(args.path_gain),
            "ref_adv_gain": float(args.ref_adv_gain),
            "ref_adv_power": float(args.ref_adv_power),
            "hit_gain": float(args.hit_gain),
            "top_min_rho": float(args.top_min_rho),
            "top_max_rho": float(args.top_max_rho),
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
