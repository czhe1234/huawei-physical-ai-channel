#!/usr/bin/env python3
"""Generate a capped regime-gated residual fusion candidate.

The successful v48 submission showed that transferring the external s1
residual onto v36 is useful:

    H = H_v36 + lambda(regime) * (H_s1 - H_v9)

For more aggressive lambdas, a small number of very low-power points can receive
an enormous relative perturbation because ``||H_v36||`` is tiny.  This generator
keeps the same residual direction but caps the per-point perturbation amplitude:

    ||lambda * delta|| / ||H_v36|| <= max_amp_ratio

This avoids the v46-style power manipulation while making residual extrapolation
less dominated by extreme low-power outliers.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_s1style_geom_blend import geometry_cache
from generate_s1_residual_regime_fusion import regime_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--v9", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v9.npy"))
    parser.add_argument("--s1", default=str(PATHS.artifacts / "external/tmp_blend_inspect/s1.npy"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/s1_residual_capped_fusion_generate.json"))
    parser.add_argument("--clear-lambda", type=float, default=1.0)
    parser.add_argument("--light-lambda", type=float, default=1.0)
    parser.add_argument("--medium-lambda", type=float, default=2.0)
    parser.add_argument("--heavy-lambda", type=float, default=12.0)
    parser.add_argument("--max-amp-ratio", type=float, default=8.0)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

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

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=v36.shape)
    stats = {
        "clear": 0,
        "light": 0,
        "medium": 0,
        "heavy": 0,
        "lambda_sum": 0.0,
        "effective_lambda_sum": 0.0,
        "capped_count": 0,
        "delta_power_sum": 0.0,
        "base_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "max_raw_amp_ratio": 0.0,
        "max_effective_amp_ratio": 0.0,
    }

    for start in range(0, len(test_blocked), args.batch):
        end = min(start + args.batch, len(test_blocked))
        base = np.asarray(v36[start:end], dtype=np.complex64)
        delta = np.asarray(s1[start:end], dtype=np.complex64) - np.asarray(v9[start:end], dtype=np.complex64)

        base_power = np.sum(np.abs(base) ** 2, axis=(1, 2, 3), dtype=np.float64)
        delta_power = np.sum(np.abs(delta) ** 2, axis=(1, 2, 3), dtype=np.float64)

        lam = np.empty((end - start,), dtype=np.float32)
        for j, blocked in enumerate(test_blocked[start:end]):
            label = regime_label(int(blocked))
            lam[j] = lambdas[label]
            stats[label] += 1
            stats["lambda_sum"] += float(lam[j])

        raw_amp = lam.astype(np.float64) * np.sqrt(delta_power / np.maximum(base_power, 1e-30))
        scale = np.ones_like(raw_amp)
        over = raw_amp > args.max_amp_ratio
        scale[over] = args.max_amp_ratio / np.maximum(raw_amp[over], 1e-30)
        eff_lam = lam.astype(np.float64) * scale

        pred = base + eff_lam[:, None, None, None].astype(np.float32) * delta
        pred = pred.astype(np.complex64)
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out[start:end] = pred
        out.flush()

        pred_power = np.sum(np.abs(pred) ** 2, axis=(1, 2, 3), dtype=np.float64)
        stats["capped_count"] += int(np.sum(over))
        stats["effective_lambda_sum"] += float(np.sum(eff_lam))
        stats["delta_power_sum"] += float(np.sum(delta_power))
        stats["base_power_sum"] += float(np.sum(base_power))
        stats["pred_power_sum"] += float(np.sum(pred_power))
        stats["max_raw_amp_ratio"] = max(stats["max_raw_amp_ratio"], float(np.max(raw_amp)))
        effective_amp = eff_lam * np.sqrt(delta_power / np.maximum(base_power, 1e-30))
        stats["max_effective_amp_ratio"] = max(stats["max_effective_amp_ratio"], float(np.max(effective_amp)))
        print(f"generated {end}/{len(test_blocked)}", flush=True)

    n = max(len(test_blocked), 1)
    stats["lambda_mean"] = stats["lambda_sum"] / n
    stats["effective_lambda_mean"] = stats["effective_lambda_sum"] / n
    stats["delta_power_ratio"] = stats["delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_power_ratio"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)

    payload = {
        "method": "s1_residual_capped_fusion",
        "formula": "v36 + capped_lambda(regime, max_amp_ratio) * (s1 - v9)",
        "v9": args.v9,
        "s1": args.s1,
        "v36": args.v36,
        "lambdas": lambdas,
        "max_amp_ratio": float(args.max_amp_ratio),
        "stats": stats,
        "output": args.output,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
