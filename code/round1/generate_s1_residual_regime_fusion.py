#!/usr/bin/env python3
"""Generate a regime-gated residual fusion between s1, v9, and v36.

The online history gives two independent signals:

* v10/s1 is a stronger physical-local baseline than our v9 baseline.
* v36 is the strongest height-regime geometry extrapolation built on v9.

This generator transfers the residual ``s1 - v9`` onto v36 with a
regime-dependent gain:

    H = H_v36 + lambda(regime) * (H_s1 - H_v9)

It intentionally does not attenuate power and does not replace heavy geometry
neighbors.  The residual is small in heavy regions, so this is an aggressive
but controlled way to test whether the stronger s1 baseline can improve the
v36 geometry branch.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_s1style_geom_blend import geometry_cache


def regime_label(blocked):
    if blocked <= 0:
        return "clear"
    if blocked <= 25:
        return "light"
    if blocked <= 100:
        return "medium"
    return "heavy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--v9", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v9.npy"))
    parser.add_argument("--s1", default=str(PATHS.artifacts / "external/tmp_blend_inspect/s1.npy"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/s1_residual_regime_fusion_generate.json"))
    parser.add_argument("--clear-lambda", type=float, default=1.0)
    parser.add_argument("--light-lambda", type=float, default=1.0)
    parser.add_argument("--medium-lambda", type=float, default=0.75)
    parser.add_argument("--heavy-lambda", type=float, default=2.0)
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
        "delta_power_sum": 0.0,
        "base_power_sum": 0.0,
    }

    for start in range(0, len(test_blocked), args.batch):
        end = min(start + args.batch, len(test_blocked))
        base = np.asarray(v36[start:end], dtype=np.complex64)
        delta = np.asarray(s1[start:end], dtype=np.complex64) - np.asarray(v9[start:end], dtype=np.complex64)
        lam = np.empty((end - start,), dtype=np.float32)
        for j, blocked in enumerate(test_blocked[start:end]):
            label = regime_label(int(blocked))
            lam[j] = lambdas[label]
            stats[label] += 1
            stats["lambda_sum"] += float(lam[j])
        pred = base + lam[:, None, None, None] * delta
        pred = pred.astype(np.complex64)
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out[start:end] = pred
        stats["delta_power_sum"] += float(np.sum(np.abs(delta) ** 2, dtype=np.float64))
        stats["base_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        out.flush()
        print(f"generated {end}/{len(test_blocked)}", flush=True)

    stats["lambda_mean"] = stats["lambda_sum"] / max(len(test_blocked), 1)
    stats["delta_power_ratio"] = stats["delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    payload = {
        "method": "s1_residual_regime_fusion",
        "formula": "v36 + lambda(regime) * (s1 - v9)",
        "v9": args.v9,
        "s1": args.s1,
        "v36": args.v36,
        "lambdas": lambdas,
        "stats": stats,
        "output": args.output,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
