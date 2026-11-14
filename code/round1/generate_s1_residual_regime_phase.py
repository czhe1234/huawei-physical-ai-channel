#!/usr/bin/env python3
"""Generate a regime-wise residual phase calibration candidate.

v48 showed that transferring the residual ``s1 - v9`` onto v36 is useful:

    H = H_v36 + lambda(regime) * (H_s1 - H_v9)

v50 and v51 showed that simply increasing lambda or applying a large phase
rotation can over-shoot.  This generator keeps the v48 amplitude rule and only
applies small regime-wise phase rotations to the residual:

    H = H_v36 + lambda(regime) * exp(j * phi(regime)) * (H_s1 - H_v9)
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
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/s1_residual_regime_phase_generate.json"))
    parser.add_argument("--clear-lambda", type=float, default=1.0)
    parser.add_argument("--light-lambda", type=float, default=1.0)
    parser.add_argument("--medium-lambda", type=float, default=1.0)
    parser.add_argument("--heavy-lambda", type=float, default=4.0)
    parser.add_argument("--clear-phase-deg", type=float, default=0.0)
    parser.add_argument("--light-phase-deg", type=float, default=0.0)
    parser.add_argument("--medium-phase-deg", type=float, default=0.0)
    parser.add_argument("--heavy-phase-deg", type=float, default=0.0)
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
        "delta_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "base_delta_dot": {k: [0.0, 0.0] for k in lambdas},
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
            dot = np.vdot(base[j], raw_delta[j])
            stats["base_delta_dot"][label][0] += float(dot.real)
            stats["base_delta_dot"][label][1] += float(dot.imag)

        delta = raw_delta * ph[:, None, None, None]
        pred = base + lam[:, None, None, None] * delta
        pred = pred.astype(np.complex64)
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out[start:end] = pred
        out.flush()

        stats["base_power_sum"] += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
        stats["delta_power_sum"] += float(np.sum(np.abs(raw_delta) ** 2, dtype=np.float64))
        stats["pred_power_sum"] += float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        print(f"generated {end}/{len(test_blocked)}", flush=True)

    stats["lambda_mean"] = stats["lambda_sum"] / max(len(test_blocked), 1)
    stats["delta_power_ratio"] = stats["delta_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_power_ratio"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["base_delta_phase_deg"] = {}
    for label, pair in stats["base_delta_dot"].items():
        z = complex(pair[0], pair[1])
        stats["base_delta_phase_deg"][label] = float(np.angle(z, deg=True)) if abs(z) > 0 else None

    payload = {
        "method": "s1_residual_regime_phase",
        "formula": "v36 + lambda(regime) * exp(j * phi(regime)) * (s1 - v9)",
        "v9": args.v9,
        "s1": args.s1,
        "v36": args.v36,
        "lambdas": lambdas,
        "phase_deg": phase_deg,
        "stats": stats,
        "output": args.output,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
