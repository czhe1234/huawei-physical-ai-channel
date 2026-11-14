#!/usr/bin/env python3
"""Generate a high-risk multibounce-gated residual phase candidate.

v48 showed that ``v36 + lambda * (s1 - v9)`` is useful.  v50 showed that
blindly increasing lambda over-shoots.  This generator only applies a stronger,
phase-rotated residual to points whose two-bounce candidate score strongly
exceeds the best one-bounce score, while leaving the rest at the v48 rule.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_s1style_geom_blend import geometry_cache
from generate_s1_residual_regime_fusion import regime_label


def point_regime_lambdas(labels, clear, light, medium, heavy):
    vals = np.empty(len(labels), dtype=np.float32)
    vals[labels == "clear"] = clear
    vals[labels == "light"] = light
    vals[labels == "medium"] = medium
    vals[labels == "heavy"] = heavy
    return vals


def multibounce_confidence(paths):
    typ = paths["test_path_type"]
    valid = paths["test_valid"]
    score = paths["test_score"]
    one = np.where(valid & (typ == 1), score, -np.inf)
    two = np.where(valid & (typ == 2), score, -np.inf)
    best_one = np.max(one, axis=1)
    best_two = np.max(two, axis=1)
    best_two_idx = np.argmax(two, axis=1)
    excess = paths["test_excess_length"][np.arange(len(best_two_idx)), best_two_idx]
    return best_two - best_one, best_two, best_one, excess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/multibounce_path_candidates.npz"))
    parser.add_argument("--v9", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v9.npy"))
    parser.add_argument("--s1", default=str(PATHS.artifacts / "external/tmp_blend_inspect/s1.npy"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/s1_residual_phase_gate_generate.json"))
    parser.add_argument("--base-clear", type=float, default=1.0)
    parser.add_argument("--base-light", type=float, default=1.0)
    parser.add_argument("--base-medium", type=float, default=1.0)
    parser.add_argument("--base-heavy", type=float, default=4.0)
    parser.add_argument("--boost-medium", type=float, default=2.5)
    parser.add_argument("--boost-heavy", type=float, default=10.0)
    parser.add_argument("--conf-quantile", type=float, default=0.80)
    parser.add_argument("--boost-phase-deg", type=float, default=-20.0)
    parser.add_argument("--max-amp-ratio", type=float, default=10.0)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    v9 = np.load(args.v9, mmap_mode="r")
    s1 = np.load(args.s1, mmap_mode="r")
    v36 = np.load(args.v36, mmap_mode="r")
    if v9.shape != s1.shape or v9.shape != v36.shape:
        raise ValueError(f"shape mismatch: v9={v9.shape}, s1={s1.shape}, v36={v36.shape}")

    _train_blocked, test_blocked = geometry_cache(args.data, args.cache, False)
    labels = np.asarray([regime_label(int(x)) for x in test_blocked], dtype=object)
    paths = np.load(args.paths)
    conf, best_two, best_one, excess = multibounce_confidence(paths)
    conf_threshold = float(np.quantile(conf[np.isfinite(conf)], args.conf_quantile))
    boost = (
        (conf >= conf_threshold)
        & np.isfinite(best_two)
        & np.isfinite(best_one)
        & np.isfinite(excess)
        & np.isin(labels, ["medium", "heavy"])
    )

    lam = point_regime_lambdas(
        labels,
        args.base_clear,
        args.base_light,
        args.base_medium,
        args.base_heavy,
    ).astype(np.float64)
    lam[boost & (labels == "medium")] = args.boost_medium
    lam[boost & (labels == "heavy")] = args.boost_heavy
    phase = np.zeros(len(labels), dtype=np.float64)
    phase[boost] = np.deg2rad(args.boost_phase_deg)
    phase_factor = np.exp(1j * phase).astype(np.complex64)

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=v36.shape)
    stats = {
        "clear": int(np.sum(labels == "clear")),
        "light": int(np.sum(labels == "light")),
        "medium": int(np.sum(labels == "medium")),
        "heavy": int(np.sum(labels == "heavy")),
        "boost_count": int(np.sum(boost)),
        "boost_medium": int(np.sum(boost & (labels == "medium"))),
        "boost_heavy": int(np.sum(boost & (labels == "heavy"))),
        "conf_threshold": conf_threshold,
        "lambda_sum": float(np.sum(lam)),
        "effective_lambda_sum": 0.0,
        "capped_count": 0,
        "base_power_sum": 0.0,
        "delta_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "max_raw_amp_ratio": 0.0,
        "max_effective_amp_ratio": 0.0,
    }

    for start in range(0, len(labels), args.batch):
        end = min(start + args.batch, len(labels))
        base = np.asarray(v36[start:end], dtype=np.complex64)
        delta = np.asarray(s1[start:end], dtype=np.complex64) - np.asarray(v9[start:end], dtype=np.complex64)
        delta = delta * phase_factor[start:end, None, None, None]

        base_power = np.sum(np.abs(base) ** 2, axis=(1, 2, 3), dtype=np.float64)
        delta_power = np.sum(np.abs(delta) ** 2, axis=(1, 2, 3), dtype=np.float64)
        raw_amp = lam[start:end] * np.sqrt(delta_power / np.maximum(base_power, 1e-30))
        scale = np.ones_like(raw_amp)
        over = raw_amp > args.max_amp_ratio
        scale[over] = args.max_amp_ratio / np.maximum(raw_amp[over], 1e-30)
        eff_lam = lam[start:end] * scale

        pred = base + eff_lam[:, None, None, None].astype(np.float32) * delta
        pred = pred.astype(np.complex64)
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite generated block {start}:{end}")
        out[start:end] = pred
        out.flush()

        pred_power = np.sum(np.abs(pred) ** 2, axis=(1, 2, 3), dtype=np.float64)
        stats["effective_lambda_sum"] += float(np.sum(eff_lam))
        stats["capped_count"] += int(np.sum(over))
        stats["base_power_sum"] += float(np.sum(base_power))
        stats["delta_power_sum"] += float(np.sum(delta_power))
        stats["pred_power_sum"] += float(np.sum(pred_power))
        stats["max_raw_amp_ratio"] = max(stats["max_raw_amp_ratio"], float(np.max(raw_amp)))
        effective_amp = eff_lam * np.sqrt(delta_power / np.maximum(base_power, 1e-30))
        stats["max_effective_amp_ratio"] = max(stats["max_effective_amp_ratio"], float(np.max(effective_amp)))
        print(f"generated {end}/{len(labels)}", flush=True)

    n = max(len(labels), 1)
    stats["lambda_mean"] = stats["lambda_sum"] / n
    stats["effective_lambda_mean"] = stats["effective_lambda_sum"] / n
    stats["pred_power_ratio"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["delta_power_ratio"] = stats["delta_power_sum"] / max(stats["base_power_sum"], 1e-30)

    payload = {
        "method": "s1_residual_phase_gate",
        "formula": "v36 + capped_lambda(x) * (s1 - v9) * exp(j * phase(x))",
        "base_lambdas": {
            "clear": args.base_clear,
            "light": args.base_light,
            "medium": args.base_medium,
            "heavy": args.base_heavy,
        },
        "boost_lambdas": {
            "medium": args.boost_medium,
            "heavy": args.boost_heavy,
        },
        "conf_quantile": args.conf_quantile,
        "boost_phase_deg": args.boost_phase_deg,
        "max_amp_ratio": args.max_amp_ratio,
        "stats": stats,
        "output": args.output,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
