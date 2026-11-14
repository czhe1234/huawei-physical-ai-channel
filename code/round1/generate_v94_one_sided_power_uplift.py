#!/usr/bin/env python3
"""Generate v94: one-sided total-power calibration on top of v93.

v92/v93 improved the joint angle-delay support while preserving the base
per-point total power.  The base total power is not guaranteed to be optimal,
but directly replacing it with kNN pathloss is risky: train LOO diagnostics show
that the kNN total-power target would mostly reduce medium/heavy power, which is
close to the already failed dead-zone attenuation mode.

This version isolates total power as a separate variable:

  * keep v93 phase and joint AD support exactly unchanged;
  * estimate log total power from geometry-gated training neighbors;
  * only apply a small uplift when the neighbor target is above v93;
  * never downscale any point;
  * cap the amplitude gain tightly.

It is intentionally a low-risk probe for whether v93 is underpowered in the
easy/clear parts without destroying the newly useful joint AD support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import summarize
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, geom_effective_dist, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def parse_regime_values(text: str, default: float) -> dict[str, float]:
    values = {r: float(default) for r in REGIMES}
    if not text:
        return values
    for item in text.split(","):
        if not item:
            continue
        key, value = item.split(":")
        key = key.strip()
        if key not in values:
            raise ValueError(f"unknown regime {key!r}")
        values[key] = float(value)
    return values


def weighted_log_total_targets(
    train_total: np.ndarray,
    pos_train: np.ndarray,
    blocked_train: np.ndarray,
    pos_test: np.ndarray,
    blocked_test: np.ndarray,
    k: int,
    power: float,
) -> dict[str, np.ndarray]:
    valid = train_total > 0.0
    log_total = np.log(train_total + TINY)
    n = len(pos_test)
    pred = np.empty(n, dtype=np.float32)
    std = np.empty(n, dtype=np.float32)
    nn_dist = np.empty(n, dtype=np.float32)
    entropy = np.empty(n, dtype=np.float32)
    nn_weight = np.empty(n, dtype=np.float32)

    for i in range(n):
        eff = geom_effective_dist(pos_train, blocked_train, pos_test[i, :2], blocked_test[i]).copy()
        eff[~valid] = np.inf
        idx = np.argpartition(eff, k)[:k]
        idx = idx[np.argsort(eff[idx])]
        w = 1.0 / np.maximum(eff[idx], 0.30) ** power
        w = w / np.maximum(w.sum(), TINY)
        vals = log_total[idx]
        mu = float(np.sum(w * vals))
        pred[i] = mu
        std[i] = float(np.sqrt(np.sum(w * (vals - mu) ** 2)))
        nn_dist[i] = float(eff[idx[0]])
        entropy[i] = float(-(w * np.log(w + 1e-12)).sum())
        nn_weight[i] = float(w[0])
        if (i + 1) % 100 == 0:
            print(f"power targets {i+1}/{n}", flush=True)

    return {
        "log_total": pred,
        "std_log_total": std,
        "nn_dist": nn_dist,
        "entropy": entropy,
        "nn_weight": nn_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=ASSETS + "/cache_v93/v93_Round1_Test_Channel.npy")
    parser.add_argument("--output", default=str(PATHS.cache_dir(94) / "v94_Round1_Test_Channel.npy"))
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    parser.add_argument("--coll", default=ASSETS + "/truth_coll_power.npy")
    parser.add_argument("--report", default=ASSETS + "/v94_one_sided_power_uplift_generate.json")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument("--alpha", default="clear:0.10,light:0.06,medium:0.04,heavy:0.03")
    parser.add_argument("--std-full", type=float, default=0.50)
    parser.add_argument("--std-zero", type=float, default=2.00)
    parser.add_argument("--max-amp", type=float, default=1.08)
    parser.add_argument("--min-amp", type=float, default=1.00)
    parser.add_argument("--batch", type=int, default=25)
    args = parser.parse_args()

    data_dir = Path(args.data)
    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    pos_test = np.load(data_dir / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_test)
    alpha_table = parse_regime_values(args.alpha, 0.0)

    coll = np.load(args.coll, mmap_mode="r")
    train_total = np.asarray(coll.sum(axis=(1, 2, 3)), dtype=np.float64)
    targets = weighted_log_total_targets(
        train_total,
        pos_train,
        blocked_train,
        pos_test,
        blocked_test,
        args.k,
        args.power,
    )

    base = np.load(args.base, mmap_mode="r")
    if base.shape != (500, 256, 4, 192) or base.dtype != np.complex64:
        raise ValueError(f"unexpected base shape/dtype {base.shape} {base.dtype}")

    base_power = np.empty(500, dtype=np.float64)
    for start in range(0, 500, args.batch):
        end = min(start + args.batch, 500)
        h = np.asarray(base[start:end])
        base_power[start:end] = np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    log_base = np.log(base_power + TINY)

    raw_log_delta = targets["log_total"].astype(np.float64) - log_base
    positive_delta = np.maximum(raw_log_delta, 0.0)
    denom = max(args.std_zero - args.std_full, 1e-6)
    confidence = np.clip((args.std_zero - targets["std_log_total"].astype(np.float64)) / denom, 0.0, 1.0)
    alpha = np.array([alpha_table[str(r)] for r in regimes], dtype=np.float64)
    amp_scale = np.exp(0.5 * alpha * confidence * positive_delta)
    amp_scale = np.clip(amp_scale, args.min_amp, args.max_amp).astype(np.float32)

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)
    for start in range(0, 500, args.batch):
        end = min(start + args.batch, 500)
        h = np.asarray(base[start:end], dtype=np.complex64)
        scaled = h * amp_scale[start:end, None, None, None].astype(np.float32)
        if not (np.isfinite(scaled.real).all() and np.isfinite(scaled.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        out[start:end] = scaled.astype(np.complex64)
        out.flush()
        print(f"generated {end}/500", flush=True)

    out_power = base_power * amp_scale.astype(np.float64) ** 2
    by_regime = {}
    for r in REGIMES:
        mask = regimes == r
        by_regime[r] = {
            "count": int(mask.sum()),
            "amp_scale": summarize(amp_scale[mask]),
            "raw_log_target_minus_base": summarize(raw_log_delta[mask]),
            "positive_log_delta": summarize(positive_delta[mask]),
            "confidence": summarize(confidence[mask]),
        }

    payload = {
        "method": "v94_one_sided_power_uplift",
        "base": args.base,
        "operation": "one-sided per-point amplitude uplift; no downscale; v93 phase and joint AD support preserved",
        "neighbor": {"k": int(args.k), "power": float(args.power)},
        "alpha": alpha_table,
        "confidence": {"std_full": float(args.std_full), "std_zero": float(args.std_zero)},
        "clip": {"min_amp": float(args.min_amp), "max_amp": float(args.max_amp)},
        "target_stats": {
            "log_total": summarize(targets["log_total"]),
            "std_log_total": summarize(targets["std_log_total"]),
            "nn_dist": summarize(targets["nn_dist"]),
            "entropy": summarize(targets["entropy"]),
            "nn_weight": summarize(targets["nn_weight"]),
        },
        "overall": {
            "amp_scale": summarize(amp_scale),
            "raw_log_target_minus_base": summarize(raw_log_delta),
            "positive_log_delta": summarize(positive_delta),
            "confidence": summarize(confidence),
            "changed_points": int(np.sum(amp_scale > 1.0001)),
            "total_power_ratio_vs_base": float(np.sum(out_power) / np.maximum(np.sum(base_power), TINY)),
            "delta_nmse_to_base": float(np.sum(np.abs((amp_scale - 1.0).astype(np.float64)) ** 2 * base_power) / np.maximum(np.sum(base_power), TINY)),
        },
        "by_regime": by_regime,
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
