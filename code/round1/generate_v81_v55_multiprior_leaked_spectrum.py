#!/usr/bin/env python3
"""Generate v81: multi-prior leaked-spectrum calibration on top of v55.

This version uses several priors in one operation, while keeping v55 as the
phase/backbone prediction:

* official-like PAS prior from the leaked platform reshape
  B,256,4,192 -> B,2,16,8,4,192 -> B,2,192,16,4,8
  followed by FFT over (subcarrier, BS-H);
* spatial smoothness prior: sharp geometry-aware kNN predicts the target power
  shape in that leaked spectrum domain;
* H-internal structure prior: the gain is applied in the joint leaked spectrum,
  so pol/frequency/BS-H/UE/BS-V are not collapsed away;
* regime prior: clear/light are conservative, medium/heavy get stronger shape
  calibration;
* confidence prior: neighbor agreement gates the strength, so incoherent local
  targets do not fully overwrite v55.

No traditional ray tracing output is submitted.  Geometry is used only to pick
local neighbors and set coarse regime-dependent calibration strength.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np


MP = 2
MH = 16
MV = 8
N_ANT = 4
S_SC = 192
TINY = 1e-30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(PATHS.round1))
    p.add_argument("--geometry-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    p.add_argument("--base", default=str(PATHS.analysis / "probes/Round1_Test_Channel.npy"))
    p.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    p.add_argument("--out", default=str(PATHS.version_report_dir(81) / "v81_multiprior_leaked_spectrum_generate.json"))
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--power", type=float, default=3.0)
    p.add_argument("--min-dist", type=float, default=0.30)
    p.add_argument("--gain-clip", type=float, default=4.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--clear-eta", type=float, default=0.015)
    p.add_argument("--light-eta", type=float, default=0.030)
    p.add_argument("--medium-eta", type=float, default=0.060)
    p.add_argument("--heavy-eta", type=float, default=0.120)
    p.add_argument("--agreement-floor", type=float, default=0.30)
    p.add_argument("--agreement-power", type=float, default=1.0)
    p.add_argument("--progress", type=int, default=25)
    return p.parse_args()


def regime_label(blocked: float) -> str:
    if blocked <= 0:
        return "clear"
    if blocked <= 25:
        return "light"
    if blocked <= 100:
        return "medium"
    return "heavy"


def leaked_transform(h: np.ndarray) -> np.ndarray:
    """H domain B,256,4,192 -> leaked joint spectrum B,2,192,16,4,8."""

    arr = h.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    arr = arr.transpose(0, 1, 5, 2, 4, 3)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def inverse_leaked(spec: np.ndarray) -> np.ndarray:
    arr = np.fft.ifft2(spec, axes=(2, 3))
    arr = arr.transpose(0, 1, 3, 5, 4, 2)
    return arr.reshape(spec.shape[0], MP * MH * MV, N_ANT, S_SC).astype(np.complex64)


def effective_dist(pos_train: np.ndarray, blocked_train: np.ndarray, q_xy: np.ndarray, q_blocked: float) -> np.ndarray:
    d = np.hypot(pos_train[:, 0] - q_xy[0], pos_train[:, 1] - q_xy[1])
    los_q = q_blocked <= 0
    los_i = blocked_train <= 0
    return d * (1.0 + 1.0 * (los_i != los_q) + 0.5 * np.abs(blocked_train - q_blocked) / 80.0)


def idw_weights(eff: np.ndarray, k: int, power: float, min_dist: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(eff, k)[:k]
    idx = idx[np.argsort(eff[idx])]
    w = 1.0 / np.maximum(eff[idx], min_dist) ** power
    w = w / np.sum(w)
    return idx.astype(np.int64), w.astype(np.float32)


def mean_pair_cos(power_bank: np.ndarray) -> float:
    flat = power_bank.reshape(power_bank.shape[0], -1).astype(np.float64)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / np.maximum(norm, TINY)
    if flat.shape[0] <= 1:
        return 1.0
    gram = flat @ flat.T
    iu = np.triu_indices(flat.shape[0], 1)
    return float(np.mean(gram[iu]))


def main():
    args = parse_args()
    data = Path(args.data)
    pos_train = np.load(data / "Round1_Train_Pos.npy")
    pos_test = np.load(data / "Round1_Test_Pos.npy")
    train_h = np.load(data / "Round1_Train_Channel.npy", mmap_mode="r")
    base = np.load(args.base, mmap_mode="r")
    geo = np.load(args.geometry_cache)
    train_blocked = geo["train_blocked"].astype(np.float64)
    test_blocked = geo["test_blocked"].astype(np.float64)

    if base.shape != (500, 256, 4, 192):
        raise ValueError(f"unexpected base shape {base.shape}")

    eta_by_regime = {
        "clear": float(args.clear_eta),
        "light": float(args.light_eta),
        "medium": float(args.medium_eta),
        "heavy": float(args.heavy_eta),
    }

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)
    stats = {
        "clear": 0,
        "light": 0,
        "medium": 0,
        "heavy": 0,
        "eta_sum": 0.0,
        "eta_eff_sum": 0.0,
        "agreement_sum": 0.0,
        "base_power_sum": 0.0,
        "out_power_sum": 0.0,
        "gain_min": float("inf"),
        "gain_max": 0.0,
    }

    for i in range(len(pos_test)):
        label = regime_label(float(test_blocked[i]))
        stats[label] += 1
        eff = effective_dist(pos_train, train_blocked, pos_test[i, :2], float(test_blocked[i]))
        idx, w = idw_weights(eff, args.k, args.power, args.min_dist)

        neigh = np.asarray(train_h[idx], dtype=np.complex64)
        neigh_spec = leaked_transform(neigh)
        neigh_power = np.abs(neigh_spec) ** 2
        target = np.sum(w[:, None, None, None, None, None] * neigh_power, axis=0)

        base_h = np.asarray(base[i:i + 1], dtype=np.complex64)
        base_spec = leaked_transform(base_h)[0]
        base_power = np.abs(base_spec) ** 2

        # Use shape transfer only.  Keep the total leaked-domain power of v55
        # approximately fixed; v77 showed global scaling is dangerous.
        target_total = float(np.sum(target, dtype=np.float64))
        base_total = float(np.sum(base_power, dtype=np.float64))
        if target_total > 1e-30 and base_total > 1e-30:
            target = target * np.float32(base_total / target_total)

        agree = mean_pair_cos(neigh_power)
        agree_gate = max(0.0, (agree - float(args.agreement_floor)) / max(1e-6, 1.0 - float(args.agreement_floor)))
        agree_gate = float(agree_gate ** float(args.agreement_power))
        eta = float(eta_by_regime[label])
        eta_eff = eta * agree_gate

        delta = float(args.delta) * base_total / np.prod(base_power.shape)
        gain = ((target + delta) / np.maximum(base_power + delta, TINY)) ** (0.5 * eta_eff)
        gain = np.clip(gain, 1.0 / float(args.gain_clip), float(args.gain_clip)).astype(np.float32)
        pred_spec = (base_spec * gain).astype(np.complex64)
        pred = inverse_leaked(pred_spec[None])[0]
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite generated point {i}")
        out[i] = pred

        stats["eta_sum"] += eta
        stats["eta_eff_sum"] += eta_eff
        stats["agreement_sum"] += agree
        stats["base_power_sum"] += float(np.sum(np.abs(base_h) ** 2, dtype=np.float64))
        stats["out_power_sum"] += float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        stats["gain_min"] = min(stats["gain_min"], float(np.min(gain)))
        stats["gain_max"] = max(stats["gain_max"], float(np.max(gain)))

        if (i + 1) % int(args.progress) == 0 or i + 1 == len(pos_test):
            out.flush()
            print(f"generated {i + 1}/{len(pos_test)}", flush=True)

    n = max(len(pos_test), 1)
    stats["eta_mean"] = stats["eta_sum"] / n
    stats["eta_eff_mean"] = stats["eta_eff_sum"] / n
    stats["agreement_mean"] = stats["agreement_sum"] / n
    stats["power_ratio"] = stats["out_power_sum"] / max(stats["base_power_sum"], 1e-30)

    payload = {
        "method": "v81_v55_multiprior_leaked_spectrum",
        "base": args.base,
        "formula": "v55 phase/backbone + sharp geometry-kNN official-like leaked-spectrum shape gain",
        "leaked_transform": "B,256,4,192 -> B,2,16,8,4,192 -> B,2,192,16,4,8 -> fft2 axes=(2,3)",
        "k": int(args.k),
        "power": float(args.power),
        "gain_clip": float(args.gain_clip),
        "delta": float(args.delta),
        "eta_by_regime": eta_by_regime,
        "agreement_floor": float(args.agreement_floor),
        "agreement_power": float(args.agreement_power),
        "stats": stats,
        "output": args.output,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
