#!/usr/bin/env python3
"""Sweep simple calibration parameters for inverse-RT component assignment.

This script reuses the already extracted top angle-delay components from
``component_path_assignments.npz`` and tests whether the geometric path bins are
mis-calibrated by a small angle/delay offset.  It is an audit step: higher
assigned power means geometry explains more true training components.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from build_component_path_assignments import circular_abs_delta
from evaluate_local_ad_paths import MH, MV


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x.strip()]


def path_angle_bins(tx_dir, steering_scale, h_sign, v_sign, h_offset, v_offset):
    h = (h_sign * steering_scale * tx_dir[..., 1] / (2.0 * np.pi) * MH + h_offset) % MH
    v = (v_sign * steering_scale * (-tx_dir[..., 2]) / (2.0 * np.pi) * MV + v_offset) % MV
    return h.astype(np.float32), v.astype(np.float32)


def evaluate_one(comp, paths, args, steering_scale, h_sign, v_sign, delay_offset, h_offset, v_offset):
    hbin = comp["hbin"].astype(np.float32)
    vbin = comp["vbin"].astype(np.float32)
    sbin = comp["sbin"].astype(np.float32)
    comp_power = comp["component_power"].astype(np.float64)
    total_power = np.maximum(float(np.sum(comp_power)), 1e-30)

    valid = paths["train_valid"]
    path_delay = paths["train_delay_bin"].astype(np.float32) + np.float32(delay_offset)
    path_h, path_v = path_angle_bins(
        paths["train_tx_dir"].astype(np.float32),
        steering_scale,
        h_sign,
        v_sign,
        h_offset,
        v_offset,
    )

    de = np.abs(sbin[:, :, None] - path_delay[:, None, :])
    he = circular_abs_delta(hbin[:, :, None], path_h[:, None, :], MH)
    ve = circular_abs_delta(vbin[:, :, None], path_v[:, None, :], MV)
    valid3 = valid[:, None, :] & np.isfinite(de) & np.isfinite(he) & np.isfinite(ve)
    score = (
        (de / args.delay_weight) ** 2
        + (he / args.h_weight) ** 2
        + (ve / args.v_weight) ** 2
    )
    score = np.where(valid3, score, np.inf)
    best = np.argmin(score, axis=2)
    row = np.arange(best.shape[0])[:, None]
    col = np.arange(best.shape[1])[None, :]
    best_de = de[row, col, best]
    best_he = he[row, col, best]
    best_ve = ve[row, col, best]
    best_score = score[row, col, best]
    assigned = (
        np.isfinite(best_score)
        & (best_de <= args.assign_delay_threshold)
        & (best_he <= args.assign_h_threshold)
        & (best_ve <= args.assign_v_threshold)
    )
    best_type = paths["train_path_type"][row, best]
    return {
        "steering_scale": float(steering_scale),
        "steering_scale_pi": float(steering_scale / np.pi),
        "h_sign": int(h_sign),
        "v_sign": int(v_sign),
        "delay_offset": float(delay_offset),
        "h_offset": float(h_offset),
        "v_offset": float(v_offset),
        "assigned_component_frac": float(np.mean(assigned)),
        "assigned_top1_frac": float(np.mean(assigned[:, 0])),
        "assigned_power_frac": float(np.sum(comp_power[assigned]) / total_power),
        "assigned_los_power_frac": float(np.sum(comp_power[assigned & (best_type == 0)]) / total_power),
        "assigned_reflection_power_frac": float(np.sum(comp_power[assigned & (best_type == 1)]) / total_power),
        "delay_err_p50": float(np.percentile(best_de, 50)),
        "delay_err_p90": float(np.percentile(best_de, 90)),
        "h_err_p50": float(np.percentile(best_he, 50)),
        "h_err_p90": float(np.percentile(best_he, 90)),
        "v_err_p50": float(np.percentile(best_ve, 50)),
        "v_err_p90": float(np.percentile(best_ve, 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", default=str(PATHS.artifacts / "shared_features/component_path_assignments.npz"))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/inverse_rt_calibration_sweep.json"))
    parser.add_argument("--scale-pi-values", type=parse_floats, default="1.20,1.30,1.40,1.50,1.60,1.70,1.80")
    parser.add_argument("--delay-offsets", type=parse_floats, default="-1.0,-0.5,0.0,0.5,1.0")
    parser.add_argument("--h-offsets", type=parse_floats, default="-1.0,0.0,1.0")
    parser.add_argument("--v-offsets", type=parse_floats, default="-1.0,0.0,1.0")
    parser.add_argument("--h-signs", type=parse_ints, default="-1,1")
    parser.add_argument("--v-signs", type=parse_ints, default="1,-1")
    parser.add_argument("--delay-weight", type=float, default=2.0)
    parser.add_argument("--h-weight", type=float, default=2.5)
    parser.add_argument("--v-weight", type=float, default=1.5)
    parser.add_argument("--assign-delay-threshold", type=float, default=2.0)
    parser.add_argument("--assign-h-threshold", type=float, default=3.0)
    parser.add_argument("--assign-v-threshold", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    comp = np.load(args.components)
    paths = np.load(args.paths)
    results = []
    combos = [
        (scale_pi * np.pi, h_sign, v_sign, delay_offset, h_offset, v_offset)
        for scale_pi in args.scale_pi_values
        for h_sign in args.h_signs
        for v_sign in args.v_signs
        for delay_offset in args.delay_offsets
        for h_offset in args.h_offsets
        for v_offset in args.v_offsets
    ]
    for i, combo in enumerate(combos):
        results.append(evaluate_one(comp, paths, args, *combo))
        if (i + 1) % 100 == 0 or i + 1 == len(combos):
            best = max(results, key=lambda r: r["assigned_power_frac"])
            print(
                f"swept {i + 1}/{len(combos)} best_power={best['assigned_power_frac']:.6f} "
                f"scale={best['steering_scale_pi']:.2f} delay={best['delay_offset']:.2f} "
                f"hoff={best['h_offset']:.1f} voff={best['v_offset']:.1f}",
                flush=True,
            )

    results.sort(key=lambda r: r["assigned_power_frac"], reverse=True)
    payload = {
        "args": vars(args),
        "combos": len(combos),
        "best": results[0],
        "top_results": results[: args.top],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
