#!/usr/bin/env python3
"""Build a calibrated component-to-path assignment cache.

This takes extracted components from an existing assignment cache and recomputes
the geometric path bins with angle/delay offsets found by
``sweep_inverse_rt_calibration.py``.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from build_component_path_assignments import circular_abs_delta
from evaluate_local_ad_paths import MH, MV


def assign_components(component_ids, path_delay, path_h, path_v, valid, weights):
    hbin = component_ids["hbin"].astype(np.float32)
    vbin = component_ids["vbin"].astype(np.float32)
    sbin = component_ids["sbin"].astype(np.float32)
    de = np.abs(sbin[:, :, None] - path_delay[:, None, :])
    he = circular_abs_delta(hbin[:, :, None], path_h[:, None, :], MH)
    ve = circular_abs_delta(vbin[:, :, None], path_v[:, None, :], MV)
    valid3 = valid[:, None, :] & np.isfinite(de) & np.isfinite(he) & np.isfinite(ve)
    score = (de / weights[0]) ** 2 + (he / weights[1]) ** 2 + (ve / weights[2]) ** 2
    score = np.where(valid3, score, np.inf)
    best = np.argmin(score, axis=2).astype(np.int16)
    row = np.arange(best.shape[0])[:, None]
    col = np.arange(best.shape[1])[None, :]
    return {
        "best_path": best,
        "score": score[row, col, best].astype(np.float32),
        "delay_err": de[row, col, best].astype(np.float32),
        "h_err": he[row, col, best].astype(np.float32),
        "v_err": ve[row, col, best].astype(np.float32),
    }


def path_angle_bins(tx_dir, steering_scale, h_sign, v_sign, h_offset, v_offset):
    h = (h_sign * steering_scale * tx_dir[..., 1] / (2.0 * np.pi) * MH + h_offset) % MH
    v = (v_sign * steering_scale * (-tx_dir[..., 2]) / (2.0 * np.pi) * MV + v_offset) % MV
    return h.astype(np.float32), v.astype(np.float32)


def summarize(cache, assign, path_type, thresholds):
    best = assign["best_path"]
    row = np.arange(best.shape[0])[:, None]
    best_type = path_type[row, best]
    assigned = (
        (assign["delay_err"] <= thresholds[0])
        & (assign["h_err"] <= thresholds[1])
        & (assign["v_err"] <= thresholds[2])
        & np.isfinite(assign["score"])
    )
    comp_power = cache["component_power"].astype(np.float64)
    total = max(float(np.sum(comp_power)), 1e-30)
    return {
        "assigned_component_frac": float(np.mean(assigned)),
        "assigned_top1_frac": float(np.mean(assigned[:, 0])),
        "assigned_power_frac": float(np.sum(comp_power[assigned]) / total),
        "assigned_los_power_frac": float(np.sum(comp_power[assigned & (best_type == 0)]) / total),
        "assigned_reflection_power_frac": float(np.sum(comp_power[assigned & (best_type == 1)]) / total),
        "delay_err_p50": float(np.percentile(assign["delay_err"], 50)),
        "delay_err_p90": float(np.percentile(assign["delay_err"], 90)),
        "h_err_p50": float(np.percentile(assign["h_err"], 50)),
        "h_err_p90": float(np.percentile(assign["h_err"], 90)),
        "v_err_p50": float(np.percentile(assign["v_err"], 50)),
        "v_err_p90": float(np.percentile(assign["v_err"], 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(PATHS.artifacts / "shared_features/component_path_assignments.npz"))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--output", default=str(PATHS.artifacts / "shared_features/component_path_assignments_calibrated_v1.npz"))
    parser.add_argument("--summary", default=str(PATHS.analysis / "results/json/component_path_assignments_calibrated_v1_summary.json"))
    parser.add_argument("--steering-scale", type=float, default=4.71238898038469)
    parser.add_argument("--h-sign", type=int, default=-1)
    parser.add_argument("--v-sign", type=int, default=1)
    parser.add_argument("--delay-offset", type=float, default=-1.0)
    parser.add_argument("--h-offset", type=float, default=-1.0)
    parser.add_argument("--v-offset", type=float, default=1.0)
    parser.add_argument("--delay-weight", type=float, default=2.0)
    parser.add_argument("--h-weight", type=float, default=2.5)
    parser.add_argument("--v-weight", type=float, default=1.5)
    parser.add_argument("--assign-delay-threshold", type=float, default=2.0)
    parser.add_argument("--assign-h-threshold", type=float, default=3.0)
    parser.add_argument("--assign-v-threshold", type=float, default=2.0)
    args = parser.parse_args()

    base = np.load(args.base)
    paths = np.load(args.paths)
    weights = np.asarray([args.delay_weight, args.h_weight, args.v_weight], dtype=np.float32)
    thresholds = np.asarray(
        [args.assign_delay_threshold, args.assign_h_threshold, args.assign_v_threshold],
        dtype=np.float32,
    )
    path_delay = paths["train_delay_bin"].astype(np.float32) + np.float32(args.delay_offset)
    path_h, path_v = path_angle_bins(
        paths["train_tx_dir"].astype(np.float32),
        args.steering_scale,
        args.h_sign,
        args.v_sign,
        args.h_offset,
        args.v_offset,
    )
    component = {
        "hbin": base["hbin"],
        "vbin": base["vbin"],
        "sbin": base["sbin"],
    }
    assign = assign_components(component, path_delay, path_h, path_v, paths["train_valid"], weights)
    summary = summarize(base, assign, paths["train_path_type"], thresholds)
    payload = {
        "output": args.output,
        "source_components": args.base,
        "paths": args.paths,
        "calibration": {
            "steering_scale": args.steering_scale,
            "steering_scale_pi": args.steering_scale / np.pi,
            "h_sign": args.h_sign,
            "v_sign": args.v_sign,
            "delay_offset": args.delay_offset,
            "h_offset": args.h_offset,
            "v_offset": args.v_offset,
        },
        "weights": {
            "delay": args.delay_weight,
            "h": args.h_weight,
            "v": args.v_weight,
        },
        "thresholds": {
            "delay": args.assign_delay_threshold,
            "h": args.assign_h_threshold,
            "v": args.assign_v_threshold,
        },
        "summary": summary,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        component_ids=base["component_ids"],
        component_power=base["component_power"],
        total_power=base["total_power"],
        hbin=base["hbin"],
        vbin=base["vbin"],
        sbin=base["sbin"],
        best_path=assign["best_path"],
        assignment_score=assign["score"],
        delay_err=assign["delay_err"],
        h_err=assign["h_err"],
        v_err=assign["v_err"],
        path_h_bin=path_h.astype(np.float32),
        path_v_bin=path_v.astype(np.float32),
        path_delay_bin=path_delay.astype(np.float32),
        path_type=paths["train_path_type"].astype(np.int8),
        path_valid=paths["train_valid"],
        h_sign=np.asarray(args.h_sign, dtype=np.int8),
        v_sign=np.asarray(args.v_sign, dtype=np.int8),
        thresholds=thresholds,
        weights=weights,
        calibration=np.asarray(
            [args.steering_scale, args.delay_offset, args.h_offset, args.v_offset],
            dtype=np.float32,
        ),
    )
    Path(args.summary).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
