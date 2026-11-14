#!/usr/bin/env python3
"""Evaluate conservative takeover rules for two-bounce path assignments.

The plain multibounce assignment lets two-bounce candidates compete directly
with LOS/one-bounce candidates.  Diagnostics showed that this can steal some
components from more reliable one-bounce matches.  This script keeps the
single-bounce assignment unless a two-bounce candidate is clearly better.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np


def score_from_errors(delay_err, h_err, v_err, weights):
    return (
        (delay_err / weights[0]) ** 2
        + (h_err / weights[1]) ** 2
        + (v_err / weights[2]) ** 2
    )


def summarize(name, best_path, delay_err, h_err, v_err, comp_power, path_type, thresholds):
    row = np.arange(best_path.shape[0])[:, None]
    best_type = path_type[row, best_path]
    assigned = (delay_err <= thresholds[0]) & (h_err <= thresholds[1]) & (v_err <= thresholds[2])
    denom = float(np.maximum(np.sum(comp_power), 1e-30))
    out = {
        "name": name,
        "assigned_component_frac": float(np.mean(assigned)),
        "assigned_power_frac": float(np.sum(comp_power[assigned]) / denom),
        "assigned_top1_frac": float(np.mean(assigned[:, 0])),
        "delay_err_p50": float(np.percentile(delay_err, 50)),
        "delay_err_p90": float(np.percentile(delay_err, 90)),
        "h_err_p50": float(np.percentile(h_err, 50)),
        "h_err_p90": float(np.percentile(h_err, 90)),
        "v_err_p50": float(np.percentile(v_err, 50)),
        "v_err_p90": float(np.percentile(v_err, 90)),
    }
    for t, label in [(0, "los"), (1, "one_bounce"), (2, "two_bounce")]:
        mask = assigned & (best_type == t)
        raw = best_type == t
        out[f"assigned_{label}_component_frac"] = float(np.mean(mask))
        out[f"assigned_{label}_power_frac"] = float(np.sum(comp_power[mask]) / denom)
        out[f"raw_{label}_component_frac"] = float(np.mean(raw))
        out[f"raw_{label}_power_frac"] = float(np.sum(comp_power[raw]) / denom)
        out[f"top1_{label}_frac"] = float(np.mean(mask[:, 0]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default=str(PATHS.artifacts / "shared_features/component_path_assignments.npz"))
    parser.add_argument("--multi", default=str(PATHS.artifacts / "shared_features/multibounce_component_path_assignments.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/conservative_multibounce_assignment.json"))
    parser.add_argument("--relative-improvement", type=float, default=0.35)
    parser.add_argument("--absolute-improvement", type=float, default=0.20)
    parser.add_argument("--delay-improvement", type=float, default=0.25)
    parser.add_argument("--h-improvement", type=float, default=0.25)
    parser.add_argument("--v-improvement", type=float, default=0.10)
    parser.add_argument("--max-two-delay", type=float, default=1.5)
    parser.add_argument("--max-two-h", type=float, default=2.0)
    parser.add_argument("--max-two-v", type=float, default=1.5)
    args = parser.parse_args()

    s = np.load(args.single)
    m = np.load(args.multi)
    comp_power = s["component_power"]
    thresholds = m["thresholds"].astype(np.float32)
    weights = m["weights"].astype(np.float32)

    s_path = s["best_path"].astype(np.int16)
    s_de = s["delay_err"].astype(np.float32)
    s_he = s["h_err"].astype(np.float32)
    s_ve = s["v_err"].astype(np.float32)
    s_score = score_from_errors(s_de, s_he, s_ve, weights)

    # Reconstruct best two-bounce candidate from path bins and component bins.
    path_type = m["path_type"]
    two_mask = m["path_valid"] & (path_type == 2)
    hbin = m["hbin"].astype(np.float32)
    vbin = m["vbin"].astype(np.float32)
    sbin = m["sbin"].astype(np.float32)
    path_delay = m["path_delay_bin"].astype(np.float32)
    path_h = m["path_h_bin"].astype(np.float32)
    path_v = m["path_v_bin"].astype(np.float32)

    de = np.abs(sbin[:, :, None] - path_delay[:, None, :])
    dh_raw = np.abs(hbin[:, :, None] - path_h[:, None, :])
    he = np.minimum(dh_raw, 16 - dh_raw)
    dv_raw = np.abs(vbin[:, :, None] - path_v[:, None, :])
    ve = np.minimum(dv_raw, 4 - dv_raw)
    valid3 = two_mask[:, None, :] & np.isfinite(de) & np.isfinite(he) & np.isfinite(ve)
    t_score = score_from_errors(de, he, ve, weights)
    t_score = np.where(valid3, t_score, np.inf)
    t_path = np.argmin(t_score, axis=2).astype(np.int16)
    row = np.arange(t_path.shape[0])[:, None]
    col = np.arange(t_path.shape[1])[None, :]
    t_score_best = t_score[row, col, t_path].astype(np.float32)
    t_de = de[row, col, t_path].astype(np.float32)
    t_he = he[row, col, t_path].astype(np.float32)
    t_ve = ve[row, col, t_path].astype(np.float32)

    clearly_better = (
        (t_score_best < s_score * (1.0 - args.relative_improvement))
        & ((s_score - t_score_best) > args.absolute_improvement)
        & ((s_de - t_de) > args.delay_improvement)
        & ((s_he - t_he) > args.h_improvement)
        & ((s_ve - t_ve) > args.v_improvement)
        & (t_de <= args.max_two_delay)
        & (t_he <= args.max_two_h)
        & (t_ve <= args.max_two_v)
    )

    c_path = np.where(clearly_better, t_path, s_path).astype(np.int16)
    c_de = np.where(clearly_better, t_de, s_de).astype(np.float32)
    c_he = np.where(clearly_better, t_he, s_he).astype(np.float32)
    c_ve = np.where(clearly_better, t_ve, s_ve).astype(np.float32)

    rows = [
        summarize("single_base", s_path, s_de, s_he, s_ve, comp_power, s["path_type"], thresholds),
        summarize("multi_direct", m["best_path"], m["delay_err"], m["h_err"], m["v_err"], comp_power, path_type, thresholds),
        summarize("conservative_takeover", c_path, c_de, c_he, c_ve, comp_power, path_type, thresholds),
    ]
    takeover_power = float(np.sum(comp_power[clearly_better]) / np.maximum(np.sum(comp_power), 1e-30))
    payload = {
        "args": vars(args),
        "takeover_component_frac": float(np.mean(clearly_better)),
        "takeover_power_frac": takeover_power,
        "takeover_top1_frac": float(np.mean(clearly_better[:, 0])),
        "summaries": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
