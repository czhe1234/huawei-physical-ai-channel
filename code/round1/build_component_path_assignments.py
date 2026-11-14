#!/usr/bin/env python3
from project_paths import PATHS
"""Assign dominant BS angle-delay components to geometric path candidates.

This is a diagnostic/cache-building step for path-family modeling. It extracts
top BS angle-delay components from training channels, predicts angle/delay bins
for LOS/specular path candidates, and records the best candidate for each
component.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_local_ad_paths import MH, MV, S_SC, ad_transform, flat_components


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x.strip()]


def circular_abs_delta(values, centers, period):
    delta = np.abs(values - centers)
    return np.minimum(delta, period - delta)


def top_components_from_power(power, count):
    flat = power.reshape(power.shape[0], -1)
    count = min(count, flat.shape[1])
    ids = np.argpartition(flat, -count, axis=1)[:, -count:]
    rows = np.arange(flat.shape[0])[:, None]
    order = np.argsort(flat[rows, ids], axis=1)[:, ::-1]
    ids = ids[rows, order]
    vals = flat[rows, ids]
    return ids.astype(np.int32), vals.astype(np.float32)


def path_angle_bins(tx_dir, steering_scale, h_sign, v_sign):
    """Map path departure direction to FFT angle-bin coordinates."""

    h = (h_sign * steering_scale * tx_dir[..., 1] / (2.0 * np.pi) * MH) % MH
    # Existing steering experiments used v axis = -unit_z.
    v = (v_sign * steering_scale * (-tx_dir[..., 2]) / (2.0 * np.pi) * MV) % MV
    return h.astype(np.float32), v.astype(np.float32)


def assign_components(component_ids, path_delay, path_h, path_v, valid, weights):
    hbin, vbin, sbin = flat_components(component_ids.reshape(-1))
    hbin = hbin.reshape(component_ids.shape).astype(np.float32)
    vbin = vbin.reshape(component_ids.shape).astype(np.float32)
    sbin = sbin.reshape(component_ids.shape).astype(np.float32)

    de = np.abs(sbin[:, :, None] - path_delay[:, None, :])
    he = circular_abs_delta(hbin[:, :, None], path_h[:, None, :], MH)
    ve = circular_abs_delta(vbin[:, :, None], path_v[:, None, :], MV)
    valid3 = valid[:, None, :] & np.isfinite(de) & np.isfinite(he) & np.isfinite(ve)
    score = (
        (de / weights["delay"]) ** 2
        + (he / weights["h"]) ** 2
        + (ve / weights["v"]) ** 2
    )
    score = np.where(valid3, score, np.inf)
    best = np.argmin(score, axis=2).astype(np.int16)
    row = np.arange(component_ids.shape[0])[:, None]
    col = np.arange(component_ids.shape[1])[None, :]
    return {
        "hbin": hbin.astype(np.int16),
        "vbin": vbin.astype(np.int16),
        "sbin": sbin.astype(np.int16),
        "best_path": best,
        "score": score[row, col, best].astype(np.float32),
        "delay_err": de[row, col, best].astype(np.float32),
        "h_err": he[row, col, best].astype(np.float32),
        "v_err": ve[row, col, best].astype(np.float32),
    }


def summarize_assignment(name, assignment, comp_power, total_power, path_type, thresholds):
    best = assignment["best_path"]
    row = np.arange(best.shape[0])[:, None]
    best_type = path_type[row, best]
    assigned = (
        (assignment["delay_err"] <= thresholds["delay"])
        & (assignment["h_err"] <= thresholds["h"])
        & (assignment["v_err"] <= thresholds["v"])
    )
    power_sum = np.maximum(np.sum(comp_power), 1e-30)
    assigned_power = float(np.sum(comp_power[assigned]) / power_sum)
    los_power = float(np.sum(comp_power[assigned & (best_type == 0)]) / power_sum)
    refl_power = float(np.sum(comp_power[assigned & (best_type == 1)]) / power_sum)
    top1_assigned = float(np.mean(assigned[:, 0]))
    top1_reflection = float(np.mean(assigned[:, 0] & (best_type[:, 0] == 1)))
    return {
        "name": name,
        "component_power_fraction_of_total_p50": float(np.percentile(np.sum(comp_power, axis=1) / np.maximum(total_power, 1e-30), 50)),
        "component_power_fraction_of_total_p90": float(np.percentile(np.sum(comp_power, axis=1) / np.maximum(total_power, 1e-30), 90)),
        "assigned_component_frac": float(np.mean(assigned)),
        "assigned_top1_frac": top1_assigned,
        "assigned_power_frac": assigned_power,
        "assigned_los_power_frac": los_power,
        "assigned_reflection_power_frac": refl_power,
        "top1_reflection_frac": top1_reflection,
        "delay_err_p50": float(np.percentile(assignment["delay_err"], 50)),
        "delay_err_p90": float(np.percentile(assignment["delay_err"], 90)),
        "h_err_p50": float(np.percentile(assignment["h_err"], 50)),
        "h_err_p90": float(np.percentile(assignment["h_err"], 90)),
        "v_err_p50": float(np.percentile(assignment["v_err"], 50)),
        "v_err_p90": float(np.percentile(assignment["v_err"], 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--output", default=str(PATHS.artifacts / "shared_features/component_path_assignments.npz"))
    parser.add_argument("--summary", default=str(PATHS.analysis / "results/json/component_path_assignments_summary.json"))
    parser.add_argument("--component-count", type=int, default=64)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--h-signs", default="1,-1")
    parser.add_argument("--v-signs", default="1,-1")
    parser.add_argument("--delay-weight", type=float, default=2.0)
    parser.add_argument("--h-weight", type=float, default=2.5)
    parser.add_argument("--v-weight", type=float, default=1.5)
    parser.add_argument("--assign-delay-threshold", type=float, default=2.0)
    parser.add_argument("--assign-h-threshold", type=float, default=3.0)
    parser.add_argument("--assign-v-threshold", type=float, default=2.0)
    args = parser.parse_args()

    channel = np.load(Path(args.data) / "Round1_Train_Channel.npy", mmap_mode="r")
    paths = np.load(args.paths)
    path_delay = paths["train_delay_bin"].astype(np.float32)
    path_valid = paths["train_valid"]
    path_type = paths["train_path_type"]
    tx_dir = paths["train_tx_dir"].astype(np.float32)

    n = channel.shape[0]
    k = int(args.component_count)
    comp_ids = np.empty((n, k), dtype=np.int32)
    comp_power = np.empty((n, k), dtype=np.float32)
    total_power = np.empty(n, dtype=np.float64)
    for start in range(0, n, args.batch):
        end = min(start + args.batch, n)
        ad = ad_transform(np.asarray(channel[start:end]))
        power = np.sum(np.abs(ad) ** 2, axis=(1, 4), dtype=np.float64)
        ids, vals = top_components_from_power(power, k)
        comp_ids[start:end] = ids
        comp_power[start:end] = vals
        total_power[start:end] = np.sum(power, axis=(1, 2, 3))
        print(f"extracted components {end}/{n}", flush=True)

    weights = {"delay": args.delay_weight, "h": args.h_weight, "v": args.v_weight}
    thresholds = {
        "delay": args.assign_delay_threshold,
        "h": args.assign_h_threshold,
        "v": args.assign_v_threshold,
    }

    variants = []
    for h_sign in parse_ints(args.h_signs):
        for v_sign in parse_ints(args.v_signs):
            ph, pv = path_angle_bins(tx_dir, args.steering_scale, h_sign, v_sign)
            assign = assign_components(comp_ids, path_delay, ph, pv, path_valid, weights)
            summary = summarize_assignment(
                f"h{h_sign}_v{v_sign}",
                assign,
                comp_power,
                total_power,
                path_type,
                thresholds,
            )
            summary.update({"h_sign": int(h_sign), "v_sign": int(v_sign)})
            variants.append((summary["assigned_power_frac"], summary, assign, ph, pv))

    variants.sort(key=lambda x: x[0], reverse=True)
    best_score, best_summary, best_assign, best_path_h, best_path_v = variants[0]
    print("best assignment variant:", json.dumps(best_summary, indent=2), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        component_ids=comp_ids,
        component_power=comp_power,
        total_power=total_power.astype(np.float32),
        hbin=best_assign["hbin"],
        vbin=best_assign["vbin"],
        sbin=best_assign["sbin"],
        best_path=best_assign["best_path"],
        assignment_score=best_assign["score"],
        delay_err=best_assign["delay_err"],
        h_err=best_assign["h_err"],
        v_err=best_assign["v_err"],
        path_h_bin=best_path_h.astype(np.float32),
        path_v_bin=best_path_v.astype(np.float32),
        path_delay_bin=path_delay.astype(np.float32),
        path_type=path_type.astype(np.int8),
        path_valid=path_valid,
        h_sign=np.asarray(best_summary["h_sign"], dtype=np.int8),
        v_sign=np.asarray(best_summary["v_sign"], dtype=np.int8),
        thresholds=np.asarray([thresholds["delay"], thresholds["h"], thresholds["v"]], dtype=np.float32),
        weights=np.asarray([weights["delay"], weights["h"], weights["v"]], dtype=np.float32),
    )

    payload = {
        "output": str(out),
        "component_count": k,
        "weights": weights,
        "thresholds": thresholds,
        "best_variant": best_summary,
        "variants": [v[1] for v in variants],
    }
    with open(args.summary, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
