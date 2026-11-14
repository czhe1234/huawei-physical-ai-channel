#!/usr/bin/env python3
"""Diagnose Sionna mesh path candidates against training-channel delay stats."""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def compute_delay_stats(channel: np.ndarray, batch: int):
    n = channel.shape[0]
    s = channel.shape[-1]
    top_delay = np.empty(n, dtype=np.float32)
    centroid = np.empty(n, dtype=np.float32)
    total_power = np.empty(n, dtype=np.float64)
    top_frac = np.empty(n, dtype=np.float32)
    taps = np.arange(s, dtype=np.float64)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        h = np.asarray(channel[start:end])
        cir = np.fft.ifft(h, axis=-1)
        pdp = np.sum(np.abs(cir) ** 2, axis=(1, 2), dtype=np.float64)
        total = np.sum(pdp, axis=1)
        top = np.argmax(pdp, axis=1)
        top_delay[start:end] = top.astype(np.float32)
        centroid[start:end] = (pdp @ taps / np.maximum(total, 1e-30)).astype(np.float32)
        total_power[start:end] = total
        top_frac[start:end] = (pdp[np.arange(end - start), top] / np.maximum(total, 1e-30)).astype(np.float32)
        print(f"computed delay stats {end}/{n}", flush=True)
    return top_delay, centroid, total_power, top_frac


def percentile_dict(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def summarize_delay_errors(name, target, delay, valid, path_type):
    err = np.where(valid & np.isfinite(delay), np.abs(delay - target[:, None]), np.inf)
    best_idx = np.argmin(err, axis=1)
    best_err = err[np.arange(len(err)), best_idx]
    best_type = path_type[np.arange(len(err)), best_idx]
    best_finite = np.isfinite(best_err)
    los_err = err[:, 0]
    ref_err = np.min(err[:, 1:], axis=1) if err.shape[1] > 1 else np.full(len(err), np.inf)
    ref_finite = np.isfinite(ref_err)
    return {
        f"{name}_los_abs_err": percentile_dict(los_err),
        f"{name}_best_abs_err": percentile_dict(best_err[best_finite]),
        f"{name}_best_ref_abs_err": percentile_dict(ref_err[ref_finite]),
        f"{name}_best_reflection_frac": float(np.mean(best_type[best_finite] == 1)),
        f"{name}_best_within_0p5tap_frac": float(np.mean(best_err[best_finite] <= 0.5)),
        f"{name}_best_within_1tap_frac": float(np.mean(best_err[best_finite] <= 1.0)),
        f"{name}_best_within_2tap_frac": float(np.mean(best_err[best_finite] <= 2.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/sionna_path_candidates.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/sionna_path_candidates_diagnostic.json"))
    parser.add_argument("--batch", type=int, default=25)
    args = parser.parse_args()

    h = np.load(Path(args.data) / "Round1_Train_Channel.npy", mmap_mode="r")
    top_delay, centroid, total_power, top_frac = compute_delay_stats(h, args.batch)
    z = np.load(args.paths, allow_pickle=True)
    delay = z["train_delay_bin"]
    valid = z["train_valid"]
    path_type = z["train_path_type"]
    los_count = z["train_los_hit_count"]
    score = z["train_score"]
    excess = z["train_excess_length"]
    material_id = z["train_material_id"]
    mesh_id = z["train_mesh_id"]
    material_names = np.asarray(z["material_names"]).astype(str)
    mesh_object = np.asarray(z["mesh_object"]).astype(str)

    payload = {
        "points": int(len(top_delay)),
        "top_delay": percentile_dict(top_delay),
        "delay_centroid": percentile_dict(centroid),
        "top_delay_frac": percentile_dict(top_frac),
        "log10_power": percentile_dict(np.log10(np.maximum(total_power, 1e-30))),
    }
    payload.update(summarize_delay_errors("top_delay", top_delay, delay, valid, path_type))
    payload.update(summarize_delay_errors("centroid", centroid, delay, valid, path_type))

    best_err = np.min(np.where(valid, np.abs(delay - top_delay[:, None]), np.inf), axis=1)
    top_ref_valid = valid[:, 1]
    payload["correlations"] = {
        "log_power_vs_los_hit_count": corr(np.log10(np.maximum(total_power, 1e-30)), los_count),
        "log_power_vs_top_reflection_score": corr(
            np.log10(np.maximum(total_power, 1e-30)), np.where(top_ref_valid, score[:, 1], np.nan)
        ),
        "log_power_vs_top_reflection_excess": corr(
            np.log10(np.maximum(total_power, 1e-30)), np.where(top_ref_valid, excess[:, 1], np.nan)
        ),
        "best_delay_err_vs_los_hit_count": corr(best_err, los_count),
        "best_delay_err_vs_top_reflection_excess": corr(best_err, np.where(top_ref_valid, excess[:, 1], np.nan)),
    }

    ref_best_idx = 1 + np.argmin(np.where(valid[:, 1:], np.abs(delay[:, 1:] - top_delay[:, None]), np.inf), axis=1)
    ref_ok = np.isfinite(np.where(valid, np.abs(delay - top_delay[:, None]), np.inf)[np.arange(len(valid)), ref_best_idx])
    best_mesh = mesh_id[np.arange(len(mesh_id)), ref_best_idx][ref_ok]
    best_mat = material_id[np.arange(len(material_id)), ref_best_idx][ref_ok]
    mat_counts = Counter(int(x) for x in best_mat if int(x) >= 0)
    mesh_counts = Counter(int(x) for x in best_mesh if int(x) >= 0)
    payload["best_reflection_material_counts"] = {
        material_names[k]: int(v) for k, v in mat_counts.most_common()
    }
    payload["best_reflection_objects"] = [
        {"mesh_id": int(k), "object": mesh_object[k], "count": int(v)}
        for k, v in mesh_counts.most_common(20)
    ]

    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
