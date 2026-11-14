#!/usr/bin/env python3
from project_paths import PATHS
"""Diagnose whether geometric path candidates explain dominant delay taps."""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_delay_stats(channel, batch):
    n = channel.shape[0]
    s = channel.shape[-1]
    top_delay = np.empty(n, dtype=np.int16)
    centroid = np.empty(n, dtype=np.float32)
    total_power = np.empty(n, dtype=np.float64)
    top_frac = np.empty(n, dtype=np.float32)
    taps = np.arange(s, dtype=np.float64)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        h = np.asarray(channel[start:end])
        cir = np.fft.ifft(h, axis=-1)
        p = np.sum(np.abs(cir) ** 2, axis=(1, 2))
        total = np.sum(p, axis=1)
        top = np.argmax(p, axis=1)
        top_delay[start:end] = top.astype(np.int16)
        centroid[start:end] = (p @ taps / np.maximum(total, 1e-30)).astype(np.float32)
        total_power[start:end] = total.astype(np.float64)
        top_frac[start:end] = (p[np.arange(end - start), top] / np.maximum(total, 1e-30)).astype(np.float32)
        print(f"computed delay stats {end}/{n}", flush=True)
    return top_delay, centroid, total_power, top_frac


def summarize_errors(name, target_delay, cand_delay, cand_valid, cand_type):
    finite = cand_valid & np.isfinite(cand_delay)
    err = np.abs(cand_delay - target_delay[:, None])
    err = np.where(finite, err, np.inf)
    best_idx = np.argmin(err, axis=1)
    best_err = err[np.arange(len(err)), best_idx]
    los_err = err[:, 0]
    valid_best = np.isfinite(best_err)
    best_type = cand_type[np.arange(len(err)), best_idx]
    return {
        f"{name}_los_abs_err_p50": float(np.percentile(los_err[np.isfinite(los_err)], 50)),
        f"{name}_los_abs_err_p90": float(np.percentile(los_err[np.isfinite(los_err)], 90)),
        f"{name}_best_abs_err_p50": float(np.percentile(best_err[valid_best], 50)),
        f"{name}_best_abs_err_p90": float(np.percentile(best_err[valid_best], 90)),
        f"{name}_best_reflection_frac": float(np.mean(best_type[valid_best] == 1)),
        f"{name}_best_within_1tap_frac": float(np.mean(best_err[valid_best] <= 1.0)),
        f"{name}_best_within_2tap_frac": float(np.mean(best_err[valid_best] <= 2.0)),
    }


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--map-features", default=str(PATHS.artifacts / "shared_features/map_features.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/path_candidates_diagnostic.json"))
    parser.add_argument("--batch", type=int, default=25)
    args = parser.parse_args()

    channel = np.load(Path(args.data) / "Round1_Train_Channel.npy", mmap_mode="r")
    top_delay, centroid, total_power, top_frac = compute_delay_stats(channel, args.batch)
    paths = np.load(args.paths)
    cand_delay = paths["train_delay_bin"]
    cand_valid = paths["train_valid"]
    cand_type = paths["train_path_type"]
    cand_score = paths["train_score"]
    cand_excess = paths["train_excess_length"]
    mapz = np.load(args.map_features)
    map_feat = mapz["train_features"]
    names = list(mapz["feature_names"])
    name_to_idx = {n: i for i, n in enumerate(names)}

    payload = {
        "points": int(len(top_delay)),
        "top_delay_p10_p50_p90": [float(x) for x in np.percentile(top_delay, [10, 50, 90])],
        "centroid_p10_p50_p90": [float(x) for x in np.percentile(centroid, [10, 50, 90])],
        "top_frac_p10_p50_p90": [float(x) for x in np.percentile(top_frac, [10, 50, 90])],
        "log_power_p10_p50_p90": [float(x) for x in np.percentile(np.log10(np.maximum(total_power, 1e-30)), [10, 50, 90])],
    }
    payload.update(summarize_errors("top_delay", top_delay.astype(np.float32), cand_delay, cand_valid, cand_type))
    payload.update(summarize_errors("centroid", centroid.astype(np.float32), cand_delay, cand_valid, cand_type))

    best_delay_err = np.min(np.where(cand_valid, np.abs(cand_delay - top_delay[:, None]), np.inf), axis=1)
    top_ref_valid = cand_valid[:, 1]
    payload["correlations"] = {
        "log_power_vs_los_min_dist": corr(np.log10(np.maximum(total_power, 1e-30)), map_feat[:, name_to_idx["los_min_dist"]]),
        "log_power_vs_los_frac_r0p5": corr(np.log10(np.maximum(total_power, 1e-30)), map_feat[:, name_to_idx["los_frac_r0p5"]]),
        "log_power_vs_top_reflection_score": corr(np.log10(np.maximum(total_power, 1e-30)), np.where(top_ref_valid, cand_score[:, 1], np.nan)),
        "top_delay_err_vs_los_min_dist": corr(best_delay_err, map_feat[:, name_to_idx["los_min_dist"]]),
        "top_delay_err_vs_top_reflection_excess": corr(best_delay_err, np.where(top_ref_valid, cand_excess[:, 1], np.nan)),
    }

    # Which planes are often selected as the best delay explanation?
    err = np.where(cand_valid, np.abs(cand_delay - top_delay[:, None]), np.inf)
    best_idx = np.argmin(err, axis=1)
    plane_id = paths["train_plane_id"][np.arange(len(best_idx)), best_idx]
    ref_plane = plane_id[plane_id >= 0]
    if len(ref_plane):
        uniq, counts = np.unique(ref_plane, return_counts=True)
        order = np.argsort(-counts)[:20]
        payload["top_best_delay_planes"] = [
            {"plane_id": int(uniq[i]), "count": int(counts[i])}
            for i in order
        ]
    else:
        payload["top_best_delay_planes"] = []

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
