#!/usr/bin/env python3
"""Fit an official-only global delay law on high-depth point-cloud paths."""

from __future__ import annotations

import json
from itertools import product

import numpy as np
from scipy.stats import pearsonr, spearmanr

import analyze_v208_strict_historical_replay as v208
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 343
VERSION = "v343_pointcloud_global_delay_law"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_OUT = OUT_DIR / "v343_pointcloud_test_d.npy"

PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
LOCAL_TRUTH = v208.TEST_TRUTH

N_DELAY = 192
BATCH = 8
SCORE_SCALE = (0.25, 0.5, 0.75, 1.0)
LENGTH_POWER = (0.0, 0.75, 1.5)
DEPTH_PENALTY = (0.0, 0.5, 1.0, 1.5)
DIRECT_BIAS = (0.0, 1.5, 3.0)
SMOOTH_SIGMA = (0.0, 0.5, 1.0)


def link_normalized_global_d(path) -> np.ndarray:
    h = np.load(path, mmap_mode="r")
    output = np.empty((len(h), N_DELAY), dtype=np.float32)
    for start in range(0, len(h), BATCH):
        end = min(start + BATCH, len(h))
        block = np.asarray(h[start:end], dtype=np.complex64)
        power = np.abs(np.fft.ifft(block, axis=-1)).astype(np.float32) ** 2
        power /= np.maximum(power.sum(axis=-1, keepdims=True), 1e-30)
        output[start:end] = power.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
    output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-30)
    return output


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1, dtype=np.float64)
    norm = np.sqrt(
        np.sum(a * a, axis=1, dtype=np.float64)
        * np.sum(b * b, axis=1, dtype=np.float64)
    )
    return (numerator / np.maximum(norm, 1e-30)).astype(np.float32)


def smooth_circular(value: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return value
    frequency = np.fft.rfftfreq(N_DELAY)
    multiplier = np.exp(-2.0 * (np.pi * sigma * frequency) ** 2)
    output = np.fft.irfft(
        np.fft.rfft(value, axis=1) * multiplier[None, :], n=N_DELAY, axis=1
    ).astype(np.float32)
    return np.maximum(output, 0.0)


def render(
    valid: np.ndarray,
    delay: np.ndarray,
    length: np.ndarray,
    depth: np.ndarray,
    score: np.ndarray,
    score_scale: float,
    length_power: float,
    depth_penalty: float,
    direct_bias: float,
    sigma: float,
) -> np.ndarray:
    log_weight = (
        score_scale * np.clip(score, -24.0, 4.0)
        - length_power * np.log(np.maximum(length, 1.0))
        - depth_penalty * depth.astype(np.float32)
        + direct_bias * (depth == 0)
    )
    log_weight[~valid] = -1e9
    row_max = np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(np.clip(log_weight - row_max, -60.0, 0.0)).astype(np.float32)
    weight[~valid] = 0.0
    empty = weight.sum(axis=1) <= 1e-20
    if np.any(empty):
        fallback = np.argmin(np.where(valid[empty], length[empty], np.inf), axis=1)
        weight[np.flatnonzero(empty), fallback] = 1.0
    lo = np.floor(delay).astype(np.int64) % N_DELAY
    fraction = (delay - np.floor(delay)).astype(np.float32)
    out = np.zeros((len(valid), N_DELAY), dtype=np.float32)
    rows = np.repeat(np.arange(len(valid), dtype=np.int64), valid.shape[1])
    np.add.at(out, (rows, lo.reshape(-1)), (weight * (1.0 - fraction)).reshape(-1))
    np.add.at(out, (rows, ((lo + 1) % N_DELAY).reshape(-1)), (weight * fraction).reshape(-1))
    out = smooth_circular(out, sigma)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-30)
    return out


def arrays(cache: np.lib.npyio.NpzFile, prefix: str) -> tuple[np.ndarray, ...]:
    valid = cache[f"{prefix}_valid"].astype(bool)
    delay = np.nan_to_num(
        cache[f"{prefix}_delay_bin_physical"].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    length = np.nan_to_num(
        cache[f"{prefix}_path_length"].astype(np.float32),
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    depth = cache[f"{prefix}_interaction_count"].astype(np.int64)
    score = np.nan_to_num(
        cache[f"{prefix}_score"].astype(np.float32), nan=-24.0, neginf=-24.0, posinf=4.0
    )
    return valid, delay, length, depth, score


def main() -> None:
    path_cache = np.load(PATH_CACHE)
    train_path = arrays(path_cache, "train")
    test_path = arrays(path_cache, "test")
    train_truth = link_normalized_global_d(TRAIN_H)
    test_truth = link_normalized_global_d(LOCAL_TRUTH)
    rows: list[dict[str, float]] = []
    best_spec = None
    best_train = -np.inf
    best_test_prediction = None
    for score_scale, length_power, depth_penalty, direct_bias, sigma in product(
        SCORE_SCALE, LENGTH_POWER, DEPTH_PENALTY, DIRECT_BIAS, SMOOTH_SIGMA
    ):
        train_prediction = render(
            *train_path,
            score_scale,
            length_power,
            depth_penalty,
            direct_bias,
            sigma,
        )
        train_score = float(np.mean(row_cos(train_prediction, train_truth)))
        test_prediction = render(
            *test_path,
            score_scale,
            length_power,
            depth_penalty,
            direct_bias,
            sigma,
        )
        test_score = float(np.mean(row_cos(test_prediction, test_truth)))
        spec = {
            "score_scale": score_scale,
            "length_power": length_power,
            "depth_penalty": depth_penalty,
            "direct_bias": direct_bias,
            "sigma": sigma,
        }
        rows.append({**spec, "official_train_cos": train_score, "frozen_local_test_cos": test_score})
        if train_score > best_train:
            best_train = train_score
            best_spec = spec
            best_test_prediction = test_prediction
    assert best_spec is not None and best_test_prediction is not None
    np.save(TARGET_OUT, best_test_prediction.astype(np.float32))
    train_scores = np.asarray([row["official_train_cos"] for row in rows])
    test_scores = np.asarray([row["frozen_local_test_cos"] for row in rows])
    official_order = np.argsort(train_scores)[::-1]
    local_order = np.argsort(test_scores)[::-1]
    report = {
        "method": VERSION,
        "status": "analysis_only_physical_target",
        "primary_hypothesis": (
            "A global attenuation law over official point-cloud high-depth paths can provide "
            "the missing absolute 192-bin D target without synthetic labels."
        ),
        "selection_policy": "all parameters selected only by official train global-D cosine",
        "grid_size": len(rows),
        "cross_domain_rank": {
            "pearson": float(pearsonr(train_scores, test_scores).statistic),
            "spearman": float(spearmanr(train_scores, test_scores).statistic),
        },
        "official_best": rows[int(official_order[0])],
        "official_top10": [rows[int(index)] for index in official_order[:10]],
        "diagnostic_local_best": rows[int(local_order[0])],
        "official_score_distribution": summarize(train_scores),
        "local_score_distribution": summarize(test_scores),
        "outputs": {
            "test_d": PATHS.relative(TARGET_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
