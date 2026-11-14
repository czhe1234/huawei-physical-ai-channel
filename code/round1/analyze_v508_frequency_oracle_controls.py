#!/usr/bin/env python3
"""Falsify v507 with wrong-delay and harder spectral holdout controls."""

from __future__ import annotations

import json

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v503_fixed_material_sionna_version_inverse as v503
import analyze_v506_path_mechanism_union_oracle as v506
import analyze_v507_complex_path_frequency_oracle as v507
from project_paths import PATHS


VERSION_NUMBER = 508
VERSION = "v508_frequency_oracle_controls"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v508_frequency_control_metrics.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"

PATH_LIMITS = (8, 16, 32, 64)
SOURCE_NAMES = ("true_union", "same_regime_shuffle", "shift_plus3bins", "uniform_grid", "random_grid")
CV_NAMES = ("even_odd", "contiguous_4fold")
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TINY = 1e-30


def selected_coefficients(
    matrix: np.ndarray,
    target: np.ndarray,
    interaction_count: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[1] <= limit:
        selected = np.arange(matrix.shape[1], dtype=np.int64)
    else:
        coefficients = v507.ridge_coefficients(matrix, target)
        energy = np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)
        score = np.log(np.maximum(energy, TINY)) - 1e-6 * interaction_count
        selected = np.argsort(score)[-limit:].astype(np.int64)
    coefficients = v507.ridge_coefficients(matrix[:, selected], target)
    return selected, coefficients


def cross_predict(
    matrix: np.ndarray,
    target: np.ndarray,
    interaction_count: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    limit: int,
) -> np.ndarray:
    output = np.empty_like(target)
    for train_rows, test_rows in folds:
        selected, coefficients = selected_coefficients(
            matrix[train_rows], target[train_rows], interaction_count, limit
        )
        output[test_rows] = matrix[test_rows][:, selected] @ coefficients
    return output


def folds() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    all_rows = np.arange(v507.N_SUBCARRIERS, dtype=np.int64)
    even = all_rows[::2]
    odd = all_rows[1::2]
    block = []
    for start in range(0, v507.N_SUBCARRIERS, 48):
        test = np.arange(start, start + 48, dtype=np.int64)
        train = np.setdiff1d(all_rows, test, assume_unique=True)
        block.append((train, test))
    return {"even_odd": [(even, odd), (odd, even)], "contiguous_4fold": block}


def fixed_sources() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    uniform_center = np.linspace(0.0, v507.N_SUBCARRIERS - 1.0, 128)
    uniform_tau = uniform_center / (
        v507.N_SUBCARRIERS * v507.SUBCARRIER_SPACING
    )
    rng = np.random.default_rng(50801)
    random_center = np.sort(rng.uniform(0.0, v507.N_SUBCARRIERS, size=128))
    random_tau = random_center / (
        v507.N_SUBCARRIERS * v507.SUBCARRIER_SPACING
    )
    return {
        "uniform_grid": (uniform_tau.astype(np.float64), np.zeros(128, dtype=np.float32)),
        "random_grid": (random_tau.astype(np.float64), np.zeros(128, dtype=np.float32)),
    }


def summarize(value: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(value, dtype=np.float64)[mask]
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.quantile(selected, 0.10)),
        "p90": float(np.quantile(selected, 0.90)),
    }


def main() -> None:
    indices = np.asarray(
        json.loads(V503_REPORT.read_text(encoding="utf-8"))["indices"],
        dtype=np.int64,
    )
    union = v507.load_union(indices)
    truth = np.load(TRAIN_H, mmap_mode="r")
    regimes = v501.regime_codes()[indices]
    cv_folds = folds()
    fixed = fixed_sources()

    true_delays = []
    true_depths = []
    for row in range(len(indices)):
        valid = union["valid"][row].astype(bool)
        tau, depth = v507.unique_delays(
            union["tau"][row, valid], union["interaction_count"][row, valid]
        )
        true_delays.append(tau)
        true_depths.append(depth)
    shuffle_row = np.empty(len(indices), dtype=np.int64)
    for regime in range(4):
        rows = np.flatnonzero(regimes == regime)
        shuffle_row[rows] = np.roll(rows, 1)

    shape = (
        len(SOURCE_NAMES),
        len(PATH_LIMITS),
        len(CV_NAMES),
        len(indices),
    )
    pdp = np.zeros(shape, dtype=np.float32)
    leaked = np.zeros(shape, dtype=np.float32)
    coherence = np.zeros(shape, dtype=np.float32)
    traceback_like = np.zeros(shape, dtype=np.float32)
    nmse = np.zeros(shape, dtype=np.float32)

    for row, point_id in enumerate(indices):
        channel = np.asarray(truth[point_id], dtype=np.complex64)
        target = channel.reshape(-1, v507.N_SUBCARRIERS).T.astype(np.complex128)
        source_data = {
            "true_union": (true_delays[row], true_depths[row]),
            "same_regime_shuffle": (
                true_delays[int(shuffle_row[row])],
                true_depths[int(shuffle_row[row])],
            ),
            "shift_plus3bins": (
                true_delays[row]
                + 3.0 / (v507.N_SUBCARRIERS * v507.SUBCARRIER_SPACING),
                true_depths[row],
            ),
            **fixed,
        }
        for source_index, source_name in enumerate(SOURCE_NAMES):
            tau, depth = source_data[source_name]
            if len(tau) == 0:
                continue
            matrix = v507.basis(tau)
            for limit_index, limit in enumerate(PATH_LIMITS):
                for cv_index, cv_name in enumerate(CV_NAMES):
                    prediction = cross_predict(
                        matrix, target, depth, cv_folds[cv_name], limit
                    )
                    predicted_channel = prediction.T.reshape(channel.shape)
                    metric = v507.complex_metrics(predicted_channel, channel)
                    pdp[source_index, limit_index, cv_index, row] = metric["pdp"]
                    leaked[source_index, limit_index, cv_index, row] = metric["leaked_full"]
                    coherence[source_index, limit_index, cv_index, row] = metric[
                        "complex_coherence"
                    ]
                    traceback_like[source_index, limit_index, cv_index, row] = metric[
                        "traceback_like"
                    ]
                    nmse[source_index, limit_index, cv_index, row] = metric["nmse"]
        if (row + 1) % 10 == 0 or row + 1 == len(indices):
            print(f"v508 controls {row + 1}/{len(indices)}", flush=True)

    summary = []
    for source_index, source_name in enumerate(SOURCE_NAMES):
        for limit_index, limit in enumerate(PATH_LIMITS):
            for cv_index, cv_name in enumerate(CV_NAMES):
                row = {
                    "source": source_name,
                    "path_limit": limit,
                    "cross_validation": cv_name,
                    "overall": {},
                    "heavy": {},
                }
                for key, value in (
                    ("pdp", pdp),
                    ("leaked_full", leaked),
                    ("complex_coherence", coherence),
                    ("traceback_like", traceback_like),
                    ("nmse", nmse),
                ):
                    local = value[source_index, limit_index, cv_index]
                    row["overall"][key] = summarize(
                        local, np.ones(len(indices), dtype=bool)
                    )
                    row["heavy"][key] = summarize(local, regimes == 3)
                summary.append(row)

    primary = next(
        row
        for row in summary
        if row["source"] == "true_union"
        and row["path_limit"] == 32
        and row["cross_validation"] == "contiguous_4fold"
    )
    controls = [
        row
        for row in summary
        if row["source"] != "true_union"
        and row["path_limit"] == 32
        and row["cross_validation"] == "contiguous_4fold"
    ]
    best_control = max(
        controls, key=lambda row: row["heavy"]["traceback_like"]["mean"]
    )
    margin = (
        primary["heavy"]["traceback_like"]["mean"]
        - best_control["heavy"]["traceback_like"]["mean"]
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=indices,
        regimes=regimes,
        source_names=np.asarray(SOURCE_NAMES),
        path_limits=np.asarray(PATH_LIMITS),
        cv_names=np.asarray(CV_NAMES),
        pdp=pdp,
        leaked_full=leaked,
        complex_coherence=coherence,
        traceback_like=traceback_like,
        nmse=nmse,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "v507 is configuration-informative only if true union delays outperform shuffled, "
            "shifted, uniform, and random delay dictionaries under harder contiguous holdout."
        ),
        "sources": list(SOURCE_NAMES),
        "path_limits": list(PATH_LIMITS),
        "cross_validation": {
            "even_odd": "v507 replication",
            "contiguous_4fold": "four disjoint 48-subcarrier holdout blocks",
        },
        "summary": summary,
        "primary_comparison": {
            "true_union_k32_block4": primary,
            "best_negative_control_k32_block4": best_control,
            "heavy_traceback_margin": float(margin),
            "v507_is_configuration_discriminative": bool(margin >= 0.02),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "point_metrics": PATHS.relative(POINT_METRICS),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "primary_heavy_traceback": primary["heavy"]["traceback_like"]["mean"],
                "best_control": best_control["source"],
                "best_control_heavy_traceback": best_control["heavy"]["traceback_like"]["mean"],
                "margin": margin,
                "discriminative": margin >= 0.02,
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
