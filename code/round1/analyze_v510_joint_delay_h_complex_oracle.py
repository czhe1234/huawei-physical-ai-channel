#!/usr/bin/env python3
"""Cross-frequency oracle with Sionna delay and horizontal-angle atoms."""

from __future__ import annotations

import json
import math

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v507_complex_path_frequency_oracle as v507
import analyze_v508_frequency_oracle_controls as v508
from project_paths import PATHS


VERSION_NUMBER = 510
VERSION = "v510_joint_delay_h_complex_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v510_joint_delay_h_metrics.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"
V508_REPORT = PATHS.version_report_dir(508) / "v508_frequency_oracle_controls.json"

CONFIGS = (
    ("true_h_k32", "true", 32),
    ("true_h_k64", "true", 64),
    ("shift_h_plus3_k64", "shift3", 64),
    ("shuffle_delay_h_k64", "shuffle", 64),
)
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TINY = 1e-30


def path_candidates(
    tau: np.ndarray, phi_t: np.ndarray, interaction_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delay_center = np.mod(
        v507.N_SUBCARRIERS * v507.SUBCARRIER_SPACING * tau.astype(np.float64),
        v507.N_SUBCARRIERS,
    )
    h_center = np.mod(
        8.0 * np.sin(phi_t.astype(np.float64) - math.radians(-30.0)), 16.0
    )
    key_delay = np.rint(delay_center * 8.0).astype(np.int32)
    key_h = np.rint(h_center * 4.0).astype(np.int32)
    order = np.lexsort((tau, interaction_count))
    selected = []
    seen: set[tuple[int, int]] = set()
    for index in order:
        key = (int(key_delay[index]), int(key_h[index]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(int(index))
    keep = np.asarray(selected, dtype=np.int64)
    return (
        tau[keep].astype(np.float64),
        h_center[keep].astype(np.float64),
        interaction_count[keep].astype(np.float32),
    )


def joint_basis(tau: np.ndarray, h_center: np.ndarray) -> np.ndarray:
    frequency = v507.basis(tau)
    element = np.arange(16, dtype=np.float64)
    horizontal = np.exp(
        2j * np.pi * h_center[:, None] * element[None, :] / 16.0
    ).astype(np.complex128)
    return (
        frequency[:, None, :] * horizontal.T[None, :, :]
    ).reshape(v507.N_SUBCARRIERS * 16, len(tau))


def target_matrix(channel: np.ndarray) -> np.ndarray:
    h6 = np.asarray(channel, dtype=np.complex64).reshape(2, 16, 8, 4, 192)
    return h6.transpose(4, 1, 0, 2, 3).reshape(192 * 16, 2 * 8 * 4).astype(np.complex128)


def channel_from_matrix(value: np.ndarray) -> np.ndarray:
    h6 = value.reshape(192, 16, 2, 8, 4).transpose(2, 1, 3, 4, 0)
    return h6.reshape(256, 4, 192).astype(np.complex64)


def frequency_row_indices(frequencies: np.ndarray) -> np.ndarray:
    horizontal = np.arange(16, dtype=np.int64)
    return (frequencies[:, None] * 16 + horizontal[None, :]).reshape(-1)


def select_and_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    interaction_count: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_matrix = matrix[train_rows]
    train_target = target[train_rows]
    if matrix.shape[1] <= limit:
        selected = np.arange(matrix.shape[1], dtype=np.int64)
    else:
        coefficients = v507.ridge_coefficients(train_matrix, train_target)
        energy = np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)
        score = np.log(np.maximum(energy, TINY)) - 1e-6 * interaction_count
        selected = np.argsort(score)[-limit:].astype(np.int64)
    coefficients = v507.ridge_coefficients(
        train_matrix[:, selected], train_target
    )
    return selected, coefficients


def block_cross_prediction(
    matrix: np.ndarray,
    target: np.ndarray,
    interaction_count: np.ndarray,
    limit: int,
) -> np.ndarray:
    output = np.empty_like(target)
    all_frequency = np.arange(192, dtype=np.int64)
    for start in range(0, 192, 48):
        test_frequency = np.arange(start, start + 48, dtype=np.int64)
        train_frequency = np.setdiff1d(
            all_frequency, test_frequency, assume_unique=True
        )
        train_rows = frequency_row_indices(train_frequency)
        test_rows = frequency_row_indices(test_frequency)
        selected, coefficients = select_and_fit(
            matrix, target, train_rows, interaction_count, limit
        )
        output[test_rows] = matrix[test_rows][:, selected] @ coefficients
    return output


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
    metric_names = ("pdp", "leaked_full", "complex_coherence", "nmse", "traceback_like")
    metrics = {
        name: np.zeros((len(CONFIGS), len(indices)), dtype=np.float32)
        for name in metric_names
    }
    candidate_count = np.zeros(len(indices), dtype=np.int32)

    for row, point_id in enumerate(indices):
        valid = union["valid"][row].astype(bool)
        tau, h_center, depth = path_candidates(
            union["tau"][row, valid],
            union["phi_t"][row, valid],
            union["interaction_count"][row, valid],
        )
        candidate_count[row] = len(tau)
        if len(tau) == 0:
            continue
        channel = np.asarray(truth[point_id], dtype=np.complex64)
        target = target_matrix(channel)
        rng = np.random.default_rng(51001 + int(point_id))
        shuffled_h = h_center[rng.permutation(len(h_center))]
        source_h = {
            "true": h_center,
            "shift3": np.mod(h_center + 3.0, 16.0),
            "shuffle": shuffled_h,
        }
        for config_index, (_, source, limit) in enumerate(CONFIGS):
            matrix = joint_basis(tau, source_h[source])
            prediction = block_cross_prediction(matrix, target, depth, limit)
            predicted_channel = channel_from_matrix(prediction)
            result = v507.complex_metrics(predicted_channel, channel)
            for name in metric_names:
                metrics[name][config_index, row] = result[name]
        if (row + 1) % 10 == 0 or row + 1 == len(indices):
            print(f"v510 joint delay-H {row + 1}/{len(indices)}", flush=True)

    summary = []
    for config_index, (name, source, limit) in enumerate(CONFIGS):
        row = {
            "name": name,
            "source": source,
            "path_limit": limit,
            "overall": {},
            "heavy": {},
        }
        for metric_name, value in metrics.items():
            local = value[config_index]
            row["overall"][metric_name] = summarize(
                local, np.ones(len(indices), dtype=bool)
            )
            row["heavy"][metric_name] = summarize(local, regimes == 3)
        summary.append(row)

    true64 = next(row for row in summary if row["name"] == "true_h_k64")
    shifted = next(row for row in summary if row["name"] == "shift_h_plus3_k64")
    shuffled = next(row for row in summary if row["name"] == "shuffle_delay_h_k64")
    v508_report = json.loads(V508_REPORT.read_text(encoding="utf-8"))
    delay_only64 = next(
        row
        for row in v508_report["summary"]
        if row["source"] == "true_union"
        and row["path_limit"] == 64
        and row["cross_validation"] == "contiguous_4fold"
    )
    heavy_margin = true64["heavy"]["traceback_like"]["mean"] - max(
        shifted["heavy"]["traceback_like"]["mean"],
        shuffled["heavy"]["traceback_like"]["mean"],
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=indices,
        regimes=regimes,
        config_names=np.asarray([value[0] for value in CONFIGS]),
        candidate_count=candidate_count,
        **metrics,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Sionna delay and departure-H pairs should predict held-out contiguous frequency "
            "blocks when only V/polarization/UE complex coefficients remain free."
        ),
        "method_details": {
            "cross_validation": "four contiguous 48-subcarrier holdout blocks",
            "free_coefficients": "one complex coefficient per path, TX-pol, V-element, and UE-port",
            "fixed_structure": "path delay and 16-element BS-H steering",
            "controls": ["all H centers +3 bins", "within-point delay/H permutation"],
        },
        "point_count": int(len(indices)),
        "candidate_count": summarize(
            candidate_count, np.ones(len(indices), dtype=bool)
        ),
        "summary": summary,
        "delay_only_reference_k64": delay_only64,
        "decision": {
            "true_h_margin_vs_best_control_heavy": float(heavy_margin),
            "sionna_delay_h_pairing_is_informative": bool(heavy_margin >= 0.02),
            "h_constraint_retained_fraction_of_delay_only_heavy": float(
                true64["heavy"]["traceback_like"]["mean"]
                / max(delay_only64["heavy"]["traceback_like"]["mean"], TINY)
            ),
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
                "true_h_k64_overall": true64["overall"]["traceback_like"]["mean"],
                "true_h_k64_heavy": true64["heavy"]["traceback_like"]["mean"],
                "shift_control_heavy": shifted["heavy"]["traceback_like"]["mean"],
                "shuffle_control_heavy": shuffled["heavy"]["traceback_like"]["mean"],
                "delay_only_k64_heavy": delay_only64["heavy"]["traceback_like"]["mean"],
                "margin": heavy_margin,
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
