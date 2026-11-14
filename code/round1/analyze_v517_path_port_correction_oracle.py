#!/usr/bin/env python3
"""Locate the minimum per-path port structure needed beyond one complex scalar."""

from __future__ import annotations

import json

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v507_complex_path_frequency_oracle as v507
import analyze_v510_joint_delay_h_complex_oracle as v510
from project_paths import PATHS


VERSION_NUMBER = 517
VERSION = "v517_path_port_correction_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v517_path_port_oracle_metrics.npz"
COEFFICIENT_CACHE = PATHS.cache_dir(516) / "v516_path_coefficients_points160.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"

MODE_NAMES = (
    "source_atom_fit",
    "scalar_complex",
    "scalar_energy",
    "pol_complex",
    "vertical_complex",
    "ue_complex",
    "pol_vertical_complex",
    "pol_ue_complex",
    "vertical_ue_complex",
    "separable_pol_vertical_ue_complex",
    "full_coefficients",
)
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TINY = 1e-30


def optimal_group_scalar(
    source: np.ndarray,
    target: np.ndarray,
    keep_axes: tuple[int, ...],
) -> np.ndarray:
    reduce_axes = tuple(axis for axis in range(source.ndim) if axis not in keep_axes)
    numerator = np.sum(np.conj(source) * target, axis=reduce_axes, keepdims=True)
    denominator = np.maximum(
        np.sum(np.abs(source) ** 2, axis=reduce_axes, keepdims=True), TINY
    )
    return numerator / denominator


def energy_group_scalar(
    source: np.ndarray,
    target: np.ndarray,
    keep_axes: tuple[int, ...],
) -> np.ndarray:
    reduce_axes = tuple(axis for axis in range(source.ndim) if axis not in keep_axes)
    source_energy = np.sum(np.abs(source) ** 2, axis=reduce_axes, keepdims=True)
    target_energy = np.sum(np.abs(target) ** 2, axis=reduce_axes, keepdims=True)
    return np.sqrt(target_energy / np.maximum(source_energy, TINY))


def separable_correction(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    # source/target: path, polarization, vertical element, UE port
    count = len(source)
    pol = np.ones((count, 2, 1, 1), dtype=np.complex128)
    vertical = np.ones((count, 1, 8, 1), dtype=np.complex128)
    ue = np.ones((count, 1, 1, 4), dtype=np.complex128)
    for _ in range(12):
        local = source * vertical * ue
        pol = optimal_group_scalar(local, target, (0, 1))
        local = source * pol * ue
        vertical = optimal_group_scalar(local, target, (0, 2))
        local = source * pol * vertical
        ue = optimal_group_scalar(local, target, (0, 3))
    return source * pol * vertical * ue


def corrected_sets(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    source4 = source.reshape(len(source), 2, 8, 4).astype(np.complex128)
    target4 = target.reshape(len(target), 2, 8, 4).astype(np.complex128)
    scalar = optimal_group_scalar(source4, target4, (0,))
    scalar_energy = energy_group_scalar(source4, target4, (0,))
    corrections = {
        "source_atom_fit": source4,
        "scalar_complex": source4 * scalar,
        "scalar_energy": source4 * scalar_energy,
        "pol_complex": source4 * optimal_group_scalar(source4, target4, (0, 1)),
        "vertical_complex": source4
        * optimal_group_scalar(source4, target4, (0, 2)),
        "ue_complex": source4 * optimal_group_scalar(source4, target4, (0, 3)),
        "pol_vertical_complex": source4
        * optimal_group_scalar(source4, target4, (0, 1, 2)),
        "pol_ue_complex": source4
        * optimal_group_scalar(source4, target4, (0, 1, 3)),
        "vertical_ue_complex": source4
        * optimal_group_scalar(source4, target4, (0, 2, 3)),
        "separable_pol_vertical_ue_complex": separable_correction(source4, target4),
        "full_coefficients": target4,
    }
    return {name: value.reshape(len(source), 64) for name, value in corrections.items()}


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
    for path in (COEFFICIENT_CACHE, TRAIN_H):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(COEFFICIENT_CACHE) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    point_ids = data["indices"].astype(np.int64)
    regimes = v501.regime_codes()[point_ids]
    truth = np.load(TRAIN_H, mmap_mode="r")
    metric_names = ("pdp", "leaked_full", "complex_coherence", "nmse", "traceback_like")
    metrics = {
        name: np.zeros((len(MODE_NAMES), len(point_ids)), dtype=np.float32)
        for name in metric_names
    }

    for row, point_id in enumerate(point_ids):
        valid = data["valid"][row].astype(bool)
        tau = data["tau"][row, valid]
        h_center = data["h_center"][row, valid]
        source = data["source_coefficients"][row, valid]
        official = data["official_coefficients"][row, valid]
        matrix = v510.joint_basis(tau, h_center)
        target_channel = np.asarray(truth[int(point_id)], dtype=np.complex64)
        candidates = corrected_sets(source, official)
        for mode_index, mode in enumerate(MODE_NAMES):
            prediction = v510.channel_from_matrix(matrix @ candidates[mode])
            result = v507.complex_metrics(prediction, target_channel)
            for name in metric_names:
                metrics[name][mode_index, row] = result[name]
        if (row + 1) % 10 == 0 or row + 1 == len(point_ids):
            print(f"v517 oracle {row + 1}/{len(point_ids)}", flush=True)

    summary = []
    for mode_index, mode in enumerate(MODE_NAMES):
        row = {"mode": mode, "overall": {}, "heavy": {}}
        for name, values in metrics.items():
            row["overall"][name] = summarize(
                values[mode_index], np.ones(len(point_ids), dtype=bool)
            )
            row["heavy"][name] = summarize(values[mode_index], regimes == 3)
        summary.append(row)

    source = next(row for row in summary if row["mode"] == "source_atom_fit")
    full = next(row for row in summary if row["mode"] == "full_coefficients")
    source_score = source["overall"]["traceback_like"]["mean"]
    full_gain = full["overall"]["traceback_like"]["mean"] - source_score
    ranked = sorted(
        (
            {
                "mode": row["mode"],
                "overall_gain": row["overall"]["traceback_like"]["mean"]
                - source_score,
                "heavy_gain": row["heavy"]["traceback_like"]["mean"]
                - source["heavy"]["traceback_like"]["mean"],
                "fraction_of_full_gap": (
                    row["overall"]["traceback_like"]["mean"] - source_score
                )
                / max(full_gain, TINY),
            }
            for row in summary
            if row["mode"] != "source_atom_fit"
        ),
        key=lambda row: row["overall_gain"],
        reverse=True,
    )
    best_compact = next(
        row
        for row in ranked
        if row["mode"] not in {"full_coefficients", "separable_pol_vertical_ue_complex"}
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=point_ids,
        regimes=regimes,
        mode_names=np.asarray(MODE_NAMES),
        **metrics,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The large full-coefficient ceiling is concentrated in a small structured subset "
            "of polarization, vertical-array, and UE-port corrections rather than requiring "
            "64 unrelated complex values per path."
        ),
        "point_count": int(len(point_ids)),
        "method_details": {
            "shared_atoms": "the same 32 source-selected delay-H atoms as v516",
            "oracle_rule": "all correction factors are fit independently per point/path from official H",
            "separable_iterations": 12,
        },
        "summary": summary,
        "ranked_gap_recovery": ranked,
        "decision": {
            "full_coefficient_gain": float(full_gain),
            "best_compact_mode": best_compact,
            "compact_structure_is_sufficient": bool(
                best_compact["fraction_of_full_gap"] >= 0.60
            ),
            "next_model_target": best_compact["mode"],
        },
        "outputs": {
            "point_metrics": PATHS.relative(POINT_METRICS),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_score": source_score,
                "full_score": full["overall"]["traceback_like"]["mean"],
                "ranked": ranked,
                "next_model_target": report["decision"]["next_model_target"],
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
