#!/usr/bin/env python3
"""Screen candidate-centered horizontal angular spread around Sionna paths."""

from __future__ import annotations

import json
import math

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v507_complex_path_frequency_oracle as v507
import analyze_v510_joint_delay_h_complex_oracle as v510
from project_paths import PATHS


VERSION_NUMBER = 511
VERSION = "v511_candidate_angular_spread_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v511_angular_spread_metrics.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"
V508_REPORT = PATHS.version_report_dir(508) / "v508_frequency_oracle_controls.json"

MODES = {
    "exact": np.asarray([0.0]),
    "spread_0p5": np.asarray([-0.5, 0.0, 0.5]),
    "spread_1": np.asarray([-1.0, 0.0, 1.0]),
    "spread_2": np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
    "all_16": None,
    "shift_plus3": np.asarray([3.0]),
}
DELAY_LIMIT = 32
ATOM_PRELIMIT = 96
ATOM_LIMIT = 32
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TINY = 1e-30


def select_screen_indices(indices: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    selected = []
    for regime in range(4):
        rows = np.flatnonzero(regimes == regime)
        selected.extend(rows[::2][:20].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def delay_angle_groups(
    tau: np.ndarray, phi_t: np.ndarray, interaction_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    delay_center = np.mod(
        v507.N_SUBCARRIERS * v507.SUBCARRIER_SPACING * tau.astype(np.float64),
        v507.N_SUBCARRIERS,
    )
    h_center = np.mod(
        8.0 * np.sin(phi_t.astype(np.float64) - math.radians(-30.0)), 16.0
    )
    key = np.rint(delay_center * 8.0).astype(np.int32)
    groups: dict[int, list[int]] = {}
    for index, value in enumerate(key):
        groups.setdefault(int(value), []).append(index)
    tau_out = []
    depth_out = []
    h_groups = []
    for local_key in sorted(groups):
        ids = np.asarray(groups[local_key], dtype=np.int64)
        representative = ids[
            np.lexsort((tau[ids], interaction_count[ids]))[0]
        ]
        tau_out.append(float(tau[representative]))
        depth_out.append(float(interaction_count[representative]))
        h_groups.append(h_center[ids].astype(np.float64))
    return (
        np.asarray(tau_out, dtype=np.float64),
        np.asarray(depth_out, dtype=np.float32),
        h_groups,
    )


def angle_candidates(
    selected_delays: np.ndarray,
    tau: np.ndarray,
    depth: np.ndarray,
    h_groups: list[np.ndarray],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau_out = []
    h_out = []
    depth_out = []
    offsets = MODES[mode]
    for delay_id in selected_delays:
        if offsets is None:
            centers = np.arange(16, dtype=np.float64)
        else:
            centers = (
                h_groups[int(delay_id)][:, None] + offsets[None, :]
            ).reshape(-1)
            centers = np.mod(centers, 16.0)
        keys = np.rint(centers * 4.0).astype(np.int32)
        _, unique_index = np.unique(keys, return_index=True)
        centers = centers[np.sort(unique_index)]
        tau_out.extend([float(tau[int(delay_id)])] * len(centers))
        h_out.extend(centers.tolist())
        depth_out.extend([float(depth[int(delay_id)])] * len(centers))
    return (
        np.asarray(tau_out, dtype=np.float64),
        np.asarray(h_out, dtype=np.float64),
        np.asarray(depth_out, dtype=np.float32),
    )


def matched_filter_preselect(
    matrix: np.ndarray, target: np.ndarray, interaction_count: np.ndarray
) -> np.ndarray:
    correlation = matrix.conj().T @ target
    score = np.sum(np.abs(correlation) ** 2, axis=1, dtype=np.float64)
    score /= np.maximum(np.sum(np.abs(matrix) ** 2, axis=0), TINY)
    score = np.log(np.maximum(score, TINY)) - 1e-6 * interaction_count
    limit = min(ATOM_PRELIMIT, len(score))
    return np.argsort(score)[-limit:].astype(np.int64)


def fit_atoms(
    matrix: np.ndarray,
    target: np.ndarray,
    interaction_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    preselected = matched_filter_preselect(matrix, target, interaction_count)
    coefficients = v507.ridge_coefficients(matrix[:, preselected], target)
    energy = np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)
    local_limit = min(ATOM_LIMIT, len(preselected))
    selected_local = np.argsort(energy)[-local_limit:]
    selected = preselected[selected_local]
    coefficients = v507.ridge_coefficients(matrix[:, selected], target)
    return selected, coefficients


def block_prediction(
    tau: np.ndarray,
    depth: np.ndarray,
    h_groups: list[np.ndarray],
    target: np.ndarray,
    mode: str,
) -> np.ndarray:
    output = np.empty_like(target)
    all_frequency = np.arange(192, dtype=np.int64)
    delay_matrix = v507.basis(tau)
    for start in range(0, 192, 48):
        test_frequency = np.arange(start, start + 48, dtype=np.int64)
        train_frequency = np.setdiff1d(
            all_frequency, test_frequency, assume_unique=True
        )
        delay_train_rows = train_frequency
        delay_target = target.reshape(192, 16, -1).reshape(192, -1)
        if len(tau) <= DELAY_LIMIT:
            selected_delays = np.arange(len(tau), dtype=np.int64)
        else:
            coefficients = v507.ridge_coefficients(
                delay_matrix[delay_train_rows], delay_target[delay_train_rows]
            )
            energy = np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)
            score = np.log(np.maximum(energy, TINY)) - 1e-6 * depth
            selected_delays = np.argsort(score)[-DELAY_LIMIT:].astype(np.int64)
        atom_tau, atom_h, atom_depth = angle_candidates(
            selected_delays, tau, depth, h_groups, mode
        )
        matrix = v510.joint_basis(atom_tau, atom_h)
        train_rows = v510.frequency_row_indices(train_frequency)
        test_rows = v510.frequency_row_indices(test_frequency)
        selected_atoms, coefficients = fit_atoms(
            matrix[train_rows], target[train_rows], atom_depth
        )
        output[test_rows] = matrix[test_rows][:, selected_atoms] @ coefficients
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
    full_indices = np.asarray(
        json.loads(V503_REPORT.read_text(encoding="utf-8"))["indices"],
        dtype=np.int64,
    )
    full_regimes = v501.regime_codes()[full_indices]
    screen_rows = select_screen_indices(full_indices, full_regimes)
    indices = full_indices[screen_rows]
    regimes = full_regimes[screen_rows]
    full_union = v507.load_union(full_indices)
    union = {
        key: value[screen_rows] if isinstance(value, np.ndarray) and len(value) == len(full_indices) else value
        for key, value in full_union.items()
    }
    truth = np.load(TRAIN_H, mmap_mode="r")
    metric_names = ("pdp", "leaked_full", "complex_coherence", "nmse", "traceback_like")
    metrics = {
        name: np.zeros((len(MODES), len(indices)), dtype=np.float32)
        for name in metric_names
    }
    candidate_count = np.zeros((len(MODES), len(indices)), dtype=np.int32)
    mode_names = list(MODES)

    for row, point_id in enumerate(indices):
        valid = union["valid"][row].astype(bool)
        tau, depth, h_groups = delay_angle_groups(
            union["tau"][row, valid],
            union["phi_t"][row, valid],
            union["interaction_count"][row, valid],
        )
        channel = np.asarray(truth[point_id], dtype=np.complex64)
        target = v510.target_matrix(channel)
        selected_all = np.arange(min(len(tau), DELAY_LIMIT), dtype=np.int64)
        for mode_index, mode in enumerate(mode_names):
            preview_tau, _, _ = angle_candidates(
                selected_all, tau, depth, h_groups, mode
            )
            candidate_count[mode_index, row] = len(preview_tau)
            prediction = block_prediction(tau, depth, h_groups, target, mode)
            predicted_channel = v510.channel_from_matrix(prediction)
            result = v507.complex_metrics(predicted_channel, channel)
            for name in metric_names:
                metrics[name][mode_index, row] = result[name]
        if (row + 1) % 5 == 0 or row + 1 == len(indices):
            print(f"v511 angular spread {row + 1}/{len(indices)}", flush=True)

    summary = []
    for mode_index, mode in enumerate(mode_names):
        row = {"mode": mode, "overall": {}, "heavy": {}}
        for name, value in metrics.items():
            local = value[mode_index]
            row["overall"][name] = summarize(
                local, np.ones(len(indices), dtype=bool)
            )
            row["heavy"][name] = summarize(local, regimes == 3)
        row["candidate_count"] = summarize(
            candidate_count[mode_index], np.ones(len(indices), dtype=bool)
        )
        summary.append(row)

    exact = next(row for row in summary if row["mode"] == "exact")
    spread1 = next(row for row in summary if row["mode"] == "spread_1")
    spread2 = next(row for row in summary if row["mode"] == "spread_2")
    all16 = next(row for row in summary if row["mode"] == "all_16")
    shift = next(row for row in summary if row["mode"] == "shift_plus3")
    v508_report = json.loads(V508_REPORT.read_text(encoding="utf-8"))
    delay_only64 = next(
        row
        for row in v508_report["summary"]
        if row["source"] == "true_union"
        and row["path_limit"] == 64
        and row["cross_validation"] == "contiguous_4fold"
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=indices,
        regimes=regimes,
        mode_names=np.asarray(mode_names),
        candidate_count=candidate_count,
        **metrics,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "If official angular structure is a local broadening of Sionna path angles, small "
            "candidate-centered H-bin offsets should recover most of the delay-only ceiling."
        ),
        "point_count": int(len(indices)),
        "selection": "20 points per regime from the v503 160-point balanced audit",
        "method_details": {
            "delay_limit": DELAY_LIMIT,
            "atom_prelimit": ATOM_PRELIMIT,
            "atom_limit": ATOM_LIMIT,
            "cross_validation": "four contiguous 48-subcarrier holdout blocks",
        },
        "summary": summary,
        "delay_only_k64_reference_160_points": delay_only64,
        "decision": {
            "spread1_heavy_gain_vs_exact": float(
                spread1["heavy"]["traceback_like"]["mean"]
                - exact["heavy"]["traceback_like"]["mean"]
            ),
            "spread2_heavy_gain_vs_exact": float(
                spread2["heavy"]["traceback_like"]["mean"]
                - exact["heavy"]["traceback_like"]["mean"]
            ),
            "all16_heavy_gain_vs_spread2": float(
                all16["heavy"]["traceback_like"]["mean"]
                - spread2["heavy"]["traceback_like"]["mean"]
            ),
            "exact_margin_vs_shift_control": float(
                exact["heavy"]["traceback_like"]["mean"]
                - shift["heavy"]["traceback_like"]["mean"]
            ),
            "local_angular_spread_is_supported": bool(
                spread1["heavy"]["traceback_like"]["mean"]
                - exact["heavy"]["traceback_like"]["mean"]
                >= 0.02
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
                mode: next(row for row in summary if row["mode"] == mode)["heavy"][
                    "traceback_like"
                ]["mean"]
                for mode in mode_names
            }
            | {"report": PATHS.relative(REPORT)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
