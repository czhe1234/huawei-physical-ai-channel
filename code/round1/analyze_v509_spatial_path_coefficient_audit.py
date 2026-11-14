#!/usr/bin/env python3
"""Compare official-H fitted spatial phase slopes with Sionna path angles."""

from __future__ import annotations

import json
import math

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v507_complex_path_frequency_oracle as v507
import analyze_v508_frequency_oracle_controls as v508
from project_paths import PATHS


VERSION_NUMBER = 509
VERSION = "v509_spatial_path_coefficient_audit"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v509_spatial_path_point_metrics.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"

PATH_LIMIT = 32
ORIENTATION_GRID_DEG = np.arange(-45.0, -14.999, 0.25, dtype=np.float64)
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TINY = 1e-30


def circular_distance(left: float, right: np.ndarray, size: float) -> np.ndarray:
    delta = np.abs(np.asarray(right, dtype=np.float64) - float(left))
    return np.minimum(delta, size - delta)


def delay_groups(
    tau: np.ndarray,
    phi_t: np.ndarray,
    interaction_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    center = np.mod(
        v507.N_SUBCARRIERS * v507.SUBCARRIER_SPACING * tau.astype(np.float64),
        v507.N_SUBCARRIERS,
    )
    key = np.rint(center * 8.0).astype(np.int32)
    groups: dict[int, list[int]] = {}
    for index, value in enumerate(key):
        groups.setdefault(int(value), []).append(index)
    representatives = []
    depths = []
    phi_groups = []
    for local_key in sorted(groups):
        ids = np.asarray(groups[local_key], dtype=np.int64)
        representative = ids[
            np.lexsort((tau[ids], interaction_count[ids]))[0]
        ]
        representatives.append(float(tau[representative]))
        depths.append(float(interaction_count[representative]))
        phi_groups.append(phi_t[ids].astype(np.float64))
    return (
        np.asarray(representatives, dtype=np.float64),
        np.asarray(depths, dtype=np.float32),
        phi_groups,
    )


def fitted_spatial_signature(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = coefficients.reshape(len(coefficients), 2, 16, 8, 4)
    h_cross = np.sum(
        tensor[:, :, 1:, :, :] * np.conj(tensor[:, :, :-1, :, :]),
        axis=(1, 2, 3, 4),
    )
    v_cross = np.sum(
        tensor[:, :, :, 1:, :] * np.conj(tensor[:, :, :, :-1, :]),
        axis=(1, 2, 3, 4),
    )
    h_step = np.angle(h_cross)
    v_step = np.angle(v_cross)
    h_coord = np.arange(16, dtype=np.float64) - 7.5
    v_coord = np.arange(8, dtype=np.float64) - 3.5
    phase = (
        h_step[:, None, None] * h_coord[None, :, None]
        + v_step[:, None, None] * v_coord[None, None, :]
    )
    demodulated = tensor * np.exp(-1j * phase[:, None, :, :, None])
    center = np.mean(demodulated, axis=(2, 3), keepdims=True)
    reconstruction = center * np.exp(1j * phase[:, None, :, :, None])
    coherence = np.sum(np.abs(reconstruction) ** 2, axis=(1, 2, 3, 4)) / np.maximum(
        np.sum(np.abs(tensor) ** 2, axis=(1, 2, 3, 4)), TINY
    )
    energy = np.sum(np.abs(tensor) ** 2, axis=(1, 2, 3, 4), dtype=np.float64)
    return h_step.astype(np.float64), coherence.astype(np.float64), energy.astype(np.float64)


def expected_h(phi: np.ndarray, orientation_deg: float) -> np.ndarray:
    return np.mod(
        8.0 * np.sin(phi - math.radians(orientation_deg)), 16.0
    )


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

    component_point = []
    component_regime = []
    component_h_step = []
    component_weight = []
    component_coherence = []
    component_phi_groups: list[np.ndarray] = []
    point_plane_coherence = np.zeros(len(indices), dtype=np.float32)
    point_component_count = np.zeros(len(indices), dtype=np.int32)

    for row, point_id in enumerate(indices):
        valid = union["valid"][row].astype(bool)
        tau, depth, phi_groups = delay_groups(
            union["tau"][row, valid],
            union["phi_t"][row, valid],
            union["interaction_count"][row, valid],
        )
        if len(tau) == 0:
            continue
        channel = np.asarray(truth[point_id], dtype=np.complex64)
        target = channel.reshape(-1, v507.N_SUBCARRIERS).T.astype(np.complex128)
        matrix = v507.basis(tau)
        selected, coefficients = v508.selected_coefficients(
            matrix, target, depth, PATH_LIMIT
        )
        h_step, coherence, energy = fitted_spatial_signature(coefficients)
        energy /= max(float(np.sum(energy)), TINY)
        point_plane_coherence[row] = float(np.sum(energy * coherence))
        point_component_count[row] = len(selected)
        for local, source_id in enumerate(selected):
            if energy[local] <= 1e-8:
                continue
            component_point.append(row)
            component_regime.append(int(regimes[row]))
            component_h_step.append(float(h_step[local]))
            component_weight.append(float(energy[local]))
            component_coherence.append(float(coherence[local]))
            component_phi_groups.append(phi_groups[int(source_id)])
        if (row + 1) % 20 == 0 or row + 1 == len(indices):
            print(f"v509 spatial coefficient {row + 1}/{len(indices)}", flush=True)

    component_point_array = np.asarray(component_point, dtype=np.int64)
    component_regime_array = np.asarray(component_regime, dtype=np.int8)
    h_step_array = np.asarray(component_h_step, dtype=np.float64)
    weight_array = np.asarray(component_weight, dtype=np.float64)
    coherence_array = np.asarray(component_coherence, dtype=np.float64)

    objectives = np.zeros((2, len(ORIENTATION_GRID_DEG), 5), dtype=np.float64)
    weight_sums = np.zeros(5, dtype=np.float64)
    for regime in range(4):
        weight_sums[regime + 1] = np.sum(weight_array[component_regime_array == regime])
    weight_sums[0] = np.sum(weight_array)
    for sign_index, sign in enumerate((1.0, -1.0)):
        fitted_h = np.mod(sign * h_step_array * 16.0 / (2.0 * np.pi), 16.0)
        for orientation_index, orientation in enumerate(ORIENTATION_GRID_DEG):
            error = np.empty(len(fitted_h), dtype=np.float64)
            for component_index, phi_group in enumerate(component_phi_groups):
                candidates = expected_h(phi_group, float(orientation))
                error[component_index] = float(
                    np.min(circular_distance(fitted_h[component_index], candidates, 16.0))
                )
            objectives[sign_index, orientation_index, 0] = np.sum(weight_array * error) / max(
                weight_sums[0], TINY
            )
            for regime in range(4):
                mask = component_regime_array == regime
                objectives[sign_index, orientation_index, regime + 1] = np.sum(
                    weight_array[mask] * error[mask]
                ) / max(weight_sums[regime + 1], TINY)

    clear_objective = objectives[:, :, 1]
    sign_index, orientation_index = np.unravel_index(
        np.argmin(clear_objective), clear_objective.shape
    )
    fitted_sign = (1.0, -1.0)[sign_index]
    best_clear_orientation = float(ORIENTATION_GRID_DEG[orientation_index])
    fitted_h = np.mod(fitted_sign * h_step_array * 16.0 / (2.0 * np.pi), 16.0)

    baseline_error = np.empty(len(fitted_h), dtype=np.float64)
    shifted_error = np.empty(len(fitted_h), dtype=np.float64)
    for component_index, phi_group in enumerate(component_phi_groups):
        candidates = expected_h(phi_group, -30.0)
        baseline_error[component_index] = float(
            np.min(circular_distance(fitted_h[component_index], candidates, 16.0))
        )
        shifted_error[component_index] = float(
            np.min(
                circular_distance(
                    fitted_h[component_index], np.mod(candidates + 3.0, 16.0), 16.0
                )
            )
        )

    point_h_error = np.zeros(len(indices), dtype=np.float32)
    point_h_within1 = np.zeros(len(indices), dtype=np.float32)
    for row in range(len(indices)):
        mask = component_point_array == row
        if not np.any(mask):
            continue
        local_weight = weight_array[mask]
        local_weight /= max(float(np.sum(local_weight)), TINY)
        point_h_error[row] = float(np.sum(local_weight * baseline_error[mask]))
        point_h_within1[row] = float(np.sum(local_weight * (baseline_error[mask] <= 1.0)))

    component_summary = {}
    for code, name in enumerate(("overall",) + REGIME_NAMES):
        mask = (
            np.ones(len(component_regime_array), dtype=bool)
            if code == 0
            else component_regime_array == code - 1
        )
        local_weight = weight_array[mask]
        local_weight /= max(float(np.sum(local_weight)), TINY)
        component_summary[name] = {
            "component_count": int(np.sum(mask)),
            "energy_weighted_h_error_bins_at_minus30": float(
                np.sum(local_weight * baseline_error[mask])
            ),
            "energy_weighted_shift3_control_error": float(
                np.sum(local_weight * shifted_error[mask])
            ),
            "energy_within_0p5_h_bin": float(
                np.sum(local_weight * (baseline_error[mask] <= 0.5))
            ),
            "energy_within_1_h_bin": float(
                np.sum(local_weight * (baseline_error[mask] <= 1.0))
            ),
            "energy_weighted_plane_wave_coherence": float(
                np.sum(local_weight * coherence_array[mask])
            ),
        }

    orientation_summary = {}
    group_names = ("overall",) + REGIME_NAMES
    for group_index, name in enumerate(group_names):
        local = objectives[sign_index, :, group_index]
        best_index = int(np.argmin(local))
        minus30_index = int(np.argmin(np.abs(ORIENTATION_GRID_DEG + 30.0)))
        orientation_summary[name] = {
            "best_orientation_deg": float(ORIENTATION_GRID_DEG[best_index]),
            "best_h_error_bins": float(local[best_index]),
            "minus30_h_error_bins": float(local[minus30_index]),
        }

    point_groups = {"overall": {}}
    for key, value in (
        ("plane_wave_coherence", point_plane_coherence),
        ("h_error_bins", point_h_error),
        ("h_energy_within_1bin", point_h_within1),
        ("component_count", point_component_count),
    ):
        point_groups["overall"][key] = summarize(
            value, np.ones(len(indices), dtype=bool)
        )
    for regime, name in enumerate(REGIME_NAMES):
        point_groups[name] = {}
        mask = regimes == regime
        for key, value in (
            ("plane_wave_coherence", point_plane_coherence),
            ("h_error_bins", point_h_error),
            ("h_energy_within_1bin", point_h_within1),
            ("component_count", point_component_count),
        ):
            point_groups[name][key] = summarize(value, mask)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=indices,
        regimes=regimes,
        point_plane_coherence=point_plane_coherence,
        point_h_error=point_h_error,
        point_h_within1=point_h_within1,
        point_component_count=point_component_count,
        component_point=component_point_array,
        component_regime=component_regime_array,
        component_h_step=h_step_array.astype(np.float32),
        component_weight=weight_array.astype(np.float32),
        component_coherence=coherence_array.astype(np.float32),
        component_h_error=baseline_error.astype(np.float32),
        orientation_grid_deg=ORIENTATION_GRID_DEG.astype(np.float32),
        orientation_objectives=objectives.astype(np.float32),
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The official-H complex path coefficients should remain plane-wave-like across "
            "the BS array and align with Sionna departure azimuths if the array orientation "
            "and path geometry are correct."
        ),
        "point_count": int(len(indices)),
        "path_limit": PATH_LIMIT,
        "fitted_h_step_sign": fitted_sign,
        "clear_selected_orientation_deg": best_clear_orientation,
        "orientation_scan": orientation_summary,
        "component_metrics": component_summary,
        "point_metrics_summary": point_groups,
        "decision": {
            "minus30_orientation_supported": bool(
                abs(orientation_summary["overall"]["best_orientation_deg"] + 30.0) <= 1.0
            ),
            "sionna_h_angles_supported": bool(
                component_summary["overall"]["energy_within_1_h_bin"] >= 0.80
            ),
            "single_plane_wave_model_supported": bool(
                component_summary["overall"]["energy_weighted_plane_wave_coherence"] >= 0.80
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
                "fitted_sign": fitted_sign,
                "overall_best_orientation": orientation_summary["overall"]["best_orientation_deg"],
                "overall_h_error": component_summary["overall"]["energy_weighted_h_error_bins_at_minus30"],
                "overall_energy_within1": component_summary["overall"]["energy_within_1_h_bin"],
                "overall_plane_coherence": component_summary["overall"]["energy_weighted_plane_wave_coherence"],
                "heavy_h_error": component_summary["heavy"]["energy_weighted_h_error_bins_at_minus30"],
                "heavy_plane_coherence": component_summary["heavy"]["energy_weighted_plane_wave_coherence"],
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
