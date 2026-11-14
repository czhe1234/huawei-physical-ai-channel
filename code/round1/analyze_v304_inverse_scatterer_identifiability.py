#!/usr/bin/env python3
"""Infer effective scatterers from official-H path delay and BS angle."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import DBSCAN
from sklearn.linear_model import HuberRegressor

import analyze_v265_scatterer_delay_angle_oracle as v265
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 304
VERSION = "v304_inverse_scatterer_identifiability"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SCATTERER_CACHE = OUT_DIR / "v304_inferred_scatterers.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

BS = v265.BS
ORIENTATION_DEGREES = -30.0
SUBCARRIER_SPACING = 30_000.0
SPEED_OF_LIGHT = 299_792_458.0
METERS_PER_BIN = SPEED_OF_LIGHT / (v297.N_DELAY * SUBCARRIER_SPACING)
MIN_PATH_MASS = 2e-3


def circular_center(profile: np.ndarray) -> np.ndarray:
    bins = np.arange(profile.shape[-1], dtype=np.float64)
    vector = np.sum(
        profile * np.exp(2j * np.pi * bins / profile.shape[-1]), axis=-1
    )
    return (
        np.angle(vector) % (2.0 * np.pi)
    ) * profile.shape[-1] / (2.0 * np.pi)


def horizontal_center(profile: np.ndarray) -> np.ndarray:
    peak = np.argmax(profile, axis=-1)
    output = peak.astype(np.float64)
    log_profile = np.log(np.maximum(profile, 1e-30))
    flat_peak = peak.reshape(-1)
    flat = log_profile.reshape(-1, profile.shape[-1])
    refined = np.empty(len(flat), dtype=np.float64)
    for row, index in enumerate(flat_peak):
        left = flat[row, (index - 1) % profile.shape[-1]]
        center = flat[row, index]
        right = flat[row, (index + 1) % profile.shape[-1]]
        denominator = left - 2.0 * center + right
        offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (left - right) / denominator
        refined[row] = (index + np.clip(offset, -0.5, 0.5)) % profile.shape[-1]
    output[...] = refined.reshape(output.shape)
    return output.astype(np.float32)


def expected_h_bin(azimuth: np.ndarray) -> np.ndarray:
    relative = azimuth - math.radians(ORIENTATION_DEGREES)
    return np.mod(np.sin(relative) * 8.0, 16.0).astype(np.float32)


def circular_bin_distance(value: np.ndarray, target: np.ndarray, size: float) -> np.ndarray:
    delta = np.abs(np.asarray(value) - np.asarray(target))
    return np.minimum(delta, size - delta)


def select_los_components(
    train_pos: np.ndarray,
    delay: np.ndarray,
    mass: np.ndarray,
    valid: np.ndarray,
    horizontal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = train_pos[:, :2] - BS[:2]
    azimuth = np.arctan2(vector[:, 1], vector[:, 0])
    expected = expected_h_bin(azimuth)
    centers = horizontal_center(horizontal)
    selected = np.zeros(len(train_pos), dtype=np.int64)
    confidence = np.zeros(len(train_pos), dtype=np.float32)
    angle_error = np.zeros(len(train_pos), dtype=np.float32)
    bins = np.arange(16, dtype=np.float32)
    for index in range(len(train_pos)):
        active = np.flatnonzero(valid[index])
        if len(active) == 0:
            continue
        distance = circular_bin_distance(centers[index, active], expected[index], 16.0)
        kernel = np.exp(-0.5 * (circular_bin_distance(bins, expected[index], 16.0) / 0.8) ** 2)
        capture = horizontal[index, active] @ kernel
        score = capture * np.sqrt(np.maximum(mass[index, active], 1e-12)) * np.exp(-0.35 * distance)
        choice = int(active[int(np.argmax(score))])
        selected[index] = choice
        confidence[index] = float(np.max(score))
        angle_error[index] = float(circular_bin_distance(centers[index, choice], expected[index], 16.0))
    return selected, confidence, angle_error


def fit_delay_calibration(
    distance: np.ndarray,
    delay: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    x = distance[mask, None]
    y = delay[mask]
    model = HuberRegressor(epsilon=1.35, alpha=1e-4).fit(x, y)
    prediction = model.predict(x)
    residual = y - prediction
    return {
        "slope_bins_per_meter": float(model.coef_[0]),
        "intercept_bins": float(model.intercept_),
        "pearson": float(pearsonr(distance[mask], delay[mask]).statistic),
        "spearman": float(spearmanr(distance[mask], delay[mask]).statistic),
        "residual": summarize(residual),
        "count": int(np.sum(mask)),
    }


def h_bin_azimuth_solutions(center: float) -> tuple[float, float]:
    signed = center if center <= 8.0 else center - 16.0
    value = float(np.clip(signed / 8.0, -1.0, 1.0))
    first = math.asin(value)
    second = math.pi - first
    orientation = math.radians(ORIENTATION_DEGREES)
    return first + orientation, second + orientation


def reflection_point(
    ue_xy: np.ndarray,
    total_length: float,
    azimuth: float,
) -> np.ndarray | None:
    vector = np.asarray(ue_xy, dtype=np.float64) - BS[:2]
    direct = float(np.linalg.norm(vector))
    direction = np.asarray([math.cos(azimuth), math.sin(azimuth)], dtype=np.float64)
    denominator = 2.0 * (total_length - float(np.dot(direction, vector)))
    if total_length <= direct + 1e-6 or denominator <= 1e-9:
        return None
    distance = (total_length**2 - direct**2) / denominator
    if distance <= 0.0 or distance >= total_length:
        return None
    return BS[:2] + distance * direction


def wall_distance(points: np.ndarray, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = patches[:, :2].astype(np.float64)
    c = patches[:, 2].astype(np.float64)
    t_min = patches[:, 3].astype(np.float64)
    t_max = patches[:, 4].astype(np.float64)
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    best_distance = np.empty(len(points), dtype=np.float32)
    best_patch = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), 512):
        end = min(start + 512, len(points))
        value = points[start:end]
        normal_error = np.abs(value @ normal.T - c[None, :])
        coordinate = value @ tangent.T
        tangent_error = np.maximum(
            np.maximum(t_min[None, :] - coordinate, coordinate - t_max[None, :]), 0.0
        )
        distance = np.sqrt(normal_error**2 + tangent_error**2)
        index = np.argmin(distance, axis=1)
        best_distance[start:end] = distance[np.arange(len(value)), index]
        best_patch[start:end] = index.astype(np.int32)
    return best_distance, best_patch


def infer_scatterers(
    mode: str,
    train_pos: np.ndarray,
    delay: np.ndarray,
    mass: np.ndarray,
    valid: np.ndarray,
    h_center: np.ndarray,
    los_component: np.ndarray,
    calibration: dict[str, object],
    patches: np.ndarray,
) -> dict[str, np.ndarray]:
    points = []
    point_ids = []
    component_ids = []
    weights = []
    total_lengths = []
    for point_id in range(len(train_pos)):
        direct = float(np.linalg.norm(train_pos[point_id, :2] - BS[:2]))
        los_delay = float(delay[point_id, los_component[point_id]])
        for component_id in np.flatnonzero(valid[point_id]):
            if component_id == los_component[point_id] or mass[point_id, component_id] < MIN_PATH_MASS:
                continue
            current_delay = float(delay[point_id, component_id])
            if mode == "relative_first_path":
                excess_bin = float(
                    (current_delay - los_delay + v297.N_DELAY / 2.0)
                    % v297.N_DELAY
                    - v297.N_DELAY / 2.0
                )
                if excess_bin <= 0.0:
                    continue
                total_length = direct + excess_bin * METERS_PER_BIN
            else:
                slope = float(calibration["slope_bins_per_meter"])
                intercept = float(calibration["intercept_bins"])
                if slope <= 1e-6:
                    continue
                total_length = (current_delay - intercept) / slope
            if total_length <= direct + 1.0 or total_length > 2500.0:
                continue
            candidates = [
                reflection_point(train_pos[point_id, :2], total_length, azimuth)
                for azimuth in h_bin_azimuth_solutions(float(h_center[point_id, component_id]))
            ]
            candidates = [value for value in candidates if value is not None]
            if not candidates:
                continue
            candidate_array = np.asarray(candidates)
            distances, _patch = wall_distance(candidate_array, patches)
            selected = int(np.argmin(distances))
            points.append(candidate_array[selected])
            point_ids.append(point_id)
            component_ids.append(component_id)
            weights.append(float(mass[point_id, component_id]))
            total_lengths.append(total_length)
    if not points:
        return {
            "point": np.empty((0, 2), dtype=np.float32),
            "point_id": np.empty(0, dtype=np.int32),
            "component_id": np.empty(0, dtype=np.int16),
            "weight": np.empty(0, dtype=np.float32),
            "total_length": np.empty(0, dtype=np.float32),
            "wall_distance": np.empty(0, dtype=np.float32),
            "wall_patch": np.empty(0, dtype=np.int32),
        }
    point = np.asarray(points, dtype=np.float32)
    distance, patch = wall_distance(point, patches)
    return {
        "point": point,
        "point_id": np.asarray(point_ids, dtype=np.int32),
        "component_id": np.asarray(component_ids, dtype=np.int16),
        "weight": np.asarray(weights, dtype=np.float32),
        "total_length": np.asarray(total_lengths, dtype=np.float32),
        "wall_distance": distance,
        "wall_patch": patch,
    }


def scatterer_summary(data: dict[str, np.ndarray]) -> dict[str, object]:
    points = data["point"]
    if len(points) == 0:
        return {"count": 0}
    labels = DBSCAN(eps=2.0, min_samples=5).fit_predict(points)
    clustered = labels >= 0
    tree = cKDTree(points)
    distance, index = tree.query(points, k=2)
    different_source = data["point_id"][index[:, 1]] != data["point_id"]
    return {
        "count": int(len(points)),
        "wall_distance": summarize(data["wall_distance"]),
        "wall_within_0.5m": float(np.mean(data["wall_distance"] <= 0.5)),
        "wall_within_1m": float(np.mean(data["wall_distance"] <= 1.0)),
        "wall_within_2m": float(np.mean(data["wall_distance"] <= 2.0)),
        "wall_within_5m": float(np.mean(data["wall_distance"] <= 5.0)),
        "cluster_count_eps2_min5": int(len(set(labels[clustered]))),
        "clustered_fraction": float(np.mean(clustered)),
        "nearest_other_distance": summarize(distance[different_source, 1]),
    }


def main() -> None:
    component = np.load(v297.COMPONENT_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    delay = component["delay"].astype(np.float32)
    mass = component["weight"].astype(np.float32)
    valid = component["valid"].astype(bool)
    horizontal = fingerprint["horizontal"].astype(np.float32)
    h_center = horizontal_center(horizontal)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    (
        _all_pos,
        _signatures,
        blocker_count,
        _scene_features,
        _footprint,
        _z_max,
        _x_min,
        _y_min,
        _resolution,
    ) = v274.load_geometry()
    scene = np.load(v274.SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float64)
    los_component, los_confidence, los_angle_error = select_los_components(
        train_pos, delay, mass, valid, horizontal
    )
    los_delay = delay[np.arange(len(delay)), los_component]
    direct_distance = np.linalg.norm(train_pos[:, :3] - BS[None, :], axis=1)
    clear = blocker_count[: len(train_pos)] == 0
    confidence_cut = np.quantile(los_confidence[clear], 0.35)
    calibration_mask = clear & (los_confidence >= confidence_cut) & (los_angle_error <= 1.5)
    calibration = fit_delay_calibration(
        direct_distance, los_delay.astype(np.float64), calibration_mask
    )
    calibration["expected_slope_bins_per_meter"] = 1.0 / METERS_PER_BIN
    calibration["los_delay"] = summarize(los_delay[calibration_mask])
    calibration["los_angle_error_bins"] = summarize(los_angle_error[calibration_mask])
    calibration["los_confidence"] = summarize(los_confidence[calibration_mask])

    inferred = {
        mode: infer_scatterers(
            mode,
            train_pos,
            delay,
            mass,
            valid,
            h_center,
            los_component,
            calibration,
            patches,
        )
        for mode in ("relative_first_path", "absolute_delay")
    }
    np.savez_compressed(
        SCATTERER_CACHE,
        relative_point=inferred["relative_first_path"]["point"],
        relative_point_id=inferred["relative_first_path"]["point_id"],
        relative_component_id=inferred["relative_first_path"]["component_id"],
        relative_weight=inferred["relative_first_path"]["weight"],
        relative_wall_distance=inferred["relative_first_path"]["wall_distance"],
        relative_wall_patch=inferred["relative_first_path"]["wall_patch"],
        absolute_point=inferred["absolute_delay"]["point"],
        absolute_point_id=inferred["absolute_delay"]["point_id"],
        absolute_component_id=inferred["absolute_delay"]["component_id"],
        absolute_weight=inferred["absolute_delay"]["weight"],
        absolute_wall_distance=inferred["absolute_delay"]["wall_distance"],
        absolute_wall_patch=inferred["absolute_delay"]["wall_patch"],
        los_component=los_component,
        los_confidence=los_confidence,
        los_angle_error=los_angle_error,
    )
    summaries = {mode: scatterer_summary(value) for mode, value in inferred.items()}
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "parameters": {
            "bs": BS.tolist(),
            "array_orientation_degrees": ORIENTATION_DEGREES,
            "subcarrier_spacing_hz": SUBCARRIER_SPACING,
            "meters_per_delay_bin": METERS_PER_BIN,
        },
        "los_calibration": calibration,
        "scatterer_modes": summaries,
        "identifiability_rule": (
            "Proceed only if one delay convention yields physically plausible LOS "
            "calibration and inferred points show strong repeated wall alignment."
        ),
        "outputs": {
            "scatterers": PATHS.relative(SCATTERER_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
