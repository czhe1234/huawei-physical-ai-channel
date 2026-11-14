#!/usr/bin/env python3
"""Match recovered official-H paths to official-pointcloud reflection paths."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

import analyze_v265_scatterer_delay_angle_oracle as v265
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 305
VERSION = "v305_pointcloud_path_identity_match"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MATCH_CACHE = OUT_DIR / "v305_train_component_patch_matches.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
SHUFFLE_REPEATS = 20
LENGTH_SCALE_BINS = 0.50
H_SCALE_BINS = 0.75
MIN_COMPONENT_MASS = 2e-3


def expected_h_bin(reflection_point: np.ndarray) -> np.ndarray:
    vector = reflection_point[..., :2] - v265.BS[:2]
    azimuth = np.arctan2(vector[..., 1], vector[..., 0])
    relative = azimuth - math.radians(v304.ORIENTATION_DEGREES)
    return np.mod(np.sin(relative) * 8.0, 16.0).astype(np.float32)


def circular_distance(left: np.ndarray, right: np.ndarray, size: float) -> np.ndarray:
    delta = np.abs(np.asarray(left) - np.asarray(right))
    return np.minimum(delta, size - delta)


def load_observations() -> dict[str, np.ndarray]:
    component = np.load(v297.COMPONENT_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    inverse = np.load(v304.SCATTERER_CACHE)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    delay = component["delay"].astype(np.float32)
    mass = component["weight"].astype(np.float32)
    valid = component["valid"].astype(bool)
    h_center = v304.horizontal_center(fingerprint["horizontal"].astype(np.float32))
    los_component = inverse["los_component"].astype(np.int64)
    path_length = (
        delay - float(calibration["intercept_bins"])
    ) / float(calibration["slope_bins_per_meter"])
    return {
        "delay": delay,
        "mass": mass,
        "valid": valid,
        "h_center": h_center,
        "los_component": los_component,
        "path_length": path_length.astype(np.float32),
    }


def match_point(
    observed_length: np.ndarray,
    observed_h: np.ndarray,
    candidate_length: np.ndarray,
    candidate_h: np.ndarray,
    candidate_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.flatnonzero(candidate_valid)
    if len(observed_length) == 0 or len(candidates) == 0:
        empty = np.empty(0, dtype=np.float32)
        return np.empty(0, dtype=np.int64), empty, empty, empty
    length_error = np.abs(
        observed_length[:, None] - candidate_length[candidates][None, :]
    ) / v304.METERS_PER_BIN
    h_error = circular_distance(
        observed_h[:, None], candidate_h[candidates][None, :], 16.0
    )
    cost = (length_error / LENGTH_SCALE_BINS) ** 2 + (h_error / H_SCALE_BINS) ** 2
    row, column = linear_sum_assignment(cost)
    selected = candidates[column]
    return (
        selected.astype(np.int64),
        length_error[row, column].astype(np.float32),
        h_error[row, column].astype(np.float32),
        cost[row, column].astype(np.float32),
    )


def collect_matches(
    observations: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    source_permutation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    point_ids = []
    component_ids = []
    candidate_ids = []
    patch_ids = []
    masses = []
    length_errors = []
    h_errors = []
    costs = []
    candidate_h = expected_h_bin(candidates["reflection_point"])
    count = len(observations["delay"])
    if source_permutation is None:
        source_permutation = np.arange(count)
    for point_id in range(count):
        source_id = int(source_permutation[point_id])
        active = np.flatnonzero(
            observations["valid"][source_id]
            & (observations["mass"][source_id] >= MIN_COMPONENT_MASS)
        )
        active = active[active != observations["los_component"][source_id]]
        if len(active) == 0:
            continue
        candidate_valid = candidates["score"][point_id] > 0.0
        selected, length_error, h_error, cost = match_point(
            observations["path_length"][source_id, active],
            observations["h_center"][source_id, active],
            candidates["path_length"][point_id],
            candidate_h[point_id],
            candidate_valid,
        )
        if len(selected) == 0:
            continue
        matched_active = active[: len(selected)]
        point_ids.extend([point_id] * len(selected))
        component_ids.extend(matched_active.tolist())
        candidate_ids.extend(selected.tolist())
        patch_ids.extend(candidates["patch_id"][point_id, selected].tolist())
        masses.extend(observations["mass"][source_id, matched_active].tolist())
        length_errors.extend(length_error.tolist())
        h_errors.extend(h_error.tolist())
        costs.extend(cost.tolist())
    return {
        "point_id": np.asarray(point_ids, dtype=np.int32),
        "component_id": np.asarray(component_ids, dtype=np.int16),
        "candidate_id": np.asarray(candidate_ids, dtype=np.int16),
        "patch_id": np.asarray(patch_ids, dtype=np.int32),
        "mass": np.asarray(masses, dtype=np.float32),
        "length_error_bins": np.asarray(length_errors, dtype=np.float32),
        "h_error_bins": np.asarray(h_errors, dtype=np.float32),
        "cost": np.asarray(costs, dtype=np.float32),
    }


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(
        np.sum(np.asarray(value, dtype=np.float64) * np.asarray(weight, dtype=np.float64))
        / np.maximum(np.sum(weight, dtype=np.float64), 1e-30)
    )


def match_summary(match: dict[str, np.ndarray]) -> dict[str, object]:
    mass = match["mass"]
    good_025 = (match["length_error_bins"] <= 0.25) & (match["h_error_bins"] <= 0.5)
    good_050 = (match["length_error_bins"] <= 0.50) & (match["h_error_bins"] <= 1.0)
    patch, occurrence = np.unique(match["patch_id"], return_counts=True)
    return {
        "count": int(len(mass)),
        "length_error_bins": summarize(match["length_error_bins"]),
        "h_error_bins": summarize(match["h_error_bins"]),
        "cost": summarize(match["cost"]),
        "mass_weighted_length_error": weighted_mean(match["length_error_bins"], mass),
        "mass_weighted_h_error": weighted_mean(match["h_error_bins"], mass),
        "mass_fraction_length025_h050": weighted_mean(good_025.astype(np.float32), mass),
        "mass_fraction_length050_h100": weighted_mean(good_050.astype(np.float32), mass),
        "unique_patch_count": int(len(patch)),
        "patch_occurrence": summarize(occurrence),
    }


def main() -> None:
    for path in (TRAIN_CANDIDATES, v297.COMPONENT_CACHE, v300.FINGERPRINT_CACHE, v304.REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    observations = load_observations()
    candidate_data = np.load(TRAIN_CANDIDATES)
    candidates = {key: candidate_data[key] for key in candidate_data.files}
    actual = collect_matches(observations, candidates)
    np.savez_compressed(MATCH_CACHE, **actual)
    rng = np.random.default_rng(305)
    shuffled_summaries = []
    shuffled_costs = []
    shuffled_good = []
    for repeat in range(SHUFFLE_REPEATS):
        permutation = rng.permutation(len(observations["delay"]))
        shuffled = collect_matches(observations, candidates, permutation)
        summary = match_summary(shuffled)
        shuffled_summaries.append(
            {
                "repeat": repeat,
                "mass_weighted_length_error": summary["mass_weighted_length_error"],
                "mass_weighted_h_error": summary["mass_weighted_h_error"],
                "mass_fraction_length025_h050": summary["mass_fraction_length025_h050"],
                "mass_fraction_length050_h100": summary["mass_fraction_length050_h100"],
            }
        )
        shuffled_costs.append(weighted_mean(shuffled["cost"], shuffled["mass"]))
        shuffled_good.append(summary["mass_fraction_length050_h100"])
        print(f"v305 shuffled control {repeat + 1}/{SHUFFLE_REPEATS}", flush=True)

    actual_summary = match_summary(actual)
    actual_weighted_cost = weighted_mean(actual["cost"], actual["mass"])
    control = {
        "mass_weighted_cost": summarize(np.asarray(shuffled_costs)),
        "mass_fraction_length050_h100": summarize(np.asarray(shuffled_good)),
        "actual_vs_control_cost_ratio": float(
            actual_weighted_cost / np.maximum(np.mean(shuffled_costs), 1e-30)
        ),
        "actual_good_minus_control_mean": float(
            actual_summary["mass_fraction_length050_h100"] - np.mean(shuffled_good)
        ),
        "repeats": shuffled_summaries,
    }
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Recovered absolute path length and departure angle identify the same "
            "official-pointcloud reflection candidate significantly better than shuffled controls."
        ),
        "parameters": {
            "candidate_count_per_point": int(candidates["path_length"].shape[1]),
            "length_scale_bins": LENGTH_SCALE_BINS,
            "h_scale_bins": H_SCALE_BINS,
            "minimum_component_mass": MIN_COMPONENT_MASS,
            "shuffle_repeats": SHUFFLE_REPEATS,
        },
        "actual": actual_summary,
        "control": control,
        "outputs": {
            "matches": PATHS.relative(MATCH_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
