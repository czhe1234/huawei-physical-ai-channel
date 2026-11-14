#!/usr/bin/env python3
"""Match official-H components to high-depth official-pointcloud geometry paths."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v309_unmatched_effective_scatterers as v309
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 312
VERSION = "v312_highdepth_path_identity_match"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MATCH_CACHE = OUT_DIR / "v312_highdepth_component_matches.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
SHUFFLE_REPEATS = 20
MIN_INTERACTION_COUNT = 2
LENGTH_SCALE_BINS = 0.50
H_SCALE_BINS = 0.75
GOOD_LENGTH_ERROR = 0.50
GOOD_H_ERROR = 1.00


def candidate_h(paths: dict[str, np.ndarray]) -> np.ndarray:
    vertices = paths["train_vertices"].astype(np.float32)
    first = vertices[:, :, 0].copy()
    direct = paths["train_interaction_count"] == 0
    first[direct] = np.broadcast_to(paths["train_pos"][:, None, :], first.shape)[direct]
    return v305.expected_h_bin(first)


def one_bounce_good_mask(point_count: int, component_count: int) -> np.ndarray:
    match = np.load(v305.MATCH_CACHE)
    good = (
        (match["length_error_bins"] <= v309.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v309.GOOD_H_ERROR)
    )
    mask = np.zeros((point_count, component_count), dtype=bool)
    mask[
        match["point_id"][good].astype(np.int64),
        match["component_id"][good].astype(np.int64),
    ] = True
    return mask


def collect_matches(
    observations: dict[str, np.ndarray],
    paths: dict[str, np.ndarray],
    path_h: np.ndarray,
    one_bounce_good: np.ndarray,
    source_permutation: np.ndarray | None = None,
    unmatched_only: bool = True,
) -> tuple[dict[str, np.ndarray], float]:
    count = len(observations["delay"])
    if source_permutation is None:
        source_permutation = np.arange(count)
    result: dict[str, list[float | int]] = {
        "point_id": [],
        "source_id": [],
        "component_id": [],
        "candidate_id": [],
        "interaction_count": [],
        "mass": [],
        "length_error_bins": [],
        "h_error_bins": [],
        "cost": [],
    }
    total_mass = 0.0
    for point_id in range(count):
        source_id = int(source_permutation[point_id])
        active = np.flatnonzero(
            observations["valid"][source_id]
            & (observations["mass"][source_id] >= v305.MIN_COMPONENT_MASS)
        )
        active = active[active != observations["los_component"][source_id]]
        if unmatched_only:
            active = active[~one_bounce_good[source_id, active]]
        if len(active) == 0:
            continue
        total_mass += float(np.sum(observations["mass"][source_id, active], dtype=np.float64))
        candidate = np.flatnonzero(
            paths["train_valid"][point_id]
            & (paths["train_interaction_count"][point_id] >= MIN_INTERACTION_COUNT)
        )
        if len(candidate) == 0:
            continue
        length_error = np.abs(
            observations["path_length"][source_id, active, None]
            - paths["train_path_length"][point_id, candidate][None, :]
        ) / v304.METERS_PER_BIN
        h_error = v305.circular_distance(
            observations["h_center"][source_id, active, None],
            path_h[point_id, candidate][None, :],
            16.0,
        )
        cost = (length_error / LENGTH_SCALE_BINS) ** 2 + (h_error / H_SCALE_BINS) ** 2
        row, column = linear_sum_assignment(cost)
        selected = candidate[column]
        result["point_id"].extend([point_id] * len(row))
        result["source_id"].extend([source_id] * len(row))
        result["component_id"].extend(active[row].tolist())
        result["candidate_id"].extend(selected.tolist())
        result["interaction_count"].extend(
            paths["train_interaction_count"][point_id, selected].tolist()
        )
        result["mass"].extend(observations["mass"][source_id, active[row]].tolist())
        result["length_error_bins"].extend(length_error[row, column].tolist())
        result["h_error_bins"].extend(h_error[row, column].tolist())
        result["cost"].extend(cost[row, column].tolist())
    integer_keys = ("point_id", "source_id", "component_id", "candidate_id", "interaction_count")
    output = {
        key: np.asarray(value, dtype=np.int32 if key in integer_keys else np.float32)
        for key, value in result.items()
    }
    return output, total_mass


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(
        np.sum(np.asarray(value, dtype=np.float64) * np.asarray(weight, dtype=np.float64))
        / np.maximum(np.sum(weight, dtype=np.float64), 1e-30)
    )


def summarize_match(match: dict[str, np.ndarray], total_mass: float) -> dict[str, object]:
    mass = match["mass"]
    good = (
        (match["length_error_bins"] <= GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= GOOD_H_ERROR)
    )
    strict = (
        (match["length_error_bins"] <= 0.25)
        & (match["h_error_bins"] <= 0.50)
    )
    good_mass = float(np.sum(mass[good], dtype=np.float64))
    strict_mass = float(np.sum(mass[strict], dtype=np.float64))
    depth, occurrence = np.unique(match["interaction_count"][good], return_counts=True)
    return {
        "count": int(len(mass)),
        "total_observed_mass": float(total_mass),
        "length_error_bins": summarize(match["length_error_bins"]),
        "h_error_bins": summarize(match["h_error_bins"]),
        "mass_weighted_length_error": weighted_mean(match["length_error_bins"], mass),
        "mass_weighted_h_error": weighted_mean(match["h_error_bins"], mass),
        "matched_mass_fraction_strict": strict_mass / max(total_mass, 1e-30),
        "matched_mass_fraction_good": good_mass / max(total_mass, 1e-30),
        "good_match_depth": {
            "depth": depth.tolist(),
            "count": occurrence.tolist(),
            "summary": summarize(match["interaction_count"][good]),
        },
    }


def main() -> None:
    for path in (PATH_CACHE, v305.MATCH_CACHE, v297.COMPONENT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    observations = v305.load_observations()
    path_data = np.load(PATH_CACHE)
    paths = {key: path_data[key] for key in path_data.files}
    path_h = candidate_h(paths)
    one_bounce_good = one_bounce_good_mask(
        len(observations["delay"]), observations["delay"].shape[1]
    )

    actual_unmatched, actual_unmatched_mass = collect_matches(
        observations, paths, path_h, one_bounce_good, unmatched_only=True
    )
    actual_all, actual_all_mass = collect_matches(
        observations, paths, path_h, one_bounce_good, unmatched_only=False
    )
    np.savez_compressed(MATCH_CACHE, **actual_unmatched)
    actual_unmatched_summary = summarize_match(actual_unmatched, actual_unmatched_mass)
    actual_all_summary = summarize_match(actual_all, actual_all_mass)

    rng = np.random.default_rng(312)
    control_rows = []
    for repeat in range(SHUFFLE_REPEATS):
        permutation = rng.permutation(len(observations["delay"]))
        shuffled, shuffled_mass = collect_matches(
            observations,
            paths,
            path_h,
            one_bounce_good,
            source_permutation=permutation,
            unmatched_only=True,
        )
        summary = summarize_match(shuffled, shuffled_mass)
        control_rows.append(
            {
                "repeat": repeat,
                "mass_weighted_length_error": summary["mass_weighted_length_error"],
                "mass_weighted_h_error": summary["mass_weighted_h_error"],
                "matched_mass_fraction_strict": summary["matched_mass_fraction_strict"],
                "matched_mass_fraction_good": summary["matched_mass_fraction_good"],
            }
        )
        print(f"v312 shuffled control {repeat + 1}/{SHUFFLE_REPEATS}", flush=True)
    control_good = np.asarray([row["matched_mass_fraction_good"] for row in control_rows])
    control_strict = np.asarray([row["matched_mass_fraction_strict"] for row in control_rows])
    control = {
        "matched_mass_fraction_good": summarize(control_good),
        "matched_mass_fraction_strict": summarize(control_strict),
        "actual_good_minus_control": float(
            actual_unmatched_summary["matched_mass_fraction_good"] - np.mean(control_good)
        ),
        "actual_strict_minus_control": float(
            actual_unmatched_summary["matched_mass_fraction_strict"] - np.mean(control_strict)
        ),
        "repeats": control_rows,
    }
    passed = (
        actual_unmatched_summary["matched_mass_fraction_good"]
        >= float(np.mean(control_good)) + 0.20
        and actual_unmatched_summary["matched_mass_fraction_strict"]
        >= float(np.mean(control_strict)) + 0.15
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Recovered absolute length and BS departure angle identify depth>=2 "
            "official-pointcloud Sionna geometry paths for components not explained by one-bounce walls."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H continuous path observations",
                "official train coordinates and point-cloud-derived geometry paths",
                "Sionna PathSolver geometry/visibility only",
            ],
            "does_not_use": [
                "Paths.cfr or path coefficients",
                "synthetic channel labels or teacher output",
                "test truth or local replay truth",
            ],
        },
        "parameters": {
            "minimum_interaction_count": MIN_INTERACTION_COUNT,
            "length_scale_bins": LENGTH_SCALE_BINS,
            "h_scale_bins": H_SCALE_BINS,
            "good_length_error": GOOD_LENGTH_ERROR,
            "good_h_error": GOOD_H_ERROR,
            "shuffle_repeats": SHUFFLE_REPEATS,
        },
        "actual_unmatched_one_bounce": actual_unmatched_summary,
        "actual_all_non_los": actual_all_summary,
        "shuffle_control": control,
        "decision_rule": {
            "good_mass_margin_over_shuffle": "+0.20",
            "strict_mass_margin_over_shuffle": "+0.15",
            "passed": bool(passed),
        },
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
