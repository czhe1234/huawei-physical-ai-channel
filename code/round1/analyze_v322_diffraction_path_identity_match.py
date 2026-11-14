#!/usr/bin/env python3
"""Match residual official components to type-8 diffraction geometry paths."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v309_unmatched_effective_scatterers as v309
import analyze_v312_highdepth_path_identity_match as v312
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 322
VERSION = "v322_diffraction_path_identity_match"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MATCH_CACHE = OUT_DIR / "v322_diffraction_component_matches.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
PATH_CACHE = PATHS.cache_dir(321) / "v321_highdepth_diffraction_paths.npz"

SHUFFLE_REPEATS = 20
LENGTH_SCALE_BINS = 0.50
H_SCALE_BINS = 0.75
GOOD_LENGTH_ERROR = 0.50
GOOD_H_ERROR = 1.00


def explained_mask(point_count: int, component_count: int) -> np.ndarray:
    mask = np.zeros((point_count, component_count), dtype=bool)
    one = np.load(v305.MATCH_CACHE)
    good_one = (
        (one["length_error_bins"] <= v309.GOOD_LENGTH_ERROR)
        & (one["h_error_bins"] <= v309.GOOD_H_ERROR)
    )
    mask[
        one["point_id"][good_one].astype(np.int64),
        one["component_id"][good_one].astype(np.int64),
    ] = True
    high = np.load(v312.MATCH_CACHE)
    good_high = (
        (high["length_error_bins"] <= v312.GOOD_LENGTH_ERROR)
        & (high["h_error_bins"] <= v312.GOOD_H_ERROR)
    )
    mask[
        high["point_id"][good_high].astype(np.int64),
        high["component_id"][good_high].astype(np.int64),
    ] = True
    return mask


def candidate_h(paths: dict[str, np.ndarray]) -> np.ndarray:
    vertices = np.nan_to_num(paths["train_vertices"].astype(np.float32), nan=0.0)
    first = vertices[:, :, 0]
    return v305.expected_h_bin(first)


def collect_matches(
    observations: dict[str, np.ndarray],
    paths: dict[str, np.ndarray],
    path_h: np.ndarray,
    explained: np.ndarray,
    source_permutation: np.ndarray | None = None,
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
            & ~explained[source_id]
        )
        active = active[active != observations["los_component"][source_id]]
        if len(active) == 0:
            continue
        total_mass += float(np.sum(observations["mass"][source_id, active], dtype=np.float64))
        candidate = np.flatnonzero(
            paths["train_valid"][point_id]
            & ((paths["train_interaction_mask"][point_id] & 8) != 0)
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
    integer = {"point_id", "source_id", "component_id", "candidate_id", "interaction_count"}
    output = {
        key: np.asarray(value, dtype=np.int32 if key in integer else np.float32)
        for key, value in result.items()
    }
    return output, total_mass


def summary(match: dict[str, np.ndarray], total_mass: float) -> dict[str, object]:
    mass = match["mass"]
    good = (
        (match["length_error_bins"] <= GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= GOOD_H_ERROR)
    )
    strict = (
        (match["length_error_bins"] <= 0.25)
        & (match["h_error_bins"] <= 0.50)
    )
    return {
        "count": int(len(mass)),
        "total_observed_mass": float(total_mass),
        "length_error_bins": summarize(match["length_error_bins"]),
        "h_error_bins": summarize(match["h_error_bins"]),
        "matched_mass_fraction_good": float(
            np.sum(mass[good], dtype=np.float64) / max(total_mass, 1e-30)
        ),
        "matched_mass_fraction_strict": float(
            np.sum(mass[strict], dtype=np.float64) / max(total_mass, 1e-30)
        ),
        "good_depth": summarize(match["interaction_count"][good]),
    }


def main() -> None:
    observations = v305.load_observations()
    path_data = np.load(PATH_CACHE)
    paths = {key: path_data[key] for key in path_data.files}
    path_h = candidate_h(paths)
    explained = explained_mask(len(observations["delay"]), observations["delay"].shape[1])
    actual, actual_mass = collect_matches(observations, paths, path_h, explained)
    np.savez_compressed(MATCH_CACHE, **actual)
    actual_summary = summary(actual, actual_mass)

    rng = np.random.default_rng(322)
    control_rows = []
    for repeat in range(SHUFFLE_REPEATS):
        shuffled, shuffled_mass = collect_matches(
            observations,
            paths,
            path_h,
            explained,
            source_permutation=rng.permutation(len(observations["delay"])),
        )
        current = summary(shuffled, shuffled_mass)
        control_rows.append(
            {
                "repeat": repeat,
                "matched_mass_fraction_good": current["matched_mass_fraction_good"],
                "matched_mass_fraction_strict": current["matched_mass_fraction_strict"],
            }
        )
        print(f"v322 shuffle {repeat + 1}/{SHUFFLE_REPEATS}", flush=True)
    control_good = np.asarray([row["matched_mass_fraction_good"] for row in control_rows])
    control_strict = np.asarray([row["matched_mass_fraction_strict"] for row in control_rows])
    good_margin = float(actual_summary["matched_mass_fraction_good"] - np.mean(control_good))
    strict_margin = float(
        actual_summary["matched_mass_fraction_strict"] - np.mean(control_strict)
    )
    passed = good_margin >= 0.20 and strict_margin >= 0.15
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Type-8 paths from the high-budget official-pointcloud solver explain components "
            "remaining after one-bounce and specular/refraction high-depth matches."
        ),
        "candidate_coverage": {
            "train_type8_path_count": int(
                np.sum(paths["train_valid"] & ((paths["train_interaction_mask"] & 8) != 0))
            ),
            "train_ue_with_type8": int(
                np.sum(
                    np.any(
                        paths["train_valid"]
                        & ((paths["train_interaction_mask"] & 8) != 0),
                        axis=1,
                    )
                )
            ),
        },
        "actual_residual": actual_summary,
        "shuffle_control": {
            "matched_mass_fraction_good": summarize(control_good),
            "matched_mass_fraction_strict": summarize(control_strict),
            "actual_good_minus_control": good_margin,
            "actual_strict_minus_control": strict_margin,
            "repeats": control_rows,
        },
        "decision_rule": {
            "good_mass_margin": ">=0.20",
            "strict_mass_margin": ">=0.15",
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
