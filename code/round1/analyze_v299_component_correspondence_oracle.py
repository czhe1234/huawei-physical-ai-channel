#!/usr/bin/env python3
"""Diagnose coverage, ambiguity, and affine motion of continuous delay sets."""

from __future__ import annotations

import json

import numpy as np

import analyze_v254_route_topology_spectral_field as v254
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 299
VERSION = "v299_component_correspondence_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

OUTER_SEEDS = (299, 1299, 2299, 3299, 4299)
OUTER_COUNT = 320
COVERAGE_THRESHOLDS = (0.10, 0.25, 0.50, 1.00)
AFFINE_MATCH_THRESHOLD = 1.00
AFFINE_RIDGE = 1e-3
MAX_DELAY_GRADIENT = 0.03


def signed_circular_delta(value: np.ndarray, reference: float) -> np.ndarray:
    return (
        np.asarray(value, dtype=np.float64)
        - float(reference)
        + v297.N_DELAY / 2.0
    ) % v297.N_DELAY - v297.N_DELAY / 2.0


def fit_affine_at_query(
    delta_xy: np.ndarray,
    observed_delay: np.ndarray,
    weight: np.ndarray,
    reference: float,
) -> float:
    unwrapped = reference + signed_circular_delta(observed_delay, reference)
    design = np.concatenate(
        [np.ones((len(delta_xy), 1), dtype=np.float64), delta_xy], axis=1
    )
    root_weight = np.sqrt(np.maximum(weight, 1e-12))
    weighted_design = design * root_weight[:, None]
    weighted_target = unwrapped * root_weight
    penalty = np.diag([0.0, AFFINE_RIDGE, AFFINE_RIDGE])
    coefficient = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    gradient = coefficient[1:]
    norm = float(np.linalg.norm(gradient))
    if norm > MAX_DELAY_GRADIENT:
        coefficient[1:] *= MAX_DELAY_GRADIENT / norm
        coefficient[0] = float(
            np.sum(weight * (unwrapped - delta_xy @ coefficient[1:]))
            / np.maximum(np.sum(weight), 1e-20)
        )
    return float(coefficient[0] % v297.N_DELAY)


def analyze_fold(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    all_pos: np.ndarray,
    signatures: np.ndarray,
    blocker_count: np.ndarray,
    footprint: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    delay: np.ndarray,
    mass: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    raw = v274.build_raw_set(
        query_ids,
        bank_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    result: dict[str, list[float]] = {
        "nearest_distance": [],
        "nearest_spatial_distance": [],
        "weighted_mean_distance": [],
        "affine_distance": [],
        "affine_gain": [],
        "matched_neighbor_count": [],
        "candidate_ambiguity_025": [],
        "candidate_ambiguity_050": [],
    }
    for threshold in COVERAGE_THRESHOLDS:
        result[f"coverage_{threshold:.2f}"] = []
        result[f"neighbor_persistence_{threshold:.2f}"] = []

    for row, query_id in enumerate(query_ids):
        query_valid = valid[query_id]
        query_delay = delay[query_id, query_valid]
        query_mass = mass[query_id, query_valid]
        if len(query_delay) == 0 or float(np.sum(query_mass)) <= 1e-20:
            continue
        query_mass = query_mass / np.sum(query_mass)
        neighbor_ids = raw.neighbor_ids[row]
        neighbor_delay = delay[neighbor_ids]
        neighbor_mass = mass[neighbor_ids]
        neighbor_valid = valid[neighbor_ids]
        flat_delay = neighbor_delay[neighbor_valid]
        flat_owner = np.broadcast_to(
            np.arange(len(neighbor_ids))[:, None], neighbor_valid.shape
        )[neighbor_valid]
        flat_mass = neighbor_mass[neighbor_valid]
        if len(flat_delay) == 0:
            continue

        component_nearest = []
        component_spatial = []
        component_mean = []
        component_affine = []
        component_affine_gain = []
        component_match_count = []
        ambiguity_025 = []
        ambiguity_050 = []
        coverage = {threshold: [] for threshold in COVERAGE_THRESHOLDS}
        persistence = {threshold: [] for threshold in COVERAGE_THRESHOLDS}
        for target_delay, target_mass in zip(query_delay, query_mass):
            distance = np.abs(signed_circular_delta(flat_delay, float(target_delay)))
            nearest = float(np.min(distance))
            component_nearest.append((target_mass, nearest))
            ambiguity_025.append((target_mass, float(np.sum(distance <= 0.25))))
            ambiguity_050.append((target_mass, float(np.sum(distance <= 0.50))))
            for threshold in COVERAGE_THRESHOLDS:
                covered = nearest <= threshold
                coverage[threshold].append((target_mass, float(covered)))
                owners = np.unique(flat_owner[distance <= threshold])
                persistence[threshold].append((target_mass, float(len(owners))))

            matched_delay = []
            matched_xy = []
            matched_weight = []
            matched_spatial_distance = []
            for neighbor_row, neighbor_id in enumerate(neighbor_ids):
                current_valid = neighbor_valid[neighbor_row]
                if not np.any(current_valid):
                    continue
                current_delay = neighbor_delay[neighbor_row, current_valid]
                current_mass = neighbor_mass[neighbor_row, current_valid]
                current_distance = np.abs(
                    signed_circular_delta(current_delay, float(target_delay))
                )
                selected = int(np.argmin(current_distance))
                if current_distance[selected] > AFFINE_MATCH_THRESHOLD:
                    continue
                matched_delay.append(float(current_delay[selected]))
                matched_xy.append(
                    all_pos[neighbor_id, :2] - all_pos[query_id, :2]
                )
                matched_weight.append(
                    float(raw.base_weights[row, neighbor_row] * current_mass[selected])
                )
                matched_spatial_distance.append(
                    float(np.linalg.norm(all_pos[neighbor_id, :2] - all_pos[query_id, :2]))
                )
            if not matched_delay:
                continue
            matched_delay_array = np.asarray(matched_delay, dtype=np.float64)
            matched_xy_array = np.asarray(matched_xy, dtype=np.float64)
            matched_weight_array = np.asarray(matched_weight, dtype=np.float64)
            matched_weight_array /= np.maximum(np.sum(matched_weight_array), 1e-20)
            nearest_spatial_index = int(np.argmin(matched_spatial_distance))
            nearest_spatial_prediction = matched_delay_array[nearest_spatial_index]
            mean_prediction = (
                float(target_delay)
                + float(
                    np.sum(
                        matched_weight_array
                        * signed_circular_delta(matched_delay_array, float(target_delay))
                    )
                )
            ) % v297.N_DELAY
            nearest_spatial_error = float(
                abs(signed_circular_delta(nearest_spatial_prediction, float(target_delay)))
            )
            mean_error = float(
                abs(signed_circular_delta(mean_prediction, float(target_delay)))
            )
            if len(matched_delay_array) >= 3:
                affine_prediction = fit_affine_at_query(
                    matched_xy_array,
                    matched_delay_array,
                    matched_weight_array,
                    float(target_delay),
                )
                affine_error = float(
                    abs(signed_circular_delta(affine_prediction, float(target_delay)))
                )
            else:
                affine_error = mean_error
            component_spatial.append((target_mass, nearest_spatial_error))
            component_mean.append((target_mass, mean_error))
            component_affine.append((target_mass, affine_error))
            component_affine_gain.append((target_mass, mean_error - affine_error))
            component_match_count.append((target_mass, float(len(matched_delay_array))))

        def weighted(values: list[tuple[float, float]]) -> float:
            if not values:
                return float("nan")
            weight_array = np.asarray([value[0] for value in values])
            value_array = np.asarray([value[1] for value in values])
            return float(np.sum(weight_array * value_array) / np.maximum(np.sum(weight_array), 1e-20))

        result["nearest_distance"].append(weighted(component_nearest))
        result["nearest_spatial_distance"].append(weighted(component_spatial))
        result["weighted_mean_distance"].append(weighted(component_mean))
        result["affine_distance"].append(weighted(component_affine))
        result["affine_gain"].append(weighted(component_affine_gain))
        result["matched_neighbor_count"].append(weighted(component_match_count))
        result["candidate_ambiguity_025"].append(weighted(ambiguity_025))
        result["candidate_ambiguity_050"].append(weighted(ambiguity_050))
        for threshold in COVERAGE_THRESHOLDS:
            result[f"coverage_{threshold:.2f}"].append(weighted(coverage[threshold]))
            result[f"neighbor_persistence_{threshold:.2f}"].append(
                weighted(persistence[threshold])
            )
    return {
        key: np.asarray(value, dtype=np.float32)
        for key, value in result.items()
    }


def main() -> None:
    if not v297.COMPONENT_CACHE.is_file():
        raise FileNotFoundError(v297.COMPONENT_CACHE)
    component = np.load(v297.COMPONENT_CACHE)
    delay = component["delay"].astype(np.float32)
    mass = component["weight"].astype(np.float32)
    valid = component["valid"].astype(bool)
    targets = np.load(v297.TARGET_CACHE)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    (
        all_pos,
        signatures,
        blocker_count,
        _scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = v274.load_geometry()

    fold_rows = []
    combined: dict[str, list[np.ndarray]] = {}
    for fold, seed in enumerate(OUTER_SEEDS):
        query_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        bank_ids = np.setdiff1d(valid_ids, query_ids)
        values = analyze_fold(
            query_ids,
            bank_ids,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
            delay,
            mass,
            valid,
        )
        summary = {
            key: summarize(value[np.isfinite(value)]) for key, value in values.items()
        }
        fold_rows.append({"fold": fold, "seed": seed, "count": int(len(query_ids)), "summary": summary})
        for key, value in values.items():
            combined.setdefault(key, []).append(value[np.isfinite(value)])
        print(
            f"v299 fold {fold + 1}/5 coverage025={summary['coverage_0.25']['mean']:.6f} "
            f"nearest={summary['nearest_distance']['mean']:.6f} "
            f"affine={summary['affine_distance']['mean']:.6f}",
            flush=True,
        )

    overall = {
        key: summarize(np.concatenate(value)) for key, value in combined.items()
    }
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "diagnostic_truth_policy": (
            "Held-out query component labels are used only to measure bank coverage, "
            "ambiguity, and an oracle-seeded affine upper bound; they are never model inputs."
        ),
        "parameters": {
            "neighbors": v274.K_NEIGHBORS,
            "coverage_thresholds_bins": list(COVERAGE_THRESHOLDS),
            "affine_match_threshold_bins": AFFINE_MATCH_THRESHOLD,
            "max_delay_gradient_bins_per_meter": MAX_DELAY_GRADIENT,
        },
        "fold_rows": fold_rows,
        "overall": overall,
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
