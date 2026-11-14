#!/usr/bin/env python3
"""Audit cross-UE reuse of effective virtual centers inferred from official H."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import analyze_e143_effective_surface_identity as e143
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import MH, S_SC, circular_delta


VERSION = "Round2-E144-virtual-center-reuse"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "virtual_center_reuse"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_virtual_center_reuse"
REPORT = ANALYSIS_DIR / "virtual_center_reuse.json"
OUTPUT = ARTIFACT_DIR / "virtual_center_reuse_metrics.npz"
E143_PATHS = (
    PATHS.artifacts
    / "round2"
    / "effective_surface_identity"
    / "effective_surface_paths.npz"
)
FULL_FOLD = e143.FULL_FOLD
METERS_PER_DELAY_BIN = e143.METERS_PER_DELAY_BIN
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    cluster_radius_m: float = 3.0
    minimum_fit_points: int = 5
    control_neighbors: int = 48
    control_repeats: int = 8
    maximum_center_radius_m: float = 6.0
    horizontal_scale: float = 1.5
    delay_scale: float = 2.0
    coordinate_cost_limit: float = 0.25
    required_coverage_margin: float = 0.20
    required_positive_folds: int = 5
    required_coordinate_good_mass: float = 0.50
    required_coordinate_margin: float = 0.20
    seed: int = 20_260_809_144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-repeats", type=int, default=Config.control_repeats)
    parser.add_argument("--seed", type=int, default=Config.seed)
    return parser.parse_args()


def horizontal_solutions(value: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    signed = np.where(value <= 8.0, value, value - 16.0)
    principal = np.arcsin(np.clip(signed / 8.0, -1.0, 1.0))
    return np.stack((principal + orientation, math.pi - principal + orientation), axis=1)


def effective_center_candidates(
    coordinate: np.ndarray,
    point_id: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    horizontal_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bs_xy = bs_positions[site[point_id], :2]
    ue_xy = positions[point_id, :2]
    vector = ue_xy - bs_xy
    direct = np.linalg.norm(vector, axis=1)
    total_length = coordinate[:, 2].astype(np.float64) * METERS_PER_DELAY_BIN
    azimuth = horizontal_solutions(coordinate[:, 0], horizontal_orientation[point_id])
    direction = np.stack((np.cos(azimuth), np.sin(azimuth)), axis=2)
    denominator = 2.0 * (
        total_length[:, None] - np.sum(direction * vector[:, None], axis=2)
    )
    numerator = np.square(total_length) - np.square(direct)
    distance = np.divide(
        numerator[:, None],
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1e-9,
    )
    valid = (
        np.isfinite(distance)
        & (total_length[:, None] > direct[:, None])
        & (distance > 0.0)
        & (distance < total_length[:, None])
    )
    center = bs_xy[:, None] + distance[..., None] * direction
    center[~valid] = np.nan
    return center.astype(np.float32), valid


def cluster_fit_centers(
    candidate: np.ndarray,
    valid: np.ndarray,
    path_rows: np.ndarray,
    point_id: np.ndarray,
    mass: np.ndarray,
    site: np.ndarray,
    config: Config,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    centers: dict[int, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    for site_value in np.unique(site[point_id[path_rows]]):
        site_value = int(site_value)
        local_path = path_rows[site[point_id[path_rows]] == site_value]
        local_candidate = candidate[local_path]
        local_valid = valid[local_path]
        flat_xy = local_candidate.reshape(-1, 2)
        flat_valid = local_valid.reshape(-1)
        flat_path = np.repeat(local_path, 2)
        finite = flat_valid & np.isfinite(flat_xy).all(axis=1)
        xy = flat_xy[finite]
        source_path = flat_path[finite]
        if len(xy) == 0:
            centers[site_value] = np.empty((0, 2), dtype=np.float32)
            diagnostics[str(site_value)] = {"candidate_count": 0, "center_count": 0}
            continue
        labels = DBSCAN(
            eps=config.cluster_radius_m,
            min_samples=config.minimum_fit_points,
            n_jobs=-1,
        ).fit_predict(xy)
        rows: list[np.ndarray] = []
        supports: list[int] = []
        radii: list[float] = []
        for label in np.unique(labels[labels >= 0]):
            member = np.flatnonzero(labels == label)
            member_path = source_path[member]
            unique_points = np.unique(point_id[member_path])
            if len(unique_points) < config.minimum_fit_points:
                continue
            weight = mass[member_path].astype(np.float64)
            center = np.sum(xy[member] * weight[:, None], axis=0) / max(
                float(np.sum(weight)), TINY
            )
            radius = float(np.quantile(np.linalg.norm(xy[member] - center[None], axis=1), 0.90))
            if radius > config.maximum_center_radius_m:
                continue
            rows.append(center.astype(np.float32))
            supports.append(int(len(unique_points)))
            radii.append(radius)
        centers[site_value] = (
            np.stack(rows).astype(np.float32)
            if rows
            else np.empty((0, 2), dtype=np.float32)
        )
        diagnostics[str(site_value)] = {
            "candidate_count": int(len(xy)),
            "raw_cluster_count": int(len(np.unique(labels[labels >= 0]))),
            "center_count": int(len(rows)),
            "support_points_mean": float(np.mean(supports)) if supports else 0.0,
            "radius_p90_mean_m": float(np.mean(radii)) if radii else 0.0,
        }
    return centers, diagnostics


def predict_coordinate(
    center: np.ndarray,
    point_id: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    horizontal_orientation: np.ndarray,
) -> np.ndarray:
    bs = bs_positions[site[point_id]]
    ue = positions[point_id]
    bs_to_center = center - bs[:, :2]
    center_to_ue = ue[:, :2] - center
    length = np.linalg.norm(bs_to_center, axis=1) + np.linalg.norm(center_to_ue, axis=1)
    azimuth = np.arctan2(bs_to_center[:, 1], bs_to_center[:, 0])
    return np.column_stack(
        (
            np.mod(8.0 * np.sin(azimuth - horizontal_orientation[point_id]), MH),
            np.mod(length / METERS_PER_DELAY_BIN, S_SC),
        )
    ).astype(np.float32)


def evaluate_fold(
    fold: int,
    coordinate: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    point_id: np.ndarray,
    mass: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    horizontal_orientation: np.ndarray,
    fold_id: np.ndarray,
    config: Config,
) -> tuple[dict[str, float | int | object], np.ndarray, np.ndarray]:
    fit_paths = np.flatnonzero(fold_id[point_id] != fold)
    held_paths = np.flatnonzero(fold_id[point_id] == fold)
    centers, diagnostics = cluster_fit_centers(
        candidate,
        valid,
        fit_paths,
        point_id,
        mass,
        site,
        config,
    )
    nearest_distance = np.full(len(held_paths), np.inf, dtype=np.float64)
    selected_center = np.full((len(held_paths), 2), np.nan, dtype=np.float32)
    for site_value in np.unique(site[point_id[held_paths]]):
        site_value = int(site_value)
        local = np.flatnonzero(site[point_id[held_paths]] == site_value)
        source = centers.get(site_value, np.empty((0, 2), dtype=np.float32))
        if len(source) == 0:
            continue
        tree = cKDTree(source)
        local_candidate = candidate[held_paths[local]]
        local_valid = valid[held_paths[local]]
        flat = local_candidate.reshape(-1, 2)
        flat_valid = local_valid.reshape(-1) & np.isfinite(flat).all(axis=1)
        distance = np.full(len(flat), np.inf, dtype=np.float64)
        index = np.zeros(len(flat), dtype=np.int64)
        if np.any(flat_valid):
            distance[flat_valid], index[flat_valid] = tree.query(
                flat[flat_valid], k=1, workers=-1
            )
        distance = distance.reshape(len(local), 2)
        index = index.reshape(len(local), 2)
        solution = np.argmin(distance, axis=1)
        rows = np.arange(len(local))
        nearest_distance[local] = distance[rows, solution]
        selected_center[local] = source[index[rows, solution]]

    held_mass = mass[held_paths]
    covered = nearest_distance <= config.cluster_radius_m
    predicted = np.full((len(held_paths), 2), np.nan, dtype=np.float32)
    if np.any(covered):
        predicted[covered] = predict_coordinate(
            selected_center[covered],
            point_id[held_paths[covered]],
            positions,
            site,
            bs_positions,
            horizontal_orientation,
        )
    h_error = np.abs(
        circular_delta(predicted[:, 0], coordinate[held_paths, 0], MH)
    )
    d_error = np.abs(
        circular_delta(predicted[:, 1], coordinate[held_paths, 2], S_SC)
    )
    cost = np.square(h_error / config.horizontal_scale) + np.square(
        d_error / config.delay_scale
    )
    coordinate_good = covered & np.isfinite(cost) & (cost <= config.coordinate_cost_limit)
    denominator = float(np.sum(held_mass))
    result: dict[str, float | int | object] = {
        "fold": fold,
        "fit_path_count": int(len(fit_paths)),
        "held_path_count": int(len(held_paths)),
        "center_count": int(sum(len(value) for value in centers.values())),
        "covered_mass_fraction": float(np.sum(held_mass[covered]) / max(denominator, TINY)),
        "coordinate_good_mass_fraction": float(
            np.sum(held_mass[coordinate_good]) / max(denominator, TINY)
        ),
        "covered_path_fraction": float(np.mean(covered)),
        "center_diagnostics": diagnostics,
    }
    return result, nearest_distance.astype(np.float32), cost.astype(np.float32)


def evaluate_all_folds(
    coordinate: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    point_id: np.ndarray,
    mass: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    horizontal_orientation: np.ndarray,
    fold_id: np.ndarray,
    config: Config,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    fold_results: list[dict[str, float | int | object]] = []
    nearest = np.full(len(point_id), np.nan, dtype=np.float32)
    cost = np.full(len(point_id), np.nan, dtype=np.float32)
    for fold in range(5):
        result, fold_distance, fold_cost = evaluate_fold(
            fold,
            coordinate,
            candidate,
            valid,
            point_id,
            mass,
            positions,
            site,
            bs_positions,
            horizontal_orientation,
            fold_id,
            config,
        )
        held = np.flatnonzero(fold_id[point_id] == fold)
        nearest[held] = fold_distance
        cost[held] = fold_cost
        fold_results.append(result)
    denominator = float(np.sum(mass))
    covered = nearest <= config.cluster_radius_m
    coordinate_good = covered & np.isfinite(cost) & (cost <= config.coordinate_cost_limit)
    return (
        {
            "covered_mass_fraction": float(np.sum(mass[covered]) / max(denominator, TINY)),
            "coordinate_good_mass_fraction": float(
                np.sum(mass[coordinate_good]) / max(denominator, TINY)
            ),
            "covered_path_fraction": float(np.mean(covered)),
            "center_count_mean": float(
                np.mean([float(value["center_count"]) for value in fold_results])
            ),
            "folds": fold_results,
        },
        nearest,
        cost,
    )


def control_summary(values: list[dict[str, object]]) -> dict[str, object]:
    return {
        key: {
            "mean": float(np.mean([float(value[key]) for value in values])),
            "std": float(np.std([float(value[key]) for value in values])),
            "min": float(np.min([float(value[key]) for value in values])),
            "max": float(np.max([float(value[key]) for value in values])),
        }
        for key in (
            "covered_mass_fraction",
            "coordinate_good_mass_fraction",
            "covered_path_fraction",
            "center_count_mean",
        )
    }


def main() -> None:
    args = parse_args()
    config = Config(control_repeats=args.control_repeats, seed=args.seed)
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    setup, positions, _, site, _ = load_positions()
    paths = np.load(E143_PATHS, allow_pickle=False)
    point_id = paths["point_id"].astype(np.int32)
    coordinate = paths["coordinate"].astype(np.float64)
    mass = paths["mass"].astype(np.float64)
    horizontal = paths["horizontal_orientation"].astype(np.float64)
    fold_id = np.load(FULL_FOLD, allow_pickle=False).astype(np.int8)
    direct_length = np.linalg.norm(
        positions[point_id] - setup.bs_positions[site[point_id]], axis=1
    )

    actual_candidate, actual_valid = effective_center_candidates(
        coordinate,
        point_id,
        positions,
        site,
        setup.bs_positions,
        horizontal,
    )
    actual, actual_nearest, actual_cost = evaluate_all_folds(
        coordinate,
        actual_candidate,
        actual_valid,
        point_id,
        mass,
        positions,
        site,
        setup.bs_positions,
        horizontal,
        fold_id,
        config,
    )

    controls, matched = e143.matched_control_indices(
        point_id,
        coordinate,
        mass,
        direct_length,
        site,
        config.control_repeats,
        config.control_neighbors,
        config.seed,
    )
    control_results: list[dict[str, object]] = []
    first_control_nearest = np.empty(0, dtype=np.float32)
    first_control_cost = np.empty(0, dtype=np.float32)
    for repeat, source in enumerate(controls):
        control_coordinate = coordinate[source]
        control_candidate, control_valid = effective_center_candidates(
            control_coordinate,
            point_id,
            positions,
            site,
            setup.bs_positions,
            horizontal,
        )
        result, nearest, cost = evaluate_all_folds(
            control_coordinate,
            control_candidate,
            control_valid,
            point_id,
            mass,
            positions,
            site,
            setup.bs_positions,
            horizontal,
            fold_id,
            config,
        )
        control_results.append(result)
        if repeat == 0:
            first_control_nearest = nearest
            first_control_cost = cost
        print(f"E144 virtual-center control {repeat + 1}/{config.control_repeats}", flush=True)

    controls_aggregate = control_summary(control_results)
    coverage_margin = float(actual["covered_mass_fraction"]) - float(
        controls_aggregate["covered_mass_fraction"]["mean"]
    )
    coordinate_margin = float(actual["coordinate_good_mass_fraction"]) - float(
        controls_aggregate["coordinate_good_mass_fraction"]["mean"]
    )
    positive_coverage_folds = int(
        sum(
            float(actual["folds"][fold]["covered_mass_fraction"])
            > float(
                np.mean(
                    [
                        result["folds"][fold]["covered_mass_fraction"]
                        for result in control_results
                    ]
                )
            )
            for fold in range(5)
        )
    )
    gate = {
        "required_coverage_margin": config.required_coverage_margin,
        "required_positive_folds": config.required_positive_folds,
        "required_coordinate_good_mass": config.required_coordinate_good_mass,
        "required_coordinate_margin": config.required_coordinate_margin,
        "coverage_margin": coverage_margin,
        "positive_coverage_folds": positive_coverage_folds,
        "coordinate_good_mass": float(actual["coordinate_good_mass_fraction"]),
        "coordinate_margin": coordinate_margin,
    }
    gate["passed"] = bool(
        coverage_margin >= config.required_coverage_margin
        and positive_coverage_folds >= config.required_positive_folds
        and float(actual["coordinate_good_mass_fraction"])
        >= config.required_coordinate_good_mass
        and coordinate_margin >= config.required_coordinate_margin
    )

    np.savez_compressed(
        OUTPUT,
        point_id=point_id,
        coordinate=coordinate.astype(np.float32),
        mass=mass.astype(np.float32),
        actual_candidate=actual_candidate,
        actual_valid=actual_valid,
        actual_nearest_center_distance=actual_nearest,
        actual_coordinate_cost=actual_cost,
        first_control_source=controls[0],
        first_control_nearest_center_distance=first_control_nearest,
        first_control_coordinate_cost=first_control_cost,
    )
    report = {
        "experiment": VERSION,
        "status": "information_gate_passed" if gate["passed"] else "information_gate_failed",
        "official_inputs_only": True,
        "sionna_or_v12_used": False,
        "primary_hypothesis": (
            "Official-H path delay and horizontal departure angle form reusable "
            "effective virtual centers even when they do not align to physical walls."
        ),
        "config": asdict(config),
        "input": {
            "path_count": int(len(point_id)),
            "point_count": int(len(np.unique(point_id))),
            "valid_candidate_fraction": float(np.mean(actual_valid)),
        },
        "matched_control": matched,
        "actual": actual,
        "controls": controls_aggregate,
        "gate": gate,
        "decision": {
            "train_virtual_center_ai": bool(gate["passed"]),
            "next_if_passed": (
                "Train a fit-only center existence/energy model and require +0.01 "
                "through a frozen complete-channel renderer before any candidate."
            ),
            "next_if_failed": (
                "Terminate the current path-identity branch and select a non-path "
                "information source; do not tune cluster radius or support count."
            ),
        },
        "outputs": {
            "metrics": PATHS.relative(OUTPUT),
            "metrics_bytes": int(OUTPUT.stat().st_size),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
