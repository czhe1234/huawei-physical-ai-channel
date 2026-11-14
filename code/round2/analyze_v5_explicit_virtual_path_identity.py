#!/usr/bin/env python3
"""Test map-assigned path identities before deploying carrier-phase transport."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze_v4_map_virtual_source_phase import (
    COMPONENT_CACHE,
    METERS_PER_DELAY_BIN,
    REPORT as MAP_PHASE_REPORT,
    Config as MapConfig,
    build_wall_bank,
    image_sources,
    rotate_source,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import MH, S_SC, TINY, circular_delta


VERSION = "Round2-E30-explicit-virtual-path-identity"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_explicit_virtual_path_identity"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_v5_explicit_virtual_path_identity"
)
ASSIGNMENT_CACHE = ARTIFACT_DIR / "foldwise_component_virtual_path_identity.npz"
REPORT = ANALYSIS_DIR / "explicit_virtual_path_identity.json"
FOLDS = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)

UNASSIGNED = -2
LOS_ID = -1


@dataclass(frozen=True)
class Config:
    paths: int = 16
    delay_gate: float = 0.55
    horizontal_gate: float = 1.35
    normalized_cost_gate: float = 5.0
    maximum_candidates: int = 96
    donor_count: int = 24
    minimum_observations: int = 4
    maximum_donor_distance_m: float = 55.0
    distance_scale_floor_m: float = 5.0
    wrong_beta_offset: float = 0.20
    seed: int = 20_260_805


def summarize(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    array = array[finite]
    if len(array) == 0:
        return {"count": 0}
    output: dict[str, float | int] = {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }
    if weights is not None:
        selected_weights = np.asarray(weights, dtype=np.float64)[finite]
        output["weighted_mean"] = float(
            np.sum(selected_weights * array)
            / max(float(np.sum(selected_weights)), TINY)
        )
    return output


def load_components() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cached = np.load(COMPONENT_CACHE)
    return (
        cached["coordinates"].astype(np.float32),
        cached["coefficients"].astype(np.complex64),
        cached["point_power"].astype(np.float64),
    )


def load_fold_parameters() -> dict[tuple[int, int], dict[str, float]]:
    payload = json.loads(MAP_PHASE_REPORT.read_text(encoding="utf-8"))
    return {
        (int(row["fold"]), int(row["site"])): {
            "beta": float(row["beta_rad_per_m"]),
            "orientation": math.radians(float(row["orientation_degrees"])),
        }
        for row in payload["diagnostics"]["beta_calibration"]
    }


def horizontal_bin(azimuth: np.ndarray, orientation: float) -> np.ndarray:
    return np.mod(8.0 * np.sin(azimuth - orientation), MH)


def wall_geometry_for_point(
    point: np.ndarray,
    bs_position: np.ndarray,
    sources: np.ndarray,
    wall_bank,
    map_config: MapConfig,
    orientation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = point[None] - sources
    length = np.linalg.norm(delta, axis=1)
    denominator = np.einsum("ci,ci->c", delta[:, :2], wall_bank.normal)
    numerator = wall_bank.offset - np.einsum(
        "ci,ci->c", sources[:, :2], wall_bank.normal
    )
    parameter = numerator / np.where(
        np.abs(denominator) > 1e-12, denominator, np.nan
    )
    reflection = sources + parameter[:, None] * delta
    tangent = np.column_stack([-wall_bank.normal[:, 1], wall_bank.normal[:, 0]])
    tangent_coordinate = np.einsum(
        "ci,ci->c", reflection[:, :2], tangent
    )
    valid = (
        np.isfinite(parameter)
        & (parameter > 0.0)
        & (parameter < 1.0)
        & (tangent_coordinate >= wall_bank.tangent_min - map_config.tangent_margin)
        & (tangent_coordinate <= wall_bank.tangent_max + map_config.tangent_margin)
        & (reflection[:, 2] >= wall_bank.z_min - map_config.height_margin)
        & (reflection[:, 2] <= wall_bank.z_max + map_config.height_margin)
    )
    departure = reflection[:, :2] - bs_position[None, :2]
    azimuth = np.arctan2(departure[:, 1], departure[:, 0])
    return length, horizontal_bin(azimuth, orientation), valid


def circular_abs(value: np.ndarray, target: np.ndarray, period: float) -> np.ndarray:
    return np.abs(circular_delta(value, target, period))


def assign_point_components(
    point: np.ndarray,
    bs_position: np.ndarray,
    component_coordinates: np.ndarray,
    sources: np.ndarray,
    wall_bank,
    map_config: MapConfig,
    orientation: float,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wall_length, wall_h, wall_valid = wall_geometry_for_point(
        point, bs_position, sources, wall_bank, map_config, orientation
    )
    component_delay = component_coordinates[:, 2]
    component_h = component_coordinates[:, 0]
    wall_delay = wall_length / METERS_PER_DELAY_BIN
    delay_error = circular_abs(
        component_delay[:, None], wall_delay[None], S_SC
    )
    horizontal_error = circular_abs(component_h[:, None], wall_h[None], MH)
    normalized = (
        (delay_error / config.delay_gate) ** 2
        + (horizontal_error / config.horizontal_gate) ** 2
    )
    candidate_score = np.min(normalized, axis=0)
    candidate_ids = np.flatnonzero(
        wall_valid
        & (np.min(delay_error, axis=0) <= config.delay_gate)
        & (np.min(horizontal_error, axis=0) <= config.horizontal_gate)
    )
    if len(candidate_ids) > config.maximum_candidates:
        candidate_ids = candidate_ids[
            np.argsort(candidate_score[candidate_ids])[: config.maximum_candidates]
        ]

    direct_length = float(np.linalg.norm(point - bs_position))
    direct_azimuth = math.atan2(
        float(point[1] - bs_position[1]), float(point[0] - bs_position[0])
    )
    path_ids = np.concatenate(
        [np.asarray([LOS_ID], dtype=np.int32), candidate_ids.astype(np.int32)]
    )
    path_length = np.concatenate([[direct_length], wall_length[candidate_ids]])
    path_h = np.concatenate(
        [
            np.asarray([horizontal_bin(np.asarray(direct_azimuth), orientation)]),
            wall_h[candidate_ids],
        ]
    )
    path_delay = path_length / METERS_PER_DELAY_BIN
    local_delay_error = circular_abs(
        component_delay[:, None], path_delay[None], S_SC
    )
    local_h_error = circular_abs(component_h[:, None], path_h[None], MH)
    cost = (
        (local_delay_error / config.delay_gate) ** 2
        + (local_h_error / config.horizontal_gate) ** 2
    )
    left, right = linear_sum_assignment(cost)

    assigned = np.full(config.paths, UNASSIGNED, dtype=np.int32)
    assigned_length = np.full(config.paths, np.nan, dtype=np.float32)
    assigned_delay_error = np.full(config.paths, np.nan, dtype=np.float32)
    assigned_h_error = np.full(config.paths, np.nan, dtype=np.float32)
    for component, candidate in zip(left, right):
        accepted = (
            local_delay_error[component, candidate] <= config.delay_gate
            and local_h_error[component, candidate] <= config.horizontal_gate
            and cost[component, candidate] <= config.normalized_cost_gate
        )
        if accepted:
            assigned[component] = path_ids[candidate]
            assigned_length[component] = np.float32(path_length[candidate])
            assigned_delay_error[component] = np.float32(
                local_delay_error[component, candidate]
            )
            assigned_h_error[component] = np.float32(
                local_h_error[component, candidate]
            )
    return assigned, assigned_length, assigned_delay_error, assigned_h_error


def build_assignments(
    config: Config,
    positions: np.ndarray,
    site: np.ndarray,
    coordinates: np.ndarray,
    point_power: np.ndarray,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if ASSIGNMENT_CACHE.is_file() and not rebuild:
        return {key: value for key, value in np.load(ASSIGNMENT_CACHE).items()}
    setup, _, _, _, _ = load_positions()
    map_config = MapConfig()
    wall_bank = build_wall_bank(map_config)
    fold_parameters = load_fold_parameters()
    path_id = np.full((5, len(positions), config.paths), UNASSIGNED, dtype=np.int32)
    path_length = np.full((5, len(positions), config.paths), np.nan, dtype=np.float32)
    delay_error = np.full_like(path_length, np.nan)
    horizontal_error = np.full_like(path_length, np.nan)

    for fold in range(5):
        for site_value in range(2):
            ids = np.flatnonzero((site == site_value) & (point_power > TINY))
            source = image_sources(wall_bank, setup.bs_positions[site_value])
            orientation = fold_parameters[(fold, site_value)]["orientation"]
            for row, point_id in enumerate(ids):
                assigned = assign_point_components(
                    positions[point_id],
                    setup.bs_positions[site_value],
                    coordinates[point_id, : config.paths],
                    source,
                    wall_bank,
                    map_config,
                    orientation,
                    config,
                )
                path_id[fold, point_id] = assigned[0]
                path_length[fold, point_id] = assigned[1]
                delay_error[fold, point_id] = assigned[2]
                horizontal_error[fold, point_id] = assigned[3]
                if (row + 1) % 500 == 0:
                    print(
                        f"virtual identity fold={fold} site={site_value} "
                        f"points={row + 1}/{len(ids)}",
                        flush=True,
                    )
    arrays = {
        "path_id": path_id,
        "path_length": path_length,
        "delay_error": delay_error,
        "horizontal_error": horizontal_error,
    }
    np.savez_compressed(ASSIGNMENT_CACHE, **arrays)
    return arrays


def unit_vector(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-20)


def weighted_unit_average(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    output = np.sum(weight[:, None] * value, axis=0)
    return unit_vector(output)


def phase_cosine(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction_unit = unit_vector(prediction)
    truth_unit = unit_vector(truth)
    return float(np.real(np.vdot(prediction_unit, truth_unit)))


def projective_cosine(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(abs(np.vdot(unit_vector(prediction), unit_vector(truth))))


def source_lengths(
    ids: np.ndarray,
    path_id: int,
    positions: np.ndarray,
    bs_position: np.ndarray,
    sources: np.ndarray,
    rotated: bool,
) -> np.ndarray:
    if path_id == LOS_ID:
        return np.linalg.norm(positions[ids] - bs_position[None], axis=1)
    source = sources[path_id]
    if rotated:
        source = rotate_source(source, bs_position)
    return np.linalg.norm(positions[ids] - source[None], axis=1)


def evaluate_phase_law(
    config: Config,
    positions: np.ndarray,
    site: np.ndarray,
    coefficients: np.ndarray,
    assignments: dict[str, np.ndarray],
) -> dict[str, object]:
    setup, _, _, _, _ = load_positions()
    wall_bank = build_wall_bank(MapConfig())
    sources = {
        site_value: image_sources(wall_bank, setup.bs_positions[site_value])
        for site_value in range(2)
    }
    fold_parameters = load_fold_parameters()
    folds = np.load(FOLDS)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    component_energy = np.sum(np.abs(coefficients[:, : config.paths]) ** 2, axis=2)

    rows: list[dict[str, float | int | str]] = []
    held_total_energy = 0.0
    assigned_energy = 0.0
    evaluable_energy = 0.0
    assigned_count = 0
    evaluable_count = 0

    for fold in range(5):
        fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
        held_ids = query_ids[fold_id == fold]
        for site_value in range(2):
            fit_site = fit_ids[site[fit_ids] == site_value]
            held_site = held_ids[site[held_ids] == site_value]
            identity_bank: dict[int, list[tuple[int, int]]] = {}
            for point_id in fit_site:
                for component, identity in enumerate(
                    assignments["path_id"][fold, point_id]
                ):
                    if identity != UNASSIGNED:
                        identity_bank.setdefault(int(identity), []).append(
                            (int(point_id), int(component))
                        )

            beta = fold_parameters[(fold, site_value)]["beta"]
            bs_position = setup.bs_positions[site_value]
            for query_id in held_site:
                held_total_energy += float(np.sum(component_energy[query_id]))
                for component, identity in enumerate(
                    assignments["path_id"][fold, query_id]
                ):
                    energy = float(component_energy[query_id, component])
                    if identity == UNASSIGNED or energy <= TINY:
                        continue
                    assigned_count += 1
                    assigned_energy += energy
                    observations = identity_bank.get(int(identity), [])
                    if len(observations) < config.minimum_observations:
                        continue
                    ids = np.asarray([value[0] for value in observations], dtype=np.int64)
                    parts = np.asarray([value[1] for value in observations], dtype=np.int64)
                    distance = np.linalg.norm(
                        positions[ids, :2] - positions[query_id, None, :2], axis=1
                    )
                    keep = distance <= config.maximum_donor_distance_m
                    ids = ids[keep]
                    parts = parts[keep]
                    distance = distance[keep]
                    if len(ids) < config.minimum_observations:
                        continue
                    order = np.argsort(distance)[: config.donor_count]
                    ids = ids[order]
                    parts = parts[order]
                    distance = distance[order]
                    scale = max(float(np.median(distance)), config.distance_scale_floor_m)
                    weight = np.exp(-0.5 * (distance / (1.5 * scale)) ** 2)
                    weight /= max(float(np.sum(weight)), TINY)

                    observed = coefficients[ids, parts]
                    observed_unit = observed / np.maximum(
                        np.linalg.norm(observed, axis=1, keepdims=True), 1e-20
                    )
                    truth = coefficients[query_id, component]
                    donor_length = assignments["path_length"][fold, ids, parts]
                    query_length = float(
                        assignments["path_length"][fold, query_id, component]
                    )
                    latent = weighted_unit_average(
                        observed_unit * np.exp(1j * beta * donor_length)[:, None],
                        weight,
                    )
                    correct = latent * np.exp(-1j * beta * query_length)
                    no_transport = weighted_unit_average(observed_unit, weight)
                    nearest = observed_unit[0]
                    wrong_beta = weighted_unit_average(
                        observed_unit
                        * np.exp(
                            1j * (beta + config.wrong_beta_offset) * donor_length
                        )[:, None],
                        weight,
                    ) * np.exp(
                        -1j * (beta + config.wrong_beta_offset) * query_length
                    )

                    row: dict[str, float | int | str] = {
                        "fold": fold,
                        "site": site_value,
                        "kind": "los" if identity == LOS_ID else "wall",
                        "identity": int(identity),
                        "energy": energy,
                        "observations": int(len(ids)),
                        "nearest_distance_m": float(distance[0]),
                        "correct": phase_cosine(correct, truth),
                        "no_transport": phase_cosine(no_transport, truth),
                        "nearest": phase_cosine(nearest, truth),
                        "wrong_beta": phase_cosine(wrong_beta, truth),
                        "correct_projective": projective_cosine(correct, truth),
                    }
                    if identity != LOS_ID:
                        rotated_donor_length = source_lengths(
                            ids,
                            int(identity),
                            positions,
                            bs_position,
                            sources[site_value],
                            True,
                        )
                        rotated_query_length = float(
                            source_lengths(
                                np.asarray([query_id]),
                                int(identity),
                                positions,
                                bs_position,
                                sources[site_value],
                                True,
                            )[0]
                        )
                        rotated_latent = weighted_unit_average(
                            observed_unit
                            * np.exp(1j * beta * rotated_donor_length)[:, None],
                            weight,
                        )
                        rotated = rotated_latent * np.exp(
                            -1j * beta * rotated_query_length
                        )
                        row["rotated_control"] = phase_cosine(rotated, truth)
                    rows.append(row)
                    evaluable_count += 1
                    evaluable_energy += energy

    def subset_payload(selected: list[dict[str, float | int | str]]) -> dict[str, object]:
        if not selected:
            return {"count": 0}
        weight = np.asarray([float(row["energy"]) for row in selected])
        names = ("correct", "no_transport", "nearest", "wrong_beta")
        output: dict[str, object] = {
            name: summarize(
                np.asarray([float(row[name]) for row in selected]), weight
            )
            for name in names
        }
        if all("rotated_control" in row for row in selected):
            output["rotated_control"] = summarize(
                np.asarray([float(row["rotated_control"]) for row in selected]),
                weight,
            )
        correct = np.asarray([float(row["correct"]) for row in selected])
        nearest = np.asarray([float(row["nearest"]) for row in selected])
        wrong_beta = np.asarray([float(row["wrong_beta"]) for row in selected])
        output["weighted_margin_vs_nearest"] = float(
            np.sum(weight * (correct - nearest)) / max(float(np.sum(weight)), TINY)
        )
        output["weighted_margin_vs_wrong_beta"] = float(
            np.sum(weight * (correct - wrong_beta))
            / max(float(np.sum(weight)), TINY)
        )
        output["correct_projective"] = summarize(
            np.asarray([float(row["correct_projective"]) for row in selected]), weight
        )
        return output

    fold_payload = []
    for fold in range(5):
        selected = [row for row in rows if int(row["fold"]) == fold]
        fold_payload.append({"fold": fold, **subset_payload(selected)})
    wall_rows = [row for row in rows if row["kind"] == "wall"]
    los_rows = [row for row in rows if row["kind"] == "los"]
    wall_payload = subset_payload(wall_rows)
    if wall_rows:
        weight = np.asarray([float(row["energy"]) for row in wall_rows])
        correct = np.asarray([float(row["correct"]) for row in wall_rows])
        rotated = np.asarray([float(row["rotated_control"]) for row in wall_rows])
        wall_payload["weighted_margin_vs_rotated_control"] = float(
            np.sum(weight * (correct - rotated)) / max(float(np.sum(weight)), TINY)
        )

    fold_by_kind = {
        kind: [
            {
                "fold": fold,
                **subset_payload(
                    [
                        row
                        for row in rows
                        if int(row["fold"]) == fold and row["kind"] == kind
                    ]
                ),
            }
            for fold in range(5)
        ]
        for kind in ("los", "wall")
    }

    fold_margins = np.asarray(
        [float(row.get("weighted_margin_vs_nearest", np.nan)) for row in fold_payload]
    )
    overall = subset_payload(rows)
    passed = bool(
        evaluable_energy / max(held_total_energy, TINY) >= 0.10
        and float(overall["weighted_margin_vs_nearest"]) >= 0.05
        and int(np.sum(fold_margins > 0.0)) >= 4
        and float(wall_payload.get("weighted_margin_vs_rotated_control", -np.inf))
        >= 0.05
    )
    return {
        "held_top_path_energy": held_total_energy,
        "assigned_count": assigned_count,
        "assigned_energy_fraction": assigned_energy / max(held_total_energy, TINY),
        "evaluable_count": evaluable_count,
        "evaluable_energy_fraction": evaluable_energy / max(held_total_energy, TINY),
        "overall": overall,
        "los": subset_payload(los_rows),
        "wall": wall_payload,
        "fold": fold_payload,
        "fold_by_kind": fold_by_kind,
        "decision_rule": {
            "minimum_evaluable_energy_fraction": 0.10,
            "minimum_correct_margin_vs_nearest": 0.05,
            "minimum_positive_folds": 4,
            "minimum_wall_margin_vs_rotated": 0.05,
            "passed": passed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _, positions, _, site, _ = load_positions()
    coordinates, coefficients, point_power = load_components()
    assignments = build_assignments(
        config, positions, site, coordinates, point_power, args.rebuild
    )
    evaluation = evaluate_phase_law(
        config, positions, site, coefficients, assignments
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "A globally consistent LOS/wall identity inferred from official delay-angle "
            "components and the official map makes geometric carrier-phase transport "
            "stable across complete spatial holes."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "held_truth_usage": (
            "Held component support and identity are used only for this explicitly marked "
            "phase-law oracle; all transported coefficients come from outer-fit points."
        ),
        "config": asdict(config),
        "evaluation": evaluation,
        "outputs": {"assignments": PATHS.relative(ASSIGNMENT_CACHE)},
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
