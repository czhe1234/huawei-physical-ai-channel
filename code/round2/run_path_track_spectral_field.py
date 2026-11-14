#!/usr/bin/env python3
"""Predict Round 2 spectra from fit-only continuous component tracks."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from analyze_v4_path_track_spectral_targets import (
    MH,
    MP,
    MV,
    N_UE,
    S_SC,
    TRAIN_AD,
    TRAIN_PAS,
    TRAIN_PDP,
    cosine,
    periodic_delta,
    render_noncoherent,
    summarize,
)
from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_path_track_spectral_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_path_track_spectral_field"
REPORT = ANALYSIS_DIR / "path_track_spectral_field.json"
PREDICTIONS = ARTIFACT_DIR / "oof_path_track_predictions.npz"

COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FOLD_CACHE = (
    PATHS.artifacts
    / "round2"
    / "foundation"
    / "round2_spatial_hole_folds.npz"
)

TINY = 1e-30


@dataclass(frozen=True)
class Config:
    component_count: int = 20
    output_paths: int = 24
    graph_neighbors: int = 10
    graph_max_distance_m: float = 9.0
    donor_count: int = 96
    local_scale_m: float = 12.0
    minimum_fraction: float = 1e-5
    affine_ridge: float = 2.0
    presence_powers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)
        self.points: dict[int, set[int]] = {}

    def find(self, value: int) -> int:
        parent = self.parent
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    def union(self, left: int, right: int, left_point: int, right_point: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        left_points = self.points.get(left_root, {left_point})
        right_points = self.points.get(right_root, {right_point})
        if not left_points.isdisjoint(right_points):
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
            left_points, right_points = right_points, left_points
        self.parent[right_root] = left_root
        if len(left_points) < len(right_points):
            left_points, right_points = right_points, left_points
        left_points.update(right_points)
        self.points[left_root] = left_points
        self.points.pop(right_root, None)
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def component_energy(coefficients: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(coefficients) ** 2, axis=-1, dtype=np.float64)


def valid_components(
    energy: np.ndarray,
    point_power: np.ndarray,
    component_count: int,
    minimum_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    fraction = np.divide(
        energy,
        np.maximum(point_power[:, None], TINY),
        out=np.zeros_like(energy),
        where=point_power[:, None] > TINY,
    )
    slots = np.arange(energy.shape[1])[None, :]
    valid = (
        (point_power[:, None] > TINY)
        & (slots < component_count)
        & (fraction >= minimum_fraction)
    )
    return valid, fraction


def support_match(
    coordinates_left: np.ndarray,
    coefficients_left: np.ndarray,
    fractions_left: np.ndarray,
    coordinates_right: np.ndarray,
    coefficients_right: np.ndarray,
    fractions_right: np.ndarray,
    spatial_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_left = np.abs(coefficients_left) ** 2
    energy_right = np.abs(coefficients_right) ** 2
    profile_left = np.sqrt(energy_left)
    profile_right = np.sqrt(energy_right)
    profile_left /= np.maximum(np.linalg.norm(profile_left, axis=1, keepdims=True), 1e-20)
    profile_right /= np.maximum(np.linalg.norm(profile_right, axis=1, keepdims=True), 1e-20)
    profile_similarity = np.clip(profile_left @ profile_right.T, 0.0, 1.0)

    horizontal = np.abs(
        periodic_delta(
            coordinates_left[:, None, 0], coordinates_right[None, :, 0], MH
        )
    )
    vertical = np.abs(
        periodic_delta(
            coordinates_left[:, None, 1], coordinates_right[None, :, 1], MV
        )
    )
    delay = np.abs(
        periodic_delta(
            coordinates_left[:, None, 2], coordinates_right[None, :, 2], S_SC
        )
    )
    horizontal_limit = 0.85 + 0.23 * spatial_distance
    vertical_limit = 0.50 + 0.14 * spatial_distance
    delay_limit = 0.90 + 0.38 * spatial_distance
    log_ratio = np.abs(
        np.log(np.maximum(fractions_left[:, None], 1e-10))
        - np.log(np.maximum(fractions_right[None, :], 1e-10))
    )
    cost = (
        (horizontal / max(horizontal_limit, 0.5)) ** 2
        + (vertical / max(vertical_limit, 0.35)) ** 2
        + (delay / max(delay_limit, 0.5)) ** 2
        + 1.25 * (1.0 - profile_similarity)
        + 0.08 * np.minimum(log_ratio, 8.0)
    )
    allowed = (
        (horizontal <= horizontal_limit)
        & (vertical <= vertical_limit)
        & (delay <= delay_limit)
        & (profile_similarity >= 0.30)
    )
    cost[~allowed] = 1e6
    rows, columns = linear_sum_assignment(cost)
    selected_cost = cost[rows, columns]
    if cost.shape[1] >= 2:
        row_sorted = np.sort(cost, axis=1)
        row_margin = row_sorted[:, 1] - row_sorted[:, 0]
    else:
        row_margin = np.full(cost.shape[0], np.inf, dtype=np.float64)
    if cost.shape[0] >= 2:
        column_sorted = np.sort(cost, axis=0)
        column_margin = column_sorted[1, :] - column_sorted[0, :]
    else:
        column_margin = np.full(cost.shape[1], np.inf, dtype=np.float64)
    unambiguous = (row_margin[rows] >= 0.06) & (column_margin[columns] >= 0.06)
    unambiguous |= selected_cost < 0.22
    accepted = (selected_cost < 2.2) & unambiguous
    return rows[accepted], columns[accepted], cost[rows[accepted], columns[accepted]]


def build_fold_tracks(
    fit_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    valid: np.ndarray,
    fractions: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, dict[str, float | int]]:
    point_count = len(positions)
    slots = config.component_count
    dsu = DisjointSet(point_count * slots)
    edge_count = 0
    pair_count = 0
    accepted_cost: list[float] = []

    for site_value in np.unique(site[fit_ids]):
        bank = fit_ids[site[fit_ids] == site_value]
        take = min(config.graph_neighbors + 1, len(bank))
        distance, neighbor = cKDTree(positions[bank, :2]).query(
            positions[bank, :2], k=take, workers=-1
        )
        if take == 1:
            distance = distance[:, None]
            neighbor = neighbor[:, None]
        for local_row, point_id in enumerate(bank):
            for rank in range(1, take):
                other_id = int(bank[int(neighbor[local_row, rank])])
                separation = float(distance[local_row, rank])
                if other_id <= int(point_id) or separation > config.graph_max_distance_m:
                    continue
                left_slots = np.flatnonzero(valid[point_id, :slots])
                right_slots = np.flatnonzero(valid[other_id, :slots])
                if len(left_slots) == 0 or len(right_slots) == 0:
                    continue
                pair_count += 1
                left_rows, right_rows, costs = support_match(
                    coordinates[point_id, left_slots],
                    coefficients[point_id, left_slots],
                    fractions[point_id, left_slots],
                    coordinates[other_id, right_slots],
                    coefficients[other_id, right_slots],
                    fractions[other_id, right_slots],
                    separation,
                )
                for left_row, right_row, cost in zip(left_rows, right_rows, costs):
                    left_slot = int(left_slots[int(left_row)])
                    right_slot = int(right_slots[int(right_row)])
                    merged = dsu.union(
                        int(point_id) * slots + left_slot,
                        other_id * slots + right_slot,
                        int(point_id),
                        other_id,
                    )
                    if merged:
                        edge_count += 1
                        accepted_cost.append(float(cost))

    root = np.full((point_count, slots), -1, dtype=np.int32)
    for point_id in fit_ids:
        for slot in np.flatnonzero(valid[point_id, :slots]):
            root[point_id, slot] = dsu.find(int(point_id) * slots + int(slot))
    active_roots = root[root >= 0]
    unique_roots, counts = np.unique(active_roots, return_counts=True)
    diagnostics: dict[str, float | int] = {
        "fit_count": int(len(fit_ids)),
        "pair_count": int(pair_count),
        "edge_count": int(edge_count),
        "track_count": int(len(unique_roots)),
        "multi_point_track_count": int(np.sum(counts >= 2)),
        "track_size_median": float(np.median(counts)) if len(counts) else 0.0,
        "track_size_p90": float(np.quantile(counts, 0.90)) if len(counts) else 0.0,
        "accepted_cost_mean": float(np.mean(accepted_cost)) if accepted_cost else 0.0,
    }
    return root, diagnostics


def weighted_intercept(
    values: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    clamp_margin: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-10)
    if len(values) < 3:
        return float(np.sum(weights * values) / np.sum(weights))
    scale = max(float(np.median(np.linalg.norm(offsets, axis=1))), 1.0)
    design = np.column_stack(
        [np.ones(len(offsets)), offsets[:, 0] / scale, offsets[:, 1] / scale]
    )
    gram = design.T @ (weights[:, None] * design)
    gram += np.diag([1e-8, ridge, ridge])
    right = design.T @ (weights * values)
    try:
        coefficients = np.linalg.solve(gram, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, right, rcond=1e-8)[0]
    return float(
        np.clip(coefficients[0], np.min(values) - clamp_margin, np.max(values) + clamp_margin)
    )


def predict_track_candidates(
    query_position: np.ndarray,
    donor_ids: np.ndarray,
    donor_distances: np.ndarray,
    positions: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    valid: np.ndarray,
    fractions: np.ndarray,
    roots: np.ndarray,
    config: Config,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float, float]], dict[str, float]]:
    scale = config.local_scale_m
    point_weights = np.exp(-0.5 * (donor_distances / scale) ** 2) / np.sqrt(
        np.maximum(donor_distances, 1.0)
    )
    point_weights /= max(float(np.sum(point_weights)), TINY)
    observations: dict[int, list[tuple[int, int, float]]] = {}
    for donor_rank, donor_id in enumerate(donor_ids):
        for slot in np.flatnonzero(valid[donor_id, : config.component_count]):
            root = int(roots[donor_id, slot])
            if root < 0:
                continue
            observations.setdefault(root, []).append(
                (int(donor_id), int(slot), float(point_weights[donor_rank]))
            )

    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    observed_counts: list[int] = []
    presences: list[float] = []
    for rows in observations.values():
        strongest_by_point: dict[int, tuple[int, float]] = {}
        for donor_id, slot, spatial_weight in rows:
            current = strongest_by_point.get(donor_id)
            if current is None or fractions[donor_id, slot] > fractions[donor_id, current[0]]:
                strongest_by_point[donor_id] = (slot, spatial_weight)
        ids = np.asarray(list(strongest_by_point), dtype=np.int64)
        slots = np.asarray([strongest_by_point[int(point_id)][0] for point_id in ids])
        weights = np.asarray(
            [strongest_by_point[int(point_id)][1] for point_id in ids], dtype=np.float64
        )
        if len(ids) < 2:
            if len(ids) == 0 or np.min(np.linalg.norm(positions[ids, :2] - query_position[:2], axis=1)) > 7.5:
                continue
        presence = float(np.clip(np.sum(weights), 0.0, 1.0))
        if presence < 0.015:
            continue
        offsets = positions[ids, :2] - query_position[None, :2]
        local_coordinates = coordinates[ids, slots]
        reference = local_coordinates[int(np.argmin(np.linalg.norm(offsets, axis=1)))]
        predicted_coordinate = np.zeros(3, dtype=np.float32)
        for axis, period, margin in ((0, MH, 0.75), (1, MV, 0.50), (2, S_SC, 1.0)):
            unwrapped = reference[axis] + periodic_delta(
                local_coordinates[:, axis], reference[axis], period
            )
            predicted_coordinate[axis] = np.float32(
                weighted_intercept(
                    unwrapped,
                    offsets,
                    weights,
                    config.affine_ridge,
                    margin,
                )
                % period
            )

        conditional_log_fraction = weighted_intercept(
            np.log(np.maximum(fractions[ids, slots], 1e-10)),
            offsets,
            weights,
            config.affine_ridge,
            1.0,
        )
        path_energy = np.abs(coefficients[ids, slots]) ** 2
        profile = path_energy / np.maximum(path_energy.sum(axis=1, keepdims=True), TINY)
        port_profile = np.sum(weights[:, None] * profile, axis=0)
        port_profile /= max(float(np.sum(port_profile)), TINY)
        conditional_fraction = float(math.exp(conditional_log_fraction))
        candidates.append(
            (
                predicted_coordinate,
                port_profile.astype(np.float32),
                conditional_fraction,
            )
        )
        observed_counts.append(len(ids))
        presences.append(presence)

    diagnostics = {
        "candidate_count": float(len(candidates)),
        "mean_observations": float(np.mean(observed_counts)) if observed_counts else 0.0,
        "mean_presence": float(np.mean(presences)) if presences else 0.0,
    }
    # Presence is returned separately through a fourth value encoded below.
    output = [
        (coordinate, profile, fraction, presence)
        for (coordinate, profile, fraction), presence in zip(candidates, presences)
    ]
    return output, diagnostics


def render_candidate_set(
    candidates: list[tuple[np.ndarray, np.ndarray, float, float]],
    presence_power: float,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.zeros((config.output_paths, 3), dtype=np.float32)
    coefficients = np.zeros((config.output_paths, MP * N_UE), dtype=np.complex64)
    ranked: list[tuple[float, np.ndarray, np.ndarray]] = []
    for coordinate, profile, conditional_fraction, presence in candidates:
        score = conditional_fraction * max(presence, 1e-6) ** presence_power
        ranked.append((score, coordinate, profile))
    ranked.sort(key=lambda row: row[0], reverse=True)
    for output_slot, (score, coordinate, profile) in enumerate(
        ranked[: config.output_paths]
    ):
        coordinates[output_slot] = coordinate
        coefficients[output_slot] = np.sqrt(
            np.maximum(score * profile, 0.0)
        ).astype(np.complex64)
    return coordinates, coefficients


def predict_oof(
    positions: np.ndarray,
    site: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    point_power: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    config: Config,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    valid, fractions = valid_components(
        component_energy(coefficients),
        point_power,
        config.component_count,
        config.minimum_fraction,
    )
    folds = np.load(FOLD_CACHE)
    variants = {
        f"track_presence_p{power:g}": (
            np.zeros((len(query_ids), config.output_paths, 3), dtype=np.float32),
            np.zeros(
                (len(query_ids), config.output_paths, MP * N_UE), dtype=np.complex64
            ),
        )
        for power in config.presence_powers
    }
    nearest_coordinates = np.zeros(
        (len(query_ids), config.output_paths, 3), dtype=np.float32
    )
    nearest_coefficients = np.zeros(
        (len(query_ids), config.output_paths, MP * N_UE), dtype=np.complex64
    )
    fold_diagnostics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, float]] = []

    for fold in np.unique(fold_id):
        rows = np.flatnonzero(fold_id == fold)
        fit_ids = folds[f"fold{int(fold)}_fit"].astype(np.int64)
        roots, graph_diagnostics = build_fold_tracks(
            fit_ids,
            positions,
            site,
            coordinates,
            coefficients,
            valid,
            fractions,
            config,
        )
        graph_diagnostics["fold"] = int(fold)
        fold_diagnostics.append(graph_diagnostics)
        for site_value in np.unique(site[query_ids[rows]]):
            selected_rows = rows[site[query_ids[rows]] == site_value]
            bank = fit_ids[(site[fit_ids] == site_value) & (point_power[fit_ids] > TINY)]
            take = min(config.donor_count, len(bank))
            distance, local = cKDTree(positions[bank, :2]).query(
                positions[query_ids[selected_rows], :2], k=take, workers=-1
            )
            if take == 1:
                distance = distance[:, None]
                local = local[:, None]
            donor_table = bank[local]
            for local_row, output_row in enumerate(selected_rows):
                donors = donor_table[local_row]
                nearest = int(donors[0])
                nearest_coordinates[output_row] = coordinates[
                    nearest, : config.output_paths
                ]
                nearest_coefficients[output_row] = coefficients[
                    nearest, : config.output_paths
                ]
                candidates, diagnostics = predict_track_candidates(
                    positions[query_ids[output_row]],
                    donors,
                    distance[local_row],
                    positions,
                    coordinates,
                    coefficients,
                    valid,
                    fractions,
                    roots,
                    config,
                )
                prediction_rows.append(diagnostics)
                for power in config.presence_powers:
                    predicted_coordinates, predicted_coefficients = render_candidate_set(
                        candidates, power, config
                    )
                    variant_coordinates, variant_coefficients = variants[
                        f"track_presence_p{power:g}"
                    ]
                    variant_coordinates[output_row] = predicted_coordinates
                    variant_coefficients[output_row] = predicted_coefficients
        print(f"path-track field fold={int(fold)} rows={len(rows)}", flush=True)

    variants["nearest_component_set"] = (
        nearest_coordinates, nearest_coefficients
    )
    diagnostics: dict[str, object] = {
        "fold_graphs": fold_diagnostics,
        "prediction": {
            key: float(np.mean([row[key] for row in prediction_rows]))
            for key in prediction_rows[0]
        },
    }
    return variants, diagnostics


def evaluate_variants(
    variants: dict[str, tuple[np.ndarray, np.ndarray]],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    query_site: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    truth_ad = np.load(TRAIN_AD, mmap_mode="r")
    truth_pas = np.load(TRAIN_PAS, mmap_mode="r")
    truth_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    arrays = {
        name: {
            "ad": np.zeros(len(query_ids), dtype=np.float64),
            "pas": np.zeros(len(query_ids), dtype=np.float64),
            "pdp": np.zeros(len(query_ids), dtype=np.float64),
        }
        for name in variants
    }
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        truth_ids = query_ids[start:end]
        block_ad = np.asarray(truth_ad[truth_ids], dtype=np.float32)
        block_pas = np.asarray(truth_pas[truth_ids], dtype=np.float32)
        block_pdp = np.asarray(truth_pdp[truth_ids], dtype=np.float32)
        for name, (coordinates, coefficients) in variants.items():
            predicted_ad, predicted_pas, predicted_pdp = render_noncoherent(
                coordinates[start:end], coefficients[start:end], device
            )
            arrays[name]["ad"][start:end] = cosine(
                predicted_ad, block_ad, (1, 2, 3)
            )
            arrays[name]["pas"][start:end] = cosine(
                predicted_pas, block_pas, 1
            ).mean(axis=(1, 2), dtype=np.float64)
            arrays[name]["pdp"][start:end] = cosine(
                predicted_pdp, block_pdp, 3
            ).mean(axis=(1, 2), dtype=np.float64)
        print(f"path-track field evaluation {end}/{len(query_ids)}", flush=True)
    return {
        name: {
            metric: summarize(values, fold_id, query_site)
            for metric, values in rows.items()
        }
        for name, rows in arrays.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    component_cache = np.load(COMPONENTS)
    coordinates = component_cache["coordinates"].astype(np.float32)
    coefficients = component_cache["coefficients"].astype(np.complex64)
    point_power = component_cache["point_power"].astype(np.float64)
    _, positions, _, train_site, _ = load_positions()
    config = Config()

    variants, diagnostics = predict_oof(
        positions,
        train_site,
        coordinates,
        coefficients,
        point_power,
        query_ids,
        fold_id,
        config,
    )
    np.savez_compressed(
        PREDICTIONS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        **{
            f"{name}_{part}": values[index]
            for name, values in variants.items()
            for index, part in enumerate(("coordinates", "coefficients"))
        },
    )
    results = evaluate_variants(
        variants,
        query_ids,
        fold_id,
        train_site[query_ids],
        torch.device(args.device),
        args.batch_size,
    )
    ranking = sorted(
        (
            {
                "name": name,
                "pas_pdp_mean": 0.5
                * (row["pas"]["mean"] + row["pdp"]["mean"]),
            }
            for name, row in results.items()
        ),
        key=lambda row: row["pas_pdp_mean"],
        reverse=True,
    )
    payload = {
        "experiment": "Round2-E9-fit-only-path-track-spectral-field",
        "status": "analysis_only",
        "primary_hypothesis": (
            "A fit-only component graph can preserve path identity through a spatial "
            "hole better than nearest-anchor component matching."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes",
        "config": asdict(config),
        "diagnostics": diagnostics,
        "results": results,
        "ranking": ranking,
        "prediction_cache": PATHS.relative(PREDICTIONS),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
