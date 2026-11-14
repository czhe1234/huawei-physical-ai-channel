#!/usr/bin/env python3
"""Audit effective surface identities inferred only from official H and the map."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
from scipy.spatial import cKDTree

import analyze_v4_map_path_support_ceiling as map_support
import analyze_v4_map_virtual_source_phase as map_phase
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import MH, MV, S_SC, circular_delta, component_energy


VERSION = "Round2-E143-effective-surface-identity"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "effective_surface_identity"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_effective_surface_identity"
REPORT = ANALYSIS_DIR / "effective_surface_identity.json"
OUTPUT = ARTIFACT_DIR / "effective_surface_paths.npz"
COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
FULL_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
LOS_LABELS = (
    PATHS.artifacts
    / "round2"
    / "geometric_los_inventory"
    / "geometric_los_labels.npz"
)

METERS_PER_DELAY_BIN = map_phase.METERS_PER_DELAY_BIN
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    minimum_component_fraction: float = 0.005
    minimum_excess_length_m: float = 0.50
    wall_sample_spacing_m: float = 1.0
    nearest_wall_samples: int = 256
    nearest_validation_points: int = 128
    control_neighbors: int = 48
    control_repeats: int = 8
    surface_distance_m: float = 2.0
    minimum_fit_points_per_signature: int = 5
    wall_alignment_margin: float = 0.15
    ordered_reuse_minimum: float = 0.35
    ordered_reuse_margin: float = 0.15
    specular_cosine_margin: float = 0.10
    seed: int = 20_260_809_143


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-repeats", type=int, default=Config.control_repeats)
    parser.add_argument("--seed", type=int, default=Config.seed)
    return parser.parse_args()


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(value[valid] * weight[valid]) / np.sum(weight[valid]))


def weighted_quantile(
    value: np.ndarray, weight: np.ndarray, quantiles: tuple[float, ...]
) -> list[float]:
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        return [float("nan") for _ in quantiles]
    order = np.argsort(value[valid])
    ordered_value = value[valid][order]
    ordered_weight = weight[valid][order]
    cumulative = np.cumsum(ordered_weight)
    cumulative /= cumulative[-1]
    return [
        float(ordered_value[min(np.searchsorted(cumulative, q), len(order) - 1)])
        for q in quantiles
    ]


def angle_solutions(value: np.ndarray, scale: float, orientation: np.ndarray) -> np.ndarray:
    signed = np.where(value <= scale, value, value - 2.0 * scale)
    principal = np.arcsin(np.clip(signed / scale, -1.0, 1.0))
    return np.stack((principal + orientation, math.pi - principal + orientation), axis=1)


def build_orientations(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    fold_id: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    point_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    horizontal = np.zeros(len(positions), dtype=np.float64)
    vertical = np.zeros(len(positions), dtype=np.float64)
    records: dict[str, object] = {}
    for fold in range(5):
        fit = np.flatnonzero(fold_id != fold)
        records[str(fold)] = {}
        for site_value in np.unique(site):
            site_value = int(site_value)
            h_value, v_value, count = map_support.calibrate_array(
                fit,
                site_value,
                positions,
                site,
                bs_positions[site_value],
                coordinates,
                coefficients,
                point_power,
            )
            rows = (fold_id == fold) & (site == site_value)
            horizontal[rows] = h_value
            vertical[rows] = v_value
            records[str(fold)][str(site_value)] = {
                "horizontal_degrees": float(np.degrees(h_value)),
                "vertical_degrees": float(np.degrees(v_value)),
                "fit_direct_path_count": int(count),
            }
    return horizontal, vertical, records


def select_paths(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    geometric_los: np.ndarray,
    config: Config,
) -> dict[str, np.ndarray]:
    energy = component_energy(coefficients).astype(np.float64)
    total = energy.sum(axis=1, keepdims=True)
    fraction = np.divide(
        energy,
        np.maximum(total, TINY),
        out=np.zeros_like(energy),
    )
    bs = bs_positions[site]
    direct_vector = positions - bs
    direct_length = np.linalg.norm(direct_vector, axis=1)
    azimuth = np.arctan2(direct_vector[:, 1], direct_vector[:, 0])
    elevation = np.arctan2(
        direct_vector[:, 2], np.maximum(np.linalg.norm(direct_vector[:, :2], axis=1), 1e-9)
    )
    expected = np.column_stack(
        (
            np.mod(8.0 * np.sin(azimuth - horizontal), MH),
            np.mod(4.0 * np.sin(elevation - vertical), MV),
            np.mod(direct_length / METERS_PER_DELAY_BIN, S_SC),
        )
    )
    error = np.abs(
        circular_delta(coordinates, expected[:, None, :], np.asarray((MH, MV, S_SC)))
    )
    direct_cost = (
        np.square(error[..., 0] / 0.8)
        + np.square(error[..., 1] / 0.8)
        + np.square(error[..., 2] / 0.35)
        - 0.02 * np.log(np.maximum(fraction, 1e-12))
    )
    direct_component = np.argmin(direct_cost, axis=1)
    direct_match = (
        geometric_los
        & (error[np.arange(len(positions)), direct_component, 0] <= 0.8)
        & (error[np.arange(len(positions)), direct_component, 1] <= 0.8)
        & (error[np.arange(len(positions)), direct_component, 2] <= 0.35)
    )

    path_length = coordinates[..., 2].astype(np.float64) * METERS_PER_DELAY_BIN
    active = (
        (fraction >= config.minimum_component_fraction)
        & (path_length >= direct_length[:, None] + config.minimum_excess_length_m)
    )
    active[np.flatnonzero(direct_match), direct_component[direct_match]] = False
    point_id, component_id = np.nonzero(active)
    selected_fraction = fraction[point_id, component_id]
    return {
        "point_id": point_id.astype(np.int32),
        "component_id": component_id.astype(np.int16),
        "coordinate": coordinates[point_id, component_id].astype(np.float64),
        "mass": selected_fraction.astype(np.float64),
        "direct_length": direct_length[point_id].astype(np.float64),
        "selected_mass_fraction": np.asarray(
            np.sum(selected_fraction) / np.maximum(np.sum(fraction), TINY)
        ),
        "direct_match_count": np.asarray(np.sum(direct_match)),
    }


def sample_walls(
    wall: map_phase.WallBank, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    normal = map_phase.canonicalize_normals(wall.normal)
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    for wall_id in range(len(wall.offset)):
        length = max(float(wall.tangent_max[wall_id] - wall.tangent_min[wall_id]), 0.0)
        count = max(int(math.ceil(length / spacing)) + 1, 2)
        coordinate = np.linspace(
            wall.tangent_min[wall_id], wall.tangent_max[wall_id], count
        )
        xy = (
            wall.offset[wall_id] * normal[wall_id][None]
            + coordinate[:, None] * tangent[wall_id][None]
        )
        samples.append(xy)
        ids.append(np.full(count, wall_id, dtype=np.int32))
    return np.concatenate(samples).astype(np.float64), np.concatenate(ids)


def exact_wall_distance(
    point: np.ndarray,
    candidate_id: np.ndarray,
    wall: map_phase.WallBank,
) -> np.ndarray:
    normal = np.asarray(wall.normal[candidate_id], dtype=np.float64)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    tangent = np.stack((-normal[..., 1], normal[..., 0]), axis=-1)
    normal_error = np.abs(
        np.sum(point[:, None, :2] * normal, axis=-1) - wall.offset[candidate_id]
    )
    tangent_coordinate = np.sum(point[:, None, :2] * tangent, axis=-1)
    tangent_error = np.maximum(
        np.maximum(
            wall.tangent_min[candidate_id] - tangent_coordinate,
            tangent_coordinate - wall.tangent_max[candidate_id],
        ),
        0.0,
    )
    z_error = np.maximum(
        np.maximum(
            wall.z_min[candidate_id] - point[:, None, 2],
            point[:, None, 2] - wall.z_max[candidate_id],
        ),
        0.0,
    )
    return np.sqrt(np.square(normal_error) + np.square(tangent_error) + np.square(z_error))


def nearest_walls(
    points: np.ndarray,
    wall: map_phase.WallBank,
    sample_tree: cKDTree,
    sample_wall_id: np.ndarray,
    neighbors: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.full(len(points), np.inf, dtype=np.float64)
    wall_id = np.full(len(points), -1, dtype=np.int32)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        _, sample_index = sample_tree.query(
            points[start:stop, :2], k=min(neighbors, len(sample_wall_id)), workers=-1
        )
        if sample_index.ndim == 1:
            sample_index = sample_index[:, None]
        candidates = sample_wall_id[sample_index]
        local_distance = exact_wall_distance(points[start:stop], candidates, wall)
        best = np.argmin(local_distance, axis=1)
        distance[start:stop] = local_distance[np.arange(stop - start), best]
        wall_id[start:stop] = candidates[np.arange(stop - start), best]
    return distance, wall_id


def validate_nearest_walls(
    points: np.ndarray,
    approximate_distance: np.ndarray,
    approximate_id: np.ndarray,
    wall: map_phase.WallBank,
    count: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(points), size=min(count, len(points)), replace=False)
    all_ids = np.arange(len(wall.offset), dtype=np.int32)
    exact_distance = np.empty(len(rows), dtype=np.float64)
    exact_id = np.empty(len(rows), dtype=np.int32)
    for local, row in enumerate(rows):
        values = exact_wall_distance(points[row : row + 1], all_ids[None], wall)[0]
        exact_id[local] = int(np.argmin(values))
        exact_distance[local] = values[exact_id[local]]
    distance_error = np.abs(approximate_distance[rows] - exact_distance)
    return {
        "count": int(len(rows)),
        "wall_id_agreement": float(np.mean(approximate_id[rows] == exact_id)),
        "distance_max_abs_error_m": float(np.max(distance_error, initial=0.0)),
        "distance_mean_abs_error_m": float(np.mean(distance_error)),
    }


def infer_effective_surface(
    coordinate: np.ndarray,
    point_id: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    wall: map_phase.WallBank,
    sample_tree: cKDTree,
    sample_wall_id: np.ndarray,
    config: Config,
) -> dict[str, np.ndarray]:
    point_site = site[point_id]
    bs = bs_positions[point_site]
    ue = positions[point_id]
    vector = ue - bs
    direct = np.linalg.norm(vector, axis=1)
    total_length = coordinate[:, 2] * METERS_PER_DELAY_BIN
    azimuth = angle_solutions(coordinate[:, 0], 8.0, horizontal[point_id])
    elevation = angle_solutions(coordinate[:, 1], 4.0, vertical[point_id])
    directions = np.empty((len(coordinate), 4, 3), dtype=np.float64)
    slot = 0
    for h_index in range(2):
        for v_index in range(2):
            cosine = np.cos(elevation[:, v_index])
            directions[:, slot, 0] = cosine * np.cos(azimuth[:, h_index])
            directions[:, slot, 1] = cosine * np.sin(azimuth[:, h_index])
            directions[:, slot, 2] = np.sin(elevation[:, v_index])
            slot += 1
    denominator = 2.0 * (
        total_length[:, None] - np.sum(directions * vector[:, None], axis=2)
    )
    numerator = np.square(total_length) - np.square(direct)
    ray_distance = np.divide(
        numerator[:, None],
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1e-9,
    )
    valid = (
        np.isfinite(ray_distance)
        & (total_length[:, None] >= direct[:, None] + config.minimum_excess_length_m)
        & (ray_distance > 0.0)
        & (ray_distance < total_length[:, None])
    )
    candidates = bs[:, None] + ray_distance[..., None] * directions
    flat = candidates.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    flat[~flat_valid] = np.asarray((1e6, 1e6, 1e6), dtype=np.float64)
    flat_distance, flat_wall = nearest_walls(
        flat,
        wall,
        sample_tree,
        sample_wall_id,
        config.nearest_wall_samples,
    )
    candidate_distance = flat_distance.reshape(len(coordinate), 4)
    candidate_distance[~valid] = np.inf
    selected = np.argmin(candidate_distance, axis=1)
    rows = np.arange(len(coordinate))
    selected_valid = np.isfinite(candidate_distance[rows, selected])
    selected_point = candidates[rows, selected]
    selected_wall = flat_wall.reshape(len(coordinate), 4)[rows, selected]
    selected_distance = candidate_distance[rows, selected]
    selected_point[~selected_valid] = np.nan
    selected_wall[~selected_valid] = -1

    normal = np.zeros((len(coordinate), 2), dtype=np.float64)
    good = selected_wall >= 0
    normal[good] = map_phase.canonicalize_normals(wall.normal[selected_wall[good]])
    bs_side = np.sum(bs[:, :2] * normal, axis=1) - np.where(
        good, wall.offset[np.maximum(selected_wall, 0)], 0.0
    )
    ue_side = np.sum(ue[:, :2] * normal, axis=1) - np.where(
        good, wall.offset[np.maximum(selected_wall, 0)], 0.0
    )
    side_code = (bs_side >= 0.0).astype(np.int32) * 2 + (ue_side >= 0.0).astype(np.int32)
    signature = np.where(good, selected_wall * 4 + side_code, -1).astype(np.int32)

    incoming = selected_point - bs
    outgoing = ue - selected_point
    incoming /= np.maximum(np.linalg.norm(incoming, axis=1, keepdims=True), 1e-12)
    outgoing /= np.maximum(np.linalg.norm(outgoing, axis=1, keepdims=True), 1e-12)
    normal_3d = np.column_stack((normal, np.zeros(len(normal))))
    reflected = incoming - 2.0 * np.sum(incoming * normal_3d, axis=1, keepdims=True) * normal_3d
    specular_cosine = np.sum(reflected * outgoing, axis=1)
    specular_cosine[~good] = np.nan
    return {
        "point": selected_point.astype(np.float32),
        "wall_distance": selected_distance.astype(np.float32),
        "wall_id": selected_wall.astype(np.int32),
        "signature": signature,
        "specular_cosine": specular_cosine.astype(np.float32),
        "solution": selected.astype(np.int8),
        "valid": selected_valid,
        "all_candidate_points": candidates.astype(np.float32),
        "all_candidate_distance": candidate_distance.astype(np.float32),
    }


def matched_control_indices(
    point_id: np.ndarray,
    coordinate: np.ndarray,
    mass: np.ndarray,
    direct_length: np.ndarray,
    site: np.ndarray,
    repeats: int,
    neighbors: int,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    outputs = [np.empty(len(point_id), dtype=np.int64) for _ in range(repeats)]
    source_site = site[point_id]
    feature = np.column_stack(
        (
            direct_length / 20.0,
            coordinate[:, 2],
            np.log(np.maximum(mass, 1e-8)),
        )
    )
    for site_value in np.unique(source_site):
        rows = np.flatnonzero(source_site == site_value)
        tree = cKDTree(feature[rows])
        _, local = tree.query(
            feature[rows], k=min(neighbors, len(rows)), workers=-1
        )
        if local.ndim == 1:
            local = local[:, None]
        candidates = rows[local]
        different = point_id[candidates] != point_id[rows, None]
        for repeat in range(repeats):
            random_score = rng.random(candidates.shape)
            random_score[~different] = np.inf
            choice = np.argmin(random_score, axis=1)
            selected = candidates[np.arange(len(rows)), choice]
            missing = ~np.any(different, axis=1)
            selected[missing] = rows[missing]
            outputs[repeat][rows] = selected
    first = outputs[0]
    return outputs, {
        "same_point_fraction": float(np.mean(point_id[first] == point_id)),
        "median_direct_length_difference_m": float(
            np.median(np.abs(direct_length[first] - direct_length))
        ),
        "median_delay_difference_bin": float(
            np.median(np.abs(coordinate[first, 2] - coordinate[:, 2]))
        ),
        "median_log_mass_difference": float(
            np.median(
                np.abs(
                    np.log(np.maximum(mass[first], 1e-8))
                    - np.log(np.maximum(mass, 1e-8))
                )
            )
        ),
    }


def alignment_metrics(result: dict[str, np.ndarray], mass: np.ndarray) -> dict[str, object]:
    distance = result["wall_distance"].astype(np.float64)
    specular = result["specular_cosine"].astype(np.float64)
    q10, q50, q90 = weighted_quantile(distance, mass, (0.10, 0.50, 0.90))
    return {
        "valid_mass_fraction": weighted_mean(result["valid"].astype(float), mass),
        "wall_distance_weighted_mean_m": weighted_mean(distance, mass),
        "wall_distance_weighted_p10_m": q10,
        "wall_distance_weighted_median_m": q50,
        "wall_distance_weighted_p90_m": q90,
        "wall_within_0p5m_mass": weighted_mean((distance <= 0.5).astype(float), mass),
        "wall_within_1m_mass": weighted_mean((distance <= 1.0).astype(float), mass),
        "wall_within_2m_mass": weighted_mean((distance <= 2.0).astype(float), mass),
        "wall_within_5m_mass": weighted_mean((distance <= 5.0).astype(float), mass),
        "specular_cosine_weighted_mean": weighted_mean(specular, mass),
        "specular_cosine_above_0p8_mass": weighted_mean((specular >= 0.8).astype(float), mass),
    }


def reusable_signature_metrics(
    signature: np.ndarray,
    wall_distance: np.ndarray,
    point_id: np.ndarray,
    mass: np.ndarray,
    fold_id: np.ndarray,
    config: Config,
) -> dict[str, object]:
    high_confidence = (
        (signature >= 0)
        & np.isfinite(wall_distance)
        & (wall_distance <= config.surface_distance_m)
    )
    fold_rows: list[dict[str, float | int]] = []
    reusable_total = 0.0
    mass_total = 0.0
    for fold in range(5):
        held = fold_id[point_id] == fold
        fit = ~held
        fit_rows = np.flatnonzero(fit & high_confidence)
        pair = np.unique(
            np.column_stack((signature[fit_rows], point_id[fit_rows])), axis=0
        )
        unique_signature, count = np.unique(pair[:, 0], return_counts=True) if len(pair) else (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )
        reusable = unique_signature[count >= config.minimum_fit_points_per_signature]
        held_reusable = held & high_confidence & np.isin(signature, reusable)
        numerator = float(np.sum(mass[held_reusable]))
        denominator = float(np.sum(mass[held]))
        reusable_total += numerator
        mass_total += denominator
        fold_rows.append(
            {
                "fold": fold,
                "held_path_count": int(np.sum(held)),
                "fit_reusable_signature_count": int(len(reusable)),
                "reusable_mass_fraction": numerator / max(denominator, TINY),
            }
        )
    return {
        "surface_distance_m": config.surface_distance_m,
        "minimum_fit_points_per_signature": config.minimum_fit_points_per_signature,
        "reusable_mass_fraction": reusable_total / max(mass_total, TINY),
        "folds": fold_rows,
    }


def aggregate_controls(values: list[dict[str, object]]) -> dict[str, object]:
    keys = [key for key, value in values[0].items() if isinstance(value, (int, float))]
    return {
        key: {
            "mean": float(np.mean([float(value[key]) for value in values])),
            "std": float(np.std([float(value[key]) for value in values])),
            "min": float(np.min([float(value[key]) for value in values])),
            "max": float(np.max([float(value[key]) for value in values])),
        }
        for key in keys
    }


def main() -> None:
    args = parse_args()
    config = Config(control_repeats=args.control_repeats, seed=args.seed)
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    setup, positions, _, site, _ = load_positions()
    component = np.load(COMPONENTS, allow_pickle=False)
    coordinates = component["coordinates"].astype(np.float32)
    coefficients = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    fold_id = np.load(FULL_FOLD, allow_pickle=False).astype(np.int8)
    geometric_los = np.load(LOS_LABELS, allow_pickle=False)["train_center_clear"].astype(bool)

    horizontal, vertical, orientation_records = build_orientations(
        positions,
        site,
        setup.bs_positions,
        fold_id,
        coordinates,
        coefficients,
        point_power,
    )
    selected = select_paths(
        positions,
        site,
        setup.bs_positions,
        coordinates,
        coefficients,
        horizontal,
        vertical,
        geometric_los,
        config,
    )
    point_id = selected["point_id"]
    coordinate = selected["coordinate"]
    mass = selected["mass"]

    wall = map_phase.build_wall_bank(map_phase.Config())
    wall_sample, sample_wall_id = sample_walls(wall, config.wall_sample_spacing_m)
    sample_tree = cKDTree(wall_sample)
    actual = infer_effective_surface(
        coordinate,
        point_id,
        positions,
        site,
        setup.bs_positions,
        horizontal,
        vertical,
        wall,
        sample_tree,
        sample_wall_id,
        config,
    )
    nearest_validation = validate_nearest_walls(
        actual["point"][actual["valid"]],
        actual["wall_distance"][actual["valid"]],
        actual["wall_id"][actual["valid"]],
        wall,
        config.nearest_validation_points,
        config.seed,
    )

    control_indices, control_match = matched_control_indices(
        point_id,
        coordinate,
        mass,
        selected["direct_length"],
        site,
        config.control_repeats,
        config.control_neighbors,
        config.seed,
    )
    control_results: list[dict[str, np.ndarray]] = []
    control_alignment: list[dict[str, object]] = []
    control_reuse: list[dict[str, object]] = []
    for repeat, source in enumerate(control_indices):
        result = infer_effective_surface(
            coordinate[source],
            point_id,
            positions,
            site,
            setup.bs_positions,
            horizontal,
            vertical,
            wall,
            sample_tree,
            sample_wall_id,
            config,
        )
        control_results.append(result)
        control_alignment.append(alignment_metrics(result, mass))
        control_reuse.append(
            reusable_signature_metrics(
                result["signature"],
                result["wall_distance"],
                point_id,
                mass,
                fold_id,
                config,
            )
        )
        print(f"E143 matched geometry control {repeat + 1}/{config.control_repeats}", flush=True)

    actual_alignment = alignment_metrics(actual, mass)
    actual_reuse = reusable_signature_metrics(
        actual["signature"],
        actual["wall_distance"],
        point_id,
        mass,
        fold_id,
        config,
    )
    control_alignment_summary = aggregate_controls(control_alignment)
    control_reuse_fraction = np.asarray(
        [float(value["reusable_mass_fraction"]) for value in control_reuse]
    )
    wall_margin = float(actual_alignment["wall_within_2m_mass"]) - float(
        control_alignment_summary["wall_within_2m_mass"]["mean"]
    )
    reuse_margin = float(actual_reuse["reusable_mass_fraction"]) - float(
        np.mean(control_reuse_fraction)
    )
    specular_margin = float(actual_alignment["specular_cosine_weighted_mean"]) - float(
        control_alignment_summary["specular_cosine_weighted_mean"]["mean"]
    )
    positive_reuse_folds = int(
        sum(
            float(actual_reuse["folds"][fold]["reusable_mass_fraction"])
            > float(np.mean([value["folds"][fold]["reusable_mass_fraction"] for value in control_reuse]))
            for fold in range(5)
        )
    )
    gate = {
        "required_wall_alignment_margin": config.wall_alignment_margin,
        "required_ordered_reuse": config.ordered_reuse_minimum,
        "required_ordered_reuse_margin": config.ordered_reuse_margin,
        "required_specular_cosine_margin": config.specular_cosine_margin,
        "required_positive_reuse_folds": 5,
        "wall_alignment_margin": wall_margin,
        "ordered_reuse": float(actual_reuse["reusable_mass_fraction"]),
        "ordered_reuse_margin": reuse_margin,
        "specular_cosine_margin": specular_margin,
        "positive_reuse_folds": positive_reuse_folds,
    }
    gate["passed"] = bool(
        wall_margin >= config.wall_alignment_margin
        and float(actual_reuse["reusable_mass_fraction"]) >= config.ordered_reuse_minimum
        and reuse_margin >= config.ordered_reuse_margin
        and specular_margin >= config.specular_cosine_margin
        and positive_reuse_folds == 5
    )

    np.savez_compressed(
        OUTPUT,
        point_id=point_id,
        component_id=selected["component_id"],
        coordinate=coordinate.astype(np.float32),
        mass=mass.astype(np.float32),
        horizontal_orientation=horizontal.astype(np.float32),
        vertical_orientation=vertical.astype(np.float32),
        effective_point=actual["point"],
        wall_distance=actual["wall_distance"],
        wall_id=actual["wall_id"],
        ordered_signature=actual["signature"],
        specular_cosine=actual["specular_cosine"],
        selected_solution=actual["solution"],
        valid=actual["valid"],
        first_control_source=control_indices[0],
        first_control_wall_distance=control_results[0]["wall_distance"],
        first_control_wall_id=control_results[0]["wall_id"],
        first_control_ordered_signature=control_results[0]["signature"],
    )
    report = {
        "experiment": VERSION,
        "status": "information_gate_passed" if gate["passed"] else "information_gate_failed",
        "official_inputs_only": True,
        "sionna_or_v12_used": False,
        "primary_hypothesis": (
            "Absolute delay and BS departure angle in official H identify a reusable, "
            "ordered BS-side/surface/UE-side interaction state beyond flat map features."
        ),
        "config": asdict(config),
        "input": {
            "train_points": int(len(positions)),
            "selected_path_count": int(len(point_id)),
            "selected_component_mass_fraction_of_all_points": float(
                selected["selected_mass_fraction"]
            ),
            "geometric_los_direct_component_excluded": int(selected["direct_match_count"]),
            "surface_family_count": int(len(wall.offset)),
            "wall_sample_count": int(len(wall_sample)),
        },
        "strict_array_calibration": orientation_records,
        "nearest_wall_acceleration_validation": nearest_validation,
        "matched_control": control_match,
        "actual_alignment": actual_alignment,
        "control_alignment": control_alignment_summary,
        "actual_ordered_signature_reuse": actual_reuse,
        "control_ordered_signature_reuse": {
            "mean": float(np.mean(control_reuse_fraction)),
            "std": float(np.std(control_reuse_fraction)),
            "min": float(np.min(control_reuse_fraction)),
            "max": float(np.max(control_reuse_fraction)),
            "fold_mean": [
                float(np.mean([value["folds"][fold]["reusable_mass_fraction"] for value in control_reuse]))
                for fold in range(5)
            ],
        },
        "gate": gate,
        "decision": {
            "run_frozen_topology_endpoint_ceiling": bool(gate["passed"]),
            "next_if_passed": (
                "Use only fit-fold surface signatures to build a query surface-state "
                "predictor, then require at least +0.01 through a frozen full-channel interface."
            ),
            "next_if_failed": (
                "Do not train a topology model; the inverse surface identity is not "
                "distinguishable from matched urban-wall controls."
            ),
        },
        "outputs": {
            "path_cache": PATHS.relative(OUTPUT),
            "path_cache_bytes": int(OUTPUT.stat().st_size),
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
