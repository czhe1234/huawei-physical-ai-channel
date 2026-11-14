#!/usr/bin/env python3
"""Test official-map image sources as fit-only path phase references for Round 2."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from data import load_positions
from evaluate_v3_anchored_projection import TARGET_AD, TARGET_PAS, TARGET_PDP, target_batch
from official_metrics import point_metric_statistics
from pre_sionna_spectral import project_channel
from project_paths import PATHS
from run_path_component_transport import (
    MH,
    MV,
    PORTS,
    S_SC,
    TINY,
    ComponentCache,
    Config as TransportConfig,
    circular_delta,
    component_energy,
    estimate_wavevector,
    match_components,
    preserve_power,
    synthesize_batch,
    wavevector_grid,
)
from run_rbf_spectral_field import ACTIONS, OOF_BASE, OOF_META, statistics


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_map_virtual_source_phase"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_map_virtual_source_phase"
REPORT = ANALYSIS_DIR / "map_virtual_source_phase.json"
WALL_CACHE = ARTIFACT_DIR / "round2_wall_planes.npz"
PREDICTION_CACHE = ARTIFACT_DIR / "oof_map_virtual_source_components.npz"
COMPONENT_CACHE = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
TRANSPORT_PREDICTION = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "oof_path_parameters_k16_os2_ref2_n40_dir3600_r1.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
V3_METRICS = (
    PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / "oof_projection_metrics.npz"
)
GEOMETRY_CACHE = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_geometry_visibility_features.npz"
)

LIGHT_SPEED = 299_792_458.0
SUBCARRIER_SPACING = 30_000.0
METERS_PER_DELAY_BIN = LIGHT_SPEED / (S_SC * SUBCARRIER_SPACING)
ROTATED_CONTROL_RADIANS = math.radians(37.0)


@dataclass(frozen=True)
class Config:
    paths: int = 16
    donors: int = 40
    vertical_normal_z: float = 0.10
    plane_round_decimals: int = 3
    minimum_plane_area: float = 2.0
    tangent_margin: float = 1.0
    height_margin: float = 1.0
    query_delay_gate: float = 0.65
    donor_delay_rmse_gate: float = 0.45
    donor_valid_fraction: float = 0.65
    maximum_candidates: int = 160
    minimum_observations: int = 6
    map_margins: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10)
    beta_min: float = 135.0
    beta_max: float = 145.0
    beta_coarse_step: float = 0.005
    beta_fine_step: float = 0.0002
    seed: int = 20_260_805


@dataclass(frozen=True)
class WallBank:
    normal: np.ndarray
    offset: np.ndarray
    tangent_min: np.ndarray
    tangent_max: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    area: np.ndarray
    face_count: np.ndarray


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def circular_abs(value: np.ndarray, target: np.ndarray, period: float) -> np.ndarray:
    return np.abs(circular_delta(value, target, period))


def load_component_cache() -> ComponentCache:
    cached = np.load(COMPONENT_CACHE)
    return ComponentCache(
        coordinates=cached["coordinates"].astype(np.float32),
        coefficients=cached["coefficients"].astype(np.complex64),
        residual_fraction=cached["residual_fraction"].astype(np.float32),
        point_power=cached["point_power"].astype(np.float64),
    )


def canonicalize_normals(normal: np.ndarray) -> np.ndarray:
    output = np.asarray(normal, dtype=np.float64).copy()
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    flip = (output[:, 0] < 0.0) | (
        (np.abs(output[:, 0]) < 1e-10) & (output[:, 1] < 0.0)
    )
    output[flip] *= -1.0
    return output


def build_wall_bank(config: Config) -> WallBank:
    if WALL_CACHE.is_file():
        cached = np.load(WALL_CACHE)
        cache_config = json.loads(str(cached["config_json"]))
        required = {
            "vertical_normal_z": config.vertical_normal_z,
            "plane_round_decimals": config.plane_round_decimals,
            "minimum_plane_area": config.minimum_plane_area,
        }
        if all(cache_config.get(key) == value for key, value in required.items()):
            return WallBank(
                normal=cached["normal"].astype(np.float64),
                offset=cached["offset"].astype(np.float64),
                tangent_min=cached["tangent_min"].astype(np.float64),
                tangent_max=cached["tangent_max"].astype(np.float64),
                z_min=cached["z_min"].astype(np.float64),
                z_max=cached["z_max"].astype(np.float64),
                area=cached["area"].astype(np.float64),
                face_count=cached["face_count"].astype(np.int32),
            )

    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(mesh).__name__}")
    face_normal = np.asarray(mesh.face_normals, dtype=np.float64)
    face_area = np.asarray(mesh.area_faces, dtype=np.float64)
    triangle = np.asarray(mesh.triangles, dtype=np.float64)
    center = np.asarray(mesh.triangles_center, dtype=np.float64)
    selected = np.flatnonzero(
        (np.abs(face_normal[:, 2]) < config.vertical_normal_z) & (face_area > 0.0)
    )
    normal = canonicalize_normals(face_normal[selected, :2])
    offset = np.sum(normal * center[selected, :2], axis=1)
    key = np.round(
        np.column_stack([normal, offset]), decimals=config.plane_round_decimals
    )
    _, inverse = np.unique(key, axis=0, return_inverse=True)

    rows: list[tuple[np.ndarray, float, float, float, float, float, float, int]] = []
    for group in range(int(np.max(inverse)) + 1):
        local = np.flatnonzero(inverse == group)
        area = float(np.sum(face_area[selected[local]]))
        if area < config.minimum_plane_area:
            continue
        group_normal = np.sum(
            face_area[selected[local], None] * normal[local], axis=0
        )
        group_normal /= max(float(np.linalg.norm(group_normal)), 1e-12)
        group_offset = float(
            np.sum(face_area[selected[local]] * offset[local])
            / max(float(np.sum(face_area[selected[local]])), 1e-12)
        )
        vertices = triangle[selected[local]].reshape(-1, 3)
        tangent = np.asarray([-group_normal[1], group_normal[0]], dtype=np.float64)
        tangent_coordinate = vertices[:, :2] @ tangent
        z_min = float(np.min(vertices[:, 2]))
        z_max = float(np.max(vertices[:, 2]))
        if z_max < 0.5 or z_max - z_min < 0.25:
            continue
        rows.append(
            (
                group_normal,
                group_offset,
                float(np.min(tangent_coordinate)),
                float(np.max(tangent_coordinate)),
                z_min,
                z_max,
                area,
                int(len(local)),
            )
        )

    rows.sort(key=lambda row: row[6], reverse=True)
    bank = WallBank(
        normal=np.stack([row[0] for row in rows]).astype(np.float64),
        offset=np.asarray([row[1] for row in rows], dtype=np.float64),
        tangent_min=np.asarray([row[2] for row in rows], dtype=np.float64),
        tangent_max=np.asarray([row[3] for row in rows], dtype=np.float64),
        z_min=np.asarray([row[4] for row in rows], dtype=np.float64),
        z_max=np.asarray([row[5] for row in rows], dtype=np.float64),
        area=np.asarray([row[6] for row in rows], dtype=np.float64),
        face_count=np.asarray([row[7] for row in rows], dtype=np.int32),
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        WALL_CACHE,
        normal=bank.normal.astype(np.float32),
        offset=bank.offset.astype(np.float32),
        tangent_min=bank.tangent_min.astype(np.float32),
        tangent_max=bank.tangent_max.astype(np.float32),
        z_min=bank.z_min.astype(np.float32),
        z_max=bank.z_max.astype(np.float32),
        area=bank.area.astype(np.float32),
        face_count=bank.face_count,
        config_json=np.asarray(json.dumps(asdict(config))),
    )
    return bank


def image_sources(bank: WallBank, bs_position: np.ndarray) -> np.ndarray:
    signed = bs_position[:2] @ bank.normal.T - bank.offset
    output = np.repeat(np.asarray(bs_position, dtype=np.float64)[None], len(bank.offset), axis=0)
    output[:, :2] -= 2.0 * signed[:, None] * bank.normal
    return output


def wall_geometry(
    positions: np.ndarray,
    sources: np.ndarray,
    bank: WallBank,
    candidate_ids: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Return path lengths and finite-wall validity as [point, candidate]."""

    point = np.asarray(positions, dtype=np.float64)
    source = sources[candidate_ids]
    normal = bank.normal[candidate_ids]
    offset = bank.offset[candidate_ids]
    delta = point[:, None, :] - source[None, :, :]
    length = np.linalg.norm(delta, axis=2)
    denominator = np.einsum("pci,ci->pc", delta[:, :, :2], normal)
    numerator = offset - np.einsum("ci,ci->c", source[:, :2], normal)
    parameter = numerator[None, :] / np.where(
        np.abs(denominator) > 1e-12, denominator, np.nan
    )
    reflection = source[None, :, :] + parameter[:, :, None] * delta
    tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
    tangent_coordinate = np.einsum("pci,ci->pc", reflection[:, :, :2], tangent)
    valid = (
        np.isfinite(parameter)
        & (parameter > 0.0)
        & (parameter < 1.0)
        & (tangent_coordinate >= bank.tangent_min[candidate_ids][None] - config.tangent_margin)
        & (tangent_coordinate <= bank.tangent_max[candidate_ids][None] + config.tangent_margin)
        & (reflection[:, :, 2] >= bank.z_min[candidate_ids][None] - config.height_margin)
        & (reflection[:, :, 2] <= bank.z_max[candidate_ids][None] + config.height_margin)
    )
    return length, valid


def direct_components(
    fit_ids: np.ndarray,
    site_value: int,
    positions: np.ndarray,
    site: np.ndarray,
    bs_position: np.ndarray,
    cache: ComponentCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = fit_ids[(site[fit_ids] == site_value) & (cache.point_power[fit_ids] > TINY)]
    coordinates = cache.coordinates[ids]
    coefficients = cache.coefficients[ids]
    energy = component_energy(coefficients)
    direct_length = np.linalg.norm(positions[ids] - bs_position[None], axis=1)
    direct_delay = direct_length / METERS_PER_DELAY_BIN
    delay_error = circular_abs(coordinates[:, :, 2], direct_delay[:, None], S_SC)
    provisional = np.argmin(
        delay_error - 0.02 * np.log(np.maximum(energy, 1e-30)), axis=1
    )
    keep = delay_error[np.arange(len(ids)), provisional] <= 0.40
    ids = ids[keep]
    direct_length = direct_length[keep]
    provisional = provisional[keep]
    coordinates = coordinates[keep]
    coefficients = coefficients[keep]
    energy = energy[keep]

    azimuth = np.arctan2(
        positions[ids, 1] - bs_position[1], positions[ids, 0] - bs_position[0]
    )
    observed_h = coordinates[np.arange(len(ids)), provisional, 0]
    selected_energy = energy[np.arange(len(ids)), provisional]
    orientation_grid = np.linspace(-math.pi, math.pi, 1441)
    orientation_error = np.empty(len(orientation_grid), dtype=np.float64)
    orientation_weight = np.sqrt(np.maximum(selected_energy, 1e-30))
    for row, orientation in enumerate(orientation_grid):
        expected_h = np.mod(8.0 * np.sin(azimuth - orientation), MH)
        orientation_error[row] = np.average(
            circular_abs(observed_h, expected_h, MH), weights=orientation_weight
        )
    orientation = float(orientation_grid[int(np.argmin(orientation_error))])
    expected_h = np.mod(8.0 * np.sin(azimuth - orientation), MH)
    delay_error = circular_abs(
        coordinates[:, :, 2], direct_length[:, None] / METERS_PER_DELAY_BIN, S_SC
    )
    horizontal_error = circular_abs(coordinates[:, :, 0], expected_h[:, None], MH)
    energy_fraction = energy / np.maximum(np.sum(energy, axis=1, keepdims=True), 1e-30)
    score = (
        delay_error / 0.25
        + horizontal_error / 0.80
        - 0.15 * np.log(np.maximum(energy_fraction, 1e-8))
    )
    selected = np.argmin(score, axis=1)
    keep = (
        (delay_error[np.arange(len(ids)), selected] <= 0.35)
        & (horizontal_error[np.arange(len(ids)), selected] <= 1.20)
        & (energy_fraction[np.arange(len(ids)), selected] >= 0.05)
    )
    ids = ids[keep]
    selected = selected[keep]
    return (
        ids,
        direct_length[keep],
        coefficients[keep, selected],
        np.asarray([orientation], dtype=np.float64),
    )


def beta_objective(
    beta: np.ndarray,
    phase: np.ndarray,
    delta_length: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(beta), dtype=np.float64)
    for start in range(0, len(beta), 512):
        end = min(start + 512, len(beta))
        residual = phase[:, None] + delta_length[:, None] * beta[None, start:end]
        output[start:end] = np.abs(
            np.sum(weight[:, None] * np.exp(1j * residual), axis=0)
        )
    return output


def calibrate_beta(
    fit_ids: np.ndarray,
    site_value: int,
    positions: np.ndarray,
    site: np.ndarray,
    bs_position: np.ndarray,
    cache: ComponentCache,
    config: Config,
) -> tuple[float, dict[str, float | int]]:
    ids, length, coefficient, orientation = direct_components(
        fit_ids, site_value, positions, site, bs_position, cache
    )
    unit = coefficient / np.maximum(np.linalg.norm(coefficient, axis=1, keepdims=True), 1e-20)
    norm = np.linalg.norm(coefficient, axis=1)
    distance, local = cKDTree(positions[ids, :2]).query(
        positions[ids, :2], k=min(7, len(ids)), workers=-1
    )
    left: list[int] = []
    right: list[int] = []
    for row in range(len(ids)):
        for separation, neighbor in zip(distance[row, 1:], local[row, 1:]):
            neighbor = int(neighbor)
            if 1.0 <= float(separation) <= 10.0 and row < neighbor:
                left.append(row)
                right.append(neighbor)
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    inner = np.sum(unit[left_array] * np.conj(unit[right_array]), axis=1)
    coherence = np.abs(inner)
    keep = coherence >= 0.60
    phase = np.angle(inner[keep]).astype(np.float64)
    delta_length = (length[left_array[keep]] - length[right_array[keep]]).astype(np.float64)
    weight = (
        coherence[keep]
        * np.sqrt(np.maximum(norm[left_array[keep]] * norm[right_array[keep]], 1e-30))
    ).astype(np.float64)
    weight /= max(float(np.sum(weight)), TINY)
    coarse = np.arange(
        config.beta_min,
        config.beta_max + 0.5 * config.beta_coarse_step,
        config.beta_coarse_step,
    )
    coarse_score = beta_objective(coarse, phase, delta_length, weight)
    coarse_best = float(coarse[int(np.argmax(coarse_score))])
    fine = np.arange(
        coarse_best - config.beta_coarse_step,
        coarse_best + config.beta_coarse_step + 0.5 * config.beta_fine_step,
        config.beta_fine_step,
    )
    fine_score = beta_objective(fine, phase, delta_length, weight)
    best_index = int(np.argmax(fine_score))
    beta = float(fine[best_index])
    return beta, {
        "direct_component_count": int(len(ids)),
        "pair_count": int(len(phase)),
        "orientation_degrees": float(np.degrees(orientation[0])),
        "beta_rad_per_m": beta,
        "frequency_ghz": beta * LIGHT_SPEED / (2.0 * math.pi) / 1e9,
        "fit_phase_coherence": float(fine_score[best_index]),
        "pair_port_coherence": float(np.sum(weight * coherence[keep])),
    }


def split_phase_score(
    unit: np.ndarray,
    phase: np.ndarray,
    weights: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all-observation and spatial half-to-half coherence per candidate."""

    rotated = unit[:, None, :] * phase[:, :, None]
    all_latent = np.sum(weights[:, None, None] * rotated, axis=0)
    all_score = np.linalg.norm(all_latent, axis=1) / max(float(np.sum(weights)), TINY)
    centered = positions - np.average(positions, axis=0, weights=weights)
    if len(positions) < 4 or np.linalg.norm(centered) < 1e-10:
        return all_score, all_score.copy()
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ right[0]
    order = np.argsort(projection)
    left_mask = np.zeros(len(positions), dtype=bool)
    left_mask[order[: len(order) // 2]] = True
    right_mask = ~left_mask
    if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
        return all_score, all_score.copy()
    left = np.sum(weights[left_mask, None, None] * rotated[left_mask], axis=0)
    right_value = np.sum(weights[right_mask, None, None] * rotated[right_mask], axis=0)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right_value, axis=1)
    cross = np.real(np.sum(np.conj(left) * right_value, axis=1))
    split_score = cross / np.maximum(denominator, TINY)
    return all_score, split_score


def rotate_source(source: np.ndarray, bs_position: np.ndarray) -> np.ndarray:
    cosine = math.cos(ROTATED_CONTROL_RADIANS)
    sine = math.sin(ROTATED_CONTROL_RADIANS)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    output = np.asarray(source, dtype=np.float64).copy()
    output[:2] = bs_position[:2] + rotation @ (output[:2] - bs_position[:2])
    return output


def predict_point_source(
    query_position: np.ndarray,
    current_coefficient: np.ndarray,
    observation_ids: np.ndarray,
    observation_components: np.ndarray,
    observation_match: np.ndarray,
    positions: np.ndarray,
    cache: ComponentCache,
    source_position: np.ndarray,
    beta: float,
) -> np.ndarray:
    offsets = positions[observation_ids, :2] - query_position[None, :2]
    distance = np.linalg.norm(offsets, axis=1)
    donor_scale = max(float(np.median(distance)), 1.0)
    weights = (
        np.exp(-0.5 * (distance / max(1.5 * donor_scale, 1.0)) ** 2)
        * np.maximum(observation_match, 0.10)
    )
    weights /= max(float(np.sum(weights)), TINY)
    coefficient = cache.coefficients[observation_ids, observation_components]
    unit = coefficient / np.maximum(np.linalg.norm(coefficient, axis=1, keepdims=True), 1e-20)
    donor_length = np.linalg.norm(
        positions[observation_ids] - source_position[None], axis=1
    )
    query_length = float(np.linalg.norm(query_position - source_position))
    latent = np.sum(
        weights[:, None] * unit * np.exp(1j * beta * donor_length)[:, None], axis=0
    )
    latent /= max(float(np.linalg.norm(latent)), 1e-20)
    prediction = latent * np.exp(-1j * beta * query_length)
    prediction *= max(float(np.linalg.norm(current_coefficient)), 1e-20)
    return prediction.astype(np.complex64)


def select_strict_los_path(
    query_position: np.ndarray,
    bs_position: np.ndarray,
    orientation_degrees: float,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
) -> int:
    direct_length = float(np.linalg.norm(query_position - bs_position))
    expected_delay = direct_length / METERS_PER_DELAY_BIN
    azimuth = math.atan2(
        query_position[1] - bs_position[1], query_position[0] - bs_position[0]
    )
    orientation = math.radians(orientation_degrees)
    expected_h = float(np.mod(8.0 * math.sin(azimuth - orientation), MH))
    delay_error = circular_abs(coordinates[:, 2], expected_delay, S_SC)
    horizontal_error = circular_abs(coordinates[:, 0], expected_h, MH)
    energy = component_energy(coefficients)
    fraction = energy / max(float(np.sum(energy)), TINY)
    score = (
        delay_error / 0.25
        + horizontal_error / 0.80
        - 0.15 * np.log(np.maximum(fraction, 1e-8))
    )
    selected = int(np.argmin(score))
    if (
        delay_error[selected] > 0.35
        or horizontal_error[selected] > 1.20
        or fraction[selected] < 0.05
    ):
        return -1
    return selected


def path_observations(
    anchor_id: int,
    donor_ids: np.ndarray,
    cache: ComponentCache,
    config: TransportConfig,
) -> list[list[tuple[int, int, float]]]:
    anchor_coordinates = cache.coordinates[anchor_id, : config.transported_components]
    anchor_coefficients = cache.coefficients[anchor_id, : config.transported_components]
    rows: list[list[tuple[int, int, float]]] = [
        [(anchor_id, path, 1.0)] for path in range(config.transported_components)
    ]
    for donor_id in donor_ids[1:]:
        donor_id = int(donor_id)
        matched, _, coherence = match_components(
            anchor_coordinates,
            anchor_coefficients,
            cache.coordinates[donor_id],
            cache.coefficients[donor_id],
            config,
        )
        for path, component in enumerate(matched):
            if component >= 0:
                rows[path].append((donor_id, int(component), float(coherence[path])))
    return rows


def source_candidates(
    query_position: np.ndarray,
    predicted_delay: float,
    sources: np.ndarray,
    bank: WallBank,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.arange(len(bank.offset), dtype=np.int64)
    length, valid = wall_geometry(query_position[None], sources, bank, ids, config)
    delay_error = circular_abs(length[0] / METERS_PER_DELAY_BIN, predicted_delay, S_SC)
    keep = np.flatnonzero(valid[0] & (delay_error <= config.query_delay_gate))
    if len(keep) > config.maximum_candidates:
        priority = delay_error[keep] - 0.01 * np.log1p(bank.area[keep])
        keep = keep[np.argsort(priority)[: config.maximum_candidates]]
    return keep, length[0, keep], delay_error[keep]


def predict_source_path(
    query_position: np.ndarray,
    predicted_delay: float,
    current_coefficient: np.ndarray,
    observation_ids: np.ndarray,
    observation_components: np.ndarray,
    observation_match: np.ndarray,
    positions: np.ndarray,
    cache: ComponentCache,
    bs_position: np.ndarray,
    sources: np.ndarray,
    bank: WallBank,
    beta: float,
    transport_config: TransportConfig,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    offsets = positions[observation_ids, :2] - query_position[None, :2]
    distance = np.linalg.norm(offsets, axis=1)
    donor_scale = max(float(np.median(distance)), 1.0)
    spatial = np.exp(-0.5 * (distance / max(1.5 * donor_scale, 1.0)) ** 2)
    weights = spatial * np.maximum(observation_match, 0.10)
    weights /= max(float(np.sum(weights)), TINY)
    coefficient = cache.coefficients[observation_ids, observation_components]
    unit = coefficient / np.maximum(np.linalg.norm(coefficient, axis=1, keepdims=True), 1e-20)

    grid = wavevector_grid(transport_config)
    radial = query_position[:2] - bs_position[:2]
    radial /= max(float(np.linalg.norm(radial)), 1e-8)
    fallback = -beta * radial
    wavevector, _, _ = estimate_wavevector(coefficient, offsets, weights, grid, fallback)
    wave_phase = np.exp(-1j * (offsets @ wavevector))[:, None]
    wave_all, wave_split = split_phase_score(
        unit, wave_phase, weights, positions[observation_ids, :2]
    )
    wave_score = float(0.45 * wave_all[0] + 0.55 * wave_split[0])

    candidate_ids, query_length, _ = source_candidates(
        query_position, predicted_delay, sources, bank, config
    )
    direct_length = float(np.linalg.norm(query_position - bs_position))
    direct_error = float(
        circular_abs(direct_length / METERS_PER_DELAY_BIN, predicted_delay, S_SC)
    )
    candidate_kind = np.ones(len(candidate_ids) + int(direct_error <= config.query_delay_gate), dtype=np.int8)
    if direct_error <= config.query_delay_gate:
        candidate_kind[0] = 0
        candidate_ids = np.concatenate([np.asarray([-1], dtype=np.int64), candidate_ids])
        query_length = np.concatenate([[direct_length], query_length])
    if len(candidate_ids) == 0:
        return current_coefficient, current_coefficient, {
            "candidate_count": 0,
            "selected_kind": -1,
            "score_margin": -np.inf,
            "wave_score": wave_score,
            "map_score": -np.inf,
            "delay_rmse": np.inf,
            "valid_fraction": 0.0,
        }

    wall_mask = candidate_ids >= 0
    path_length = np.empty((len(observation_ids), len(candidate_ids)), dtype=np.float64)
    valid = np.ones_like(path_length, dtype=bool)
    if np.any(wall_mask):
        wall_length, wall_valid = wall_geometry(
            positions[observation_ids],
            sources,
            bank,
            candidate_ids[wall_mask],
            config,
        )
        path_length[:, wall_mask] = wall_length
        valid[:, wall_mask] = wall_valid
    if np.any(~wall_mask):
        path_length[:, ~wall_mask] = np.linalg.norm(
            positions[observation_ids] - bs_position[None], axis=1
        )[:, None]

    observed_delay = cache.coordinates[observation_ids, observation_components, 2]
    delay_error = circular_abs(
        path_length / METERS_PER_DELAY_BIN,
        observed_delay[:, None],
        S_SC,
    )
    valid_weight = weights[:, None] * valid
    valid_fraction = np.sum(valid_weight, axis=0) / max(float(np.sum(weights)), TINY)
    delay_rmse = np.sqrt(
        np.sum(valid_weight * delay_error**2, axis=0)
        / np.maximum(np.sum(valid_weight, axis=0), TINY)
    )
    eligible = (
        (valid_fraction >= config.donor_valid_fraction)
        & (delay_rmse <= config.donor_delay_rmse_gate)
    )
    phase = np.exp(1j * beta * path_length)
    phase[~valid] = 0.0
    all_score, split_score = split_phase_score(
        unit,
        phase,
        weights,
        positions[observation_ids, :2],
    )
    phase_score = 0.45 * all_score + 0.55 * split_score
    ranking_score = (
        phase_score
        - 0.02 * (delay_rmse / config.donor_delay_rmse_gate) ** 2
        - 0.02 * (1.0 - valid_fraction)
    )
    ranking_score[~eligible] = -np.inf
    if not np.any(np.isfinite(ranking_score)):
        return current_coefficient, current_coefficient, {
            "candidate_count": int(len(candidate_ids)),
            "selected_kind": -1,
            "score_margin": -np.inf,
            "wave_score": wave_score,
            "map_score": -np.inf,
            "delay_rmse": np.inf,
            "valid_fraction": 0.0,
        }
    selected = int(np.argmax(ranking_score))
    selected_valid_weight = weights * valid[:, selected]
    latent = np.sum(
        selected_valid_weight[:, None]
        * unit
        * np.exp(1j * beta * path_length[:, selected])[:, None],
        axis=0,
    )
    latent_norm = max(float(np.linalg.norm(latent)), 1e-20)
    predicted = latent / latent_norm * np.exp(-1j * beta * query_length[selected])
    predicted *= max(float(np.linalg.norm(current_coefficient)), 1e-20)

    if candidate_ids[selected] < 0:
        rotated = predicted.copy()
    else:
        rotated_source = rotate_source(sources[candidate_ids[selected]], bs_position)
        rotated_donor_length = np.linalg.norm(
            positions[observation_ids] - rotated_source[None], axis=1
        )
        rotated_query_length = float(np.linalg.norm(query_position - rotated_source))
        rotated_latent = np.sum(
            selected_valid_weight[:, None]
            * unit
            * np.exp(1j * beta * rotated_donor_length)[:, None],
            axis=0,
        )
        rotated_latent /= max(float(np.linalg.norm(rotated_latent)), 1e-20)
        rotated = rotated_latent * np.exp(-1j * beta * rotated_query_length)
        rotated *= max(float(np.linalg.norm(current_coefficient)), 1e-20)

    return predicted.astype(np.complex64), rotated.astype(np.complex64), {
        "candidate_count": int(len(candidate_ids)),
        "selected_kind": int(candidate_kind[selected]),
        "selected_wall_id": int(candidate_ids[selected]),
        "score_margin": float(phase_score[selected] - wave_score),
        "wave_score": wave_score,
        "map_score": float(phase_score[selected]),
        "ranking_score": float(ranking_score[selected]),
        "map_all_coherence": float(all_score[selected]),
        "map_split_coherence": float(split_score[selected]),
        "delay_rmse": float(delay_rmse[selected]),
        "valid_fraction": float(valid_fraction[selected]),
    }


def support_match(
    predicted_coordinates: np.ndarray,
    true_coordinates: np.ndarray,
) -> np.ndarray:
    delta_h = circular_delta(
        true_coordinates[None, :, 0], predicted_coordinates[:, None, 0], MH
    )
    delta_v = circular_delta(
        true_coordinates[None, :, 1], predicted_coordinates[:, None, 1], MV
    )
    delta_d = circular_delta(
        true_coordinates[None, :, 2], predicted_coordinates[:, None, 2], S_SC
    )
    cost = (delta_h / 1.5) ** 2 + (delta_v / 1.0) ** 2 + (delta_d / 2.0) ** 2
    left, right = linear_sum_assignment(cost)
    output = np.full(len(predicted_coordinates), -1, dtype=np.int16)
    for predicted, truth in zip(left, right):
        if (
            cost[predicted, truth] <= 9.0
            and abs(delta_h[predicted, truth]) <= 3.0
            and abs(delta_v[predicted, truth]) <= 2.0
            and abs(delta_d[predicted, truth]) <= 4.0
        ):
            output[predicted] = truth
    return output


def path_metrics(
    coordinates: np.ndarray,
    variants: dict[str, np.ndarray],
    cache: ComponentCache,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
) -> dict[str, object]:
    accumulators = {
        name: {
            "truth_energy": np.zeros(len(query_ids), dtype=np.float64),
            "error": np.zeros(len(query_ids), dtype=np.float64),
            "signed": np.zeros(len(query_ids), dtype=np.float64),
            "absolute": np.zeros(len(query_ids), dtype=np.float64),
            "phase_error": np.zeros(len(query_ids), dtype=np.float64),
        }
        for name in variants
    }
    matched_energy = np.zeros(len(query_ids), dtype=np.float64)
    total_energy = np.zeros(len(query_ids), dtype=np.float64)
    for row, point_id in enumerate(query_ids):
        match = support_match(coordinates[row], cache.coordinates[point_id])
        true_energy = component_energy(cache.coefficients[point_id])
        total_energy[row] = float(np.sum(true_energy))
        accepted = np.flatnonzero(match >= 0)
        truth_index = match[accepted].astype(np.int64)
        truth = cache.coefficients[point_id, truth_index]
        energy = np.sum(np.abs(truth) ** 2, axis=1, dtype=np.float64)
        matched_energy[row] = float(np.sum(energy))
        for name, coefficient in variants.items():
            prediction = coefficient[row, accepted]
            inner = np.sum(np.conj(prediction) * truth, axis=1)
            denominator = np.maximum(
                np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1),
                TINY,
            )
            accumulators[name]["truth_energy"][row] = float(np.sum(energy))
            accumulators[name]["error"][row] = float(
                np.sum(np.abs(prediction - truth) ** 2, dtype=np.float64)
            )
            accumulators[name]["signed"][row] = float(
                np.sum(energy * np.real(inner) / denominator)
            )
            accumulators[name]["absolute"][row] = float(
                np.sum(energy * np.abs(inner) / denominator)
            )
            accumulators[name]["phase_error"][row] = float(
                np.sum(energy * np.abs(np.angle(inner)))
            )

    def summarize_rows(name: str, mask: np.ndarray) -> dict[str, float | int]:
        values = accumulators[name]
        energy = float(np.sum(values["truth_energy"][mask]))
        return {
            "count": int(np.sum(mask)),
            "matched_energy": energy,
            "coefficient_nmse": float(np.sum(values["error"][mask]) / max(energy, TINY)),
            "signed_coherence": float(np.sum(values["signed"][mask]) / max(energy, TINY)),
            "absolute_coherence": float(np.sum(values["absolute"][mask]) / max(energy, TINY)),
            "phase_error_radians": float(
                np.sum(values["phase_error"][mask]) / max(energy, TINY)
            ),
        }

    output: dict[str, object] = {
        "support_matched_energy_fraction": float(
            np.sum(matched_energy) / max(float(np.sum(total_energy)), TINY)
        ),
        "variants": {},
    }
    for name in variants:
        output["variants"][name] = {
            "all": summarize_rows(name, np.ones(len(query_ids), dtype=bool)),
            "fold": [
                {"fold": int(fold), **summarize_rows(name, fold_id == fold)}
                for fold in np.unique(fold_id)
            ],
        }
    return output


def blend_coefficient_direction(
    current: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    current_norm = np.linalg.norm(current, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    current_unit = current / np.maximum(current_norm, 1e-20)
    target_unit = target / np.maximum(target_norm, 1e-20)
    mixed = (1.0 - alpha) * current_unit + alpha * target_unit
    mixed /= np.maximum(np.linalg.norm(mixed, axis=-1, keepdims=True), 1e-20)
    return (mixed * current_norm).astype(np.complex64)


def add_visibility_los_variants(
    variants: dict[str, np.ndarray],
    query_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    if not GEOMETRY_CACHE.is_file():
        raise FileNotFoundError(GEOMETRY_CACHE)
    geometry = np.load(GEOMETRY_CACHE)
    names = geometry["names"].astype(str).tolist()
    train_geometry = geometry["train"]
    center_clear = train_geometry[query_ids, names.index("center_hit_count")] == 0.0
    fan_clear = train_geometry[query_ids, names.index("fan_clear_fraction")] >= 0.60
    current = variants["free_wave"]
    calibrated = variants["strict_los_calibrated"]
    old_beta = variants["strict_los_beta_140p4_control"]
    for gate_name, gate in (("center_clear", center_clear), ("fan_clear_060", fan_clear)):
        for alpha in (0.25, 0.50, 1.00):
            blended = blend_coefficient_direction(current, calibrated, alpha)
            output = current.copy()
            output[gate] = blended[gate]
            variants[f"strict_los_{gate_name}_alpha_{alpha:.2f}"] = output
        control = current.copy()
        control[gate] = old_beta[gate]
        variants[f"strict_los_{gate_name}_beta_140p4_control"] = control
    return variants


def build_predictions(
    cache: ComponentCache,
    wall_bank: WallBank,
    config: Config,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object], np.ndarray, np.ndarray]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    transport = np.load(TRANSPORT_PREDICTION)
    if not np.array_equal(query_ids, transport["query_ids"].astype(np.int64)):
        raise RuntimeError("OOF query order differs from path-transport prediction")
    coordinates = transport["coordinates"].astype(np.float32)
    current = transport["coefficients"].astype(np.complex64)
    variants = {
        "free_wave": current.copy(),
        "strict_los_calibrated": current.copy(),
        "strict_los_beta_140p4_control": current.copy(),
        "map_all_eligible": current.copy(),
        "direct_all_eligible": current.copy(),
        "wall_all_eligible": current.copy(),
        "rotated_control_all_eligible": current.copy(),
    }
    for margin in config.map_margins:
        variants[f"map_margin_{margin:.2f}"] = current.copy()
    variants["rotated_control_margin_0.05"] = current.copy()

    setup, positions, _, site, _ = load_positions()
    folds = np.load(FOLD_CACHE)
    transport_config = TransportConfig(
        transported_components=config.paths,
        donor_count=config.donors,
        direction_bins=3600,
        refinement_sweeps=2,
    )
    beta_rows: list[dict[str, float | int]] = []
    diagnostic_rows: list[dict[str, float | int]] = []
    strict_los_rows: list[dict[str, float | int]] = []
    valid = cache.point_power > TINY
    for held_fold in np.unique(fold_id):
        fit_ids = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
        fold_rows = np.flatnonzero(fold_id == held_fold)
        for site_value in np.unique(site[query_ids[fold_rows]]):
            beta, beta_report = calibrate_beta(
                fit_ids,
                int(site_value),
                positions,
                site,
                setup.bs_positions[int(site_value)],
                cache,
                config,
            )
            beta_rows.append(
                {"fold": int(held_fold), "site": int(site_value), **beta_report}
            )
            orientation_degrees = float(beta_report["orientation_degrees"])
            selected_rows = fold_rows[site[query_ids[fold_rows]] == site_value]
            bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
            _, local = cKDTree(positions[bank, :2]).query(
                positions[query_ids[selected_rows], :2],
                k=min(config.donors, len(bank)),
                workers=-1,
            )
            donor_table = bank[local]
            sources = image_sources(wall_bank, setup.bs_positions[int(site_value)])
            for local_row, output_row in enumerate(selected_rows):
                point_id = int(query_ids[output_row])
                observation_table = path_observations(
                    int(donor_table[local_row, 0]),
                    donor_table[local_row],
                    cache,
                    transport_config,
                )
                strict_los_path = select_strict_los_path(
                    positions[point_id],
                    setup.bs_positions[int(site_value)],
                    orientation_degrees,
                    coordinates[output_row],
                    current[output_row],
                )
                for path, observations in enumerate(observation_table):
                    if len(observations) < config.minimum_observations:
                        continue
                    observation_ids = np.asarray([row[0] for row in observations], dtype=np.int64)
                    observation_components = np.asarray(
                        [row[1] for row in observations], dtype=np.int64
                    )
                    observation_match = np.asarray(
                        [row[2] for row in observations], dtype=np.float64
                    )
                    if path == strict_los_path:
                        variants["strict_los_calibrated"][output_row, path] = predict_point_source(
                            positions[point_id],
                            current[output_row, path],
                            observation_ids,
                            observation_components,
                            observation_match,
                            positions,
                            cache,
                            setup.bs_positions[int(site_value)],
                            beta,
                        )
                        variants["strict_los_beta_140p4_control"][output_row, path] = (
                            predict_point_source(
                                positions[point_id],
                                current[output_row, path],
                                observation_ids,
                                observation_components,
                                observation_match,
                                positions,
                                cache,
                                setup.bs_positions[int(site_value)],
                                140.40,
                            )
                        )
                        strict_los_rows.append(
                            {
                                "fold": int(held_fold),
                                "site": int(site_value),
                                "output_row": int(output_row),
                                "path": int(path),
                                "observations": int(len(observations)),
                            }
                        )
                    predicted, rotated, diagnostics = predict_source_path(
                        positions[point_id],
                        float(coordinates[output_row, path, 2]),
                        current[output_row, path],
                        observation_ids,
                        observation_components,
                        observation_match,
                        positions,
                        cache,
                        setup.bs_positions[int(site_value)],
                        sources,
                        wall_bank,
                        beta,
                        transport_config,
                        config,
                    )
                    diagnostics.update(
                        {
                            "fold": int(held_fold),
                            "site": int(site_value),
                            "output_row": int(output_row),
                            "path": int(path),
                            "observations": int(len(observations)),
                        }
                    )
                    diagnostic_rows.append(diagnostics)
                    selected_kind = int(diagnostics["selected_kind"])
                    if selected_kind >= 0:
                        variants["map_all_eligible"][output_row, path] = predicted
                        variants["rotated_control_all_eligible"][output_row, path] = rotated
                    if selected_kind == 0:
                        variants["direct_all_eligible"][output_row, path] = predicted
                    elif selected_kind == 1:
                        variants["wall_all_eligible"][output_row, path] = predicted
                    margin = float(diagnostics["score_margin"])
                    for threshold in config.map_margins:
                        if margin >= threshold:
                            variants[f"map_margin_{threshold:.2f}"][output_row, path] = predicted
                    if margin >= 0.05:
                        variants["rotated_control_margin_0.05"][output_row, path] = rotated
            print(
                f"map virtual source fold={int(held_fold)} site={int(site_value)} "
                f"rows={len(selected_rows)} beta={beta:.6f}",
                flush=True,
            )

    diagnostics = {
        "beta_calibration": beta_rows,
        "path_count": int(len(diagnostic_rows)),
        "candidate_count": summarize(
            np.asarray([row["candidate_count"] for row in diagnostic_rows])
        ),
        "score_margin": summarize(
            np.asarray([row["score_margin"] for row in diagnostic_rows])
        ),
        "delay_rmse": summarize(np.asarray([row["delay_rmse"] for row in diagnostic_rows])),
        "valid_fraction": summarize(
            np.asarray([row["valid_fraction"] for row in diagnostic_rows])
        ),
        "selected_direct_count": int(
            np.sum([row["selected_kind"] == 0 for row in diagnostic_rows])
        ),
        "selected_wall_count": int(
            np.sum([row["selected_kind"] == 1 for row in diagnostic_rows])
        ),
        "replacement_count": {
            f"{margin:.2f}": int(
                np.sum([row["score_margin"] >= margin for row in diagnostic_rows])
            )
            for margin in config.map_margins
        },
        "strict_los_count": int(len(strict_los_rows)),
        "strict_los_fold_count": {
            str(int(fold)): int(np.sum([row["fold"] == fold for row in strict_los_rows]))
            for fold in np.unique(fold_id)
        },
    }
    diagnostic_keys = (
        "fold",
        "site",
        "output_row",
        "path",
        "observations",
        "candidate_count",
        "selected_kind",
        "selected_wall_id",
        "score_margin",
        "wave_score",
        "map_score",
        "ranking_score",
        "map_all_coherence",
        "map_split_coherence",
        "delay_rmse",
        "valid_fraction",
    )
    np.savez_compressed(
        PREDICTION_CACHE,
        coordinates=coordinates,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
        config_json=np.asarray(json.dumps(asdict(config))),
        **{
            f"diagnostic_{key}": np.asarray(
                [row.get(key, np.nan) for row in diagnostic_rows]
            )
            for key in diagnostic_keys
        },
        **{f"coefficient_{name}": value for name, value in variants.items()},
    )
    return coordinates, variants, diagnostics, query_ids, fold_id


def metric_payload(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    return {
        "all": statistics(pas, pdp, error, energy),
        "fold": [
            {"fold": int(fold), **statistics(pas, pdp, error, energy, fold_id == fold)}
            for fold in np.unique(fold_id)
        ],
        "site": {
            str(int(value)): statistics(pas, pdp, error, energy, site == value)
            for value in np.unique(site)
        },
    }


@torch.inference_mode()
def full_channel_evaluation(
    coordinates: np.ndarray,
    variants: dict[str, np.ndarray],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    selected_names = (
        "free_wave",
        "strict_los_calibrated",
        "strict_los_center_clear_alpha_1.00",
        "strict_los_fan_clear_060_alpha_1.00",
        "strict_los_fan_clear_060_beta_140p4_control",
    )
    variants = {name: variants[name] for name in selected_names}
    _, _, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    base_all = np.load(OOF_BASE, mmap_mode="r")
    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    v3_metrics = np.load(V3_METRICS)
    selected_action = v3_metrics["selected_action"].astype(np.int64)
    selected_scale = v3_metrics["selected_scale"].astype(np.float32)
    evaluation_names = ["identity"]
    for name in variants:
        evaluation_names.append(f"{name}_complex_0.25")
        evaluation_names.append(f"{name}_complex_0.50")
    delta_names = (
        "strict_los_center_clear_alpha_1.00",
        "strict_los_fan_clear_060_alpha_1.00",
        "strict_los_fan_clear_060_beta_140p4_control",
    )
    for name in delta_names:
        for alpha in (0.25, 0.50, 1.00):
            evaluation_names.append(f"{name}_delta_{alpha:.2f}")
    arrays = {
        name: {
            "pas": np.zeros(len(query_ids), dtype=np.float64),
            "pdp": np.zeros(len(query_ids), dtype=np.float64),
            "error": np.zeros(len(query_ids), dtype=np.float64),
            "energy": np.zeros(len(query_ids), dtype=np.float64),
        }
        for name in evaluation_names
    }
    raw_error = {name: np.zeros(len(query_ids), dtype=np.float64) for name in variants}
    raw_energy = np.zeros(len(query_ids), dtype=np.float64)

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(base_all[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64)
        ad, pas, pdp = target_batch(ad_cache, pas_cache, pdp_cache, start, end, device)
        candidate_channels = {
            name: synthesize_batch(coordinates[start:end], coefficient[start:end], device)
            for name, coefficient in variants.items()
        }
        raw_energy[start:end] = np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64)
        for name, channel in candidate_channels.items():
            raw_error[name][start:end] = np.sum(
                np.abs(channel.cpu().numpy() - truth) ** 2,
                axis=(1, 2, 3),
                dtype=np.float64,
            )

        carriers: dict[str, torch.Tensor] = {"identity": base}
        for name, channel in candidate_channels.items():
            aligned = preserve_power(channel, base)
            for alpha in (0.25, 0.50):
                output_name = f"{name}_complex_{alpha:.2f}"
                if output_name not in arrays:
                    continue
                mixed = preserve_power((1.0 - alpha) * base + alpha * aligned, base)
                carriers[output_name] = mixed
        free_channel = candidate_channels["free_wave"]
        for name in delta_names:
            delta = candidate_channels[name] - free_channel
            for alpha in (0.25, 0.50, 1.00):
                output_name = f"{name}_delta_{alpha:.2f}"
                carriers[output_name] = preserve_power(base + alpha * delta, base)

        for name, carrier in carriers.items():
            projected = torch.empty_like(carrier)
            action_rows = selected_action[start:end]
            for action_index in np.unique(action_rows):
                rows = np.flatnonzero(action_rows == action_index)
                row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
                action = ACTIONS[int(action_index)]
                projected[row_tensor] = project_channel(
                    carrier[row_tensor],
                    ad[row_tensor],
                    pas[row_tensor],
                    pdp[row_tensor],
                    action.alpha_ad,
                    action.alpha_pas,
                    action.alpha_pdp,
                )
            projected *= torch.as_tensor(
                selected_scale[start:end], dtype=torch.float32, device=device
            )[:, None, None, None]
            metric = point_metric_statistics(projected.cpu().numpy(), truth)
            arrays[name]["pas"][start:end] = metric.pas
            arrays[name]["pdp"][start:end] = metric.pdp
            arrays[name]["error"][start:end] = metric.squared_error
            arrays[name]["energy"][start:end] = metric.truth_energy
        if end == len(query_ids) or end % 20 == 0:
            print(f"full map phase evaluation {end}/{len(query_ids)}", flush=True)

    results = {
        name: metric_payload(
            values["pas"],
            values["pdp"],
            values["error"],
            values["energy"],
            fold_id,
            query_site,
        )
        for name, values in arrays.items()
    }
    reference = float(results["identity"]["all"]["score"])
    for name in results:
        results[name]["delta_vs_v3"] = float(results[name]["all"]["score"] - reference)
        results[name]["fold_delta_vs_v3"] = [
            float(current["score"] - baseline["score"])
            for current, baseline in zip(results[name]["fold"], results["identity"]["fold"])
        ]
    return {
        "raw_component_channel_nmse": {
            name: float(np.sum(error) / np.sum(raw_energy)) for name, error in raw_error.items()
        },
        "variants": results,
        "ranking": sorted(
            results, key=lambda name: float(results[name]["all"]["score"]), reverse=True
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--full-evaluation", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config()
    for required in (COMPONENT_CACHE, TRANSPORT_PREDICTION, OOF_META, FOLD_CACHE):
        if not required.is_file():
            raise FileNotFoundError(required)
    cache = load_component_cache()
    wall_bank = build_wall_bank(config)
    if args.reuse_predictions and PREDICTION_CACHE.is_file():
        stored = np.load(PREDICTION_CACHE)
        coordinates = stored["coordinates"].astype(np.float32)
        query_ids = stored["query_ids"].astype(np.int64)
        fold_id = stored["fold_id"].astype(np.int8)
        diagnostics = json.loads(str(stored["diagnostics_json"]))
        variants = {
            key.removeprefix("coefficient_"): stored[key].astype(np.complex64)
            for key in stored.files
            if key.startswith("coefficient_")
        }
    else:
        coordinates, variants, diagnostics, query_ids, fold_id = build_predictions(
            cache, wall_bank, config
        )
    variants = add_visibility_los_variants(variants, query_ids)
    path_result = path_metrics(coordinates, variants, cache, query_ids, fold_id)
    payload: dict[str, object] = {
        "experiment": "Round2-E8-map-conditioned-virtual-source-phase",
        "status": "analysis_only_complete",
        "primary_hypothesis": (
            "Official-map wall image sources provide the missing absolute path-length phase "
            "reference for a subset of tracked delay-angle components and outperform both a "
            "free local plane wave and a rotated-wall control across strict spatial holes."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "wall_bank": {
            "plane_count": int(len(wall_bank.offset)),
            "area": summarize(wall_bank.area),
            "face_count": summarize(wall_bank.face_count),
            "cache": str(WALL_CACHE.relative_to(PATHS.root)),
        },
        "diagnostics": diagnostics,
        "path_evaluation": path_result,
        "full_channel_evaluation": None,
        "outputs": {
            "prediction_cache": str(PREDICTION_CACHE.relative_to(PATHS.root)),
            "report": str(REPORT.relative_to(PATHS.root)),
            "candidate_zip": None,
        },
    }
    if args.full_evaluation:
        payload["full_channel_evaluation"] = full_channel_evaluation(
            coordinates,
            variants,
            query_ids,
            fold_id,
            torch.device(args.device),
            args.batch_size,
        )
    payload["seconds"] = float(time.time() - started)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
