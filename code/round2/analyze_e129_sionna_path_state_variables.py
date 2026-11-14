#!/usr/bin/env python3
"""Discover path-mass state variables from fit-only Sionna trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

import analyze_path_candidate_information_audit as e84
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E129-Sionna-path-state-variable-discovery"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
SIONNA_STATE = (
    PATHS.artifacts
    / "round2"
    / "sionna_multiseed"
    / "primary_d16_base_v4lambda_1m"
    / "full"
    / "state.json"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_path_state_variables"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sionna_path_state_variables"
DATASET_CACHE = ARTIFACT_DIR / "fit_only_plane_trajectory_dataset.npz"
METRICS_CACHE = ARTIFACT_DIR / "path_state_variable_outer0_metrics.npz"
REPORT = ANALYSIS_DIR / "path_state_variable_outer0_audit.json"

OUTER = 0
INNER = 1
PERIODS = np.asarray((16.0, 8.0, 192.0), dtype=np.float64)
TINY = 1e-30
ACTIVE_MASS = 1e-5


@dataclass(frozen=True)
class Config:
    neighbors: int = 32
    candidates: int = 128
    spatial_scale_m: float = 12.0
    plane_round_decimals: int = 3
    ridge: float = 0.08
    maximum_iterations: int = 140
    maximum_leaf_nodes: int = 31
    minimum_samples_leaf: int = 36
    learning_rate: float = 0.06
    l2_regularization: float = 1.0
    random_state: int = 260_808_129


@dataclass
class PointState:
    family: np.ndarray
    fine: np.ndarray
    mass: np.ndarray
    coordinate: np.ndarray
    path_length: np.ndarray
    stability: np.ndarray
    fine_diversity: np.ndarray
    fine_dominance: np.ndarray
    depth: np.ndarray
    interaction_histogram: np.ndarray
    plane_diversity: np.ndarray
    rank_fraction: np.ndarray
    entropy: np.ndarray
    top_mass: np.ndarray
    top_margin: np.ndarray
    family_count: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=140)
    return parser.parse_args()


def circular_delta(value: np.ndarray, reference: np.ndarray, period: float) -> np.ndarray:
    return np.remainder(value - reference + period / 2.0, period) - period / 2.0


def circular_mean(value: np.ndarray, weight: np.ndarray, period: float) -> float:
    phase = 2.0 * np.pi * np.asarray(value, dtype=np.float64) / period
    direction = np.sum(np.asarray(weight, dtype=np.float64) * np.exp(1j * phase))
    if abs(direction) <= TINY:
        return float(np.asarray(value, dtype=np.float64)[np.argmax(weight)] % period)
    return float((np.angle(direction) * period / (2.0 * np.pi)) % period)


def update_hash(current: np.ndarray, value: np.ndarray) -> None:
    with np.errstate(over="ignore"):
        current ^= np.asarray(value, dtype=np.uint64) + np.uint64(
            0x9E3779B97F4A7C15
        )
        current *= np.uint64(1099511628211)


def sequence_hashes(
    source: np.ndarray,
    depth: np.ndarray,
    interactions: np.ndarray,
    identities: np.ndarray,
) -> np.ndarray:
    output = np.full(source.shape, np.uint64(1469598103934665603), dtype=np.uint64)
    update_hash(output, source)
    update_hash(output, depth)
    for step in range(interactions.shape[2]):
        active = step < depth
        update_hash(output, np.where(active, interactions[..., step], 0))
        update_hash(output, np.where(active, identities[..., step], 0))
    return output


def canonical_face_planes(decimals: int) -> tuple[np.ndarray, dict[str, int]]:
    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh).__name__)
    normal = np.array(mesh.face_normals, dtype=np.float64, copy=True)
    center = np.asarray(mesh.triangles_center, dtype=np.float64)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    undecided = np.ones(len(normal), dtype=bool)
    for axis in range(3):
        flip = undecided & (normal[:, axis] < 0.0)
        normal[flip] *= -1.0
        undecided &= np.abs(normal[:, axis]) < 1e-10
    offset = np.sum(normal * center, axis=1)
    key = np.round(np.column_stack((normal, offset)), decimals=decimals)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    return inverse.astype(np.int32), {
        "face_count": int(len(normal)),
        "plane_family_count": int(np.max(inverse) + 1),
    }


def path_lengths(
    vertices: np.ndarray,
    depth: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    block: int = 128,
) -> np.ndarray:
    output = np.zeros(depth.shape, dtype=np.float32)
    steps = np.arange(vertices.shape[2], dtype=np.int64)[None, None, :]
    for start in range(0, len(depth), block):
        end = min(start + block, len(depth))
        local_vertices = np.asarray(vertices[start:end], dtype=np.float64)
        local_depth = np.asarray(depth[start:end], dtype=np.int64)
        bs = bs_positions[sites[start:end]].astype(np.float64)
        ue = positions[start:end].astype(np.float64)
        previous = np.empty_like(local_vertices)
        previous[:, :, 0] = bs[:, None]
        previous[:, :, 1:] = local_vertices[:, :, :-1]
        active = steps < local_depth[:, :, None]
        segment = np.linalg.norm(local_vertices - previous, axis=3)
        total = np.sum(segment * active, axis=2)
        last_index = np.maximum(local_depth - 1, 0)
        last = np.take_along_axis(
            local_vertices, last_index[:, :, None, None], axis=2
        )[:, :, 0]
        last = np.where((local_depth > 0)[:, :, None], last, bs[:, None])
        total += np.linalg.norm(ue[:, None] - last, axis=2)
        output[start:end] = total.astype(np.float32)
    return output


def point_states(
    plane_id: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    configuration: dict[str, object],
) -> tuple[list[PointState], dict[str, object]]:
    mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    source = np.load(PATH_CACHE / "source.npy", mmap_mode="r")
    depth = np.asarray(
        np.load(PATH_CACHE / "interaction_count.npy", mmap_mode="r"), dtype=np.int64
    )
    interactions = np.load(PATH_CACHE / "interactions.npy", mmap_mode="r")
    primitives = np.load(PATH_CACHE / "primitives.npy", mmap_mode="r")
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    angles = np.load(PATH_CACHE / "angles.npy", mmap_mode="r")
    energy_fraction = np.load(PATH_CACHE / "energy_fraction.npy", mmap_mode="r")
    presence = np.load(PATH_CACHE / "presence.npy", mmap_mode="r")
    vertices = np.load(PATH_CACHE / "vertices.npy", mmap_mode="r")

    base_mask = np.asarray(mask & (source == 0), dtype=bool)
    maximum_recurrence = int(np.max(presence[base_mask], initial=1))
    recurrence_scale = math.log1p(max(maximum_recurrence, 1))
    active_step = np.arange(primitives.shape[2])[None, None, :] < depth[..., None]
    active_base_step = base_mask[..., None] & active_step
    maximum_primitive = int(np.max(primitives[active_base_step], initial=0))
    primitive_to_plane = np.empty(max(maximum_primitive + 1, len(plane_id)), dtype=np.int32)
    primitive_to_plane[: len(plane_id)] = plane_id
    if maximum_primitive >= len(plane_id):
        extra = maximum_primitive + 1 - len(plane_id)
        primitive_to_plane[len(plane_id) : maximum_primitive + 1] = (
            int(np.max(plane_id)) + 1 + np.arange(extra, dtype=np.int32)
        )
    plane_sequence = np.zeros(primitives.shape, dtype=np.int32)
    for step in range(primitives.shape[2]):
        local = np.minimum(primitives[..., step], maximum_primitive)
        plane_sequence[..., step] = primitive_to_plane[local]
    family_hash = sequence_hashes(source, depth, interactions, plane_sequence)
    fine_hash = sequence_hashes(source, depth, interactions, primitives)
    coordinate = e84.sionna_coordinates(
        tau, angles, sites, configuration, vertical_alignment=2.0
    )
    length = path_lengths(vertices, depth, positions, sites, bs_positions)

    output: list[PointState] = []
    family_counts: list[int] = []
    for point in range(len(mask)):
        slots = np.flatnonzero(base_mask[point])
        if len(slots) == 0:
            empty = np.empty(0, dtype=np.float32)
            output.append(
                PointState(
                    family=np.empty(0, dtype=np.uint64),
                    fine=np.empty(0, dtype=np.uint64),
                    mass=empty,
                    coordinate=np.empty((0, 3), dtype=np.float32),
                    path_length=empty,
                    stability=empty,
                    fine_diversity=empty,
                    fine_dominance=empty,
                    depth=empty,
                    interaction_histogram=np.empty((0, 4), dtype=np.float32),
                    plane_diversity=empty,
                    rank_fraction=empty,
                    entropy=empty,
                    top_mass=empty,
                    top_margin=empty,
                    family_count=empty,
                )
            )
            family_counts.append(0)
            continue
        local_family = family_hash[point, slots]
        unique, inverse = np.unique(local_family, return_inverse=True)
        raw_quality = np.asarray(
            energy_fraction[point, slots], dtype=np.float64
        ) * np.sqrt(np.maximum(np.asarray(presence[point, slots]), 1.0))
        family_mass = np.zeros(len(unique), dtype=np.float64)
        np.add.at(family_mass, inverse, raw_quality)
        family_mass /= max(float(np.sum(family_mass)), TINY)
        family_coordinate = np.zeros((len(unique), 3), dtype=np.float32)
        family_length = np.zeros(len(unique), dtype=np.float32)
        family_stability = np.zeros(len(unique), dtype=np.float32)
        family_fine = np.zeros(len(unique), dtype=np.uint64)
        family_fine_diversity = np.zeros(len(unique), dtype=np.float32)
        family_fine_dominance = np.zeros(len(unique), dtype=np.float32)
        family_depth = np.zeros(len(unique), dtype=np.float32)
        family_histogram = np.zeros((len(unique), 4), dtype=np.float32)
        family_plane_diversity = np.zeros(len(unique), dtype=np.float32)
        for group in range(len(unique)):
            member = np.flatnonzero(inverse == group)
            local_slots = slots[member]
            weight = raw_quality[member]
            if float(np.sum(weight)) <= TINY:
                weight = np.ones(len(member), dtype=np.float64)
            weight = weight / np.sum(weight)
            for axis, period in enumerate(PERIODS):
                family_coordinate[group, axis] = circular_mean(
                    coordinate[point, local_slots, axis], weight, float(period)
                )
            family_length[group] = float(np.sum(weight * length[point, local_slots]))
            family_stability[group] = float(
                np.sum(
                    weight
                    * np.log1p(np.asarray(presence[point, local_slots], dtype=np.float64))
                    / recurrence_scale
                )
            )
            local_fine = fine_hash[point, local_slots]
            fine_unique, fine_inverse = np.unique(local_fine, return_inverse=True)
            fine_mass = np.zeros(len(fine_unique), dtype=np.float64)
            np.add.at(fine_mass, fine_inverse, raw_quality[member])
            family_fine[group] = fine_unique[int(np.argmax(fine_mass))]
            family_fine_diversity[group] = float(len(fine_unique))
            family_fine_dominance[group] = float(
                np.max(fine_mass) / max(float(np.sum(fine_mass)), TINY)
            )
            representative = int(local_slots[int(np.argmax(raw_quality[member]))])
            local_depth = int(depth[point, representative])
            family_depth[group] = float(local_depth)
            if local_depth:
                local_interaction = np.asarray(
                    interactions[point, representative, :local_depth], dtype=np.int64
                )
                for code in range(1, 5):
                    family_histogram[group, code - 1] = float(
                        np.sum(local_interaction == code) / local_depth
                    )
                family_plane_diversity[group] = float(
                    len(
                        np.unique(
                            plane_sequence[point, representative, :local_depth]
                        )
                    )
                    / local_depth
                )
        order = np.argsort(-family_mass, kind="stable")
        inverse_rank = np.empty(len(order), dtype=np.float32)
        inverse_rank[order] = np.arange(len(order), dtype=np.float32)
        rank_fraction = inverse_rank / max(len(order) - 1, 1)
        entropy_value = float(
            -np.sum(family_mass * np.log(np.maximum(family_mass, TINY)))
            / max(math.log(max(len(family_mass), 2)), TINY)
        )
        top_mass_value = float(np.max(family_mass, initial=0.0))
        sorted_mass = np.sort(family_mass)[::-1]
        margin_value = float(
            sorted_mass[0] - (sorted_mass[1] if len(sorted_mass) > 1 else 0.0)
        )
        output.append(
            PointState(
                family=unique.astype(np.uint64),
                fine=family_fine,
                mass=family_mass.astype(np.float32),
                coordinate=family_coordinate,
                path_length=family_length,
                stability=family_stability,
                fine_diversity=family_fine_diversity,
                fine_dominance=family_fine_dominance,
                depth=family_depth,
                interaction_histogram=family_histogram,
                plane_diversity=family_plane_diversity,
                rank_fraction=rank_fraction,
                entropy=np.full(len(unique), entropy_value, dtype=np.float32),
                top_mass=np.full(len(unique), top_mass_value, dtype=np.float32),
                top_margin=np.full(len(unique), margin_value, dtype=np.float32),
                family_count=np.full(len(unique), len(unique), dtype=np.float32),
            )
        )
        family_counts.append(len(unique))
    return output, {
        "point_count": int(len(output)),
        "base_path_count": int(np.sum(base_mask)),
        "maximum_solver_recurrence": maximum_recurrence,
        "maximum_sionna_primitive": maximum_primitive,
        "non_ply_primitive_ids": int(max(maximum_primitive + 1 - len(plane_id), 0)),
        "family_count": {
            "mean": float(np.mean(family_counts)),
            "median": float(np.median(family_counts)),
            "min": int(np.min(family_counts)),
            "max": int(np.max(family_counts)),
        },
    }


def spatial_weights(distance: np.ndarray, scale: float) -> np.ndarray:
    value = np.exp(-0.5 * np.square(distance / scale)) / np.sqrt(
        np.maximum(distance, 1.0)
    )
    return value / max(float(np.sum(value)), TINY)


def weighted_statistics(value: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(weight * value) / max(float(np.sum(weight)), TINY))
    variance = float(
        np.sum(weight * np.square(value - mean)) / max(float(np.sum(weight)), TINY)
    )
    return mean, math.sqrt(max(variance, 0.0))


def local_linear(
    offsets: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
    ridge: float,
) -> tuple[float, float, float, float, float]:
    if len(value) < 3:
        mean, std = weighted_statistics(value, weight)
        return mean, 0.0, 0.0, std, 0.0
    design = np.column_stack((np.ones(len(offsets)), offsets))
    weighted_design = design * weight[:, None]
    gram = design.T @ weighted_design
    gram[1:, 1:] += np.eye(2) * ridge
    right = design.T @ (weight * value)
    coefficient = np.linalg.solve(gram + np.eye(3) * 1e-9, right)
    residual = value - design @ coefficient
    rmse = math.sqrt(float(np.sum(weight * residual**2) / np.sum(weight)))
    mean = float(np.sum(weight * value) / np.sum(weight))
    total = float(np.sum(weight * np.square(value - mean)))
    r2 = 1.0 - float(np.sum(weight * residual**2)) / max(total, 1e-12)
    return (
        float(coefficient[0]),
        float(coefficient[1]),
        float(coefficient[2]),
        rmse,
        float(np.clip(r2, -2.0, 1.0)),
    )


def query_neighbors(
    positions: np.ndarray,
    sites: np.ndarray,
    fit_rows: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids = np.empty((len(positions), count), dtype=np.int32)
    donor_distance = np.empty((len(positions), count), dtype=np.float32)
    fit_set = set(map(int, fit_rows))
    for site_value in np.unique(sites):
        pool = fit_rows[sites[fit_rows] == site_value]
        rows = np.flatnonzero(sites == site_value)
        tree = cKDTree(positions[pool, :2])
        take = min(count + 1, len(pool))
        distance, local = tree.query(positions[rows, :2], k=take, workers=-1)
        distance = np.atleast_2d(distance)
        local = np.atleast_2d(local)
        for output_row, point, local_distance, local_id in zip(
            rows, rows, distance, local
        ):
            selected: list[int] = []
            selected_distance: list[float] = []
            for current_distance, current_local in zip(local_distance, local_id):
                donor = int(pool[int(current_local)])
                if donor == int(point) and int(point) in fit_set:
                    continue
                selected.append(donor)
                selected_distance.append(float(current_distance))
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise RuntimeError("insufficient fit-only neighbors")
            donor_ids[output_row] = np.asarray(selected, dtype=np.int32)
            donor_distance[output_row] = np.asarray(
                selected_distance, dtype=np.float32
            )
    return donor_ids, donor_distance


def feature_schema() -> tuple[list[str], dict[str, list[int]]]:
    names: list[str] = []
    blocks: dict[str, list[int]] = {}

    def add(block: str, local_names: list[str]) -> None:
        start = len(names)
        names.extend(local_names)
        blocks[block] = list(range(start, len(names)))

    add(
        "unary",
        [
            "anchor_log_mass",
            "anchor_sqrt_mass",
            "anchor_rank_fraction",
            "site",
            "query_x",
            "query_y",
            "bs_distance",
            "bs_bearing_sin",
            "bs_bearing_cos",
            "coordinate_h_sin",
            "coordinate_h_cos",
            "coordinate_v_sin",
            "coordinate_v_cos",
            "coordinate_d_sin",
            "coordinate_d_cos",
            "depth",
            "interaction_1",
            "interaction_2",
            "interaction_3",
            "interaction_4",
            "plane_diversity",
            "path_length",
            "excess_length",
        ],
    )
    add(
        "boundary",
        [
            "active_fraction",
            "weighted_active_fraction",
            "nearest_active",
            "nearest_inactive",
            "signed_boundary_margin",
            "active_within_3m",
            "active_within_6m",
            "active_within_12m",
            "active_within_24m",
            "active_distance_mean",
            "active_distance_std",
            "active_distance_max",
            "nearest_active_gap",
        ],
    )
    add(
        "stability",
        [
            "path_recurrence_mean",
            "path_recurrence_std",
            "path_recurrence_min",
            "path_recurrence_max",
            "fine_diversity_mean",
            "fine_diversity_std",
            "fine_dominance_mean",
            "fine_dominance_std",
            "mass_coefficient_variation",
            "coordinate_h_std",
            "coordinate_v_std",
            "coordinate_d_std",
            "path_length_std",
        ],
    )
    jacobian = [
        "log_mass_intercept",
        "log_mass_grad_x",
        "log_mass_grad_y",
        "log_mass_rmse",
        "log_mass_r2",
    ]
    for axis in ("h", "v", "d"):
        jacobian.extend(
            [
                f"coordinate_{axis}_intercept_delta",
                f"coordinate_{axis}_grad_x",
                f"coordinate_{axis}_grad_y",
                f"coordinate_{axis}_rmse",
                f"coordinate_{axis}_r2",
            ]
        )
    jacobian.extend(
        [
            "path_length_intercept",
            "path_length_grad_x",
            "path_length_grad_y",
            "path_length_rmse",
            "path_length_r2",
            "coordinate_jacobian_norm",
            "coordinate_hv_jacobian_det",
        ]
    )
    add("jacobian", jacobian)
    add(
        "competition",
        [
            "donor_rank_mean",
            "donor_rank_std",
            "donor_entropy_mean",
            "donor_entropy_std",
            "donor_top_mass_mean",
            "donor_top_mass_std",
            "donor_top_margin_mean",
            "donor_top_margin_std",
            "donor_family_count_mean",
            "donor_family_count_std",
            "query_anchor_entropy",
            "query_anchor_top_mass",
            "query_anchor_top_margin",
            "coordinate_neighbor_count_tight",
            "coordinate_neighbor_count_medium",
            "coordinate_neighbor_mass_tight",
            "coordinate_neighbor_mass_medium",
        ],
    )
    return names, blocks


def candidate_features(
    point: int,
    states: list[PointState],
    donor_ids: np.ndarray,
    donor_distance: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    config: Config,
) -> dict[str, np.ndarray]:
    donors = donor_ids[point]
    distance = np.asarray(donor_distance[point], dtype=np.float64)
    point_weight = spatial_weights(distance, config.spatial_scale_m)
    gathered: dict[int, list[tuple[int, int]]] = {}
    for donor_rank, donor in enumerate(donors):
        for local in range(len(states[int(donor)].family)):
            family = int(states[int(donor)].family[local])
            gathered.setdefault(family, []).append((donor_rank, local))
    if not gathered:
        return {
            "family": np.empty(0, dtype=np.uint64),
            "features": np.empty((0, len(feature_schema()[0])), dtype=np.float32),
            "predicted_coordinate": np.empty((0, 3), dtype=np.float32),
            "target_mass": np.empty(0, dtype=np.float32),
            "target_coordinate": np.empty((0, 3), dtype=np.float32),
            "target_present": np.empty(0, dtype=bool),
            "anchor": np.empty(0, dtype=np.float32),
        }
    anchor_by_family: dict[int, float] = {}
    for family, observations in gathered.items():
        anchor_by_family[family] = float(
            sum(
                point_weight[donor_rank]
                * states[int(donors[donor_rank])].mass[local]
                for donor_rank, local in observations
            )
        )
    selected_families = sorted(
        gathered,
        key=lambda family: (-anchor_by_family[family], family),
    )[: config.candidates]
    anchor = np.asarray(
        [anchor_by_family[family] for family in selected_families], dtype=np.float64
    )
    anchor /= max(float(np.sum(anchor)), TINY)
    anchor_order = np.argsort(-anchor, kind="stable")
    anchor_rank = np.empty(len(anchor), dtype=np.float64)
    anchor_rank[anchor_order] = np.arange(len(anchor), dtype=np.float64)
    anchor_rank /= max(len(anchor) - 1, 1)
    anchor_entropy = float(
        -np.sum(anchor * np.log(np.maximum(anchor, TINY)))
        / max(math.log(max(len(anchor), 2)), TINY)
    )
    sorted_anchor = np.sort(anchor)[::-1]
    anchor_top = float(sorted_anchor[0])
    anchor_margin = float(
        sorted_anchor[0] - (sorted_anchor[1] if len(sorted_anchor) > 1 else 0.0)
    )

    names, _ = feature_schema()
    features = np.zeros((len(selected_families), len(names)), dtype=np.float32)
    predicted_coordinate = np.zeros((len(selected_families), 3), dtype=np.float32)
    target_mass = np.zeros(len(selected_families), dtype=np.float32)
    target_coordinate = np.zeros((len(selected_families), 3), dtype=np.float32)
    target_present = np.zeros(len(selected_families), dtype=bool)
    target_lookup = {
        int(family): index for index, family in enumerate(states[point].family)
    }
    query_relative = (positions[point, :2] - center[:2]) / scale[:2]
    bs_delta = positions[point, :2] - bs_positions[int(sites[point]), :2]
    bs_distance = float(np.linalg.norm(bs_delta))
    bearing = math.atan2(float(bs_delta[1]), float(bs_delta[0]))
    direct_length = float(
        np.linalg.norm(positions[point] - bs_positions[int(sites[point])])
    )

    for row, family in enumerate(selected_families):
        observations = gathered[family]
        active_rank = np.asarray([item[0] for item in observations], dtype=np.int64)
        active_local = np.asarray([item[1] for item in observations], dtype=np.int64)
        active_donor = donors[active_rank].astype(np.int64)
        active_distance = distance[active_rank]
        active_weight = point_weight[active_rank]
        active_weight /= max(float(np.sum(active_weight)), TINY)
        mass_value = np.asarray(
            [states[d].mass[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        coordinate_value = np.stack(
            [states[d].coordinate[l] for d, l in zip(active_donor, active_local)]
        ).astype(np.float64)
        length_value = np.asarray(
            [states[d].path_length[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        stability_value = np.asarray(
            [states[d].stability[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        fine_diversity_value = np.asarray(
            [states[d].fine_diversity[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        fine_dominance_value = np.asarray(
            [states[d].fine_dominance[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        donor_rank_value = np.asarray(
            [states[d].rank_fraction[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        donor_entropy_value = np.asarray(
            [states[d].entropy[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        donor_top_value = np.asarray(
            [states[d].top_mass[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        donor_margin_value = np.asarray(
            [states[d].top_margin[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        donor_count_value = np.asarray(
            [states[d].family_count[l] for d, l in zip(active_donor, active_local)],
            dtype=np.float64,
        )
        nearest_observation = int(np.argmin(active_distance))
        descriptor_donor = int(active_donor[nearest_observation])
        descriptor_local = int(active_local[nearest_observation])
        descriptor = states[descriptor_donor]
        for axis, period in enumerate(PERIODS):
            predicted_coordinate[row, axis] = circular_mean(
                coordinate_value[:, axis], active_weight, float(period)
            )

        inactive = np.ones(config.neighbors, dtype=bool)
        inactive[np.unique(active_rank)] = False
        nearest_active = float(np.min(active_distance))
        nearest_inactive = (
            float(np.min(distance[inactive])) if np.any(inactive) else float(distance[-1] + 12.0)
        )
        sorted_active_distance = np.sort(np.unique(active_distance))
        nearest_gap = float(
            sorted_active_distance[1] - sorted_active_distance[0]
            if len(sorted_active_distance) > 1
            else 12.0
        )
        mass_mean, mass_std = weighted_statistics(mass_value, active_weight)
        coordinate_std = np.asarray(
            [
                math.sqrt(
                    max(
                        float(
                            np.sum(
                                active_weight
                                * np.square(
                                    circular_delta(
                                        coordinate_value[:, axis],
                                        predicted_coordinate[row, axis],
                                        float(PERIODS[axis]),
                                    )
                                )
                            )
                        ),
                        0.0,
                    )
                )
                for axis in range(3)
            ]
        )
        offsets = positions[active_donor, :2].astype(np.float64) - positions[
            point, :2
        ]
        log_mass_fit = local_linear(
            offsets,
            np.log(np.maximum(mass_value, 1e-8)),
            active_weight,
            config.ridge,
        )
        coordinate_fit: list[tuple[float, float, float, float, float]] = []
        for axis, period in enumerate(PERIODS):
            local_delta = circular_delta(
                coordinate_value[:, axis],
                predicted_coordinate[row, axis],
                float(period),
            )
            coordinate_fit.append(
                local_linear(offsets, local_delta, active_weight, config.ridge)
            )
        length_fit = local_linear(
            offsets, length_value, active_weight, config.ridge
        )
        coordinate_gradient = np.asarray(
            [[value[1], value[2]] for value in coordinate_fit], dtype=np.float64
        )
        jacobian_norm = float(np.linalg.norm(coordinate_gradient / PERIODS[:, None]))
        hv_det = float(np.linalg.det(coordinate_gradient[:2]))

        unary = [
            math.log(max(anchor[row], 1e-8)),
            math.sqrt(max(anchor[row], 0.0)),
            anchor_rank[row],
            float(sites[point]),
            float(query_relative[0]),
            float(query_relative[1]),
            bs_distance / 250.0,
            math.sin(bearing),
            math.cos(bearing),
        ]
        for axis, period in enumerate(PERIODS):
            phase = 2.0 * np.pi * predicted_coordinate[row, axis] / period
            unary.extend((math.sin(phase), math.cos(phase)))
        unary.extend(
            [
                float(descriptor.depth[descriptor_local] / 16.0),
                *descriptor.interaction_histogram[descriptor_local].astype(float),
                float(descriptor.plane_diversity[descriptor_local]),
                float(np.sum(active_weight * length_value) / 500.0),
                float(
                    (np.sum(active_weight * length_value) - direct_length) / 300.0
                ),
            ]
        )
        boundary = [
            len(np.unique(active_rank)) / config.neighbors,
            float(np.sum(point_weight[np.unique(active_rank)])),
            nearest_active / 50.0,
            nearest_inactive / 50.0,
            (nearest_inactive - nearest_active) / 50.0,
            *[
                float(np.sum(np.unique(active_distance) <= radius) / config.neighbors)
                for radius in (3.0, 6.0, 12.0, 24.0)
            ],
            float(np.sum(active_weight * active_distance) / 50.0),
            float(
                math.sqrt(
                    np.sum(
                        active_weight
                        * np.square(
                            active_distance - np.sum(active_weight * active_distance)
                        )
                    )
                )
                / 50.0
            ),
            float(np.max(active_distance) / 50.0),
            nearest_gap / 50.0,
        ]
        stability_mean, stability_std = weighted_statistics(
            stability_value, active_weight
        )
        fine_diversity_mean, fine_diversity_std = weighted_statistics(
            np.log1p(fine_diversity_value), active_weight
        )
        fine_dominance_mean, fine_dominance_std = weighted_statistics(
            fine_dominance_value, active_weight
        )
        length_mean, length_std = weighted_statistics(length_value, active_weight)
        stability = [
            stability_mean,
            stability_std,
            float(np.min(stability_value)),
            float(np.max(stability_value)),
            fine_diversity_mean,
            fine_diversity_std,
            fine_dominance_mean,
            fine_dominance_std,
            mass_std / max(mass_mean, 1e-8),
            *(coordinate_std / PERIODS).tolist(),
            length_std / 300.0,
        ]
        jacobian = list(log_mass_fit)
        for axis, value in enumerate(coordinate_fit):
            jacobian.extend(
                [
                    value[0] / PERIODS[axis],
                    value[1] / PERIODS[axis],
                    value[2] / PERIODS[axis],
                    value[3] / PERIODS[axis],
                    value[4],
                ]
            )
        jacobian.extend(
            [
                length_fit[0] / 500.0,
                length_fit[1] / 100.0,
                length_fit[2] / 100.0,
                length_fit[3] / 300.0,
                length_fit[4],
                jacobian_norm,
                hv_det / (PERIODS[0] * PERIODS[1]),
            ]
        )
        competition = []
        for value in (
            donor_rank_value,
            donor_entropy_value,
            donor_top_value,
            donor_margin_value,
            donor_count_value / 128.0,
        ):
            competition.extend(weighted_statistics(value, active_weight))
        competition.extend(
            [anchor_entropy, anchor_top, anchor_margin, 0.0, 0.0, 0.0, 0.0]
        )
        features[row] = np.asarray(
            unary + boundary + stability + jacobian + competition,
            dtype=np.float32,
        )
        target_index = target_lookup.get(family)
        if target_index is not None:
            target_present[row] = True
            target_mass[row] = states[point].mass[target_index]
            target_coordinate[row] = states[point].coordinate[target_index]
        else:
            target_coordinate[row] = predicted_coordinate[row]

    # Candidate competition in predicted H/V/delay space is defined after all rows exist.
    _, blocks = feature_schema()
    competition_start = blocks["competition"][0]
    for row in range(len(selected_families)):
        delta = circular_delta(
            predicted_coordinate,
            predicted_coordinate[row],
            PERIODS,
        )
        cost = np.sum(np.square(delta / np.asarray((1.5, 1.0, 2.0))), axis=1)
        other = np.arange(len(cost)) != row
        tight = other & (cost <= 0.25)
        medium = other & (cost <= 1.0)
        features[row, competition_start + 13 : competition_start + 17] = (
            float(np.sum(tight)),
            float(np.sum(medium)),
            float(np.sum(anchor[tight])),
            float(np.sum(anchor[medium])),
        )
    return {
        "family": np.asarray(selected_families, dtype=np.uint64),
        "features": features,
        "predicted_coordinate": predicted_coordinate,
        "target_mass": target_mass,
        "target_coordinate": target_coordinate,
        "target_present": target_present,
        "anchor": anchor.astype(np.float32),
    }


def build_dataset(config: Config, smoke: bool) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    setup, positions, _, sites, _ = load_positions()
    fold_id = np.load(FOLD_CACHE).astype(np.int8)
    fit_rows = np.flatnonzero((fold_id != OUTER) & (fold_id != INNER))
    plane_id, plane_report = canonical_face_planes(config.plane_round_decimals)
    state = json.loads(SIONNA_STATE.read_text(encoding="utf-8"))
    states, state_report = point_states(
        plane_id,
        positions,
        sites,
        setup.bs_positions,
        state["configuration"],
    )
    donor_ids, donor_distance = query_neighbors(
        positions, sites, fit_rows, config.neighbors
    )
    if smoke:
        selected = np.concatenate(
            [np.flatnonzero(fold_id == fold)[:24] for fold in range(5)]
        )
    else:
        selected = np.arange(len(positions), dtype=np.int64)
    center = positions.mean(axis=0)
    scale = np.maximum(positions.std(axis=0), 1.0)
    rows: list[dict[str, np.ndarray]] = []
    point_offsets = [0]
    truth_family: list[np.ndarray] = []
    truth_mass: list[np.ndarray] = []
    truth_coordinate: list[np.ndarray] = []
    for index, point in enumerate(selected):
        local = candidate_features(
            int(point),
            states,
            donor_ids,
            donor_distance,
            positions,
            sites,
            setup.bs_positions,
            center,
            scale,
            config,
        )
        rows.append(local)
        point_offsets.append(point_offsets[-1] + len(local["family"]))
        truth_family.append(states[int(point)].family)
        truth_mass.append(states[int(point)].mass)
        truth_coordinate.append(states[int(point)].coordinate)
        if (index + 1) % 200 == 0 or index + 1 == len(selected):
            print(f"trajectory dataset {index + 1}/{len(selected)}", flush=True)
    names, blocks = feature_schema()
    truth_offsets = np.cumsum(
        np.asarray([0] + [len(value) for value in truth_family], dtype=np.int64)
    )
    dataset = {
        "query_ids": selected.astype(np.int32),
        "point_offsets": np.asarray(point_offsets, dtype=np.int64),
        "family": np.concatenate([row["family"] for row in rows]),
        "features": np.concatenate([row["features"] for row in rows]),
        "predicted_coordinate": np.concatenate(
            [row["predicted_coordinate"] for row in rows]
        ),
        "target_mass": np.concatenate([row["target_mass"] for row in rows]),
        "target_coordinate": np.concatenate(
            [row["target_coordinate"] for row in rows]
        ),
        "target_present": np.concatenate(
            [row["target_present"] for row in rows]
        ),
        "anchor": np.concatenate([row["anchor"] for row in rows]),
        "truth_offsets": truth_offsets,
        "truth_family": np.concatenate(truth_family),
        "truth_mass": np.concatenate(truth_mass),
        "truth_coordinate": np.concatenate(truth_coordinate),
        "fold_id": fold_id[selected],
        "site": sites[selected].astype(np.int8),
        "feature_names": np.asarray(names),
        "block_json": np.asarray(json.dumps(blocks)),
    }
    if not np.all(np.isfinite(dataset["features"])):
        raise FloatingPointError("non-finite trajectory features")
    target_total = np.asarray(
        [
            np.sum(
                dataset["target_mass"][
                    dataset["point_offsets"][row] : dataset["point_offsets"][row + 1]
                ]
            )
            for row in range(len(selected))
        ]
    )
    report = {
        "plane_identity": plane_report,
        "state": state_report,
        "query_count": int(len(selected)),
        "candidate_rows": int(len(dataset["target_mass"])),
        "candidate_count": {
            "mean": float(np.mean(np.diff(dataset["point_offsets"]))),
            "min": int(np.min(np.diff(dataset["point_offsets"]))),
            "max": int(np.max(np.diff(dataset["point_offsets"]))),
        },
        "target_mass_coverage": {
            "mean": float(np.mean(target_total)),
            "median": float(np.median(target_total)),
            "p10": float(np.quantile(target_total, 0.10)),
        },
    }
    return dataset, report


def save_dataset(dataset: dict[str, np.ndarray]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATASET_CACHE, **dataset)


def load_dataset() -> dict[str, np.ndarray]:
    cached = np.load(DATASET_CACHE, allow_pickle=False)
    return {name: cached[name] for name in cached.files}


def point_rows(dataset: dict[str, np.ndarray], point_mask: np.ndarray) -> np.ndarray:
    output = np.zeros(len(dataset["target_mass"]), dtype=bool)
    selected = np.flatnonzero(point_mask)
    for point in selected:
        output[
            dataset["point_offsets"][point] : dataset["point_offsets"][point + 1]
        ] = True
    return output


def train_regressor(
    features: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    rows: np.ndarray,
    columns: list[int],
    config: Config,
    seed_offset: int,
) -> HistGradientBoostingRegressor:
    active = target[rows] > ACTIVE_MASS
    positive_fraction = float(np.mean(active))
    weight = np.where(
        active,
        0.5 / max(positive_fraction, 1e-6),
        0.5 / max(1.0 - positive_fraction, 1e-6),
    )
    weight *= 0.5 + 0.5 * np.sqrt(
        np.maximum(target[rows], 0.0)
        / max(float(np.mean(target[rows][active])) if np.any(active) else 1.0, 1e-8)
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.maximum_iterations,
        max_leaf_nodes=config.maximum_leaf_nodes,
        min_samples_leaf=config.minimum_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.random_state + seed_offset,
    )
    model.fit(
        features[rows][:, columns],
        np.sqrt(np.maximum(target[rows], 0.0))
        - np.sqrt(np.maximum(anchor[rows], 0.0)),
        sample_weight=weight,
    )
    return model


def apply_residual_prediction(
    residual: np.ndarray,
    anchor: np.ndarray,
    offsets: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = np.maximum(
        np.sqrt(np.maximum(np.asarray(anchor, dtype=np.float64), 0.0))
        + alpha * np.asarray(residual, dtype=np.float64),
        0.0,
    ) ** 2
    for point in range(len(offsets) - 1):
        start, end = int(offsets[point]), int(offsets[point + 1])
        total = float(np.sum(output[start:end]))
        if total <= TINY:
            output[start:end] = anchor[start:end]
            total = float(np.sum(output[start:end]))
        output[start:end] /= max(total, TINY)
    return output.astype(np.float32)


def add_bilinear(
    output: np.ndarray,
    coordinate: np.ndarray,
    mass: np.ndarray,
    first_period: int,
) -> None:
    first = np.remainder(coordinate[:, 0], first_period)
    delay = np.remainder(coordinate[:, 1], int(PERIODS[2]))
    first0 = np.floor(first).astype(np.int64)
    delay0 = np.floor(delay).astype(np.int64)
    first_fraction = first - first0
    delay_fraction = delay - delay0
    for first_step, first_weight in (
        (0, 1.0 - first_fraction),
        (1, first_fraction),
    ):
        for delay_step, delay_weight in (
            (0, 1.0 - delay_fraction),
            (1, delay_fraction),
        ):
            np.add.at(
                output,
                (
                    (first0 + first_step) % first_period,
                    (delay0 + delay_step) % int(PERIODS[2]),
                ),
                mass * first_weight * delay_weight,
            )


def spectrum_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = math.sqrt(float(np.sum(left**2) * np.sum(right**2)))
    return float(np.sum(left * right) / (denominator if denominator > 0.0 else 1e-12))


def evaluate_prediction(
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    point_mask: np.ndarray,
    coordinate_mode: str,
) -> dict[str, object]:
    offsets = dataset["point_offsets"]
    truth_offsets = dataset["truth_offsets"]
    affinities: list[float] = []
    top16: list[float] = []
    spectral: list[float] = []
    site_affinity: dict[int, list[float]] = {0: [], 1: []}
    selected_points = np.flatnonzero(point_mask)
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for point in selected_points:
        start, end = int(offsets[point]), int(offsets[point + 1])
        target = dataset["target_mass"][start:end].astype(np.float64)
        predicted = prediction[start:end].astype(np.float64)
        affinity = float(np.sum(np.sqrt(np.maximum(target * predicted, 0.0))))
        affinities.append(affinity)
        site_affinity[int(dataset["site"][point])].append(affinity)
        selected = np.argsort(-predicted, kind="stable")[:16]
        top16.append(float(np.sum(target[selected])))
        labels.append(target > ACTIVE_MASS)
        scores.append(predicted)

        truth_start, truth_end = int(truth_offsets[point]), int(
            truth_offsets[point + 1]
        )
        true_mass = dataset["truth_mass"][truth_start:truth_end].astype(np.float64)
        true_coordinate = dataset["truth_coordinate"][truth_start:truth_end].astype(
            np.float64
        )
        pred_coordinate = (
            dataset["target_coordinate"][start:end]
            if coordinate_mode == "mass_isolation"
            else dataset["predicted_coordinate"][start:end]
        ).astype(np.float64)
        truth_hd = np.zeros((16, 192), dtype=np.float64)
        truth_vd = np.zeros((8, 192), dtype=np.float64)
        predicted_hd = np.zeros_like(truth_hd)
        predicted_vd = np.zeros_like(truth_vd)
        if len(true_mass):
            add_bilinear(truth_hd, true_coordinate[:, (0, 2)], true_mass, 16)
            add_bilinear(truth_vd, true_coordinate[:, (1, 2)], true_mass, 8)
        if len(predicted):
            add_bilinear(predicted_hd, pred_coordinate[:, (0, 2)], predicted, 16)
            add_bilinear(predicted_vd, pred_coordinate[:, (1, 2)], predicted, 8)
        spectral.append(
            0.5
            * (
                spectrum_cosine(predicted_hd, truth_hd)
                + spectrum_cosine(predicted_vd, truth_vd)
            )
        )
    flat_label = np.concatenate(labels) if labels else np.empty(0, dtype=bool)
    flat_score = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    return {
        "point_count": int(len(selected_points)),
        "candidate_count": int(len(flat_label)),
        "existence_ap": float(average_precision_score(flat_label, flat_score))
        if np.any(flat_label)
        else 0.0,
        "mass_affinity": float(np.mean(affinities)) if affinities else 0.0,
        "top16_mass_recall": float(np.mean(top16)) if top16 else 0.0,
        "angle_delay_cosine": float(np.mean(spectral)) if spectral else 0.0,
        "selection_score": float(
            0.5 * np.mean(affinities) + 0.5 * np.mean(spectral)
        )
        if affinities
        else 0.0,
        "site_mass_affinity": {
            str(site): float(np.mean(values)) if values else 0.0
            for site, values in site_affinity.items()
        },
    }


def shuffled_features(
    features: np.ndarray,
    dataset: dict[str, np.ndarray],
    columns: list[int],
    mode: str,
    seed: int,
) -> np.ndarray:
    output = np.array(features, copy=True)
    rng = np.random.default_rng(seed)
    fold_id = dataset["fold_id"]
    offsets = dataset["point_offsets"]
    if mode == "candidate":
        for point in range(len(offsets) - 1):
            start, end = int(offsets[point]), int(offsets[point + 1])
            if end - start > 1:
                permutation = rng.permutation(np.arange(start, end))
                output[start:end, columns] = features[permutation][:, columns]
        return output
    if mode != "trajectory":
        raise ValueError(mode)
    point_index = np.empty(len(features), dtype=np.int32)
    for point in range(len(offsets) - 1):
        point_index[int(offsets[point]) : int(offsets[point + 1])] = point
    unary_names, blocks = feature_schema()
    del unary_names
    depth_column = blocks["unary"][15]
    anchor_column = blocks["unary"][0]
    depth_bin = np.clip(np.floor(features[:, depth_column] * 4.0), 0, 3).astype(
        np.int8
    )
    anchor_bin = np.clip(
        np.floor((features[:, anchor_column] + 18.0) / 3.0), 0, 5
    ).astype(np.int8)
    for split in ((2, 3, 4), (INNER,), (OUTER,)):
        split_points = np.isin(fold_id, split)
        rows = np.flatnonzero(split_points[point_index])
        local_site = dataset["site"][point_index[rows]]
        for site_value in (0, 1):
            for depth_value in range(4):
                for anchor_value in range(6):
                    local = rows[
                        (local_site == site_value)
                        & (depth_bin[rows] == depth_value)
                        & (anchor_bin[rows] == anchor_value)
                    ]
                    if len(local) > 1:
                        permutation = rng.permutation(local)
                        output[np.ix_(local, columns)] = features[
                            np.ix_(permutation, columns)
                        ]
    return output


def run_models(
    dataset: dict[str, np.ndarray], config: Config
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    names = dataset["feature_names"].astype(str).tolist()
    blocks = json.loads(str(dataset["block_json"]))
    features = dataset["features"].astype(np.float32)
    target = dataset["target_mass"].astype(np.float32)
    anchor = dataset["anchor"].astype(np.float32)
    fit_points = np.isin(dataset["fold_id"], (2, 3, 4))
    inner_points = dataset["fold_id"] == INNER
    outer_points = dataset["fold_id"] == OUTER
    fit_rows = point_rows(dataset, fit_points)

    anchor_prediction = apply_residual_prediction(
        np.zeros(len(anchor), dtype=np.float32),
        anchor,
        dataset["point_offsets"],
        0.0,
    )
    anchor_inner = evaluate_prediction(
        dataset, anchor_prediction, inner_points, "mass_isolation"
    )
    anchor_outer = evaluate_prediction(
        dataset, anchor_prediction, outer_points, "mass_isolation"
    )
    variants: dict[str, list[int]] = {"unary": blocks["unary"]}
    for block in ("boundary", "stability", "jacobian", "competition"):
        variants[block] = blocks["unary"] + blocks[block]
    variants["all"] = blocks["unary"] + sum(
        [blocks[name] for name in ("boundary", "stability", "jacobian", "competition")],
        [],
    )
    predictions: dict[str, np.ndarray] = {"anchor": anchor_prediction}
    inner_results: dict[str, object] = {"anchor": anchor_inner}
    selected_alphas: dict[str, float] = {"anchor": 0.0}
    alpha_grid = (0.0, 0.25, 0.50, 0.75, 1.0)
    for index, (variant, columns) in enumerate(variants.items()):
        model = train_regressor(
            features,
            target,
            anchor,
            fit_rows,
            columns,
            config,
            seed_offset=100 * index,
        )
        residual = model.predict(features[:, columns])
        alpha_metrics: list[tuple[float, dict[str, object], np.ndarray]] = []
        for alpha in alpha_grid:
            local_prediction = apply_residual_prediction(
                residual, anchor, dataset["point_offsets"], alpha
            )
            local_metrics = evaluate_prediction(
                dataset, local_prediction, inner_points, "mass_isolation"
            )
            alpha_metrics.append((alpha, local_metrics, local_prediction))
        selected_alpha, selected_metrics, prediction = max(
            alpha_metrics,
            key=lambda row: (row[1]["selection_score"], -row[0]),
        )
        predictions[variant] = prediction
        inner_results[variant] = {
            **selected_metrics,
            "selected_alpha": float(selected_alpha),
            "alpha_selection": {
                str(alpha): metrics["selection_score"]
                for alpha, metrics, _ in alpha_metrics
            },
        }
        selected_alphas[variant] = float(selected_alpha)
        print(
            f"variant={variant} inner={inner_results[variant]['selection_score']:.6f}",
            flush=True,
        )
    selected = max(
        (name for name in variants if name != "unary"),
        key=lambda name: inner_results[name]["selection_score"],
    )
    selected_columns = variants[selected]
    dynamic_columns = [column for column in selected_columns if column not in blocks["unary"]]
    controls: dict[str, object] = {}
    for control_index, mode in enumerate(("candidate", "trajectory")):
        controlled = shuffled_features(
            features,
            dataset,
            dynamic_columns,
            mode,
            config.random_state + 10_000 + control_index,
        )
        model = train_regressor(
            controlled,
            target,
            anchor,
            fit_rows,
            selected_columns,
            config,
            seed_offset=10_000 + control_index,
        )
        residual = model.predict(controlled[:, selected_columns])
        control_alpha_rows: list[tuple[float, dict[str, object], np.ndarray]] = []
        for alpha in alpha_grid:
            local_prediction = apply_residual_prediction(
                residual, anchor, dataset["point_offsets"], alpha
            )
            local_metrics = evaluate_prediction(
                dataset, local_prediction, inner_points, "mass_isolation"
            )
            control_alpha_rows.append((alpha, local_metrics, local_prediction))
        control_alpha, control_inner, prediction = max(
            control_alpha_rows,
            key=lambda row: (row[1]["selection_score"], -row[0]),
        )
        predictions[f"{mode}_shuffle"] = prediction
        controls[mode] = {
            "selected_alpha": float(control_alpha),
            "inner": control_inner,
            "outer": evaluate_prediction(
                dataset, prediction, outer_points, "mass_isolation"
            ),
        }
    outer_results = {
        name: evaluate_prediction(dataset, prediction, outer_points, "mass_isolation")
        for name, prediction in predictions.items()
        if name in ("anchor", "unary", selected)
    }
    rollout = evaluate_prediction(
        dataset, predictions[selected], outer_points, "rollout"
    )
    reference = max(
        ("anchor", "unary"),
        key=lambda name: inner_results[name]["selection_score"],
    )
    selected_outer = outer_results[selected]
    reference_outer = outer_results[reference]
    gate = {
        "minimum_mass_affinity_gain": 0.015,
        "minimum_angle_delay_gain": 0.010,
        "minimum_control_mass_margin": 0.010,
        "minimum_each_site_mass_gain": 0.005,
        "mass_affinity_gain": float(
            selected_outer["mass_affinity"] - reference_outer["mass_affinity"]
        ),
        "angle_delay_gain": float(
            selected_outer["angle_delay_cosine"]
            - reference_outer["angle_delay_cosine"]
        ),
        "candidate_shuffle_mass_margin": float(
            selected_outer["mass_affinity"]
            - controls["candidate"]["outer"]["mass_affinity"]
        ),
        "trajectory_shuffle_mass_margin": float(
            selected_outer["mass_affinity"]
            - controls["trajectory"]["outer"]["mass_affinity"]
        ),
        "site_mass_gains": {
            site: float(
                selected_outer["site_mass_affinity"][site]
                - reference_outer["site_mass_affinity"][site]
            )
            for site in ("0", "1")
        },
    }
    gate["passed"] = bool(
        gate["mass_affinity_gain"] >= gate["minimum_mass_affinity_gain"]
        and gate["angle_delay_gain"] >= gate["minimum_angle_delay_gain"]
        and gate["candidate_shuffle_mass_margin"]
        >= gate["minimum_control_mass_margin"]
        and gate["trajectory_shuffle_mass_margin"]
        >= gate["minimum_control_mass_margin"]
        and min(gate["site_mass_gains"].values())
        >= gate["minimum_each_site_mass_gain"]
    )
    report = {
        "feature_count": int(len(names)),
        "feature_blocks": {name: [names[index] for index in value] for name, value in blocks.items()},
        "inner_selection": inner_results,
        "selected_variable_family": selected,
        "selected_alpha": selected_alphas[selected],
        "locked_reference": reference,
        "outer_results": outer_results,
        "controls": controls,
        "selected_outer_rollout_coordinate": rollout,
        "gate": gate,
    }
    return report, predictions


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(
        candidates=24 if args.smoke else 128,
        neighbors=12 if args.smoke else 32,
        maximum_iterations=4 if args.smoke else args.max_iterations,
        minimum_samples_leaf=8 if args.smoke else 36,
    )
    required = (PATH_CACHE, SIONNA_STATE, FOLD_CACHE, PATHS.round2 / "Round2_Map.ply")
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if DATASET_CACHE.is_file() and not args.force_dataset and not args.smoke:
        dataset = load_dataset()
        dataset_report: dict[str, object] = {"status": "reused"}
    else:
        dataset, dataset_report = build_dataset(config, args.smoke)
        if not args.smoke:
            save_dataset(dataset)
    model_report, predictions = run_models(dataset, config)
    report = {
        "experiment": VERSION,
        "status": "smoke_complete" if args.smoke else "information_audit_complete",
        "primary_hypothesis": (
            "Fit-only surface-family trajectories expose boundary, identity stability, "
            "path-geometry Jacobian, or same-point competition variables that predict "
            "held path mass beyond static candidate interpolation."
        ),
        "sionna_role": "offline variable discovery on training coordinates only",
        "test_sionna_loaded": False,
        "official_h_loaded": False,
        "candidate_generated": False,
        "split": {"outer": OUTER, "inner": INNER, "fit_folds": [2, 3, 4]},
        "config": asdict(config),
        "dataset": dataset_report,
        **model_report,
        "seconds": float(time.time() - started),
    }
    if not args.smoke:
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            METRICS_CACHE,
            query_ids=dataset["query_ids"],
            fold_id=dataset["fold_id"],
            site=dataset["site"],
            point_offsets=dataset["point_offsets"],
            target_mass=dataset["target_mass"],
            anchor_prediction=predictions["anchor"],
            unary_prediction=predictions["unary"],
            selected_prediction=predictions[model_report["selected_variable_family"]],
            candidate_shuffle_prediction=predictions["candidate_shuffle"],
            trajectory_shuffle_prediction=predictions["trajectory_shuffle"],
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
