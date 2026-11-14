#!/usr/bin/env python3
"""Train the E205 Clean ordered-map-path mass and port calibrator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e124_one_to_one_path_identity_ceiling as e124
import analyze_e146_map_propagation_graph_coverage as e146
import analyze_e147_height_representation_contrast as e147
import build_v7_complex_carrier_bank as carrier_bank
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from carrier_calibration import estimate_carrier
from data import load_positions
from official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics_torch,
)
from project_paths import PATHS
from train_full_path_pas_slice_gate import project_joint
import train_e150_v7_pathwise_carrier_operator as e150


VERSION = "Round2-E205-ordered-map-path-calibrator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e205_ordered_map_path_calibrator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e205_ordered_map_path_calibrator"
COMMON_DIR = ARTIFACT_DIR / "common"
COMMON_STATE = COMMON_DIR / "state.json"
TARGET_PAS = COMMON_DIR / "official_pas_target_f32.npy"
TARGET_PDP = COMMON_DIR / "official_pdp_target_f32.npy"
ORDERED_POINTS = COMMON_DIR / "ordered_points_f32.npy"
ORDERED_Z_MIN = COMMON_DIR / "ordered_z_min_f32.npy"
ORDERED_Z_MAX = COMMON_DIR / "ordered_z_max_f32.npy"
ORDERED_TYPE = COMMON_DIR / "ordered_type_i8.npy"
ORDERED_MASK = COMMON_DIR / "ordered_mask.npy"
ORDERED_STRENGTH = COMMON_DIR / "ordered_strength_f32.npy"
COMMON_CANDIDATE_FEATURE = COMMON_DIR / "candidate_feature_f16.npy"
COMMON_POINT_FEATURE = COMMON_DIR / "point_feature_f32.npy"
COMMON_SEQUENCE_FEATURE = COMMON_DIR / "sequence_feature_f16.npy"
FULL_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
FULL_REPORT = ANALYSIS_DIR / "fixed500_report.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
BASE_V7 = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "canonical_v7_complete.npy"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FULL_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PORTS = MP * N_UE
CANDIDATES = 256
SEQUENCE_STEPS = 3
CARRIERS = 17
PERIODS = np.asarray((MH, MV, S_SC), dtype=np.float32)
TINY = 1e-30
EXPECTED_V7_SCORE = 0.644456345600551
FORCE_BASE_GLOBAL_IDS: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Config:
    validation_fold_offset: int = 1
    carrier_neighbors: int = 32
    carrier_batch_size: int = 64
    train_batch_size: int = 64
    score_batch_size: int = 16
    hidden: int = 96
    heads: int = 4
    sequence_layers: int = 1
    set_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.08
    epochs: int = 48
    validation_interval: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    score_limit: float = 4.0
    port_limit: float = 2.0
    spectral_weight: float = 1.0
    mass_weight: float = 0.75
    port_weight: float = 0.20
    existence_weight: float = 0.05
    residual_weight: float = 0.002
    pas_beta: float = 0.20
    pdp_beta: float = 0.20
    projection_cycles: int = 4
    projection_order: str = "pas_pdp"
    required_score: float = 0.677521
    maximum_component_drop: float = 0.001
    required_positive_folds: int = 4
    minimum_fold_delta: float = -0.001
    seed: int = 20_260_814_205


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "fold", "aggregate"), default="prepare"
    )
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--carrier-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_array(directory: Path, name: str, value: np.ndarray) -> None:
    np.save(directory / f"{name}.npy", value)


def fold_dir(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}"


def fold_cache_dir(outer: int) -> Path:
    return fold_dir(outer) / "cache"


def fold_report_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}.json"


def fold_channel_path(outer: int) -> Path:
    return fold_dir(outer) / "held_channel.npy"


def fold_endpoint_path(outer: int) -> Path:
    return fold_dir(outer) / "held_endpoint.npz"


def fold_last_checkpoint_path(outer: int) -> Path:
    return fold_dir(outer) / "last_checkpoint.pt"


def fold_best_checkpoint_path(outer: int) -> Path:
    return fold_dir(outer) / "best_checkpoint.pt"


def required_inputs() -> tuple[Path, ...]:
    return (
        TRAIN_CHANNEL,
        BASE_V7,
        OOF_META,
        FULL_FOLD,
        e114.COMPONENTS,
        e146.CANDIDATE_CACHE,
        e147.METADATA_CACHE,
        e146.GRAPH_CACHE,
        e146.LOS_LABELS,
        e146.MAP_PATH,
    )


def canonical_full_fold() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical = metadata["fold_id"].astype(np.int8)
    fold = np.load(FULL_FOLD, allow_pickle=False).astype(np.int8)
    fold = np.array(fold, copy=True)
    fold[query_ids] = canonical
    if not np.array_equal(fold[query_ids], canonical):
        raise RuntimeError("canonical fold override failed")
    return fold, query_ids, canonical


def gpu_record(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "cuda": False}
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "cuda": True,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def route_points_batch(
    routes: np.ndarray,
    ue: np.ndarray,
    bs: np.ndarray,
    graph: e146.GraphData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    route_array = np.asarray(routes, dtype=np.int64)
    if route_array.ndim != 2 or route_array.shape[1] not in (1, 2, 3):
        raise ValueError(f"invalid route array shape {route_array.shape}")
    count, depth = route_array.shape
    node_xy = graph.node_xy[route_array]
    bs_xy = np.broadcast_to(bs[None, None, :2], (count, 1, 2))
    ue_xy = np.broadcast_to(ue[None, None, :2], (count, 1, 2))
    xy = np.concatenate((bs_xy, node_xy, ue_xy), axis=1)
    horizontal = np.linalg.norm(np.diff(xy, axis=1), axis=2)
    total_horizontal = np.maximum(horizontal.sum(axis=1), 1e-9)
    cumulative = np.cumsum(horizontal, axis=1)[:, :depth]
    ideal_z = bs[2] + (ue[2] - bs[2]) * cumulative / total_horizontal[:, None]
    low = graph.node_z_min[route_array] + 0.20
    high = graph.node_z_max[route_array] - 0.20
    z = np.clip(ideal_z, low, high)
    z = np.minimum(
        np.maximum(z, graph.node_z_min[route_array]), graph.node_z_max[route_array]
    )
    points = np.concatenate((node_xy, z[..., None]), axis=2)
    full = np.concatenate(
        (
            np.broadcast_to(bs[None, None], (count, 1, 3)),
            points,
            np.broadcast_to(ue[None, None], (count, 1, 3)),
        ),
        axis=1,
    )
    segment = np.diff(full, axis=1)
    length = np.linalg.norm(segment, axis=2).sum(axis=1)
    unit_xy = np.diff(xy, axis=1) / np.maximum(horizontal[..., None], 1e-9)
    turn = np.zeros(count, dtype=np.float64)
    if unit_xy.shape[1] > 1:
        turn = np.sum(
            1.0
            - np.clip(
                np.sum(unit_xy[:, :-1] * unit_xy[:, 1:], axis=2), -1.0, 1.0
            ),
            axis=1,
        )
    return points, length, turn


def build_route_reverse(
    route_bank: list[list[list[tuple[float, tuple[int, ...]]]]]
) -> list[list[dict[int, np.ndarray]]]:
    output: list[list[dict[int, np.ndarray]]] = []
    for site_bank in route_bank:
        site_output: list[dict[int, np.ndarray]] = [dict() for _ in range(4)]
        for depth in range(1, 4):
            grouped: dict[int, set[tuple[int, ...]]] = {}
            for states in site_bank[depth]:
                for _, route in states:
                    grouped.setdefault(int(route[0]), set()).add(tuple(route))
            site_output[depth] = {
                key: np.asarray(sorted(value), dtype=np.int32)
                for key, value in grouped.items()
            }
        output.append(site_output)
    return output


def common_config_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "candidate_config": e146.candidate_config_payload(e146.Config()),
        "graph_config": e146.graph_config_payload(e146.Config()),
        "candidate_budget": CANDIDATES,
        "sequence_steps": SEQUENCE_STEPS,
        "official_target": "cosine-sufficient PAS/PDP computed directly from Train H",
    }


def build_ordered_geometry(force: bool) -> dict[str, Any]:
    if COMMON_STATE.is_file() and not force:
        state = json.loads(COMMON_STATE.read_text(encoding="utf-8"))
        if state.get("config") == common_config_payload() and all(
            path.is_file()
            for path in (
                ORDERED_POINTS,
                ORDERED_Z_MIN,
                ORDERED_Z_MAX,
                ORDERED_TYPE,
                ORDERED_MASK,
                ORDERED_STRENGTH,
                COMMON_CANDIDATE_FEATURE,
                COMMON_POINT_FEATURE,
                COMMON_SEQUENCE_FEATURE,
            )
        ):
            return {"reused": True, **state["geometry"]}

    started = time.time()
    setup, positions, _, site, _ = load_positions()
    candidate_file = np.load(e146.CANDIDATE_CACHE, allow_pickle=False)
    candidates = {
        key: candidate_file[key]
        for key in candidate_file.files
        if key != "config_json"
    }
    if candidates["mask"].shape != (len(positions), CANDIDATES):
        raise ValueError(f"unexpected E146 candidate shape {candidates['mask'].shape}")
    mesh = e146.load_mesh()
    graph, graph_record = e146.build_graph(
        mesh, setup.bs_positions, e146.Config(), force=False
    )
    route_bank = e146.build_route_bank(graph, setup.bs_positions, e146.Config())
    base_config = e146.Config()
    wall = e146.map_phase.build_wall_bank(e146.map_phase.Config())
    reflection_sources = {
        site_value: e146.map_phase.image_sources(
            wall, setup.bs_positions[site_value]
        )
        for site_value in range(len(setup.bs_positions))
    }
    corner_count = min(base_config.query_corner_neighbors, len(graph.node_xy))
    _, nearest_corner = cKDTree(graph.node_xy).query(
        positions[:, :2], k=corner_count, workers=-1
    )
    if corner_count == 1:
        nearest_corner = nearest_corner[:, None]

    shape = (len(positions), CANDIDATES, SEQUENCE_STEPS)
    points = np.zeros((*shape, 3), dtype=np.float32)
    z_min = np.zeros(shape, dtype=np.float32)
    z_max = np.zeros(shape, dtype=np.float32)
    interaction_type = np.zeros(shape, dtype=np.int8)
    sequence_mask = np.zeros(shape, dtype=bool)
    strength = np.zeros(shape, dtype=np.float32)
    path_mask = candidates["mask"].astype(bool)
    path_type = candidates["path_type"].astype(np.int8)
    depth_array = candidates["depth"].astype(np.int8)
    maximum_length_error = 0.0
    maximum_score_error = 0.0
    maximum_first_error = 0.0
    missing_routes = 0

    for row, ue in enumerate(positions):
        site_value = int(site[row])
        bs = setup.bs_positions[site_value]
        direct_length = float(np.linalg.norm(ue - bs))
        theta = candidates["theta"][row].astype(np.float64)
        phi = candidates["phi"][row].astype(np.float64)
        target_unit = np.column_stack(
            (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )
        )

        ground_slots = np.flatnonzero(
            path_mask[row] & (path_type[row] == e146.TYPE_GROUND)
        )
        for local, slot in enumerate(ground_slots):
            height = (0.0, -0.5)[local]
            direction, length = e146.ground_candidate(ue, bs, height)
            unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
            length_error = abs(length - float(candidates["path_length"][row, slot]))
            direction_error = float(np.linalg.norm(unit - target_unit[slot]))
            maximum_length_error = max(maximum_length_error, length_error)
            maximum_first_error = max(maximum_first_error, direction_error)
            if length_error > 5e-4 or direction_error > 5e-4:
                raise RuntimeError(
                    f"ground reconstruction mismatch row={row} slot={slot} "
                    f"length={length_error} direction={direction_error}"
                )
            points[row, slot, 0] = (bs + direction).astype(np.float32)
            z_min[row, slot, 0] = height
            z_max[row, slot, 0] = height
            interaction_type[row, slot, 0] = 1
            sequence_mask[row, slot, 0] = True

        reflection_slots = np.flatnonzero(
            path_mask[row] & (path_type[row] == e146.TYPE_REFLECTION)
        )
        if len(reflection_slots):
            reflected = e146.reflection_candidates(
                ue,
                bs,
                wall,
                reflection_sources[site_value],
                base_config,
            )
            if len(reflected) < len(reflection_slots):
                raise RuntimeError(
                    f"insufficient reflection reconstruction row={row}: "
                    f"{len(reflected)} < {len(reflection_slots)}"
                )
            reflected_score = np.asarray([value[0] for value in reflected])
            reflected_direction = np.stack([value[1] for value in reflected])
            reflected_length = np.asarray([value[2] for value in reflected])
            reflected_wall = np.asarray([value[3] for value in reflected], dtype=np.int64)
            reflected_unit = reflected_direction / np.maximum(
                np.linalg.norm(reflected_direction, axis=1, keepdims=True), 1e-12
            )
            direction_cost = np.linalg.norm(
                target_unit[reflection_slots, None] - reflected_unit[None], axis=2
            )
            cost = (
                1000.0
                * np.abs(
                    candidates["path_length"][row, reflection_slots, None]
                    - reflected_length[None]
                )
                + 100.0
                * np.abs(
                    candidates["score"][row, reflection_slots, None]
                    - reflected_score[None]
                )
                + 1000.0 * direction_cost
            )
            local_slot, local_reflection = linear_sum_assignment(cost)
            for slot_index, reflection_index in zip(
                local_slot, local_reflection, strict=True
            ):
                slot = int(reflection_slots[slot_index])
                wall_id = int(reflected_wall[reflection_index])
                length_error = abs(
                    float(reflected_length[reflection_index])
                    - float(candidates["path_length"][row, slot])
                )
                score_error = abs(
                    float(reflected_score[reflection_index])
                    - float(candidates["score"][row, slot])
                )
                direction_error = float(direction_cost[slot_index, reflection_index])
                maximum_length_error = max(maximum_length_error, length_error)
                maximum_score_error = max(maximum_score_error, score_error)
                maximum_first_error = max(maximum_first_error, direction_error)
                if (
                    length_error > 5e-4
                    or score_error > 5e-4
                    or direction_error > 5e-4
                ):
                    raise RuntimeError(
                        f"reflection reconstruction mismatch row={row} slot={slot} "
                        f"length={length_error} score={score_error} "
                        f"direction={direction_error}"
                    )
                points[row, slot, 0] = (
                    bs + reflected_direction[reflection_index]
                ).astype(np.float32)
                z_min[row, slot, 0] = float(wall.z_min[wall_id])
                z_max[row, slot, 0] = float(wall.z_max[wall_id])
                interaction_type[row, slot, 0] = 2
                sequence_mask[row, slot, 0] = True
                strength[row, slot, 0] = float(
                    np.log1p(max(float(wall.area[wall_id]), 0.0)) / 10.0
                )

        for depth in range(1, SEQUENCE_STEPS + 1):
            slots = np.flatnonzero(
                path_mask[row]
                & (path_type[row] >= e146.TYPE_CORNER_1)
                & (depth_array[row] == depth)
            )
            if len(slots) == 0:
                continue
            route_set: set[tuple[int, ...]] = set()
            for endpoint in nearest_corner[row]:
                for _, route in route_bank[site_value][depth][int(endpoint)]:
                    route_set.add(tuple(route))
            routes = np.asarray(sorted(route_set), dtype=np.int32)
            if len(routes) < len(slots):
                missing_routes += len(slots)
                continue
            route_points, route_length, route_turn = route_points_batch(
                routes, ue, bs, graph
            )
            route_score = (
                route_length
                - direct_length
                + base_config.interaction_penalty_m * depth
                + base_config.turn_penalty_m * route_turn
                - 0.02 * graph.node_strength[routes].sum(axis=1)
            )
            target_length = candidates["path_length"][row, slots].astype(np.float64)
            target_score = candidates["score"][row, slots].astype(np.float64)
            route_direction = route_points[:, 0] - bs[None]
            route_unit = route_direction / np.maximum(
                np.linalg.norm(route_direction, axis=1, keepdims=True), 1e-12
            )
            direction_error = np.linalg.norm(
                target_unit[slots, None] - route_unit[None], axis=2
            )
            cost = (
                1000.0 * np.abs(target_length[:, None] - route_length[None])
                + 100.0 * np.abs(target_score[:, None] - route_score[None])
                + 1000.0 * direction_error
            )
            local_slot, local_route = linear_sum_assignment(cost)
            if len(local_slot) != len(slots):
                raise RuntimeError(f"incomplete corner assignment row={row}")
            for slot_index, route_index in zip(local_slot, local_route, strict=True):
                slot = int(slots[slot_index])
                route = routes[route_index]
                assigned_points = route_points[route_index]
                length_error = abs(
                    float(route_length[route_index])
                    - float(candidates["path_length"][row, slot])
                )
                score_error = abs(
                    float(route_score[route_index])
                    - float(candidates["score"][row, slot])
                )
                first_value_error = float(direction_error[slot_index, route_index])
                maximum_length_error = max(maximum_length_error, length_error)
                maximum_score_error = max(maximum_score_error, score_error)
                maximum_first_error = max(maximum_first_error, first_value_error)
                if (
                    length_error > 5e-4
                    or score_error > 5e-4
                    or first_value_error > 5e-4
                ):
                    raise RuntimeError(
                        "corner route reconstruction mismatch "
                        f"row={row} slot={slot} depth={depth} "
                        f"length={length_error} score={score_error} first={first_value_error}"
                    )
                points[row, slot, :depth] = assigned_points.astype(np.float32)
                z_min[row, slot, :depth] = graph.node_z_min[route].astype(np.float32)
                z_max[row, slot, :depth] = graph.node_z_max[route].astype(np.float32)
                interaction_type[row, slot, :depth] = 3
                sequence_mask[row, slot, :depth] = True
                strength[row, slot, :depth] = (
                    graph.node_strength[route] / 180.0
                ).astype(np.float32)
        if (row + 1) % 250 == 0 or row + 1 == len(positions):
            print(f"E205 ordered geometry {row + 1}/{len(positions)}", flush=True)

    if missing_routes:
        raise RuntimeError(f"failed to reconstruct {missing_routes} corner routes")
    if not np.array_equal(sequence_mask.sum(axis=2), depth_array.astype(np.int64)):
        mismatch = int(
            np.sum(sequence_mask.sum(axis=2) != depth_array.astype(np.int64))
        )
        raise RuntimeError(f"ordered depth mismatch for {mismatch} candidates")
    candidate_feature, point_feature, sequence_feature = build_common_features(
        candidates,
        points,
        z_min,
        z_max,
        strength,
        sequence_mask,
        positions,
        site,
        setup.bs_positions,
    )
    temporary = COMMON_DIR.with_name(COMMON_DIR.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    save_array(temporary, "ordered_points_f32", points)
    save_array(temporary, "ordered_z_min_f32", z_min)
    save_array(temporary, "ordered_z_max_f32", z_max)
    save_array(temporary, "ordered_type_i8", interaction_type)
    save_array(temporary, "ordered_mask", sequence_mask)
    save_array(temporary, "ordered_strength_f32", strength)
    save_array(temporary, "candidate_feature_f16", candidate_feature.astype(np.float16))
    save_array(temporary, "point_feature_f32", point_feature)
    save_array(temporary, "sequence_feature_f16", sequence_feature.astype(np.float16))
    geometry_record = {
        "point_count": int(len(positions)),
        "candidate_count": CANDIDATES,
        "valid_candidate_count": int(np.sum(path_mask)),
        "ordered_interaction_count": int(np.sum(sequence_mask)),
        "candidate_feature_dim": int(candidate_feature.shape[-1]),
        "point_feature_dim": int(point_feature.shape[-1]),
        "sequence_feature_dim": int(sequence_feature.shape[-1]),
        "maximum_length_error": maximum_length_error,
        "maximum_score_error": maximum_score_error,
        "maximum_direction_error": maximum_first_error,
        "graph": graph_record,
        "seconds": float(time.time() - started),
    }
    state = {
        "experiment": VERSION,
        "status": "geometry_complete",
        "config": common_config_payload(),
        "geometry": geometry_record,
    }
    atomic_json(temporary / "state.json", state)
    if COMMON_DIR.exists():
        shutil.rmtree(COMMON_DIR)
    os.replace(temporary, COMMON_DIR)
    return {"reused": False, **geometry_record}


def build_common_features(
    candidates: dict[str, np.ndarray],
    points: np.ndarray,
    z_min: np.ndarray,
    z_max: np.ndarray,
    strength: np.ndarray,
    sequence_mask: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = candidates["mask"].astype(bool)
    path_type = candidates["path_type"].astype(np.int64)
    depth = candidates["depth"].astype(np.float32)
    path_length = candidates["path_length"].astype(np.float32)
    score = np.where(mask, candidates["score"], 0.0).astype(np.float32)
    source = bs_positions[site]
    direct = np.linalg.norm(positions - source, axis=1).astype(np.float32)
    one_hot = np.eye(len(e146.TYPE_NAMES), dtype=np.float32)[
        np.clip(path_type, 0, len(e146.TYPE_NAMES) - 1)
    ]
    one_hot *= mask[..., None]
    candidate_feature = np.concatenate(
        (
            np.sin(candidates["theta"])[..., None],
            np.cos(candidates["theta"])[..., None],
            np.sin(candidates["phi"])[..., None],
            np.cos(candidates["phi"])[..., None],
            np.sin(2.0 * np.pi * candidates["delay"] / S_SC)[..., None],
            np.cos(2.0 * np.pi * candidates["delay"] / S_SC)[..., None],
            one_hot,
            (depth / 3.0)[..., None],
            (np.log1p(path_length) / 6.0)[..., None],
            ((path_length - direct[:, None]) / 100.0)[..., None],
            (
                np.sign(score) * np.log1p(np.abs(score)) / 6.0
            )[..., None],
            (path_length / np.maximum(direct[:, None], 1.0))[..., None],
        ),
        axis=2,
    ).astype(np.float32)
    candidate_feature[~mask] = 0.0

    offset = positions - source
    distance = np.linalg.norm(offset, axis=1)
    unit = offset / np.maximum(distance[:, None], 1e-12)
    point_feature = np.column_stack(
        (
            offset / 100.0,
            distance / 100.0,
            np.log1p(distance),
            unit,
            site == 0,
            site == 1,
        )
    ).astype(np.float32)

    count = len(positions)
    sequence_parts = []
    bs = source[:, None, :]
    ue = positions[:, None, :]
    for step in range(SEQUENCE_STEPS):
        current = points[:, :, step].astype(np.float32)
        previous = (
            np.broadcast_to(bs, current.shape)
            if step == 0
            else points[:, :, step - 1].astype(np.float32)
        )
        following_point = (
            np.broadcast_to(ue, current.shape)
            if step + 1 == SEQUENCE_STEPS
            else points[:, :, step + 1].astype(np.float32)
        )
        last = sequence_mask[:, :, step] & (
            (step + 1 == SEQUENCE_STEPS) | (~sequence_mask[:, :, min(step + 1, 2)])
        )
        following = np.where(last[..., None], np.broadcast_to(ue, current.shape), following_point)
        incoming = current - previous
        outgoing = following - current
        incoming_length = np.linalg.norm(incoming, axis=2, keepdims=True)
        outgoing_length = np.linalg.norm(outgoing, axis=2, keepdims=True)
        incoming_unit = incoming / np.maximum(incoming_length, 1e-6)
        outgoing_unit = outgoing / np.maximum(outgoing_length, 1e-6)
        turn = np.sum(incoming_unit * outgoing_unit, axis=2, keepdims=True)
        local = np.concatenate(
            (
                (current - np.broadcast_to(bs, current.shape)) / 100.0,
                (current - np.broadcast_to(ue, current.shape)) / 100.0,
                incoming_unit,
                outgoing_unit,
                np.log1p(incoming_length) / 5.0,
                np.log1p(outgoing_length) / 5.0,
                turn,
                (z_min[:, :, step : step + 1] / 50.0),
                (z_max[:, :, step : step + 1] / 50.0),
                ((z_max[:, :, step : step + 1] - z_min[:, :, step : step + 1]) / 50.0),
                strength[:, :, step : step + 1],
                np.full((*current.shape[:2], 1), step / 2.0, dtype=np.float32),
            ),
            axis=2,
        ).astype(np.float32)
        local *= sequence_mask[:, :, step : step + 1]
        sequence_parts.append(local)
    sequence_feature = np.stack(sequence_parts, axis=2)
    return candidate_feature, point_feature, sequence_feature


def normalized_torch(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), torch.zeros_like(value))


def build_official_targets(device: torch.device, force: bool, batch_size: int = 16) -> dict[str, Any]:
    if TARGET_PAS.is_file() and TARGET_PDP.is_file() and not force:
        return {
            "reused": True,
            "pas_shape": list(np.load(TARGET_PAS, mmap_mode="r").shape),
            "pdp_shape": list(np.load(TARGET_PDP, mmap_mode="r").shape),
        }
    started = time.time()
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    temporary_pas = TARGET_PAS.with_suffix(TARGET_PAS.suffix + f".tmp.{os.getpid()}")
    temporary_pdp = TARGET_PDP.with_suffix(TARGET_PDP.suffix + f".tmp.{os.getpid()}")
    pas_output = np.lib.format.open_memmap(
        temporary_pas,
        mode="w+",
        dtype=np.float32,
        shape=(len(train_h), MP, MH, MV, N_UE),
    )
    pdp_output = np.lib.format.open_memmap(
        temporary_pdp,
        mode="w+",
        dtype=np.float32,
        shape=(len(train_h), MP, N_UE, S_SC),
    )
    for start in range(0, len(train_h), batch_size):
        end = min(start + batch_size, len(train_h))
        channel = torch.as_tensor(
            np.array(train_h[start:end], dtype=np.complex64, copy=True), device=device
        )
        shaped = channel.reshape(-1, MP, MH, MV, N_UE, S_SC)
        pas_power = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
        pas_target = normalized_torch(pas_power, (1, 2, 3)).mean(dim=5)
        delay_power = torch.fft.ifft(shaped, dim=5).abs().square().float()
        pdp_target = normalized_torch(delay_power, (5,)).mean(dim=(2, 3))
        pas_output[start:end] = pas_target.cpu().numpy().astype(np.float32)
        pdp_output[start:end] = pdp_target.cpu().numpy().astype(np.float32)
        print(f"E205 official targets {end}/{len(train_h)}", flush=True)
    pas_output.flush()
    pdp_output.flush()
    os.replace(temporary_pas, TARGET_PAS)
    os.replace(temporary_pdp, TARGET_PDP)
    return {
        "reused": False,
        "pas_shape": list(pas_output.shape),
        "pdp_shape": list(pdp_output.shape),
        "seconds": float(time.time() - started),
    }


def score_channels(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, float | int], PointMetricStatistics]:
    parts: list[PointMetricStatistics] = []
    for start in range(0, len(prediction), batch_size):
        end = min(start + batch_size, len(prediction))
        parts.append(point_metric_statistics_torch(prediction[start:end], truth[start:end]))
    statistics = PointMetricStatistics(
        pas=np.concatenate([part.pas for part in parts]),
        pdp=np.concatenate([part.pdp for part in parts]),
        squared_error=np.concatenate([part.squared_error for part in parts]),
        truth_energy=np.concatenate([part.truth_energy for part in parts]),
    )
    return aggregate_statistics(statistics), statistics


def prepare(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in required_inputs():
        if not path.is_file():
            raise FileNotFoundError(path)
    geometry = build_ordered_geometry(args.force)
    targets = build_official_targets(device, args.force)
    _, query_ids, _ = canonical_full_fold()
    base = torch.as_tensor(np.array(np.load(BASE_V7, mmap_mode="r"), copy=True), device=device)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    baseline, _ = score_channels(base, truth, Config().score_batch_size)
    if abs(float(baseline["score"]) - EXPECTED_V7_SCORE) > 1e-8:
        raise RuntimeError(
            f"V7 identity mismatch {baseline['score']} != {EXPECTED_V7_SCORE}"
        )
    smoke = model_smoke(device)
    state = json.loads(COMMON_STATE.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "prepare_complete",
            "targets": targets,
            "baseline": baseline,
            "smoke": smoke,
            "gpu": gpu_record(device),
            "prepare_seconds": float(time.time() - started),
        }
    )
    atomic_json(COMMON_STATE, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return state


def candidate_coordinate(
    outer: int,
    fit_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    setup: Any,
    component: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    los = np.load(e146.LOS_LABELS, allow_pickle=False)["train_center_clear"].astype(bool)
    observations = e146.select_los_observations(
        positions,
        site,
        setup.bs_positions,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
        los,
    )
    horizontal_site, vertical_site, record = e150.fit_strict_array_axes(
        outer, fit_ids, observations, site, e146.Config()
    )
    candidate_file = np.load(e146.CANDIDATE_CACHE, allow_pickle=False)
    candidates = {
        key: candidate_file[key]
        for key in candidate_file.files
        if key != "config_json"
    }
    coordinate = e146.map_coordinates_linear_array(
        candidates, horizontal_site[site], vertical_site[site]
    ).astype(np.float32)
    return coordinate, record


def build_fold_cache(
    outer: int,
    config: Config,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    directory = fold_cache_dir(outer)
    state_path = directory / "state.json"
    if state_path.is_file() and not force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "cache_complete":
            return {"reused": True, **state}
    started = time.time()
    stage_timing: dict[str, float] = {}
    fold, query_ids, canonical = canonical_full_fold()
    validation_fold = (outer + config.validation_fold_offset) % 5
    fit_ids = np.flatnonzero((fold != outer) & (fold != validation_fold)).astype(np.int64)
    if FORCE_BASE_GLOBAL_IDS:
        excluded = np.fromiter(FORCE_BASE_GLOBAL_IDS, dtype=np.int64)
        fit_ids = fit_ids[~np.isin(fit_ids, excluded)]
    validation_ids = query_ids[canonical == validation_fold]
    held_ids = query_ids[canonical == outer]
    if np.intersect1d(fit_ids, np.concatenate((validation_ids, held_ids))).size:
        raise RuntimeError("E205 fit rows overlap validation/held canonical labels")

    setup, positions, _, site, _ = load_positions()
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    phase_started = time.time()
    coordinate, axis_record = candidate_coordinate(
        outer, fit_ids, positions, site, setup, component
    )
    stage_timing["array_axis_and_coordinate"] = float(time.time() - phase_started)
    candidate_file = np.load(e146.CANDIDATE_CACHE, allow_pickle=False)
    mask = candidate_file["mask"].astype(bool)

    phase_started = time.time()
    distances, neighbors = e150.strict_neighbors(
        positions,
        site,
        fit_ids,
        np.arange(len(positions), dtype=np.int64),
        config.carrier_neighbors,
    )
    stage_timing["strict_neighbors"] = float(time.time() - phase_started)
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    phase_started = time.time()
    calibration = estimate_carrier(
        train_h,
        positions,
        site,
        setup.bs_positions,
        fit_ids,
        component["point_power"].astype(np.float64),
        max_pairs_per_site=500,
        seed=config.seed + outer * 101,
    )
    base_configs = carrier_bank.carrier_configs()
    configs_by_site = {
        int(site_value): [
            replace(
                template,
                beta=calibration[int(site_value)].beta,
                gamma=calibration[int(site_value)].gamma,
            )
            for template in base_configs
        ]
        for site_value in np.unique(site)
    }
    stage_timing["carrier_calibration"] = float(time.time() - phase_started)

    temporary = directory.with_name(directory.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    coordinate_output = np.lib.format.open_memmap(
        temporary / "coordinate.npy",
        mode="w+",
        dtype=np.float16,
        shape=coordinate.shape,
    )
    coordinate_output[:] = coordinate.astype(np.float16)
    base_log_mass_output = np.lib.format.open_memmap(
        temporary / "base_log_mass.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(positions), CANDIDATES),
    )
    base_log_port_output = np.lib.format.open_memmap(
        temporary / "base_log_port.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(positions), CANDIDATES, PORTS),
    )
    field_feature_output = np.lib.format.open_memmap(
        temporary / "field_feature.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(positions), CANDIDATES, 38),
    )
    point_field_output = np.lib.format.open_memmap(
        temporary / "point_field_feature.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(positions), 24),
    )

    phase_started = time.time()
    train_device_started = time.time()
    train_device = torch.as_tensor(np.asarray(train_h), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stage_timing["train_h_host_to_device"] = float(time.time() - train_device_started)
    mask_device = torch.as_tensor(mask, device=device)
    first_batch_seconds = None
    for start in range(0, len(positions), config.carrier_batch_size):
        batch_started = time.time()
        end = min(start + config.carrier_batch_size, len(positions))
        predicted = e150.batched_carrier_predictions(
            train_device,
            neighbors[start:end],
            positions[neighbors[start:end]],
            positions[start:end],
            distances[start:end],
            setup.bs_positions,
            site[start:end],
            configs_by_site,
        )
        carrier_power = predicted.abs().square().sum(
            dim=(2, 3, 4), dtype=torch.float64
        ).to(torch.float32)
        field = e150.complex_angle_delay(predicted.reshape(-1, *predicted.shape[2:]))
        local_coordinate = torch.as_tensor(coordinate[start:end], device=device)
        repeated_coordinate = (
            local_coordinate[:, None]
            .expand(-1, CARRIERS, -1, -1)
            .reshape((end - start) * CARRIERS, CANDIDATES, 3)
        )
        sampled = e150.periodic_trilinear_sample(field, repeated_coordinate)
        sampled = sampled.reshape(end - start, CARRIERS, CANDIDATES, PORTS).permute(
            0, 2, 1, 3
        )
        energy_port = sampled.abs().square().float()
        energy = energy_port.sum(dim=3).clamp_min(TINY)
        log_absolute = torch.log(energy) - torch.log(carrier_power.clamp_min(TINY))[:, None]
        local_mask = mask_device[start:end]
        masked_absolute = torch.where(
            local_mask[..., None],
            log_absolute,
            torch.full_like(log_absolute, -torch.inf),
        )
        log_mass_carrier = masked_absolute - torch.logsumexp(
            masked_absolute, dim=1, keepdim=True
        )
        base_log_mass = torch.logsumexp(log_mass_carrier, dim=2) - math.log(CARRIERS)
        base_log_mass = torch.where(
            local_mask, base_log_mass, torch.full_like(base_log_mass, -30.0)
        )
        port_profile = energy_port / energy[..., None]
        carrier_weight = torch.softmax(log_absolute, dim=2)
        base_port = (carrier_weight[..., None] * port_profile).sum(dim=2)
        base_port = base_port / base_port.sum(dim=2, keepdim=True).clamp_min(TINY)
        base_log_port = torch.log(base_port.clamp_min(1e-8))
        absolute_mean = log_absolute.mean(dim=2)
        absolute_std = log_absolute.std(dim=2)
        absolute_range = log_absolute.amax(dim=2) - log_absolute.amin(dim=2)
        agreement = torch.exp(-absolute_std)
        port_mean = port_profile.mean(dim=2)
        port_std = port_profile.std(dim=2)
        field_feature = torch.cat(
            (
                log_mass_carrier.clamp(-20.0, 0.0),
                absolute_mean.clamp(-30.0, 5.0)[..., None],
                absolute_std.clamp(0.0, 10.0)[..., None],
                absolute_range.clamp(0.0, 20.0)[..., None],
                agreement[..., None],
                port_mean,
                port_std,
                base_log_mass[..., None].clamp(-20.0, 0.0),
            ),
            dim=2,
        )
        if field_feature.shape[2] != 38:
            raise RuntimeError(f"field feature dimension {field_feature.shape}")
        field_feature = torch.where(
            local_mask[..., None], field_feature, torch.zeros_like(field_feature)
        )
        log_power = torch.log(carrier_power.clamp_min(TINY))
        centered_power = log_power - log_power.mean(dim=1, keepdim=True)
        local_distance = distances[start:end]
        point_field = np.column_stack(
            (
                centered_power.cpu().numpy(),
                log_power.mean(dim=1).cpu().numpy(),
                log_power.std(dim=1).cpu().numpy(),
                (log_power.amax(dim=1) - log_power.amin(dim=1)).cpu().numpy(),
                np.mean(local_distance, axis=1),
                np.median(local_distance, axis=1),
                np.quantile(local_distance, 0.90, axis=1),
                np.max(local_distance, axis=1),
            )
        ).astype(np.float32)
        base_log_mass_output[start:end] = base_log_mass.cpu().numpy().astype(np.float16)
        base_log_port_output[start:end] = base_log_port.cpu().numpy().astype(np.float16)
        field_feature_output[start:end] = field_feature.cpu().numpy().astype(np.float16)
        point_field_output[start:end] = point_field
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if first_batch_seconds is None:
            first_batch_seconds = float(time.time() - batch_started)
        print(f"E205 outer={outer} carrier field {end}/{len(positions)}", flush=True)
        del predicted, field, sampled, energy_port, energy
    stage_timing["carrier_field_total"] = float(time.time() - phase_started)
    stage_timing["carrier_first_batch"] = float(first_batch_seconds or 0.0)
    del train_device, mask_device
    if device.type == "cuda":
        torch.cuda.empty_cache()

    phase_started = time.time()
    targets, assignment = e124.build_one_to_one_targets(
        coordinate[fit_ids],
        mask[fit_ids],
        component["coordinates"][fit_ids].astype(np.float32),
        component["coefficients"][fit_ids].astype(np.complex64),
        component["point_power"][fit_ids].astype(np.float64),
    )
    target_mass = np.zeros((len(positions), CANDIDATES), dtype=np.float16)
    target_port = np.full(
        (len(positions), CANDIDATES, PORTS), 1.0 / PORTS, dtype=np.float16
    )
    label_valid = np.zeros(len(positions), dtype=bool)
    target_mass[fit_ids] = targets["mass_normalized"].astype(np.float16)
    target_port[fit_ids] = targets["port"].astype(np.float16)
    label_valid[fit_ids] = targets["matched_fraction"] > 0.0
    save_array(temporary, "target_mass", target_mass)
    save_array(temporary, "target_port", target_port)
    save_array(temporary, "label_valid", label_valid)
    save_array(temporary, "fit_ids", fit_ids.astype(np.int32))
    save_array(temporary, "validation_ids", validation_ids.astype(np.int32))
    save_array(temporary, "held_ids", held_ids.astype(np.int32))
    stage_timing["one_to_one_labels"] = float(time.time() - phase_started)

    for output in (
        coordinate_output,
        base_log_mass_output,
        base_log_port_output,
        field_feature_output,
        point_field_output,
    ):
        output.flush()
    state = {
        "experiment": VERSION,
        "status": "cache_complete",
        "outer": outer,
        "validation_fold": validation_fold,
        "fit_count": int(len(fit_ids)),
        "validation_count": int(len(validation_ids)),
        "held_count": int(len(held_ids)),
        "canonical_fold_override": True,
        "zero_label_overlap": bool(
            np.intersect1d(fit_ids, np.concatenate((validation_ids, held_ids))).size == 0
        ),
        "candidate_donors_exclude_outer_and_validation": True,
        "array_axis_fit": axis_record,
        "carrier_calibration": {
            str(key): value.to_dict() for key, value in calibration.items()
        },
        "assignment": assignment,
        "valid_label_count": int(np.sum(label_valid)),
        "timing": stage_timing,
        "estimated_carrier_seconds_from_first_batch": float(
            (first_batch_seconds or 0.0)
            * math.ceil(len(positions) / config.carrier_batch_size)
        ),
        "gpu": gpu_record(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "seconds": float(time.time() - started),
    }
    atomic_json(temporary / "state.json", state)
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(temporary, directory)
    return {"reused": False, **state}


class OrderedPathCalibrator(nn.Module):
    def __init__(
        self,
        candidate_dim: int,
        point_dim: int,
        sequence_dim: int,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.candidate_input = nn.Sequential(
            nn.Linear(candidate_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.point_input = nn.Sequential(
            nn.Linear(point_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.sequence_input = nn.Linear(sequence_dim, hidden)
        self.interaction_embedding = nn.Embedding(4, hidden)
        self.step_embedding = nn.Embedding(SEQUENCE_STEPS, hidden)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            sequence_layer, num_layers=config.sequence_layers
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer, num_layers=config.set_layers
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.mass_head = nn.Linear(hidden, 1)
        self.port_head = nn.Linear(hidden, PORTS)
        for layer in (self.mass_head, self.port_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        candidate = self.candidate_input(batch["candidate_feature"])
        point = self.point_input(batch["point_feature"])
        sequence = self.sequence_input(batch["sequence_feature"])
        step = torch.arange(SEQUENCE_STEPS, device=sequence.device)
        sequence = (
            sequence
            + self.interaction_embedding(batch["sequence_type"].clamp(0, 3))
            + self.step_embedding(step)[None, None]
        )
        batch_size, candidates, steps, hidden = sequence.shape
        flat_mask = batch["sequence_mask"].reshape(batch_size * candidates, steps)
        effective_mask = flat_mask.clone()
        empty = ~effective_mask.any(dim=1)
        effective_mask[empty, 0] = True
        sequence = self.sequence_encoder(
            sequence.reshape(batch_size * candidates, steps, hidden),
            src_key_padding_mask=~effective_mask,
        )
        weight = flat_mask.to(sequence.dtype)
        pooled = (sequence * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pooled = pooled.reshape(batch_size, candidates, hidden)
        hidden_state = self.fusion(
            torch.cat(
                (candidate, pooled, point[:, None].expand(-1, candidates, -1)), dim=2
            )
        )
        hidden_state = self.set_encoder(
            hidden_state, src_key_padding_mask=~batch["mask"]
        )
        hidden_state = self.output_norm(hidden_state)
        mass_residual = self.config.score_limit * torch.tanh(
            self.mass_head(hidden_state)[..., 0]
        )
        port_residual = self.config.port_limit * torch.tanh(
            self.port_head(hidden_state)
        )
        mass_logits = batch["base_log_mass"] + mass_residual
        mass_logits = torch.where(
            batch["mask"], mass_logits, torch.full_like(mass_logits, -torch.inf)
        )
        mass = torch.softmax(mass_logits, dim=1)
        port = torch.softmax(batch["base_log_port"] + port_residual, dim=2)
        return {
            "mass": mass,
            "port": port,
            "mass_residual": mass_residual,
            "port_residual": port_residual,
        }


class FoldData:
    def __init__(self, outer: int, device: torch.device) -> None:
        self.outer = outer
        self.device = device
        self.cache_dir = fold_cache_dir(outer)
        candidate_common = np.load(COMMON_CANDIDATE_FEATURE, mmap_mode="r")
        point_common = np.load(COMMON_POINT_FEATURE, mmap_mode="r")
        sequence = np.load(COMMON_SEQUENCE_FEATURE, mmap_mode="r")
        field = np.load(self.cache_dir / "field_feature.npy", mmap_mode="r")
        point_field = np.load(
            self.cache_dir / "point_field_feature.npy", mmap_mode="r"
        )
        coordinate_np = np.asarray(
            np.load(self.cache_dir / "coordinate.npy", mmap_mode="r"), dtype=np.float32
        )
        coordinate_phase = 2.0 * np.pi * coordinate_np / PERIODS[None, None]
        coordinate_feature = np.concatenate(
            (np.sin(coordinate_phase), np.cos(coordinate_phase)), axis=2
        ).astype(np.float32)
        candidate_feature = np.concatenate(
            (
                np.asarray(candidate_common, dtype=np.float32),
                np.asarray(field, dtype=np.float32),
                coordinate_feature,
            ),
            axis=2,
        )
        point_feature = np.concatenate(
            (
                np.asarray(point_common, dtype=np.float32),
                np.asarray(point_field, dtype=np.float32),
            ),
            axis=1,
        )
        candidate_mask = np.load(e146.CANDIDATE_CACHE, allow_pickle=False)[
            "mask"
        ].astype(bool)
        self.fit_ids_np = np.load(self.cache_dir / "fit_ids.npy").astype(np.int64)
        if FORCE_BASE_GLOBAL_IDS:
            excluded = np.fromiter(FORCE_BASE_GLOBAL_IDS, dtype=np.int64)
            self.fit_ids_np = self.fit_ids_np[
                ~np.isin(self.fit_ids_np, excluded)
            ]
        self.validation_ids_np = np.load(
            self.cache_dir / "validation_ids.npy"
        ).astype(np.int64)
        self.held_ids_np = np.load(self.cache_dir / "held_ids.npy").astype(np.int64)
        valid_values = candidate_feature[self.fit_ids_np][
            candidate_mask[self.fit_ids_np]
        ]
        candidate_mean = valid_values.mean(axis=0, dtype=np.float64).astype(np.float32)
        candidate_std = valid_values.std(axis=0, dtype=np.float64).astype(np.float32)
        candidate_std[candidate_std < 1e-4] = 1.0
        point_mean = point_feature[self.fit_ids_np].mean(axis=0, dtype=np.float64).astype(
            np.float32
        )
        point_std = point_feature[self.fit_ids_np].std(axis=0, dtype=np.float64).astype(
            np.float32
        )
        point_std[point_std < 1e-4] = 1.0
        candidate_feature = (candidate_feature - candidate_mean) / candidate_std
        candidate_feature = np.where(
            candidate_mask[..., None], candidate_feature.clip(-8.0, 8.0), 0.0
        )
        point_feature = ((point_feature - point_mean) / point_std).clip(-8.0, 8.0)
        self.normalization = {
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
            "point_mean": point_mean,
            "point_std": point_std,
        }
        self.mask = torch.as_tensor(candidate_mask, device=device)
        self.coordinate = torch.as_tensor(coordinate_np, device=device)
        self.candidate_feature = torch.as_tensor(candidate_feature, device=device)
        self.point_feature = torch.as_tensor(point_feature, device=device)
        self.sequence_feature = torch.as_tensor(
            np.asarray(sequence, dtype=np.float32), device=device
        )
        self.sequence_type = torch.as_tensor(
            np.asarray(np.load(ORDERED_TYPE, mmap_mode="r"), dtype=np.int64),
            device=device,
        )
        self.sequence_mask = torch.as_tensor(
            np.asarray(np.load(ORDERED_MASK, mmap_mode="r"), dtype=bool), device=device
        )
        self.base_log_mass = torch.as_tensor(
            np.asarray(
                np.load(self.cache_dir / "base_log_mass.npy", mmap_mode="r"),
                dtype=np.float32,
            ),
            device=device,
        )
        self.base_log_port = torch.as_tensor(
            np.asarray(
                np.load(self.cache_dir / "base_log_port.npy", mmap_mode="r"),
                dtype=np.float32,
            ),
            device=device,
        )
        self.target_mass = torch.as_tensor(
            np.asarray(
                np.load(self.cache_dir / "target_mass.npy", mmap_mode="r"),
                dtype=np.float32,
            ),
            device=device,
        )
        self.target_port = torch.as_tensor(
            np.asarray(
                np.load(self.cache_dir / "target_port.npy", mmap_mode="r"),
                dtype=np.float32,
            ),
            device=device,
        )
        self.label_valid = torch.as_tensor(
            np.load(self.cache_dir / "label_valid.npy").astype(bool), device=device
        )
        self.target_pas = torch.as_tensor(
            np.asarray(np.load(TARGET_PAS, mmap_mode="r"), dtype=np.float32), device=device
        )
        self.target_pdp = torch.as_tensor(
            np.asarray(np.load(TARGET_PDP, mmap_mode="r"), dtype=np.float32), device=device
        )
        self.horizontal, self.vertical, self.delay = spectral_kernels(self.coordinate)

    def batch(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "rows": rows,
            "mask": self.mask[rows],
            "coordinate": self.coordinate[rows],
            "candidate_feature": self.candidate_feature[rows],
            "point_feature": self.point_feature[rows],
            "sequence_feature": self.sequence_feature[rows],
            "sequence_type": self.sequence_type[rows],
            "sequence_mask": self.sequence_mask[rows],
            "base_log_mass": self.base_log_mass[rows],
            "base_log_port": self.base_log_port[rows],
            "target_mass": self.target_mass[rows],
            "target_port": self.target_port[rows],
            "label_valid": self.label_valid[rows],
            "target_pas": self.target_pas[rows],
            "target_pdp": self.target_pdp[rows],
            "horizontal": self.horizontal[rows],
            "vertical": self.vertical[rows],
            "delay": self.delay[rows],
        }


def spectral_kernels(
    coordinate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizontal_index = torch.arange(MH, dtype=torch.float32, device=coordinate.device)
    vertical_index = torch.arange(MV, dtype=torch.float32, device=coordinate.device)
    delay_index = torch.arange(S_SC, dtype=torch.float32, device=coordinate.device)
    horizontal_wave = torch.exp(
        2j * torch.pi * coordinate[..., 0, None] * horizontal_index / MH
    )
    vertical_wave = torch.exp(
        2j * torch.pi * coordinate[..., 1, None] * vertical_index / MV
    )
    delay_wave = torch.exp(
        -2j * torch.pi * coordinate[..., 2, None] * delay_index / S_SC
    )
    horizontal = torch.fft.fft(horizontal_wave, dim=-1).abs().square().float()
    vertical = torch.fft.fft(vertical_wave, dim=-1).abs().square().float()
    delay = torch.fft.ifft(delay_wave, dim=-1).abs().square().float()
    horizontal /= horizontal.sum(dim=-1, keepdim=True).clamp_min(TINY)
    vertical /= vertical.sum(dim=-1, keepdim=True).clamp_min(TINY)
    delay /= delay.sum(dim=-1, keepdim=True).clamp_min(TINY)
    return horizontal, vertical, delay


def render_compact(
    mass: torch.Tensor,
    port: torch.Tensor,
    horizontal: torch.Tensor,
    vertical: torch.Tensor,
    delay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    energy = mass[..., None, None] * port.reshape(len(mass), CANDIDATES, MP, N_UE)
    pas = torch.einsum("bkpu,bkh,bkv->bphvu", energy, horizontal, vertical)
    pas = normalized_torch(pas, (1, 2, 3))
    pdp = torch.einsum("bkpu,bkd->bpud", energy, delay)
    pdp = normalized_torch(pdp, (3,))
    return pas, pdp


def cosine_sufficient(
    prediction: torch.Tensor,
    target: torch.Tensor,
    norm_dims: tuple[int, ...],
    average_dims: tuple[int, ...],
) -> torch.Tensor:
    dot = (prediction * target).sum(dim=norm_dims, dtype=torch.float64)
    prediction_norm = torch.sqrt(
        prediction.square().sum(dim=norm_dims, dtype=torch.float64)
    )
    target_norm = torch.sqrt(target.square().sum(dim=norm_dims, dtype=torch.float64))
    denominator = prediction_norm * target_norm
    denominator = torch.where(
        denominator == 0.0, torch.full_like(denominator, 1e-12), denominator
    )
    return (dot / denominator).mean(dim=average_dims, dtype=torch.float64)


def training_objective(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    pas, pdp = render_compact(
        output["mass"],
        output["port"],
        batch["horizontal"],
        batch["vertical"],
        batch["delay"],
    )
    pas_score = cosine_sufficient(pas, batch["target_pas"], (1, 2, 3), (1,))
    pdp_score = cosine_sufficient(pdp, batch["target_pdp"], (3,), (1, 2))
    spectral = 0.5 * (pas_score + pdp_score)
    label_valid = batch["label_valid"]
    if bool(label_valid.any()):
        target_mass = batch["target_mass"][label_valid]
        predicted_mass = output["mass"][label_valid]
        mass_affinity = (
            torch.sqrt(target_mass.clamp_min(0.0))
            * torch.sqrt(predicted_mass.clamp_min(1e-12))
        ).sum(dim=1)
        weight = target_mass / target_mass.sum(dim=1, keepdim=True).clamp_min(TINY)
        port_loss = -(
            batch["target_port"][label_valid].clamp_min(0.0)
            * torch.log(output["port"][label_valid].clamp_min(1e-8))
        ).sum(dim=2)
        port_loss = (weight * port_loss).sum(dim=1)
        positive = target_mass > 0.0
        active_logits = output["mass_residual"][label_valid]
        positive_loss = F.binary_cross_entropy_with_logits(
            active_logits[positive], torch.ones_like(active_logits[positive])
        ) if bool(positive.any()) else torch.zeros((), device=pas.device)
        negative = (~positive) & batch["mask"][label_valid]
        negative_loss = F.binary_cross_entropy_with_logits(
            active_logits[negative], torch.zeros_like(active_logits[negative])
        ) if bool(negative.any()) else torch.zeros((), device=pas.device)
        existence = 0.5 * (positive_loss + negative_loss)
        mass_term = 1.0 - mass_affinity.mean()
        port_term = port_loss.mean()
    else:
        mass_term = torch.zeros((), device=pas.device)
        port_term = torch.zeros((), device=pas.device)
        existence = torch.zeros((), device=pas.device)
    residual = (
        output["mass_residual"].square().mean()
        + output["port_residual"].square().mean()
    )
    loss = (
        config.spectral_weight * (1.0 - spectral.mean())
        + config.mass_weight * mass_term
        + config.port_weight * port_term
        + config.existence_weight * existence
        + config.residual_weight * residual
    )
    return loss, {
        "loss": float(loss.detach()),
        "pas": float(pas_score.mean().detach()),
        "pdp": float(pdp_score.mean().detach()),
        "mass": float(1.0 - mass_term.detach()),
        "port_loss": float(port_term.detach()),
        "existence": float(existence.detach()),
    }


def compact_to_full_endpoint(
    pas: torch.Tensor, pdp: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    full_pas = pas[..., None].expand(-1, -1, -1, -1, -1, S_SC)
    full_pdp = (
        pdp[:, :, None, None, :, :]
        .expand(-1, -1, MH, MV, -1, -1)
        .reshape(len(pdp), MP * MH * MV, N_UE, S_SC)
    )
    return full_pas, full_pdp


def endpoint_target(
    base: torch.Tensor,
    endpoint: torch.Tensor,
    beta: float,
    dims: tuple[int, ...],
) -> torch.Tensor:
    mixed = (
        (1.0 - beta) * torch.sqrt(base.clamp_min(0.0))
        + beta * torch.sqrt(endpoint.clamp_min(0.0))
    ).square()
    return normalized_torch(mixed, dims)


def apply_endpoint(
    base: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    desired_pas = endpoint_target(base_pas, endpoint_pas, config.pas_beta, (1, 2, 3))
    desired_pdp = endpoint_target(base_pdp, endpoint_pdp, config.pdp_beta, (3,))
    return project_joint(
        base,
        desired_pas,
        desired_pdp,
        config.projection_cycles,
        config.projection_order,
    )


@torch.inference_mode()
def predict_endpoint(
    model: OrderedPathCalibrator,
    data: FoldData,
    rows: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mass = np.empty((len(rows), CANDIDATES), dtype=np.float32)
    port = np.empty((len(rows), CANDIDATES, PORTS), dtype=np.float32)
    pas = np.empty((len(rows), MP, MH, MV, N_UE), dtype=np.float32)
    pdp = np.empty((len(rows), MP, N_UE, S_SC), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        local = torch.as_tensor(rows[start:end], dtype=torch.long, device=data.device)
        batch = data.batch(local)
        output = model(batch)
        local_pas, local_pdp = render_compact(
            output["mass"],
            output["port"],
            batch["horizontal"],
            batch["vertical"],
            batch["delay"],
        )
        mass[start:end] = output["mass"].cpu().numpy().astype(np.float32)
        port[start:end] = output["port"].cpu().numpy().astype(np.float32)
        pas[start:end] = local_pas.cpu().numpy().astype(np.float32)
        pdp[start:end] = local_pdp.cpu().numpy().astype(np.float32)
    coordinate = data.coordinate[torch.as_tensor(rows, device=data.device)].cpu().numpy()
    return mass, coordinate.astype(np.float32), port, pas, pdp


@torch.inference_mode()
def complete_evaluation(
    model: OrderedPathCalibrator | None,
    data: FoldData,
    global_ids: np.ndarray,
    config: Config,
    collect_channel: bool,
) -> tuple[
    dict[str, float | int],
    dict[str, float | int],
    np.ndarray | None,
    dict[str, np.ndarray] | None,
]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    lookup = np.full(4000, -1, dtype=np.int64)
    lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    local = lookup[global_ids]
    if np.any(local < 0):
        raise ValueError("complete evaluation requested non-canonical rows")
    base_np = np.load(BASE_V7, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    selected_parts: list[PointMetricStatistics] = []
    baseline_parts: list[PointMetricStatistics] = []
    channel_output = (
        np.empty((len(global_ids), MP * MH * MV, N_UE, S_SC), dtype=np.complex64)
        if collect_channel
        else None
    )
    endpoint_payload = None
    force_base_ids = (
        np.fromiter(FORCE_BASE_GLOBAL_IDS, dtype=np.int64)
        if FORCE_BASE_GLOBAL_IDS
        else np.empty(0, dtype=np.int64)
    )
    if model is not None:
        mass, coordinate, port, pas, pdp = predict_endpoint(
            model, data, global_ids, config.train_batch_size
        )
        endpoint_payload = {
            "mass": mass,
            "coordinate": coordinate,
            "port": port,
        }
    for start in range(0, len(global_ids), config.score_batch_size):
        end = min(start + config.score_batch_size, len(global_ids))
        base = torch.as_tensor(
            np.array(base_np[local[start:end]], dtype=np.complex64, copy=True),
            device=data.device,
        )
        truth = torch.as_tensor(
            np.asarray(truth_all[global_ids[start:end]], dtype=np.complex64),
            device=data.device,
        )
        if model is None:
            candidate = base
        else:
            endpoint_pas, endpoint_pdp = compact_to_full_endpoint(
                torch.as_tensor(pas[start:end], device=data.device),
                torch.as_tensor(pdp[start:end], device=data.device),
            )
            candidate = apply_endpoint(base, endpoint_pas, endpoint_pdp, config)
        if len(force_base_ids):
            fallback = np.isin(global_ids[start:end], force_base_ids)
            if np.any(fallback):
                fallback_t = torch.as_tensor(fallback, device=data.device)
                candidate = candidate.clone()
                candidate[fallback_t] = base[fallback_t]
        selected_parts.append(point_metric_statistics_torch(candidate, truth))
        baseline_parts.append(point_metric_statistics_torch(base, truth))
        if channel_output is not None:
            channel_output[start:end] = candidate.cpu().numpy().astype(np.complex64)
    selected_statistics = PointMetricStatistics(
        pas=np.concatenate([part.pas for part in selected_parts]),
        pdp=np.concatenate([part.pdp for part in selected_parts]),
        squared_error=np.concatenate([part.squared_error for part in selected_parts]),
        truth_energy=np.concatenate([part.truth_energy for part in selected_parts]),
    )
    baseline_statistics = PointMetricStatistics(
        pas=np.concatenate([part.pas for part in baseline_parts]),
        pdp=np.concatenate([part.pdp for part in baseline_parts]),
        squared_error=np.concatenate([part.squared_error for part in baseline_parts]),
        truth_energy=np.concatenate([part.truth_energy for part in baseline_parts]),
    )
    return (
        aggregate_statistics(selected_statistics),
        aggregate_statistics(baseline_statistics),
        channel_output,
        endpoint_payload,
    )


def metric_delta(
    selected: dict[str, float | int], baseline: dict[str, float | int]
) -> dict[str, float]:
    return {
        "pas": float(selected["pas"] - baseline["pas"]),
        "pdp": float(selected["pdp"] - baseline["pdp"]),
        "nmse_score": float(selected["nmse_score"] - baseline["nmse_score"]),
        "score": float(selected["score"] - baseline["score"]),
    }


def make_model(data: FoldData, config: Config) -> OrderedPathCalibrator:
    return OrderedPathCalibrator(
        data.candidate_feature.shape[-1],
        data.point_feature.shape[-1],
        data.sequence_feature.shape[-1],
        config,
    ).to(data.device)


def model_smoke(device: torch.device) -> dict[str, Any]:
    config = Config(hidden=32, heads=4, feedforward=64, set_layers=1)
    model = OrderedPathCalibrator(12, 8, 10, config).to(device)
    mask = torch.ones(2, 8, dtype=torch.bool, device=device)
    sequence_mask = torch.zeros(2, 8, SEQUENCE_STEPS, dtype=torch.bool, device=device)
    sequence_mask[:, 1:, 0] = True
    batch = {
        "mask": mask,
        "candidate_feature": torch.randn(2, 8, 12, device=device),
        "point_feature": torch.randn(2, 8, device=device),
        "sequence_feature": torch.randn(2, 8, SEQUENCE_STEPS, 10, device=device),
        "sequence_type": torch.zeros(2, 8, SEQUENCE_STEPS, dtype=torch.long, device=device),
        "sequence_mask": sequence_mask,
        "base_log_mass": torch.log_softmax(torch.randn(2, 8, device=device), dim=1),
        "base_log_port": torch.log_softmax(torch.randn(2, 8, PORTS, device=device), dim=2),
    }
    output = model(batch)
    loss = output["mass"].square().mean() + output["port"].square().mean()
    loss.backward()
    finite_gradient = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if not finite_gradient:
        raise FloatingPointError("E205 smoke produced non-finite gradients")
    return {
        "mass_shape": list(output["mass"].shape),
        "port_shape": list(output["port"].shape),
        "mass_sum_error": float((output["mass"].sum(dim=1) - 1.0).abs().max()),
        "port_sum_error": float((output["port"].sum(dim=2) - 1.0).abs().max()),
        "finite_gradient": finite_gradient,
    }


def train_fold(
    outer: int,
    config: Config,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    report_path = fold_report_path(outer)
    if report_path.is_file() and fold_channel_path(outer).is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            print(json.dumps({"status": "reused", "outer": outer}, indent=2))
            return report
    started = time.time()
    fold_dir(outer).mkdir(parents=True, exist_ok=True)
    cache_record = build_fold_cache(outer, config, device, force=False)
    data_started = time.time()
    data = FoldData(outer, device)
    data_seconds = float(time.time() - data_started)
    model = make_model(data, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    baseline_validation, _, _, _ = complete_evaluation(
        None, data, data.validation_ids_np, config, collect_channel=False
    )
    best_score = float(baseline_validation["score"])
    best_epoch = 0
    best_metrics = baseline_validation
    best_state = None
    history: list[dict[str, Any]] = []
    start_epoch = 0
    last_path = fold_last_checkpoint_path(outer)
    if last_path.is_file() and not force:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("outer") == outer and checkpoint.get("config") == asdict(config):
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"])
            best_score = float(checkpoint["best_score"])
            best_epoch = int(checkpoint["best_epoch"])
            best_metrics = checkpoint["best_metrics"]
            best_state = checkpoint["best_state"]
            history = checkpoint["history"]
            torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
            print(f"E205 outer={outer} resume epoch={start_epoch}", flush=True)

    fit_ids = data.fit_ids_np
    for epoch in range(start_epoch + 1, config.epochs + 1):
        model.train()
        rng = np.random.default_rng(config.seed + outer * 1000 + epoch)
        order = fit_ids[rng.permutation(len(fit_ids))]
        sums: dict[str, float] = {}
        batch_count = 0
        for start in range(0, len(order), config.train_batch_size):
            rows = torch.as_tensor(
                order[start : start + config.train_batch_size],
                dtype=torch.long,
                device=device,
            )
            batch = data.batch(rows)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, values = training_objective(output, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"E205 outer={outer} epoch={epoch} non-finite loss"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + value
            batch_count += 1
        record: dict[str, Any] = {
            "epoch": epoch,
            "train": {key: value / batch_count for key, value in sums.items()},
        }
        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            validation, _, _, _ = complete_evaluation(
                model, data, data.validation_ids_np, config, collect_channel=False
            )
            record["validation"] = validation
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_epoch = epoch
                best_metrics = validation
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                atomic_torch_save(
                    fold_best_checkpoint_path(outer),
                    {
                        "outer": outer,
                        "epoch": epoch,
                        "config": asdict(config),
                        "state_dict": best_state,
                        "validation": validation,
                    },
                )
            history.append(record)
            atomic_torch_save(
                last_path,
                {
                    "outer": outer,
                    "epoch": epoch,
                    "config": asdict(config),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "best_state": best_state,
                    "history": history,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device)
                    if device.type == "cuda"
                    else None,
                },
            )
            print(
                f"E205 outer={outer} epoch={epoch} "
                f"train={record['train']['loss']:.6f} "
                f"validation={validation['score']:.6f} best={best_score:.6f}",
                flush=True,
            )
        else:
            history.append(record)

    selected_model: OrderedPathCalibrator | None
    if best_epoch == 0 or best_state is None:
        selected_model = None
    else:
        model.load_state_dict(best_state)
        selected_model = model
    selected, baseline, channel, endpoint = complete_evaluation(
        selected_model, data, data.held_ids_np, config, collect_channel=True
    )
    if channel is None:
        raise RuntimeError("held channel was not collected")
    temporary_channel = fold_channel_path(outer).with_suffix(
        f".npy.tmp.{os.getpid()}"
    )
    with temporary_channel.open("wb") as handle:
        np.save(handle, channel.astype(np.complex64))
    os.replace(temporary_channel, fold_channel_path(outer))
    endpoint_payload = {
        "global_ids": data.held_ids_np.astype(np.int32),
        "fallback": np.asarray(best_epoch == 0),
    }
    if endpoint is not None:
        endpoint_payload.update(endpoint)
    temporary_endpoint = fold_endpoint_path(outer).with_suffix(
        f".npz.tmp.{os.getpid()}"
    )
    with temporary_endpoint.open("wb") as handle:
        np.savez_compressed(handle, **endpoint_payload)
    os.replace(temporary_endpoint, fold_endpoint_path(outer))
    delta = metric_delta(selected, baseline)
    report = {
        "experiment": VERSION,
        "status": "complete",
        "outer": outer,
        "validation_fold": (outer + config.validation_fold_offset) % 5,
        "configuration": asdict(config),
        "cache": cache_record,
        "split": {
            "fit_count": int(len(data.fit_ids_np)),
            "validation_count": int(len(data.validation_ids_np)),
            "held_count": int(len(data.held_ids_np)),
            "zero_label_overlap": True,
        },
        "selection": {
            "epoch0_is_v7_fallback": True,
            "best_epoch": best_epoch,
            "validation_baseline": baseline_validation,
            "validation_selected": best_metrics,
        },
        "held": {"selected": selected, "baseline": baseline, "delta": delta},
        "history": history,
        "outputs": {
            "channel": PATHS.relative(fold_channel_path(outer)),
            "endpoint": PATHS.relative(fold_endpoint_path(outer)),
            "best_checkpoint": PATHS.relative(fold_best_checkpoint_path(outer))
            if best_epoch > 0
            else None,
        },
        "performance": {
            "data_load_seconds": data_seconds,
            "total_seconds": float(time.time() - started),
            "gpu": gpu_record(device),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "measured_bottleneck": "strict 17-carrier field cache"
            if float(cache_record.get("seconds", 0.0)) > float(time.time() - started) / 2.0
            else "mass/port set-model training",
        },
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def aggregate(device: torch.device, config: Config) -> dict[str, Any]:
    started = time.time()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    output = np.lib.format.open_memmap(
        FULL_CHANNEL,
        mode="w+",
        dtype=np.complex64,
        shape=(len(query_ids), MP * MH * MV, N_UE, S_SC),
    )
    coverage = np.zeros(len(query_ids), dtype=np.int8)
    fold_reports = []
    for outer in range(5):
        report_path = fold_report_path(outer)
        if not report_path.is_file() or not fold_channel_path(outer).is_file():
            raise FileNotFoundError(f"missing E205 outer={outer} result")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise RuntimeError(f"E205 outer={outer} is not complete")
        global_ids = np.load(fold_endpoint_path(outer), allow_pickle=False)[
            "global_ids"
        ].astype(np.int64)
        expected = query_ids[fold_id == outer]
        if not np.array_equal(global_ids, expected):
            raise RuntimeError(f"E205 outer={outer} canonical row mismatch")
        rows = np.flatnonzero(fold_id == outer)
        output[rows] = np.load(fold_channel_path(outer), mmap_mode="r")
        coverage[rows] += 1
        fold_reports.append(
            {
                "outer": outer,
                "count": int(len(rows)),
                "best_epoch": int(report["selection"]["best_epoch"]),
                "selected": report["held"]["selected"],
                "baseline": report["held"]["baseline"],
                "delta": report["held"]["delta"],
                "performance": report["performance"],
            }
        )
    output.flush()
    if not np.all(coverage == 1):
        raise RuntimeError(f"invalid fixed500 coverage {np.unique(coverage, return_counts=True)}")
    prediction = torch.as_tensor(np.asarray(output), device=device)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    selected, _ = score_channels(prediction, truth, config.score_batch_size)
    base = torch.as_tensor(
        np.array(np.load(BASE_V7, mmap_mode="r"), dtype=np.complex64, copy=True),
        device=device,
    )
    baseline, _ = score_channels(base, truth, config.score_batch_size)
    if abs(float(baseline["score"]) - EXPECTED_V7_SCORE) > 1e-8:
        raise RuntimeError("aggregate V7 baseline mismatch")
    delta = metric_delta(selected, baseline)
    positive_folds = int(sum(row["delta"]["score"] > 0.0 for row in fold_reports))
    worst_fold_delta = float(min(row["delta"]["score"] for row in fold_reports))
    component_protection = bool(
        delta["pas"] >= -config.maximum_component_drop
        and delta["pdp"] >= -config.maximum_component_drop
        and delta["nmse_score"] >= -config.maximum_component_drop
    )
    passed = bool(
        float(selected["score"]) >= config.required_score
        and positive_folds >= config.required_positive_folds
        and worst_fold_delta >= config.minimum_fold_delta
        and component_protection
    )
    total_fold_seconds = sum(
        float(row["performance"]["total_seconds"]) for row in fold_reports
    )
    report = {
        "experiment": VERSION,
        "stage": "aggregate",
        "status": "fixed500_passed" if passed else "fixed500_failed",
        "configuration": asdict(config),
        "baseline": baseline,
        "selected": selected,
        "delta": delta,
        "folds": fold_reports,
        "gate": {
            "required_score": config.required_score,
            "required_positive_folds": config.required_positive_folds,
            "minimum_fold_delta": config.minimum_fold_delta,
            "maximum_component_drop": config.maximum_component_drop,
            "positive_folds": positive_folds,
            "worst_fold_delta": worst_fold_delta,
            "component_protection": component_protection,
            "passed": passed,
        },
        "data_lineage": {
            "official_train_h": True,
            "official_train_positions": True,
            "official_map_setup": True,
            "canonical_v7_base": PATHS.relative(BASE_V7),
            "sionna_used": False,
            "v8_v15_derived_inputs_used": False,
            "predicted_heads": ["path_mass", "8_port_power"],
            "frozen": ["path_angle", "path_delay", "v7_phase", "point_power"],
        },
        "performance": {
            "aggregate_seconds": float(time.time() - started),
            "sum_fold_wall_seconds": total_fold_seconds,
            "parallel_wall_time_recorded_by_runner": None,
            "gpu": gpu_record(device),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "outputs": {"channel": PATHS.relative(FULL_CHANNEL)},
        "failure_conclusion_if_failed": (
            "The current 256-path official-map candidate family plus ordered interactions "
            "and strict Clean carrier evidence is insufficient to recover V12-level score "
            "through mass/port calibration alone."
        ),
    }
    atomic_json(FULL_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    config = Config(
        carrier_batch_size=args.carrier_batch_size,
        train_batch_size=args.train_batch_size,
        epochs=args.epochs,
    )
    if args.outer not in range(5):
        raise ValueError(f"invalid outer fold {args.outer}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    torch.manual_seed(config.seed + args.outer)
    np.random.seed((config.seed + args.outer) % (2**32 - 1))
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args, device)
    elif args.stage == "fold":
        for path in required_inputs():
            if not path.is_file():
                raise FileNotFoundError(path)
        if not COMMON_STATE.is_file():
            raise FileNotFoundError(COMMON_STATE)
        train_fold(args.outer, config, device, args.force)
    else:
        aggregate(device, config)


if __name__ == "__main__":
    main()
