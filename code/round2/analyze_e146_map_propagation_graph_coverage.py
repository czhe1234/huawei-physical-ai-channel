#!/usr/bin/env python3
"""Audit finite-budget path coverage from an official-map propagation graph."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
from scipy.optimize import differential_evolution, linear_sum_assignment
from scipy.spatial import cKDTree
import trimesh

import analyze_e143_effective_surface_identity as e143
import analyze_v4_map_virtual_source_phase as map_phase
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import MH, MV, S_SC, component_energy


VERSION = "Round2-E146-map-propagation-graph-coverage"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "map_propagation_graph_coverage"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_map_propagation_graph_coverage"
GRAPH_CACHE = ARTIFACT_DIR / "official_map_propagation_graph.npz"
CANDIDATE_CACHE = ARTIFACT_DIR / "train_map_graph_candidates_k256.npz"
METRICS = ARTIFACT_DIR / "map_graph_coverage_metrics.npz"
REPORT = ANALYSIS_DIR / "map_graph_coverage.json"
MAP_PATH = PATHS.round2 / "Round2_Map.ply"
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

TYPE_DIRECT = 0
TYPE_GROUND = 1
TYPE_REFLECTION = 2
TYPE_CORNER_1 = 3
TYPE_CORNER_2 = 4
TYPE_CORNER_3 = 5
TYPE_NAMES = (
    "direct",
    "ground",
    "one_reflection",
    "one_corner",
    "two_corner",
    "three_corner",
)
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    candidate_budget: int = 256
    ground_quota: int = 2
    reflection_quota: int = 64
    corner_quota: int = 189
    corner_rounding_m: float = 0.20
    minimum_corner_height_m: float = 1.0
    minimum_corner_angle_deg: float = 20.0
    maximum_graph_edge_m: float = 90.0
    graph_neighbors: int = 32
    graph_routes_per_depth: int = 3
    maximum_corner_depth: int = 3
    query_corner_neighbors: int = 192
    visibility_height_m: float = 2.25
    visibility_endpoint_margin_m: float = 0.75
    maximum_excess_length_m: float = 180.0
    reflection_tangent_margin_m: float = 0.75
    reflection_height_margin_m: float = 0.75
    interaction_penalty_m: float = 3.0
    turn_penalty_m: float = 4.0
    horizontal_scale: float = 1.5
    vertical_scale: float = 1.0
    delay_scale: float = 2.0
    match_cost_limit: float = 1.0
    match_bandwidth: float = 0.25
    minimum_truth_fraction: float = 1e-5
    control_repeats: int = 8
    control_distance_strata: int = 8
    array_fit_maxiter: int = 300
    array_fit_popsize: int = 18
    seed: int = 20_260_809_146


@dataclass(frozen=True)
class GraphData:
    node_xy: np.ndarray
    node_z_min: np.ndarray
    node_z_max: np.ndarray
    node_strength: np.ndarray
    edge_left: np.ndarray
    edge_right: np.ndarray
    edge_length: np.ndarray
    bs_visible: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-graph", action="store_true")
    parser.add_argument("--force-candidates", action="store_true")
    parser.add_argument("--control-repeats", type=int, default=Config.control_repeats)
    return parser.parse_args()


def circular_delta(value: np.ndarray, target: np.ndarray, period: float) -> np.ndarray:
    return np.remainder(value - target + period / 2.0, period) - period / 2.0


def direction_angles(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = np.asarray(direction, dtype=np.float64)
    unit /= np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-12)
    theta = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
    phi = np.arctan2(unit[..., 1], unit[..., 0])
    return theta, phi


def load_mesh() -> trimesh.Trimesh:
    mesh = trimesh.load(MAP_PATH, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(mesh).__name__}")
    return mesh


def extract_corner_nodes(
    mesh: trimesh.Trimesh, config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    adjacent = np.asarray(mesh.face_adjacency, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    endpoints = vertices[edges]
    vector = endpoints[:, 1] - endpoints[:, 0]
    length = np.linalg.norm(vector, axis=1)
    first = normals[adjacent[:, 0]]
    second = normals[adjacent[:, 1]]
    vertical = np.abs(vector[:, 2]) / np.maximum(length, 1e-12) >= 0.90
    wall_pair = (np.abs(first[:, 2]) <= 0.20) & (np.abs(second[:, 2]) <= 0.20)
    angle = np.degrees(
        np.arccos(np.clip(np.abs(np.sum(first * second, axis=1)), -1.0, 1.0))
    )
    keep = (
        vertical
        & wall_pair
        & (length >= config.minimum_corner_height_m)
        & (angle >= config.minimum_corner_angle_deg)
    )
    if not np.any(keep):
        raise RuntimeError("official mesh produced no vertical corner nodes")
    xy = endpoints[keep, :, :2].mean(axis=1)
    z_min_raw = endpoints[keep, :, 2].min(axis=1)
    z_max_raw = endpoints[keep, :, 2].max(axis=1)
    angle_raw = angle[keep]
    key = np.round(xy / config.corner_rounding_m).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    count = int(np.max(inverse)) + 1
    frequency = np.bincount(inverse, minlength=count).astype(np.float64)
    node_xy = np.column_stack(
        [
            np.bincount(inverse, weights=xy[:, axis], minlength=count)
            / np.maximum(frequency, 1.0)
            for axis in range(2)
        ]
    )
    node_z_min = np.full(count, np.inf, dtype=np.float64)
    node_z_max = np.full(count, -np.inf, dtype=np.float64)
    node_strength = np.zeros(count, dtype=np.float64)
    np.minimum.at(node_z_min, inverse, z_min_raw)
    np.maximum.at(node_z_max, inverse, z_max_raw)
    np.maximum.at(node_strength, inverse, angle_raw)
    return node_xy, node_z_min, node_z_max, node_strength


def node_height(
    z_min: np.ndarray, z_max: np.ndarray, preferred: float
) -> np.ndarray:
    low = np.minimum(z_min + 0.20, z_max)
    high = np.maximum(z_max - 0.20, low)
    return np.clip(np.full_like(low, preferred), low, high)


def segment_visibility(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    target: np.ndarray,
    margin: float,
    batch_size: int = 4096,
) -> np.ndarray:
    origin = np.asarray(origin, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    vector = target - origin
    length = np.linalg.norm(vector, axis=1)
    valid = length > 2.0 * margin + 1e-6
    clear = np.ones(len(origin), dtype=bool)
    clear[~valid] = False
    for start in range(0, len(origin), batch_size):
        stop = min(start + batch_size, len(origin))
        local_length = length[start:stop]
        direction = vector[start:stop] / np.maximum(local_length[:, None], 1e-12)
        location, ray, _ = mesh.ray.intersects_location(
            origin[start:stop], direction, multiple_hits=True
        )
        if len(ray) == 0:
            continue
        ray = np.asarray(ray, dtype=np.int64)
        distance = np.linalg.norm(location - origin[start:stop][ray], axis=1)
        interior = (distance > margin) & (distance < local_length[ray] - margin)
        if np.any(interior):
            clear[start + np.unique(ray[interior])] = False
    return clear


def graph_config_payload(config: Config) -> dict[str, float | int]:
    keys = (
        "corner_rounding_m",
        "minimum_corner_height_m",
        "minimum_corner_angle_deg",
        "maximum_graph_edge_m",
        "graph_neighbors",
        "visibility_height_m",
        "visibility_endpoint_margin_m",
    )
    return {key: getattr(config, key) for key in keys}


def build_graph(
    mesh: trimesh.Trimesh,
    bs_positions: np.ndarray,
    config: Config,
    force: bool,
) -> tuple[GraphData, dict[str, object]]:
    expected = graph_config_payload(config)
    if GRAPH_CACHE.is_file() and not force:
        cached = np.load(GRAPH_CACHE, allow_pickle=False)
        cached_config = json.loads(str(cached["config_json"]))
        if cached_config == expected:
            graph = GraphData(
                node_xy=cached["node_xy"].astype(np.float64),
                node_z_min=cached["node_z_min"].astype(np.float64),
                node_z_max=cached["node_z_max"].astype(np.float64),
                node_strength=cached["node_strength"].astype(np.float64),
                edge_left=cached["edge_left"].astype(np.int32),
                edge_right=cached["edge_right"].astype(np.int32),
                edge_length=cached["edge_length"].astype(np.float64),
                bs_visible=cached["bs_visible"].astype(bool),
            )
            return graph, {"reused": True, **graph_summary(graph)}

    node_xy, z_min, z_max, strength = extract_corner_nodes(mesh, config)
    tree = cKDTree(node_xy)
    neighbor_count = min(config.graph_neighbors + 1, len(node_xy))
    distance, neighbor = tree.query(node_xy, k=neighbor_count, workers=-1)
    if neighbor_count == 1:
        distance = distance[:, None]
        neighbor = neighbor[:, None]
    left = np.repeat(np.arange(len(node_xy), dtype=np.int32), neighbor_count - 1)
    right = neighbor[:, 1:].reshape(-1).astype(np.int32)
    pair_distance = distance[:, 1:].reshape(-1)
    keep = (left < right) & (pair_distance <= config.maximum_graph_edge_m)
    pair = np.column_stack((left[keep], right[keep]))
    pair, unique = np.unique(pair, axis=0, return_index=True)
    pair_distance = pair_distance[keep][unique]
    height = node_height(z_min, z_max, config.visibility_height_m)
    edge_origin = np.column_stack((node_xy[pair[:, 0]], height[pair[:, 0]]))
    edge_target = np.column_stack((node_xy[pair[:, 1]], height[pair[:, 1]]))
    visible = segment_visibility(
        mesh,
        edge_origin,
        edge_target,
        config.visibility_endpoint_margin_m,
    )
    pair = pair[visible]
    pair_distance = pair_distance[visible]

    node_point = np.column_stack((node_xy, height))
    bs_visible = np.zeros((len(bs_positions), len(node_xy)), dtype=bool)
    for site_value, bs in enumerate(bs_positions):
        origin = np.repeat(np.asarray(bs, dtype=np.float64)[None], len(node_xy), axis=0)
        bs_visible[site_value] = segment_visibility(
            mesh,
            origin,
            node_point,
            config.visibility_endpoint_margin_m,
        )

    graph = GraphData(
        node_xy=node_xy,
        node_z_min=z_min,
        node_z_max=z_max,
        node_strength=strength,
        edge_left=pair[:, 0].astype(np.int32),
        edge_right=pair[:, 1].astype(np.int32),
        edge_length=pair_distance.astype(np.float64),
        bs_visible=bs_visible,
    )
    np.savez_compressed(
        GRAPH_CACHE,
        node_xy=graph.node_xy.astype(np.float32),
        node_z_min=graph.node_z_min.astype(np.float32),
        node_z_max=graph.node_z_max.astype(np.float32),
        node_strength=graph.node_strength.astype(np.float32),
        edge_left=graph.edge_left,
        edge_right=graph.edge_right,
        edge_length=graph.edge_length.astype(np.float32),
        bs_visible=graph.bs_visible,
        config_json=np.asarray(json.dumps(expected, sort_keys=True)),
    )
    return graph, {"reused": False, **graph_summary(graph)}


def graph_summary(graph: GraphData) -> dict[str, object]:
    degree = np.bincount(
        np.concatenate((graph.edge_left, graph.edge_right)),
        minlength=len(graph.node_xy),
    )
    return {
        "node_count": int(len(graph.node_xy)),
        "edge_count": int(len(graph.edge_left)),
        "degree_mean": float(np.mean(degree)),
        "degree_median": float(np.median(degree)),
        "isolated_node_fraction": float(np.mean(degree == 0)),
        "bs_visible_count": graph.bs_visible.sum(axis=1).astype(int).tolist(),
    }


def build_route_bank(
    graph: GraphData, bs_positions: np.ndarray, config: Config
) -> list[list[list[tuple[float, tuple[int, ...]]]]]:
    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(len(graph.node_xy))
    ]
    for left, right, length in zip(
        graph.edge_left, graph.edge_right, graph.edge_length
    ):
        adjacency[int(left)].append((int(right), float(length)))
        adjacency[int(right)].append((int(left), float(length)))

    output: list[list[list[tuple[float, tuple[int, ...]]]]] = []
    for site_value, bs in enumerate(bs_positions):
        by_depth: list[list[list[tuple[float, tuple[int, ...]]]]] = [
            [[] for _ in range(len(graph.node_xy))]
            for _ in range(config.maximum_corner_depth + 1)
        ]
        start_length = np.linalg.norm(graph.node_xy - bs[None, :2], axis=1)
        for node in np.flatnonzero(graph.bs_visible[site_value]):
            by_depth[1][int(node)] = [(float(start_length[node]), (int(node),))]
        for depth in range(2, config.maximum_corner_depth + 1):
            proposals: list[list[tuple[float, tuple[int, ...]]]] = [
                [] for _ in range(len(graph.node_xy))
            ]
            for previous_node, states in enumerate(by_depth[depth - 1]):
                for previous_length, route in states:
                    for next_node, edge_length in adjacency[previous_node]:
                        if next_node in route:
                            continue
                        proposals[next_node].append(
                            (previous_length + edge_length, route + (next_node,))
                        )
            for node, states in enumerate(proposals):
                if not states:
                    continue
                states.sort(key=lambda item: item[0])
                selected: list[tuple[float, tuple[int, ...]]] = []
                first_nodes: set[int] = set()
                for state in states:
                    first = state[1][0]
                    if first in first_nodes and len(selected) > 0:
                        continue
                    selected.append(state)
                    first_nodes.add(first)
                    if len(selected) >= config.graph_routes_per_depth:
                        break
                by_depth[depth][node] = selected
        output.append(by_depth)
    return output


def query_visible_corners(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    graph: GraphData,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    count = min(config.query_corner_neighbors, len(graph.node_xy))
    _, neighbor = cKDTree(graph.node_xy).query(
        positions[:, :2], k=count, workers=-1
    )
    if count == 1:
        neighbor = neighbor[:, None]
    neighbor = np.asarray(neighbor, dtype=np.int32)
    height = node_height(
        graph.node_z_min[neighbor],
        graph.node_z_max[neighbor],
        config.visibility_height_m,
    )
    origin = np.repeat(positions[:, None, :], count, axis=1).reshape(-1, 3)
    target = np.concatenate(
        (graph.node_xy[neighbor], height[..., None]), axis=2
    ).reshape(-1, 3)
    visible = segment_visibility(
        mesh,
        origin,
        target,
        config.visibility_endpoint_margin_m,
    ).reshape(len(positions), count)
    return neighbor, visible, {
        "query_neighbor_count": int(count),
        "visible_fraction": float(np.mean(visible)),
        "visible_count_mean": float(np.mean(visible.sum(axis=1))),
        "visible_count_min": int(np.min(visible.sum(axis=1))),
        "visible_count_max": int(np.max(visible.sum(axis=1))),
    }


def route_geometry(
    route: tuple[int, ...],
    ue: np.ndarray,
    bs: np.ndarray,
    graph: GraphData,
) -> tuple[float, np.ndarray, float]:
    xy = np.vstack((bs[None, :2], graph.node_xy[np.asarray(route)], ue[None, :2]))
    horizontal = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total_horizontal = max(float(np.sum(horizontal)), 1e-9)
    cumulative = np.cumsum(horizontal)[:-1]
    ideal_z = bs[2] + (ue[2] - bs[2]) * cumulative / total_horizontal
    route_array = np.asarray(route, dtype=np.int64)
    z = np.clip(
        ideal_z,
        graph.node_z_min[route_array] + 0.20,
        graph.node_z_max[route_array] - 0.20,
    )
    z = np.minimum(np.maximum(z, graph.node_z_min[route_array]), graph.node_z_max[route_array])
    points = np.column_stack((graph.node_xy[route_array], z))
    full = np.vstack((bs[None], points, ue[None]))
    segment = np.diff(full, axis=0)
    length = float(np.sum(np.linalg.norm(segment, axis=1)))
    unit_xy = np.diff(xy, axis=0) / np.maximum(horizontal[:, None], 1e-9)
    turn = 0.0
    if len(unit_xy) > 1:
        turn = float(
            np.sum(
                1.0
                - np.clip(np.sum(unit_xy[:-1] * unit_xy[1:], axis=1), -1.0, 1.0)
            )
        )
    return length, segment[0], turn


def append_candidate(
    arrays: dict[str, np.ndarray],
    row: int,
    slot: int,
    direction: np.ndarray,
    path_length: float,
    path_type: int,
    depth: int,
    score: float,
) -> None:
    theta, phi = direction_angles(np.asarray(direction)[None])
    arrays["theta"][row, slot] = theta[0]
    arrays["phi"][row, slot] = phi[0]
    arrays["delay"][row, slot] = np.mod(
        path_length / map_phase.METERS_PER_DELAY_BIN, S_SC
    )
    arrays["path_length"][row, slot] = path_length
    arrays["score"][row, slot] = score
    arrays["path_type"][row, slot] = path_type
    arrays["depth"][row, slot] = depth
    arrays["mask"][row, slot] = True


def reflection_candidates(
    ue: np.ndarray,
    bs: np.ndarray,
    wall: map_phase.WallBank,
    sources: np.ndarray,
    config: Config,
) -> list[tuple[float, np.ndarray, float, int]]:
    delta = ue[None] - sources
    length = np.linalg.norm(delta, axis=1)
    denominator = np.sum(delta[:, :2] * wall.normal, axis=1)
    numerator = wall.offset - np.sum(sources[:, :2] * wall.normal, axis=1)
    parameter = numerator / np.where(np.abs(denominator) > 1e-12, denominator, np.nan)
    point = sources + parameter[:, None] * delta
    tangent = np.column_stack((-wall.normal[:, 1], wall.normal[:, 0]))
    tangent_coordinate = np.sum(point[:, :2] * tangent, axis=1)
    valid = (
        np.isfinite(parameter)
        & (parameter > 0.0)
        & (parameter < 1.0)
        & (
            tangent_coordinate
            >= wall.tangent_min - config.reflection_tangent_margin_m
        )
        & (
            tangent_coordinate
            <= wall.tangent_max + config.reflection_tangent_margin_m
        )
        & (point[:, 2] >= wall.z_min - config.reflection_height_margin_m)
        & (point[:, 2] <= wall.z_max + config.reflection_height_margin_m)
    )
    direct = float(np.linalg.norm(ue - bs))
    excess = length - direct
    valid &= (excess >= -1e-5) & (excess <= config.maximum_excess_length_m)
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return []
    width = np.maximum(wall.tangent_max - wall.tangent_min, 1e-6)
    edge = np.minimum(
        tangent_coordinate - wall.tangent_min,
        wall.tangent_max - tangent_coordinate,
    ) / width
    rank_score = (
        excess[ids]
        - 0.45 * np.log1p(wall.area[ids])
        - 1.0 * np.clip(edge[ids], -1.0, 1.0)
    )
    order = ids[np.argsort(rank_score)]
    output: list[tuple[float, np.ndarray, float, int]] = []
    for wall_id in order:
        direction = point[wall_id] - bs
        if any(
            abs(length[wall_id] - item[2]) < 0.10
            and np.dot(
                direction / max(float(np.linalg.norm(direction)), 1e-12),
                item[1] / max(float(np.linalg.norm(item[1])), 1e-12),
            )
            > 0.9995
            for item in output
        ):
            continue
        output.append(
            (float(rank_score[np.flatnonzero(ids == wall_id)[0]]), direction, float(length[wall_id]), int(wall_id))
        )
        if len(output) >= config.reflection_quota:
            break
    return output


def ground_candidate(ue: np.ndarray, bs: np.ndarray, height: float) -> tuple[np.ndarray, float]:
    image = np.array(bs, copy=True)
    image[2] = 2.0 * height - bs[2]
    vector = ue - image
    parameter = (height - image[2]) / vector[2]
    point = image + parameter * vector
    return point - bs, float(np.linalg.norm(vector))


def candidate_config_payload(config: Config) -> dict[str, object]:
    keys = (
        "candidate_budget",
        "ground_quota",
        "reflection_quota",
        "corner_quota",
        "maximum_corner_depth",
        "query_corner_neighbors",
        "maximum_excess_length_m",
        "interaction_penalty_m",
        "turn_penalty_m",
    )
    return {key: getattr(config, key) for key in keys}


def build_candidates(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    wall: map_phase.WallBank,
    graph: GraphData,
    route_bank: list[list[list[tuple[float, tuple[int, ...]]]]],
    query_corner: np.ndarray,
    query_visible: np.ndarray,
    config: Config,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    expected = candidate_config_payload(config)
    if CANDIDATE_CACHE.is_file() and not force:
        cached = np.load(CANDIDATE_CACHE, allow_pickle=False)
        if json.loads(str(cached["config_json"])) == expected:
            arrays = {key: cached[key] for key in cached.files if key != "config_json"}
            return arrays, {"reused": True, **candidate_summary(arrays)}

    if 1 + config.ground_quota + config.reflection_quota + config.corner_quota != config.candidate_budget:
        raise ValueError("candidate family quotas do not sum to candidate_budget")
    shape = (len(positions), config.candidate_budget)
    arrays = {
        "theta": np.zeros(shape, dtype=np.float32),
        "phi": np.zeros(shape, dtype=np.float32),
        "delay": np.zeros(shape, dtype=np.float32),
        "path_length": np.zeros(shape, dtype=np.float32),
        "score": np.full(shape, np.inf, dtype=np.float32),
        "path_type": np.full(shape, -1, dtype=np.int8),
        "depth": np.zeros(shape, dtype=np.int8),
        "mask": np.zeros(shape, dtype=bool),
    }
    sources = {
        site_value: map_phase.image_sources(wall, bs_positions[site_value])
        for site_value in range(len(bs_positions))
    }
    corner_start = 1 + config.ground_quota + config.reflection_quota
    for row, ue in enumerate(positions):
        site_value = int(site[row])
        bs = bs_positions[site_value]
        direct_vector = ue - bs
        direct_length = float(np.linalg.norm(direct_vector))
        append_candidate(
            arrays, row, 0, direct_vector, direct_length, TYPE_DIRECT, 0, 0.0
        )
        for local, height in enumerate((0.0, -0.5)[: config.ground_quota]):
            direction, length = ground_candidate(ue, bs, height)
            append_candidate(
                arrays,
                row,
                1 + local,
                direction,
                length,
                TYPE_GROUND,
                1,
                length - direct_length,
            )

        reflected = reflection_candidates(
            ue, bs, wall, sources[site_value], config
        )
        reflection_start = 1 + config.ground_quota
        for local, (score, direction, length, _) in enumerate(reflected):
            append_candidate(
                arrays,
                row,
                reflection_start + local,
                direction,
                length,
                TYPE_REFLECTION,
                1,
                score,
            )

        graph_candidates: list[tuple[float, np.ndarray, float, int]] = []
        for endpoint, visible in zip(query_corner[row], query_visible[row]):
            if not visible:
                continue
            endpoint = int(endpoint)
            for depth in range(1, config.maximum_corner_depth + 1):
                for _, route in route_bank[site_value][depth][endpoint]:
                    length, direction, turn = route_geometry(route, ue, bs, graph)
                    excess = length - direct_length
                    if excess < -1e-5 or excess > config.maximum_excess_length_m:
                        continue
                    score = (
                        excess
                        + config.interaction_penalty_m * depth
                        + config.turn_penalty_m * turn
                        - 0.02 * float(np.sum(graph.node_strength[np.asarray(route)]))
                    )
                    graph_candidates.append((score, direction, length, depth))
        graph_candidates.sort(key=lambda item: item[0])
        selected_graph: list[tuple[float, np.ndarray, float, int]] = []
        for candidate in graph_candidates:
            direction = candidate[1]
            unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
            if any(
                abs(candidate[2] - kept[2]) < 0.10
                and np.dot(
                    unit,
                    kept[1] / max(float(np.linalg.norm(kept[1])), 1e-12),
                )
                > 0.9995
                for kept in selected_graph
            ):
                continue
            selected_graph.append(candidate)
            if len(selected_graph) >= config.corner_quota:
                break
        for local, (score, direction, length, depth) in enumerate(selected_graph):
            append_candidate(
                arrays,
                row,
                corner_start + local,
                direction,
                length,
                TYPE_CORNER_1 + depth - 1,
                depth,
                score,
            )
        if (row + 1) % 250 == 0 or row + 1 == len(positions):
            print(f"E146 candidates {row + 1}/{len(positions)}", flush=True)

    np.savez_compressed(
        CANDIDATE_CACHE,
        **arrays,
        config_json=np.asarray(json.dumps(expected, sort_keys=True)),
    )
    return arrays, {"reused": False, **candidate_summary(arrays)}


def candidate_summary(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    mask = arrays["mask"].astype(bool)
    path_type = arrays["path_type"].astype(np.int8)
    return {
        "count_mean": float(np.mean(mask.sum(axis=1))),
        "count_median": float(np.median(mask.sum(axis=1))),
        "count_min": int(np.min(mask.sum(axis=1))),
        "count_max": int(np.max(mask.sum(axis=1))),
        "points_at_budget_fraction": float(np.mean(mask.sum(axis=1) == mask.shape[1])),
        "family_count_mean": {
            TYPE_NAMES[value]: float(np.mean(np.sum(mask & (path_type == value), axis=1)))
            for value in range(len(TYPE_NAMES))
        },
    }


def map_coordinates(
    arrays: dict[str, np.ndarray], horizontal: np.ndarray, vertical: np.ndarray
) -> np.ndarray:
    elevation = math.pi / 2.0 - arrays["theta"].astype(np.float64)
    phi = arrays["phi"].astype(np.float64)
    coordinate = np.stack(
        (
            np.mod(8.0 * np.sin(phi - horizontal[:, None]), MH),
            np.mod(4.0 * np.sin(elevation - vertical[:, None]), MV),
            arrays["delay"].astype(np.float64),
        ),
        axis=2,
    )
    return coordinate.astype(np.float32)


def select_los_observations(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    point_power: np.ndarray,
    geometric_los: np.ndarray,
) -> dict[str, np.ndarray]:
    energy = component_energy(coefficients).astype(np.float64)
    component_total = energy.sum(axis=1, keepdims=True, dtype=np.float64)
    fraction = np.divide(
        energy,
        np.maximum(component_total, TINY),
        out=np.zeros_like(energy),
    )
    vector = positions - bs_positions[site]
    distance = np.linalg.norm(vector, axis=1)
    unit = vector / np.maximum(distance[:, None], 1e-12)
    expected_delay = distance / map_phase.METERS_PER_DELAY_BIN
    delay_error = np.abs(
        circular_delta(coordinates[:, :, 2], expected_delay[:, None], S_SC)
    )
    significant = fraction >= 0.05
    score = np.where(
        significant,
        delay_error - 0.02 * np.log(np.maximum(fraction, 1e-12)),
        np.inf,
    )
    selected = np.argmin(score, axis=1)
    selected_error = delay_error[np.arange(len(positions)), selected]
    valid = (
        geometric_los
        & (point_power > TINY)
        & np.any(significant, axis=1)
        & (selected_error <= 0.35)
    )
    return {
        "valid": valid,
        "selected": selected.astype(np.int16),
        "unit": unit,
        "horizontal": coordinates[np.arange(len(positions)), selected, 0].astype(
            np.float64
        ),
        "vertical": coordinates[np.arange(len(positions)), selected, 1].astype(
            np.float64
        ),
        "weight": np.sqrt(
            np.maximum(energy[np.arange(len(positions)), selected], TINY)
        ),
        "delay_error": selected_error,
    }


def wrapped_axis_objective(
    parameter: np.ndarray,
    direction: np.ndarray,
    observed: np.ndarray,
    weight: np.ndarray,
    period: float,
) -> float:
    residual = 2.0 * math.pi * (
        direction @ parameter[:3] + parameter[3] - observed
    ) / period
    return float(np.sum(weight * (1.0 - np.cos(residual))))


def fit_unconstrained_axis(
    direction: np.ndarray,
    observed: np.ndarray,
    weight: np.ndarray,
    period: float,
    vector_bound: float,
    seed: int,
    config: Config,
) -> np.ndarray:
    normalized_weight = weight / max(float(np.sum(weight)), TINY)
    result = differential_evolution(
        wrapped_axis_objective,
        [(-vector_bound, vector_bound)] * 3 + [(-period / 2.0, period / 2.0)],
        args=(direction, observed, normalized_weight, period),
        seed=seed % (2**32 - 1),
        popsize=config.array_fit_popsize,
        maxiter=config.array_fit_maxiter,
        tol=1e-9,
        polish=True,
        workers=1,
    )
    return np.asarray(result.x, dtype=np.float64)


def spherical_vector(azimuth: float, elevation: float, norm: float) -> np.ndarray:
    cosine = math.cos(elevation)
    return norm * np.asarray(
        (cosine * math.cos(azimuth), cosine * math.sin(azimuth), math.sin(elevation)),
        dtype=np.float64,
    )


def fit_fixed_norm_axis(
    direction: np.ndarray,
    observed: np.ndarray,
    weight: np.ndarray,
    period: float,
    norm: float,
    seed: int,
    config: Config,
) -> np.ndarray:
    normalized_weight = weight / max(float(np.sum(weight)), TINY)

    def objective(parameter: np.ndarray) -> float:
        vector = spherical_vector(parameter[0], parameter[1], norm)
        full = np.concatenate((vector, parameter[2:3]))
        return wrapped_axis_objective(
            full, direction, observed, normalized_weight, period
        )

    result = differential_evolution(
        objective,
        [(-math.pi, math.pi), (-math.pi / 2.0, math.pi / 2.0), (-period / 2.0, period / 2.0)],
        seed=seed % (2**32 - 1),
        popsize=config.array_fit_popsize,
        maxiter=config.array_fit_maxiter,
        tol=1e-9,
        polish=True,
        workers=1,
    )
    vector = spherical_vector(result.x[0], result.x[1], norm)
    return np.concatenate((vector, result.x[2:3]))


def axis_error(
    parameter: np.ndarray,
    direction: np.ndarray,
    observed: np.ndarray,
    period: float,
) -> np.ndarray:
    predicted = np.mod(direction @ parameter[:3] + parameter[3], period)
    return np.abs(circular_delta(predicted, observed, period))


def fit_fold_array_mapping(
    observations: dict[str, np.ndarray],
    fold: np.ndarray,
    site: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    horizontal = np.zeros((len(observations["unit"]), 4), dtype=np.float64)
    vertical = np.zeros_like(horizontal)
    records: dict[str, object] = {}
    axes = (
        ("horizontal", MH, 12.0, horizontal),
        ("vertical", MV, 40.0, vertical),
    )
    valid = observations["valid"].astype(bool)
    for fold_value in np.unique(fold):
        fit = valid & (fold != fold_value)
        held = valid & (fold == fold_value)
        site_counts = {
            int(site_value): int(np.sum(fit & (site == site_value)))
            for site_value in np.unique(site)
        }
        anchor_site = max(site_counts, key=site_counts.get)
        records[str(int(fold_value))] = {
            "anchor_site": int(anchor_site),
            "fit_los_count": int(np.sum(fit)),
            "held_los_count": int(np.sum(held)),
            "sites": {},
        }
        for axis_index, (name, period, bound, output) in enumerate(axes):
            anchor_rows = np.flatnonzero(fit & (site == anchor_site))
            unconstrained = fit_unconstrained_axis(
                observations["unit"][anchor_rows],
                observations[name][anchor_rows],
                observations["weight"][anchor_rows],
                float(period),
                bound,
                config.seed + 1000 * int(fold_value) + 100 * axis_index,
                config,
            )
            norm = float(np.linalg.norm(unconstrained[:3]))
            records[str(int(fold_value))][f"{name}_shared_vector_norm"] = norm
            for site_value in np.unique(site):
                site_value = int(site_value)
                fit_rows = np.flatnonzero(fit & (site == site_value))
                held_rows = np.flatnonzero(held & (site == site_value))
                if len(fit_rows) < 12:
                    raise RuntimeError(
                        f"insufficient fold-out LOS rows fold={fold_value} site={site_value}"
                    )
                parameter = fit_fixed_norm_axis(
                    observations["unit"][fit_rows],
                    observations[name][fit_rows],
                    observations["weight"][fit_rows],
                    float(period),
                    norm,
                    config.seed
                    + 10_000 * int(fold_value)
                    + 1000 * site_value
                    + 10 * axis_index,
                    config,
                )
                output[(fold == fold_value) & (site == site_value)] = parameter
                fit_error = axis_error(
                    parameter,
                    observations["unit"][fit_rows],
                    observations[name][fit_rows],
                    float(period),
                )
                held_error = axis_error(
                    parameter,
                    observations["unit"][held_rows],
                    observations[name][held_rows],
                    float(period),
                )
                site_record = records[str(int(fold_value))]["sites"].setdefault(
                    str(site_value), {"fit_count": int(len(fit_rows)), "held_count": int(len(held_rows))}
                )
                site_record[f"{name}_parameter"] = parameter.tolist()
                site_record[f"{name}_fit_error_p50_p90"] = (
                    np.quantile(fit_error, (0.5, 0.9)).tolist()
                    if len(fit_error)
                    else []
                )
                site_record[f"{name}_held_error_p50_p90"] = (
                    np.quantile(held_error, (0.5, 0.9)).tolist()
                    if len(held_error)
                    else []
                )
    return horizontal, vertical, records


def map_coordinates_linear_array(
    arrays: dict[str, np.ndarray],
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> np.ndarray:
    theta = arrays["theta"].astype(np.float64)
    phi = arrays["phi"].astype(np.float64)
    sine = np.sin(theta)
    direction = np.stack(
        (sine * np.cos(phi), sine * np.sin(phi), np.cos(theta)), axis=2
    )
    h = np.einsum("pki,pi->pk", direction, horizontal[:, :3]) + horizontal[:, 3, None]
    v = np.einsum("pki,pi->pk", direction, vertical[:, :3]) + vertical[:, 3, None]
    return np.stack(
        (np.mod(h, MH), np.mod(v, MV), arrays["delay"].astype(np.float64)),
        axis=2,
    ).astype(np.float32)


def build_control_permutations(
    positions: np.ndarray,
    site: np.ndarray,
    fold: np.ndarray,
    bs_positions: np.ndarray,
    config: Config,
) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    distance = np.linalg.norm(positions - bs_positions[site], axis=1)
    output = [np.arange(len(positions), dtype=np.int32) for _ in range(config.control_repeats)]
    for fold_value in np.unique(fold):
        for site_value in np.unique(site):
            rows = np.flatnonzero((fold == fold_value) & (site == site_value))
            distance_order = rows[np.argsort(distance[rows])]
            strata = np.array_split(distance_order, config.control_distance_strata)
            for stratum in strata:
                if len(stratum) <= 1:
                    continue
                center = bs_positions[int(site_value), :2]
                angle = np.arctan2(
                    positions[stratum, 1] - center[1],
                    positions[stratum, 0] - center[0],
                )
                ordered = stratum[np.argsort(angle)]
                for repeat in range(config.control_repeats):
                    fraction = 0.37 + 0.26 * repeat / max(config.control_repeats - 1, 1)
                    shift = max(1, min(len(ordered) - 1, int(round(fraction * len(ordered)))))
                    output[repeat][ordered] = np.roll(ordered, shift)
    diagnostics = []
    for source in output:
        diagnostics.append(
            {
                "self_fraction": float(np.mean(source == np.arange(len(source)))),
                "median_spatial_separation_m": float(
                    np.median(
                        np.linalg.norm(
                            positions[source, :2] - positions[:, :2], axis=1
                        )
                    )
                ),
                "median_bs_distance_difference_m": float(
                    np.median(np.abs(distance[source] - distance))
                ),
            }
        )
    return output, diagnostics


def control_arrays(
    arrays: dict[str, np.ndarray], source: np.ndarray, config: Config
) -> dict[str, np.ndarray]:
    output = {key: np.array(value, copy=True) for key, value in arrays.items()}
    interaction_start = 1 + config.ground_quota
    for key in output:
        output[key][:, interaction_start:] = arrays[key][source, interaction_start:]
    return output


def support_cost(
    candidate: np.ndarray, truth: np.ndarray, config: Config
) -> np.ndarray:
    h = circular_delta(candidate[:, None, 0], truth[None, :, 0], MH)
    v = circular_delta(candidate[:, None, 1], truth[None, :, 1], MV)
    d = circular_delta(candidate[:, None, 2], truth[None, :, 2], S_SC)
    return (
        np.square(h / config.horizontal_scale)
        + np.square(v / config.vertical_scale)
        + np.square(d / config.delay_scale)
    )


def evaluate_coverage(
    candidate_coordinate: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_type: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_energy: np.ndarray,
    truth_power: np.ndarray,
    config: Config,
) -> dict[str, np.ndarray | float]:
    count = len(candidate_coordinate)
    truth_total = truth_energy.sum(axis=1, dtype=np.float64)
    truth_fraction = np.divide(
        truth_energy,
        np.maximum(truth_total[:, None], TINY),
        out=np.zeros_like(truth_energy, dtype=np.float64),
    )
    fraction_of_power = np.divide(
        truth_energy,
        np.maximum(truth_power[:, None], TINY),
        out=np.zeros_like(truth_energy, dtype=np.float64),
    )
    valid_truth = fraction_of_power >= config.minimum_truth_fraction
    point_coverage = np.zeros(count, dtype=np.float64)
    point_quality = np.zeros(count, dtype=np.float64)
    nearest_full = np.zeros(count, dtype=np.float64)
    nearest_hd = np.zeros(count, dtype=np.float64)
    nearest_d = np.zeros(count, dtype=np.float64)
    family_mass = np.zeros((count, len(TYPE_NAMES)), dtype=np.float64)
    matched_count = np.zeros(count, dtype=np.int16)
    for row in np.flatnonzero(truth_total > TINY):
        candidate_ids = np.flatnonzero(candidate_mask[row])
        truth_ids = np.flatnonzero(valid_truth[row])
        if len(candidate_ids) == 0 or len(truth_ids) == 0:
            continue
        candidate = candidate_coordinate[row, candidate_ids]
        truth = truth_coordinate[row, truth_ids]
        cost = support_cost(candidate, truth, config)
        h = circular_delta(candidate[:, None, 0], truth[None, :, 0], MH)
        d = circular_delta(candidate[:, None, 2], truth[None, :, 2], S_SC)
        hd_cost = np.square(h / config.horizontal_scale) + np.square(
            d / config.delay_scale
        )
        d_cost = np.square(d / config.delay_scale)
        mass = truth_fraction[row, truth_ids]
        nearest_full[row] = float(np.sum(mass * (np.min(cost, axis=0) <= config.match_cost_limit)))
        nearest_hd[row] = float(np.sum(mass * (np.min(hd_cost, axis=0) <= config.match_cost_limit)))
        nearest_d[row] = float(np.sum(mass * (np.min(d_cost, axis=0) <= config.match_cost_limit)))
        utility = mass[None] * np.exp(-0.5 * cost / config.match_bandwidth)
        utility[cost > config.match_cost_limit] = 0.0
        candidate_local, truth_local = linear_sum_assignment(-utility)
        accepted = utility[candidate_local, truth_local] > 0.0
        candidate_local = candidate_local[accepted]
        truth_local = truth_local[accepted]
        if len(candidate_local) == 0:
            continue
        assigned_mass = mass[truth_local]
        assigned_cost = cost[candidate_local, truth_local]
        point_coverage[row] = float(np.sum(assigned_mass))
        point_quality[row] = float(
            np.sum(assigned_mass * np.exp(-0.5 * assigned_cost / config.match_bandwidth))
        )
        matched_count[row] = len(candidate_local)
        assigned_type = candidate_type[row, candidate_ids[candidate_local]]
        for type_value in range(len(TYPE_NAMES)):
            family_mass[row, type_value] = float(
                np.sum(assigned_mass[assigned_type == type_value])
            )
    nonzero = truth_total > TINY
    weight = truth_total[nonzero]

    def weighted(value: np.ndarray) -> float:
        return float(np.sum(value[nonzero] * weight) / max(float(np.sum(weight)), TINY))

    return {
        "point_coverage": point_coverage.astype(np.float32),
        "point_quality": point_quality.astype(np.float32),
        "nearest_full": nearest_full.astype(np.float32),
        "nearest_hd": nearest_hd.astype(np.float32),
        "nearest_d": nearest_d.astype(np.float32),
        "family_mass": family_mass.astype(np.float32),
        "matched_count": matched_count,
        "global_energy_coverage": weighted(point_coverage),
        "global_quality_coverage": weighted(point_quality),
        "global_nearest_full": weighted(nearest_full),
        "global_nearest_hd": weighted(nearest_hd),
        "global_nearest_d": weighted(nearest_d),
    }


def fold_summary(
    value: np.ndarray,
    fold: np.ndarray,
    truth_energy: np.ndarray,
) -> list[float]:
    total = truth_energy.sum(axis=1, dtype=np.float64)
    output = []
    for fold_value in np.unique(fold):
        rows = (fold == fold_value) & (total > TINY)
        output.append(
            float(
                np.sum(value[rows] * total[rows])
                / max(float(np.sum(total[rows])), TINY)
            )
        )
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(control_repeats=args.control_repeats)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for required in (MAP_PATH, COMPONENTS, FULL_FOLD, LOS_LABELS):
        if not required.is_file():
            raise FileNotFoundError(required)

    setup, positions, _, site, _ = load_positions()
    component = np.load(COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_coefficient = component["coefficients"].astype(np.complex64)
    truth_energy = component_energy(truth_coefficient).astype(np.float64)
    truth_power = component["point_power"].astype(np.float64)
    fold = np.load(FULL_FOLD, allow_pickle=False).astype(np.int8)
    if len(fold) != len(positions):
        raise ValueError(f"fold length {len(fold)} != position length {len(positions)}")

    mesh = load_mesh()
    wall = map_phase.build_wall_bank(map_phase.Config())
    graph, graph_record = build_graph(
        mesh, setup.bs_positions, config, args.force_graph
    )
    route_bank = build_route_bank(graph, setup.bs_positions, config)
    query_corner, query_visible, visibility_record = query_visible_corners(
        mesh, positions, graph, config
    )
    candidates, candidate_record = build_candidates(
        positions,
        site,
        setup.bs_positions,
        wall,
        graph,
        route_bank,
        query_corner,
        query_visible,
        config,
        args.force_candidates,
    )

    horizontal, vertical, calibration = e143.build_orientations(
        positions,
        site,
        setup.bs_positions,
        fold,
        truth_coordinate,
        truth_coefficient,
        truth_power,
    )
    actual_coordinate = map_coordinates(candidates, horizontal, vertical)
    actual = evaluate_coverage(
        actual_coordinate,
        candidates["mask"],
        candidates["path_type"],
        truth_coordinate,
        truth_energy,
        truth_power,
        config,
    )

    geometric_los = np.load(LOS_LABELS, allow_pickle=False)[
        "train_center_clear"
    ].astype(bool)
    los_observations = select_los_observations(
        positions,
        site,
        setup.bs_positions,
        truth_coordinate,
        truth_coefficient,
        truth_power,
        geometric_los,
    )
    horizontal_axis, vertical_axis, axis_calibration = fit_fold_array_mapping(
        los_observations, fold, site, config
    )
    linear_coordinate = map_coordinates_linear_array(
        candidates, horizontal_axis, vertical_axis
    )
    linear_actual = evaluate_coverage(
        linear_coordinate,
        candidates["mask"],
        candidates["path_type"],
        truth_coordinate,
        truth_energy,
        truth_power,
        config,
    )

    permutations, control_diagnostics = build_control_permutations(
        positions, site, fold, setup.bs_positions, config
    )
    controls = []
    linear_controls = []
    for repeat, source in enumerate(permutations):
        control = control_arrays(candidates, source, config)
        control_coordinate = map_coordinates(control, horizontal, vertical)
        metrics = evaluate_coverage(
            control_coordinate,
            control["mask"],
            control["path_type"],
            truth_coordinate,
            truth_energy,
            truth_power,
            config,
        )
        controls.append(metrics)
        linear_control_coordinate = map_coordinates_linear_array(
            control, horizontal_axis, vertical_axis
        )
        linear_metrics = evaluate_coverage(
            linear_control_coordinate,
            control["mask"],
            control["path_type"],
            truth_coordinate,
            truth_energy,
            truth_power,
            config,
        )
        linear_controls.append(linear_metrics)
        print(
            f"E146 control {repeat + 1}/{len(permutations)} "
            f"orientation={metrics['global_energy_coverage']:.6f} "
            f"linear={linear_metrics['global_energy_coverage']:.6f}",
            flush=True,
        )

    control_point = np.mean(
        np.stack([item["point_coverage"] for item in linear_controls]), axis=0
    )
    actual_fold = fold_summary(
        np.asarray(linear_actual["point_coverage"]), fold, truth_energy
    )
    control_fold = fold_summary(control_point, fold, truth_energy)
    fold_delta = np.asarray(actual_fold) - np.asarray(control_fold)
    control_global = float(
        np.mean([float(item["global_energy_coverage"]) for item in linear_controls])
    )
    global_margin = float(linear_actual["global_energy_coverage"]) - control_global
    family_actual = np.sum(
        np.asarray(linear_actual["family_mass"], dtype=np.float64)
        * truth_energy.sum(axis=1, keepdims=True),
        axis=0,
    ) / max(float(np.sum(truth_energy)), TINY)
    family_control = np.mean(
        np.stack(
            [
                np.sum(
                    np.asarray(item["family_mass"], dtype=np.float64)
                    * truth_energy.sum(axis=1, keepdims=True),
                    axis=0,
                )
                / max(float(np.sum(truth_energy)), TINY)
                for item in linear_controls
            ]
        ),
        axis=0,
    )

    np.savez_compressed(
        METRICS,
        fold=fold,
        site=site.astype(np.int8),
        orientation_point_coverage=np.asarray(actual["point_coverage"]),
        orientation_nearest_full=np.asarray(actual["nearest_full"]),
        orientation_nearest_hd=np.asarray(actual["nearest_hd"]),
        orientation_nearest_d=np.asarray(actual["nearest_d"]),
        actual_point_coverage=np.asarray(linear_actual["point_coverage"]),
        actual_point_quality=np.asarray(linear_actual["point_quality"]),
        actual_nearest_full=np.asarray(linear_actual["nearest_full"]),
        actual_nearest_hd=np.asarray(linear_actual["nearest_hd"]),
        actual_nearest_d=np.asarray(linear_actual["nearest_d"]),
        actual_family_mass=np.asarray(linear_actual["family_mass"]),
        control_point_coverage=control_point.astype(np.float32),
        control_repeat_coverage=np.asarray(
            [item["global_energy_coverage"] for item in linear_controls], dtype=np.float64
        ),
        horizontal_orientation=horizontal.astype(np.float32),
        vertical_orientation=vertical.astype(np.float32),
        horizontal_array_parameter=horizontal_axis.astype(np.float32),
        vertical_array_parameter=vertical_axis.astype(np.float32),
    )

    passed = bool(
        float(linear_actual["global_energy_coverage"]) >= 0.85
        and global_margin >= 0.15
        and np.all(fold_delta > 0.0)
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_only_information_gate",
        "primary_hypothesis": (
            "A finite official-map visibility graph contains the delay-angle locations of "
            "the dominant official-H paths beyond direct and one-bounce geometry."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "v12_used": False,
        "config": asdict(config),
        "graph": graph_record,
        "query_visibility": visibility_record,
        "candidates": candidate_record,
        "orientation_array_calibration": calibration,
        "los_wrapped_linear_array_calibration": {
            "los_observation_count": int(np.sum(los_observations["valid"])),
            "folds": axis_calibration,
            "strict_fold_out": True,
        },
        "matching": {
            "assignment": "mass-weighted one-to-one Hungarian",
            "one_candidate_at_most_one_truth": True,
            "one_truth_at_most_one_candidate": True,
            "support_scales_h_v_d": [
                config.horizontal_scale,
                config.vertical_scale,
                config.delay_scale,
            ],
            "cost_limit": config.match_cost_limit,
            "bandwidth": config.match_bandwidth,
        },
        "orientation_mapping_baseline": {
            "global_energy_coverage": float(actual["global_energy_coverage"]),
            "global_quality_coverage": float(actual["global_quality_coverage"]),
            "nearest_full_coverage": float(actual["global_nearest_full"]),
            "nearest_h_delay_coverage": float(actual["global_nearest_hd"]),
            "nearest_delay_coverage": float(actual["global_nearest_d"]),
        },
        "actual": {
            "global_energy_coverage": float(linear_actual["global_energy_coverage"]),
            "global_quality_coverage": float(linear_actual["global_quality_coverage"]),
            "nearest_full_coverage": float(linear_actual["global_nearest_full"]),
            "nearest_h_delay_coverage": float(linear_actual["global_nearest_hd"]),
            "nearest_delay_coverage": float(linear_actual["global_nearest_d"]),
            "fold_energy_coverage": actual_fold,
            "matched_count_nonzero_mean": float(
                np.mean(np.asarray(linear_actual["matched_count"])[truth_power > TINY])
            ),
            "family_energy_coverage": {
                name: float(value) for name, value in zip(TYPE_NAMES, family_actual)
            },
        },
        "matched_control": {
            "global_energy_coverage_mean": control_global,
            "global_energy_coverage_repeats": [
                float(item["global_energy_coverage"]) for item in linear_controls
            ],
            "fold_energy_coverage": control_fold,
            "diagnostics": control_diagnostics,
            "family_energy_coverage": {
                name: float(value) for name, value in zip(TYPE_NAMES, family_control)
            },
        },
        "causal_margin": {
            "global_percentage_points": 100.0 * global_margin,
            "fold_percentage_points": (100.0 * fold_delta).tolist(),
            "positive_folds": int(np.sum(fold_delta > 0.0)),
        },
        "decision": {
            "passed": passed,
            "required_energy_coverage": 0.85,
            "required_control_margin": 0.15,
            "required_positive_folds": 5,
            "train_e147_topk_wave_packet_network": passed,
        },
        "outputs": {
            "graph": PATHS.relative(GRAPH_CACHE),
            "candidates": PATHS.relative(CANDIDATE_CACHE),
            "metrics": PATHS.relative(METRICS),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
