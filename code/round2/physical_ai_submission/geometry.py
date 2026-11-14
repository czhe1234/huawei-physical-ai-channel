"""Official-map geometry descriptors used by the path model."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from scipy.sparse import csgraph
import trimesh

from .config import RuntimePaths
from .data import load_official_data
from .io_utils import atomic_json


GRID_RESOLUTION_M = 1.0
GRID_MARGIN_M = 24.0
WALL_DILATION_CELLS = 1
FAN_OFFSETS_M = np.asarray((-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0))
ROUTE_BINS = 48
PANORAMA_DIRECTIONS = 32
PANORAMA_HEIGHTS_M = (2.0, 8.0)
PANORAMA_MAX_DISTANCE_M = 48.0


def _grid_bounds(points: np.ndarray) -> tuple[float, float, int, int]:
    lower = np.floor(np.min(points[:, :2], axis=0) - GRID_MARGIN_M)
    upper = np.ceil(np.max(points[:, :2], axis=0) + GRID_MARGIN_M)
    width = int(round((upper[0] - lower[0]) / GRID_RESOLUTION_M)) + 1
    height = int(round((upper[1] - lower[1]) / GRID_RESOLUTION_M)) + 1
    return float(lower[0]), float(lower[1]), height, width


def _xy_to_rc(
    xy: np.ndarray, x_min: float, y_min: float, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    column = np.rint((xy[:, 0] - x_min) / GRID_RESOLUTION_M).astype(np.int64)
    row = np.rint((xy[:, 1] - y_min) / GRID_RESOLUTION_M).astype(np.int64)
    return np.clip(row, 0, height - 1), np.clip(column, 0, width - 1)


def _rasterize_walls(
    mesh: trimesh.Trimesh,
    x_min: float,
    y_min: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    z_max = np.max(triangles[:, :, 2], axis=1)
    keep = (np.abs(normals[:, 2]) < 0.65) & (z_max > 2.0)
    triangles = triangles[keep]
    z_max = z_max[keep]
    wall_count = np.zeros((height, width), dtype=np.float32)
    wall_height = np.zeros((height, width), dtype=np.float32)
    for left, right in ((0, 1), (1, 2), (2, 0)):
        start = triangles[:, left, :2]
        end = triangles[:, right, :2]
        length = np.linalg.norm(end - start, axis=1)
        steps = np.maximum(np.ceil(length / 0.45).astype(np.int32), 1)
        for step_index in range(int(np.max(steps)) + 1):
            valid = steps >= step_index
            if not np.any(valid):
                break
            fraction = step_index / np.maximum(steps[valid], 1)
            samples = start[valid] + fraction[:, None] * (end[valid] - start[valid])
            row, column = _xy_to_rc(samples, x_min, y_min, height, width)
            np.add.at(wall_count, (row, column), 1.0)
            np.maximum.at(wall_height, (row, column), z_max[valid].astype(np.float32))
    occupancy = wall_count > 0.0
    if WALL_DILATION_CELLS:
        occupancy = ndimage.binary_dilation(occupancy, iterations=WALL_DILATION_CELLS)
    return occupancy, wall_count, wall_height


def _grid_graph(cost: np.ndarray) -> sparse.csr_matrix:
    height, width = cost.shape
    ids = np.arange(height * width, dtype=np.int64).reshape(height, width)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for delta_row, delta_column, step in (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ):
        row_start = max(0, -delta_row)
        row_end = min(height, height - delta_row)
        column_start = max(0, -delta_column)
        column_end = min(width, width - delta_column)
        source = ids[row_start:row_end, column_start:column_end].ravel()
        target = ids[
            row_start + delta_row : row_end + delta_row,
            column_start + delta_column : column_end + delta_column,
        ].ravel()
        source_cost = cost[row_start:row_end, column_start:column_end].ravel()
        target_cost = cost[
            row_start + delta_row : row_end + delta_row,
            column_start + delta_column : column_end + delta_column,
        ].ravel()
        row_parts.append(source)
        column_parts.append(target)
        data_parts.append((step * 0.5 * (source_cost + target_cost)).astype(np.float32))
    return sparse.csr_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(height * width, height * width),
        dtype=np.float32,
    )


def _trace_path(node: int, predecessor: np.ndarray, maximum: int) -> np.ndarray:
    path: list[int] = []
    current = int(node)
    for _ in range(maximum):
        path.append(current)
        parent = int(predecessor[current])
        if parent < 0:
            break
        current = parent
    path.reverse()
    return np.asarray(path, dtype=np.int64)


def _point_features(
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    x_min: float,
    y_min: float,
    cost: np.ndarray,
    occupancy: np.ndarray,
    clearance: np.ndarray,
    distance: np.ndarray,
    predecessor: np.ndarray,
) -> np.ndarray:
    height, width = cost.shape
    row, column = _xy_to_rc(positions[:, :2], x_min, y_min, height, width)
    output = np.zeros((len(positions), 15), dtype=np.float32)
    for point_index in range(len(positions)):
        site = int(sites[point_index])
        node = int(row[point_index] * width + column[point_index])
        path = _trace_path(node, predecessor[site], height * width)
        direct = float(
            np.linalg.norm(positions[point_index, :2] - bs_positions[site, :2])
        )
        geodesic = float(distance[site, node] * GRID_RESOLUTION_M)
        path_row = path // width
        path_column = path % width
        if len(path) >= 2:
            step_row = np.diff(path_row).astype(np.float32)
            step_column = np.diff(path_column).astype(np.float32)
            angle = np.arctan2(step_row, step_column)
            angle_delta = (
                np.abs(np.angle(np.exp(1j * np.diff(angle))))
                if len(angle) > 1
                else np.empty(0, dtype=np.float32)
            )
            turns = float(np.sum(angle_delta > np.deg2rad(35.0)))
            first = float(angle[0])
            last = float(angle[-1])
        else:
            turns = first = last = 0.0
        path_clearance = clearance[path_row, path_column] * GRID_RESOLUTION_M
        path_cost = cost[path_row, path_column]
        output[point_index] = np.asarray(
            (
                geodesic,
                direct,
                geodesic - direct,
                geodesic / max(direct, 1e-6),
                turns,
                turns / max(len(path), 1),
                math.sin(first),
                math.cos(first),
                math.sin(last),
                math.cos(last),
                float(np.min(path_clearance)),
                float(np.mean(path_clearance)),
                float(np.mean(path_cost)),
                float(np.mean(occupancy[path_row, path_column])),
                float(len(path) * GRID_RESOLUTION_M),
            ),
            dtype=np.float32,
        )
    return output


def _unique_distances(values: np.ndarray, tolerance: float = 0.35) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return ordered[np.concatenate(([True], np.diff(ordered) > tolerance))]


def _base_visibility_geometry(
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    serving = bs_positions[np.asarray(sites, dtype=np.int64)]
    relative = positions - serving
    distance = np.linalg.norm(relative, axis=1)
    horizontal = np.linalg.norm(relative[:, :2], axis=1)
    azimuth = np.arctan2(relative[:, 1], relative[:, 0])
    elevation = np.arctan2(relative[:, 2], np.maximum(horizontal, 1e-9))
    one_hot = np.column_stack((sites == 0, sites == 1)).astype(np.float64)
    return np.column_stack(
        (
            positions,
            relative,
            distance,
            np.log1p(distance),
            np.sin(azimuth),
            np.cos(azimuth),
            np.sin(elevation),
            np.cos(elevation),
            one_hot,
        )
    ).astype(np.float32)


def _visibility_ray_features(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    origins_center = bs_positions[np.asarray(sites, dtype=np.int64)]
    delta_xy = positions[:, :2] - origins_center[:, :2]
    norm_xy = np.maximum(np.linalg.norm(delta_xy, axis=1, keepdims=True), 1e-9)
    perpendicular = np.column_stack((-delta_xy[:, 1], delta_xy[:, 0])) / norm_xy
    offsets = np.asarray((-4.0, -2.0, 0.0, 2.0, 4.0), dtype=np.float64)
    targets = np.repeat(positions[:, None, :], len(offsets), axis=1)
    targets[:, :, :2] += perpendicular[:, None] * offsets[None, :, None]
    origins = np.repeat(origins_center[:, None, :], len(offsets), axis=1)
    flat_origins = origins.reshape(-1, 3)
    vectors = targets.reshape(-1, 3) - flat_origins
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(lengths[:, None], 1e-12)
    raw = np.zeros((len(flat_origins), 4), dtype=np.float32)
    raw[:, 1:3] = 1.0
    for start in range(0, len(flat_origins), batch_size):
        end = min(start + batch_size, len(flat_origins))
        locations, ray_ids, _ = mesh.ray.intersects_location(
            flat_origins[start:end], directions[start:end], multiple_hits=True
        )
        if len(ray_ids):
            global_ids = ray_ids.astype(np.int64) + start
            distances = np.linalg.norm(locations - flat_origins[global_ids], axis=1)
            for ray in np.unique(global_ids):
                selected = distances[global_ids == ray]
                selected = selected[
                    (selected > 0.50) & (selected < lengths[ray] - 0.75)
                ]
                selected = _unique_distances(selected)
                if not len(selected):
                    continue
                raw[ray, 0] = np.float32(len(selected))
                raw[ray, 1] = np.float32(selected[0] / lengths[ray])
                raw[ray, 2] = np.float32(selected[-1] / lengths[ray])
                raw[ray, 3] = np.float32((selected[-1] - selected[0]) / lengths[ray])
    fan = raw.reshape(len(positions), len(offsets), 4)
    center = fan[:, 2]
    hit_count = fan[:, :, 0]
    first = fan[:, :, 1]
    last = fan[:, :, 2]
    span = fan[:, :, 3]
    return np.column_stack(
        (
            center,
            np.mean(hit_count, axis=1),
            np.min(hit_count, axis=1),
            np.max(hit_count, axis=1),
            np.mean(hit_count == 0.0, axis=1),
            np.mean(first, axis=1),
            np.min(first, axis=1),
            np.mean(last, axis=1),
            np.mean(span, axis=1),
            np.std(hit_count, axis=1),
        )
    ).astype(np.float32)


def _local_surface_features(mesh: trimesh.Trimesh, positions: np.ndarray) -> np.ndarray:
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    distances, indices = cKDTree(centers[:, :2]).query(
        positions[:, :2], k=16, workers=-1
    )
    nearby_height = centers[indices, 2]
    nearby_normal_z = np.abs(normals[indices, 2])
    vertical = np.flatnonzero(np.abs(normals[:, 2]) < 0.45)
    if len(vertical):
        vertical_distance, vertical_local = cKDTree(centers[vertical, :2]).query(
            positions[:, :2], k=8, workers=-1
        )
        vertical_height = centers[vertical[vertical_local], 2]
    else:
        vertical_distance = np.full((len(positions), 8), 1e3, dtype=np.float64)
        vertical_height = np.zeros((len(positions), 8), dtype=np.float64)
    return np.column_stack(
        (
            distances[:, 0],
            np.mean(distances, axis=1),
            np.mean(nearby_height, axis=1),
            np.max(nearby_height, axis=1),
            np.std(nearby_height, axis=1),
            np.mean(nearby_normal_z < 0.45, axis=1),
            vertical_distance[:, 0],
            np.mean(vertical_distance, axis=1),
            np.mean(vertical_height, axis=1),
            np.max(vertical_height, axis=1),
        )
    ).astype(np.float32)


def _route_profiles(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    serving = bs_positions[np.asarray(sites, dtype=np.int64)]
    delta = positions[:, :2] - serving[:, :2]
    norm = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
    perpendicular = np.column_stack((-delta[:, 1], delta[:, 0])) / norm
    targets = np.repeat(positions[:, None, :], len(FAN_OFFSETS_M), axis=1)
    targets[:, :, :2] += perpendicular[:, None] * FAN_OFFSETS_M[None, :, None]
    origins = np.repeat(serving[:, None, :], len(FAN_OFFSETS_M), axis=1)
    vectors = targets - origins
    lengths = np.linalg.norm(vectors, axis=2).reshape(-1)
    directions = (
        vectors / np.maximum(np.linalg.norm(vectors, axis=2)[..., None], 1e-12)
    ).reshape(-1, 3)
    origins = origins.reshape(-1, 3)
    hit = np.zeros((len(origins), ROUTE_BINS), dtype=np.float32)
    occupancy = np.zeros_like(hit)
    count = np.zeros(len(origins), dtype=np.float32)
    for start in range(0, len(origins), batch_size):
        end = min(start + batch_size, len(origins))
        locations, ray_ids, _ = mesh.ray.intersects_location(
            origins[start:end], directions[start:end], multiple_hits=True
        )
        if len(ray_ids):
            global_ray = ray_ids.astype(np.int64) + start
            distances = np.linalg.norm(locations - origins[global_ray], axis=1)
            for ray in np.unique(global_ray):
                local = distances[global_ray == ray]
                local = local[(local > 0.50) & (local < lengths[ray] - 0.75)]
                local = _unique_distances(local)
                if not len(local):
                    continue
                bins = np.floor(
                    np.clip(local / lengths[ray], 0.0, 1.0 - 1e-8) * ROUTE_BINS
                ).astype(np.int64)
                hit[ray, bins] = 1.0
                count[ray] = float(len(local))
                for pair in range(0, len(bins) - 1, 2):
                    occupancy[ray, int(bins[pair]) : int(bins[pair + 1]) + 1] = 1.0
    shape = (len(positions), len(FAN_OFFSETS_M), ROUTE_BINS)
    return (
        hit.reshape(shape),
        occupancy.reshape(shape),
        count.reshape(len(positions), -1),
    )


def _panorama_profiles(
    mesh: trimesh.Trimesh, positions: np.ndarray, batch_size: int
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, PANORAMA_DIRECTIONS, endpoint=False)
    horizontal = np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles))
    )
    profiles = np.ones(
        (len(positions), len(PANORAMA_HEIGHTS_M), PANORAMA_DIRECTIONS),
        dtype=np.float32,
    )
    for height_index, height in enumerate(PANORAMA_HEIGHTS_M):
        origins = np.repeat(positions[:, None, :], PANORAMA_DIRECTIONS, axis=1)
        origins[:, :, 2] = height
        directions = np.broadcast_to(horizontal, origins.shape)
        flat_origins = origins.reshape(-1, 3)
        flat_directions = directions.reshape(-1, 3)
        output = np.ones(len(flat_origins), dtype=np.float32)
        for start in range(0, len(flat_origins), batch_size):
            end = min(start + batch_size, len(flat_origins))
            locations, ray_ids, _ = mesh.ray.intersects_location(
                flat_origins[start:end], flat_directions[start:end], multiple_hits=False
            )
            if len(ray_ids):
                ids = start + ray_ids.astype(np.int64)
                distance = np.linalg.norm(locations - flat_origins[ids], axis=1)
                output[ids] = np.minimum(
                    distance / PANORAMA_MAX_DISTANCE_M, 1.0
                ).astype(np.float32)
        profiles[:, height_index] = output.reshape(len(positions), -1)
    return profiles


def _route_statistics(
    hit: np.ndarray,
    occupancy: np.ndarray,
    count: np.ndarray,
    panorama: np.ndarray,
) -> np.ndarray:
    opposite = np.minimum(panorama, np.roll(panorama, panorama.shape[-1] // 2, axis=-1))
    return np.column_stack(
        (
            hit.mean(axis=(1, 2)),
            hit[:, hit.shape[1] // 2].mean(axis=1),
            occupancy.mean(axis=(1, 2)),
            occupancy[:, occupancy.shape[1] // 2].mean(axis=1),
            count.mean(axis=1),
            count.max(axis=1),
            count.std(axis=1),
            panorama.mean(axis=(1, 2)),
            panorama.min(axis=(1, 2)),
            panorama.std(axis=(1, 2)),
            np.mean(panorama < 0.999, axis=(1, 2)),
            opposite.mean(axis=(1, 2)),
            opposite.max(axis=(1, 2)),
        )
    ).astype(np.float32)


def _assign_regime(
    train_features: np.ndarray,
    query_features: np.ndarray,
    train_sites: np.ndarray,
    query_sites: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    columns = np.asarray((13, 12, 3, 17, 25, 11), dtype=np.int64)
    weights = np.asarray((0.25, 0.23, 0.20, 0.14, 0.10, -0.08), dtype=np.float32)
    train_risk = np.zeros(len(train_features), dtype=np.float32)
    query_risk = np.zeros(len(query_features), dtype=np.float32)
    train_regime = np.zeros(len(train_features), dtype=np.int8)
    query_regime = np.zeros(len(query_features), dtype=np.int8)
    diagnostics: dict[str, Any] = {"sites": {}}
    for site in np.unique(train_sites):
        fit = np.flatnonzero(train_sites == site)
        query = np.flatnonzero(query_sites == site)
        reference = train_features[fit][:, columns]
        median = np.median(reference, axis=0)
        q25, q75 = np.quantile(reference, (0.25, 0.75), axis=0)
        scale = np.maximum(q75 - q25, 1e-6)
        train_standard = np.clip(
            (train_features[fit][:, columns] - median) / scale, -2.5, 3.5
        )
        query_standard = np.clip(
            (query_features[query][:, columns] - median) / scale, -2.5, 3.5
        )
        train_risk[fit] = train_standard @ weights
        query_risk[query] = query_standard @ weights
        thresholds = np.quantile(train_risk[fit], (0.25, 0.50, 0.75))
        train_regime[fit] = np.digitize(train_risk[fit], thresholds).astype(np.int8)
        query_regime[query] = np.digitize(query_risk[query], thresholds).astype(np.int8)
        diagnostics["sites"][str(int(site))] = {
            "thresholds": thresholds.tolist(),
            "train_count": int(len(fit)),
            "query_counts": [
                int(np.sum(query_regime[query] == code)) for code in range(4)
            ],
        }
    return train_risk, query_risk, train_regime, query_regime, diagnostics


def build_geometry_regime(
    paths: RuntimePaths,
    force: bool = False,
    ray_batch_size: int = 8192,
) -> dict[str, np.ndarray]:
    output = paths.work_dir / "geometry_regime.npz"
    if output.is_file() and not force:
        with np.load(output, allow_pickle=False) as loaded:
            required = {
                "train_features",
                "test_features",
                "train_visibility",
                "test_visibility",
                "train_route_occupancy",
                "test_route_occupancy",
                "train_panorama",
                "test_panorama",
            }
            if required.issubset(loaded.files):
                return {
                    name: np.array(loaded[name], copy=True) for name in loaded.files
                }
    started = time.time()
    data = load_official_data(paths.data_dir)
    positions = np.concatenate((data.train_positions, data.test_positions), axis=0)
    sites = np.concatenate((data.train_sites, data.test_sites), axis=0)
    mesh = trimesh.load(data.map_path, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh).__name__)
    route_hit, route_occupancy, route_count = _route_profiles(
        mesh, positions, sites, data.setup.bs_positions, ray_batch_size
    )
    panorama = _panorama_profiles(mesh, positions, ray_batch_size)
    # The frozen models were trained from the persisted float16 map profiles.
    route_hit = route_hit.astype(np.float16)
    route_occupancy = route_occupancy.astype(np.float16)
    route_count = route_count.astype(np.float16)
    panorama = panorama.astype(np.float16)
    route = _route_statistics(
        route_hit.astype(np.float32),
        route_occupancy.astype(np.float32),
        route_count.astype(np.float32),
        panorama.astype(np.float32),
    )
    visibility = np.concatenate(
        (
            _base_visibility_geometry(positions, sites, data.setup.bs_positions),
            _visibility_ray_features(
                mesh, positions, sites, data.setup.bs_positions, ray_batch_size
            ),
            _local_surface_features(mesh, positions),
        ),
        axis=1,
    ).astype(np.float32)

    bounds = np.concatenate((positions, data.setup.bs_positions), axis=0)
    x_min, y_min, height, width = _grid_bounds(bounds)
    occupancy, wall_count, wall_height = _rasterize_walls(
        mesh, x_min, y_min, height, width
    )
    clearance = ndimage.distance_transform_edt(~occupancy).astype(np.float32)
    cost = (
        1.0
        + 3.0 * occupancy.astype(np.float32)
        + 0.30 * np.clip(np.log1p(wall_count), 0.0, 5.0)
        + 0.20 * np.clip((wall_height - 2.0) / 18.0, 0.0, 3.0)
    ).astype(np.float32)
    graph = _grid_graph(cost)
    distance = np.empty(
        (len(data.setup.bs_positions), height * width), dtype=np.float32
    )
    predecessor = np.empty(
        (len(data.setup.bs_positions), height * width), dtype=np.int32
    )
    bs_row, bs_column = _xy_to_rc(
        data.setup.bs_positions[:, :2], x_min, y_min, height, width
    )
    for site in range(len(data.setup.bs_positions)):
        source = int(bs_row[site] * width + bs_column[site])
        local_distance, local_predecessor = csgraph.dijkstra(
            graph, directed=True, indices=source, return_predecessors=True
        )
        distance[site] = local_distance.astype(np.float32)
        predecessor[site] = local_predecessor.astype(np.int32)
    geodesic = _point_features(
        positions,
        sites,
        data.setup.bs_positions,
        x_min,
        y_min,
        cost,
        occupancy,
        clearance,
        distance,
        predecessor,
    )
    features = np.concatenate((geodesic, route), axis=1).astype(np.float32)
    split = len(data.train_positions)
    train_risk, test_risk, train_regime, test_regime, diagnostics = _assign_regime(
        features[:split], features[split:], data.train_sites, data.test_sites
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        train_features=features[:split],
        test_features=features[split:],
        train_visibility=visibility[:split],
        test_visibility=visibility[split:],
        train_route_hit=route_hit[:split].astype(np.float16),
        test_route_hit=route_hit[split:].astype(np.float16),
        train_route_occupancy=route_occupancy[:split].astype(np.float16),
        test_route_occupancy=route_occupancy[split:].astype(np.float16),
        train_route_count=route_count[:split].astype(np.float16),
        test_route_count=route_count[split:].astype(np.float16),
        train_panorama=panorama[:split].astype(np.float16),
        test_panorama=panorama[split:].astype(np.float16),
        train_risk=train_risk,
        test_risk=test_risk,
        train_regime=train_regime,
        test_regime=test_regime,
    )
    temporary.replace(output)
    report = {
        "stage": "geometry_regime",
        "status": "complete",
        "feature_shape": [list(features[:split].shape), list(features[split:].shape)],
        "visibility_shape": [
            list(visibility[:split].shape),
            list(visibility[split:].shape),
        ],
        "grid_shape": [height, width],
        "wall_fraction": float(np.mean(occupancy)),
        "regime": diagnostics,
        "wall_seconds": float(time.time() - started),
        "measured_bottleneck": "map ray intersections and grid shortest paths",
    }
    atomic_json(paths.report_dir / "geometry_regime.json", report)
    return {
        "train_features": features[:split],
        "test_features": features[split:],
        "train_visibility": visibility[:split],
        "test_visibility": visibility[split:],
        "train_route_hit": route_hit[:split].astype(np.float16),
        "test_route_hit": route_hit[split:].astype(np.float16),
        "train_route_occupancy": route_occupancy[:split].astype(np.float16),
        "test_route_occupancy": route_occupancy[split:].astype(np.float16),
        "train_route_count": route_count[:split].astype(np.float16),
        "test_route_count": route_count[split:].astype(np.float16),
        "train_panorama": panorama[:split].astype(np.float16),
        "test_panorama": panorama[split:].astype(np.float16),
        "train_risk": train_risk,
        "test_risk": test_risk,
        "train_regime": train_regime,
        "test_regime": test_regime,
    }
