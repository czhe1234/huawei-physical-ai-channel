#!/usr/bin/env python3
"""Build dual-site geodesic and corridor features from the official map."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
import trimesh

from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "regime_lifecycle_support"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_regime_lifecycle_support"
CACHE = ARTIFACT_DIR / "geodesic_corridor_features.npz"
REPORT = ANALYSIS_DIR / "geodesic_corridor_features.json"
ROUTE_CACHE = (
    PATHS.artifacts / "round2" / "route_topology" / "round2_route_topology_features.npz"
)

GRID_RESOLUTION_M = 1.0
GRID_MARGIN_M = 24.0
WALL_DILATION_CELLS = 1


def _grid_bounds(points: np.ndarray) -> tuple[float, float, int, int]:
    lower = np.floor(np.min(points[:, :2], axis=0) - GRID_MARGIN_M)
    upper = np.ceil(np.max(points[:, :2], axis=0) + GRID_MARGIN_M)
    width = int(round((upper[0] - lower[0]) / GRID_RESOLUTION_M)) + 1
    height = int(round((upper[1] - lower[1]) / GRID_RESOLUTION_M)) + 1
    return float(lower[0]), float(lower[1]), height, width


def _xy_to_rc(
    xy: np.ndarray, x_min: float, y_min: float, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    col = np.rint((xy[:, 0] - x_min) / GRID_RESOLUTION_M).astype(np.int64)
    row = np.rint((xy[:, 1] - y_min) / GRID_RESOLUTION_M).astype(np.int64)
    return np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)


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
    vertical = (np.abs(normals[:, 2]) < 0.65) & (z_max > 2.0)
    triangles = triangles[vertical]
    z_max = z_max[vertical]

    wall_count = np.zeros((height, width), dtype=np.float32)
    wall_height = np.zeros((height, width), dtype=np.float32)
    for edge_start, edge_end in ((0, 1), (1, 2), (2, 0)):
        start = triangles[:, edge_start, :2]
        end = triangles[:, edge_end, :2]
        length = np.linalg.norm(end - start, axis=1)
        steps = np.maximum(np.ceil(length / 0.45).astype(np.int32), 1)
        for step_index in range(int(np.max(steps)) + 1):
            valid = steps >= step_index
            if not np.any(valid):
                break
            fraction = step_index / np.maximum(steps[valid], 1)
            samples = start[valid] + fraction[:, None] * (end[valid] - start[valid])
            row, col = _xy_to_rc(samples, x_min, y_min, height, width)
            np.add.at(wall_count, (row, col), 1.0)
            np.maximum.at(wall_height, (row, col), z_max[valid].astype(np.float32))

    occupancy = wall_count > 0.0
    if WALL_DILATION_CELLS > 0:
        occupancy = ndimage.binary_dilation(
            occupancy, iterations=WALL_DILATION_CELLS
        )
    return occupancy, wall_count, wall_height


def _grid_graph(cost: np.ndarray) -> sparse.csr_matrix:
    height, width = cost.shape
    ids = np.arange(height * width, dtype=np.int64).reshape(height, width)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for delta_row, delta_col, step in (
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
        col_start = max(0, -delta_col)
        col_end = min(width, width - delta_col)
        source = ids[row_start:row_end, col_start:col_end].ravel()
        target = ids[
            row_start + delta_row : row_end + delta_row,
            col_start + delta_col : col_end + delta_col,
        ].ravel()
        source_cost = cost[row_start:row_end, col_start:col_end].ravel()
        target_cost = cost[
            row_start + delta_row : row_end + delta_row,
            col_start + delta_col : col_end + delta_col,
        ].ravel()
        row_parts.append(source)
        col_parts.append(target)
        data_parts.append(
            (step * 0.5 * (source_cost + target_cost)).astype(np.float32)
        )
    return sparse.csr_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
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


def _route_statistics(route: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    hit = np.asarray(route[f"{prefix}_route_hit"], dtype=np.float32)
    occupancy = np.asarray(route[f"{prefix}_route_occupancy"], dtype=np.float32)
    count = np.asarray(route[f"{prefix}_route_count"], dtype=np.float32)
    panorama = np.asarray(route[f"{prefix}_panorama"], dtype=np.float32)
    opposite = np.minimum(panorama, np.roll(panorama, panorama.shape[-1] // 2, axis=-1))
    return np.column_stack(
        [
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
        ]
    ).astype(np.float32)


def _point_features(
    positions: np.ndarray,
    site: np.ndarray,
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
    row, col = _xy_to_rc(positions[:, :2], x_min, y_min, height, width)
    output = np.zeros((len(positions), 15), dtype=np.float32)
    maximum = height * width
    for point_index in range(len(positions)):
        site_value = int(site[point_index])
        node = int(row[point_index] * width + col[point_index])
        path = _trace_path(node, predecessor[site_value], maximum)
        direct = float(
            np.linalg.norm(
                positions[point_index, :2] - bs_positions[site_value, :2]
            )
        )
        geodesic = float(distance[site_value, node] * GRID_RESOLUTION_M)
        path_row = path // width
        path_col = path % width
        if len(path) >= 2:
            step_row = np.diff(path_row).astype(np.float32)
            step_col = np.diff(path_col).astype(np.float32)
            angle = np.arctan2(step_row, step_col)
            angle_delta = (
                np.abs(np.angle(np.exp(1j * np.diff(angle))))
                if len(angle) > 1
                else np.empty(0, dtype=np.float32)
            )
            turns = float(np.sum(angle_delta > np.deg2rad(35.0)))
            first = float(angle[0])
            last = float(angle[-1])
        else:
            turns = 0.0
            first = 0.0
            last = 0.0
        path_clearance = clearance[path_row, path_col] * GRID_RESOLUTION_M
        path_cost = cost[path_row, path_col]
        path_occupancy = occupancy[path_row, path_col]
        output[point_index] = np.asarray(
            [
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
                float(np.mean(path_occupancy)),
                float(len(path) * GRID_RESOLUTION_M),
            ],
            dtype=np.float32,
        )
    return output


def build_features(force: bool = False) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if CACHE.is_file() and not force:
        cached = np.load(CACHE, allow_pickle=False)
        return (
            cached["train_features"].astype(np.float32),
            cached["test_features"].astype(np.float32),
            tuple(cached["feature_names"].astype(str).tolist()),
        )
    if not ROUTE_CACHE.is_file():
        raise FileNotFoundError(
            f"missing {PATHS.relative(ROUTE_CACHE)}; run build_route_topology_features.py"
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    all_positions = np.concatenate([train_positions, test_positions], axis=0)
    all_site = np.concatenate([train_site, test_site], axis=0)
    bounds_points = np.concatenate([all_positions, setup.bs_positions], axis=0)
    x_min, y_min, height, width = _grid_bounds(bounds_points)
    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh).__name__)
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
    distance = np.empty((len(setup.bs_positions), height * width), dtype=np.float32)
    predecessor = np.empty((len(setup.bs_positions), height * width), dtype=np.int32)
    bs_row, bs_col = _xy_to_rc(
        setup.bs_positions[:, :2], x_min, y_min, height, width
    )
    for site_value in range(len(setup.bs_positions)):
        source = int(bs_row[site_value] * width + bs_col[site_value])
        site_distance, site_predecessor = csgraph.dijkstra(
            graph,
            directed=True,
            indices=source,
            return_predecessors=True,
        )
        distance[site_value] = site_distance.astype(np.float32)
        predecessor[site_value] = site_predecessor.astype(np.int32)

    geodesic = _point_features(
        all_positions,
        all_site,
        setup.bs_positions,
        x_min,
        y_min,
        cost,
        occupancy,
        clearance,
        distance,
        predecessor,
    )
    route = np.load(ROUTE_CACHE, allow_pickle=False)
    train_route = _route_statistics(route, "train")
    test_route = _route_statistics(route, "test")
    route_features = np.concatenate([train_route, test_route], axis=0)
    features = np.concatenate([geodesic, route_features], axis=1).astype(np.float32)
    names = (
        "geodesic_cost_distance",
        "direct_distance",
        "geodesic_excess",
        "geodesic_ratio",
        "turn_count",
        "turn_rate",
        "first_direction_sin",
        "first_direction_cos",
        "last_direction_sin",
        "last_direction_cos",
        "bottleneck_clearance",
        "mean_clearance",
        "mean_path_cost",
        "path_wall_fraction",
        "path_length",
        "route_hit_fraction",
        "center_route_hit_fraction",
        "route_occupancy_fraction",
        "center_route_occupancy_fraction",
        "route_hit_count_mean",
        "route_hit_count_max",
        "route_hit_count_std",
        "panorama_mean_clearance",
        "panorama_min_clearance",
        "panorama_clearance_std",
        "panorama_blocked_fraction",
        "opposite_clearance_mean",
        "opposite_clearance_max",
    )
    split = len(train_positions)
    np.savez_compressed(
        CACHE,
        train_features=features[:split],
        test_features=features[split:],
        feature_names=np.asarray(names, dtype="U48"),
        x_min=np.float32(x_min),
        y_min=np.float32(y_min),
        grid_resolution_m=np.float32(GRID_RESOLUTION_M),
        grid_shape=np.asarray([height, width], dtype=np.int32),
        wall_fraction=np.float32(np.mean(occupancy)),
    )
    return features[:split], features[split:], names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    train, test, names = build_features(force=args.force)
    report = {
        "experiment": "Round2-E41-geodesic-corridor-features",
        "status": "feature_cache_complete",
        "official_inputs_only": True,
        "sionna_used": False,
        "parameters": {
            "grid_resolution_m": GRID_RESOLUTION_M,
            "grid_margin_m": GRID_MARGIN_M,
            "wall_dilation_cells": WALL_DILATION_CELLS,
        },
        "features": list(names),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "finite": bool(np.isfinite(train).all() and np.isfinite(test).all()),
        "output": PATHS.relative(CACHE),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
