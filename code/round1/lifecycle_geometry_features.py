"""Point-cloud-compatible geometry features for path lifecycle models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import analyze_v366_local_eikonal_pdp as v366


RAY_DISTANCE = 100.0
RAY_STEP = 2.0
FIXED_DIRECTIONS = 8


@dataclass
class RasterScene:
    height: np.ndarray
    x_min: float
    y_min: float
    resolution: float


def raster_from_boxes(
    boxes: np.ndarray,
    x_min: float = -150.0,
    x_max: float = 150.0,
    y_min: float = -150.0,
    y_max: float = 150.0,
    resolution: float = 2.0,
) -> RasterScene:
    width = int(math.ceil((x_max - x_min) / resolution))
    height = int(math.ceil((y_max - y_min) / resolution))
    grid = np.zeros((height, width), dtype=np.float32)
    for cx, cy, box_width, box_depth, box_height in boxes:
        ix0 = max(int(math.floor((cx - box_width / 2.0 - x_min) / resolution)), 0)
        ix1 = min(int(math.ceil((cx + box_width / 2.0 - x_min) / resolution)), width)
        iy0 = max(int(math.floor((cy - box_depth / 2.0 - y_min) / resolution)), 0)
        iy1 = min(int(math.ceil((cy + box_depth / 2.0 - y_min) / resolution)), height)
        if ix1 > ix0 and iy1 > iy0:
            grid[iy0:iy1, ix0:ix1] = np.maximum(
                grid[iy0:iy1, ix0:ix1], float(box_height)
            )
    return RasterScene(grid, x_min, y_min, resolution)


def raster_from_pointcloud_scene(scene: dict[str, np.ndarray]) -> RasterScene:
    return RasterScene(
        np.nan_to_num(scene["grid_z_max"], nan=0.0).astype(np.float32),
        float(scene["x_min"]),
        float(scene["y_min"]),
        float(scene["resolution"]),
    )


def sample_height(scene: RasterScene, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.floor((xy[:, 0] - scene.x_min) / scene.resolution).astype(np.int64)
    iy = np.floor((xy[:, 1] - scene.y_min) / scene.resolution).astype(np.int64)
    valid = (
        (ix >= 0)
        & (ix < scene.height.shape[1])
        & (iy >= 0)
        & (iy < scene.height.shape[0])
    )
    safe_x = np.clip(ix, 0, scene.height.shape[1] - 1)
    safe_y = np.clip(iy, 0, scene.height.shape[0] - 1)
    return scene.height[safe_y, safe_x], valid


def ray_summary(
    scene: RasterScene,
    origin: np.ndarray,
    direction: np.ndarray,
    ue_z: float,
    distance: float = RAY_DISTANCE,
) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.zeros(6, dtype=np.float32)
    unit = direction / norm
    samples = np.arange(RAY_STEP, distance + 0.5 * RAY_STEP, RAY_STEP)
    xy = origin[None, :] + samples[:, None] * unit[None, :]
    height, valid = sample_height(scene, xy)
    occupied = valid & (height >= ue_z + 0.2)
    if np.any(occupied):
        first = float(samples[int(np.argmax(occupied))]) / max(distance, 1e-8)
        occupied_height = height[occupied]
        max_height = float(np.max(occupied_height)) / 40.0
        mean_height = float(np.mean(occupied_height)) / 40.0
    else:
        first = 1.0
        max_height = 0.0
        mean_height = 0.0
    starts = occupied & ~np.r_[False, occupied[:-1]]
    near = samples <= min(30.0, distance)
    return np.asarray(
        [
            first,
            float(np.mean(occupied)),
            float(np.mean(occupied[near])) if np.any(near) else 0.0,
            min(float(np.sum(starts)) / 8.0, 1.0),
            max_height,
            mean_height,
        ],
        dtype=np.float32,
    )


def local_density(scene: RasterScene, point: np.ndarray, radius: float) -> tuple[float, float]:
    ix = int(math.floor((point[0] - scene.x_min) / scene.resolution))
    iy = int(math.floor((point[1] - scene.y_min) / scene.resolution))
    cells = int(math.ceil(radius / scene.resolution))
    x0, x1 = max(ix - cells, 0), min(ix + cells + 1, scene.height.shape[1])
    y0, y1 = max(iy - cells, 0), min(iy + cells + 1, scene.height.shape[0])
    patch = scene.height[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0
    return float(np.mean(patch > 0.2)), float(np.max(patch)) / 40.0


def point_features(
    scene: RasterScene,
    positions: np.ndarray,
    bs: np.ndarray,
) -> np.ndarray:
    rows = []
    angles = np.linspace(0.0, 2.0 * np.pi, FIXED_DIRECTIONS, endpoint=False)
    fixed = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    for point in positions:
        to_bs = bs[:2] - point[:2]
        distance = float(np.linalg.norm(to_bs))
        radial = to_bs / max(distance, 1e-8)
        features = [
            to_bs[0] / 300.0,
            to_bs[1] / 300.0,
            distance / 300.0,
            radial[0],
            radial[1],
            float(point[2]) / 40.0,
            float(bs[2]) / 40.0,
        ]
        features.extend(
            ray_summary(
                scene,
                point[:2],
                to_bs,
                float(point[2]),
                distance=max(distance, RAY_STEP),
            ).tolist()
        )
        for direction in fixed:
            summary = ray_summary(scene, point[:2], direction, float(point[2]))
            features.extend([float(summary[0]), float(summary[1]), float(summary[4])])
        for radius in (12.0, 30.0, 60.0):
            features.extend(local_density(scene, point[:2], radius))
        rows.append(features)
    return np.asarray(rows, dtype=np.float32)


def family_feature(
    *,
    scene: RasterScene,
    point: np.ndarray,
    bs: np.ndarray,
    point_feature: np.ndarray,
    predicted_delay: float,
    predicted_gradient: np.ndarray,
    support_xy: np.ndarray,
    support_mass: np.ndarray,
    fit_delay_rmse: float,
    frequency: float,
) -> np.ndarray:
    gradient_norm = float(np.linalg.norm(predicted_gradient))
    if gradient_norm > 1e-8:
        direction = predicted_gradient / gradient_norm
    else:
        radial = point[:2] - bs[:2]
        direction = radial / max(float(np.linalg.norm(radial)), 1e-8)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    radial = point[:2] - bs[:2]
    radial /= max(float(np.linalg.norm(radial)), 1e-8)
    orientation_dot = float(np.dot(direction, radial))
    orientation_cross = float(direction[0] * radial[1] - direction[1] * radial[0])

    support_distance = np.linalg.norm(support_xy - point[None, :2], axis=1)
    ordered = np.sort(support_distance)
    centered = support_xy - np.mean(support_xy, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(support_xy), 1)
    eigenvalue = np.linalg.eigvalsh(covariance)
    support_statistics = [
        ordered[0] / 150.0,
        float(np.mean(ordered[: min(4, len(ordered))])) / 150.0,
        float(np.mean(ordered[: min(8, len(ordered))])) / 150.0,
        math.log1p(len(support_xy)) / 6.0,
        math.sqrt(max(float(eigenvalue[-1]), 0.0)) / 150.0,
        math.sqrt(max(float(eigenvalue[0]), 0.0)) / 150.0,
        float(np.mean(support_mass)),
        float(np.std(support_mass)),
        float(np.max(support_mass)),
        float(fit_delay_rmse) / 0.5,
    ]
    path_statistics = [
        float(predicted_delay) / 24.0,
        float(predicted_gradient[0]) / v366.MAX_GRADIENT,
        float(predicted_gradient[1]) / v366.MAX_GRADIENT,
        gradient_norm / v366.MAX_GRADIENT,
        orientation_dot,
        orientation_cross,
        (float(predicted_delay) / v366.THEORETICAL_GRADIENT) / 800.0,
        float(frequency) / 7.0e9,
    ]
    terminal = []
    for ray_direction in (direction, -direction, perpendicular, -perpendicular):
        terminal.extend(
            ray_summary(
                scene,
                point[:2],
                ray_direction,
                float(point[2]),
            ).tolist()
        )
    return np.concatenate(
        [
            np.asarray(point_feature, dtype=np.float32),
            np.asarray(path_statistics, dtype=np.float32),
            np.asarray(support_statistics, dtype=np.float32),
            np.asarray(terminal, dtype=np.float32),
        ]
    ).astype(np.float32)


def feature_dimensions() -> dict[str, int]:
    point = 7 + 6 + 3 * FIXED_DIRECTIONS + 2 * 3
    return {
        "point": point,
        "path": 8,
        "support": 10,
        "terminal": 24,
        "total": point + 8 + 10 + 24,
    }
