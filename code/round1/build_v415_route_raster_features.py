#!/usr/bin/env python3
"""Build BS-UE aligned propagation-profile rasters from the official point cloud."""

from __future__ import annotations

import json

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 415
VERSION = "v415_route_raster_features"
CACHE = PATHS.shared_features / "v415_route_raster_features.npz"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / "v415_route_raster_feature_build.json"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"

LONGITUDINAL_SAMPLES = 160
LATERAL_SAMPLES = 41
LATERAL_RADIUS = 20.0
BATCH = 20
BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)


def sample_grid(
    grid: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    valid: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    output = np.full(xi.shape, fill, dtype=np.float32)
    output[valid] = grid[yi[valid], xi[valid]]
    return output


def build_raster(
    positions: np.ndarray,
    scene: np.lib.npyio.NpzFile,
) -> np.ndarray:
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    grid_count = scene["grid_count"].astype(np.float32)
    grid_z_min = scene["grid_z_min"].astype(np.float32)
    grid_z_max = scene["grid_z_max"].astype(np.float32)
    grid_z_mean = scene["grid_z_mean"].astype(np.float32)
    grid_vertical = scene["grid_vertical_count"].astype(np.float32)
    grid_horizontal = scene["grid_horizontal_count"].astype(np.float32)
    grid_wall_nx = scene["grid_wall_nx"].astype(np.float32)
    grid_wall_ny = scene["grid_wall_ny"].astype(np.float32)
    ny, nx = grid_count.shape
    t = np.linspace(0.01, 0.99, LONGITUDINAL_SAMPLES, dtype=np.float32)
    lateral = np.linspace(
        -LATERAL_RADIUS, LATERAL_RADIUS, LATERAL_SAMPLES, dtype=np.float32
    )
    output = np.empty(
        (len(positions), 12, LONGITUDINAL_SAMPLES, LATERAL_SAMPLES),
        dtype=np.float16,
    )
    for start in range(0, len(positions), BATCH):
        end = min(start + BATCH, len(positions))
        ue = positions[start:end].astype(np.float32)
        route = ue[:, :2] - BS[None, :2]
        route_length = np.maximum(np.linalg.norm(route, axis=1), 1e-5)
        direction = route / route_length[:, None]
        normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
        center = (
            BS[None, None, :2]
            + t[None, :, None] * route[:, None, :]
        )
        query = (
            center[:, :, None, :]
            + lateral[None, None, :, None] * normal[:, None, None, :]
        )
        xi = np.floor((query[..., 0] - x_min) / resolution).astype(np.int64)
        yi = np.floor((query[..., 1] - y_min) / resolution).astype(np.int64)
        valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        count = sample_grid(grid_count, xi, yi, valid)
        z_min = sample_grid(grid_z_min, xi, yi, valid)
        z_max = sample_grid(grid_z_max, xi, yi, valid)
        z_mean = sample_grid(grid_z_mean, xi, yi, valid)
        vertical = sample_grid(grid_vertical, xi, yi, valid)
        horizontal = sample_grid(grid_horizontal, xi, yi, valid)
        wall_nx = sample_grid(grid_wall_nx, xi, yi, valid)
        wall_ny = sample_grid(grid_wall_ny, xi, yi, valid)
        wall_route = wall_nx * direction[:, None, None, 0] + wall_ny * direction[:, None, None, 1]
        wall_cross = wall_nx * normal[:, None, None, 0] + wall_ny * normal[:, None, None, 1]
        los_height = BS[2] + t[None, :] * (ue[:, 2:3] - BS[2])
        los_surplus = (z_max - los_height[:, :, None]) / 40.0
        occupied = count > 0.0
        channel = np.stack(
            [
                np.log1p(count) / 10.0,
                np.log1p(vertical) / 8.0,
                np.log1p(horizontal) / 8.0,
                z_max / 40.0,
                z_mean / 40.0,
                np.maximum(z_max - z_min, 0.0) / 40.0,
                wall_route,
                wall_cross,
                occupied.astype(np.float32),
                los_surplus,
                np.broadcast_to(t[None, :, None], count.shape),
                np.broadcast_to(
                    lateral[None, None, :] / LATERAL_RADIUS, count.shape
                ),
            ],
            axis=1,
        )
        channel = np.nan_to_num(channel, nan=0.0, posinf=4.0, neginf=-4.0)
        output[start:end] = np.clip(channel, -4.0, 4.0).astype(np.float16)
        print(f"v415 route raster {end}/{len(positions)}", flush=True)
    return output


def main() -> None:
    for path in (
        SCENE_CACHE,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scene = np.load(SCENE_CACHE)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train = build_raster(train_pos, scene)
    test = build_raster(test_pos, scene)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise FloatingPointError("route raster contains non-finite values")
    np.savez_compressed(
        CACHE,
        train=train,
        test=test,
        longitudinal=np.linspace(0.01, 0.99, LONGITUDINAL_SAMPLES).astype(np.float32),
        lateral=np.linspace(
            -LATERAL_RADIUS, LATERAL_RADIUS, LATERAL_SAMPLES
        ).astype(np.float32),
        channel_names=np.asarray(
            [
                "log_count",
                "log_vertical_count",
                "log_horizontal_count",
                "z_max",
                "z_mean",
                "z_range",
                "wall_normal_route",
                "wall_normal_cross",
                "occupied",
                "los_height_surplus",
                "longitudinal_coordinate",
                "lateral_coordinate",
            ]
        ),
    )
    report = {
        "method": VERSION,
        "source": "official Round1_Map.ply raster cache only",
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "dtype": str(train.dtype),
        "finite": True,
        "parameters": {
            "longitudinal_samples": LONGITUDINAL_SAMPLES,
            "lateral_samples": LATERAL_SAMPLES,
            "lateral_radius_m": LATERAL_RADIUS,
            "pointcloud_resolution_m": float(scene["resolution"]),
        },
        "outputs": {"cache": PATHS.relative(CACHE)},
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
