"""Geometry and visibility features derived from the official Round 2 map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "pre_sionna"
GEOMETRY_CACHE = ARTIFACT_DIR / "round2_geometry_visibility_features.npz"


@dataclass(frozen=True)
class GeometryFeatureBundle:
    train: np.ndarray
    test: np.ndarray
    names: tuple[str, ...]


def _unique_distances(values: np.ndarray, tolerance: float = 0.35) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    keep = np.concatenate([[True], np.diff(ordered) > tolerance])
    return ordered[keep]


def _ray_features(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    site_ids: np.ndarray,
    bs_positions: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Cast a five-ray fan from the serving BS to each UE position."""

    positions = np.asarray(positions, dtype=np.float64)
    origins_center = bs_positions[np.asarray(site_ids, dtype=np.int64)]
    delta_xy = positions[:, :2] - origins_center[:, :2]
    norm_xy = np.maximum(np.linalg.norm(delta_xy, axis=1, keepdims=True), 1e-9)
    perpendicular = np.column_stack([-delta_xy[:, 1], delta_xy[:, 0]]) / norm_xy
    offsets = np.asarray([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float64)

    targets = np.repeat(positions[:, None, :], len(offsets), axis=1)
    targets[:, :, :2] += perpendicular[:, None, :] * offsets[None, :, None]
    origins = np.repeat(origins_center[:, None, :], len(offsets), axis=1)
    flat_origins = origins.reshape(-1, 3)
    flat_targets = targets.reshape(-1, 3)
    vectors = flat_targets - flat_origins
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(lengths[:, None], 1e-12)

    raw = np.zeros((len(flat_origins), 4), dtype=np.float32)
    raw[:, 1] = 1.0
    raw[:, 2] = 1.0
    for start in range(0, len(flat_origins), batch_size):
        end = min(start + batch_size, len(flat_origins))
        locations, ray_ids, _ = mesh.ray.intersects_location(
            flat_origins[start:end],
            directions[start:end],
            multiple_hits=True,
        )
        if len(ray_ids):
            global_ids = ray_ids.astype(np.int64) + start
            distances = np.linalg.norm(locations - flat_origins[global_ids], axis=1)
            for local_ray in np.unique(global_ids):
                ray_distances = distances[global_ids == local_ray]
                valid = (ray_distances > 0.50) & (ray_distances < lengths[local_ray] - 0.75)
                unique = _unique_distances(ray_distances[valid])
                if len(unique) == 0:
                    continue
                raw[local_ray, 0] = np.float32(len(unique))
                raw[local_ray, 1] = np.float32(unique[0] / lengths[local_ray])
                raw[local_ray, 2] = np.float32(unique[-1] / lengths[local_ray])
                raw[local_ray, 3] = np.float32((unique[-1] - unique[0]) / lengths[local_ray])
        print(f"geometry rays {end}/{len(flat_origins)}", flush=True)

    fan = raw.reshape(len(positions), len(offsets), 4)
    center = fan[:, 2]
    hit_count = fan[:, :, 0]
    first = fan[:, :, 1]
    last = fan[:, :, 2]
    span = fan[:, :, 3]
    return np.column_stack(
        [
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
        ]
    ).astype(np.float32)


def _local_surface_features(mesh: trimesh.Trimesh, positions: np.ndarray) -> np.ndarray:
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    all_tree = cKDTree(centers[:, :2])
    distances, indices = all_tree.query(positions[:, :2], k=16, workers=-1)
    nearby_height = centers[indices, 2]
    nearby_normal_z = np.abs(normals[indices, 2])

    vertical = np.flatnonzero(np.abs(normals[:, 2]) < 0.45)
    if len(vertical):
        vertical_tree = cKDTree(centers[vertical, :2])
        vertical_distance, vertical_local = vertical_tree.query(
            positions[:, :2], k=8, workers=-1
        )
        vertical_height = centers[vertical[vertical_local], 2]
    else:
        vertical_distance = np.full((len(positions), 8), 1e3, dtype=np.float64)
        vertical_height = np.zeros((len(positions), 8), dtype=np.float64)

    return np.column_stack(
        [
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
        ]
    ).astype(np.float32)


def _base_geometry(
    positions: np.ndarray,
    site_ids: np.ndarray,
    bs_positions: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    serving = bs_positions[np.asarray(site_ids, dtype=np.int64)]
    relative = positions - serving
    distance = np.linalg.norm(relative, axis=1)
    horizontal = np.linalg.norm(relative[:, :2], axis=1)
    azimuth = np.arctan2(relative[:, 1], relative[:, 0])
    elevation = np.arctan2(relative[:, 2], np.maximum(horizontal, 1e-9))
    one_hot = np.column_stack([site_ids == 0, site_ids == 1]).astype(np.float64)
    return np.column_stack(
        [
            positions,
            relative,
            distance,
            np.log1p(distance),
            np.sin(azimuth),
            np.cos(azimuth),
            np.sin(elevation),
            np.cos(elevation),
            one_hot,
        ]
    ).astype(np.float32)


def build_geometry_features(
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    bs_positions: np.ndarray,
    cache_path: Path = GEOMETRY_CACHE,
) -> GeometryFeatureBundle:
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if cached["train"].shape[0] == len(train_positions) and cached["test"].shape[0] == len(test_positions):
            return GeometryFeatureBundle(
                train=cached["train"].astype(np.float32),
                test=cached["test"].astype(np.float32),
                names=tuple(cached["names"].astype(str).tolist()),
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(mesh).__name__}")

    positions = np.concatenate([train_positions, test_positions], axis=0)
    sites = np.concatenate([train_site, test_site], axis=0)
    base = _base_geometry(positions, sites, bs_positions)
    rays = _ray_features(mesh, positions, sites, bs_positions)
    local = _local_surface_features(mesh, positions)
    values = np.concatenate([base, rays, local], axis=1).astype(np.float32)
    names = (
        "x", "y", "z", "dx_bs", "dy_bs", "dz_bs", "bs_distance", "log_bs_distance",
        "sin_azimuth", "cos_azimuth", "sin_elevation", "cos_elevation", "site0", "site1",
        "center_hit_count", "center_first_fraction", "center_last_fraction", "center_blocked_span",
        "fan_hit_mean", "fan_hit_min", "fan_hit_max", "fan_clear_fraction", "fan_first_mean",
        "fan_first_min", "fan_last_mean", "fan_span_mean", "fan_hit_std", "surface_nearest_xy",
        "surface_mean16_xy", "surface_mean_height", "surface_max_height", "surface_height_std",
        "surface_vertical_fraction", "wall_nearest_xy", "wall_mean8_xy", "wall_mean_height",
        "wall_max_height",
    )
    if values.shape[1] != len(names):
        raise RuntimeError(f"geometry feature/name mismatch: {values.shape[1]} != {len(names)}")
    split = len(train_positions)
    np.savez_compressed(
        cache_path,
        train=values[:split],
        test=values[split:],
        names=np.asarray(names, dtype="U40"),
    )
    return GeometryFeatureBundle(values[:split], values[split:], names)
