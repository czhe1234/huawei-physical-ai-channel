#!/usr/bin/env python3
"""Build high-resolution official-map route and local-wall fingerprints."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import trimesh

from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "route_topology"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_route_topology"
CACHE = ARTIFACT_DIR / "round2_route_topology_features.npz"
REPORT = ANALYSIS_DIR / "route_topology_features.json"

FAN_OFFSETS_M = np.asarray((-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0))
ROUTE_BINS = 48
PANORAMA_DIRECTIONS = 32
PANORAMA_HEIGHTS_M = (2.0, 8.0)
PANORAMA_MAX_DISTANCE_M = 48.0


def unique_distances(values: np.ndarray, tolerance: float = 0.35) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    keep = np.concatenate([[True], np.diff(ordered) > tolerance])
    return ordered[keep]


def build_route_rays(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    serving = bs_positions[np.asarray(site, dtype=np.int64)]
    delta_xy = positions[:, :2] - serving[:, :2]
    norm_xy = np.maximum(np.linalg.norm(delta_xy, axis=1, keepdims=True), 1e-9)
    perpendicular = np.column_stack([-delta_xy[:, 1], delta_xy[:, 0]]) / norm_xy
    targets = np.repeat(positions[:, None, :], len(FAN_OFFSETS_M), axis=1)
    targets[:, :, :2] += (
        perpendicular[:, None, :] * FAN_OFFSETS_M[None, :, None]
    )
    origins = np.repeat(serving[:, None, :], len(FAN_OFFSETS_M), axis=1)
    vectors = targets - origins
    lengths = np.linalg.norm(vectors, axis=2)
    directions = vectors / np.maximum(lengths[:, :, None], 1e-12)
    return origins.reshape(-1, 3), directions.reshape(-1, 3), lengths.reshape(-1)


def route_profiles(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins, directions, lengths = build_route_rays(positions, site, bs_positions)
    hit_profile = np.zeros((len(origins), ROUTE_BINS), dtype=np.float32)
    occupancy_profile = np.zeros_like(hit_profile)
    hit_count = np.zeros(len(origins), dtype=np.float32)
    for start in range(0, len(origins), batch_size):
        end = min(start + batch_size, len(origins))
        locations, ray_ids, _ = mesh.ray.intersects_location(
            origins[start:end], directions[start:end], multiple_hits=True
        )
        if len(ray_ids):
            global_ray = ray_ids.astype(np.int64) + start
            distance = np.linalg.norm(locations - origins[global_ray], axis=1)
            for ray in np.unique(global_ray):
                selected = distance[global_ray == ray]
                selected = selected[
                    (selected > 0.50) & (selected < lengths[ray] - 0.75)
                ]
                selected = unique_distances(selected)
                if len(selected) == 0:
                    continue
                fraction = np.clip(selected / lengths[ray], 0.0, 1.0 - 1e-8)
                bins = np.floor(fraction * ROUTE_BINS).astype(np.int64)
                hit_profile[ray, bins] = 1.0
                hit_count[ray] = float(len(selected))
                for pair_start in range(0, len(bins) - 1, 2):
                    left = int(bins[pair_start])
                    right = int(bins[pair_start + 1])
                    occupancy_profile[ray, left : right + 1] = 1.0
        print(f"route rays {end}/{len(origins)}", flush=True)
    shape = (len(positions), len(FAN_OFFSETS_M), ROUTE_BINS)
    return (
        hit_profile.reshape(shape),
        occupancy_profile.reshape(shape),
        hit_count.reshape(len(positions), len(FAN_OFFSETS_M)),
    )


def panorama_profiles(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    angles = np.linspace(
        0.0, 2.0 * np.pi, PANORAMA_DIRECTIONS, endpoint=False, dtype=np.float64
    )
    horizontal = np.column_stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)]
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
                flat_origins[start:end],
                flat_directions[start:end],
                multiple_hits=False,
            )
            if len(ray_ids):
                distance = np.linalg.norm(
                    locations - flat_origins[start + ray_ids.astype(np.int64)], axis=1
                )
                output[start + ray_ids.astype(np.int64)] = np.minimum(
                    distance / PANORAMA_MAX_DISTANCE_M, 1.0
                ).astype(np.float32)
            print(
                f"panorama h={height:g} {end}/{len(flat_origins)}", flush=True
            )
        profiles[:, height_index] = output.reshape(
            len(positions), PANORAMA_DIRECTIONS
        )
    return profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE.is_file() and not args.force:
        cached = np.load(CACHE)
        payload = {
            "status": "reused",
            "cache": PATHS.relative(CACHE),
            "arrays": {
                key: {"shape": list(cached[key].shape), "dtype": str(cached[key].dtype)}
                for key in cached.files
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    setup, train_positions, test_positions, train_site, test_site = load_positions()
    positions = np.concatenate([train_positions, test_positions], axis=0)
    site = np.concatenate([train_site, test_site], axis=0)
    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh).__name__)

    route_hit, route_occupancy, route_count = route_profiles(
        mesh, positions, site, setup.bs_positions, args.batch_size
    )
    panorama = panorama_profiles(mesh, positions, args.batch_size)
    split = len(train_positions)
    np.savez_compressed(
        CACHE,
        train_route_hit=route_hit[:split].astype(np.float16),
        test_route_hit=route_hit[split:].astype(np.float16),
        train_route_occupancy=route_occupancy[:split].astype(np.float16),
        test_route_occupancy=route_occupancy[split:].astype(np.float16),
        train_route_count=route_count[:split].astype(np.float16),
        test_route_count=route_count[split:].astype(np.float16),
        train_panorama=panorama[:split].astype(np.float16),
        test_panorama=panorama[split:].astype(np.float16),
        fan_offsets_m=FAN_OFFSETS_M.astype(np.float32),
        panorama_heights_m=np.asarray(PANORAMA_HEIGHTS_M, dtype=np.float32),
    )
    report = {
        "experiment": "Round2-E11-route-topology-features",
        "status": "feature_cache_complete",
        "official_inputs_only": True,
        "sionna_used": False,
        "parameters": {
            "fan_offsets_m": FAN_OFFSETS_M.tolist(),
            "route_bins": ROUTE_BINS,
            "panorama_directions": PANORAMA_DIRECTIONS,
            "panorama_heights_m": list(PANORAMA_HEIGHTS_M),
            "panorama_max_distance_m": PANORAMA_MAX_DISTANCE_M,
        },
        "diagnostics": {
            "route_hit_fraction": float(np.mean(route_hit > 0.0)),
            "route_occupancy_fraction": float(np.mean(route_occupancy > 0.0)),
            "route_count_mean": float(np.mean(route_count)),
            "panorama_blocked_fraction": float(np.mean(panorama < 1.0)),
        },
        "output": PATHS.relative(CACHE),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
