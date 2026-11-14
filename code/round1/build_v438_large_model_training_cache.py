#!/usr/bin/env python3
"""Build geometry and compact target memmaps for v438 large-model training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

import build_v415_route_raster_features as route_builder
from project_paths import PATHS


VERSION_NUMBER = 438
VERSION = "v438_large_model_training_cache"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
SOURCE = PATHS.cache_dir(437)
POSITIONS_SOURCE = SOURCE / "positions.npz"
SHARD_ROOT = SOURCE / "shards"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"

COUNT = 100_000
SHARD_SIZE = 1_000
MAX_PATHS = 128
MAX_DEPTH = 16
DELAY_BINS = 192
DEPTH_BINS = 17
DELAY_SCALE = DELAY_BINS * 30_000.0


def memmap_path(name: str) -> Path:
    return OUT_DIR / f"{name}.npy"


def write_memmap(name: str, dtype, shape: tuple[int, ...]) -> np.memmap:
    final = memmap_path(name)
    partial = final.with_name(final.stem + ".partial.npy")
    if partial.exists():
        partial.unlink()
    return np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)


def finish_memmap(name: str, value: np.memmap) -> None:
    value.flush()
    del value
    final = memmap_path(name)
    partial = final.with_name(final.stem + ".partial.npy")
    partial.replace(final)


def path_histograms(path_data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, ...]:
    valid = path_data["valid"].astype(bool)
    tau = path_data["tau"].astype(np.float32)
    mass = path_data["normalized_power"].astype(np.float32)
    depth = path_data["interaction_count"].astype(np.int64)
    count = len(valid)
    delay_hist = np.zeros((count, DELAY_BINS), dtype=np.float32)
    depth_hist = np.zeros((count, DEPTH_BINS), dtype=np.float32)
    los_fraction = np.zeros(count, dtype=np.float32)
    path_count = np.sum(valid, axis=1).astype(np.int16)
    for row in range(count):
        ids = np.flatnonzero(valid[row])
        if not len(ids):
            continue
        weight = mass[row, ids]
        weight_sum = float(np.sum(weight))
        if weight_sum <= 0.0:
            continue
        weight = weight / weight_sum
        delay = np.mod(tau[row, ids] * DELAY_SCALE, DELAY_BINS)
        left = np.floor(delay).astype(np.int64)
        fraction = delay - left
        np.add.at(delay_hist[row], left, weight * (1.0 - fraction))
        np.add.at(delay_hist[row], (left + 1) % DELAY_BINS, weight * fraction)
        clipped_depth = np.clip(depth[row, ids], 0, DEPTH_BINS - 1)
        np.add.at(depth_hist[row], clipped_depth, weight)
        los_fraction[row] = float(np.sum(weight[clipped_depth == 0]))
    return (
        delay_hist.astype(np.float16),
        depth_hist.astype(np.float16),
        path_count,
        los_fraction.astype(np.float16),
    )


def build_route(positions: np.ndarray) -> dict[str, object]:
    route_path = memmap_path("route_raster")
    if route_path.is_file():
        route = np.load(route_path, mmap_mode="r")
        if route.shape != (COUNT, 12, 160, 41) or route.dtype != np.float16:
            raise ValueError(f"Invalid existing route raster {route.shape}/{route.dtype}")
        return {"status": "reused", "shape": list(route.shape)}
    scene = np.load(SCENE_CACHE)
    route_builder.BATCH = 256
    started = time.time()
    route = route_builder.build_raster(positions, scene)
    if route.shape != (COUNT, 12, 160, 41) or route.dtype != np.float16:
        raise ValueError(f"Unexpected route raster {route.shape}/{route.dtype}")
    if not np.isfinite(route).all():
        raise FloatingPointError("Route raster is non-finite")
    partial = route_path.with_name(route_path.stem + ".partial.npy")
    np.save(partial, route)
    partial.replace(route_path)
    return {
        "status": "generated",
        "shape": list(route.shape),
        "seconds": float(time.time() - started),
    }


def validate_cache() -> dict[str, object]:
    expected = {
        "positions": ((COUNT, 3), np.dtype(np.float32)),
        "split": ((COUNT,), np.dtype(np.int8)),
        "route_raster": ((COUNT, 12, 160, 41), np.dtype(np.float16)),
        "joint_sh": ((COUNT, 192, 16), np.dtype(np.float16)),
        "pdp": ((COUNT, 192), np.dtype(np.float16)),
        "path_delay": ((COUNT, 192), np.dtype(np.float16)),
        "path_depth": ((COUNT, 17), np.dtype(np.float16)),
        "path_count": ((COUNT,), np.dtype(np.int16)),
        "los_fraction": ((COUNT,), np.dtype(np.float16)),
        "log_power": ((COUNT,), np.dtype(np.float32)),
        "zero_channel": ((COUNT,), np.dtype(np.bool_)),
    }
    sizes = {}
    for name, (shape, dtype) in expected.items():
        path = memmap_path(name)
        value = np.load(path, mmap_mode="r")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{name} has {value.shape}/{value.dtype}, expected {shape}/{dtype}")
        sizes[name] = path.stat().st_size
    split = np.load(memmap_path("split"), mmap_mode="r")
    zero = np.load(memmap_path("zero_channel"), mmap_mode="r")
    path_count = np.load(memmap_path("path_count"), mmap_mode="r")
    return {
        "status": "pass",
        "bytes": int(sum(sizes.values())),
        "file_bytes": sizes,
        "split_counts": np.bincount(split, minlength=3).astype(int).tolist(),
        "zero_channel_count": int(np.sum(zero)),
        "mean_path_count": float(np.mean(path_count)),
    }


def main() -> None:
    for path in (POSITIONS_SOURCE, SCENE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source = np.load(POSITIONS_SOURCE)
    positions = source["positions"].astype(np.float32)
    split = source["pretrain_split"].astype(np.int8)
    if positions.shape != (COUNT, 3) or split.shape != (COUNT,):
        raise ValueError("v437 position manifest has unexpected shape")

    simple = {"positions": positions, "split": split}
    for name, value in simple.items():
        path = memmap_path(name)
        if not path.is_file():
            np.save(path, value)

    route_report = build_route(positions)
    target_names = (
        "joint_sh",
        "pdp",
        "path_delay",
        "path_depth",
        "path_count",
        "los_fraction",
        "log_power",
        "zero_channel",
    )
    if all(memmap_path(name).is_file() for name in target_names):
        target_status = "reused"
    else:
        buffers = {
            "joint_sh": write_memmap("joint_sh", np.float16, (COUNT, 192, 16)),
            "pdp": write_memmap("pdp", np.float16, (COUNT, 192)),
            "path_delay": write_memmap("path_delay", np.float16, (COUNT, 192)),
            "path_depth": write_memmap("path_depth", np.float16, (COUNT, 17)),
            "path_count": write_memmap("path_count", np.int16, (COUNT,)),
            "los_fraction": write_memmap("los_fraction", np.float16, (COUNT,)),
            "log_power": write_memmap("log_power", np.float32, (COUNT,)),
            "zero_channel": write_memmap("zero_channel", np.bool_, (COUNT,)),
        }
        for shard_id in range(COUNT // SHARD_SIZE):
            start = shard_id * SHARD_SIZE
            end = start + SHARD_SIZE
            spectral = np.load(
                SHARD_ROOT / f"shard_{shard_id:05d}" / "spectral_targets.npz"
            )
            paths = np.load(
                SHARD_ROOT / f"shard_{shard_id:05d}" / "path_targets.npz"
            )
            delay, depth, count, los = path_histograms(paths)
            buffers["joint_sh"][start:end] = spectral["joint_sh"]
            buffers["pdp"][start:end] = spectral["pdp"]
            buffers["path_delay"][start:end] = delay
            buffers["path_depth"][start:end] = depth
            buffers["path_count"][start:end] = count
            buffers["los_fraction"][start:end] = los
            buffers["log_power"][start:end] = spectral["log_point_power"]
            buffers["zero_channel"][start:end] = (
                spectral["log_point_power"] <= np.log(1e-30) + 1e-4
            )
            print(f"v438 compact targets {end}/{COUNT}", flush=True)
        for name in target_names:
            finish_memmap(name, buffers.pop(name))
        target_status = "generated"

    validation = validate_cache()
    report = {
        "method": VERSION,
        "status": "training_cache_ready",
        "source": {
            "v437": PATHS.relative(SOURCE),
            "geometry": "official Round1_Map.ply point-cloud raster only",
        },
        "route": route_report,
        "targets": target_status,
        "validation": validation,
        "outputs": {
            name: PATHS.relative(memmap_path(name))
            for name in (
                "positions",
                "split",
                "route_raster",
                *target_names,
            )
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
