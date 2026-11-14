#!/usr/bin/env python3
"""Build power-free, geometry-ordered path tokens from the v442 dataset."""

from __future__ import annotations

import json
import time

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 445
VERSION = "v445_geometry_ordered_path_tokens"
SOURCE = PATHS.cache_dir(442) / "shards"
POSITION_SOURCE = PATHS.cache_dir(443) / "positions.npy"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"

COUNT = 200_000
SHARD_SIZE = 1_000
MAX_PATHS = 128
FEATURES = 28
BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
LIGHT_SPEED = 299_792_458.0
DELAY_SCALE = 192.0 * 30_000.0


def open_output(name: str, dtype, shape: tuple[int, ...]) -> np.memmap:
    path = OUT_DIR / f"{name}.npy"
    partial = path.with_name(path.stem + ".partial.npy")
    if partial.exists():
        partial.unlink()
    return np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)


def finish(name: str, value: np.memmap) -> None:
    value.flush()
    del value
    path = OUT_DIR / f"{name}.npy"
    path.with_name(path.stem + ".partial.npy").replace(path)


def first_last_vertices(
    vertices: np.ndarray, interactions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    active = interactions != 0
    count = np.sum(active, axis=2)
    first_index = np.argmax(active, axis=2)
    last_index = np.maximum(count - 1, 0)
    row = np.arange(len(vertices))[:, None]
    path = np.arange(vertices.shape[1])[None, :]
    first = vertices[row, path, first_index]
    last = vertices[row, path, last_index]
    missing = count == 0
    first[missing] = BS
    last[missing] = BS
    return first, last


def geometry_order(data: np.lib.npyio.NpzFile, row: int) -> np.ndarray:
    valid = np.flatnonzero(data["valid"][row])
    if not len(valid):
        return valid
    # Original rows are power-ranked. Geometry sorting removes rank leakage.
    local = np.lexsort(
        (
            data["phi_r"][row, valid].astype(np.float32),
            data["phi_t"][row, valid].astype(np.float32),
            data["interaction_count"][row, valid].astype(np.int16),
            data["tau"][row, valid].astype(np.float64),
        )
    )
    return valid[local]


def main() -> None:
    for path in (SOURCE, POSITION_SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.load(POSITION_SOURCE, mmap_mode="r")
    feature = open_output("path_candidate_feature", np.float16, (COUNT, MAX_PATHS, FEATURES))
    valid_out = open_output("path_candidate_valid", np.bool_, (COUNT, MAX_PATHS))
    mass_out = open_output("path_candidate_mass", np.float16, (COUNT, MAX_PATHS))
    feature[:] = 0
    valid_out[:] = False
    mass_out[:] = 0
    started = time.time()
    valid_total = 0
    maximum_mass_error = 0.0
    for shard_id in range(COUNT // SHARD_SIZE):
        start = shard_id * SHARD_SIZE
        end = start + SHARD_SIZE
        data = np.load(SOURCE / f"shard_{shard_id:05d}" / "path_targets.npz")
        interactions = data["interactions"].astype(np.int8)
        first, last = first_last_vertices(data["vertices"].astype(np.float32), interactions)
        for local in range(SHARD_SIZE):
            order = geometry_order(data, local)
            count = len(order)
            if not count:
                continue
            global_id = start + local
            tau = data["tau"][local, order].astype(np.float32)
            delay = tau * DELAY_SCALE
            length = tau * LIGHT_SPEED
            ue = np.asarray(positions[global_id], dtype=np.float32)
            direct = float(np.linalg.norm(ue - BS))
            depth = data["interaction_count"][local, order].astype(np.float32)
            mask = data["interaction_mask"][local, order].astype(np.int16)
            angles = [
                data[name][local, order].astype(np.float32)
                for name in ("theta_t", "phi_t", "theta_r", "phi_r")
            ]
            angle_feature = np.column_stack(
                [value for angle in angles for value in (np.sin(angle), np.cos(angle))]
            )
            first_value = first[local, order]
            last_value = last[local, order]
            global_vertex = np.column_stack(
                [
                    (first_value[:, 0] - 150.0) / 150.0,
                    (first_value[:, 1] + 50.0) / 200.0,
                    first_value[:, 2] / 50.0,
                    (last_value[:, 0] - 150.0) / 150.0,
                    (last_value[:, 1] + 50.0) / 200.0,
                    last_value[:, 2] / 50.0,
                ]
            )
            relative_vertex = np.column_stack(
                [
                    (first_value[:, 0] - ue[0]) / 300.0,
                    (first_value[:, 1] - ue[1]) / 300.0,
                    (last_value[:, 0] - ue[0]) / 300.0,
                    (last_value[:, 1] - ue[1]) / 300.0,
                ]
            )
            current = np.column_stack(
                [
                    tau * 1e6 / 8.0,
                    delay / 48.0,
                    length / 600.0,
                    np.maximum(length - direct, 0.0) / 500.0,
                    depth / 16.0,
                    (depth == 0).astype(np.float32),
                    *[((mask & bit) != 0).astype(np.float32) for bit in (1, 2, 4, 8)],
                    angle_feature,
                    global_vertex,
                    relative_vertex,
                ]
            ).astype(np.float32)
            if current.shape != (count, FEATURES):
                raise ValueError(f"Unexpected feature shape {current.shape}")
            mass = data["normalized_power"][local, order].astype(np.float32)
            mass /= max(float(np.sum(mass)), 1e-30)
            feature[global_id, :count] = np.clip(current, -4.0, 4.0).astype(np.float16)
            valid_out[global_id, :count] = True
            mass_out[global_id, :count] = mass.astype(np.float16)
            valid_total += count
            maximum_mass_error = max(maximum_mass_error, abs(float(np.sum(mass)) - 1.0))
        if (shard_id + 1) % 20 == 0:
            print(f"v445 path tokens {end}/{COUNT}", flush=True)
    for name, value in (
        ("path_candidate_feature", feature),
        ("path_candidate_valid", valid_out),
        ("path_candidate_mass", mass_out),
    ):
        finish(name, value)
    outputs = {
        name: OUT_DIR / f"{name}.npy"
        for name in ("path_candidate_feature", "path_candidate_valid", "path_candidate_mass")
    }
    validation = {}
    for name, path in outputs.items():
        value = np.load(path, mmap_mode="r")
        validation[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.isfinite(value).all()),
            "bytes": int(path.stat().st_size),
        }
    report = {
        "method": VERSION,
        "status": "geometry_ordered_power_free_inputs_ready",
        "source": PATHS.relative(SOURCE),
        "input_exclusions": ["log_power", "normalized_power", "complex_reference", "power_rank"],
        "target_only": "normalized_power copied after independent geometry ordering",
        "count": COUNT,
        "valid_paths": int(valid_total),
        "mean_paths": float(valid_total / COUNT),
        "maximum_target_mass_error": float(maximum_mass_error),
        "seconds": float(time.time() - started),
        "validation": validation,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
