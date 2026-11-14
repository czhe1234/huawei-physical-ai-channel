#!/usr/bin/env python3
"""Build point-cloud component supervision from v437 Sionna path vertices."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 439
VERSION = "v439_component_path_supervision"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
SOURCE = PATHS.cache_dir(437)
BASE_CACHE = PATHS.cache_dir(438)
SHARD_ROOT = SOURCE / "shards"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"

COUNT = 100_000
SHARD_SIZE = 1_000
COMPONENT_COUNT = 80
SPECIAL_COUNT = 2
TOTAL_COMPONENTS = COMPONENT_COUNT + SPECIAL_COUNT
LOS_COMPONENT = COMPONENT_COUNT
UNMATCHED_COMPONENT = COMPONENT_COUNT + 1
ACTIVE_DELAY_BINS = 32
H_BINS = 16
DELAY_SCALE = 192 * 30_000.0
BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)


def memmap_path(name: str) -> Path:
    return OUT_DIR / f"{name}.npy"


def create_memmap(name: str, dtype, shape: tuple[int, ...]) -> np.memmap:
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


def component_geometry(scene: np.lib.npyio.NpzFile) -> np.ndarray:
    table = scene["component_table"].astype(np.float32)
    columns = scene["component_columns"].tolist()
    by_name = {str(name): index for index, name in enumerate(columns)}
    label = table[:, by_name["label"]].astype(np.int64)
    if len(table) != COMPONENT_COUNT or not np.array_equal(
        np.sort(label), np.arange(1, COMPONENT_COUNT + 1)
    ):
        raise ValueError("Point-cloud component labels are not contiguous 1..80")
    order = np.argsort(label)
    table = table[order]
    resolution = float(scene["resolution"])
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    x0 = x_min + table[:, by_name["x0"]] * resolution
    x1 = x_min + (table[:, by_name["x1"]] + 1.0) * resolution
    y0 = y_min + table[:, by_name["y0"]] * resolution
    y1 = y_min + (table[:, by_name["y1"]] + 1.0) * resolution
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    labels = scene["component_labels"].astype(np.int32)
    wall_nx = scene["grid_wall_nx"].astype(np.float64)
    wall_ny = scene["grid_wall_ny"].astype(np.float64)
    vertical = scene["grid_vertical_count"].astype(np.float64)
    normal_x = np.zeros(COMPONENT_COUNT, dtype=np.float32)
    normal_y = np.zeros(COMPONENT_COUNT, dtype=np.float32)
    concentration = np.zeros(COMPONENT_COUNT, dtype=np.float32)
    for component in range(1, COMPONENT_COUNT + 1):
        mask = labels == component
        weight = np.maximum(vertical[mask], 1.0)
        sx = float(np.sum(wall_nx[mask] * weight))
        sy = float(np.sum(wall_ny[mask] * weight))
        denominator = max(float(np.sum(weight)), 1e-8)
        normal_x[component - 1] = sx / denominator
        normal_y[component - 1] = sy / denominator
        concentration[component - 1] = np.hypot(sx, sy) / denominator
    return np.column_stack(
        [
            center_x,
            center_y,
            x0,
            x1,
            y0,
            y1,
            table[:, by_name["z_max"]],
            table[:, by_name["z_mean"]],
            table[:, by_name["cell_count"]],
            normal_x,
            normal_y,
            concentration,
        ]
    ).astype(np.float32)


def build_component_features(positions: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    center = geometry[:, :2].astype(np.float32)
    x0, x1, y0, y1 = (geometry[:, index] for index in range(2, 6))
    z_max = geometry[:, 6]
    z_mean = geometry[:, 7]
    cell_count = geometry[:, 8]
    normal_x = geometry[:, 9]
    normal_y = geometry[:, 10]
    concentration = geometry[:, 11]
    bs_delta = center - BS[None, :2]
    bs_distance = np.maximum(np.linalg.norm(bs_delta, axis=1), 1e-4)
    bs_angle = np.arctan2(bs_delta[:, 1], bs_delta[:, 0])
    bs_box_dx = np.maximum(np.maximum(x0 - BS[0], BS[0] - x1), 0.0)
    bs_box_dy = np.maximum(np.maximum(y0 - BS[1], BS[1] - y1), 0.0)
    bs_box_distance = np.hypot(bs_box_dx, bs_box_dy)

    output = np.empty((len(positions), COMPONENT_COUNT, 32), dtype=np.float16)
    static = np.column_stack(
        [
            (center[:, 0] - 150.0) / 180.0,
            (center[:, 1] + 50.0) / 220.0,
            (x1 - x0) / 200.0,
            (y1 - y0) / 200.0,
            z_max / 40.0,
            z_mean / 40.0,
            np.log1p(cell_count) / 10.0,
            normal_x,
            normal_y,
            concentration,
        ]
    ).astype(np.float32)
    route = positions[:, None, :2] - BS[None, None, :2]
    route_norm2 = np.maximum(np.sum(route * route, axis=2), 1e-6)
    for start in range(0, len(positions), 512):
        end = min(start + 512, len(positions))
        ue = positions[start:end, :2].astype(np.float32)
        delta_ue = center[None, :, :] - ue[:, None, :]
        distance_ue = np.maximum(np.linalg.norm(delta_ue, axis=2), 1e-4)
        angle_ue = np.arctan2(delta_ue[..., 1], delta_ue[..., 0])
        path_length = bs_distance[None, :] + distance_ue
        direct = np.maximum(
            np.linalg.norm(positions[start:end, :2] - BS[None, :2], axis=1),
            1e-4,
        )
        excess = np.maximum(path_length - direct[:, None], 0.0)
        ue_box_dx = np.maximum(
            np.maximum(x0[None, :] - ue[:, None, 0], ue[:, None, 0] - x1[None, :]),
            0.0,
        )
        ue_box_dy = np.maximum(
            np.maximum(y0[None, :] - ue[:, None, 1], ue[:, None, 1] - y1[None, :]),
            0.0,
        )
        ue_box_distance = np.hypot(ue_box_dx, ue_box_dy)
        relative = center[None, :, :] - BS[None, None, :2]
        projection = np.clip(
            np.sum(relative * route[start:end], axis=2) / route_norm2[start:end],
            0.0,
            1.0,
        )
        closest = BS[None, None, :2] + projection[..., None] * route[start:end]
        transverse = np.linalg.norm(center[None, :, :] - closest, axis=2)
        los_height = BS[2] + projection * (positions[start:end, 2:3] - BS[2])
        dynamic = np.stack(
            [
                delta_ue[..., 0] / 400.0,
                delta_ue[..., 1] / 400.0,
                distance_ue / 400.0,
                np.log1p(distance_ue) / 6.0,
                np.broadcast_to(bs_delta[None, :, 0] / 400.0, distance_ue.shape),
                np.broadcast_to(bs_delta[None, :, 1] / 400.0, distance_ue.shape),
                np.broadcast_to(bs_distance[None, :] / 400.0, distance_ue.shape),
                path_length / 800.0,
                excess / 400.0,
                np.broadcast_to(np.sin(bs_angle)[None, :], distance_ue.shape),
                np.broadcast_to(np.cos(bs_angle)[None, :], distance_ue.shape),
                np.sin(angle_ue),
                np.cos(angle_ue),
                projection,
                transverse / 200.0,
                ue_box_distance / 200.0,
                np.broadcast_to(bs_box_distance[None, :] / 200.0, distance_ue.shape),
                los_height / 40.0,
                (z_max[None, :] - los_height) / 40.0,
                np.exp(-transverse / 20.0),
                np.exp(-distance_ue / 40.0),
                np.broadcast_to(np.exp(-bs_distance)[None, :], distance_ue.shape),
            ],
            axis=2,
        )
        output[start:end] = np.concatenate(
            [np.broadcast_to(static[None, :, :], (end - start, *static.shape)), dynamic],
            axis=2,
        ).astype(np.float16)
        print(f"v439 component features {end}/{len(positions)}", flush=True)
    return output


def build_path_targets(scene: np.lib.npyio.NpzFile) -> dict[str, object]:
    buffers = {
        "component_mass": create_memmap(
            "component_mass", np.float16, (COUNT, TOTAL_COMPONENTS)
        ),
        "component_delay": create_memmap(
            "component_delay",
            np.float16,
            (COUNT, TOTAL_COMPONENTS, ACTIVE_DELAY_BINS),
        ),
        "component_h": create_memmap(
            "component_h", np.float16, (COUNT, TOTAL_COMPONENTS, H_BINS)
        ),
        "component_match_fraction": create_memmap(
            "component_match_fraction", np.float16, (COUNT,)
        ),
    }
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    labels = scene["component_labels"].astype(np.int32)
    ny, nx = labels.shape
    total_interaction_mass = 0.0
    matched_interaction_mass = 0.0
    class_mass = np.zeros(TOTAL_COMPONENTS, dtype=np.float64)
    started = time.time()
    for shard_id in range(COUNT // SHARD_SIZE):
        start = shard_id * SHARD_SIZE
        end = start + SHARD_SIZE
        paths = np.load(SHARD_ROOT / f"shard_{shard_id:05d}" / "path_targets.npz")
        valid = paths["valid"].astype(bool)
        depth = paths["interaction_count"].astype(np.int16)
        weight = paths["normalized_power"].astype(np.float32)
        tau = paths["tau"].astype(np.float32)
        first = paths["vertices"][:, :, 0, :].astype(np.float32)
        h_fingerprint = paths["tx_h_fingerprint"].astype(np.float32)
        row, path = np.nonzero(valid & np.isfinite(tau) & (weight > 0.0))
        component = np.full(len(row), UNMATCHED_COMPONENT, dtype=np.int64)
        is_los = depth[row, path] == 0
        component[is_los] = LOS_COMPONENT
        interaction = ~is_los
        xyz = first[row, path]
        finite = np.isfinite(xyz).all(axis=1)
        ix = np.zeros(len(row), dtype=np.int64)
        iy = np.zeros(len(row), dtype=np.int64)
        ix[finite] = np.floor((xyz[finite, 0] - x_min) / resolution).astype(np.int64)
        iy[finite] = np.floor((xyz[finite, 1] - y_min) / resolution).astype(np.int64)
        inside = finite & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        grid_label = np.zeros(len(row), dtype=np.int32)
        grid_label[inside] = labels[iy[inside], ix[inside]]
        matched = interaction & (grid_label >= 1) & (grid_label <= COMPONENT_COUNT)
        component[matched] = grid_label[matched] - 1

        current_mass = np.zeros((SHARD_SIZE, TOTAL_COMPONENTS), dtype=np.float32)
        current_delay = np.zeros(
            (SHARD_SIZE, TOTAL_COMPONENTS, ACTIVE_DELAY_BINS), dtype=np.float32
        )
        current_h = np.zeros(
            (SHARD_SIZE, TOTAL_COMPONENTS, H_BINS), dtype=np.float32
        )
        current_weight = weight[row, path]
        np.add.at(current_mass, (row, component), current_weight)
        delay = np.mod(tau[row, path] * DELAY_SCALE, 192.0)
        delay = np.clip(delay, 0.0, ACTIVE_DELAY_BINS - 1.0001)
        left = np.floor(delay).astype(np.int64)
        fraction = delay - left
        right = np.minimum(left + 1, ACTIVE_DELAY_BINS - 1)
        np.add.at(
            current_delay,
            (row, component, left),
            current_weight * (1.0 - fraction),
        )
        np.add.at(
            current_delay,
            (row, component, right),
            current_weight * fraction,
        )
        for h_bin in range(H_BINS):
            np.add.at(
                current_h[:, :, h_bin],
                (row, component),
                current_weight * h_fingerprint[row, path, h_bin],
            )
        denominator = np.maximum(np.sum(current_mass, axis=1, keepdims=True), 1e-12)
        current_mass /= denominator
        current_delay /= denominator[:, :, None]
        current_h /= denominator[:, :, None]
        interaction_mass = np.sum(
            current_weight[interaction], dtype=np.float64
        )
        matched_mass = np.sum(current_weight[matched], dtype=np.float64)
        row_interaction = np.zeros(SHARD_SIZE, dtype=np.float32)
        row_matched = np.zeros(SHARD_SIZE, dtype=np.float32)
        np.add.at(row_interaction, row[interaction], current_weight[interaction])
        np.add.at(row_matched, row[matched], current_weight[matched])
        match_fraction = row_matched / np.maximum(row_interaction, 1e-12)
        buffers["component_mass"][start:end] = current_mass.astype(np.float16)
        buffers["component_delay"][start:end] = current_delay.astype(np.float16)
        buffers["component_h"][start:end] = current_h.astype(np.float16)
        buffers["component_match_fraction"][start:end] = match_fraction.astype(np.float16)
        class_mass += np.sum(current_mass, axis=0, dtype=np.float64)
        total_interaction_mass += interaction_mass
        matched_interaction_mass += matched_mass
        print(f"v439 component targets {end}/{COUNT}", flush=True)
    for name in tuple(buffers):
        finish_memmap(name, buffers.pop(name))
    return {
        "seconds": float(time.time() - started),
        "matched_interaction_mass_fraction": float(
            matched_interaction_mass / max(total_interaction_mass, 1e-12)
        ),
        "active_component_count": int(np.sum(class_mass[:COMPONENT_COUNT] > 0.0)),
        "los_mean_mass": float(class_mass[LOS_COMPONENT] / COUNT),
        "unmatched_mean_mass": float(class_mass[UNMATCHED_COMPONENT] / COUNT),
    }


def validate() -> dict[str, object]:
    expected = {
        "component_features": ((COUNT, COMPONENT_COUNT, 32), np.dtype(np.float16)),
        "component_mass": ((COUNT, TOTAL_COMPONENTS), np.dtype(np.float16)),
        "component_delay": (
            (COUNT, TOTAL_COMPONENTS, ACTIVE_DELAY_BINS),
            np.dtype(np.float16),
        ),
        "component_h": ((COUNT, TOTAL_COMPONENTS, H_BINS), np.dtype(np.float16)),
        "component_match_fraction": ((COUNT,), np.dtype(np.float16)),
        "component_geometry": ((COMPONENT_COUNT, 12), np.dtype(np.float32)),
    }
    result: dict[str, object] = {"status": "pass", "files": {}}
    for name, (shape, dtype) in expected.items():
        path = memmap_path(name)
        value = np.load(path, mmap_mode="r")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{name}: {value.shape}/{value.dtype}, expected {shape}/{dtype}")
        flat = value.reshape(-1)
        index = np.linspace(0, len(flat) - 1, min(100_000, len(flat)), dtype=np.int64)
        if not np.isfinite(flat[index]).all():
            raise FloatingPointError(f"{name} contains non-finite sampled values")
        result["files"][name] = {
            "shape": list(shape),
            "dtype": str(dtype),
            "bytes": int(path.stat().st_size),
        }
    result["bytes"] = int(
        sum(value["bytes"] for value in result["files"].values())
    )
    return result


def main() -> None:
    for path in (
        SCENE_CACHE,
        BASE_CACHE / "positions.npy",
        SHARD_ROOT / "shard_00000" / "path_targets.npz",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scene = np.load(SCENE_CACHE)
    geometry_path = memmap_path("component_geometry")
    if geometry_path.is_file():
        geometry = np.load(geometry_path).astype(np.float32)
    else:
        geometry = component_geometry(scene)
        np.save(geometry_path, geometry)
    feature_path = memmap_path("component_features")
    if feature_path.is_file():
        feature_status: dict[str, object] = {"status": "reused"}
    else:
        positions = np.load(BASE_CACHE / "positions.npy", mmap_mode="r")
        started = time.time()
        features = build_component_features(np.asarray(positions), geometry)
        partial = feature_path.with_name(feature_path.stem + ".partial.npy")
        np.save(partial, features)
        partial.replace(feature_path)
        feature_status = {
            "status": "generated",
            "seconds": float(time.time() - started),
        }
    target_names = (
        "component_mass",
        "component_delay",
        "component_h",
        "component_match_fraction",
    )
    if all(memmap_path(name).is_file() for name in target_names):
        target_status: dict[str, object] = {"status": "reused"}
    else:
        target_status = {"status": "generated", **build_path_targets(scene)}
    report = {
        "method": VERSION,
        "status": "component_supervision_ready",
        "source": {
            "paths": PATHS.relative(SOURCE),
            "geometry": PATHS.relative(SCENE_CACHE),
        },
        "classes": {
            "pointcloud_components": COMPONENT_COUNT,
            "los_class": LOS_COMPONENT,
            "unmatched_class": UNMATCHED_COMPONENT,
            "active_delay_bins": ACTIVE_DELAY_BINS,
        },
        "features": feature_status,
        "targets": target_status,
        "validation": validate(),
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
