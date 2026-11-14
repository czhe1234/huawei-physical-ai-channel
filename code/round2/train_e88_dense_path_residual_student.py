#!/usr/bin/env python3
"""Train and strictly evaluate the E88 dense path-residual student."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import trimesh

from build_geodesic_corridor_features import (
    _grid_bounds,
    _rasterize_walls,
)
from generate_e88_dense_random_sionna_teacher import SEEDS
from project_paths import PATHS
from round2.data import load_positions
from train_offline_sionna_geometry_student import (
    CORRIDOR_OFFSETS_M,
    CORRIDOR_SAMPLES,
    OOF_META,
    POLAR_DIRECTIONS,
    POLAR_RADII_M,
    fusion_metrics,
    position_features,
)
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E88-dense-path-residual-student"
DENSE_ROOT = (
    PATHS.artifacts
    / "round2"
    / "dense_random_sionna_teacher"
    / "dense_n60000_s3"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "dense_path_residual_student"
GEOMETRY_DIR = ARTIFACT_DIR / "geometry"
PRETRAIN_CHECKPOINT = ARTIFACT_DIR / "dense_path_pretrain.pt"
SEED_AGREEMENT = ARTIFACT_DIR / "dense_seed_agreement_f32.npy"
OOF_PREDICTIONS = ARTIFACT_DIR / "strict_oof_dense_path_endpoints.npz"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_dense_path_residual_student"
REPORT = ANALYSIS_DIR / "dense_path_residual_student.json"
INFORMATION_REPORT = ANALYSIS_DIR / "dense_teacher_information_ceiling.json"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
TINY = 1e-30
FIXED_TEACHER_METRIC_VERSION = 2


@dataclass(frozen=True)
class Config:
    hidden: int = 512
    path_slots: int = 32
    slot_layers: int = 3
    heads: int = 8
    feedforward: int = 1024
    dropout: float = 0.08
    residual_limit: float = 1.5
    dense_spectral_mix: float = 0.75
    grid_resolutions: tuple[int, ...] = (64, 128, 256, 512, 1024)
    grid_features: int = 4
    pretrain_epochs: int = 36
    finetune_epochs: int = 64
    validation_interval: int = 4
    pretrain_batch_size: int = 192
    finetune_batch_size: int = 128
    pretrain_learning_rate: float = 2e-4
    finetune_learning_rate: float = 1.2e-4
    weight_decay: float = 2e-4
    path_spectral_weight: float = 0.10
    support_weight: float = 0.30
    port_weight: float = 0.02
    residual_weight: float = 1e-6
    teacher_retain: float = 0.12
    random_state: int = 2026080888


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "pretrain", "screen", "strict"), default="prepare"
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--path-slots", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=36)
    parser.add_argument("--finetune-epochs", type=int, default=64)
    parser.add_argument("--pretrain-batch-size", type=int, default=192)
    parser.add_argument("--finetune-batch-size", type=int, default=128)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-pretrain", action="store_true")
    return parser.parse_args()


def geometry_paths() -> dict[str, Path]:
    return {
        "dense_polar": GEOMETRY_DIR / "dense_polar_f16.npy",
        "dense_corridor": GEOMETRY_DIR / "dense_corridor_f16.npy",
        "dense_scalar": GEOMETRY_DIR / "dense_map_scalar_f32.npy",
        "train_polar": GEOMETRY_DIR / "train_polar_f16.npy",
        "train_corridor": GEOMETRY_DIR / "train_corridor_f16.npy",
        "train_scalar": GEOMETRY_DIR / "train_map_scalar_f32.npy",
        "test_polar": GEOMETRY_DIR / "test_polar_f16.npy",
        "test_corridor": GEOMETRY_DIR / "test_corridor_f16.npy",
        "test_scalar": GEOMETRY_DIR / "test_map_scalar_f32.npy",
        "normalization": GEOMETRY_DIR / "normalization.npz",
        "metadata": GEOMETRY_DIR / "metadata.json",
    }


def required_teacher_files(seed_count: int = 3) -> tuple[Path, ...]:
    output = [DENSE_ROOT / "positions.npy", DENSE_ROOT / "site.npy"]
    for seed in SEEDS[:seed_count]:
        prefix = DENSE_ROOT / f"seed_{seed}"
        output.extend(
            (
                prefix.with_name(prefix.name + "_coordinate_f16.npy"),
                prefix.with_name(prefix.name + "_mass_f16.npy"),
                prefix.with_name(prefix.name + "_port_f16.npy"),
                prefix.with_name(prefix.name + "_mask.npy"),
                prefix.with_name(prefix.name + "_pas_f16.npy"),
                prefix.with_name(prefix.name + "_pdp_f16.npy"),
                prefix.with_name(prefix.name + "_done.npy"),
            )
        )
    return tuple(output)


def build_seed_agreement(seed_count: int = 3) -> dict[str, Any]:
    if SEED_AGREEMENT.is_file():
        value = np.load(SEED_AGREEMENT, mmap_mode="r")
        return {
            "status": "reused",
            "path": PATHS.relative(SEED_AGREEMENT),
            "count": int(len(value)),
            "mean": float(np.mean(value)),
        }
    seeds = tuple(SEEDS[:seed_count])
    pas = []
    pdp = []
    for seed in seeds:
        prefix = DENSE_ROOT / f"seed_{seed}"
        pas.append(np.load(prefix.with_name(prefix.name + "_pas_f16.npy"), mmap_mode="r"))
        pdp.append(np.load(prefix.with_name(prefix.name + "_pdp_f16.npy"), mmap_mode="r"))
    output = np.lib.format.open_memmap(
        SEED_AGREEMENT, mode="w+", dtype=np.float32, shape=(len(pas[0]),)
    )
    pairs = tuple(
        (left, right)
        for left in range(seed_count)
        for right in range(left + 1, seed_count)
    )
    for start in range(0, len(output), 512):
        end = min(start + 512, len(output))
        local_scores = []
        for left, right in pairs:
            left_pas = np.asarray(pas[left][start:end], dtype=np.float64)
            right_pas = np.asarray(pas[right][start:end], dtype=np.float64)
            pas_dot = np.sum(left_pas * right_pas, axis=(1, 2, 3))
            pas_norm = np.sqrt(
                np.sum(left_pas**2, axis=(1, 2, 3))
                * np.sum(right_pas**2, axis=(1, 2, 3))
            )
            pas_score = np.divide(
                pas_dot,
                np.where(pas_norm > 0.0, pas_norm, 1e-12),
            ).mean(axis=1)
            left_pdp = np.asarray(pdp[left][start:end], dtype=np.float64)
            right_pdp = np.asarray(pdp[right][start:end], dtype=np.float64)
            pdp_dot = np.sum(left_pdp * right_pdp, axis=3)
            pdp_norm = np.sqrt(
                np.sum(left_pdp**2, axis=3) * np.sum(right_pdp**2, axis=3)
            )
            pdp_score = np.divide(
                pdp_dot,
                np.where(pdp_norm > 0.0, pdp_norm, 1e-12),
            ).mean(axis=(1, 2))
            local_scores.append(0.5 * (pas_score + pdp_score))
        output[start:end] = np.mean(local_scores, axis=0).astype(np.float32)
        if end % 4096 < 512 or end == len(output):
            output.flush()
            print(f"E88 seed agreement {end}/{len(output)}", flush=True)
    output.flush()
    return {
        "status": "created",
        "path": PATHS.relative(SEED_AGREEMENT),
        "count": int(len(output)),
        "mean": float(np.mean(output)),
    }


def _normalized_root_blend(
    value: np.ndarray,
    weight: np.ndarray,
    norm_axes: tuple[int, ...],
) -> np.ndarray:
    # value is [seed, point, neighbor, ...], weight is [point, neighbor].
    root = np.sqrt(np.maximum(value, 0.0, dtype=np.float32))
    blended = np.einsum(
        "pk,spk...->p...", weight / value.shape[0], root, optimize=True
    )
    blended = np.square(np.maximum(blended, 0.0), dtype=np.float32)
    total = blended.sum(axis=norm_axes, keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    return np.divide(
        blended,
        np.maximum(total, TINY),
        out=np.zeros_like(blended),
        where=total > 0.0,
    )


def dense_information_ceiling(device: torch.device, seed_count: int = 3) -> dict[str, Any]:
    if INFORMATION_REPORT.is_file():
        return json.loads(INFORMATION_REPORT.read_text(encoding="utf-8"))
    from scipy.spatial import cKDTree

    _, train_positions, _, train_site, _ = load_positions()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    dense_positions = np.load(DENSE_ROOT / "positions.npy")
    dense_site = np.load(DENSE_ROOT / "site.npy")
    maximum_neighbors = 16
    distances = np.empty((len(query_ids), maximum_neighbors), dtype=np.float64)
    neighbors = np.empty((len(query_ids), maximum_neighbors), dtype=np.int64)
    for site_value in (0, 1):
        query_rows = np.flatnonzero(train_site[query_ids] == site_value)
        bank = np.flatnonzero(dense_site == site_value)
        local_distance, local_neighbor = cKDTree(dense_positions[bank, :2]).query(
            train_positions[query_ids[query_rows], :2], k=maximum_neighbors
        )
        distances[query_rows] = local_distance
        neighbors[query_rows] = bank[local_neighbor]

    teacher = np.load(
        PATHS.artifacts
        / "round2"
        / "offline_sionna_geometry_student"
        / "raw_sionna_sufficient_teacher.npz",
        allow_pickle=False,
    )
    target_pas = teacher["pas"][query_ids].astype(np.float32)
    target_pdp = teacher["pdp"][query_ids].astype(np.float32)
    seed_pas = []
    seed_pdp = []
    for seed in SEEDS[:seed_count]:
        prefix = DENSE_ROOT / f"seed_{seed}"
        seed_pas.append(
            np.asarray(
                np.load(prefix.with_name(prefix.name + "_pas_f16.npy"), mmap_mode="r")[
                    neighbors
                ],
                dtype=np.float32,
            )
        )
        seed_pdp.append(
            np.asarray(
                np.load(prefix.with_name(prefix.name + "_pdp_f16.npy"), mmap_mode="r")[
                    neighbors
                ],
                dtype=np.float32,
            )
        )
    seed_pas_array = np.stack(seed_pas)
    seed_pdp_array = np.stack(seed_pdp)

    def summarize(pas: np.ndarray, pdp: np.ndarray) -> dict[str, Any]:
        values = spectral_point_score(
            torch.as_tensor(pas, device=device),
            torch.as_tensor(pdp, device=device),
            torch.as_tensor(target_pas, device=device),
            torch.as_tensor(target_pdp, device=device),
        )
        point_pas = values[0].cpu().numpy()
        point_pdp = values[1].cpu().numpy()
        rows = {
            "all": np.ones(len(query_ids), dtype=bool),
            "site0": train_site[query_ids] == 0,
            "site1": train_site[query_ids] == 1,
        }
        rows.update({f"fold{fold}": fold_id == fold for fold in range(5)})
        return {
            name: {
                "count": int(mask.sum()),
                "pas": float(point_pas[mask].mean()),
                "pdp": float(point_pdp[mask].mean()),
                "mean": float(0.5 * (point_pas[mask] + point_pdp[mask]).mean()),
            }
            for name, mask in rows.items()
        }

    knn = {}
    for count in (1, 4, 8, 16):
        weight = 1.0 / np.maximum(distances[:, :count], 0.08) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        pas = _normalized_root_blend(
            seed_pas_array[:, :, :count], weight, (1, 2, 3)
        )
        pdp = _normalized_root_blend(
            seed_pdp_array[:, :, :count], weight, (3,)
        )
        knn[f"k{count}"] = summarize(pas, pdp)

    e87 = np.load(
        PATHS.artifacts
        / "round2"
        / "offline_sionna_geometry_student"
        / "strict_oof_student_endpoints.npz",
        allow_pickle=False,
    )
    if not np.array_equal(e87["query_ids"].astype(np.int64), query_ids):
        raise ValueError("E87 query order changed")
    baseline = summarize(
        e87["map_teacher_pas"].astype(np.float32),
        e87["map_teacher_pdp"].astype(np.float32),
    )
    report = {
        "experiment": VERSION,
        "status": "information_ceiling_complete",
        "target": "E87 24-seed Sionna path-spectrum teacher at fixed 500 points",
        "seed_count": seed_count,
        "distance_m": {
            "nearest_median": float(np.median(distances[:, 0])),
            "nearest_p90": float(np.quantile(distances[:, 0], 0.9)),
            "neighbor16_median": float(np.median(distances[:, 15])),
        },
        "dense_knn": knn,
        "E87_map_teacher_baseline": baseline,
    }
    INFORMATION_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def sample_grid(
    grid: np.ndarray,
    xy: np.ndarray,
    x_min: float,
    y_min: float,
) -> np.ndarray:
    original = xy.shape[:-1]
    flat = xy.reshape(-1, 2)
    col = np.rint(flat[:, 0] - x_min).astype(np.int64)
    row = np.rint(flat[:, 1] - y_min).astype(np.int64)
    row = np.clip(row, 0, grid.shape[0] - 1)
    col = np.clip(col, 0, grid.shape[1] - 1)
    return np.asarray(grid[row, col]).reshape(original)


def map_scalar_summary(polar: np.ndarray, corridor: np.ndarray) -> np.ndarray:
    blocks = []
    for value in (polar, corridor):
        axes = tuple(range(2, value.ndim))
        blocks.extend(
            (
                value.mean(axis=axes),
                value.std(axis=axes),
                value.min(axis=axes),
                value.max(axis=axes),
            )
        )
    center_index = int(np.argmin(np.abs(CORRIDOR_OFFSETS_M)))
    center = corridor[:, :, center_index]
    blocks.extend(
        (
            center.mean(axis=2),
            center.std(axis=2),
            center.min(axis=2),
            center.max(axis=2),
        )
    )
    directional = polar.mean(axis=2)
    angles = np.linspace(0.0, 2.0 * np.pi, POLAR_DIRECTIONS, endpoint=False)
    for harmonic in range(1, 5):
        blocks.append(
            np.mean(directional * np.cos(harmonic * angles)[None, None], axis=2)
        )
        blocks.append(
            np.mean(directional * np.sin(harmonic * angles)[None, None], axis=2)
        )
    return np.concatenate(blocks, axis=1).astype(np.float32)


def open_memmap(path: Path, dtype: Any, shape: tuple[int, ...], force: bool) -> np.memmap:
    if force:
        path.unlink(missing_ok=True)
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"geometry cache mismatch: {path}")
        return value
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value[...] = 0
    value.flush()
    return value


def build_geometry_cache(force: bool) -> dict[str, Any]:
    paths = geometry_paths()
    expected = tuple(paths.values())
    if not force and all(path.is_file() for path in expected):
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))

    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    dense_positions = np.load(DENSE_ROOT / "positions.npy").astype(np.float32)
    dense_site = np.load(DENSE_ROOT / "site.npy").astype(np.int8)
    position_groups = {
        "dense": (dense_positions, dense_site),
        "train": (train_positions.astype(np.float32), train_site),
        "test": (test_positions.astype(np.float32), test_site),
    }
    all_positions = np.concatenate(
        (dense_positions, train_positions, test_positions, setup.bs_positions), axis=0
    )
    x_min, y_min, height, width = _grid_bounds(all_positions)
    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh).__name__)
    occupancy, wall_count, wall_height = _rasterize_walls(
        mesh, x_min, y_min, height, width
    )
    clearance = ndimage.distance_transform_edt(~occupancy).astype(np.float32)
    grids = (
        occupancy.astype(np.float32),
        np.clip(np.log1p(wall_count) / 5.0, 0.0, 1.0),
        np.clip(wall_height / 40.0, 0.0, 1.0),
        np.clip(clearance / 24.0, 0.0, 1.0),
    )
    polar_angles = np.linspace(
        0.0, 2.0 * np.pi, POLAR_DIRECTIONS, endpoint=False, dtype=np.float32
    )
    fractions = np.linspace(0.0, 1.0, CORRIDOR_SAMPLES, dtype=np.float32)
    output_shapes = {}
    for name, (positions, site) in position_groups.items():
        count = len(positions)
        polar_store = open_memmap(
            paths[f"{name}_polar"],
            np.float16,
            (count, 4, len(POLAR_RADII_M), POLAR_DIRECTIONS),
            force,
        )
        corridor_store = open_memmap(
            paths[f"{name}_corridor"],
            np.float16,
            (count, 4, len(CORRIDOR_OFFSETS_M), CORRIDOR_SAMPLES),
            force,
        )
        scalar_store: np.memmap | None = None
        for start in range(0, count, 256):
            end = min(start + 256, count)
            point = positions[start:end, :2]
            bs = setup.bs_positions[site[start:end], :2].astype(np.float32)
            toward_bs = bs - point
            distance = np.maximum(np.linalg.norm(toward_bs, axis=1), 1e-6)
            bearing = np.arctan2(toward_bs[:, 1], toward_bs[:, 0])
            direction = np.stack(
                (
                    np.cos(bearing[:, None] + polar_angles[None]),
                    np.sin(bearing[:, None] + polar_angles[None]),
                ),
                axis=-1,
            )
            polar_xy = (
                point[:, None, None, :]
                + POLAR_RADII_M[None, :, None, None] * direction[:, None]
            )
            route = point - bs
            route_unit = route / distance[:, None]
            perpendicular = np.stack((-route_unit[:, 1], route_unit[:, 0]), axis=1)
            center = bs[:, None] + fractions[None, :, None] * route[:, None]
            corridor_xy = (
                center[:, None]
                + CORRIDOR_OFFSETS_M[None, :, None, None]
                * perpendicular[:, None, None]
            )
            local_polar = np.empty(
                (end - start, 4, len(POLAR_RADII_M), POLAR_DIRECTIONS),
                dtype=np.float32,
            )
            local_corridor = np.empty(
                (end - start, 4, len(CORRIDOR_OFFSETS_M), CORRIDOR_SAMPLES),
                dtype=np.float32,
            )
            for channel, grid in enumerate(grids):
                local_polar[:, channel] = sample_grid(
                    grid, polar_xy, x_min, y_min
                )
                local_corridor[:, channel] = sample_grid(
                    grid, corridor_xy, x_min, y_min
                )
            summary = map_scalar_summary(local_polar, local_corridor)
            if scalar_store is None:
                scalar_store = open_memmap(
                    paths[f"{name}_scalar"],
                    np.float32,
                    (count, summary.shape[1]),
                    force,
                )
            polar_store[start:end] = local_polar.astype(np.float16)
            corridor_store[start:end] = local_corridor.astype(np.float16)
            scalar_store[start:end] = summary
            if end % 2048 < 256 or end == count:
                print(f"E88 geometry {name} {end}/{count}", flush=True)
        polar_store.flush()
        corridor_store.flush()
        assert scalar_store is not None
        scalar_store.flush()
        output_shapes[name] = {
            "polar": list(polar_store.shape),
            "corridor": list(corridor_store.shape),
            "scalar": list(scalar_store.shape),
        }

    dense_position = position_features(dense_positions, dense_site, setup.bs_positions)
    dense_scalar = np.load(paths["dense_scalar"], mmap_mode="r")
    position_mean = dense_position.mean(axis=0).astype(np.float32)
    position_std = dense_position.std(axis=0).astype(np.float32)
    position_std[position_std < 1e-5] = 1.0
    scalar_mean = np.asarray(dense_scalar, dtype=np.float64).mean(axis=0).astype(np.float32)
    scalar_std = np.asarray(dense_scalar, dtype=np.float64).std(axis=0).astype(np.float32)
    scalar_std[scalar_std < 1e-5] = 1.0
    np.savez(
        paths["normalization"],
        position_mean=position_mean,
        position_std=position_std,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
    )
    metadata = {
        "experiment": VERSION,
        "status": "geometry_complete",
        "grid": {
            "x_min": x_min,
            "y_min": y_min,
            "height": height,
            "width": width,
        },
        "outputs": output_shapes,
        "normalization": PATHS.relative(paths["normalization"]),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


class GeometryStore:
    def __init__(self, split: str, device: torch.device) -> None:
        if split not in ("dense", "train", "test"):
            raise ValueError(split)
        paths = geometry_paths()
        setup, train_positions, test_positions, train_site, test_site = load_positions()
        if split == "dense":
            self.positions = np.load(DENSE_ROOT / "positions.npy").astype(np.float32)
            self.site = np.load(DENSE_ROOT / "site.npy").astype(np.int8)
        elif split == "train":
            self.positions = train_positions.astype(np.float32)
            self.site = train_site
        else:
            self.positions = test_positions.astype(np.float32)
            self.site = test_site
        norm = np.load(paths["normalization"], allow_pickle=False)
        position = position_features(self.positions, self.site, setup.bs_positions)
        scalar = np.load(paths[f"{split}_scalar"], mmap_mode="r")
        self.position = np.clip(
            (position - norm["position_mean"]) / norm["position_std"], -8.0, 8.0
        ).astype(np.float32)
        self.scalar = np.clip(
            (np.asarray(scalar) - norm["scalar_mean"]) / norm["scalar_std"],
            -8.0,
            8.0,
        ).astype(np.float32)
        self.polar = np.load(paths[f"{split}_polar"], mmap_mode="r")
        self.corridor = np.load(paths[f"{split}_corridor"], mmap_mode="r")
        dense_positions = np.load(DENSE_ROOT / "positions.npy").astype(np.float32)
        dense_site = np.load(DENSE_ROOT / "site.npy").astype(np.int8)
        xy_min = np.stack(
            [dense_positions[dense_site == value, :2].min(axis=0) for value in range(2)]
        )
        xy_max = np.stack(
            [dense_positions[dense_site == value, :2].max(axis=0) for value in range(2)]
        )
        local_min = xy_min[self.site]
        local_span = np.maximum(xy_max[self.site] - local_min, 1e-6)
        self.xy_unit = np.clip(
            2.0 * (self.positions[:, :2] - local_min) / local_span - 1.0,
            -1.05,
            1.05,
        ).astype(np.float32)
        self.device = device

    def batch(self, rows: np.ndarray, use_map: bool = True) -> dict[str, torch.Tensor]:
        position = torch.as_tensor(self.position[rows], device=self.device)
        scalar = torch.as_tensor(self.scalar[rows], device=self.device)
        polar = torch.as_tensor(
            np.asarray(self.polar[rows], dtype=np.float32), device=self.device
        )
        corridor = torch.as_tensor(
            np.asarray(self.corridor[rows], dtype=np.float32), device=self.device
        )
        if not use_map:
            scalar = torch.zeros_like(scalar)
            polar = torch.zeros_like(polar)
            corridor = torch.zeros_like(corridor)
        return {
            "position": position,
            "scalar": scalar,
            "polar": polar,
            "corridor": corridor,
            "xy_unit": torch.as_tensor(self.xy_unit[rows], device=self.device),
            "site": torch.as_tensor(self.site[rows].astype(np.int64), device=self.device),
        }


def conv_encoder(channels: int, output: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, 64, 3, padding=1),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv2d(64, 96, 3, stride=(1, 2), padding=1),
        nn.GroupNorm(8, 96),
        nn.SiLU(),
        nn.Conv2d(96, 128, 3, stride=(2, 2), padding=1),
        nn.GroupNorm(8, 128),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d((2, 4)),
        nn.Flatten(),
        nn.Linear(128 * 8, output),
        nn.SiLU(),
    )


class MultiResolutionSpatialField(nn.Module):
    def __init__(self, config: Config, output: int) -> None:
        super().__init__()
        self.grids = nn.ParameterList(
            nn.Parameter(
                0.01
                * torch.randn(
                    2,
                    config.grid_features,
                    resolution,
                    resolution,
                )
            )
            for resolution in config.grid_resolutions
        )
        self.output = nn.Sequential(
            nn.Linear(len(config.grid_resolutions) * config.grid_features, output),
            nn.SiLU(),
            nn.LayerNorm(output),
        )

    def forward(self, xy_unit: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        levels = []
        for parameter in self.grids:
            sampled = torch.zeros(
                len(xy_unit), parameter.shape[1], device=xy_unit.device, dtype=parameter.dtype
            )
            for site_value in (0, 1):
                rows = torch.nonzero(site == site_value, as_tuple=True)[0]
                if not len(rows):
                    continue
                coordinates = xy_unit[rows].reshape(1, -1, 1, 2).to(parameter.dtype)
                local = torch.nn.functional.grid_sample(
                    parameter[site_value : site_value + 1],
                    coordinates,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                sampled[rows] = local[0, :, :, 0].transpose(0, 1)
            levels.append(sampled)
        return self.output(torch.cat(levels, dim=1))


class DensePathResidualStudent(nn.Module):
    def __init__(self, position_dim: int, scalar_dim: int, config: Config) -> None:
        super().__init__()
        self.config = config
        branch = config.hidden // 4
        self.scalar = nn.Sequential(
            nn.Linear(position_dim + scalar_dim, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, branch),
            nn.SiLU(),
        )
        self.polar = conv_encoder(4, branch)
        self.corridor = conv_encoder(4, branch)
        self.spatial_field = MultiResolutionSpatialField(config, branch)
        self.fusion = nn.Sequential(
            nn.Linear(4 * branch, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.slot_queries = nn.Parameter(
            torch.randn(config.path_slots, config.hidden) / math.sqrt(config.hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_encoder = nn.TransformerEncoder(layer, num_layers=config.slot_layers)
        self.coordinate_head = nn.Linear(config.hidden, 3)
        self.mass_head = nn.Linear(config.hidden, 1)
        self.port_head = nn.Linear(config.hidden, MP * N_UE)
        self.pas_residual = nn.Linear(config.hidden, MP * MH * MV * N_UE)
        self.pdp_residual = nn.Linear(config.hidden, MP * N_UE * S_SC)
        self.valid_head = nn.Linear(config.hidden, 1)
        self.reliability_head = nn.Linear(config.hidden, 1)
        nn.init.zeros_(self.pas_residual.weight)
        nn.init.zeros_(self.pas_residual.bias)
        nn.init.zeros_(self.pdp_residual.weight)
        nn.init.zeros_(self.pdp_residual.bias)

    @staticmethod
    def kernels(coordinate: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = torch.arange(MH, dtype=torch.float32, device=coordinate.device)
        v = torch.arange(MV, dtype=torch.float32, device=coordinate.device)
        d = torch.arange(S_SC, dtype=torch.float32, device=coordinate.device)
        h_wave = torch.exp(2j * torch.pi * coordinate[..., 0, None] * h / MH)
        v_wave = torch.exp(2j * torch.pi * coordinate[..., 1, None] * v / MV)
        d_wave = torch.exp(-2j * torch.pi * coordinate[..., 2, None] * d / S_SC)
        horizontal = torch.fft.fft(h_wave, dim=-1).abs().square().float()
        vertical = torch.fft.fft(v_wave, dim=-1).abs().square().float()
        delay = torch.fft.ifft(d_wave, dim=-1).abs().square().float()
        horizontal /= horizontal.sum(dim=-1, keepdim=True).clamp_min(TINY)
        vertical /= vertical.sum(dim=-1, keepdim=True).clamp_min(TINY)
        delay /= delay.sum(dim=-1, keepdim=True).clamp_min(TINY)
        return horizontal, vertical, delay

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context = self.fusion(
            torch.cat(
                (
                    self.scalar(torch.cat((batch["position"], batch["scalar"]), dim=1)),
                    self.polar(batch["polar"]),
                    self.corridor(batch["corridor"]),
                    self.spatial_field(batch["xy_unit"], batch["site"]),
                ),
                dim=1,
            )
        )
        slots = context[:, None] + self.slot_queries[None]
        slots = self.slot_encoder(slots)
        coordinate_unit = torch.sigmoid(self.coordinate_head(slots)).float()
        scale = torch.tensor((MH, MV, S_SC), device=slots.device)
        coordinate = coordinate_unit * scale
        mass = torch.softmax(self.mass_head(slots)[..., 0].float(), dim=1)
        port = torch.softmax(
            self.port_head(slots).float().reshape(-1, self.config.path_slots, MP * N_UE),
            dim=2,
        ).reshape(-1, self.config.path_slots, MP, N_UE)
        horizontal, vertical, delay = self.kernels(coordinate)
        energy = mass[..., None, None] * port
        path_pas = torch.einsum(
            "bkpu,bkh,bkv->bphvu", energy, horizontal, vertical
        )
        path_pas /= path_pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
        path_pdp = torch.einsum("bkpu,bkd->bpud", energy, delay)
        path_pdp /= path_pdp.sum(dim=3, keepdim=True).clamp_min(TINY)
        pas_logits = self.pas_residual(context).float().reshape(
            -1, N_UE, MP * MH * MV
        )
        dense_pas = torch.softmax(pas_logits, dim=2).reshape(
            -1, N_UE, MP, MH, MV
        ).permute(0, 2, 3, 4, 1)
        pdp_logits = self.pdp_residual(context).float().reshape(
            -1, MP, N_UE, S_SC
        )
        dense_pdp = torch.softmax(pdp_logits, dim=3)
        mix = self.config.dense_spectral_mix
        pas = (1.0 - mix) * path_pas + mix * dense_pas
        pdp = (1.0 - mix) * path_pdp + mix * dense_pdp
        valid_logits = self.valid_head(context)[..., 0].float()
        return {
            "coordinate": coordinate,
            "coordinate_unit": coordinate_unit,
            "mass": mass,
            "port": port,
            "path_pas": path_pas,
            "path_pdp": path_pdp,
            "dense_pas": dense_pas,
            "dense_pdp": dense_pdp,
            "pas": pas,
            "pdp": pdp,
            "pas_residual": pas_logits,
            "pdp_residual": pdp_logits,
            "valid_logits": valid_logits,
            "valid": torch.sigmoid(valid_logits),
            "reliability": torch.sigmoid(
                self.reliability_head(context)[..., 0].float()
            ),
        }


def spectral_point_score(
    pas: torch.Tensor,
    pdp: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pas_dot = (pas * target_pas).sum(dim=(1, 2, 3), dtype=torch.float64)
    pas_product = (
        pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * target_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    pas_norm = torch.sqrt(pas_product.clamp_min(TINY))
    pas_score = (
        pas_dot
        / torch.where(
            pas_product > 0.0, pas_norm, torch.full_like(pas_norm, 1e-12)
        )
    ).mean(dim=1)
    pdp_dot = (pdp * target_pdp).sum(dim=3, dtype=torch.float64)
    pdp_product = (
        pdp.square().sum(dim=3, dtype=torch.float64)
        * target_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_norm = torch.sqrt(pdp_product.clamp_min(TINY))
    pdp_score = (
        pdp_dot
        / torch.where(
            pdp_product > 0.0, pdp_norm, torch.full_like(pdp_norm, 1e-12)
        )
    ).mean(dim=(1, 2))
    return pas_score, pdp_score, 0.5 * (pas_score + pdp_score)


def endpoint_summary(
    pas: np.ndarray,
    pdp: np.ndarray,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    fold_id: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    values = spectral_point_score(
        torch.as_tensor(pas, device=device),
        torch.as_tensor(pdp, device=device),
        truth_pas,
        truth_pdp,
    )
    point_pas = values[0].cpu().numpy()
    point_pdp = values[1].cpu().numpy()
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        if not np.any(mask):
            continue
        folds.append(
            {
                "fold": fold,
                "count": int(mask.sum()),
                "pas": float(point_pas[mask].mean()),
                "pdp": float(point_pdp[mask].mean()),
                "mean": float(0.5 * (point_pas[mask] + point_pdp[mask]).mean()),
            }
        )
    return {
        "pas": float(point_pas.mean()),
        "pdp": float(point_pdp.mean()),
        "mean": float(0.5 * (point_pas + point_pdp).mean()),
        "folds": folds,
    }


def periodic_support_loss(
    output: dict[str, torch.Tensor],
    teacher_coordinate: torch.Tensor,
    teacher_mass: torch.Tensor,
    teacher_port: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.tensor((MH, MV, S_SC), device=teacher_coordinate.device)
    target = teacher_coordinate / scale
    prediction = output["coordinate_unit"]
    delta = torch.abs(prediction[:, :, None] - target[:, None])
    delta = torch.minimum(delta, 1.0 - delta)
    distance = (delta.square() * torch.tensor((1.0, 1.0, 0.65), device=delta.device)).sum(dim=3)
    distance = torch.where(
        teacher_mask[:, None], distance, torch.full_like(distance, 1e3)
    )
    teacher_min = distance.min(dim=1).values
    teacher_weight = teacher_mass / teacher_mass.sum(dim=1, keepdim=True).clamp_min(TINY)
    coverage = (teacher_weight * teacher_min).sum(dim=1)
    prediction_min, nearest = distance.min(dim=2)
    precision = (output["mass"] * prediction_min).sum(dim=1)
    support = (coverage + precision).mean()
    nearest_port = torch.gather(
        teacher_port,
        1,
        nearest[..., None, None].expand(-1, -1, MP, N_UE),
    )
    port = (
        output["mass"][..., None, None]
        * (
            torch.sqrt(output["port"].clamp_min(0.0))
            - torch.sqrt(nearest_port.clamp_min(0.0))
        ).square()
    ).mean()
    return support, port


def spectral_loss(
    pas: torch.Tensor,
    pdp: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    _, _, score = spectral_point_score(pas, pdp, target_pas, target_pdp)
    if torch.any(valid):
        return (1.0 - score[valid]).mean()
    return pas.sum() * 0.0


class DenseTeacherStore:
    def __init__(self, seed_count: int = 3) -> None:
        self.seeds = tuple(SEEDS[:seed_count])
        self.values: dict[int, dict[str, np.ndarray]] = {}
        for seed in self.seeds:
            prefix = DENSE_ROOT / f"seed_{seed}"
            self.values[int(seed)] = {
                name: np.load(prefix.with_name(prefix.name + suffix), mmap_mode="r")
                for name, suffix in {
                    "coordinate": "_coordinate_f16.npy",
                    "mass": "_mass_f16.npy",
                    "port": "_port_f16.npy",
                    "mask": "_mask.npy",
                    "pas": "_pas_f16.npy",
                    "pdp": "_pdp_f16.npy",
                    "done": "_done.npy",
                }.items()
            }
            if not np.all(self.values[int(seed)]["done"]):
                raise RuntimeError(f"dense teacher seed {seed} is incomplete")
        self.agreement = np.load(SEED_AGREEMENT, mmap_mode="r")

    def batch(
        self, rows: np.ndarray, seed_index: np.ndarray, device: torch.device
    ) -> dict[str, torch.Tensor]:
        shape_coordinate = self.values[int(self.seeds[0])]["coordinate"].shape[1:]
        coordinate = np.empty((len(rows), *shape_coordinate), dtype=np.float32)
        mass = np.empty((len(rows), shape_coordinate[0]), dtype=np.float32)
        port = np.empty((len(rows), shape_coordinate[0], MP, N_UE), dtype=np.float32)
        mask = np.empty((len(rows), shape_coordinate[0]), dtype=bool)
        pas = np.empty((len(rows), MP, MH, MV, N_UE), dtype=np.float32)
        pdp = np.empty((len(rows), MP, N_UE, S_SC), dtype=np.float32)
        for local_seed, seed in enumerate(self.seeds):
            selected = np.flatnonzero(seed_index == local_seed)
            if not len(selected):
                continue
            source = self.values[int(seed)]
            local_rows = rows[selected]
            coordinate[selected] = source["coordinate"][local_rows]
            mass[selected] = source["mass"][local_rows]
            port[selected] = source["port"][local_rows]
            mask[selected] = source["mask"][local_rows]
            pas[selected] = source["pas"][local_rows]
            pdp[selected] = source["pdp"][local_rows]
        return {
            "coordinate": torch.as_tensor(coordinate, device=device),
            "mass": torch.as_tensor(mass, device=device),
            "port": torch.as_tensor(port, device=device),
            "mask": torch.as_tensor(mask, device=device),
            "pas": torch.as_tensor(pas, device=device),
            "pdp": torch.as_tensor(pdp, device=device),
            "reliability": torch.as_tensor(
                np.asarray(self.agreement[rows], dtype=np.float32), device=device
            ),
        }


def random_objective(
    output: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = teacher["mask"].any(dim=1)
    final_loss = spectral_loss(
        output["pas"], output["pdp"], teacher["pas"], teacher["pdp"], valid
    )
    path_loss = spectral_loss(
        output["path_pas"],
        output["path_pdp"],
        teacher["pas"],
        teacher["pdp"],
        valid,
    )
    if torch.any(valid):
        support, port = periodic_support_loss(
            {key: value[valid] for key, value in output.items()},
            teacher["coordinate"][valid],
            teacher["mass"][valid],
            teacher["port"][valid],
            teacher["mask"][valid],
        )
    else:
        support = final_loss * 0.0
        port = final_loss * 0.0
    valid_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["valid_logits"], valid.float()
    )
    reliability_loss = torch.nn.functional.mse_loss(
        output["reliability"], teacher["reliability"].clamp(0.0, 1.0)
    )
    residual = 0.5 * (
        output["pas_residual"].square().mean()
        + output["pdp_residual"].square().mean()
    )
    loss = (
        final_loss
        + config.path_spectral_weight * path_loss
        + config.support_weight * support
        + config.port_weight * port
        + config.residual_weight * residual
        + 0.05 * valid_loss
        + 0.05 * reliability_loss
    )
    return loss, {
        "final_loss": float(final_loss.detach().item()),
        "path_loss": float(path_loss.detach().item()),
        "support_loss": float(support.detach().item()),
        "valid_loss": float(valid_loss.detach().item()),
        "reliability_loss": float(reliability_loss.detach().item()),
    }


@torch.inference_mode()
def evaluate_random(
    model: DensePathResidualStudent,
    geometry: GeometryStore,
    teacher: DenseTeacherStore,
    rows: np.ndarray,
    config: Config,
) -> dict[str, float]:
    model.eval()
    scores = []
    path_scores = []
    valid_accuracy = []
    for start in range(0, len(rows), config.pretrain_batch_size):
        end = min(start + config.pretrain_batch_size, len(rows))
        local = rows[start:end]
        # Validation uses the mean score over all independent teacher seeds.
        batch = geometry.batch(local)
        output = model(batch)
        seed_scores = []
        seed_path_scores = []
        seed_valid = []
        for seed_index in range(len(teacher.seeds)):
            target = teacher.batch(
                local,
                np.full(len(local), seed_index, dtype=np.int64),
                geometry.device,
            )
            _, _, score = spectral_point_score(
                output["pas"], output["pdp"], target["pas"], target["pdp"]
            )
            _, _, path_score = spectral_point_score(
                output["path_pas"],
                output["path_pdp"],
                target["pas"],
                target["pdp"],
            )
            valid = target["mask"].any(dim=1)
            seed_scores.append(score)
            seed_path_scores.append(path_score)
            seed_valid.append(valid)
        scores.append(torch.stack(seed_scores).mean(dim=0).cpu().numpy())
        path_scores.append(torch.stack(seed_path_scores).mean(dim=0).cpu().numpy())
        target_valid = torch.stack(seed_valid).float().mean(dim=0) >= 0.5
        valid_accuracy.append(
            ((output["valid"] >= 0.5) == target_valid).float().cpu().numpy()
        )
    return {
        "spectral_mean": float(np.concatenate(scores).mean()),
        "path_spectral_mean": float(np.concatenate(path_scores).mean()),
        "valid_accuracy": float(np.concatenate(valid_accuracy).mean()),
    }


@torch.inference_mode()
def evaluate_pretrain_on_fixed_teacher(
    model: DensePathResidualStudent,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    geometry = GeometryStore("train", device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    teacher = np.load(
        PATHS.artifacts
        / "round2"
        / "offline_sionna_geometry_student"
        / "raw_sionna_sufficient_teacher.npz",
        allow_pickle=False,
    )
    target_pas = torch.as_tensor(teacher["pas"][query_ids].astype(np.float32), device=device)
    target_pdp = torch.as_tensor(teacher["pdp"][query_ids].astype(np.float32), device=device)
    rng = np.random.default_rng(config.random_state + 77)
    shuffled_global = np.array(query_ids, copy=True)
    for site_value in (0, 1):
        local = np.flatnonzero(geometry.site[query_ids] == site_value)
        shuffled_global[local] = query_ids[rng.permutation(local)]
    branches = {
        "final": {
            "pas": np.empty((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32),
            "pdp": np.empty((len(query_ids), MP, N_UE, S_SC), dtype=np.float32),
        },
        "path_only": {
            "pas": np.empty((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32),
            "pdp": np.empty((len(query_ids), MP, N_UE, S_SC), dtype=np.float32),
        },
        "spatial_field_shuffle": {
            "pas": np.empty((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32),
            "pdp": np.empty((len(query_ids), MP, N_UE, S_SC), dtype=np.float32),
        },
    }
    model.eval()
    for start in range(0, len(query_ids), config.finetune_batch_size):
        end = min(start + config.finetune_batch_size, len(query_ids))
        local = query_ids[start:end]
        batch = geometry.batch(local)
        output = model(batch)
        branches["final"]["pas"][start:end] = output["pas"].cpu().numpy()
        branches["final"]["pdp"][start:end] = output["pdp"].cpu().numpy()
        branches["path_only"]["pas"][start:end] = output["path_pas"].cpu().numpy()
        branches["path_only"]["pdp"][start:end] = output["path_pdp"].cpu().numpy()
        shuffled_batch = geometry.batch(local)
        shuffled_batch["xy_unit"] = torch.as_tensor(
            geometry.xy_unit[shuffled_global[start:end]], device=device
        )
        shuffled = model(shuffled_batch)
        branches["spatial_field_shuffle"]["pas"][start:end] = shuffled["pas"].cpu().numpy()
        branches["spatial_field_shuffle"]["pdp"][start:end] = shuffled["pdp"].cpu().numpy()
    return {
        name: endpoint_summary(
            value["pas"],
            value["pdp"],
            target_pas,
            target_pdp,
            fold_id,
            device,
        )
        for name, value in branches.items()
    }


def build_model(geometry: GeometryStore, config: Config) -> DensePathResidualStudent:
    return DensePathResidualStudent(
        geometry.position.shape[1], geometry.scalar.shape[1], config
    ).to(geometry.device)


def train_pretrain(config: Config, device: torch.device, force: bool) -> dict[str, Any]:
    if PRETRAIN_CHECKPOINT.is_file() and not force:
        checkpoint = torch.load(PRETRAIN_CHECKPOINT, map_location="cpu", weights_only=False)
        if checkpoint.get("config") == asdict(config):
            if checkpoint.get("fixed_teacher_metric_version") != FIXED_TEACHER_METRIC_VERSION:
                geometry = GeometryStore("train", device)
                model = build_model(geometry, config)
                model.load_state_dict(checkpoint["state_dict"])
                checkpoint["fixed_teacher_evaluation"] = (
                    evaluate_pretrain_on_fixed_teacher(model, config, device)
                )
                checkpoint["fixed_teacher_metric_version"] = (
                    FIXED_TEACHER_METRIC_VERSION
                )
                torch.save(checkpoint, PRETRAIN_CHECKPOINT)
                del model
            return {
                "status": "reused",
                "checkpoint": PATHS.relative(PRETRAIN_CHECKPOINT),
                "best_epoch": checkpoint["best_epoch"],
                "best_validation": checkpoint["best_validation"],
                "parameter_count": checkpoint["parameter_count"],
                "fixed_teacher_evaluation": checkpoint["fixed_teacher_evaluation"],
            }
    geometry = GeometryStore("dense", device)
    teacher = DenseTeacherStore(3)
    rows = np.arange(len(geometry.positions), dtype=np.int64)
    validation = rows[rows % 20 == 0]
    fit = rows[rows % 20 != 0]
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    model = build_model(geometry, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain_learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.pretrain_epochs,
        eta_min=config.pretrain_learning_rate * 0.05,
    )
    rng = np.random.default_rng(config.random_state + 1)
    best_score = -math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = []
    for epoch in range(1, config.pretrain_epochs + 1):
        model.train()
        order = rng.permutation(fit)
        losses = []
        for start in range(0, len(order), config.pretrain_batch_size):
            local = order[start : start + config.pretrain_batch_size]
            seed_index = rng.integers(0, len(teacher.seeds), size=len(local))
            batch = geometry.batch(local)
            target = teacher.batch(local, seed_index, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(batch)
                loss, _ = random_objective(output, target, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite E88 pretrain loss at epoch={epoch} start={start}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite E88 pretrain gradient at epoch={epoch} start={start}"
                )
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()
        if epoch % config.validation_interval == 0 or epoch == config.pretrain_epochs:
            validation_metrics = evaluate_random(
                model, geometry, teacher, validation, config
            )
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                **validation_metrics,
            }
            history.append(row)
            if row["spectral_mean"] > best_score:
                best_score = row["spectral_mean"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E88 pretrain epoch={epoch}/{config.pretrain_epochs} "
                f"loss={row['loss']:.6f} spectrum={row['spectral_mean']:.6f} "
                f"path={row['path_spectral_mean']:.6f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    fixed_teacher_evaluation = evaluate_pretrain_on_fixed_teacher(
        model, config, device
    )
    payload = {
        "experiment": VERSION,
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_validation": best_score,
        "history": history,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "fixed_teacher_evaluation": fixed_teacher_evaluation,
        "fixed_teacher_metric_version": FIXED_TEACHER_METRIC_VERSION,
        "sionna_inputs_required_for_inference": False,
    }
    torch.save(payload, PRETRAIN_CHECKPOINT)
    return {
        "status": "trained",
        "checkpoint": PATHS.relative(PRETRAIN_CHECKPOINT),
        "best_epoch": best_epoch,
        "best_validation": best_score,
        "parameter_count": payload["parameter_count"],
        "fixed_teacher_evaluation": fixed_teacher_evaluation,
        "history": history,
    }


@torch.inference_mode()
def predict_official(
    model: DensePathResidualStudent,
    geometry: GeometryStore,
    rows: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    output = {
        "pas": np.empty((len(rows), MP, MH, MV, N_UE), dtype=np.float32),
        "pdp": np.empty((len(rows), MP, N_UE, S_SC), dtype=np.float32),
        "path_pas": np.empty((len(rows), MP, MH, MV, N_UE), dtype=np.float32),
        "path_pdp": np.empty((len(rows), MP, N_UE, S_SC), dtype=np.float32),
        "valid": np.empty(len(rows), dtype=np.float32),
        "reliability": np.empty(len(rows), dtype=np.float32),
    }
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        local = rows[start:end]
        values = model(geometry.batch(local))
        for name in output:
            output[name][start:end] = values[name].cpu().numpy().astype(np.float32)
    return output


def official_objective(
    output: dict[str, torch.Tensor],
    official_pas: torch.Tensor,
    official_pdp: torch.Tensor,
    teacher_pas: torch.Tensor,
    teacher_pdp: torch.Tensor,
    valid: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    official = spectral_loss(
        output["pas"], output["pdp"], official_pas, official_pdp, valid
    )
    teacher = spectral_loss(
        output["path_pas"],
        output["path_pdp"],
        teacher_pas,
        teacher_pdp,
        valid,
    )
    valid_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["valid_logits"], valid.float()
    )
    with torch.no_grad():
        _, _, teacher_agreement = spectral_point_score(
            teacher_pas, teacher_pdp, official_pas, official_pdp
        )
    reliability_loss = torch.nn.functional.mse_loss(
        output["reliability"], teacher_agreement.float().clamp(0.0, 1.0)
    )
    residual = 0.5 * (
        output["pas_residual"].square().mean()
        + output["pdp_residual"].square().mean()
    )
    return (
        official
        + config.teacher_retain * teacher
        + config.residual_weight * residual
        + 0.05 * valid_loss
        + 0.05 * reliability_loss
    )


@torch.inference_mode()
def official_endpoint_score(
    model: DensePathResidualStudent,
    geometry: GeometryStore,
    rows: np.ndarray,
    official_pas: torch.Tensor,
    official_pdp: torch.Tensor,
    config: Config,
) -> dict[str, float]:
    prediction = predict_official(model, geometry, rows, config.finetune_batch_size)
    values = spectral_point_score(
        torch.as_tensor(prediction["pas"], device=geometry.device),
        torch.as_tensor(prediction["pdp"], device=geometry.device),
        official_pas[torch.as_tensor(rows, device=geometry.device)],
        official_pdp[torch.as_tensor(rows, device=geometry.device)],
    )
    return {
        "pas": float(values[0].mean().item()),
        "pdp": float(values[1].mean().item()),
        "mean": float(values[2].mean().item()),
    }


def train_official_fold(
    *,
    outer: int,
    fit: np.ndarray,
    validation: np.ndarray,
    geometry: GeometryStore,
    official_pas: torch.Tensor,
    official_pdp: torch.Tensor,
    teacher_pas: torch.Tensor,
    teacher_pdp: torch.Tensor,
    power: np.ndarray,
    config: Config,
) -> tuple[DensePathResidualStudent, dict[str, Any]]:
    checkpoint = torch.load(PRETRAIN_CHECKPOINT, map_location="cpu", weights_only=False)
    fold_seed = config.random_state + 10_000 * outer + 17
    torch.manual_seed(fold_seed)
    if geometry.device.type == "cuda":
        torch.cuda.manual_seed_all(fold_seed)
    model = build_model(geometry, config)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.finetune_learning_rate,
        weight_decay=config.weight_decay,
        fused=geometry.device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.finetune_epochs,
        eta_min=config.finetune_learning_rate * 0.05,
    )
    rng = np.random.default_rng(config.random_state + 10_000 * outer)
    best_score = -math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, config.finetune_epochs + 1):
        model.train()
        order = rng.permutation(fit)
        losses = []
        for start in range(0, len(order), config.finetune_batch_size):
            local_np = order[start : start + config.finetune_batch_size]
            local = torch.as_tensor(local_np, device=geometry.device)
            batch = geometry.batch(local_np)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=geometry.device.type,
                dtype=torch.bfloat16,
                enabled=geometry.device.type == "cuda",
            ):
                output = model(batch)
                loss = official_objective(
                    output,
                    official_pas[local],
                    official_pdp[local],
                    teacher_pas[local],
                    teacher_pdp[local],
                    torch.as_tensor(power[local_np] > 0.0, device=geometry.device),
                    config,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite E88 official loss outer={outer} epoch={epoch} start={start}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite E88 official gradient outer={outer} epoch={epoch} start={start}"
                )
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()
        if epoch % config.validation_interval == 0 or epoch == config.finetune_epochs:
            metrics = official_endpoint_score(
                model,
                geometry,
                validation,
                official_pas,
                official_pdp,
                config,
            )
            row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
            history.append(row)
            if row["mean"] > best_score:
                best_score = row["mean"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E88 outer={outer} epoch={epoch}/{config.finetune_epochs} "
                f"loss={row['loss']:.6f} validation={row['mean']:.6f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return model, {
        "outer": outer,
        "fit_count": int(len(fit)),
        "validation_count": int(len(validation)),
        "best_epoch": best_epoch,
        "best_validation": best_score,
        "history": history,
    }


def run_strict(
    config: Config, device: torch.device, only_outer: int | None
) -> dict[str, Any]:
    geometry = GeometryStore("train", device)
    teacher_cache = np.load(
        PATHS.artifacts
        / "round2"
        / "offline_sionna_geometry_student"
        / "raw_sionna_sufficient_teacher.npz",
        allow_pickle=False,
    )
    official_pas = torch.as_tensor(
        np.array(np.load(e85.TARGET_PAS, mmap_mode="r"), dtype=np.float32, copy=True),
        device=device,
    )
    official_pdp = torch.as_tensor(
        np.array(np.load(e85.TARGET_PDP, mmap_mode="r"), dtype=np.float32, copy=True),
        device=device,
    )
    teacher_pas = torch.as_tensor(teacher_cache["pas"].astype(np.float32), device=device)
    teacher_pdp = torch.as_tensor(teacher_cache["pdp"].astype(np.float32), device=device)
    power = np.asarray(np.load(e85.POINT_POWER), dtype=np.float64)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    global_to_query = np.full(len(geometry.positions), -1, dtype=np.int64)
    global_to_query[query_ids] = np.arange(len(query_ids))
    names = ("final", "path_only")
    predictions = {
        name: {
            "pas": np.zeros((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32),
            "pdp": np.zeros((len(query_ids), MP, N_UE, S_SC), dtype=np.float32),
        }
        for name in names
    }
    confidence = np.zeros((len(query_ids), 2), dtype=np.float32)
    all_rows = np.arange(len(geometry.positions), dtype=np.int64)
    outers = range(5) if only_outer is None else (only_outer,)
    training = []
    for outer in outers:
        validation_fold = (outer + 1) % 5
        held = query_ids[fold_id == outer]
        validation = query_ids[fold_id == validation_fold]
        excluded = np.zeros(len(geometry.positions), dtype=bool)
        excluded[held] = True
        excluded[validation] = True
        fit = all_rows[~excluded]
        model, record = train_official_fold(
            outer=outer,
            fit=fit,
            validation=validation,
            geometry=geometry,
            official_pas=official_pas,
            official_pdp=official_pdp,
            teacher_pas=teacher_pas,
            teacher_pdp=teacher_pdp,
            power=power,
            config=config,
        )
        held_output = predict_official(
            model, geometry, held, config.finetune_batch_size
        )
        target_rows = global_to_query[held]
        predictions["final"]["pas"][target_rows] = held_output["pas"]
        predictions["final"]["pdp"][target_rows] = held_output["pdp"]
        predictions["path_only"]["pas"][target_rows] = held_output["path_pas"]
        predictions["path_only"]["pdp"][target_rows] = held_output["path_pdp"]
        confidence[target_rows, 0] = held_output["valid"]
        confidence[target_rows, 1] = held_output["reliability"]
        checkpoint_path = ARTIFACT_DIR / f"outer{outer}_dense_path_student.pt"
        torch.save(
            {
                "experiment": VERSION,
                "config": asdict(config),
                "outer": outer,
                "state_dict": model.state_dict(),
                "sionna_inputs_required_for_inference": False,
            },
            checkpoint_path,
        )
        training.append(
            {
                **record,
                "held_count": int(len(held)),
                "checkpoint": PATHS.relative(checkpoint_path),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    completed = np.isin(fold_id, np.asarray(tuple(outers), dtype=np.int8))
    selected_ids = query_ids[completed]
    selected_folds = fold_id[completed]
    truth_pas = official_pas[torch.as_tensor(selected_ids, device=device)]
    truth_pdp = official_pdp[torch.as_tensor(selected_ids, device=device)]
    summaries = {}
    for name in names:
        summaries[name] = endpoint_summary(
            predictions[name]["pas"][completed],
            predictions[name]["pdp"][completed],
            truth_pas,
            truth_pdp,
            selected_folds,
            device,
        )
    if only_outer is None:
        np.savez_compressed(
            OOF_PREDICTIONS,
            query_ids=query_ids.astype(np.int32),
            fold_id=fold_id,
            final_pas=predictions["final"]["pas"].astype(np.float16),
            final_pdp=predictions["final"]["pdp"].astype(np.float16),
            path_pas=predictions["path_only"]["pas"].astype(np.float16),
            path_pdp=predictions["path_only"]["pdp"].astype(np.float16),
            confidence=confidence.astype(np.float16),
        )
        fusion = fusion_metrics(
            {
                name: (predictions[name]["pas"], predictions[name]["pdp"])
                for name in names
            },
            query_ids,
            fold_id,
            device,
        )
        fixed = fusion["branches"]["final"]["fixed_beta_020"]
        decision = {
            "v12_score_to_match": fusion["V12"]["score"],
            "student_score": fixed["metrics"]["score"],
            "delta_vs_v11": fixed["delta_vs_v11"],
            "positive_folds": fixed["positive_folds"],
            "passed_deployment_gate": bool(
                fixed["metrics"]["score"] >= fusion["V12"]["score"]
                and fixed["positive_folds"] == 5
            ),
        }
    else:
        fusion = None
        decision = {"full_strict_completed": False}
    return {
        "endpoint": summaries,
        "fusion": fusion,
        "decision": decision,
        "training": training,
        "oof_predictions": PATHS.relative(OOF_PREDICTIONS) if only_outer is None else None,
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.outer < 5:
        raise ValueError("--outer must be in [0, 4]")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    config = Config(
        hidden=args.hidden,
        path_slots=args.path_slots,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        finetune_batch_size=args.finetune_batch_size,
    )
    started = time.time()
    missing_base = [
        path
        for path in (
            DENSE_ROOT / "positions.npy",
            DENSE_ROOT / "site.npy",
            PATHS.round2 / "Round2_Map.ply",
            e85.TARGET_PAS,
            e85.TARGET_PDP,
            e85.POINT_POWER,
        )
        if not path.exists()
    ]
    if missing_base:
        raise FileNotFoundError([PATHS.relative(path) for path in missing_base])
    geometry = build_geometry_cache(args.force_geometry)
    if args.stage == "prepare":
        print(json.dumps(geometry, ensure_ascii=False, indent=2), flush=True)
        return
    missing_teacher = [path for path in required_teacher_files() if not path.exists()]
    if missing_teacher:
        raise FileNotFoundError([PATHS.relative(path) for path in missing_teacher])
    agreement = build_seed_agreement()
    information = dense_information_ceiling(device)
    pretrain = train_pretrain(config, device, args.force_pretrain)
    if args.stage == "pretrain":
        print(
            json.dumps(
                {
                    "seed_agreement": agreement,
                    "information_ceiling": information,
                    "pretrain": pretrain,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    result = run_strict(
        config, device, args.outer if args.stage == "screen" else None
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Dense new-position Sionna supervision plus a path-state decoder can predict "
            "a reliable correction to the official-only complex anchor across unseen holes."
        ),
        "configuration": asdict(config),
        "geometry": geometry,
        "seed_agreement": agreement,
        "information_ceiling": information,
        "pretrain": pretrain,
        **result,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "endpoint": result["endpoint"],
                "fusion": result["fusion"],
                "decision": result["decision"],
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
