#!/usr/bin/env python3
"""Learn a distance-baseline channel transfer function constrained by geometry."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch import nn
import trimesh

from analyze_e86_surface_endpoint_v11_blend import endpoint_target
from analyze_v4_exact_joint_projection import (
    pas_distribution,
    pdp_distribution,
    truth_views,
)
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from spatial_hole_cv import build_spatial_hole_folds
from train_full_path_pas_slice_gate import project_joint
import train_offline_sionna_geometry_student as e87
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E109-geometry-conditioned-transfer-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "geometry_conditioned_transfer_operator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_geometry_conditioned_transfer_operator"
DELAY_ANGLE_CACHE = ARTIFACT_DIR / "link_delay_angle_geometry_f16.npz"
SCREEN_REPORT = ANALYSIS_DIR / "outer0_screen.json"
STRICT_REPORT = ANALYSIS_DIR / "strict_five_hole.json"
STRICT_ENDPOINTS = ARTIFACT_DIR / "strict_oof_endpoints.npz"

OOF_META = e87.OOF_META
V11_CHANNEL = e87.V11_CHANNEL
V12_CHANNEL = e87.V12_CHANNEL
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PAS_SIZE = MP * MH * MV * N_UE
PDP_SIZE = MP * N_UE * S_SC
TINY = 1e-30

DELAY_BINS = 48
ANGLE_BINS = 64
MAX_EXCESS_M = 600.0
MAP_CHANNELS = 6
MODES = (
    "distance_only",
    "legacy_geometry",
    "full_geometry",
    "shuffled_full_geometry",
)


@dataclass(frozen=True)
class Config:
    hidden: int = 384
    branch: int = 96
    residual_blocks: int = 4
    dropout: float = 0.08
    epochs: int = 56
    validation_interval: int = 4
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    residual_limit: float = 3.0
    power_limit: float = 4.0
    power_weight: float = 0.03
    zero_weight: float = 0.08
    residual_weight: float = 0.001
    distance_bins: int = 24
    random_state: int = 20260808109


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "screen", "strict"), default="screen"
    )
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=56)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def required_inputs() -> tuple:
    return (
        PATHS.round2 / "Round2_Map.ply",
        PATHS.round2 / "Round2_Train_Channel.npy",
        PATHS.round2 / "Round2_Train_Pos.npy",
        PATHS.round2 / "Round2_Test_Pos.npy",
        e87.GEOMETRY_CACHE,
        e87.ROUTE_CACHE,
        e87.GEODESIC_CACHE,
        e87.VISIBILITY_CACHE,
        e85.TARGET_PAS,
        e85.TARGET_PDP,
        e85.POINT_POWER,
        OOF_META,
        V11_CHANNEL,
        V12_CHANNEL,
    )


def ensure_inputs() -> None:
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E109 inputs: {missing}")


def normalized_sum(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = np.sum(array, axis=axes, keepdims=True, dtype=np.float64)
    return np.where(total > 0.0, array / np.maximum(total, TINY), 0.0).astype(
        np.float32
    )


def torch_normalized_sum(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    positive = value.clamp_min(0.0)
    total = positive.sum(dim=dims, keepdim=True)
    return torch.where(total > 0.0, positive / total.clamp_min(TINY), torch.zeros_like(value))


def link_scalar_features(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
) -> np.ndarray:
    serving = bs_positions[site]
    delta = np.asarray(positions, dtype=np.float64) - serving
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
    horizontal = np.maximum(np.linalg.norm(delta[:, :2], axis=1), 1e-6)
    bearing = np.arctan2(delta[:, 1], delta[:, 0])
    elevation = np.arctan2(delta[:, 2], horizontal)
    columns = [
        delta[:, 0] / 200.0,
        delta[:, 1] / 200.0,
        delta[:, 2] / 40.0,
        distance / 250.0,
        np.log1p(distance) / math.log(251.0),
        1.0 / np.sqrt(distance),
        np.sin(bearing),
        np.cos(bearing),
        np.sin(elevation),
        np.cos(elevation),
        (site == 0).astype(np.float64),
        (site == 1).astype(np.float64),
    ]
    for harmonic in (2, 3, 4):
        columns.extend((np.sin(harmonic * bearing), np.cos(harmonic * bearing)))
    for wavelength in (8.0, 16.0, 32.0, 64.0, 128.0):
        phase = 2.0 * np.pi * distance / wavelength
        columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(columns).astype(np.float32)


def _delay_angle_map_for_link(
    point: np.ndarray,
    bs: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    z_min: float,
    z_scale: float,
) -> np.ndarray:
    direct = point - bs
    direct_distance = max(float(np.linalg.norm(direct)), 1e-6)
    direct_bearing = math.atan2(float(direct[1]), float(direct[0]))

    from_bs = centers - bs[None, :]
    to_ue = point[None, :] - centers
    first = np.maximum(np.linalg.norm(from_bs, axis=1), 1e-6)
    second = np.maximum(np.linalg.norm(to_ue, axis=1), 1e-6)
    excess = np.maximum(first + second - direct_distance, 0.0)
    valid = excess <= MAX_EXCESS_M
    if not np.any(valid):
        return np.zeros((MAP_CHANNELS, DELAY_BINS, ANGLE_BINS), dtype=np.float16)

    local_centers = centers[valid]
    local_normals = normals[valid]
    local_areas = areas[valid]
    local_from_bs = from_bs[valid]
    local_to_ue = to_ue[valid]
    local_first = first[valid]
    local_second = second[valid]
    local_excess = excess[valid]

    angle = np.arctan2(local_from_bs[:, 1], local_from_bs[:, 0]) - direct_bearing
    angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
    delay_coordinate = np.log1p(local_excess) / math.log1p(MAX_EXCESS_M)
    delay_bin = np.minimum((delay_coordinate * DELAY_BINS).astype(np.int64), DELAY_BINS - 1)
    angle_bin = np.minimum(
        (((angle + np.pi) / (2.0 * np.pi)) * ANGLE_BINS).astype(np.int64),
        ANGLE_BINS - 1,
    )
    flat = delay_bin * ANGLE_BINS + angle_bin
    size = DELAY_BINS * ANGLE_BINS

    to_bs_unit = -local_from_bs / local_first[:, None]
    to_ue_unit = local_to_ue / local_second[:, None]
    bisector = to_bs_unit + to_ue_unit
    bisector /= np.maximum(np.linalg.norm(bisector, axis=1, keepdims=True), 1e-6)
    specular = np.abs(np.sum(local_normals * bisector, axis=1))
    incidence = np.sqrt(
        np.abs(np.sum(local_normals * to_bs_unit, axis=1))
        * np.abs(np.sum(local_normals * to_ue_unit, axis=1))
    )
    height = np.clip((local_centers[:, 2] - z_min) / z_scale, 0.0, 1.0)
    path_weight = local_areas / np.sqrt((1.0 + local_first) * (1.0 + local_second))

    area_sum = np.bincount(flat, weights=local_areas, minlength=size)
    weighted_spec = np.bincount(
        flat, weights=local_areas * specular, minlength=size
    )
    weighted_incidence = np.bincount(
        flat, weights=local_areas * incidence, minlength=size
    )
    weighted_height = np.bincount(
        flat, weights=local_areas * height, minlength=size
    )
    propagation_weight = np.bincount(flat, weights=path_weight, minlength=size)
    max_specular = np.zeros(size, dtype=np.float64)
    np.maximum.at(max_specular, flat, specular)

    denominator = np.maximum(area_sum, 1e-9)
    output = np.stack(
        (
            np.clip(np.log1p(area_sum) / 6.0, 0.0, 1.0),
            np.clip(weighted_spec / denominator, 0.0, 1.0),
            np.clip(max_specular, 0.0, 1.0),
            np.clip(weighted_incidence / denominator, 0.0, 1.0),
            np.clip(weighted_height / denominator, 0.0, 1.0),
            np.clip(np.log1p(100.0 * propagation_weight) / 5.0, 0.0, 1.0),
        ),
        axis=0,
    )
    return output.reshape(MAP_CHANNELS, DELAY_BINS, ANGLE_BINS).astype(np.float16)


def build_delay_angle_cache(force: bool) -> dict[str, object]:
    if DELAY_ANGLE_CACHE.is_file() and not force:
        cached = np.load(DELAY_ANGLE_CACHE, allow_pickle=False)
        return {
            "status": "reused",
            "path": PATHS.relative(DELAY_ANGLE_CACHE),
            "train_shape": list(cached["train"].shape),
            "test_shape": list(cached["test"].shape),
        }

    setup, train_positions, test_positions, train_site, test_site = load_positions()
    positions = np.concatenate((train_positions, test_positions), axis=0).astype(np.float64)
    site = np.concatenate((train_site, test_site), axis=0)
    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh).__name__)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    valid_faces = np.isfinite(centers).all(axis=1) & np.isfinite(normals).all(axis=1)
    valid_faces &= np.isfinite(areas) & (areas > 1e-8)
    centers = centers[valid_faces]
    normals = normals[valid_faces]
    areas = areas[valid_faces]
    z_min = float(np.quantile(centers[:, 2], 0.01))
    z_scale = max(float(np.quantile(centers[:, 2], 0.99) - z_min), 1.0)

    maps = np.empty(
        (len(positions), MAP_CHANNELS, DELAY_BINS, ANGLE_BINS), dtype=np.float16
    )
    for row, (point, local_site) in enumerate(zip(positions, site)):
        maps[row] = _delay_angle_map_for_link(
            point,
            setup.bs_positions[int(local_site)],
            centers,
            normals,
            areas,
            z_min,
            z_scale,
        )
        if (row + 1) % 100 == 0 or row + 1 == len(positions):
            print(f"E109 delay-angle geometry {row + 1}/{len(positions)}", flush=True)

    split = len(train_positions)
    np.savez_compressed(
        DELAY_ANGLE_CACHE,
        train=maps[:split],
        test=maps[split:],
        delay_bins=np.int32(DELAY_BINS),
        angle_bins=np.int32(ANGLE_BINS),
        max_excess_m=np.float32(MAX_EXCESS_M),
        face_count=np.int32(len(centers)),
    )
    return {
        "status": "created",
        "path": PATHS.relative(DELAY_ANGLE_CACHE),
        "train_shape": list(maps[:split].shape),
        "test_shape": list(maps[split:].shape),
        "face_count": int(len(centers)),
    }


class FeatureStore:
    def __init__(self, device: torch.device) -> None:
        setup, positions, test_positions, site, test_site = load_positions()
        self.setup = setup
        self.positions = positions.astype(np.float32)
        self.test_positions = test_positions.astype(np.float32)
        self.site = site.astype(np.int8)
        self.test_site = test_site.astype(np.int8)
        self.scalar_raw = link_scalar_features(positions, site, setup.bs_positions)

        geodesic = np.load(e87.GEODESIC_CACHE, allow_pickle=False)["train_features"]
        visibility = np.load(e87.VISIBILITY_CACHE, allow_pickle=False)["train"]
        self.map_scalar_raw = np.concatenate((geodesic, visibility), axis=1).astype(
            np.float32
        )

        route = np.load(e87.ROUTE_CACHE, allow_pickle=False)
        hit = route["train_route_hit"].astype(np.float32)
        occupancy = route["train_route_occupancy"].astype(np.float32)
        count = route["train_route_count"].astype(np.float32)
        offsets = route["fan_offsets_m"].astype(np.float32)
        fraction = (np.arange(hit.shape[2], dtype=np.float32) + 0.5) / hit.shape[2]
        route_tensor = np.stack(
            (
                hit,
                occupancy,
                np.broadcast_to(np.log1p(count)[..., None] / 3.0, hit.shape),
                np.broadcast_to(offsets[None, :, None] / 12.0, hit.shape),
                np.broadcast_to(fraction[None, None, :], hit.shape),
            ),
            axis=1,
        ).astype(np.float32)
        panorama = route["train_panorama"].astype(np.float32)
        panorama_tensor = np.concatenate((panorama, 1.0 - panorama), axis=1)
        legacy = np.load(e87.GEOMETRY_CACHE, allow_pickle=False)
        delay_angle = np.load(DELAY_ANGLE_CACHE, allow_pickle=False)

        target_pas = np.asarray(np.load(e85.TARGET_PAS, mmap_mode="r"), dtype=np.float32)
        target_pdp = np.asarray(np.load(e85.TARGET_PDP, mmap_mode="r"), dtype=np.float32)
        self.target_pas_numpy = normalized_sum(target_pas, (1, 2, 3))
        self.target_pdp_numpy = normalized_sum(target_pdp, (3,))
        self.power_numpy = np.asarray(np.load(e85.POINT_POWER), dtype=np.float64)
        self.nonzero_numpy = self.power_numpy > 0.0
        self.distance_numpy = np.linalg.norm(
            positions - setup.bs_positions[site], axis=1
        ).astype(np.float64)

        self.target_pas = torch.as_tensor(self.target_pas_numpy, device=device)
        self.target_pdp = torch.as_tensor(self.target_pdp_numpy, device=device)
        self.log_power = torch.as_tensor(
            np.log(np.maximum(self.power_numpy, 1e-30)).astype(np.float32), device=device
        )
        self.zero_target = torch.as_tensor(
            (~self.nonzero_numpy).astype(np.float32), device=device
        )
        self.route = torch.as_tensor(route_tensor, device=device)
        self.panorama = torch.as_tensor(panorama_tensor, device=device)
        self.polar = torch.as_tensor(
            legacy["train_polar"].astype(np.float32), device=device
        )
        self.corridor = torch.as_tensor(
            legacy["train_corridor"].astype(np.float32), device=device
        )
        self.delay_angle = torch.as_tensor(
            delay_angle["train"].astype(np.float32), device=device
        )
        self.device = device

    @staticmethod
    def normalization(value: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(value[rows], dtype=np.float64)
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std[std < 1e-5] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def scalar_tensors(
        self, fit_rows: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
        scalar_mean, scalar_std = self.normalization(self.scalar_raw, fit_rows)
        map_mean, map_std = self.normalization(self.map_scalar_raw, fit_rows)
        scalar = torch.as_tensor(
            np.clip((self.scalar_raw - scalar_mean) / scalar_std, -8.0, 8.0),
            device=self.device,
        )
        map_scalar = torch.as_tensor(
            np.clip((self.map_scalar_raw - map_mean) / map_std, -8.0, 8.0),
            device=self.device,
        )
        return scalar, map_scalar, {
            "scalar_mean": scalar_mean,
            "scalar_std": scalar_std,
            "map_mean": map_mean,
            "map_std": map_std,
        }

    def batch(
        self,
        rows: torch.Tensor,
        scalar: torch.Tensor,
        map_scalar: torch.Tensor,
        mode: str,
        map_rows: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        use_legacy = mode != "distance_only"
        use_delay_angle = mode in ("full_geometry", "shuffled_full_geometry")
        if not use_legacy:
            return {
                "scalar": scalar[rows],
                "map_scalar": torch.zeros_like(map_scalar[rows]),
                "route": torch.zeros_like(self.route[rows]),
                "panorama": torch.zeros_like(self.panorama[rows]),
                "polar": torch.zeros_like(self.polar[rows]),
                "corridor": torch.zeros_like(self.corridor[rows]),
                "delay_angle": torch.zeros_like(self.delay_angle[rows]),
            }
        selected = map_rows[rows]
        return {
            "scalar": scalar[rows],
            "map_scalar": map_scalar[selected],
            "route": self.route[selected],
            "panorama": self.panorama[selected],
            "polar": self.polar[selected],
            "corridor": self.corridor[selected],
            "delay_angle": (
                self.delay_angle[selected]
                if use_delay_angle
                else torch.zeros_like(self.delay_angle[selected])
            ),
        }


def _interpolate_bins(
    centers: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    if len(centers) == 1:
        return np.broadcast_to(values[0], (len(query), *values.shape[1:])).copy()
    upper = np.searchsorted(centers, query, side="right")
    upper = np.clip(upper, 1, len(centers) - 1)
    lower = upper - 1
    denominator = np.maximum(centers[upper] - centers[lower], 1e-6)
    weight = np.clip((query - centers[lower]) / denominator, 0.0, 1.0)
    reshape = (len(query),) + (1,) * (values.ndim - 1)
    return (
        (1.0 - weight.reshape(reshape)) * values[lower]
        + weight.reshape(reshape) * values[upper]
    )


def distance_baseline(
    store: FeatureStore,
    fit_rows: np.ndarray,
    bins: int,
) -> dict[str, torch.Tensor]:
    count = len(store.positions)
    pas = np.empty_like(store.target_pas_numpy)
    pdp = np.empty_like(store.target_pdp_numpy)
    log_power = np.empty(count, dtype=np.float32)
    zero_logit = np.empty(count, dtype=np.float32)
    for site in np.unique(store.site):
        local_fit = fit_rows[
            (store.site[fit_rows] == site) & store.nonzero_numpy[fit_rows]
        ]
        if len(local_fit) < bins:
            raise RuntimeError(f"site {site} has only {len(local_fit)} nonzero fit rows")
        ordered = local_fit[np.argsort(store.distance_numpy[local_fit])]
        groups = [group for group in np.array_split(ordered, bins) if len(group)]
        centers = np.asarray(
            [np.mean(store.distance_numpy[group]) for group in groups], dtype=np.float64
        )
        pas_values = np.stack(
            [np.mean(store.target_pas_numpy[group], axis=0) for group in groups]
        )
        pdp_values = np.stack(
            [np.mean(store.target_pdp_numpy[group], axis=0) for group in groups]
        )
        power_values = np.asarray(
            [np.median(np.log(store.power_numpy[group])) for group in groups],
            dtype=np.float32,
        )

        zero_fit = fit_rows[store.site[fit_rows] == site]
        zero_ordered = zero_fit[np.argsort(store.distance_numpy[zero_fit])]
        zero_groups = [group for group in np.array_split(zero_ordered, bins) if len(group)]
        zero_centers = np.asarray(
            [np.mean(store.distance_numpy[group]) for group in zero_groups],
            dtype=np.float64,
        )
        zero_rate = np.asarray(
            [np.mean(~store.nonzero_numpy[group]) for group in zero_groups],
            dtype=np.float64,
        )
        zero_rate = np.clip(zero_rate, 1e-3, 1.0 - 1e-3)
        zero_values = np.log(zero_rate / (1.0 - zero_rate)).astype(np.float32)

        rows = np.flatnonzero(store.site == site)
        query_distance = store.distance_numpy[rows]
        pas[rows] = _interpolate_bins(centers, pas_values, query_distance)
        pdp[rows] = _interpolate_bins(centers, pdp_values, query_distance)
        log_power[rows] = _interpolate_bins(
            centers, power_values[:, None], query_distance
        )[:, 0]
        zero_logit[rows] = _interpolate_bins(
            zero_centers, zero_values[:, None], query_distance
        )[:, 0]

    pas = normalized_sum(pas, (1, 2, 3))
    pdp = normalized_sum(pdp, (3,))
    return {
        "pas": torch.as_tensor(pas, device=store.device),
        "pdp": torch.as_tensor(pdp, device=store.device),
        "log_power": torch.as_tensor(log_power, device=store.device),
        "zero_logit": torch.as_tensor(zero_logit, device=store.device),
    }


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.input = nn.Linear(hidden, 2 * hidden)
        self.output = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, right = self.input(self.norm(value)).chunk(2, dim=-1)
        return value + self.dropout(self.output(torch.nn.functional.silu(left) * right))


def conv2d_encoder(channels: int, output: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, 48, 3, padding=1),
        nn.GroupNorm(8, 48),
        nn.SiLU(),
        nn.Conv2d(48, 96, 3, stride=(1, 2), padding=1),
        nn.GroupNorm(8, 96),
        nn.SiLU(),
        nn.Conv2d(96, 128, 3, stride=(2, 2), padding=1),
        nn.GroupNorm(8, 128),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d((2, 4)),
        nn.Flatten(),
        nn.Linear(128 * 2 * 4, output),
        nn.SiLU(),
    )


class GeometryConditionedTransferOperator(nn.Module):
    def __init__(self, scalar_dim: int, map_dim: int, config: Config) -> None:
        super().__init__()
        branch = config.branch
        self.scalar = nn.Sequential(
            nn.Linear(scalar_dim + map_dim, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, branch),
            nn.SiLU(),
        )
        self.route = conv2d_encoder(5, branch)
        self.polar = conv2d_encoder(4, branch)
        self.corridor = conv2d_encoder(4, branch)
        self.delay_angle = conv2d_encoder(MAP_CHANNELS, branch)
        self.panorama = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128 * 4, branch),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(6 * branch, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden, config.dropout)
            for _ in range(config.residual_blocks)
        )
        self.pas_head = nn.Linear(config.hidden, PAS_SIZE)
        self.pdp_head = nn.Linear(config.hidden, PDP_SIZE)
        self.power_head = nn.Linear(config.hidden, 1)
        self.zero_head = nn.Linear(config.hidden, 1)
        self.config = config
        self.initialize_identity()

    def initialize_identity(self) -> None:
        for head in (self.pas_head, self.pdp_head, self.power_head, self.zero_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        baseline: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hidden = torch.cat(
            (
                self.scalar(torch.cat((batch["scalar"], batch["map_scalar"]), dim=1)),
                self.route(batch["route"]),
                self.panorama(batch["panorama"]),
                self.polar(batch["polar"]),
                self.corridor(batch["corridor"]),
                self.delay_angle(batch["delay_angle"]),
            ),
            dim=1,
        )
        hidden = self.fusion(hidden)
        for block in self.blocks:
            hidden = block(hidden)

        pas_delta = self.config.residual_limit * torch.tanh(
            self.pas_head(hidden).reshape(-1, MP, MH, MV, N_UE)
        )
        pdp_delta = self.config.residual_limit * torch.tanh(
            self.pdp_head(hidden).reshape(-1, MP, N_UE, S_SC)
        )
        pas = torch_normalized_sum(
            baseline["pas"] * torch.exp(pas_delta), (1, 2, 3)
        )
        pdp = torch_normalized_sum(baseline["pdp"] * torch.exp(pdp_delta), (3,))
        power_delta = self.config.power_limit * torch.tanh(
            self.power_head(hidden).squeeze(1)
        )
        return {
            "pas": pas,
            "pdp": pdp,
            "log_power": baseline["log_power"] + power_delta,
            "zero_logit": baseline["zero_logit"] + self.zero_head(hidden).squeeze(1),
            "pas_delta": pas_delta,
            "pdp_delta": pdp_delta,
            "power_delta": power_delta,
        }


def cosine_score(
    prediction_pas: torch.Tensor,
    prediction_pdp: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pas_dot = (prediction_pas * target_pas).sum(dim=(1, 2, 3), dtype=torch.float64)
    pas_norm_product = (
        prediction_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * target_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    pas_denominator = torch.sqrt(
        torch.where(
            pas_norm_product == 0.0,
            torch.full_like(pas_norm_product, 1e-24),
            pas_norm_product,
        )
    )
    pas = (pas_dot / pas_denominator).mean(dim=1)

    pdp_dot = (prediction_pdp * target_pdp).sum(dim=3, dtype=torch.float64)
    pdp_norm_product = (
        prediction_pdp.square().sum(dim=3, dtype=torch.float64)
        * target_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_denominator = torch.sqrt(
        torch.where(
            pdp_norm_product == 0.0,
            torch.full_like(pdp_norm_product, 1e-24),
            pdp_norm_product,
        )
    )
    pdp = (pdp_dot / pdp_denominator).mean(dim=(1, 2))
    return pas.float(), pdp.float(), (0.5 * (pas + pdp)).float()


def transfer_loss(
    prediction: dict[str, torch.Tensor],
    store: FeatureStore,
    rows: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    pas, pdp, score = cosine_score(
        prediction["pas"],
        prediction["pdp"],
        store.target_pas[rows],
        store.target_pdp[rows],
    )
    nonzero = store.zero_target[rows] < 0.5
    spectral = 1.0 - score[nonzero].mean() if torch.any(nonzero) else score.mean() * 0.0
    power = torch.nn.functional.smooth_l1_loss(
        prediction["log_power"][nonzero],
        store.log_power[rows][nonzero],
    ) if torch.any(nonzero) else spectral * 0.0
    zero = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction["zero_logit"], store.zero_target[rows]
    )
    regularization = (
        prediction["pas_delta"].square().mean()
        + prediction["pdp_delta"].square().mean()
        + 0.25 * prediction["power_delta"].square().mean()
    )
    return (
        spectral
        + config.power_weight * power
        + config.zero_weight * zero
        + config.residual_weight * regularization
    )


def mode_map_rows(store: FeatureStore, mode: str, seed: int) -> torch.Tensor:
    rows = np.arange(len(store.positions), dtype=np.int64)
    if mode == "shuffled_full_geometry":
        rng = np.random.default_rng(seed)
        for site in np.unique(store.site):
            local = np.flatnonzero(store.site == site)
            rows[local] = rng.permutation(local)
    return torch.as_tensor(rows, device=store.device)


@torch.inference_mode()
def predict_rows(
    model: GeometryConditionedTransferOperator,
    store: FeatureStore,
    rows: np.ndarray,
    scalar: torch.Tensor,
    map_scalar: torch.Tensor,
    baseline: dict[str, torch.Tensor],
    mode: str,
    map_rows: torch.Tensor,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {
        "pas": [],
        "pdp": [],
        "log_power": [],
        "zero_logit": [],
    }
    for start in range(0, len(rows), batch_size):
        local = torch.as_tensor(rows[start : start + batch_size], device=store.device)
        batch = store.batch(local, scalar, map_scalar, mode, map_rows)
        base = {key: value[local] for key, value in baseline.items()}
        prediction = model(batch, base)
        for key in outputs:
            outputs[key].append(prediction[key].detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in outputs.items()}


def endpoint_metrics(
    prediction: dict[str, np.ndarray],
    store: FeatureStore,
    rows: np.ndarray,
) -> dict[str, float | int]:
    pas, pdp, mean = cosine_score(
        torch.as_tensor(prediction["pas"]),
        torch.as_tensor(prediction["pdp"]),
        torch.as_tensor(store.target_pas_numpy[rows]),
        torch.as_tensor(store.target_pdp_numpy[rows]),
    )
    nonzero = store.nonzero_numpy[rows]
    if np.any(nonzero):
        power_mae = float(
            np.mean(
                np.abs(
                    prediction["log_power"][nonzero]
                    - np.log(store.power_numpy[rows][nonzero])
                )
            )
        )
    else:
        power_mae = float("nan")
    zero_truth = (~nonzero).astype(np.int8)
    zero_probability = expit(prediction["zero_logit"])
    zero_auc = (
        float(roc_auc_score(zero_truth, zero_probability))
        if len(np.unique(zero_truth)) == 2
        else float("nan")
    )
    return {
        "count": int(len(rows)),
        "pas": float(pas.mean().item()),
        "pdp": float(pdp.mean().item()),
        "mean": float(mean.mean().item()),
        "nonzero_count": int(np.sum(nonzero)),
        "log_power_mae": power_mae,
        "zero_auc": zero_auc,
    }


def train_model(
    store: FeatureStore,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    baseline: dict[str, torch.Tensor],
    mode: str,
    config: Config,
    max_epochs: int,
    seed: int,
    select_epoch: bool,
) -> tuple[GeometryConditionedTransferOperator, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    scalar, map_scalar, normalization = store.scalar_tensors(train_rows)
    map_rows = mode_map_rows(store, mode, seed + 701)
    model = GeometryConditionedTransferOperator(
        scalar.shape[1], map_scalar.shape[1], config
    ).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, object]] = []
    initial = predict_rows(
        model,
        store,
        validation_rows,
        scalar,
        map_scalar,
        baseline,
        mode,
        map_rows,
        config.batch_size,
    )
    initial_metric = endpoint_metrics(initial, store, validation_rows)
    history.append({"epoch": 0, "validation": initial_metric, "loss": None})
    best_metric = float(initial_metric["mean"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(seed + 11)

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), config.batch_size):
            local = torch.as_tensor(
                order[start : start + config.batch_size], device=store.device
            )
            batch = store.batch(local, scalar, map_scalar, mode, map_rows)
            base = {key: value[local] for key, value in baseline.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch, base)
            loss = transfer_loss(prediction, store, local, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite E109 loss mode={mode} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

        if epoch % config.validation_interval == 0 or epoch == max_epochs:
            predicted = predict_rows(
                model,
                store,
                validation_rows,
                scalar,
                map_scalar,
                baseline,
                mode,
                map_rows,
                config.batch_size,
            )
            metric = endpoint_metrics(predicted, store, validation_rows)
            record = {
                "epoch": epoch,
                "validation": metric,
                "loss": float(np.mean(losses)),
            }
            history.append(record)
            if select_epoch and float(metric["mean"]) > best_metric:
                best_metric = float(metric["mean"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E109 mode={mode} epoch={epoch} loss={record['loss']:.6f} "
                f"validation={metric['mean']:.6f}",
                flush=True,
            )

    if select_epoch:
        model.load_state_dict(best_state)
    else:
        best_epoch = max_epochs
        best_metric = float(history[-1]["validation"]["mean"])
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_mean": best_metric,
        "history": history,
        "normalization": {
            key: {
                "shape": list(value.shape),
                "finite": bool(np.isfinite(value).all()),
            }
            for key, value in normalization.items()
        },
    }


def run_outer(
    store: FeatureStore,
    folds: list,
    outer: int,
    mode: str,
    config: Config,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    validation_fold = (outer + 1) % len(folds)
    outer_fold = folds[outer]
    validation = folds[validation_fold]
    screen_rows = np.intersect1d(
        outer_fold.fit_indices, validation.fit_indices, assume_unique=False
    )
    baseline_screen = distance_baseline(store, screen_rows, config.distance_bins)
    _, screen = train_model(
        store,
        screen_rows,
        validation.query_indices,
        baseline_screen,
        mode,
        config,
        config.epochs,
        config.random_state + outer * 101 + MODES.index(mode) * 1009,
        True,
    )
    selected_epoch = int(screen["best_epoch"])

    baseline_refit = distance_baseline(
        store, outer_fold.fit_indices, config.distance_bins
    )
    refit_model, refit = train_model(
        store,
        outer_fold.fit_indices,
        outer_fold.query_indices,
        baseline_refit,
        mode,
        config,
        selected_epoch,
        config.random_state + outer * 101 + MODES.index(mode) * 1009 + 53,
        False,
    )
    scalar, map_scalar, _ = store.scalar_tensors(outer_fold.fit_indices)
    map_rows = mode_map_rows(
        store,
        mode,
        config.random_state + outer * 101 + MODES.index(mode) * 1009 + 53 + 701,
    )
    held_prediction = predict_rows(
        refit_model,
        store,
        outer_fold.query_indices,
        scalar,
        map_scalar,
        baseline_refit,
        mode,
        map_rows,
        config.batch_size,
    )
    held_metric = endpoint_metrics(held_prediction, store, outer_fold.query_indices)
    return held_prediction, {
        "outer": int(outer),
        "validation_fold": int(validation_fold),
        "screen_count": int(len(screen_rows)),
        "refit_count": int(len(outer_fold.fit_indices)),
        "held_count": int(len(outer_fold.query_indices)),
        "selected_epoch": selected_epoch,
        "screen": screen,
        "refit": refit,
        "held": held_metric,
    }


def projection_metrics(
    base_path,
    endpoint_pas: np.ndarray,
    endpoint_pdp: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    beta: float,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, object]:
    base_np = np.load(base_path, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    output = np.empty(base_np.shape, dtype=np.complex64)
    for start in range(0, len(base_np), batch_size):
        end = min(start + batch_size, len(base_np))
        base = torch.as_tensor(np.asarray(base_np[start:end]), device=device)
        pas = torch.as_tensor(endpoint_pas[start:end], device=device)
        pdp = torch.as_tensor(endpoint_pdp[start:end], device=device)
        expanded_pas = pas.unsqueeze(-1).expand(-1, MP, MH, MV, N_UE, S_SC)
        expanded_pdp = (
            pdp[:, :, None, None, :, :]
            .expand(-1, MP, MH, MV, N_UE, S_SC)
            .reshape(-1, MP * MH * MV, N_UE, S_SC)
        )
        desired_pas = endpoint_target(
            pas_distribution(base), expanded_pas, beta, "sqrt", (1, 2, 3)
        )
        desired_pdp = endpoint_target(
            pdp_distribution(base), expanded_pdp, beta, "sqrt", (3,)
        )
        output[start:end] = (
            project_joint(base, desired_pas, desired_pdp, 4, "pas_pdp")
            .detach()
            .cpu()
            .numpy()
            .astype(np.complex64)
        )
    truth = np.asarray(truth_all[query_ids], dtype=np.complex64)
    truth_tensor = torch.as_tensor(truth, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    values = point_metrics(
        torch.as_tensor(output, device=device),
        truth_tensor,
        truth_pas,
        truth_pdp,
    )
    base_values = point_metrics(
        torch.as_tensor(np.asarray(base_np), device=device),
        truth_tensor,
        truth_pas,
        truth_pdp,
    )
    selected = statistics(*values)
    baseline = statistics(*base_values)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        local = statistics(*values, mask)
        local_base = statistics(*base_values, mask)
        folds.append(
            {
                "fold": fold,
                "selected": local,
                "baseline": local_base,
                "delta": float(local["score"] - local_base["score"]),
            }
        )
    return {
        "selected": selected,
        "baseline": baseline,
        "delta": float(selected["score"] - baseline["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


def run_screen(store: FeatureStore, config: Config, outer: int) -> dict[str, object]:
    _, train_positions, test_positions, train_site, test_site = load_positions()
    folds = build_spatial_hole_folds(
        train_positions, test_positions, train_site, test_site
    )
    modes = {}
    for mode in MODES:
        prediction, report = run_outer(store, folds, outer, mode, config)
        modes[mode] = report
        modes[mode]["prediction_finite"] = bool(
            all(np.isfinite(value).all() for value in prediction.values())
        )
    full = float(modes["full_geometry"]["held"]["mean"])
    distance = float(modes["distance_only"]["held"]["mean"])
    legacy = float(modes["legacy_geometry"]["held"]["mean"])
    shuffled = float(modes["shuffled_full_geometry"]["held"]["mean"])
    return {
        "experiment": VERSION,
        "status": "outer_screen_complete",
        "outer": int(outer),
        "config": asdict(config),
        "modes": modes,
        "margins": {
            "full_over_distance": full - distance,
            "full_over_legacy": full - legacy,
            "full_over_shuffled": full - shuffled,
        },
        "screen_gate": {
            "minimum_full_over_distance": 0.005,
            "minimum_full_over_shuffled": 0.005,
            "passed": bool(full - distance >= 0.005 and full - shuffled >= 0.005),
        },
    }


def run_strict(store: FeatureStore, config: Config, device: torch.device) -> dict[str, object]:
    _, train_positions, test_positions, train_site, test_site = load_positions()
    folds = build_spatial_hole_folds(
        train_positions, test_positions, train_site, test_site
    )
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    row_lookup = np.full(len(store.positions), -1, dtype=np.int64)
    row_lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)

    stitched = {
        mode: {
            "pas": np.empty((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32),
            "pdp": np.empty((len(query_ids), MP, N_UE, S_SC), dtype=np.float32),
            "log_power": np.empty(len(query_ids), dtype=np.float32),
            "zero_logit": np.empty(len(query_ids), dtype=np.float32),
        }
        for mode in MODES
    }
    fold_reports = {mode: [] for mode in MODES}
    for outer in range(5):
        for mode in MODES:
            prediction, report = run_outer(store, folds, outer, mode, config)
            target_rows = row_lookup[folds[outer].query_indices]
            if np.any(target_rows < 0):
                raise RuntimeError(f"outer {outer} query rows missing from canonical OOF")
            for key in stitched[mode]:
                stitched[mode][key][target_rows] = prediction[key]
            fold_reports[mode].append(report)

    endpoint_reports = {}
    projection_reports = {}
    for mode in MODES:
        endpoint = endpoint_metrics(stitched[mode], store, query_ids)
        fold_endpoint = []
        for fold in range(5):
            mask = fold_id == fold
            local_prediction = {key: value[mask] for key, value in stitched[mode].items()}
            fold_endpoint.append(
                {"fold": fold, **endpoint_metrics(local_prediction, store, query_ids[mask])}
            )
        endpoint_reports[mode] = {"overall": endpoint, "folds": fold_endpoint}
        projection_reports[mode] = {
            "from_v11_beta020": projection_metrics(
                V11_CHANNEL,
                stitched[mode]["pas"],
                stitched[mode]["pdp"],
                query_ids,
                fold_id,
                0.20,
                device,
            ),
            "from_v12_beta010": projection_metrics(
                V12_CHANNEL,
                stitched[mode]["pas"],
                stitched[mode]["pdp"],
                query_ids,
                fold_id,
                0.10,
                device,
            ),
        }

    np.savez_compressed(
        STRICT_ENDPOINTS,
        query_ids=query_ids,
        fold_id=fold_id,
        **{
            f"{mode}_{key}": value.astype(np.float16)
            for mode, endpoint in stitched.items()
            for key, value in endpoint.items()
        },
    )
    full_endpoint = float(endpoint_reports["full_geometry"]["overall"]["mean"])
    distance_endpoint = float(endpoint_reports["distance_only"]["overall"]["mean"])
    shuffled_endpoint = float(
        endpoint_reports["shuffled_full_geometry"]["overall"]["mean"]
    )
    deployment = projection_reports["full_geometry"]["from_v12_beta010"]
    information_passed = bool(
        full_endpoint - distance_endpoint >= 0.005
        and full_endpoint - shuffled_endpoint >= 0.005
    )
    deployment_passed = bool(
        deployment["delta"] >= 0.001
        and deployment["positive_folds"] >= 4
        and deployment["worst_fold_delta"] >= -0.001
    )
    return {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "A straight-line distance baseline plus link-conditioned geometry can learn "
            "a transferable channel function better than distance alone or shuffled geometry."
        ),
        "config": asdict(config),
        "fold_training": fold_reports,
        "endpoints": endpoint_reports,
        "complete_channel_projection": projection_reports,
        "margins": {
            "full_over_distance_endpoint": full_endpoint - distance_endpoint,
            "full_over_shuffled_endpoint": full_endpoint - shuffled_endpoint,
        },
        "decision": {
            "information_gate_passed": information_passed,
            "deployment_gate_passed": deployment_passed,
            "generate_v13": bool(information_passed and deployment_passed),
        },
        "outputs": {
            "endpoints": PATHS.relative(STRICT_ENDPOINTS),
            "finite": bool(
                all(
                    np.isfinite(value).all()
                    for endpoint in stitched.values()
                    for value in endpoint.values()
                )
            ),
        },
    }


def write_report(path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_inputs()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prepare = build_delay_angle_cache(args.force_prepare)
    if args.stage == "prepare":
        payload = {
            "experiment": VERSION,
            "status": "prepare_complete",
            "geometry": prepare,
            "seconds": time.time() - started,
        }
        write_report(ANALYSIS_DIR / "prepare.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = Config(batch_size=args.batch_size, epochs=args.epochs)
    store = FeatureStore(device)
    if args.stage == "screen":
        payload = run_screen(store, config, args.outer)
        path = SCREEN_REPORT
    else:
        payload = run_strict(store, config, device)
        path = STRICT_REPORT
    payload["geometry"] = prepare
    payload["seconds"] = time.time() - started
    write_report(path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
