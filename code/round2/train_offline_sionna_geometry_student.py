#!/usr/bin/env python3
"""Distill Sionna spectra into a map-only student and evaluate strict holes."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import trimesh

from analyze_e86_surface_endpoint_v11_blend import endpoint_target
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from build_geodesic_corridor_features import (
    _grid_bounds,
    _rasterize_walls,
    _xy_to_rc,
)
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E87-offline-sionna-geometry-student"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "offline_sionna_geometry_student"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_offline_sionna_geometry_student"
GEOMETRY_CACHE = ARTIFACT_DIR / "official_map_geometry_rasters.npz"
TEACHER_CACHE = ARTIFACT_DIR / "raw_sionna_sufficient_teacher.npz"
OOF_PREDICTIONS = ARTIFACT_DIR / "strict_oof_student_endpoints.npz"
REPORT = ANALYSIS_DIR / "offline_sionna_geometry_student.json"

ROUTE_CACHE = (
    PATHS.artifacts / "round2" / "route_topology" / "round2_route_topology_features.npz"
)
GEODESIC_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
VISIBILITY_CACHE = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
V11_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "canonical_500_version_replay"
    / "V11_Canonical_500_OOF_Channel.npy"
)
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PAS_SIZE = MP * MH * MV * N_UE
PDP_SIZE = MP * N_UE * S_SC
TINY = 1e-30

POLAR_RADII_M = np.linspace(2.0, 64.0, 16, dtype=np.float32)
POLAR_DIRECTIONS = 64
CORRIDOR_OFFSETS_M = np.linspace(-30.0, 30.0, 13, dtype=np.float32)
CORRIDOR_SAMPLES = 64


@dataclass(frozen=True)
class Config:
    hidden: int = 512
    residual_blocks: int = 5
    dropout: float = 0.08
    pretrain_epochs: int = 48
    finetune_epochs: int = 72
    validation_interval: int = 4
    batch_size: int = 128
    learning_rate: float = 4e-4
    finetune_learning_rate: float = 2e-4
    weight_decay: float = 2e-4
    teacher_retain: float = 0.25
    random_state: int = 2026080787


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "screen", "strict"), default="strict"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--pretrain-epochs", type=int, default=48)
    parser.add_argument("--finetune-epochs", type=int, default=72)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def required_inputs() -> tuple:
    return (
        PATHS.round2 / "Round2_Map.ply",
        PATHS.round2 / "Round2_Train_Channel.npy",
        ROUTE_CACHE,
        GEODESIC_CACHE,
        VISIBILITY_CACHE,
        OOF_META,
        V11_CHANNEL,
        V12_CHANNEL,
        e85.PATH_CACHE,
        e85.BASE_COORDINATE,
        e85.PORT_PRIOR,
        e85.TARGET_PAS,
        e85.TARGET_PDP,
        e85.POINT_POWER,
    )


def sample_grid(
    grid: np.ndarray,
    xy: np.ndarray,
    x_min: float,
    y_min: float,
) -> np.ndarray:
    original_shape = xy.shape[:-1]
    flat = xy.reshape(-1, 2)
    row, col = _xy_to_rc(
        flat, x_min, y_min, grid.shape[0], grid.shape[1]
    )
    return np.asarray(grid[row, col]).reshape(original_shape)


def build_geometry_cache(force: bool) -> dict[str, object]:
    if GEOMETRY_CACHE.is_file() and not force:
        cached = np.load(GEOMETRY_CACHE, allow_pickle=False)
        return {
            "status": "reused",
            "path": PATHS.relative(GEOMETRY_CACHE),
            "arrays": {
                key: {"shape": list(cached[key].shape), "dtype": str(cached[key].dtype)}
                for key in cached.files
            },
        }

    setup, train_positions, test_positions, train_site, test_site = load_positions()
    positions = np.concatenate((train_positions, test_positions), axis=0)
    site = np.concatenate((train_site, test_site), axis=0)
    bounds = np.concatenate((positions, setup.bs_positions), axis=0)
    x_min, y_min, height, width = _grid_bounds(bounds)
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

    polar = np.empty(
        (len(positions), len(grids), len(POLAR_RADII_M), POLAR_DIRECTIONS),
        dtype=np.float16,
    )
    corridor = np.empty(
        (
            len(positions),
            len(grids),
            len(CORRIDOR_OFFSETS_M),
            CORRIDOR_SAMPLES,
        ),
        dtype=np.float16,
    )
    angles = np.linspace(
        0.0, 2.0 * np.pi, POLAR_DIRECTIONS, endpoint=False, dtype=np.float32
    )
    fractions = np.linspace(0.0, 1.0, CORRIDOR_SAMPLES, dtype=np.float32)
    for start in range(0, len(positions), 128):
        end = min(start + 128, len(positions))
        point = positions[start:end, :2].astype(np.float32)
        bs = setup.bs_positions[site[start:end], :2].astype(np.float32)
        toward_bs = bs - point
        distance = np.maximum(np.linalg.norm(toward_bs, axis=1), 1e-6)
        bearing = np.arctan2(toward_bs[:, 1], toward_bs[:, 0])
        direction = np.stack(
            (
                np.cos(bearing[:, None] + angles[None]),
                np.sin(bearing[:, None] + angles[None]),
            ),
            axis=-1,
        )
        polar_xy = (
            point[:, None, None, :]
            + POLAR_RADII_M[None, :, None, None] * direction[:, None, :, :]
        )

        route = point - bs
        route_unit = route / distance[:, None]
        perpendicular = np.stack((-route_unit[:, 1], route_unit[:, 0]), axis=1)
        center = (
            bs[:, None, :]
            + fractions[None, :, None] * route[:, None, :]
        )
        corridor_xy = (
            center[:, None, :, :]
            + CORRIDOR_OFFSETS_M[None, :, None, None]
            * perpendicular[:, None, None, :]
        )
        for channel, grid in enumerate(grids):
            polar[start:end, channel] = sample_grid(
                grid, polar_xy, x_min, y_min
            ).astype(np.float16)
            corridor[start:end, channel] = sample_grid(
                grid, corridor_xy, x_min, y_min
            ).astype(np.float16)
        print(f"E87 official-map rasters {end}/{len(positions)}", flush=True)

    split = len(train_positions)
    np.savez_compressed(
        GEOMETRY_CACHE,
        train_polar=polar[:split],
        test_polar=polar[split:],
        train_corridor=corridor[:split],
        test_corridor=corridor[split:],
        polar_radii_m=POLAR_RADII_M,
        corridor_offsets_m=CORRIDOR_OFFSETS_M,
        x_min=np.float32(x_min),
        y_min=np.float32(y_min),
        grid_shape=np.asarray((height, width), dtype=np.int32),
    )
    return {
        "status": "created",
        "path": PATHS.relative(GEOMETRY_CACHE),
        "train_polar_shape": list(polar[:split].shape),
        "train_corridor_shape": list(corridor[:split].shape),
        "test_polar_shape": list(polar[split:].shape),
        "test_corridor_shape": list(corridor[split:].shape),
    }


@torch.inference_mode()
def build_teacher_cache(
    device: torch.device, batch_size: int, force: bool
) -> dict[str, object]:
    if TEACHER_CACHE.is_file() and not force:
        cached = np.load(TEACHER_CACHE, allow_pickle=False)
        return {
            "status": "reused",
            "path": PATHS.relative(TEACHER_CACHE),
            "arrays": {
                key: {"shape": list(cached[key].shape), "dtype": str(cached[key].dtype)}
                for key in cached.files
            },
        }

    fraction = np.load(e85.PATH_CACHE / "energy_fraction.npy", mmap_mode="r")
    presence = np.load(e85.PATH_CACHE / "presence.npy", mmap_mode="r")
    coordinate = np.load(e85.BASE_COORDINATE, mmap_mode="r")
    port = np.load(e85.PORT_PRIOR, mmap_mode="r")
    count = len(fraction)
    teacher_pas = np.empty((count, MP, MH, MV, N_UE), dtype=np.float16)
    teacher_pdp = np.empty((count, MP, N_UE, S_SC), dtype=np.float16)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        local_fraction = np.asarray(fraction[start:end], dtype=np.float32)
        local_presence = np.asarray(presence[start:end], dtype=np.float32)
        anchored = np.maximum(local_fraction, 1e-12) * np.sqrt(
            np.maximum(local_presence, 1.0) / 24.0
        )
        mass = anchored / np.maximum(anchored.sum(axis=1, keepdims=True), TINY)
        local_port = np.asarray(port[start:end], dtype=np.float32)
        local_port /= np.maximum(
            local_port.sum(axis=(2, 3), keepdims=True), TINY
        )
        output = {
            "mass": torch.as_tensor(mass, device=device),
            "port": torch.as_tensor(local_port, device=device),
            "coordinate": torch.as_tensor(
                np.asarray(coordinate[start:end], dtype=np.float32), device=device
            ),
        }
        pas, pdp = e85.render_spectra(output)
        teacher_pas[start:end] = pas.cpu().numpy().astype(np.float16)
        teacher_pdp[start:end] = pdp.cpu().numpy().astype(np.float16)
        print(f"E87 Sionna teacher {end}/{count}", flush=True)
    np.savez_compressed(TEACHER_CACHE, pas=teacher_pas, pdp=teacher_pdp)
    return {
        "status": "created",
        "path": PATHS.relative(TEACHER_CACHE),
        "pas_shape": list(teacher_pas.shape),
        "pdp_shape": list(teacher_pdp.shape),
    }


def position_features(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
) -> np.ndarray:
    serving = bs_positions[site]
    delta = positions - serving
    distance = np.maximum(np.linalg.norm(delta[:, :2], axis=1), 1e-6)
    bearing = np.arctan2(delta[:, 1], delta[:, 0])
    columns = [
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        delta[:, 0],
        delta[:, 1],
        delta[:, 2],
        distance,
        np.log1p(distance),
        np.sin(bearing),
        np.cos(bearing),
        (site == 0).astype(np.float64),
        (site == 1).astype(np.float64),
    ]
    wavelengths = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    for value in (positions[:, 0], positions[:, 1], delta[:, 0], delta[:, 1]):
        for wavelength in wavelengths:
            phase = 2.0 * np.pi * value / wavelength
            columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(columns).astype(np.float32)


class FeatureStore:
    def __init__(self, device: torch.device) -> None:
        setup, positions, _, site, _ = load_positions()
        self.positions = positions.astype(np.float32)
        self.site = site.astype(np.int8)
        self.position_raw = position_features(positions, site, setup.bs_positions)

        geodesic = np.load(GEODESIC_CACHE, allow_pickle=False)["train_features"]
        visibility = np.load(VISIBILITY_CACHE, allow_pickle=False)["train"]
        self.map_scalar_raw = np.concatenate((geodesic, visibility), axis=1).astype(
            np.float32
        )

        route = np.load(ROUTE_CACHE, allow_pickle=False)
        hit = route["train_route_hit"].astype(np.float32)
        occupancy = route["train_route_occupancy"].astype(np.float32)
        count = route["train_route_count"].astype(np.float32)
        offsets = route["fan_offsets_m"].astype(np.float32)
        fraction = (
            np.arange(hit.shape[2], dtype=np.float32) + 0.5
        ) / float(hit.shape[2])
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
        geometry = np.load(GEOMETRY_CACHE, allow_pickle=False)

        teacher = np.load(TEACHER_CACHE, allow_pickle=False)
        self.teacher_pas = torch.as_tensor(
            teacher["pas"].astype(np.float32), device=device
        )
        self.teacher_pdp = torch.as_tensor(
            teacher["pdp"].astype(np.float32), device=device
        )
        self.official_pas = torch.as_tensor(
            np.array(np.load(e85.TARGET_PAS, mmap_mode="r"), dtype=np.float32, copy=True),
            device=device,
        )
        self.official_pdp = torch.as_tensor(
            np.array(np.load(e85.TARGET_PDP, mmap_mode="r"), dtype=np.float32, copy=True),
            device=device,
        )
        self.power = np.asarray(np.load(e85.POINT_POWER), dtype=np.float64)
        self.route = torch.as_tensor(route_tensor, device=device)
        self.panorama = torch.as_tensor(panorama_tensor, device=device)
        self.polar = torch.as_tensor(
            geometry["train_polar"].astype(np.float32), device=device
        )
        self.corridor = torch.as_tensor(
            geometry["train_corridor"].astype(np.float32), device=device
        )
        self.device = device

    @staticmethod
    def normalization(
        value: np.ndarray, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(value[rows], dtype=np.float64)
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std[std < 1e-5] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def normalized_scalars(
        self, fit_rows: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
        position_mean, position_std = self.normalization(self.position_raw, fit_rows)
        map_mean, map_std = self.normalization(self.map_scalar_raw, fit_rows)
        position = torch.as_tensor(
            np.clip((self.position_raw - position_mean) / position_std, -8.0, 8.0),
            device=self.device,
        )
        map_scalar = torch.as_tensor(
            np.clip((self.map_scalar_raw - map_mean) / map_std, -8.0, 8.0),
            device=self.device,
        )
        return position, map_scalar, {
            "position_mean": position_mean,
            "position_std": position_std,
            "map_mean": map_mean,
            "map_std": map_std,
        }

    def batch(
        self,
        rows: torch.Tensor,
        position: torch.Tensor,
        map_scalar: torch.Tensor,
        use_map: bool,
        map_rows: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if map_rows is None:
            map_rows = rows
        if use_map:
            return {
                "position": position[rows],
                "map_scalar": map_scalar[map_rows],
                "route": self.route[map_rows],
                "panorama": self.panorama[map_rows],
                "polar": self.polar[map_rows],
                "corridor": self.corridor[map_rows],
            }
        return {
            "position": position[rows],
            "map_scalar": torch.zeros_like(map_scalar[rows]),
            "route": torch.zeros_like(self.route[rows]),
            "panorama": torch.zeros_like(self.panorama[rows]),
            "polar": torch.zeros_like(self.polar[rows]),
            "corridor": torch.zeros_like(self.corridor[rows]),
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
        update = self.output(torch.nn.functional.silu(left) * right)
        return value + self.dropout(update)


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


class GeometrySpectralStudent(nn.Module):
    def __init__(
        self,
        position_dim: int,
        map_dim: int,
        config: Config,
    ) -> None:
        super().__init__()
        branch = config.hidden // 4
        self.scalar = nn.Sequential(
            nn.Linear(position_dim + map_dim, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, branch),
            nn.SiLU(),
        )
        self.route = conv2d_encoder(5, branch)
        self.polar = conv2d_encoder(4, branch)
        self.corridor = conv2d_encoder(4, branch)
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
            nn.Linear(5 * branch, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden, config.dropout)
            for _ in range(config.residual_blocks)
        )
        self.pas_head = nn.Linear(config.hidden, PAS_SIZE)
        self.pdp_head = nn.Linear(config.hidden, PDP_SIZE)

    def initialize_bias(
        self, pas_mean: torch.Tensor, pdp_mean: torch.Tensor
    ) -> None:
        with torch.no_grad():
            self.pas_head.bias.copy_(
                torch.log(pas_mean.permute(3, 0, 1, 2).reshape(-1).clamp_min(1e-8))
            )
            self.pdp_head.bias.copy_(
                torch.log(pdp_mean.reshape(-1).clamp_min(1e-8))
            )
            nn.init.zeros_(self.pas_head.weight)
            nn.init.zeros_(self.pdp_head.weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.cat(
            (
                self.scalar(torch.cat((batch["position"], batch["map_scalar"]), dim=1)),
                self.route(batch["route"]),
                self.panorama(batch["panorama"]),
                self.polar(batch["polar"]),
                self.corridor(batch["corridor"]),
            ),
            dim=1,
        )
        hidden = self.fusion(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        pas = torch.softmax(
            self.pas_head(hidden).reshape(-1, N_UE, MP * MH * MV), dim=2
        )
        pas = pas.reshape(-1, N_UE, MP, MH, MV).permute(0, 2, 3, 4, 1)
        pdp = torch.softmax(
            self.pdp_head(hidden).reshape(-1, MP, N_UE, S_SC), dim=3
        )
        return pas.float(), pdp.float()


def spectral_score(
    prediction_pas: torch.Tensor,
    prediction_pdp: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pas_norm = torch.sqrt(
        prediction_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    pas_dot = (prediction_pas * target_pas).sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    pas = (pas_dot / pas_norm.clamp_min(TINY)).mean(dim=1)
    pdp_norm = torch.sqrt(
        prediction_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_dot = (prediction_pdp * target_pdp).sum(dim=3, dtype=torch.float64)
    pdp = (pdp_dot / pdp_norm.clamp_min(TINY)).mean(dim=(1, 2))
    return pas, pdp, 0.5 * (pas + pdp)


def spectral_loss(
    prediction_pas: torch.Tensor,
    prediction_pdp: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
) -> torch.Tensor:
    _, _, score = spectral_score(
        prediction_pas, prediction_pdp, target_pas, target_pdp
    )
    pas_hellinger = (
        torch.sqrt(prediction_pas.clamp_min(0.0))
        - torch.sqrt(target_pas.clamp_min(0.0))
    ).square().mean(dim=(1, 2, 3, 4))
    pdp_hellinger = (
        torch.sqrt(prediction_pdp.clamp_min(0.0))
        - torch.sqrt(target_pdp.clamp_min(0.0))
    ).square().mean(dim=(1, 2, 3))
    return (1.0 - score + 0.05 * (pas_hellinger + pdp_hellinger)).mean()


@torch.inference_mode()
def predict(
    model: GeometrySpectralStudent,
    store: FeatureStore,
    rows: np.ndarray,
    position: torch.Tensor,
    map_scalar: torch.Tensor,
    use_map: bool,
    batch_size: int,
    map_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output_pas = np.empty((len(rows), MP, MH, MV, N_UE), dtype=np.float32)
    output_pdp = np.empty((len(rows), MP, N_UE, S_SC), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        local = torch.as_tensor(rows[start:end], device=store.device)
        local_map = (
            None
            if map_rows is None
            else torch.as_tensor(map_rows[start:end], device=store.device)
        )
        batch = store.batch(
            local, position, map_scalar, use_map=use_map, map_rows=local_map
        )
        with torch.autocast(
            device_type=store.device.type,
            dtype=torch.bfloat16,
            enabled=store.device.type == "cuda",
        ):
            pas, pdp = model(batch)
        output_pas[start:end] = pas.cpu().numpy().astype(np.float32)
        output_pdp[start:end] = pdp.cpu().numpy().astype(np.float32)
    return output_pas, output_pdp


@torch.inference_mode()
def evaluate_model(
    model: GeometrySpectralStudent,
    store: FeatureStore,
    rows: np.ndarray,
    position: torch.Tensor,
    map_scalar: torch.Tensor,
    use_map: bool,
    batch_size: int,
) -> dict[str, float]:
    pas, pdp = predict(
        model, store, rows, position, map_scalar, use_map, batch_size
    )
    prediction_pas = torch.as_tensor(pas, device=store.device)
    prediction_pdp = torch.as_tensor(pdp, device=store.device)
    target_rows = torch.as_tensor(rows, device=store.device)
    official = spectral_score(
        prediction_pas,
        prediction_pdp,
        store.official_pas[target_rows],
        store.official_pdp[target_rows],
    )
    teacher = spectral_score(
        prediction_pas,
        prediction_pdp,
        store.teacher_pas[target_rows],
        store.teacher_pdp[target_rows],
    )
    return {
        "official_pas": float(official[0].mean().item()),
        "official_pdp": float(official[1].mean().item()),
        "official_mean": float(official[2].mean().item()),
        "teacher_pas": float(teacher[0].mean().item()),
        "teacher_pdp": float(teacher[1].mean().item()),
        "teacher_mean": float(teacher[2].mean().item()),
    }


def train_phase(
    model: GeometrySpectralStudent,
    store: FeatureStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    position: torch.Tensor,
    map_scalar: torch.Tensor,
    use_map: bool,
    config: Config,
    phase: str,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
        fused=store.device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.06
    )
    rng = np.random.default_rng(seed)
    train_rows = fit_rows[store.power[fit_rows] > 0.0]
    best_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), config.batch_size):
            local_np = order[start : start + config.batch_size]
            local = torch.as_tensor(local_np, device=store.device)
            batch = store.batch(local, position, map_scalar, use_map=use_map)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=store.device.type,
                dtype=torch.bfloat16,
                enabled=store.device.type == "cuda",
            ):
                prediction_pas, prediction_pdp = model(batch)
                teacher_loss = spectral_loss(
                    prediction_pas,
                    prediction_pdp,
                    store.teacher_pas[local],
                    store.teacher_pdp[local],
                )
                if phase == "teacher":
                    loss = teacher_loss
                elif phase == "official":
                    official_loss = spectral_loss(
                        prediction_pas,
                        prediction_pdp,
                        store.official_pas[local],
                        store.official_pdp[local],
                    )
                    loss = official_loss
                elif phase == "calibrate":
                    official_loss = spectral_loss(
                        prediction_pas,
                        prediction_pdp,
                        store.official_pas[local],
                        store.official_pdp[local],
                    )
                    loss = (
                        (1.0 - config.teacher_retain) * official_loss
                        + config.teacher_retain * teacher_loss
                    )
                else:
                    raise ValueError(phase)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()

        if epoch % config.validation_interval == 0 or epoch == epochs:
            validation = evaluate_model(
                model,
                store,
                validation_rows,
                position,
                map_scalar,
                use_map,
                config.batch_size,
            )
            selection = (
                validation["teacher_mean"]
                if phase == "teacher"
                else validation["official_mean"]
            )
            record = {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)),
                **validation,
            }
            history.append(record)
            if selection > best_score:
                best_score = selection
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E87 phase={phase} map={int(use_map)} epoch={epoch}/{epochs} "
                f"loss={record['loss']:.6f} official={record['official_mean']:.6f} "
                f"teacher={record['teacher_mean']:.6f}",
                flush=True,
            )
    return history, best_state, best_epoch


def train_branch(
    name: str,
    store: FeatureStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    config: Config,
    seed: int,
) -> tuple[GeometrySpectralStudent, dict[str, object], dict[str, np.ndarray]]:
    use_map = name != "coordinate_teacher"
    use_teacher = name != "map_official"
    position, map_scalar, normalization = store.normalized_scalars(fit_rows)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GeometrySpectralStudent(
        store.position_raw.shape[1], store.map_scalar_raw.shape[1], config
    ).to(store.device)
    initial_target_pas = (
        store.teacher_pas if use_teacher else store.official_pas
    )
    initial_target_pdp = (
        store.teacher_pdp if use_teacher else store.official_pdp
    )
    fit_tensor = torch.as_tensor(
        fit_rows[store.power[fit_rows] > 0.0], device=store.device
    )
    model.initialize_bias(
        initial_target_pas[fit_tensor].mean(dim=0),
        initial_target_pdp[fit_tensor].mean(dim=0),
    )
    histories: dict[str, object] = {}
    if use_teacher:
        teacher_history, teacher_state, teacher_epoch = train_phase(
            model,
            store,
            fit_rows,
            validation_rows,
            position,
            map_scalar,
            use_map,
            config,
            "teacher",
            config.pretrain_epochs,
            config.learning_rate,
            seed + 101,
        )
        model.load_state_dict(teacher_state)
        histories["teacher"] = teacher_history
        histories["teacher_best_epoch"] = teacher_epoch
        phase = "calibrate"
        epochs = config.finetune_epochs
    else:
        phase = "official"
        epochs = config.pretrain_epochs + config.finetune_epochs

    official_history, best_state, best_epoch = train_phase(
        model,
        store,
        fit_rows,
        validation_rows,
        position,
        map_scalar,
        use_map,
        config,
        phase,
        epochs,
        config.finetune_learning_rate,
        seed + 303,
    )
    model.load_state_dict(best_state)
    histories["official"] = official_history
    histories["official_best_epoch"] = best_epoch
    histories["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    return model, histories, {**normalization, "position": position, "map": map_scalar}


def shuffled_map_rows(
    rows: np.ndarray, site: np.ndarray, seed: int
) -> np.ndarray:
    output = np.array(rows, copy=True)
    rng = np.random.default_rng(seed)
    for site_value in np.unique(site[rows]):
        local = np.flatnonzero(site[rows] == site_value)
        if len(local) > 1:
            output[local] = rows[rng.permutation(local)]
    return output


def endpoint_summary(
    pas: np.ndarray,
    pdp: np.ndarray,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    fold_id: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    values = spectral_score(
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


def expanded_endpoint(
    pas: np.ndarray, pdp: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_tensor = torch.as_tensor(pas, device=device)
    pas_tensor = pas_tensor[..., None].expand(-1, -1, -1, -1, -1, S_SC)
    pdp_tensor = torch.as_tensor(pdp, device=device)
    pdp_tensor = (
        pdp_tensor[:, :, None, :, :]
        .expand(-1, -1, MH * MV, -1, -1)
        .reshape(-1, MP * MH * MV, N_UE, S_SC)
    )
    return pas_tensor, pdp_tensor


@torch.inference_mode()
def fusion_metrics(
    endpoints: dict[str, tuple[np.ndarray, np.ndarray]],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    base = torch.as_tensor(np.array(np.load(V11_CHANNEL, mmap_mode="r"), copy=True), device=device)
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    baseline_values = point_metrics(base, truth, truth_pas, truth_pdp)
    baseline = statistics(*baseline_values)

    v12 = torch.as_tensor(
        np.array(np.load(V12_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    v12_values = point_metrics(v12, truth, truth_pas, truth_pdp)
    v12_summary = statistics(*v12_values)

    output: dict[str, object] = {
        "V11": baseline,
        "V12": {
            **v12_summary,
            "delta_vs_v11": float(v12_summary["score"] - baseline["score"]),
        },
        "branches": {},
    }
    for name, (pas, pdp) in endpoints.items():
        endpoint_pas, endpoint_pdp = expanded_endpoint(pas, pdp, device)
        rows = []
        for beta in (0.05, 0.10, 0.20, 0.30, 0.40):
            desired_pas = endpoint_target(
                base_pas, endpoint_pas, beta, "sqrt", (1, 2, 3)
            )
            desired_pdp = endpoint_target(
                base_pdp, endpoint_pdp, beta, "sqrt", (3,)
            )
            prediction = project_joint(
                base, desired_pas, desired_pdp, 4, "pas_pdp"
            )
            values = point_metrics(prediction, truth, truth_pas, truth_pdp)
            selected = statistics(*values)
            folds = []
            for fold in range(5):
                mask = fold_id == fold
                local = statistics(*values, mask)
                local_base = statistics(*baseline_values, mask)
                folds.append(
                    {
                        "fold": fold,
                        "delta": float(local["score"] - local_base["score"]),
                    }
                )
            rows.append(
                {
                    "beta": beta,
                    "metrics": selected,
                    "delta_vs_v11": float(selected["score"] - baseline["score"]),
                    "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
                    "worst_fold_delta": float(min(row["delta"] for row in folds)),
                    "folds": folds,
                }
            )
            del prediction
        fixed = next(row for row in rows if row["beta"] == 0.20)
        best = max(rows, key=lambda row: float(row["metrics"]["score"]))
        output["branches"][name] = {"fixed_beta_020": fixed, "best_scan": best, "scan": rows}
        del endpoint_pas, endpoint_pdp
    return output


def run_strict(config: Config, device: torch.device, only_outer: int | None) -> dict[str, object]:
    store = FeatureStore(device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    global_to_query = np.full(len(store.positions), -1, dtype=np.int64)
    global_to_query[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    branches = ("map_official", "coordinate_teacher", "map_teacher")
    predictions = {
        name: {
            "pas": np.zeros((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32),
            "pdp": np.zeros((len(query_ids), MP, N_UE, S_SC), dtype=np.float32),
        }
        for name in branches
    }
    predictions["map_teacher_shuffle"] = {
        "pas": np.zeros((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32),
        "pdp": np.zeros((len(query_ids), MP, N_UE, S_SC), dtype=np.float32),
    }
    training: list[dict[str, object]] = []
    all_rows = np.arange(len(store.positions), dtype=np.int64)
    resumed_outers: set[int] = set()
    if only_outer is None and OOF_PREDICTIONS.is_file() and REPORT.is_file():
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        previous_config = previous.get("configuration")
        previous_decision = previous.get("decision", {})
        if previous_config == asdict(config) and not previous_decision.get(
            "full_strict_completed", False
        ):
            cached = np.load(OOF_PREDICTIONS, allow_pickle=False)
            if np.array_equal(cached["query_ids"].astype(np.int64), query_ids):
                completed_query = cached["completed_fold"].astype(bool)
                resumed_outers = set(
                    int(value) for value in np.unique(fold_id[completed_query])
                )
                for name in predictions:
                    predictions[name]["pas"] = cached[f"{name}_pas"].astype(
                        np.float32
                    )
                    predictions[name]["pdp"] = cached[f"{name}_pdp"].astype(
                        np.float32
                    )
                training = list(previous.get("training", []))
                print(
                    f"E87 resume completed outers={sorted(resumed_outers)}",
                    flush=True,
                )
    requested_outers = set(range(5)) if only_outer is None else {only_outer}
    outers = tuple(sorted(requested_outers - resumed_outers))
    for outer in outers:
        validation_fold = (outer + 1) % 5
        held = query_ids[fold_id == outer]
        validation = query_ids[fold_id == validation_fold]
        excluded = np.zeros(len(store.positions), dtype=bool)
        excluded[held] = True
        excluded[validation] = True
        fit = all_rows[~excluded]
        print(
            f"E87 outer={outer} fit={len(fit)} validation={len(validation)} held={len(held)}",
            flush=True,
        )
        for branch_index, name in enumerate(branches):
            seed = config.random_state + 10000 * outer + 1000 * branch_index
            model, history, state = train_branch(
                name, store, fit, validation, config, seed
            )
            use_map = name != "coordinate_teacher"
            pas, pdp = predict(
                model,
                store,
                held,
                state["position"],
                state["map"],
                use_map,
                config.batch_size,
            )
            target_rows = global_to_query[held]
            predictions[name]["pas"][target_rows] = pas
            predictions[name]["pdp"][target_rows] = pdp
            checkpoint = ARTIFACT_DIR / f"outer{outer}_{name}.pt"
            torch.save(
                {
                    "experiment": VERSION,
                    "branch": name,
                    "outer": outer,
                    "validation_fold": validation_fold,
                    "config": asdict(config),
                    "state_dict": model.state_dict(),
                    "position_mean": state["position_mean"],
                    "position_std": state["position_std"],
                    "map_mean": state["map_mean"],
                    "map_std": state["map_std"],
                    "sionna_inputs_required_for_inference": False,
                },
                checkpoint,
            )
            row: dict[str, object] = {
                "outer": outer,
                "validation_fold": validation_fold,
                "branch": name,
                "fit_count": int(len(fit)),
                "held_count": int(len(held)),
                "history": history,
                "checkpoint": PATHS.relative(checkpoint),
            }
            if name == "map_teacher":
                shuffled = shuffled_map_rows(
                    held, store.site, config.random_state + 70000 + outer
                )
                shuffled_pas, shuffled_pdp = predict(
                    model,
                    store,
                    held,
                    state["position"],
                    state["map"],
                    True,
                    config.batch_size,
                    shuffled,
                )
                predictions["map_teacher_shuffle"]["pas"][target_rows] = shuffled_pas
                predictions["map_teacher_shuffle"]["pdp"][target_rows] = shuffled_pdp
                row["shuffle_map_rows"] = shuffled.astype(np.int32).tolist()
            training.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    completed_outers = resumed_outers | set(outers)
    selected_query = np.isin(
        fold_id, np.asarray(sorted(completed_outers), dtype=np.int8)
    )
    selected_ids = query_ids[selected_query]
    selected_folds = fold_id[selected_query]
    truth_pas = store.official_pas[
        torch.as_tensor(selected_ids, device=device)
    ]
    truth_pdp = store.official_pdp[
        torch.as_tensor(selected_ids, device=device)
    ]
    summaries = {}
    endpoint_payload: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id,
        "completed_fold": selected_query.astype(np.uint8),
    }
    endpoints_for_fusion = {}
    for name, value in predictions.items():
        local_pas = value["pas"][selected_query]
        local_pdp = value["pdp"][selected_query]
        summaries[name] = endpoint_summary(
            local_pas,
            local_pdp,
            truth_pas,
            truth_pdp,
            selected_folds,
            device,
        )
        endpoint_payload[f"{name}_pas"] = value["pas"].astype(np.float16)
        endpoint_payload[f"{name}_pdp"] = value["pdp"].astype(np.float16)
        if only_outer is None:
            endpoints_for_fusion[name] = (value["pas"], value["pdp"])
    np.savez_compressed(OOF_PREDICTIONS, **endpoint_payload)

    fusion = (
        fusion_metrics(endpoints_for_fusion, query_ids, fold_id, device)
        if only_outer is None
        else None
    )
    decision: dict[str, object] = {"full_strict_completed": only_outer is None}
    if fusion is not None:
        fixed = fusion["branches"]["map_teacher"]["fixed_beta_020"]
        endpoint_map = summaries["map_teacher"]["mean"]
        endpoint_coord = summaries["coordinate_teacher"]["mean"]
        endpoint_official = summaries["map_official"]["mean"]
        endpoint_shuffle = summaries["map_teacher_shuffle"]["mean"]
        decision.update(
            {
                "minimum_map_margin": 0.001,
                "minimum_teacher_margin": 0.001,
                "minimum_shuffle_margin": 0.001,
                "minimum_fused_gain": 0.002,
                "map_margin_over_coordinate": float(endpoint_map - endpoint_coord),
                "teacher_margin_over_official_only": float(endpoint_map - endpoint_official),
                "correct_map_margin_over_shuffle": float(endpoint_map - endpoint_shuffle),
                "fixed_fused_gain": fixed["delta_vs_v11"],
                "fixed_positive_folds": fixed["positive_folds"],
                "v12_score_to_match": fusion["V12"]["score"],
                "fixed_student_score": fixed["metrics"]["score"],
            }
        )
        decision["passed_information_gate"] = bool(
            decision["map_margin_over_coordinate"] >= decision["minimum_map_margin"]
            and decision["teacher_margin_over_official_only"]
            >= decision["minimum_teacher_margin"]
            and decision["correct_map_margin_over_shuffle"]
            >= decision["minimum_shuffle_margin"]
            and decision["fixed_fused_gain"] >= decision["minimum_fused_gain"]
            and decision["fixed_positive_folds"] >= 4
        )
        decision["passed_deployment_gate"] = bool(
            decision["passed_information_gate"]
            and decision["fixed_student_score"] >= decision["v12_score_to_match"]
            and decision["fixed_positive_folds"] == 5
        )
    return {
        "endpoint": summaries,
        "fusion": fusion,
        "decision": decision,
        "training": training,
        "oof_predictions": PATHS.relative(OOF_PREDICTIONS),
    }


def main() -> None:
    args = parse_args()
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    if not 0 <= args.outer < 5:
        raise ValueError("outer must be in [0, 4]")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    config = Config(
        hidden=args.hidden,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
    )
    started = time.time()
    geometry = build_geometry_cache(args.force_prepare)
    teacher = build_teacher_cache(device, args.batch_size, args.force_prepare)
    prepare = {"geometry": geometry, "teacher": teacher}
    if args.stage == "prepare":
        print(json.dumps(prepare, ensure_ascii=False, indent=2), flush=True)
        return
    result = run_strict(
        config,
        device,
        args.outer if args.stage == "screen" else None,
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A student using only official-map geometry and coordinates can retain useful "
            "Sionna spectral knowledge after official-H calibration and improve V11 without "
            "Sionna inputs at inference."
        ),
        "inference_boundary": {
            "allowed": ["official map", "Round2 Setup", "train/test coordinates", "model weights"],
            "sionna_runtime_input": False,
            "sionna_checkpoint_member": False,
            "sionna_role": "offline PAS/PDP teacher during pretraining only",
        },
        "configuration": asdict(config),
        "prepare": prepare,
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
