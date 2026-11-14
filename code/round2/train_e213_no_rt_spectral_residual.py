#!/usr/bin/env python3
"""Train a no-RT spectral residual model on the historical V16 fold protocol."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
from torch import nn

from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics_torch,
)
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    normalized,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E213-no-rt-spectral-residual"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e213_no_rt_spectral_residual"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e213_no_rt_spectral_residual"
PREPARE_REPORT = ANALYSIS_DIR / "prepare_report.json"
FINAL_REPORT = ANALYSIS_DIR / "fixed500_report.json"

TRAIN_H = PATHS.round2 / "Round2_Train_Channel.npy"
TRAIN_BASE_H = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "full4000_v7_core.npy"
)
CANONICAL_BASE_H = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "canonical_v7_complete.npy"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FULL_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
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
MAP_RASTER_CACHE = (
    PATHS.artifacts
    / "round2"
    / "offline_sionna_geometry_student"
    / "official_map_geometry_rasters.npz"
)
E160_GROUPED_DIR = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "correction_capacity"
)
E160_CORE_PAS = E160_GROUPED_DIR / "full4000_core_grouped_pas_f16.npy"
E160_CORE_PDP = E160_GROUPED_DIR / "full4000_core_grouped_pdp_f16.npy"
OFFICIAL_TARGET_DIR = (
    PATHS.artifacts / "round2" / "surface_interaction_energy_model"
)
SOURCE_OFFICIAL_PAS = OFFICIAL_TARGET_DIR / "official_pas_cosine_sufficient_target_f32.npy"
SOURCE_OFFICIAL_PDP = OFFICIAL_TARGET_DIR / "official_pdp_cosine_sufficient_target_f32.npy"
SOURCE_OFFICIAL_POWER = OFFICIAL_TARGET_DIR / "official_point_power_f64.npy"

TRAIN_BASE_PAS = ARTIFACT_DIR / "train_v7_compact_pas_f32.npy"
TRAIN_BASE_PDP = ARTIFACT_DIR / "train_v7_compact_pdp_f32.npy"
CANONICAL_BASE_PAS = ARTIFACT_DIR / "canonical_v7_compact_pas_f32.npy"
CANONICAL_BASE_PDP = ARTIFACT_DIR / "canonical_v7_compact_pdp_f32.npy"
OFFICIAL_PAS = ARTIFACT_DIR / "official_train_compact_pas_f32.npy"
OFFICIAL_PDP = ARTIFACT_DIR / "official_train_compact_pdp_f32.npy"
OFFICIAL_POWER = ARTIFACT_DIR / "official_train_power_f64.npy"
OUTPUT_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PAS_SIZE = MP * MH * MV * N_UE
PDP_SIZE = MP * N_UE * S_SC
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    hidden: int = 512
    residual_blocks: int = 5
    dropout: float = 0.08
    epochs: int = 80
    validation_interval: int = 4
    batch_size: int = 128
    channel_batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 2e-4
    max_log_ratio: float = 3.0
    residual_penalty: float = 1e-3
    pas_action_scale: float = 0.20
    pdp_action_scale: float = 0.20
    projection_cycles: int = 4
    random_state: int = 20260816213


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "aggregate", "status"), required=True)
    parser.add_argument("--outer", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--channel-batch-size", type=int, default=Config.channel_batch_size)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def required_inputs() -> tuple[Path, ...]:
    return (
        TRAIN_H,
        TRAIN_BASE_H,
        CANONICAL_BASE_H,
        OOF_META,
        FULL_FOLD,
        ROUTE_CACHE,
        GEODESIC_CACHE,
        VISIBILITY_CACHE,
        MAP_RASTER_CACHE,
        E160_CORE_PAS,
        E160_CORE_PDP,
        SOURCE_OFFICIAL_PAS,
        SOURCE_OFFICIAL_PDP,
        SOURCE_OFFICIAL_POWER,
    )


def set_seed(seed: int) -> None:
    bounded_seed = seed % (2**32)
    np.random.seed(bounded_seed)
    torch.manual_seed(bounded_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bounded_seed)


def normalize_compact_pas(value: torch.Tensor) -> torch.Tensor:
    total = value.clamp_min(0.0).sum(dim=(1, 2, 3), keepdim=True)
    return torch.where(total > TINY, value.clamp_min(0.0) / total.clamp_min(TINY), 0.0)


def normalize_compact_pdp(value: torch.Tensor) -> torch.Tensor:
    total = value.clamp_min(0.0).sum(dim=3, keepdim=True)
    return torch.where(total > TINY, value.clamp_min(0.0) / total.clamp_min(TINY), 0.0)


def normalized_direction(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    norm = torch.sqrt(value.square().sum(dim=dims, keepdim=True, dtype=torch.float64))
    return torch.where(norm > 0.0, value / norm.clamp_min(TINY), torch.zeros_like(value))


@torch.inference_mode()
def compact_spectra(channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shaped = channel.reshape(-1, MP, MH, MV, N_UE, S_SC)
    pas_power = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    pas = normalized_direction(pas_power, (1, 2, 3)).mean(dim=5)
    delay_power = torch.fft.ifft(shaped, dim=5).abs().square().float()
    pdp = normalized_direction(delay_power, (5,)).mean(dim=(2, 3))
    power = channel.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return pas.float(), pdp.float(), power


@torch.inference_mode()
def base_compact_spectra(
    channel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(channel)
    pas = pas_distribution(channel).reshape(batch, MP, MH, MV, N_UE, S_SC)
    pas = pas.reshape(batch, MP, MH, MV, N_UE, 8, 24).mean(dim=6)
    pas = normalized(pas, (1, 2, 3)).mean(dim=5)
    pas = normalize_compact_pas(pas)

    pdp = pdp_distribution(channel).reshape(batch, MP, MH, MV, N_UE, S_SC)
    pdp = pdp.reshape(batch, MP, 4, 4, 2, 4, N_UE, S_SC)
    pdp = normalized(pdp.mean(dim=(3, 5)), (5,)).mean(dim=(2, 3))
    pdp = normalize_compact_pdp(pdp)
    power = channel.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return pas.float(), pdp.float(), power


def compact_paths(prefix: str) -> tuple[Path, Path, Path | None]:
    if prefix == "train_base":
        return TRAIN_BASE_PAS, TRAIN_BASE_PDP, None
    if prefix == "canonical_base":
        return CANONICAL_BASE_PAS, CANONICAL_BASE_PDP, None
    if prefix == "official":
        return OFFICIAL_PAS, OFFICIAL_PDP, OFFICIAL_POWER
    raise ValueError(prefix)


def build_compact_cache(
    source_path: Path,
    prefix: str,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, object]:
    pas_path, pdp_path, power_path = compact_paths(prefix)
    expected_count = len(np.load(source_path, mmap_mode="r"))
    ready = pas_path.is_file() and pdp_path.is_file() and (
        power_path is None or power_path.is_file()
    )
    if ready and not force:
        pas = np.load(pas_path, mmap_mode="r")
        pdp = np.load(pdp_path, mmap_mode="r")
        if pas.shape == (expected_count, MP, MH, MV, N_UE) and pdp.shape == (
            expected_count,
            MP,
            N_UE,
            S_SC,
        ):
            return {
                "status": "reused",
                "pas": PATHS.relative(pas_path),
                "pdp": PATHS.relative(pdp_path),
                "count": expected_count,
            }

    source = np.load(source_path, mmap_mode="r")
    partial_pas = pas_path.with_name(f".{pas_path.name}.partial-{os.getpid()}")
    partial_pdp = pdp_path.with_name(f".{pdp_path.name}.partial-{os.getpid()}")
    partial_power = (
        None
        if power_path is None
        else power_path.with_name(f".{power_path.name}.partial-{os.getpid()}")
    )
    pas_output = np.lib.format.open_memmap(
        partial_pas,
        mode="w+",
        dtype=np.float32,
        shape=(expected_count, MP, MH, MV, N_UE),
    )
    pdp_output = np.lib.format.open_memmap(
        partial_pdp,
        mode="w+",
        dtype=np.float32,
        shape=(expected_count, MP, N_UE, S_SC),
    )
    power_output = (
        None
        if partial_power is None
        else np.lib.format.open_memmap(
            partial_power, mode="w+", dtype=np.float64, shape=(expected_count,)
        )
    )
    started = time.time()
    for start in range(0, expected_count, batch_size):
        end = min(start + batch_size, expected_count)
        local = torch.as_tensor(
            np.array(source[start:end], dtype=np.complex64, copy=True), device=device
        )
        if prefix == "canonical_base":
            pas, pdp, power = base_compact_spectra(local)
        else:
            pas, pdp, power = compact_spectra(local)
        pas_output[start:end] = pas.cpu().numpy()
        pdp_output[start:end] = pdp.cpu().numpy()
        if power_output is not None:
            power_output[start:end] = power.cpu().numpy()
        if end == expected_count or end % 500 == 0:
            pas_output.flush()
            pdp_output.flush()
            if power_output is not None:
                power_output.flush()
            print(f"E213 compact {prefix} {end}/{expected_count}", flush=True)
    pas_output.flush()
    pdp_output.flush()
    if power_output is not None:
        power_output.flush()
    del pas_output, pdp_output, power_output
    os.replace(partial_pas, pas_path)
    os.replace(partial_pdp, pdp_path)
    if partial_power is not None and power_path is not None:
        os.replace(partial_power, power_path)
    return {
        "status": "created",
        "pas": PATHS.relative(pas_path),
        "pdp": PATHS.relative(pdp_path),
        "count": expected_count,
        "seconds": float(time.time() - started),
    }


def build_train_base_from_grouped(force: bool) -> dict[str, object]:
    ready = TRAIN_BASE_PAS.is_file() and TRAIN_BASE_PDP.is_file()
    if ready and not force:
        pas = np.load(TRAIN_BASE_PAS, mmap_mode="r")
        pdp = np.load(TRAIN_BASE_PDP, mmap_mode="r")
        if pas.shape == (4000, MP, MH, MV, N_UE) and pdp.shape == (
            4000,
            MP,
            N_UE,
            S_SC,
        ):
            return {
                "status": "reused",
                "pas": PATHS.relative(TRAIN_BASE_PAS),
                "pdp": PATHS.relative(TRAIN_BASE_PDP),
                "count": 4000,
            }
    source_pas = np.load(E160_CORE_PAS, mmap_mode="r")
    source_pdp = np.load(E160_CORE_PDP, mmap_mode="r")
    partial_pas = TRAIN_BASE_PAS.with_name(
        f".{TRAIN_BASE_PAS.name}.partial-{os.getpid()}"
    )
    partial_pdp = TRAIN_BASE_PDP.with_name(
        f".{TRAIN_BASE_PDP.name}.partial-{os.getpid()}"
    )
    pas_output = np.lib.format.open_memmap(
        partial_pas,
        mode="w+",
        dtype=np.float32,
        shape=(4000, MP, MH, MV, N_UE),
    )
    pdp_output = np.lib.format.open_memmap(
        partial_pdp,
        mode="w+",
        dtype=np.float32,
        shape=(4000, MP, N_UE, S_SC),
    )
    started = time.time()
    for start in range(0, 4000, 128):
        end = min(start + 128, 4000)
        local_pas = np.asarray(source_pas[start:end], dtype=np.float32).mean(axis=5)
        local_pdp = np.asarray(source_pdp[start:end], dtype=np.float32).mean(axis=(2, 3))
        pas_total = local_pas.sum(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
        pdp_total = local_pdp.sum(axis=3, keepdims=True, dtype=np.float64)
        pas_output[start:end] = np.divide(
            local_pas,
            np.maximum(pas_total, TINY),
            out=np.zeros_like(local_pas),
            where=pas_total > 0.0,
        )
        pdp_output[start:end] = np.divide(
            local_pdp,
            np.maximum(pdp_total, TINY),
            out=np.zeros_like(local_pdp),
            where=pdp_total > 0.0,
        )
    pas_output.flush()
    pdp_output.flush()
    del pas_output, pdp_output
    os.replace(partial_pas, TRAIN_BASE_PAS)
    os.replace(partial_pdp, TRAIN_BASE_PDP)
    return {
        "status": "created_from_e160_grouped",
        "pas": PATHS.relative(TRAIN_BASE_PAS),
        "pdp": PATHS.relative(TRAIN_BASE_PDP),
        "count": 4000,
        "seconds": float(time.time() - started),
    }


def copy_official_targets(force: bool) -> dict[str, object]:
    sources = (SOURCE_OFFICIAL_PAS, SOURCE_OFFICIAL_PDP, SOURCE_OFFICIAL_POWER)
    targets = (OFFICIAL_PAS, OFFICIAL_PDP, OFFICIAL_POWER)
    expected = (
        (4000, MP, MH, MV, N_UE),
        (4000, MP, N_UE, S_SC),
        (4000,),
    )
    ready = all(path.is_file() for path in targets)
    if ready and not force:
        arrays = [np.load(path, mmap_mode="r") for path in targets]
        if all(array.shape == shape for array, shape in zip(arrays, expected)):
            return {
                "status": "reused",
                "pas": PATHS.relative(OFFICIAL_PAS),
                "pdp": PATHS.relative(OFFICIAL_PDP),
                "power": PATHS.relative(OFFICIAL_POWER),
            }
    started = time.time()
    for source_path, target_path, shape in zip(sources, targets, expected):
        source = np.load(source_path, mmap_mode="r")
        partial = target_path.with_name(f".{target_path.name}.partial-{os.getpid()}")
        output = np.lib.format.open_memmap(
            partial, mode="w+", dtype=source.dtype, shape=shape
        )
        for start in range(0, len(source), 256):
            end = min(start + 256, len(source))
            output[start:end] = source[start:end]
        output.flush()
        del output
        os.replace(partial, target_path)
    return {
        "status": "copied_official_only_targets",
        "pas": PATHS.relative(OFFICIAL_PAS),
        "pdp": PATHS.relative(OFFICIAL_PDP),
        "power": PATHS.relative(OFFICIAL_POWER),
        "seconds": float(time.time() - started),
    }


def compact_identity_error(device: torch.device, batch_size: int) -> dict[str, float]:
    source = np.load(TRAIN_H, mmap_mode="r")
    ids = np.asarray((0, 37, 1024, 2048, 3999), dtype=np.int64)
    local = torch.as_tensor(np.array(source[ids], copy=True), device=device)
    pas, pdp, power = compact_spectra(local)
    cached_pas = torch.as_tensor(np.asarray(np.load(OFFICIAL_PAS, mmap_mode="r")[ids]), device=device)
    cached_pdp = torch.as_tensor(np.asarray(np.load(OFFICIAL_PDP, mmap_mode="r")[ids]), device=device)
    cached_power = torch.as_tensor(np.asarray(np.load(OFFICIAL_POWER, mmap_mode="r")[ids]), device=device)
    del batch_size
    return {
        "pas_max_abs": float((pas - cached_pas).abs().max().item()),
        "pdp_max_abs": float((pdp - cached_pdp).abs().max().item()),
        "power_max_abs": float((power - cached_power).abs().max().item()),
    }


def fold_lineage() -> dict[str, object]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    broad_fold = np.load(FULL_FOLD, allow_pickle=False).astype(np.int8)
    matched = broad_fold[query_ids] == canonical_fold
    return {
        "query_count": int(len(query_ids)),
        "matched_fold_count": int(matched.sum()),
        "label_access_risk_count": int((~matched).sum()),
        "per_outer": [
            {
                "outer": outer,
                "count": int(np.sum(canonical_fold == outer)),
                "matched": int(np.sum((canonical_fold == outer) & matched)),
                "label_access_risk": int(np.sum((canonical_fold == outer) & (~matched))),
            }
            for outer in range(5)
        ],
    }


def prepare(config: Config, device: torch.device, force: bool) -> dict[str, object]:
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    caches = {
        "train_base": build_train_base_from_grouped(force),
        "canonical_base": build_compact_cache(
            CANONICAL_BASE_H,
            "canonical_base",
            device,
            config.channel_batch_size,
            force,
        ),
        "official": copy_official_targets(force),
    }
    identity = compact_identity_error(device, config.channel_batch_size)
    lineage = fold_lineage()
    if lineage["matched_fold_count"] != 124 or lineage["label_access_risk_count"] != 376:
        raise RuntimeError(f"unexpected historical fold lineage: {lineage}")
    report = {
        "experiment": VERSION,
        "status": "complete",
        "config": asdict(config),
        "caches": caches,
        "compact_identity": identity,
        "lineage": lineage,
        "seconds": float(time.time() - started),
        "device": str(device),
        "exit_code": 0,
    }
    atomic_json(PREPARE_REPORT, report)
    return report


def position_features(
    positions: np.ndarray, site: np.ndarray, bs_positions: np.ndarray
) -> np.ndarray:
    delta = positions - bs_positions[site]
    distance = np.linalg.norm(delta, axis=1)
    bearing = np.arctan2(delta[:, 1], delta[:, 0])
    columns: list[np.ndarray] = [
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
        self.map_scalar_raw = np.concatenate((geodesic, visibility), axis=1).astype(np.float32)

        route = np.load(ROUTE_CACHE, allow_pickle=False)
        hit = route["train_route_hit"].astype(np.float32)
        occupancy = route["train_route_occupancy"].astype(np.float32)
        count = route["train_route_count"].astype(np.float32)
        offsets = route["fan_offsets_m"].astype(np.float32)
        fraction = (np.arange(hit.shape[2], dtype=np.float32) + 0.5) / float(hit.shape[2])
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
        geometry = np.load(MAP_RASTER_CACHE, allow_pickle=False)

        self.base_pas = torch.as_tensor(
            np.array(np.load(TRAIN_BASE_PAS, mmap_mode="r"), copy=True), device=device
        )
        self.base_pdp = torch.as_tensor(
            np.array(np.load(TRAIN_BASE_PDP, mmap_mode="r"), copy=True), device=device
        )
        self.target_pas = torch.as_tensor(
            np.array(np.load(OFFICIAL_PAS, mmap_mode="r"), copy=True), device=device
        )
        self.target_pdp = torch.as_tensor(
            np.array(np.load(OFFICIAL_PDP, mmap_mode="r"), copy=True), device=device
        )
        self.power = np.asarray(np.load(OFFICIAL_POWER, mmap_mode="r"), dtype=np.float64)
        self.route = torch.as_tensor(route_tensor, device=device)
        self.panorama = torch.as_tensor(
            np.concatenate((panorama, 1.0 - panorama), axis=1), device=device
        )
        self.polar = torch.as_tensor(geometry["train_polar"].astype(np.float32), device=device)
        self.corridor = torch.as_tensor(
            geometry["train_corridor"].astype(np.float32), device=device
        )
        self.device = device

    @staticmethod
    def normalization(value: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(value[rows], dtype=np.float64)
        mean = selected.mean(axis=0).astype(np.float32)
        std = selected.std(axis=0).astype(np.float32)
        std[std < 1e-5] = 1.0
        return mean, std

    def normalized_scalars(self, rows: np.ndarray) -> dict[str, object]:
        position_mean, position_std = self.normalization(self.position_raw, rows)
        map_mean, map_std = self.normalization(self.map_scalar_raw, rows)
        return {
            "position_mean": position_mean,
            "position_std": position_std,
            "map_mean": map_mean,
            "map_std": map_std,
            "position": torch.as_tensor(
                np.clip((self.position_raw - position_mean) / position_std, -8.0, 8.0),
                device=self.device,
            ),
            "map_scalar": torch.as_tensor(
                np.clip((self.map_scalar_raw - map_mean) / map_std, -8.0, 8.0),
                device=self.device,
            ),
        }

    def batch(
        self,
        rows: torch.Tensor,
        normalization_state: dict[str, object],
        base_pas: torch.Tensor | None = None,
        base_pdp: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "position": normalization_state["position"][rows],
            "map_scalar": normalization_state["map_scalar"][rows],
            "route": self.route[rows],
            "panorama": self.panorama[rows],
            "polar": self.polar[rows],
            "corridor": self.corridor[rows],
            "base_pas": self.base_pas[rows] if base_pas is None else base_pas,
            "base_pdp": self.base_pdp[rows] if base_pdp is None else base_pdp,
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


def conv1d_encoder(channels: int, output: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(channels, 64, 7, padding=3),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv1d(64, 96, 7, stride=2, padding=3),
        nn.GroupNorm(8, 96),
        nn.SiLU(),
        nn.Conv1d(96, 128, 5, stride=2, padding=2),
        nn.GroupNorm(8, 128),
        nn.SiLU(),
        nn.AdaptiveAvgPool1d(8),
        nn.Flatten(),
        nn.Linear(128 * 8, output),
        nn.SiLU(),
    )


class SpectralResidualModel(nn.Module):
    def __init__(self, position_dim: int, map_dim: int, config: Config) -> None:
        super().__init__()
        branch = config.hidden // 4
        self.max_log_ratio = config.max_log_ratio
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
        self.panorama = conv1d_encoder(4, branch)
        self.pas_encoder = conv2d_encoder(MP * N_UE, branch)
        self.pdp_encoder = conv1d_encoder(MP * N_UE, branch)
        self.fusion = nn.Sequential(
            nn.Linear(7 * branch, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden, config.dropout)
            for _ in range(config.residual_blocks)
        )
        self.pas_head = nn.Linear(config.hidden, PAS_SIZE)
        self.pdp_head = nn.Linear(config.hidden, PDP_SIZE)
        nn.init.zeros_(self.pas_head.weight)
        nn.init.zeros_(self.pas_head.bias)
        nn.init.zeros_(self.pdp_head.weight)
        nn.init.zeros_(self.pdp_head.bias)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_base_pas = batch["base_pas"].float()
        input_base_pdp = batch["base_pdp"].float()
        base_pas = normalize_compact_pas(input_base_pas)
        base_pdp = normalize_compact_pdp(input_base_pdp)
        pas_image = base_pas.permute(0, 1, 4, 2, 3).reshape(-1, MP * N_UE, MH, MV)
        pdp_line = base_pdp.reshape(-1, MP * N_UE, S_SC)
        hidden = torch.cat(
            (
                self.scalar(torch.cat((batch["position"], batch["map_scalar"]), dim=1)),
                self.route(batch["route"]),
                self.panorama(batch["panorama"]),
                self.polar(batch["polar"]),
                self.corridor(batch["corridor"]),
                self.pas_encoder(pas_image),
                self.pdp_encoder(pdp_line),
            ),
            dim=1,
        )
        hidden = self.fusion(hidden)
        for block in self.blocks:
            hidden = block(hidden)

        pas_delta = self.max_log_ratio * torch.tanh(
            self.pas_head(hidden).reshape(-1, N_UE, MP * MH * MV)
        )
        pdp_delta = self.max_log_ratio * torch.tanh(
            self.pdp_head(hidden).reshape(-1, MP, N_UE, S_SC)
        )
        if not torch.is_grad_enabled() and not bool(
            torch.any(pas_delta) or torch.any(pdp_delta)
        ):
            return input_base_pas, input_base_pdp, pas_delta.float(), pdp_delta.float()

        pas_logits = torch.log(
            base_pas.permute(0, 4, 1, 2, 3).reshape(-1, N_UE, MP * MH * MV).clamp_min(1e-8)
        ) + pas_delta
        pas = torch.softmax(pas_logits, dim=2).reshape(-1, N_UE, MP, MH, MV)
        pas = pas.permute(0, 2, 3, 4, 1)

        pdp_logits = torch.log(base_pdp.clamp_min(1e-8)) + pdp_delta
        pdp = torch.softmax(pdp_logits, dim=3)
        return pas.float(), pdp.float(), pas_delta.float(), pdp_delta.float()


def spectral_cosines(
    prediction_pas: torch.Tensor,
    prediction_pdp: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_num = (prediction_pas * target_pas).sum(dim=(1, 2, 3), dtype=torch.float64)
    pas_den = torch.sqrt(
        prediction_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * target_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    pas = (pas_num / pas_den.clamp_min(1e-12)).mean(dim=1)
    pdp_num = (prediction_pdp * target_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = torch.sqrt(
        prediction_pdp.square().sum(dim=3, dtype=torch.float64)
        * target_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp = (pdp_num / pdp_den.clamp_min(1e-12)).mean(dim=(1, 2))
    return pas, pdp


def training_loss(
    prediction_pas: torch.Tensor,
    prediction_pdp: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    pas_delta: torch.Tensor,
    pdp_delta: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    pas, pdp = spectral_cosines(prediction_pas, prediction_pdp, target_pas, target_pdp)
    target_pas_probability = normalize_compact_pas(target_pas)
    target_pdp_probability = normalize_compact_pdp(target_pdp)
    pas_hellinger = (
        torch.sqrt(prediction_pas.clamp_min(1e-8))
        - torch.sqrt(target_pas_probability.clamp_min(1e-8))
    ).square().mean(dim=(1, 2, 3, 4))
    pdp_hellinger = (
        torch.sqrt(prediction_pdp.clamp_min(1e-8))
        - torch.sqrt(target_pdp_probability.clamp_min(1e-8))
    ).square().mean(dim=(1, 2, 3))
    residual = 0.5 * (pas_delta.square().mean(dim=(1, 2)) + pdp_delta.square().mean(dim=(1, 2, 3)))
    return (
        1.0
        - 0.5 * (pas + pdp)
        + 0.05 * (pas_hellinger + pdp_hellinger)
        + config.residual_penalty * residual
    ).mean()


@torch.inference_mode()
def predict_endpoints(
    model: SpectralResidualModel,
    store: FeatureStore,
    rows: np.ndarray,
    normalization_state: dict[str, object],
    batch_size: int,
    base_pas: np.ndarray | None = None,
    base_pdp: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output_pas = np.empty((len(rows), MP, MH, MV, N_UE), dtype=np.float32)
    output_pdp = np.empty((len(rows), MP, N_UE, S_SC), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        local_rows = torch.as_tensor(rows[start:end], device=store.device)
        local_pas = (
            None
            if base_pas is None
            else torch.as_tensor(base_pas[start:end], device=store.device)
        )
        local_pdp = (
            None
            if base_pdp is None
            else torch.as_tensor(base_pdp[start:end], device=store.device)
        )
        batch = store.batch(local_rows, normalization_state, local_pas, local_pdp)
        with torch.autocast(
            device_type=store.device.type,
            dtype=torch.bfloat16,
            enabled=store.device.type == "cuda",
        ):
            pas, pdp, _, _ = model(batch)
        output_pas[start:end] = pas.cpu().numpy()
        output_pdp[start:end] = pdp.cpu().numpy()
    return output_pas, output_pdp


def corrected_targets(
    base: torch.Tensor,
    base_compact_pas: torch.Tensor,
    base_compact_pdp: torch.Tensor,
    prediction_pas: torch.Tensor,
    prediction_pdp: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_pas = pas_distribution(base)
    compact_pas = normalize_compact_pas(base_compact_pas)
    pas_ratio = torch.sqrt(
        prediction_pas.clamp_min(1e-8) / compact_pas.clamp_min(1e-8)
    )
    pas_factor = (
        (1.0 - config.pas_action_scale) + config.pas_action_scale * pas_ratio
    ).square()[..., None]
    desired_pas = normalized(full_pas * pas_factor, (1, 2, 3), fallback=full_pas)

    full_pdp = pdp_distribution(base)
    shaped_pdp = full_pdp.reshape(-1, MP, MH, MV, N_UE, S_SC)
    compact_pdp = normalize_compact_pdp(base_compact_pdp)
    pdp_ratio = torch.sqrt(
        prediction_pdp.clamp_min(1e-8) / compact_pdp.clamp_min(1e-8)
    )
    pdp_factor = (
        (1.0 - config.pdp_action_scale) + config.pdp_action_scale * pdp_ratio
    ).square()[:, :, None, None, :, :]
    desired_pdp = normalized(
        shaped_pdp * pdp_factor, (5,), fallback=shaped_pdp
    ).reshape_as(full_pdp)
    return desired_pas, desired_pdp


def project_channel(
    base: torch.Tensor,
    base_compact_pas: torch.Tensor,
    base_compact_pdp: torch.Tensor,
    prediction_pas: torch.Tensor,
    prediction_pdp: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    identity = torch.all(
        prediction_pas == base_compact_pas, dim=(1, 2, 3, 4)
    ) & torch.all(prediction_pdp == base_compact_pdp, dim=(1, 2, 3))
    if bool(torch.all(identity)):
        return base

    desired_pas, desired_pdp = corrected_targets(
        base,
        base_compact_pas,
        base_compact_pdp,
        prediction_pas,
        prediction_pdp,
        config,
    )
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    output = base
    for _ in range(config.projection_cycles):
        output = exact_pas_projection(output, desired_pas)
        output = exact_pdp_projection(output, desired_pdp)
        output = preserve_point_power(output, reference_power)
    return torch.where(identity[:, None, None, None], base, output)


def concatenate_statistics(parts: list[PointMetricStatistics]) -> PointMetricStatistics:
    return PointMetricStatistics(
        pas=np.concatenate([part.pas for part in parts]),
        pdp=np.concatenate([part.pdp for part in parts]),
        squared_error=np.concatenate([part.squared_error for part in parts]),
        truth_energy=np.concatenate([part.truth_energy for part in parts]),
    )


class ValidationBank:
    def __init__(self, rows: np.ndarray, device: torch.device) -> None:
        base_source = np.load(TRAIN_BASE_H, mmap_mode="r")
        truth_source = np.load(TRAIN_H, mmap_mode="r")
        self.rows = rows
        self.base = torch.as_tensor(
            np.array(base_source[rows], dtype=np.complex64, copy=True), device=device
        )
        self.truth = torch.as_tensor(
            np.array(truth_source[rows], dtype=np.complex64, copy=True), device=device
        )
        self.device = device


@torch.inference_mode()
def evaluate_complete(
    model: SpectralResidualModel,
    store: FeatureStore,
    bank: ValidationBank,
    normalization_state: dict[str, object],
    config: Config,
) -> dict[str, float | int]:
    prediction_pas, prediction_pdp = predict_endpoints(
        model,
        store,
        bank.rows,
        normalization_state,
        config.batch_size,
    )
    base_pas = store.base_pas[torch.as_tensor(bank.rows, device=store.device)]
    base_pdp = store.base_pdp[torch.as_tensor(bank.rows, device=store.device)]
    prediction_pas_tensor = torch.as_tensor(prediction_pas, device=store.device)
    prediction_pdp_tensor = torch.as_tensor(prediction_pdp, device=store.device)
    parts: list[PointMetricStatistics] = []
    for start in range(0, len(bank.rows), config.channel_batch_size):
        end = min(start + config.channel_batch_size, len(bank.rows))
        candidate = project_channel(
            bank.base[start:end],
            base_pas[start:end],
            base_pdp[start:end],
            prediction_pas_tensor[start:end],
            prediction_pdp_tensor[start:end],
            config,
        )
        parts.append(point_metric_statistics_torch(candidate, bank.truth[start:end]))
    return aggregate_statistics(concatenate_statistics(parts))


def model_checkpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_model.pt"


def endpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_endpoints.npz"


def fold_report_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}.json"


def failure_report_path(outer: int | None, stage: str) -> Path:
    suffix = stage if outer is None else f"{stage}_outer{outer}"
    return ANALYSIS_DIR / f"failure_{suffix}.json"


def train_phase(
    model: SpectralResidualModel,
    store: FeatureStore,
    train_rows: np.ndarray,
    normalization_state: dict[str, object],
    config: Config,
    epochs: int,
    seed: int,
    validation_bank: ValidationBank | None,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, object]], dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=store.device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.06
    )
    rng = np.random.default_rng(seed)
    active_rows = train_rows[store.power[train_rows] > 0.0]
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history: list[dict[str, object]] = []
    training_seconds = 0.0
    validation_seconds = 0.0
    best_score = -math.inf
    if validation_bank is not None:
        started = time.time()
        baseline = evaluate_complete(model, store, validation_bank, normalization_state, config)
        validation_seconds += time.time() - started
        best_score = float(baseline["score"])
        history.append({"epoch": 0, "loss": None, "validation": baseline})
        print(f"E213 epoch=0 validation={best_score:.9f}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(active_rows)
        losses = []
        phase_started = time.time()
        for start in range(0, len(order), config.batch_size):
            local = torch.as_tensor(order[start : start + config.batch_size], device=store.device)
            batch = store.batch(local, normalization_state)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=store.device.type,
                dtype=torch.bfloat16,
                enabled=store.device.type == "cuda",
            ):
                prediction_pas, prediction_pdp, pas_delta, pdp_delta = model(batch)
                loss = training_loss(
                    prediction_pas,
                    prediction_pdp,
                    store.target_pas[local],
                    store.target_pdp[local],
                    pas_delta,
                    pdp_delta,
                    config,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        training_seconds += time.time() - phase_started
        scheduler.step()

        if validation_bank is not None and (
            epoch % config.validation_interval == 0 or epoch == epochs
        ):
            started = time.time()
            validation = evaluate_complete(
                model, store, validation_bank, normalization_state, config
            )
            validation_seconds += time.time() - started
            score = float(validation["score"])
            record = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation": validation,
            }
            history.append(record)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E213 epoch={epoch}/{epochs} loss={record['loss']:.6f} "
                f"validation={score:.9f} best_epoch={best_epoch}",
                flush=True,
            )
    return best_state, best_epoch, history, {
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
    }


def endpoint_summary(
    prediction_pas: np.ndarray,
    prediction_pdp: np.ndarray,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
) -> dict[str, float]:
    pas, pdp = spectral_cosines(
        torch.as_tensor(prediction_pas, device=target_pas.device),
        torch.as_tensor(prediction_pdp, device=target_pas.device),
        target_pas,
        target_pdp,
    )
    return {
        "pas": float(pas.mean().item()),
        "pdp": float(pdp.mean().item()),
        "mean": float((0.5 * (pas + pdp)).mean().item()),
    }


def train_outer(config: Config, outer: int, device: torch.device, force: bool) -> dict[str, object]:
    if outer not in range(5):
        raise ValueError("outer must be in [0, 4]")
    report_path = fold_report_path(outer)
    if report_path.is_file() and endpoint_path(outer).is_file() and not force:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete" and previous.get("config") == asdict(config):
            print(json.dumps({"status": "reused", "outer": outer}), flush=True)
            return previous
    if not PREPARE_REPORT.is_file():
        raise FileNotFoundError("prepare stage has not completed")

    started = time.time()
    set_seed(config.random_state + 10000 * outer)
    store = FeatureStore(device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    broad_fold = np.load(FULL_FOLD, allow_pickle=False).astype(np.int8)
    validation_fold = (outer + 1) % 5
    screen_fit = np.flatnonzero((broad_fold != outer) & (broad_fold != validation_fold))
    validation_rows = np.flatnonzero(broad_fold == validation_fold)
    refit_rows = np.flatnonzero(broad_fold != outer)
    held_mask = canonical_fold == outer
    held_ids = query_ids[held_mask]

    screen_normalization = store.normalized_scalars(screen_fit)
    validation_bank = ValidationBank(validation_rows, device)
    set_seed(config.random_state + 10000 * outer + 11)
    screen_model = SpectralResidualModel(
        store.position_raw.shape[1], store.map_scalar_raw.shape[1], config
    ).to(device)
    _, selected_epoch, screen_history, screen_timing = train_phase(
        screen_model,
        store,
        screen_fit,
        screen_normalization,
        config,
        config.epochs,
        config.random_state + 10000 * outer + 101,
        validation_bank,
    )
    del validation_bank, screen_model, screen_normalization
    if device.type == "cuda":
        torch.cuda.empty_cache()

    refit_normalization = store.normalized_scalars(refit_rows)
    set_seed(config.random_state + 10000 * outer + 29)
    model = SpectralResidualModel(
        store.position_raw.shape[1], store.map_scalar_raw.shape[1], config
    ).to(device)
    _, _, _, refit_timing = train_phase(
        model,
        store,
        refit_rows,
        refit_normalization,
        config,
        selected_epoch,
        config.random_state + 10000 * outer + 303,
        None,
    )

    canonical_pas_all = np.load(CANONICAL_BASE_PAS, mmap_mode="r")
    canonical_pdp_all = np.load(CANONICAL_BASE_PDP, mmap_mode="r")
    canonical_rows = np.flatnonzero(held_mask)
    base_pas = np.asarray(canonical_pas_all[canonical_rows], dtype=np.float32)
    base_pdp = np.asarray(canonical_pdp_all[canonical_rows], dtype=np.float32)
    prediction_pas, prediction_pdp = predict_endpoints(
        model,
        store,
        held_ids,
        refit_normalization,
        config.batch_size,
        base_pas,
        base_pdp,
    )
    target_rows = torch.as_tensor(held_ids, device=device)
    endpoint = endpoint_summary(
        prediction_pas,
        prediction_pdp,
        store.target_pas[target_rows],
        store.target_pdp[target_rows],
    )
    checkpoint = {
        "experiment": VERSION,
        "outer": outer,
        "validation_fold": validation_fold,
        "selected_epoch": selected_epoch,
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "position_mean": refit_normalization["position_mean"],
        "position_std": refit_normalization["position_std"],
        "map_mean": refit_normalization["map_mean"],
        "map_std": refit_normalization["map_std"],
        "rt_runtime_input": False,
        "rt_training_label": False,
    }
    atomic_torch_save(model_checkpoint_path(outer), checkpoint)
    atomic_npz(
        endpoint_path(outer),
        query_ids=held_ids.astype(np.int32),
        canonical_rows=canonical_rows.astype(np.int32),
        canonical_fold=np.full(len(held_ids), outer, dtype=np.int8),
        broad_fold=broad_fold[held_ids].astype(np.int8),
        pas=prediction_pas.astype(np.float32),
        pdp=prediction_pdp.astype(np.float32),
    )
    label_access_risk = broad_fold[held_ids] != outer
    report = {
        "experiment": VERSION,
        "status": "complete",
        "outer": outer,
        "validation_fold": validation_fold,
        "config": asdict(config),
        "counts": {
            "screen_fit": int(len(screen_fit)),
            "validation": int(len(validation_rows)),
            "refit": int(len(refit_rows)),
            "held": int(len(held_ids)),
            "held_label_access_risk": int(label_access_risk.sum()),
            "held_lineage_match": int((~label_access_risk).sum()),
        },
        "selected_epoch": selected_epoch,
        "screen_history": screen_history,
        "endpoint": endpoint,
        "timing": {
            "screen": screen_timing,
            "refit": refit_timing,
            "wall_seconds": float(time.time() - started),
        },
        "resources": {
            "device": str(device),
            "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "checkpoint": PATHS.relative(model_checkpoint_path(outer)),
        "endpoints": PATHS.relative(endpoint_path(outer)),
        "exit_code": 0,
    }
    atomic_json(report_path, report)
    return report


def aggregate(config: Config, device: torch.device) -> dict[str, object]:
    started = time.time()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    prediction_pas = np.zeros((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32)
    prediction_pdp = np.zeros((len(query_ids), MP, N_UE, S_SC), dtype=np.float32)
    seen = np.zeros(len(query_ids), dtype=np.uint8)
    fold_reports = []
    for outer in range(5):
        report_path = fold_report_path(outer)
        if not report_path.is_file() or not endpoint_path(outer).is_file():
            raise FileNotFoundError(f"missing completed outer {outer}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("config") != asdict(config):
            raise RuntimeError(f"invalid outer report {outer}")
        payload = np.load(endpoint_path(outer), allow_pickle=False)
        rows = payload["canonical_rows"].astype(np.int64)
        if not np.array_equal(query_ids[rows], payload["query_ids"].astype(np.int64)):
            raise RuntimeError(f"query identity mismatch outer {outer}")
        if not np.all(canonical_fold[rows] == outer):
            raise RuntimeError(f"canonical fold mismatch outer {outer}")
        if np.any(seen[rows]):
            raise RuntimeError(f"duplicate predictions outer {outer}")
        prediction_pas[rows] = payload["pas"].astype(np.float32)
        prediction_pdp[rows] = payload["pdp"].astype(np.float32)
        seen[rows] = 1
        fold_reports.append(report)
    if not np.all(seen == 1):
        raise RuntimeError("canonical predictions do not cover all 500 rows")
    label_access_risk = sum(
        int(report["counts"]["held_label_access_risk"]) for report in fold_reports
    )
    if label_access_risk != 376:
        raise RuntimeError(f"unexpected label access risk count: {label_access_risk}")

    base_source = np.load(CANONICAL_BASE_H, mmap_mode="r")
    truth_source = np.load(TRAIN_H, mmap_mode="r")
    base_pas_source = np.load(CANONICAL_BASE_PAS, mmap_mode="r")
    base_pdp_source = np.load(CANONICAL_BASE_PDP, mmap_mode="r")
    partial_output = OUTPUT_CHANNEL.with_name(f".{OUTPUT_CHANNEL.name}.partial-{os.getpid()}")
    output = np.lib.format.open_memmap(
        partial_output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(query_ids), 256, N_UE, S_SC),
    )
    candidate_parts: list[PointMetricStatistics] = []
    baseline_parts: list[PointMetricStatistics] = []
    for start in range(0, len(query_ids), config.channel_batch_size):
        end = min(start + config.channel_batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(base_source[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_source[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        candidate = project_channel(
            base,
            torch.as_tensor(np.asarray(base_pas_source[start:end]), device=device),
            torch.as_tensor(np.asarray(base_pdp_source[start:end]), device=device),
            torch.as_tensor(prediction_pas[start:end], device=device),
            torch.as_tensor(prediction_pdp[start:end], device=device),
            config,
        )
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        candidate_parts.append(point_metric_statistics_torch(candidate, truth))
        baseline_parts.append(point_metric_statistics_torch(base, truth))
        if end == len(query_ids) or end % 100 == 0:
            output.flush()
            print(f"E213 aggregate channel {end}/{len(query_ids)}", flush=True)
    output.flush()
    del output
    os.replace(partial_output, OUTPUT_CHANNEL)

    candidate_statistics = concatenate_statistics(candidate_parts)
    baseline_statistics = concatenate_statistics(baseline_parts)
    candidate = aggregate_statistics(candidate_statistics)
    baseline = aggregate_statistics(baseline_statistics)
    folds = []
    for outer in range(5):
        mask = canonical_fold == outer
        local_candidate = aggregate_statistics(candidate_statistics, mask)
        local_baseline = aggregate_statistics(baseline_statistics, mask)
        folds.append(
            {
                "outer": outer,
                "candidate": local_candidate,
                "baseline": local_baseline,
                "delta": float(local_candidate["score"] - local_baseline["score"]),
            }
        )
    target_pas = torch.as_tensor(
        np.asarray(np.load(OFFICIAL_PAS, mmap_mode="r")[query_ids]), device=device
    )
    target_pdp = torch.as_tensor(
        np.asarray(np.load(OFFICIAL_PDP, mmap_mode="r")[query_ids]), device=device
    )
    endpoint = endpoint_summary(
        prediction_pas, prediction_pdp, target_pas, target_pdp
    )
    wall_seconds = float(time.time() - started)
    report = {
        "experiment": VERSION,
        "status": "complete",
        "protocol": {
            "name": "historical_v16_compatibility",
            "query_count": 500,
            "fold_lineage_match": 124,
            "label_access_risk": label_access_risk,
            "strict_score_used_for_gate": False,
        },
        "config": asdict(config),
        "baseline_v7": baseline,
        "candidate": candidate,
        "delta_vs_v7": {
            key: float(candidate[key] - baseline[key])
            for key in ("pas", "pdp", "nmse_score", "score")
        },
        "v16_reference": {
            "pas": 0.6769415758539015,
            "pdp": 0.7803473980573273,
            "nmse": 0.9331251955178451,
            "nmse_score": 0.5172970702149068,
            "score": 0.6863750036074729,
        },
        "success_threshold": 0.680000,
        "passed": bool(candidate["score"] >= 0.680000),
        "endpoint": endpoint,
        "folds": folds,
        "fold_reports": [PATHS.relative(fold_report_path(outer)) for outer in range(5)],
        "output_channel": PATHS.relative(OUTPUT_CHANNEL),
        "runtime": {
            "aggregate_wall_seconds": wall_seconds,
            "fold_wall_seconds_max": float(
                max(report["timing"]["wall_seconds"] for report in fold_reports)
            ),
            "fold_wall_seconds_sum": float(
                sum(report["timing"]["wall_seconds"] for report in fold_reports)
            ),
            "measured_bottleneck": "reported per fold from training and complete-H validation timings",
        },
        "resources": {
            "aggregate_device": str(device),
            "aggregate_cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "aggregate_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "exit_code": 0,
    }
    atomic_json(FINAL_REPORT, report)
    return report


def status(config: Config) -> dict[str, object]:
    return {
        "prepare": PREPARE_REPORT.is_file(),
        "outers": [
            {
                "outer": outer,
                "report": fold_report_path(outer).is_file(),
                "endpoints": endpoint_path(outer).is_file(),
                "checkpoint": model_checkpoint_path(outer).is_file(),
            }
            for outer in range(5)
        ],
        "aggregate": FINAL_REPORT.is_file(),
        "config": asdict(config),
    }


def main() -> None:
    args = parse_args()
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        channel_batch_size=args.channel_batch_size,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "prepare":
            result = prepare(config, device, args.force)
        elif args.stage == "train":
            if args.outer is None:
                raise ValueError("--outer is required for train")
            result = train_outer(config, args.outer, device, args.force)
        elif args.stage == "aggregate":
            result = aggregate(config, device)
        else:
            result = status(config)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False), flush=True)
    except Exception as error:
        payload = {
            "experiment": VERSION,
            "status": "failed",
            "stage": args.stage,
            "outer": args.outer,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "exit_code": 1,
        }
        atomic_json(failure_report_path(args.outer, args.stage), payload)
        raise


if __name__ == "__main__":
    main()
