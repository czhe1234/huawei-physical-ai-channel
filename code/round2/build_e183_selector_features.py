#!/usr/bin/env python3
"""Build deployment-visible Clean features for E183 hole-specific selectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import time
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree
import torch

import analyze_e153_v7_v12_peak_actions as e153
import analyze_e167_clean_test_hole_difficulty as e167
import analyze_e180_clean_point_action_oracle as e180
from data import infer_site_ids, load_setup
from project_paths import PATHS
from run_v6_regime_component_lifecycle import physical_risk_and_regime
import score_e183_clean_action_labels as e183_labels


VERSION = "Round2-E183-hole-specific-selector-features"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e183_hole_specific_selector" / "features"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e183_hole_specific_selector"
REPORT_PATH = ANALYSIS_DIR / "feature_build_report.json"

WORKSPACE_ROOT = (
    PATHS.artifacts / "round2" / "e183_strict_action_labels" / "workspaces"
)
LIBRARY_PATH = (
    PATHS.artifacts
    / "round2"
    / "e183_test_hole_specialization"
    / "matched_pseudo_hole_library.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
CANONICAL_FEATURES = ARTIFACT_DIR / "canonical_fixed500_clean_features.npz"

SPECIALIZED_FAMILIES = (
    "S0H0",
    "S0H1",
    "S0H2",
    "S0H3",
    "S1H1",
    "S1H3",
)
ACTION_NAMES = e183_labels.ACTION_NAMES
PRIMARY_CARRIER_INDICES = e183_labels.CARRIER_INDICES
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        default="",
        help="Comma-separated family IDs; empty skips family feature generation.",
    )
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()


def json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def normalized_rows(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def hellinger_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    coefficient = np.sum(
        np.sqrt(np.maximum(left * right, 0.0)), axis=1, dtype=np.float64
    )
    return np.maximum(0.0, 1.0 - coefficient)


def finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise RuntimeError(f"non-finite feature array: {name}")
    return array


def device_report(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"device": str(device), "cuda": False}
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(device),
        "cuda": True,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def channel_maps_tensor(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shaped = value.reshape(len(value), 2, 16, 8, 4, 192)
    angle = torch.fft.fft2(shaped, dim=(2, 3)).abs().square()
    angle = angle.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2)
    angle = angle / angle.sum(dim=(2, 3), keepdim=True).clamp_min(TINY)
    delay = torch.fft.ifft(value, dim=-1).abs().square().reshape(
        len(value), 2, 16, 8, 4, 192
    )
    delay = delay.mean(dim=(2, 3))
    delay = delay / delay.sum(dim=3, keepdim=True).clamp_min(TINY)
    power = value.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return angle.float(), delay.float(), torch.log10(power.clamp_min(TINY)).float()


def load_v7_to_device(
    reader: Callable[[int, int], np.ndarray],
    count: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    v7 = torch.empty((count, 256, 4, 192), dtype=torch.complex64, device=device)
    angle = np.empty((count, 4, 16, 8), dtype=np.float32)
    delay = np.empty((count, 2, 4, 192), dtype=np.float32)
    log_power = np.empty(count, dtype=np.float32)
    io_seconds = 0.0
    gpu_seconds = 0.0
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        io_started = time.perf_counter()
        host = np.asarray(reader(start, end), dtype=np.complex64)
        io_seconds += time.perf_counter() - io_started
        if not np.isfinite(host.real).all() or not np.isfinite(host.imag).all():
            raise RuntimeError("V7 contains non-finite values")
        gpu_started = time.perf_counter()
        block = torch.as_tensor(np.array(host, copy=True), device=device)
        v7[start:end] = block
        local_angle, local_delay, local_power = channel_maps_tensor(block)
        angle[start:end] = local_angle.cpu().numpy()
        delay[start:end] = local_delay.cpu().numpy()
        log_power[start:end] = local_power.cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gpu_seconds += time.perf_counter() - gpu_started
    return v7, angle, delay, log_power, {
        "v7_wall_seconds": time.perf_counter() - started,
        "v7_io_seconds": io_seconds,
        "v7_gpu_seconds": gpu_seconds,
    }


def maps_from_reader(
    reader: Callable[[int, int], np.ndarray],
    count: int,
    device: torch.device,
    batch_size: int,
    v7: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    angle = np.empty((count, 4, 16, 8), dtype=np.float32)
    delay = np.empty((count, 2, 4, 192), dtype=np.float32)
    log_power = np.empty(count, dtype=np.float32)
    delta_ratio = np.empty(count, dtype=np.float32)
    complex_cosine = np.empty(count, dtype=np.float32)
    io_seconds = 0.0
    gpu_seconds = 0.0
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        io_started = time.perf_counter()
        host = np.asarray(reader(start, end), dtype=np.complex64)
        io_seconds += time.perf_counter() - io_started
        if not np.isfinite(host.real).all() or not np.isfinite(host.imag).all():
            raise RuntimeError(f"action contains non-finite values at {start}:{end}")
        gpu_started = time.perf_counter()
        block = torch.as_tensor(np.array(host, copy=True), device=device)
        local_angle, local_delay, local_power = channel_maps_tensor(block)
        reference = v7[start:end]
        reference_energy = reference.abs().square().sum(
            dim=(1, 2, 3), dtype=torch.float64
        )
        action_energy = block.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
        difference = (block - reference).abs().square().sum(
            dim=(1, 2, 3), dtype=torch.float64
        )
        inner = (reference.conj() * block).sum(dim=(1, 2, 3), dtype=torch.complex128)
        denominator = torch.sqrt(reference_energy * action_energy)
        cosine = torch.where(
            denominator > 0.0,
            inner.real / denominator,
            torch.zeros_like(denominator),
        )
        angle[start:end] = local_angle.cpu().numpy()
        delay[start:end] = local_delay.cpu().numpy()
        log_power[start:end] = local_power.cpu().numpy()
        delta_ratio[start:end] = (
            difference / reference_energy.clamp_min(TINY)
        ).float().cpu().numpy()
        complex_cosine[start:end] = cosine.float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gpu_seconds += time.perf_counter() - gpu_started
    return angle, delay, log_power, delta_ratio, complex_cosine, {
        "io_seconds": io_seconds,
        "gpu_seconds": gpu_seconds,
    }


def spectral_summary(
    angle: np.ndarray, delay: np.ndarray, log_power: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    values = np.column_stack(
        [
            e153.entropy(angle, (2, 3)).mean(axis=1),
            e153.top_mass(angle, (2, 3), 1).mean(axis=1),
            e153.top_mass(angle, (2, 3), 4).mean(axis=1),
            e153.entropy(delay, (3,)).mean(axis=(1, 2)),
            e153.top_mass(delay, (3,), 1).mean(axis=(1, 2)),
            e153.top_mass(delay, (3,), 4).mean(axis=(1, 2)),
            np.asarray(log_power, dtype=np.float64),
        ]
    ).astype(np.float32)
    names = [
        "angle_entropy",
        "angle_top1",
        "angle_top4",
        "delay_entropy",
        "delay_top1",
        "delay_top4",
        "log_power",
    ]
    return finite_array(values, "spectral_summary"), names


def action_descriptor(
    angle: np.ndarray,
    delay: np.ndarray,
    log_power: np.ndarray,
    delta_ratio: np.ndarray,
    complex_cosine: np.ndarray,
    v7_angle: np.ndarray,
    v7_delay: np.ndarray,
    v7_log_power: np.ndarray,
    v7_summary: np.ndarray,
) -> np.ndarray:
    summary, _ = spectral_summary(angle, delay, log_power)
    angle_distance = e153.hellinger_squared(angle, v7_angle, (2, 3)).mean(axis=1)
    delay_distance = e153.hellinger_squared(delay, v7_delay, (3,)).mean(axis=(1, 2))
    output = np.column_stack(
        [
            angle_distance,
            delay_distance,
            summary,
            summary[:, 0] - v7_summary[:, 0],
            summary[:, 3] - v7_summary[:, 3],
            np.asarray(log_power, dtype=np.float64) - np.asarray(v7_log_power),
            np.asarray(delta_ratio, dtype=np.float64),
            np.asarray(complex_cosine, dtype=np.float64),
        ]
    ).astype(np.float32)
    return finite_array(output, "action_descriptor")


ACTION_DESCRIPTOR_NAMES = (
    "angle_distance_v7",
    "delay_distance_v7",
    "angle_entropy",
    "angle_top1",
    "angle_top4",
    "delay_entropy",
    "delay_top1",
    "delay_top4",
    "log_power",
    "angle_entropy_delta",
    "delay_entropy_delta",
    "log_power_delta",
    "complex_delta_energy_ratio",
    "complex_real_cosine_v7",
)


def compute_action_features(
    *,
    count: int,
    action_readers: list[Callable[[int, int], np.ndarray]],
    carrier_bank: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    if len(action_readers) != len(ACTION_NAMES):
        raise RuntimeError("13 action readers are required")
    started = time.perf_counter()
    v7, v7_angle, v7_delay, v7_log_power, v7_timing = load_v7_to_device(
        action_readers[0], count, device, batch_size
    )
    v7_summary, summary_names = spectral_summary(v7_angle, v7_delay, v7_log_power)

    carrier_angle = np.empty((17, count, 4, 16, 8), dtype=np.float16)
    carrier_delay = np.empty((17, count, 2, 4, 192), dtype=np.float16)
    carrier_power = np.empty((17, count), dtype=np.float32)
    carrier_rel: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    io_seconds = float(v7_timing["v7_io_seconds"])
    gpu_seconds = float(v7_timing["v7_gpu_seconds"])
    for carrier_index in range(17):
        reader = lambda start, end, index=carrier_index: np.asarray(
            carrier_bank[index, start:end], dtype=np.complex64
        )
        angle, delay, power, ratio, cosine, timing = maps_from_reader(
            reader, count, device, batch_size, v7
        )
        carrier_angle[carrier_index] = angle.astype(np.float16)
        carrier_delay[carrier_index] = delay.astype(np.float16)
        carrier_power[carrier_index] = power
        if carrier_index in PRIMARY_CARRIER_INDICES:
            carrier_rel[carrier_index] = (ratio, cosine)
        io_seconds += timing["io_seconds"]
        gpu_seconds += timing["gpu_seconds"]
        print(f"E183 selector carrier maps {carrier_index + 1}/17", flush=True)

    carrier_angle_f32 = np.asarray(carrier_angle, dtype=np.float32)
    carrier_delay_f32 = np.asarray(carrier_delay, dtype=np.float32)
    consensus_angle = e153.normalize(carrier_angle_f32.mean(axis=0), (2, 3))
    consensus_delay = e153.normalize(carrier_delay_f32.mean(axis=0), 3)
    angle_distance = e153.hellinger_squared(
        carrier_angle_f32, consensus_angle[None], (3, 4)
    ).mean(axis=2)
    delay_distance = e153.hellinger_squared(
        carrier_delay_f32, consensus_delay[None], (4,)
    ).mean(axis=(2, 3))
    carrier_context = np.column_stack(
        [
            angle_distance.mean(axis=0),
            angle_distance.max(axis=0),
            delay_distance.mean(axis=0),
            delay_distance.max(axis=0),
            carrier_power.std(axis=0, dtype=np.float64),
            e153.hellinger_squared(
                consensus_angle, v7_angle, (2, 3)
            ).mean(axis=1),
            e153.hellinger_squared(consensus_delay, v7_delay, (3,)).mean(
                axis=(1, 2)
            ),
        ]
    ).astype(np.float32)
    carrier_context_names = [
        "carrier_angle_dispersion_mean",
        "carrier_angle_dispersion_max",
        "carrier_delay_dispersion_mean",
        "carrier_delay_dispersion_max",
        "carrier_log_power_std",
        "v7_carrier_consensus_angle_distance",
        "v7_carrier_consensus_delay_distance",
    ]

    descriptors = np.zeros(
        (count, len(ACTION_NAMES), len(ACTION_DESCRIPTOR_NAMES)), dtype=np.float32
    )
    descriptors[:, 0] = action_descriptor(
        v7_angle,
        v7_delay,
        v7_log_power,
        np.zeros(count, dtype=np.float32),
        np.ones(count, dtype=np.float32),
        v7_angle,
        v7_delay,
        v7_log_power,
        v7_summary,
    )
    for action_index in range(1, 8):
        angle, delay, power, ratio, cosine, timing = maps_from_reader(
            action_readers[action_index], count, device, batch_size, v7
        )
        descriptors[:, action_index] = action_descriptor(
            angle,
            delay,
            power,
            ratio,
            cosine,
            v7_angle,
            v7_delay,
            v7_log_power,
            v7_summary,
        )
        io_seconds += timing["io_seconds"]
        gpu_seconds += timing["gpu_seconds"]
        print(
            f"E183 selector action maps {action_index + 1}/13 {ACTION_NAMES[action_index]}",
            flush=True,
        )
    for action_index, carrier_index in enumerate(PRIMARY_CARRIER_INDICES, start=8):
        ratio, cosine = carrier_rel[carrier_index]
        descriptors[:, action_index] = action_descriptor(
            carrier_angle_f32[carrier_index],
            carrier_delay_f32[carrier_index],
            carrier_power[carrier_index],
            ratio,
            cosine,
            v7_angle,
            v7_delay,
            v7_log_power,
            v7_summary,
        )

    point_spectrum = np.column_stack([v7_summary, carrier_context]).astype(np.float32)
    point_names = [f"v7_{name}" for name in summary_names] + carrier_context_names
    del carrier_angle, carrier_delay, carrier_angle_f32, carrier_delay_f32, v7
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        finite_array(point_spectrum, "point_spectrum"),
        finite_array(descriptors, "action_descriptors"),
        point_names,
        {
            "wall_seconds": time.perf_counter() - started,
            "host_action_io_seconds": io_seconds,
            "gpu_transform_seconds": gpu_seconds,
        },
    )


def build_spatial_context(
    *,
    setup: object,
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    train_site: np.ndarray,
    query_site: np.ndarray,
    train_pas: np.ndarray,
    train_pdp: np.ndarray,
    geometry_query: np.ndarray,
    geometry_names: list[str],
    route_train: np.ndarray,
    route_query: np.ndarray,
    route_names: list[str],
    group_id: np.ndarray,
    fit_by_group: dict[int, np.ndarray],
    v7_angle: np.ndarray,
    v7_delay: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    count = len(query_positions)
    neighbor = {
        name: np.zeros(count, dtype=np.float32)
        for name in (
            "nearest_m",
            "mean8_m",
            "mean16_m",
            "effective_count",
            "angle_dispersion",
            "delay_dispersion",
            "angle_distance_v7",
            "delay_distance_v7",
            "angle_variation_per_m",
            "delay_variation_per_m",
        )
    }
    map_risk = np.zeros(count, dtype=np.float32)
    map_regime = np.zeros(count, dtype=np.int8)
    train_angle = normalized_rows(train_pas).reshape(len(train_pas), 2, 16, 8).sum(
        axis=1
    )
    train_angle = normalized_rows(train_angle.reshape(len(train_angle), -1)).reshape(
        len(train_angle), 16, 8
    )
    train_delay = normalized_rows(train_pdp)
    query_angle = e153.normalize(v7_angle.mean(axis=1), (1, 2))
    query_delay = e153.normalize(v7_delay.mean(axis=(1, 2)), 1)

    for group in sorted(fit_by_group):
        group_rows = np.flatnonzero(group_id == group)
        if not len(group_rows):
            continue
        fit_indices = np.asarray(fit_by_group[group], dtype=np.int64)
        _, risk, _, regime, _ = physical_risk_and_regime(
            route_train,
            route_query[group_rows],
            train_site,
            query_site[group_rows],
            fit_indices,
            tuple(route_names),
        )
        map_risk[group_rows] = risk
        map_regime[group_rows] = regime
        for site in np.unique(query_site[group_rows]):
            rows = group_rows[query_site[group_rows] == site]
            fit = fit_indices[train_site[fit_indices] == site]
            if len(fit) < 2:
                raise RuntimeError(f"insufficient external neighbors for group={group} site={site}")
            take = min(16, len(fit))
            distance, local = cKDTree(train_positions[fit, :2]).query(
                query_positions[rows, :2], k=take, workers=-1
            )
            distance = np.asarray(distance, dtype=np.float64)
            local = np.asarray(local, dtype=np.int64)
            if take == 1:
                distance = distance[:, None]
                local = local[:, None]
            ids = fit[local]
            weight = 1.0 / np.maximum(distance, 1e-3) ** 2
            weight /= np.maximum(weight.sum(axis=1, keepdims=True), TINY)
            consensus_angle = np.sum(
                train_angle[ids] * weight[:, :, None, None], axis=1
            )
            consensus_angle = normalized_rows(
                consensus_angle.reshape(len(rows), -1)
            ).reshape(len(rows), 16, 8)
            consensus_delay = np.sum(train_delay[ids] * weight[:, :, None], axis=1)
            consensus_delay = normalized_rows(consensus_delay)
            angle_local = train_angle[ids].reshape(len(rows), take, -1)
            delay_local = train_delay[ids]
            angle_consensus_flat = consensus_angle.reshape(len(rows), -1)
            angle_dispersion = np.sum(
                weight
                * np.stack(
                    [
                        hellinger_rows(angle_local[:, index], angle_consensus_flat)
                        for index in range(take)
                    ],
                    axis=1,
                ),
                axis=1,
            )
            delay_dispersion = np.sum(
                weight
                * np.stack(
                    [
                        hellinger_rows(delay_local[:, index], consensus_delay)
                        for index in range(take)
                    ],
                    axis=1,
                ),
                axis=1,
            )
            neighbor["nearest_m"][rows] = distance[:, 0]
            neighbor["mean8_m"][rows] = distance[:, : min(8, take)].mean(axis=1)
            neighbor["mean16_m"][rows] = distance.mean(axis=1)
            neighbor["effective_count"][rows] = 1.0 / np.sum(weight * weight, axis=1)
            neighbor["angle_dispersion"][rows] = angle_dispersion
            neighbor["delay_dispersion"][rows] = delay_dispersion
            neighbor["angle_distance_v7"][rows] = hellinger_rows(
                angle_consensus_flat, query_angle[rows].reshape(len(rows), -1)
            )
            neighbor["delay_distance_v7"][rows] = hellinger_rows(
                consensus_delay, query_delay[rows]
            )
            scale = np.maximum(distance.mean(axis=1), 1e-3)
            neighbor["angle_variation_per_m"][rows] = angle_dispersion / scale
            neighbor["delay_variation_per_m"][rows] = delay_dispersion / scale

    delta_bs = query_positions[:, :2] - setup.bs_positions[query_site, :2]
    bs_distance = np.linalg.norm(delta_bs, axis=1)
    azimuth = np.arctan2(delta_bs[:, 1], delta_bs[:, 0])
    position = np.column_stack(
        [
            query_positions[:, 0],
            query_positions[:, 1],
            delta_bs[:, 0],
            delta_bs[:, 1],
            bs_distance,
            np.sin(azimuth),
            np.cos(azimuth),
            query_site,
            map_risk,
            np.eye(4, dtype=np.float32)[map_regime],
        ]
    ).astype(np.float32)
    position_names = [
        "x",
        "y",
        "dx_bs",
        "dy_bs",
        "bs_distance",
        "sin_bs_azimuth",
        "cos_bs_azimuth",
        "site",
        "v7_map_risk",
        "map_regime_clear",
        "map_regime_light",
        "map_regime_medium",
        "map_regime_heavy",
    ]
    neighbor_names = [f"neighbor_{name}" for name in neighbor]
    output = np.column_stack(
        [
            position,
            np.column_stack([neighbor[name] for name in neighbor]),
            np.asarray(geometry_query, dtype=np.float32),
            np.asarray(route_query, dtype=np.float32),
        ]
    ).astype(np.float32)
    names = (
        position_names
        + neighbor_names
        + [f"geometry_{name}" for name in geometry_names]
        + [f"route_{name}" for name in route_names]
    )
    output = np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
    return finite_array(output, "spatial_context"), names, neighbor["nearest_m"]


def principal_coordinates(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(xy, dtype=np.float64) - np.mean(xy, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    _, vectors = np.linalg.eigh(covariance)
    major = vectors[:, -1]
    minor = vectors[:, 0]
    projected = np.column_stack([centered @ major, centered @ minor])
    scale = np.maximum(np.max(np.abs(projected), axis=0), 1e-6)
    normalized = projected / scale[None]
    radius = np.linalg.norm(normalized, axis=1)
    return normalized.astype(np.float32), radius.astype(np.float32)


def library_rows() -> dict[str, object]:
    library = np.load(LIBRARY_PATH, allow_pickle=False)
    return {name: np.asarray(library[name]) for name in library.files}


def family_occurrence_features(
    family: str,
    mapping: dict[str, np.ndarray],
    query_positions: np.ndarray,
    nearest_distance: np.ndarray,
    library: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    target_ids = [str(value) for value in library["target_hole_ids"].tolist()]
    target_index = target_ids.index(family)
    library_rows_for_family = np.flatnonzero(library["target_index"] == target_index)
    library_by_points: dict[tuple[int, ...], int] = {}
    point_flat = library["point_ids_flat"]
    point_offsets = library["point_offsets"]
    for row in library_rows_for_family:
        key = tuple(
            sorted(
                np.asarray(
                    point_flat[point_offsets[row] : point_offsets[row + 1]],
                    dtype=np.int64,
                ).tolist()
            )
        )
        library_by_points[key] = int(row)

    flat_rows = np.asarray(mapping["hole_test_rows_flat"], dtype=np.int64)
    offsets = np.asarray(mapping["hole_offsets"], dtype=np.int64)
    hidden_global = np.asarray(mapping["test_global_ids"], dtype=np.int64)
    global_flat = np.asarray(mapping["hole_global_ids_flat"], dtype=np.int64)
    occurrence = np.empty((len(flat_rows), 46), dtype=np.float32)
    occurrence_hole = np.empty(len(flat_rows), dtype=np.int16)
    matched_library_rows: list[int] = []
    for hole in range(30):
        start, end = int(offsets[hole]), int(offsets[hole + 1])
        point_rows = flat_rows[start:end]
        global_ids = global_flat[start:end]
        key = tuple(sorted(global_ids.tolist()))
        if key not in library_by_points:
            raise RuntimeError(f"{family} analog hole {hole} does not match E183 library")
        library_row = library_by_points[key]
        matched_library_rows.append(library_row)
        profile = np.asarray(library["standardized_features"][library_row], dtype=np.float32)
        local_xy, radius = principal_coordinates(query_positions[point_rows, :2])
        local_depth = nearest_distance[point_rows]
        normalized_depth = local_depth / max(float(np.max(local_depth)), 1e-6)
        count_feature = np.full(len(point_rows), math.log1p(len(point_rows)), dtype=np.float32)
        match_cost = np.full(
            len(point_rows), float(library["match_cost"][library_row]), dtype=np.float32
        )
        occurrence[start:end] = np.column_stack(
            [
                np.repeat(profile[None], len(point_rows), axis=0),
                match_cost,
                local_xy,
                radius,
                normalized_depth,
                count_feature,
            ]
        )
        occurrence_hole[start:end] = hole
        if not np.array_equal(hidden_global[point_rows], global_ids):
            raise RuntimeError(f"{family} mapping point order changed for hole {hole}")
    names = [f"hole_profile_{name}" for name in library["feature_names"].tolist()] + [
        "hole_match_cost",
        "hole_local_major",
        "hole_local_minor",
        "hole_local_radius",
        "hole_normalized_depth",
        "hole_log_point_count",
    ]
    return (
        flat_rows.astype(np.int32),
        occurrence_hole,
        finite_array(occurrence, f"{family}_occurrence"),
        names,
        {
            "target_index": target_index,
            "matched_library_rows": matched_library_rows,
            "library_match_unique_count": len(set(matched_library_rows)),
        },
    )


def family_action_context(
    family: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    workspace = WORKSPACE_ROOT / family
    mapping_npz = np.load(
        workspace / "artifacts" / "round2" / "e183_mapping" / "family_mapping.npz",
        allow_pickle=False,
    )
    count = len(mapping_npz["test_global_ids"])
    provider = e183_labels.ActionProvider(workspace, count)
    lineage = provider.validate_lineage_reports()
    action_files: dict[str, np.ndarray] = {}
    for action_name, path in provider.files.items():
        value = np.load(path, mmap_mode="r")
        if value.shape != (count, 256, 4, 192) or value.dtype != np.complex64:
            raise RuntimeError(f"invalid family action {family}/{action_name}")
        action_files[action_name] = value
    bank = np.load(provider.bank_path, mmap_mode="r")
    if bank.shape != (17, count, 256, 4, 192) or bank.dtype != np.complex64:
        raise RuntimeError(f"invalid family carrier bank {family}: {bank.shape}/{bank.dtype}")
    readers: list[Callable[[int, int], np.ndarray]] = []
    for index, action_name in enumerate(ACTION_NAMES):
        if index < 8:
            readers.append(
                lambda start, end, name=action_name: np.asarray(
                    action_files[name][start:end], dtype=np.complex64
                )
            )
        else:
            carrier = PRIMARY_CARRIER_INDICES[index - 8]
            readers.append(
                lambda start, end, carrier_index=carrier: np.asarray(
                    bank[carrier_index, start:end], dtype=np.complex64
                )
            )
    point_spectrum, descriptors, point_names, timing = compute_action_features(
        count=count,
        action_readers=readers,
        carrier_bank=bank,
        device=device,
        batch_size=batch_size,
    )
    return point_spectrum, descriptors, point_names, {
        "lineage": lineage,
        "timing": timing,
    }


def canonical_action_context(
    device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object], e180.ActionProvider]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = np.asarray(metadata["query_ids"], dtype=np.int64)
    fold_id = np.asarray(metadata["fold_id"], dtype=np.int8)
    train_positions = np.load(PATHS.round2 / "Round2_Train_Pos.npy")
    setup = load_setup()
    train_site, _, _ = infer_site_ids(train_positions, train_positions, setup.bs_positions)
    site = train_site[query_ids]
    actions = [action for action in e180.build_actions(fold_id, site) if action.primary]
    if tuple(action.action_id for action in actions) != ACTION_NAMES:
        raise RuntimeError("canonical 13-action identity changed")
    provider = e180.ActionProvider(actions)
    readers = [
        lambda start, end, name=action.action_id: provider.block(name, start, end)
        for action in actions
    ]
    point_spectrum, descriptors, point_names, timing = compute_action_features(
        count=len(query_ids),
        action_readers=readers,
        carrier_bank=provider.carrier_bank,
        device=device,
        batch_size=batch_size,
    )
    return point_spectrum, descriptors, point_names, {"timing": timing}, provider


def build_family_features(
    family: str,
    device: torch.device,
    batch_size: int,
    force: bool,
    library: dict[str, object],
) -> dict[str, object]:
    output_path = ARTIFACT_DIR / f"{family}_clean_features.npz"
    if output_path.is_file() and not force:
        with np.load(output_path, allow_pickle=False) as cached:
            return {
                "family": family,
                "status": "cached",
                "point_count": int(len(cached["point_context"])),
                "occurrence_count": int(len(cached["occurrence_point_rows"])),
                "path": PATHS.relative(output_path),
            }
    started = time.perf_counter()
    workspace = WORKSPACE_ROOT / family
    mapping_npz = np.load(
        workspace / "artifacts" / "round2" / "e183_mapping" / "family_mapping.npz",
        allow_pickle=False,
    )
    mapping = {name: np.asarray(mapping_npz[name]) for name in mapping_npz.files}
    external_global = mapping["external_global_ids"].astype(np.int64)
    hidden_global = mapping["test_global_ids"].astype(np.int64)
    original_positions = np.load(PATHS.round2 / "Round2_Train_Pos.npy")
    setup = load_setup()
    original_site, _, _ = infer_site_ids(
        original_positions, original_positions, setup.bs_positions
    )
    train_site = original_site[external_global]
    query_site = original_site[hidden_global]
    local_root = workspace / "artifacts" / "round2"
    local_data = workspace / "datasets" / "round2"
    train_positions = np.load(local_data / "Round2_Train_Pos.npy")
    query_positions = np.load(local_data / "Round2_Test_Pos.npy")
    if not np.allclose(train_positions, original_positions[external_global]):
        raise RuntimeError(f"{family} external position mapping changed")
    if not np.allclose(query_positions, original_positions[hidden_global]):
        raise RuntimeError(f"{family} hidden position mapping changed")

    point_spectrum, action_descriptors, spectrum_names, action_report = (
        family_action_context(family, device, batch_size)
    )
    descriptors = np.load(
        local_root / "pre_sionna" / "round2_train_spectral_descriptors.npz",
        allow_pickle=False,
    )
    geometry = np.load(
        local_root / "pre_sionna" / "round2_geometry_visibility_features.npz",
        allow_pickle=False,
    )
    route = np.load(
        local_root / "regime_lifecycle_support" / "geodesic_corridor_features.npz",
        allow_pickle=False,
    )
    v7_angle = np.zeros((len(query_positions), 4, 16, 8), dtype=np.float32)
    v7_delay = np.zeros((len(query_positions), 2, 4, 192), dtype=np.float32)
    # The action descriptor stores only summaries; read V7 once more for neighbor-shape context.
    provider = e183_labels.ActionProvider(workspace, len(query_positions))
    provider.validate_lineage_reports()
    v7_file = np.load(provider.files["V7"], mmap_mode="r")
    for start in range(0, len(query_positions), batch_size):
        end = min(start + batch_size, len(query_positions))
        block = torch.as_tensor(
            np.array(v7_file[start:end], dtype=np.complex64, copy=True), device=device
        )
        angle, delay, _ = channel_maps_tensor(block)
        v7_angle[start:end] = angle.cpu().numpy()
        v7_delay[start:end] = delay.cpu().numpy()
    spatial, spatial_names, nearest = build_spatial_context(
        setup=setup,
        train_positions=train_positions,
        query_positions=query_positions,
        train_site=train_site,
        query_site=query_site,
        train_pas=np.asarray(descriptors["global_pas"], dtype=np.float32),
        train_pdp=np.asarray(descriptors["global_pdp"], dtype=np.float32),
        geometry_query=np.asarray(geometry["test"], dtype=np.float32),
        geometry_names=[str(value) for value in geometry["names"].tolist()],
        route_train=np.asarray(route["train_features"], dtype=np.float32),
        route_query=np.asarray(route["test_features"], dtype=np.float32),
        route_names=[str(value) for value in route["feature_names"].tolist()],
        group_id=np.zeros(len(query_positions), dtype=np.int8),
        fit_by_group={0: np.arange(len(train_positions), dtype=np.int64)},
        v7_angle=v7_angle,
        v7_delay=v7_delay,
    )
    point_context = np.column_stack([spatial, point_spectrum]).astype(np.float32)
    point_names = spatial_names + spectrum_names
    occurrence_rows, occurrence_hole, occurrence, occurrence_names, occurrence_audit = (
        family_occurrence_features(
            family, mapping, query_positions, nearest, library
        )
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        family=np.asarray(family),
        action_names=np.asarray(ACTION_NAMES, dtype="U64"),
        hidden_global_ids=hidden_global.astype(np.int32),
        point_context=point_context,
        point_context_names=np.asarray(point_names, dtype="U96"),
        action_descriptors=action_descriptors,
        action_descriptor_names=np.asarray(ACTION_DESCRIPTOR_NAMES, dtype="U64"),
        occurrence_point_rows=occurrence_rows,
        occurrence_hole_index=occurrence_hole,
        occurrence_features=occurrence,
        occurrence_feature_names=np.asarray(occurrence_names, dtype="U96"),
        hole_offsets=np.asarray(mapping["hole_offsets"], dtype=np.int64),
        hole_global_ids_flat=np.asarray(mapping["hole_global_ids_flat"], dtype=np.int32),
    )
    return {
        "family": family,
        "status": "built",
        "point_count": len(point_context),
        "occurrence_count": len(occurrence_rows),
        "point_feature_count": point_context.shape[1],
        "action_feature_count": action_descriptors.shape[2],
        "occurrence_feature_count": occurrence.shape[1],
        "occurrence_audit": occurrence_audit,
        "action_context": action_report,
        "wall_seconds": time.perf_counter() - started,
        "path": PATHS.relative(output_path),
    }


def canonical_occurrence_features(
    query_ids: np.ndarray,
    query_positions: np.ndarray,
    nearest_distance: np.ndarray,
    library: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    _, pseudo_holes, _ = e167.build_holes()
    global_to_oof = np.full(4000, -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    family_by_row = np.full(len(query_ids), "", dtype="U8")
    hole_index_by_row = np.full(len(query_ids), -1, dtype=np.int16)
    occurrence = np.empty((len(query_ids), 46), dtype=np.float32)
    target_ids = [str(value) for value in library["target_hole_ids"].tolist()]
    for hole in pseudo_holes:
        family = f"S{hole.site}H{hole.ordinal}"
        target_index = target_ids.index(family)
        rows = global_to_oof[hole.point_ids]
        if np.any(rows < 0):
            raise RuntimeError(f"canonical hole {hole.hole_id} escaped fixed500")
        profile = np.asarray(library["canonical_features"][target_index], dtype=np.float32)
        local_xy, radius = principal_coordinates(query_positions[rows, :2])
        depth = nearest_distance[rows]
        normalized_depth = depth / max(float(np.max(depth)), 1e-6)
        occurrence[rows] = np.column_stack(
            [
                np.repeat(profile[None], len(rows), axis=0),
                np.full(len(rows), float(library["canonical_cost"][target_index])),
                local_xy,
                radius,
                normalized_depth,
                np.full(len(rows), math.log1p(len(rows))),
            ]
        )
        family_by_row[rows] = family
        hole_index_by_row[rows] = target_index
    if np.any(hole_index_by_row < 0) or np.any(family_by_row == ""):
        raise RuntimeError("canonical occurrence mapping does not cover fixed500")
    names = [f"hole_profile_{name}" for name in library["feature_names"].tolist()] + [
        "hole_match_cost",
        "hole_local_major",
        "hole_local_minor",
        "hole_local_radius",
        "hole_normalized_depth",
        "hole_log_point_count",
    ]
    return (
        np.arange(len(query_ids), dtype=np.int32),
        hole_index_by_row,
        finite_array(occurrence, "canonical_occurrence"),
        names,
        family_by_row,
    )


def build_canonical_features(
    device: torch.device,
    batch_size: int,
    force: bool,
    library: dict[str, object],
) -> dict[str, object]:
    if CANONICAL_FEATURES.is_file() and not force:
        with np.load(CANONICAL_FEATURES, allow_pickle=False) as cached:
            return {
                "status": "cached",
                "point_count": int(len(cached["point_context"])),
                "path": PATHS.relative(CANONICAL_FEATURES),
            }
    started = time.perf_counter()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = np.asarray(metadata["query_ids"], dtype=np.int64)
    fold_id = np.asarray(metadata["fold_id"], dtype=np.int8)
    setup = load_setup()
    train_positions = np.load(PATHS.round2 / "Round2_Train_Pos.npy")
    query_positions = train_positions[query_ids]
    train_site, _, _ = infer_site_ids(train_positions, train_positions, setup.bs_positions)
    query_site = train_site[query_ids]
    point_spectrum, action_descriptors, spectrum_names, action_report, provider = (
        canonical_action_context(device, batch_size)
    )
    v7_angle = np.zeros((len(query_ids), 4, 16, 8), dtype=np.float32)
    v7_delay = np.zeros((len(query_ids), 2, 4, 192), dtype=np.float32)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        block = torch.as_tensor(provider.block("V7", start, end), device=device)
        angle, delay, _ = channel_maps_tensor(block)
        v7_angle[start:end] = angle.cpu().numpy()
        v7_delay[start:end] = delay.cpu().numpy()

    descriptors = np.load(
        PATHS.artifacts
        / "round2"
        / "pre_sionna"
        / "round2_train_spectral_descriptors.npz",
        allow_pickle=False,
    )
    geometry = np.load(
        PATHS.artifacts
        / "round2"
        / "pre_sionna"
        / "round2_geometry_visibility_features.npz",
        allow_pickle=False,
    )
    route = np.load(
        PATHS.artifacts
        / "round2"
        / "regime_lifecycle_support"
        / "geodesic_corridor_features.npz",
        allow_pickle=False,
    )
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_by_group = {
        fold: np.asarray(folds[f"fold{fold}_fit"], dtype=np.int64)
        for fold in range(5)
    }
    spatial, spatial_names, nearest = build_spatial_context(
        setup=setup,
        train_positions=train_positions,
        query_positions=query_positions,
        train_site=train_site,
        query_site=query_site,
        train_pas=np.asarray(descriptors["global_pas"], dtype=np.float32),
        train_pdp=np.asarray(descriptors["global_pdp"], dtype=np.float32),
        geometry_query=np.asarray(geometry["train"], dtype=np.float32)[query_ids],
        geometry_names=[str(value) for value in geometry["names"].tolist()],
        route_train=np.asarray(route["train_features"], dtype=np.float32),
        route_query=np.asarray(route["train_features"], dtype=np.float32)[query_ids],
        route_names=[str(value) for value in route["feature_names"].tolist()],
        group_id=fold_id,
        fit_by_group=fit_by_group,
        v7_angle=v7_angle,
        v7_delay=v7_delay,
    )
    point_context = np.column_stack([spatial, point_spectrum]).astype(np.float32)
    point_names = spatial_names + spectrum_names
    occurrence_rows, occurrence_hole, occurrence, occurrence_names, family_by_row = (
        canonical_occurrence_features(
            query_ids, query_positions, nearest, library
        )
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CANONICAL_FEATURES,
        action_names=np.asarray(ACTION_NAMES, dtype="U64"),
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=query_site,
        family_by_row=family_by_row,
        point_context=point_context,
        point_context_names=np.asarray(point_names, dtype="U96"),
        action_descriptors=action_descriptors,
        action_descriptor_names=np.asarray(ACTION_DESCRIPTOR_NAMES, dtype="U64"),
        occurrence_point_rows=occurrence_rows,
        occurrence_hole_index=occurrence_hole,
        occurrence_features=occurrence,
        occurrence_feature_names=np.asarray(occurrence_names, dtype="U96"),
    )
    return {
        "status": "built",
        "point_count": len(query_ids),
        "point_feature_count": point_context.shape[1],
        "action_feature_count": action_descriptors.shape[2],
        "occurrence_feature_count": occurrence.shape[1],
        "action_context": action_report,
        "wall_seconds": time.perf_counter() - started,
        "path": PATHS.relative(CANONICAL_FEATURES),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    families = tuple(value for value in args.families.split(",") if value)
    unknown = set(families) - set(SPECIALIZED_FAMILIES)
    if unknown:
        raise ValueError(f"unsupported specialized families: {sorted(unknown)}")
    if not families and not args.canonical:
        raise ValueError("select --families and/or --canonical")
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    library = library_rows()
    family_reports = []
    for family in families:
        family_reports.append(
            build_family_features(
                family, device, args.batch_size, args.force, library
            )
        )
    canonical_report = (
        build_canonical_features(device, args.batch_size, args.force, library)
        if args.canonical
        else None
    )
    report = {
        "experiment": VERSION,
        "status": "feature_build_complete",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "hidden_truth_read": False,
        "families": family_reports,
        "canonical": canonical_report,
        "feature_lineage": {
            "point_context": "official positions/Map/Setup, external Train H spectra, strict V7 and 17 Clean carrier predictions",
            "action_context": "13 strict Clean candidate H tensors compared with strict V7; no GT metric",
            "hole_context": "E183 Clean matching profiles and geometry of the complete hole",
        },
        "performance": {
            "wall_seconds": time.perf_counter() - started,
            "device": device_report(device),
            "peak_torch_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024.0**2),
            "expected_bottleneck": "sequential action/carrier H reads",
        },
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    previous = {}
    if REPORT_PATH.is_file():
        previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    merged_families = {
        str(item["family"]): item
        for item in previous.get("families", [])
        if isinstance(item, dict) and "family" in item
    }
    merged_families.update({str(item["family"]): item for item in family_reports})
    report["families"] = [merged_families[key] for key in sorted(merged_families)]
    if canonical_report is None and previous.get("canonical") is not None:
        report["canonical"] = previous["canonical"]
    REPORT_PATH.write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_value(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
