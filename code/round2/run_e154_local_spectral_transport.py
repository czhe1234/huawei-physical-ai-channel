#!/usr/bin/env python3
"""Test whether official-neighbor peak trajectories can safely correct V7 spectra."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import analyze_e153_v7_v12_peak_actions as e153
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
from run_rbf_spectral_field import OOF_META, statistics
from run_v6_regime_component_lifecycle import REGIME_CACHE, REGIME_NAMES
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E154-local-spectral-energy-transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "local_spectral_transport"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_local_spectral_transport"
REPORT = ANALYSIS_DIR / "local_spectral_transport.json"
METRICS = ARTIFACT_DIR / "local_spectral_transport_metrics.npz"
COMPACT_CACHE = ARTIFACT_DIR / "official_train_compact_spectra.npz"
PEAK_CACHE = ARTIFACT_DIR / "official_train_peak_tracks.npz"

V7_CHANNEL = e153.V7_CHANNEL
TRAIN_CHANNEL = e153.TRAIN_CHANNEL
TRAIN_PAS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_pas_link_support_f16.npy"
)
TRAIN_PDP = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_pdp_link_support_f16.npy"
)
NEIGHBOR_CACHE = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)

TINY = 1e-30
VARIANT_NAMES = ("identity", "pas_transport", "pdp_transport", "joint_pas_pdp")


@dataclass(frozen=True)
class Config:
    neighbor_count: int = 24
    maximum_peaks: int = 8
    minimum_observations: int = 4
    affine_ridge: float = 0.20
    minimum_confidence: float = 0.18
    shift_threshold: float = 0.35
    angle_fraction: float = 0.65
    delay_fraction: float = 0.55
    angle_residual_scale: float = 0.75
    delay_residual_scale: float = 1.25
    minimum_mass_ratio: float = 0.50
    maximum_mass_ratio: float = 1.50
    cycles: int = 4
    order: str = "pas_pdp"


@dataclass(frozen=True)
class PeakAction:
    coordinate: tuple[int, ...]
    shift: tuple[int, ...]
    confidence: float
    mass_ratio: float
    peak_mass: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def normalize(value: np.ndarray, axes: int | tuple[int, ...]) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = output.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        output,
        np.maximum(total, TINY),
        out=np.zeros_like(output),
        where=total > 0.0,
    )


def load_compact_train_spectra() -> tuple[np.ndarray, np.ndarray]:
    if COMPACT_CACHE.is_file():
        cached = np.load(COMPACT_CACHE, allow_pickle=False)
        return cached["angle"].astype(np.float32), cached["delay"].astype(np.float32)

    pas_source = np.load(TRAIN_PAS, mmap_mode="r")
    pdp_source = np.load(TRAIN_PDP, mmap_mode="r")
    count = len(pas_source)
    angle = np.empty((count, e153.e138.N_UE, e153.e138.MH, e153.e138.MV), dtype=np.float32)
    delay = np.empty(
        (count, e153.e138.MP, e153.e138.N_UE, e153.e138.S_SC), dtype=np.float32
    )
    for start in range(0, count, 32):
        stop = min(start + 32, count)
        pas = np.asarray(pas_source[start:stop], dtype=np.float32).reshape(
            stop - start,
            e153.e138.MP,
            e153.e138.MH,
            e153.e138.MV,
            e153.e138.N_UE,
            e153.e138.S_SC,
        )
        pdp = np.asarray(pdp_source[start:stop], dtype=np.float32).reshape(
            stop - start,
            e153.e138.MP,
            e153.e138.MH,
            e153.e138.MV,
            e153.e138.N_UE,
            e153.e138.S_SC,
        )
        angle[start:stop] = np.transpose(
            pas.sum(axis=1, dtype=np.float64).mean(axis=-1), (0, 3, 1, 2)
        ).astype(np.float32)
        delay[start:stop] = pdp.mean(axis=(2, 3), dtype=np.float64).astype(np.float32)
    angle = normalize(angle, (2, 3))
    delay = normalize(delay, 3)
    np.savez_compressed(
        COMPACT_CACHE, angle=angle.astype(np.float16), delay=delay.astype(np.float16)
    )
    return angle, delay


def load_peak_cache(
    train_angle: np.ndarray, train_delay: np.ndarray, maximum_peaks: int
) -> dict[str, np.ndarray]:
    if PEAK_CACHE.is_file():
        return {
            key: np.asarray(value)
            for key, value in np.load(PEAK_CACHE, allow_pickle=False).items()
        }

    count = len(train_angle)
    angle_coordinate = np.full(
        (count, e153.e138.N_UE, maximum_peaks, 2), -1, dtype=np.int16
    )
    angle_mass = np.zeros(
        (count, e153.e138.N_UE, maximum_peaks), dtype=np.float32
    )
    angle_count = np.zeros((count, e153.e138.N_UE), dtype=np.uint8)
    delay_coordinate = np.full(
        (count, e153.e138.MP, e153.e138.N_UE, maximum_peaks),
        -1,
        dtype=np.int16,
    )
    delay_mass = np.zeros(
        (count, e153.e138.MP, e153.e138.N_UE, maximum_peaks),
        dtype=np.float32,
    )
    delay_count = np.zeros(
        (count, e153.e138.MP, e153.e138.N_UE), dtype=np.uint8
    )

    for point in range(count):
        for ue in range(e153.e138.N_UE):
            peaks = e153.extract_angle_peaks(train_angle[point, ue])[:maximum_peaks]
            angle_count[point, ue] = len(peaks)
            for index, peak in enumerate(peaks):
                angle_coordinate[point, ue, index] = peak.coordinate
                angle_mass[point, ue, index] = peak.mass
        for polarization in range(e153.e138.MP):
            for ue in range(e153.e138.N_UE):
                peaks = e153.extract_delay_peaks(
                    train_delay[point, polarization, ue]
                )[:maximum_peaks]
                delay_count[point, polarization, ue] = len(peaks)
                for index, peak in enumerate(peaks):
                    delay_coordinate[point, polarization, ue, index] = peak.coordinate[0]
                    delay_mass[point, polarization, ue, index] = peak.mass
        if (point + 1) % 500 == 0:
            print(f"E154官方峰缓存 {point + 1}/{count}", flush=True)

    np.savez_compressed(
        PEAK_CACHE,
        angle_coordinate=angle_coordinate,
        angle_mass=angle_mass.astype(np.float16),
        angle_count=angle_count,
        delay_coordinate=delay_coordinate,
        delay_mass=delay_mass.astype(np.float16),
        delay_count=delay_count,
    )
    return {
        "angle_coordinate": angle_coordinate,
        "angle_mass": angle_mass,
        "angle_count": angle_count,
        "delay_coordinate": delay_coordinate,
        "delay_mass": delay_mass,
        "delay_count": delay_count,
    }


def signed_periodic_delta(destination: float, source: float, period: int) -> float:
    return float((destination - source + period / 2.0) % period - period / 2.0)


def rounded_unit_shift(value: float, threshold: float) -> int:
    if abs(value) < threshold:
        return 0
    return 1 if value > 0.0 else -1


def robust_affine_intercept(
    offsets: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> tuple[float, float]:
    weight = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    weight /= max(float(weight.sum()), TINY)
    scale = math.sqrt(
        max(float(np.sum(weight * np.sum(offsets * offsets, axis=1))), 1.0)
    )
    design = np.column_stack(
        (np.ones(len(offsets)), offsets[:, 0] / scale, offsets[:, 1] / scale)
    )
    penalty = np.diag((1e-8, ridge, ridge))
    coefficient = np.linalg.solve(
        design.T @ (weight[:, None] * design) + penalty,
        design.T @ (weight * values),
    )
    residual = design @ coefficient - values
    for _ in range(2):
        residual_scale = max(
            1.4826 * float(np.median(np.abs(residual - np.median(residual)))),
            0.10,
        )
        robust = 1.0 / np.maximum(1.0, np.abs(residual) / (2.5 * residual_scale))
        local_weight = weight * robust
        local_weight /= max(float(local_weight.sum()), TINY)
        coefficient = np.linalg.solve(
            design.T @ (local_weight[:, None] * design) + penalty,
            design.T @ (local_weight * values),
        )
        residual = design @ coefficient - values
    rms = math.sqrt(float(np.sum(weight * residual * residual)))
    return float(coefficient[0]), rms


def directional_balance(offsets: np.ndarray, weights: np.ndarray) -> float:
    radius = np.linalg.norm(offsets, axis=1)
    valid = radius > 1e-6
    if np.sum(valid) < 2:
        return 0.0
    weight = np.maximum(weights[valid], 0.0)
    weight /= max(float(weight.sum()), TINY)
    unit = offsets[valid] / radius[valid, None]
    resultant = np.linalg.norm(np.sum(weight[:, None] * unit, axis=0))
    return float(np.clip(1.0 - resultant, 0.0, 1.0))


def angle_neighbor_matches(
    base_peaks: list[e153.Peak],
    neighbor_id: int,
    ue: int,
    peak_cache: dict[str, np.ndarray],
) -> list[int]:
    count = int(peak_cache["angle_count"][neighbor_id, ue])
    coordinate = peak_cache["angle_coordinate"][neighbor_id, ue, :count]
    mass = peak_cache["angle_mass"][neighbor_id, ue, :count]
    available = set(range(count))
    output = [-1] * len(base_peaks)
    for base_index, peak in enumerate(base_peaks):
        best: tuple[float, int] | None = None
        for candidate in available:
            dh = abs(
                signed_periodic_delta(
                    float(coordinate[candidate, 0]),
                    float(peak.coordinate[0]),
                    e153.e138.MH,
                )
            )
            dv = abs(
                signed_periodic_delta(
                    float(coordinate[candidate, 1]),
                    float(peak.coordinate[1]),
                    e153.e138.MV,
                )
            )
            if dh > e153.ANGLE_MATCH_H or dv > e153.ANGLE_MATCH_V:
                continue
            distance = math.sqrt(
                (dh / e153.ANGLE_MATCH_H) ** 2
                + (dv / e153.ANGLE_MATCH_V) ** 2
            )
            mass_cost = 0.12 * abs(
                math.log(max(float(mass[candidate]), 1e-6) / max(peak.mass, 1e-6))
            )
            score = distance + mass_cost
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None and best[0] <= 1.35:
            output[base_index] = best[1]
            available.remove(best[1])
    return output


def delay_neighbor_matches(
    base_peaks: list[e153.Peak],
    neighbor_id: int,
    polarization: int,
    ue: int,
    peak_cache: dict[str, np.ndarray],
) -> list[int]:
    count = int(peak_cache["delay_count"][neighbor_id, polarization, ue])
    coordinate = peak_cache["delay_coordinate"][
        neighbor_id, polarization, ue, :count
    ]
    mass = peak_cache["delay_mass"][neighbor_id, polarization, ue, :count]
    available = set(range(count))
    output = [-1] * len(base_peaks)
    for base_index, peak in enumerate(base_peaks):
        best: tuple[float, int] | None = None
        for candidate in available:
            distance = abs(
                signed_periodic_delta(
                    float(coordinate[candidate]),
                    float(peak.coordinate[0]),
                    e153.e138.S_SC,
                )
            )
            if distance > e153.DELAY_MATCH_BINS:
                continue
            mass_cost = 0.12 * abs(
                math.log(max(float(mass[candidate]), 1e-6) / max(peak.mass, 1e-6))
            )
            score = distance / e153.DELAY_MATCH_BINS + mass_cost
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None and best[0] <= 1.35:
            output[base_index] = best[1]
            available.remove(best[1])
    return output


def action_confidence(
    presence: float,
    residual: float,
    residual_scale: float,
    balance: float,
    risk_scale: float,
) -> float:
    value = presence * math.exp(-0.5 * (residual / residual_scale) ** 2)
    value *= 0.45 + 0.55 * balance
    return float(np.clip(value * risk_scale, 0.0, 1.0))


def angle_actions(
    base: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_offsets: np.ndarray,
    distance_weights: np.ndarray,
    ue: int,
    peak_cache: dict[str, np.ndarray],
    risk_scale: float,
    config: Config,
) -> list[PeakAction]:
    base_peaks = e153.extract_angle_peaks(base)[: config.maximum_peaks]
    if not base_peaks:
        return []
    observations: list[list[tuple[int, np.ndarray, float, float, float]]] = [
        [] for _ in base_peaks
    ]
    for local, neighbor_id in enumerate(neighbor_ids):
        matches = angle_neighbor_matches(base_peaks, int(neighbor_id), ue, peak_cache)
        count = int(peak_cache["angle_count"][neighbor_id, ue])
        coordinate = peak_cache["angle_coordinate"][neighbor_id, ue, :count]
        mass = peak_cache["angle_mass"][neighbor_id, ue, :count]
        for peak_index, candidate in enumerate(matches):
            if candidate < 0:
                continue
            peak = base_peaks[peak_index]
            dh = signed_periodic_delta(
                float(coordinate[candidate, 0]),
                float(peak.coordinate[0]),
                e153.e138.MH,
            )
            dv = signed_periodic_delta(
                float(coordinate[candidate, 1]),
                float(peak.coordinate[1]),
                e153.e138.MV,
            )
            observations[peak_index].append(
                (
                    local,
                    neighbor_offsets[local],
                    dh,
                    dv,
                    math.log(max(float(mass[candidate]), 1e-6)),
                )
            )

    actions: list[PeakAction] = []
    total_weight = max(float(distance_weights.sum()), TINY)
    for peak_index, peak in enumerate(base_peaks):
        rows = observations[peak_index]
        if len(rows) < config.minimum_observations:
            continue
        local_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
        offsets = np.stack([row[1] for row in rows]).astype(np.float64)
        dh = np.asarray([row[2] for row in rows], dtype=np.float64)
        dv = np.asarray([row[3] for row in rows], dtype=np.float64)
        log_mass = np.asarray([row[4] for row in rows], dtype=np.float64)
        weights = distance_weights[local_ids].astype(np.float64)
        presence = float(weights.sum() / total_weight)
        pred_h, residual_h = robust_affine_intercept(
            offsets, dh, weights, config.affine_ridge
        )
        pred_v, residual_v = robust_affine_intercept(
            offsets, dv, weights, config.affine_ridge
        )
        pred_log_mass, _ = robust_affine_intercept(
            offsets, log_mass, weights, config.affine_ridge
        )
        residual = math.sqrt(
            (residual_h / e153.ANGLE_MATCH_H) ** 2
            + (residual_v / e153.ANGLE_MATCH_V) ** 2
        )
        balance = directional_balance(offsets, weights)
        confidence = action_confidence(
            presence,
            residual,
            config.angle_residual_scale,
            balance,
            risk_scale,
        )
        if confidence < config.minimum_confidence:
            continue
        predicted_mass = math.exp(pred_log_mass) * math.sqrt(max(presence, 1e-6))
        mass_ratio = float(
            np.clip(
                predicted_mass / max(peak.mass, 1e-6),
                config.minimum_mass_ratio,
                config.maximum_mass_ratio,
            )
        )
        actions.append(
            PeakAction(
                coordinate=peak.coordinate,
                shift=(
                    rounded_unit_shift(pred_h, config.shift_threshold),
                    rounded_unit_shift(pred_v, config.shift_threshold),
                ),
                confidence=confidence,
                mass_ratio=mass_ratio,
                peak_mass=peak.mass,
            )
        )
    return actions


def delay_actions(
    base: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_offsets: np.ndarray,
    distance_weights: np.ndarray,
    polarization: int,
    ue: int,
    peak_cache: dict[str, np.ndarray],
    risk_scale: float,
    config: Config,
) -> list[PeakAction]:
    base_peaks = e153.extract_delay_peaks(base)[: config.maximum_peaks]
    if not base_peaks:
        return []
    observations: list[list[tuple[int, np.ndarray, float, float]]] = [
        [] for _ in base_peaks
    ]
    for local, neighbor_id in enumerate(neighbor_ids):
        matches = delay_neighbor_matches(
            base_peaks, int(neighbor_id), polarization, ue, peak_cache
        )
        count = int(peak_cache["delay_count"][neighbor_id, polarization, ue])
        coordinate = peak_cache["delay_coordinate"][
            neighbor_id, polarization, ue, :count
        ]
        mass = peak_cache["delay_mass"][neighbor_id, polarization, ue, :count]
        for peak_index, candidate in enumerate(matches):
            if candidate < 0:
                continue
            delta = signed_periodic_delta(
                float(coordinate[candidate]),
                float(base_peaks[peak_index].coordinate[0]),
                e153.e138.S_SC,
            )
            observations[peak_index].append(
                (
                    local,
                    neighbor_offsets[local],
                    delta,
                    math.log(max(float(mass[candidate]), 1e-6)),
                )
            )

    actions: list[PeakAction] = []
    total_weight = max(float(distance_weights.sum()), TINY)
    for peak_index, peak in enumerate(base_peaks):
        rows = observations[peak_index]
        if len(rows) < config.minimum_observations:
            continue
        local_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
        offsets = np.stack([row[1] for row in rows]).astype(np.float64)
        delta = np.asarray([row[2] for row in rows], dtype=np.float64)
        log_mass = np.asarray([row[3] for row in rows], dtype=np.float64)
        weights = distance_weights[local_ids].astype(np.float64)
        presence = float(weights.sum() / total_weight)
        pred_delay, residual = robust_affine_intercept(
            offsets, delta, weights, config.affine_ridge
        )
        pred_log_mass, _ = robust_affine_intercept(
            offsets, log_mass, weights, config.affine_ridge
        )
        balance = directional_balance(offsets, weights)
        confidence = action_confidence(
            presence,
            residual,
            config.delay_residual_scale,
            balance,
            risk_scale,
        )
        if confidence < config.minimum_confidence:
            continue
        predicted_mass = math.exp(pred_log_mass) * math.sqrt(max(presence, 1e-6))
        mass_ratio = float(
            np.clip(
                predicted_mass / max(peak.mass, 1e-6),
                config.minimum_mass_ratio,
                config.maximum_mass_ratio,
            )
        )
        actions.append(
            PeakAction(
                coordinate=peak.coordinate,
                shift=(rounded_unit_shift(pred_delay, config.shift_threshold),),
                confidence=confidence,
                mass_ratio=mass_ratio,
                peak_mass=peak.mass,
            )
        )
    return actions


def apply_angle_actions(
    base: np.ndarray, actions: list[PeakAction], fraction: float
) -> tuple[np.ndarray, dict[str, float]]:
    target = np.asarray(base, dtype=np.float64).copy()
    source = np.asarray(base, dtype=np.float64)
    assigned = np.zeros(source.shape, dtype=bool)
    moved_mass = 0.0
    scaled_mass = 0.0
    for action in sorted(actions, key=lambda item: item.peak_mass, reverse=True):
        indices: list[tuple[int, int]] = []
        h, v = action.coordinate
        for dh in (-1, 0, 1):
            for dv in (-1, 0, 1):
                index = ((h + dh) % e153.e138.MH, (v + dv) % e153.e138.MV)
                if not assigned[index]:
                    indices.append(index)
                    assigned[index] = True
        action_fraction = fraction * action.confidence
        for index in indices:
            amount = min(target[index], action_fraction * source[index])
            target[index] -= amount
            destination = (
                (index[0] + action.shift[0]) % e153.e138.MH,
                (index[1] + action.shift[1]) % e153.e138.MV,
            )
            target[destination] += amount * action.mass_ratio
            if action.shift != (0, 0):
                moved_mass += amount
            scaled_mass += amount * abs(action.mass_ratio - 1.0)
    total = float(target.sum())
    if total <= TINY:
        return np.asarray(base, dtype=np.float32), {"moved_mass": 0.0, "scaled_mass": 0.0}
    return (target / total).astype(np.float32), {
        "moved_mass": moved_mass,
        "scaled_mass": scaled_mass,
    }


def apply_delay_actions(
    base: np.ndarray, actions: list[PeakAction], fraction: float
) -> tuple[np.ndarray, dict[str, float]]:
    target = np.asarray(base, dtype=np.float64).copy()
    source = np.asarray(base, dtype=np.float64)
    assigned = np.zeros(source.shape, dtype=bool)
    moved_mass = 0.0
    scaled_mass = 0.0
    for action in sorted(actions, key=lambda item: item.peak_mass, reverse=True):
        indices = []
        delay = action.coordinate[0]
        for offset in range(-2, 3):
            index = (delay + offset) % e153.e138.S_SC
            if not assigned[index]:
                indices.append(index)
                assigned[index] = True
        action_fraction = fraction * action.confidence
        for index in indices:
            amount = min(target[index], action_fraction * source[index])
            target[index] -= amount
            destination = (index + action.shift[0]) % e153.e138.S_SC
            target[destination] += amount * action.mass_ratio
            if action.shift != (0,):
                moved_mass += amount
            scaled_mass += amount * abs(action.mass_ratio - 1.0)
    total = float(target.sum())
    if total <= TINY:
        return np.asarray(base, dtype=np.float32), {"moved_mass": 0.0, "scaled_mass": 0.0}
    return (target / total).astype(np.float32), {
        "moved_mass": moved_mass,
        "scaled_mass": scaled_mass,
    }


def build_transport_targets(
    v7_angle: np.ndarray,
    v7_delay: np.ndarray,
    neighbor_ids: np.ndarray,
    positions: np.ndarray,
    query_ids: np.ndarray,
    site: np.ndarray,
    regime: np.ndarray,
    peak_cache: dict[str, np.ndarray],
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    count = len(query_ids)
    target_angle = np.empty_like(v7_angle, dtype=np.float32)
    target_delay = np.empty_like(v7_delay, dtype=np.float32)
    diagnostics = {
        key: np.zeros(count, dtype=np.float32)
        for key in (
            "angle_action_count",
            "delay_action_count",
            "angle_shift_count",
            "delay_shift_count",
            "angle_moved_mass",
            "delay_moved_mass",
            "angle_scaled_mass",
            "delay_scaled_mass",
        )
    }
    regime_scale = np.asarray((1.00, 1.00, 0.85, 0.65), dtype=np.float64)
    site_scale = np.asarray((1.00, 0.85), dtype=np.float64)

    for row in range(count):
        selected = neighbor_ids[row, : config.neighbor_count].astype(np.int64)
        offsets = positions[selected, :2] - positions[query_ids[row], None, :2]
        distance = np.linalg.norm(offsets, axis=1)
        weights = 1.0 / np.maximum(distance, 1e-3) ** 2
        risk_scale = float(regime_scale[regime[row]] * site_scale[site[row]])
        for ue in range(e153.e138.N_UE):
            actions = angle_actions(
                v7_angle[row, ue],
                selected,
                offsets,
                weights,
                ue,
                peak_cache,
                risk_scale,
                config,
            )
            target_angle[row, ue], detail = apply_angle_actions(
                v7_angle[row, ue], actions, config.angle_fraction
            )
            diagnostics["angle_action_count"][row] += len(actions)
            diagnostics["angle_shift_count"][row] += sum(
                action.shift != (0, 0) for action in actions
            )
            diagnostics["angle_moved_mass"][row] += detail["moved_mass"]
            diagnostics["angle_scaled_mass"][row] += detail["scaled_mass"]
        for polarization in range(e153.e138.MP):
            for ue in range(e153.e138.N_UE):
                actions = delay_actions(
                    v7_delay[row, polarization, ue],
                    selected,
                    offsets,
                    weights,
                    polarization,
                    ue,
                    peak_cache,
                    risk_scale,
                    config,
                )
                target_delay[row, polarization, ue], detail = apply_delay_actions(
                    v7_delay[row, polarization, ue], actions, config.delay_fraction
                )
                diagnostics["delay_action_count"][row] += len(actions)
                diagnostics["delay_shift_count"][row] += sum(
                    action.shift != (0,) for action in actions
                )
                diagnostics["delay_moved_mass"][row] += detail["moved_mass"]
                diagnostics["delay_scaled_mass"][row] += detail["scaled_mass"]
        if (row + 1) % 50 == 0:
            print(f"E154轨迹运输 {row + 1}/{count}", flush=True)
    return target_angle, target_delay, diagnostics


def endpoint_cosine(
    prediction: np.ndarray, truth: np.ndarray, vector_axes: tuple[int, ...]
) -> np.ndarray:
    untouched = tuple(
        axis for axis in range(prediction.ndim) if axis not in vector_axes
    )
    left = np.transpose(prediction, (*untouched, *vector_axes)).reshape(
        *[prediction.shape[axis] for axis in untouched], -1
    )
    right = np.transpose(truth, (*untouched, *vector_axes)).reshape(
        *[truth.shape[axis] for axis in untouched], -1
    )
    numerator = np.sum(left * right, axis=-1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1, dtype=np.float64)
        * np.sum(right * right, axis=-1, dtype=np.float64)
    )
    return np.divide(
        numerator,
        np.where(denominator == 0.0, 1e-12, denominator),
    ).mean(axis=tuple(range(1, len(untouched))), dtype=np.float64)


def full_targets(
    base: torch.Tensor,
    target_angle: torch.Tensor,
    target_delay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_pas = pas_distribution(base)
    shaped_pas = current_pas
    current_angle = shaped_pas.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2)
    angle_ratio = target_angle / current_angle.clamp_min(TINY)
    desired_pas = shaped_pas * angle_ratio.permute(0, 2, 3, 1)[:, None, :, :, :, None]
    desired_pas = normalized(desired_pas, (1, 2, 3), fallback=current_pas)

    current_pdp = pdp_distribution(base)
    shaped_pdp = current_pdp.reshape(
        len(base),
        e153.e138.MP,
        e153.e138.MH,
        e153.e138.MV,
        e153.e138.N_UE,
        e153.e138.S_SC,
    )
    current_delay = shaped_pdp.mean(dim=(2, 3))
    delay_ratio = target_delay / current_delay.clamp_min(TINY)
    desired_pdp = shaped_pdp * delay_ratio[:, :, None, None, :, :]
    desired_pdp = normalized(desired_pdp, (5,), fallback=shaped_pdp)
    return desired_pas, desired_pdp.reshape_as(current_pdp)


def project_variant(
    base: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
    variant: str,
    config: Config,
) -> torch.Tensor:
    if variant == "identity":
        return base
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    output = base
    order = config.order.split("_")
    for _ in range(config.cycles):
        for domain in order:
            if domain == "pas" and variant in ("pas_transport", "joint_pas_pdp"):
                output = exact_pas_projection(output, desired_pas)
            elif domain == "pdp" and variant in ("pdp_transport", "joint_pas_pdp"):
                output = exact_pdp_projection(output, desired_pdp)
        output = preserve_point_power(output, reference_power)
    return output


def evaluate_variants(
    v7_channel: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    target_angle: np.ndarray,
    target_delay: np.ndarray,
    device: torch.device,
    batch_size: int,
    config: Config,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metric_arrays = {
        name: [np.empty(len(query_ids), dtype=np.float64) for _ in range(4)]
        for name in VARIANT_NAMES
    }
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(v7_channel[start:stop], dtype=np.complex64, copy=True),
            device=device,
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:stop]], dtype=np.complex64, copy=True),
            device=device,
        )
        angle = torch.as_tensor(target_angle[start:stop], device=device)
        delay = torch.as_tensor(target_delay[start:stop], device=device)
        with torch.inference_mode():
            desired_pas, desired_pdp = full_targets(base, angle, delay)
            truth_pas = pas_distribution(truth)
            truth_pdp = pdp_distribution(truth)
            for name in VARIANT_NAMES:
                prediction = project_variant(
                    base, desired_pas, desired_pdp, name, config
                )
                values = point_metrics(prediction, truth, truth_pas, truth_pdp)
                for destination, source in zip(metric_arrays[name], values, strict=True):
                    destination[start:stop] = source
        if stop % 50 == 0 or stop == len(query_ids):
            print(f"E154完整评分 {stop}/{len(query_ids)}", flush=True)

    results: dict[str, object] = {}
    identity = statistics(*metric_arrays["identity"])
    if abs(float(identity["score"]) - 0.6444563457343335) > 1e-8:
        raise RuntimeError(f"V7 score mismatch: {identity}")
    for name in VARIANT_NAMES:
        overall = statistics(*metric_arrays[name])
        folds = [
            {"fold": fold, **statistics(*metric_arrays[name], fold_id == fold)}
            for fold in range(5)
        ]
        delta = float(overall["score"] - identity["score"])
        fold_delta = [
            float(
                folds[fold]["score"]
                - statistics(*metric_arrays["identity"], fold_id == fold)["score"]
            )
            for fold in range(5)
        ]
        results[name] = {
            "all": overall,
            "folds": folds,
            "delta_vs_v7": delta,
            "fold_delta_vs_v7": fold_delta,
            "positive_folds": int(np.sum(np.asarray(fold_delta) > 0.0)),
        }
    flattened = {
        f"{name}_{metric}": metric_arrays[name][index]
        for name in VARIANT_NAMES
        for index, metric in enumerate(("pas", "pdp", "error", "energy"))
    }
    return results, flattened


def finite_payload(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_payload(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_payload(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    required = (
        V7_CHANNEL,
        TRAIN_CHANNEL,
        TRAIN_PAS,
        TRAIN_PDP,
        NEIGHBOR_CACHE,
        OOF_META,
        REGIME_CACHE,
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    neighbor_cache = np.load(NEIGHBOR_CACHE, allow_pickle=False)
    if not np.array_equal(neighbor_cache["query_ids"].astype(np.int64), query_ids):
        raise RuntimeError("E154 neighbor cache query IDs do not match canonical 500")
    neighbor_ids = neighbor_cache["neighbor_ids"].astype(np.int64)
    _, positions, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)

    train_angle, train_delay = load_compact_train_spectra()
    peak_cache = load_peak_cache(train_angle, train_delay, config.maximum_peaks)
    v7_channel = np.load(V7_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    v7_angle, v7_delay, _ = e153.channel_group_maps(
        v7_channel, device, args.batch_size
    )
    truth_angle = train_angle[query_ids]
    truth_delay = train_delay[query_ids]

    target_angle, target_delay, diagnostics = build_transport_targets(
        v7_angle,
        v7_delay,
        neighbor_ids,
        positions,
        query_ids,
        site,
        regime,
        peak_cache,
        config,
    )
    angle_before = endpoint_cosine(v7_angle, truth_angle, (2, 3))
    angle_after = endpoint_cosine(target_angle, truth_angle, (2, 3))
    delay_before = endpoint_cosine(v7_delay, truth_delay, (3,))
    delay_after = endpoint_cosine(target_delay, truth_delay, (3,))

    results, metric_arrays = evaluate_variants(
        v7_channel,
        truth_all,
        query_ids,
        fold_id,
        target_angle,
        target_delay,
        device,
        args.batch_size,
        config,
    )
    primary = results["joint_pas_pdp"]
    gate = {
        "minimum_score_gain": 0.005,
        "minimum_positive_folds": 5,
        "score_gain": float(primary["delta_vs_v7"]),
        "positive_folds": int(primary["positive_folds"]),
    }
    gate["passed"] = bool(
        gate["score_gain"] >= gate["minimum_score_gain"]
        and gate["positive_folds"] >= gate["minimum_positive_folds"]
    )

    endpoint = {
        "angle_v7": float(np.mean(angle_before)),
        "angle_transport": float(np.mean(angle_after)),
        "angle_delta": float(np.mean(angle_after - angle_before)),
        "delay_v7": float(np.mean(delay_before)),
        "delay_transport": float(np.mean(delay_after)),
        "delay_delta": float(np.mean(delay_after - delay_before)),
        "angle_fold_delta": [
            float(np.mean((angle_after - angle_before)[fold_id == fold]))
            for fold in range(5)
        ],
        "delay_fold_delta": [
            float(np.mean((delay_after - delay_before)[fold_id == fold]))
            for fold in range(5)
        ],
    }
    action_summary = {
        key: {
            "mean": float(np.mean(value)),
            "median": float(np.median(value)),
            "p90": float(np.quantile(value, 0.90)),
        }
        for key, value in diagnostics.items()
    }
    report = {
        "experiment": VERSION,
        "status": "baseline_complete_network_not_started"
        if not gate["passed"]
        else "baseline_passed_network_required",
        "objective": (
            "Use fit-only official-neighbor peak trajectories to move, attenuate, or "
            "delete existing V7 spectral peaks without adding free peaks."
        ),
        "protocol": {
            "query_count": len(query_ids),
            "fold_count": 5,
            "official_inputs_only": True,
            "test_sionna_used": False,
            "v12_output_used": False,
            "free_peak_birth_allowed": False,
            "network_training_started": bool(gate["passed"]),
        },
        "config": asdict(config),
        "endpoint": endpoint,
        "actions": action_summary,
        "results": results,
        "gate": gate,
        "decision": {
            "train_action_network": bool(gate["passed"]),
            "generate_candidate": False,
            "reason": (
                "The deterministic trajectory endpoint passed the full-score gate."
                if gate["passed"]
                else "Neighbor peak trajectories did not pass the pre-registered full-score gate."
            ),
        },
        "seconds": float(time.time() - started),
    }
    if not finite_payload(report):
        raise RuntimeError("E154 report contains non-finite values")

    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=site,
        regime=regime,
        target_angle=target_angle.astype(np.float16),
        target_delay=target_delay.astype(np.float16),
        angle_before=angle_before,
        angle_after=angle_after,
        delay_before=delay_before,
        delay_after=delay_after,
        **diagnostics,
        **metric_arrays,
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in VARIANT_NAMES:
        result = results[name]
        print(
            f"E154 {name}: score={result['all']['score']:.6f} "
            f"delta={result['delta_vs_v7']:+.6f} "
            f"positive_folds={result['positive_folds']}/5",
            flush=True,
        )
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
