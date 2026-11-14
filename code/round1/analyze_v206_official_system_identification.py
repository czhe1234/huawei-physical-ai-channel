#!/usr/bin/env python3
"""Infer array, delay, and spatial-validation constraints from official train H."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v206_official_system_identification"
REPORT_DIR = PATHS.version_report_dir(206)
REPORT = REPORT_DIR / "v206_official_system_identification.json"
OUT_DIR = PATHS.cache_dir(206)
SPLIT_CACHE = OUT_DIR / "v206_spatial_calibration_split.npz"
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
LIGHT_SPEED = 299_792_458.0


def stratified_sample(indices: np.ndarray, labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) <= count:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    names = ("clear", "light", "medium", "heavy")
    available = {name: indices[labels[indices] == name] for name in names}
    raw = {name: count * len(value) / len(indices) for name, value in available.items()}
    allocation = {name: min(len(available[name]), int(math.floor(raw[name]))) for name in names}
    while sum(allocation.values()) < count:
        choices = [name for name in names if allocation[name] < len(available[name])]
        name = max(choices, key=lambda item: raw[item] - allocation[item])
        allocation[name] += 1
    selected = np.concatenate(
        [rng.choice(available[name], size=allocation[name], replace=False) for name in names]
    )
    return np.sort(selected.astype(np.int32))


def build_spatial_split(
    positions: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    count_per_split: int = 384,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    xy = positions[:, :2]
    tile = np.floor((xy - np.min(xy, axis=0, keepdims=True)) / 55.0).astype(np.int32)
    tile_id = tile[:, 0] * (int(np.max(tile[:, 1])) + 1) + tile[:, 1]
    valid_indices = np.flatnonzero(valid)
    target = np.asarray([np.mean(labels[valid] == name) for name in ("clear", "light", "medium", "heavy")])
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    splitter = GroupShuffleSplit(n_splits=128, test_size=0.42, random_state=206)
    for train_local, validation_local in splitter.split(valid_indices, groups=tile_id[valid_indices]):
        train_pool = valid_indices[train_local]
        validation_pool = valid_indices[validation_local]
        if min(len(train_pool), len(validation_pool)) < count_per_split:
            continue
        train_fraction = np.asarray(
            [np.mean(labels[train_pool] == name) for name in ("clear", "light", "medium", "heavy")]
        )
        validation_fraction = np.asarray(
            [np.mean(labels[validation_pool] == name) for name in ("clear", "light", "medium", "heavy")]
        )
        objective = float(np.sum(np.abs(train_fraction - target)) + np.sum(np.abs(validation_fraction - target)))
        if best is None or objective < best[0]:
            best = (objective, train_pool, validation_pool)
    if best is None:
        raise RuntimeError("Could not construct spatially disjoint calibration and validation pools")
    _, calibration_pool, validation_pool = best
    calibration = stratified_sample(calibration_pool, labels, count_per_split, 2061)
    validation = stratified_sample(validation_pool, labels, count_per_split, 2062)
    if np.intersect1d(tile_id[calibration], tile_id[validation]).size:
        raise AssertionError("Spatial tiles overlap between calibration and validation")
    details = {
        "tile_size_m": 55.0,
        "calibration_count": int(len(calibration)),
        "validation_count": int(len(validation)),
        "calibration_tile_count": int(len(np.unique(tile_id[calibration]))),
        "validation_tile_count": int(len(np.unique(tile_id[validation]))),
        "calibration_regimes": {
            name: int(np.sum(labels[calibration] == name)) for name in ("clear", "light", "medium", "heavy")
        },
        "validation_regimes": {
            name: int(np.sum(labels[validation] == name)) for name in ("clear", "light", "medium", "heavy")
        },
    }
    return calibration, validation, details


def adjacent_phase(value: np.ndarray, axis: int) -> np.ndarray:
    left = [slice(None)] * value.ndim
    right = [slice(None)] * value.ndim
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    product = np.conj(value[tuple(left)]) * value[tuple(right)]
    reduce_axes = tuple(index for index in range(1, value.ndim))
    return np.angle(np.sum(product, axis=reduce_axes, dtype=np.complex128)) / np.pi


def fit_horizontal_axis(unit_direction: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    coefficient = np.linalg.lstsq(unit_direction, observed, rcond=None)[0]
    prediction = unit_direction @ coefficient
    horizontal = coefficient.copy()
    horizontal[2] = 0.0
    horizontal /= np.linalg.norm(horizontal)
    normal_a = np.asarray([horizontal[1], -horizontal[0], 0.0])
    normal_b = -normal_a
    median_direction = np.median(unit_direction, axis=0)
    normal = normal_a if np.dot(normal_a, median_direction) >= np.dot(normal_b, median_direction) else normal_b
    return {
        "fitted_axis": coefficient.tolist(),
        "horizontal_axis_normalized": horizontal.tolist(),
        "boresight_normal": normal.tolist(),
        "boresight_azimuth_deg": float(np.degrees(np.arctan2(normal[1], normal[0]))),
        "rmse": float(np.sqrt(np.mean((prediction - observed) ** 2))),
        "correlation": float(np.corrcoef(prediction, observed)[0, 1]),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"]
    labels = regime_of(blocked.astype(np.float32)).astype(str)
    channels = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")

    power = np.load(PATHS.shared_features / "train_total_power.npy").astype(np.float64)
    nonzero = power > 0.0
    calibration, validation, split_details = build_spatial_split(positions, labels, nonzero)
    np.savez_compressed(
        SPLIT_CACHE,
        calibration_indices=calibration,
        validation_indices=validation,
    )

    clear = np.flatnonzero((blocked == 0) & nonzero)
    clear_power = power[clear]
    clear = clear[clear_power >= np.quantile(clear_power, 0.25)]
    h = np.asarray(channels[clear], dtype=np.complex64)
    h7 = h.reshape(len(h), 2, 16, 8, 2, 2, 192)
    direction = positions[clear] - BS_POSITION
    distance = np.linalg.norm(direction, axis=1)
    unit = direction / distance[:, None]

    horizontal_phase = adjacent_phase(h7, axis=2)
    phase_fit = fit_horizontal_axis(unit, horizontal_phase)

    pas = np.sum(np.abs(np.fft.fft2(h7, axes=(2, 3))) ** 2, axis=(1, 4, 5, 6))
    peak = np.asarray(np.unravel_index(np.argmax(pas.reshape(len(h), -1), axis=1), (16, 8))).T
    horizontal_peak = (np.fft.fftfreq(16) * 2.0)[peak[:, 0]]
    peak_fit = fit_horizontal_axis(unit, horizontal_peak)

    delay_power = np.sum(np.abs(np.fft.ifft(h, axis=-1)) ** 2, axis=(1, 2))
    delay_peak = np.argmax(delay_power, axis=1)
    slope, intercept = np.polyfit(distance, delay_peak, 1)
    inferred_spacing = slope * LIGHT_SPEED / 192.0
    spacing_candidates = {}
    for spacing in (15_000.0, 30_000.0, 60_000.0, 120_000.0):
        predicted = distance / LIGHT_SPEED * 192.0 * spacing
        offset = float(np.median(delay_peak - predicted))
        spacing_candidates[str(int(spacing))] = {
            "offset_bins": offset,
            "median_absolute_error_bins": float(np.median(np.abs(delay_peak - (predicted + offset)))),
        }

    ue_product = np.sum(
        np.conj(h7[:, :, :, :, :, 0, :]) * h7[:, :, :, :, :, 1, :],
        axis=(1, 2, 3, 4, 5),
        dtype=np.complex128,
    )
    ue_direction_cosine = np.angle(ue_product) / np.pi
    fixed_vertical = (BS_POSITION[2] - positions[clear, 2]) / distance
    ue_orientation = {
        "observed_abs_median": float(np.median(np.abs(ue_direction_cosine))),
        "track_bs_zero_median_absolute_error": float(np.median(np.abs(ue_direction_cosine))),
        "fixed_vertical_positive_median_absolute_error": float(
            np.median(np.abs(ue_direction_cosine - fixed_vertical))
        ),
        "fixed_vertical_negative_median_absolute_error": float(
            np.median(np.abs(ue_direction_cosine + fixed_vertical))
        ),
        "interpretation": "The near-zero UE vertical phase strongly favors a receiver that tracks the BS.",
    }

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official clear-point phase and delay geometry directly constrain array orientation, UE orientation, "
            "and subcarrier spacing before any Sionna parameter search."
        ),
        "official_exact_zero_count": int(np.sum(~nonzero)),
        "system_identification_count": int(len(clear)),
        "spatial_split": split_details,
        "bs_horizontal_adjacent_phase": phase_fit,
        "bs_horizontal_fft_peak": peak_fit,
        "delay": {
            "distance_range_m": [float(np.min(distance)), float(np.max(distance))],
            "slope_bins_per_m": float(slope),
            "intercept_bins": float(intercept),
            "inferred_subcarrier_spacing_hz": float(inferred_spacing),
            "distance_peak_spearman": float(spearmanr(distance, delay_peak).statistic),
            "candidates": spacing_candidates,
        },
        "ue_orientation": ue_orientation,
        "fixed_constraints_for_search": {
            "port_layout": "TX [polarization,horizontal,vertical], RX [polarization,vertical]",
            "tx_azimuth_center_deg": float(
                np.mean([phase_fit["boresight_azimuth_deg"], peak_fit["boresight_azimuth_deg"]])
            ),
            "rx_orientation": "track_bs",
            "subcarrier_spacing_hz": 30_000,
            "materials": "canonical Sionna Etoile ITU defaults",
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "spatial_split": PATHS.relative(SPLIT_CACHE),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
