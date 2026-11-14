#!/usr/bin/env python3
"""Test a carrier-wavenumber angular-spectrum kernel on Round 2 phase holes."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E27-angular-spectrum-phase-kernel"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_angular_spectrum_phase_kernel"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_angular_spectrum_phase_kernel"
SKETCH = ARTIFACT_DIR / "official_channel_frequency_sketch.npy"
PAIR_CACHE = ARTIFACT_DIR / "official_spatial_pair_cross.npz"
MODEL_CACHE = ARTIFACT_DIR / "oof_angular_spectrum_models.npz"
INNER_CACHE = ARTIFACT_DIR / "oof_phase_kernel_inner_products.npz"
REPORT = ANALYSIS_DIR / "angular_spectrum_phase_kernel.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
POWER_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_train_point_power.npy"
SPLITS = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
FOUNDATION_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_foundation_benchmark"
    / "round2_foundation_benchmark.json"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
V4_OOF_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "v5_static_link_phase_calibration"
    / "v4_nested_oof_channel.npy"
)

TINY = 1e-30


@dataclass(frozen=True)
class Config:
    sketch_subcarriers: int = 16
    pair_neighbors: int = 8
    pair_minimum_distance_m: float = 0.20
    pair_maximum_distance_m: float = 12.0
    directions: int = 72
    azimuth_sectors: int = 8
    minimum_sector_pairs: int = 160
    smoothness: float = 0.08
    sum_constraint: float = 12.0
    maximum_neighbors: int = 24
    seed: int = 20_260_805


def frequency_indices(config: Config, subcarriers: int) -> np.ndarray:
    return np.unique(
        np.rint(
            np.linspace(0, subcarriers - 1, config.sketch_subcarriers)
        ).astype(np.int64)
    )


def build_sketch(config: Config, rebuild: bool) -> np.ndarray:
    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    frequencies = frequency_indices(config, channel.shape[-1])
    expected = (len(channel), channel.shape[1] * channel.shape[2] * len(frequencies))
    if SKETCH.is_file() and not rebuild:
        cached = np.load(SKETCH, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.complex64:
            return cached
    output = np.lib.format.open_memmap(
        SKETCH, mode="w+", dtype=np.complex64, shape=expected
    )
    for start in range(0, len(channel), 16):
        end = min(start + 16, len(channel))
        block = np.array(channel[start:end], dtype=np.complex64, copy=True)
        output[start:end] = block[..., frequencies].reshape(end - start, -1)
        if end % 400 == 0 or end == len(channel):
            output.flush()
            print(f"phase sketch {end}/{len(channel)}", flush=True)
    return np.load(SKETCH, mmap_mode="r")


def unique_spatial_pairs(
    positions: np.ndarray,
    site: np.ndarray,
    power: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    for site_value in np.unique(site):
        ids = np.flatnonzero((site == site_value) & (power > TINY))
        tree = cKDTree(positions[ids, :2])
        distance, local = tree.query(
            positions[ids, :2], k=config.pair_neighbors + 1, workers=-1
        )
        left = np.repeat(ids, config.pair_neighbors)
        right = ids[local[:, 1:].reshape(-1)]
        separation = distance[:, 1:].reshape(-1)
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        pairs = np.column_stack([low, high])
        keep = (
            (separation >= config.pair_minimum_distance_m)
            & (separation <= config.pair_maximum_distance_m)
            & (low != high)
        )
        pairs = pairs[keep]
        separation = separation[keep]
        pairs, unique_index = np.unique(pairs, axis=0, return_index=True)
        pair_parts.append(pairs)
        distance_parts.append(separation[unique_index])
    joined = np.concatenate(pair_parts)
    return (
        joined[:, 0].astype(np.int64),
        joined[:, 1].astype(np.int64),
        np.concatenate(distance_parts).astype(np.float32),
    )


def build_pair_cache(config: Config, rebuild: bool) -> dict[str, np.ndarray]:
    if PAIR_CACHE.is_file() and not rebuild:
        return {key: value for key, value in np.load(PAIR_CACHE).items()}
    _, positions, _, site, _ = load_positions()
    power = np.load(POWER_CACHE).astype(np.float64)
    sketch = build_sketch(config, rebuild)
    left, right, distance = unique_spatial_pairs(positions, site, power, config)
    cross = np.empty(len(left), dtype=np.complex64)
    sketch_power = np.sum(np.abs(sketch) ** 2, axis=1, dtype=np.float64)
    for start in range(0, len(left), 1024):
        end = min(start + 1024, len(left))
        first = np.asarray(sketch[left[start:end]], dtype=np.complex64)
        second = np.asarray(sketch[right[start:end]], dtype=np.complex64)
        numerator = np.sum(np.conj(first) * second, axis=1, dtype=np.complex128)
        denominator = np.sqrt(
            sketch_power[left[start:end]] * sketch_power[right[start:end]]
        )
        cross[start:end] = (numerator / np.maximum(denominator, TINY)).astype(
            np.complex64
        )
    payload = {
        "left": left,
        "right": right,
        "distance": distance,
        "cross": cross,
        "frequency_indices": frequency_indices(config, 192),
    }
    np.savez_compressed(PAIR_CACHE, **payload)
    return payload


def local_coordinates(
    left_position: np.ndarray,
    right_position: np.ndarray,
    bs_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    midpoint = 0.5 * (left_position[:, :2] + right_position[:, :2])
    radial = midpoint - bs_position[None, :2]
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    tangent = np.column_stack([-radial[:, 1], radial[:, 0]])
    delta = right_position[:, :2] - left_position[:, :2]
    radial_delta = np.sum(delta * radial, axis=1)
    tangent_delta = np.sum(delta * tangent, axis=1)
    azimuth = np.arctan2(radial[:, 1], radial[:, 0])
    return radial_delta, tangent_delta, azimuth


def direction_matrix(
    radial_delta: np.ndarray,
    tangent_delta: np.ndarray,
    beta: float,
    angles: np.ndarray,
) -> np.ndarray:
    projection = (
        radial_delta[:, None] * np.cos(angles)[None, :]
        + tangent_delta[:, None] * np.sin(angles)[None, :]
    )
    return np.exp(-1j * beta * projection)


def fit_nonnegative_spectrum(
    radial_delta: np.ndarray,
    tangent_delta: np.ndarray,
    target: np.ndarray,
    pair_weight: np.ndarray,
    beta: float,
    config: Config,
) -> np.ndarray:
    angles = np.linspace(-np.pi, np.pi, config.directions, endpoint=False)
    design = direction_matrix(radial_delta, tangent_delta, beta, angles)
    weight = np.sqrt(np.maximum(pair_weight, 1e-4))
    real_design = np.concatenate(
        [weight[:, None] * design.real, weight[:, None] * design.imag], axis=0
    )
    real_target = np.concatenate([weight * target.real, weight * target.imag])
    second_difference = (
        -2.0 * np.eye(config.directions)
        + np.roll(np.eye(config.directions), 1, axis=1)
        + np.roll(np.eye(config.directions), -1, axis=1)
    )
    augmented_design = np.concatenate(
        [
            real_design,
            math.sqrt(config.smoothness) * second_difference,
            np.full((1, config.directions), config.sum_constraint),
        ],
        axis=0,
    )
    augmented_target = np.concatenate(
        [real_target, np.zeros(config.directions), [config.sum_constraint]]
    )
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(0.0, np.inf),
        method="trf",
        max_iter=300,
        lsmr_tol="auto",
    )
    spectrum = np.maximum(result.x, 0.0)
    total = float(np.sum(spectrum))
    if total <= TINY:
        spectrum.fill(1.0 / len(spectrum))
    else:
        spectrum /= total
    return spectrum.astype(np.float32)


def model_diagnostics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    target_magnitude = np.abs(target)
    valid = (target_magnitude > 1e-6) & (np.abs(prediction) > 1e-6)
    phase_cosine = np.real(
        np.conj(prediction[valid]) * target[valid]
    ) / np.maximum(np.abs(prediction[valid]) * target_magnitude[valid], TINY)
    return {
        "count": int(len(target)),
        "complex_mse": float(np.mean(np.abs(prediction - target) ** 2)),
        "phase_cosine": float(np.mean(phase_cosine)) if np.any(valid) else 0.0,
        "target_magnitude_mean": float(np.mean(target_magnitude)),
        "prediction_magnitude_mean": float(np.mean(np.abs(prediction))),
    }


def carrier_beta_by_fold() -> dict[tuple[int, int], float]:
    report = json.loads(FOUNDATION_REPORT.read_text(encoding="utf-8"))
    return {
        (int(fold), int(site)): float(values["beta"])
        for fold, site_values in report["carrier_fit_only_by_fold"].items()
        for site, values in site_values.items()
    }


def fit_models(
    pairs: dict[str, np.ndarray], config: Config, rebuild: bool
) -> tuple[dict[tuple[int, int], dict[str, np.ndarray]], dict[str, object]]:
    if MODEL_CACHE.is_file() and not rebuild:
        cached = np.load(MODEL_CACHE)
        models: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for fold in range(5):
            for site in range(2):
                prefix = f"fold{fold}_site{site}"
                models[(fold, site)] = {
                    "global": cached[f"{prefix}_global"],
                    "sectors": cached[f"{prefix}_sectors"],
                    "beta": cached[f"{prefix}_beta"],
                }
        diagnostics = json.loads(str(cached["diagnostics_json"].item()))
        return models, diagnostics

    setup, positions, _, site, _ = load_positions()
    power = np.load(POWER_CACHE).astype(np.float64)
    split = np.load(SPLITS)
    left = pairs["left"].astype(np.int64)
    right = pairs["right"].astype(np.int64)
    target = pairs["cross"].astype(np.complex128)
    beta_lookup = carrier_beta_by_fold()
    angles = np.linspace(-np.pi, np.pi, config.directions, endpoint=False)
    models: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    diagnostics: dict[str, object] = {"rows": []}
    model_arrays: dict[str, np.ndarray] = {}

    for fold in range(5):
        fit = split[f"fold{fold}_fit"].astype(np.int64)
        in_fit = np.zeros(len(positions), dtype=bool)
        in_fit[fit] = True
        for site_value in np.unique(site):
            mask = (
                in_fit[left]
                & in_fit[right]
                & (site[left] == site_value)
                & (site[right] == site_value)
            )
            local_left = left[mask]
            local_right = right[mask]
            local_target = target[mask]
            radial_delta, tangent_delta, azimuth = local_coordinates(
                positions[local_left],
                positions[local_right],
                setup.bs_positions[int(site_value)],
            )
            geometric_power = np.sqrt(power[local_left] * power[local_right])
            positive = geometric_power[geometric_power > TINY]
            reference_power = float(np.median(positive)) if len(positive) else 1.0
            pair_weight = np.sqrt(np.clip(geometric_power / reference_power, 0.1, 10.0))
            pair_weight *= np.sqrt(np.clip(np.abs(local_target), 0.03, 1.0))
            beta = beta_lookup[(fold, int(site_value))]
            global_spectrum = fit_nonnegative_spectrum(
                radial_delta,
                tangent_delta,
                local_target,
                pair_weight,
                beta,
                config,
            )
            sector_index = np.floor(
                ((azimuth + np.pi) % (2.0 * np.pi))
                * config.azimuth_sectors
                / (2.0 * np.pi)
            ).astype(np.int64)
            sector_spectra = np.empty(
                (config.azimuth_sectors, config.directions), dtype=np.float32
            )
            for sector in range(config.azimuth_sectors):
                sector_mask = sector_index == sector
                if int(np.sum(sector_mask)) < config.minimum_sector_pairs:
                    sector_spectra[sector] = global_spectrum
                else:
                    sector_spectra[sector] = fit_nonnegative_spectrum(
                        radial_delta[sector_mask],
                        tangent_delta[sector_mask],
                        local_target[sector_mask],
                        pair_weight[sector_mask],
                        beta,
                        config,
                    )
            design = direction_matrix(radial_delta, tangent_delta, beta, angles)
            global_prediction = design @ global_spectrum
            sector_prediction = np.sum(
                design * sector_spectra[sector_index], axis=1
            )
            radial_prediction = np.exp(-1j * beta * radial_delta)
            uniform_prediction = design.mean(axis=1)
            rotated_prediction = np.sum(
                design * np.roll(sector_spectra[sector_index], 7, axis=1), axis=1
            )
            prefix = f"fold{fold}_site{int(site_value)}"
            model = {
                "global": global_spectrum,
                "sectors": sector_spectra,
                "beta": np.asarray(beta, dtype=np.float64),
            }
            models[(fold, int(site_value))] = model
            model_arrays[f"{prefix}_global"] = global_spectrum
            model_arrays[f"{prefix}_sectors"] = sector_spectra
            model_arrays[f"{prefix}_beta"] = model["beta"]
            diagnostics["rows"].append(
                {
                    "fold": fold,
                    "site": int(site_value),
                    "pair_count": int(len(local_target)),
                    "beta": beta,
                    "global": model_diagnostics(local_target, global_prediction),
                    "sector": model_diagnostics(local_target, sector_prediction),
                    "radial": model_diagnostics(local_target, radial_prediction),
                    "uniform": model_diagnostics(local_target, uniform_prediction),
                    "rotated_37deg_control": model_diagnostics(
                        local_target, rotated_prediction
                    ),
                    "global_effective_directions": float(
                        1.0 / np.sum(global_spectrum.astype(np.float64) ** 2)
                    ),
                }
            )
            print(
                f"angular spectrum fold={fold} site={int(site_value)} "
                f"pairs={len(local_target)}",
                flush=True,
            )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_CACHE,
        **model_arrays,
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    return models, diagnostics


def idw_weights(distance: np.ndarray) -> np.ndarray:
    value = np.power(np.maximum(distance, 1e-4), -2.0)
    return value / np.maximum(np.sum(value), TINY)


def kernel_values(
    donor_position: np.ndarray,
    query_position: np.ndarray,
    bs_position: np.ndarray,
    model: dict[str, np.ndarray],
    config: Config,
) -> dict[str, np.ndarray]:
    query = np.broadcast_to(query_position[None, :], donor_position.shape)
    radial_delta, tangent_delta, azimuth = local_coordinates(
        donor_position, query, bs_position
    )
    angles = np.linspace(-np.pi, np.pi, config.directions, endpoint=False)
    design = direction_matrix(
        radial_delta, tangent_delta, float(model["beta"]), angles
    )
    sector_index = np.floor(
        ((azimuth + np.pi) % (2.0 * np.pi))
        * config.azimuth_sectors
        / (2.0 * np.pi)
    ).astype(np.int64)
    sector = np.sum(design * model["sectors"][sector_index], axis=1)
    global_value = design @ model["global"]
    uniform = np.mean(design, axis=1)
    rotated = np.sum(
        design * np.roll(model["sectors"][sector_index], 7, axis=1), axis=1
    )
    radial = np.exp(-1j * float(model["beta"]) * radial_delta)
    return {
        "global": global_value,
        "sector": sector,
        "uniform_control": uniform,
        "rotated_control": rotated,
        "radial_control": radial,
        "sector_conjugate_control": np.conj(sector),
    }


def normalized_coefficients(
    distance: np.ndarray, kernel: np.ndarray, mode: str
) -> np.ndarray:
    spatial = idw_weights(distance)
    if mode == "phase":
        value = kernel / np.maximum(np.abs(kernel), 1e-6)
        coefficient = spatial * value
    elif mode == "cov":
        coefficient = spatial * kernel
        coefficient /= np.maximum(np.sum(np.abs(coefficient)), TINY)
    else:
        raise ValueError(mode)
    return coefficient.astype(np.complex64)


def variant_names() -> list[str]:
    names: list[str] = []
    for neighbors in (8, 16, 24):
        for source in ("global", "sector"):
            for mode in ("phase", "cov"):
                names.append(f"{source}_{mode}_k{neighbors}")
    names.extend(
        [
            "radial_control_phase_k16",
            "uniform_control_phase_k16",
            "rotated_control_phase_k16",
            "sector_conjugate_control_phase_k16",
        ]
    )
    return names


def coefficient_bank(
    distance: np.ndarray,
    kernels: dict[str, np.ndarray],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for name in variant_names():
        source_mode, neighbors_text = name.rsplit("_k", 1)
        neighbors = int(neighbors_text)
        source, mode = source_mode.rsplit("_", 1)
        rows.append(
            normalized_coefficients(
                distance[:neighbors], kernels[source][:neighbors], mode
            )
        )
    maximum = max(len(row) for row in rows)
    output = np.zeros((len(rows), maximum), dtype=np.complex64)
    for index, row in enumerate(rows):
        output[index, : len(row)] = row
    return output


@torch.inference_mode()
def generate_inner_products(
    models: dict[tuple[int, int], dict[str, np.ndarray]],
    config: Config,
    device: torch.device,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if INNER_CACHE.is_file() and not rebuild:
        return {key: value for key, value in np.load(INNER_CACHE).items()}
    setup, positions, _, site, _ = load_positions()
    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    power = np.load(POWER_CACHE).astype(np.float64)
    split = np.load(SPLITS)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    reference = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    names = variant_names()
    arrays: dict[str, np.ndarray] = {
        "predicted_energy": np.zeros((len(names), len(query_ids)), dtype=np.float64),
        "predicted_truth_cross_real": np.zeros(
            (len(names), len(query_ids)), dtype=np.float64
        ),
        "predicted_truth_cross_imag": np.zeros(
            (len(names), len(query_ids)), dtype=np.float64
        ),
        "reference_predicted_cross_real": np.zeros(
            (len(names), len(query_ids)), dtype=np.float64
        ),
        "reference_predicted_cross_imag": np.zeros(
            (len(names), len(query_ids)), dtype=np.float64
        ),
        "truth_energy": np.zeros(len(query_ids), dtype=np.float64),
        "reference_energy": np.zeros(len(query_ids), dtype=np.float64),
        "reference_truth_cross_real": np.zeros(len(query_ids), dtype=np.float64),
        "reference_truth_cross_imag": np.zeros(len(query_ids), dtype=np.float64),
    }
    row_by_id = np.full(len(positions), -1, dtype=np.int64)
    row_by_id[query_ids] = np.arange(len(query_ids))

    for fold in range(5):
        fit = split[f"fold{fold}_fit"].astype(np.int64)
        held = query_ids[fold_id == fold]
        for site_value in np.unique(site[held]):
            fit_site = fit[(site[fit] == site_value) & (power[fit] > TINY)]
            held_site = held[site[held] == site_value]
            tree = cKDTree(positions[fit_site, :2])
            distances, local = tree.query(
                positions[held_site, :2], k=config.maximum_neighbors, workers=-1
            )
            donors = fit_site[local]
            model = models[(fold, int(site_value))]
            for local_row, query_id in enumerate(held_site):
                output_row = int(row_by_id[query_id])
                kernels = kernel_values(
                    positions[donors[local_row]],
                    positions[query_id],
                    setup.bs_positions[int(site_value)],
                    model,
                    config,
                )
                coefficients = coefficient_bank(distances[local_row], kernels)
                donor_tensor = torch.as_tensor(
                    np.array(
                        channel[donors[local_row]], dtype=np.complex64, copy=True
                    ).reshape(config.maximum_neighbors, -1),
                    device=device,
                )
                coefficient_tensor = torch.as_tensor(coefficients, device=device)
                prediction = coefficient_tensor @ donor_tensor
                truth = torch.as_tensor(
                    np.array(channel[query_id], dtype=np.complex64, copy=True).reshape(-1),
                    device=device,
                )
                ref = torch.as_tensor(
                    np.array(reference[output_row], dtype=np.complex64, copy=True).reshape(-1),
                    device=device,
                )
                predicted_truth = torch.sum(torch.conj(prediction) * truth[None], dim=1)
                reference_predicted = torch.sum(
                    torch.conj(ref[None]) * prediction, dim=1
                )
                arrays["predicted_energy"][:, output_row] = (
                    prediction.abs().square().sum(dim=1).double().cpu().numpy()
                )
                arrays["predicted_truth_cross_real"][:, output_row] = (
                    predicted_truth.real.double().cpu().numpy()
                )
                arrays["predicted_truth_cross_imag"][:, output_row] = (
                    predicted_truth.imag.double().cpu().numpy()
                )
                arrays["reference_predicted_cross_real"][:, output_row] = (
                    reference_predicted.real.double().cpu().numpy()
                )
                arrays["reference_predicted_cross_imag"][:, output_row] = (
                    reference_predicted.imag.double().cpu().numpy()
                )
                arrays["truth_energy"][output_row] = float(
                    truth.abs().square().sum().double().cpu()
                )
                arrays["reference_energy"][output_row] = float(
                    ref.abs().square().sum().double().cpu()
                )
                reference_truth = torch.sum(torch.conj(ref) * truth)
                arrays["reference_truth_cross_real"][output_row] = float(
                    reference_truth.real.double().cpu()
                )
                arrays["reference_truth_cross_imag"][output_row] = float(
                    reference_truth.imag.double().cpu()
                )
                if (output_row + 1) % 25 == 0:
                    print(f"phase kernel OOF row={output_row + 1}/500", flush=True)
    arrays["variant_names"] = np.asarray(names)
    arrays["query_ids"] = query_ids
    arrays["fold_id"] = fold_id
    arrays["site"] = site[query_ids].astype(np.int8)
    np.savez_compressed(INNER_CACHE, **arrays)
    return arrays


def point_errors(arrays: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = arrays["variant_names"].tolist()
    truth_energy = arrays["truth_energy"]
    reference_energy = arrays["reference_energy"]
    reference_truth = arrays["reference_truth_cross_real"]
    baseline_error = np.maximum(
        reference_energy + truth_energy - 2.0 * reference_truth, 0.0
    )
    candidate_names = ["identity"]
    errors = [baseline_error]
    alphas = (0.10, 0.25, 0.50, 1.00)
    for index, name in enumerate(names):
        predicted_energy = arrays["predicted_energy"][index]
        normalize = np.sqrt(
            reference_energy / np.maximum(predicted_energy, TINY)
        )
        predicted_truth = normalize * arrays["predicted_truth_cross_real"][index]
        reference_predicted = (
            normalize * arrays["reference_predicted_cross_real"][index]
        )
        for alpha in alphas:
            mixed_energy = (
                ((1.0 - alpha) ** 2 + alpha**2) * reference_energy
                + 2.0 * alpha * (1.0 - alpha) * reference_predicted
            )
            mixed_scale = np.sqrt(
                reference_energy / np.maximum(mixed_energy, TINY)
            )
            mixed_truth = mixed_scale * (
                (1.0 - alpha) * reference_truth + alpha * predicted_truth
            )
            error = np.maximum(
                reference_energy + truth_energy - 2.0 * mixed_truth, 0.0
            )
            candidate_names.append(f"{name}_a{alpha:.2f}")
            errors.append(error)
    return candidate_names, np.stack(errors)


def nmse(error: np.ndarray, energy: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(error[mask]) / np.maximum(np.sum(energy[mask]), TINY))


def summarize_candidates(
    names: list[str], errors: np.ndarray, arrays: dict[str, np.ndarray]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    energy = arrays["truth_energy"]
    fold_id = arrays["fold_id"]
    site = arrays["site"]
    baseline_nmse = nmse(errors[0], energy, np.ones(len(energy), dtype=bool))
    baseline_score = 1.0 / (1.0 + baseline_nmse)
    rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        overall = nmse(errors[index], energy, np.ones(len(energy), dtype=bool))
        folds = [
            nmse(errors[index], energy, fold_id == fold)
            for fold in np.unique(fold_id)
        ]
        rows.append(
            {
                "name": name,
                "nmse": overall,
                "equivalent_total_gain_if_pas_pdp_fixed": float(
                    0.2 * (1.0 / (1.0 + overall) - baseline_score)
                ),
                "fold_nmse": folds,
                "positive_fold_count": int(
                    sum(
                        value
                        < nmse(errors[0], energy, fold_id == fold)
                        for value, fold in zip(folds, np.unique(fold_id))
                    )
                ),
            }
        )
    rows.sort(key=lambda row: float(row["nmse"]))

    deployable = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if "control" not in name and "conjugate" not in name
        ],
        dtype=np.int64,
    )
    selected = np.zeros(len(energy), dtype=np.int64)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            train_mask = (fold_id != held_fold) & (site == site_value)
            held_mask = (fold_id == held_fold) & (site == site_value)
            training_nmse = np.asarray(
                [nmse(errors[index], energy, train_mask) for index in deployable]
            )
            chosen = int(deployable[int(np.argmin(training_nmse))])
            selected[held_mask] = chosen
            decisions.append(
                {
                    "held_fold": int(held_fold),
                    "site": int(site_value),
                    "candidate": names[chosen],
                    "training_nmse": float(np.min(training_nmse)),
                    "held_nmse": nmse(errors[chosen], energy, held_mask),
                    "held_count": int(np.sum(held_mask)),
                }
            )
    point = np.arange(len(energy))
    nested_error = errors[selected, point]
    nested_nmse = nmse(nested_error, energy, np.ones(len(energy), dtype=bool))
    nested_fold = [
        nmse(nested_error, energy, fold_id == fold) for fold in np.unique(fold_id)
    ]
    baseline_fold = [
        nmse(errors[0], energy, fold_id == fold) for fold in np.unique(fold_id)
    ]
    nested = {
        "baseline_nmse": baseline_nmse,
        "nmse": nested_nmse,
        "equivalent_total_gain_if_pas_pdp_fixed": float(
            0.2 * (1.0 / (1.0 + nested_nmse) - baseline_score)
        ),
        "fold_nmse": nested_fold,
        "fold_nmse_delta": [
            float(new - old) for new, old in zip(nested_fold, baseline_fold)
        ],
        "positive_fold_count": int(
            sum(new < old for new, old in zip(nested_fold, baseline_fold))
        ),
        "decisions": decisions,
    }
    return rows, nested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-sketch", action="store_true")
    parser.add_argument("--rebuild-models", action="store_true")
    parser.add_argument("--rebuild-inner-products", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = build_pair_cache(config, args.rebuild_sketch)
    models, diagnostics = fit_models(pairs, config, args.rebuild_models)
    arrays: dict[str, np.ndarray] | None = None
    candidate_rows: list[dict[str, object]] | None = None
    nested: dict[str, object] | None = None
    if not args.fit_only:
        arrays = generate_inner_products(
            models,
            config,
            torch.device(args.device),
            args.rebuild_inner_products,
        )
        names, errors = point_errors(arrays)
        candidate_rows, nested = summarize_candidates(names, errors, arrays)
    report = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "A carrier-wavenumber-constrained angular spectrum learned from official-H "
            "spatial cross-correlations predicts rapid aggregate complex phase across "
            "complete holes without assigning brittle path identities."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": (
            "fixed five complete spatial holes; each angular spectrum uses only pair "
            "cross-correlations whose two endpoints are in that fold's fit set"
        ),
        "config": asdict(config),
        "pair_count": int(len(pairs["left"])),
        "model_diagnostics": diagnostics,
        "nested_nmse_selection": nested,
        "top_candidates": candidate_rows[:20] if candidate_rows is not None else None,
        "outputs": {
            "sketch": PATHS.relative(SKETCH),
            "pairs": PATHS.relative(PAIR_CACHE),
            "models": PATHS.relative(MODEL_CACHE),
            "inner_products": PATHS.relative(INNER_CACHE),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pair_count": report["pair_count"],
                "nested_nmse_selection": nested,
                "top_candidates": report["top_candidates"][:5]
                if report["top_candidates"] is not None
                else None,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
