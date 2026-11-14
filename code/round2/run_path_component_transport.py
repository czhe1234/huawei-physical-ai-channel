#!/usr/bin/env python3
"""Track delay-angle components and transport their complex phase across Round 2 holes."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from data import load_positions
from evaluate_v3_anchored_projection import TARGET_AD, TARGET_PAS, TARGET_PDP, target_batch
from official_metrics import point_metric_statistics
from pre_sionna_spectral import project_channel
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, OOF_BASE, OOF_META, statistics


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_path_component_transport"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_path_component_transport"
REPORT = ANALYSIS_DIR / "path_component_transport.json"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
V3_METRICS = (
    PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / "oof_projection_metrics.npz"
)

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PORTS = MP * N_UE
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    components: int = 24
    transported_components: int = 16
    oversample: int = 2
    refinement_sweeps: int = 2
    donor_count: int = 40
    direction_bins: int = 180
    beta: float = 140.40
    radius_fractions: tuple[float, ...] = (0.55, 0.75, 0.90, 1.00)
    match_h_scale: float = 1.5
    match_v_scale: float = 1.0
    match_d_scale: float = 2.0
    match_cost_limit: float = 11.0
    match_coherence_floor: float = 0.10
    affine_ridge: float = 2.0
    seed: int = 20_260_805


@dataclass(frozen=True)
class ComponentCache:
    coordinates: np.ndarray
    coefficients: np.ndarray
    residual_fraction: np.ndarray
    point_power: np.ndarray


def component_cache_path(config: Config) -> Path:
    return ARTIFACT_DIR / (
        f"train_components_k{config.components}_os{config.oversample}.npz"
    )


def refined_component_cache_path(config: Config) -> Path:
    return ARTIFACT_DIR / (
        f"train_components_k{config.components}_os{config.oversample}"
        f"_ref{config.refinement_sweeps}.npz"
    )


def carrier_path(config: Config) -> Path:
    return ARTIFACT_DIR / (
        f"oof_path_carrier_k{config.transported_components}_os{config.oversample}"
        f"_ref{config.refinement_sweeps}_n{config.donor_count}"
        f"_dir{config.direction_bins}_{radius_tag(config)}.npy"
    )


def prediction_path(config: Config) -> Path:
    return ARTIFACT_DIR / (
        f"oof_path_parameters_k{config.transported_components}_os{config.oversample}"
        f"_ref{config.refinement_sweeps}_n{config.donor_count}"
        f"_dir{config.direction_bins}_{radius_tag(config)}.npz"
    )


def radius_tag(config: Config) -> str:
    return "r" + "_".join(f"{value:.3f}".rstrip("0").rstrip(".") for value in config.radius_fractions)


def report_path(config: Config) -> Path:
    return ANALYSIS_DIR / (
        f"path_component_transport_k{config.transported_components}_os{config.oversample}"
        f"_ref{config.refinement_sweeps}_n{config.donor_count}"
        f"_dir{config.direction_bins}_{radius_tag(config)}.json"
    )


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def circular_delta(destination: np.ndarray, source: np.ndarray, period: float) -> np.ndarray:
    return (np.asarray(destination) - np.asarray(source) + period / 2.0) % period - period / 2.0


def unit_complex(value: torch.Tensor) -> torch.Tensor:
    return value / value.abs().clamp_min(1e-20)


def _component_atom(
    coordinates: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return unit-norm atoms for coordinates expressed in native FFT bins."""

    horizontal = torch.arange(MH, dtype=torch.float32, device=device)
    vertical = torch.arange(MV, dtype=torch.float32, device=device)
    subcarrier = torch.arange(S_SC, dtype=torch.float32, device=device)
    phase = (
        2.0
        * math.pi
        * coordinates[..., 0, None, None, None]
        * horizontal[None, None, :, None, None]
        / MH
        + 2.0
        * math.pi
        * coordinates[..., 1, None, None, None]
        * vertical[None, None, None, :, None]
        / MV
        - 2.0
        * math.pi
        * coordinates[..., 2, None, None, None]
        * subcarrier[None, None, None, None, :]
        / S_SC
    )
    return torch.exp(1j * phase).to(dtype) / math.sqrt(MH * MV * S_SC)


@torch.inference_mode()
def extract_components(config: Config, device: torch.device, batch_size: int) -> ComponentCache:
    cache_path = component_cache_path(config)
    if cache_path.is_file():
        cached = np.load(cache_path)
        if (
            int(cached["components"]) == config.components
            and int(cached["oversample"]) == config.oversample
        ):
            return ComponentCache(
                coordinates=cached["coordinates"].astype(np.float32),
                coefficients=cached["coefficients"].astype(np.complex64),
                residual_fraction=cached["residual_fraction"].astype(np.float32),
                point_power=cached["point_power"].astype(np.float64),
            )

    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    count = len(channel)
    coordinates = np.zeros((count, config.components, 3), dtype=np.float32)
    coefficients = np.zeros((count, config.components, PORTS), dtype=np.complex64)
    residual_fraction = np.ones(count, dtype=np.float32)
    point_power = np.zeros(count, dtype=np.float64)
    os = config.oversample

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        source = torch.as_tensor(
            np.asarray(channel[start:end], dtype=np.complex64), device=device
        )
        source = (
            source.reshape(-1, MP, MH, MV, N_UE, S_SC)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(-1, PORTS, MH, MV, S_SC)
        )
        residual = source.clone()
        source_energy = source.abs().square().sum(dim=(1, 2, 3, 4))
        point_power[start:end] = source_energy.cpu().numpy().astype(np.float64)

        batch_coordinates = torch.zeros(
            (len(source), config.components, 3), dtype=torch.float32, device=device
        )
        batch_coefficients = torch.zeros(
            (len(source), config.components, PORTS), dtype=torch.complex64, device=device
        )
        for component in range(config.components):
            transformed = torch.fft.ifft(
                torch.fft.fft2(
                    residual,
                    s=(MH * os, MV * os),
                    dim=(-3, -2),
                ),
                n=S_SC * os,
                dim=-1,
            )
            power = transformed.abs().square().sum(dim=1).reshape(len(source), -1)
            flat_index = torch.argmax(power, dim=1)
            delay = flat_index % (S_SC * os)
            partial = flat_index // (S_SC * os)
            vertical = partial % (MV * os)
            horizontal = partial // (MV * os)
            selected = torch.stack([horizontal, vertical, delay], dim=1).float() / os
            atom = _component_atom(selected[:, None, :], residual.dtype, device)[:, 0]
            coefficient = torch.sum(
                torch.conj(atom[:, None]) * residual,
                dim=(-3, -2, -1),
            )
            residual = residual - coefficient[:, :, None, None, None] * atom[:, None]
            batch_coordinates[:, component] = selected
            batch_coefficients[:, component] = coefficient

        component_energy = batch_coefficients.abs().square().sum(dim=2)
        order = torch.argsort(component_energy, dim=1, descending=True)
        batch_coordinates = torch.gather(
            batch_coordinates, 1, order[:, :, None].expand(-1, -1, 3)
        )
        batch_coefficients = torch.gather(
            batch_coefficients, 1, order[:, :, None].expand(-1, -1, PORTS)
        )
        residual_energy = residual.abs().square().sum(dim=(1, 2, 3, 4))
        fraction = torch.where(
            source_energy > 0.0,
            residual_energy / source_energy.clamp_min(1e-30),
            torch.ones_like(source_energy),
        )
        coordinates[start:end] = batch_coordinates.cpu().numpy()
        coefficients[start:end] = batch_coefficients.cpu().numpy()
        residual_fraction[start:end] = fraction.cpu().numpy().astype(np.float32)
        print(
            f"component extraction {end}/{count} "
            f"median_explained={np.median(1.0 - residual_fraction[start:end]):.6f}",
            flush=True,
        )

    np.savez_compressed(
        cache_path,
        coordinates=coordinates,
        coefficients=coefficients,
        residual_fraction=residual_fraction,
        point_power=point_power,
        components=np.int32(config.components),
        oversample=np.int32(config.oversample),
        source_shape=np.asarray(channel.shape, dtype=np.int32),
    )
    return ComponentCache(coordinates, coefficients, residual_fraction, point_power)


def component_energy(coefficients: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(coefficients) ** 2, axis=-1, dtype=np.float64)


def _clamped_axis_estimate(
    sequence: torch.Tensor,
    initial: torch.Tensor,
    period: int,
    sign: float,
    limit: float,
) -> torch.Tensor:
    cross = torch.sum(
        torch.conj(sequence[..., :-1]) * sequence[..., 1:],
        dim=tuple(range(1, sequence.ndim)),
    )
    raw = torch.remainder(
        sign * torch.angle(cross) * period / (2.0 * math.pi),
        period,
    )
    delta = torch.remainder(raw - initial + period / 2.0, period) - period / 2.0
    return torch.remainder(initial + delta.clamp(-limit, limit), period)


def _refine_one_atom(
    residual: torch.Tensor,
    initial: torch.Tensor,
    iterations: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    estimate = initial.clone()
    horizontal_index = torch.arange(MH, dtype=torch.float32, device=residual.device)
    vertical_index = torch.arange(MV, dtype=torch.float32, device=residual.device)
    subcarrier_index = torch.arange(S_SC, dtype=torch.float32, device=residual.device)
    for _ in range(iterations):
        horizontal = torch.exp(
            1j * 2.0 * math.pi * estimate[:, 0, None] * horizontal_index[None] / MH
        ) / math.sqrt(MH)
        vertical = torch.exp(
            1j * 2.0 * math.pi * estimate[:, 1, None] * vertical_index[None] / MV
        ) / math.sqrt(MV)
        delay_sequence = torch.einsum(
            "bchvs,bh,bv->bcs", residual, torch.conj(horizontal), torch.conj(vertical)
        )
        estimate[:, 2] = _clamped_axis_estimate(
            delay_sequence, initial[:, 2], S_SC, -1.0, 0.45
        )

        delay = torch.exp(
            -1j * 2.0 * math.pi * estimate[:, 2, None] * subcarrier_index[None] / S_SC
        ) / math.sqrt(S_SC)
        horizontal_sequence = torch.einsum(
            "bchvs,bv,bs->bch", residual, torch.conj(vertical), torch.conj(delay)
        )
        estimate[:, 0] = _clamped_axis_estimate(
            horizontal_sequence, initial[:, 0], MH, 1.0, 0.45
        )

        horizontal = torch.exp(
            1j * 2.0 * math.pi * estimate[:, 0, None] * horizontal_index[None] / MH
        ) / math.sqrt(MH)
        vertical_sequence = torch.einsum(
            "bchvs,bh,bs->bcv", residual, torch.conj(horizontal), torch.conj(delay)
        )
        estimate[:, 1] = _clamped_axis_estimate(
            vertical_sequence, initial[:, 1], MV, 1.0, 0.45
        )

    atom = _component_atom(estimate[:, None], residual.dtype, residual.device)[:, 0]
    coefficient = torch.sum(
        torch.conj(atom[:, None]) * residual,
        dim=(-3, -2, -1),
    )
    return estimate, coefficient, atom


@torch.inference_mode()
def refine_components(
    raw: ComponentCache,
    config: Config,
    device: torch.device,
    batch_size: int,
) -> ComponentCache:
    if config.refinement_sweeps <= 0:
        return raw
    output_path = refined_component_cache_path(config)
    if output_path.is_file():
        cached = np.load(output_path)
        if int(cached["refinement_sweeps"]) == config.refinement_sweeps:
            return ComponentCache(
                cached["coordinates"].astype(np.float32),
                cached["coefficients"].astype(np.complex64),
                cached["residual_fraction"].astype(np.float32),
                cached["point_power"].astype(np.float64),
            )

    source_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    coordinates = raw.coordinates.copy()
    coefficients = raw.coefficients.copy()
    residual_fraction = np.ones(len(source_all), dtype=np.float32)
    for start in range(0, len(source_all), batch_size):
        end = min(start + batch_size, len(source_all))
        source = torch.as_tensor(
            np.array(source_all[start:end], dtype=np.complex64, copy=True), device=device
        )
        source = (
            source.reshape(-1, MP, MH, MV, N_UE, S_SC)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(-1, PORTS, MH, MV, S_SC)
        )
        coordinate = torch.as_tensor(
            coordinates[start:end], dtype=torch.float32, device=device
        ).clone()
        coefficient = torch.as_tensor(
            coefficients[start:end], dtype=torch.complex64, device=device
        ).clone()
        initial = coordinate.clone()
        atoms = _component_atom(coordinate, torch.complex64, device)
        model = torch.einsum("bkc,bkhvs->bchvs", coefficient, atoms)
        for _ in range(config.refinement_sweeps):
            for component in range(config.components):
                residual_plus = (
                    source
                    - model
                    + coefficient[:, component, :, None, None, None]
                    * atoms[:, component, None]
                )
                estimate, updated_coefficient, updated_atom = _refine_one_atom(
                    residual_plus,
                    initial[:, component],
                )
                model = (
                    model
                    - coefficient[:, component, :, None, None, None]
                    * atoms[:, component, None]
                    + updated_coefficient[:, :, None, None, None] * updated_atom[:, None]
                )
                coordinate[:, component] = estimate
                coefficient[:, component] = updated_coefficient
                atoms[:, component] = updated_atom
        energy = coefficient.abs().square().sum(dim=2)
        order = torch.argsort(energy, dim=1, descending=True)
        coordinate = torch.gather(coordinate, 1, order[:, :, None].expand(-1, -1, 3))
        coefficient = torch.gather(
            coefficient, 1, order[:, :, None].expand(-1, -1, PORTS)
        )
        coordinates[start:end] = coordinate.cpu().numpy()
        coefficients[start:end] = coefficient.cpu().numpy()
        residual = source - model
        source_energy = source.abs().square().sum(dim=(1, 2, 3, 4))
        residual_energy = residual.abs().square().sum(dim=(1, 2, 3, 4))
        residual_fraction[start:end] = torch.where(
            source_energy > 0.0,
            residual_energy / source_energy.clamp_min(1e-30),
            torch.ones_like(source_energy),
        ).cpu().numpy()
        print(
            f"component refinement {end}/{len(source_all)} "
            f"median_explained={np.median(1.0 - residual_fraction[start:end]):.6f}",
            flush=True,
        )
    np.savez_compressed(
        output_path,
        coordinates=coordinates,
        coefficients=coefficients,
        residual_fraction=residual_fraction,
        point_power=raw.point_power,
        components=np.int32(config.components),
        oversample=np.int32(config.oversample),
        refinement_sweeps=np.int32(config.refinement_sweeps),
    )
    return ComponentCache(coordinates, coefficients, residual_fraction, raw.point_power)


def match_components(
    anchor_coordinates: np.ndarray,
    anchor_coefficients: np.ndarray,
    other_coordinates: np.ndarray,
    other_coefficients: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_norm = np.linalg.norm(anchor_coefficients, axis=1)
    other_norm = np.linalg.norm(other_coefficients, axis=1)
    coherence = np.abs(anchor_coefficients.conj() @ other_coefficients.T)
    coherence /= np.maximum(anchor_norm[:, None] * other_norm[None, :], TINY)
    delta_h = circular_delta(
        other_coordinates[None, :, 0], anchor_coordinates[:, None, 0], MH
    )
    delta_v = circular_delta(
        other_coordinates[None, :, 1], anchor_coordinates[:, None, 1], MV
    )
    delta_d = circular_delta(
        other_coordinates[None, :, 2], anchor_coordinates[:, None, 2], S_SC
    )
    log_ratio = np.abs(
        np.log(np.maximum(other_norm[None, :], 1e-20))
        - np.log(np.maximum(anchor_norm[:, None], 1e-20))
    )
    cost = (
        (delta_h / config.match_h_scale) ** 2
        + (delta_v / config.match_v_scale) ** 2
        + (delta_d / config.match_d_scale) ** 2
        + 2.0 * (1.0 - coherence)
        + 0.08 * np.minimum(log_ratio, 5.0) ** 2
    )
    left, right = linear_sum_assignment(cost)
    matched = np.full(len(anchor_coordinates), -1, dtype=np.int16)
    matched_cost = np.full(len(anchor_coordinates), np.inf, dtype=np.float32)
    matched_coherence = np.zeros(len(anchor_coordinates), dtype=np.float32)
    for anchor_index, other_index in zip(left, right):
        accepted = (
            cost[anchor_index, other_index] <= config.match_cost_limit
            and coherence[anchor_index, other_index] >= config.match_coherence_floor
            and abs(delta_h[anchor_index, other_index]) <= 3.0
            and abs(delta_v[anchor_index, other_index]) <= 2.0
            and abs(delta_d[anchor_index, other_index]) <= 4.0
        )
        if accepted:
            matched[anchor_index] = other_index
            matched_cost[anchor_index] = cost[anchor_index, other_index]
            matched_coherence[anchor_index] = coherence[anchor_index, other_index]
    return matched, matched_cost, matched_coherence


def pair_trackability(
    cache: ComponentCache,
    positions: np.ndarray,
    site: np.ndarray,
    config: Config,
) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    distance_edges = (0.0, 3.0, 6.0, 10.0, 16.0)
    pair_rows: list[tuple[int, int, float]] = []
    valid = cache.point_power > TINY
    for site_value in np.unique(site):
        ids = np.flatnonzero((site == site_value) & valid)
        distance, local = cKDTree(positions[ids, :2]).query(
            positions[ids, :2], k=min(10, len(ids)), workers=-1
        )
        seen: set[tuple[int, int]] = set()
        for row, source in enumerate(ids):
            for column in range(1, local.shape[1]):
                target = int(ids[local[row, column]])
                pair = (min(int(source), target), max(int(source), target))
                if pair in seen:
                    continue
                seen.add(pair)
                separation = float(distance[row, column])
                if separation <= distance_edges[-1]:
                    pair_rows.append((pair[0], pair[1], separation))
    if len(pair_rows) > 8000:
        chosen = np.sort(rng.choice(len(pair_rows), size=8000, replace=False))
        pair_rows = [pair_rows[index] for index in chosen]

    bins: dict[str, dict[str, list[float]]] = {}
    for lower, upper in zip(distance_edges[:-1], distance_edges[1:]):
        bins[f"{lower:g}-{upper:g}m"] = {
            "coverage": [],
            "coherence": [],
            "h_delta": [],
            "v_delta": [],
            "d_delta": [],
        }
    path_count = config.transported_components
    for left, right, separation in pair_rows:
        key = next(
            f"{lower:g}-{upper:g}m"
            for lower, upper in zip(distance_edges[:-1], distance_edges[1:])
            if lower < separation <= upper
        )
        anchor_coordinates = cache.coordinates[left, :path_count]
        anchor_coefficients = cache.coefficients[left, :path_count]
        matched, _, coherence = match_components(
            anchor_coordinates,
            anchor_coefficients,
            cache.coordinates[right],
            cache.coefficients[right],
            config,
        )
        energy = component_energy(anchor_coefficients)
        accepted = matched >= 0
        bins[key]["coverage"].append(
            float(np.sum(energy[accepted]) / max(float(np.sum(energy)), TINY))
        )
        if np.any(accepted):
            weight = energy[accepted]
            weight /= max(float(np.sum(weight)), TINY)
            other_index = matched[accepted].astype(np.int64)
            other_coordinates = cache.coordinates[right, other_index]
            anchor_selected = anchor_coordinates[accepted]
            bins[key]["coherence"].append(float(np.sum(weight * coherence[accepted])))
            bins[key]["h_delta"].append(
                float(np.sum(weight * np.abs(circular_delta(other_coordinates[:, 0], anchor_selected[:, 0], MH))))
            )
            bins[key]["v_delta"].append(
                float(np.sum(weight * np.abs(circular_delta(other_coordinates[:, 1], anchor_selected[:, 1], MV))))
            )
            bins[key]["d_delta"].append(
                float(np.sum(weight * np.abs(circular_delta(other_coordinates[:, 2], anchor_selected[:, 2], S_SC))))
            )
    return {
        "pair_count": len(pair_rows),
        "distance_bins": {
            key: {name: summarize(np.asarray(values)) for name, values in row.items()}
            for key, row in bins.items()
        },
    }


def weighted_intercept(
    values: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    max_extrapolation: float | None = None,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)
    if len(values) < 3:
        return float(np.sum(weights * values) / np.sum(weights))
    scale = max(float(np.median(np.linalg.norm(offsets, axis=1))), 1.0)
    design = np.column_stack([np.ones(len(values)), offsets[:, 0] / scale, offsets[:, 1] / scale])
    gram = design.T @ (weights[:, None] * design)
    gram += np.diag([1e-8, ridge, ridge])
    right = design.T @ (weights * values)
    try:
        coefficients = np.linalg.solve(gram, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, right, rcond=1e-8)[0]
    output = float(coefficients[0])
    if max_extrapolation is not None:
        low = float(np.min(values) - max_extrapolation)
        high = float(np.max(values) + max_extrapolation)
        output = float(np.clip(output, low, high))
    return output


def wavevector_grid(config: Config) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * math.pi, config.direction_bins, endpoint=False)
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    rows = [
        config.beta * radius * direction for radius in config.radius_fractions
    ]
    return np.concatenate([np.zeros((1, 2)), *rows], axis=0).astype(np.float64)


def estimate_wavevector(
    coefficients: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    norms = np.linalg.norm(coefficients, axis=1)
    valid = norms > 1e-20
    if np.sum(valid) < 4:
        return fallback.astype(np.float64), 0.0, 0.0
    unit = coefficients[valid] / norms[valid, None]
    local_weights = np.asarray(weights[valid], dtype=np.float64)
    local_weights /= max(float(np.sum(local_weights)), TINY)
    phase = np.exp(-1j * (offsets[valid] @ grid.T))
    summary = phase.T @ (local_weights[:, None] * unit)
    score = np.linalg.norm(summary, axis=1)
    best = int(np.argmax(score))
    if best == 0:
        selected = fallback.astype(np.float64)
    else:
        selected = grid[best]
    sorted_score = np.sort(score)
    runner_up = float(sorted_score[-2]) if len(sorted_score) > 1 else 0.0
    return selected, float(score[best]), float(score[best] - runner_up)


def predict_query_components(
    query_position: np.ndarray,
    bs_position: np.ndarray,
    donor_ids: np.ndarray,
    positions: np.ndarray,
    cache: ComponentCache,
    config: Config,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    anchor_id = int(donor_ids[0])
    path_count = config.transported_components
    anchor_coordinates = cache.coordinates[anchor_id, :path_count]
    anchor_coefficients = cache.coefficients[anchor_id, :path_count]
    observations: list[list[tuple[int, int, float, float]]] = [
        [(anchor_id, path, 0.0, 1.0)] for path in range(path_count)
    ]

    for donor_id in donor_ids[1:]:
        donor_id = int(donor_id)
        matched, costs, coherence = match_components(
            anchor_coordinates,
            anchor_coefficients,
            cache.coordinates[donor_id],
            cache.coefficients[donor_id],
            config,
        )
        for path, component in enumerate(matched):
            if component >= 0:
                observations[path].append(
                    (donor_id, int(component), float(costs[path]), float(coherence[path]))
                )

    output_coordinates = np.zeros((path_count, 3), dtype=np.float32)
    output_coefficients = np.zeros((path_count, PORTS), dtype=np.complex64)
    path_observation_count = np.zeros(path_count, dtype=np.int16)
    path_wave_coherence = np.zeros(path_count, dtype=np.float32)
    path_wave_margin = np.zeros(path_count, dtype=np.float32)

    radial = query_position[:2] - bs_position[:2]
    radial /= max(float(np.linalg.norm(radial)), 1e-8)
    fallback = -config.beta * radial
    donor_scale = max(
        float(np.median(np.linalg.norm(positions[donor_ids, :2] - query_position[None, :2], axis=1))),
        1.0,
    )

    for path, rows in enumerate(observations):
        ids = np.asarray([row[0] for row in rows], dtype=np.int64)
        components = np.asarray([row[1] for row in rows], dtype=np.int64)
        match_coherence = np.asarray([row[3] for row in rows], dtype=np.float64)
        offsets = positions[ids, :2] - query_position[None, :2]
        distance = np.linalg.norm(offsets, axis=1)
        spatial_weight = np.exp(-0.5 * (distance / max(1.5 * donor_scale, 1.0)) ** 2)
        weights = spatial_weight * np.maximum(match_coherence, 0.10)
        coordinates = cache.coordinates[ids, components]
        coefficients = cache.coefficients[ids, components]
        path_observation_count[path] = len(rows)

        for axis, period, extrapolation in ((0, MH, 0.75), (1, MV, 0.50), (2, S_SC, 0.75)):
            unwrapped = anchor_coordinates[path, axis] + circular_delta(
                coordinates[:, axis], anchor_coordinates[path, axis], period
            )
            predicted = weighted_intercept(
                unwrapped,
                offsets,
                weights,
                config.affine_ridge,
                extrapolation,
            )
            output_coordinates[path, axis] = np.float32(predicted % period)

        wavevector, coherence, margin = estimate_wavevector(
            coefficients,
            offsets,
            weights,
            grid,
            fallback,
        )
        transported = coefficients * np.exp(-1j * (offsets @ wavevector))[:, None]
        vector = np.sum(weights[:, None] * transported, axis=0) / max(float(np.sum(weights)), TINY)
        observed_log_norm = np.log(np.maximum(np.linalg.norm(coefficients, axis=1), 1e-20))
        predicted_log_norm = weighted_intercept(
            observed_log_norm,
            offsets,
            weights,
            config.affine_ridge,
            0.75,
        )
        vector_norm = max(float(np.linalg.norm(vector)), 1e-20)
        vector *= math.exp(predicted_log_norm) / vector_norm
        output_coefficients[path] = vector.astype(np.complex64)
        path_wave_coherence[path] = coherence
        path_wave_margin[path] = margin

    path_energy = component_energy(output_coefficients)
    energy_weight = path_energy / max(float(np.sum(path_energy)), TINY)
    diagnostics = {
        "mean_observations": float(np.sum(energy_weight * path_observation_count)),
        "mean_wave_coherence": float(np.sum(energy_weight * path_wave_coherence)),
        "mean_wave_margin": float(np.sum(energy_weight * path_wave_margin)),
        "well_tracked_energy": float(
            np.sum(energy_weight[(path_observation_count >= 6) & (path_wave_coherence >= 0.35)])
        ),
    }
    return output_coordinates, output_coefficients, diagnostics


def predict_oof_parameters(
    cache: ComponentCache,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    output_path = prediction_path(config)
    if output_path.is_file():
        cached = np.load(output_path)
        return (
            cached["coordinates"].astype(np.float32),
            cached["coefficients"].astype(np.complex64),
            json.loads(str(cached["diagnostics_json"])),
        )

    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    folds = np.load(FOLD_CACHE)
    output_coordinates = np.zeros(
        (len(query_ids), config.transported_components, 3), dtype=np.float32
    )
    output_coefficients = np.zeros(
        (len(query_ids), config.transported_components, PORTS), dtype=np.complex64
    )
    diagnostic_rows: list[dict[str, float]] = []
    grid = wavevector_grid(config)
    valid = cache.point_power > TINY

    for held_fold in np.unique(fold_id):
        rows = np.flatnonzero(fold_id == held_fold)
        fit_ids = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
        for site_value in np.unique(site[query_ids[rows]]):
            selected_rows = rows[site[query_ids[rows]] == site_value]
            bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
            _, local = cKDTree(positions[bank, :2]).query(
                positions[query_ids[selected_rows], :2],
                k=min(config.donor_count, len(bank)),
                workers=-1,
            )
            donor_table = bank[local]
            for local_row, output_row in enumerate(selected_rows):
                coordinates, coefficients, diagnostics = predict_query_components(
                    positions[query_ids[output_row]],
                    bs_positions[int(site_value)],
                    donor_table[local_row],
                    positions,
                    cache,
                    config,
                    grid,
                )
                output_coordinates[output_row] = coordinates
                output_coefficients[output_row] = coefficients
                diagnostic_rows.append(diagnostics)
        print(f"path parameter transport fold={int(held_fold)} rows={len(rows)}", flush=True)

    diagnostics = {
        key: summarize(np.asarray([row[key] for row in diagnostic_rows]))
        for key in diagnostic_rows[0]
    }
    np.savez_compressed(
        output_path,
        coordinates=output_coordinates,
        coefficients=output_coefficients,
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        config_json=np.asarray(json.dumps(asdict(config))),
    )
    return output_coordinates, output_coefficients, diagnostics


@torch.inference_mode()
def synthesize_batch(
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    coordinate_tensor = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
    coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.complex64, device=device)
    atom = _component_atom(coordinate_tensor, coefficient_tensor.dtype, device)
    channel = torch.einsum("bkc,bkhvs->bchvs", coefficient_tensor, atom)
    return (
        channel.reshape(-1, MP, N_UE, MH, MV, S_SC)
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(-1, MP * MH * MV, N_UE, S_SC)
    )


def preserve_power(channel: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    source_power = channel.abs().square().sum(dim=(1, 2, 3))
    reference_power = reference.abs().square().sum(dim=(1, 2, 3))
    scale = torch.sqrt(reference_power / source_power.clamp_min(1e-30))
    return channel * scale[:, None, None, None]


def angle_delay_phase_blend(
    base: torch.Tensor,
    carrier: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    base_shaped = base.reshape(-1, MP, MH, MV, N_UE, S_SC)
    carrier_shaped = carrier.reshape_as(base_shaped)
    base_ad = torch.fft.ifft(torch.fft.fft2(base_shaped, dim=(2, 3)), dim=-1)
    carrier_ad = torch.fft.ifft(torch.fft.fft2(carrier_shaped, dim=(2, 3)), dim=-1)
    carrier_power = carrier_ad.abs().square().sum(dim=(1, 4), keepdim=True)
    scale = carrier_power.mean(dim=(2, 3, 5), keepdim=True).clamp_min(1e-30)
    reliability = carrier_power / (carrier_power + 0.05 * scale)
    weight = (alpha * reliability).clamp(0.0, 1.0)
    direction = unit_complex((1.0 - weight) * unit_complex(base_ad) + weight * unit_complex(carrier_ad))
    blended = base_ad.abs() * direction
    restored = torch.fft.ifft2(torch.fft.fft(blended, dim=-1), dim=(2, 3))
    return restored.reshape_as(base)


def variant_carriers(
    base: torch.Tensor,
    path: torch.Tensor,
) -> dict[str, torch.Tensor]:
    aligned_path = preserve_power(path, base)
    variants: dict[str, torch.Tensor] = {"identity": base, "path_power_locked": aligned_path}
    for alpha in (0.25, 0.50, 0.75):
        mixed = (1.0 - alpha) * base + alpha * aligned_path
        variants[f"complex_blend_{alpha:.2f}"] = preserve_power(mixed, base)
        variants[f"ad_phase_blend_{alpha:.2f}"] = angle_delay_phase_blend(base, aligned_path, alpha)
    variants["ad_phase_blend_1.00"] = angle_delay_phase_blend(base, aligned_path, 1.0)
    return variants


def metric_payload(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    all_values = statistics(pas, pdp, error, energy)
    fold_values = [
        {"fold": int(fold), **statistics(pas, pdp, error, energy, fold_id == fold)}
        for fold in np.unique(fold_id)
    ]
    site_values = {
        str(int(value)): statistics(pas, pdp, error, energy, site == value)
        for value in np.unique(site)
    }
    return {"all": all_values, "fold": fold_values, "site": site_values}


@torch.inference_mode()
def evaluate_transport(
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    cache: ComponentCache,
    config: Config,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    base_all = np.load(OOF_BASE, mmap_mode="r")
    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    v3_metrics = np.load(V3_METRICS)
    selected_action = v3_metrics["selected_action"].astype(np.int64)
    selected_scale = v3_metrics["selected_scale"].astype(np.float32)

    names = [
        "identity",
        "path_power_locked",
        "complex_blend_0.25",
        "complex_blend_0.50",
        "complex_blend_0.75",
        "ad_phase_blend_0.25",
        "ad_phase_blend_0.50",
        "ad_phase_blend_0.75",
        "ad_phase_blend_1.00",
    ]
    arrays = {
        name: {
            "pas": np.zeros(len(query_ids), dtype=np.float64),
            "pdp": np.zeros(len(query_ids), dtype=np.float64),
            "error": np.zeros(len(query_ids), dtype=np.float64),
            "energy": np.zeros(len(query_ids), dtype=np.float64),
        }
        for name in names
    }
    extraction_error = np.zeros(len(query_ids), dtype=np.float64)
    truth_energy = np.zeros(len(query_ids), dtype=np.float64)
    raw_path_error = np.zeros(len(query_ids), dtype=np.float64)

    output = np.lib.format.open_memmap(
        carrier_path(config),
        mode="w+",
        dtype=np.complex64,
        shape=(len(query_ids), MP * MH * MV, N_UE, S_SC),
    )
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.asarray(base_all[start:end], dtype=np.complex64), device=device
        )
        truth = np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64)
        path = synthesize_batch(coordinates[start:end], coefficients[start:end], device)
        output[start:end] = path.cpu().numpy()

        true_components = synthesize_batch(
            cache.coordinates[query_ids[start:end]],
            cache.coefficients[query_ids[start:end]],
            device,
        ).cpu().numpy()
        extraction_error[start:end] = np.sum(
            np.abs(true_components - truth) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        truth_energy[start:end] = np.sum(
            np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        raw_path_error[start:end] = np.sum(
            np.abs(path.cpu().numpy() - truth) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )

        ad, pas, pdp = target_batch(ad_cache, pas_cache, pdp_cache, start, end, device)
        variants = variant_carriers(base, path)
        for name, carrier in variants.items():
            projected = torch.empty_like(carrier)
            action_rows = selected_action[start:end]
            for action_index in np.unique(action_rows):
                rows = np.flatnonzero(action_rows == action_index)
                row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
                action = ACTIONS[int(action_index)]
                projected[row_tensor] = project_channel(
                    carrier[row_tensor],
                    ad[row_tensor],
                    pas[row_tensor],
                    pdp[row_tensor],
                    action.alpha_ad,
                    action.alpha_pas,
                    action.alpha_pdp,
                )
            projected *= torch.as_tensor(
                selected_scale[start:end], dtype=torch.float32, device=device
            )[:, None, None, None]
            metric = point_metric_statistics(projected.cpu().numpy(), truth)
            arrays[name]["pas"][start:end] = metric.pas
            arrays[name]["pdp"][start:end] = metric.pdp
            arrays[name]["error"][start:end] = metric.squared_error
            arrays[name]["energy"][start:end] = metric.truth_energy
        print(f"path transport projection {end}/{len(query_ids)}", flush=True)
    output.flush()

    results = {
        name: metric_payload(
            values["pas"],
            values["pdp"],
            values["error"],
            values["energy"],
            fold_id,
            query_site,
        )
        for name, values in arrays.items()
    }
    reference = float(results["identity"]["all"]["score"])
    for name in results:
        results[name]["delta_vs_v3"] = float(results[name]["all"]["score"] - reference)
        identity_folds = results["identity"]["fold"]
        results[name]["fold_delta_vs_v3"] = [
            float(current["score"] - baseline["score"])
            for current, baseline in zip(results[name]["fold"], identity_folds)
        ]
    ranking = sorted(results, key=lambda name: float(results[name]["all"]["score"]), reverse=True)
    return {
        "extraction_nmse": float(np.sum(extraction_error) / np.sum(truth_energy)),
        "raw_path_nmse": float(np.sum(raw_path_error) / np.sum(truth_energy)),
        "variants": results,
        "ranking": ranking,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--skip-pair-diagnostics", action="store_true")
    parser.add_argument("--components", type=int, default=24)
    parser.add_argument("--transported-components", type=int, default=16)
    parser.add_argument("--donor-count", type=int, default=40)
    parser.add_argument("--direction-bins", type=int, default=180)
    parser.add_argument("--radius-fractions", default="0.55,0.75,0.90,1.00")
    parser.add_argument("--refinement-sweeps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(
        components=args.components,
        transported_components=args.transported_components,
        donor_count=args.donor_count,
        direction_bins=args.direction_bins,
        radius_fractions=tuple(float(value) for value in args.radius_fractions.split(",")),
        refinement_sweeps=args.refinement_sweeps,
    )
    device = torch.device(args.device)
    raw_cache = extract_components(config, device, args.batch_size)
    cache = refine_components(raw_cache, config, device, args.batch_size)
    _, positions, _, site, _ = load_positions()
    setup, _, _, _, _ = load_positions()

    payload: dict[str, object] = {
        "experiment": "Round2-E7-path-component-transport",
        "status": "analysis_in_progress" if args.extract_only else "analysis_only",
        "primary_hypothesis": (
            "A small number of off-grid delay-angle atoms can be matched across nearby official "
            "train positions; estimating a per-path 2-D wavevector then transports the rapid "
            "carrier phase more reliably than regressing a mixed point-level complex factor."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "component_extraction": {
            "explained_fraction": summarize(1.0 - cache.residual_fraction[cache.point_power > TINY]),
            "valid_points": int(np.sum(cache.point_power > TINY)),
            "cache": str(component_cache_path(config).relative_to(PATHS.root)),
            "refined_cache": str(refined_component_cache_path(config).relative_to(PATHS.root)),
        },
    }
    if not args.skip_pair_diagnostics:
        payload["pair_trackability"] = pair_trackability(cache, positions, site, config)
    if not args.extract_only:
        predicted_coordinates, predicted_coefficients, transport_diagnostics = predict_oof_parameters(
            cache,
            positions,
            site,
            setup.bs_positions,
            config,
        )
        payload["transport_diagnostics"] = transport_diagnostics
        payload["evaluation"] = evaluate_transport(
            predicted_coordinates,
            predicted_coefficients,
            cache,
            config,
            device,
            args.batch_size,
        )
        payload["status"] = "analysis_only_complete"
    payload["seconds"] = float(time.time() - started)
    selected_report = report_path(config)
    selected_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
