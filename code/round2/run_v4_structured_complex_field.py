#!/usr/bin/env python3
"""Learn a fold-isolated structured complex carrier correction for Round 2 V3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from complex_baselines import BaselineConfig, predict_one
from data import load_positions
from evaluate_v3_anchored_projection import target_batch
from official_metrics import PointMetricStatistics, aggregate_statistics, point_metric_statistics
from pre_sionna_spectral import project_channel
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, OOF_BASE, OOF_META, optimal_scale, scaled_error
from run_v3_anchored_neural_rbf import (
    AnchoredWeightCorrector,
    Config as V3Config,
    DESCRIPTORS,
    GEOMETRY,
    build_train_contexts,
)


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_structured_complex_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_structured_complex_field"
REPORT = ANALYSIS_DIR / "structured_complex_field.json"

FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
FOUNDATION_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_foundation_benchmark"
    / "round2_foundation_benchmark.json"
)
POINT_POWER = (
    PATHS.artifacts / "round2" / "foundation" / "round2_train_point_power.npy"
)
V3_DIR = PATHS.artifacts / "round2" / "v3_anchored_neural_rbf"
V3_METRICS = V3_DIR / "oof_projection_metrics.npz"
V3_WEIGHTS = V3_DIR / "oof_epoch6_corrected_weights.npz"
V3_AD = V3_DIR / "oof_ad_pas_f16.npy"
V3_PAS = V3_DIR / "oof_pas_neural_f16.npy"
V3_PDP = V3_DIR / "oof_pdp_neural_f16.npy"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
TINY = 1e-20


@dataclass(frozen=True)
class TrainConfig:
    samples_per_site: int = 600
    hidden: int = 256
    epochs: int = 160
    batch_size: int = 128
    learning_rate: float = 8e-4
    weight_decay: float = 2e-5
    dropout: float = 0.05
    max_log_amplitude: float = 2.5
    seed: int = 20_260_805


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_carrier_configs() -> dict[int, BaselineConfig]:
    return {
        0: BaselineConfig(
            name="affine_k24_steer_axis90_v8pi",
            method="radial_affine",
            k=24,
            ridge=0.30,
            steering_alpha=1.0,
            horizontal_axis_deg=90.0,
            vertical_scale=8.0 * np.pi,
        ),
        1: BaselineConfig(
            name="radial_idw_k8",
            method="radial_idw",
            k=8,
        ),
    }


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def select_pseudo_queries(
    fit_ids: np.ndarray,
    valid: np.ndarray,
    site: np.ndarray,
    point_power: np.ndarray,
    count_per_site: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    fractions = np.asarray((0.15, 0.20, 0.25, 0.40), dtype=np.float64)
    for site_value in np.unique(site[fit_ids]):
        bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
        log_power = np.log10(np.maximum(point_power[bank], 1e-30))
        edges = np.quantile(log_power, (0.25, 0.50, 0.75))
        bins = np.digitize(log_power, edges, right=False)
        target = np.floor(fractions * count_per_site).astype(int)
        target[-1] += count_per_site - int(np.sum(target))
        site_rows: list[np.ndarray] = []
        unused = np.ones(len(bank), dtype=bool)
        for bin_value, requested in enumerate(target):
            candidates = np.flatnonzero(bins == bin_value)
            take = min(requested, len(candidates))
            choice = rng.choice(candidates, size=take, replace=False)
            site_rows.append(bank[choice])
            unused[choice] = False
        current = sum(len(rows) for rows in site_rows)
        if current < count_per_site:
            candidates = np.flatnonzero(unused)
            choice = rng.choice(
                candidates,
                size=min(count_per_site - current, len(candidates)),
                replace=False,
            )
            site_rows.append(bank[choice])
        selected.append(np.concatenate(site_rows))
    output = np.concatenate(selected).astype(np.int64)
    rng.shuffle(output)
    return output


def context_features(
    distances: np.ndarray,
    base_weights: np.ndarray,
    pas_weights: np.ndarray,
    pdp_weights: np.ndarray,
    width: int = 16,
) -> np.ndarray:
    distance = np.asarray(distances[:, :width], dtype=np.float32)
    base = np.asarray(base_weights[:, :width], dtype=np.float32)
    pas = np.asarray(pas_weights[:, :width], dtype=np.float32)
    pdp = np.asarray(pdp_weights[:, :width], dtype=np.float32)
    summaries = np.column_stack(
        [
            distance[:, 0] / 20.0,
            np.mean(distance, axis=1) / 20.0,
            np.std(distance, axis=1) / 20.0,
            np.sum(np.abs(base), axis=1),
            np.sum(np.abs(pas), axis=1),
            np.sum(np.abs(pdp), axis=1),
            np.max(np.abs(pas - base), axis=1),
            np.max(np.abs(pdp - base), axis=1),
        ]
    ).astype(np.float32)
    return np.concatenate(
        [
            distance / 20.0,
            np.log1p(distance),
            base,
            pas,
            pdp,
            summaries,
        ],
        axis=1,
    ).astype(np.float32)


def _normalized_power(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / value.sum(dim=dim, keepdim=True).clamp_min(TINY)


def _adjacent_coherence(
    channel: torch.Tensor,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = channel.narrow(axis, 0, channel.shape[axis] - 1)
    right = channel.narrow(axis, 1, channel.shape[axis] - 1)
    group_axes = (2, 3, 5)
    numerator = torch.sum(torch.conj(left) * right, dim=group_axes)
    denominator = torch.sqrt(
        torch.sum(left.abs().square(), dim=group_axes)
        * torch.sum(right.abs().square(), dim=group_axes)
    ).clamp_min(TINY)
    grouped = numerator / denominator
    global_numerator = numerator.sum(dim=(1, 2))
    global_denominator = torch.sqrt(
        torch.sum(left.abs().square(), dim=(1, 2, 3, 4, 5))
        * torch.sum(right.abs().square(), dim=(1, 2, 3, 4, 5))
    ).clamp_min(TINY)
    return grouped, global_numerator / global_denominator


@torch.no_grad()
def channel_features(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(-1, MP, MH, MV, N_UE, S_SC)
    total_power = shaped.abs().square().sum(dim=(1, 2, 3, 4, 5))
    group_power = shaped.abs().square().sum(dim=(2, 3, 5))
    group_fraction = group_power / total_power[:, None, None].clamp_min(TINY)
    features = [
        torch.log10(total_power.clamp_min(1e-30))[:, None],
        torch.log((group_fraction * (MP * N_UE)).clamp_min(1e-8)).reshape(
            len(shaped), -1
        ),
    ]
    for axis in (2, 3, 5):
        grouped, global_value = _adjacent_coherence(shaped, axis)
        features.extend(
            [
                global_value.real[:, None],
                global_value.imag[:, None],
                grouped.real.reshape(len(shaped), -1),
                grouped.imag.reshape(len(shaped), -1),
            ]
        )

    angle = torch.fft.fft2(shaped, dim=(2, 3)).abs().square()
    horizontal = _normalized_power(angle.sum(dim=(1, 3, 4, 5)), 1)
    vertical = _normalized_power(angle.sum(dim=(1, 2, 4, 5)), 1)
    delay = torch.fft.ifft(shaped, dim=5).abs().square().sum(dim=(1, 2, 3, 4))
    delay = delay.reshape(len(shaped), 24, 8).sum(dim=2)
    delay = _normalized_power(delay, 1)
    features.extend([horizontal, vertical, delay])
    return torch.cat(features, dim=1).float()


def apply_axis_ramp(
    channel: torch.Tensor,
    slope: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    index = torch.arange(
        channel.shape[axis],
        dtype=channel.real.dtype,
        device=channel.device,
    )
    ramp = torch.exp(1j * slope[:, None] * index[None, :])
    shape = [len(channel)] + [1] * (channel.ndim - 1)
    shape[axis] = channel.shape[axis]
    return channel * ramp.reshape(shape)


def best_axis_ramp(
    source: torch.Tensor,
    truth: torch.Tensor,
    axis: int,
    nfft: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_axes = tuple(index for index in range(1, source.ndim) if index != axis)
    cross = torch.sum(torch.conj(source) * truth, dim=reduce_axes)
    spectrum = torch.fft.fft(cross, n=nfft, dim=1)
    best = torch.argmax(spectrum.abs(), dim=1)
    slope = 2.0 * math.pi * best.to(source.real.dtype) / float(nfft)
    slope = torch.remainder(slope + math.pi, 2.0 * math.pi) - math.pi
    return apply_axis_ramp(source, slope, axis), slope


@torch.no_grad()
def structured_oracle(
    source: torch.Tensor,
    truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    shaped_source = source.reshape(-1, MP, MH, MV, N_UE, S_SC)
    shaped_truth = truth.reshape_as(shaped_source)
    output = shaped_source.clone()
    slopes = {
        2: torch.zeros(len(source), device=source.device),
        3: torch.zeros(len(source), device=source.device),
        5: torch.zeros(len(source), device=source.device),
    }
    for _ in range(2):
        for axis, nfft in ((5, 3072), (2, 512), (3, 256)):
            output, delta = best_axis_ramp(output, shaped_truth, axis, nfft)
            slopes[axis] = torch.remainder(
                slopes[axis] + delta + math.pi,
                2.0 * math.pi,
            ) - math.pi

    numerator = torch.sum(
        torch.conj(output) * shaped_truth,
        dim=(2, 3, 5),
    )
    denominator = torch.sum(output.abs().square(), dim=(2, 3, 5)).clamp_min(TINY)
    factor = numerator / denominator
    corrected = output * factor[:, :, None, None, :, None]
    source_power = shaped_source.abs().square().sum(dim=(1, 2, 3, 4, 5))
    corrected_power = corrected.abs().square().sum(dim=(1, 2, 3, 4, 5))
    common_scale = torch.sqrt(
        source_power / corrected_power.clamp_min(TINY)
    )
    corrected = corrected * common_scale[:, None, None, None, None, None]
    factor = factor * common_scale[:, None, None]

    baseline_error = (shaped_source - shaped_truth).abs().square().sum(
        dim=(1, 2, 3, 4, 5)
    )
    oracle_error = (corrected - shaped_truth).abs().square().sum(
        dim=(1, 2, 3, 4, 5)
    )
    truth_power = shaped_truth.abs().square().sum(dim=(1, 2, 3, 4, 5))
    slope_values = torch.stack([slopes[2], slopes[3], slopes[5]], dim=1)
    log_amplitude = torch.log(factor.abs().clamp_min(1e-8)).reshape(len(source), -1)
    phase = torch.angle(factor).reshape(len(source), -1)
    return slope_values, log_amplitude, phase, {
        "baseline_error": baseline_error,
        "oracle_error": oracle_error,
        "truth_power": truth_power,
    }


class StructuredComplexField(nn.Module):
    def __init__(self, input_size: int, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(input_size, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Dropout(config.dropout),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden, config.hidden),
                    nn.SiLU(),
                    nn.Linear(config.hidden, 19),
                )
                for _ in range(2)
            ]
        )
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        site: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.trunk(features)
        raw = torch.zeros(
            (len(features), 19),
            dtype=features.dtype,
            device=features.device,
        )
        for site_value, head in enumerate(self.heads):
            mask = site == site_value
            if torch.any(mask):
                raw[mask] = head(latent[mask])
        slope_scale = torch.as_tensor(
            (math.pi, math.pi, 0.35),
            dtype=features.dtype,
            device=features.device,
        )
        slopes = torch.tanh(raw[:, :3]) * slope_scale[None, :]
        log_amplitude = torch.tanh(raw[:, 3:11]) * self.config.max_log_amplitude
        phase = torch.tanh(raw[:, 11:19]) * math.pi
        return slopes, log_amplitude, phase


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / weight.sum().clamp_min(TINY)


def parameter_loss(
    prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    slopes, log_amplitude, phase = prediction
    target_slopes, target_log_amplitude, target_phase = target
    slope_scale = torch.as_tensor(
        (math.pi, math.pi, 0.35),
        dtype=slopes.dtype,
        device=slopes.device,
    )
    slope_error = torch.nn.functional.smooth_l1_loss(
        slopes / slope_scale[None, :],
        target_slopes / slope_scale[None, :],
        reduction="none",
        beta=0.15,
    ).mean(dim=1)
    amplitude_error = torch.nn.functional.smooth_l1_loss(
        log_amplitude,
        target_log_amplitude,
        reduction="none",
        beta=0.25,
    ).mean(dim=1)
    phase_error = (1.0 - torch.cos(phase - target_phase)).mean(dim=1)
    regularization = (
        (slopes / slope_scale[None, :]).square().mean(dim=1)
        + 0.25 * log_amplitude.square().mean(dim=1)
        + 0.10 * phase.square().mean(dim=1)
    )
    total = weighted_mean(
        1.5 * slope_error + 0.35 * amplitude_error + phase_error
        + 0.002 * regularization,
        weight,
    )
    return total, {
        "slope": float(weighted_mean(slope_error, weight).detach().cpu()),
        "amplitude": float(weighted_mean(amplitude_error, weight).detach().cpu()),
        "phase": float(weighted_mean(phase_error, weight).detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def pseudo_dataset_path(fold: int, config: TrainConfig) -> Path:
    return ARTIFACT_DIR / f"fold{fold}_pseudo_n{config.samples_per_site}_per_site.npz"


def load_v3_model(
    fold: int,
    device: torch.device,
) -> tuple[AnchoredWeightCorrector, np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        V3_DIR / f"fold{fold}_final.pt",
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint["epoch"]) != 6:
        raise RuntimeError(f"fold {fold} V3 checkpoint is not epoch 6")
    config = V3Config(**checkpoint["config"])
    geometry_raw = np.load(GEOMETRY)["train"].astype(np.float32)[:, 3:]
    geometry = (
        (geometry_raw - checkpoint["feature_mean"][None, :])
        / checkpoint["feature_std"][None, :]
    ).astype(np.float32)
    model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, geometry, np.asarray(config.guard_radii_m, dtype=np.float32)


@torch.no_grad()
def pseudo_v3_weights(
    fold: int,
    query_ids: np.ndarray,
    guard_index: np.ndarray,
    fit_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        V3_DIR / f"fold{fold}_final.pt",
        map_location=device,
        weights_only=False,
    )
    config = V3Config(**checkpoint["config"])
    table = build_train_contexts(fold, positions, site, valid, fit_ids, config)
    row_map = np.full(len(positions), -1, dtype=np.int64)
    row_map[table.query_ids] = np.arange(len(table.query_ids), dtype=np.int64)
    rows = row_map[query_ids]
    if np.any(rows < 0):
        raise RuntimeError("pseudo query missing from V3 context table")
    donor_ids = table.donor_ids[rows, guard_index].astype(np.int64)
    base_weights = table.weights[rows, guard_index].astype(np.float32)
    distances = table.distances[rows, guard_index].astype(np.float32)

    descriptors = np.load(DESCRIPTORS)
    spectra_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    geometry_raw = np.load(GEOMETRY)["train"].astype(np.float32)[:, 3:]
    geometry_numpy = (
        (geometry_raw - checkpoint["feature_mean"][None, :])
        / checkpoint["feature_std"][None, :]
    ).astype(np.float32)
    geometry = torch.as_tensor(geometry_numpy, dtype=torch.float32, device=device)
    positions_tensor = torch.as_tensor(
        positions[:, :2], dtype=torch.float32, device=device
    )
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    pas_weights = np.empty_like(base_weights)
    pdp_weights = np.empty_like(base_weights)
    for start in range(0, len(query_ids), 128):
        end = min(start + 128, len(query_ids))
        local_query = torch.as_tensor(
            query_ids[start:end], dtype=torch.long, device=device
        )
        local_donor = torch.as_tensor(
            donor_ids[start:end], dtype=torch.long, device=device
        )
        local_distance = torch.as_tensor(
            distances[start:end], dtype=torch.float32, device=device
        )
        local_base = torch.as_tensor(
            base_weights[start:end], dtype=torch.float32, device=device
        )
        _, _, diagnostics = model(
            geometry[local_query],
            positions_tensor[local_query],
            geometry[local_donor],
            positions_tensor[local_donor],
            spectra[local_donor],
            local_distance,
            local_base,
        )
        pas_weights[start:end] = diagnostics["pas_weights"].cpu().numpy()
        pdp_weights[start:end] = diagnostics["pdp_weights"].cpu().numpy()
    return donor_ids, distances, base_weights, pas_weights, pdp_weights


def build_pseudo_dataset(
    fold: int,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, object]:
    path = pseudo_dataset_path(fold, config)
    if path.is_file():
        cached = np.load(path)
        if int(cached["samples_per_site"]) == config.samples_per_site:
            return {
                "path": PATHS.relative(path),
                "count": int(len(cached["query_ids"])),
                "reused": True,
                "baseline_nmse": float(
                    np.sum(cached["baseline_error"], dtype=np.float64)
                    / np.sum(cached["truth_power"], dtype=np.float64)
                ),
                "oracle_nmse": float(
                    np.sum(cached["oracle_error"], dtype=np.float64)
                    / np.sum(cached["truth_power"], dtype=np.float64)
                ),
            }

    setup, positions, _, site, _ = load_positions()
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = np.load(POINT_POWER).astype(np.float64)
    geometry = np.load(GEOMETRY)["train"].astype(np.float32)
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    folds = np.load(FOLD_CACHE)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    query_ids = select_pseudo_queries(
        fit_ids,
        valid,
        site,
        point_power,
        config.samples_per_site,
        config.seed + fold * 1009,
    )
    rng = np.random.default_rng(config.seed + fold * 1009)
    guard_probability = np.asarray((0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05))
    guard_index = rng.choice(
        np.arange(len(guard_probability)),
        size=len(query_ids),
        p=guard_probability,
    ).astype(np.int8)
    donor_ids, distances, base_weights, pas_weights, pdp_weights = pseudo_v3_weights(
        fold,
        query_ids,
        guard_index,
        fit_ids,
        positions,
        site,
        valid,
        device,
    )

    foundation = json.loads(FOUNDATION_REPORT.read_text(encoding="utf-8"))
    carrier = foundation["carrier_fit_only_by_fold"][str(fold)]
    configs = current_carrier_configs()
    static_context = context_features(
        distances, base_weights, pas_weights, pdp_weights
    )
    channel_feature_parts: list[np.ndarray] = []
    slope_parts: list[np.ndarray] = []
    amplitude_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    baseline_error_parts: list[np.ndarray] = []
    oracle_error_parts: list[np.ndarray] = []
    truth_power_parts: list[np.ndarray] = []
    source_batch: list[np.ndarray] = []
    truth_batch: list[np.ndarray] = []

    def flush_batch() -> None:
        if not source_batch:
            return
        source = torch.as_tensor(np.stack(source_batch), device=device)
        truth = torch.as_tensor(np.stack(truth_batch), device=device)
        channel_feature_parts.append(channel_features(source).cpu().numpy())
        slopes, log_amplitude, phase, diagnostic = structured_oracle(source, truth)
        slope_parts.append(slopes.cpu().numpy())
        amplitude_parts.append(log_amplitude.cpu().numpy())
        phase_parts.append(phase.cpu().numpy())
        baseline_error_parts.append(diagnostic["baseline_error"].cpu().numpy())
        oracle_error_parts.append(diagnostic["oracle_error"].cpu().numpy())
        truth_power_parts.append(diagnostic["truth_power"].cpu().numpy())
        source_batch.clear()
        truth_batch.clear()

    for row, query_id in enumerate(query_ids):
        site_value = int(site[query_id])
        template = configs[site_value]
        local_carrier = carrier[str(site_value)]
        local_config = replace(
            template,
            beta=float(local_carrier["beta"]),
            gamma=float(local_carrier["gamma"]),
        )
        ids = donor_ids[row, : local_config.k]
        source, _ = predict_one(
            np.asarray(train_channel[ids], dtype=np.complex64),
            positions[ids],
            positions[query_id],
            distances[row, : local_config.k],
            setup.bs_positions[site_value],
            local_config,
        )
        source_batch.append(source)
        truth_batch.append(np.asarray(train_channel[query_id], dtype=np.complex64))
        if len(source_batch) == 8:
            flush_batch()
        if (row + 1) % 100 == 0 or row + 1 == len(query_ids):
            print(
                f"structured pseudo fold={fold} {row + 1}/{len(query_ids)}",
                flush=True,
            )
    flush_batch()

    features = np.concatenate(
        [
            geometry[query_ids],
            np.concatenate(channel_feature_parts),
            static_context,
        ],
        axis=1,
    ).astype(np.float32)
    slopes = np.concatenate(slope_parts).astype(np.float32)
    log_amplitude = np.clip(
        np.concatenate(amplitude_parts),
        -config.max_log_amplitude,
        config.max_log_amplitude,
    ).astype(np.float32)
    phase = np.concatenate(phase_parts).astype(np.float32)
    baseline_error = np.concatenate(baseline_error_parts).astype(np.float64)
    oracle_error = np.concatenate(oracle_error_parts).astype(np.float64)
    truth_power = np.concatenate(truth_power_parts).astype(np.float64)
    np.savez_compressed(
        path,
        query_ids=query_ids.astype(np.int32),
        site=site[query_ids].astype(np.int8),
        guard_index=guard_index,
        features=features,
        slopes=slopes,
        log_amplitude=log_amplitude,
        phase=phase,
        baseline_error=baseline_error,
        oracle_error=oracle_error,
        truth_power=truth_power,
        samples_per_site=np.int32(config.samples_per_site),
    )
    return {
        "path": PATHS.relative(path),
        "count": int(len(query_ids)),
        "reused": False,
        "feature_size": int(features.shape[1]),
        "baseline_nmse": float(np.sum(baseline_error) / np.sum(truth_power)),
        "oracle_nmse": float(np.sum(oracle_error) / np.sum(truth_power)),
        "oracle_improvement_fraction": float(
            1.0 - np.sum(oracle_error) / np.sum(baseline_error)
        ),
    }


@torch.no_grad()
def evaluate_parameter_model(
    model: StructuredComplexField,
    features: torch.Tensor,
    site: torch.Tensor,
    target: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    rows: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    prediction = model(features[rows], site[rows])
    _, values = parameter_loss(
        prediction,
        tuple(value[rows] for value in target),
        weight[rows],
    )
    return values


def train_parameter_model(
    fold: int,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, object]:
    set_seed(config.seed + fold * 1009)
    dataset = np.load(pseudo_dataset_path(fold, config))
    query_ids = dataset["query_ids"].astype(np.int64)
    site_numpy = dataset["site"].astype(np.int64)
    features_numpy = dataset["features"].astype(np.float32)
    slopes_numpy = dataset["slopes"].astype(np.float32)
    log_amplitude_numpy = dataset["log_amplitude"].astype(np.float32)
    phase_numpy = dataset["phase"].astype(np.float32)
    power_numpy = dataset["truth_power"].astype(np.float64)

    validation_mask = ((query_ids * 2_654_435_761 + fold * 104_729) % 7) == 0
    if np.sum(validation_mask) < 64:
        validation_mask[np.arange(0, len(query_ids), 7)] = True
    train_mask = ~validation_mask
    mean = np.mean(features_numpy[train_mask], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(features_numpy[train_mask], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    normalized = np.clip((features_numpy - mean) / std, -8.0, 8.0).astype(np.float32)
    median_power = float(np.median(power_numpy[train_mask]))
    weight_numpy = np.sqrt(power_numpy / max(median_power, 1e-30))
    weight_numpy = np.clip(weight_numpy, 0.20, 20.0)
    weight_numpy /= np.mean(weight_numpy[train_mask])

    features = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    site = torch.as_tensor(site_numpy, dtype=torch.long, device=device)
    target = (
        torch.as_tensor(slopes_numpy, dtype=torch.float32, device=device),
        torch.as_tensor(log_amplitude_numpy, dtype=torch.float32, device=device),
        torch.as_tensor(phase_numpy, dtype=torch.float32, device=device),
    )
    weight = torch.as_tensor(weight_numpy, dtype=torch.float32, device=device)
    train_rows = np.flatnonzero(train_mask)
    validation_rows = torch.as_tensor(
        np.flatnonzero(validation_mask), dtype=torch.long, device=device
    )
    model = StructuredComplexField(features.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed + fold * 1009)
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        train_losses = []
        for start in range(0, len(order), config.batch_size):
            rows = torch.as_tensor(
                order[start : start + config.batch_size],
                dtype=torch.long,
                device=device,
            )
            prediction = model(features[rows], site[rows])
            loss, _ = parameter_loss(
                prediction,
                tuple(value[rows] for value in target),
                weight[rows],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation = evaluate_parameter_model(
            model,
            features,
            site,
            target,
            weight,
            validation_rows,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation": validation,
        }
        history.append(row)
        if validation["total"] < best_loss:
            best_loss = validation["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"structured model fold={fold} epoch={epoch}/{config.epochs} "
                f"train={row['train_loss']:.6f} valid={validation['total']:.6f} "
                f"best={best_loss:.6f}@{best_epoch}",
                flush=True,
            )
    checkpoint = ARTIFACT_DIR / f"fold{fold}_structured_complex.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": asdict(config),
            "feature_mean": mean,
            "feature_std": std,
            "feature_size": int(features.shape[1]),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
        checkpoint,
    )
    return {
        "fold": fold,
        "train_count": int(np.sum(train_mask)),
        "validation_count": int(np.sum(validation_mask)),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "checkpoint": PATHS.relative(checkpoint),
        "history": history,
    }


@torch.no_grad()
def build_oof_features(device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    _, positions, _, _, _ = load_positions()
    geometry = np.load(GEOMETRY)["train"].astype(np.float32)[query_ids]
    weights = np.load(V3_WEIGHTS)
    if not np.array_equal(weights["query_ids"], query_ids):
        raise RuntimeError("V3 corrected weight/query order mismatch")
    donor_ids = weights["donor_ids"].astype(np.int64)
    distances = np.linalg.norm(
        positions[donor_ids, :2] - positions[query_ids, None, :2],
        axis=2,
    ).astype(np.float32)
    static_context = context_features(
        distances,
        weights["base_weights"],
        weights["pas_weights"],
        weights["pdp_weights"],
    )
    base = np.load(OOF_BASE, mmap_mode="r")
    channel_parts = []
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        channel = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        channel_parts.append(channel_features(channel).cpu().numpy())
        print(f"structured OOF features {end}/{len(query_ids)}", flush=True)
    features = np.concatenate(
        [geometry, np.concatenate(channel_parts), static_context],
        axis=1,
    ).astype(np.float32)
    return query_ids, features


@torch.no_grad()
def predict_oof_parameters(
    config: TrainConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_ids, features = build_oof_features(device, batch_size)
    metadata = np.load(OOF_META)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site_numpy = train_site[query_ids].astype(np.int64)
    slopes = np.zeros((len(query_ids), 3), dtype=np.float32)
    log_amplitude = np.zeros((len(query_ids), 8), dtype=np.float32)
    phase = np.zeros((len(query_ids), 8), dtype=np.float32)
    for fold in np.unique(fold_id):
        rows = np.flatnonzero(fold_id == fold)
        checkpoint = torch.load(
            ARTIFACT_DIR / f"fold{int(fold)}_structured_complex.pt",
            map_location=device,
            weights_only=False,
        )
        normalized = np.clip(
            (features[rows] - checkpoint["feature_mean"][None, :])
            / checkpoint["feature_std"][None, :],
            -8.0,
            8.0,
        ).astype(np.float32)
        model = StructuredComplexField(
            int(checkpoint["feature_size"]),
            TrainConfig(**checkpoint["config"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            local = rows[start:end]
            values = model(
                torch.as_tensor(normalized[start:end], device=device),
                torch.as_tensor(site_numpy[local], dtype=torch.long, device=device),
            )
            slopes[local] = values[0].cpu().numpy()
            log_amplitude[local] = values[1].cpu().numpy()
            phase[local] = values[2].cpu().numpy()
    np.savez_compressed(
        ARTIFACT_DIR / "oof_structured_parameters.npz",
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=site_numpy.astype(np.int8),
        slopes=slopes,
        log_amplitude=log_amplitude,
        phase=phase,
    )
    return slopes, log_amplitude, phase


def apply_structured_correction(
    source: torch.Tensor,
    slopes: torch.Tensor,
    log_amplitude: torch.Tensor,
    phase: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if strength <= 0.0:
        return source
    shaped = source.reshape(-1, MP, MH, MV, N_UE, S_SC)
    source_power = shaped.abs().square().sum(dim=(1, 2, 3, 4, 5))
    output = apply_axis_ramp(shaped, strength * slopes[:, 0], 2)
    output = apply_axis_ramp(output, strength * slopes[:, 1], 3)
    output = apply_axis_ramp(output, strength * slopes[:, 2], 5)
    factor = torch.exp(
        strength * log_amplitude + 1j * strength * phase
    ).reshape(-1, MP, N_UE)
    output = output * factor[:, :, None, None, :, None]
    output_power = output.abs().square().sum(dim=(1, 2, 3, 4, 5))
    scale = torch.sqrt(source_power / output_power.clamp_min(TINY))
    return (output * scale[:, None, None, None, None, None]).reshape_as(source)


def aggregate_arrays(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    truth_energy: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    return aggregate_statistics(
        PointMetricStatistics(pas, pdp, error, truth_energy),
        mask,
    )


def global_relative_phase_by_fold(
    config: TrainConfig,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = np.zeros((len(fold_id), MP * N_UE), dtype=np.float32)
    records: list[dict[str, object]] = []
    for fold in np.unique(fold_id):
        dataset = np.load(pseudo_dataset_path(int(fold), config))
        dataset_site = dataset["site"].astype(np.int8)
        phase = dataset["phase"].astype(np.float64)
        relative = np.angle(np.exp(1j * (phase - phase[:, :1])))
        power = dataset["truth_power"].astype(np.float64)
        for site_value in np.unique(site):
            sample_mask = dataset_site == site_value
            cap = float(np.quantile(power[sample_mask], 0.99))
            weight = np.minimum(power[sample_mask], cap)
            circular = np.sum(
                weight[:, None] * np.exp(1j * relative[sample_mask]),
                axis=0,
            )
            value = np.angle(circular).astype(np.float32)
            held = (fold_id == fold) & (site == site_value)
            output[held] = value
            records.append(
                {
                    "held_fold": int(fold),
                    "site": int(site_value),
                    "sample_count": int(np.sum(sample_mask)),
                    "relative_phase": value.tolist(),
                }
            )
    return output, records


def evaluate_oof(
    config: TrainConfig,
    device: torch.device,
    batch_size: int,
    strengths: tuple[float, ...],
) -> dict[str, object]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids]
    v3_metrics = np.load(V3_METRICS)
    selected_action = v3_metrics["selected_action"].astype(np.int16)
    fixed_v3_scale = v3_metrics["selected_scale"].astype(np.float32)
    slopes, log_amplitude, phase = predict_oof_parameters(
        config, device, batch_size
    )
    global_relative_phase, global_phase_records = global_relative_phase_by_fold(
        config, fold_id, site
    )
    base = np.load(OOF_BASE, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    ad_cache = np.load(V3_AD, mmap_mode="r")
    pas_cache = np.load(V3_PAS, mmap_mode="r")
    pdp_cache = np.load(V3_PDP, mmap_mode="r")
    count = len(query_ids)
    variants = {
        "ramps_only": (True, False, False),
        "amplitude_only": (False, True, False),
        "polue_phase_only": (False, False, True),
        "ramps_amplitude": (True, True, False),
        "full": (True, True, True),
        "relative_polue_phase_site1": (False, False, True),
        "centered_polue_phase_site1": (False, False, True),
        "common_phase_site1": (False, False, True),
        "global_relative_phase_site1": (False, False, True),
    }
    run_specs: list[tuple[str, float, tuple[bool, bool, bool]]] = [
        ("identity", 0.0, (False, False, False))
    ]
    run_specs.extend(
        (variant, strength, enabled)
        for variant, enabled in variants.items()
        for strength in strengths
        if strength > 0.0
    )
    stores = {
        f"{variant}_a{strength:.2f}": {
            "variant": variant,
            "strength": strength,
            "pas": np.empty(count, dtype=np.float64),
            "pdp": np.empty(count, dtype=np.float64),
            "predicted_energy": np.empty(count, dtype=np.float64),
            "truth_energy": np.empty(count, dtype=np.float64),
            "cross": np.empty(count, dtype=np.float64),
        }
        for variant, strength, _ in run_specs
    }
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        ad, pas, pdp = target_batch(
            ad_cache, pas_cache, pdp_cache, start, end, device
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        truth = np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64)
        local_slopes = torch.as_tensor(slopes[start:end], device=device)
        local_log_amplitude = torch.as_tensor(log_amplitude[start:end], device=device)
        local_phase = torch.as_tensor(phase[start:end], device=device)
        local_global_phase = torch.as_tensor(
            global_relative_phase[start:end], device=device
        )
        local_site = torch.as_tensor(site[start:end], dtype=torch.long, device=device)
        for variant, strength, enabled in run_specs:
            local_slope = local_slopes if enabled[0] else torch.zeros_like(local_slopes)
            local_amplitude = (
                local_log_amplitude
                if enabled[1]
                else torch.zeros_like(local_log_amplitude)
            )
            local_group_phase = (
                local_phase if enabled[2] else torch.zeros_like(local_phase)
            )
            if variant == "relative_polue_phase_site1":
                local_group_phase = torch.angle(
                    torch.exp(1j * (local_phase - local_phase[:, :1]))
                )
                local_group_phase = torch.where(
                    (local_site == 1)[:, None],
                    local_group_phase,
                    torch.zeros_like(local_group_phase),
                )
            elif variant == "centered_polue_phase_site1":
                center = torch.angle(torch.mean(torch.exp(1j * local_phase), dim=1))
                local_group_phase = torch.angle(
                    torch.exp(1j * (local_phase - center[:, None]))
                )
                local_group_phase = torch.where(
                    (local_site == 1)[:, None],
                    local_group_phase,
                    torch.zeros_like(local_group_phase),
                )
            elif variant == "common_phase_site1":
                center = torch.angle(torch.mean(torch.exp(1j * local_phase), dim=1))
                local_group_phase = center[:, None].expand_as(local_phase)
                local_group_phase = torch.where(
                    (local_site == 1)[:, None],
                    local_group_phase,
                    torch.zeros_like(local_group_phase),
                )
            elif variant == "global_relative_phase_site1":
                local_group_phase = torch.where(
                    (local_site == 1)[:, None],
                    local_global_phase,
                    torch.zeros_like(local_global_phase),
                )
            corrected = apply_structured_correction(
                base_tensor,
                local_slope,
                local_amplitude,
                local_group_phase,
                strength,
            )
            prediction = np.empty_like(np.asarray(base[start:end]))
            local_actions = selected_action[start:end]
            for action_index in np.unique(local_actions):
                rows = np.flatnonzero(local_actions == action_index)
                action = ACTIONS[int(action_index)]
                with torch.inference_mode():
                    projected = project_channel(
                        corrected[rows],
                        ad[rows],
                        pas[rows],
                        pdp[rows],
                        action.alpha_ad,
                        action.alpha_pas,
                        action.alpha_pdp,
                    )
                prediction[rows] = projected.cpu().numpy()
            point = point_metric_statistics(prediction, truth)
            store = stores[f"{variant}_a{strength:.2f}"]
            store["pas"][start:end] = point.pas
            store["pdp"][start:end] = point.pdp
            store["truth_energy"][start:end] = point.truth_energy
            store["predicted_energy"][start:end] = np.sum(
                np.abs(prediction) ** 2,
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            store["cross"][start:end] = np.real(
                np.sum(
                    prediction * np.conj(truth),
                    axis=(1, 2, 3),
                    dtype=np.complex128,
                )
            )
        print(f"structured OOF projection {end}/{count}", flush=True)

    results: dict[str, object] = {}
    metric_payload: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id,
        "site": site,
    }
    for key, store in stores.items():
        error = scaled_error(
            store["predicted_energy"],
            store["truth_energy"],
            store["cross"],
            fixed_v3_scale,
        )
        results[key] = {
            "variant": store["variant"],
            "strength": store["strength"],
            "overall": aggregate_arrays(
                store["pas"], store["pdp"], error, store["truth_energy"]
            ),
            "folds": [
                {
                    "fold": int(fold),
                    **aggregate_arrays(
                        store["pas"],
                        store["pdp"],
                        error,
                        store["truth_energy"],
                        fold_id == fold,
                    ),
                }
                for fold in np.unique(fold_id)
            ],
            "sites": {
                str(int(site_value)): aggregate_arrays(
                    store["pas"],
                    store["pdp"],
                    error,
                    store["truth_energy"],
                    site == site_value,
                )
                for site_value in np.unique(site)
            },
            "scale_policy": "frozen V3 nested selected_scale",
        }
        metric_payload[f"strength_{key}_pas"] = store["pas"]
        metric_payload[f"strength_{key}_pdp"] = store["pdp"]
        metric_payload[f"strength_{key}_error"] = error
        metric_payload[f"strength_{key}_truth_energy"] = store["truth_energy"]
        metric_payload[f"strength_{key}_predicted_energy"] = store["predicted_energy"]
        metric_payload[f"strength_{key}_cross"] = store["cross"]
    np.savez_compressed(
        ARTIFACT_DIR / "oof_structured_projection_metrics.npz",
        **metric_payload,
    )
    return {
        "parameter_summary": {
            "horizontal_slope": summarize(slopes[:, 0]),
            "vertical_slope": summarize(slopes[:, 1]),
            "delay_slope": summarize(slopes[:, 2]),
            "log_amplitude": summarize(log_amplitude.reshape(-1)),
            "phase": summarize(phase.reshape(-1)),
        },
        "strengths": results,
        "global_relative_phase_records": global_phase_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--samples-per-site", type=int, default=TrainConfig.samples_per_site)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--strengths", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--evaluation-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        samples_per_site=args.samples_per_site,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    device = torch.device(args.device)
    selected_folds = tuple(
        int(value) for value in args.folds.split(",") if value.strip()
    )
    strengths = tuple(float(value) for value in args.strengths.split(","))
    if args.evaluation_only:
        datasets = []
        models = []
        for fold in selected_folds:
            dataset_path = pseudo_dataset_path(fold, config)
            checkpoint_path = ARTIFACT_DIR / f"fold{fold}_structured_complex.pt"
            if not dataset_path.is_file():
                raise FileNotFoundError(dataset_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            cached = np.load(dataset_path)
            baseline_error = float(
                np.sum(cached["baseline_error"], dtype=np.float64)
            )
            oracle_error = float(np.sum(cached["oracle_error"], dtype=np.float64))
            truth_power = float(np.sum(cached["truth_power"], dtype=np.float64))
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            datasets.append(
                {
                    "path": PATHS.relative(dataset_path),
                    "count": int(len(cached["query_ids"])),
                    "feature_size": int(cached["features"].shape[1]),
                    "baseline_nmse": baseline_error / truth_power,
                    "oracle_nmse": oracle_error / truth_power,
                    "oracle_improvement_fraction": 1.0 - oracle_error / baseline_error,
                    "reused": True,
                }
            )
            models.append(
                {
                    "fold": fold,
                    "checkpoint": PATHS.relative(checkpoint_path),
                    "best_epoch": int(checkpoint["best_epoch"]),
                    "best_validation_loss": float(
                        checkpoint["best_validation_loss"]
                    ),
                    "reused": True,
                }
            )
    else:
        datasets = [build_pseudo_dataset(fold, config, device) for fold in selected_folds]
        models = [train_parameter_model(fold, config, device) for fold in selected_folds]
    if selected_folds != (0, 1, 2, 3, 4):
        evaluation: dict[str, object] = {
            "status": "skipped_partial_folds",
        }
    else:
        evaluation = evaluate_oof(config, device, args.batch_size, strengths)
    payload = {
        "experiment": "Round2-E6-V4-structured-complex-field",
        "status": "analysis_only",
        "primary_hypothesis": (
            "A fold-isolated field of H/V/delay phase ramps and polarization-by-UE "
            "complex coefficients can improve the V3 complex carrier while the frozen "
            "V3 projection preserves PAS/PDP."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "leakage_control": (
            "Each outer-fold model is trained only on pseudo holes constructed from that "
            "fold's fit IDs; no OOF label or donor channel that can contain held-fold H is "
            "used to train the corresponding model."
        ),
        "config": asdict(config),
        "pseudo_datasets": datasets,
        "models": models,
        "evaluation": evaluation,
        "v3_reference": {
            "pas": 0.5811764771365105,
            "pdp": 0.7490299601831307,
            "nmse": 0.9526311808008533,
            "score": 0.6345084718235374,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
