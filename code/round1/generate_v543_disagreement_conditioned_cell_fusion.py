#!/usr/bin/env python3
"""Train and deploy disagreement-conditioned cell-wise physical-spectrum fusion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v531_expanded_sbi_configuration_library as v531
import analyze_v541_corrected_full_library_sbi as v541
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import generate_v533_expanded_physical_sbi_action_match as v533
import generate_v541_corrected_union_stack_action_match as v541_deploy
import train_v469_neural_sionna_calibrator as v469
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 543
VERSION = "v543_disagreement_conditioned_cell_fusion"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUTPUT = CACHE_DIR / "v543_Round1_Test_Channel.npy"
TARGET = CACHE_DIR / "v543_cellwise_joint_target.npy"
ACTIONS = CACHE_DIR / "v543_cellwise_fusion_actions.npz"
OOF_CACHE = CACHE_DIR / "v543_cellwise_fusion_oof.npz"
CHECKPOINT = CHECKPOINT_DIR / "cellwise_sionna_fusion.pt"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

V531_REPORT = PATHS.version_report_dir(531) / "v531_expanded_sbi_configuration_library.json"
V541_REPORT = PATHS.version_report_dir(541) / "v541_old_corrected_union_sbi.json"
V541_ACTIONS = PATHS.cache_dir(541) / "v541_corrected_union_stack_actions.npz"
V541_OUTPUT = PATHS.cache_dir(541) / "v541_Round1_Test_Channel.npy"
V540_OUTPUT = PATHS.cache_dir(540) / "v540_Round1_Test_Channel.npy"

REGIMES = tuple(v494.REGIMES)
FLOOR = 1e-10
SEED = 54301
OLD_SOURCE_COUNT = 9
HEIGHT = 192
WIDTH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--width", type=int, default=72)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-fold", type=int, default=0)
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    right = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


def spectrum_scores(truth: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    truth_grid = np.asarray(truth, dtype=np.float32).reshape(-1, HEIGHT, WIDTH)
    pred_grid = np.asarray(prediction, dtype=np.float32).reshape(-1, HEIGHT, WIDTH)
    truth_delay = normalize_rows(truth_grid.sum(axis=2))
    pred_delay = normalize_rows(pred_grid.sum(axis=2))
    truth_angle = normalize_rows(truth_grid.sum(axis=1))
    pred_angle = normalize_rows(pred_grid.sum(axis=1))
    joint = row_cosine(truth_grid, pred_grid)
    delay = row_cosine(truth_delay, pred_delay)
    angle = row_cosine(truth_angle, pred_angle)
    weighted = 0.50 * joint + 0.25 * delay + 0.25 * angle
    return {"joint": joint, "delay": delay, "angle": angle, "weighted": weighted}


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def log_blend(anchor: np.ndarray, prediction: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    alpha_array = np.asarray(alpha, dtype=np.float32)
    if alpha_array.ndim == 1:
        alpha_array = alpha_array[:, None]
    delta = np.clip(
        np.log(np.maximum(prediction, FLOOR)) - np.log(np.maximum(anchor, FLOOR)),
        -2.0,
        2.0,
    )
    return normalize_rows(anchor * np.exp(alpha_array * delta).astype(np.float32))


def blend_local(
    baseline: np.ndarray,
    target: np.ndarray,
    quality: np.ndarray,
    regimes: np.ndarray,
    alpha: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    effective = np.asarray(
        [alpha[str(regime)] for regime in regimes], dtype=np.float32
    ) * quality
    effective = np.clip(effective, 0.0, 1.0)
    prediction = normalize_rows(
        baseline * (1.0 - effective[:, None]) + target * effective[:, None]
    )
    return prediction, effective


def prior_coefficients(
    effective: np.ndarray,
    regimes: np.ndarray,
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    source_count = len(next(iter(weights.values())))
    output = np.zeros((len(effective), source_count + 1), dtype=np.float32)
    output[:, 0] = 1.0 - effective
    for regime in REGIMES:
        local = regimes == regime
        output[local, 1:] = effective[local, None] * weights[regime][None, :]
    output /= np.maximum(output.sum(axis=1, keepdims=True), TINY)
    return output


def load_sources() -> tuple[
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    list[str],
]:
    truth, old_train, old_quality, old_names = v531.load_expanded_sources()
    corrected_truth, corrected_train, corrected_quality, corrected_names = (
        v541.load_corrected_sources()
    )
    if not np.allclose(truth, corrected_truth):
        raise ValueError("old and corrected truth arrays differ")
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    test_regimes = regime_of(geometry["test_blocked"]).astype("U8")
    old_test, old_test_quality = v533.load_test_sources(old_names)
    corrected_test, corrected_test_quality, corrected_test_names = (
        v541_deploy.load_corrected_test_sources(test_regimes)
    )
    if corrected_test_names != corrected_names:
        raise ValueError("corrected source order mismatch")
    train_names = [f"old_{name}" for name in old_names] + list(corrected_names)
    train_sources = list(old_train) + list(corrected_train)
    test_sources = list(old_test) + list(corrected_test)
    train_quality = np.maximum(old_quality, corrected_quality).astype(np.float32)
    test_quality = np.maximum(old_test_quality, corrected_test_quality).astype(np.float32)
    return (
        truth.astype(np.float32),
        train_sources,
        test_sources,
        train_quality,
        test_quality,
        train_names,
    )


def base_position_features(
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    positions = np.asarray(positions, dtype=np.float32)
    rel = positions - np.asarray([50.0, 0.0, 25.0], dtype=np.float32)[None, :]
    distance = np.linalg.norm(rel, axis=1)
    azimuth = np.arctan2(rel[:, 1], rel[:, 0])
    features = [
        positions,
        rel,
        distance[:, None],
        np.log1p(distance)[:, None],
        np.sin(azimuth)[:, None],
        np.cos(azimuth)[:, None],
        blocked[:, None],
        np.log1p(np.maximum(blocked, 0.0))[:, None],
    ]
    names = [
        "x",
        "y",
        "z",
        "rel_x",
        "rel_y",
        "rel_z",
        "distance",
        "log_distance",
        "azimuth_sin",
        "azimuth_cos",
        "blocked",
        "log_blocked",
    ]
    xy = positions[:, :2]
    for wavelength in (20.0, 40.0, 80.0, 160.0):
        phase = np.float32(2.0 * math.pi / wavelength) * xy
        features.extend([np.sin(phase), np.cos(phase)])
        names.extend(
            [
                f"sin_x_{int(wavelength)}",
                f"sin_y_{int(wavelength)}",
                f"cos_x_{int(wavelength)}",
                f"cos_y_{int(wavelength)}",
            ]
        )
    one_hot = np.stack(
        [(regimes == regime).astype(np.float32) for regime in REGIMES], axis=1
    )
    features.append(one_hot)
    names.extend([f"regime_{regime}" for regime in REGIMES])
    return np.concatenate(features, axis=1).astype(np.float32), names


def load_static_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked).astype("U8")
    test_regimes = regime_of(test_blocked).astype("U8")
    train_base, base_names = base_position_features(
        train_positions, train_blocked, train_regimes
    )
    test_base, _ = base_position_features(test_positions, test_blocked, test_regimes)

    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    train = np.concatenate(
        [train_base, geodesic["train_features"], pointcloud["train_features"]], axis=1
    ).astype(np.float32)
    test = np.concatenate(
        [test_base, geodesic["test_features"], pointcloud["test_features"]], axis=1
    ).astype(np.float32)
    names = (
        base_names
        + [f"geodesic_{value}" for value in geodesic["feature_names"].astype(str)]
        + [f"pointcloud_{value}" for value in pointcloud["feature_names"].astype(str)]
    )
    train = np.nan_to_num(train, nan=0.0, posinf=1e6, neginf=-1e6)
    test = np.nan_to_num(test, nan=0.0, posinf=1e6, neginf=-1e6)
    return train, test, names


def dynamic_features(
    bank: np.ndarray,
    anchor: np.ndarray,
    prior: np.ndarray,
    quality: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    source = np.asarray(bank[:, 1:], dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)
    log_source = np.log(np.maximum(source, FLOOR))
    entropy = -np.sum(source * log_source, axis=2) / np.float32(math.log(source.shape[2]))
    top32 = np.partition(source, -32, axis=2)[:, :, -32:].sum(axis=2)
    anchor_norm = np.sqrt(np.sum(anchor * anchor, axis=1, dtype=np.float64))
    source_norm = np.sqrt(np.sum(source * source, axis=2, dtype=np.float64))
    source_anchor = np.sum(source * anchor[:, None, :], axis=2, dtype=np.float64)
    source_anchor /= np.maximum(source_norm * anchor_norm[:, None], 1e-30)
    old_mean = normalize_rows(source[:, :OLD_SOURCE_COUNT].mean(axis=1))
    corrected_mean = normalize_rows(source[:, OLD_SOURCE_COUNT:].mean(axis=1))
    disagreement = np.std(log_source, axis=1)
    old_corrected_cos = row_cosine(old_mean, corrected_mean)
    baseline_anchor_cos = row_cosine(bank[:, 0], anchor)
    summary = np.column_stack(
        [
            quality,
            old_corrected_cos,
            baseline_anchor_cos,
            disagreement.mean(axis=1),
            np.quantile(disagreement, 0.75, axis=1),
            np.quantile(disagreement, 0.95, axis=1),
            np.max(disagreement, axis=1),
        ]
    ).astype(np.float32)
    output = np.concatenate(
        [
            prior,
            entropy.astype(np.float32),
            top32.astype(np.float32),
            source_anchor.astype(np.float32),
            summary,
        ],
        axis=1,
    ).astype(np.float32)
    names = (
        [f"prior_{index}" for index in range(prior.shape[1])]
        + [f"source_entropy_{index}" for index in range(source.shape[1])]
        + [f"source_top32_{index}" for index in range(source.shape[1])]
        + [f"source_anchor_cosine_{index}" for index in range(source.shape[1])]
        + [
            "union_quality",
            "old_corrected_cosine",
            "baseline_anchor_cosine",
            "log_source_std_mean",
            "log_source_std_p75",
            "log_source_std_p95",
            "log_source_std_max",
        ]
    )
    return output, names


def standardize_pair(
    train: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.quantile(train, 0.005, axis=0)
    upper = np.quantile(train, 0.995, axis=0)
    clipped_train = np.clip(train, lower, upper)
    clipped_query = np.clip(query, lower, upper)
    mean = clipped_train.mean(axis=0)
    std = np.maximum(clipped_train.std(axis=0), 1e-4)
    return (
        np.clip((clipped_train - mean) / std, -8.0, 8.0).astype(np.float32),
        np.clip((clipped_query - mean) / std, -8.0, 8.0).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def calibrated_test_sources(
    truth: np.ndarray,
    train_sources: list[np.ndarray],
    test_sources: list[np.ndarray],
    fit_ids: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> np.ndarray:
    query_ids = np.arange(len(test_sources[0]), dtype=np.int64)
    output = []
    for train_source, test_source in zip(train_sources, test_sources, strict=True):
        ratio = v432.fit_ratio(truth, train_source, fit_ids, train_regimes)
        output.append(v432.apply_ratio(test_source, query_ids, test_regimes, ratio))
    return np.stack(output, axis=0).astype(np.float32)


def test_neighbor_baseline(
    truth: np.ndarray,
    fit_ids: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> np.ndarray:
    positions = np.concatenate([train_positions, test_positions], axis=0)
    blocked = np.concatenate([train_blocked, test_blocked])
    regimes = np.concatenate([train_regimes, test_regimes])
    target = np.zeros((len(positions), truth.shape[1]), dtype=np.float32)
    target[: len(truth)] = truth
    query_ids = np.arange(len(truth), len(positions), dtype=np.int64)
    return normalize_rows(
        v432.neighbor_prediction(
            fit_ids, query_ids, positions, blocked, regimes, target, k=48
        )
    )


def fit_fold_data(
    outer: int,
    truth: np.ndarray,
    train_sources: list[np.ndarray],
    test_sources: list[np.ndarray],
    quality_train: np.ndarray,
    quality_test: np.ndarray,
    fold_id: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> dict[str, object]:
    source_count = len(train_sources)
    query_ids = np.flatnonzero(fold_id == outer)
    fit_ids = np.flatnonzero(fold_id != outer)
    inner_baseline = np.zeros_like(truth)
    inner_candidates = np.zeros(
        (source_count, len(truth), truth.shape[1]), dtype=np.float32
    )
    for inner in range(5):
        if inner == outer:
            continue
        inner_query = np.flatnonzero(fold_id == inner)
        inner_fit = np.flatnonzero((fold_id != outer) & (fold_id != inner))
        inner_baseline[inner_query] = normalize_rows(
            v432.neighbor_prediction(
                inner_fit,
                inner_query,
                train_positions,
                train_blocked,
                train_regimes,
                truth,
                k=48,
            )
        )
        inner_candidates[:, inner_query] = v494.calibrated_sources(
            truth, train_sources, inner_fit, inner_query, train_regimes
        )

    query_baseline = normalize_rows(
        v432.neighbor_prediction(
            fit_ids,
            query_ids,
            train_positions,
            train_blocked,
            train_regimes,
            truth,
            k=48,
        )
    )
    query_candidates = v494.calibrated_sources(
        truth, train_sources, fit_ids, query_ids, train_regimes
    )
    weights: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        regime_fit = fit_ids[train_regimes[fit_ids] == regime]
        weights[regime] = v494.fit_convex_weights(
            truth, inner_candidates, regime_fit
        )
    inner_target = np.zeros_like(truth)
    inner_target[fit_ids] = v494.weighted_target(
        inner_candidates, fit_ids, train_regimes, weights
    )
    query_target = v494.weighted_target(
        query_candidates,
        np.arange(len(query_ids), dtype=np.int64),
        train_regimes[query_ids],
        weights,
    )
    alpha = v494.select_alpha(
        truth,
        inner_baseline,
        inner_target,
        quality_train,
        fit_ids,
        train_regimes,
    )
    fit_anchor, fit_effective = blend_local(
        inner_baseline[fit_ids],
        inner_target[fit_ids],
        quality_train[fit_ids],
        train_regimes[fit_ids],
        alpha,
    )
    query_anchor, query_effective = blend_local(
        query_baseline,
        query_target,
        quality_train[query_ids],
        train_regimes[query_ids],
        alpha,
    )
    fit_prior = prior_coefficients(fit_effective, train_regimes[fit_ids], weights)
    query_prior = prior_coefficients(
        query_effective, train_regimes[query_ids], weights
    )
    fit_bank = np.concatenate(
        [inner_baseline[fit_ids][None, ...], inner_candidates[:, fit_ids]], axis=0
    ).transpose(1, 0, 2)
    query_bank = np.concatenate(
        [query_baseline[None, ...], query_candidates], axis=0
    ).transpose(1, 0, 2)

    test_candidates = calibrated_test_sources(
        truth,
        train_sources,
        test_sources,
        fit_ids,
        train_regimes,
        test_regimes,
    )
    test_baseline = test_neighbor_baseline(
        truth,
        fit_ids,
        train_positions,
        test_positions,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
    )
    test_target = v494.weighted_target(
        test_candidates,
        np.arange(len(test_positions), dtype=np.int64),
        test_regimes,
        weights,
    )
    test_anchor, test_effective = blend_local(
        test_baseline,
        test_target,
        quality_test,
        test_regimes,
        alpha,
    )
    test_prior = prior_coefficients(test_effective, test_regimes, weights)
    test_bank = np.concatenate(
        [test_baseline[None, ...], test_candidates], axis=0
    ).transpose(1, 0, 2)
    return {
        "fit_ids": fit_ids,
        "query_ids": query_ids,
        "weights": weights,
        "alpha": alpha,
        "fit_bank": fit_bank,
        "query_bank": query_bank,
        "test_bank": test_bank,
        "fit_anchor": fit_anchor,
        "query_anchor": query_anchor,
        "test_anchor": test_anchor,
        "fit_prior": fit_prior,
        "query_prior": query_prior,
        "test_prior": test_prior,
    }


class CircularConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(value, (self.pad,) * 4, mode="circular"))


class FusionBlock(nn.Module):
    def __init__(self, width: int, point_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = CircularConv2d(width, width)
        self.conv2 = CircularConv2d(width, width)
        self.film = nn.Linear(point_dim, 2 * width)

    def forward(self, value: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(point).chunk(2, dim=1)
        work = self.conv1(F.gelu(self.norm1(value)))
        work = work * (1.0 + 0.15 * torch.tanh(scale)[:, :, None, None])
        work = work + 0.15 * torch.tanh(bias)[:, :, None, None]
        work = self.conv2(F.gelu(self.norm2(work)))
        return value + work


class CellFusionNet(nn.Module):
    def __init__(
        self,
        source_count: int,
        feature_dim: int,
        width: int,
        blocks: int,
        old_source_count: int = OLD_SOURCE_COUNT,
    ) -> None:
        super().__init__()
        self.source_count = source_count
        self.old_source_count = old_source_count
        point_dim = 192
        self.point = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(384, point_dim),
            nn.GELU(),
        )
        input_channels = 2 * source_count + 8
        self.input = CircularConv2d(input_channels, width)
        self.blocks = nn.ModuleList(
            [FusionBlock(width, point_dim) for _ in range(blocks)]
        )
        self.output = CircularConv2d(width, source_count + 2)
        nn.init.zeros_(self.output.conv.weight)
        nn.init.zeros_(self.output.conv.bias)
        with torch.no_grad():
            self.output.conv.bias[source_count] = -2.0

    def map_features(self, bank: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        log_bank = torch.log(bank.clamp_min(FLOOR))
        sqrt_bank = torch.sqrt((bank * float(HEIGHT * WIDTH)).clamp_min(0.0)).clamp_max(8.0)
        mean = bank.mean(dim=1, keepdim=True)
        log_std = log_bank.std(dim=1, keepdim=True)
        maximum = bank.max(dim=1, keepdim=True).values
        old_start = 1
        old_stop = old_start + self.old_source_count
        old_mean = bank[:, old_start:old_stop].mean(dim=1, keepdim=True)
        corrected_mean = bank[:, old_stop:].mean(dim=1, keepdim=True)
        old_corrected = torch.log(old_mean.clamp_min(FLOOR)) - torch.log(
            corrected_mean.clamp_min(FLOOR)
        )
        return torch.cat(
            [
                log_bank / 10.0,
                sqrt_bank / 3.0,
                torch.log(anchor.clamp_min(FLOOR))[:, None] / 10.0,
                torch.sqrt(anchor * float(HEIGHT * WIDTH)).clamp_max(8.0)[:, None] / 3.0,
                torch.log(mean.clamp_min(FLOOR)) / 10.0,
                log_std / 5.0,
                torch.log(maximum.clamp_min(FLOOR)) / 10.0,
                torch.log(old_mean.clamp_min(FLOOR)) / 10.0,
                torch.log(corrected_mean.clamp_min(FLOOR)) / 10.0,
                old_corrected / 5.0,
            ],
            dim=1,
        )

    def forward(
        self,
        bank: torch.Tensor,
        anchor: torch.Tensor,
        prior: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point = self.point(features)
        value = self.input(self.map_features(bank, anchor))
        for block in self.blocks:
            value = block(value, point)
        raw = self.output(F.gelu(value))
        delta = 1.75 * torch.tanh(raw[:, : self.source_count] / 2.0)
        gate = 0.25 + 0.50 * torch.sigmoid(
            raw[:, self.source_count : self.source_count + 1]
        )
        residual = 0.18 * torch.tanh(raw[:, self.source_count + 1 :] / 2.0)
        smoothed_prior = 0.97 * prior + 0.03 / float(self.source_count)
        selector = F.softmax(
            torch.log(smoothed_prior.clamp_min(1e-6))[:, :, None, None] + delta,
            dim=1,
        )
        selected = torch.sum(selector * bank, dim=1, keepdim=True)
        fused = ((1.0 - gate) * anchor[:, None] + gate * selected) * torch.exp(residual)
        fused = fused.clamp_min(FLOOR)
        prediction = fused / fused.sum(dim=(2, 3), keepdim=True).clamp_min(FLOOR)
        return prediction[:, 0], gate[:, 0], selector


def fusion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    flat_prediction = prediction.flatten(1)
    flat_target = target.flatten(1)
    joint = F.cosine_similarity(flat_prediction, flat_target, dim=1)
    delay = F.cosine_similarity(prediction.sum(dim=2), target.sum(dim=2), dim=1)
    angle = F.cosine_similarity(prediction.sum(dim=1), target.sum(dim=1), dim=1)
    root_error = F.smooth_l1_loss(
        torch.sqrt(prediction.clamp_min(FLOOR)),
        torch.sqrt(target.clamp_min(FLOOR)),
        reduction="none",
    ).mean(dim=(1, 2))
    loss = (
        0.50 * (1.0 - joint)
        + 0.25 * (1.0 - delay)
        + 0.25 * (1.0 - angle)
        + 0.025 * root_error
    ).mean()
    return loss, {
        "joint": joint.mean(),
        "delay": delay.mean(),
        "angle": angle.mean(),
        "gate": gate.mean(),
    }


def train_model(
    bank: np.ndarray,
    anchor: np.ndarray,
    prior: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    width: int,
    blocks: int,
    seed: int,
) -> tuple[CellFusionNet, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CellFusionNet(
        source_count=bank.shape[1],
        feature_dim=features.shape[1],
        width=width,
        blocks=blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=2e-5
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    use_amp = False
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(bank))
        model.train()
        totals = {"loss": 0.0, "joint": 0.0, "delay": 0.0, "angle": 0.0, "gate": 0.0}
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            bank_tensor = torch.from_numpy(bank[ids]).to(device).reshape(
                -1, bank.shape[1], HEIGHT, WIDTH
            )
            anchor_tensor = torch.from_numpy(anchor[ids]).to(device).reshape(
                -1, HEIGHT, WIDTH
            )
            prior_tensor = torch.from_numpy(prior[ids]).to(device)
            feature_tensor = torch.from_numpy(features[ids]).to(device)
            target_tensor = torch.from_numpy(target[ids]).to(device).reshape(
                -1, HEIGHT, WIDTH
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                prediction, gate, _ = model(
                    bank_tensor, anchor_tensor, prior_tensor, feature_tensor
                )
                loss, metrics = fusion_loss(prediction, target_tensor, gate)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite v543 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(ids)
            totals["loss"] += float(loss.detach()) * count
            for key in ("joint", "delay", "angle", "gate"):
                totals[key] += float(metrics[key].detach()) * count
        scheduler.step()
        row = {
            "epoch": float(epoch),
            **{key: value / len(bank) for key, value in totals.items()},
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        if row["loss"] < best_loss - 1e-6:
            best_loss = row["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 8 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d}/{epochs} loss={row['loss']:.6f} "
                f"joint={row['joint']:.6f} delay={row['delay']:.6f} "
                f"angle={row['angle']:.6f} gate={row['gate']:.4f}",
                flush=True,
            )
        if stale_epochs >= 16:
            print(
                f"early stop after {epoch} epochs; best training loss={best_loss:.6f}",
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError("v543 training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    return model, history


def predict_model(
    model: CellFusionNet,
    bank: np.ndarray,
    anchor: np.ndarray,
    prior: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty((len(bank), HEIGHT * WIDTH), dtype=np.float32)
    gate_mean = np.empty(len(bank), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bank), batch_size):
            stop = min(start + batch_size, len(bank))
            bank_tensor = torch.from_numpy(bank[start:stop]).to(device).reshape(
                -1, bank.shape[1], HEIGHT, WIDTH
            )
            anchor_tensor = torch.from_numpy(anchor[start:stop]).to(device).reshape(
                -1, HEIGHT, WIDTH
            )
            prior_tensor = torch.from_numpy(prior[start:stop]).to(device)
            feature_tensor = torch.from_numpy(features[start:stop]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction, gate, _ = model(
                    bank_tensor, anchor_tensor, prior_tensor, feature_tensor
                )
            output[start:stop] = prediction.float().cpu().numpy().reshape(stop - start, -1)
            gate_mean[start:stop] = gate.float().mean(dim=(1, 2)).cpu().numpy()
    return normalize_rows(output), gate_mean


def nearest_cell_oracle(truth: np.ndarray, bank: np.ndarray) -> np.ndarray:
    output = np.empty_like(truth)
    for start in range(0, len(truth), 64):
        stop = min(start + 64, len(truth))
        local_truth = truth[start:stop, None, :]
        local_bank = bank[start:stop]
        index = np.argmin(np.abs(local_bank - local_truth), axis=1)
        output[start:stop] = np.take_along_axis(
            local_bank, index[:, None, :], axis=1
        )[:, 0]
    return normalize_rows(output)


def select_strengths(
    truth: np.ndarray,
    anchor: np.ndarray,
    prediction: np.ndarray,
    regimes: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, dict[str, object]]:
    grid = np.linspace(0.0, 1.25, 26, dtype=np.float32)
    selected: dict[str, float] = {}
    grid_rows: dict[str, list[dict[str, float]]] = {}
    for regime in REGIMES:
        local = regimes == regime
        rows = []
        best_alpha = 0.0
        best_score = -np.inf
        for alpha in grid:
            candidate = log_blend(anchor[local], prediction[local], float(alpha))
            metric = spectrum_scores(truth[local], candidate)
            score = float(np.mean(metric["weighted"]))
            rows.append({"alpha": float(alpha), "weighted": score})
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        selected[regime] = best_alpha
        grid_rows[regime] = rows
    per_point = np.asarray([selected[str(value)] for value in regimes], dtype=np.float32)
    final = log_blend(anchor, prediction, per_point)
    return selected, final, {"grid": grid.tolist(), "by_regime": grid_rows}


def compare_to(path: Path) -> dict[str, float]:
    require(path)
    left = np.load(OUTPUT, mmap_mode="r")
    right = np.load(path, mmap_mode="r")
    error = 0.0
    energy = 0.0
    cosine = 0.0
    count = 0
    for start in range(0, len(left), 5):
        stop = min(start + 5, len(left))
        a = np.asarray(left[start:stop], dtype=np.complex64)
        b = np.asarray(right[start:stop], dtype=np.complex64)
        error += float(np.sum(np.abs(a - b) ** 2, dtype=np.float64))
        energy += float(np.sum(np.abs(b) ** 2, dtype=np.float64))
        inner = np.sum(np.conj(b) * a, axis=(1, 2, 3), dtype=np.complex128)
        norm = np.sqrt(
            np.sum(np.abs(a) ** 2, axis=(1, 2, 3), dtype=np.float64)
            * np.sum(np.abs(b) ** 2, axis=(1, 2, 3), dtype=np.float64)
        )
        cosine += float(np.sum(np.abs(inner) / np.maximum(norm, 1e-30)))
        count += stop - start
    return {
        "nmse": error / max(energy, 1e-30),
        "complex_cosine": cosine / max(count, 1),
    }


def write_package(report: dict[str, object]) -> None:
    neural_checkpoint = PATHS.checkpoints / v469.VERSION / "neural_sionna_calibrator.pt"
    for path in (OUTPUT, CHECKPOINT, neural_checkpoint):
        require(path)
    dependencies = (
        "generate_v543_disagreement_conditioned_cell_fusion.py",
        "generate_v541_corrected_union_stack_action_match.py",
        "generate_v541_corrected_full_library_spectra.py",
        "analyze_v541_corrected_full_library_sbi.py",
        "analyze_v531_expanded_sbi_configuration_library.py",
        "analyze_v529_sbi_configuration_posterior.py",
        "analyze_v494_sionna_configuration_stacking.py",
        "generate_v533_expanded_physical_sbi_action_match.py",
        "generate_v496_sionna_configuration_selector_joint.py",
        "train_v469_neural_sionna_calibrator.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "analyze_v208_strict_historical_replay.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(CHECKPOINT, "model/cellwise_sionna_fusion.pt")
        archive.write(neural_checkpoint, "model/neural_sionna_calibrator.pt")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(output[start : start + 5])).all()
        for start in range(0, len(output), 5)
    )
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "sha256": v533.digest(ZIP_OUT, "sha256"),
        "md5": v533.digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    args = parse_args()
    required = (
        V531_REPORT,
        V541_REPORT,
        V541_ACTIONS,
        V541_OUTPUT,
        V540_OUTPUT,
        v434.OOF_CACHE,
        v458.BASE,
        v458.CALIBRATED_LINK_TARGET,
        v458.LINK_QUALITY,
        v458.POWER_ACTIONS,
        v458.JOINT_ACTIONS,
        v469.JOINT_PRED,
    )
    for path in required:
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")
    print(f"device={device}", flush=True)

    truth, train_sources, test_sources, quality_train, quality_test, source_names = (
        load_sources()
    )
    source_count = len(source_names)
    if source_count != 18:
        raise ValueError(f"expected 18 physical sources, found {source_count}")
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked).astype("U8")
    test_regimes = regime_of(test_blocked).astype("U8")
    fold_id = np.load(v434.OOF_CACHE)["fold_id"].astype(np.int16)
    static_train, static_test, static_names = load_static_features()

    epochs = 8 if args.smoke else args.epochs
    if args.smoke_fold < 0 or args.smoke_fold > 4:
        raise ValueError("--smoke-fold must be in [0, 4]")
    outer_folds = (args.smoke_fold,) if args.smoke else tuple(range(5))
    oof_prediction = np.zeros_like(truth)
    oof_anchor = np.zeros_like(truth)
    oof_oracle = np.zeros_like(truth)
    oof_shuffle = np.zeros_like(truth)
    oof_gate = np.zeros(len(truth), dtype=np.float32)
    test_predictions: list[np.ndarray] = []
    test_anchors: list[np.ndarray] = []
    fold_rows: list[dict[str, object]] = []
    checkpoint_folds: list[dict[str, object]] = []

    for outer in outer_folds:
        print(f"building fold {outer + 1}/5", flush=True)
        fold = fit_fold_data(
            outer,
            truth,
            train_sources,
            test_sources,
            quality_train,
            quality_test,
            fold_id,
            train_positions,
            test_positions,
            train_blocked,
            test_blocked,
            train_regimes,
            test_regimes,
        )
        fit_ids = np.asarray(fold["fit_ids"], dtype=np.int64)
        query_ids = np.asarray(fold["query_ids"], dtype=np.int64)
        fit_dynamic, dynamic_names = dynamic_features(
            fold["fit_bank"], fold["fit_anchor"], fold["fit_prior"], quality_train[fit_ids]
        )
        query_dynamic, _ = dynamic_features(
            fold["query_bank"], fold["query_anchor"], fold["query_prior"], quality_train[query_ids]
        )
        test_dynamic, _ = dynamic_features(
            fold["test_bank"], fold["test_anchor"], fold["test_prior"], quality_test
        )
        fit_raw_features = np.concatenate(
            [static_train[fit_ids], fit_dynamic], axis=1
        ).astype(np.float32)
        query_raw_features = np.concatenate(
            [static_train[query_ids], query_dynamic], axis=1
        ).astype(np.float32)
        test_raw_features = np.concatenate(
            [static_test, test_dynamic], axis=1
        ).astype(np.float32)
        fit_features, query_features, feature_mean, feature_std = standardize_pair(
            fit_raw_features, query_raw_features
        )
        test_features = np.clip(
            (test_raw_features - feature_mean) / feature_std, -8.0, 8.0
        ).astype(np.float32)
        print(
            f"training fold {outer + 1}/5 fit={len(fit_ids)} query={len(query_ids)} "
            f"features={fit_features.shape[1]}",
            flush=True,
        )
        model, history = train_model(
            np.asarray(fold["fit_bank"], dtype=np.float32),
            np.asarray(fold["fit_anchor"], dtype=np.float32),
            np.asarray(fold["fit_prior"], dtype=np.float32),
            fit_features,
            truth[fit_ids],
            device,
            epochs,
            args.batch_size,
            args.width,
            args.blocks,
            SEED + outer,
        )
        query_prediction, query_gate = predict_model(
            model,
            np.asarray(fold["query_bank"], dtype=np.float32),
            np.asarray(fold["query_anchor"], dtype=np.float32),
            np.asarray(fold["query_prior"], dtype=np.float32),
            query_features,
            device,
        )
        test_prediction, _ = predict_model(
            model,
            np.asarray(fold["test_bank"], dtype=np.float32),
            np.asarray(fold["test_anchor"], dtype=np.float32),
            np.asarray(fold["test_prior"], dtype=np.float32),
            test_features,
            device,
        )
        rng = np.random.default_rng(SEED + 100 + outer)
        shuffled_bank = np.asarray(fold["query_bank"], dtype=np.float32).copy()
        shuffled_bank[:, 1:] = shuffled_bank[rng.permutation(len(shuffled_bank)), 1:]
        shuffled_prediction, _ = predict_model(
            model,
            shuffled_bank,
            np.asarray(fold["query_anchor"], dtype=np.float32),
            np.asarray(fold["query_prior"], dtype=np.float32),
            query_features,
            device,
        )
        oracle = nearest_cell_oracle(
            truth[query_ids], np.asarray(fold["query_bank"], dtype=np.float32)
        )
        oof_prediction[query_ids] = query_prediction
        oof_anchor[query_ids] = fold["query_anchor"]
        oof_oracle[query_ids] = oracle
        oof_shuffle[query_ids] = shuffled_prediction
        oof_gate[query_ids] = query_gate
        test_predictions.append(test_prediction)
        test_anchors.append(np.asarray(fold["test_anchor"], dtype=np.float32))
        anchor_metric = spectrum_scores(truth[query_ids], fold["query_anchor"])
        model_metric = spectrum_scores(truth[query_ids], query_prediction)
        oracle_metric = spectrum_scores(truth[query_ids], oracle)
        shuffle_metric = spectrum_scores(truth[query_ids], shuffled_prediction)
        fold_row = {
            "fold": outer,
            "fit": int(len(fit_ids)),
            "query": int(len(query_ids)),
            "anchor": {key: float(np.mean(value)) for key, value in anchor_metric.items()},
            "model": {key: float(np.mean(value)) for key, value in model_metric.items()},
            "model_gain": {
                key: float(np.mean(model_metric[key] - anchor_metric[key]))
                for key in anchor_metric
            },
            "cell_oracle": {key: float(np.mean(value)) for key, value in oracle_metric.items()},
            "cell_oracle_gain": {
                key: float(np.mean(oracle_metric[key] - anchor_metric[key]))
                for key in anchor_metric
            },
            "shuffled_same_point_control": {
                key: float(np.mean(value)) for key, value in shuffle_metric.items()
            },
            "gate_mean": float(np.mean(query_gate)),
            "train_final": history[-1],
        }
        fold_rows.append(fold_row)
        checkpoint_folds.append(
            {
                "fold": outer,
                "fit_ids": fit_ids,
                "weights": {key: value for key, value in fold["weights"].items()},
                "alpha": fold["alpha"],
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
        )
        print(
            f"fold {outer + 1}/5 weighted gain="
            f"{fold_row['model_gain']['weighted']:+.6f} "
            f"joint={fold_row['model_gain']['joint']:+.6f} "
            f"oracle={fold_row['cell_oracle_gain']['weighted']:+.6f}",
            flush=True,
        )
        del model, fold
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.smoke:
        print(json.dumps({"status": "smoke_complete", "folds": fold_rows}, indent=2))
        return

    selected_strengths, calibrated_oof, strength_search = select_strengths(
        truth, oof_anchor, oof_prediction, train_regimes
    )
    calibrated_scores = spectrum_scores(truth, calibrated_oof)
    anchor_scores = spectrum_scores(truth, oof_anchor)
    model_scores = spectrum_scores(truth, oof_prediction)
    oracle_scores = spectrum_scores(truth, oof_oracle)
    shuffle_scores = spectrum_scores(truth, oof_shuffle)
    fold_gains = [
        float(
            np.mean(
                calibrated_scores["weighted"][fold_id == fold]
                - anchor_scores["weighted"][fold_id == fold]
            )
        )
        for fold in range(5)
    ]

    test_prediction_stack = np.stack(test_predictions, axis=0)
    test_anchor_stack = np.stack(test_anchors, axis=0)
    test_strength = np.asarray(
        [selected_strengths[str(value)] for value in test_regimes], dtype=np.float32
    )
    calibrated_test_folds = np.stack(
        [
            log_blend(test_anchor_stack[index], test_prediction_stack[index], test_strength)
            for index in range(5)
        ],
        axis=0,
    )
    final_physical = normalize_rows(np.mean(calibrated_test_folds, axis=0))

    v541_actions = np.load(V541_ACTIONS)
    six_final = normalize_rows(v541_actions["six_final"].astype(np.float32))
    v541_final = normalize_rows(v541_actions["new_final"].astype(np.float32))
    desired_action = np.clip(
        np.log(np.maximum(final_physical, FLOOR))
        - np.log(np.maximum(six_final, FLOOR)),
        -1.5,
        1.5,
    ).astype(np.float32)
    deployment = np.load(v458.JOINT_ACTIONS)
    deployment_regimes = deployment["regimes"].astype("U8")
    if not np.array_equal(deployment_regimes, test_regimes):
        raise ValueError("v458 and v543 regime labels differ")
    point_quality = deployment["point_quality"].astype(np.float32)
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    deployment_alpha = v458.joint_alpha(profile, test_regimes, point_quality)
    target_log_delta = np.zeros_like(desired_action)
    active = deployment_alpha > 1e-6
    target_log_delta[active] = desired_action[active] / deployment_alpha[active, None]
    target_log_delta = np.clip(target_log_delta, -1.5, 1.5)
    neural = normalize_rows(np.load(v469.JOINT_PRED, mmap_mode="r"))
    corrected_target = normalize_rows(
        neural * np.exp(target_log_delta).astype(np.float32)
    )
    np.save(TARGET, corrected_target.astype(np.float32))
    np.savez_compressed(
        ACTIONS,
        source_names=np.asarray(source_names),
        selected_strengths=np.asarray([selected_strengths[key] for key in REGIMES]),
        test_regimes=test_regimes,
        physical_target=final_physical,
        v541_final=v541_final,
        six_final=six_final,
        desired_action=desired_action,
        deployment_alpha=deployment_alpha,
        target_log_delta=target_log_delta,
        fold_model_disagreement=np.std(test_prediction_stack, axis=0).mean(axis=1),
    )
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        regimes=train_regimes,
        truth=truth,
        anchor=oof_anchor,
        prediction=oof_prediction,
        calibrated=calibrated_oof,
        cell_oracle=oof_oracle,
        shuffled=oof_shuffle,
        gate_mean=oof_gate,
        **{f"anchor_{key}": value for key, value in anchor_scores.items()},
        **{f"prediction_{key}": value for key, value in model_scores.items()},
        **{f"calibrated_{key}": value for key, value in calibrated_scores.items()},
        **{f"oracle_{key}": value for key, value in oracle_scores.items()},
    )
    torch.save(
        {
            "version": VERSION,
            "source_names": source_names,
            "static_feature_names": static_names,
            "dynamic_feature_names": dynamic_names,
            "model": {
                "source_count": source_count + 1,
                "feature_dim": int(len(static_names) + len(dynamic_names)),
                "width": args.width,
                "blocks": args.blocks,
                "old_source_count": OLD_SOURCE_COUNT,
            },
            "epochs": epochs,
            "selected_strengths": selected_strengths,
            "folds": checkpoint_folds,
        },
        CHECKPOINT,
    )

    v533.OUTPUT = OUTPUT
    rendering = v533.render(corrected_target)
    comparison = {
        "v540": compare_to(V540_OUTPUT),
        "v541": compare_to(V541_OUTPUT),
        "v533": compare_to(PATHS.cache_dir(533) / "v533_Round1_Test_Channel.npy"),
    }
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The 18 old/corrected Sionna sources contain locally correct delay-angle cells, "
            "but point-level convex weights bind correct and incorrect cells together. A bounded "
            "FiLM CNN can use source disagreement and official geometry to select source weights "
            "per cell while retaining the deterministic v541 stack as its anchor."
        ),
        "new_information": [
            "cell-wise disagreement across 18 old/corrected Sionna spectra",
            "old-versus-corrected local agreement",
            "official point-cloud and geodesic geometry",
            "five-fold model disagreement on the hidden test coordinates",
        ],
        "evidence_boundary": {
            "outer_split": "five mutually exclusive spatial folds",
            "query_truth_used_for_training": False,
            "test_sionna": "all 18 complete test-coordinate normalized joint S-H spectra",
            "online_feedback_used_for_parameters": False,
            "local_metrics": "diagnostic only; online score is final evidence",
        },
        "model": {
            "architecture": "bounded per-cell source selector plus log-spectrum residual",
            "grid": [HEIGHT, WIDTH],
            "source_bank": source_count + 1,
            "physical_sources": source_count,
            "feature_dim": int(len(static_names) + len(dynamic_names)),
            "width": args.width,
            "blocks": args.blocks,
            "epochs_per_fold": epochs,
            "batch_size": args.batch_size,
            "device": str(device),
            "selected_strengths": selected_strengths,
        },
        "oof": {
            "anchor": {key: summarize(value) for key, value in anchor_scores.items()},
            "raw_model": {key: summarize(value) for key, value in model_scores.items()},
            "calibrated_model": {key: summarize(value) for key, value in calibrated_scores.items()},
            "cell_oracle": {key: summarize(value) for key, value in oracle_scores.items()},
            "shuffled_source_control": {key: summarize(value) for key, value in shuffle_scores.items()},
            "calibrated_gain": {
                key: float(np.mean(calibrated_scores[key] - anchor_scores[key]))
                for key in anchor_scores
            },
            "cell_oracle_gain": {
                key: float(np.mean(oracle_scores[key] - anchor_scores[key]))
                for key in anchor_scores
            },
            "fold_weighted_gains": fold_gains,
            "fold_wins": int(sum(value > 0.0 for value in fold_gains)),
            "gate_mean": summarize(oof_gate),
            "strength_search": strength_search,
        },
        "folds": fold_rows,
        "renderer": {
            "base": PATHS.relative(v458.BASE),
            "joint_anchor": PATHS.relative(v469.JOINT_PRED),
            "joint_profile": "tail085",
            "link_scale": 0.40,
            "power_scale": 10.0,
            "cycles": v458.JOINT_CYCLES,
            "preserved": [
                "v458 calibrated linkwise PDP",
                "v458 point power",
                "base complex phase",
            ],
        },
        "action": {
            "physical_cosine_to_v541": float(
                np.mean(row_cosine(final_physical, v541_final))
            ),
            "desired_action_abs_mean": float(np.mean(np.abs(desired_action))),
            "desired_action_abs_p90": float(np.quantile(np.abs(desired_action), 0.90)),
            "desired_action_abs_p99": float(np.quantile(np.abs(desired_action), 0.99)),
        },
        "rendering": rendering,
        "comparison": comparison,
        "online_score": None,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET),
            "actions": PATHS.relative(ACTIONS),
            "oof": PATHS.relative(OOF_CACHE),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "archive": PATHS.relative(ZIP_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    write_package(report)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "archive": PATHS.relative(ZIP_OUT),
                "oof_weighted_gain": report["oof"]["calibrated_gain"]["weighted"],
                "oof_joint_gain": report["oof"]["calibrated_gain"]["joint"],
                "fold_wins": report["oof"]["fold_wins"],
                "cell_oracle_gain": report["oof"]["cell_oracle_gain"]["weighted"],
                "selected_strengths": selected_strengths,
                "replay": rendering["replay"]["score_traceback_leaked"],
                "comparison": comparison,
                "sha256": report["integrity"]["sha256"],
                "md5": report["integrity"]["md5"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
