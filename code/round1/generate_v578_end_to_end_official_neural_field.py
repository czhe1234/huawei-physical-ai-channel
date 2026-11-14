#!/usr/bin/env python3
"""Train and package v578, an official-data end-to-end neural channel field.

The inference graph starts from the released map, train channels, train
positions, and test positions. Historical prediction arrays and external test
teachers are intentionally not inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

import build_map_features as map_features
from project_paths import PATHS


VERSION_NUMBER = 578
VERSION = "v578_end_to_end_official_neural_field"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION

OUTPUT = OUT_DIR / "v578_Round1_Test_Channel.npy"
MAP_CACHE = OUT_DIR / "v578_official_map_features.npz"
DESCRIPTOR_CACHE = OUT_DIR / "v578_train_channel_descriptors.npz"
CHECKPOINT = CHECKPOINT_DIR / "v578_model.pt"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BS_POS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
M_H = 16
M_V = 8
M_P = 2
N_UE = 4
N_SC = 192
N_TX = M_H * M_V * M_P
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    k_neighbors: int = 24
    validation_count: int = 500
    split_seed: int = 20260703
    train_seed: int = 57801
    epochs: int = 60
    patience: int = 18
    eval_every: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 2e-5
    beta: float = 140.3
    gamma: float = 0.15
    steering_scale: float = 4.71238898038
    logit_delta_limit: float = 2.5
    phase_delta_limit: float = 0.30
    power_delta_limit: float = 0.65


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def ensure_map_cache(data_dir: Path, cache: Path, force: bool = False) -> None:
    if cache.is_file() and not force:
        return
    train_pos = np.load(data_dir / "Round1_Train_Pos.npy")
    test_pos = np.load(data_dir / "Round1_Test_Pos.npy")
    vertices = map_features.read_ply_memmap(data_dir / "Round1_Map.ply")
    wall_points, wall_normals = map_features.select_wall_points(
        vertices, -50.0, 300.0, -250.0, 150.0
    )
    tree3 = cKDTree(wall_points)
    tree2 = cKDTree(wall_points[:, :2])
    thresholds = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    samples = np.linspace(0.03, 0.97, 96, dtype=np.float64)
    train_line = map_features.query_line_features(
        tree3, wall_points, train_pos, samples, thresholds, 128
    )
    test_line = map_features.query_line_features(
        tree3, wall_points, test_pos, samples, thresholds, 128
    )
    train_rx = map_features.query_receiver_features(
        tree3, tree2, wall_points, wall_normals, train_pos, 128
    )
    test_rx = map_features.query_receiver_features(
        tree3, tree2, wall_points, wall_normals, test_pos, 128
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        train_features=np.concatenate([train_line, train_rx], axis=1).astype(
            np.float32
        ),
        test_features=np.concatenate([test_line, test_rx], axis=1).astype(
            np.float32
        ),
        feature_names=np.asarray(map_features.make_feature_names(thresholds)),
        wall_point_count=np.asarray(len(wall_points), dtype=np.int64),
    )


def safe_distribution(value: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = np.sum(array, axis=axis, keepdims=True, dtype=np.float64)
    return (array / np.maximum(total, TINY)).astype(np.float32)


def circular_features(distribution: np.ndarray, harmonics: int) -> np.ndarray:
    value = np.asarray(distribution, dtype=np.float64)
    bins = value.shape[1]
    angle = 2.0 * math.pi * np.arange(bins, dtype=np.float64) / bins
    rows: list[np.ndarray] = []
    for order in range(1, harmonics + 1):
        moment = value @ np.exp(1j * order * angle)
        rows.extend([moment.real, moment.imag])
    peak = np.argmax(value, axis=1)
    peak_angle = angle[peak]
    entropy = -np.sum(
        value * np.log(np.maximum(value, 1e-30)), axis=1, dtype=np.float64
    ) / math.log(max(bins, 2))
    rows.extend(
        [
            np.max(value, axis=1),
            entropy,
            np.sin(peak_angle),
            np.cos(peak_angle),
        ]
    )
    return np.stack(rows, axis=1).astype(np.float32)


def ensure_descriptor_cache(
    data_dir: Path, cache: Path, force: bool = False, batch_size: int = 16
) -> None:
    if cache.is_file() and not force:
        return
    channel = np.load(data_dir / "Round1_Train_Channel.npy", mmap_mode="r")
    descriptors: list[np.ndarray] = []
    log_power_parts: list[np.ndarray] = []
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        h = np.asarray(channel[start:end], dtype=np.complex64)
        power = np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        log_power = np.log(np.maximum(power, 1e-30)).astype(np.float32)

        cir = np.fft.ifft(h, axis=-1)
        delay = np.sum(np.abs(cir) ** 2, axis=(1, 2), dtype=np.float64)
        delay = safe_distribution(delay, (1,))
        delay_feature = circular_features(delay, 4)

        array = h.reshape(len(h), M_P, M_H, M_V, N_UE, N_SC)
        angle = np.fft.fftn(array, axes=(2, 3))
        angle_power = np.sum(
            np.abs(angle) ** 2, axis=(1, 4, 5), dtype=np.float64
        )
        angle_power = safe_distribution(angle_power, (1, 2))
        horizontal = np.sum(angle_power, axis=2, dtype=np.float64).astype(
            np.float32
        )
        vertical = np.sum(angle_power, axis=1, dtype=np.float64).astype(
            np.float32
        )
        horizontal_feature = circular_features(horizontal, 3)
        vertical_feature = circular_features(vertical, 3)

        element_power = np.abs(array) ** 2
        polar = np.sum(element_power, axis=(2, 3, 4, 5), dtype=np.float64)
        ue = np.sum(element_power, axis=(1, 2, 3, 5), dtype=np.float64)
        polar = safe_distribution(polar, (1,))
        ue = safe_distribution(ue, (1,))
        descriptors.append(
            np.concatenate(
                [delay_feature, horizontal_feature, vertical_feature, polar, ue],
                axis=1,
            ).astype(np.float32)
        )
        log_power_parts.append(log_power)
        if end % 200 == 0 or end == len(channel):
            print(f"v578 descriptors {end}/{len(channel)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        descriptors=np.concatenate(descriptors, axis=0),
        log_power=np.concatenate(log_power_parts, axis=0),
    )


def base_point_features(position: np.ndarray, map_value: np.ndarray) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float64)
    relative = pos - BS_POS[None, :]
    radial = np.linalg.norm(relative, axis=1)
    unit = relative / np.maximum(radial[:, None], 1e-12)
    azimuth = np.arctan2(relative[:, 1], relative[:, 0])
    elevation = np.arctan2(relative[:, 2], np.linalg.norm(relative[:, :2], axis=1))
    physical = np.column_stack(
        [
            pos[:, 0] / 100.0,
            pos[:, 1] / 100.0,
            pos[:, 2] / 25.0,
            radial / 250.0,
            np.log1p(radial) / math.log(301.0),
            unit,
            np.sin(azimuth),
            np.cos(azimuth),
            np.sin(2.0 * azimuth),
            np.cos(2.0 * azimuth),
            np.sin(elevation),
            np.cos(elevation),
        ]
    )
    return np.concatenate([physical, map_value], axis=1).astype(np.float32)


def fit_standardizer(value: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(value[ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(value[ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean[None]) / std[None]).astype(np.float32)


def make_split(count: int, validation_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    validation = np.sort(order[:validation_count]).astype(np.int64)
    fit = np.sort(order[validation_count:]).astype(np.int64)
    return fit, validation


def neighbor_table(
    query_pos: np.ndarray,
    bank_pos: np.ndarray,
    bank_ids: np.ndarray,
    k: int,
    query_global_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    extra = 1 if query_global_ids is not None else 0
    distances, local = cKDTree(bank_pos[:, :2]).query(
        query_pos[:, :2], k=min(k + extra, len(bank_ids)), workers=-1
    )
    if distances.ndim == 1:
        distances = distances[:, None]
        local = local[:, None]
    absolute = bank_ids[local]
    output_ids = np.empty((len(query_pos), k), dtype=np.int64)
    output_distances = np.empty((len(query_pos), k), dtype=np.float32)
    for row in range(len(query_pos)):
        keep = np.ones(absolute.shape[1], dtype=bool)
        if query_global_ids is not None:
            keep &= absolute[row] != query_global_ids[row]
        chosen = absolute[row, keep][:k]
        chosen_distance = distances[row, keep][:k]
        if len(chosen) != k:
            raise RuntimeError("insufficient non-self neighbors")
        output_ids[row] = chosen
        output_distances[row] = chosen_distance
    return output_ids, output_distances


class NeuralNeighborField(nn.Module):
    def __init__(self, point_dim: int, descriptor_dim: int, config: Config):
        super().__init__()
        self.config = config
        hidden = 96
        self.query_encoder = nn.Sequential(
            nn.Linear(point_dim, 160),
            nn.SiLU(),
            nn.LayerNorm(160),
            nn.Linear(160, hidden),
        )
        self.donor_encoder = nn.Sequential(
            nn.Linear(point_dim + descriptor_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, hidden),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(hidden * 2 + 10, 192),
            nn.SiLU(),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 2),
        )
        self.power_head = nn.Sequential(
            nn.Linear(hidden * 2 + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)
        nn.init.zeros_(self.power_head[-1].weight)
        nn.init.zeros_(self.power_head[-1].bias)

        h_coord = np.tile(
            np.arange(M_H, dtype=np.float32) - (M_H - 1) / 2.0,
            M_P * M_V,
        ).reshape(M_P, M_V, M_H).transpose(0, 2, 1).reshape(-1)
        v_coord = np.tile(
            np.arange(M_V, dtype=np.float32) - (M_V - 1) / 2.0,
            M_P * M_H,
        ).reshape(M_P, M_H, M_V).reshape(-1)
        self.register_buffer("h_coord", torch.from_numpy(h_coord))
        self.register_buffer("v_coord", torch.from_numpy(v_coord))
        self.register_buffer(
            "subcarrier", torch.linspace(-0.5, 0.5, N_SC, dtype=torch.float32)
        )
        self.register_buffer("bs_pos", torch.from_numpy(BS_POS.copy()))

    def forward(
        self,
        query_feature: torch.Tensor,
        query_position: torch.Tensor,
        donor_feature: torch.Tensor,
        donor_descriptor: torch.Tensor,
        donor_position: torch.Tensor,
        donor_log_power_feature: torch.Tensor,
        donor_log_power_raw: torch.Tensor,
        donor_channel: torch.Tensor,
        distance: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, neighbors = distance.shape
        query_latent = self.query_encoder(query_feature)
        donor_latent = self.donor_encoder(
            torch.cat([donor_feature, donor_descriptor], dim=2)
        )

        delta = donor_position - query_position[:, None, :]
        direction = delta[:, :, :2] / distance[:, :, None].clamp_min(1e-4)
        query_radial = torch.linalg.vector_norm(
            query_position - self.bs_pos[None], dim=1
        )
        donor_radial = torch.linalg.vector_norm(
            donor_position - self.bs_pos[None, None], dim=2
        )
        bearing = torch.atan2(delta[:, :, 1], delta[:, :, 0])
        pair_numeric = torch.stack(
            [
                delta[:, :, 0] / 20.0,
                delta[:, :, 1] / 20.0,
                distance / 20.0,
                torch.log1p(distance),
                (donor_radial - query_radial[:, None]) / 20.0,
                direction[:, :, 0],
                direction[:, :, 1],
                torch.sin(bearing),
                torch.cos(bearing),
                donor_log_power_feature,
            ],
            dim=2,
        )
        query_pair = query_latent[:, None, :].expand(-1, neighbors, -1)
        raw_pair = self.pair_head(
            torch.cat([query_pair, donor_latent, pair_numeric], dim=2)
        )
        logit_delta = self.config.logit_delta_limit * torch.tanh(raw_pair[:, :, 0])
        phase_delta = self.config.phase_delta_limit * torch.tanh(raw_pair[:, :, 1])
        prior = -2.0 * torch.log(distance.clamp_min(0.05))
        weight = torch.softmax(prior + logit_delta, dim=1)

        query_unit = (query_position - self.bs_pos[None]) / query_radial[:, None].clamp_min(
            1e-6
        )
        donor_unit = (donor_position - self.bs_pos[None, None]) / donor_radial[
            :, :, None
        ].clamp_min(1e-6)
        delta_h = query_unit[:, None, 1] - donor_unit[:, :, 1]
        delta_v = donor_unit[:, :, 2] - query_unit[:, None, 2]
        steering_arg = self.config.steering_scale * (
            delta_h[:, :, None] * self.h_coord[None, None, :]
            + delta_v[:, :, None] * self.v_coord[None, None, :]
        )
        steering = torch.polar(torch.ones_like(steering_arg), steering_arg)

        beta = self.config.beta + self.config.gamma * self.subcarrier
        radial_arg = (
            (donor_radial - query_radial[:, None])[:, :, None] * beta[None, None]
            + phase_delta[:, :, None]
        )
        radial = torch.polar(torch.ones_like(radial_arg), radial_arg)
        aligned = (
            donor_channel
            * steering[:, :, :, None, None]
            * radial[:, :, None, None, :]
        )
        prediction = torch.sum(weight[:, :, None, None, None] * aligned, dim=1)

        pooled_donor = torch.sum(weight[:, :, None] * donor_latent, dim=1)
        weighted_log_power = torch.sum(weight * donor_log_power_raw, dim=1)
        power_delta = self.config.power_delta_limit * torch.tanh(
            self.power_head(
                torch.cat(
                    [query_latent, pooled_donor, weighted_log_power[:, None]], dim=1
                )
            ).squeeze(1)
        )
        target_log_power = weighted_log_power + power_delta
        current_power = torch.sum(torch.abs(prediction) ** 2, dim=(1, 2, 3))
        scale = torch.sqrt(
            torch.exp(target_log_power) / current_power.clamp_min(1e-30)
        )
        prediction = prediction * scale[:, None, None, None]
        return prediction, {
            "weight": weight,
            "logit_delta": logit_delta,
            "phase_delta": phase_delta,
            "power_delta": power_delta,
            "target_log_power": target_log_power,
        }


def cosine_with_zero_rule(left: torch.Tensor, right: torch.Tensor, dim: int) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=dim)
    left_square = torch.sum(left.square(), dim=dim)
    right_square = torch.sum(right.square(), dim=dim)
    norm_product_square = left_square * right_square
    denominator = torch.sqrt(norm_product_square + 1e-24)
    denominator = torch.where(
        norm_product_square > 0.0,
        denominator,
        torch.full_like(denominator, 1e-12),
    )
    return numerator / denominator


def metric_terms(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_cir = torch.fft.ifft(prediction, dim=-1)
    truth_cir = torch.fft.ifft(truth, dim=-1)
    pred_pdp = torch.abs(pred_cir).square().reshape(-1, N_SC)
    truth_pdp = torch.abs(truth_cir).square().reshape(-1, N_SC)
    pred_pdp = pred_pdp / pred_pdp.sum(dim=1, keepdim=True).clamp_min(1e-30)
    truth_pdp = truth_pdp / truth_pdp.sum(dim=1, keepdim=True).clamp_min(1e-30)
    pdp = cosine_with_zero_rule(pred_pdp, truth_pdp, 1).reshape(len(truth), -1).mean(1)

    pred_array = prediction.reshape(-1, M_P, M_H, M_V, N_UE, N_SC)
    truth_array = truth.reshape(-1, M_P, M_H, M_V, N_UE, N_SC)
    pred_angle = torch.fft.fftn(pred_array, dim=(2, 3))
    truth_angle = torch.fft.fftn(truth_array, dim=(2, 3))
    pred_pas = torch.sum(torch.abs(pred_angle).square(), dim=1)
    truth_pas = torch.sum(torch.abs(truth_angle).square(), dim=1)
    pred_pas = pred_pas.permute(0, 3, 4, 1, 2).reshape(-1, M_H * M_V)
    truth_pas = truth_pas.permute(0, 3, 4, 1, 2).reshape(-1, M_H * M_V)
    pred_pas = pred_pas / pred_pas.sum(dim=1, keepdim=True).clamp_min(1e-30)
    truth_pas = truth_pas / truth_pas.sum(dim=1, keepdim=True).clamp_min(1e-30)
    pas = cosine_with_zero_rule(pred_pas, truth_pas, 1).reshape(len(truth), -1).mean(1)

    error = torch.sum(torch.abs(prediction - truth).square())
    truth_energy = torch.sum(torch.abs(truth).square())
    nmse = error / truth_energy.clamp_min(1e-30)
    inverse_nmse = 1.0 / (1.0 + nmse)
    score = 0.4 * pas.mean() + 0.4 * pdp.mean() + 0.2 * inverse_nmse
    return score, pas, pdp, error, truth_energy


def training_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    score, pas, pdp, error, truth_energy = metric_terms(prediction, truth)
    predicted_log_power = torch.log(
        torch.sum(torch.abs(prediction).square(), dim=(1, 2, 3)).clamp_min(1e-30)
    )
    truth_log_power = torch.log(
        torch.sum(torch.abs(truth).square(), dim=(1, 2, 3)).clamp_min(1e-30)
    )
    power_loss = F.smooth_l1_loss(predicted_log_power, truth_log_power, beta=0.5)
    regularization = (
        2e-4 * diagnostics["logit_delta"].square().mean()
        + 3e-4 * diagnostics["phase_delta"].square().mean()
        + 2e-4 * diagnostics["power_delta"].square().mean()
    )
    loss = 1.0 - score + 0.025 * power_loss + regularization
    return loss, {
        "loss": float(loss.detach().cpu()),
        "score": float(score.detach().cpu()),
        "pas": float(pas.mean().detach().cpu()),
        "pdp": float(pdp.mean().detach().cpu()),
        "nmse": float((error / truth_energy.clamp_min(1e-30)).detach().cpu()),
        "power_loss": float(power_loss.detach().cpu()),
    }


def model_batch(
    model: NeuralNeighborField,
    query_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    distances: np.ndarray,
    static_feature: torch.Tensor,
    descriptor: torch.Tensor,
    position: torch.Tensor,
    log_power_normalized: torch.Tensor,
    log_power_raw: torch.Tensor,
    channel: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    query_tensor = torch.as_tensor(query_ids, dtype=torch.long, device=channel.device)
    neighbor_tensor = torch.as_tensor(
        neighbor_ids, dtype=torch.long, device=channel.device
    )
    distance_tensor = torch.as_tensor(
        distances, dtype=torch.float32, device=channel.device
    )
    return model(
        static_feature[query_tensor],
        position[query_tensor],
        static_feature[neighbor_tensor],
        descriptor[neighbor_tensor],
        position[neighbor_tensor],
        log_power_normalized[neighbor_tensor],
        log_power_raw[neighbor_tensor],
        channel[neighbor_tensor],
        distance_tensor,
    )


@torch.no_grad()
def evaluate_model(
    model: NeuralNeighborField,
    query_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    distances: np.ndarray,
    static_feature: torch.Tensor,
    descriptor: torch.Tensor,
    position: torch.Tensor,
    log_power_normalized: torch.Tensor,
    log_power_raw: torch.Tensor,
    channel: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    pas_values: list[np.ndarray] = []
    pdp_values: list[np.ndarray] = []
    error = 0.0
    truth_energy = 0.0
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        prediction, _ = model_batch(
            model,
            query_ids[start:end],
            neighbor_ids[start:end],
            distances[start:end],
            static_feature,
            descriptor,
            position,
            log_power_normalized,
            log_power_raw,
            channel,
        )
        truth = channel[
            torch.as_tensor(query_ids[start:end], dtype=torch.long, device=channel.device)
        ]
        _score, pas, pdp, local_error, local_energy = metric_terms(prediction, truth)
        pas_values.append(pas.detach().cpu().numpy())
        pdp_values.append(pdp.detach().cpu().numpy())
        error += float(local_error.detach().cpu())
        truth_energy += float(local_energy.detach().cpu())
    pas = float(np.mean(np.concatenate(pas_values)))
    pdp = float(np.mean(np.concatenate(pdp_values)))
    nmse = float(error / max(truth_energy, TINY))
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "inverse_nmse": float(1.0 / (1.0 + nmse)),
        "score": float(0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse)),
    }


def train_model(
    model: NeuralNeighborField,
    fit_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    distances: np.ndarray,
    static_feature: torch.Tensor,
    descriptor: torch.Tensor,
    position: torch.Tensor,
    log_power_normalized: torch.Tensor,
    log_power_raw: torch.Tensor,
    channel: torch.Tensor,
    config: Config,
    epochs: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], int, dict[str, float] | None]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.train_seed)
    history: list[dict[str, object]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_metric: dict[str, float] | None = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(fit_ids))
        running: list[dict[str, float]] = []
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            query = fit_ids[rows]
            prediction, diagnostics = model_batch(
                model,
                query,
                neighbor_ids[rows],
                distances[rows],
                static_feature,
                descriptor,
                position,
                log_power_normalized,
                log_power_raw,
                channel,
            )
            truth = channel[
                torch.as_tensor(query, dtype=torch.long, device=channel.device)
            ]
            loss, row = training_loss(prediction, truth, diagnostics)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running.append(row)

        train_summary = {
            key: float(np.mean([row[key] for row in running])) for key in running[0]
        }
        item: dict[str, object] = {"epoch": epoch, "train": train_summary}
        if validation is not None and (
            epoch == 1 or epoch % config.eval_every == 0 or epoch == epochs
        ):
            validation_ids, validation_neighbors, validation_distances = validation
            metric = evaluate_model(
                model,
                validation_ids,
                validation_neighbors,
                validation_distances,
                static_feature,
                descriptor,
                position,
                log_power_normalized,
                log_power_raw,
                channel,
                config.batch_size,
            )
            item["validation"] = metric
            if best_metric is None or metric["score"] > best_metric["score"] + 1e-7:
                best_metric = metric
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += config.eval_every
            print(
                f"v578 epoch={epoch} train={train_summary['score']:.6f} "
                f"validation={metric['score']:.6f} best={best_epoch}",
                flush=True,
            )
            if stale >= config.patience:
                history.append(item)
                break
        else:
            print(
                f"v578 epoch={epoch} train={train_summary['score']:.6f}",
                flush=True,
            )
        history.append(item)
    if validation is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        best_epoch = epochs
    return best_state, history, best_epoch, best_metric


def validate_official_setup(data_dir: Path) -> None:
    setup_path = data_dir / "Round1_Setup.json"
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    expected = {
        "M": N_TX,
        "M_H": M_H,
        "M_V": M_V,
        "M_P": M_P,
        "N": N_UE,
        "S": N_SC,
    }
    for key, value in expected.items():
        if int(setup[key]) != value:
            raise ValueError(f"unexpected Setup {key}={setup[key]!r}; expected {value}")
    if not np.allclose(np.asarray(setup["X"], dtype=np.float32), BS_POS):
        raise ValueError(f"unexpected Setup base-station position {setup['X']!r}")
    if not np.allclose(np.asarray(setup["w"], dtype=np.float64), [0.4, 0.4, 0.2]):
        raise ValueError(f"unexpected Setup score weights {setup['w']!r}")


def load_official_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.memmap]:
    validate_official_setup(data_dir)
    train_pos = np.load(data_dir / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(data_dir / "Round1_Test_Pos.npy").astype(np.float32)
    train_h = np.load(data_dir / "Round1_Train_Channel.npy", mmap_mode="r")
    if train_h.shape != (2000, N_TX, N_UE, N_SC):
        raise ValueError(f"unexpected train channel shape {train_h.shape}")
    return train_pos, test_pos, train_h


def prepare_features(
    data_dir: Path,
    fit_ids: np.ndarray,
    force_features: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ensure_map_cache(data_dir, MAP_CACHE, force_features)
    ensure_descriptor_cache(data_dir, DESCRIPTOR_CACHE, force_features)
    train_pos = np.load(data_dir / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(data_dir / "Round1_Test_Pos.npy").astype(np.float32)
    map_data = np.load(MAP_CACHE)
    descriptor_data = np.load(DESCRIPTOR_CACHE)
    raw_train_static = base_point_features(train_pos, map_data["train_features"])
    raw_test_static = base_point_features(test_pos, map_data["test_features"])
    raw_descriptor = descriptor_data["descriptors"].astype(np.float32)
    raw_log_power = descriptor_data["log_power"].astype(np.float32)
    static_mean, static_std = fit_standardizer(raw_train_static, fit_ids)
    descriptor_mean, descriptor_std = fit_standardizer(raw_descriptor, fit_ids)
    log_power_mean = np.asarray(np.mean(raw_log_power[fit_ids]), dtype=np.float32)
    log_power_std = np.asarray(
        max(float(np.std(raw_log_power[fit_ids])), 1e-4), dtype=np.float32
    )
    train_static = standardize(raw_train_static, static_mean, static_std)
    test_static = standardize(raw_test_static, static_mean, static_std)
    descriptor = standardize(raw_descriptor, descriptor_mean, descriptor_std)
    log_power = ((raw_log_power - log_power_mean) / log_power_std).astype(np.float32)
    return (
        train_static,
        test_static,
        descriptor,
        log_power,
        static_mean,
        static_std,
        descriptor_mean,
        descriptor_std,
    )


def train_pipeline(
    data_dir: Path, device: torch.device, config: Config, force_features: bool
) -> dict[str, object]:
    train_pos, test_pos, train_h_memmap = load_official_arrays(data_dir)
    fit_ids, validation_ids = make_split(
        len(train_pos), config.validation_count, config.split_seed
    )
    (
        train_static,
        _test_static,
        descriptor,
        log_power,
        static_mean,
        static_std,
        descriptor_mean,
        descriptor_std,
    ) = prepare_features(data_dir, fit_ids, force_features)

    fit_neighbors, fit_distances = neighbor_table(
        train_pos[fit_ids],
        train_pos[fit_ids],
        fit_ids,
        config.k_neighbors,
        fit_ids,
    )
    validation_neighbors, validation_distances = neighbor_table(
        train_pos[validation_ids],
        train_pos[fit_ids],
        fit_ids,
        config.k_neighbors,
    )

    print("v578 loading official train H on GPU", flush=True)
    channel = torch.from_numpy(np.asarray(train_h_memmap)).to(device)
    static_tensor = torch.from_numpy(train_static).to(device)
    descriptor_tensor = torch.from_numpy(descriptor).to(device)
    position_tensor = torch.from_numpy(train_pos).to(device)
    log_power_tensor = torch.from_numpy(log_power).to(device)
    raw_log_power_tensor = torch.from_numpy(
        np.load(DESCRIPTOR_CACHE)["log_power"].astype(np.float32)
    ).to(device)

    set_seed(config.train_seed)
    validation_model = NeuralNeighborField(
        train_static.shape[1], descriptor.shape[1], config
    ).to(device)
    baseline = evaluate_model(
        validation_model,
        validation_ids,
        validation_neighbors,
        validation_distances,
        static_tensor,
        descriptor_tensor,
        position_tensor,
        log_power_tensor,
        raw_log_power_tensor,
        channel,
        config.batch_size,
    )
    best_state, history, best_epoch, best_validation = train_model(
        validation_model,
        fit_ids,
        fit_neighbors,
        fit_distances,
        static_tensor,
        descriptor_tensor,
        position_tensor,
        log_power_tensor,
        raw_log_power_tensor,
        channel,
        config,
        config.epochs,
        (validation_ids, validation_neighbors, validation_distances),
    )
    validation_model.load_state_dict(best_state)
    best_validation = evaluate_model(
        validation_model,
        validation_ids,
        validation_neighbors,
        validation_distances,
        static_tensor,
        descriptor_tensor,
        position_tensor,
        log_power_tensor,
        raw_log_power_tensor,
        channel,
        config.batch_size,
    )

    all_ids = np.arange(len(train_pos), dtype=np.int64)
    (
        final_train_static,
        final_test_static,
        final_descriptor,
        final_log_power,
        final_static_mean,
        final_static_std,
        final_descriptor_mean,
        final_descriptor_std,
    ) = prepare_features(data_dir, all_ids, False)
    raw_log_power = np.load(DESCRIPTOR_CACHE)["log_power"].astype(np.float32)
    final_log_power_mean = np.asarray(np.mean(raw_log_power), dtype=np.float32)
    final_log_power_std = np.asarray(
        max(float(np.std(raw_log_power)), 1e-4), dtype=np.float32
    )

    final_neighbors, final_distances = neighbor_table(
        train_pos,
        train_pos,
        all_ids,
        config.k_neighbors,
        all_ids,
    )
    final_static_tensor = torch.from_numpy(final_train_static).to(device)
    final_descriptor_tensor = torch.from_numpy(final_descriptor).to(device)
    final_log_power_tensor = torch.from_numpy(final_log_power).to(device)
    set_seed(config.train_seed + 1000)
    final_model = NeuralNeighborField(
        final_train_static.shape[1], final_descriptor.shape[1], config
    ).to(device)
    final_state, final_history, _unused_epoch, _unused_metric = train_model(
        final_model,
        all_ids,
        final_neighbors,
        final_distances,
        final_static_tensor,
        final_descriptor_tensor,
        position_tensor,
        final_log_power_tensor,
        raw_log_power_tensor,
        channel,
        config,
        max(best_epoch, 1),
        None,
    )
    final_model.load_state_dict(final_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": VERSION,
            "config": asdict(config),
            "model_state": final_model.state_dict(),
            "point_dim": int(final_train_static.shape[1]),
            "descriptor_dim": int(final_descriptor.shape[1]),
            "static_mean": final_static_mean,
            "static_std": final_static_std,
            "descriptor_mean": final_descriptor_mean,
            "descriptor_std": final_descriptor_std,
            "log_power_mean": final_log_power_mean,
            "log_power_std": final_log_power_std,
            "selected_epoch": int(max(best_epoch, 1)),
            "split_seed": config.split_seed,
            "train_seed": config.train_seed,
        },
        CHECKPOINT,
    )
    del channel
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return {
        "fit_count": int(len(fit_ids)),
        "validation_count": int(len(validation_ids)),
        "baseline_validation": baseline,
        "best_validation": best_validation,
        "validation_gain": float(best_validation["score"] - baseline["score"]),
        "selected_epoch": int(max(best_epoch, 1)),
        "history": history,
        "final_history": final_history,
        "checkpoint": PATHS.relative(CHECKPOINT),
        "test_count": int(len(test_pos)),
    }


def inference_pipeline(
    data_dir: Path,
    device: torch.device,
    checkpoint_path: Path,
    output_path: Path,
    force_features: bool,
) -> dict[str, object]:
    train_pos, test_pos, train_h_memmap = load_official_arrays(data_dir)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config(**payload["config"])
    ensure_map_cache(data_dir, MAP_CACHE, force_features)
    ensure_descriptor_cache(data_dir, DESCRIPTOR_CACHE, force_features)
    map_data = np.load(MAP_CACHE)
    descriptor_data = np.load(DESCRIPTOR_CACHE)
    raw_train_static = base_point_features(train_pos, map_data["train_features"])
    raw_test_static = base_point_features(test_pos, map_data["test_features"])
    train_static = standardize(
        raw_train_static, payload["static_mean"], payload["static_std"]
    )
    test_static = standardize(
        raw_test_static, payload["static_mean"], payload["static_std"]
    )
    descriptor = standardize(
        descriptor_data["descriptors"].astype(np.float32),
        payload["descriptor_mean"],
        payload["descriptor_std"],
    )
    log_power = (
        (descriptor_data["log_power"].astype(np.float32) - payload["log_power_mean"])
        / payload["log_power_std"]
    ).astype(np.float32)

    all_ids = np.arange(len(train_pos), dtype=np.int64)
    test_neighbors, test_distances = neighbor_table(
        test_pos,
        train_pos,
        all_ids,
        config.k_neighbors,
    )
    combined_static = np.concatenate([train_static, test_static], axis=0)
    combined_position = np.concatenate([train_pos, test_pos], axis=0)
    test_global_ids = np.arange(len(train_pos), len(train_pos) + len(test_pos))

    print("v578 inference loading official train H on GPU", flush=True)
    channel = torch.from_numpy(np.asarray(train_h_memmap)).to(device)
    static_tensor = torch.from_numpy(combined_static).to(device)
    descriptor_tensor = torch.from_numpy(descriptor).to(device)
    position_tensor = torch.from_numpy(combined_position).to(device)
    log_power_tensor = torch.from_numpy(log_power).to(device)
    raw_log_power_tensor = torch.from_numpy(
        descriptor_data["log_power"].astype(np.float32)
    ).to(device)
    model = NeuralNeighborField(
        int(payload["point_dim"]), int(payload["descriptor_dim"]), config
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), N_TX, N_UE, N_SC),
    )
    power_ratio = np.empty(len(test_pos), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(test_pos), config.batch_size):
            end = min(start + config.batch_size, len(test_pos))
            query_tensor = torch.as_tensor(
                test_global_ids[start:end], dtype=torch.long, device=device
            )
            neighbor_tensor = torch.as_tensor(
                test_neighbors[start:end], dtype=torch.long, device=device
            )
            distance_tensor = torch.as_tensor(
                test_distances[start:end], dtype=torch.float32, device=device
            )
            prediction, diagnostics = model(
                static_tensor[query_tensor],
                position_tensor[query_tensor],
                static_tensor[neighbor_tensor],
                descriptor_tensor[neighbor_tensor],
                position_tensor[neighbor_tensor],
                log_power_tensor[neighbor_tensor],
                raw_log_power_tensor[neighbor_tensor],
                channel[neighbor_tensor],
                distance_tensor,
            )
            array = prediction.detach().cpu().numpy().astype(np.complex64)
            output[start:end] = array
            donor_power = np.exp(
                np.sum(
                    diagnostics["weight"].detach().cpu().numpy()
                    * descriptor_data["log_power"][test_neighbors[start:end]],
                    axis=1,
                )
            )
            predicted_power = np.sum(
                np.abs(array) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            power_ratio[start:end] = (predicted_power / np.maximum(donor_power, TINY)).astype(
                np.float32
            )
            if end % 50 == 0 or end == len(test_pos):
                output.flush()
                print(f"v578 inference {end}/{len(test_pos)}", flush=True)
    output.flush()
    return {
        "output": PATHS.relative(output_path),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(
            all(
                np.isfinite(np.asarray(output[start : start + 8])).all()
                for start in range(0, len(output), 8)
            )
        ),
        "power_ratio_vs_weighted_donors": summarize(power_ratio),
    }


def package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v578_end_to_end_official_neural_field.py",
        "build_map_features.py",
        "project_paths.py",
    )
    for name in dependencies:
        if not (PATHS.code / name).is_file():
            raise FileNotFoundError(PATHS.code / name)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not OUTPUT.is_file():
        raise FileNotFoundError(OUTPUT)
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(".hw-project", b"")
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(CHECKPOINT, "model/v578_model.pt")
        archive.write(REPORT, "model/v578_report.json")
        archive.writestr(
            "model/INFERENCE.txt",
            (
                "Run generate_v578_end_to_end_official_neural_field.py --mode infer "
                "--checkpoint model/v578_model.pt --data-dir <official round1 directory>.\n"
            ).encode("ascii"),
        )
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    integrity = {
        "zip": PATHS.relative(ZIP_OUT),
        "zip_bytes": int(ZIP_OUT.stat().st_size),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    report["integrity"] = integrity
    return integrity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "train", "infer"), default="all")
    parser.add_argument("--data-dir", type=Path, default=PATHS.round1)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--no-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(epochs=args.epochs, batch_size=args.batch_size)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    training = None
    if args.mode in ("all", "train"):
        training = train_pipeline(args.data_dir, device, config, args.force_features)
    inference = None
    if args.mode in ("all", "infer"):
        inference = inference_pipeline(
            args.data_dir,
            device,
            args.checkpoint,
            args.output,
            args.force_features,
        )
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending" if args.mode == "all" else "analysis_only",
        "primary_hypothesis": (
            "A neural attention field can learn how to combine physically phase-aligned "
            "official train channels from released geometry, while direct PAS/PDP/NMSE "
            "optimization keeps the complete inference graph reproducible from official inputs."
        ),
        "official_inputs": [
            "datasets/round1/Round1_Train_Channel.npy",
            "datasets/round1/Round1_Train_Pos.npy",
            "datasets/round1/Round1_Test_Pos.npy",
            "datasets/round1/Round1_Map.ply",
            "datasets/round1/Round1_Setup.json",
        ],
        "forbidden_inputs_not_used": [
            "historical Round1_Test_Channel.npy predictions",
            "versions/*.zip teacher outputs",
            "test-coordinate Sionna channels",
            "hidden labels or online-score-derived per-point actions",
        ],
        "architecture": {
            "type": "geometry-conditioned neural neighbor attention",
            "trainable_outputs": [
                "neighbor attention correction",
                "bounded residual carrier phase",
                "bounded point-power correction",
            ],
            "physical_inductive_bias": [
                "BS-range carrier de-rotation",
                "BS-array steering alignment",
                "official point-cloud LOS and receiver-wall features",
            ],
            "loss": "0.4*PAS + 0.4*PDP + 0.2/(1+NMSE), plus weak power and action regularization",
        },
        "config": asdict(config),
        "training": training,
        "inference": inference,
        "outputs": {
            "checkpoint": PATHS.relative(args.checkpoint),
            "channel": PATHS.relative(args.output),
            "report": PATHS.relative(REPORT),
            "archive": PATHS.relative(ZIP_OUT) if args.mode == "all" else None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    integrity = None
    if args.mode == "all" and not args.no_package:
        integrity = package(report)
        REPORT.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "training": training,
                "inference": inference,
                "integrity": integrity,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
