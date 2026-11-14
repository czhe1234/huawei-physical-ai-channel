#!/usr/bin/env python3
"""Diagnose whether a complete boundary field predicts held-hole PAS."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn

from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    fixed_blend,
    pas_distribution,
    preserve_point_power,
)
from data import load_positions
from official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics,
)
from project_paths import PATHS
from run_rbf_spectral_field import FOLD_CACHE, OOF_META
from run_v3_grouped_link_refinement import (
    GROUPED_PAS,
    GROUPED_PDP,
    TRAIN_PAS,
)


VERSION = "Round2-E175-complete-boundary-field-pas"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e175_boundary_field_pas"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e175_boundary_field_pas"
REPORT = ANALYSIS_DIR / "boundary_field_pas.json"
HOLE_CSV = ANALYSIS_DIR / "boundary_field_pas_per_hole.csv"
FIXED_CSV = ANALYSIS_DIR / "boundary_field_pas_fixed500.csv"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
MAP_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
V7_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "canonical_v7_complete.npy"
)
AD_CACHE = ARTIFACT_DIR / "official_train_angle_delay_amplitude_f16.npy"
PREDICTIONS = ARTIFACT_DIR / "canonical_grouped_pas_predictions.npz"
POINT_METRICS = ARTIFACT_DIR / "canonical_point_metrics.npz"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PAS_GROUPS = 32
PAS_BINS = 256
PDP_GROUPS = 8
PDP_BINS = 192
FREQUENCY_GROUPS = 8
FREQUENCY_GROUP_SIZE = S_SC // FREQUENCY_GROUPS
AD_DELAY_GROUPS = 48
AD_DELAY_GROUP_SIZE = S_SC // AD_DELAY_GROUPS
BOUNDARY_POINTS = 64
BOUNDARY_SECTORS = 16
TRAIN_HOLE_SIZES = (24, 40, 64, 80)
TRAIN_EPISODES = 240
TRAIN_QUERY_BATCH = 32
INNER_STEPS = 600
VALIDATION_INTERVAL = 100
CAPACITY_STEPS = 400
CAPACITY_GROUPED_THRESHOLD = 0.985
CAPACITY_CEILING_GAP = 0.015
PROJECTION_ALPHA = 0.20
TINY = 1e-12


@dataclass(frozen=True)
class Config:
    hidden: int = 128
    heads: int = 4
    cross_layers: int = 2
    dropout: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.5
    seed: int = 20260811175


@dataclass(frozen=True)
class HoleEpisode:
    query_ids: np.ndarray
    boundary_ids: np.ndarray
    center_xy: np.ndarray


@dataclass
class FieldStore:
    grouped_pas: torch.Tensor
    grouped_pdp: torch.Tensor
    angle_delay: torch.Tensor
    positions: torch.Tensor
    point_features: torch.Tensor
    nonzero: np.ndarray


@dataclass
class SharedFields:
    grouped_pas: torch.Tensor
    grouped_pdp: torch.Tensor
    angle_delay: torch.Tensor
    positions: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("cache", "capacity", "formal", "score", "all"),
        default="all",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--outers", default="0,1,2,3,4")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def normalized_amplitude(value: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    norm = torch.sqrt(value.square().sum(dim=dim, keepdim=True).clamp_min(TINY))
    return value / norm


def grouped_cosine(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    prediction_power = prediction.square()
    truth_power = truth.square()
    numerator = (prediction_power * truth_power).sum(dim=-1)
    product = prediction_power.square().sum(dim=-1) * truth_power.square().sum(dim=-1)
    denominator = torch.sqrt(product)
    denominator = torch.where(
        denominator > 0.0, denominator, torch.full_like(denominator, 1e-12)
    )
    return numerator / denominator


def grouped_to_full_pas(amplitude: torch.Tensor) -> torch.Tensor:
    power = amplitude.square().reshape(-1, N_UE, FREQUENCY_GROUPS, PAS_BINS)
    return power.repeat_interleave(FREQUENCY_GROUP_SIZE, dim=2)


def grouped_to_pas_distribution(amplitude: torch.Tensor) -> torch.Tensor:
    full = grouped_to_full_pas(amplitude)
    return full.reshape(-1, N_UE, S_SC, MP, MH, MV).permute(0, 3, 4, 5, 1, 2)


def official_pas_cosine_from_distribution(
    prediction: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    prediction = prediction.reshape(len(prediction), N_UE, S_SC, PAS_BINS)
    truth = truth.permute(0, 2, 3, 1).contiguous()
    numerator = (prediction * truth).sum(dim=-1)
    product = prediction.square().sum(dim=-1) * truth.square().sum(dim=-1)
    denominator = torch.sqrt(product)
    denominator = torch.where(
        denominator > 0.0, denominator, torch.full_like(denominator, 1e-12)
    )
    return numerator / denominator


def build_angle_delay_cache(device: torch.device) -> dict[str, float | int | str]:
    if AD_CACHE.is_file():
        cached = np.load(AD_CACHE, mmap_mode="r")
        if cached.shape == (4000, N_UE, MP, MH, MV, AD_DELAY_GROUPS):
            return {
                "path": PATHS.relative(AD_CACHE),
                "shape": list(cached.shape),
                "dtype": str(cached.dtype),
                "reused": True,
            }

    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    output = np.lib.format.open_memmap(
        AD_CACHE,
        mode="w+",
        dtype=np.float16,
        shape=(len(channel), N_UE, MP, MH, MV, AD_DELAY_GROUPS),
    )
    quantization_cosines: list[float] = []
    for start in range(0, len(channel), 16):
        end = min(start + 16, len(channel))
        value = torch.as_tensor(
            np.asarray(channel[start:end], dtype=np.complex64),
            dtype=torch.complex64,
            device=device,
        ).reshape(-1, MP, MH, MV, N_UE, S_SC)
        transformed = torch.fft.ifft(
            torch.fft.fft2(value, dim=(2, 3), norm="ortho"),
            dim=-1,
            norm="ortho",
        )
        energy = transformed.abs().square().reshape(
            -1, MP, MH, MV, N_UE, AD_DELAY_GROUPS, AD_DELAY_GROUP_SIZE
        )
        energy = energy.sum(dim=-1).permute(0, 4, 1, 2, 3, 5).contiguous()
        total = energy.sum(dim=(2, 3, 4, 5), keepdim=True)
        amplitude = torch.sqrt(energy / total.clamp_min(TINY))
        amplitude = torch.where(total > TINY, amplitude, torch.zeros_like(amplitude))
        quantized = amplitude.to(torch.float16).to(torch.float32)
        cosine = grouped_cosine(
            quantized.reshape(len(amplitude), N_UE, -1),
            amplitude.reshape(len(amplitude), N_UE, -1),
        )
        valid = total.reshape(len(amplitude), N_UE) > TINY
        if torch.any(valid):
            quantization_cosines.append(float(cosine[valid].mean().cpu()))
        output[start:end] = amplitude.cpu().numpy().astype(np.float16)
        if end % 400 == 0 or end == len(channel):
            print(f"E175 AD cache {end}/{len(channel)}", flush=True)
    output.flush()
    return {
        "path": PATHS.relative(AD_CACHE),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "reused": False,
        "mean_fp16_power_cosine": float(np.mean(quantization_cosines)),
    }


def select_boundary(
    query_ids: np.ndarray,
    donor_pool: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    count: int = BOUNDARY_POINTS,
) -> HoleEpisode:
    query_ids = np.unique(np.asarray(query_ids, dtype=np.int64))
    site_values = np.unique(site[query_ids])
    if len(site_values) != 1:
        raise ValueError("a boundary episode must contain one site")
    site_value = int(site_values[0])
    candidates = np.asarray(donor_pool, dtype=np.int64)
    candidates = candidates[(site[candidates] == site_value) & ~np.isin(candidates, query_ids)]
    if len(candidates) < count:
        raise RuntimeError(f"site {site_value} has only {len(candidates)} boundary candidates")

    query_xy = positions[query_ids, :2]
    center = np.mean(query_xy, axis=0)
    tree = cKDTree(query_xy)
    boundary_distance = tree.query(positions[candidates, :2], k=1, workers=-1)[0]
    order = np.argsort(boundary_distance)
    candidate_limit = min(len(order), max(count * 8, 256))
    candidates = candidates[order[:candidate_limit]]
    boundary_distance = boundary_distance[order[:candidate_limit]]
    delta = positions[candidates, :2] - center[None]
    angles = (np.arctan2(delta[:, 1], delta[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    sectors = np.floor(angles / (2.0 * np.pi / BOUNDARY_SECTORS)).astype(np.int64)

    selected: list[int] = []
    per_sector = max(1, count // BOUNDARY_SECTORS)
    for sector in range(BOUNDARY_SECTORS):
        rows = np.flatnonzero(sectors == sector)
        rows = rows[np.argsort(boundary_distance[rows])]
        selected.extend(candidates[rows[:per_sector]].tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < count:
        for point in candidates[np.argsort(boundary_distance)]:
            point = int(point)
            if point not in selected:
                selected.append(point)
            if len(selected) == count:
                break
    selected_array = np.asarray(selected[:count], dtype=np.int64)
    return HoleEpisode(query_ids=query_ids, boundary_ids=selected_array, center_xy=center)


def build_training_episodes(
    donor_pool: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    nonzero: np.ndarray,
    seed: int,
    count: int = TRAIN_EPISODES,
) -> list[HoleEpisode]:
    rng = np.random.default_rng(seed)
    output: list[HoleEpisode] = []
    pools = {value: donor_pool[site[donor_pool] == value] for value in (0, 1)}
    trees = {value: cKDTree(positions[pools[value], :2]) for value in (0, 1)}
    attempts = 0
    while len(output) < count and attempts < count * 20:
        attempts += 1
        site_value = int(rng.integers(0, 2))
        pool = pools[site_value]
        centers = pool[nonzero[pool]]
        if len(centers) == 0:
            continue
        center_id = int(rng.choice(centers))
        size = int(rng.choice(TRAIN_HOLE_SIZES))
        local = np.asarray(
            trees[site_value].query(positions[center_id, :2], k=min(size, len(pool)))[1]
        ).reshape(-1)
        query_ids = pool[local]
        if np.sum(nonzero[query_ids]) < min(16, len(query_ids)):
            continue
        try:
            output.append(select_boundary(query_ids, donor_pool, positions, site))
        except RuntimeError:
            continue
    if len(output) != count:
        raise RuntimeError(f"built only {len(output)}/{count} training pseudo-holes")
    return output


def build_point_features(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    map_raw: np.ndarray,
    fit_ids: np.ndarray,
) -> np.ndarray:
    map_mean = np.mean(map_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    map_std = np.maximum(
        np.std(map_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    map_value = np.clip((map_raw - map_mean) / map_std, -6.0, 6.0)
    serving = bs_positions[site]
    delta = positions - serving
    distance = np.maximum(np.linalg.norm(delta[:, :2], axis=1), 1e-6)
    bearing = np.arctan2(delta[:, 1], delta[:, 0])
    return np.concatenate(
        (
            delta[:, :2] / 200.0,
            distance[:, None] / 250.0,
            np.sin(bearing)[:, None],
            np.cos(bearing)[:, None],
            (site == 0)[:, None].astype(np.float32),
            (site == 1)[:, None].astype(np.float32),
            map_value,
        ),
        axis=1,
    ).astype(np.float32)


class AFieldEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.pas = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pdp = nn.Sequential(
            nn.Conv1d(PDP_GROUPS, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Linear(160, hidden, bias=False)

    def forward(self, pas: torch.Tensor, pdp: torch.Tensor) -> torch.Tensor:
        pas_value = pas.reshape(-1, PAS_GROUPS, MP, MH, MV).reshape(-1, 64, MH, MV)
        pas_value = self.pas(pas_value).flatten(1)
        pdp_value = self.pdp(pdp).flatten(1)
        return self.output(torch.cat((pas_value, pdp_value), dim=1))


class BFieldEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(8, 24, kernel_size=3, stride=(2, 2, 4), padding=1, bias=False),
            nn.GELU(),
            nn.Conv3d(24, 48, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.output = nn.Linear(48, hidden, bias=False)

    def forward(self, angle_delay: torch.Tensor) -> torch.Tensor:
        value = angle_delay.reshape(-1, N_UE * MP, MH, MV, AD_DELAY_GROUPS)
        return self.output(self.encoder(value).flatten(1))


class CrossBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.key_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.feed_norm = nn.LayerNorm(hidden)
        self.feed = nn.Sequential(
            nn.Linear(hidden, hidden * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        update, _ = self.attention(
            self.query_norm(query),
            self.key_norm(key_value),
            self.key_norm(key_value),
            need_weights=False,
        )
        query = query + update
        return query + self.feed(self.feed_norm(query))


class BoundaryFieldPAS(nn.Module):
    def __init__(
        self, representation: str, point_feature_size: int, config: Config
    ) -> None:
        super().__init__()
        self.representation = representation
        self.config = config
        self.field = AFieldEncoder(config.hidden) if representation == "A" else BFieldEncoder(config.hidden)
        static_size = point_feature_size + 2
        self.boundary_static = nn.Sequential(
            nn.Linear(static_size, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        self.query_static = nn.Sequential(
            nn.Linear(static_size, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        self.pair = nn.Sequential(
            nn.Linear(6, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
        )
        self.cross = nn.ModuleList(
            CrossBlock(config.hidden, config.heads, config.dropout)
            for _ in range(config.cross_layers)
        )
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden,
                nhead=config.heads,
                dim_feedforward=config.hidden * 3,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden * 2),
            nn.GELU(),
            nn.Linear(config.hidden * 2, PAS_GROUPS * PAS_BINS),
        )

    def forward(
        self,
        boundary_pas: torch.Tensor,
        boundary_pdp: torch.Tensor,
        boundary_ad: torch.Tensor,
        boundary_features: torch.Tensor,
        query_features: torch.Tensor,
        boundary_xy: torch.Tensor,
        query_xy: torch.Tensor,
        center_xy: torch.Tensor,
        field_enabled: bool,
    ) -> torch.Tensor:
        if field_enabled:
            if self.representation == "A":
                field = self.field(boundary_pas, boundary_pdp)
            else:
                field = self.field(boundary_ad)
        else:
            field = torch.zeros(
                (len(boundary_xy), self.config.hidden),
                dtype=query_features.dtype,
                device=query_features.device,
            )

        boundary_center = (boundary_xy - center_xy[None]) / 32.0
        query_center = (query_xy - center_xy[None]) / 32.0
        boundary = field + self.boundary_static(
            torch.cat((boundary_features, boundary_center), dim=1)
        )
        query = self.query_static(torch.cat((query_features, query_center), dim=1))
        delta = boundary_xy[None] - query_xy[:, None]
        distance = torch.linalg.vector_norm(delta, dim=2).clamp_min(1e-6)
        pair = torch.stack(
            (
                delta[:, :, 0] / 32.0,
                delta[:, :, 1] / 32.0,
                distance / 32.0,
                torch.sin(torch.atan2(delta[:, :, 1], delta[:, :, 0])),
                torch.cos(torch.atan2(delta[:, :, 1], delta[:, :, 0])),
                torch.log1p(distance) / 4.0,
            ),
            dim=2,
        )
        key_value = boundary[None].expand(len(query), -1, -1) + self.pair(pair)
        query = query[:, None]
        for block in self.cross:
            query = block(query, key_value)
        query = self.self_attention(query[:, 0][None])[0]
        amplitude = F.softplus(self.decoder(query)).reshape(-1, PAS_GROUPS, PAS_BINS)
        return normalized_amplitude(amplitude, -1)


def episode_tensors(
    episode: HoleEpisode,
    store: FieldStore,
    representation: str,
    variant: str,
    rng: np.random.Generator | None,
    query_limit: int | None,
) -> tuple[torch.Tensor, ...]:
    query_ids = episode.query_ids[store.nonzero[episode.query_ids]]
    if query_limit is not None and len(query_ids) > query_limit:
        if rng is None:
            query_ids = query_ids[:query_limit]
        else:
            query_ids = np.sort(rng.choice(query_ids, size=query_limit, replace=False))
    boundary_ids = episode.boundary_ids.copy()
    if variant == "shuffled_field":
        if rng is None:
            local = np.random.default_rng(int(np.sum(boundary_ids) + len(query_ids) * 104729))
            boundary_field_ids = boundary_ids[local.permutation(len(boundary_ids))]
        else:
            boundary_field_ids = boundary_ids[rng.permutation(len(boundary_ids))]
    else:
        boundary_field_ids = boundary_ids

    device = store.positions.device
    query_index = torch.as_tensor(query_ids, dtype=torch.long, device=device)
    boundary_index = torch.as_tensor(boundary_ids, dtype=torch.long, device=device)
    field_index = torch.as_tensor(boundary_field_ids, dtype=torch.long, device=device)
    center = torch.as_tensor(episode.center_xy, dtype=torch.float32, device=device)
    return (
        store.grouped_pas[field_index],
        store.grouped_pdp[field_index],
        store.angle_delay[field_index],
        store.point_features[boundary_index],
        store.point_features[query_index],
        store.positions[boundary_index, :2],
        store.positions[query_index, :2],
        center,
        store.grouped_pas[query_index],
        query_ids,
    )


def model_prediction(
    model: BoundaryFieldPAS,
    episode: HoleEpisode,
    store: FieldStore,
    representation: str,
    variant: str,
    rng: np.random.Generator | None,
    query_limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    values = episode_tensors(
        episode, store, representation, variant, rng, query_limit
    )
    prediction = model(*values[:8], field_enabled=variant != "coordinate_only")
    return prediction, values[8], values[9]


@torch.no_grad()
def evaluate_episodes_grouped(
    model: BoundaryFieldPAS,
    episodes: list[HoleEpisode],
    store: FieldStore,
    representation: str,
    variant: str,
) -> dict[str, float | int]:
    model.eval()
    values: list[np.ndarray] = []
    for episode in episodes:
        prediction, truth, _ = model_prediction(
            model, episode, store, representation, variant, None, None
        )
        values.append(grouped_cosine(prediction, truth).mean(dim=1).cpu().numpy())
    score = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    return {"pas": float(np.mean(score)), "count": int(len(score))}


def train_model(
    representation: str,
    variant: str,
    episodes: list[HoleEpisode],
    store: FieldStore,
    config: Config,
    steps: int,
    seed: int,
    validation: list[HoleEpisode] | None = None,
) -> tuple[BoundaryFieldPAS, list[dict[str, float]], int]:
    set_seed(seed)
    model = BoundaryFieldPAS(
        representation, store.point_features.shape[1], config
    ).to(store.positions.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_step = steps
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    rolling: list[float] = []
    for step in range(1, steps + 1):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        model.train()
        prediction, truth, _ = model_prediction(
            model,
            episode,
            store,
            representation,
            variant,
            rng,
            TRAIN_QUERY_BATCH,
        )
        cosine = grouped_cosine(prediction, truth)
        loss = 1.0 - cosine.mean()
        loss = loss + 0.05 * (prediction - truth).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        rolling.append(float(loss.detach().cpu()))

        if step % VALIDATION_INTERVAL == 0 or step == steps:
            record = {"step": float(step), "train_loss": float(np.mean(rolling))}
            rolling.clear()
            if validation is not None:
                result = evaluate_episodes_grouped(
                    model, validation, store, representation, variant
                )
                record["inner_pas"] = float(result["pas"])
                if result["pas"] > best_score:
                    best_score = float(result["pas"])
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
            history.append(record)
            print(
                f"E175 rep={representation} variant={variant} step={step}/{steps} "
                + " ".join(f"{key}={value:.6f}" for key, value in record.items() if key != "step"),
                flush=True,
            )
    if validation is not None:
        if best_state is None:
            raise RuntimeError("validation did not produce a finite checkpoint")
        model.load_state_dict(best_state)
    return model, history, best_step


@torch.no_grad()
def predict_episode(
    model: BoundaryFieldPAS,
    episode: HoleEpisode,
    store: FieldStore,
    representation: str,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction, _, query_ids = model_prediction(
        model, episode, store, representation, variant, None, None
    )
    return query_ids, prediction.cpu().numpy().astype(np.float32)


def mean_prediction(
    episode: HoleEpisode,
    store: FieldStore,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    query_ids = episode.query_ids
    boundary = torch.as_tensor(
        episode.boundary_ids, dtype=torch.long, device=store.positions.device
    )
    if representation == "A":
        power = store.grouped_pas[boundary].square().mean(dim=0, keepdim=True)
        amplitude = torch.sqrt(power.clamp_min(0.0))
        amplitude = normalized_amplitude(amplitude, -1)
    else:
        energy = store.angle_delay[boundary].square().mean(dim=0)
        angle = energy.sum(dim=-1).reshape(N_UE, MP * MH * MV)
        angle = angle / angle.sum(dim=-1, keepdim=True).clamp_min(TINY)
        amplitude = torch.sqrt(angle).repeat_interleave(FREQUENCY_GROUPS, dim=0)[None]
    prediction = amplitude.expand(len(query_ids), -1, -1)
    return query_ids, prediction.cpu().numpy().astype(np.float32)


def load_shared_fields(device: torch.device, positions: np.ndarray) -> SharedFields:
    return SharedFields(
        grouped_pas=torch.as_tensor(
            np.asarray(np.load(GROUPED_PAS, mmap_mode="r"), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ),
        grouped_pdp=torch.as_tensor(
            np.asarray(np.load(GROUPED_PDP, mmap_mode="r"), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ),
        angle_delay=torch.as_tensor(
            np.asarray(np.load(AD_CACHE, mmap_mode="r"), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ),
        positions=torch.as_tensor(positions, dtype=torch.float32, device=device),
    )


def prepare_store(
    device: torch.device,
    shared: SharedFields,
    point_features: np.ndarray,
    nonzero: np.ndarray,
) -> FieldStore:
    return FieldStore(
        grouped_pas=shared.grouped_pas,
        grouped_pdp=shared.grouped_pdp,
        angle_delay=shared.angle_delay,
        positions=shared.positions,
        point_features=torch.as_tensor(point_features, dtype=torch.float32, device=device),
        nonzero=nonzero,
    )


def capacity_gate(
    device: torch.device,
    config: Config,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    map_raw: np.ndarray,
    nonzero: np.ndarray,
    canonical_ids: np.ndarray,
    shared: SharedFields,
) -> dict[str, object]:
    all_ids = np.arange(len(positions), dtype=np.int64)
    pool = all_ids[~np.isin(all_ids, canonical_ids)]
    episodes = build_training_episodes(
        pool, positions, site, nonzero, config.seed + 17, count=1
    )
    base_episode = episodes[0]
    valid_queries = base_episode.query_ids[nonzero[base_episode.query_ids]][:16]
    episode = HoleEpisode(valid_queries, base_episode.boundary_ids, base_episode.center_xy)
    point_features = build_point_features(positions, site, bs_positions, map_raw, pool)
    store = prepare_store(device, shared, point_features, nonzero)
    truth_full = torch.as_tensor(
        np.asarray(np.load(TRAIN_PAS, mmap_mode="r")[valid_queries], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    truth_grouped = store.grouped_pas[
        torch.as_tensor(valid_queries, dtype=torch.long, device=device)
    ]
    ceiling_pas = float(
        official_pas_cosine_from_distribution(
            grouped_to_full_pas(truth_grouped), truth_full
        ).mean().cpu()
    )
    records: dict[str, object] = {
        "query_ids": valid_queries.tolist(),
        "boundary_ids": episode.boundary_ids.tolist(),
        "representation_ceiling_pas": ceiling_pas,
    }
    passed = True
    for representation in ("A", "B"):
        model, history, _ = train_model(
            representation,
            "spatial_field",
            [episode],
            store,
            config,
            CAPACITY_STEPS,
            config.seed + (101 if representation == "A" else 202),
        )
        query_ids, prediction = predict_episode(
            model, episode, store, representation, "spatial_field"
        )
        predicted = torch.as_tensor(prediction, device=device)
        grouped_pas = float(grouped_cosine(predicted, truth_grouped).mean().cpu())
        official_pas = float(
            official_pas_cosine_from_distribution(
                grouped_to_full_pas(predicted), truth_full
            ).mean().cpu()
        )
        local_passed = bool(
            grouped_pas >= CAPACITY_GROUPED_THRESHOLD
            and ceiling_pas - official_pas <= CAPACITY_CEILING_GAP
        )
        passed = passed and local_passed
        checkpoint = ARTIFACT_DIR / f"capacity_{representation}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "representation": representation,
            },
            checkpoint,
        )
        records[representation] = {
            "grouped_pas": grouped_pas,
            "official_pas": official_pas,
            "ceiling_gap": float(ceiling_pas - official_pas),
            "passed": local_passed,
            "history": history,
            "checkpoint": PATHS.relative(checkpoint),
        }
        if not np.array_equal(query_ids, valid_queries):
            raise RuntimeError("capacity query order changed")
    records["passed"] = passed
    return records


def canonical_holes(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    return {
        (site_value, fold): query_ids[
            (fold_id == fold) & (site[query_ids] == site_value)
        ]
        for fold in range(5)
        for site_value in range(2)
    }


def formal_run(
    device: torch.device,
    config: Config,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    map_raw: np.ndarray,
    nonzero: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    selected_outers: list[int],
    shared: SharedFields,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    holes = canonical_holes(query_ids, fold_id, site)
    row_map = np.full(len(positions), -1, dtype=np.int64)
    row_map[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    variants = [
        "A_coordinate_only",
        "A_shuffled_field",
        "A_mean_field",
        "A_spatial_field",
        "B_coordinate_only",
        "B_shuffled_field",
        "B_mean_field",
        "B_spatial_field",
    ]
    predictions = {
        name: np.zeros((len(query_ids), PAS_GROUPS, PAS_BINS), dtype=np.float32)
        for name in variants
    }
    training_records: list[dict[str, object]] = []
    all_ids = np.arange(len(positions), dtype=np.int64)

    for outer in selected_outers:
        inner = (outer + 1) % 5
        outer_ids = query_ids[fold_id == outer]
        inner_ids = query_ids[fold_id == inner]
        inner_pool = all_ids[~np.isin(all_ids, np.concatenate((outer_ids, inner_ids)))]
        outer_pool = all_ids[~np.isin(all_ids, outer_ids)]
        inner_features = build_point_features(
            positions, site, bs_positions, map_raw, inner_pool
        )
        inner_store = prepare_store(device, shared, inner_features, nonzero)
        inner_episodes = build_training_episodes(
            inner_pool,
            positions,
            site,
            nonzero,
            config.seed + outer * 1009,
        )
        validation_episodes = [
            select_boundary(holes[(site_value, inner)], inner_pool, positions, site)
            for site_value in range(2)
        ]

        outer_features = build_point_features(
            positions, site, bs_positions, map_raw, outer_pool
        )
        outer_store = prepare_store(device, shared, outer_features, nonzero)
        outer_episodes = build_training_episodes(
            outer_pool,
            positions,
            site,
            nonzero,
            config.seed + outer * 1009 + 50000,
        )
        held_episodes = [
            select_boundary(holes[(site_value, outer)], outer_pool, positions, site)
            for site_value in range(2)
        ]

        for representation in ("A", "B"):
            for variant in ("coordinate_only", "shuffled_field", "spatial_field"):
                seed = config.seed + outer * 10007 + (0 if representation == "A" else 100000) + {
                    "coordinate_only": 11,
                    "shuffled_field": 22,
                    "spatial_field": 33,
                }[variant]
                _, history, selected_step = train_model(
                    representation,
                    variant,
                    inner_episodes,
                    inner_store,
                    config,
                    INNER_STEPS,
                    seed,
                    validation_episodes,
                )
                model, retrain_history, _ = train_model(
                    representation,
                    variant,
                    outer_episodes,
                    outer_store,
                    config,
                    selected_step,
                    seed + 1,
                )
                checkpoint = ARTIFACT_DIR / f"outer{outer}_{representation}_{variant}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(config),
                        "outer": outer,
                        "inner": inner,
                        "representation": representation,
                        "variant": variant,
                        "selected_step": selected_step,
                    },
                    checkpoint,
                )
                key = f"{representation}_{variant}"
                for episode in held_episodes:
                    local_ids, local_prediction = predict_episode(
                        model, episode, outer_store, representation, variant
                    )
                    predictions[key][row_map[local_ids]] = local_prediction
                training_records.append(
                    {
                        "outer": outer,
                        "inner": inner,
                        "representation": representation,
                        "variant": variant,
                        "selected_step": selected_step,
                        "inner_history": history,
                        "retrain_history": retrain_history,
                        "checkpoint": PATHS.relative(checkpoint),
                    }
                )

            mean_key = f"{representation}_mean_field"
            for episode in held_episodes:
                local_ids, local_prediction = mean_prediction(
                    episode, outer_store, representation
                )
                predictions[mean_key][row_map[local_ids]] = local_prediction

        del inner_store, outer_store
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if set(selected_outers) != set(range(5)):
        return {"training": training_records, "complete": False}, predictions
    return {"training": training_records, "complete": True}, predictions


def replay_formal_predictions(
    device: torch.device,
    config: Config,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    map_raw: np.ndarray,
    nonzero: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    shared: SharedFields,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Replay saved formal checkpoints after a scoring-only implementation fix."""

    holes = canonical_holes(query_ids, fold_id, site)
    row_map = np.full(len(positions), -1, dtype=np.int64)
    row_map[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    variants = [
        "A_coordinate_only",
        "A_shuffled_field",
        "A_mean_field",
        "A_spatial_field",
        "B_coordinate_only",
        "B_shuffled_field",
        "B_mean_field",
        "B_spatial_field",
    ]
    predictions = {
        name: np.zeros((len(query_ids), PAS_GROUPS, PAS_BINS), dtype=np.float32)
        for name in variants
    }
    records: list[dict[str, object]] = []
    all_ids = np.arange(len(positions), dtype=np.int64)

    for outer in range(5):
        outer_ids = query_ids[fold_id == outer]
        outer_pool = all_ids[~np.isin(all_ids, outer_ids)]
        outer_features = build_point_features(
            positions, site, bs_positions, map_raw, outer_pool
        )
        outer_store = prepare_store(device, shared, outer_features, nonzero)
        held_episodes = [
            select_boundary(holes[(site_value, outer)], outer_pool, positions, site)
            for site_value in range(2)
        ]
        for representation in ("A", "B"):
            for variant in ("coordinate_only", "shuffled_field", "spatial_field"):
                checkpoint = ARTIFACT_DIR / f"outer{outer}_{representation}_{variant}.pt"
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                model = BoundaryFieldPAS(
                    representation, outer_store.point_features.shape[1], config
                ).to(device)
                model.load_state_dict(saved["state_dict"])
                key = f"{representation}_{variant}"
                for episode in held_episodes:
                    local_ids, local_prediction = predict_episode(
                        model, episode, outer_store, representation, variant
                    )
                    predictions[key][row_map[local_ids]] = local_prediction
                records.append(
                    {
                        "outer": outer,
                        "inner": int(saved["inner"]),
                        "representation": representation,
                        "variant": variant,
                        "selected_step": int(saved["selected_step"]),
                        "checkpoint": PATHS.relative(checkpoint),
                    }
                )
                del model

            mean_key = f"{representation}_mean_field"
            for episode in held_episodes:
                local_ids, local_prediction = mean_prediction(
                    episode, outer_store, representation
                )
                predictions[mean_key][row_map[local_ids]] = local_prediction
        del outer_store

    return {
        "training": records,
        "complete": True,
        "scoring_replay_from_saved_checkpoints": True,
    }, predictions


def direct_pas_tables(
    predictions: dict[str, np.ndarray],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    truth_pas = np.load(TRAIN_PAS, mmap_mode="r")
    point_values: dict[str, np.ndarray] = {}
    for name, value in predictions.items():
        rows: list[np.ndarray] = []
        for start in range(0, len(query_ids), 16):
            end = min(start + 16, len(query_ids))
            predicted = torch.as_tensor(value[start:end], dtype=torch.float32, device=device)
            truth = torch.as_tensor(
                np.asarray(truth_pas[query_ids[start:end]], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            rows.append(
                official_pas_cosine_from_distribution(
                    grouped_to_full_pas(predicted), truth
                )
                .mean(dim=(1, 2))
                .cpu()
                .numpy()
            )
        point_values[name] = np.concatenate(rows)

    v7 = np.load(V7_CHANNEL, mmap_mode="r")
    truth_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    v7_point = point_metric_statistics(
        np.asarray(v7, dtype=np.complex64),
        np.asarray(truth_channel[query_ids], dtype=np.complex64),
    ).pas
    point_values["V7"] = np.asarray(v7_point, dtype=np.float64)

    hole_rows: list[dict[str, object]] = []
    for fold in range(5):
        for site_value in range(2):
            mask = (fold_id == fold) & (site[query_ids] == site_value)
            for name, value in point_values.items():
                hole_rows.append(
                    {
                        "hole_id": f"P_S{site_value}F{fold}",
                        "site": site_value,
                        "fold": fold,
                        "method": name,
                        "count": int(np.sum(mask)),
                        "pas": float(np.mean(value[mask])),
                        "delta_to_v7": float(
                            np.mean(value[mask]) - np.mean(point_values["V7"][mask])
                        ),
                    }
                )

    fixed_rows = [
        {
            "method": name,
            "count": int(len(value)),
            "pas": float(np.mean(value)),
            "delta_to_v7": float(np.mean(value) - np.mean(point_values["V7"])),
        }
        for name, value in point_values.items()
    ]
    return hole_rows, fixed_rows, point_values


def projected_complete_metrics(
    predictions: dict[str, np.ndarray],
    query_ids: np.ndarray,
    device: torch.device,
) -> list[dict[str, object]]:
    v7 = np.load(V7_CHANNEL, mmap_mode="r")
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    rows: list[dict[str, object]] = []
    baseline = aggregate_statistics(
        point_metric_statistics(
            np.asarray(v7, dtype=np.complex64),
            np.asarray(truth[query_ids], dtype=np.complex64),
        )
    )
    rows.append({"method": "V7", **baseline, "delta_to_v7": 0.0})
    for name, value in predictions.items():
        metrics_parts: dict[str, list[np.ndarray]] = {
            "pas": [],
            "pdp": [],
            "squared_error": [],
            "truth_energy": [],
        }
        for start in range(0, len(query_ids), 4):
            end = min(start + 4, len(query_ids))
            carrier = torch.as_tensor(
                np.asarray(v7[start:end], dtype=np.complex64),
                dtype=torch.complex64,
                device=device,
            )
            target_amplitude = torch.as_tensor(
                value[start:end], dtype=torch.float32, device=device
            )
            desired_target = grouped_to_pas_distribution(target_amplitude)
            desired = fixed_blend(
                pas_distribution(carrier), desired_target, PROJECTION_ALPHA, (1, 2, 3)
            )
            reference_power = carrier.abs().square().sum(dim=(1, 2, 3))
            projected = preserve_point_power(
                exact_pas_projection(carrier, desired), reference_power
            )
            local = point_metric_statistics(
                projected.cpu().numpy().astype(np.complex64),
                np.asarray(truth[query_ids[start:end]], dtype=np.complex64),
            )
            metrics_parts["pas"].append(local.pas)
            metrics_parts["pdp"].append(local.pdp)
            metrics_parts["squared_error"].append(local.squared_error)
            metrics_parts["truth_energy"].append(local.truth_energy)
        payload = PointMetricStatistics(
            **{key: np.concatenate(value) for key, value in metrics_parts.items()}
        )
        aggregate = aggregate_statistics(payload)
        rows.append(
            {
                "method": name,
                **aggregate,
                "delta_to_v7": float(aggregate["score"] - baseline["score"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def success_gate(
    hole_rows: list[dict[str, object]], fixed_rows: list[dict[str, object]]
) -> dict[str, object]:
    hole_lookup = {(row["hole_id"], row["method"]): row for row in hole_rows}
    fixed_lookup = {row["method"]: row for row in fixed_rows}
    hard = ["P_S1F1", "P_S1F2", "P_S1F3", "P_S1F4"]
    representation_results: dict[str, object] = {}
    passed_any = False
    for representation in ("A", "B"):
        spatial = f"{representation}_spatial_field"
        coordinate = f"{representation}_coordinate_only"
        shuffled = f"{representation}_shuffled_field"
        gains = [float(hole_lookup[(hole, spatial)]["delta_to_v7"]) for hole in hard]
        coordinate_margin = [
            float(hole_lookup[(hole, spatial)]["pas"])
            - float(hole_lookup[(hole, coordinate)]["pas"])
            for hole in hard
        ]
        shuffled_margin = [
            float(hole_lookup[(hole, spatial)]["pas"])
            - float(hole_lookup[(hole, shuffled)]["pas"])
            for hole in hard
        ]
        passed = bool(
            sum(value >= 0.020 for value in gains) >= 2
            and sum(value >= 0.0 for value in gains) >= 3
            and min(gains) >= -0.010
            and sum(
                left >= 0.015 and right >= 0.015
                for left, right in zip(coordinate_margin, shuffled_margin)
            )
            >= 2
            and float(fixed_lookup[spatial]["pas"])
            - float(fixed_lookup[coordinate]["pas"])
            >= 0.005
            and float(fixed_lookup[spatial]["pas"])
            - float(fixed_lookup[shuffled]["pas"])
            >= 0.005
        )
        passed_any = passed_any or passed
        representation_results[representation] = {
            "hard_hole_gain_vs_v7": dict(zip(hard, gains)),
            "hard_hole_margin_vs_coordinate": dict(zip(hard, coordinate_margin)),
            "hard_hole_margin_vs_shuffled": dict(zip(hard, shuffled_margin)),
            "fixed500_margin_vs_coordinate": float(
                fixed_lookup[spatial]["pas"] - fixed_lookup[coordinate]["pas"]
            ),
            "fixed500_margin_vs_shuffled": float(
                fixed_lookup[spatial]["pas"] - fixed_lookup[shuffled]["pas"]
            ),
            "passed": passed,
        }
    return {"passed": passed_any, "representations": representation_results}


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    set_seed(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    required = (
        TRAIN_CHANNEL,
        TRAIN_PAS,
        GROUPED_PAS,
        GROUPED_PDP,
        MAP_FEATURES,
        FOLD_CACHE,
        OOF_META,
        V7_CHANNEL,
    )
    missing = [PATHS.relative(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing E175 inputs: {missing}")

    cache_record = build_angle_delay_cache(device)
    if args.stage == "cache":
        print(json.dumps(cache_record, ensure_ascii=False, indent=2), flush=True)
        return

    setup, positions, _, site, _ = load_positions()
    positions = positions.astype(np.float64)
    site = site.astype(np.int8)
    map_raw = np.load(MAP_FEATURES, allow_pickle=False)["train_features"].astype(np.float32)
    grouped_pas_np = np.load(GROUPED_PAS, mmap_mode="r")
    nonzero = np.square(np.asarray(grouped_pas_np, dtype=np.float32)).sum(axis=(1, 2)) > TINY
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    shared = load_shared_fields(device, positions)

    if args.stage == "score":
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        capacity = previous["capacity"]
    else:
        capacity = capacity_gate(
            device,
            config,
            positions,
            site,
            setup.bs_positions,
            map_raw,
            nonzero,
            query_ids,
            shared,
        )
    if args.stage == "capacity" or not capacity["passed"]:
        payload = {
            "experiment": VERSION,
            "status": "capacity_passed" if capacity["passed"] else "capacity_failed",
            "official_inputs_only": True,
            "sionna_used": False,
            "v8_v12_used": False,
            "config": asdict(config),
            "cache": cache_record,
            "capacity": capacity,
            "seconds": float(time.time() - started),
            "device": str(device),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else 0,
        }
        REPORT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return

    selected_outers = [int(value) for value in args.outers.split(",") if value.strip()]
    if args.stage == "score":
        selected_outers = list(range(5))
        formal, predictions = replay_formal_predictions(
            device,
            config,
            positions,
            site,
            setup.bs_positions,
            map_raw,
            nonzero,
            query_ids,
            fold_id,
            shared,
        )
    else:
        formal, predictions = formal_run(
            device,
            config,
            positions,
            site,
            setup.bs_positions,
            map_raw,
            nonzero,
            query_ids,
            fold_id,
            selected_outers,
            shared,
        )
    np.savez_compressed(
        PREDICTIONS,
        query_ids=query_ids,
        fold_id=fold_id,
        **{name: value.astype(np.float16) for name, value in predictions.items()},
    )
    payload: dict[str, object] = {
        "experiment": VERSION,
        "status": "partial_formal" if not formal["complete"] else "formal_complete",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "config": asdict(config),
        "cache": cache_record,
        "capacity": capacity,
        "formal": formal,
        "selected_outers": selected_outers,
        "scoring_interface_fix": args.stage == "score",
        "outputs": {"predictions": PATHS.relative(PREDICTIONS)},
    }
    if formal["complete"]:
        hole_rows, fixed_rows, point_values = direct_pas_tables(
            predictions, query_ids, fold_id, site, device
        )
        complete_rows = projected_complete_metrics(predictions, query_ids, device)
        gate = success_gate(hole_rows, fixed_rows)
        write_csv(HOLE_CSV, hole_rows)
        write_csv(FIXED_CSV, fixed_rows)
        np.savez_compressed(
            POINT_METRICS,
            query_ids=query_ids,
            fold_id=fold_id,
            **{name: value.astype(np.float32) for name, value in point_values.items()},
        )
        payload.update(
            {
                "status": "analysis_complete_gate_passed"
                if gate["passed"]
                else "analysis_complete_gate_failed",
                "direct_pas": {"fixed500": fixed_rows, "per_hole": hole_rows},
                "frozen_v7_pas_projection_complete_metrics": complete_rows,
                "success_gate": gate,
                "outputs": {
                    **payload["outputs"],
                    "point_metrics": PATHS.relative(POINT_METRICS),
                    "per_hole_csv": PATHS.relative(HOLE_CSV),
                    "fixed500_csv": PATHS.relative(FIXED_CSV),
                },
            }
        )
    payload["seconds"] = float(time.time() - started)
    payload["device"] = str(device)
    payload["peak_gpu_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    )
    payload["measured_bottleneck"] = "boundary-set model training"
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
