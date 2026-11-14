#!/usr/bin/env python3
"""Measure how local official-H PAS predictability decays with distance."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn

from data import load_positions
from project_paths import PATHS
from run_v3_grouped_link_refinement import GROUPED_PAS, GROUPED_PDP, TRAIN_PAS
from train_e175_boundary_field_pas import AD_CACHE
from train_full_path_pdp_link_gate import FOLD_IDS


VERSION = "Round2-E176-local-pas-distance-curve"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e176_local_pas_distance_curve"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e176_local_pas_distance_curve"
REPORT = ANALYSIS_DIR / "local_pas_distance_curve.json"
COMPLEX_AD_CACHE = ARTIFACT_DIR / "official_train_complex_angle_delay_c64.npy"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
CURVE_CSV = ANALYSIS_DIR / "distance_curve.csv"
FOLD_CSV = ANALYSIS_DIR / "distance_curve_by_fold.csv"
CURVE_PNG = ANALYSIS_DIR / "distance_pas_curve.png"
COMPLEX_MARGIN_PNG = ANALYSIS_DIR / "complex_minus_energy_pas.png"
FORMAL_METRICS = ARTIFACT_DIR / "formal_group_metrics.npz"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PAS_GROUPS = 32
PAS_BINS = 256
PDP_GROUPS = 8
AD_DELAY_GROUPS = 48
CAPACITY_PAIRS = 32
CAPACITY_STEPS = 500
CAPACITY_GROUPED_THRESHOLD = 0.985
CAPACITY_CEILING_GAP = 0.015
FORMAL_STEPS = 600
VALIDATION_INTERVAL = 100
PAIR_BATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 32
SCORE_BATCH_SIZE = 4
MAX_SOURCES_PER_TARGET_BIN = 4
CHECKERBOARD_CELL_M = 0.5
DISTANCE_BINS = (
    ("0-1m", 0.0, 1.0),
    ("1-2m", 1.0, 2.0),
    ("2-4m", 2.0, 4.0),
    ("4-6m", 4.0, 6.0),
    ("6-10m", 6.0, 10.0),
    (">10m", 10.0, math.inf),
)
METHODS = (
    "coordinate_only",
    "A_noncoherent",
    "A_shuffled_field",
    "B_complex",
    "B_shuffled_field",
    "neighbor_mean",
    "nearest_copy",
)
TINY = 1e-12


@dataclass(frozen=True)
class Config:
    hidden: int = 128
    dropout: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.5
    residual_bound: float = 6.0
    seed: int = 20260811176


@dataclass
class SharedFields:
    grouped_pas: torch.Tensor
    grouped_pdp: torch.Tensor
    noncoherent_ad: torch.Tensor
    complex_ad: torch.Tensor
    full_pas: torch.Tensor
    positions: torch.Tensor
    site: torch.Tensor
    bs_positions: torch.Tensor


@dataclass
class PairTable:
    source_ids: np.ndarray
    target_ids: np.ndarray
    distances: np.ndarray
    bin_ids: np.ndarray
    fold_ids: np.ndarray
    site_ids: np.ndarray
    group_ids: np.ndarray
    group_target_ids: np.ndarray
    group_bin_ids: np.ndarray
    group_fold_ids: np.ndarray
    group_site_ids: np.ndarray
    group_nearest_source_ids: np.ndarray
    group_distance_median: np.ndarray
    group_distance_p90: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("cache", "capacity", "formal", "all"), default="all"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def channel_to_angle_delay(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(-1, MP, MH, MV, N_UE, S_SC)
    angle = torch.fft.fft2(shaped, dim=(2, 3), norm="ortho")
    return torch.fft.ifft(angle, dim=-1, norm="ortho")


def angle_delay_to_channel(angle_delay: torch.Tensor) -> torch.Tensor:
    frequency = torch.fft.fft(angle_delay, dim=-1, norm="ortho")
    channel = torch.fft.ifft2(frequency, dim=(2, 3), norm="ortho")
    return channel.reshape(-1, MP * MH * MV, N_UE, S_SC)


def build_complex_ad_cache(device: torch.device) -> dict[str, object]:
    expected_shape = (4000, MP, MH, MV, N_UE, S_SC)
    if COMPLEX_AD_CACHE.is_file():
        cached = np.load(COMPLEX_AD_CACHE, mmap_mode="r")
        if cached.shape == expected_shape and cached.dtype == np.complex64:
            return {
                "path": PATHS.relative(COMPLEX_AD_CACHE),
                "shape": list(cached.shape),
                "dtype": str(cached.dtype),
                "reused": True,
            }

    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    output = np.lib.format.open_memmap(
        COMPLEX_AD_CACHE,
        mode="w+",
        dtype=np.complex64,
        shape=expected_shape,
    )
    for start in range(0, len(channel), 8):
        end = min(start + 8, len(channel))
        source = torch.as_tensor(
            np.asarray(channel[start:end], dtype=np.complex64),
            dtype=torch.complex64,
            device=device,
        )
        output[start:end] = channel_to_angle_delay(source).cpu().numpy().astype(
            np.complex64
        )
        if end % 400 == 0 or end == len(channel):
            print(f"E176 complex AD cache {end}/{len(channel)}", flush=True)
    output.flush()

    check_ids = np.asarray([0, 417, 1211, 1999, 2000, 2417, 3211, 3999])
    source_np = np.asarray(channel[check_ids], dtype=np.complex64)
    cached_np = np.asarray(np.load(COMPLEX_AD_CACHE, mmap_mode="r")[check_ids])
    restored = angle_delay_to_channel(
        torch.as_tensor(cached_np, dtype=torch.complex64, device=device)
    ).cpu().numpy()
    relative_l2 = float(
        np.sqrt(np.sum(np.abs(restored - source_np) ** 2, dtype=np.float64))
        / max(np.sqrt(np.sum(np.abs(source_np) ** 2, dtype=np.float64)), TINY)
    )
    maximum_absolute = float(np.max(np.abs(restored - source_np)))
    if relative_l2 > 1e-6:
        raise RuntimeError(f"complex AD identity failed: relative_l2={relative_l2}")
    return {
        "path": PATHS.relative(COMPLEX_AD_CACHE),
        "shape": list(expected_shape),
        "dtype": "complex64",
        "reused": False,
        "identity_relative_l2": relative_l2,
        "identity_maximum_absolute": maximum_absolute,
    }


def grouped_cosine(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=-1)
    product = prediction.square().sum(dim=-1) * truth.square().sum(dim=-1)
    denominator = torch.sqrt(product)
    denominator = torch.where(
        denominator > 0.0, denominator, torch.full_like(denominator, 1e-12)
    )
    return numerator / denominator


def grouped_to_full_pas(amplitude: torch.Tensor) -> torch.Tensor:
    value = amplitude.reshape(-1, N_UE, 8, PAS_BINS)
    power = value.square().repeat_interleave(S_SC // 8, dim=2)
    return power.reshape(-1, N_UE, S_SC, PAS_BINS)


def official_pas_cosine_from_distribution(
    prediction: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    truth_value = truth.permute(0, 2, 3, 1).contiguous()
    numerator = (prediction * truth_value).sum(dim=-1)
    product = prediction.square().sum(dim=-1) * truth_value.square().sum(dim=-1)
    denominator = torch.sqrt(product)
    denominator = torch.where(
        denominator > 0.0, denominator, torch.full_like(denominator, 1e-12)
    )
    return numerator / denominator


def normalize_amplitude(value: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(value.square().sum(dim=-1, keepdim=True).clamp_min(TINY))
    return value / norm


def pair_geometry(
    source_ids: torch.Tensor, target_ids: torch.Tensor, fields: SharedFields
) -> torch.Tensor:
    source = fields.positions[source_ids]
    target = fields.positions[target_ids]
    delta = target[:, :2] - source[:, :2]
    distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-6)
    bearing = torch.atan2(delta[:, 1], delta[:, 0])
    site = fields.site[target_ids]
    bs = fields.bs_positions[site]
    source_bs = source[:, :2] - bs[:, :2]
    target_bs = target[:, :2] - bs[:, :2]
    source_distance = torch.linalg.vector_norm(source_bs, dim=1).clamp_min(1e-6)
    target_distance = torch.linalg.vector_norm(target_bs, dim=1).clamp_min(1e-6)
    return torch.cat(
        (
            delta / 20.0,
            (distance / 20.0)[:, None],
            torch.sin(bearing)[:, None],
            torch.cos(bearing)[:, None],
            source_bs / 200.0,
            target_bs / 200.0,
            (source_distance / 250.0)[:, None],
            (target_distance / 250.0)[:, None],
            (site == 0)[:, None].float(),
            (site == 1)[:, None].float(),
        ),
        dim=1,
    )


class NoncoherentEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.pas = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pdp = nn.Sequential(
            nn.Conv1d(PDP_GROUPS, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.angle_delay = nn.Sequential(
            nn.Conv3d(8, 24, kernel_size=3, stride=(2, 2, 4), padding=1, bias=False),
            nn.GELU(),
            nn.Conv3d(24, 40, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.output = nn.Linear(64 + 48 + 40, hidden)

    def forward(
        self, pas: torch.Tensor, pdp: torch.Tensor, angle_delay: torch.Tensor
    ) -> torch.Tensor:
        pas_value = pas.reshape(-1, PAS_GROUPS, MP, MH, MV).reshape(
            -1, 64, MH, MV
        )
        pas_value = self.pas(pas_value).flatten(1)
        pdp_value = self.pdp(pdp).flatten(1)
        ad_value = angle_delay.reshape(-1, N_UE * MP, MH, MV, AD_DELAY_GROUPS)
        ad_value = self.angle_delay(ad_value).flatten(1)
        return self.output(torch.cat((pas_value, pdp_value, ad_value), dim=1))


class ComplexEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 2, 4), padding=1, bias=False),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.output = nn.Linear(64, hidden)

    def forward(self, angle_delay: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(
            angle_delay.abs().square().sum(dim=(1, 2, 3, 4, 5), keepdim=True)
            .clamp_min(TINY)
        )
        value = angle_delay / scale
        packed = torch.stack((value.real, value.imag), dim=5)
        packed = packed.permute(0, 1, 4, 5, 2, 3, 6).contiguous()
        packed = packed.reshape(-1, MP * N_UE * 2, MH, MV, S_SC)
        return self.output(self.encoder(packed).flatten(1))


class LocalPASOperator(nn.Module):
    def __init__(self, representation: str, config: Config) -> None:
        super().__init__()
        self.representation = representation
        self.config = config
        if representation == "A":
            self.field: nn.Module | None = NoncoherentEncoder(config.hidden)
        elif representation == "B":
            self.field = ComplexEncoder(config.hidden)
        elif representation == "coordinate":
            self.field = None
        else:
            raise ValueError(f"unknown representation: {representation}")
        self.geometry = nn.Sequential(
            nn.Linear(13, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.hidden * 2),
            nn.Linear(config.hidden * 2, config.hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden * 2, config.hidden),
            nn.GELU(),
        )
        self.output = nn.Linear(config.hidden, PAS_GROUPS * PAS_BINS)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        field_ids: torch.Tensor,
        fields: SharedFields,
    ) -> torch.Tensor:
        if self.representation == "A":
            if self.field is None:
                raise RuntimeError("A encoder is missing")
            field = self.field(
                fields.grouped_pas[field_ids],
                fields.grouped_pdp[field_ids],
                fields.noncoherent_ad[field_ids],
            )
        elif self.representation == "B":
            if self.field is None:
                raise RuntimeError("B encoder is missing")
            field = self.field(fields.complex_ad[field_ids])
        else:
            field = torch.zeros(
                (len(source_ids), self.config.hidden),
                dtype=fields.positions.dtype,
                device=fields.positions.device,
            )
        geometry = self.geometry(pair_geometry(source_ids, target_ids, fields))
        residual = self.output(self.fusion(torch.cat((field, geometry), dim=1)))
        residual = self.config.residual_bound * torch.tanh(
            residual.reshape(-1, PAS_GROUPS, PAS_BINS)
        )
        if self.representation == "coordinate":
            base_power = torch.full_like(residual, 1.0 / PAS_BINS)
        else:
            base_power = fields.grouped_pas[field_ids].square()
        logits = torch.log(base_power.clamp_min(1e-8)) + residual
        power = torch.softmax(logits, dim=-1)
        return torch.sqrt(power.clamp_min(0.0))


def load_shared_fields(
    device: torch.device,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
) -> SharedFields:
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
        noncoherent_ad=torch.as_tensor(
            np.asarray(np.load(AD_CACHE, mmap_mode="r"), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ),
        complex_ad=torch.as_tensor(
            np.asarray(np.load(COMPLEX_AD_CACHE, mmap_mode="r"), dtype=np.complex64),
            dtype=torch.complex64,
            device=device,
        ),
        full_pas=torch.as_tensor(
            np.asarray(np.load(TRAIN_PAS, mmap_mode="r")),
            dtype=torch.float16,
            device=device,
        ),
        positions=torch.as_tensor(positions, dtype=torch.float32, device=device),
        site=torch.as_tensor(site, dtype=torch.long, device=device),
        bs_positions=torch.as_tensor(
            bs_positions, dtype=torch.float32, device=device
        ),
    )


def select_capacity_pairs(
    positions: np.ndarray,
    site: np.ndarray,
    nonzero: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates: list[tuple[int, int, float]] = []
    for site_value in range(2):
        ids = np.flatnonzero((site == site_value) & nonzero)
        xy = positions[ids, :2]
        pairs = cKDTree(xy).query_pairs(2.0, output_type="ndarray")
        for left, right in pairs:
            distance = float(np.linalg.norm(xy[left] - xy[right]))
            candidates.append((int(ids[left]), int(ids[right]), distance))
            candidates.append((int(ids[right]), int(ids[left]), distance))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    selected: list[tuple[int, int, float]] = []
    target_seen: set[int] = set()
    site_count = {0: 0, 1: 0}
    for source, target, distance in candidates:
        target_site = int(site[target])
        if target in target_seen or site_count[target_site] >= CAPACITY_PAIRS // 2:
            continue
        selected.append((source, target, distance))
        target_seen.add(target)
        site_count[target_site] += 1
        if len(selected) == CAPACITY_PAIRS:
            break
    if len(selected) != CAPACITY_PAIRS:
        raise RuntimeError(f"only found {len(selected)} unique capacity pairs")
    return (
        np.asarray([value[0] for value in selected], dtype=np.int64),
        np.asarray([value[1] for value in selected], dtype=np.int64),
        np.asarray([value[2] for value in selected], dtype=np.float32),
    )


def train_capacity_model(
    representation: str,
    fields: SharedFields,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    truth_pas: np.ndarray,
    config: Config,
) -> tuple[LocalPASOperator, list[dict[str, float]]]:
    set_seed(config.seed + (0 if representation == "A" else 10000))
    model = LocalPASOperator(representation, config).to(fields.positions.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    source = torch.as_tensor(source_ids, dtype=torch.long, device=fields.positions.device)
    target = torch.as_tensor(target_ids, dtype=torch.long, device=fields.positions.device)
    truth = torch.as_tensor(truth_pas, dtype=torch.float32, device=fields.positions.device)
    history: list[dict[str, float]] = []
    rolling: list[float] = []
    for step in range(1, CAPACITY_STEPS + 1):
        model.train()
        prediction = model(source, target, source, fields)
        cosine = grouped_cosine(prediction, truth)
        loss = 1.0 - cosine.mean()
        loss = loss + 0.02 * (prediction - truth).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        rolling.append(float(loss.detach().cpu()))
        if step % 100 == 0:
            record = {"step": float(step), "train_loss": float(np.mean(rolling))}
            history.append(record)
            rolling.clear()
            print(
                f"E176 capacity rep={representation} step={step}/{CAPACITY_STEPS} "
                f"train_loss={record['train_loss']:.6f}",
                flush=True,
            )
    return model, history


@torch.no_grad()
def capacity_gate(
    device: torch.device,
    config: Config,
    positions: np.ndarray,
    site: np.ndarray,
    fields: SharedFields,
) -> dict[str, object]:
    grouped_pas_np = np.asarray(np.load(GROUPED_PAS, mmap_mode="r"), dtype=np.float32)
    nonzero = np.square(grouped_pas_np).sum(axis=(1, 2)) > TINY
    source_ids, target_ids, distances = select_capacity_pairs(
        positions, site, nonzero, config.seed
    )
    truth_grouped = grouped_pas_np[target_ids]
    truth_full = torch.as_tensor(
        np.asarray(np.load(TRAIN_PAS, mmap_mode="r")[target_ids], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    ceiling = official_pas_cosine_from_distribution(
        grouped_to_full_pas(
            torch.as_tensor(truth_grouped, dtype=torch.float32, device=device)
        ),
        truth_full,
    ).mean()
    result: dict[str, object] = {
        "source_ids": source_ids.tolist(),
        "target_ids": target_ids.tolist(),
        "distances": distances.tolist(),
        "distance_mean": float(np.mean(distances)),
        "distance_max": float(np.max(distances)),
        "representation_ceiling_pas": float(ceiling.cpu()),
    }
    passed = True
    for representation in ("A", "B"):
        with torch.enable_grad():
            model, history = train_capacity_model(
                representation,
                fields,
                source_ids,
                target_ids,
                truth_grouped,
                config,
            )
        prediction = model(
            torch.as_tensor(source_ids, dtype=torch.long, device=device),
            torch.as_tensor(target_ids, dtype=torch.long, device=device),
            torch.as_tensor(source_ids, dtype=torch.long, device=device),
            fields,
        )
        grouped_pas = float(
            grouped_cosine(
                prediction,
                torch.as_tensor(truth_grouped, dtype=torch.float32, device=device),
            )
            .mean()
            .cpu()
        )
        official_pas = float(
            official_pas_cosine_from_distribution(
                grouped_to_full_pas(prediction), truth_full
            )
            .mean()
            .cpu()
        )
        ceiling_gap = float(ceiling.cpu()) - official_pas
        local_passed = bool(
            grouped_pas >= CAPACITY_GROUPED_THRESHOLD
            and ceiling_gap <= CAPACITY_CEILING_GAP
        )
        checkpoint = ARTIFACT_DIR / f"capacity_{representation}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "representation": representation,
                "source_ids": source_ids,
                "target_ids": target_ids,
            },
            checkpoint,
        )
        result[representation] = {
            "grouped_pas": grouped_pas,
            "official_pas": official_pas,
            "ceiling_gap": ceiling_gap,
            "passed": local_passed,
            "history": history,
            "checkpoint": PATHS.relative(checkpoint),
        }
        passed = passed and local_passed
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["passed"] = passed
    return result


def checkerboard_anchor_mask(positions: np.ndarray) -> np.ndarray:
    cell_x = np.floor(positions[:, 0] / CHECKERBOARD_CELL_M).astype(np.int64)
    cell_y = np.floor(positions[:, 1] / CHECKERBOARD_CELL_M).astype(np.int64)
    return ((cell_x + 2 * cell_y) % 2) == 0


def distance_bin_id(distance: np.ndarray) -> np.ndarray:
    output = np.full(len(distance), -1, dtype=np.int8)
    for bin_id, (_, lower, upper) in enumerate(DISTANCE_BINS):
        if math.isinf(upper):
            selected = distance >= lower
        else:
            selected = (distance >= lower) & (distance < upper)
        output[selected] = bin_id
    return output


def quantile_source_rows(
    source_ids: np.ndarray,
    distances: np.ndarray,
    logarithmic: bool,
) -> np.ndarray:
    order = np.lexsort((source_ids, distances))
    if len(order) <= MAX_SOURCES_PER_TARGET_BIN:
        return order
    values = np.log(np.maximum(distances[order], 1e-6)) if logarithmic else distances[order]
    targets = np.linspace(
        float(values[0]),
        float(values[-1]),
        MAX_SOURCES_PER_TARGET_BIN,
    )
    chosen: list[int] = []
    for target in targets:
        candidates = np.argsort(np.abs(values - target), kind="stable")
        for candidate in candidates:
            row = int(order[int(candidate)])
            if row not in chosen:
                chosen.append(row)
                break
    if len(chosen) < MAX_SOURCES_PER_TARGET_BIN:
        for row in order:
            value = int(row)
            if value not in chosen:
                chosen.append(value)
            if len(chosen) == MAX_SOURCES_PER_TARGET_BIN:
                break
    return np.asarray(chosen, dtype=np.int64)


def build_fold_pair_table(
    fold: int,
    positions: np.ndarray,
    site: np.ndarray,
    broad_fold: np.ndarray,
    anchor_mask: np.ndarray,
    nonzero: np.ndarray,
) -> PairTable:
    source_rows: list[int] = []
    target_rows: list[int] = []
    distance_rows: list[float] = []
    bin_rows: list[int] = []
    fold_rows: list[int] = []
    site_rows: list[int] = []
    group_rows: list[int] = []
    group_target: list[int] = []
    group_bin: list[int] = []
    group_fold: list[int] = []
    group_site: list[int] = []
    group_nearest: list[int] = []
    group_median: list[float] = []
    group_p90: list[float] = []

    group_id = 0
    for site_value in sorted(np.unique(site).tolist()):
        anchors = np.flatnonzero(
            (broad_fold == fold)
            & (site == site_value)
            & anchor_mask
            & nonzero
        )
        targets = np.flatnonzero(
            (broad_fold == fold)
            & (site == site_value)
            & ~anchor_mask
            & nonzero
        )
        if len(anchors) == 0 or len(targets) == 0:
            continue
        anchor_xy = positions[anchors, :2]
        for target_id in targets:
            distances = np.linalg.norm(anchor_xy - positions[target_id, :2], axis=1)
            bins = distance_bin_id(distances)
            for bin_id in range(len(DISTANCE_BINS)):
                local = np.flatnonzero(bins == bin_id)
                if len(local) == 0:
                    continue
                picked_local = local[
                    quantile_source_rows(
                        anchors[local],
                        distances[local],
                        logarithmic=bin_id == len(DISTANCE_BINS) - 1,
                    )
                ]
                selected_sources = anchors[picked_local]
                selected_distances = distances[picked_local]
                nearest_row = int(np.argmin(selected_distances))
                source_rows.extend(selected_sources.tolist())
                target_rows.extend([int(target_id)] * len(selected_sources))
                distance_rows.extend(selected_distances.astype(np.float64).tolist())
                bin_rows.extend([bin_id] * len(selected_sources))
                fold_rows.extend([fold] * len(selected_sources))
                site_rows.extend([int(site_value)] * len(selected_sources))
                group_rows.extend([group_id] * len(selected_sources))
                group_target.append(int(target_id))
                group_bin.append(bin_id)
                group_fold.append(fold)
                group_site.append(int(site_value))
                group_nearest.append(int(selected_sources[nearest_row]))
                group_median.append(float(np.median(selected_distances)))
                group_p90.append(float(np.quantile(selected_distances, 0.90)))
                group_id += 1

    return PairTable(
        source_ids=np.asarray(source_rows, dtype=np.int64),
        target_ids=np.asarray(target_rows, dtype=np.int64),
        distances=np.asarray(distance_rows, dtype=np.float32),
        bin_ids=np.asarray(bin_rows, dtype=np.int8),
        fold_ids=np.asarray(fold_rows, dtype=np.int8),
        site_ids=np.asarray(site_rows, dtype=np.int8),
        group_ids=np.asarray(group_rows, dtype=np.int64),
        group_target_ids=np.asarray(group_target, dtype=np.int64),
        group_bin_ids=np.asarray(group_bin, dtype=np.int8),
        group_fold_ids=np.asarray(group_fold, dtype=np.int8),
        group_site_ids=np.asarray(group_site, dtype=np.int8),
        group_nearest_source_ids=np.asarray(group_nearest, dtype=np.int64),
        group_distance_median=np.asarray(group_median, dtype=np.float32),
        group_distance_p90=np.asarray(group_p90, dtype=np.float32),
    )


def combine_pair_tables(tables: list[PairTable]) -> PairTable:
    if not tables:
        raise ValueError("at least one pair table is required")
    pair_values: dict[str, list[np.ndarray]] = {
        "source_ids": [],
        "target_ids": [],
        "distances": [],
        "bin_ids": [],
        "fold_ids": [],
        "site_ids": [],
        "group_ids": [],
    }
    group_values: dict[str, list[np.ndarray]] = {
        "group_target_ids": [],
        "group_bin_ids": [],
        "group_fold_ids": [],
        "group_site_ids": [],
        "group_nearest_source_ids": [],
        "group_distance_median": [],
        "group_distance_p90": [],
    }
    group_offset = 0
    for table in tables:
        for name in pair_values:
            value = getattr(table, name)
            if name == "group_ids":
                value = value + group_offset
            pair_values[name].append(value)
        for name in group_values:
            group_values[name].append(getattr(table, name))
        group_offset += len(table.group_target_ids)
    return PairTable(
        **{
            name: np.concatenate(values) if values else np.zeros(0)
            for name, values in {**pair_values, **group_values}.items()
        }
    )


def shuffled_field_ids(table: PairTable, seed: int) -> np.ndarray:
    output = table.source_ids.copy()
    rng = np.random.default_rng(seed)
    for site_value in np.unique(table.site_ids):
        for bin_id in range(len(DISTANCE_BINS)):
            rows = np.flatnonzero(
                (table.site_ids == site_value) & (table.bin_ids == bin_id)
            )
            if len(rows) <= 1:
                continue
            permutation = rng.permutation(len(rows))
            if np.all(permutation == np.arange(len(rows))):
                permutation = np.roll(permutation, 1)
            output[rows] = table.source_ids[rows[permutation]]
    return output


def pair_table_audit(table: PairTable) -> dict[str, object]:
    bins: list[dict[str, object]] = []
    for bin_id, (label, _, _) in enumerate(DISTANCE_BINS):
        pair_mask = table.bin_ids == bin_id
        group_mask = table.group_bin_ids == bin_id
        values = table.distances[pair_mask]
        bins.append(
            {
                "bin": label,
                "pairs": int(np.sum(pair_mask)),
                "targets": int(np.sum(group_mask)),
                "distance_median_m": float(np.median(values)) if len(values) else None,
                "distance_p90_m": float(np.quantile(values, 0.90)) if len(values) else None,
            }
        )
    return {
        "pairs": int(len(table.source_ids)),
        "target_bin_groups": int(len(table.group_target_ids)),
        "unique_targets": int(len(np.unique(table.group_target_ids))),
        "bins": bins,
    }


def method_representation(method: str) -> str:
    if method == "coordinate_only":
        return "coordinate"
    if method.startswith("A_"):
        return "A"
    if method.startswith("B_"):
        return "B"
    raise ValueError(f"method has no trainable representation: {method}")


def method_field_ids(
    method: str,
    table: PairTable,
    shuffled: np.ndarray,
) -> np.ndarray:
    if method.endswith("shuffled_field"):
        return shuffled
    return table.source_ids


def sample_pair_rows(
    table: PairTable,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    available = [
        np.flatnonzero(table.bin_ids == bin_id)
        for bin_id in range(len(DISTANCE_BINS))
    ]
    valid_bins = np.asarray(
        [bin_id for bin_id, rows in enumerate(available) if len(rows)], dtype=np.int64
    )
    selected_bins = rng.choice(valid_bins, size=batch_size, replace=True)
    rows = np.empty(batch_size, dtype=np.int64)
    for index, bin_id in enumerate(selected_bins):
        pool = available[int(bin_id)]
        rows[index] = int(pool[rng.integers(0, len(pool))])
    return rows


@torch.no_grad()
def predict_grouped_pas(
    model: LocalPASOperator,
    table: PairTable,
    field_ids: np.ndarray,
    fields: SharedFields,
) -> torch.Tensor:
    model.eval()
    device = fields.positions.device
    count = len(table.group_target_ids)
    power_sum = torch.zeros(
        (count, PAS_GROUPS, PAS_BINS), dtype=torch.float32, device=device
    )
    pair_count = torch.zeros(count, dtype=torch.float32, device=device)
    for start in range(0, len(table.source_ids), PREDICTION_BATCH_SIZE):
        end = min(start + PREDICTION_BATCH_SIZE, len(table.source_ids))
        source = torch.as_tensor(
            table.source_ids[start:end], dtype=torch.long, device=device
        )
        target = torch.as_tensor(
            table.target_ids[start:end], dtype=torch.long, device=device
        )
        field = torch.as_tensor(field_ids[start:end], dtype=torch.long, device=device)
        group = torch.as_tensor(
            table.group_ids[start:end], dtype=torch.long, device=device
        )
        prediction = model(source, target, field, fields)
        power_sum.index_add_(0, group, prediction.square())
        pair_count.index_add_(0, group, torch.ones_like(group, dtype=torch.float32))
    power = power_sum / pair_count[:, None, None].clamp_min(1.0)
    amplitude = torch.sqrt(power.clamp_min(0.0))
    return normalize_amplitude(amplitude)


@torch.no_grad()
def baseline_grouped_pas(
    table: PairTable,
    fields: SharedFields,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = fields.positions.device
    count = len(table.group_target_ids)
    power_sum = torch.zeros(
        (count, PAS_GROUPS, PAS_BINS), dtype=torch.float32, device=device
    )
    pair_count = torch.zeros(count, dtype=torch.float32, device=device)
    for start in range(0, len(table.source_ids), 256):
        end = min(start + 256, len(table.source_ids))
        source = torch.as_tensor(
            table.source_ids[start:end], dtype=torch.long, device=device
        )
        group = torch.as_tensor(
            table.group_ids[start:end], dtype=torch.long, device=device
        )
        power_sum.index_add_(0, group, fields.grouped_pas[source].square())
        pair_count.index_add_(0, group, torch.ones_like(group, dtype=torch.float32))
    mean_amplitude = normalize_amplitude(
        torch.sqrt(
            (power_sum / pair_count[:, None, None].clamp_min(1.0)).clamp_min(0.0)
        )
    )
    nearest = torch.as_tensor(
        table.group_nearest_source_ids, dtype=torch.long, device=device
    )
    nearest_amplitude = normalize_amplitude(fields.grouped_pas[nearest])
    return mean_amplitude, nearest_amplitude


@torch.no_grad()
def direct_official_pas(
    prediction: torch.Tensor,
    target_ids: np.ndarray,
    fields: SharedFields,
) -> np.ndarray:
    values: list[np.ndarray] = []
    device = fields.positions.device
    for start in range(0, len(target_ids), 16):
        end = min(start + 16, len(target_ids))
        target = torch.as_tensor(
            target_ids[start:end], dtype=torch.long, device=device
        )
        score = official_pas_cosine_from_distribution(
            grouped_to_full_pas(prediction[start:end].float()),
            fields.full_pas[target].float(),
        ).mean(dim=(1, 2))
        values.append(score.cpu().numpy().astype(np.float64))
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)


def summarize_direct_pas(
    point_pas: np.ndarray,
    table: PairTable,
) -> dict[str, object]:
    bins: list[dict[str, object]] = []
    bin_means: list[float] = []
    for bin_id, (label, _, _) in enumerate(DISTANCE_BINS):
        mask = table.group_bin_ids == bin_id
        mean = float(np.mean(point_pas[mask])) if np.any(mask) else float("nan")
        bins.append({"bin": label, "count": int(np.sum(mask)), "pas": mean})
        if np.isfinite(mean):
            bin_means.append(mean)
    return {
        "macro_pas": float(np.mean(bin_means)) if bin_means else float("nan"),
        "point_weighted_pas": float(np.mean(point_pas)) if len(point_pas) else float("nan"),
        "bins": bins,
    }


def evaluate_model_direct(
    model: LocalPASOperator,
    table: PairTable,
    field_ids: np.ndarray,
    fields: SharedFields,
) -> tuple[torch.Tensor, np.ndarray, dict[str, object]]:
    prediction = predict_grouped_pas(model, table, field_ids, fields)
    point_pas = direct_official_pas(prediction, table.group_target_ids, fields)
    return prediction, point_pas, summarize_direct_pas(point_pas, table)


def train_formal_model(
    method: str,
    train_table: PairTable,
    train_field_ids: np.ndarray,
    validation_table: PairTable | None,
    validation_field_ids: np.ndarray | None,
    fields: SharedFields,
    config: Config,
    steps: int,
    seed: int,
) -> tuple[LocalPASOperator, list[dict[str, object]], int]:
    set_seed(seed)
    representation = method_representation(method)
    model = LocalPASOperator(representation, config).to(fields.positions.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed + 17)
    history: list[dict[str, object]] = []
    rolling: list[float] = []
    best_score = -math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None

    def validate(step: int) -> None:
        nonlocal best_score, best_step, best_state
        if validation_table is None or validation_field_ids is None:
            return
        _, _, summary = evaluate_model_direct(
            model, validation_table, validation_field_ids, fields
        )
        record: dict[str, object] = {
            "step": step,
            "train_loss": float(np.mean(rolling)) if rolling else None,
            "inner_macro_pas": float(summary["macro_pas"]),
            "inner_point_weighted_pas": float(summary["point_weighted_pas"]),
            "inner_bins": summary["bins"],
        }
        history.append(record)
        score = float(summary["macro_pas"])
        if score > best_score:
            best_score = score
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"E176 formal method={method} step={step}/{steps} "
            f"macro_pas={score:.6f}",
            flush=True,
        )

    if validation_table is not None:
        validate(0)
    for step in range(1, steps + 1):
        rows = sample_pair_rows(train_table, rng, PAIR_BATCH_SIZE)
        source = torch.as_tensor(
            train_table.source_ids[rows],
            dtype=torch.long,
            device=fields.positions.device,
        )
        target = torch.as_tensor(
            train_table.target_ids[rows],
            dtype=torch.long,
            device=fields.positions.device,
        )
        field = torch.as_tensor(
            train_field_ids[rows], dtype=torch.long, device=fields.positions.device
        )
        truth = fields.grouped_pas[target]
        model.train()
        prediction = model(source, target, field, fields)
        cosine = grouped_cosine(prediction, truth)
        loss = 1.0 - cosine.mean()
        loss = loss + 0.02 * (prediction - truth).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        rolling.append(float(loss.detach().cpu()))
        if validation_table is not None and (
            step % VALIDATION_INTERVAL == 0 or step == steps
        ):
            validate(step)
            rolling.clear()
    if validation_table is not None:
        if best_state is None:
            raise RuntimeError("formal validation did not produce a checkpoint")
        model.load_state_dict(best_state)
    return model, history, best_step


def aggregate_metric_arrays(
    pas: np.ndarray,
    pdp: np.ndarray,
    squared_error: np.ndarray,
    truth_energy: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    count = int(np.sum(mask))
    if count == 0:
        return {
            "count": 0,
            "pas": float("nan"),
            "pdp": float("nan"),
            "nmse": float("nan"),
            "nmse_score": float("nan"),
            "score": float("nan"),
        }
    pas_mean = float(np.mean(pas[mask], dtype=np.float64))
    pdp_mean = float(np.mean(pdp[mask], dtype=np.float64))
    error = float(np.sum(squared_error[mask], dtype=np.float64))
    energy = float(np.sum(truth_energy[mask], dtype=np.float64))
    nmse = error / max(energy, 1e-30)
    nmse_score = 1.0 / (1.0 + nmse)
    return {
        "count": count,
        "pas": pas_mean,
        "pdp": pdp_mean,
        "nmse": nmse,
        "nmse_score": nmse_score,
        "score": 0.4 * pas_mean + 0.4 * pdp_mean + 0.2 * nmse_score,
    }


@torch.no_grad()
def projected_complete_metrics(
    predictions: dict[str, torch.Tensor],
    table: PairTable,
    fields: SharedFields,
) -> dict[str, dict[str, np.ndarray]]:
    count = len(table.group_target_ids)
    output = {
        method: {
            "pas": np.zeros(count, dtype=np.float64),
            "pdp": np.zeros(count, dtype=np.float64),
            "squared_error": np.zeros(count, dtype=np.float64),
            "truth_energy": np.zeros(count, dtype=np.float64),
        }
        for method in predictions
    }
    device = fields.positions.device
    for start in range(0, count, SCORE_BATCH_SIZE):
        end = min(start + SCORE_BATCH_SIZE, count)
        source = torch.as_tensor(
            table.group_nearest_source_ids[start:end],
            dtype=torch.long,
            device=device,
        )
        target = torch.as_tensor(
            table.group_target_ids[start:end], dtype=torch.long, device=device
        )
        source_h = angle_delay_to_channel(fields.complex_ad[source])
        truth_h = angle_delay_to_channel(fields.complex_ad[target])
        source_shaped = source_h.reshape(-1, MP, MH, MV, N_UE, S_SC)
        source_angle = torch.fft.fft2(source_shaped, dim=(2, 3), norm="ortho")
        source_flat = source_angle.reshape(-1, MP * MH * MV, N_UE, S_SC)
        source_total = source_flat.abs().square().sum(dim=1, keepdim=True)
        source_phase = torch.where(
            source_flat.abs() > 1e-20,
            source_flat / source_flat.abs().clamp_min(1e-20),
            torch.ones_like(source_flat),
        )
        truth_delay_power = torch.fft.ifft(truth_h, dim=-1).abs().square()
        truth_energy = truth_h.abs().square().sum(dim=(1, 2, 3))
        truth_pas = fields.full_pas[target].float()

        for method, prediction in predictions.items():
            desired = grouped_to_full_pas(prediction[start:end].float())
            desired_flat = desired.permute(0, 3, 1, 2).contiguous()
            magnitude = torch.sqrt(
                (desired_flat * source_total).clamp_min(0.0)
            )
            projected_angle = (source_phase * magnitude).reshape(
                -1, MP, MH, MV, N_UE, S_SC
            )
            projected_h = torch.fft.ifft2(
                projected_angle, dim=(2, 3), norm="ortho"
            ).reshape(-1, MP * MH * MV, N_UE, S_SC)

            actual_pas = projected_angle.abs().square().reshape(
                -1, MP * MH * MV, N_UE, S_SC
            ).permute(0, 2, 3, 1).contiguous()
            pas = official_pas_cosine_from_distribution(
                actual_pas, truth_pas
            ).mean(dim=(1, 2))
            predicted_delay_power = torch.fft.ifft(
                projected_h, dim=-1
            ).abs().square()
            numerator = (predicted_delay_power * truth_delay_power).sum(dim=-1)
            product = (
                predicted_delay_power.square().sum(dim=-1)
                * truth_delay_power.square().sum(dim=-1)
            )
            denominator = torch.sqrt(product)
            denominator = torch.where(
                denominator > 0.0,
                denominator,
                torch.full_like(denominator, 1e-12),
            )
            pdp = (numerator / denominator).mean(dim=(1, 2))
            squared_error = (projected_h - truth_h).abs().square().sum(
                dim=(1, 2, 3)
            )
            output[method]["pas"][start:end] = pas.cpu().numpy()
            output[method]["pdp"][start:end] = pdp.cpu().numpy()
            output[method]["squared_error"][start:end] = (
                squared_error.cpu().numpy()
            )
            output[method]["truth_energy"][start:end] = truth_energy.cpu().numpy()
    return output


def test_distance_statistics(
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
) -> dict[str, object]:
    distances = np.empty(len(test_positions), dtype=np.float64)
    per_site: list[dict[str, object]] = []
    for site_value in sorted(np.unique(train_site).tolist()):
        train_ids = np.flatnonzero(train_site == site_value)
        test_ids = np.flatnonzero(test_site == site_value)
        local = cKDTree(train_positions[train_ids, :2]).query(
            test_positions[test_ids, :2], k=1, workers=-1
        )[0]
        distances[test_ids] = local
        per_site.append(
            {
                "site": int(site_value),
                "count": int(len(local)),
                "mean_m": float(np.mean(local)),
                "median_m": float(np.median(local)),
                "p90_m": float(np.quantile(local, 0.90)),
                "max_m": float(np.max(local)),
            }
        )
    return {
        "proxy": "same-site nearest official Train distance",
        "overall": {
            "count": int(len(distances)),
            "mean_m": float(np.mean(distances)),
            "median_m": float(np.median(distances)),
            "p90_m": float(np.quantile(distances, 0.90)),
            "max_m": float(np.max(distances)),
        },
        "per_site": per_site,
        "distances_m": distances.tolist(),
    }


def fold_bin_rows(
    metrics: dict[str, dict[str, np.ndarray]],
    group_bin_ids: np.ndarray,
    group_fold_ids: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for bin_id, (label, _, _) in enumerate(DISTANCE_BINS):
            mask = (group_fold_ids == fold) & (group_bin_ids == bin_id)
            for method in METHODS:
                result = aggregate_metric_arrays(
                    metrics[method]["pas"],
                    metrics[method]["pdp"],
                    metrics[method]["squared_error"],
                    metrics[method]["truth_energy"],
                    mask,
                )
                rows.append(
                    {
                        "fold": fold,
                        "bin_id": bin_id,
                        "bin": label,
                        "method": method,
                        **result,
                    }
                )
    return rows


def overall_bin_rows(
    metrics: dict[str, dict[str, np.ndarray]],
    group_bin_ids: np.ndarray,
    fold_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bin_id, (label, _, _) in enumerate(DISTANCE_BINS):
        mask = group_bin_ids == bin_id
        for method in METHODS:
            result = aggregate_metric_arrays(
                metrics[method]["pas"],
                metrics[method]["pdp"],
                metrics[method]["squared_error"],
                metrics[method]["truth_energy"],
                mask,
            )
            fold_pas = np.asarray(
                [
                    row["pas"]
                    for row in fold_rows
                    if row["bin_id"] == bin_id and row["method"] == method
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "bin_id": bin_id,
                    "bin": label,
                    "method": method,
                    **result,
                    "fold_pas_std": float(np.nanstd(fold_pas, ddof=0)),
                    "fold_pas_min": float(np.nanmin(fold_pas)),
                    "fold_pas_max": float(np.nanmax(fold_pas)),
                }
            )
    return rows


def metric_lookup(
    rows: list[dict[str, object]],
    method: str,
    bin_id: int,
    key: str,
    fold: int | None = None,
) -> float:
    for row in rows:
        if row["method"] != method or row["bin_id"] != bin_id:
            continue
        if fold is not None and row.get("fold") != fold:
            continue
        return float(row[key])
    raise KeyError((method, bin_id, key, fold))


def distance_gate_decision(
    overall_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> dict[str, object]:
    decisions: dict[str, object] = {}
    for representation, correct, shuffled in (
        ("A", "A_noncoherent", "A_shuffled_field"),
        ("B", "B_complex", "B_shuffled_field"),
    ):
        bin_records: list[dict[str, object]] = []
        continuous = True
        radius_bin: int | None = None
        first_failed: int | None = None
        for bin_id, (label, _, _) in enumerate(DISTANCE_BINS):
            correct_pas = metric_lookup(overall_rows, correct, bin_id, "pas")
            coordinate_pas = metric_lookup(
                overall_rows, "coordinate_only", bin_id, "pas"
            )
            shuffled_pas = metric_lookup(overall_rows, shuffled, bin_id, "pas")
            mean_pas = metric_lookup(overall_rows, "neighbor_mean", bin_id, "pas")
            fold_delta = np.asarray(
                [
                    metric_lookup(fold_rows, correct, bin_id, "pas", fold)
                    - metric_lookup(
                        fold_rows, "neighbor_mean", bin_id, "pas", fold
                    )
                    for fold in range(5)
                ]
            )
            margins = {
                "versus_coordinate": correct_pas - coordinate_pas,
                "versus_shuffled": correct_pas - shuffled_pas,
                "versus_neighbor_mean": correct_pas - mean_pas,
                "neighbor_positive_folds": int(np.sum(fold_delta > 0.0)),
            }
            passed = bool(
                margins["versus_coordinate"] >= 0.05
                and margins["versus_shuffled"] >= 0.03
                and margins["versus_neighbor_mean"] >= 0.015
                and margins["neighbor_positive_folds"] >= 4
            )
            if continuous and passed:
                radius_bin = bin_id
            else:
                if continuous and not passed:
                    first_failed = bin_id
                continuous = False
            bin_records.append(
                {
                    "bin": label,
                    "pas": correct_pas,
                    **margins,
                    "passed": passed,
                }
            )
        decisions[representation] = {
            "bins": bin_records,
            "predictive_radius_bin": (
                DISTANCE_BINS[radius_bin][0] if radius_bin is not None else None
            ),
            "predictive_radius_upper_m": (
                DISTANCE_BINS[radius_bin][2]
                if radius_bin is not None and not math.isinf(DISTANCE_BINS[radius_bin][2])
                else None
            ),
            "first_failed_bin": (
                DISTANCE_BINS[first_failed][0] if first_failed is not None else None
            ),
            "one_to_two_passed": bool(bin_records[1]["passed"]),
        }

    a_radius = next(
        (
            index
            for index in reversed(range(len(DISTANCE_BINS)))
            if decisions["A"]["predictive_radius_bin"] == DISTANCE_BINS[index][0]
        ),
        -1,
    )
    b_radius = next(
        (
            index
            for index in reversed(range(len(DISTANCE_BINS)))
            if decisions["B"]["predictive_radius_bin"] == DISTANCE_BINS[index][0]
        ),
        -1,
    )
    extension_bin = a_radius + 1
    complex_extension = False
    extension_record: dict[str, object] | None = None
    if b_radius >= a_radius + 1 and 0 <= extension_bin < len(DISTANCE_BINS):
        b_minus_a = metric_lookup(
            overall_rows, "B_complex", extension_bin, "pas"
        ) - metric_lookup(overall_rows, "A_noncoherent", extension_bin, "pas")
        fold_delta = np.asarray(
            [
                metric_lookup(
                    fold_rows, "B_complex", extension_bin, "pas", fold
                )
                - metric_lookup(
                    fold_rows, "A_noncoherent", extension_bin, "pas", fold
                )
                for fold in range(5)
            ]
        )
        extension_record = {
            "bin": DISTANCE_BINS[extension_bin][0],
            "B_minus_A_pas": float(b_minus_a),
            "positive_folds": int(np.sum(fold_delta > 0.0)),
        }
        complex_extension = bool(
            b_minus_a >= 0.015 and int(np.sum(fold_delta > 0.0)) >= 4
        )
    decisions["complex_extends_radius"] = complex_extension
    decisions["complex_extension_test"] = extension_record
    return decisions


def write_curve_csvs(
    overall_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    overall_fields = [
        "bin_id",
        "bin",
        "method",
        "count",
        "pas",
        "pdp",
        "nmse",
        "nmse_score",
        "score",
        "fold_pas_std",
        "fold_pas_min",
        "fold_pas_max",
    ]
    with CURVE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=overall_fields)
        writer.writeheader()
        writer.writerows(overall_rows)
    fold_fields = [
        "fold",
        "bin_id",
        "bin",
        "method",
        "count",
        "pas",
        "pdp",
        "nmse",
        "nmse_score",
        "score",
    ]
    with FOLD_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(fold_rows)


def plot_curves(
    overall_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(DISTANCE_BINS), dtype=np.float64)
    labels = [value[0] for value in DISTANCE_BINS]
    style = {
        "coordinate_only": ("Coordinate only", "#6b7280", "--"),
        "neighbor_mean": ("Neighbor mean", "#111827", ":"),
        "A_shuffled_field": ("A shuffled", "#f59e0b", "--"),
        "A_noncoherent": ("A energy AD", "#d97706", "-"),
        "B_shuffled_field": ("B shuffled", "#60a5fa", "--"),
        "B_complex": ("B complex AD", "#2563eb", "-"),
    }
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for method, (name, color, line_style) in style.items():
        mean = np.asarray(
            [metric_lookup(overall_rows, method, bin_id, "pas") for bin_id in range(6)]
        )
        std = np.asarray(
            [
                np.std(
                    [
                        metric_lookup(fold_rows, method, bin_id, "pas", fold)
                        for fold in range(5)
                    ]
                )
                for bin_id in range(6)
            ]
        )
        axis.errorbar(
            x,
            mean,
            yerr=std,
            label=name,
            color=color,
            linestyle=line_style,
            marker="o",
            capsize=3,
            linewidth=2,
        )
    axis.set_xticks(x, labels)
    axis.set_xlabel("Source-target distance")
    axis.set_ylabel("Official PAS")
    axis.set_title("Round2 E176: spatial PAS predictability by distance")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(CURVE_PNG, dpi=180)
    plt.close(figure)

    delta_mean = np.asarray(
        [
            metric_lookup(overall_rows, "B_complex", bin_id, "pas")
            - metric_lookup(overall_rows, "A_noncoherent", bin_id, "pas")
            for bin_id in range(6)
        ]
    )
    delta_fold = np.asarray(
        [
            [
                metric_lookup(fold_rows, "B_complex", bin_id, "pas", fold)
                - metric_lookup(
                    fold_rows, "A_noncoherent", bin_id, "pas", fold
                )
                for fold in range(5)
            ]
            for bin_id in range(6)
        ]
    )
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.errorbar(
        x,
        delta_mean,
        yerr=np.std(delta_fold, axis=1),
        color="#2563eb",
        marker="o",
        capsize=3,
        linewidth=2,
    )
    axis.axhline(0.0, color="#111827", linewidth=1)
    axis.axhline(0.015, color="#dc2626", linewidth=1, linestyle="--")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Source-target distance")
    axis.set_ylabel("Official PAS: B complex minus A energy")
    axis.set_title("Round2 E176: complex representation margin")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(COMPLEX_MARGIN_PNG, dpi=180)
    plt.close(figure)


def formal_run(
    device: torch.device,
    config: Config,
    setup: object,
    positions: np.ndarray,
    test_positions: np.ndarray,
    site: np.ndarray,
    test_site: np.ndarray,
    fields: SharedFields,
) -> dict[str, object]:
    formal_started = time.time()
    broad_fold = np.asarray(np.load(FOLD_IDS), dtype=np.int8)
    grouped_pas_np = np.asarray(np.load(GROUPED_PAS, mmap_mode="r"), dtype=np.float32)
    nonzero = np.square(grouped_pas_np).sum(axis=(1, 2)) > TINY
    anchor_mask = checkerboard_anchor_mask(positions)

    pair_started = time.time()
    fold_tables = [
        build_fold_pair_table(
            fold, positions, site, broad_fold, anchor_mask, nonzero
        )
        for fold in range(5)
    ]
    pair_seconds = time.time() - pair_started
    fold_audits = []
    for fold, table in enumerate(fold_tables):
        audit = pair_table_audit(table)
        audit.update(
            {
                "fold": fold,
                "anchors_all": int(np.sum((broad_fold == fold) & anchor_mask)),
                "targets_all": int(np.sum((broad_fold == fold) & ~anchor_mask)),
                "anchors_nonzero": int(
                    np.sum((broad_fold == fold) & anchor_mask & nonzero)
                ),
                "targets_nonzero": int(
                    np.sum((broad_fold == fold) & ~anchor_mask & nonzero)
                ),
            }
        )
        fold_audits.append(audit)

    training_started = time.time()
    training_records: list[dict[str, object]] = []
    outer_predictions: list[dict[str, torch.Tensor]] = []
    outer_tables: list[PairTable] = []
    trainable_methods = METHODS[:5]
    for outer in range(5):
        inner = (outer + 1) % 5
        inner_train_folds = [fold for fold in range(5) if fold not in (outer, inner)]
        retrain_folds = [fold for fold in range(5) if fold != outer]
        inner_train = combine_pair_tables([fold_tables[fold] for fold in inner_train_folds])
        inner_validation = fold_tables[inner]
        retrain = combine_pair_tables([fold_tables[fold] for fold in retrain_folds])
        held = fold_tables[outer]
        inner_train_shuffle = shuffled_field_ids(
            inner_train, config.seed + outer * 100003 + 101
        )
        inner_validation_shuffle = shuffled_field_ids(
            inner_validation, config.seed + outer * 100003 + 202
        )
        retrain_shuffle = shuffled_field_ids(
            retrain, config.seed + outer * 100003 + 303
        )
        held_shuffle = shuffled_field_ids(
            held, config.seed + outer * 100003 + 404
        )
        held_predictions: dict[str, torch.Tensor] = {}

        for method_index, method in enumerate(trainable_methods):
            train_fields = method_field_ids(
                method, inner_train, inner_train_shuffle
            )
            validation_fields = method_field_ids(
                method, inner_validation, inner_validation_shuffle
            )
            seed = config.seed + outer * 10007 + method_index * 1009
            selection_model, history, selected_step = train_formal_model(
                method,
                inner_train,
                train_fields,
                inner_validation,
                validation_fields,
                fields,
                config,
                FORMAL_STEPS,
                seed,
            )
            del selection_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            retrain_fields = method_field_ids(method, retrain, retrain_shuffle)
            model, retrain_history, _ = train_formal_model(
                method,
                retrain,
                retrain_fields,
                None,
                None,
                fields,
                config,
                selected_step,
                seed + 1,
            )
            held_fields = method_field_ids(method, held, held_shuffle)
            prediction, point_pas, direct_summary = evaluate_model_direct(
                model, held, held_fields, fields
            )
            held_predictions[method] = prediction
            checkpoint = ARTIFACT_DIR / f"outer{outer}_{method}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "outer": outer,
                    "inner": inner,
                    "method": method,
                    "selected_step": selected_step,
                },
                checkpoint,
            )
            training_records.append(
                {
                    "outer": outer,
                    "inner": inner,
                    "method": method,
                    "selected_step": selected_step,
                    "selection_history": history,
                    "retrain_history": retrain_history,
                    "held_direct": direct_summary,
                    "held_direct_pas_mean": float(np.mean(point_pas)),
                    "checkpoint": PATHS.relative(checkpoint),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        neighbor_mean, nearest_copy = baseline_grouped_pas(held, fields)
        held_predictions["neighbor_mean"] = neighbor_mean
        held_predictions["nearest_copy"] = nearest_copy
        outer_predictions.append(held_predictions)
        outer_tables.append(held)
        print(f"E176 formal outer={outer} training complete", flush=True)
    training_seconds = time.time() - training_started

    scoring_started = time.time()
    metric_parts: dict[str, dict[str, list[np.ndarray]]] = {
        method: {
            "pas": [],
            "pdp": [],
            "squared_error": [],
            "truth_energy": [],
        }
        for method in METHODS
    }
    metadata_parts: dict[str, list[np.ndarray]] = {
        "target_ids": [],
        "bin_ids": [],
        "fold_ids": [],
        "site_ids": [],
        "nearest_source_ids": [],
        "distance_median": [],
        "distance_p90": [],
    }
    for outer, (predictions, table) in enumerate(zip(outer_predictions, outer_tables)):
        local = projected_complete_metrics(predictions, table, fields)
        for method in METHODS:
            for key in metric_parts[method]:
                metric_parts[method][key].append(local[method][key])
        metadata_parts["target_ids"].append(table.group_target_ids)
        metadata_parts["bin_ids"].append(table.group_bin_ids)
        metadata_parts["fold_ids"].append(table.group_fold_ids)
        metadata_parts["site_ids"].append(table.group_site_ids)
        metadata_parts["nearest_source_ids"].append(
            table.group_nearest_source_ids
        )
        metadata_parts["distance_median"].append(table.group_distance_median)
        metadata_parts["distance_p90"].append(table.group_distance_p90)
        print(f"E176 formal outer={outer} complete-H scoring complete", flush=True)
    metrics = {
        method: {
            key: np.concatenate(values)
            for key, values in method_values.items()
        }
        for method, method_values in metric_parts.items()
    }
    metadata = {
        key: np.concatenate(values) for key, values in metadata_parts.items()
    }
    scoring_seconds = time.time() - scoring_started

    fold_rows = fold_bin_rows(metrics, metadata["bin_ids"], metadata["fold_ids"])
    overall_rows = overall_bin_rows(metrics, metadata["bin_ids"], fold_rows)
    decision = distance_gate_decision(overall_rows, fold_rows)
    test_stats = test_distance_statistics(
        positions, test_positions, site, test_site
    )
    radius_values: list[float] = []
    radius_exceeds_10 = False
    for representation in ("A", "B"):
        label = decision[representation]["predictive_radius_bin"]
        upper = decision[representation]["predictive_radius_upper_m"]
        if label == ">10m":
            radius_exceeds_10 = True
        elif upper is not None:
            radius_values.append(float(upper))
    max_radius = max(radius_values, default=None)
    decision["test_depth_comparison"] = {
        "test_nearest_train_median_m": test_stats["overall"]["median_m"],
        "largest_predictive_radius_upper_m": max_radius,
        "predictive_radius_exceeds_10m": radius_exceeds_10,
        "test_median_exceeds_radius": (
            False
            if radius_exceeds_10
            else (
                bool(test_stats["overall"]["median_m"] > max_radius)
                if max_radius is not None
                else True
            )
        ),
    }

    write_curve_csvs(overall_rows, fold_rows)
    plot_curves(overall_rows, fold_rows)
    np.savez_compressed(
        FORMAL_METRICS,
        **metadata,
        **{
            f"{method}_{key}": value
            for method, method_values in metrics.items()
            for key, value in method_values.items()
        },
    )
    return {
        "status": "formal_complete",
        "split": {
            "broad_fold_path": PATHS.relative(FOLD_IDS),
            "checkerboard_cell_m": CHECKERBOARD_CELL_M,
            "anchor_formula": "(floor(x/0.5)+2*floor(y/0.5))%2==0",
            "source_target_nonzero_only": True,
            "folds": fold_audits,
        },
        "training": training_records,
        "overall_distance_curve": overall_rows,
        "fold_distance_curve": fold_rows,
        "decision": decision,
        "test_distance_statistics": test_stats,
        "outputs": {
            "curve_csv": PATHS.relative(CURVE_CSV),
            "fold_csv": PATHS.relative(FOLD_CSV),
            "curve_png": PATHS.relative(CURVE_PNG),
            "complex_margin_png": PATHS.relative(COMPLEX_MARGIN_PNG),
            "metrics_npz": PATHS.relative(FORMAL_METRICS),
        },
        "timing_seconds": {
            "pair_build": pair_seconds,
            "training": training_seconds,
            "complete_h_scoring": scoring_seconds,
            "formal_total": time.time() - formal_started,
        },
    }


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

    required = (TRAIN_CHANNEL, TRAIN_PAS, GROUPED_PAS, GROUPED_PDP, AD_CACHE)
    if args.stage in ("formal", "all"):
        required = (*required, FOLD_IDS)
    missing = [PATHS.relative(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing E176 inputs: {missing}")

    cache = build_complex_ad_cache(device)
    if args.stage == "cache":
        print(json.dumps(cache, ensure_ascii=False, indent=2), flush=True)
        return

    loading_started = time.time()
    setup, positions, test_positions, site, test_site = load_positions()
    positions = positions.astype(np.float64)
    test_positions = test_positions.astype(np.float64)
    site = site.astype(np.int8)
    test_site = test_site.astype(np.int8)
    fields = load_shared_fields(device, positions, site, setup.bs_positions)
    data_loading_seconds = time.time() - loading_started

    capacity: dict[str, object] | None = None
    existing: dict[str, object] = {}
    if REPORT.is_file():
        try:
            existing = json.loads(REPORT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    if args.stage in ("capacity", "all"):
        capacity = capacity_gate(device, config, positions, site, fields)
    elif args.stage == "formal":
        previous_capacity = existing.get("capacity")
        if not isinstance(previous_capacity, dict) or not previous_capacity.get("passed"):
            raise RuntimeError("formal stage requires the recorded passed capacity gate")
        capacity = previous_capacity

    formal: dict[str, object] | None = None
    if args.stage in ("formal", "all"):
        if capacity is None or not capacity.get("passed"):
            raise RuntimeError("capacity gate failed; formal distance curve is blocked")
        formal = formal_run(
            device,
            config,
            setup,
            positions,
            test_positions,
            site,
            test_site,
            fields,
        )

    timing_candidates = {
        "data loading and host-to-device transfer": data_loading_seconds,
    }
    if formal is not None:
        timing_candidates.update(
            {
                "CPU pair construction": float(
                    formal["timing_seconds"]["pair_build"]
                ),
                "GPU model training": float(formal["timing_seconds"]["training"]),
                "GPU complete-H scoring": float(
                    formal["timing_seconds"]["complete_h_scoring"]
                ),
            }
        )
    measured_bottleneck = max(timing_candidates, key=timing_candidates.get)
    payload = {
        "experiment": VERSION,
        "status": (
            formal["status"]
            if formal is not None
            else ("capacity_passed" if capacity and capacity["passed"] else "capacity_failed")
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "config": asdict(config),
        "cache": cache,
        "capacity": capacity,
        "formal": formal,
        "seconds": float(time.time() - started),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else 0,
        "data_loading_seconds": data_loading_seconds,
        "timing_candidates_seconds": timing_candidates,
        "measured_bottleneck": measured_bottleneck,
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
