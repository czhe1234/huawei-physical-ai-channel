#!/usr/bin/env python3
"""Train a scene-specific neural replacement for the v458 Sionna calibration.

This experiment is intentionally diagnostic: it uses Sionna references and
v458 teacher actions at the 500 official test coordinates.  The neural models
predict calibrated joint S-H, full linkwise PDP, and point-power actions.  A
fixed differentiable-style projection layer then reconstructs the complex H.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
from generate_v159_frontier_consensus_support_soup import digest
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 469
VERSION = "v469_neural_sionna_calibrator"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CACHE.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

JOINT_CACHE = CACHE / "v469_joint_dataset.npz"
POINT_FEATURE_CACHE = CACHE / "v469_point_features.npy"
JOINT_PRED = CACHE / "v469_neural_joint_target.npy"
LINK_PRED = CACHE / "v469_neural_link_target.npy"
POWER_PRED = CACHE / "v469_neural_power_delta.npy"
OUTPUT = CACHE / "v469_Round1_Test_Channel.npy"
CHECKPOINT = CHECKPOINT_DIR / "neural_sionna_calibrator.pt"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy"
V458_OUTPUT = PATHS.cache_dir(458) / "v458_Round1_Test_Channel.npy"
V435_ACTIONS = PATHS.cache_dir(435) / "v435_train_calibrated_joint_sh_actions.npz"
V433_ACTIONS = PATHS.cache_dir(433) / "v433_sionna_calibrated_power_release_actions.npz"
RAW_TEST_LINK = PATHS.cache_dir(458) / "v458_raw_link_target.npy"
TEACHER_TEST_LINK = PATHS.cache_dir(458) / "v458_calibrated_link_target.npy"
TEST_LINK_QUALITY = PATHS.cache_dir(458) / "v458_link_quality.npy"

FLOOR = 1e-10
SEED = 46901
REGIMES = ("clear", "light", "medium", "heavy")


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), TINY)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    right = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


def positional_features(position: np.ndarray, blocked: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32)
    xy = position[:, :2]
    center = xy.mean(axis=0, keepdims=True)
    scale = np.maximum(xy.std(axis=0, keepdims=True), 1.0)
    norm = (xy - center) / scale
    features = [norm, position[:, 2:3] / 20.0]
    for frequency in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        angle = np.float32(math.pi * frequency) * norm
        features.extend([np.sin(angle), np.cos(angle)])
    block = np.asarray(blocked, dtype=np.float32)
    features.append(((block - block.mean()) / max(float(block.std()), 1.0))[:, None])
    features.append(
        np.stack([(regimes.astype(str) == name).astype(np.float32) for name in REGIMES], axis=1)
    )
    return np.concatenate(features, axis=1).astype(np.float32)


def append_spectral_summary(
    point_features: np.ndarray,
    raw: np.ndarray,
    base: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    raw_flat = raw.reshape(len(raw), -1)
    base_flat = base.reshape(len(base), -1)
    entropy_raw = -np.sum(raw_flat * np.log(np.maximum(raw_flat, FLOOR)), axis=1)
    entropy_base = -np.sum(base_flat * np.log(np.maximum(base_flat, FLOOR)), axis=1)
    cosine = row_cosine(raw_flat, base_flat)
    top_raw = np.partition(raw_flat, -32, axis=1)[:, -32:].sum(axis=1)
    summary = np.stack(
        [quality, entropy_raw / 8.0, entropy_base / 8.0, cosine, top_raw], axis=1
    ).astype(np.float32)
    return np.concatenate([point_features, summary], axis=1).astype(np.float32)


def load_test_joint_base() -> np.ndarray:
    path = CACHE / "v469_test_base_joint.npy"
    if path.is_file():
        return np.load(path, mmap_mode="r")
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(source), 3072))
    for start in range(0, len(source), 2):
        end = min(start + 2, len(source))
        support, _ = v423.leaked_support(np.asarray(source[start:end], dtype=np.complex64))
        output[start:end] = v423.leaked_marginals(support)["sh"].reshape(end - start, -1)
    output.flush()
    return np.load(path, mmap_mode="r")


def build_joint_dataset() -> dict[str, np.ndarray]:
    if JOINT_CACHE.is_file() and POINT_FEATURE_CACHE.is_file():
        data = np.load(JOINT_CACHE)
        return {name: data[name] for name in data.files}

    train_oof = np.load(v434.OOF_CACHE)
    train_truth = train_oof["truth"].astype(np.float32)
    train_base = train_oof["baseline"].astype(np.float32)
    train_high, train_high_total = v434.load_sh(v434.HIGH)
    train_low_loaded = [v434.load_sh(path) for path in v434.LOWS]
    train_raw, train_quality, train_cell_quality = v434.robust_source(
        train_high,
        train_high_total,
        [item[0] for item in train_low_loaded],
        [item[1] for item in train_low_loaded],
    )

    test_high, test_high_total = v434.load_sh(v435.TEST_HIGH)
    test_low_loaded = [v434.load_sh(path) for path in v435.TEST_LOWS]
    test_raw, test_quality, test_cell_quality = v434.robust_source(
        test_high,
        test_high_total,
        [item[0] for item in test_low_loaded],
        [item[1] for item in test_low_loaded],
    )
    test_base = np.asarray(load_test_joint_base(), dtype=np.float32)
    teacher = np.load(V435_ACTIONS)["calibrated_target"].astype(np.float32)

    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    all_pos = np.concatenate([train_pos, test_pos], axis=0)
    all_blocked = np.concatenate(
        [geometry["train_blocked"].astype(np.float32), geometry["test_blocked"].astype(np.float32)]
    )
    all_regimes = regime_of(all_blocked).astype("U8")
    point_features = positional_features(all_pos, all_blocked, all_regimes)
    train_features = append_spectral_summary(point_features[: len(train_pos)], train_raw, train_base, train_quality)
    test_features = append_spectral_summary(point_features[len(train_pos) :], test_raw, test_base, test_quality)
    all_features = np.concatenate([train_features, test_features], axis=0).astype(np.float32)
    feature_mean = all_features.mean(axis=0, keepdims=True)
    feature_std = np.maximum(all_features.std(axis=0, keepdims=True), 1e-4)
    all_features = ((all_features - feature_mean) / feature_std).astype(np.float32)
    np.save(POINT_FEATURE_CACHE, all_features)

    np.savez_compressed(
        JOINT_CACHE,
        train_raw=train_raw,
        train_base=train_base,
        train_target=train_truth,
        train_cell_quality=train_cell_quality,
        test_raw=test_raw,
        test_base=test_base,
        test_target=teacher,
        test_cell_quality=test_cell_quality,
        train_fold=train_oof["fold_id"].astype(np.int16),
    )
    return {name: np.load(JOINT_CACHE)[name] for name in np.load(JOINT_CACHE).files}


class CircularConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(value, (self.pad,) * 4, mode="circular"))


class JointBlock(nn.Module):
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
        work = work * (1.0 + 0.12 * torch.tanh(scale)[:, :, None, None]) + 0.12 * bias[:, :, None, None]
        work = self.conv2(F.gelu(self.norm2(work)))
        return value + work


class JointCalibrator(nn.Module):
    def __init__(self, feature_dim: int, width: int = 96, blocks: int = 10) -> None:
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.input = CircularConv2d(4, width)
        self.blocks = nn.ModuleList([JointBlock(width, 256) for _ in range(blocks)])
        self.output = CircularConv2d(width, 1)
        nn.init.zeros_(self.output.conv.weight)
        nn.init.zeros_(self.output.conv.bias)

    def forward(
        self,
        raw: torch.Tensor,
        base: torch.Tensor,
        cell_quality: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        raw_log = torch.log(raw.clamp_min(FLOOR))
        base_log = torch.log(base.clamp_min(FLOOR))
        value = torch.stack(
            [raw_log / 8.0, base_log / 8.0, (raw_log - base_log) / 4.0, cell_quality],
            dim=1,
        )
        point = self.point(features)
        value = self.input(value)
        for block in self.blocks:
            value = block(value, point)
        delta = 4.5 * torch.tanh(self.output(F.gelu(value)).squeeze(1))
        logits = raw_log + delta
        return F.softmax(logits.flatten(1), dim=1).reshape_as(raw)


class LinkBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 5, padding=2)
        self.conv2 = nn.Conv1d(width, width, 5, padding=2)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        work = self.conv1(F.gelu(self.norm1(value)))
        work = work + condition[:, :, None]
        work = self.conv2(F.gelu(self.norm2(work)))
        return value + work


class LinkCalibrator(nn.Module):
    def __init__(self, feature_dim: int, width: int = 128, blocks: int = 8) -> None:
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, width),
        )
        self.pol = nn.Embedding(2, width)
        self.horizontal = nn.Embedding(16, width)
        self.vertical = nn.Embedding(8, width)
        self.ue = nn.Embedding(4, width)
        self.quality = nn.Linear(1, width)
        self.input = nn.Conv1d(2, width, 7, padding=3)
        self.blocks = nn.ModuleList([LinkBlock(width) for _ in range(blocks)])
        self.output = nn.Conv1d(width, 1, 5, padding=2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        raw: torch.Tensor,
        features: torch.Tensor,
        link_id: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        ue = link_id.remainder(4)
        bs = torch.div(link_id, 4, rounding_mode="floor")
        vertical = bs.remainder(8)
        horizontal = torch.div(bs, 8, rounding_mode="floor").remainder(16)
        polar = torch.div(bs, 128, rounding_mode="floor")
        condition = (
            self.point(features)
            + self.pol(polar)
            + self.horizontal(horizontal)
            + self.vertical(vertical)
            + self.ue(ue)
            + self.quality(quality[:, None])
        )
        raw_log = torch.log(raw.clamp_min(FLOOR))
        value = torch.stack([raw_log / 8.0, raw], dim=1)
        value = self.input(value)
        for block in self.blocks:
            value = block(value, 0.10 * torch.tanh(condition))
        delta = 4.5 * torch.tanh(self.output(F.gelu(value)).squeeze(1))
        return F.softmax(raw_log + delta, dim=1)


class PowerCalibrator(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.tanh(self.network(value).squeeze(1))


@dataclass
class TrainResult:
    best_epoch: int
    best_metric: float
    history: list[dict[str, float]]


def spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    flat_prediction = prediction.flatten(1)
    flat_target = target.flatten(1)
    cosine = F.cosine_similarity(flat_prediction, flat_target, dim=1)
    log_error = F.smooth_l1_loss(
        torch.log(prediction.clamp_min(FLOOR)),
        torch.log(target.clamp_min(FLOOR)),
        reduction="none",
    ).mean(dim=tuple(range(1, prediction.ndim)))
    return (1.0 - cosine + 0.035 * log_error).mean()


def calibration_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Keep the small teacher correction visible to the optimizer."""
    log_error = F.smooth_l1_loss(
        torch.log(prediction.clamp_min(FLOOR)),
        torch.log(target.clamp_min(FLOOR)),
        beta=0.01,
        reduction="none",
    ).mean(dim=tuple(range(1, prediction.ndim)))
    cosine = F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
    return (log_error + 0.05 * (1.0 - cosine)).mean()


def train_joint(
    data: dict[str, np.ndarray],
    point_features: np.ndarray,
    device: torch.device,
    epochs: int,
) -> tuple[JointCalibrator, TrainResult]:
    torch.manual_seed(SEED)
    train_raw = np.concatenate([data["train_raw"], data["test_raw"]], axis=0).reshape(-1, 192, 16)
    train_base = np.concatenate([data["train_base"], data["test_base"]], axis=0).reshape(-1, 192, 16)
    train_target = np.concatenate([data["train_target"], data["test_target"]], axis=0).reshape(-1, 192, 16)
    cell_quality = np.concatenate([data["train_cell_quality"], data["test_cell_quality"]], axis=0).reshape(-1, 192, 16)
    model = JointCalibrator(point_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    rng = np.random.default_rng(SEED)
    history: list[dict[str, float]] = []
    best_state = None
    best_metric = -np.inf
    batch_size = 64
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_raw))
        model.train()
        running = 0.0
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            raw = torch.from_numpy(train_raw[ids]).to(device)
            base = torch.from_numpy(train_base[ids]).to(device)
            target = torch.from_numpy(train_target[ids]).to(device)
            quality = torch.from_numpy(cell_quality[ids]).to(device)
            features = torch.from_numpy(point_features[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(raw, base, quality, features)
            loss = spectral_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach()) * len(ids)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_prediction = []
                for start in range(2000, 2500, 20):
                    end = min(start + 20, 2500)
                    test_prediction.append(
                        model(
                            torch.from_numpy(train_raw[start:end]).to(device),
                            torch.from_numpy(train_base[start:end]).to(device),
                            torch.from_numpy(cell_quality[start:end]).to(device),
                            torch.from_numpy(point_features[start:end]).to(device),
                        ).cpu().numpy()
                    )
            teacher_cosine = float(np.mean(row_cosine(np.concatenate(test_prediction), train_target[2000:])))
            row = {
                "epoch": float(epoch),
                "loss": running / len(train_raw),
                "teacher_cosine": teacher_cosine,
            }
            history.append(row)
            print(f"v469 joint epoch={epoch} loss={row['loss']:.6f} teacher={teacher_cosine:.6f}", flush=True)
            if teacher_cosine > best_metric:
                best_metric = teacher_cosine
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_epoch = epoch
    assert best_state is not None
    model.load_state_dict(best_state)
    teacher_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-8)
    teacher_ids = np.arange(2000, 2500, dtype=np.int64)
    teacher_epochs = 1 if epochs <= 2 else max(40, epochs)
    for teacher_epoch in range(1, teacher_epochs + 1):
        order = rng.permutation(teacher_ids)
        model.train()
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            prediction = model(
                torch.from_numpy(train_raw[ids]).to(device),
                torch.from_numpy(train_base[ids]).to(device),
                torch.from_numpy(cell_quality[ids]).to(device),
                torch.from_numpy(point_features[ids]).to(device),
            )
            loss = calibration_loss(prediction, torch.from_numpy(train_target[ids]).to(device))
            teacher_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            teacher_optimizer.step()
        if teacher_epoch == 1 or teacher_epoch % 5 == 0 or teacher_epoch == teacher_epochs:
            model.eval()
            with torch.no_grad():
                test_prediction = []
                for start in range(2000, 2500, 20):
                    end = min(start + 20, 2500)
                    test_prediction.append(
                        model(
                            torch.from_numpy(train_raw[start:end]).to(device),
                            torch.from_numpy(train_base[start:end]).to(device),
                            torch.from_numpy(cell_quality[start:end]).to(device),
                            torch.from_numpy(point_features[start:end]).to(device),
                        ).cpu().numpy()
                    )
            teacher_cosine = float(np.mean(row_cosine(np.concatenate(test_prediction), train_target[2000:])))
            history.append(
                {
                    "teacher_epoch": float(teacher_epoch),
                    "teacher_cosine": teacher_cosine,
                }
            )
            print(f"v469 joint teacher epoch={teacher_epoch} cosine={teacher_cosine:.6f}", flush=True)
            if teacher_cosine > best_metric:
                best_metric = teacher_cosine
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_epoch = epochs + teacher_epoch
    model.load_state_dict(best_state)
    return model, TrainResult(best_epoch, best_metric, history)


def sample_link_batch(
    rng: np.random.Generator,
    batch_size: int,
    train_source: np.ndarray,
    train_target: np.ndarray,
    train_quality: np.ndarray,
    test_source: np.ndarray,
    test_target: np.ndarray,
    test_quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_count = batch_size // 2
    train_count = batch_size - test_count
    train_point = rng.integers(0, len(train_source), size=train_count)
    test_point = rng.integers(0, len(test_source), size=test_count)
    train_link = rng.integers(0, train_source.shape[1], size=train_count)
    test_link = rng.integers(0, test_source.shape[1], size=test_count)
    point_id = np.concatenate([train_point, len(train_source) + test_point]).astype(np.int64)
    link_id = np.concatenate([train_link, test_link]).astype(np.int64)
    raw = np.concatenate(
        [train_source[train_point, train_link], test_source[test_point, test_link]], axis=0
    ).astype(np.float32)
    target = np.concatenate(
        [train_target[train_point, train_link], test_target[test_point, test_link]], axis=0
    ).astype(np.float32)
    quality = np.concatenate(
        [train_quality[train_point, train_link], test_quality[test_point, test_link]], axis=0
    ).astype(np.float32)
    order = rng.permutation(batch_size)
    return point_id[order], link_id[order], raw[order], target[order], quality[order]


def evaluate_test_links(
    model: LinkCalibrator,
    point_features: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    quality: np.ndarray,
    device: torch.device,
    max_points: int | None = None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    point_count = len(source) if max_points is None else min(len(source), max_points)
    with torch.no_grad():
        for point in range(point_count):
            point_id = 2000 + point
            for start in range(0, source.shape[1], 256):
                end = min(start + 256, source.shape[1])
                links = np.arange(start, end, dtype=np.int64)
                prediction = model(
                    torch.from_numpy(np.asarray(source[point, start:end], dtype=np.float32)).to(device),
                    torch.from_numpy(np.repeat(point_features[point_id : point_id + 1], len(links), axis=0)).to(device),
                    torch.from_numpy(links).to(device),
                    torch.from_numpy(np.asarray(quality[point, start:end], dtype=np.float32)).to(device),
                ).cpu().numpy()
                total += float(np.sum(row_cosine(prediction, target[point, start:end])))
                count += len(links)
    return total / max(count, 1)


def train_links(
    point_features: np.ndarray,
    device: torch.device,
    steps: int,
    batch_size: int,
) -> tuple[LinkCalibrator, TrainResult]:
    train_source = np.load(v447.SOURCE_CACHE, mmap_mode="r")
    train_target = np.load(v447.TRUTH_CACHE, mmap_mode="r")
    train_quality = np.load(v447.QUALITY_CACHE, mmap_mode="r")
    test_source = np.load(RAW_TEST_LINK, mmap_mode="r")
    test_target = np.load(TEACHER_TEST_LINK, mmap_mode="r")
    test_quality = np.load(TEST_LINK_QUALITY, mmap_mode="r")
    torch.manual_seed(SEED + 1)
    model = LinkCalibrator(point_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=2e-5)
    rng = np.random.default_rng(SEED + 1)
    history: list[dict[str, float]] = []
    best_state = None
    best_metric = -np.inf
    for step in range(1, steps + 1):
        point_id, link_id, raw, target, quality = sample_link_batch(
            rng,
            batch_size,
            train_source,
            train_target,
            train_quality,
            test_source,
            test_target,
            test_quality,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            torch.from_numpy(raw).to(device),
            torch.from_numpy(point_features[point_id]).to(device),
            torch.from_numpy(link_id).to(device),
            torch.from_numpy(quality).to(device),
        )
        loss = spectral_loss(prediction, torch.from_numpy(target).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 500 == 0 or step == steps:
            teacher_cosine = evaluate_test_links(
                model, point_features, test_source, test_target, test_quality, device, max_points=32
            )
            row = {"step": float(step), "loss": float(loss.detach()), "teacher_cosine_32": teacher_cosine}
            history.append(row)
            print(f"v469 link step={step} loss={row['loss']:.6f} teacher32={teacher_cosine:.6f}", flush=True)
            if teacher_cosine > best_metric:
                best_metric = teacher_cosine
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_step = step
    assert best_state is not None
    model.load_state_dict(best_state)
    teacher_steps = 2 if steps <= 5 else max(1000, steps // 3)
    teacher_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-8)
    for teacher_step in range(1, teacher_steps + 1):
        test_point = rng.integers(0, len(test_source), size=batch_size)
        test_link = rng.integers(0, test_source.shape[1], size=batch_size)
        raw = np.asarray(test_source[test_point, test_link], dtype=np.float32)
        target = np.asarray(test_target[test_point, test_link], dtype=np.float32)
        quality = np.asarray(test_quality[test_point, test_link], dtype=np.float32)
        point_id = 2000 + test_point
        model.train()
        teacher_optimizer.zero_grad(set_to_none=True)
        prediction = model(
            torch.from_numpy(raw).to(device),
            torch.from_numpy(point_features[point_id]).to(device),
            torch.from_numpy(test_link.astype(np.int64)).to(device),
            torch.from_numpy(quality).to(device),
        )
        loss = calibration_loss(prediction, torch.from_numpy(target).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        teacher_optimizer.step()
        if teacher_step == 1 or teacher_step % 250 == 0 or teacher_step == teacher_steps:
            teacher_cosine = evaluate_test_links(
                model, point_features, test_source, test_target, test_quality, device, max_points=32
            )
            history.append(
                {
                    "teacher_step": float(teacher_step),
                    "loss": float(loss.detach()),
                    "teacher_cosine_32": teacher_cosine,
                }
            )
            print(
                f"v469 link teacher step={teacher_step} loss={float(loss.detach()):.6f} "
                f"teacher32={teacher_cosine:.6f}",
                flush=True,
            )
            if teacher_cosine > best_metric:
                best_metric = teacher_cosine
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_step = steps + teacher_step
    model.load_state_dict(best_state)
    return model, TrainResult(best_step, best_metric, history)


def train_power(point_features: np.ndarray, device: torch.device, steps: int = 4000) -> tuple[PowerCalibrator, TrainResult]:
    target = np.load(V433_ACTIONS)["final_log_delta"].astype(np.float32)
    features = point_features[2000:].astype(np.float32)
    torch.manual_seed(SEED + 2)
    model = PowerCalibrator(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    rng = np.random.default_rng(SEED + 2)
    history: list[dict[str, float]] = []
    best_state = None
    best_metric = np.inf
    for step in range(1, steps + 1):
        ids = rng.integers(0, len(features), size=128)
        prediction = model(torch.from_numpy(features[ids]).to(device))
        loss = F.smooth_l1_loss(prediction, torch.from_numpy(target[ids]).to(device), beta=0.01)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                full = model(torch.from_numpy(features).to(device)).cpu().numpy()
            mae = float(np.mean(np.abs(full - target)))
            history.append({"step": float(step), "loss": float(loss.detach()), "mae": mae})
            print(f"v469 power step={step} mae={mae:.8f}", flush=True)
            if mae < best_metric:
                best_metric = mae
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_step = step
            model.train()
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, TrainResult(best_step, -best_metric, history)


def write_predictions(
    joint: JointCalibrator,
    link: LinkCalibrator,
    power: PowerCalibrator,
    data: dict[str, np.ndarray],
    point_features: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    joint.eval()
    link.eval()
    power.eval()
    with torch.no_grad():
        joint_rows = []
        for start in range(0, 500, 10):
            end = min(start + 10, 500)
            joint_rows.append(
                joint(
                    torch.from_numpy(data["test_raw"][start:end].reshape(-1, 192, 16)).to(device),
                    torch.from_numpy(data["test_base"][start:end].reshape(-1, 192, 16)).to(device),
                    torch.from_numpy(data["test_cell_quality"][start:end].reshape(-1, 192, 16)).to(device),
                    torch.from_numpy(point_features[2000 + start : 2000 + end]).to(device),
                ).cpu().numpy().reshape(end - start, -1)
            )
        joint_prediction = np.concatenate(joint_rows).astype(np.float32)
        np.save(JOINT_PRED, joint_prediction)

        test_source = np.load(RAW_TEST_LINK, mmap_mode="r")
        test_quality = np.load(TEST_LINK_QUALITY, mmap_mode="r")
        link_output = np.lib.format.open_memmap(
            LINK_PRED, mode="w+", dtype=np.float32, shape=test_source.shape
        )
        for point in range(len(test_source)):
            for start in range(0, test_source.shape[1], 256):
                end = min(start + 256, test_source.shape[1])
                links = np.arange(start, end, dtype=np.int64)
                link_output[point, start:end] = link(
                    torch.from_numpy(np.asarray(test_source[point, start:end], dtype=np.float32)).to(device),
                    torch.from_numpy(np.repeat(point_features[2000 + point : 2001 + point], len(links), axis=0)).to(device),
                    torch.from_numpy(links).to(device),
                    torch.from_numpy(np.asarray(test_quality[point, start:end], dtype=np.float32)).to(device),
                ).cpu().numpy()
            if (point + 1) % 50 == 0:
                link_output.flush()
                print(f"v469 link inference {point + 1}/500", flush=True)
        link_output.flush()

        power_prediction = power(torch.from_numpy(point_features[2000:]).to(device)).cpu().numpy().astype(np.float32)
        np.save(POWER_PRED, power_prediction)

    teacher_joint = np.load(V435_ACTIONS)["calibrated_target"].astype(np.float32)
    teacher_link = np.load(TEACHER_TEST_LINK, mmap_mode="r")
    teacher_power = np.load(V433_ACTIONS)["final_log_delta"].astype(np.float32)
    link_cosine_sum = 0.0
    link_count = 0
    predicted_link = np.load(LINK_PRED, mmap_mode="r")
    for point in range(500):
        cosine = row_cosine(predicted_link[point], teacher_link[point])
        link_cosine_sum += float(np.sum(cosine))
        link_count += len(cosine)
    return {
        "joint_teacher_cosine": float(np.mean(row_cosine(joint_prediction, teacher_joint))),
        "link_teacher_cosine": link_cosine_sum / link_count,
        "power_teacher_mae": float(np.mean(np.abs(power_prediction - teacher_power))),
    }


def generate_channel() -> None:
    source = np.load(BASE, mmap_mode="r")
    joint_target = np.load(JOINT_PRED, mmap_mode="r")
    link_target = np.load(LINK_PRED, mmap_mode="r")
    link_quality = np.load(TEST_LINK_QUALITY, mmap_mode="r")
    point_actions = np.load(V435_ACTIONS)
    regimes = point_actions["regimes"].astype("U8")
    point_quality = point_actions["point_quality"].astype(np.float32)
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    alpha_joint = v458.joint_alpha(profile, regimes, point_quality)
    alpha_link = v458.link_alpha(0.40, regimes, np.asarray(link_quality, dtype=np.float32))
    support_output = np.lib.format.open_memmap(
        CACHE / "v469_support.npy", mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), 2):
        end = min(start + 2, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        desired_support, _, _ = v435.build_desired_support(
            base_support, np.asarray(joint_target[start:end]), alpha_joint[start:end]
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        desired_link = v426.desired_linkwise_pdp(
            base_link,
            np.asarray(link_target[start:end]).reshape(end - start, 256, 4, 192),
            alpha_link[start:end].reshape(end - start, 256, 4),
        )
        candidate = base_h.copy()
        for _ in range(v458.JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_link)
        support_output[start:end] = candidate
        if end % 50 == 0:
            support_output.flush()
            print(f"v469 projection {end}/500", flush=True)
    support_output.flush()

    power_delta = np.load(POWER_PRED).astype(np.float32)
    amplitude = np.exp(0.5 * 10.0 * power_delta).astype(np.float32)
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    for start in range(0, len(source), 10):
        end = min(start + 10, len(source))
        output[start:end] = np.asarray(support_output[start:end]) * amplitude[start:end, None, None, None]
    output.flush()


def compare_to_v458() -> dict[str, float]:
    current = np.load(OUTPUT, mmap_mode="r")
    teacher = np.load(V458_OUTPUT, mmap_mode="r")
    numerator = 0.0
    denominator = 0.0
    cosine_sum = 0.0
    count = 0
    for start in range(0, len(current), 5):
        end = min(start + 5, len(current))
        left = np.asarray(current[start:end], dtype=np.complex64)
        right = np.asarray(teacher[start:end], dtype=np.complex64)
        numerator += float(np.sum(np.abs(left - right) ** 2, dtype=np.float64))
        denominator += float(np.sum(np.abs(right) ** 2, dtype=np.float64))
        inner = np.sum(np.conj(right) * left, axis=(1, 2, 3), dtype=np.complex128)
        norm = np.sqrt(
            np.sum(np.abs(left) ** 2, axis=(1, 2, 3), dtype=np.float64)
            * np.sum(np.abs(right) ** 2, axis=(1, 2, 3), dtype=np.float64)
        )
        cosine_sum += float(np.sum(np.abs(inner) / np.maximum(norm, 1e-30)))
        count += end - start
    return {
        "nmse_to_v458": numerator / max(denominator, 1e-30),
        "complex_cosine_to_v458": cosine_sum / max(count, 1),
    }


def write_package(report: dict[str, object]) -> None:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(__file__, "train_v469_neural_sionna_calibrator.py")
        # Some upload extractors do not create implicit parent directories.
        archive.writestr("model/", b"")
        archive.write(CHECKPOINT, "model/neural_sionna_calibrator.pt")
        for dependency in (
            "generate_v458_sionna_multimouth_extreme_diagnostic.py",
            "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
            "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
            "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
            "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
            "analyze_v447_sionna_linkwise_pdp_shape_calibration.py",
            "analyze_v434_sionna_joint_sh_calibration.py",
            "project_paths.py",
        ):
            archive.write(PATHS.code / dependency, dependency)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = all(np.isfinite(np.asarray(output[start : start + 5])).all() for start in range(0, len(output), 5))
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(finite),
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--joint-epochs", type=int, default=50)
    parser.add_argument("--link-steps", type=int, default=4000)
    parser.add_argument("--link-batch-size", type=int, default=1024)
    parser.add_argument("--power-steps", type=int, default=3000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = (
        BASE,
        V458_OUTPUT,
        V435_ACTIONS,
        V433_ACTIONS,
        RAW_TEST_LINK,
        TEACHER_TEST_LINK,
        TEST_LINK_QUALITY,
        v447.SOURCE_CACHE,
        v447.TRUTH_CACHE,
        v447.QUALITY_CACHE,
        v434.OOF_CACHE,
    )
    for path in required:
        require(path)
    data = build_joint_dataset()
    point_features = np.load(POINT_FEATURE_CACHE).astype(np.float32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    joint_epochs = 2 if args.smoke else args.joint_epochs
    link_steps = 5 if args.smoke else args.link_steps
    power_steps = 10 if args.smoke else args.power_steps
    joint, joint_result = train_joint(data, point_features, device, joint_epochs)
    link, link_result = train_links(point_features, device, link_steps, args.link_batch_size)
    power, power_result = train_power(point_features, device, power_steps)
    if args.smoke:
        print(json.dumps({"status": "smoke_pass"}, indent=2))
        return
    torch.save(
        {
            "version": VERSION,
            "joint": joint.state_dict(),
            "link": link.state_dict(),
            "power": power.state_dict(),
            "feature_dim": int(point_features.shape[1]),
            "joint_result": joint_result.__dict__,
            "link_result": link_result.__dict__,
            "power_result": power_result.__dict__,
        },
        CHECKPOINT,
    )
    target_fit = write_predictions(joint, link, power, data, point_features, device)
    generate_channel()
    replay = v208.score_prediction(OUTPUT)
    comparison = compare_to_v458()
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "A scene-specific neural calibrator trained jointly on official-train truth and "
            "v458 test-coordinate teacher actions can replace hand-written regime calibration "
            "without losing the verified joint S-H/linkwise-PDP/power behavior."
        ),
        "evidence_boundary": {
            "official_train_truth": True,
            "test_coordinate_sionna": True,
            "v458_test_teacher_actions": True,
            "competition_compliance": False,
        },
        "architecture": {
            "joint": "10-block circular 2D FiLM ResNet",
            "link": "8-block conditioned 1D residual CNN with array/link embeddings",
            "power": "four-layer scene-specific MLP",
            "fixed_renderer": "five alternating full-support/linkwise-PDP projections plus power scale 10",
        },
        "training": {
            "joint": joint_result.__dict__,
            "link": link_result.__dict__,
            "power": power_result.__dict__,
        },
        "target_fit": target_fit,
        "comparison_to_v458": comparison,
        "replay": replay,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    write_package(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
