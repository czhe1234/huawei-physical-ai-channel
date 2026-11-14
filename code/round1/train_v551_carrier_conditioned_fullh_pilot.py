#!/usr/bin/env python3
"""Train a 400-point carrier-conditioned full-channel residual pilot."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v401_dualdomain_pas_preserving_confocal as v401
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 551
VERSION = "v551_carrier_conditioned_fullh_pilot"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT = CHECKPOINT_DIR / "carrier_conditioned_fullh_pilot.pt"
SELECTION = CACHE_DIR / "v551_pilot_selection.npz"
SOURCE_FEATURES = CACHE_DIR / "v551_pilot_sionna_log_power.npy"
SOURCE_POINT_POWER = CACHE_DIR / "v551_pilot_sionna_log_point_power.npy"

TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
BASE_H = PATHS.cache_dir(471) / "v471_train_v98like_base_h.npy"
POINT_FEATURES = PATHS.cache_dir(469) / "v469_point_features.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
SOURCES = (
    ("high_6p0_d16", PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"),
    (
        "low_2p6_d5",
        PATHS.cache_dir(205)
        / "benchmark_sector_m30_track_d5_f2p6"
        / "train_clean_H.npy",
    ),
    (
        "low_6p0_d5",
        PATHS.cache_dir(205)
        / "benchmark_sector_m30_track_d5_f6p0"
        / "train_clean_H.npy",
    ),
    (
        "low_6p7_d5",
        PATHS.cache_dir(205)
        / "benchmark_sector_m30_track_d5_f6p7"
        / "train_clean_H.npy",
    ),
)
REGIMES = ("clear", "light", "medium", "heavy")
SEED = 55101
ROWS_PER_FOLD = 80
OUTER_FOLD = 4
INNER_FOLD = 3
TINY = 1e-12
AMP_LOG_POWER_MAX = 2.0
POINT_LOG_POWER_MAX = 1.5
PHASE_MAX = math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def allocate_counts(total: int, labels: np.ndarray) -> dict[str, int]:
    fractions = {name: float(np.mean(labels == name)) for name in REGIMES}
    raw = {name: total * fractions[name] for name in REGIMES}
    counts = {name: int(math.floor(raw[name])) for name in REGIMES}
    for name in sorted(
        REGIMES,
        key=lambda value: raw[value] - counts[value],
        reverse=True,
    ):
        if sum(counts.values()) >= total:
            break
        counts[name] += 1
    return counts


def build_selection() -> dict[str, np.ndarray]:
    positions = np.load(TRAIN_POS).astype(np.float32)
    geometry = np.load(GEOMETRY)
    regimes = regime_of(geometry["train_blocked"]).astype("U8")
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    target_counts = allocate_counts(ROWS_PER_FOLD, regimes)
    rng = np.random.default_rng(SEED)
    selected = []
    selected_fold = []
    for fold_id, fold_ids in enumerate(folds):
        local = []
        for regime in REGIMES:
            candidates = fold_ids[regimes[fold_ids] == regime].copy()
            rng.shuffle(candidates)
            take = min(target_counts[regime], len(candidates))
            local.extend(candidates[:take].tolist())
        remaining = np.setdiff1d(fold_ids, np.asarray(local, dtype=np.int64))
        rng.shuffle(remaining)
        if len(local) < ROWS_PER_FOLD:
            local.extend(remaining[: ROWS_PER_FOLD - len(local)].tolist())
        local_ids = np.sort(np.asarray(local[:ROWS_PER_FOLD], dtype=np.int64))
        if len(local_ids) != ROWS_PER_FOLD:
            raise ValueError(f"fold {fold_id} selection has {len(local_ids)} rows")
        selected.append(local_ids)
        selected_fold.append(np.full(ROWS_PER_FOLD, fold_id, dtype=np.int16))
    selected_ids = np.concatenate(selected)
    fold_id = np.concatenate(selected_fold)
    if len(np.unique(selected_ids)) != 5 * ROWS_PER_FOLD:
        raise ValueError("pilot selection must contain 400 unique rows")
    np.savez_compressed(
        SELECTION,
        selected_ids=selected_ids,
        fold_id=fold_id,
        regimes=regimes[selected_ids],
        seed=np.asarray(SEED),
    )
    return {
        "selected_ids": selected_ids,
        "fold_id": fold_id,
        "regimes": regimes[selected_ids],
    }


def load_selection() -> dict[str, np.ndarray]:
    if not SELECTION.is_file():
        return build_selection()
    data = np.load(SELECTION)
    return {
        "selected_ids": data["selected_ids"].astype(np.int64),
        "fold_id": data["fold_id"].astype(np.int16),
        "regimes": data["regimes"].astype("U8"),
    }


def build_source_cache(
    selection: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> None:
    if SOURCE_FEATURES.is_file() and SOURCE_POINT_POWER.is_file() and not rebuild:
        return
    selected_ids = selection["selected_ids"]
    source_arrays = [np.load(path, mmap_mode="r") for _, path in SOURCES]
    output = np.lib.format.open_memmap(
        SOURCE_FEATURES,
        mode="w+",
        dtype=np.float16,
        shape=(len(selected_ids), len(SOURCES), 1024, 192),
    )
    point_power = np.lib.format.open_memmap(
        SOURCE_POINT_POWER,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected_ids), len(SOURCES)),
    )
    for start in range(0, len(selected_ids), batch_size):
        end = min(start + batch_size, len(selected_ids))
        ids = selected_ids[start:end]
        for source_id, source in enumerate(source_arrays):
            channel = torch.from_numpy(np.asarray(source[ids], dtype=np.complex64)).to(
                device
            )
            cir = torch.fft.ifft(channel, dim=-1)
            power = cir.abs().square().float()
            total = power.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
            normalized = power / total
            feature = torch.log(normalized.clamp_min(1e-12)).clamp(-24.0, 0.0)
            output[start:end, source_id] = (
                feature.reshape(end - start, 1024, 192).cpu().numpy().astype(np.float16)
            )
            point_power[start:end, source_id] = (
                torch.log(total.reshape(-1)).cpu().numpy().astype(np.float32)
            )
            del channel, cir, power, total, normalized, feature
        if end == len(selected_ids) or end % 80 == 0:
            output.flush()
            point_power.flush()
            print(f"v551 source cache {end}/{len(selected_ids)}", flush=True)
    output.flush()
    point_power.flush()


def link_coordinates(device: torch.device) -> tuple[torch.Tensor, ...]:
    link_id = torch.arange(1024, device=device)
    ue = link_id.remainder(4)
    bs = torch.div(link_id, 4, rounding_mode="floor")
    vertical = bs.remainder(8)
    horizontal = torch.div(bs, 8, rounding_mode="floor").remainder(16)
    polar = torch.div(bs, 128, rounding_mode="floor")
    return polar, horizontal, vertical, ue


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(
            width,
            width,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        work = self.conv1(F.gelu(self.norm1(value)))
        work = work + 0.12 * torch.tanh(condition)[:, :, None]
        work = self.conv2(F.gelu(self.norm2(work)))
        return value + work


class CarrierConditionedResidualNet(nn.Module):
    def __init__(self, point_dim: int, width: int) -> None:
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(point_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, width),
        )
        self.polar = nn.Embedding(2, width)
        self.horizontal = nn.Embedding(16, width)
        self.vertical = nn.Embedding(8, width)
        self.ue = nn.Embedding(4, width)
        self.input = nn.Conv1d(7, width, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8, 1, 2)]
        )
        self.output = nn.Conv1d(width, 2, kernel_size=5, padding=2)
        self.point_power = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.point_power.weight)
        nn.init.zeros_(self.point_power.bias)

    def forward(
        self,
        base_cir: torch.Tensor,
        source_log_power: torch.Tensor,
        point_features: torch.Tensor,
        coordinates: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = len(base_cir)
        base_links = base_cir.reshape(batch, 1024, 192)
        base_power = base_links.abs().square().float()
        normalized = base_power / base_power.sum(dim=(1, 2), keepdim=True).clamp_min(
            TINY
        )
        base_log = torch.log(normalized.clamp_min(1e-12)).clamp(-24.0, 0.0)
        base_unit = base_links / base_links.abs().clamp_min(1e-12)
        inputs = torch.cat(
            [
                base_log[:, :, None, :],
                base_unit.real[:, :, None, :],
                base_unit.imag[:, :, None, :],
                source_log_power.permute(0, 2, 1, 3),
            ],
            dim=2,
        ).reshape(batch * 1024, 7, 192)

        point_context = self.point(point_features)
        polar, horizontal, vertical, ue = coordinates
        link_context = (
            self.polar(polar)
            + self.horizontal(horizontal)
            + self.vertical(vertical)
            + self.ue(ue)
        )
        condition = (point_context[:, None, :] + link_context[None, :, :]).reshape(
            batch * 1024, -1
        )

        value = self.input(inputs)
        for block in self.blocks:
            value = block(value, condition)
        raw = self.output(F.gelu(value)).reshape(batch, 1024, 2, 192)
        amplitude_delta = (AMP_LOG_POWER_MAX * torch.tanh(raw[:, :, 0])).float()
        phase_delta = (PHASE_MAX * torch.tanh(raw[:, :, 1])).float()
        point_delta = (
            POINT_LOG_POWER_MAX * torch.tanh(self.point_power(point_context))
        ).float()
        log_power_delta = amplitude_delta + point_delta[:, None, :]
        candidate_links = base_links * torch.exp(0.5 * log_power_delta)
        candidate_links = candidate_links * torch.polar(
            torch.ones_like(phase_delta), phase_delta
        )
        candidate_cir = candidate_links.reshape_as(base_cir)
        candidate_h = torch.fft.fft(candidate_cir, dim=-1)
        diagnostics = {
            "amplitude_delta": amplitude_delta,
            "phase_delta": phase_delta,
            "point_delta": point_delta,
        }
        return candidate_h, diagnostics


def cosine(left: torch.Tensor, right: torch.Tensor, dim: int) -> torch.Tensor:
    left_scale = torch.amax(left.abs(), dim=dim, keepdim=True).detach()
    right_scale = torch.amax(right.abs(), dim=dim, keepdim=True).detach()
    left_scaled = left / torch.where(
        left_scale > 0, left_scale, torch.ones_like(left_scale)
    )
    right_scaled = right / torch.where(
        right_scale > 0, right_scale, torch.ones_like(right_scale)
    )
    numerator = torch.sum(left_scaled * right_scaled, dim=dim)
    denominator = torch.linalg.vector_norm(
        left_scaled, dim=dim
    ) * torch.linalg.vector_norm(right_scaled, dim=dim)
    positive = denominator > 0
    denominator = torch.where(
        positive,
        denominator,
        torch.full_like(denominator, 1e-12),
    )
    return numerator / denominator


def score_tensors(
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = len(prediction)
    prediction6 = prediction.reshape(batch, 2, 16, 8, 4, 192)
    truth6 = truth.reshape(batch, 2, 16, 8, 4, 192)
    prediction_pas = torch.fft.fft2(prediction6, dim=(2, 3)).abs().square()
    truth_pas = torch.fft.fft2(truth6, dim=(2, 3)).abs().square()
    prediction_pas = prediction_pas.reshape(batch, 256, 4, 192)
    truth_pas = truth_pas.reshape(batch, 256, 4, 192)
    pas = cosine(prediction_pas, truth_pas, 1).mean(dim=(1, 2))

    prediction_cir = torch.fft.ifft(prediction, dim=-1)
    truth_cir = torch.fft.ifft(truth, dim=-1)
    prediction_pdp = prediction_cir.abs().square()
    truth_pdp = truth_cir.abs().square()
    pdp = cosine(prediction_pdp, truth_pdp, 3).mean(dim=(1, 2))

    point_error = torch.sum((prediction - truth).abs().square(), dim=(1, 2, 3))
    point_energy = torch.sum(truth.abs().square(), dim=(1, 2, 3))
    nmse = point_error.sum() / point_energy.sum().clamp_min(1e-12)
    nmse_score = 1.0 / (1.0 + nmse)
    official = 0.4 * pas.mean() + 0.4 * pdp.mean() + 0.2 * nmse_score
    return {
        "pas": pas,
        "pdp": pdp,
        "point_error": point_error,
        "point_energy": point_energy,
        "nmse": nmse,
        "nmse_score": nmse_score,
        "official_score": official,
    }


@dataclass
class BatchStore:
    selected_ids: np.ndarray
    base: np.ndarray
    truth: np.ndarray
    source: np.ndarray
    point_features: np.ndarray
    device: torch.device

    def load(
        self,
        rows: np.ndarray,
        feature_override: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, ...]:
        ids = self.selected_ids[rows]
        base_h = torch.from_numpy(np.asarray(self.base[ids], dtype=np.complex64)).to(
            self.device
        )
        truth_h = torch.from_numpy(np.asarray(self.truth[ids], dtype=np.complex64)).to(
            self.device
        )
        base_cir = torch.fft.ifft(base_h, dim=-1)
        local_source = self.source[
            rows if feature_override is None else feature_override
        ]
        source = torch.from_numpy(np.asarray(local_source, dtype=np.float32)).to(
            self.device
        )
        point = torch.from_numpy(
            np.asarray(self.point_features[rows], dtype=np.float32)
        ).to(self.device)
        return base_h, base_cir, truth_h, source, point


def build_point_features(
    selection: dict[str, np.ndarray], fit_rows: np.ndarray
) -> tuple[np.ndarray, dict[str, list[float]]]:
    selected_ids = selection["selected_ids"]
    cached = np.load(POINT_FEATURES).astype(np.float32)[:2000]
    positions = np.load(TRAIN_POS).astype(np.float32)
    geometry = np.load(GEOMETRY)
    blocked = geometry["train_blocked"].astype(np.float32)
    base = np.load(BASE_H, mmap_mode="r")
    base_log_power = np.empty(len(selected_ids), dtype=np.float32)
    for start in range(0, len(selected_ids), 16):
        end = min(start + 16, len(selected_ids))
        values = np.asarray(base[selected_ids[start:end]], dtype=np.complex64)
        base_log_power[start:end] = np.log(
            np.maximum(
                np.sum(np.abs(values) ** 2, axis=(1, 2, 3), dtype=np.float64),
                1e-30,
            )
        ).astype(np.float32)
    source_power = np.load(SOURCE_POINT_POWER).astype(np.float32)
    point = np.concatenate(
        [
            cached[selected_ids],
            positions[selected_ids],
            blocked[selected_ids, None],
            base_log_power[:, None],
            source_power,
        ],
        axis=1,
    ).astype(np.float32)
    mean = point[fit_rows].mean(axis=0, keepdims=True)
    std = np.maximum(point[fit_rows].std(axis=0, keepdims=True), 1e-4)
    normalized = ((point - mean) / std).astype(np.float32)
    return normalized, {
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
    }


def summarize_scores(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    if mask is None:
        mask = np.ones(len(pas), dtype=bool)
    nmse = float(np.sum(error[mask]) / max(np.sum(energy[mask]), 1e-30))
    nmse_score = 1.0 / (1.0 + nmse)
    mean_pas = float(np.mean(pas[mask]))
    mean_pdp = float(np.mean(pdp[mask]))
    return {
        "count": int(np.sum(mask)),
        "pas": mean_pas,
        "pdp": mean_pdp,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "official_score": 0.4 * mean_pas + 0.4 * mean_pdp + 0.2 * nmse_score,
    }


def score_delta(
    candidate: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("pas", "pdp", "global_nmse", "nmse_score", "official_score")
    }


@torch.no_grad()
def evaluate(
    model: CarrierConditionedResidualNet | None,
    store: BatchStore,
    rows: np.ndarray,
    coordinates: tuple[torch.Tensor, ...],
    batch_size: int,
    feature_permutation: np.ndarray | None = None,
) -> dict[str, np.ndarray | dict[str, float]]:
    pas = np.empty(len(rows), dtype=np.float32)
    pdp = np.empty(len(rows), dtype=np.float32)
    error = np.empty(len(rows), dtype=np.float64)
    energy = np.empty(len(rows), dtype=np.float64)
    amplitude = np.zeros(len(rows), dtype=np.float32)
    phase = np.zeros(len(rows), dtype=np.float32)
    point_delta = np.zeros(len(rows), dtype=np.float32)
    if model is not None:
        model.eval()
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        local_rows = rows[start:end]
        override = None
        if feature_permutation is not None:
            override = feature_permutation[start:end]
        base_h, base_cir, truth_h, source, point = store.load(local_rows, override)
        if model is None:
            prediction = base_h
            diagnostics = None
        else:
            prediction, diagnostics = model(base_cir, source, point, coordinates)
            prediction = prediction.to(torch.complex64)
        scores = score_tensors(prediction, truth_h)
        pas[start:end] = scores["pas"].float().cpu().numpy()
        pdp[start:end] = scores["pdp"].float().cpu().numpy()
        error[start:end] = scores["point_error"].double().cpu().numpy()
        energy[start:end] = scores["point_energy"].double().cpu().numpy()
        if diagnostics is not None:
            amplitude[start:end] = (
                diagnostics["amplitude_delta"]
                .abs()
                .mean(dim=(1, 2))
                .float()
                .cpu()
                .numpy()
            )
            phase[start:end] = (
                diagnostics["phase_delta"].abs().mean(dim=(1, 2)).float().cpu().numpy()
            )
            point_delta[start:end] = (
                diagnostics["point_delta"].reshape(-1).float().cpu().numpy()
            )
    return {
        "pas": pas,
        "pdp": pdp,
        "error": error,
        "energy": energy,
        "amplitude_abs_mean": amplitude,
        "phase_abs_mean": phase,
        "point_log_power_delta": point_delta,
        "summary": summarize_scores(pas, pdp, error, energy),
    }


def train_model(
    store: BatchStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    point_dim: int,
    coordinates: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    epochs: int,
) -> tuple[CarrierConditionedResidualNet, dict[str, object]]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = CarrierConditionedResidualNet(point_dim, args.width).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    baseline_validation = None
    if validation_rows is not None:
        baseline_validation = evaluate(
            None,
            store,
            validation_rows,
            coordinates,
            args.batch_size,
        )["summary"]
    best_score = (
        float(baseline_validation["official_score"])
        if baseline_validation is not None
        else -float("inf")
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    stale = 0
    rng = np.random.default_rng(SEED)
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses = []
        scores = []
        for start in range(0, len(order), args.batch_size):
            local_rows = order[start : start + args.batch_size]
            _, base_cir, truth_h, source, point = store.load(local_rows)
            optimizer.zero_grad(set_to_none=True)
            prediction, diagnostics = model(base_cir, source, point, coordinates)
            prediction = prediction.to(torch.complex64)
            metrics = score_tensors(prediction, truth_h)
            regularizer = (
                1e-5 * diagnostics["amplitude_delta"].square().mean()
                + 1e-5 * diagnostics["phase_delta"].square().mean()
                + 1e-5 * diagnostics["point_delta"].square().mean()
            )
            loss = 1.0 - metrics["official_score"] + regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}, rows={local_rows.tolist()}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            scores.append(float(metrics["official_score"].detach()))
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_batch_score": float(np.mean(scores)),
        }
        if validation_rows is not None:
            validation = evaluate(
                model,
                store,
                validation_rows,
                coordinates,
                args.batch_size,
            )["summary"]
            validation_score = float(validation["official_score"])
            row["validation"] = validation
            row["validation_delta"] = score_delta(validation, baseline_validation)
            if validation_score > best_score + 1e-5:
                best_score = validation_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_score": row["train_batch_score"],
                    "validation_score": (
                        row.get("validation", {}).get("official_score")
                        if validation_rows is not None
                        else None
                    ),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if validation_rows is not None and epoch >= 8 and stale >= args.patience:
            break
    if validation_rows is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
        best_score = (
            float(history[-1]["train_batch_score"]) if history else float("nan")
        )
    return model, {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "baseline_validation": baseline_validation,
        "history": history,
        "seconds": float(time.time() - started),
    }


def regime_breakdown(
    result: dict[str, np.ndarray | dict[str, float]],
    regimes: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        regime: summarize_scores(
            result["pas"],
            result["pdp"],
            result["error"],
            result["energy"],
            regimes == regime,
        )
        for regime in REGIMES
        if np.any(regimes == regime)
    }


def shuffled_rows(
    rows: np.ndarray,
    regimes: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(rows), dtype=np.int64)
    rng = np.random.default_rng(SEED + 17)
    local_regimes = regimes[rows]
    for regime in REGIMES:
        local = np.flatnonzero(local_regimes == regime)
        output[local] = rng.permutation(rows[local])
    return output


def main() -> None:
    args = parse_args()
    for path in (
        TRAIN_H,
        TRAIN_POS,
        BASE_H,
        POINT_FEATURES,
        GEOMETRY,
        *(path for _, path in SOURCES),
    ):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    started = time.time()
    selection = load_selection()
    build_source_cache(
        selection,
        device,
        args.cache_batch_size,
        args.rebuild_cache,
    )
    fold_id = selection["fold_id"]
    fit_rows = np.flatnonzero(fold_id <= 2).astype(np.int64)
    inner_rows = np.flatnonzero(fold_id == INNER_FOLD).astype(np.int64)
    outer_rows = np.flatnonzero(fold_id == OUTER_FOLD).astype(np.int64)

    point_features, normalization = build_point_features(selection, fit_rows)
    store = BatchStore(
        selected_ids=selection["selected_ids"],
        base=np.load(BASE_H, mmap_mode="r"),
        truth=np.load(TRAIN_H, mmap_mode="r"),
        source=np.load(SOURCE_FEATURES, mmap_mode="r"),
        point_features=point_features,
        device=device,
    )
    coordinates = link_coordinates(device)
    baseline_inner = evaluate(None, store, inner_rows, coordinates, args.batch_size)
    baseline_outer = evaluate(None, store, outer_rows, coordinates, args.batch_size)

    selection_model, selection_training = train_model(
        store,
        fit_rows,
        inner_rows,
        point_features.shape[1],
        coordinates,
        args,
        args.max_epochs,
    )
    selected_epoch = int(selection_training["best_epoch"])
    if selected_epoch == 0:
        final_model = selection_model
        final_training = {
            "epochs": 0,
            "seconds": 0.0,
            "reason": "inner validation selected the unchanged carrier",
        }
    else:
        final_fit_rows = np.concatenate([fit_rows, inner_rows])
        final_features, final_normalization = build_point_features(
            selection, final_fit_rows
        )
        store.point_features = final_features
        final_model, final_training_detail = train_model(
            store,
            final_fit_rows,
            None,
            final_features.shape[1],
            coordinates,
            args,
            selected_epoch,
        )
        final_normalization["selection_fit"] = normalization
        normalization = final_normalization
        final_training = {
            "epochs": selected_epoch,
            **final_training_detail,
        }

    outer = evaluate(
        final_model,
        store,
        outer_rows,
        coordinates,
        args.batch_size,
    )
    permutation = shuffled_rows(outer_rows, selection["regimes"])
    outer_shuffled = evaluate(
        final_model,
        store,
        outer_rows,
        coordinates,
        args.batch_size,
        feature_permutation=permutation,
    )
    baseline_summary = baseline_outer["summary"]
    outer_summary = outer["summary"]
    shuffled_summary = outer_shuffled["summary"]
    delta_vs_base = score_delta(outer_summary, baseline_summary)
    source_margin = float(
        outer_summary["official_score"] - shuffled_summary["official_score"]
    )
    scale_up = bool(delta_vs_base["official_score"] >= 0.005 and source_margin >= 0.001)
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "point_dim": point_features.shape[1],
            "width": args.width,
            "selected_epoch": selected_epoch,
            "selection": {key: value.tolist() for key, value in selection.items()},
            "normalization": normalization,
            "sources": [name for name, _ in SOURCES],
        },
        CHECKPOINT,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A residual model trained from the same strict OOF carrier that it edits, "
            "and conditioned on multi-configuration Sionna power plus geometry, can "
            "jointly improve official PAS, PDP, and NMSE at unseen positions."
        ),
        "pilot_design": {
            "rows": 400,
            "rows_per_fold": ROWS_PER_FOLD,
            "fit_folds": [0, 1, 2],
            "inner_selection_fold": INNER_FOLD,
            "outer_unseen_fold": OUTER_FOLD,
            "fit_rows": len(fit_rows),
            "inner_rows": len(inner_rows),
            "outer_rows": len(outer_rows),
            "selection_seed": SEED,
            "carrier": PATHS.relative(BASE_H),
            "source_names": [name for name, _ in SOURCES],
            "uses_online_score": False,
        },
        "model": {
            "architecture": (
                "point-conditioned shared link-delay residual CNN with pol/H/V/UE "
                "embeddings; predicts cell log-power, point power, and phase residuals"
            ),
            "width": args.width,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "precision": "float32",
            "loss": "1 - (0.4*PAS + 0.4*PDP + 0.2/(1+NMSE))",
            "amplitude_log_power_cap": AMP_LOG_POWER_MAX,
            "point_log_power_cap": POINT_LOG_POWER_MAX,
            "phase_cap_radians": PHASE_MAX,
        },
        "selection_training": selection_training,
        "final_training": final_training,
        "inner_baseline": baseline_inner["summary"],
        "outer": {
            "baseline": baseline_summary,
            "candidate": outer_summary,
            "delta": delta_vs_base,
            "source_shuffled_control": shuffled_summary,
            "candidate_minus_shuffled": source_margin,
            "baseline_by_regime": regime_breakdown(
                baseline_outer, selection["regimes"][outer_rows]
            ),
            "candidate_by_regime": regime_breakdown(
                outer, selection["regimes"][outer_rows]
            ),
            "actions": {
                "amplitude_abs_mean": float(np.mean(outer["amplitude_abs_mean"])),
                "phase_abs_mean": float(np.mean(outer["phase_abs_mean"])),
                "point_log_power_delta_mean": float(
                    np.mean(outer["point_log_power_delta"])
                ),
                "point_log_power_delta_abs_mean": float(
                    np.mean(np.abs(outer["point_log_power_delta"]))
                ),
            },
        },
        "decision": {
            "scale_up_recommended": scale_up,
            "required_outer_delta": 0.005,
            "required_source_margin": 0.001,
            "observed_outer_delta": delta_vs_base["official_score"],
            "observed_source_margin": source_margin,
        },
        "runtime": {
            "seconds": float(time.time() - started),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "outputs": {
            "selection": PATHS.relative(SELECTION),
            "source_features": PATHS.relative(SOURCE_FEATURES),
            "source_point_power": PATHS.relative(SOURCE_POINT_POWER),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_epoch": selected_epoch,
                "outer_baseline": baseline_summary,
                "outer_candidate": outer_summary,
                "outer_delta": delta_vs_base,
                "source_margin": source_margin,
                "scale_up_recommended": scale_up,
                "runtime_seconds": report["runtime"]["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
