#!/usr/bin/env python3
"""Scale the v551 carrier-conditioned full-H model to all official points."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import train_v551_carrier_conditioned_fullh_pilot as v551
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 552
VERSION = "v552_full_carrier_conditioned_fullh"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE_DIR / "v552_Round1_Test_Channel.npy"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_ensemble_candidate_pending.zip"
BUNDLE = CHECKPOINT_DIR / "v552_inference_bundle.pt"

TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TEST_POS = PATHS.round1 / "Round1_Test_Pos.npy"
TRAIN_BASE = PATHS.cache_dir(471) / "v471_train_v98like_base_h.npy"
TEST_BASE = PATHS.cache_dir(469) / "v469_Round1_Test_Channel.npy"
POINT_FEATURES = PATHS.cache_dir(469) / "v469_point_features.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"

TRAIN_SOURCES = (
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
TEST_SOURCES = (
    ("high_6p0_d16", PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"),
    (
        "low_2p6_d5",
        PATHS.cache_dir(205)
        / "benchmark_sector_m30_track_d5_f2p6"
        / "test_clean_H.npy",
    ),
    (
        "low_6p0_d5",
        PATHS.cache_dir(205)
        / "benchmark_sector_m30_track_d5_f6p0"
        / "test_clean_H.npy",
    ),
    (
        "low_6p7_d5",
        PATHS.cache_dir(205)
        / "benchmark_sector_m30_track_d5_f6p7"
        / "test_clean_H.npy",
    ),
)

TRAIN_SOURCE_FEATURES = CACHE_DIR / "v552_train_sionna_log_power.npy"
TEST_SOURCE_FEATURES = CACHE_DIR / "v552_test_sionna_log_power.npy"
TRAIN_SOURCE_POINT_POWER = CACHE_DIR / "v552_train_sionna_log_point_power.npy"
TEST_SOURCE_POINT_POWER = CACHE_DIR / "v552_test_sionna_log_point_power.npy"
TRAIN_SOURCE_DONE = CACHE_DIR / "v552_train_source_cache.done.json"
TEST_SOURCE_DONE = CACHE_DIR / "v552_test_source_cache.done.json"
RAW_POINT_FEATURES = CACHE_DIR / "v552_raw_point_features.npy"

REGIMES = v551.REGIMES
SEED = 55201
ALPHA_GRID = (0.25, 0.50, 0.75, 1.00)
TINY = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def cache_valid(
    feature_path: Path,
    power_path: Path,
    marker_path: Path,
    rows: int,
) -> bool:
    if not (feature_path.is_file() and power_path.is_file() and marker_path.is_file()):
        return False
    try:
        feature = np.load(feature_path, mmap_mode="r")
        power = np.load(power_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return feature.shape == (rows, 4, 1024, 192) and power.shape == (rows, 4)


def build_source_cache(
    source_paths: tuple[tuple[str, Path], ...],
    feature_path: Path,
    power_path: Path,
    marker_path: Path,
    rows: int,
    label: str,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> None:
    if not rebuild and cache_valid(feature_path, power_path, marker_path, rows):
        print(f"v552 reuse {label} source cache", flush=True)
        return
    for _, path in source_paths:
        require(path)
    sources = [np.load(path, mmap_mode="r") for _, path in source_paths]
    if any(len(source) != rows for source in sources):
        raise ValueError(f"{label} source row count mismatch")
    output = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(rows, len(source_paths), 1024, 192),
    )
    point_power = np.lib.format.open_memmap(
        power_path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, len(source_paths)),
    )
    for start in range(0, rows, batch_size):
        end = min(start + batch_size, rows)
        for source_id, source in enumerate(sources):
            channel = torch.from_numpy(
                np.asarray(source[start:end], dtype=np.complex64)
            ).to(device)
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
        if end == rows or end % 200 == 0:
            output.flush()
            point_power.flush()
            print(f"v552 {label} source cache {end}/{rows}", flush=True)
    output.flush()
    point_power.flush()
    marker_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "shape": [rows, len(source_paths), 1024, 192],
                "source_names": [name for name, _ in source_paths],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def channel_log_power(path: Path, rows: int) -> np.ndarray:
    channel = np.load(path, mmap_mode="r")
    if len(channel) != rows:
        raise ValueError(f"unexpected row count for {path}")
    output = np.empty(rows, dtype=np.float32)
    for start in range(0, rows, 8):
        end = min(start + 8, rows)
        values = np.asarray(channel[start:end], dtype=np.complex64)
        output[start:end] = np.log(
            np.maximum(
                np.sum(np.abs(values) ** 2, axis=(1, 2, 3), dtype=np.float64),
                1e-30,
            )
        ).astype(np.float32)
    return output


def build_raw_point_features(rebuild: bool) -> np.ndarray:
    if RAW_POINT_FEATURES.is_file() and not rebuild:
        value = np.load(RAW_POINT_FEATURES, mmap_mode="r")
        if value.shape[0] == 2500:
            return value
    cached = np.load(POINT_FEATURES).astype(np.float32)
    train_pos = np.load(TRAIN_POS).astype(np.float32)
    test_pos = np.load(TEST_POS).astype(np.float32)
    geometry = np.load(GEOMETRY)
    blocked = np.concatenate(
        [geometry["train_blocked"], geometry["test_blocked"]]
    ).astype(np.float32)
    base_log_power = np.concatenate(
        [
            channel_log_power(TRAIN_BASE, len(train_pos)),
            channel_log_power(TEST_BASE, len(test_pos)),
        ]
    )
    source_power = np.concatenate(
        [
            np.load(TRAIN_SOURCE_POINT_POWER).astype(np.float32),
            np.load(TEST_SOURCE_POINT_POWER).astype(np.float32),
        ],
        axis=0,
    )
    positions = np.concatenate([train_pos, test_pos], axis=0)
    value = np.concatenate(
        [
            cached,
            positions,
            blocked[:, None],
            base_log_power[:, None],
            source_power,
        ],
        axis=1,
    ).astype(np.float32)
    np.save(RAW_POINT_FEATURES, value)
    return np.load(RAW_POINT_FEATURES, mmap_mode="r")


def normalization(
    raw_features: np.ndarray, fit_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(raw_features[fit_ids], dtype=np.float32).mean(
        axis=0, keepdims=True
    )
    std = np.maximum(
        np.asarray(raw_features[fit_ids], dtype=np.float32).std(axis=0, keepdims=True),
        1e-4,
    )
    return mean.astype(np.float32), std.astype(np.float32)


class TrainStore:
    def __init__(self, raw_features: np.ndarray, device: torch.device) -> None:
        self.base = np.load(TRAIN_BASE, mmap_mode="r")
        self.truth = np.load(TRAIN_H, mmap_mode="r")
        self.source = np.load(TRAIN_SOURCE_FEATURES, mmap_mode="r")
        self.raw_features = raw_features
        self.device = device

    def load(
        self,
        ids: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        source_ids: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, ...]:
        base_h = torch.from_numpy(np.asarray(self.base[ids], dtype=np.complex64)).to(
            self.device
        )
        truth_h = torch.from_numpy(np.asarray(self.truth[ids], dtype=np.complex64)).to(
            self.device
        )
        local_source_ids = ids if source_ids is None else source_ids
        source = torch.from_numpy(
            np.asarray(self.source[local_source_ids], dtype=np.float32)
        ).to(self.device)
        point = (np.asarray(self.raw_features[ids], dtype=np.float32) - mean) / std
        point_tensor = torch.from_numpy(point.astype(np.float32)).to(self.device)
        return base_h, torch.fft.ifft(base_h, dim=-1), truth_h, source, point_tensor


class TestStore:
    def __init__(self, raw_features: np.ndarray, device: torch.device) -> None:
        self.base = np.load(TEST_BASE, mmap_mode="r")
        self.source = np.load(TEST_SOURCE_FEATURES, mmap_mode="r")
        self.raw_features = raw_features[2000:]
        self.device = device

    def load_base(self, ids: np.ndarray) -> tuple[torch.Tensor, ...]:
        base_h = torch.from_numpy(np.asarray(self.base[ids], dtype=np.complex64)).to(
            self.device
        )
        source = torch.from_numpy(np.asarray(self.source[ids], dtype=np.float32)).to(
            self.device
        )
        return base_h, torch.fft.ifft(base_h, dim=-1), source

    def point(self, ids: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        value = (np.asarray(self.raw_features[ids], dtype=np.float32) - mean) / std
        return torch.from_numpy(value.astype(np.float32)).to(self.device)


def apply_diagnostics(
    base_cir: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    batch = len(base_cir)
    base_links = base_cir.reshape(batch, 1024, 192)
    amplitude = alpha * (
        diagnostics["amplitude_delta"] + diagnostics["point_delta"][:, None, :]
    )
    phase = alpha * diagnostics["phase_delta"]
    links = base_links * torch.exp(0.5 * amplitude)
    links = links * torch.polar(torch.ones_like(phase), phase)
    return torch.fft.fft(links.reshape_as(base_cir), dim=-1).to(torch.complex64)


def checkpoint_path(fold: int) -> Path:
    return CHECKPOINT_DIR / f"v552_fold{fold}.pt"


def train_fold(
    fold: int,
    fit_ids: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    store: TrainStore,
    coordinates: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
) -> tuple[v551.CarrierConditionedResidualNet, dict[str, object]]:
    path = checkpoint_path(fold)
    if path.is_file() and not args.retrain:
        saved = torch.load(path, map_location=store.device, weights_only=False)
        model = v551.CarrierConditionedResidualNet(
            int(saved["point_dim"]), int(saved["width"])
        ).to(store.device)
        model.load_state_dict(saved["state_dict"])
        print(f"v552 reuse fold {fold} checkpoint", flush=True)
        return model, saved

    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    model = v551.CarrierConditionedResidualNet(
        store.raw_features.shape[1], args.width
    ).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(SEED + fold)
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(fit_ids)
        losses = []
        scores = []
        grad_norms = []
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            _, base_cir, truth_h, source, point = store.load(ids, mean, std)
            optimizer.zero_grad(set_to_none=True)
            prediction, diagnostics = model(base_cir, source, point, coordinates)
            metrics = v551.score_tensors(prediction.to(torch.complex64), truth_h)
            regularizer = (
                1e-5 * diagnostics["amplitude_delta"].square().mean()
                + 1e-5 * diagnostics["phase_delta"].square().mean()
                + 1e-5 * diagnostics["point_delta"].square().mean()
            )
            loss = 1.0 - metrics["official_score"] + regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite v552 loss fold={fold} epoch={epoch} ids={ids.tolist()}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite v552 gradient fold={fold} epoch={epoch}"
                )
            optimizer.step()
            losses.append(float(loss.detach()))
            scores.append(float(metrics["official_score"].detach()))
            grad_norms.append(float(grad_norm.detach()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_batch_score": float(np.mean(scores)),
            "grad_norm_mean": float(np.mean(grad_norms)),
        }
        history.append(row)
        print(json.dumps({"fold": fold, **row}), flush=True)
    saved: dict[str, object] = {
        "state_dict": model.state_dict(),
        "point_dim": int(store.raw_features.shape[1]),
        "width": args.width,
        "fold": fold,
        "fit_ids": fit_ids.tolist(),
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "history": history,
        "seconds": float(time.time() - started),
    }
    torch.save(saved, path)
    return model, saved


def shuffled_source_ids(
    outer_ids: np.ndarray,
    regimes: np.ndarray,
    fold: int,
) -> np.ndarray:
    output = outer_ids.copy()
    rng = np.random.default_rng(SEED + 1000 + fold)
    for regime in REGIMES:
        local = np.flatnonzero(regimes[outer_ids] == regime)
        output[local] = rng.permutation(outer_ids[local])
    return output


def empty_metrics(rows: int) -> dict[str, np.ndarray]:
    return {
        "pas": np.empty(rows, dtype=np.float32),
        "pdp": np.empty(rows, dtype=np.float32),
        "error": np.empty(rows, dtype=np.float64),
        "energy": np.empty(rows, dtype=np.float64),
    }


def assign_scores(
    target: dict[str, np.ndarray],
    ids: np.ndarray,
    scores: dict[str, torch.Tensor],
) -> None:
    target["pas"][ids] = scores["pas"].float().cpu().numpy()
    target["pdp"][ids] = scores["pdp"].float().cpu().numpy()
    target["error"][ids] = scores["point_error"].double().cpu().numpy()
    target["energy"][ids] = scores["point_energy"].double().cpu().numpy()


@torch.no_grad()
def evaluate_fold(
    fold: int,
    model: v551.CarrierConditionedResidualNet,
    outer_ids: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    regimes: np.ndarray,
    store: TrainStore,
    coordinates: tuple[torch.Tensor, ...],
    batch_size: int,
    baseline: dict[str, np.ndarray],
    candidates: dict[float, dict[str, np.ndarray]],
    shuffled: dict[float, dict[str, np.ndarray]],
) -> dict[str, float]:
    model.eval()
    permutation = shuffled_source_ids(outer_ids, regimes, fold)
    amplitude_sum = 0.0
    phase_sum = 0.0
    point_sum = 0.0
    action_rows = 0
    for start in range(0, len(outer_ids), batch_size):
        end = min(start + batch_size, len(outer_ids))
        ids = outer_ids[start:end]
        source_ids = permutation[start:end]
        base_h, base_cir, truth_h, source, point = store.load(ids, mean, std)
        base_scores = v551.score_tensors(base_h, truth_h)
        assign_scores(baseline, ids, base_scores)
        _, diagnostics = model(base_cir, source, point, coordinates)
        _, _, _, shuffled_source, _ = store.load(ids, mean, std, source_ids=source_ids)
        _, shuffled_diagnostics = model(base_cir, shuffled_source, point, coordinates)
        for alpha in ALPHA_GRID:
            prediction = apply_diagnostics(base_cir, diagnostics, alpha)
            assign_scores(
                candidates[alpha], ids, v551.score_tensors(prediction, truth_h)
            )
            shuffled_prediction = apply_diagnostics(
                base_cir, shuffled_diagnostics, alpha
            )
            assign_scores(
                shuffled[alpha],
                ids,
                v551.score_tensors(shuffled_prediction, truth_h),
            )
        amplitude_sum += float(diagnostics["amplitude_delta"].abs().sum().cpu())
        phase_sum += float(diagnostics["phase_delta"].abs().sum().cpu())
        point_sum += float(diagnostics["point_delta"].abs().sum().cpu())
        action_rows += len(ids)
    return {
        "amplitude_abs_mean": amplitude_sum / (action_rows * 1024 * 192),
        "phase_abs_mean": phase_sum / (action_rows * 1024 * 192),
        "point_delta_abs_mean": point_sum / action_rows,
    }


def summary(
    metrics: dict[str, np.ndarray], mask: np.ndarray | None = None
) -> dict[str, float]:
    return v551.summarize_scores(
        metrics["pas"],
        metrics["pdp"],
        metrics["error"],
        metrics["energy"],
        mask,
    )


def delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return v551.score_delta(candidate, baseline)


def load_fold_models(
    device: torch.device,
) -> tuple[list[v551.CarrierConditionedResidualNet], list[dict[str, object]]]:
    models = []
    checkpoints = []
    for fold in range(5):
        saved = torch.load(
            checkpoint_path(fold), map_location=device, weights_only=False
        )
        model = v551.CarrierConditionedResidualNet(
            int(saved["point_dim"]), int(saved["width"])
        ).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        models.append(model)
        checkpoints.append(saved)
    return models, checkpoints


@torch.no_grad()
def predict_test(
    alpha: float,
    test_store: TestStore,
    models: list[v551.CarrierConditionedResidualNet],
    checkpoints: list[dict[str, object]],
    coordinates: tuple[torch.Tensor, ...],
    batch_size: int,
) -> dict[str, float]:
    base = np.load(TEST_BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.complex64,
        shape=base.shape,
    )
    amplitude_sum = 0.0
    phase_sum = 0.0
    point_signed_sum = 0.0
    point_abs_sum = 0.0
    action_rows = 0
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        ids = np.arange(start, end, dtype=np.int64)
        _, base_cir, source = test_store.load_base(ids)
        prediction_sum = None
        for model, saved in zip(models, checkpoints, strict=True):
            mean = np.asarray(saved["mean"], dtype=np.float32)[None, :]
            std = np.asarray(saved["std"], dtype=np.float32)[None, :]
            point = test_store.point(ids, mean, std)
            _, diagnostics = model(base_cir, source, point, coordinates)
            prediction = apply_diagnostics(base_cir, diagnostics, alpha)
            prediction_sum = (
                prediction if prediction_sum is None else prediction_sum + prediction
            )
            amplitude_sum += float(diagnostics["amplitude_delta"].abs().sum().cpu())
            phase_sum += float(diagnostics["phase_delta"].abs().sum().cpu())
            point_signed_sum += float(diagnostics["point_delta"].sum().cpu())
            point_abs_sum += float(diagnostics["point_delta"].abs().sum().cpu())
            action_rows += len(ids)
        output[start:end] = (
            (prediction_sum / len(models)).cpu().numpy().astype(np.complex64)
        )
        if end == len(base) or end % 100 == 0:
            output.flush()
            print(f"v552 test inference {end}/{len(base)}", flush=True)
    output.flush()
    return {
        "amplitude_abs_mean_unscaled": amplitude_sum / (action_rows * 1024 * 192),
        "phase_abs_mean_unscaled": phase_sum / (action_rows * 1024 * 192),
        "point_delta_mean_unscaled": point_signed_sum / action_rows,
        "point_delta_abs_mean_unscaled": point_abs_sum / action_rows,
    }


def compare_to_base() -> dict[str, float | list[float]]:
    output = np.load(OUTPUT, mmap_mode="r")
    base = np.load(TEST_BASE, mmap_mode="r")
    error_sum = 0.0
    base_sum = 0.0
    numerator = 0.0
    output_sum = 0.0
    ratios = []
    point_nmse = []
    max_abs = 0.0
    finite = True
    for start in range(0, len(output), 5):
        end = min(start + 5, len(output))
        candidate = np.asarray(output[start:end], dtype=np.complex64)
        reference = np.asarray(base[start:end], dtype=np.complex64)
        finite = finite and bool(
            np.isfinite(candidate.real).all() and np.isfinite(candidate.imag).all()
        )
        difference = candidate - reference
        candidate_power = np.sum(
            np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        base_power = np.sum(np.abs(reference) ** 2, axis=(1, 2, 3), dtype=np.float64)
        local_error = np.sum(np.abs(difference) ** 2, axis=(1, 2, 3), dtype=np.float64)
        error_sum += float(np.sum(local_error))
        base_sum += float(np.sum(base_power))
        output_sum += float(np.sum(candidate_power))
        numerator += float(
            np.real(np.vdot(reference.reshape(-1), candidate.reshape(-1)))
        )
        ratios.extend((candidate_power / np.maximum(base_power, 1e-30)).tolist())
        point_nmse.extend((local_error / np.maximum(base_power, 1e-30)).tolist())
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
    ratios_array = np.asarray(ratios, dtype=np.float64)
    point_nmse_array = np.asarray(point_nmse, dtype=np.float64)
    return {
        "finite": finite,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "nmse_to_v469": error_sum / max(base_sum, 1e-30),
        "complex_cosine_to_v469": numerator
        / max(math.sqrt(base_sum * output_sum), 1e-30),
        "max_abs_difference": max_abs,
        "power_ratio_mean": float(np.mean(ratios_array)),
        "power_ratio_p10": float(np.quantile(ratios_array, 0.10)),
        "power_ratio_median": float(np.median(ratios_array)),
        "power_ratio_p90": float(np.quantile(ratios_array, 0.90)),
        "point_nmse_mean": float(np.mean(point_nmse_array)),
        "point_nmse_p90": float(np.quantile(point_nmse_array, 0.90)),
    }


def save_bundle(
    checkpoints: list[dict[str, object]],
    alpha: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "version": VERSION,
            "alpha": alpha,
            "source_names": [name for name, _ in TRAIN_SOURCES],
            "train_carrier": PATHS.relative(TRAIN_BASE),
            "test_carrier": PATHS.relative(TEST_BASE),
            "width": args.width,
            "folds": checkpoints,
        },
        BUNDLE,
    )


def package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "train_v552_full_carrier_conditioned_fullh.py",
        "train_v551_carrier_conditioned_fullh_pilot.py",
        "analyze_v401_dualdomain_pas_preserving_confocal.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    require(OUTPUT)
    require(BUNDLE)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(BUNDLE, "model/v552_inference_bundle.pt")
        for fold in range(5):
            archive.write(checkpoint_path(fold), f"model/v552_fold{fold}.pt")
        archive.write(REPORT, "model/v552_full_carrier_conditioned_fullh.json")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    return {
        "archive": PATHS.relative(ZIP_OUT),
        "zip_crc": True,
        "zip_members": members,
        "zip_bytes": int(ZIP_OUT.stat().st_size),
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    args = parse_args()
    required = (
        TRAIN_H,
        TRAIN_POS,
        TEST_POS,
        TRAIN_BASE,
        TEST_BASE,
        POINT_FEATURES,
        GEOMETRY,
        *(path for _, path in TRAIN_SOURCES),
        *(path for _, path in TEST_SOURCES),
    )
    for path in required:
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
        torch.backends.cuda.matmul.allow_tf32 = True

    started = time.time()
    build_source_cache(
        TRAIN_SOURCES,
        TRAIN_SOURCE_FEATURES,
        TRAIN_SOURCE_POINT_POWER,
        TRAIN_SOURCE_DONE,
        2000,
        "train",
        device,
        args.cache_batch_size,
        args.rebuild_cache,
    )
    build_source_cache(
        TEST_SOURCES,
        TEST_SOURCE_FEATURES,
        TEST_SOURCE_POINT_POWER,
        TEST_SOURCE_DONE,
        500,
        "test",
        device,
        args.cache_batch_size,
        args.rebuild_cache,
    )
    raw_features = build_raw_point_features(args.rebuild_cache)
    positions = np.load(TRAIN_POS).astype(np.float32)
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    geometry = np.load(GEOMETRY)
    regimes = regime_of(geometry["train_blocked"]).astype("U8")
    train_store = TrainStore(raw_features, device)
    coordinates = v551.link_coordinates(device)

    baseline = empty_metrics(2000)
    candidates = {alpha: empty_metrics(2000) for alpha in ALPHA_GRID}
    shuffled = {alpha: empty_metrics(2000) for alpha in ALPHA_GRID}
    fold_records = []
    checkpoints = []
    models = []
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, outer_ids).astype(np.int64)
        mean, std = normalization(raw_features, fit_ids)
        model, saved = train_fold(
            fold, fit_ids, mean, std, train_store, coordinates, args
        )
        actions = evaluate_fold(
            fold,
            model,
            outer_ids,
            mean,
            std,
            regimes,
            train_store,
            coordinates,
            args.batch_size,
            baseline,
            candidates,
            shuffled,
        )
        fold_records.append(
            {
                "fold": fold,
                "fit_count": len(fit_ids),
                "outer_count": len(outer_ids),
                "checkpoint": PATHS.relative(checkpoint_path(fold)),
                "training_seconds": float(saved["seconds"]),
                "actions": actions,
            }
        )
        checkpoints.append(saved)
        models.append(model)
        print(f"v552 completed fold {fold}", flush=True)

    baseline_summary = summary(baseline)
    alpha_results = {}
    for alpha in ALPHA_GRID:
        candidate_summary = summary(candidates[alpha])
        shuffled_summary = summary(shuffled[alpha])
        alpha_results[str(alpha)] = {
            "candidate": candidate_summary,
            "delta": delta(candidate_summary, baseline_summary),
            "shuffled_source": shuffled_summary,
            "candidate_minus_shuffled": float(
                candidate_summary["official_score"] - shuffled_summary["official_score"]
            ),
        }
    selected_alpha = max(
        ALPHA_GRID,
        key=lambda value: alpha_results[str(value)]["candidate"]["official_score"],
    )
    selected_metrics = candidates[selected_alpha]
    selected_summary = alpha_results[str(selected_alpha)]["candidate"]
    selected_delta = alpha_results[str(selected_alpha)]["delta"]
    selected_shuffled = alpha_results[str(selected_alpha)]["shuffled_source"]
    source_margin = alpha_results[str(selected_alpha)]["candidate_minus_shuffled"]

    fold_deltas = []
    for fold, outer_ids in enumerate(folds):
        base_fold = summary(baseline, np.isin(all_ids, outer_ids))
        candidate_fold = summary(selected_metrics, np.isin(all_ids, outer_ids))
        fold_records[fold]["baseline"] = base_fold
        fold_records[fold]["candidate"] = candidate_fold
        fold_records[fold]["delta"] = delta(candidate_fold, base_fold)
        fold_deltas.append(fold_records[fold]["delta"]["official_score"])
    fold_wins = int(np.sum(np.asarray(fold_deltas) > 0))
    by_regime = {}
    for regime in REGIMES:
        mask = regimes == regime
        base_regime = summary(baseline, mask)
        candidate_regime = summary(selected_metrics, mask)
        by_regime[regime] = {
            "baseline": base_regime,
            "candidate": candidate_regime,
            "delta": delta(candidate_regime, base_regime),
        }

    test_store = TestStore(raw_features, device)
    test_actions = predict_test(
        selected_alpha,
        test_store,
        models,
        checkpoints,
        coordinates,
        args.batch_size,
    )
    comparison = compare_to_base()
    save_bundle(checkpoints, selected_alpha, args)
    scale_up_evidence = bool(
        selected_delta["official_score"] >= 0.005
        and source_margin >= 0.001
        and fold_wins >= 4
    )
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The v551 same-carrier full-H residual model scales to all official "
            "positions and transfers its multi-configuration Sionna information to "
            "the semantically matched v469 test carrier."
        ),
        "new_information": [
            "same-carrier joint amplitude, phase, and point-power learning",
            "four same-coordinate Sionna link-delay power sources",
            "five-model spatial-fold ensemble",
        ],
        "evidence_boundary": {
            "train_truth": "official 2000-point H",
            "train_carrier": PATHS.relative(TRAIN_BASE),
            "test_carrier": PATHS.relative(TEST_BASE),
            "uses_online_score_for_selection": False,
            "source_names": [name for name, _ in TRAIN_SOURCES],
        },
        "model": {
            "architecture": (
                "v551 point-conditioned shared link-delay residual CNN with "
                "pol/H/V/UE embeddings and amplitude, phase, point-power heads"
            ),
            "width": args.width,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "precision": "float32",
            "fold_models": 5,
            "alpha_grid": list(ALPHA_GRID),
            "selected_alpha": selected_alpha,
        },
        "oof": {
            "baseline": baseline_summary,
            "alpha_results": alpha_results,
            "selected_candidate": selected_summary,
            "selected_delta": selected_delta,
            "selected_shuffled_source": selected_shuffled,
            "source_margin": source_margin,
            "fold_wins": fold_wins,
            "folds": fold_records,
            "by_regime": by_regime,
            "passes_v551_scale_up_criteria": scale_up_evidence,
        },
        "deployment": {
            "ensemble": "complex mean of five spatial-fold model predictions",
            "test_actions": test_actions,
            "comparison_to_v469": comparison,
            "output": PATHS.relative(OUTPUT),
            "bundle": PATHS.relative(BUNDLE),
        },
        "runtime": {
            "seconds_before_packaging": float(time.time() - started),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.skip_package:
        report["integrity"] = package(report)
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_alpha": selected_alpha,
                "oof_baseline": baseline_summary,
                "oof_candidate": selected_summary,
                "oof_delta": selected_delta,
                "source_margin": source_margin,
                "fold_wins": fold_wins,
                "passes_scale_up_criteria": scale_up_evidence,
                "comparison_to_v469": comparison,
                "archive": report.get("integrity", {}).get("archive"),
                "runtime_seconds": time.time() - started,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
