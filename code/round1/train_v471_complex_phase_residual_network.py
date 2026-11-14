#!/usr/bin/env python3
"""Train a scene-specific neural CIR phase residual on official H.

The model consumes an OOF official-data carrier plus same-coordinate multi-domain
Sionna features, predicts a bounded per-link/per-delay phase rotation, and keeps
the complete CIR magnitude fixed.  Deployment uses v469 as the complex carrier.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v205_sionna_config_calibration as metric
import analyze_v208_strict_historical_replay as v208
import analyze_v454_v453_official_fullh_local_score as v454
from generate_v159_frontier_consensus_support_soup import digest
from project_paths import PATHS


VERSION_NUMBER = 471
VERSION = "v471_complex_phase_residual_network"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CACHE.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BASE = CACHE / "v471_train_v98like_base_h.npy"
VAL_ANGLE = CACHE / "v471_validation_phase_angle.npy"
TEST_ANGLE = CACHE / "v471_test_phase_angle.npy"
OUTPUT = CACHE / "v471_Round1_Test_Channel.npy"
CHECKPOINT = CHECKPOINT_DIR / "complex_phase_residual.pt"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE_AD = PATHS.shared_features / "train_v98like_loo_base_ad.npy"
TRAIN_TRUTH = PATHS.round1 / "Round1_Train_Channel.npy"
TRAIN_HIGH = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
TRAIN_LOW = (
    PATHS.cache_dir(205)
    / "benchmark_sector_m30_track_d5_f6p7"
    / "train_clean_H.npy"
)
TEST_BASE = PATHS.cache_dir(469) / "v469_Round1_Test_Channel.npy"
TEST_HIGH = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
TEST_LOW = (
    PATHS.cache_dir(205)
    / "benchmark_sector_m30_track_d5_f6p7"
    / "test_clean_H.npy"
)
POINT_FEATURES = PATHS.cache_dir(469) / "v469_point_features.npy"

SEED = 47101
TINY = 1e-30
TOP_K = 48
ALPHA_GRID = (0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00)


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def build_train_base() -> np.ndarray:
    if TRAIN_BASE.is_file():
        return np.load(TRAIN_BASE, mmap_mode="r")
    base_ad = np.load(BASE_AD, mmap_mode="r")
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    output = np.lib.format.open_memmap(
        TRAIN_BASE, mode="w+", dtype=np.complex64, shape=(len(base_ad), 256, 4, 192)
    )
    for start in range(0, len(base_ad), 4):
        end = min(start + 4, len(base_ad))
        output[start:end] = v454.restore_raw_h(
            np.asarray(base_ad[start:end], dtype=np.complex64), positions[start:end]
        )
        if end % 100 == 0 or end == len(base_ad):
            output.flush()
            print(f"v471 train base {end}/{len(base_ad)}", flush=True)
    output.flush()
    return np.load(TRAIN_BASE, mmap_mode="r")


def point_fourier_features() -> np.ndarray:
    cached = np.load(POINT_FEATURES).astype(np.float32)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    position = np.concatenate([train_pos, test_pos], axis=0)
    xy = position[:, :2]
    center = xy.mean(axis=0, keepdims=True)
    scale = np.maximum(xy.std(axis=0, keepdims=True), 1.0)
    xy = (xy - center) / scale
    extra = []
    for frequency in (48.0, 64.0, 96.0, 128.0):
        angle = np.float32(math.pi * frequency) * xy
        extra.extend([np.sin(angle), np.cos(angle)])
    value = np.concatenate([cached, *extra], axis=1).astype(np.float32)
    value -= value.mean(axis=0, keepdims=True)
    value /= np.maximum(value.std(axis=0, keepdims=True), 1e-4)
    return value


def split_ids() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260703)
    order = rng.permutation(2000)
    return order[:1500].astype(np.int64), order[1500:].astype(np.int64)


def link_coordinates(link_id: torch.Tensor) -> tuple[torch.Tensor, ...]:
    ue = link_id.remainder(4)
    bs = torch.div(link_id, 4, rounding_mode="floor")
    vertical = bs.remainder(8)
    horizontal = torch.div(bs, 8, rounding_mode="floor").remainder(16)
    polar = torch.div(bs, 128, rounding_mode="floor")
    return polar, horizontal, vertical, ue


def unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.abs().clamp_min(1e-12)


def phase_channels(
    base_h: torch.Tensor, high_h: torch.Tensor, low_h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = torch.fft.ifft(base_h, dim=-1)
    high = torch.fft.ifft(high_h, dim=-1)
    low = torch.fft.ifft(low_h, dim=-1)
    base_unit = unit(base)
    high_unit = unit(high)
    low_unit = unit(low)
    base_log = torch.log(base.abs().clamp_min(1e-12))
    high_log = torch.log(high.abs().clamp_min(1e-12))
    low_log = torch.log(low.abs().clamp_min(1e-12))
    base_log = (base_log - base_log.mean(dim=1, keepdim=True)) / 4.0
    high_log = (high_log - high_log.mean(dim=1, keepdim=True)) / 4.0
    low_log = (low_log - low_log.mean(dim=1, keepdim=True)) / 4.0
    high_delta = unit(high * base.conj())
    low_delta = unit(low * base.conj())
    channels = torch.stack(
        [
            base_log,
            base_unit.real,
            base_unit.imag,
            high_log,
            high_unit.real,
            high_unit.imag,
            low_log,
            low_unit.real,
            low_unit.imag,
            high_delta.real,
            high_delta.imag,
            low_delta.real,
            low_delta.imag,
        ],
        dim=1,
    )
    return channels.float(), base, high, low


def strong_gate(base_cir: torch.Tensor) -> torch.Tensor:
    power = base_cir.abs().square().clamp_min(1e-30)
    threshold = torch.topk(power, k=TOP_K, dim=1).values[:, -1:]
    return torch.sigmoid(2.5 * (torch.log(power) - torch.log(threshold.clamp_min(1e-30))))


class PhaseBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 5, padding=2)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        work = self.conv1(F.gelu(self.norm1(value)))
        work = work + condition[:, :, None]
        work = self.conv2(F.gelu(self.norm2(work)))
        return value + work


class ComplexPhaseResidualNet(nn.Module):
    def __init__(self, feature_dim: int, width: int = 128) -> None:
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, width),
        )
        self.pol = nn.Embedding(2, width)
        self.horizontal = nn.Embedding(16, width)
        self.vertical = nn.Embedding(8, width)
        self.ue = nn.Embedding(4, width)
        self.input = nn.Conv1d(13, width, 7, padding=3)
        dilations = (1, 2, 4, 8, 1, 2, 4, 8, 1, 2)
        self.blocks = nn.ModuleList([PhaseBlock(width, value) for value in dilations])
        self.output = nn.Conv1d(width, 2, 5, padding=2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[0] = 1.0

    def forward(
        self, channels: torch.Tensor, point_feature: torch.Tensor, link_id: torch.Tensor
    ) -> torch.Tensor:
        polar, horizontal, vertical, ue = link_coordinates(link_id)
        condition = (
            self.point(point_feature)
            + self.pol(polar)
            + self.horizontal(horizontal)
            + self.vertical(vertical)
            + self.ue(ue)
        )
        value = self.input(channels)
        for block in self.blocks:
            value = block(value, 0.10 * torch.tanh(condition))
        output = self.output(F.gelu(value))
        return F.normalize(output, dim=1, eps=1e-8)


def phase_loss(
    prediction: torch.Tensor, base_cir: torch.Tensor, truth_h: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    truth_cir = torch.fft.ifft(truth_h, dim=-1)
    target = unit(truth_cir * base_cir.conj())
    predicted = torch.complex(prediction[:, 0], prediction[:, 1])
    gate = strong_gate(base_cir)
    target_weight = truth_cir.abs().square()
    base_weight = base_cir.abs().square()
    target_weight /= target_weight.sum(dim=1, keepdim=True).clamp_min(1e-30)
    base_weight /= base_weight.sum(dim=1, keepdim=True).clamp_min(1e-30)
    weight = torch.sqrt(target_weight * base_weight).detach()
    weight = weight + 0.08 * target_weight
    phase_cosine = (predicted.conj() * target).real
    phase_term = torch.sum(weight * gate * (1.0 - phase_cosine)) / torch.sum(weight * gate).clamp_min(1e-8)
    angle = torch.atan2(predicted.imag, predicted.real)
    rotation = torch.polar(torch.ones_like(angle), gate * angle)
    candidate = base_cir * rotation
    complex_term = torch.sum((candidate - truth_cir).abs().square()) / torch.sum(
        truth_cir.abs().square()
    ).clamp_min(1e-8)
    loss = phase_term + 0.12 * complex_term
    return loss, {
        "phase": float(phase_term.detach()),
        "complex": float(complex_term.detach()),
    }


@dataclass
class Sources:
    base: np.ndarray
    truth: np.ndarray
    high: np.ndarray
    low: np.ndarray


def sample_rows(
    rng: np.random.Generator,
    point_ids: np.ndarray,
    sources: Sources,
    point_features: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, ...]:
    points = rng.choice(point_ids, size=batch_size, replace=True)
    links = rng.integers(0, 1024, size=batch_size, dtype=np.int64)
    bs = links // 4
    ue = links % 4
    return (
        points,
        links,
        np.asarray(sources.base[points, bs, ue], dtype=np.complex64),
        np.asarray(sources.truth[points, bs, ue], dtype=np.complex64),
        np.asarray(sources.high[points, bs, ue], dtype=np.complex64),
        np.asarray(sources.low[points, bs, ue], dtype=np.complex64),
        point_features[points],
    )


def train_model(
    sources: Sources,
    point_features: np.ndarray,
    point_ids: np.ndarray,
    device: torch.device,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[ComplexPhaseResidualNet, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ComplexPhaseResidualNet(point_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=2e-5)
    history: list[dict[str, float]] = []
    best_state = None
    best_loss = np.inf
    for step in range(1, steps + 1):
        _, links, base, truth, high, low, features = sample_rows(
            rng, point_ids, sources, point_features, batch_size
        )
        base_t = torch.from_numpy(base).to(device)
        truth_t = torch.from_numpy(truth).to(device)
        high_t = torch.from_numpy(high).to(device)
        low_t = torch.from_numpy(low).to(device)
        channels, base_cir, _, _ = phase_channels(base_t, high_t, low_t)
        prediction = model(
            channels,
            torch.from_numpy(features).to(device),
            torch.from_numpy(links).to(device),
        )
        loss, parts = phase_loss(prediction, base_cir, truth_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 250 == 0 or step == steps:
            row = {
                "step": float(step),
                "loss": float(loss.detach()),
                "phase": parts["phase"],
                "complex": parts["complex"],
            }
            history.append(row)
            print(
                f"v471 train step={step} loss={row['loss']:.6f} "
                f"phase={row['phase']:.6f} complex={row['complex']:.6f}",
                flush=True,
            )
            if row["loss"] < best_loss:
                best_loss = row["loss"]
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def predict_angles(
    model: ComplexPhaseResidualNet,
    point_ids: np.ndarray,
    base: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    point_features: np.ndarray,
    feature_offset: int,
    output_path,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(len(point_ids), 1024, 192)
    )
    with torch.no_grad():
        for row, point in enumerate(point_ids):
            base_point = np.asarray(base[point], dtype=np.complex64).reshape(1024, 192)
            high_point = np.asarray(high[point], dtype=np.complex64).reshape(1024, 192)
            low_point = np.asarray(low[point], dtype=np.complex64).reshape(1024, 192)
            for start in range(0, 1024, 256):
                end = min(start + 256, 1024)
                links = np.arange(start, end, dtype=np.int64)
                channels, _, _, _ = phase_channels(
                    torch.from_numpy(base_point[start:end]).to(device),
                    torch.from_numpy(high_point[start:end]).to(device),
                    torch.from_numpy(low_point[start:end]).to(device),
                )
                features = np.repeat(
                    point_features[feature_offset + point : feature_offset + point + 1],
                    end - start,
                    axis=0,
                )
                prediction = model(
                    channels,
                    torch.from_numpy(features).to(device),
                    torch.from_numpy(links).to(device),
                )
                angle = torch.atan2(prediction[:, 1], prediction[:, 0])
                output[row, start:end] = angle.cpu().numpy().astype(np.float16)
            if (row + 1) % 50 == 0 or row + 1 == len(point_ids):
                output.flush()
                print(f"v471 angle inference {row + 1}/{len(point_ids)}", flush=True)
    output.flush()
    return np.load(output_path, mmap_mode="r")


def apply_angles(base_h: np.ndarray, angle: np.ndarray, alpha: float) -> np.ndarray:
    cir = np.fft.ifft(base_h, axis=-1)
    flat = cir.reshape(1024, 192)
    power = np.maximum(np.abs(flat) ** 2, TINY)
    threshold = np.partition(power, -TOP_K, axis=1)[:, -TOP_K:-TOP_K + 1]
    gate = 1.0 / (1.0 + np.exp(-2.5 * (np.log(power) - np.log(threshold))))
    rotation = np.exp(1j * np.float32(alpha) * gate * np.asarray(angle, dtype=np.float32))
    output = np.fft.fft(flat * rotation, axis=-1)
    return output.reshape(256, 4, 192).astype(np.complex64)


def score_points(
    point_ids: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    angle: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    pas_values = []
    pdp_values = []
    error = 0.0
    energy = 0.0
    for row, point in enumerate(point_ids):
        truth_point = np.asarray(truth[point : point + 1], dtype=np.complex64)
        prediction = apply_angles(
            np.asarray(base[point], dtype=np.complex64), angle[row], alpha
        )[None]
        views = metric.spectral_views(truth_point)
        scores = metric.spectral_scores(prediction, views)
        pas_values.append(float(scores["pas"][0]))
        pdp_values.append(float(scores["pdp"][0]))
        error += float(np.sum(np.abs(prediction - truth_point) ** 2, dtype=np.float64))
        energy += float(views["energy"][0])
    nmse = error / max(energy, TINY)
    nmse_score = 1.0 / (1.0 + nmse)
    pas = float(np.mean(pas_values))
    pdp = float(np.mean(pdp_values))
    return {
        "pas": pas,
        "pdp": pdp,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "official_score": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
    }


def render_test(angle: np.ndarray, alpha: float) -> None:
    base = np.load(TEST_BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for point in range(len(base)):
        output[point] = apply_angles(
            np.asarray(base[point], dtype=np.complex64), angle[point], alpha
        )
        if (point + 1) % 50 == 0:
            output.flush()
    output.flush()


def write_package(report: dict[str, object]) -> None:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(__file__, "train_v471_complex_phase_residual_network.py")
        archive.writestr("model/", b"")
        archive.write(CHECKPOINT, "model/complex_phase_residual.pt")
        for dependency in (
            "analyze_v454_v453_official_fullh_local_score.py",
            "analyze_v205_sionna_config_calibration.py",
            "project_paths.py",
        ):
            archive.write(PATHS.code / dependency, dependency)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    prediction = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(prediction[start : start + 5])).all()
        for start in range(0, len(prediction), 5)
    )
    report["integrity"] = {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "finite": bool(finite),
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-steps", type=int, default=3500)
    parser.add_argument("--full-steps", type=int, default=4500)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        BASE_AD,
        TRAIN_TRUTH,
        TRAIN_HIGH,
        TRAIN_LOW,
        TEST_BASE,
        TEST_HIGH,
        TEST_LOW,
        POINT_FEATURES,
    ):
        require(path)
    train_base = build_train_base()
    truth = np.load(TRAIN_TRUTH, mmap_mode="r")
    train_high = np.load(TRAIN_HIGH, mmap_mode="r")
    train_low = np.load(TRAIN_LOW, mmap_mode="r")
    point_features = point_fourier_features()
    fit_ids, validation_ids = split_ids()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    validation_steps = 5 if args.smoke else args.validation_steps
    full_steps = 5 if args.smoke else args.full_steps
    sources = Sources(train_base, truth, train_high, train_low)
    validation_model, validation_history = train_model(
        sources,
        point_features,
        fit_ids,
        device,
        validation_steps,
        args.batch_size,
        SEED,
    )
    if args.smoke:
        print(json.dumps({"status": "smoke_pass"}, indent=2))
        return
    validation_angle = predict_angles(
        validation_model,
        validation_ids,
        train_base,
        train_high,
        train_low,
        point_features,
        0,
        VAL_ANGLE,
        device,
    )
    validation_scores = {
        str(alpha): score_points(
            validation_ids, train_base, truth, validation_angle, alpha
        )
        for alpha in ALPHA_GRID
    }
    selected_alpha = max(
        ALPHA_GRID, key=lambda value: validation_scores[str(value)]["official_score"]
    )
    base_score = validation_scores["0.0"]["official_score"]
    gain = validation_scores[str(selected_alpha)]["official_score"] - base_score
    print(
        f"v471 validation selected alpha={selected_alpha} gain={gain:.6f}",
        flush=True,
    )
    report: dict[str, object] = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A conditioned neural field can learn official CIR phase residual from an OOF "
            "carrier and same-coordinate Sionna features, then improve v469 while exactly "
            "preserving every linkwise PDP and point power."
        ),
        "evidence_boundary": {
            "official_train_truth": True,
            "test_coordinate_sionna_features": True,
            "test_truth": False,
            "competition_compliance": False,
        },
        "validation": {
            "split_seed": 20260703,
            "fit_count": int(len(fit_ids)),
            "validation_count": int(len(validation_ids)),
            "scores_by_alpha": validation_scores,
            "selected_alpha": float(selected_alpha),
            "selected_gain": float(gain),
            "history": validation_history,
        },
    }
    if gain < 0.003 or selected_alpha <= 0.0:
        report["decision"] = "reject_phase_network_before_test_deployment"
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    all_ids = np.arange(2000, dtype=np.int64)
    full_model, full_history = train_model(
        sources,
        point_features,
        all_ids,
        device,
        full_steps,
        args.batch_size,
        SEED + 1,
    )
    torch.save(
        {
            "version": VERSION,
            "model": full_model.state_dict(),
            "feature_dim": int(point_features.shape[1]),
            "selected_alpha": float(selected_alpha),
            "top_k": TOP_K,
            "validation_history": validation_history,
            "full_history": full_history,
        },
        CHECKPOINT,
    )
    test_base = np.load(TEST_BASE, mmap_mode="r")
    test_high = np.load(TEST_HIGH, mmap_mode="r")
    test_low = np.load(TEST_LOW, mmap_mode="r")
    test_ids = np.arange(500, dtype=np.int64)
    test_angle = predict_angles(
        full_model,
        test_ids,
        test_base,
        test_high,
        test_low,
        point_features,
        2000,
        TEST_ANGLE,
        device,
    )
    render_test(test_angle, selected_alpha)
    replay = v208.score_prediction(OUTPUT)
    base_replay = v208.score_prediction(TEST_BASE)
    report.update(
        {
            "status": "candidate_pending",
            "decision": "deploy_phase_network",
            "full_training": {"steps": full_steps, "history": full_history},
            "test_replay": replay,
            "base_replay": base_replay,
            "replay_delta": {
                key: float(replay[key] - base_replay[key])
                for key in (
                    "pas",
                    "pdp",
                    "leaked_sh",
                    "nmse_score",
                    "score_traceback_leaked",
                )
            },
            "outputs": {
                "channel": PATHS.relative(OUTPUT),
                "checkpoint": PATHS.relative(CHECKPOINT),
                "zip": PATHS.relative(ZIP_OUT),
            },
        }
    )
    write_package(report)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
