#!/usr/bin/env python3
"""Learn official CIR magnitude and link-energy residuals over a complex carrier."""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v205_sionna_config_calibration as metric
import analyze_v208_strict_historical_replay as v208
import train_v471_complex_phase_residual_network as v471
from generate_v159_frontier_consensus_support_soup import digest
from project_paths import PATHS


VERSION_NUMBER = 472
VERSION = "v472_cir_magnitude_residual_network"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CACHE.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

VAL_SHAPE = CACHE / "v472_validation_link_shape.npy"
VAL_ENERGY = CACHE / "v472_validation_link_energy.npy"
TEST_SHAPE = CACHE / "v472_test_link_shape.npy"
TEST_ENERGY = CACHE / "v472_test_link_energy.npy"
OUTPUT = CACHE / "v472_Round1_Test_Channel.npy"
CHECKPOINT = CHECKPOINT_DIR / "cir_magnitude_residual.pt"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

SHAPE_ALPHAS = (0.0, 0.15, 0.30, 0.50, 0.75, 1.00)
ENERGY_ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75)
FLOOR = 1e-10
SEED = 47201


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.sum(dim=1, keepdim=True).clamp_min(1e-12)


def link_views(channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cir = torch.fft.ifft(channel, dim=-1)
    power = cir.abs().square().float()
    energy = power.sum(dim=1).clamp_min(1e-30)
    shape = power / energy[:, None]
    phase = cir / cir.abs().clamp_min(1e-12)
    return shape, energy, phase


class MagnitudeBlock(nn.Module):
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


class CIRMagnitudeResidualNet(nn.Module):
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
        self.quality = nn.Linear(2, width)
        self.input = nn.Conv1d(7, width, 7, padding=3)
        dilations = (1, 2, 4, 8, 1, 2, 4, 8, 1, 2)
        self.blocks = nn.ModuleList([MagnitudeBlock(width, value) for value in dilations])
        self.shape_head = nn.Conv1d(width, 1, 5, padding=2)
        self.energy_head = nn.Sequential(
            nn.Linear(2 * width, 256), nn.GELU(), nn.Linear(256, 1)
        )
        nn.init.zeros_(self.shape_head.weight)
        nn.init.zeros_(self.shape_head.bias)
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    def forward(
        self,
        base_shape: torch.Tensor,
        base_energy: torch.Tensor,
        high_shape: torch.Tensor,
        high_energy: torch.Tensor,
        low_shape: torch.Tensor,
        low_energy: torch.Tensor,
        point_feature: torch.Tensor,
        link_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        polar, horizontal, vertical, ue = v471.link_coordinates(link_id)
        shape_quality = F.cosine_similarity(high_shape, low_shape, dim=1)
        energy_spread = torch.abs(torch.log(high_energy) - torch.log(low_energy))
        condition = (
            self.point(point_feature)
            + self.pol(polar)
            + self.horizontal(horizontal)
            + self.vertical(vertical)
            + self.ue(ue)
            + self.quality(torch.stack([shape_quality, energy_spread / 4.0], dim=1))
        )
        log_base = torch.log(base_shape.clamp_min(FLOOR))
        log_high = torch.log(high_shape.clamp_min(FLOOR))
        log_low = torch.log(low_shape.clamp_min(FLOOR))
        channels = torch.stack(
            [
                log_base / 8.0,
                log_high / 8.0,
                log_low / 8.0,
                (log_high - log_base) / 4.0,
                (log_low - log_base) / 4.0,
                high_shape,
                low_shape,
            ],
            dim=1,
        )
        value = self.input(channels)
        for block in self.blocks:
            value = block(value, 0.10 * torch.tanh(condition))
        delta = 4.5 * torch.tanh(self.shape_head(F.gelu(value)).squeeze(1))
        prediction_shape = F.softmax(log_base + delta, dim=1)
        pooled = torch.mean(value, dim=2)
        energy_delta = 2.0 * torch.tanh(
            self.energy_head(torch.cat([pooled, condition], dim=1)).squeeze(1)
        )
        prediction_energy = base_energy * torch.exp(energy_delta)
        return prediction_shape, prediction_energy


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
        links,
        np.asarray(sources.base[points, bs, ue], dtype=np.complex64),
        np.asarray(sources.truth[points, bs, ue], dtype=np.complex64),
        np.asarray(sources.high[points, bs, ue], dtype=np.complex64),
        np.asarray(sources.low[points, bs, ue], dtype=np.complex64),
        point_features[points],
    )


def magnitude_loss(
    prediction_shape: torch.Tensor,
    prediction_energy: torch.Tensor,
    truth_shape: torch.Tensor,
    truth_energy: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    cosine = F.cosine_similarity(prediction_shape, truth_shape, dim=1)
    shape_loss = torch.mean(1.0 - cosine)
    log_shape = F.smooth_l1_loss(
        torch.log(prediction_shape.clamp_min(FLOOR)),
        torch.log(truth_shape.clamp_min(FLOOR)),
        beta=0.05,
    )
    energy_loss = F.smooth_l1_loss(
        torch.log(prediction_energy.clamp_min(1e-30)),
        torch.log(truth_energy.clamp_min(1e-30)),
        beta=0.05,
    )
    loss = shape_loss + 0.025 * log_shape + 0.06 * energy_loss
    return loss, {
        "shape": float(shape_loss.detach()),
        "log_shape": float(log_shape.detach()),
        "energy": float(energy_loss.detach()),
    }


def train_model(
    sources: Sources,
    point_features: np.ndarray,
    point_ids: np.ndarray,
    device: torch.device,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[CIRMagnitudeResidualNet, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CIRMagnitudeResidualNet(point_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.3e-3, weight_decay=2e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=2e-5)
    history: list[dict[str, float]] = []
    best_state = None
    best_loss = np.inf
    for step in range(1, steps + 1):
        links, base, truth, high, low, features = sample_rows(
            rng, point_ids, sources, point_features, batch_size
        )
        base_shape, base_energy, _ = link_views(torch.from_numpy(base).to(device))
        truth_shape, truth_energy, _ = link_views(torch.from_numpy(truth).to(device))
        high_shape, high_energy, _ = link_views(torch.from_numpy(high).to(device))
        low_shape, low_energy, _ = link_views(torch.from_numpy(low).to(device))
        prediction_shape, prediction_energy = model(
            base_shape,
            base_energy,
            high_shape,
            high_energy,
            low_shape,
            low_energy,
            torch.from_numpy(features).to(device),
            torch.from_numpy(links).to(device),
        )
        loss, parts = magnitude_loss(
            prediction_shape, prediction_energy, truth_shape, truth_energy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 250 == 0 or step == steps:
            row = {"step": float(step), "loss": float(loss.detach()), **parts}
            history.append(row)
            print(
                f"v472 train step={step} loss={row['loss']:.6f} "
                f"shape={row['shape']:.6f} energy={row['energy']:.6f}",
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


def predict_targets(
    model: CIRMagnitudeResidualNet,
    point_ids: np.ndarray,
    base: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    point_features: np.ndarray,
    feature_offset: int,
    shape_path,
    energy_path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    shape_out = np.lib.format.open_memmap(
        shape_path, mode="w+", dtype=np.float16, shape=(len(point_ids), 1024, 192)
    )
    energy_out = np.lib.format.open_memmap(
        energy_path, mode="w+", dtype=np.float32, shape=(len(point_ids), 1024)
    )
    with torch.no_grad():
        for row, point in enumerate(point_ids):
            base_point = np.asarray(base[point], dtype=np.complex64).reshape(1024, 192)
            high_point = np.asarray(high[point], dtype=np.complex64).reshape(1024, 192)
            low_point = np.asarray(low[point], dtype=np.complex64).reshape(1024, 192)
            for start in range(0, 1024, 256):
                end = min(start + 256, 1024)
                links = np.arange(start, end, dtype=np.int64)
                base_shape, base_energy, _ = link_views(
                    torch.from_numpy(base_point[start:end].copy()).to(device)
                )
                high_shape, high_energy, _ = link_views(
                    torch.from_numpy(high_point[start:end].copy()).to(device)
                )
                low_shape, low_energy, _ = link_views(
                    torch.from_numpy(low_point[start:end].copy()).to(device)
                )
                features = np.repeat(
                    point_features[feature_offset + point : feature_offset + point + 1],
                    end - start,
                    axis=0,
                )
                prediction_shape, prediction_energy = model(
                    base_shape,
                    base_energy,
                    high_shape,
                    high_energy,
                    low_shape,
                    low_energy,
                    torch.from_numpy(features).to(device),
                    torch.from_numpy(links).to(device),
                )
                shape_out[row, start:end] = prediction_shape.cpu().numpy().astype(np.float16)
                energy_out[row, start:end] = prediction_energy.cpu().numpy().astype(np.float32)
            if (row + 1) % 50 == 0 or row + 1 == len(point_ids):
                shape_out.flush()
                energy_out.flush()
                print(f"v472 target inference {row + 1}/{len(point_ids)}", flush=True)
    shape_out.flush()
    energy_out.flush()
    return np.load(shape_path, mmap_mode="r"), np.load(energy_path, mmap_mode="r")


def apply_target(
    base_h: np.ndarray,
    predicted_shape: np.ndarray,
    predicted_energy: np.ndarray,
    shape_alpha: float,
    energy_alpha: float,
) -> np.ndarray:
    cir = np.fft.ifft(base_h, axis=-1).reshape(1024, 192)
    base_abs = np.abs(cir).astype(np.float32)
    base_power = np.maximum(base_abs * base_abs, 1e-30)
    base_energy = base_power.sum(axis=1)
    base_shape = base_power / np.maximum(base_energy[:, None], 1e-30)
    target_shape = np.maximum(np.asarray(predicted_shape, dtype=np.float32), FLOOR)
    log_shape = (
        (1.0 - np.float32(shape_alpha)) * np.log(np.maximum(base_shape, FLOOR))
        + np.float32(shape_alpha) * np.log(target_shape)
    )
    desired_shape = np.exp(log_shape)
    desired_shape /= np.maximum(desired_shape.sum(axis=1, keepdims=True), 1e-30)
    desired_energy = np.exp(
        (1.0 - np.float32(energy_alpha)) * np.log(np.maximum(base_energy, 1e-30))
        + np.float32(energy_alpha)
        * np.log(np.maximum(np.asarray(predicted_energy, dtype=np.float32), 1e-30))
    )
    phase = cir / np.maximum(base_abs, 1e-30)
    magnitude = np.sqrt(np.maximum(desired_shape * desired_energy[:, None], 0.0))
    return np.fft.fft(magnitude * phase, axis=-1).reshape(256, 4, 192).astype(np.complex64)


def score_points(
    point_ids: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    predicted_shape: np.ndarray,
    predicted_energy: np.ndarray,
    shape_alpha: float,
    energy_alpha: float,
) -> dict[str, float]:
    pas_values = []
    pdp_values = []
    error = 0.0
    energy = 0.0
    for row, point in enumerate(point_ids):
        prediction = apply_target(
            np.asarray(base[point], dtype=np.complex64),
            predicted_shape[row],
            predicted_energy[row],
            shape_alpha,
            energy_alpha,
        )[None]
        truth_point = np.asarray(truth[point : point + 1], dtype=np.complex64)
        views = metric.spectral_views(truth_point)
        scores = metric.spectral_scores(prediction, views)
        pas_values.append(float(scores["pas"][0]))
        pdp_values.append(float(scores["pdp"][0]))
        error += float(np.sum(np.abs(prediction - truth_point) ** 2, dtype=np.float64))
        energy += float(views["energy"][0])
    nmse = error / max(energy, 1e-30)
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


def render_test(
    predicted_shape: np.ndarray,
    predicted_energy: np.ndarray,
    shape_alpha: float,
    energy_alpha: float,
) -> None:
    base = np.load(v471.TEST_BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for point in range(len(base)):
        output[point] = apply_target(
            np.asarray(base[point], dtype=np.complex64),
            predicted_shape[point],
            predicted_energy[point],
            shape_alpha,
            energy_alpha,
        )
        if (point + 1) % 50 == 0:
            output.flush()
    output.flush()


def write_package(report: dict[str, object]) -> None:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(__file__, "train_v472_cir_magnitude_residual_network.py")
        archive.writestr("model/", b"")
        archive.write(CHECKPOINT, "model/cir_magnitude_residual.pt")
        archive.write(PATHS.code / "train_v471_complex_phase_residual_network.py", "train_v471_complex_phase_residual_network.py")
        archive.write(PATHS.code_root / "project_paths.py", "project_paths.py")
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
        v471.TRAIN_BASE,
        v471.TRAIN_TRUTH,
        v471.TRAIN_HIGH,
        v471.TRAIN_LOW,
        v471.TEST_BASE,
        v471.TEST_HIGH,
        v471.TEST_LOW,
        v471.POINT_FEATURES,
    ):
        require(path)
    train_base = np.load(v471.TRAIN_BASE, mmap_mode="r")
    truth = np.load(v471.TRAIN_TRUTH, mmap_mode="r")
    train_high = np.load(v471.TRAIN_HIGH, mmap_mode="r")
    train_low = np.load(v471.TRAIN_LOW, mmap_mode="r")
    point_features = v471.point_fourier_features()
    fit_ids, validation_ids = v471.split_ids()
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
    validation_shape, validation_energy = predict_targets(
        validation_model,
        validation_ids,
        train_base,
        train_high,
        train_low,
        point_features,
        0,
        VAL_SHAPE,
        VAL_ENERGY,
        device,
    )
    scores = {}
    for shape_alpha in SHAPE_ALPHAS:
        for energy_alpha in ENERGY_ALPHAS:
            key = f"shape{shape_alpha:.2f}_energy{energy_alpha:.2f}"
            scores[key] = score_points(
                validation_ids,
                train_base,
                truth,
                validation_shape,
                validation_energy,
                shape_alpha,
                energy_alpha,
            )
    selected_key = max(scores, key=lambda key: scores[key]["official_score"])
    selected_shape = float(selected_key.split("_")[0].replace("shape", ""))
    selected_energy = float(selected_key.split("_")[1].replace("energy", ""))
    base_score = scores["shape0.00_energy0.00"]["official_score"]
    gain = scores[selected_key]["official_score"] - base_score
    print(
        f"v472 validation selected={selected_key} gain={gain:.6f}", flush=True
    )
    report: dict[str, object] = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official CIR magnitude and link energy are more spatially learnable than phase; "
            "a conditioned link network can approach the 0.800944 magnitude oracle while "
            "retaining the v469 complex phase."
        ),
        "validation": {
            "fit_count": int(len(fit_ids)),
            "validation_count": int(len(validation_ids)),
            "scores": scores,
            "selected_key": selected_key,
            "selected_shape_alpha": selected_shape,
            "selected_energy_alpha": selected_energy,
            "selected_gain": float(gain),
            "history": validation_history,
        },
    }
    if gain < 0.005 or (selected_shape <= 0.0 and selected_energy <= 0.0):
        report["decision"] = "reject_magnitude_network_before_test_deployment"
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
            "shape_alpha": selected_shape,
            "energy_alpha": selected_energy,
            "validation_history": validation_history,
            "full_history": full_history,
        },
        CHECKPOINT,
    )
    test_base = np.load(v471.TEST_BASE, mmap_mode="r")
    test_high = np.load(v471.TEST_HIGH, mmap_mode="r")
    test_low = np.load(v471.TEST_LOW, mmap_mode="r")
    test_ids = np.arange(500, dtype=np.int64)
    test_shape, test_energy = predict_targets(
        full_model,
        test_ids,
        test_base,
        test_high,
        test_low,
        point_features,
        2000,
        TEST_SHAPE,
        TEST_ENERGY,
        device,
    )
    render_test(test_shape, test_energy, selected_shape, selected_energy)
    replay = v208.score_prediction(OUTPUT)
    base_replay = v208.score_prediction(v471.TEST_BASE)
    report.update(
        {
            "status": "candidate_pending",
            "decision": "deploy_magnitude_network",
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
