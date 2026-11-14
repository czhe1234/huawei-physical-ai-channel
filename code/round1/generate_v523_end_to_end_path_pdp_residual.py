#!/usr/bin/env python3
"""Train the v520 renderer at full scale and inject centered path-PDP residuals into v491."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

import analyze_v501_full_official_sionna_inverse_calibration as v501
import train_v518_path_pol_vertical_correction as v518
import train_v519_end_to_end_path_spectral_renderer as v519
from project_paths import PATHS


VERSION_NUMBER = 523
VERSION = "v523_end_to_end_path_pdp_residual"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
SOURCE_DATA = PATHS.cache_dir(522) / "v522_source_path_coefficients_2500.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
BASE = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"
OUTPUT = CACHE_DIR / "v523_Round1_Test_Channel.npy"
INFERENCE_BUNDLE = CACHE_DIR / "v523_inference_bundle.pt"
ARCHIVE = PATHS.round1_versions / f"{VERSION}_diagnostic_not_recommended.zip"

NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
PATH_LIMIT = 32
TINY = 1e-20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--max-log-gain", type=float, default=1.5)
    parser.add_argument("--residual-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=523_001)
    return parser.parse_args()


def regime_codes() -> np.ndarray:
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = np.concatenate(
        [geometry["train_blocked"], geometry["test_blocked"]]
    ).astype(np.float32)
    result = np.full(len(blocked), 2, dtype=np.int8)
    result[blocked <= 0] = 0
    result[(blocked >= 1) & (blocked <= 25)] = 1
    result[(blocked > 25) & (blocked <= 100)] = 2
    result[blocked > 100] = 3
    return result


def source_base_features(data: dict[str, np.ndarray], regimes: np.ndarray) -> np.ndarray:
    valid = data["valid"].astype(bool)
    tau = data["tau"].astype(np.float64)
    h_center = data["h_center"].astype(np.float64)
    depth = data["depth"].astype(np.float64)
    source = data["source_coefficients"].astype(np.complex128)
    positions = data["positions"].astype(np.float64)
    source_energy = np.sum(np.abs(source) ** 2, axis=2, dtype=np.float64)
    source_total = np.maximum(np.sum(source_energy, axis=1, keepdims=True), TINY)
    source_fraction = source_energy / source_total
    coefficient_power = np.abs(source.reshape(len(source), PATH_LIMIT, 2, 8, 4)) ** 2
    coefficient_power /= np.maximum(
        np.sum(coefficient_power, axis=(2, 3, 4), keepdims=True), TINY
    )
    pol = np.sum(coefficient_power, axis=(3, 4))
    vertical = np.sum(coefficient_power, axis=(2, 4))
    ue = np.sum(coefficient_power, axis=(2, 3))
    coherent_sum = np.abs(np.sum(source, axis=2)) / np.maximum(
        np.sqrt(64.0 * source_energy), TINY
    )
    strongest = np.argmax(np.abs(source), axis=2)
    reference = np.take_along_axis(source, strongest[..., None], axis=2)[..., 0]
    relative = positions - v501.BS_POSITION[None, :]
    distance = np.linalg.norm(relative, axis=1)
    bearing = np.arctan2(relative[:, 1], relative[:, 0])
    point = np.concatenate(
        [
            relative / np.asarray([200.0, 200.0, 50.0]),
            (distance / 200.0)[:, None],
            np.sin(bearing)[:, None],
            np.cos(bearing)[:, None],
            np.eye(4, dtype=np.float64)[regimes],
        ],
        axis=1,
    )
    point = np.repeat(point[:, None, :], PATH_LIMIT, axis=1)
    delay_center = np.mod(
        NUM_SUBCARRIERS * SUBCARRIER_SPACING * tau,
        NUM_SUBCARRIERS,
    )
    delay_angle = 2.0 * np.pi * delay_center / NUM_SUBCARRIERS
    h_angle = 2.0 * np.pi * h_center / 16.0
    rank = np.repeat(
        (np.arange(PATH_LIMIT, dtype=np.float64) / PATH_LIMIT)[None, :, None],
        len(source),
        axis=0,
    )
    features = np.concatenate(
        [
            point,
            np.sin(delay_angle)[..., None],
            np.cos(delay_angle)[..., None],
            (delay_center / NUM_SUBCARRIERS)[..., None],
            np.sin(h_angle)[..., None],
            np.cos(h_angle)[..., None],
            (depth / 16.0)[..., None],
            np.log(np.maximum(source_energy, TINY))[..., None],
            np.log(np.maximum(source_fraction, TINY))[..., None],
            source_fraction[..., None],
            coherent_sum[..., None],
            np.cos(np.angle(reference))[..., None],
            np.sin(np.angle(reference))[..., None],
            rank,
            pol,
            vertical,
            ue,
        ],
        axis=2,
    ).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    features *= valid[..., None]
    return features


def target_units(channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h6 = channel.reshape(-1, 2, 16, 8, 4, 192)
    delay_power = torch.abs(torch.fft.ifft(h6, dim=-1)) ** 2
    delay_norm = torch.sqrt(
        torch.sum(delay_power * delay_power, dim=-1, keepdim=True) + 1e-20
    )
    pdp = delay_power / delay_norm
    leaked = h6.permute(0, 1, 5, 2, 4, 3)
    leaked_power = torch.abs(torch.fft.fft2(leaked, dim=(2, 3))) ** 2
    leaked_norm = torch.sqrt(
        torch.sum(leaked_power * leaked_power, dim=(2, 3), keepdim=True) + 1e-20
    )
    return pdp, leaked_power / leaked_norm


def render(
    tau: torch.Tensor,
    h_center: torch.Tensor,
    source: torch.Tensor,
    log_gain: torch.Tensor,
) -> torch.Tensor:
    corrected = source.reshape(-1, PATH_LIMIT, 2, 8, 4)
    corrected = corrected * torch.exp(log_gain.reshape(-1, PATH_LIMIT, 2, 8, 1))
    frequency = (
        torch.arange(NUM_SUBCARRIERS, device=tau.device, dtype=torch.float32)
        - NUM_SUBCARRIERS // 2
    ) * SUBCARRIER_SPACING
    delay = torch.exp(-2j * math.pi * frequency[None, :, None] * tau[:, None, :])
    element = torch.arange(16, device=tau.device, dtype=torch.float32)
    horizontal = torch.exp(
        2j * math.pi * element[None, :, None] * h_center[:, None, :] / 16.0
    )
    basis = delay[:, :, None, :] * horizontal[:, None, :, :]
    return torch.einsum("bfhk,bkpvu->bpfhvu", basis, corrected)


def load_batch(
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    features: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(features[ids]).to(device),
        torch.from_numpy(data["valid"][ids]).to(device),
        torch.from_numpy(data["tau"][ids].astype(np.float32)).to(device),
        torch.from_numpy(data["h_center"][ids].astype(np.float32)).to(device),
        torch.from_numpy(data["source_coefficients"][ids]).to(device),
    )


def train_models(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    features: np.ndarray,
) -> tuple[list[dict[str, torch.Tensor]], np.ndarray, np.ndarray, list[list[float]]]:
    device = torch.device(args.device)
    valid = data["valid"][:2000]
    selected = features[:2000][valid]
    mean = selected.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(selected.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
    normalized = ((features - mean) / std).astype(np.float32)
    normalized *= data["valid"][..., None]
    truth = np.load(TRAIN_H, mmap_mode="r")
    states = []
    histories = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for model_id in range(args.ensemble_size):
        seed = args.seed + model_id
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = v519.PathGainTransformer(features.shape[2], args.max_log_gain).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(2000)
            running = 0.0
            for start in range(0, len(order), args.batch_size):
                ids = order[start : start + args.batch_size]
                tensors = load_batch(ids, data, normalized, device)
                target_channel = torch.from_numpy(
                    np.asarray(truth[ids], dtype=np.complex64)
                ).to(device)
                target_pdp, target_pas = target_units(target_channel)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(*tensors)
                loss = v519.spectral_loss(prediction, target_pdp, target_pas)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running += float(loss.detach().cpu()) * len(ids)
            epoch_loss = running / 2000
            history.append(epoch_loss)
            print(
                f"v523 model={model_id} epoch={epoch}/{args.epochs} "
                f"train_spectral={1.0 - epoch_loss:.6f}",
                flush=True,
            )
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        states.append(state)
        histories.append(history)
        torch.save(
            {
                "state_dict": state,
                "feature_mean": mean,
                "feature_std": std,
                "input_dim": features.shape[2],
                "max_log_gain": args.max_log_gain,
                "seed": seed,
            },
            CHECKPOINT_DIR / f"model_{model_id}.pt",
        )
        del model
        torch.cuda.empty_cache()
    return states, mean, std, histories


@torch.no_grad()
def predict_residual_gains(
    args: argparse.Namespace,
    states: list[dict[str, torch.Tensor]],
    mean: np.ndarray,
    std: np.ndarray,
    data: dict[str, np.ndarray],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    normalized = ((features - mean) / std).astype(np.float32)
    normalized *= data["valid"][..., None]
    residuals = []
    centers = []
    for state in states:
        model = v519.PathGainTransformer(features.shape[2], args.max_log_gain).to(device)
        model.load_state_dict(state)
        model.eval()
        all_gain = np.zeros((2500, PATH_LIMIT, 16), dtype=np.float32)
        for start in range(0, 2500, 64):
            end = min(start + 64, 2500)
            gain = model.gains(
                torch.from_numpy(normalized[start:end]).to(device),
                torch.from_numpy(data["valid"][start:end]).to(device),
            )
            all_gain[start:end] = gain.cpu().numpy()
        train_valid = data["valid"][:2000]
        denominator = np.maximum(np.sum(train_valid), 1)
        center = np.sum(
            all_gain[:2000] * train_valid[..., None], axis=(0, 1)
        ) / denominator
        centers.append(center.astype(np.float32))
        residuals.append(all_gain[2000:] - center[None, None, :])
        del model
        torch.cuda.empty_cache()
    residual = np.mean(residuals, axis=0)
    residual = np.clip(residual, -args.residual_clip, args.residual_clip)
    return residual.astype(np.float32), np.asarray(centers, dtype=np.float32)


@torch.no_grad()
def path_pdp_targets(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    residual_gain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    base_output = np.empty((500, 256, 4, 192), dtype=np.float16)
    corrected_output = np.empty_like(base_output)
    for start in range(0, 500, 8):
        end = min(start + 8, 500)
        ids = np.arange(2000 + start, 2000 + end)
        tau = torch.from_numpy(data["tau"][ids].astype(np.float32)).to(device)
        h_center = torch.from_numpy(data["h_center"][ids].astype(np.float32)).to(device)
        source = torch.from_numpy(data["source_coefficients"][ids]).to(device)
        zero = torch.zeros((len(ids), PATH_LIMIT, 16), device=device)
        residual = torch.from_numpy(residual_gain[start:end]).to(device)
        for destination, gain in ((base_output, zero), (corrected_output, residual)):
            prediction = render(tau, h_center, source, gain)
            h6 = prediction.permute(0, 1, 3, 4, 5, 2)
            power = torch.abs(torch.fft.ifft(h6, dim=-1)) ** 2
            power = power / torch.clamp(
                torch.sum(power, dim=-1, keepdim=True), min=1e-20
            )
            destination[start:end] = (
                power.reshape(len(ids), 256, 4, 192).cpu().numpy().astype(np.float16)
            )
        print(f"v523 path targets {end}/500", flush=True)
    return base_output, corrected_output


def generate_candidate(
    base_path_pdp: np.ndarray,
    corrected_path_pdp: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    base = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.complex64,
        shape=base.shape,
    )
    alpha_by_regime = np.asarray([0.15, 0.25, 0.45, 0.65], dtype=np.float32)
    action_l1 = np.zeros(500, dtype=np.float32)
    power_ratio = np.zeros(500, dtype=np.float64)
    for start in range(0, 500, 2):
        end = min(start + 2, 500)
        channel = np.asarray(base[start:end], dtype=np.complex64)
        cir = np.fft.ifft(channel, axis=-1).astype(np.complex64)
        cir_power = np.abs(cir).astype(np.float32) ** 2
        link_energy = np.sum(cir_power, axis=-1, keepdims=True, dtype=np.float64)
        current = cir_power / np.maximum(link_energy, TINY)
        source = np.asarray(base_path_pdp[start:end], dtype=np.float32)
        corrected = np.asarray(corrected_path_pdp[start:end], dtype=np.float32)
        floor = 1e-6 / 192.0
        log_ratio = np.clip(
            np.log(np.maximum(corrected, floor))
            - np.log(np.maximum(source, floor)),
            -2.0,
            2.0,
        )
        alpha = alpha_by_regime[regimes[start:end]][:, None, None, None]
        target = current * np.exp(alpha * log_ratio)
        target /= np.maximum(np.sum(target, axis=-1, keepdims=True), TINY)
        action_l1[start:end] = np.mean(
            np.sum(np.abs(target - current), axis=-1), axis=(1, 2)
        )
        phase = cir / np.maximum(np.abs(cir), 1e-20)
        new_cir = phase * np.sqrt(target * link_energy).astype(np.float32)
        prediction = np.fft.fft(new_cir, axis=-1).astype(np.complex64)
        output[start:end] = prediction
        old_power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
        new_power = np.sum(np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
        power_ratio[start:end] = new_power / np.maximum(old_power, TINY)
    output.flush()
    return output, {
        "alpha_by_regime": alpha_by_regime.tolist(),
        "action_l1_mean": float(np.mean(action_l1)),
        "action_l1_by_regime": {
            name: float(np.mean(action_l1[regimes == code]))
            for code, name in enumerate(("clear", "light", "medium", "heavy"))
        },
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
    }


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def package() -> dict[str, object]:
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(Path(__file__), Path(__file__).name)
        archive.write(PATHS.code_root / "project_paths.py", "project_paths.py")
        archive.write(INFERENCE_BUNDLE, "model/v523_inference_bundle.pt")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"bad ZIP member: {bad}")
    return {
        "zip_crc": True,
        "zip_members": members,
        "zip_size": int(ARCHIVE.stat().st_size),
        "sha256": digest(ARCHIVE, "sha256"),
        "md5": digest(ARCHIVE, "md5"),
    }


def main() -> None:
    args = parse_args()
    for path in (SOURCE_DATA, TRAIN_H, BASE):
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_DATA) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    regimes = regime_codes()
    base_features = source_base_features(data, regimes)
    features = v518.input_features(
        base_features, data["source_coefficients"].astype(np.complex128)
    )
    states, mean, std, histories = train_models(args, data, features)
    residual_gain, centers = predict_residual_gains(
        args, states, mean, std, data, features
    )
    torch.save(
        {
            "states": states,
            "feature_mean": mean,
            "feature_std": std,
            "global_log_gain_centers": centers,
            "test_residual_log_gain": residual_gain.astype(np.float16),
            "parameters": vars(args),
        },
        INFERENCE_BUNDLE,
    )
    base_path_pdp, corrected_path_pdp = path_pdp_targets(
        args, data, residual_gain
    )
    output, action = generate_candidate(
        base_path_pdp, corrected_path_pdp, regimes[2000:]
    )
    integrity = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.all(np.isfinite(output))),
    }
    if integrity["shape"] != [500, 256, 4, 192] or integrity["dtype"] != "complex64":
        raise ValueError(integrity)
    integrity.update(package())
    report = {
        "method": VERSION,
        "status": "diagnostic_not_recommended_noncompliant",
        "primary_hypothesis": (
            "The v520 five-fold path-dependent PDP gain transfers to all test points when "
            "only the geometry-dependent log-gain residual is injected into v491."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "uses Sionna path coefficients generated at official test coordinates",
        },
        "base": {"version": 491, "online_score": 0.739634},
        "training": {
            "train_points": 2000,
            "ensemble_size": args.ensemble_size,
            "epochs": args.epochs,
            "histories": histories,
            "last_train_spectral": [float(1.0 - history[-1]) for history in histories],
        },
        "residual_gain": {
            "mean": float(np.mean(residual_gain)),
            "std": float(np.std(residual_gain)),
            "min": float(np.min(residual_gain)),
            "max": float(np.max(residual_gain)),
            "global_centers": centers.tolist(),
        },
        "action": action,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "inference_bundle": PATHS.relative(INFERENCE_BUNDLE),
            "archive": PATHS.relative(ARCHIVE),
            "report": PATHS.relative(REPORT),
        },
        "integrity": integrity,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
