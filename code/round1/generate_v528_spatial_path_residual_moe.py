#!/usr/bin/env python3
"""Deploy validated clear and spatial light/medium path-residual experts on v491."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

import analyze_v501_full_official_sionna_inverse_calibration as v501
import generate_v523_end_to_end_path_pdp_residual as v523
import generate_v524_carrier_aware_path_pdp_residual as v524
import generate_v526_full_carrier_residual_renderer as v526
import generate_v527_focused_carrier_residual_renderer as v527
import train_v519_end_to_end_path_spectral_renderer as v519
import train_v525_carrier_residual_renderer_pilot as v525
from project_paths import PATHS


VERSION_NUMBER = 528
VERSION = "v528_spatial_path_residual_moe"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE_DIR / "v528_Round1_Test_Channel.npy"
ACTION_CACHE = CACHE_DIR / "v528_path_pdp_actions.npz"
INFERENCE_BUNDLE = CACHE_DIR / "v528_inference_bundle.pt"
ARCHIVE = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"
OLD_TEST_TARGET = PATHS.cache_dir(458) / "v458_calibrated_link_target.npy"
SOURCE_DATA = PATHS.cache_dir(522) / "v522_source_path_coefficients_2500.npz"
V523_BUNDLE = PATHS.cache_dir(523) / "v523_inference_bundle.pt"
V527_CHECKPOINTS = PATHS.checkpoints / "v527_focused_carrier_residual_renderer"
V524_REPORT = PATHS.version_report_dir(524) / "v524_carrier_aware_path_pdp_residual.json"
V527_REPORT = PATHS.version_report_dir(527) / "v527_focused_carrier_residual_renderer.json"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TEST_POS = PATHS.round1 / "Round1_Test_Pos.npy"

CLEAR_ALPHA = 0.5
POSITIVE_FOLDS = (1, 3, 4)
TINY = 1e-20


def test_fold_codes(train_positions: np.ndarray, test_positions: np.ndarray, train_folds: np.ndarray) -> np.ndarray:
    train_xy = train_positions[:, :2]
    test_xy = test_positions[:, :2]
    origin = np.min(train_xy, axis=0)
    train_tiles = np.floor((train_xy - origin[None, :]) / 55.0).astype(np.int32)
    test_tiles = np.floor((test_xy - origin[None, :]) / 55.0).astype(np.int32)
    mapping: dict[tuple[int, int], int] = {}
    for tile, fold in zip(train_tiles, train_folds):
        key = (int(tile[0]), int(tile[1]))
        previous = mapping.setdefault(key, int(fold))
        if previous != int(fold):
            raise ValueError(f"train tile spans folds: {key}")
    output = np.full(len(test_positions), -1, dtype=np.int8)
    missing = []
    for row, tile in enumerate(test_tiles):
        key = (int(tile[0]), int(tile[1]))
        if key in mapping:
            output[row] = mapping[key]
        else:
            missing.append(row)
    if missing:
        distance = np.sum(
            (test_xy[missing, None, :] - train_xy[None, :, :]) ** 2, axis=2
        )
        output[np.asarray(missing)] = train_folds[np.argmin(distance, axis=1)]
    return output


@torch.no_grad()
def spatial_expert_log_ratio(
    data: dict[str, np.ndarray],
    raw_features: np.ndarray,
    test_folds: np.ndarray,
    test_regimes: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = np.zeros((500, 256, 4, 192), dtype=np.float32)
    rows = []
    for fold in POSITIVE_FOLDS:
        checkpoint_path = V527_CHECKPOINTS / f"fold_{fold}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = v519.PathGainTransformer(
            int(checkpoint["input_dim"]), float(checkpoint["max_log_gain"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        features = (
            (raw_features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        ).astype(np.float32)
        features *= data["valid"][..., None]
        local_test = np.flatnonzero(
            (test_folds == fold) & np.isin(test_regimes, (1, 2))
        )
        for start in range(0, len(local_test), 8):
            rows_local = local_test[start : start + 8]
            ids = rows_local + 2000
            gains = model.gains(
                torch.from_numpy(features[ids]).to(device),
                torch.from_numpy(data["valid"][ids]).to(device),
            )
            zero = torch.zeros_like(gains)
            tau = torch.from_numpy(data["tau"][ids].astype(np.float32)).to(device)
            h_center = torch.from_numpy(data["h_center"][ids].astype(np.float32)).to(device)
            source_coefficients = torch.from_numpy(data["source_coefficients"][ids]).to(device)
            source = v525.normalized_link_pdp(
                v525.render_from_gains(tau, h_center, source_coefficients, zero)
            )
            corrected = v525.normalized_link_pdp(
                v525.render_from_gains(tau, h_center, source_coefficients, gains)
            )
            floor = 1e-6 / 192.0
            ratio = torch.clamp(
                torch.log(torch.clamp(corrected, min=floor))
                - torch.log(torch.clamp(source, min=floor)),
                -2.0,
                2.0,
            )
            output[rows_local] = ratio.cpu().numpy()
        rows.append(
            {
                "fold": fold,
                "test_points": int(len(local_test)),
                "best_epoch": int(checkpoint["best_epoch"]),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output, rows


def write_candidate(
    clear_neural_target: np.ndarray,
    spatial_ratio: np.ndarray,
    test_regimes: np.ndarray,
    test_folds: np.ndarray,
) -> dict[str, object]:
    base = np.load(BASE, mmap_mode="r")
    old_target = np.load(OLD_TEST_TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    action = np.zeros(500, dtype=np.float32)
    branch = np.zeros(500, dtype=np.int8)
    power_ratio = np.zeros(500, dtype=np.float64)
    for start in range(0, 500, 2):
        end = min(start + 2, 500)
        channel = np.asarray(base[start:end], dtype=np.complex64)
        cir = np.fft.ifft(channel, axis=-1).astype(np.complex64)
        current_power = np.abs(cir).astype(np.float32) ** 2
        energy = np.sum(current_power, axis=-1, keepdims=True, dtype=np.float64)
        current = current_power / np.maximum(energy, TINY)
        ratio = np.asarray(spatial_ratio[start:end], dtype=np.float32).copy()
        clear = test_regimes[start:end] == 0
        if np.any(clear):
            old = np.asarray(old_target[start:end], dtype=np.float32).reshape(
                end - start, 256, 4, 192
            )
            neural = clear_neural_target[start:end]
            floor = np.float32(1e-6 / 192.0)
            clear_ratio = np.clip(
                np.log(np.maximum(neural, floor))
                - np.log(np.maximum(old, floor)),
                -2.0,
                2.0,
            )
            ratio[clear] = CLEAR_ALPHA * clear_ratio[clear]
            branch[start:end][clear] = 1
        spatial = (
            np.isin(test_regimes[start:end], (1, 2))
            & np.isin(test_folds[start:end], POSITIVE_FOLDS)
        )
        branch[start:end][spatial] = 2
        target = v524.normalize_links(current * np.exp(ratio))
        action[start:end] = np.mean(
            np.sum(np.abs(target - current), axis=-1), axis=(1, 2)
        )
        phase = cir / np.maximum(np.abs(cir), TINY)
        new_cir = phase * np.sqrt(target * energy).astype(np.float32)
        prediction = np.fft.fft(new_cir, axis=-1).astype(np.complex64)
        output[start:end] = prediction
        old_power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
        new_power = np.sum(np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
        power_ratio[start:end] = new_power / np.maximum(old_power, TINY)
    output.flush()
    np.savez_compressed(
        ACTION_CACHE,
        test_regimes=test_regimes,
        test_folds=test_folds,
        branch=branch,
        action_l1=action,
    )
    return {
        "clear_points": int(np.sum(branch == 1)),
        "spatial_light_medium_points": int(np.sum(branch == 2)),
        "untouched_points": int(np.sum(branch == 0)),
        "action_l1_mean": float(np.mean(action)),
        "action_l1_by_branch": {
            name: float(np.mean(action[branch == code])) if np.any(branch == code) else 0.0
            for code, name in enumerate(("untouched", "clear", "spatial_light_medium"))
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
    dependencies = (
        "generate_v528_spatial_path_residual_moe.py",
        "generate_v527_focused_carrier_residual_renderer.py",
        "generate_v526_full_carrier_residual_renderer.py",
        "train_v525_carrier_residual_renderer_pilot.py",
        "generate_v524_carrier_aware_path_pdp_residual.py",
        "generate_v523_end_to_end_path_pdp_residual.py",
        "train_v519_end_to_end_path_spectral_renderer.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
        archive.writestr("model/", b"")
        archive.write(INFERENCE_BUNDLE, "model/v528_inference_bundle.pt")
        archive.write(V523_BUNDLE, "model/v523_inference_bundle.pt")
        archive.write(SOURCE_DATA, "model/v522_source_path_coefficients_2500.npz")
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
    required = (
        BASE,
        OLD_TEST_TARGET,
        SOURCE_DATA,
        V523_BUNDLE,
        V524_REPORT,
        V527_REPORT,
        TRAIN_POS,
        TEST_POS,
        *[V527_CHECKPOINTS / f"fold_{fold}.pt" for fold in POSITIVE_FOLDS],
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_DATA) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    regimes = v524.regimes_all()
    train_positions = np.load(TRAIN_POS).astype(np.float32)
    test_positions = np.load(TEST_POS).astype(np.float32)
    train_folds = v501.spatial_fold_codes(train_positions, regimes[:2000])
    test_folds = test_fold_codes(train_positions, test_positions, train_folds)
    raw_features = v526.features_all(data, regimes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spatial_ratio, experts = spatial_expert_log_ratio(
        data, raw_features, test_folds, regimes[2000:], device
    )
    clear_neural_target = v524.v523_test_neural_pdp(data, regimes)
    action = write_candidate(
        clear_neural_target, spatial_ratio, regimes[2000:], test_folds
    )
    checkpoints = {
        fold: torch.load(
            V527_CHECKPOINTS / f"fold_{fold}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for fold in POSITIVE_FOLDS
    }
    torch.save(
        {
            "spatial_experts": checkpoints,
            "positive_folds": POSITIVE_FOLDS,
            "clear_alpha": CLEAR_ALPHA,
            "test_fold_codes": test_folds,
        },
        INFERENCE_BUNDLE,
    )
    value = np.load(OUTPUT, mmap_mode="r")
    integrity = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(
            all(
                np.isfinite(np.asarray(value[start : start + 8])).all()
                for start in range(0, len(value), 8)
            )
        ),
    }
    integrity.update(package())
    report524 = json.loads(V524_REPORT.read_text(encoding="utf-8"))
    report527 = json.loads(V527_REPORT.read_text(encoding="utf-8"))
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": "Disjoint clear and spatial light/medium path residual experts retain only the regimes and map folds with OOF evidence beyond v458/v491.",
        "base": {"version": 491, "online_score": 0.739634},
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "uses Sionna path coefficients generated at official test coordinates",
        },
        "evidence": {
            "clear_fixed_alpha": CLEAR_ALPHA,
            "v524_clear_nested_gain": report524["oof"]["by_regime"]["clear"]["gain"],
            "v527_global_oof_gain": report527["oof"]["gain"],
            "v527_margin_vs_shuffled": report527["oof"]["margin_vs_shuffled"],
            "positive_folds": list(POSITIVE_FOLDS),
            "experts": experts,
        },
        "action": action,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_CACHE),
            "inference_bundle": PATHS.relative(INFERENCE_BUNDLE),
            "archive": PATHS.relative(ARCHIVE),
            "report": PATHS.relative(REPORT),
        },
        "integrity": integrity,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
