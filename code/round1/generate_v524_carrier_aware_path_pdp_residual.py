#!/usr/bin/env python3
"""Inject only the path-network PDP residual not already present in v458/v491."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import analyze_v501_full_official_sionna_inverse_calibration as v501
import generate_v523_end_to_end_path_pdp_residual as v523
import train_v518_path_pol_vertical_correction as v518
import train_v519_end_to_end_path_spectral_renderer as v519
from project_paths import PATHS


VERSION_NUMBER = 524
VERSION = "v524_carrier_aware_path_pdp_residual"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = CACHE_DIR / "v524_Round1_Test_Channel.npy"
ACTION_CACHE = CACHE_DIR / "v524_path_pdp_actions.npz"
ARCHIVE = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

COEFFICIENTS_160 = PATHS.cache_dir(520) / "v520_single_config_coefficients_points160.npz"
V520_CHECKPOINTS = PATHS.checkpoints / "v520_single_config_path_renderer_alignment"
V523_BUNDLE = PATHS.cache_dir(523) / "v523_inference_bundle.pt"
SOURCE_DATA = PATHS.cache_dir(522) / "v522_source_path_coefficients_2500.npz"
BASE = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"
OLD_TEST_TARGET = PATHS.cache_dir(458) / "v458_calibrated_link_target.npy"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"

ALPHA_GRID = np.linspace(0.0, 1.5, 31, dtype=np.float32)
REGIME_NAMES = ("clear", "light", "medium", "heavy")
PATH_LIMIT = 32
TINY = 1e-20


def normalize_links(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(np.sum(value, axis=-1, keepdims=True), TINY)


def link_cosine(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    numerator = np.sum(truth * prediction, axis=-1, dtype=np.float64)
    left = np.sum(truth * truth, axis=-1, dtype=np.float64)
    right = np.sum(prediction * prediction, axis=-1, dtype=np.float64)
    return (numerator / np.maximum(np.sqrt(left * right), TINY)).astype(np.float32)


def interpolate_log_ratio(
    base: np.ndarray, target: np.ndarray, alpha: float | np.ndarray
) -> np.ndarray:
    floor = np.float32(1e-6 / base.shape[-1])
    ratio = np.clip(
        np.log(np.maximum(target, floor)) - np.log(np.maximum(base, floor)),
        -2.0,
        2.0,
    )
    return normalize_links(base * np.exp(np.asarray(alpha, dtype=np.float32) * ratio))


def regimes_all() -> np.ndarray:
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = np.concatenate(
        [geometry["train_blocked"], geometry["test_blocked"]]
    ).astype(np.float32)
    output = np.full(len(blocked), 2, dtype=np.int8)
    output[blocked <= 0] = 0
    output[(blocked >= 1) & (blocked <= 25)] = 1
    output[(blocked > 25) & (blocked <= 100)] = 2
    output[blocked > 100] = 3
    return output


def v520_features(
    data: dict[str, np.ndarray], point_ids: np.ndarray, regimes: np.ndarray
) -> np.ndarray:
    positions = np.load(TRAIN_POS).astype(np.float32)[point_ids]
    base, _, _, _ = v519.v516.path_features(data, positions, regimes)
    return v518.input_features(
        base, data["source_coefficients"].astype(np.complex128)
    )


@torch.no_grad()
def infer_v520_oof_pdp(
    data: dict[str, np.ndarray], features: np.ndarray, fold_codes: np.ndarray
) -> np.ndarray:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = np.zeros((len(features), 256, 4, 192), dtype=np.float16)
    for fold in sorted(np.unique(fold_codes).tolist()):
        checkpoint_path = V520_CHECKPOINTS / f"fold_{int(fold)}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = v519.PathGainTransformer(
            int(checkpoint["input_dim"]), float(checkpoint["max_log_gain"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        normalized = (
            (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        ).astype(np.float32)
        normalized *= data["valid"][..., None]
        ids = np.flatnonzero(fold_codes == fold)
        for start in range(0, len(ids), 8):
            local = ids[start : start + 8]
            prediction = model(
                torch.from_numpy(normalized[local]).to(device),
                torch.from_numpy(data["valid"][local]).to(device),
                torch.from_numpy(data["tau"][local].astype(np.float32)).to(device),
                torch.from_numpy(data["h_center"][local].astype(np.float32)).to(device),
                torch.from_numpy(data["source_coefficients"][local]).to(device),
            )
            channel = prediction.permute(0, 1, 3, 4, 5, 2)
            power = torch.abs(torch.fft.ifft(channel, dim=-1)) ** 2
            power /= torch.clamp(torch.sum(power, dim=-1, keepdim=True), min=TINY)
            output[local] = power.reshape(len(local), 256, 4, 192).cpu().numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def calibrated_old_oof(
    point_ids: np.ndarray, fold_codes: np.ndarray, all_fold_codes: np.ndarray, regimes: np.ndarray
) -> np.ndarray:
    truth = np.load(v447.TRUTH_CACHE, mmap_mode="r")
    source = np.load(v447.SOURCE_CACHE, mmap_mode="r")
    output = np.empty((len(point_ids), 1024, 192), dtype=np.float16)
    all_ids = np.arange(len(all_fold_codes), dtype=np.int64)
    for fold in sorted(np.unique(fold_codes).tolist()):
        query_rows = np.flatnonzero(fold_codes == fold)
        query_ids = point_ids[query_rows]
        fit_ids = all_ids[all_fold_codes != fold]
        for start in range(0, source.shape[1], 16):
            end = min(start + 16, source.shape[1])
            ratios = v447.fit_ratio_chunk(
                truth, source, fit_ids, np.asarray(REGIME_NAMES)[regimes[:2000]], start, end
            )
            values = np.asarray(source[query_ids, start:end], dtype=np.float32)
            local = np.empty_like(values)
            for code, name in enumerate(REGIME_NAMES):
                mask = regimes[query_ids] == code
                if np.any(mask):
                    local[mask] = normalize_links(
                        values[mask] * np.exp(ratios[name])[None]
                    )
            output[query_rows, start:end] = local.astype(np.float16)
    return output


def nested_alpha_selection(
    truth: np.ndarray,
    old: np.ndarray,
    neural: np.ndarray,
    fold_codes: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    baseline = np.mean(link_cosine(truth, old), axis=1)
    grid_scores = np.empty((len(ALPHA_GRID), len(truth)), dtype=np.float32)
    for index, alpha in enumerate(ALPHA_GRID):
        candidate = interpolate_log_ratio(old, neural, float(alpha))
        grid_scores[index] = np.mean(link_cosine(truth, candidate), axis=1)

    selected_alpha = np.zeros(len(truth), dtype=np.float32)
    selected_score = np.zeros(len(truth), dtype=np.float32)
    fold_rows = []
    for fold in sorted(np.unique(fold_codes).tolist()):
        fit = fold_codes != fold
        query = fold_codes == fold
        regime_alphas: dict[str, float] = {}
        for code, name in enumerate(REGIME_NAMES):
            fit_regime = fit & (regimes == code)
            if np.sum(fit_regime) < 8:
                fit_regime = fit
            means = np.mean(grid_scores[:, fit_regime], axis=1)
            best = int(np.argmax(means))
            regime_alphas[name] = float(ALPHA_GRID[best])
            local = query & (regimes == code)
            selected_alpha[local] = ALPHA_GRID[best]
            selected_score[local] = grid_scores[best, local]
        fold_rows.append(
            {
                "fold": int(fold),
                "count": int(np.sum(query)),
                "alphas": regime_alphas,
                "baseline": float(np.mean(baseline[query])),
                "selected": float(np.mean(selected_score[query])),
                "gain": float(np.mean(selected_score[query] - baseline[query])),
            }
        )

    deployment_alpha = []
    for code, name in enumerate(REGIME_NAMES):
        local = regimes == code
        means = np.mean(grid_scores[:, local], axis=1)
        deployment_alpha.append(float(ALPHA_GRID[int(np.argmax(means))]))
    gain = selected_score - baseline
    report = {
        "baseline_mean": float(np.mean(baseline)),
        "selected_mean": float(np.mean(selected_score)),
        "nested_gain": float(np.mean(gain)),
        "positive_folds": int(sum(row["gain"] > 0.0 for row in fold_rows)),
        "positive_points": float(np.mean(gain > 0.0)),
        "deployment_alpha_by_regime": dict(zip(REGIME_NAMES, deployment_alpha)),
        "folds": fold_rows,
        "by_regime": {
            name: {
                "count": int(np.sum(regimes == code)),
                "gain": float(np.mean(gain[regimes == code])),
                "positive_points": float(np.mean(gain[regimes == code] > 0.0)),
            }
            for code, name in enumerate(REGIME_NAMES)
        },
    }
    return np.asarray(deployment_alpha, dtype=np.float32), report


@torch.no_grad()
def v523_test_neural_pdp(
    source_data: dict[str, np.ndarray], regimes: np.ndarray
) -> np.ndarray:
    bundle = torch.load(V523_BUNDLE, map_location="cpu", weights_only=False)
    residual = np.asarray(bundle["test_residual_log_gain"], dtype=np.float32)
    args = type("Args", (), {"device": "cuda:0" if torch.cuda.is_available() else "cpu"})()
    _, corrected = v523.path_pdp_targets(args, source_data, residual)
    return np.asarray(corrected, dtype=np.float32)


def write_candidate(
    neural_target: np.ndarray, alpha_by_regime: np.ndarray, test_regimes: np.ndarray
) -> dict[str, object]:
    base = np.load(BASE, mmap_mode="r")
    old_target = np.load(OLD_TEST_TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    action = np.zeros(len(base), dtype=np.float32)
    power_ratio = np.zeros(len(base), dtype=np.float64)
    for start in range(0, len(base), 2):
        end = min(start + 2, len(base))
        channel = np.asarray(base[start:end], dtype=np.complex64)
        cir = np.fft.ifft(channel, axis=-1).astype(np.complex64)
        current_power = np.abs(cir).astype(np.float32) ** 2
        energy = np.sum(current_power, axis=-1, keepdims=True, dtype=np.float64)
        current = current_power / np.maximum(energy, TINY)
        old = np.asarray(old_target[start:end], dtype=np.float32).reshape(
            end - start, 256, 4, 192
        )
        neural = neural_target[start:end]
        alpha = alpha_by_regime[test_regimes[start:end]][:, None, None, None]
        floor = np.float32(1e-6 / 192.0)
        residual = np.clip(
            np.log(np.maximum(neural, floor)) - np.log(np.maximum(old, floor)),
            -2.0,
            2.0,
        )
        target = normalize_links(current * np.exp(alpha * residual))
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
        alpha_by_regime=alpha_by_regime,
        test_regimes=test_regimes,
        action_l1=action,
    )
    return {
        "alpha_by_regime": dict(zip(REGIME_NAMES, alpha_by_regime.astype(float).tolist())),
        "action_l1_mean": float(np.mean(action)),
        "action_l1_by_regime": {
            name: float(np.mean(action[test_regimes == code]))
            for code, name in enumerate(REGIME_NAMES)
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
        archive.write(V523_BUNDLE, "model/v523_inference_bundle.pt")
        archive.write(ACTION_CACHE, "model/v524_path_pdp_actions.npz")
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
        COEFFICIENTS_160,
        V523_BUNDLE,
        SOURCE_DATA,
        BASE,
        OLD_TEST_TARGET,
        v447.TRUTH_CACHE,
        v447.SOURCE_CACHE,
        TRAIN_POS,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    regimes = regimes_all()
    positions = np.load(TRAIN_POS).astype(np.float32)
    all_fold_codes = v501.spatial_fold_codes(positions, regimes[:2000])

    with np.load(COEFFICIENTS_160) as loaded:
        data160 = {name: np.asarray(loaded[name]) for name in loaded.files}
    point_ids = data160["indices"].astype(np.int64)
    fold_codes = all_fold_codes[point_ids]
    local_regimes = regimes[point_ids]
    features = v520_features(data160, point_ids, local_regimes)
    neural_oof = infer_v520_oof_pdp(data160, features, fold_codes).reshape(
        len(point_ids), 1024, 192
    ).astype(np.float32)
    old_oof = calibrated_old_oof(
        point_ids, fold_codes, all_fold_codes, regimes
    ).astype(np.float32)
    truth = np.asarray(np.load(v447.TRUTH_CACHE, mmap_mode="r")[point_ids], dtype=np.float32)
    alpha_by_regime, oof = nested_alpha_selection(
        truth, old_oof, neural_oof, fold_codes, local_regimes
    )
    passed = bool(oof["nested_gain"] >= 0.005 and oof["positive_folds"] >= 4)
    if not passed:
        report = {
            "method": VERSION,
            "status": "analysis_rejected_no_candidate",
            "primary_hypothesis": "The learned path renderer adds PDP information beyond v458's existing official-train calibration.",
            "oof": oof,
            "deployment_gate": {"pass": False, "minimum_gain": 0.005, "minimum_positive_folds": 4},
        }
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return

    with np.load(SOURCE_DATA) as loaded:
        source_data = {name: np.asarray(loaded[name]) for name in loaded.files}
    neural_test = v523_test_neural_pdp(source_data, regimes)
    action = write_candidate(neural_test, alpha_by_regime, regimes[2000:])
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
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": "The learned path renderer adds PDP information beyond v458's existing official-train calibration.",
        "base": {"version": 491, "online_score": 0.739634},
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "uses Sionna path coefficients generated at official test coordinates",
        },
        "oof": oof,
        "deployment_gate": {"pass": True, "minimum_gain": 0.005, "minimum_positive_folds": 4},
        "action": action,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_CACHE),
            "archive": PATHS.relative(ARCHIVE),
            "report": PATHS.relative(REPORT),
        },
        "integrity": integrity,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
