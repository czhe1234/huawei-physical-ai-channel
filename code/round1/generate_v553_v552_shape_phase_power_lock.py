#!/usr/bin/env python3
"""Deploy v552 shape and phase residuals with exact carrier point-power locking."""

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
import train_v552_full_carrier_conditioned_fullh as v552
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 553
VERSION = "v553_v552_shape_phase_v469_power_lock"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUTPUT = CACHE_DIR / "v553_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"
BUNDLE = CHECKPOINT_DIR / "v553_inference_bundle.pt"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

TRAIN_H = v552.TRAIN_H
TRAIN_POS = v552.TRAIN_POS
TRAIN_BASE = v552.TRAIN_BASE
TEST_BASE = v552.TEST_BASE
GEOMETRY = v552.GEOMETRY
RAW_POINT_FEATURES = v552.RAW_POINT_FEATURES
ALPHA_GRID = v552.ALPHA_GRID
REGIMES = v552.REGIMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
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


def apply_shape_phase(
    base_cir: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    batch = len(base_cir)
    base_links = base_cir.reshape(batch, 1024, 192)
    amplitude = alpha * diagnostics["amplitude_delta"]
    phase = alpha * diagnostics["phase_delta"]
    links = base_links * torch.exp(0.5 * amplitude)
    links = links * torch.polar(torch.ones_like(phase), phase)
    return torch.fft.fft(links.reshape_as(base_cir), dim=-1).to(torch.complex64)


def lock_point_power(
    candidate: torch.Tensor,
    carrier: torch.Tensor,
) -> torch.Tensor:
    candidate_power = candidate.abs().square().sum(dim=(1, 2, 3))
    carrier_power = carrier.abs().square().sum(dim=(1, 2, 3))
    scale = torch.sqrt(carrier_power.clamp_min(0.0) / candidate_power.clamp_min(1e-30))
    scale = torch.where(
        carrier_power > 0,
        scale,
        torch.zeros_like(scale),
    )
    return (candidate * scale[:, None, None, None]).to(torch.complex64)


def empty_metrics(rows: int) -> dict[str, np.ndarray]:
    return v552.empty_metrics(rows)


def assign_scores(
    target: dict[str, np.ndarray],
    ids: np.ndarray,
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> None:
    v552.assign_scores(target, ids, v551.score_tensors(prediction, truth))


def summarize(
    metrics: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    return v552.summary(metrics, mask)


def score_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return v552.delta(candidate, baseline)


@torch.no_grad()
def evaluate_oof(
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    raw_features = np.load(RAW_POINT_FEATURES, mmap_mode="r")
    positions = np.load(TRAIN_POS).astype(np.float32)
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    geometry = np.load(GEOMETRY)
    regimes = regime_of(geometry["train_blocked"]).astype("U8")
    store = v552.TrainStore(raw_features, device)
    coordinates = v551.link_coordinates(device)

    baseline = empty_metrics(len(all_ids))
    candidates = {alpha: empty_metrics(len(all_ids)) for alpha in ALPHA_GRID}
    fold_rows: list[dict[str, object]] = []
    max_power_error = {alpha: 0.0 for alpha in ALPHA_GRID}

    for fold, outer_ids in enumerate(folds):
        saved = torch.load(
            v552.checkpoint_path(fold),
            map_location=device,
            weights_only=False,
        )
        model = v551.CarrierConditionedResidualNet(
            int(saved["point_dim"]),
            int(saved["width"]),
        ).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        mean = np.asarray(saved["mean"], dtype=np.float32)[None, :]
        std = np.asarray(saved["std"], dtype=np.float32)[None, :]

        for start in range(0, len(outer_ids), batch_size):
            ids = outer_ids[start : start + batch_size]
            base_h, base_cir, truth_h, source, point = store.load(
                ids,
                mean,
                std,
            )
            assign_scores(baseline, ids, base_h, truth_h)
            _, diagnostics = model(base_cir, source, point, coordinates)
            base_power = base_h.abs().square().sum(dim=(1, 2, 3))
            for alpha in ALPHA_GRID:
                prediction = apply_shape_phase(base_cir, diagnostics, alpha)
                prediction = lock_point_power(prediction, base_h)
                assign_scores(candidates[alpha], ids, prediction, truth_h)
                prediction_power = prediction.abs().square().sum(dim=(1, 2, 3))
                relative_error = (
                    prediction_power - base_power
                ).abs() / base_power.clamp_min(1e-30)
                max_power_error[alpha] = max(
                    max_power_error[alpha],
                    float(relative_error.max().cpu()),
                )
        fold_rows.append(
            {
                "fold": fold,
                "outer_count": int(len(outer_ids)),
                "checkpoint": PATHS.relative(v552.checkpoint_path(fold)),
            }
        )
        print(f"v553 OOF fold {fold} complete", flush=True)

    baseline_summary = summarize(baseline)
    alpha_results: dict[str, dict[str, object]] = {}
    for alpha in ALPHA_GRID:
        candidate_summary = summarize(candidates[alpha])
        alpha_results[str(alpha)] = {
            "candidate": candidate_summary,
            "delta": score_delta(candidate_summary, baseline_summary),
            "max_point_power_relative_error": max_power_error[alpha],
        }
    selected_alpha = max(
        ALPHA_GRID,
        key=lambda value: alpha_results[str(value)]["candidate"]["official_score"],
    )
    selected_metrics = candidates[selected_alpha]
    selected_summary = alpha_results[str(selected_alpha)]["candidate"]
    selected_delta = alpha_results[str(selected_alpha)]["delta"]

    fold_wins = 0
    for fold, outer_ids in enumerate(folds):
        mask = np.isin(all_ids, outer_ids)
        base_fold = summarize(baseline, mask)
        candidate_fold = summarize(selected_metrics, mask)
        delta_fold = score_delta(candidate_fold, base_fold)
        fold_rows[fold]["baseline"] = base_fold
        fold_rows[fold]["candidate"] = candidate_fold
        fold_rows[fold]["delta"] = delta_fold
        fold_wins += int(delta_fold["official_score"] > 0)

    by_regime = {}
    for regime in REGIMES:
        mask = regimes == regime
        base_regime = summarize(baseline, mask)
        candidate_regime = summarize(selected_metrics, mask)
        by_regime[regime] = {
            "baseline": base_regime,
            "candidate": candidate_regime,
            "delta": score_delta(candidate_regime, base_regime),
        }

    return {
        "baseline": baseline_summary,
        "alpha_results": alpha_results,
        "selected_alpha": selected_alpha,
        "selected_candidate": selected_summary,
        "selected_delta": selected_delta,
        "fold_wins": fold_wins,
        "folds": fold_rows,
        "by_regime": by_regime,
    }


@torch.no_grad()
def predict_test(
    device: torch.device,
    batch_size: int,
    alpha: float,
) -> dict[str, float]:
    raw_features = np.load(RAW_POINT_FEATURES, mmap_mode="r")
    test_store = v552.TestStore(raw_features, device)
    models, checkpoints = v552.load_fold_models(device)
    coordinates = v551.link_coordinates(device)
    base = np.load(TEST_BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.complex64,
        shape=base.shape,
    )

    amplitude_sum = 0.0
    phase_sum = 0.0
    action_rows = 0
    max_power_error = 0.0
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        ids = np.arange(start, end, dtype=np.int64)
        base_h, base_cir, source = test_store.load_base(ids)
        prediction_sum = None
        for model, saved in zip(models, checkpoints, strict=True):
            mean = np.asarray(saved["mean"], dtype=np.float32)[None, :]
            std = np.asarray(saved["std"], dtype=np.float32)[None, :]
            point = test_store.point(ids, mean, std)
            _, diagnostics = model(base_cir, source, point, coordinates)
            prediction = apply_shape_phase(base_cir, diagnostics, alpha)
            prediction_sum = (
                prediction if prediction_sum is None else prediction_sum + prediction
            )
            amplitude_sum += float(diagnostics["amplitude_delta"].abs().sum().cpu())
            phase_sum += float(diagnostics["phase_delta"].abs().sum().cpu())
            action_rows += len(ids)

        ensemble = prediction_sum / len(models)
        ensemble = lock_point_power(ensemble, base_h)
        base_power = base_h.abs().square().sum(dim=(1, 2, 3))
        output_power = ensemble.abs().square().sum(dim=(1, 2, 3))
        relative_error = (output_power - base_power).abs() / base_power.clamp_min(1e-30)
        max_power_error = max(max_power_error, float(relative_error.max().cpu()))
        output[start:end] = ensemble.cpu().numpy().astype(np.complex64)
        if end == len(base) or end % 100 == 0:
            output.flush()
            print(f"v553 test inference {end}/{len(base)}", flush=True)
    output.flush()
    return {
        "amplitude_abs_mean_unscaled": amplitude_sum / (action_rows * 1024 * 192),
        "phase_abs_mean_unscaled": phase_sum / (action_rows * 1024 * 192),
        "max_point_power_relative_error_before_numpy_roundtrip": max_power_error,
    }


def compare_outputs(
    candidate_path: Path,
    reference_path: Path,
    reference_name: str,
) -> dict[str, float | str | list[int] | bool]:
    candidate = np.load(candidate_path, mmap_mode="r")
    reference = np.load(reference_path, mmap_mode="r")
    error_sum = 0.0
    reference_sum = 0.0
    candidate_sum = 0.0
    numerator = 0.0
    ratios = []
    point_nmse = []
    finite = True
    max_abs = 0.0
    for start in range(0, len(candidate), 5):
        end = min(start + 5, len(candidate))
        left = np.asarray(candidate[start:end], dtype=np.complex64)
        right = np.asarray(reference[start:end], dtype=np.complex64)
        finite = finite and bool(np.isfinite(left).all())
        difference = left - right
        left_power = np.sum(np.abs(left) ** 2, axis=(1, 2, 3), dtype=np.float64)
        right_power = np.sum(np.abs(right) ** 2, axis=(1, 2, 3), dtype=np.float64)
        local_error = np.sum(
            np.abs(difference) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        error_sum += float(local_error.sum())
        reference_sum += float(right_power.sum())
        candidate_sum += float(left_power.sum())
        numerator += float(np.real(np.vdot(right.reshape(-1), left.reshape(-1))))
        ratios.extend((left_power / np.maximum(right_power, 1e-30)).tolist())
        point_nmse.extend((local_error / np.maximum(right_power, 1e-30)).tolist())
        max_abs = max(max_abs, float(np.max(np.abs(difference))))

    ratio = np.asarray(ratios, dtype=np.float64)
    local_nmse = np.asarray(point_nmse, dtype=np.float64)
    return {
        "reference": reference_name,
        "finite": finite,
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
        "nmse": error_sum / max(reference_sum, 1e-30),
        "complex_cosine": numerator
        / max(math.sqrt(reference_sum * candidate_sum), 1e-30),
        "max_abs_difference": max_abs,
        "power_ratio_mean": float(ratio.mean()),
        "power_ratio_p10": float(np.quantile(ratio, 0.10)),
        "power_ratio_median": float(np.median(ratio)),
        "power_ratio_p90": float(np.quantile(ratio, 0.90)),
        "power_ratio_max_abs_error": float(np.max(np.abs(ratio - 1.0))),
        "point_nmse_mean": float(local_nmse.mean()),
        "point_nmse_p90": float(np.quantile(local_nmse, 0.90)),
    }


def save_bundle(alpha: float) -> None:
    checkpoints = []
    for fold in range(5):
        checkpoints.append(
            torch.load(
                v552.checkpoint_path(fold),
                map_location="cpu",
                weights_only=False,
            )
        )
    torch.save(
        {
            "version": VERSION,
            "source_version": v552.VERSION,
            "alpha": alpha,
            "point_power_head_enabled": False,
            "final_point_power_lock": PATHS.relative(TEST_BASE),
            "source_names": [name for name, _ in v552.TRAIN_SOURCES],
            "folds": checkpoints,
        },
        BUNDLE,
    )


def package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v553_v552_shape_phase_power_lock.py",
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(BUNDLE, "model/v553_inference_bundle.pt")
        for fold in range(5):
            archive.write(
                v552.checkpoint_path(fold),
                f"model/v552_fold{fold}.pt",
            )
        archive.write(REPORT, f"model/{REPORT.name}")
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
        TRAIN_BASE,
        TEST_BASE,
        GEOMETRY,
        RAW_POINT_FEATURES,
        v552.TRAIN_SOURCE_FEATURES,
        v552.TRAIN_SOURCE_POINT_POWER,
        v552.TEST_SOURCE_FEATURES,
        v552.TEST_SOURCE_POINT_POWER,
        PATHS.cache_dir(549) / "v549_Round1_Test_Channel.npy",
        *(v552.checkpoint_path(fold) for fold in range(5)),
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
    oof = evaluate_oof(device, args.batch_size)
    selected_alpha = float(oof["selected_alpha"])
    test_actions = predict_test(device, args.batch_size, selected_alpha)
    save_bundle(selected_alpha)

    comparison_v469 = compare_outputs(OUTPUT, TEST_BASE, "v469")
    comparison_v549 = compare_outputs(
        OUTPUT,
        PATHS.cache_dir(549) / "v549_Round1_Test_Channel.npy",
        "v549",
    )
    previous_report = json.loads(v552.REPORT.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v552 failed online mainly because its point-power head over-downscaled "
            "the test domain; the frozen shape and phase residuals may transfer when "
            "point power is restored exactly to the semantically matched v469 carrier."
        ),
        "causal_change_from_v552": {
            "models_retrained": False,
            "point_power_head_enabled": False,
            "ensemble": "complex mean of five frozen v552 fold predictions",
            "final_point_power_lock": "exact v469 point power after ensemble",
            "uses_v552_online_score_to_define_hypothesis": True,
            "uses_online_score_for_alpha_selection": False,
        },
        "model": {
            "source_version": v552.VERSION,
            "fold_models": 5,
            "alpha_grid": list(ALPHA_GRID),
            "selected_alpha": selected_alpha,
            "precision": "float32 inference, complex64 output",
        },
        "oof": oof,
        "v552_reference": {
            "online_score": previous_report.get("online_score"),
            "oof_selected_candidate": previous_report["oof"]["selected_candidate"],
            "oof_selected_delta": previous_report["oof"]["selected_delta"],
        },
        "deployment": {
            "test_actions": test_actions,
            "comparison_to_v469": comparison_v469,
            "comparison_to_v549": comparison_v549,
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
                "oof_baseline": oof["baseline"],
                "oof_candidate": oof["selected_candidate"],
                "oof_delta": oof["selected_delta"],
                "fold_wins": oof["fold_wins"],
                "comparison_to_v469": comparison_v469,
                "comparison_to_v549": comparison_v549,
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
