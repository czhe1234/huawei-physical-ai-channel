#!/usr/bin/env python3
"""Apply the v462 spatial Sionna joint-support residual on top of v458."""

from __future__ import annotations

import json
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v462_spatial_sionna_residual_calibration as v462
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
from generate_v159_frontier_consensus_support_soup import digest
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 463
VERSION = "v463_spatial_sionna_joint_residual_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = OUT_DIR / "v463_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v463_spatial_joint_residual_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(458) / "v458_Round1_Test_Channel.npy"
JOINT_ACTIONS = PATHS.cache_dir(435) / "v435_train_calibrated_joint_sh_actions.npz"
TRAIN_OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
TRAIN_HIGH = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
TRAIN_LOWS = tuple(
    PATHS.cache_dir(205) / name / "train_clean_H.npy"
    for name in (
        "benchmark_sector_m30_track_d5_f2p6",
        "benchmark_sector_m30_track_d5_f6p0",
        "benchmark_sector_m30_track_d5_f6p7",
    )
)

RESIDUAL_STRENGTH = np.float32(0.90)
ACTION_ALPHA = np.float32(0.45)
ACTION_CAP = np.float32(0.48)
JOINT_CYCLES = 5
BATCH = 2
FLOOR = 1e-10


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_support(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    output /= np.maximum(
        output.sum(axis=tuple(range(1, output.ndim)), keepdims=True), TINY
    )
    return output.astype(np.float32)


def build_test_residual() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked).astype("U8")
    test_regimes = regime_of(test_blocked).astype("U8")

    truth, _ = v434.load_sh(TRAIN_OFFICIAL)
    high, high_total = v434.load_sh(TRAIN_HIGH)
    lows = [v434.load_sh(path) for path in TRAIN_LOWS]
    train_raw, _, _ = v434.robust_source(
        high,
        high_total,
        [value[0] for value in lows],
        [value[1] for value in lows],
    )
    fit_ids = np.arange(len(train_pos), dtype=np.int64)
    ratios = v432.fit_ratio(truth, train_raw, fit_ids, train_regimes)
    train_global = v432.apply_ratio(train_raw, fit_ids, train_regimes, ratios)
    train_residual = np.clip(
        np.log(np.maximum(truth, FLOOR))
        - np.log(np.maximum(train_global, FLOOR)),
        -2.5,
        2.5,
    ).astype(np.float32)

    joint = np.load(JOINT_ACTIONS)
    test_global = joint["calibrated_target"].astype(np.float32)
    point_quality = joint["point_quality"].astype(np.float32)
    positions = np.vstack([train_pos, test_pos]).astype(np.float32)
    blocked = np.concatenate([train_blocked, test_blocked]).astype(np.float32)
    regimes = np.concatenate([train_regimes, test_regimes]).astype("U8")
    source = np.vstack([train_global, test_global]).astype(np.float32)
    query_ids = np.arange(len(train_pos), len(positions), dtype=np.int64)
    neighbor_ids, weights, diagnostics = v462.conditioned_neighbor_bank(
        fit_ids,
        query_ids,
        positions,
        blocked,
        regimes,
        source,
    )
    local_residual = v462.predict_local_residual(
        train_residual, neighbor_ids, weights
    )
    return local_residual, point_quality, diagnostics


def generate(local_residual: np.ndarray, point_quality: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    action = np.clip(
        ACTION_ALPHA * point_quality, 0.0, ACTION_CAP
    ).astype(np.float32)
    diagnostics = {
        "action": action,
        "local_residual_l1": np.mean(np.abs(local_residual), axis=1).astype(np.float32),
        "desired_action_l1": np.zeros(len(source), dtype=np.float32),
        "actual_action_l1": np.zeros(len(source), dtype=np.float32),
        "pdp_lock_error": np.zeros(len(source), dtype=np.float32),
        "power_ratio": np.zeros(len(source), dtype=np.float32),
    }
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        base_sh = v423.leaked_marginals(base_support)["sh"]
        log_action = np.clip(
            RESIDUAL_STRENGTH
            * action[start:end, None]
            * local_residual[start:end],
            -1.25,
            1.25,
        ).reshape(end - start, 192, 16)
        desired_sh = base_sh * np.exp(log_action).astype(np.float32)
        desired_sh /= np.maximum(
            desired_sh.sum(axis=(1, 2), keepdims=True), TINY
        )
        ratio = desired_sh / np.maximum(base_sh, FLOOR)
        desired_support = normalize_support(
            base_support * ratio[:, None, :, :, None, None]
        )
        candidate = base_h.copy()
        for _ in range(JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"non-finite v463 block {start}:{end}")
        output[start:end] = candidate
        output.flush()

        actual_support, _ = v423.leaked_support(candidate)
        actual_sh = v423.leaked_marginals(actual_support)["sh"]
        diagnostics["desired_action_l1"][start:end] = np.sum(
            np.abs(desired_sh - base_sh), axis=(1, 2)
        ).astype(np.float32)
        diagnostics["actual_action_l1"][start:end] = np.sum(
            np.abs(actual_sh - base_sh), axis=(1, 2)
        ).astype(np.float32)
        diagnostics["pdp_lock_error"][start:end] = v424.full_linkwise_pdp_error(
            candidate, base_h
        )
        base_power = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        candidate_power = np.sum(
            np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        diagnostics["power_ratio"][start:end] = (
            candidate_power / np.maximum(base_power, TINY)
        ).astype(np.float32)
        if end % 50 == 0 or end == len(source):
            print(f"v463 projection {end}/{len(source)}", flush=True)
    return diagnostics


def write_zip(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v463_spatial_sionna_joint_residual_diagnostic.py",
        "analyze_v462_spatial_sionna_residual_calibration.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "analyze_v434_sionna_joint_sh_calibration.py",
        "analyze_v432_sionna_link_energy_calibration.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "analyze_v208_strict_historical_replay.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(output[start : start + 10])).all()
        for start in range(0, len(output), 10)
    )
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def main() -> None:
    for path in (BASE, JOINT_ACTIONS, TRAIN_OFFICIAL, TRAIN_HIGH, *TRAIN_LOWS):
        require_file(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    local_residual, point_quality, neighbor_diagnostics = build_test_residual()
    diagnostics = generate(local_residual, point_quality)
    np.savez_compressed(
        ACTION_OUT,
        local_residual=local_residual,
        point_quality=point_quality,
        **diagnostics,
    )
    baseline = v208.score_prediction(BASE)
    candidate = v208.score_prediction(OUTPUT)
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "v462 showed 5/5-fold independent gain from a spatially varying "
            "Sionna-to-official joint S-H residual field. Applying only that residual "
            "direction to v458 should improve angular-delay support without changing "
            "its online-validated full linkwise PDP or power."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "uses Sionna references traced at the 500 official test coordinates",
        },
        "oof_evidence": {
            "replacement_target": {
                "old_v434_mean": 0.7300043702125549,
                "v462_mean": 0.7485272884368896,
                "gain": 0.01852291077375412,
                "fold_wins": 5,
            },
            "deployment_matched_overlay": {
                "operator": "old selected support times exp(0.45 * quality * local residual)",
                "old_v434_mean": 0.7300043702125549,
                "selected_mean": 0.7382369041442871,
                "gain": 0.008232553489506245,
                "fold_wins": 5,
            },
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "residual_strength": float(RESIDUAL_STRENGTH),
            "action_alpha": float(ACTION_ALPHA),
            "action_cap": float(ACTION_CAP),
            "joint_cycles": JOINT_CYCLES,
            "locked": "v458 complete linkwise PDP, link energy, point power, and carrier",
        },
        "neighbor_diagnostics": neighbor_diagnostics,
        "actions": {key: summarize(value) for key, value in diagnostics.items()},
        "replay": {
            "warning": "same-coordinate v207 diagnostic reference, not hidden truth",
            "v458": baseline,
            "candidate": candidate,
            "delta": float(
                candidate["score_traceback_leaked"]
                - baseline["score_traceback_leaked"]
            ),
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    write_zip(report)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
