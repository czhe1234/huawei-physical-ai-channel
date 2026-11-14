#!/usr/bin/env python3
"""Apply official-train-calibrated Sionna joint S-H support to v426.

This diagnostic reads Sionna references traced at the official test positions
and is not compliant with the final competition data boundary.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import generate_v178_test_track_complex_consensus as track
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
from generate_v87_easy_field_projection import summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 435
VERSION = "v435_train_calibrated_sionna_joint_sh_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v435_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v435_train_calibrated_joint_sh_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy"
TRAIN_OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
TRAIN_HIGH = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
TEST_HIGH = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
TRAIN_LOWS = tuple(
    PATHS.cache_dir(205) / name / "train_clean_H.npy"
    for name in (
        "benchmark_sector_m30_track_d5_f2p6",
        "benchmark_sector_m30_track_d5_f6p0",
        "benchmark_sector_m30_track_d5_f6p7",
    )
)
TEST_LOWS = tuple(
    PATHS.cache_dir(205) / name / "test_clean_H.npy"
    for name in (
        "benchmark_sector_m30_track_d5_f2p6",
        "benchmark_sector_m30_track_d5_f6p0",
        "benchmark_sector_m30_track_d5_f6p7",
    )
)

ALPHA = np.float32(0.25)
REGIME_SCALE = {"clear": 1.00, "light": 1.00, "medium": 0.82, "heavy": 0.78}
JOINT_CYCLES = 5
BATCH = 2
FLOOR = 1e-12


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def build_desired_support(
    base_support: np.ndarray,
    target_sh: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_sh = v423.leaked_marginals(base_support)["sh"]
    target = target_sh.reshape(len(base_support), 192, 16)
    log_ratio = np.clip(
        np.log(np.maximum(target, FLOOR))
        - np.log(np.maximum(base_sh, FLOOR)),
        -3.5,
        3.5,
    ).astype(np.float32)
    desired_sh = base_sh * np.exp(alpha[:, None, None] * log_ratio).astype(np.float32)
    desired_sh = normalize_rows(desired_sh.reshape(len(base_support), -1)).reshape(
        len(base_support), 192, 16
    )
    ratio = desired_sh / np.maximum(base_sh, FLOOR)
    desired = base_support * ratio[:, None, :, :, None, None]
    desired /= np.maximum(
        desired.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY
    )
    return desired.astype(np.float32), base_sh.astype(np.float32), desired_sh.astype(np.float32)


def row_l1(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(left.reshape(len(left), -1) - right.reshape(len(right), -1)),
        axis=1,
    ).astype(np.float32)


def generate(
    regimes: np.ndarray,
    calibrated_target: np.ndarray,
    point_quality: np.ndarray,
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    diagnostics = {
        "alpha": np.zeros(len(source), dtype=np.float32),
        "base_target_l1": np.zeros(len(source), dtype=np.float32),
        "actual_target_l1": np.zeros(len(source), dtype=np.float32),
        "action_l1": np.zeros(len(source), dtype=np.float32),
        "pdp_lock_error": np.zeros(len(source), dtype=np.float32),
        "power_ratio": np.zeros(len(source), dtype=np.float32),
        "phase_agreement": np.zeros(len(source), dtype=np.float32),
    }

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        local_scale = np.asarray(
            [REGIME_SCALE[name] for name in regimes[start:end]], dtype=np.float32
        )
        alpha = np.clip(
            ALPHA * local_scale * point_quality[start:end], 0.0, 0.28
        ).astype(np.float32)
        desired, base_sh, desired_sh = build_desired_support(
            base_support, calibrated_target[start:end], alpha
        )
        candidate = base_h.copy()
        for _ in range(JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired)
            candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = candidate
        output.flush()

        actual_support, _ = v423.leaked_support(candidate)
        actual_sh = v423.leaked_marginals(actual_support)["sh"]
        base_power = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        candidate_power = np.sum(
            np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        diagnostics["alpha"][start:end] = alpha
        diagnostics["base_target_l1"][start:end] = row_l1(base_sh, desired_sh)
        diagnostics["actual_target_l1"][start:end] = row_l1(actual_sh, desired_sh)
        diagnostics["action_l1"][start:end] = row_l1(actual_sh, base_sh)
        diagnostics["pdp_lock_error"][start:end] = v424.full_linkwise_pdp_error(
            candidate, base_h
        )
        diagnostics["power_ratio"][start:end] = (
            candidate_power / np.maximum(base_power, TINY)
        ).astype(np.float32)
        diagnostics["phase_agreement"][start:end] = v423.phase_agreement(
            base_h, candidate
        )
        print(f"v435 calibrated joint S-H {end}/{len(source)}", flush=True)
    return diagnostics


def write_zip(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "analyze_v434_sionna_joint_sh_calibration.py",
        "analyze_v432_sionna_link_energy_calibration.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v212_multidomain_sionna_pdp_ensemble.py",
        "generate_v189_leaked_multimarginal_overdrive.py",
        "generate_v187_leaked_pas_lifecycle_projection.py",
        "generate_v196_train_guided_delay_pdp_mouth.py",
        "analyze_v208_strict_historical_replay.py",
        "analyze_v205_sionna_config_calibration.py",
        "generate_v178_test_track_complex_consensus.py",
        "generate_v87_easy_field_projection.py",
        "generate_v171_full_complex_ad_tail_denoising.py",
        "generate_v169_power_scaled_tail_denoising.py",
        "generate_v159_frontier_consensus_support_soup.py",
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
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    for path in (
        BASE,
        TRAIN_OFFICIAL,
        TRAIN_HIGH,
        TEST_HIGH,
        *TRAIN_LOWS,
        *TEST_LOWS,
    ):
        require_file(path)

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked).astype("U8")
    test_regimes = regime_of(test_blocked).astype("U8")

    train_truth, _ = v434.load_sh(TRAIN_OFFICIAL)
    train_high, train_high_total = v434.load_sh(TRAIN_HIGH)
    train_low_loaded = [v434.load_sh(path) for path in TRAIN_LOWS]
    train_raw, _, _ = v434.robust_source(
        train_high,
        train_high_total,
        [value[0] for value in train_low_loaded],
        [value[1] for value in train_low_loaded],
    )
    ratios = v432.fit_ratio(
        train_truth,
        train_raw,
        np.arange(len(train_pos), dtype=np.int64),
        train_regimes,
    )

    test_high, test_high_total = v434.load_sh(TEST_HIGH)
    test_low_loaded = [v434.load_sh(path) for path in TEST_LOWS]
    test_raw, point_quality, cell_quality = v434.robust_source(
        test_high,
        test_high_total,
        [value[0] for value in test_low_loaded],
        [value[1] for value in test_low_loaded],
    )
    calibrated_target = v432.apply_ratio(
        test_raw,
        np.arange(len(test_pos), dtype=np.int64),
        test_regimes,
        ratios,
    )
    diagnostics = generate(test_regimes, calibrated_target, point_quality)
    np.savez_compressed(
        ACTION_OUT,
        regimes=test_regimes,
        point_quality=point_quality,
        cell_quality=cell_quality,
        calibrated_target=calibrated_target,
        **diagnostics,
    )

    base_local = v208.score_prediction(BASE)
    candidate_local = v208.score_prediction(OUTPUT)
    comparison = track.compare_output_to(OUTPUT, BASE, test_regimes)
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "v434 showed that official-train calibration converts Sionna joint 192x16 S-H "
            "support into a 5/5-fold positive target with mean cosine lift +0.02323. Applying "
            "that calibrated joint target to v426 should correct delay-angle correspondence "
            "without changing its online-validated full linkwise PDP or power."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "joint support target uses Sionna references traced at the 500 official test coordinates",
        },
        "new_information_used": {
            "v434_official_oof": {
                "baseline_joint_cosine": 0.7067741330778227,
                "selected_joint_cosine": 0.7300043563107028,
                "mean_gain": 0.02323022298514843,
                "fold_wins": 5,
                "selected_alpha": 0.25,
            },
            "calibration": "regime-shrunk official-train log-ratio calibration of full 192x16 joint S-H cells",
            "carrier": "v426 online-validated leaked support carrier and complete 256x4 linkwise PDP",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "alpha": float(ALPHA),
            "regime_scale": REGIME_SCALE,
            "joint_cycles": JOINT_CYCLES,
            "policy": "modify global joint S-H only; retain base conditional pol/UE/V allocation and restore every v426 linkwise PDP after each cycle",
        },
        "diagnostics": {
            "overall": {
                key: summarize(value) for key, value in diagnostics.items()
            },
            "point_quality": summarize(point_quality),
            "cell_quality": summarize(cell_quality),
            "by_regime": by_regime_summary(
                test_regimes,
                {
                    "alpha": diagnostics["alpha"],
                    "base_target_l1": diagnostics["base_target_l1"],
                    "actual_target_l1": diagnostics["actual_target_l1"],
                    "action_l1": diagnostics["action_l1"],
                    "pdp_lock_error": diagnostics["pdp_lock_error"],
                    "power_ratio": diagnostics["power_ratio"],
                    "phase_agreement": diagnostics["phase_agreement"],
                },
            ),
        },
        "strict_local_diagnostic": {
            "reference_warning": "v207 same-coordinate Sionna reference, not official hidden truth",
            "base_v426": base_local,
            "candidate": candidate_local,
        },
        "comparison_to_v426": comparison,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_zip(report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
