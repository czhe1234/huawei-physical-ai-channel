#!/usr/bin/env python3
"""Release train-calibrated point power only after v426 support correction.

This diagnostic uses Sionna references traced at the official test coordinates
and is not compliant with the final competition data boundary.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import analyze_v208_strict_historical_replay as v208
import analyze_v432_sionna_link_energy_calibration as v432
import generate_v178_test_track_complex_consensus as track
from generate_v87_easy_field_projection import summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 433
VERSION = "v433_sionna_calibrated_power_release_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v433_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v433_sionna_calibrated_power_release_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy"
V426_ACTIONS = PATHS.cache_dir(426) / "v426_sionna_linkwise_pdp_completion_actions.npz"
V199_DIAG = PATHS.cache_dir(199) / "v199_test_power_diagnostics.npz"
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

MODEL_SEEDS = (43301, 43331, 43361, 43391, 43421)
REGIME_ALPHA = {"clear": 0.0, "light": 0.24, "medium": 0.68, "heavy": 0.88}
REGIME_CAP = {"clear": 0.0, "light": 0.20, "medium": 0.52, "heavy": 0.72}
BATCH = 10


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def log_point_power(path) -> np.ndarray:
    channel = np.load(path, mmap_mode="r")
    output = np.empty(len(channel), dtype=np.float32)
    for start in range(0, len(channel), BATCH):
        end = min(start + BATCH, len(channel))
        block = np.asarray(channel[start:end], dtype=np.complex64)
        power = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        output[start:end] = np.log(np.maximum(power, TINY)).astype(np.float32)
    return output


def fit_predict_power(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = []
    for seed in MODEL_SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=520,
            min_samples_leaf=6,
            max_features=0.85,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train_features, train_target)
        predictions.append(model.predict(test_features).astype(np.float32))
    stack = np.stack(predictions, axis=0).astype(np.float32)
    return (
        stack.mean(axis=0).astype(np.float32),
        stack.std(axis=0).astype(np.float32),
        stack,
    )


def build_action(
    regimes: np.ndarray,
    base_log: np.ndarray,
    model_log: np.ndarray,
    model_spread: np.ndarray,
    v426_actions: np.lib.npyio.NpzFile,
    old_diag: np.lib.npyio.NpzFile,
    low_logs: list[np.ndarray],
    high_log: np.ndarray,
) -> dict[str, np.ndarray]:
    old_target = (
        0.55 * old_diag["test_model_mean_log"].astype(np.float32)
        + 0.45 * old_diag["test_knn_log"].astype(np.float32)
    )
    raw_delta = model_log - base_log
    old_delta = old_target - base_log
    direction_agreement = (
        (np.sign(raw_delta) == np.sign(old_delta))
        | (np.abs(old_delta) < 0.08)
    )
    target_agreement = np.exp(-np.abs(model_log - old_target) / 1.25).astype(np.float32)
    model_confidence = np.exp(-model_spread / 0.42).astype(np.float32)
    domain_stack = np.stack([high_log, *low_logs], axis=1)
    domain_spread = np.std(domain_stack, axis=1).astype(np.float32)
    domain_confidence = np.exp(-domain_spread / 2.0).astype(np.float32)
    support_quality = v426_actions["point_quality"].astype(np.float32)
    link_quality = v426_actions["link_quality_mean"].astype(np.float32)
    train_confidence = old_diag["test_train_confidence"].astype(np.float32)
    conflict = old_diag["conflict"].astype(bool)

    confidence = np.clip(
        (0.20 + 0.80 * support_quality)
        * (0.35 + 0.65 * link_quality)
        * (0.45 + 0.55 * model_confidence)
        * (0.45 + 0.55 * target_agreement)
        * (0.55 + 0.45 * domain_confidence)
        * (0.60 + 0.40 * train_confidence),
        0.0,
        1.0,
    ).astype(np.float32)
    confidence[~direction_agreement] *= 0.18
    confidence[conflict] *= 0.15

    alpha = np.asarray([REGIME_ALPHA[name] for name in regimes], dtype=np.float32)
    cap = np.asarray([REGIME_CAP[name] for name in regimes], dtype=np.float32)
    log_delta = np.clip(alpha * confidence * raw_delta, -cap, cap).astype(np.float32)
    active = np.abs(log_delta) > 1e-5
    amp_factor = np.exp(0.5 * log_delta).astype(np.float32)
    return {
        "model_log_power": model_log,
        "model_spread": model_spread,
        "old_target_log_power": old_target,
        "base_log_power": base_log,
        "raw_log_delta": raw_delta,
        "old_log_delta": old_delta,
        "direction_agreement": direction_agreement,
        "target_agreement": target_agreement,
        "model_confidence": model_confidence,
        "domain_confidence": domain_confidence,
        "support_quality": support_quality,
        "link_quality": link_quality,
        "train_confidence": train_confidence,
        "conflict": conflict,
        "confidence": confidence,
        "final_log_delta": log_delta,
        "amp_factor": amp_factor,
        "active": active,
    }


def apply_action(amp_factor: np.ndarray) -> np.ndarray:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    actual = np.empty(len(source), dtype=np.float32)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        block = np.asarray(source[start:end], dtype=np.complex64)
        candidate = block * amp_factor[start:end, None, None, None]
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = candidate.astype(np.complex64)
        output.flush()
        before = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        after = np.sum(np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64)
        actual[start:end] = (after / np.maximum(before, TINY)).astype(np.float32)
        print(f"v433 calibrated power release {end}/{len(source)}", flush=True)
    return actual


def write_zip(report: dict[str, object]) -> None:
    dependencies = (
        PATHS.code / "generate_v433_sionna_calibrated_power_release_diagnostic.py",
        PATHS.code / "analyze_v432_sionna_link_energy_calibration.py",
        PATHS.code / "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        PATHS.code / "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        PATHS.code / "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        PATHS.code / "analyze_v208_strict_historical_replay.py",
        PATHS.code / "analyze_v205_sionna_config_calibration.py",
        PATHS.code / "generate_v178_test_track_complex_consensus.py",
        PATHS.code / "generate_v87_easy_field_projection.py",
        PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
        PATHS.code / "generate_v187_leaked_pas_lifecycle_projection.py",
        PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
        PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
        PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
        PATHS.code / "powerfield_common.py",
        PATHS.code_root / "project_paths.py",
    )
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source in dependencies:
            archive.write(source, source.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(output[start : start + BATCH])).all()
        for start in range(0, len(output), BATCH)
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
        V426_ACTIONS,
        V199_DIAG,
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
    regimes = regime_of(test_blocked).astype("U8")

    train_target = log_point_power(TRAIN_OFFICIAL)
    train_high = log_point_power(TRAIN_HIGH)
    test_high = log_point_power(TEST_HIGH)
    train_lows = [log_point_power(path) for path in TRAIN_LOWS]
    test_lows = [log_point_power(path) for path in TEST_LOWS]
    train_features = v432.power_features(
        train_pos, train_blocked, [train_high, *train_lows]
    )
    test_features = v432.power_features(
        test_pos, test_blocked, [test_high, *test_lows]
    )
    model_log, model_spread, model_stack = fit_predict_power(
        train_features, train_target, test_features
    )
    base_log = log_point_power(BASE)
    v426_actions = np.load(V426_ACTIONS)
    old_diag = np.load(V199_DIAG)
    action = build_action(
        regimes,
        base_log,
        model_log,
        model_spread,
        v426_actions,
        old_diag,
        test_lows,
        test_high,
    )
    actual_factor = apply_action(action["amp_factor"])
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes,
        model_predictions=model_stack,
        actual_power_factor=actual_factor,
        **action,
    )

    base_local = v208.score_prediction(BASE)
    candidate_local = v208.score_prediction(OUTPUT)
    comparison = track.compare_output_to(OUTPUT, BASE, regimes)
    active = action["active"].astype(bool)
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "v432 rejected Sionna link-energy transfer but found 5/5 official-train "
            "improvement for point total power. After v426 has corrected normalized PAS/PDP, "
            "a support-conditioned calibrated scalar power release can improve the remaining "
            "NMSE term without changing any validated normalized spectrum."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "power features use Sionna references traced at the 500 official test coordinates",
        },
        "new_information_used": {
            "v432_official_oof": {
                "baseline_log_mae": 2.371272563934326,
                "sionna_feature_log_mae": 2.0475306510925293,
                "fold_wins": 5,
                "link_energy_result": "rejected; alpha=0",
            },
            "v426_support_gate": "point and link quality from online-validated linkwise PDP completion",
            "independent_power_consensus": "v199 official-only KNN/model target gates direction conflicts",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "model_seeds": list(MODEL_SEEDS),
            "regime_alpha": REGIME_ALPHA,
            "regime_log_cap": REGIME_CAP,
            "policy": "per-point positive scalar only; preserve normalized PAS, every linkwise PDP, link allocation and complex phase",
        },
        "action": {
            "active_count": int(np.sum(active)),
            "up_count": int(np.sum(action["final_log_delta"] > 1e-5)),
            "down_count": int(np.sum(action["final_log_delta"] < -1e-5)),
            "direction_agreement_count": int(np.sum(action["direction_agreement"])),
            "conflict_count": int(np.sum(action["conflict"])),
            "final_log_delta": summarize(action["final_log_delta"]),
            "actual_power_factor": summarize(actual_factor),
            "model_spread": summarize(model_spread),
            "confidence": summarize(action["confidence"]),
            "by_regime": by_regime_summary(
                regimes,
                {
                    "active": active.astype(np.float32),
                    "final_log_delta": action["final_log_delta"],
                    "actual_power_factor": actual_factor,
                    "confidence": action["confidence"],
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
