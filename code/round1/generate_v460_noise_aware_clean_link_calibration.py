#!/usr/bin/env python3
"""Calibrate Sionna linkwise PDP against noise-floor-subtracted official train H."""

from __future__ import annotations

import json
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
from generate_v159_frontier_consensus_support_soup import digest
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 460
VERSION = "v460_noise_aware_clean_link_calibration"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DENOISED_TRUTH = OUT_DIR / "v460_denoised_official_linkwise_pdp.npy"
NOISE_FLOOR = OUT_DIR / "v460_link_noise_floor.npy"
TEST_TARGET = OUT_DIR / "v460_noise_aware_test_link_target.npy"
OUTPUT = OUT_DIR / "v460_Round1_Test_Channel.npy"
SUPPORT_OUTPUT = OUT_DIR / "v460_support.npy"
ACTION_OUT = OUT_DIR / "v460_noise_aware_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
BASE = PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy"
V458 = PATHS.cache_dir(458) / "v458_Round1_Test_Channel.npy"
JOINT_ACTIONS = PATHS.cache_dir(435) / "v435_train_calibrated_joint_sh_actions.npz"
POWER_ACTIONS = PATHS.cache_dir(433) / "v433_sionna_calibrated_power_release_actions.npz"

BATCH = 2
LINK_BATCH = 16
QUERY_BATCH = 4
JOINT_CYCLES = 5
POWER_SCALE = 10.0
LOG_FLOOR = 1e-10
ALPHA_GRID = np.linspace(0.0, 1.0, 21, dtype=np.float32)


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def build_denoised_truth() -> dict[str, float]:
    if DENOISED_TRUTH.is_file() and NOISE_FLOOR.is_file():
        floor = np.load(NOISE_FLOOR, mmap_mode="r")
        return {
            "mean_floor": float(np.mean(floor)),
            "median_floor": float(np.median(floor)),
        }
    channel = np.load(OFFICIAL, mmap_mode="r")
    links = channel.shape[1] * channel.shape[2]
    delays = channel.shape[3]
    truth_out = np.lib.format.open_memmap(
        DENOISED_TRUTH,
        mode="w+",
        dtype=np.float32,
        shape=(len(channel), links, delays),
    )
    floor_out = np.lib.format.open_memmap(
        NOISE_FLOOR, mode="w+", dtype=np.float32, shape=(len(channel), links)
    )
    removed_fraction = []
    for start in range(0, len(channel), BATCH):
        end = min(start + BATCH, len(channel))
        value = np.asarray(channel[start:end], dtype=np.complex64)
        cir = np.fft.ifft(value, axis=-1)
        power = (np.abs(cir).astype(np.float32) ** 2).reshape(end - start, links, delays)
        floor = np.median(power, axis=-1).astype(np.float32) / np.float32(np.log(2.0))
        clean = np.maximum(power - floor[..., None], 0.0)
        original_total = power.sum(axis=-1, dtype=np.float64)
        clean_total = clean.sum(axis=-1, dtype=np.float64)
        removed_fraction.append(
            (1.0 - clean_total / np.maximum(original_total, 1e-30)).reshape(-1)
        )
        shape = clean / np.maximum(clean_total[..., None], TINY)
        zero = clean_total <= 1e-30
        shape[zero] = 0.0
        truth_out[start:end] = shape.astype(np.float32)
        floor_out[start:end] = floor
        if end % 100 == 0 or end == len(channel):
            truth_out.flush()
            floor_out.flush()
            print(f"v460 denoised truth {end}/{len(channel)}", flush=True)
    removed = np.concatenate(removed_fraction)
    return {
        "mean_floor": float(np.mean(floor_out)),
        "median_floor": float(np.median(floor_out)),
        "removed_power_fraction_mean": float(np.mean(removed)),
        "removed_power_fraction_median": float(np.median(removed)),
        "removed_power_fraction_p90": float(np.quantile(removed, 0.9)),
    }


def oof_calibration(
    truth: np.ndarray,
    source: np.ndarray,
    quality: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, object]:
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    fold_id = np.full(len(positions), -1, dtype=np.int16)
    links = truth.shape[1]
    scores = np.zeros((len(ALPHA_GRID), len(positions)), dtype=np.float64)
    baseline_score = np.zeros(len(positions), dtype=np.float64)
    calibrated_score = np.zeros(len(positions), dtype=np.float64)
    for fold, query_ids in enumerate(folds):
        fold_id[query_ids] = fold
        fit_ids = np.setdiff1d(all_ids, query_ids)
        neighbor_ids, weights = v447.neighbor_bank(
            fit_ids, query_ids, positions, blocked, regimes
        )
        for start in range(0, links, LINK_BATCH):
            end = min(start + LINK_BATCH, links)
            ratios = v447.fit_ratio_chunk(
                truth, source, fit_ids, regimes, start, end
            )
            baseline = v447.baseline_chunk(
                truth, neighbor_ids, weights, start, end
            )
            calibrated = v447.apply_ratio_chunk(
                source, query_ids, regimes, ratios, start, end
            )
            truth_chunk = np.asarray(truth[query_ids, start:end], dtype=np.float32)
            quality_chunk = np.asarray(quality[query_ids, start:end], dtype=np.float32)
            baseline_score[query_ids] += v447.link_cosine(
                truth_chunk, baseline
            ).sum(axis=1)
            calibrated_score[query_ids] += v447.link_cosine(
                truth_chunk, calibrated
            ).sum(axis=1)
            for index, alpha in enumerate(ALPHA_GRID):
                effective = np.clip(alpha * quality_chunk, 0.0, 1.0)
                candidate = v447.normalize_links(
                    baseline * (1.0 - effective[..., None])
                    + calibrated * effective[..., None]
                )
                scores[index, query_ids] += v447.link_cosine(
                    truth_chunk, candidate
                ).sum(axis=1)
        print(f"v460 OOF fold {fold + 1}/5", flush=True)
    baseline_score /= links
    calibrated_score /= links
    scores /= links
    mean_by_alpha = scores.mean(axis=1)
    selected_index = int(np.argmax(mean_by_alpha))
    selected_alpha = float(ALPHA_GRID[selected_index])
    selected_score = scores[selected_index]
    fold_rows = []
    for fold in range(5):
        mask = fold_id == fold
        fold_rows.append(
            {
                "fold": fold,
                "baseline": float(np.mean(baseline_score[mask])),
                "selected": float(np.mean(selected_score[mask])),
                "gain": float(np.mean(selected_score[mask] - baseline_score[mask])),
            }
        )
    return {
        "selected_alpha": selected_alpha,
        "baseline_mean": float(np.mean(baseline_score)),
        "calibrated_mean": float(np.mean(calibrated_score)),
        "selected_mean": float(np.mean(selected_score)),
        "mean_gain": float(np.mean(selected_score - baseline_score)),
        "positive_folds": int(sum(row["gain"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "alpha_grid": ALPHA_GRID.tolist(),
        "mean_by_alpha": mean_by_alpha.tolist(),
    }


def build_test_target(
    truth: np.ndarray,
    source: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> None:
    if TEST_TARGET.is_file():
        return
    raw_test = np.load(v458.RAW_LINK_TARGET, mmap_mode="r")
    fit_ids = np.arange(len(truth), dtype=np.int64)
    target = np.lib.format.open_memmap(
        TEST_TARGET, mode="w+", dtype=np.float32, shape=raw_test.shape
    )
    for start in range(0, source.shape[1], LINK_BATCH):
        end = min(start + LINK_BATCH, source.shape[1])
        ratios = v447.fit_ratio_chunk(
            truth, source, fit_ids, train_regimes, start, end
        )
        values = np.asarray(raw_test[:, start:end], dtype=np.float32)
        output = np.empty_like(values)
        for regime in v447.REGIMES:
            mask = test_regimes == regime
            if np.any(mask):
                output[mask] = v447.normalize_links(
                    values[mask] * np.exp(ratios[regime])[None]
                )
        target[:, start:end] = output
        if end % 128 == 0 or end == source.shape[1]:
            target.flush()
            print(f"v460 test target {end}/{source.shape[1]}", flush=True)


def generate_candidate(
    selected_alpha: float, test_regimes: np.ndarray
) -> dict[str, float]:
    source = np.load(BASE, mmap_mode="r")
    target = np.load(TEST_TARGET, mmap_mode="r")
    quality = np.load(v458.LINK_QUALITY, mmap_mode="r")
    joint = np.load(JOINT_ACTIONS)
    point_quality = joint["point_quality"].astype(np.float32)
    calibrated_joint = joint["calibrated_target"].astype(np.float32)
    profile = next(value for value in v458.JOINT_PROFILES if value.name == "tail085")
    joint_alpha = v458.joint_alpha(profile, test_regimes, point_quality)
    # v458 online validated scale 0.40. Do not exceed it based only on denoised OOF.
    deployment_alpha = min(float(selected_alpha), 0.40)
    link_alpha = v458.link_alpha(
        deployment_alpha,
        test_regimes,
        np.asarray(quality, dtype=np.float32),
    )
    support = np.lib.format.open_memmap(
        SUPPORT_OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        desired_support, _, _ = v435.build_desired_support(
            base_support, calibrated_joint[start:end], joint_alpha[start:end]
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        target_link = np.asarray(target[start:end], dtype=np.float32).reshape(
            end - start, source.shape[1], source.shape[2], source.shape[3]
        )
        desired_link = v426.desired_linkwise_pdp(
            base_link,
            target_link,
            link_alpha[start:end].reshape(end - start, source.shape[1], source.shape[2]),
        )
        candidate = base_h.copy()
        for _ in range(JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_link)
        support[start:end] = candidate
        if end % 50 == 0 or end == len(source):
            support.flush()
    log_delta = np.load(POWER_ACTIONS)["final_log_delta"].astype(np.float32)
    amplitude = np.exp(0.5 * np.float32(POWER_SCALE) * log_delta).astype(np.float32)
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), 10):
        end = min(start + 10, len(source))
        value = np.asarray(support[start:end], dtype=np.complex64).copy()
        value *= amplitude[start:end, None, None, None]
        output[start:end] = value
        output.flush()
    np.savez_compressed(
        ACTION_OUT,
        test_regimes=test_regimes,
        joint_alpha=joint_alpha,
        link_alpha=link_alpha.astype(np.float16),
        selected_oof_alpha=np.float32(selected_alpha),
        deployment_alpha=np.float32(deployment_alpha),
    )
    return {
        "selected_oof_alpha": float(selected_alpha),
        "deployment_alpha": deployment_alpha,
        "joint_alpha_mean": float(np.mean(joint_alpha)),
        "link_alpha_mean": float(np.mean(link_alpha)),
    }


def write_zip(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v460_noise_aware_clean_link_calibration.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "analyze_v447_sionna_linkwise_pdp_shape_calibration.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
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
    value = np.load(OUTPUT, mmap_mode="r")
    report["integrity"] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    for path in (
        OFFICIAL,
        v447.SOURCE_CACHE,
        v447.QUALITY_CACHE,
        v458.RAW_LINK_TARGET,
        v458.LINK_QUALITY,
        BASE,
        V458,
        JOINT_ACTIONS,
        POWER_ACTIONS,
    ):
        require_file(path)
    noise = build_denoised_truth()
    truth = np.load(DENOISED_TRUTH, mmap_mode="r")
    source = np.load(v447.SOURCE_CACHE, mmap_mode="r")
    quality = np.load(v447.QUALITY_CACHE, mmap_mode="r")
    positions = np.load(TRAIN_POS).astype(np.float32)
    geometry = np.load(GEOMETRY)
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked).astype("U8")
    test_regimes = regime_of(test_blocked).astype("U8")
    oof = oof_calibration(
        truth, source, quality, positions, train_blocked, train_regimes
    )
    build_test_target(truth, source, train_regimes, test_regimes)
    action = generate_candidate(float(oof["selected_alpha"]), test_regimes)
    candidate = v208.score_prediction(OUTPUT)
    baseline = v208.score_prediction(V458)
    replay_delta = float(
        candidate["score_traceback_leaked"] - baseline["score_traceback_leaked"]
    )
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant"
        if replay_delta > 0.0
        else "rejected_analysis_only",
        "primary_hypothesis": (
            "Official train H contains additive complex Gaussian noise. Estimating and "
            "subtracting its per-link delay-domain floor before Sionna calibration should "
            "produce a target closer to the clean hidden channel than raw train PDP."
        ),
        "noise_model": {
            "estimator": "median delay power divided by ln(2), then nonnegative floor subtraction",
            **noise,
        },
        "official_train_oof": oof,
        "test_action": action,
        "baseline_v458": baseline,
        "candidate": candidate,
        "replay_delta_vs_v458": replay_delta,
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "deployment target still uses Sionna references at official test coordinates",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT) if replay_delta > 0.0 else None,
        },
    }
    if replay_delta > 0.0:
        write_zip(report)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "noise_model": report["noise_model"],
        "official_train_oof": oof,
        "test_action": action,
        "candidate": candidate,
        "replay_delta_vs_v458": report["replay_delta_vs_v458"],
        "zip": report["outputs"]["zip"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
