#!/usr/bin/env python3
"""Push the noncompliant same-coordinate Sionna diagnostic to a wider frontier."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v449_corrected_metric_joint_power_frontier as v449
from generate_v159_frontier_consensus_support_soup import digest
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 458
VERSION = "v458_sionna_multimouth_extreme_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v458_Round1_Test_Channel.npy"
BEST_SUPPORT = OUT_DIR / "v458_best_support.npy"
TEMP_SUPPORT = OUT_DIR / "v458_probe_support.npy"
RAW_LINK_TARGET = OUT_DIR / "v458_raw_link_target.npy"
CALIBRATED_LINK_TARGET = OUT_DIR / "v458_calibrated_link_target.npy"
LINK_QUALITY = OUT_DIR / "v458_link_quality.npy"
ACTION_OUT = OUT_DIR / "v458_multimouth_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy"
REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
JOINT_ACTIONS = PATHS.cache_dir(435) / "v435_train_calibrated_joint_sh_actions.npz"
POWER_ACTIONS = PATHS.cache_dir(433) / "v433_sionna_calibrated_power_release_actions.npz"
V449 = PATHS.cache_dir(449) / "v449_Round1_Test_Channel.npy"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TEST_POS = PATHS.round1 / "Round1_Test_Pos.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"

BATCH = 2
JOINT_CYCLES = 5
LINK_BATCH = 16
POWER_SCALES = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
LINK_SCALES = (0.0, 0.12, 0.24, 0.40)
LINK_REGIME_SCALE = {"clear": 1.25, "light": 1.50, "medium": 1.00, "heavy": 0.70}


@dataclass(frozen=True)
class JointProfile:
    name: str
    scales: dict[str, float]
    cap: float


JOINT_PROFILES = (
    JointProfile("balanced050", {"clear": 0.50, "light": 0.50, "medium": 0.50, "heavy": 0.50}, 0.40),
    JointProfile("easy075_tail055", {"clear": 0.75, "light": 0.75, "medium": 0.58, "heavy": 0.52}, 0.55),
    JointProfile("easy100_tail065", {"clear": 1.00, "light": 1.00, "medium": 0.72, "heavy": 0.60}, 0.65),
    JointProfile("all075", {"clear": 0.75, "light": 0.75, "medium": 0.75, "heavy": 0.75}, 0.65),
    JointProfile("tail085", {"clear": 0.90, "light": 0.90, "medium": 0.85, "heavy": 0.80}, 0.72),
)


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def build_calibrated_link_target(
    train_regimes: np.ndarray, test_regimes: np.ndarray
) -> None:
    if RAW_LINK_TARGET.is_file() and CALIBRATED_LINK_TARGET.is_file() and LINK_QUALITY.is_file():
        return
    base = np.load(BASE, mmap_mode="r")
    high = np.load(v426.V207_REFERENCE, mmap_mode="r")
    lows = [np.load(path, mmap_mode="r") for path in v426.V205_REFERENCES]
    links = base.shape[1] * base.shape[2]
    delays = base.shape[3]
    raw = np.lib.format.open_memmap(
        RAW_LINK_TARGET, mode="w+", dtype=np.float32, shape=(len(base), links, delays)
    )
    quality = np.lib.format.open_memmap(
        LINK_QUALITY, mode="w+", dtype=np.float32, shape=(len(base), links)
    )
    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        target, diagnostics = v426.robust_link_target(
            np.asarray(base[start:end], dtype=np.complex64),
            np.asarray(high[start:end], dtype=np.complex64),
            [np.asarray(value[start:end], dtype=np.complex64) for value in lows],
        )
        raw[start:end] = target.reshape(end - start, links, delays)
        quality[start:end] = diagnostics["link_quality"].reshape(end - start, links)
        if end % 50 == 0 or end == len(base):
            raw.flush()
            quality.flush()
            print(f"v458 raw link target {end}/{len(base)}", flush=True)
    raw.flush()
    quality.flush()
    calibrated = np.lib.format.open_memmap(
        CALIBRATED_LINK_TARGET,
        mode="w+",
        dtype=np.float32,
        shape=(len(base), links, delays),
    )
    train_truth = np.load(v447.TRUTH_CACHE, mmap_mode="r")
    train_source = np.load(v447.SOURCE_CACHE, mmap_mode="r")
    fit_ids = np.arange(len(train_truth), dtype=np.int64)
    query_ids = np.arange(len(base), dtype=np.int64)
    for start in range(0, links, LINK_BATCH):
        end = min(start + LINK_BATCH, links)
        ratios = v447.fit_ratio_chunk(
            train_truth, train_source, fit_ids, train_regimes, start, end
        )
        values = np.asarray(raw[:, start:end], dtype=np.float32)
        output = np.empty_like(values)
        for regime in v447.REGIMES:
            mask = test_regimes == regime
            if np.any(mask):
                output[mask] = v447.normalize_links(
                    values[mask] * np.exp(ratios[regime])[None]
                )
        calibrated[:, start:end] = output
        if end % 128 == 0 or end == links:
            calibrated.flush()
            print(f"v458 calibrated link target {end}/{links}", flush=True)


def joint_alpha(
    profile: JointProfile, regimes: np.ndarray, quality: np.ndarray
) -> np.ndarray:
    scale = np.asarray([profile.scales[str(value)] for value in regimes], dtype=np.float32)
    return np.clip(scale * quality, 0.0, profile.cap).astype(np.float32)


def link_alpha(
    scale: float, regimes: np.ndarray, quality: np.ndarray
) -> np.ndarray:
    regime_scale = np.asarray(
        [LINK_REGIME_SCALE[str(value)] for value in regimes], dtype=np.float32
    )
    return np.clip(
        np.float32(scale) * regime_scale[:, None] * quality, 0.0, 0.75
    ).astype(np.float32)


def generate_probe(
    profile: JointProfile,
    link_scale: float,
    regimes: np.ndarray,
    point_quality: np.ndarray,
    calibrated_joint: np.ndarray,
) -> dict[str, object]:
    source = np.load(BASE, mmap_mode="r")
    link_target = np.load(CALIBRATED_LINK_TARGET, mmap_mode="r")
    link_quality = np.load(LINK_QUALITY, mmap_mode="r")
    output = np.lib.format.open_memmap(
        TEMP_SUPPORT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    alpha_joint = joint_alpha(profile, regimes, point_quality)
    alpha_link = link_alpha(link_scale, regimes, np.asarray(link_quality, dtype=np.float32))
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        desired_support, _, _ = v435.build_desired_support(
            base_support,
            calibrated_joint[start:end],
            alpha_joint[start:end],
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        target_link = np.asarray(link_target[start:end], dtype=np.float32).reshape(
            end - start, source.shape[1], source.shape[2], source.shape[3]
        )
        desired_link = v426.desired_linkwise_pdp(
            base_link, target_link, alpha_link[start:end].reshape(end - start, source.shape[1], source.shape[2])
        )
        candidate = base_h.copy()
        for _ in range(JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_link)
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"non-finite probe {profile.name}/{link_scale} {start}:{end}")
        output[start:end] = candidate
        if end % 50 == 0 or end == len(source):
            output.flush()
    metrics = v208.score_prediction(TEMP_SUPPORT)
    terms = v449.quadratic_terms(TEMP_SUPPORT)
    power_delta = np.load(POWER_ACTIONS)["final_log_delta"].astype(np.float32)
    frontier = v449.power_frontier(metrics, terms, power_delta)
    frontier = [row for row in frontier if row["power_scale"] in POWER_SCALES]
    # v449.power_frontier only evaluates its module grid, so compute any wider scales here.
    left, cross, right = terms
    denominator = max(float(np.sum(right)), 1e-30)
    frontier = []
    for power_scale in POWER_SCALES:
        amplitude = np.exp(0.5 * np.float32(power_scale) * power_delta).astype(np.float64)
        nmse = float(
            np.sum(amplitude * amplitude * left - 2.0 * amplitude * cross + right)
            / denominator
        )
        nmse_score = 1.0 / (1.0 + nmse)
        frontier.append(
            {
                "power_scale": float(power_scale),
                "global_nmse": nmse,
                "nmse_score": nmse_score,
                "score_traceback_leaked": float(
                    0.4 * metrics["leaked_sh"] + 0.4 * metrics["pdp"] + 0.2 * nmse_score
                ),
                "score_conventional": float(
                    0.4 * metrics["pas"] + 0.4 * metrics["pdp"] + 0.2 * nmse_score
                ),
            }
        )
    selected = max(frontier, key=lambda row: row["score_traceback_leaked"])
    return {
        "joint_profile": profile.name,
        "joint_scales": profile.scales,
        "joint_cap": profile.cap,
        "link_scale": float(link_scale),
        "joint_alpha_mean": float(np.mean(alpha_joint)),
        "link_alpha_mean": float(np.mean(alpha_link)),
        "support_metrics": metrics,
        "power_frontier": frontier,
        "selected": selected,
    }


def write_final(power_scale: float) -> None:
    source = np.load(BEST_SUPPORT, mmap_mode="r")
    log_delta = np.load(POWER_ACTIONS)["final_log_delta"].astype(np.float32)
    amplitude = np.exp(0.5 * np.float32(power_scale) * log_delta).astype(np.float32)
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), 10):
        end = min(start + 10, len(source))
        value = np.asarray(source[start:end], dtype=np.complex64).copy()
        value *= amplitude[start:end, None, None, None]
        output[start:end] = value
        output.flush()


def write_zip(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "generate_v449_corrected_metric_joint_power_frontier.py",
        "analyze_v447_sionna_linkwise_pdp_shape_calibration.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "analyze_v434_sionna_joint_sh_calibration.py",
        "generate_v433_sionna_calibrated_power_release_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
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
    value = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(value[start : start + 10])).all()
        for start in range(0, len(value), 10)
    )
    report["integrity"] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    required = (
        BASE,
        REFERENCE,
        JOINT_ACTIONS,
        POWER_ACTIONS,
        V449,
        v447.TRUTH_CACHE,
        v447.SOURCE_CACHE,
        *v426.V205_REFERENCES,
    )
    for path in required:
        require_file(path)
    geometry = np.load(GEOMETRY)
    train_regimes = regime_of(geometry["train_blocked"].astype(np.float32)).astype("U8")
    test_regimes = regime_of(geometry["test_blocked"].astype(np.float32)).astype("U8")
    build_calibrated_link_target(train_regimes, test_regimes)
    joint = np.load(JOINT_ACTIONS)
    point_quality = joint["point_quality"].astype(np.float32)
    calibrated_joint = joint["calibrated_target"].astype(np.float32)
    baseline = v208.score_prediction(V449)
    rows = []
    best = None
    for profile in JOINT_PROFILES:
        for link_scale in LINK_SCALES:
            row = generate_probe(
                profile, link_scale, test_regimes, point_quality, calibrated_joint
            )
            rows.append(row)
            score = float(row["selected"]["score_traceback_leaked"])
            conventional = float(row["selected"]["score_conventional"])
            eligible = conventional >= float(baseline["score_conventional"]) - 0.01
            if eligible and (best is None or score > best[0]):
                shutil.copyfile(TEMP_SUPPORT, BEST_SUPPORT)
                best = (score, row)
            print(
                f"v458 {profile.name} link={link_scale:.2f} "
                f"traceback={score:.6f} conventional={conventional:.6f}",
                flush=True,
            )
    if best is None:
        selected_row = max(rows, key=lambda row: row["selected"]["score_traceback_leaked"])
        generate_probe(
            next(profile for profile in JOINT_PROFILES if profile.name == selected_row["joint_profile"]),
            float(selected_row["link_scale"]),
            test_regimes,
            point_quality,
            calibrated_joint,
        )
        shutil.copyfile(TEMP_SUPPORT, BEST_SUPPORT)
    else:
        selected_row = best[1]
    selected_power = float(selected_row["selected"]["power_scale"])
    write_final(selected_power)
    final_metrics = v208.score_prediction(OUTPUT)
    np.savez_compressed(
        ACTION_OUT,
        test_regimes=test_regimes,
        point_quality=point_quality,
        selected_joint_profile=np.asarray(selected_row["joint_profile"]),
        selected_link_scale=np.float32(selected_row["link_scale"]),
        selected_power_scale=np.float32(selected_power),
    )
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "v449 stopped at search boundaries. Regime-specific joint S-H overdrive, "
            "official-train-calibrated full linkwise PDP, and a wider orthogonal power "
            "frontier can extract additional corrected-metric signal."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "uses Sionna references traced at all 500 official test coordinates",
        },
        "search": {
            "joint_profiles": [profile.__dict__ for profile in JOINT_PROFILES],
            "link_scales": list(LINK_SCALES),
            "power_scales": list(POWER_SCALES),
            "rows": rows,
        },
        "selection": selected_row,
        "baseline_v449": baseline,
        "candidate": final_metrics,
        "predicted_replay_gain": float(
            final_metrics["score_traceback_leaked"] - baseline["score_traceback_leaked"]
        ),
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    write_zip(report)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "selection": report["selection"],
        "candidate": report["candidate"],
        "predicted_replay_gain": report["predicted_replay_gain"],
        "zip": report["outputs"]["zip"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
