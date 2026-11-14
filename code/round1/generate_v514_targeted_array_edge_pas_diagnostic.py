#!/usr/bin/env python3
"""Apply a targeted v513 array-edge PAS correction to the v491 carrier."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as v208
import generate_v178_test_track_complex_consensus as track
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 514
VERSION = "v514_targeted_array_edge_pas_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = OUT_DIR / "v514_Round1_Test_Channel.npy"
ACTIONS = OUT_DIR / "v514_targeted_array_edge_pas_actions.npz"
REFERENCE = OUT_DIR / "v514_array_edge_test_joint_reference.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ARCHIVE = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
BASE = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"

TRAIN_BASE = (
    PATHS.cache_dir(513)
    / "point_metrics"
    / "syn1_edge0_d16_3m_max100k_seed42_points160.npz"
)
TRAIN_COMBO = (
    PATHS.cache_dir(513)
    / "point_metrics"
    / "syn0_edge1_d16_3m_max100k_seed42_points160.npz"
)

BATCH = 2
JOINT_CYCLES = 5
SELECT_QUANTILE = 0.70
ALPHA_MIN = 0.15
ALPHA_MAX = 0.30
LOG_RATIO_CLIP = 2.0
FLOOR = 1e-12


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def row_l1(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(left.reshape(len(left), -1) - right.reshape(len(right), -1)),
        axis=1,
    ).astype(np.float32)


def selector_evidence() -> dict[str, object]:
    base = np.load(TRAIN_BASE)
    combo = np.load(TRAIN_COMBO)
    heavy = base["regimes"] == 3
    feature = np.abs(combo["pred_log_power"] - base["pred_log_power"])[heavy]
    pas_gain = (combo["leaked_full"] - base["leaked_full"])[heavy]
    pdp_gain = (combo["pdp"] - base["pdp"])[heavy]
    gain = 0.5 * (pas_gain + pdp_gain)
    threshold = float(np.quantile(feature, SELECT_QUANTILE))
    high = float(np.quantile(feature, 0.90))
    selected = feature >= threshold
    return {
        "heavy_count": int(np.sum(heavy)),
        "selection_quantile": SELECT_QUANTILE,
        "feature_threshold": threshold,
        "feature_high": high,
        "selected_count": int(np.sum(selected)),
        "selected_fraction": float(np.mean(selected)),
        "selected_positive_fraction": float(np.mean(gain[selected] > 0.0)),
        "selected_mean_pas_gain": float(np.mean(pas_gain[selected])),
        "selected_mean_pdp_gain": float(np.mean(pdp_gain[selected])),
        "selected_mean_combined_gain": float(np.mean(gain[selected])),
        "all_heavy_positive_fraction": float(np.mean(gain > 0.0)),
        "all_heavy_mean_combined_gain": float(np.mean(gain)),
    }


def build_policy(
    reference: np.lib.npyio.NpzFile,
    evidence: dict[str, object],
) -> dict[str, np.ndarray | float]:
    indices = reference["indices"].astype(np.int64)
    feature = np.abs(
        reference["combo_log_power"] - reference["baseline_log_power"]
    ).astype(np.float32)
    threshold = float(evidence["feature_threshold"])
    high = float(evidence["feature_high"])
    active_local = feature >= threshold
    strength = np.clip(
        (feature - threshold) / max(high - threshold, 1e-6), 0.0, 1.0
    ).astype(np.float32)
    alpha_local = np.where(
        active_local,
        ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * strength,
        0.0,
    ).astype(np.float32)
    alpha = np.zeros(500, dtype=np.float32)
    alpha[indices] = alpha_local
    reference_row = np.full(500, -1, dtype=np.int32)
    reference_row[indices] = np.arange(len(indices), dtype=np.int32)
    return {
        "indices": indices,
        "feature": feature,
        "threshold": threshold,
        "high": high,
        "active_local": active_local,
        "alpha_local": alpha_local,
        "alpha": alpha,
        "reference_row": reference_row,
    }


def render(reference, policy) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    diagnostics = {
        "alpha": np.asarray(policy["alpha"], dtype=np.float32),
        "action_l1": np.zeros(len(source), dtype=np.float32),
        "target_l1": np.zeros(len(source), dtype=np.float32),
        "pdp_lock_error": np.zeros(len(source), dtype=np.float32),
        "power_ratio": np.zeros(len(source), dtype=np.float32),
        "phase_agreement": np.zeros(len(source), dtype=np.float32),
    }
    reference_row = np.asarray(policy["reference_row"], dtype=np.int32)
    alpha = diagnostics["alpha"]
    baseline_sh = reference["baseline_sh"].astype(np.float32)
    combo_sh = reference["combo_sh"].astype(np.float32)

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        base_sh = v423.leaked_marginals(base_support)["sh"]
        target_sh = base_sh.copy()
        for local, point_id in enumerate(range(start, end)):
            row = int(reference_row[point_id])
            if row < 0 or alpha[point_id] <= 0.0:
                continue
            delta = np.clip(
                np.log(np.maximum(combo_sh[row], FLOOR))
                - np.log(np.maximum(baseline_sh[row], FLOOR)),
                -LOG_RATIO_CLIP,
                LOG_RATIO_CLIP,
            ).astype(np.float32)
            value = base_sh[local] * np.exp(delta).astype(np.float32)
            target_sh[local] = value / np.maximum(np.sum(value), TINY)
        desired, _, desired_sh = v435.build_desired_support(
            base_support,
            target_sh,
            alpha[start:end],
        )
        candidate = base_h.copy()
        for _ in range(JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired)
            candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"non-finite output block {start}:{end}")
        output[start:end] = candidate
        actual_support, _ = v423.leaked_support(candidate)
        actual_sh = v423.leaked_marginals(actual_support)["sh"]
        diagnostics["target_l1"][start:end] = row_l1(base_sh, desired_sh)
        diagnostics["action_l1"][start:end] = row_l1(base_sh, actual_sh)
        diagnostics["pdp_lock_error"][start:end] = v424.full_linkwise_pdp_error(
            candidate, base_h
        )
        base_power = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        output_power = np.sum(
            np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        diagnostics["power_ratio"][start:end] = (
            output_power / np.maximum(base_power, TINY)
        ).astype(np.float32)
        diagnostics["phase_agreement"][start:end] = v423.phase_agreement(
            base_h, candidate
        )
        if end % 50 == 0 or end == len(source):
            output.flush()
            print(f"v514 render {end}/{len(source)}", flush=True)
    output.flush()
    return diagnostics


def summarize(values: np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p50": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v514_targeted_array_edge_pas_diagnostic.py",
        "build_v514_array_edge_test_joint_reference.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v187_leaked_pas_lifecycle_projection.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
        archive.writestr("model/", b"")
        archive.write(REFERENCE, "model/v514_array_edge_test_joint_reference.npz")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(output[start : start + 5])).all()
        for start in range(0, len(output), 5)
    )
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ARCHIVE, "sha256"),
        "md5": digest(ARCHIVE, "md5"),
    }


def main() -> None:
    for path in (BASE, REFERENCE, TRAIN_BASE, TRAIN_COMBO):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    evidence = selector_evidence()
    with np.load(REFERENCE) as reference:
        policy = build_policy(reference, evidence)
        diagnostics = render(reference, policy)
        reference_summary = {
            "heavy_count": int(len(reference["indices"])),
            "frozen_train_abs_power_delta_q70": float(policy["threshold"]),
            "frozen_train_abs_power_delta_q90": float(policy["high"]),
            "active_count": int(np.sum(policy["active_local"])),
            "active_fraction_of_heavy": float(np.mean(policy["active_local"])),
        }

    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["test_blocked"]).astype("U8")
    active = diagnostics["alpha"] > 0.0
    np.savez_compressed(
        ACTIONS,
        regimes=regimes,
        active=active,
        selector_feature=np.asarray(policy["feature"], dtype=np.float32),
        selector_indices=np.asarray(policy["indices"], dtype=np.int64),
        **diagnostics,
    )
    base_replay = v208.score_prediction(BASE)
    replay = v208.score_prediction(OUTPUT)
    comparison = track.compare_output_to(OUTPUT, BASE, regimes)
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "The v513 non-synthetic-array x edge interaction contains useful heavy PAS "
            "information when applied only to heavy points exceeding the frozen official-"
            "train q70 paired-Sionna absolute log-power disagreement threshold."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "uses compact Sionna spectra traced at official test coordinates",
        },
        "base": {
            "version": 491,
            "online_score": 0.739634,
            "channel": PATHS.relative(BASE),
            "replay": base_replay,
        },
        "new_information": {
            "source": "paired 6.693GHz depth16 3M-ray synthetic/base and non-synthetic/edge joint S-H spectra",
            "selector_evidence": evidence,
            "test_policy": reference_summary,
        },
        "parameters": {
            "selection_quantile": SELECT_QUANTILE,
            "alpha_min": ALPHA_MIN,
            "alpha_max": ALPHA_MAX,
            "log_ratio_clip": LOG_RATIO_CLIP,
            "joint_cycles": JOINT_CYCLES,
            "regimes_modified": ["heavy"],
            "preserved": [
                "v491 full 256x4 linkwise PDP",
                "v491 per-link CIR magnitude and total power",
                "all clear/light/medium points",
            ],
        },
        "diagnostics": {
            "active_count": int(np.sum(active)),
            "active_by_regime": {
                name: int(np.sum(active & (regimes == name)))
                for name in ("clear", "light", "medium", "heavy")
            },
            "alpha": summarize(diagnostics["alpha"]),
            "action_l1": summarize(diagnostics["action_l1"]),
            "target_l1": summarize(diagnostics["target_l1"]),
            "pdp_lock_error": summarize(diagnostics["pdp_lock_error"]),
            "power_ratio": summarize(diagnostics["power_ratio"]),
            "phase_agreement": summarize(diagnostics["phase_agreement"]),
        },
        "strict_local_diagnostic": {
            "reference_warning": "v207 same-coordinate Sionna reference, not official hidden truth",
            "base_v491": base_replay,
            "candidate": replay,
            "delta": float(
                replay["score_traceback_leaked"]
                - base_replay["score_traceback_leaked"]
            ),
        },
        "comparison_to_v491": comparison,
        "online_score": None,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTIONS),
            "reference": PATHS.relative(REFERENCE),
            "archive": PATHS.relative(ARCHIVE),
            "report": PATHS.relative(REPORT),
        },
    }
    report["integrity"] = package(report)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
