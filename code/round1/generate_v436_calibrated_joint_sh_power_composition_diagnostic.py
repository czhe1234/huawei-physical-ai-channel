#!/usr/bin/env python3
"""Compose the independently validated v435 joint support and v433 power."""

from __future__ import annotations

import json
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as v208
import generate_v178_test_track_complex_consensus as track
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
from generate_v87_easy_field_projection import summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 436
VERSION = "v436_calibrated_joint_sh_power_composition_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v436_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v436_composition_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy"
SUPPORT = PATHS.cache_dir(435) / "v435_Round1_Test_Channel.npy"
SUPPORT_ACTIONS = PATHS.cache_dir(435) / "v435_train_calibrated_joint_sh_actions.npz"
POWER_ACTIONS = PATHS.cache_dir(433) / "v433_sionna_calibrated_power_release_actions.npz"
BATCH = 10


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def generate() -> dict[str, np.ndarray]:
    source = np.load(SUPPORT, mmap_mode="r")
    power = np.load(POWER_ACTIONS)
    amp_factor = power["amp_factor"].astype(np.float32)
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    actual_factor = np.empty(len(source), dtype=np.float32)
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
        actual_factor[start:end] = (after / np.maximum(before, TINY)).astype(np.float32)
        print(f"v436 orthogonal composition {end}/{len(source)}", flush=True)
    return {
        "amp_factor": amp_factor,
        "actual_power_factor": actual_factor,
        "final_log_delta": power["final_log_delta"].astype(np.float32),
        "power_confidence": power["confidence"].astype(np.float32),
        "power_active": power["active"].astype(bool),
    }


def write_zip(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v436_calibrated_joint_sh_power_composition_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "analyze_v434_sionna_joint_sh_calibration.py",
        "generate_v433_sionna_calibrated_power_release_diagnostic.py",
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
    for path in (BASE, SUPPORT, SUPPORT_ACTIONS, POWER_ACTIONS):
        require_file(path)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["test_blocked"].astype(np.float32)).astype("U8")
    actions = generate()
    support_actions = np.load(SUPPORT_ACTIONS)
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes,
        support_alpha=support_actions["alpha"].astype(np.float32),
        support_action_l1=support_actions["action_l1"].astype(np.float32),
        support_pdp_lock_error=support_actions["pdp_lock_error"].astype(np.float32),
        **actions,
    )

    base_local = v208.score_prediction(BASE)
    support_local = v208.score_prediction(SUPPORT)
    candidate_local = v208.score_prediction(OUTPUT)
    comparison = track.compare_output_to(OUTPUT, BASE, regimes)
    support_channel = np.load(SUPPORT, mmap_mode="r")
    output_channel = np.load(OUTPUT, mmap_mode="r")
    normalized_lock_error = np.empty(len(output_channel), dtype=np.float32)
    for start in range(0, len(output_channel), 2):
        end = min(start + 2, len(output_channel))
        normalized_lock_error[start:end] = v424.full_linkwise_pdp_error(
            np.asarray(output_channel[start:end], dtype=np.complex64),
            np.asarray(support_channel[start:end], dtype=np.complex64),
        )

    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "v434 validated train-calibrated joint S-H support and v432 independently validated "
            "train-calibrated point power. The two actions are mathematically orthogonal: v435 "
            "changes normalized joint support while preserving power, and v433 applies only a "
            "point scalar. Their composition should retain the full v435 spectral gain while "
            "adding the remaining NMSE/power gain."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "both component targets use Sionna references traced at the 500 official test coordinates",
        },
        "new_information_used": {
            "joint_support_oof": "v434 mean +0.023230 joint S-H cosine, 5/5 folds positive",
            "point_power_oof": "v432 log-MAE 2.371273 -> 2.047531, 5/5 folds positive",
            "composition_identity": "a per-point scalar leaves all normalized PAS/PDP and linkwise PDP shapes unchanged",
        },
        "components": {
            "support": PATHS.relative(SUPPORT),
            "power_actions": PATHS.relative(POWER_ACTIONS),
        },
        "diagnostics": {
            "support_alpha": summarize(support_actions["alpha"]),
            "support_action_l1": summarize(support_actions["action_l1"]),
            "power_log_delta": summarize(actions["final_log_delta"]),
            "actual_power_factor": summarize(actions["actual_power_factor"]),
            "normalized_pdp_lock_error": summarize(normalized_lock_error),
            "by_regime": by_regime_summary(
                regimes,
                {
                    "support_alpha": support_actions["alpha"],
                    "support_action_l1": support_actions["action_l1"],
                    "power_log_delta": actions["final_log_delta"],
                    "actual_power_factor": actions["actual_power_factor"],
                    "power_active": actions["power_active"].astype(np.float32),
                },
            ),
        },
        "strict_local_diagnostic": {
            "reference_warning": "v207 same-coordinate Sionna reference, not official hidden truth",
            "base_v426": base_local,
            "support_only_v435": support_local,
            "combined_v436": candidate_local,
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
