#!/usr/bin/env python3
"""Inject the calibrated multi-seed linkwise-PDP residual into v549."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as replay
import analyze_v554_conditional_linkwise_projection as conditional
import analyze_v554_multiseed_linkwise_pdp_calibration as calibration
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v533_expanded_physical_sbi_action_match as v533
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 554
VERSION = "v554_multiseed_conditional_linkwise_pdp_v549_delay_power_lock"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v554_Round1_Test_Channel.npy"
ACTIONS = CACHE_DIR / "v554_multiseed_linkwise_pdp_residual_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
BASE = PATHS.cache_dir(549) / "v549_Round1_Test_Channel.npy"
CALIBRATION_REPORT = calibration.REPORT
CONDITIONAL_REPORT = conditional.REPORT
LINK_ACTION = calibration.TEST_ACTION
PROJECTION_CYCLES = 1
BATCH = 2
TINY = 1e-30


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def row_l1(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(left.reshape(len(left), -1) - right.reshape(len(right), -1)),
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1).astype(np.float64)
    right = right.reshape(len(right), -1).astype(np.float64)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return (numerator / np.maximum(denominator, TINY)).astype(np.float32)


def compare_channels(left_path: Path, right_path: Path) -> dict[str, float]:
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
    error = 0.0
    energy = 0.0
    coherence = 0.0
    count = 0
    for start in range(0, len(left), 5):
        end = min(start + 5, len(left))
        local_left = np.asarray(left[start:end], dtype=np.complex64)
        local_right = np.asarray(right[start:end], dtype=np.complex64)
        error += float(np.sum(np.abs(local_left - local_right) ** 2, dtype=np.float64))
        local_energy = np.sum(
            np.abs(local_right) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        energy += float(np.sum(local_energy))
        inner = np.sum(
            np.conj(local_right) * local_left,
            axis=(1, 2, 3),
            dtype=np.complex128,
        )
        norm = np.sqrt(
            np.sum(np.abs(local_left) ** 2, axis=(1, 2, 3), dtype=np.float64)
            * local_energy
        )
        coherence += float(np.sum(np.abs(inner) / np.maximum(norm, TINY)))
        count += end - start
    return {
        "nmse": float(error / max(energy, TINY)),
        "complex_cosine": float(coherence / max(count, 1)),
    }


def generate(test_regimes: np.ndarray) -> dict[str, object]:
    base = np.load(BASE, mmap_mode="r")
    action = np.load(LINK_ACTION, mmap_mode="r")
    expected_action_shape = (len(base), base.shape[1] * base.shape[2], base.shape[3])
    if action.shape != expected_action_shape:
        raise ValueError(
            f"link action mismatch {action.shape} != {expected_action_shape}"
        )
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    diagnostics = {
        "desired_pdp_error": np.empty(len(base), dtype=np.float32),
        "pdp_action_l1_per_link": np.empty(len(base), dtype=np.float32),
        "pdp_cosine_to_v549": np.empty(len(base), dtype=np.float32),
        "link_energy_max_relative_error": np.empty(len(base), dtype=np.float32),
        "global_delay_max_abs_error": np.empty(len(base), dtype=np.float32),
        "point_power_ratio": np.empty(len(base), dtype=np.float32),
        "phase_agreement": np.empty(len(base), dtype=np.float32),
        "sh_l1_to_v549": np.empty(len(base), dtype=np.float32),
        "psh_l1_to_v549": np.empty(len(base), dtype=np.float32),
        "shue_l1_to_v549": np.empty(len(base), dtype=np.float32),
        "shv_l1_to_v549": np.empty(len(base), dtype=np.float32),
    }
    pas_names = ("sh", "psh", "shue", "shv")
    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        base_pdp, base_link_energy = v426.linkwise_pdp(base_h)
        local_action = np.asarray(action[start:end], dtype=np.float32).reshape(
            end - start, -1, base.shape[3]
        )
        base_links = base_pdp.reshape(end - start, -1, base.shape[3])
        raw_target = v447_normalize(
            base_links * np.exp(local_action).astype(np.float32)
        )
        desired_links = np.empty_like(raw_target)
        for row in range(end - start):
            index = start + row
            if test_regimes[index] in ("medium", "heavy"):
                desired_links[row] = conditional.conditional_shape(
                    base_links[row],
                    raw_target[row],
                    base_link_energy[row].reshape(-1),
                )
            else:
                desired_links[row] = base_links[row]
        desired_pdp = desired_links.reshape(base_pdp.shape)
        base_support, _ = v423.leaked_support(base_h)
        base_pas = v423.leaked_marginals(base_support)
        candidate = base_h.copy()
        for _ in range(PROJECTION_CYCLES):
            candidate = v423.project_pas(candidate, base_pas)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_pdp)
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"non-finite v554 block {start}:{end}")
        output[start:end] = candidate
        actual_pdp, actual_link_energy = v426.linkwise_pdp(candidate)
        actual_support, _ = v423.leaked_support(candidate)
        actual_pas = v423.leaked_marginals(actual_support)
        diagnostics["desired_pdp_error"][start:end] = np.max(
            np.abs(actual_pdp - desired_pdp), axis=(1, 2, 3)
        ).astype(np.float32)
        diagnostics["pdp_action_l1_per_link"][start:end] = row_l1(
            actual_pdp, base_pdp
        ) / np.float32(base.shape[1] * base.shape[2])
        diagnostics["pdp_cosine_to_v549"][start:end] = row_cosine(actual_pdp, base_pdp)
        diagnostics["link_energy_max_relative_error"][start:end] = np.max(
            np.abs(actual_link_energy - base_link_energy)
            / np.maximum(base_link_energy, TINY),
            axis=(1, 2),
        ).astype(np.float32)
        base_delay = np.sum(
            base_pdp * base_link_energy[..., None], axis=(1, 2), dtype=np.float64
        )
        actual_delay = np.sum(
            actual_pdp * actual_link_energy[..., None],
            axis=(1, 2),
            dtype=np.float64,
        )
        base_delay /= np.maximum(np.sum(base_delay, axis=1, keepdims=True), TINY)
        actual_delay /= np.maximum(np.sum(actual_delay, axis=1, keepdims=True), TINY)
        diagnostics["global_delay_max_abs_error"][start:end] = np.max(
            np.abs(actual_delay - base_delay), axis=1
        ).astype(np.float32)
        base_power = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        candidate_power = np.sum(
            np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        diagnostics["point_power_ratio"][start:end] = (
            candidate_power / np.maximum(base_power, TINY)
        ).astype(np.float32)
        diagnostics["phase_agreement"][start:end] = v423.phase_agreement(
            base_h, candidate
        )
        for name in pas_names:
            diagnostics[f"{name}_l1_to_v549"][start:end] = row_l1(
                actual_pas[name], base_pas[name]
            )
        if end == len(base) or end % 20 == 0:
            output.flush()
            print(
                f"v554 linkwise-PDP residual projection {end}/{len(base)}", flush=True
            )
    output.flush()
    np.savez_compressed(ACTIONS, **diagnostics)
    return {
        "projection_cycles": PROJECTION_CYCLES,
        "diagnostics": {name: summarize(value) for name, value in diagnostics.items()},
    }


def v447_normalize(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), TINY)


def write_package(report: dict[str, object]) -> None:
    neural_checkpoint = (
        PATHS.checkpoints
        / "v469_neural_sionna_calibrator"
        / "neural_sionna_calibrator.pt"
    )
    require(neural_checkpoint)
    dependencies = (
        "generate_v554_multiseed_linkwise_pdp_residual_candidate.py",
        "analyze_v554_conditional_linkwise_projection.py",
        "analyze_v554_multiseed_linkwise_pdp_calibration.py",
        "generate_v554_multiseed_linkwise_pdp_statistics.py",
        "generate_v549_multiseed_pas_residual_candidate.py",
        "analyze_v549_multiseed_pas_calibration.py",
        "generate_v549_multiseed_v207_pas_statistics.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v533_expanded_physical_sbi_action_match.py",
        "analyze_v531_expanded_sbi_configuration_library.py",
        "analyze_v529_sbi_configuration_posterior.py",
        "analyze_v494_sionna_configuration_stacking.py",
        "train_v469_neural_sionna_calibrator.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "analyze_v208_strict_historical_replay.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(neural_checkpoint, "model/neural_sionna_calibrator.pt")
        archive.write(REPORT, "model/v554_multiseed_linkwise_pdp_report.json")
        archive.write(
            CALIBRATION_REPORT,
            "model/v554_multiseed_linkwise_pdp_calibration.json",
        )
        archive.write(
            CONDITIONAL_REPORT,
            "model/v554_conditional_linkwise_projection.json",
        )
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    channel = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(channel[start : start + 5])).all()
        for start in range(0, len(channel), 5)
    )
    report["integrity"] = {
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "zip_bytes": int(ZIP_OUT.stat().st_size),
        "sha256": v533.digest(ZIP_OUT, "sha256"),
        "md5": v533.digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    for path in (BASE, CALIBRATION_REPORT, CONDITIONAL_REPORT, LINK_ACTION):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_report = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    conditional_report = json.loads(CONDITIONAL_REPORT.read_text(encoding="utf-8"))
    if not calibration_report["deployment_gate"]["pass"]:
        raise RuntimeError(
            "v554 calibration did not pass its predeclared deployment gate"
        )
    if not conditional_report["deployment_gate"]["pass"]:
        raise RuntimeError(
            "v554 conditional projection did not pass its deployment gate"
        )
    if (
        conditional_report["method_definition"]["projection_cycles"]
        != PROJECTION_CYCLES
    ):
        raise RuntimeError("v554 conditional report does not match projection cycles")
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    test_regimes = regime_of(geometry["test_blocked"].astype(np.float32)).astype("U8")
    generation = generate(test_regimes)
    base_replay = replay.score_prediction(BASE)
    candidate_replay = replay.score_prediction(OUTPUT)
    comparison = compare_channels(OUTPUT, BASE)
    selected = calibration_report["selected"]
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "A calibrated 24-realization Sionna expectation can correct which delay bins "
            "carry each link's energy at medium/heavy points while matrix balancing keeps "
            "the online-validated v549 global delay spectrum, link energy, and point power."
        ),
        "new_information": [
            "24 independent high-budget Sionna realizations for every 256x4 linkwise PDP",
            "per-link Monte Carlo variance and valid-realization coverage",
            "official-train cross-fit calibration of new versus inherited physical source",
            "same-location shuffled-source control",
            "matrix-balanced conditional link-delay allocation",
        ],
        "frozen_oof_evidence": {
            "unconstrained_calibration": selected,
            "conditional_projection": conditional_report["oof"],
        },
        "deployment": {
            "carrier": PATHS.relative(BASE),
            "mapping": (
                "apply the calibrated multi-seed / old-source differential action only "
                "to medium/heavy link-delay allocation, then balance rows and columns"
            ),
            "protected_objects": (
                "exact v549 per-link energy, point power, and global 192-bin delay mass; "
                "no direct Sionna phase or total-power injection"
            ),
            **generation,
        },
        "v207_replay_diagnostic": {
            "warning": "same-coordinate Sionna reference, not hidden online truth",
            "v549": base_replay,
            "candidate": candidate_replay,
            "traceback_delta": float(
                candidate_replay["score_traceback_leaked"]
                - base_replay["score_traceback_leaked"]
            ),
            "pdp_delta": float(candidate_replay["pdp"] - base_replay["pdp"]),
            "leaked_pas_delta": float(
                candidate_replay["leaked_sh"] - base_replay["leaked_sh"]
            ),
            "nmse_score_delta": float(
                candidate_replay["nmse_score"] - base_replay["nmse_score"]
            ),
        },
        "comparison_to_v549": comparison,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTIONS),
            "report": PATHS.relative(REPORT),
            "archive": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_package(report)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "archive": report["outputs"]["archive"],
                "oof_gain": selected["gain"],
                "oof_fold_wins": selected["fold_wins"],
                "real_minus_shuffled": selected["real_minus_shuffled"],
                "replay_delta": report["v207_replay_diagnostic"]["traceback_delta"],
                "comparison_to_v549": comparison,
                "sha256": report["integrity"]["sha256"],
                "md5": report["integrity"]["md5"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
