#!/usr/bin/env python3
"""Inject the calibrated multi-seed PAS residual into the v540 carrier."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as replay
import analyze_v549_multiseed_pas_calibration as analysis
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v533_expanded_physical_sbi_action_match as v533
from project_paths import PATHS


VERSION_NUMBER = 549
VERSION = "v549_multiseed_pas_residual_pdp_nmse_lock"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v549_Round1_Test_Channel.npy"
ACTIONS = CACHE_DIR / "v549_multiseed_pas_residual_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
BASE = PATHS.cache_dir(540) / "v540_Round1_Test_Channel.npy"
CALIBRATION_REPORT = analysis.REPORT
PAS_CYCLES = 4
BATCH = 2
ACTION_SCALE = np.float32(1.0)
FLOOR = 1e-12
TINY = 1e-30


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    output /= np.maximum(
        np.sum(output, axis=axes, keepdims=True, dtype=np.float64), TINY
    )
    return output.astype(np.float32)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def row_l1(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(left.reshape(len(left), -1) - right.reshape(len(right), -1)),
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)


def action_paths(checkpoint: int) -> dict[str, Path]:
    root = analysis.OUT_DIR / "test_targets"
    return {
        name: root / f"n{checkpoint:02d}_test_{name}_log_action.npy"
        for name in analysis.source.MARGINALS
    }


def desired_pas(
    current: dict[str, np.ndarray],
    actions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output = {}
    for name in current:
        axes = (1, 2) if name == "sh" else (1, 2, 3)
        output[name] = normalize(
            current[name] * np.exp(ACTION_SCALE * actions[name]).astype(np.float32),
            axes,
        )
    return output


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
            np.sum(
                np.abs(local_left) ** 2,
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            * local_energy
        )
        coherence += float(np.sum(np.abs(inner) / np.maximum(norm, TINY)))
        count += end - start
    return {
        "nmse": float(error / max(energy, TINY)),
        "complex_cosine": float(coherence / max(count, 1)),
    }


def generate(checkpoint: int) -> dict[str, object]:
    source_channel = np.load(BASE, mmap_mode="r")
    paths = action_paths(checkpoint)
    for path in paths.values():
        require(path)
    action_arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source_channel.shape
    )
    diagnostics = {
        "pdp_lock_error": np.empty(len(source_channel), dtype=np.float32),
        "power_ratio": np.empty(len(source_channel), dtype=np.float32),
        "phase_agreement": np.empty(len(source_channel), dtype=np.float32),
        **{
            f"{name}_action_l1": np.empty(len(source_channel), dtype=np.float32)
            for name in analysis.source.MARGINALS
        },
        **{
            f"{name}_requested_l1": np.empty(len(source_channel), dtype=np.float32)
            for name in analysis.source.MARGINALS
        },
    }

    for start in range(0, len(source_channel), BATCH):
        end = min(start + BATCH, len(source_channel))
        base_h = np.asarray(source_channel[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        current = v423.leaked_marginals(base_support)
        local_actions = {
            name: np.asarray(value[start:end], dtype=np.float32)
            for name, value in action_arrays.items()
        }
        desired = desired_pas(current, local_actions)
        candidate = base_h.copy()
        for _ in range(PAS_CYCLES):
            candidate = v423.project_pas(candidate, desired)
            candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"non-finite v549 block {start}:{end}")
        output[start:end] = candidate
        actual_support, _ = v423.leaked_support(candidate)
        actual = v423.leaked_marginals(actual_support)
        for name in analysis.source.MARGINALS:
            diagnostics[f"{name}_action_l1"][start:end] = row_l1(
                actual[name], current[name]
            )
            diagnostics[f"{name}_requested_l1"][start:end] = row_l1(
                desired[name], current[name]
            )
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
        diagnostics["phase_agreement"][start:end] = v423.phase_agreement(
            base_h, candidate
        )
        if end == len(source_channel) or end % 20 == 0:
            output.flush()
            print(
                f"v549 PAS residual projection {end}/{len(source_channel)}", flush=True
            )
    output.flush()
    np.savez_compressed(ACTIONS, checkpoint=checkpoint, **diagnostics)
    return {
        "checkpoint": checkpoint,
        "action_scale": float(ACTION_SCALE),
        "pas_cycles": PAS_CYCLES,
        "diagnostics": {
            name: summarize(values) for name, values in diagnostics.items()
        },
    }


def write_package(report: dict[str, object]) -> None:
    neural_checkpoint = (
        PATHS.checkpoints
        / "v469_neural_sionna_calibrator"
        / "neural_sionna_calibrator.pt"
    )
    require(neural_checkpoint)
    dependencies = (
        "generate_v549_multiseed_pas_residual_candidate.py",
        "analyze_v549_multiseed_pas_calibration.py",
        "generate_v549_multiseed_v207_pas_statistics.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v533_expanded_physical_sbi_action_match.py",
        "analyze_v531_expanded_sbi_configuration_library.py",
        "analyze_v529_sbi_configuration_posterior.py",
        "analyze_v494_sionna_configuration_stacking.py",
        "train_v469_neural_sionna_calibrator.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
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
        archive.write(REPORT, "model/v549_multiseed_pas_report.json")
        archive.write(
            CALIBRATION_REPORT,
            "model/v549_multiseed_pas_calibration.json",
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
    for path in (BASE, CALIBRATION_REPORT):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_report = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    selected = calibration_report["selected"]
    checkpoint = int(selected["checkpoint"])
    generation = generate(checkpoint)
    base_replay = replay.score_prediction(BASE)
    candidate_replay = replay.score_prediction(OUTPUT)
    comparison = compare_channels(OUTPUT, BASE)
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The official-train-calibrated residual of a multi-realization Sionna PAS "
            "expectation adds same-coordinate path information to v540 without replacing "
            "its stronger carrier, linkwise PDP, point power, or CIR magnitudes."
        ),
        "new_information": [
            "independent high-budget Sionna PAS Monte Carlo expectation",
            "cellwise Monte Carlo variance used as deployment confidence",
            "official-train spatial-OOF log-ratio calibration",
            "four scorer-shaped PAS marginal residuals",
        ],
        "frozen_oof_evidence": {
            "checkpoint": checkpoint,
            "composite": selected["composite"],
            "gain": selected["gain"],
            "fold_wins": selected["fold_wins"],
            "by_regime": selected["by_regime"],
            "quality_scale": selected["quality_scale"],
            "alpha": selected["alpha"],
        },
        "deployment": {
            "carrier": PATHS.relative(BASE),
            "mapping": (
                "apply calibrated-target / official-KNN-baseline log-ratio to the "
                "current v540 PAS marginals"
            ),
            "pdp_and_power_lock": (
                "restore all v540 256x4x192 CIR magnitudes after every PAS cycle"
            ),
            **generation,
        },
        "v207_replay_diagnostic": {
            "warning": "same-coordinate Sionna reference, not hidden online truth",
            "v540": base_replay,
            "candidate": candidate_replay,
            "traceback_delta": float(
                candidate_replay["score_traceback_leaked"]
                - base_replay["score_traceback_leaked"]
            ),
            "nmse_score_delta": float(
                candidate_replay["nmse_score"] - base_replay["nmse_score"]
            ),
        },
        "comparison_to_v540": comparison,
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
                "checkpoint": checkpoint,
                "oof_gain": selected["gain"],
                "replay_delta": report["v207_replay_diagnostic"]["traceback_delta"],
                "nmse_score_delta": report["v207_replay_diagnostic"][
                    "nmse_score_delta"
                ],
                "comparison_to_v540": comparison,
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
