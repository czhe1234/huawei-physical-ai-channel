#!/usr/bin/env python3
"""Package the validated Round 2 Sionna spectral calibration model as V5."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from package_round2_v4_calibrated_exact_projection import (
    MODEL_FILES as V4_MODEL_FILES,
    SOURCE_FILES as V4_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V5"
METHOD = "sionna_shared_spectral_calibration"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_official_calibration"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sionna_official_calibration"
MULTISEED_ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sionna_multiseed"
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_V5_Sionna_Calibrated_Projection_Test_Channel.npy"
CALIBRATOR_REPORT = ANALYSIS_DIR / "calibrator_oof.json"
PROJECTION_REPORT = ANALYSIS_DIR / "calibrated_projection.json"
GENERATION_REPORT = ANALYSIS_DIR / "v5_generate.json"
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)

CALIBRATOR_FILES = tuple(
    (
        ARTIFACT_DIR / f"calibrator_{domain}_mean_fold{fold}.pt",
        f"model/calibrators/calibrator_{domain}_mean_fold{fold}.pt",
    )
    for domain in ("pas", "pdp")
    for fold in range(5)
)

REPORT_FILES = (
    (
        MULTISEED_ANALYSIS_DIR / "sionna_multiseed_generation.json",
        "model/reports/sionna_multiseed_generation.json",
    ),
    (
        MULTISEED_ANALYSIS_DIR / "sionna_multiseed_validation.json",
        "model/reports/sionna_multiseed_validation.json",
    ),
    (
        ANALYSIS_DIR / "calibration_dataset.json",
        "model/reports/calibration_dataset.json",
    ),
    (
        ANALYSIS_DIR / "calibrator_screen.json",
        "model/reports/calibrator_screen.json",
    ),
    (CALIBRATOR_REPORT, "model/reports/calibrator_oof.json"),
    (PROJECTION_REPORT, "model/reports/calibrated_projection.json"),
    (GENERATION_REPORT, "model/reports/v5_generate.json"),
)

V5_SOURCE_FILES = (
    (
        PATHS.round2_code / "evaluate_v3_anchored_full_targets.py",
        "code/round2/evaluate_v3_anchored_full_targets.py",
    ),
    (
        PATHS.round2_code / "package_round2_v4_calibrated_exact_projection.py",
        "code/round2/package_round2_v4_calibrated_exact_projection.py",
    ),
    (
        PATHS.round2_code / "generate_v5_sionna_spectral_library.py",
        "code/round2/generate_v5_sionna_spectral_library.py",
    ),
    (
        PATHS.round2_code / "generate_multiseed_sionna_dataset.py",
        "code/round2/generate_multiseed_sionna_dataset.py",
    ),
    (
        PATHS.round2_code / "validate_multiseed_sionna_dataset.py",
        "code/round2/validate_multiseed_sionna_dataset.py",
    ),
    (
        PATHS.round2_code / "prepare_sionna_calibration_dataset.py",
        "code/round2/prepare_sionna_calibration_dataset.py",
    ),
    (
        PATHS.round2_code / "train_sionna_spectral_calibrator.py",
        "code/round2/train_sionna_spectral_calibrator.py",
    ),
    (
        PATHS.round2_code / "evaluate_sionna_spectral_calibrator.py",
        "code/round2/evaluate_sionna_spectral_calibrator.py",
    ),
    (
        PATHS.round2_code / "generate_round2_v5_sionna_calibrated_projection.py",
        "code/round2/generate_round2_v5_sionna_calibrated_projection.py",
    ),
    (
        PATHS.round2_code
        / "package_round2_v5_sionna_shared_spectral_calibration.py",
        "code/round2/package_round2_v5_sionna_shared_spectral_calibration.py",
    ),
)

SOURCE_FILES = (*V4_SOURCE_FILES, *V5_SOURCE_FILES)
MODEL_FILES = (*V4_MODEL_FILES, *CALIBRATOR_FILES, *REPORT_FILES)


def validate_output() -> dict[str, object]:
    setup = json.loads((PATHS.round2 / "Round2_Setup.json").read_text(encoding="utf-8"))
    expected = (
        int(setup["P_Test"]),
        int(setup["M"]),
        int(setup["N"]),
        int(setup["S"]),
    )
    output = np.load(SOURCE_OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 8):
        block = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    if output.shape != expected:
        raise ValueError(f"output shape {output.shape} != {expected}")
    if output.dtype != np.complex64:
        raise ValueError(f"output dtype {output.dtype} != complex64")
    if not finite:
        raise ValueError("output contains non-finite values")
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(SOURCE_OUTPUT.stat().st_size),
    }


def main() -> None:
    required = (SOURCE_OUTPUT, *[path for path, _ in MODEL_FILES])
    for source in (*required, *[path for path, _ in SOURCE_FILES]):
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    calibration = json.loads(CALIBRATOR_REPORT.read_text(encoding="utf-8"))
    projection = json.loads(PROJECTION_REPORT.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_REPORT.read_text(encoding="utf-8"))
    identity = projection["identity"]
    fixed = projection["fixed_best_policy"]
    nested = projection["nested"]
    fixed_gain = float(fixed["overall"]["score"] - identity["score"])
    nested_gain = float(nested["overall"]["score"] - identity["score"])
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiment": "Round2-E40",
        "sionna_used": True,
        "absolute_sionna_phase_used": False,
        "official_train_h_supervision": True,
        "calibrator_mode": "mean",
        "fold_ensemble": 5,
        "checkpoint_count": len(CALIBRATOR_FILES),
        "local_validation": {
            "v4_identity": identity,
            "fixed_policy": fixed["overall"],
            "fixed_gain_vs_v4": fixed_gain,
            "fixed_positive_fold_count": int(
                sum(row["delta_score"] > 0.0 for row in fixed["folds"])
            ),
            "nested_policy": nested["overall"],
            "nested_gain_vs_v4": nested_gain,
            "nested_positive_fold_count": int(
                sum(row["delta_score"] > 0.0 for row in nested["folds"])
            ),
            "pointwise_oracle": projection["pointwise_oracle"]["overall"],
        },
        "deployment_policy": generation["model"],
        "calibrator_config": calibration["config"],
        "output_integrity": output_integrity,
        "reproduction": (
            "Run the V4 reproduction pipeline, generate and validate the Round2-E39 "
            "24-seed Sionna dataset, prepare the E40 calibration dataset, train the five "
            "PAS and five PDP fold models, and run "
            "generate_round2_v5_sionna_calibrated_projection.py."
        ),
    }

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {ARCHIVE}")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(PATHS.root / ".hw-project", ".hw-project")
        archive.write(SOURCE_OUTPUT, "Round2_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.writestr("model/calibrators/", b"")
        archive.writestr("model/reports/", b"")
        for source, destination in MODEL_FILES:
            archive.write(source, destination)
        archive.writestr(
            "model/V5_VERSION.json",
            json.dumps(package_metadata, indent=2, ensure_ascii=False).encode("utf-8")
            + b"\n",
        )
        archive.writestr(
            "model/INFERENCE.txt",
            (package_metadata["reproduction"] + "\n").encode("ascii"),
        )
        for source, destination in SOURCE_FILES:
            archive.write(source, destination)

    expected_members = {
        "Round2_Test_Channel.npy",
        "model/full_epoch6.pt",
        "model/test_corrected_weights.npz",
        "model/full_group_calibration_model.npz",
        "model/calibrated_exact_projection_selection.json",
        "model/V5_VERSION.json",
        *[destination for _, destination in CALIBRATOR_FILES],
    }
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
        compression_types = {member.compress_type for member in archive.infolist()}
        missing = sorted(expected_members.difference(members))
        with archive.open("Round2_Test_Channel.npy") as handle:
            version = np.lib.format.read_magic(handle)
            packaged = np.lib.format._read_array_header(handle, version)
    if bad_member is not None:
        raise RuntimeError(f"ZIP CRC failure: {bad_member}")
    if missing:
        raise RuntimeError(f"ZIP is missing required members: {missing}")
    if members.count("Round2_Test_Channel.npy") != 1:
        raise RuntimeError("ZIP must contain one Round2_Test_Channel.npy at root")
    if compression_types != {zipfile.ZIP_STORED}:
        raise RuntimeError(f"unexpected ZIP compression types: {compression_types}")
    packaged_shape, _, packaged_dtype = packaged
    if tuple(packaged_shape) != tuple(output_integrity["shape"]):
        raise RuntimeError(f"packaged NPY shape mismatch: {packaged_shape}")
    if np.dtype(packaged_dtype) != np.dtype(np.complex64):
        raise RuntimeError(f"packaged NPY dtype mismatch: {packaged_dtype}")

    package_metadata["zip_integrity"] = {
        "crc": True,
        "compression": "ZIP_STORED",
        "bytes": int(ARCHIVE.stat().st_size),
        "member_count": len(members),
        "required_members_present": True,
        "checkpoint_count": len(CALIBRATOR_FILES),
        "members": members,
    }
    VERSION_REPORT.write_text(
        json.dumps(package_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(package_metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
