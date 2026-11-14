#!/usr/bin/env python3
"""Package the V7 carrier plus calibrated Sionna spectral migration as Round 2 V8."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from package_round2_v5_sionna_shared_spectral_calibration import (
    CALIBRATOR_FILES,
    REPORT_FILES as V5_REPORT_FILES,
    SOURCE_FILES as V5_SOURCE_FILES,
)
from package_round2_v7_complex_carrier_lifecycle import (
    MODEL_FILES as V7_MODEL_FILES,
    SOURCE_FILES as V7_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V8"
METHOD = "full_sionna_migration"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_sionna_migration"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_sionna_migration"
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_V8_Full_Sionna_Migration_Test_Channel.npy"
ANALYSIS_REPORT = ANALYSIS_DIR / "full_sionna_migration.json"
GENERATION_REPORT = ANALYSIS_DIR / "v8_generate.json"
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)


def unique_files(files):
    result = []
    destinations: set[str] = set()
    for source, destination in files:
        if destination in destinations:
            continue
        destinations.add(destination)
        result.append((source, destination))
    return tuple(result)


V8_SOURCE_FILES = (
    (
        PATHS.round2_code / "run_v8_full_sionna_migration.py",
        "code/round2/run_v8_full_sionna_migration.py",
    ),
    (
        PATHS.round2_code / "generate_round2_v8_full_sionna_migration.py",
        "code/round2/generate_round2_v8_full_sionna_migration.py",
    ),
    (
        PATHS.round2_code / "package_round2_v8_full_sionna_migration.py",
        "code/round2/package_round2_v8_full_sionna_migration.py",
    ),
)
V8_REPORT_FILES = (
    (ANALYSIS_REPORT, "model/reports/full_sionna_migration.json"),
    (GENERATION_REPORT, "model/reports/v8_generate.json"),
)
SOURCE_FILES = unique_files((*V7_SOURCE_FILES, *V5_SOURCE_FILES, *V8_SOURCE_FILES))
MODEL_FILES = unique_files(
    (*V7_MODEL_FILES, *CALIBRATOR_FILES, *V5_REPORT_FILES, *V8_REPORT_FILES)
)


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
        finite &= bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
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
    required = (
        SOURCE_OUTPUT,
        ANALYSIS_REPORT,
        GENERATION_REPORT,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    )
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    analysis = json.loads(ANALYSIS_REPORT.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_REPORT.read_text(encoding="utf-8"))
    identity = analysis["analysis"]["identity"]
    fixed = analysis["analysis"]["fixed_best"]
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiment": "Round2-E56",
        "sionna_used": True,
        "absolute_sionna_phase_used": False,
        "primary_hypothesis": (
            "The validated V7 complex carrier should retain its phase and power while "
            "calibrated Sionna PAS/PDP supply the missing path-energy structure."
        ),
        "local_validation": {
            "v7_identity": identity,
            "fixed_policy": fixed,
            "gain_vs_v7": float(fixed["score"] - identity["score"]),
            "pointwise_oracle": analysis["analysis"]["pointwise_oracle"]["overall"],
        },
        "deployment_policy": generation["policy"],
        "output_integrity": output_integrity,
        "output_delta_vs_v7": generation["output"],
        "reproduction": (
            "Run the V7 official-data carrier pipeline and the 24-seed Sionna spectral "
            "calibration pipeline, then apply the V8 PAS/PDP joint projection."
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
            "model/V8_VERSION.json",
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
        "model/full_complex_stack_weights.npz",
        "model/stacked_lifecycle_selection.json",
        "model/calibrators/calibrator_pas_mean_fold0.pt",
        "model/calibrators/calibrator_pdp_mean_fold4.pt",
        "model/V8_VERSION.json",
        "code/round2/generate_round2_v8_full_sionna_migration.py",
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
        "members": members,
    }
    VERSION_REPORT.write_text(
        json.dumps(package_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(package_metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
