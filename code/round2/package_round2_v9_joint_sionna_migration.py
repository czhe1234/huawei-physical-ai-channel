#!/usr/bin/env python3
"""Package the point-joint Sionna migration as Round 2 V9."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from package_round2_v8_full_sionna_migration import (
    MODEL_FILES as V8_MODEL_FILES,
    SOURCE_FILES as V8_SOURCE_FILES,
    unique_files,
)
from project_paths import PATHS


VERSION = "V9"
METHOD = "joint_sionna_migration"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v9_joint_sionna_migration"
ENDPOINT_DIR = PATHS.artifacts / "round2" / "v9_joint_sionna_endpoints"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v9_joint_sionna_migration"
ENDPOINT_ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_v9_joint_sionna_endpoints"
)
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_V9_Joint_Sionna_Migration_Test_Channel.npy"
REFINE_REPORT = ANALYSIS_DIR / "joint_migration_refine.json"
GENERATION_REPORT = ANALYSIS_DIR / "v9_generate.json"
ENDPOINT_OOF_REPORT = ENDPOINT_ANALYSIS_DIR / "joint_endpoint_oof.json"
ENDPOINT_TEST_REPORT = ENDPOINT_ANALYSIS_DIR / "joint_endpoint_test.json"
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)


V9_SOURCE_FILES = (
    (
        PATHS.round2_code / "train_v9_joint_sionna_endpoints.py",
        "code/round2/train_v9_joint_sionna_endpoints.py",
    ),
    (
        PATHS.round2_code / "run_v9_joint_sionna_migration.py",
        "code/round2/run_v9_joint_sionna_migration.py",
    ),
    (
        PATHS.round2_code / "generate_round2_v9_joint_sionna_migration.py",
        "code/round2/generate_round2_v9_joint_sionna_migration.py",
    ),
    (
        PATHS.round2_code / "package_round2_v9_joint_sionna_migration.py",
        "code/round2/package_round2_v9_joint_sionna_migration.py",
    ),
)
JOINT_CALIBRATORS = tuple(
    (
        ENDPOINT_DIR / f"joint_calibrator_{domain}_fold{fold}.pt",
        f"model/calibrators/joint_calibrator_{domain}_fold{fold}.pt",
    )
    for domain in ("pas", "pdp")
    for fold in range(5)
)
V9_REPORT_FILES = (
    (REFINE_REPORT, "model/reports/joint_migration_refine.json"),
    (GENERATION_REPORT, "model/reports/v9_generate.json"),
    (ENDPOINT_OOF_REPORT, "model/reports/joint_endpoint_oof.json"),
    (ENDPOINT_TEST_REPORT, "model/reports/joint_endpoint_test.json"),
)
SOURCE_FILES = unique_files((*V8_SOURCE_FILES, *V9_SOURCE_FILES))
MODEL_FILES = unique_files((*V8_MODEL_FILES, *JOINT_CALIBRATORS, *V9_REPORT_FILES))


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
        REFINE_REPORT,
        GENERATION_REPORT,
        ENDPOINT_OOF_REPORT,
        ENDPOINT_TEST_REPORT,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    )
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    refine = json.loads(REFINE_REPORT.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_REPORT.read_text(encoding="utf-8"))
    selected = refine["analysis"]["selected"]
    v8_reference = refine["analysis"]["v8_reference"]
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiments": ["Round2-E57", "Round2-E58"],
        "sionna_used": True,
        "absolute_sionna_phase_used": False,
        "primary_hypothesis": (
            "Point-joint calibration recovers cross-UE, cross-frequency, cross-link, "
            "and angle-delay evidence discarded by independent spectral slices."
        ),
        "local_validation": {
            "v8_reference": v8_reference,
            "selected_policy": selected,
            "gain_vs_v8": float(selected["score"] - v8_reference["score"]),
        },
        "deployment_policy": generation["policy"],
        "output_integrity": output_integrity,
        "output_diagnostics": generation["output"],
        "reproduction": (
            "Run the V7 carrier and E40 spectral calibration, train five joint PAS and "
            "five joint PDP calibrators, then apply the frozen V9 endpoint blend and "
            "four-cycle PAS-to-PDP projection."
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
            "model/V9_VERSION.json",
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
        "model/calibrators/calibrator_pas_mean_fold0.pt",
        "model/calibrators/joint_calibrator_pas_fold0.pt",
        "model/calibrators/joint_calibrator_pdp_fold4.pt",
        "model/V9_VERSION.json",
        "code/round2/generate_round2_v9_joint_sionna_migration.py",
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
