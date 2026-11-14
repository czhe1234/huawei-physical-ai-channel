#!/usr/bin/env python3
"""Package the full-data explicit-path PDP link gate as Round 2 V10."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from package_round2_v8_full_sionna_migration import unique_files
from package_round2_v9_joint_sionna_migration import (
    MODEL_FILES as V9_MODEL_FILES,
    SOURCE_FILES as V9_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V10"
METHOD = "full_path_pdp_link_gate"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_path_pdp_link_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_path_pdp_link_gate"
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_V10_Full_Path_PDP_Link_Gate_Test_Channel.npy"
TRAIN_REPORT = ANALYSIS_DIR / "full_path_pdp_link_gate_train.json"
OOF_REPORT = ANALYSIS_DIR / "full_path_pdp_link_gate_v9_oof.json"
CALIBRATOR_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_v9_path_link_action_calibrator"
    / "v9_path_link_action_calibrator.json"
)
GENERATION_REPORT = ANALYSIS_DIR / "v10_generate.json"
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)


V10_SOURCE_FILES = (
    (
        PATHS.round2_code / "analyze_joint_path_token_oracle.py",
        "code/round2/analyze_joint_path_token_oracle.py",
    ),
    (
        PATHS.round2_code / "build_explicit_path_token_cache.py",
        "code/round2/build_explicit_path_token_cache.py",
    ),
    (
        PATHS.round2_code / "train_explicit_path_pdp_predictor.py",
        "code/round2/train_explicit_path_pdp_predictor.py",
    ),
    (
        PATHS.round2_code / "build_explicit_path_model_features.py",
        "code/round2/build_explicit_path_model_features.py",
    ),
    (
        PATHS.round2_code / "train_path_pdp_link_gate.py",
        "code/round2/train_path_pdp_link_gate.py",
    ),
    (
        PATHS.round2_code / "train_full_path_pdp_link_gate.py",
        "code/round2/train_full_path_pdp_link_gate.py",
    ),
    (
        PATHS.round2_code / "generate_round2_v10_full_path_pdp_link_gate.py",
        "code/round2/generate_round2_v10_full_path_pdp_link_gate.py",
    ),
    (
        PATHS.round2_code / "package_round2_v10_full_path_pdp_link_gate.py",
        "code/round2/package_round2_v10_full_path_pdp_link_gate.py",
    ),
)
GATE_MODELS = tuple(
    (
        ARTIFACT_DIR / f"fold{fold}_full_link_gate.pt",
        f"model/path_link_gate/fold{fold}_full_link_gate.pt",
    )
    for fold in range(5)
)
V10_REPORT_FILES = (
    (TRAIN_REPORT, "model/reports/full_path_pdp_link_gate_train.json"),
    (OOF_REPORT, "model/reports/full_path_pdp_link_gate_v9_oof.json"),
    (CALIBRATOR_REPORT, "model/reports/v9_path_link_action_calibrator.json"),
    (GENERATION_REPORT, "model/reports/v10_generate.json"),
)
SOURCE_FILES = unique_files((*V9_SOURCE_FILES, *V10_SOURCE_FILES))
MODEL_FILES = unique_files((*V9_MODEL_FILES, *GATE_MODELS, *V10_REPORT_FILES))


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
        TRAIN_REPORT,
        OOF_REPORT,
        CALIBRATOR_REPORT,
        GENERATION_REPORT,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    )
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    train_report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    oof_report = json.loads(OOF_REPORT.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_REPORT.read_text(encoding="utf-8"))
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiments": [
            "Round2-E60",
            "Round2-E61",
            "Round2-E62",
            "Round2-E63",
            "Round2-E64",
            "Round2-E65",
            "Round2-E66",
            "Round2-E67",
        ],
        "sionna_used": True,
        "absolute_sionna_phase_used": False,
        "primary_hypothesis": (
            "Explicit multiseed paths contain link-specific delay-energy information "
            "missing from V9, and a gate trained on all 4000 official labels can select "
            "where that physical PDP should replace the RBF/V9 PDP prior."
        ),
        "local_validation": {
            "full_4000_link_pdp": train_report["selected"],
            "v9_oof_baseline": oof_report["baseline"],
            "v9_oof_selected": oof_report["selected"],
            "v9_oof_fold_deltas": [
                row["delta"] for row in oof_report["fold_results"]
            ],
        },
        "deployment_policy": generation["policy"],
        "gate_ensemble": generation["gate_ensemble"],
        "output_integrity": output_integrity,
        "output_diagnostics": generation["output"],
        "reproduction": (
            "Generate the 24-seed base/scatter explicit path cache, build path-token "
            "features, train five full-data link gates, reproduce V9, and apply the "
            "frozen scale-0.50 two-cycle PAS-to-PDP projection."
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
        archive.writestr("model/path_link_gate/", b"")
        archive.writestr("model/reports/", b"")
        for source, destination in MODEL_FILES:
            archive.write(source, destination)
        archive.writestr(
            "model/V10_VERSION.json",
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
        "model/calibrators/joint_calibrator_pas_fold0.pt",
        "model/path_link_gate/fold0_full_link_gate.pt",
        "model/path_link_gate/fold4_full_link_gate.pt",
        "model/V10_VERSION.json",
        "code/round2/generate_round2_v10_full_path_pdp_link_gate.py",
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
