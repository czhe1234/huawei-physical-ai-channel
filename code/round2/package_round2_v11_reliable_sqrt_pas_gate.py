#!/usr/bin/env python3
"""Package reliability-gated square-root PAS consensus as Round 2 V11."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from package_round2_v8_full_sionna_migration import unique_files
from package_round2_v10_full_path_pdp_link_gate import (
    MODEL_FILES as V10_MODEL_FILES,
    SOURCE_FILES as V10_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V11"
METHOD = "reliable_sqrt_pas_consensus"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "reliable_sqrt_pas_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_reliable_sqrt_pas_gate"
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_V11_Reliable_Sqrt_PAS_Consensus_Test_Channel.npy"
E68_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_explicit_path_pas_joint_gate"
    / "explicit_path_pas_upper_bound.json"
)
E75_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_sqrt_pas_source_consensus"
    / "sqrt_pas_source_consensus.json"
)
E76_REPORT = ANALYSIS_DIR / "reliable_sqrt_pas_gate.json"
GENERATION_REPORT = ANALYSIS_DIR / "v11_generate.json"
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)


V11_SOURCE_FILES = (
    (
        PATHS.round2_code / "analyze_v11_explicit_path_pas_upper_bound.py",
        "code/round2/analyze_v11_explicit_path_pas_upper_bound.py",
    ),
    (
        PATHS.round2_code / "analyze_v11_low_rank_path_pas_residual.py",
        "code/round2/analyze_v11_low_rank_path_pas_residual.py",
    ),
    (
        PATHS.round2_code / "analyze_v11_sqrt_pas_source_consensus.py",
        "code/round2/analyze_v11_sqrt_pas_source_consensus.py",
    ),
    (
        PATHS.round2_code / "analyze_v11_reliable_sqrt_pas_gate.py",
        "code/round2/analyze_v11_reliable_sqrt_pas_gate.py",
    ),
    (
        PATHS.round2_code / "train_full_path_pas_slice_gate.py",
        "code/round2/train_full_path_pas_slice_gate.py",
    ),
    (
        PATHS.round2_code / "generate_round2_v11_reliable_sqrt_pas_gate.py",
        "code/round2/generate_round2_v11_reliable_sqrt_pas_gate.py",
    ),
    (
        PATHS.round2_code / "package_round2_v11_reliable_sqrt_pas_gate.py",
        "code/round2/package_round2_v11_reliable_sqrt_pas_gate.py",
    ),
)
V11_REPORT_FILES = (
    (E68_REPORT, "model/reports/explicit_path_pas_upper_bound.json"),
    (E75_REPORT, "model/reports/sqrt_pas_source_consensus.json"),
    (E76_REPORT, "model/reports/reliable_sqrt_pas_gate.json"),
    (GENERATION_REPORT, "model/reports/v11_generate.json"),
)
SOURCE_FILES = unique_files((*V10_SOURCE_FILES, *V11_SOURCE_FILES))
MODEL_FILES = unique_files((*V10_MODEL_FILES, *V11_REPORT_FILES))


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
        E68_REPORT,
        E75_REPORT,
        E76_REPORT,
        GENERATION_REPORT,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    )
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    evidence = json.loads(E76_REPORT.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_REPORT.read_text(encoding="utf-8"))
    if not evidence["deployment_gate"]["passed"]:
        raise RuntimeError("E76 deployment gate did not pass")
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiments": ["Round2-E68", "Round2-E75", "Round2-E76"],
        "sionna_used": True,
        "absolute_sionna_phase_used": False,
        "primary_hypothesis": (
            "Multiple independently calibrated Sionna PAS endpoints expose a reliable "
            "consensus direction when interpolation is performed in square-root "
            "probability space and high-disagreement slices are excluded."
        ),
        "local_validation": {
            "v10_baseline": evidence["selected"]["baseline"],
            "selected": evidence["selected"]["selected"],
            "delta": evidence["selected"]["delta"],
            "fold_deltas": [
                row["delta"] for row in evidence["selected"]["folds"]
            ],
            "nested_selection": evidence["nested_selection"],
            "power_geometry_control": evidence["power_geometry_control"],
        },
        "deployment_policy": generation["policy"],
        "test_gate": generation["gate"],
        "output_integrity": output_integrity,
        "output_diagnostics": generation["output"],
        "reproduction": (
            "Reproduce V10 and four PAS endpoints, form their square-root spectral "
            "consensus, retain low-disagreement UE-frequency slices, and apply the "
            "frozen four-cycle PDP-to-PAS projection while preserving point power."
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
            "model/V11_VERSION.json",
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
        "model/reports/reliable_sqrt_pas_gate.json",
        "model/V11_VERSION.json",
        "code/round2/generate_round2_v11_reliable_sqrt_pas_gate.py",
        "code/round2/package_round2_v11_reliable_sqrt_pas_gate.py",
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
