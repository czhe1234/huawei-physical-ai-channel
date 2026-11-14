#!/usr/bin/env python3
"""Package the E86 surface endpoint blend as Round 2 V12."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from package_round2_v8_full_sionna_migration import unique_files
from package_round2_v11_reliable_sqrt_pas_gate import (
    MODEL_FILES as V11_MODEL_FILES,
    SOURCE_FILES as V11_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V12"
METHOD = "surface_endpoint_v11_blend"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "surface_interaction_energy_model"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_surface_interaction_energy_model"
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_E86_Surface_Endpoint_V11_Blend_Test_Channel.npy"
LIGHT_ENDPOINT = ARTIFACT_DIR / "e86_test_geometry_sequence_endpoint_light.npz"
FROZEN_REPORT = ANALYSIS_DIR / "e86_frozen_policy.json"
TEST_REPORT = ANALYSIS_DIR / "e86_test_generate.json"
E85_REPORT = ANALYSIS_DIR / "surface_interaction_strict_holes.json"
E86_REPORT = ANALYSIS_DIR / "e86_surface_endpoint_v11_blend.json"
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)


E86_SOURCE_FILES = (
    (PATHS.round2_code / "analyze_path_candidate_information_audit.py", "code/round2/analyze_path_candidate_information_audit.py"),
    (PATHS.round2_code / "analyze_v4_exact_joint_projection.py", "code/round2/analyze_v4_exact_joint_projection.py"),
    (PATHS.round2_code / "analyze_v4_path_track_spectral_targets.py", "code/round2/analyze_v4_path_track_spectral_targets.py"),
    (PATHS.round2_code / "train_surface_interaction_energy_model.py", "code/round2/train_surface_interaction_energy_model.py"),
    (PATHS.round2_code / "analyze_e86_surface_endpoint_v11_blend.py", "code/round2/analyze_e86_surface_endpoint_v11_blend.py"),
    (PATHS.round2_code / "evaluate_e86_frozen_policy.py", "code/round2/evaluate_e86_frozen_policy.py"),
    (PATHS.round2_code / "generate_e86_surface_endpoint_test_channel.py", "code/round2/generate_e86_surface_endpoint_test_channel.py"),
    (PATHS.round2_code / "package_round2_v12_surface_endpoint_v11_blend.py", "code/round2/package_round2_v12_surface_endpoint_v11_blend.py"),
)
E86_MODEL_FILES = tuple(
    (
        ARTIFACT_DIR / f"strict_fold{fold}_geometry_sequence.pt",
        f"model/surface_interaction/strict_fold{fold}_geometry_sequence.pt",
    )
    for fold in range(5)
)
E86_REPORT_FILES = (
    (E85_REPORT, "model/reports/surface_interaction_strict_holes.json"),
    (E86_REPORT, "model/reports/e86_surface_endpoint_v11_blend.json"),
    (FROZEN_REPORT, "model/reports/e86_frozen_policy.json"),
    (TEST_REPORT, "model/reports/e86_test_generate.json"),
    (LIGHT_ENDPOINT, "model/surface_interaction/e86_test_geometry_sequence_endpoint_light.npz"),
)
SOURCE_FILES = unique_files((*V11_SOURCE_FILES, *E86_SOURCE_FILES))
MODEL_FILES = unique_files((*V11_MODEL_FILES, *E86_MODEL_FILES, *E86_REPORT_FILES))


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
    endpoint = np.load(LIGHT_ENDPOINT, allow_pickle=False)
    expected_endpoint = {
        "mass": (expected[0], 128),
        "coordinate": (expected[0], 128, 3),
        "port": (expected[0], 128, 2, 4),
    }
    for name, shape in expected_endpoint.items():
        if endpoint[name].shape != shape or not np.isfinite(endpoint[name]).all():
            raise ValueError(f"invalid endpoint {name}: {endpoint[name].shape}")
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(SOURCE_OUTPUT.stat().st_size),
        "endpoint_shapes": {name: list(value.shape) for name, value in endpoint.items()},
    }


def main() -> None:
    required = (
        SOURCE_OUTPUT,
        LIGHT_ENDPOINT,
        FROZEN_REPORT,
        TEST_REPORT,
        E85_REPORT,
        E86_REPORT,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    )
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    frozen = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    generation = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    geometry = frozen["endpoint_results"]["geometry_sequence"]
    fold_deltas = [row["delta"] for row in geometry["folds"]]
    frozen_gate = bool(
        geometry["delta"] >= 0.005
        and len(fold_deltas) == 5
        and min(fold_deltas) > 0.0
    )
    if not frozen_gate:
        raise RuntimeError(f"E86 frozen policy gate failed: {geometry}")

    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiments": ["Round2-E84", "Round2-E85", "Round2-E86"],
        "sionna_used": True,
        "absolute_sionna_phase_used": False,
        "primary_hypothesis": (
            "A strict geometry-sequence model can provide complementary noncoherent "
            "PAS/PDP targets; injecting them weakly into the V11 complex carrier "
            "improves spectral structure without replacing phase or total power."
        ),
        "local_validation": {
            "v11_baseline": frozen["baseline"],
            "geometry_sequence_fixed": geometry["summary"],
            "delta": geometry["delta"],
            "fold_deltas": fold_deltas,
            "no_sequence_control": frozen["endpoint_results"]["no_sequence"],
            "frozen_gate_passed": frozen_gate,
        },
        "deployment_policy": generation["policy"],
        "output_integrity": output_integrity,
        "reproduction": (
            "Use the five strict spatial-hole surface-interaction checkpoints to predict "
            "test path mass, delay-angle coordinates and port energy. Render only their "
            "noncoherent PAS/PDP endpoint, blend each with V11 PAS/PDP in square-root "
            "space at 0.20, apply four PAS-to-PDP projections, and preserve V11 point power."
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
        archive.writestr("model/surface_interaction/", b"")
        archive.writestr("model/reports/", b"")
        for source, destination in MODEL_FILES:
            archive.write(source, destination)
        archive.writestr(
            "model/V12_VERSION.json",
            json.dumps(package_metadata, indent=2, ensure_ascii=False).encode("utf-8") + b"\n",
        )
        archive.writestr(
            "model/INFERENCE.txt",
            (package_metadata["reproduction"] + "\n").encode("ascii"),
        )
        for source, destination in SOURCE_FILES:
            archive.write(source, destination)

    expected_members = {
        "Round2_Test_Channel.npy",
        "model/V12_VERSION.json",
        "model/surface_interaction/strict_fold0_geometry_sequence.pt",
        "model/surface_interaction/strict_fold4_geometry_sequence.pt",
        "model/surface_interaction/e86_test_geometry_sequence_endpoint_light.npz",
        "model/reports/e86_frozen_policy.json",
        "code/round2/generate_e86_surface_endpoint_test_channel.py",
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
