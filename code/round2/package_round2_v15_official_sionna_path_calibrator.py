#!/usr/bin/env python3
"""Package the fivefold E190 official-path calibrator as Round 2 V15."""

from __future__ import annotations

import json
import zipfile

import numpy as np

import generate_e190_official_sionna_path_calibrator_test as e190_test
import train_e190_official_sionna_path_calibrator as e190
from package_round2_v8_full_sionna_migration import unique_files
from package_round2_v11_reliable_sqrt_pas_gate import (
    MODEL_FILES as V11_MODEL_FILES,
    SOURCE_FILES as V11_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V15"
METHOD = "official_sionna_path_calibrator"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
SOURCE_OUTPUT = e190_test.OUTPUT
TEST_ENDPOINT = e190_test.ENDPOINT
TEST_REPORT = e190_test.REPORT
LOCAL_REPORT = e190.FULL_REPORT
VERSION_REPORT = (
    PATHS.analysis / "results" / "version_reports" / VERSION / f"{VERSION}_{METHOD}.json"
)


E190_SOURCE_FILES = (
    (PATHS.round2_code / "analyze_global_local_path_lifecycle.py", "code/round2/analyze_global_local_path_lifecycle.py"),
    (PATHS.round2_code / "analyze_path_candidate_information_audit.py", "code/round2/analyze_path_candidate_information_audit.py"),
    (PATHS.round2_code / "analyze_e100_locked_energy_angle_correction.py", "code/round2/analyze_e100_locked_energy_angle_correction.py"),
    (PATHS.round2_code / "analyze_e86_surface_endpoint_v11_blend.py", "code/round2/analyze_e86_surface_endpoint_v11_blend.py"),
    (PATHS.round2_code / "analyze_v4_path_track_spectral_targets.py", "code/round2/analyze_v4_path_track_spectral_targets.py"),
    (PATHS.round2_code / "generate_e86_surface_endpoint_test_channel.py", "code/round2/generate_e86_surface_endpoint_test_channel.py"),
    (PATHS.round2_code / "optimize_e98_global_array_angle_mapping.py", "code/round2/optimize_e98_global_array_angle_mapping.py"),
    (PATHS.round2_code / "run_path_component_transport.py", "code/round2/run_path_component_transport.py"),
    (PATHS.round2_code / "train_e99_corrected_array_surface_model.py", "code/round2/train_e99_corrected_array_surface_model.py"),
    (PATHS.round2_code / "train_surface_interaction_energy_model.py", "code/round2/train_surface_interaction_energy_model.py"),
    (PATHS.round2_code / "train_e190_official_sionna_path_calibrator.py", "code/round2/train_e190_official_sionna_path_calibrator.py"),
    (PATHS.round2_code / "generate_e190_official_sionna_path_calibrator_test.py", "code/round2/generate_e190_official_sionna_path_calibrator_test.py"),
    (PATHS.round2_code / "package_round2_v15_official_sionna_path_calibrator.py", "code/round2/package_round2_v15_official_sionna_path_calibrator.py"),
)
E190_MODEL_FILES = tuple(
    (
        e190.refit_checkpoint_path(outer),
        f"model/official_path_calibrator/outer{outer}_refit_model.pt",
    )
    for outer in range(5)
) + tuple(
    (
        e190.screen_report_path(outer),
        f"model/reports/e190_outer{outer}_screen.json",
    )
    for outer in range(5)
) + (
    (TEST_ENDPOINT, "model/official_path_calibrator/test_fivefold_endpoint.npz"),
    (LOCAL_REPORT, "model/reports/e190_fixed500_report.json"),
    (TEST_REPORT, "model/reports/e190_test_generate.json"),
)
SOURCE_FILES = unique_files((*V11_SOURCE_FILES, *E190_SOURCE_FILES))
MODEL_FILES = unique_files((*V11_MODEL_FILES, *E190_MODEL_FILES))


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
    endpoint = np.load(TEST_ENDPOINT, allow_pickle=False)
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
        "endpoint_shapes": {name: list(endpoint[name].shape) for name in expected_endpoint},
    }


def main() -> None:
    required = (
        SOURCE_OUTPUT,
        TEST_ENDPOINT,
        TEST_REPORT,
        LOCAL_REPORT,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    )
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    local = json.loads(LOCAL_REPORT.read_text(encoding="utf-8"))
    generation = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    if local["status"] != "fixed500_failed" or bool(local["gate"]["passed"]):
        raise RuntimeError("V15 expects the explicitly user-authorized failed E190 gate")
    if not bool(generation["local_validation"]["user_authorized_failed_gate_deployment"]):
        raise RuntimeError("V15 user authorization marker is missing")

    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiments": ["Round2-E85", "Round2-E86", "Round2-E98", "Round2-E99", "Round2-E190"],
        "sionna_used": True,
        "same_position_test_sionna_paths_used": True,
        "hidden_official_test_h_used": False,
        "absolute_sionna_phase_used": False,
        "primary_hypothesis": (
            "Official-H one-to-one path supervision calibrates same-position external "
            "candidate mass, continuous angle-delay coordinates, and port energy better "
            "than the original V12 aggregate endpoint."
        ),
        "local_validation": {
            "complete_fixed500_score": local["selected"]["score"],
            "v12_complete_fixed500_score": local["baseline"]["score"],
            "delta": local["delta"]["score"],
            "fold_deltas": [row["delta"]["score"] for row in local["folds"]],
            "original_deployment_gate_passed": False,
            "online_diagnostic_requested_by_user": True,
        },
        "deployment_policy": generation["deployment_policy"],
        "data_lineage": generation["data_lineage"],
        "output_integrity": output_integrity,
        "reproduction": (
            "Load the five strict E190 refit checkpoints. For each checkpoint, apply its "
            "held-excluded E98 array mapping to the 128 same-position Test paths, predict "
            "path mass, continuous H/V/delay coordinates and eight-port energy, then "
            "ensemble mass and port by normalized arithmetic mean and coordinates by "
            "mass-weighted circular mean. Render noncoherent PAS/PDP, blend both with the "
            "V11 complex carrier at 0.20 in square-root space, and apply four PAS-to-PDP "
            "projections while preserving point power."
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
        archive.writestr("model/official_path_calibrator/", b"")
        archive.writestr("model/reports/", b"")
        for source, destination in MODEL_FILES:
            archive.write(source, destination)
        archive.writestr(
            "model/V15_VERSION.json",
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
        "model/V15_VERSION.json",
        "model/official_path_calibrator/outer0_refit_model.pt",
        "model/official_path_calibrator/outer4_refit_model.pt",
        "model/official_path_calibrator/test_fivefold_endpoint.npz",
        "model/reports/e190_fixed500_report.json",
        "model/reports/e190_test_generate.json",
        "code/round2/generate_e190_official_sionna_path_calibrator_test.py",
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
