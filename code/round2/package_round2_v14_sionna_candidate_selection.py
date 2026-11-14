#!/usr/bin/env python3
"""Package the E136 Sionna-assisted candidate selector as Round 2 V14."""

from __future__ import annotations

import json
import zipfile

import numpy as np

import analyze_e135_sionna_selection_v12_complete_score as e135
import generate_e136_sionna_candidate_selection_v12_test_channel as e136
import optimize_e98_global_array_angle_mapping as e98
from package_round2_v13_unique_path_v12_complement import E120_SOURCE_FILES
from package_round2_v8_full_sionna_migration import unique_files
from package_round2_v12_surface_endpoint_v11_blend import (
    MODEL_FILES as V12_MODEL_FILES,
    SOURCE_FILES as V12_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V14"
METHOD = "sionna_candidate_selection_v12"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
SOURCE_OUTPUT = e136.OUTPUT
TEST_ENDPOINT = e136.ENDPOINT
TEST_REPORT = e136.REPORT
LOCAL_REPORT = e135.REPORT
VERSION_REPORT = (
    PATHS.analysis / "results" / "version_reports" / VERSION / f"{VERSION}_{METHOD}.json"
)

E136_SOURCE_FILES = (
    (PATHS.round2_code / "analyze_e114_path_state_variable_ceiling.py", "code/round2/analyze_e114_path_state_variable_ceiling.py"),
    (PATHS.round2_code / "train_global_local_path_set_selector.py", "code/round2/train_global_local_path_set_selector.py"),
    (PATHS.round2_code / "analyze_e121_standalone_unique_path_carrier.py", "code/round2/analyze_e121_standalone_unique_path_carrier.py"),
    (PATHS.round2_code / "analyze_e123_complete_path_variable_system.py", "code/round2/analyze_e123_complete_path_variable_system.py"),
    (PATHS.round2_code / "analyze_e124_one_to_one_path_identity_ceiling.py", "code/round2/analyze_e124_one_to_one_path_identity_ceiling.py"),
    (PATHS.round2_code / "analyze_e131_v7_path_selection_evidence.py", "code/round2/analyze_e131_v7_path_selection_evidence.py"),
    (PATHS.round2_code / "analyze_e132_v7_path_coverage_gap.py", "code/round2/analyze_e132_v7_path_coverage_gap.py"),
    (PATHS.round2_code / "analyze_e133_v7_deformable_peak_alignment.py", "code/round2/analyze_e133_v7_deformable_peak_alignment.py"),
    (PATHS.round2_code / "analyze_e134_sionna_candidate_selection_evidence.py", "code/round2/analyze_e134_sionna_candidate_selection_evidence.py"),
    (PATHS.round2_code / "analyze_e135_sionna_selection_v12_complete_score.py", "code/round2/analyze_e135_sionna_selection_v12_complete_score.py"),
    (PATHS.round2_code / "optimize_e98_global_array_angle_mapping.py", "code/round2/optimize_e98_global_array_angle_mapping.py"),
    (PATHS.round2_code / "train_e115_complete_path_state_operator.py", "code/round2/train_e115_complete_path_state_operator.py"),
    (PATHS.round2_code / "train_e118_hierarchical_unique_path_operator.py", "code/round2/train_e118_hierarchical_unique_path_operator.py"),
    (PATHS.round2_code / "generate_e120_unique_path_v12_test_channel.py", "code/round2/generate_e120_unique_path_v12_test_channel.py"),
    (PATHS.round2_code / "generate_e136_sionna_candidate_selection_v12_test_channel.py", "code/round2/generate_e136_sionna_candidate_selection_v12_test_channel.py"),
    (PATHS.round2_code / "package_round2_v14_sionna_candidate_selection.py", "code/round2/package_round2_v14_sionna_candidate_selection.py"),
)
E136_MODEL_FILES = (
    (TEST_ENDPOINT, "model/sionna_candidate_selection/test_endpoint.npz"),
    (e98.PARAMETERS, "model/sionna_candidate_selection/strict_fold_angle_parameters.npz"),
    (LOCAL_REPORT, "model/reports/e135_sionna_selection_v12_complete_score.json"),
    (TEST_REPORT, "model/reports/e136_sionna_candidate_selection_v12_test.json"),
)
SOURCE_FILES = unique_files((*V12_SOURCE_FILES, *E120_SOURCE_FILES, *E136_SOURCE_FILES))
MODEL_FILES = unique_files((*V12_MODEL_FILES, *E136_MODEL_FILES))


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
        "pas": (expected[0], 2, 16, 8, 4),
        "pdp": (expected[0], 2, 4, 192),
        "pas_model_cosine": (expected[0],),
        "pdp_model_cosine": (expected[0],),
        "selected": (5, expected[0], 64),
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
    gate = local["gate"]
    if gate["primary_variant"] != e135.PRIMARY:
        raise RuntimeError(f"unexpected E135 primary variant: {gate['primary_variant']}")
    if bool(gate["passed"]):
        raise RuntimeError("V14 is specifically the user-requested failed-gate online diagnostic")

    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiments": ["Round2-E98", "Round2-E131", "Round2-E134", "Round2-E135", "Round2-E136"],
        "sionna_used": True,
        "same_position_test_sionna_used": True,
        "absolute_sionna_phase_used": False,
        "primary_hypothesis": (
            "E98-aligned same-position Sionna H/V/delay paths can improve the Top-15 "
            "candidate identity used to form a weak PAS/PDP endpoint for V12."
        ),
        "local_validation": {
            "v12_score": gate["v12_score"],
            "selected_score": gate["primary_score"],
            "delta": gate["delta_to_v12"],
            "positive_folds": gate["positive_folds"],
            "worst_fold_delta": gate["worst_fold_delta"],
            "deployment_gate_passed": False,
            "online_diagnostic_requested_by_user": True,
        },
        "deployment_policy": generation["deployment_policy"],
        "ensemble_agreement": generation["ensemble_agreement"],
        "output_integrity": output_integrity,
        "reproduction": (
            "For each of five strict donor-fold systems, cluster official train-H neighbor "
            "components into 64 test candidates. Rank each candidate by base mass, V7 local "
            "angle-delay energy, and E98-aligned same-position Sionna H/V/delay evidence; "
            "retain 15, render PAS/PDP, average the five endpoints, blend both into V12 at "
            "0.10 in square-root space, and apply four PAS-to-PDP projections."
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
        archive.writestr("model/sionna_candidate_selection/", b"")
        archive.writestr("model/reports/", b"")
        for source, destination in MODEL_FILES:
            archive.write(source, destination)
        archive.writestr(
            "model/V14_VERSION.json",
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
        "model/V14_VERSION.json",
        "model/sionna_candidate_selection/test_endpoint.npz",
        "model/sionna_candidate_selection/strict_fold_angle_parameters.npz",
        "model/reports/e135_sionna_selection_v12_complete_score.json",
        "model/reports/e136_sionna_candidate_selection_v12_test.json",
        "code/round2/generate_e136_sionna_candidate_selection_v12_test_channel.py",
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
