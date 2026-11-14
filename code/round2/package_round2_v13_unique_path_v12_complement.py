#!/usr/bin/env python3
"""Package the strict E120 unique-path complement as Round 2 V13."""

from __future__ import annotations

import json
import zipfile

import numpy as np

import run_e120_fivefold_unique_path_v12 as e120
from package_round2_v8_full_sionna_migration import unique_files
from package_round2_v12_surface_endpoint_v11_blend import (
    MODEL_FILES as V12_MODEL_FILES,
    SOURCE_FILES as V12_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V13"
METHOD = "unique_path_v12_complement"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "fivefold_unique_path_v12"
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_E120_Unique_Path_V12_Test_Channel.npy"
TEST_ENDPOINT = ARTIFACT_DIR / "test_fivefold_mass_coordinate_endpoint.npz"
SCREEN_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_fivefold_unique_path_v12"
    / "fivefold_unique_path_v12.json"
)
TEST_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_fivefold_unique_path_v12"
    / "fivefold_unique_path_v12_test.json"
)
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)


E120_SOURCE_FILES = (
    (
        PATHS.round2_code / "analyze_global_local_path_lifecycle.py",
        "code/round2/analyze_global_local_path_lifecycle.py",
    ),
    (
        PATHS.round2_code / "analyze_e117_unique_path_cluster_ceiling.py",
        "code/round2/analyze_e117_unique_path_cluster_ceiling.py",
    ),
    (
        PATHS.round2_code / "analyze_e119_v12_unique_path_complement.py",
        "code/round2/analyze_e119_v12_unique_path_complement.py",
    ),
    (
        PATHS.round2_code / "run_path_component_transport.py",
        "code/round2/run_path_component_transport.py",
    ),
    (
        PATHS.round2_code / "run_path_track_spectral_field.py",
        "code/round2/run_path_track_spectral_field.py",
    ),
    (
        PATHS.round2_code / "train_path_conditioned_scene_gnn.py",
        "code/round2/train_path_conditioned_scene_gnn.py",
    ),
    (
        PATHS.round2_code / "train_e115_complete_path_state_operator.py",
        "code/round2/train_e115_complete_path_state_operator.py",
    ),
    (
        PATHS.round2_code / "train_e116_nested_path_state_operator.py",
        "code/round2/train_e116_nested_path_state_operator.py",
    ),
    (
        PATHS.round2_code / "train_e118_hierarchical_unique_path_operator.py",
        "code/round2/train_e118_hierarchical_unique_path_operator.py",
    ),
    (
        PATHS.round2_code / "run_e120_fivefold_unique_path_v12.py",
        "code/round2/run_e120_fivefold_unique_path_v12.py",
    ),
    (
        PATHS.round2_code / "generate_e120_unique_path_v12_test_channel.py",
        "code/round2/generate_e120_unique_path_v12_test_channel.py",
    ),
    (
        PATHS.round2_code / "package_round2_v13_unique_path_v12_complement.py",
        "code/round2/package_round2_v13_unique_path_v12_complement.py",
    ),
)
E120_MODEL_FILES = tuple(
    (
        PATHS.artifacts
        / "round2"
        / "hierarchical_unique_path_operator"
        / f"e120_outer{outer}_hierarchical_scene.pt",
        f"model/unique_path/e120_outer{outer}_hierarchical_scene.pt",
    )
    for outer in range(5)
) + (
    (TEST_ENDPOINT, "model/unique_path/test_fivefold_mass_coordinate_endpoint.npz"),
    (SCREEN_REPORT, "model/reports/fivefold_unique_path_v12.json"),
    (TEST_REPORT, "model/reports/fivefold_unique_path_v12_test.json"),
)
SOURCE_FILES = unique_files((*V12_SOURCE_FILES, *E120_SOURCE_FILES))
MODEL_FILES = unique_files((*V12_MODEL_FILES, *E120_MODEL_FILES))


def validate_output() -> dict[str, object]:
    setup = json.loads(
        (PATHS.round2 / "Round2_Setup.json").read_text(encoding="utf-8")
    )
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
        finite &= bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
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
    }
    for name, shape in expected_endpoint.items():
        if endpoint[name].shape != shape or not np.isfinite(endpoint[name]).all():
            raise ValueError(f"invalid endpoint {name}: {endpoint[name].shape}")
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(SOURCE_OUTPUT.stat().st_size),
        "endpoint_shapes": {
            name: list(endpoint[name].shape) for name in expected_endpoint
        },
    }


def main() -> None:
    required = (
        SOURCE_OUTPUT,
        TEST_ENDPOINT,
        SCREEN_REPORT,
        TEST_REPORT,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    )
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    screen = json.loads(SCREEN_REPORT.read_text(encoding="utf-8"))
    generation = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    selected = screen["selected"]
    gate = screen["deployment_gate"]
    if selected["name"] != e120.PRIMARY_ACTION.name or not gate["passed"]:
        raise RuntimeError("E120 preregistered primary deployment gate did not pass")
    if gate["positive_folds"] < 4 or gate["worst_fold_delta"] < -0.001:
        raise RuntimeError(f"E120 fold stability gate failed: {gate}")

    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiments": [
            "Round2-E114",
            "Round2-E115",
            "Round2-E116",
            "Round2-E117",
            "Round2-E118",
            "Round2-E119",
            "Round2-E120",
        ],
        "sionna_used_by_v12_base": True,
        "sionna_used_by_new_unique_path_module": False,
        "absolute_sionna_phase_used": False,
        "primary_hypothesis": (
            "Official train-H neighbor paths contain complementary unique-path mass "
            "and coordinate information that can correct V12 PAS/PDP while preserving "
            "its complex phase carrier and point power."
        ),
        "local_validation": {
            "v12_baseline": screen["baseline"],
            "selected": selected["selected"],
            "delta": selected["delta"],
            "fold_deltas": [row["delta"] for row in selected["folds"]],
            "positive_folds": selected["positive_folds"],
            "worst_fold_delta": selected["worst_fold_delta"],
            "deployment_gate": gate,
            "diagnostic_pas_only_not_deployed": screen["diagnostic"],
        },
        "deployment_policy": generation["deployment_policy"],
        "ensemble_agreement": generation["ensemble_agreement"],
        "output_integrity": output_integrity,
        "reproduction": (
            "Decompose official train H into continuous paths. For each of five strict "
            "models, construct test evidence from its three allowed donor folds, cluster "
            "176 raw tokens into at most 64 unique paths, and predict path mass and "
            "coordinates with the stored scene-conditioned checkpoint. Render and "
            "average the five PAS/PDP endpoints, blend both into V12 at 0.10 in "
            "square-root space, apply four PAS-to-PDP projections, and preserve V12 "
            "phase and point power."
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
        archive.writestr("model/unique_path/", b"")
        archive.writestr("model/reports/", b"")
        for source, destination in MODEL_FILES:
            archive.write(source, destination)
        archive.writestr(
            "model/V13_VERSION.json",
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
        "model/V13_VERSION.json",
        "model/unique_path/e120_outer0_hierarchical_scene.pt",
        "model/unique_path/e120_outer4_hierarchical_scene.pt",
        "model/unique_path/test_fivefold_mass_coordinate_endpoint.npz",
        "model/reports/fivefold_unique_path_v12.json",
        "code/round2/generate_e120_unique_path_v12_test_channel.py",
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
