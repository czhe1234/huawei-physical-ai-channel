#!/usr/bin/env python3
"""Package the validated Round 2 calibrated exact-projection model as V4."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from project_paths import PATHS


VERSION = "V4"
METHOD = "predictability_calibrated_exact_projection"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_calibrated_exact_projection"
V3_ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_anchored_neural_rbf"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_calibrated_exact_projection"
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_V4_Calibrated_Exact_Projection_Test_Channel.npy"
SOURCE_SELECTION = ARTIFACT_DIR / "calibrated_exact_projection_selection.json"
SOURCE_CALIBRATION_MODEL = ARTIFACT_DIR / "full_group_calibration_model.npz"
SOURCE_REPORT = ANALYSIS_DIR / "calibrated_exact_projection.json"
SOURCE_CHECKPOINT = V3_ARTIFACT_DIR / "full_epoch6.pt"
SOURCE_TEST_WEIGHTS = V3_ARTIFACT_DIR / "test_corrected_weights.npz"
E21_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_v4_exact_joint_projection"
    / "exact_joint_projection.json"
)
E22_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_v4_predictability_calibrated_groups"
    / "predictability_calibrated_groups.json"
)
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)

SOURCE_FILES = (
    (PATHS.code_root / "project_paths.py", "code/project_paths.py"),
    (PATHS.common_code / "__init__.py", "code/common/__init__.py"),
    (PATHS.common_code / "official_metrics.py", "code/common/official_metrics.py"),
    (PATHS.round2_code / "__init__.py", "code/round2/__init__.py"),
    (PATHS.round2_code / "README.md", "code/round2/README.md"),
    (PATHS.round2_code / "data.py", "code/round2/data.py"),
    (PATHS.round2_code / "spatial_hole_cv.py", "code/round2/spatial_hole_cv.py"),
    (PATHS.round2_code / "carrier_calibration.py", "code/round2/carrier_calibration.py"),
    (PATHS.round2_code / "complex_baselines.py", "code/round2/complex_baselines.py"),
    (
        PATHS.round2_code / "run_foundation_benchmark.py",
        "code/round2/run_foundation_benchmark.py",
    ),
    (PATHS.round2_code / "geometry_features.py", "code/round2/geometry_features.py"),
    (
        PATHS.round2_code / "pre_sionna_spectral.py",
        "code/round2/pre_sionna_spectral.py",
    ),
    (PATHS.round2_code / "pre_sionna_targets.py", "code/round2/pre_sionna_targets.py"),
    (
        PATHS.round2_code / "run_pre_sionna_pipeline.py",
        "code/round2/run_pre_sionna_pipeline.py",
    ),
    (
        PATHS.round2_code / "run_rbf_spectral_field.py",
        "code/round2/run_rbf_spectral_field.py",
    ),
    (
        PATHS.round2_code / "run_v3_anchored_neural_rbf.py",
        "code/round2/run_v3_anchored_neural_rbf.py",
    ),
    (
        PATHS.round2_code / "run_v3_grouped_link_refinement.py",
        "code/round2/run_v3_grouped_link_refinement.py",
    ),
    (
        PATHS.round2_code / "evaluate_v3_anchored_projection.py",
        "code/round2/evaluate_v3_anchored_projection.py",
    ),
    (
        PATHS.round2_code / "build_v3_anchored_neural_rbf.py",
        "code/round2/build_v3_anchored_neural_rbf.py",
    ),
    (
        PATHS.round2_code / "analyze_v4_exact_joint_projection.py",
        "code/round2/analyze_v4_exact_joint_projection.py",
    ),
    (
        PATHS.round2_code / "analyze_v4_predictability_calibrated_groups.py",
        "code/round2/analyze_v4_predictability_calibrated_groups.py",
    ),
    (
        PATHS.round2_code / "run_v4_calibrated_exact_projection.py",
        "code/round2/run_v4_calibrated_exact_projection.py",
    ),
    (PATHS.code_root / "tools" / "check_environment.py", "code/tools/check_environment.py"),
    (
        PATHS.code_root / "tools" / "bootstrap_environment.sh",
        "code/tools/bootstrap_environment.sh",
    ),
    (PATHS.code_root / "tools" / "run_in_env.sh", "code/tools/run_in_env.sh"),
    (PATHS.root / "environment.yml", "environment.yml"),
    (PATHS.root / "environment.cuda.yml", "environment.cuda.yml"),
)

MODEL_FILES = (
    (SOURCE_CHECKPOINT, "model/full_epoch6.pt"),
    (SOURCE_TEST_WEIGHTS, "model/test_corrected_weights.npz"),
    (SOURCE_CALIBRATION_MODEL, "model/full_group_calibration_model.npz"),
    (SOURCE_SELECTION, "model/calibrated_exact_projection_selection.json"),
    (SOURCE_REPORT, "model/calibrated_exact_projection.json"),
    (E21_REPORT, "model/exact_joint_projection_ablation.json"),
    (E22_REPORT, "model/predictability_calibrated_groups_ablation.json"),
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
    for source in (*required, *(path for path, _ in SOURCE_FILES)):
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    experiment = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiment": "Round2-E23",
        "sionna_used": False,
        "official_inputs_only": True,
        "local_validation": experiment["nested_selected"]["overall"],
        "local_final_policy_oof": experiment["final_policy_oof"]["overall"],
        "local_gain_vs_v3": experiment["delta_vs_v3"]["overall_score"],
        "positive_fold_count": int(
            sum(row["score"] > 0.0 for row in experiment["delta_vs_v3"]["folds"])
        ),
        "final_policy": experiment["nested_selected"]["final_policy"],
        "output_integrity": output_integrity,
        "reproduction": (
            "Run code/tools/bootstrap_environment.sh; reproduce the Round 2 foundation, "
            "pre-Sionna, V2 and V3 artifacts; run the E22 target calibration; then run "
            "run_v4_calibrated_exact_projection.py."
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
        for source, destination in MODEL_FILES:
            archive.write(source, destination)
        archive.writestr(
            "model/V4_VERSION.json",
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
        "model/full_epoch6.pt",
        "model/test_corrected_weights.npz",
        "model/full_group_calibration_model.npz",
        "model/calibrated_exact_projection_selection.json",
        "model/V4_VERSION.json",
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
