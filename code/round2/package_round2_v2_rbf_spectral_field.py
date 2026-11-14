#!/usr/bin/env python3
"""Package the validated Round2-E3 RBF spectral field as V2."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from project_paths import PATHS


VERSION = "V2"
METHOD = "rbf_sqrt_spectral_field"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
SOURCE_OUTPUT = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "Round2_RBF_Spectral_Field_Test_Channel.npy"
)
SOURCE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_rbf_spectral_field"
    / "round2_rbf_spectral_field.json"
)
SOURCE_SELECTION = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_selection.json"
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
    (PATHS.code_root / "tools" / "check_environment.py", "code/tools/check_environment.py"),
    (
        PATHS.code_root / "tools" / "bootstrap_environment.sh",
        "code/tools/bootstrap_environment.sh",
    ),
    (PATHS.code_root / "tools" / "run_in_env.sh", "code/tools/run_in_env.sh"),
    (PATHS.root / "environment.yml", "environment.yml"),
    (PATHS.root / "environment.cuda.yml", "environment.cuda.yml"),
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
    required = (SOURCE_OUTPUT, SOURCE_REPORT, SOURCE_SELECTION)
    for source in (*required, *(path for path, _ in SOURCE_FILES)):
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    experiment = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    local_validation = experiment["validation"]["nested_selected"]
    final_policy = experiment["validation"]["final_policy"]
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiment": "Round2-E3",
        "sionna_used": False,
        "official_inputs_only": True,
        "local_validation": local_validation,
        "local_gain_vs_v1": float(local_validation["score"] - 0.5965343913062728),
        "final_policy": final_policy,
        "output_integrity": output_integrity,
        "reproduction": (
            "Run code/tools/bootstrap_environment.sh, then "
            "code/tools/run_in_env.sh python -u code/round2/run_foundation_benchmark.py, "
            "code/tools/run_in_env.sh python -u code/round2/run_pre_sionna_pipeline.py, "
            "and code/tools/run_in_env.sh python -u code/round2/run_rbf_spectral_field.py."
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
        archive.write(SOURCE_REPORT, "model/round2_rbf_spectral_field.json")
        archive.write(SOURCE_SELECTION, "model/round2_rbf_selection.json")
        archive.writestr(
            "model/V2_VERSION.json",
            json.dumps(package_metadata, indent=2, ensure_ascii=False).encode("utf-8") + b"\n",
        )
        archive.writestr(
            "model/INFERENCE.txt",
            (package_metadata["reproduction"] + "\n").encode("ascii"),
        )
        for source, destination in SOURCE_FILES:
            archive.write(source, destination)

    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
        compression_types = {member.compress_type for member in archive.infolist()}
    if bad_member is not None:
        raise RuntimeError(f"ZIP CRC failure: {bad_member}")
    if "Round2_Test_Channel.npy" not in members:
        raise RuntimeError("Round2_Test_Channel.npy is not at the ZIP root")
    if compression_types != {zipfile.ZIP_STORED}:
        raise RuntimeError(f"unexpected ZIP compression types: {compression_types}")

    package_metadata["zip_integrity"] = {
        "crc": True,
        "compression": "ZIP_STORED",
        "bytes": int(ARCHIVE.stat().st_size),
        "member_count": len(members),
        "members": members,
    }
    VERSION_REPORT.write_text(
        json.dumps(package_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(package_metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
