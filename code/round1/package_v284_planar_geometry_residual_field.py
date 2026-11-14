#!/usr/bin/env python3
"""Package the unchanged v284 strict official-data prediction for submission."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from generate_v159_frontier_consensus_support_soup import digest
from project_paths import PATHS


VERSION_NUMBER = 284
VERSION = "v284_planar_geometry_residual_field"
SOURCE = PATHS.cache_dir(VERSION_NUMBER) / "v284_Round1_Test_Channel.npy"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
PACKAGE_REPORT = REPORT_DIR / "v284_planar_geometry_residual_field_package.json"

SOURCE_FILES = [
    "package_v284_planar_geometry_residual_field.py",
    "analyze_v284_planar_geometry_residual_field.py",
    "analyze_v283_joint_geometry_path_renderer.py",
    "analyze_v279_joint_component_trajectory.py",
    "analyze_v274_planar_barrier_attention.py",
    "analyze_v265_scatterer_delay_angle_oracle.py",
    "analyze_v264_patch_scatterer_transformer.py",
    "analyze_v257_neural_pairwise_spectral_transport.py",
    "analyze_v254_route_topology_spectral_field.py",
    "analyze_v247_delaunay_spectral_field.py",
    "analyze_v239_nnls_path_energy_decoder.py",
    "analyze_v208_strict_historical_replay.py",
    "generate_v218_official_multispectrum_physics_ai.py",
    "generate_v215_pointcloud_physics_neural_field.py",
    "generate_v187_leaked_pas_lifecycle_projection.py",
    "generate_v87_easy_field_projection.py",
    "build_official_pointcloud_scene.py",
    "powerfield_common.py",
    "project_paths.py",
]

MODEL_FILES = [
    PATHS.checkpoints / VERSION / "seed_28401.pt",
    PATHS.checkpoints / VERSION / "seed_28431.pt",
    PATHS.checkpoints / VERSION / "seed_28461.pt",
    PATHS.cache_dir(284) / "v284_planar_geometry_targets.npz",
    PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz",
    PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz",
    PATHS.shared_features / "official_pointcloud_scene_cache.npz",
    PATHS.version_report_dir(265) / "v265_scatterer_delay_angle_oracle.json",
]


def validate_output(path: Path) -> dict[str, object]:
    prediction = np.load(path, mmap_mode="r")
    expected_shape = (500, 256, 4, 192)
    if prediction.shape != expected_shape or prediction.dtype != np.complex64:
        raise ValueError(
            f"invalid v284 output: {prediction.shape} {prediction.dtype}"
        )
    finite = True
    for start in range(0, len(prediction), 12):
        finite = finite and bool(
            np.isfinite(np.asarray(prediction[start : start + 12])).all()
        )
    if not finite:
        raise FloatingPointError("v284 output contains non-finite values")
    return {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "finite": finite,
    }


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    for filename in SOURCE_FILES:
        source = PATHS.code / filename
        if not source.is_file():
            raise FileNotFoundError(source)
    for model_file in MODEL_FILES:
        if not model_file.is_file():
            raise FileNotFoundError(model_file)

    output_check = validate_output(SOURCE)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(SOURCE, "Round1_Test_Channel.npy")
        for filename in SOURCE_FILES:
            archive.write(PATHS.code / filename, filename)
        for model_file in MODEL_FILES:
            archive.write(model_file, model_file.name)

    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    nested_members = [name for name in members if "/" in name.strip("/")]
    if nested_members:
        raise ValueError(f"submission ZIP must be flat: {nested_members}")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "packaging_reason": "user explicitly requested an online submission of v284",
        "model_identity": {
            "source_version": VERSION_NUMBER,
            "source_output": PATHS.relative(SOURCE),
            "source_local_score_traceback_leaked": 0.5925112281386167,
            "output_changes": "none",
        },
        "compliance_boundary": {
            "uses": [
                "official train H joint S-H/PDP labels",
                "official train/test coordinates",
                "official point-cloud wall-path geometry",
                "PyTorch planar geometry residual-field checkpoints",
                "strict official-only v239 carrier",
            ],
            "does_not_use": [
                "synthetic/Sionna channel labels",
                "v212/v217 teacher output",
                "hidden online labels",
                "RT H or path coefficients",
            ],
        },
        "integrity": {
            **output_check,
            "zip_crc": True,
            "zip_members": members,
            "sha256": digest(ZIP_OUT, "sha256"),
            "md5": digest(ZIP_OUT, "md5"),
        },
        "outputs": {
            "zip": PATHS.relative(ZIP_OUT),
            "report": PATHS.relative(PACKAGE_REPORT),
        },
    }
    PACKAGE_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
