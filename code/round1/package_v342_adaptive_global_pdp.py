#!/usr/bin/env python3
"""Package the unchanged strict v282 adaptive global-PDP prediction as v342."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from generate_v159_frontier_consensus_support_soup import digest
from project_paths import PATHS


VERSION_NUMBER = 342
VERSION = "v342_adaptive_global_pdp"
SOURCE = PATHS.cache_dir(282) / "v282_Round1_Test_Channel.npy"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PACKAGE_REPORT = REPORT_DIR / f"{VERSION}_package.json"

SOURCE_FILES = [
    "package_v342_adaptive_global_pdp.py",
    "analyze_v282_nested_v239_alpha_selector.py",
    "analyze_v281_v239_crossfit_pdp_current.py",
    "analyze_v274_planar_barrier_attention.py",
    "analyze_v239_nnls_path_energy_decoder.py",
    "analyze_v238_raw_path_conditional_angle.py",
    "analyze_v236_official_path_pdp_decoder.py",
    "analyze_v233_official_joint_ad_manifold.py",
    "analyze_v225_official_graph_residual_field.py",
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
    PATHS.cache_dir(282) / "v282_v239_alpha_actions.npz",
    PATHS.cache_dir(281) / "v281_v239_crossfit_pdp_current.npz",
    PATHS.cache_dir(239) / "v239_test_path_pdp.npy",
    PATHS.cache_dir(238) / "v238_test_anchor.npy",
    PATHS.checkpoints / "v239_nnls_path_energy_decoder" / "seed_23901.pt",
    PATHS.checkpoints / "v239_nnls_path_energy_decoder" / "seed_23931.pt",
    PATHS.checkpoints / "v239_nnls_path_energy_decoder" / "seed_23959.pt",
]


def validate_output() -> dict[str, object]:
    prediction = np.load(SOURCE, mmap_mode="r")
    if prediction.shape != (500, 256, 4, 192) or prediction.dtype != np.complex64:
        raise ValueError(f"invalid v342 output: {prediction.shape} {prediction.dtype}")
    finite = True
    for start in range(0, len(prediction), 12):
        finite = finite and bool(
            np.isfinite(np.asarray(prediction[start : start + 12])).all()
        )
    if not finite:
        raise FloatingPointError("v342 output contains non-finite values")
    return {"shape": list(prediction.shape), "dtype": str(prediction.dtype), "finite": finite}


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    for filename in SOURCE_FILES:
        if not (PATHS.code / filename).is_file():
            raise FileNotFoundError(PATHS.code / filename)
    for path in MODEL_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_check = validate_output()
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(SOURCE, "Round1_Test_Channel.npy")
        for filename in SOURCE_FILES:
            archive.write(PATHS.code / filename, filename)
        for path in MODEL_FILES:
            archive.write(path, path.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    nested = [name for name in members if "/" in name.strip("/")]
    if nested:
        raise ValueError(f"submission ZIP must be flat: {nested}")
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "model_identity": {
            "source_version": 282,
            "source_output": PATHS.relative(SOURCE),
            "output_changes": "none",
            "official_oof_pdp_delta": 0.0353250867408993,
            "official_oof_wins": 5,
            "frozen_local_score": 0.5858171973760586,
        },
        "compliance_boundary": {
            "uses": [
                "official train H",
                "official train/test coordinates",
                "official point-cloud geometry features",
                "nested official-only v239 path decoder and ExtraTrees alpha selector",
            ],
            "does_not_use": [
                "v211/v212 outputs or teacher labels",
                "synthetic training/test labels",
                "restored Etoile scene data",
                "RT H/path coefficients",
                "online-score subspace operations",
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
