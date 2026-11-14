#!/usr/bin/env python3
"""Package the unchanged v239 strict official-data local best as v246.

v246 is an online transfer probe, not a new model.  It preserves the exact
v239 output and packages the corresponding AI model checkpoints, derived
official-data features, and maintained source required for method review.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from generate_v159_frontier_consensus_support_soup import digest
from project_paths import PATHS


VERSION_NUMBER = 246
VERSION = "v246_strict_v239_online_transfer_probe"
SOURCE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"


SOURCE_FILES = [
    "package_v246_strict_v239_online_probe.py",
    "analyze_v239_nnls_path_energy_decoder.py",
    "analyze_v238_raw_path_conditional_angle.py",
    "analyze_v236_official_path_pdp_decoder.py",
    "analyze_v233_official_joint_ad_manifold.py",
    "analyze_v225_official_graph_residual_field.py",
    "generate_v218_official_multispectrum_physics_ai.py",
    "generate_v215_pointcloud_physics_neural_field.py",
    "analyze_v208_strict_historical_replay.py",
    "generate_v87_easy_field_projection.py",
    "powerfield_common.py",
    "project_paths.py",
]


MODEL_FILES = [
    PATHS.checkpoints / "v239_nnls_path_energy_decoder" / "seed_23901.pt",
    PATHS.checkpoints / "v239_nnls_path_energy_decoder" / "seed_23931.pt",
    PATHS.checkpoints / "v239_nnls_path_energy_decoder" / "seed_23959.pt",
    PATHS.cache_dir(239) / "v239_train_nnls_path_labels.npz",
    PATHS.cache_dir(239) / "v239_test_path_pdp.npy",
    PATHS.cache_dir(239) / "v239_nnls_path_energy_actions.npz",
    PATHS.shared_features / "multibounce_path_candidates.npz",
    PATHS.shared_features / "official_pointcloud_scene_cache.npz",
    PATHS.shared_features / "geodesic_support_features.npz",
    PATHS.shared_features / "height_geometry_cache.npz",
]


def validate_output(path: Path) -> dict[str, object]:
    arr = np.load(path, mmap_mode="r")
    expected_shape = (500, 256, 4, 192)
    if arr.shape != expected_shape or arr.dtype != np.complex64:
        raise ValueError(f"invalid v239 output: {arr.shape} {arr.dtype}")
    finite = True
    for start in range(0, len(arr), 12):
        finite = finite and bool(np.isfinite(np.asarray(arr[start : start + 12])).all())
    if not finite:
        raise FloatingPointError("v239 output contains non-finite values")
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite": finite}


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    for filename in SOURCE_FILES:
        path = PATHS.code / filename
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in MODEL_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_check = validate_output(SOURCE)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(SOURCE, "Round1_Test_Channel.npy")
        for filename in SOURCE_FILES:
            archive.write(PATHS.code / filename, filename)
        for path in MODEL_FILES:
            archive.write(path, path.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    nested_members = [name for name in members if "/" in name.strip("/")]
    if nested_members:
        raise ValueError(f"submission ZIP must be flat: {nested_members}")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The current v212 local replay gate is target-domain mismatched; an online "
            "submission of the unchanged strict official-data local best v239 is required "
            "to measure its actual transfer to the official evaluator."
        ),
        "model_identity": {
            "source_version": 239,
            "source_output": PATHS.relative(SOURCE),
            "source_local_score_traceback_leaked": 0.5900571130783295,
            "output_changes": "none",
        },
        "compliance_boundary": {
            "uses": [
                "official train H PDP",
                "official train/test coordinates",
                "official point-cloud geometry and multibounce path candidates",
                "strict official-data carriers",
                "PyTorch path Transformer checkpoints trained on official NNLS labels",
            ],
            "does_not_use": [
                "Sionna or synthetic channel labels",
                "v212/v217/v221/v222 teacher output",
                "synthetic replay truth",
                "hidden online labels",
                "raw RT H",
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
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
