#!/usr/bin/env python3
"""Generate the carrier-relative confocal D projection candidate."""

from __future__ import annotations

import json
import zipfile

import numpy as np

import analyze_v247_delaunay_spectral_field as v247
import analyze_v393_confocal_pointcloud_pdp as v393
import analyze_v395_mass_preserving_confocal_pdp as v395
import generate_v396_confocal_mass_preserving_d_projection as v396
from project_paths import PATHS


VERSION_NUMBER = 398
VERSION = "v398_relative_confocal_d_projection"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v398_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v398_relative_confocal_targets.npz"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
BASE = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"
V396_TARGETS = PATHS.cache_dir(396) / "v396_confocal_targets.npz"
EFFECTIVE_ALPHA = 0.5

SOURCE_FILES = list(
    dict.fromkeys(
        [
            "generate_v398_relative_confocal_d_projection.py",
            "analyze_v397_confocal_v239_carrier_transfer.py",
            *v396.SOURCE_FILES,
        ]
    )
)


def current_d() -> np.ndarray:
    source = np.load(BASE, mmap_mode="r")
    output = np.empty((len(source), 192), dtype=np.float32)
    for start in range(0, len(source), 8):
        end = min(start + 8, len(source))
        _sh, output[start:end] = v247.extract_current(
            np.asarray(source[start:end], dtype=np.complex64)
        )
    return v393.normalize(output)


def package(output_check: dict[str, object]) -> dict[str, object]:
    for filename in SOURCE_FILES:
        if not (PATHS.code / filename).is_file():
            raise FileNotFoundError(PATHS.code / filename)
    model_files = [
        v396.MODEL,
        TARGETS,
        v393.FEATURE_CACHE,
        v393.SCENE_CACHE,
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28401.pt",
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28431.pt",
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28461.pt",
        PATHS.cache_dir(284) / "v284_planar_geometry_targets.npz",
        PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz",
        PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz",
        PATHS.version_report_dir(265) / "v265_scatterer_delay_angle_oracle.json",
        PATHS.version_report_dir(395) / "v395_mass_preserving_confocal_pdp.json",
        PATHS.version_report_dir(397) / "v397_confocal_v239_carrier_transfer.json",
    ]
    for path in model_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for filename in SOURCE_FILES:
            archive.write(PATHS.code / filename, filename)
        for path in model_files:
            archive.write(path, path.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    nested = [name for name in members if "/" in name.strip("/")]
    if nested:
        raise ValueError(f"submission ZIP must be flat: {nested}")
    return {
        **output_check,
        "zip_crc": True,
        "zip_member_count": len(members),
        "zip_members": members,
        "sha256": v396.digest(ZIP_OUT, "sha256"),
        "md5": v396.digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    for path in (BASE, V396_TARGETS, v396.MODEL, v393.FEATURE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    current = current_d()
    frozen = np.load(V396_TARGETS)
    residual = frozen["residual"].astype(np.float32)
    target = v395.apply_mass_preserving_residual(
        current, residual, EFFECTIVE_ALPHA
    )
    active_mass_error = np.max(
        np.abs(
            target[:, : v393.ACTIVE_BINS].sum(axis=1)
            - current[:, : v393.ACTIVE_BINS].sum(axis=1)
        )
    )
    np.savez_compressed(
        TARGETS,
        target_d=target,
        current_d=current,
        residual=residual,
        effective_alpha=np.asarray(EFFECTIVE_ALPHA, dtype=np.float32),
    )
    v396.OUTPUT = OUTPUT
    projection = v396.project_d_only(target)
    output_check = v396.validate_output()
    integrity = package(output_check)
    target_diagnostics = {
        "target_current_l1": v396.summarize(
            np.sum(np.abs(target - current), axis=1)
        ),
        "target_current_cosine": v396.summarize(
            v393.row_cos(target, current)
        ),
        "active_mass_max_abs_error": float(active_mass_error),
        "row_sum_max_abs_error": float(
            np.max(np.abs(target.sum(axis=1) - 1.0))
        ),
        "finite": bool(np.isfinite(target).all()),
    }
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The confocal within-active-delay relative action that passed v397 improves "
            "the strict v284 carrier without replacing its absolute D state."
        ),
        "official_oof_evidence": {
            "source": "v397 strict v239-like carrier transfer",
            "full_vs_carrier_cosine": 0.009196945984815841,
            "full_vs_context_cosine": 0.002969757135745832,
            "full_vs_shuffled_cosine": 0.0037895619238930587,
            "circular_emd_reduction": 0.02514171789294592,
            "joint_fold_wins": 5,
        },
        "parameters": {
            "effective_alpha": EFFECTIVE_ALPHA,
            "active_bins": v393.ACTIVE_BINS,
            "model": PATHS.relative(v396.MODEL),
            "base": PATHS.relative(BASE),
            "target_action": "relative log-ratio on v284 current D",
            "projection": "one positive real gain per CIR delay bin",
        },
        "target_diagnostics": target_diagnostics,
        "projection_diagnostics": projection,
        "compliance_boundary": {
            "uses": [
                "official train H normalized D labels",
                "official train/test coordinates and Round1_Map point cloud",
                "frozen supervised ExtraTrees AI checkpoint",
                "strict official-only v284 full-H carrier",
            ],
            "does_not_use": [
                "Sionna/RT test target or path coefficient",
                "restored mesh/material/object labels",
                "v212/synthetic teacher",
                "hidden, local-replay, or online truth",
            ],
        },
        "preserved_quantities": {
            "v284_active_delay_total_mass": True,
            "v284_tail_total_mass": True,
            "complex_phase_within_each_delay_bin": True,
            "point_total_power": True,
            "per_delay_bin_antenna_pol_ue_allocation": True,
        },
        "integrity": integrity,
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS),
            "model": PATHS.relative(v396.MODEL),
            "zip": PATHS.relative(ZIP_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
