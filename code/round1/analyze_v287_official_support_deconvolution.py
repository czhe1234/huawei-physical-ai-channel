#!/usr/bin/env python3
"""Learn a universal deconvolution from degraded official support."""

from __future__ import annotations

import json

import numpy as np

import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v283_joint_geometry_path_renderer as v283
import analyze_v285_v239_isomorphic_joint_residual as v285
import analyze_v286_carrier_quality_matched_residual as v286
from project_paths import PATHS


VERSION_NUMBER = 287
VERSION = "v287_official_support_deconvolution"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
SAMPLES_OUT = OUT_DIR / "v287_degraded_official_support.npz"
TARGETS_OUT = OUT_DIR / "v287_joint_targets.npz"
OUTPUT = OUT_DIR / "v287_Round1_Test_Channel.npy"

BLUR_S = 0.24
BLUR_H = 0.36
POWER_TEMPERATURE = 0.95


def degrade(truth: np.ndarray) -> np.ndarray:
    delay_blurred = (
        (1.0 - BLUR_S) * truth
        + 0.5 * BLUR_S * (
            np.roll(truth, 1, axis=1) + np.roll(truth, -1, axis=1)
        )
    )
    angle_blurred = (
        (1.0 - BLUR_H) * delay_blurred
        + 0.5 * BLUR_H * (
            np.roll(delay_blurred, 1, axis=2)
            + np.roll(delay_blurred, -1, axis=2)
        )
    )
    return v279.normalize(
        np.maximum(angle_blurred, 1e-12) ** POWER_TEMPERATURE
    )


def build_deconvolution_samples(
    truth_sh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cache = np.load(v285.V281_CACHE)
    selected_rows, assigned_fold, assignment = v285.assign_unique_rows(
        cache["ids"].astype(np.int64), cache["fold"].astype(np.int16)
    )
    query_ids = cache["ids"][selected_rows].astype(np.int64)
    clean = truth_sh[query_ids]
    degraded = degrade(clean)
    test_current, _ = v274.extract_current_marginals(v285.BASE)
    report = {
        **assignment,
        "joint_current_baseline": v279.metrics(
            degraded,
            clean,
            v279.derive_d_numpy(clean),
        ),
        "degradation": {
            "selection_source": (
                "unlabeled strict v239 test-current top-k/entropy/TV/L2 statistics"
            ),
            "blur_s": BLUR_S,
            "blur_h": BLUR_H,
            "power_temperature": POWER_TEMPERATURE,
            "clean_official_means": v286.shape_means(clean),
            "degraded_official_means": v286.shape_means(degraded),
            "test_current_means": v286.shape_means(test_current),
        },
    }
    np.savez_compressed(
        SAMPLES_OUT,
        query_ids=query_ids,
        assigned_fold=assigned_fold,
        current_sh=degraded,
        clean_sh=clean,
    )
    return query_ids, assigned_fold, degraded, report


def configure_runner() -> None:
    v285.VERSION_NUMBER = VERSION_NUMBER
    v285.VERSION = VERSION
    v285.OUT_DIR = OUT_DIR
    v285.REPORT_DIR = REPORT_DIR
    v285.CHECKPOINT_DIR = CHECKPOINT_DIR
    v285.REPORT = REPORT
    v285.SAMPLES_OUT = SAMPLES_OUT
    v285.TARGETS_OUT = TARGETS_OUT
    v285.OUTPUT = OUTPUT
    v285.FINAL_SEEDS = (28701, 28731, 28761)
    v285.build_crossfit_samples = build_deconvolution_samples


def rewrite_report() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["method"] = VERSION
    report["primary_hypothesis"] = (
        "A universal AI deconvolution trained on controlled degradation of official truth "
        "can recover sharp physical support from the similarly resolved v239 carrier without "
        "learning the non-transferable component errors of a partial pseudo-current."
    )
    report["parameters"]["degradation"] = {
        "blur_s": BLUR_S,
        "blur_h": BLUR_H,
        "power_temperature": POWER_TEMPERATURE,
        "selection": "frozen from unlabeled test-carrier statistics",
    }
    report["compliance_boundary"]["uses"] = [
        "official train H S-H/PDP labels",
        "official train/test planar coordinates",
        "official point-cloud full wall-reflection geometry",
        "unlabeled strict v239 test-current statistics for degradation calibration",
        "strict v239 official-only carrier after official selection",
    ]
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    configure_runner()
    v285.main()
    rewrite_report()


if __name__ == "__main__":
    main()
