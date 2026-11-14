#!/usr/bin/env python3
"""Train the joint residual on OOF currents matched to test-carrier quality."""

from __future__ import annotations

import json

import numpy as np

import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v285_v239_isomorphic_joint_residual as v285
from project_paths import PATHS


VERSION_NUMBER = 286
VERSION = "v286_carrier_quality_matched_residual"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
SAMPLES_OUT = OUT_DIR / "v286_quality_matched_crossfit_current.npz"
TARGETS_OUT = OUT_DIR / "v286_joint_targets.npz"
OUTPUT = OUT_DIR / "v286_Round1_Test_Channel.npy"

BLUR_S = 0.12
BLUR_H = 0.24
POWER_TEMPERATURE = 0.95
ORIGINAL_BUILD = v285.build_crossfit_samples


def shape_means(sh: np.ndarray) -> dict[str, float]:
    value = v279.normalize(sh)
    flat = value.reshape(len(value), -1)
    d = value.sum(axis=2)
    h = value.sum(axis=1)
    sorted_sh = np.sort(flat, axis=1)[:, ::-1]
    sorted_d = np.sort(d, axis=1)[:, ::-1]
    eps = 1e-12
    return {
        "sh_top1": float(np.mean(sorted_sh[:, 0])),
        "sh_top4": float(np.mean(sorted_sh[:, :4].sum(axis=1))),
        "sh_top16": float(np.mean(sorted_sh[:, :16].sum(axis=1))),
        "sh_top64": float(np.mean(sorted_sh[:, :64].sum(axis=1))),
        "d_top1": float(np.mean(sorted_d[:, 0])),
        "d_top4": float(np.mean(sorted_d[:, :4].sum(axis=1))),
        "d_top16": float(np.mean(sorted_d[:, :16].sum(axis=1))),
        "sh_entropy": float(np.mean(-np.sum(flat * np.log(flat + eps), axis=1))),
        "d_entropy": float(np.mean(-np.sum(d * np.log(d + eps), axis=1))),
        "h_entropy": float(np.mean(-np.sum(h * np.log(h + eps), axis=1))),
        "tv_s": float(
            np.mean(np.sum(np.abs(value - np.roll(value, 1, axis=1)), axis=(1, 2)))
        ),
        "tv_h": float(
            np.mean(np.sum(np.abs(value - np.roll(value, 1, axis=2)), axis=(1, 2)))
        ),
        "l2": float(np.mean(np.sqrt(np.sum(flat * flat, axis=1)))),
    }


def quality_match(current: np.ndarray) -> np.ndarray:
    delay_blurred = (
        (1.0 - BLUR_S) * current
        + 0.5 * BLUR_S * (
            np.roll(current, 1, axis=1) + np.roll(current, -1, axis=1)
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


def build_quality_matched_samples(
    truth_sh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    query_ids, assigned_fold, raw_current, report = ORIGINAL_BUILD(truth_sh)
    matched = quality_match(raw_current)
    test_current, _ = v274.extract_current_marginals(v285.BASE)
    raw_baseline = report.pop("joint_current_baseline")
    report = {
        **report,
        "joint_current_baseline_raw": raw_baseline,
        "joint_current_baseline": v279.metrics(
            matched,
            truth_sh[query_ids],
            v279.derive_d_numpy(truth_sh[query_ids]),
        ),
        "quality_matching": {
            "selection_source": (
                "unlabeled strict v239 test-current top-k/entropy/TV/L2 statistics"
            ),
            "does_not_use": "local or hidden test truth",
            "blur_s": BLUR_S,
            "blur_h": BLUR_H,
            "power_temperature": POWER_TEMPERATURE,
            "raw_oof_means": shape_means(raw_current),
            "matched_oof_means": shape_means(matched),
            "test_current_means": shape_means(test_current),
        },
    }
    source = np.load(SAMPLES_OUT)
    np.savez_compressed(
        SAMPLES_OUT,
        query_ids=query_ids,
        assigned_fold=assigned_fold,
        reconstructed_anchor_sh=source["reconstructed_anchor_sh"],
        raw_current_sh=raw_current,
        current_sh=matched,
        current_d=v279.derive_d_numpy(matched),
        cached_anchor_d=source["cached_anchor_d"],
    )
    return query_ids, assigned_fold, matched, report


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
    v285.FINAL_SEEDS = (28601, 28631, 28661)
    v285.build_crossfit_samples = build_quality_matched_samples


def rewrite_report() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["method"] = VERSION
    report["primary_hypothesis"] = (
        "A residual learned from a partial v239 cross-fit current transfers only when the "
        "training current is corrupted to match the unlabeled deployed carrier's spectral "
        "quality, concentration, entropy, and circular smoothness."
    )
    report["parameters"]["quality_matching"] = {
        "blur_s": BLUR_S,
        "blur_h": BLUR_H,
        "power_temperature": POWER_TEMPERATURE,
        "selection": "frozen before model training from unlabeled test-carrier statistics",
    }
    report["compliance_boundary"]["uses"].append(
        "unlabeled strict v239 test-current shape statistics for corruption calibration"
    )
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    configure_runner()
    v285.main()
    rewrite_report()


if __name__ == "__main__":
    main()
