#!/usr/bin/env python3
"""Confirm v325 path phase transport on complete complex gain tensors."""

from __future__ import annotations

import json

import numpy as np

import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v326_complex_path_subspace_transport as v326
from project_paths import PATHS


VERSION_NUMBER = 327
VERSION = "v327_full_tensor_phase_transport"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"


def real_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = prediction.reshape(len(prediction), -1).astype(np.complex128)
    target = truth.reshape(len(truth), -1).astype(np.complex128)
    numerator = np.real(np.sum(np.conj(pred) * target, axis=1))
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    return numerator / np.maximum(denominator, 1e-30)


def absolute_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = prediction.reshape(len(prediction), -1).astype(np.complex128)
    target = truth.reshape(len(truth), -1).astype(np.complex128)
    numerator = np.abs(np.sum(np.conj(pred) * target, axis=1))
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    return numerator / np.maximum(denominator, 1e-30)


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(
        np.sum(value.astype(np.float64) * weight.astype(np.float64))
        / np.maximum(np.sum(weight, dtype=np.float64), 1e-30)
    )


def evaluate_variant(
    prediction: np.ndarray, truth: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    return {
        "real_complex_cosine": weighted_mean(real_cosine(prediction, truth), weight),
        "phase_invariant_cosine": weighted_mean(
            absolute_cosine(prediction, truth), weight
        ),
    }


def main() -> None:
    gain_data = v325.recover_center_gains()
    rows = v325.matched_rows(gain_data)
    gain = v326.recover_matched_gain_tensors(rows)
    train_xy = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    valid_points = np.unique(rows["point"])
    folds = v315.disjoint_folds(valid_points, train_xy)
    h_coordinate = np.arange(16, dtype=np.float32) - 7.5
    fold_rows = []

    for fold, query_points in enumerate(folds):
        fit_points = np.setdiff1d(valid_points, query_points)
        fit_left, fit_right, _ = v325.nearest_patch_pairs(
            rows, fit_points, fit_points, train_xy
        )
        fit_phase, fit_length, fit_weight = v325.pair_values(
            rows, fit_left, fit_right
        )
        beta = v325.fit_beta(fit_phase, fit_length, fit_weight)
        left, right, distance = v325.nearest_patch_pairs(
            rows, query_points, fit_points, train_xy
        )
        truth = gain[left]
        raw = gain[right].copy()
        delta_length = rows["length"][left] - rows["length"][right]
        scalar = raw * np.exp(-1j * beta * delta_length)[:, None, None, None, None]
        delta_h = np.remainder(
            rows["h_center"][left] - rows["h_center"][right] + 8.0, 16.0
        ) - 8.0
        h_phase = np.exp(
            1j * 2.0 * np.pi * delta_h[:, None] * h_coordinate[None, :] / 16.0
        ).astype(np.complex64)
        scalar_h = scalar * h_phase[:, None, :, None, None]
        weight = np.sqrt(rows["mass"][left] * rows["mass"][right]).astype(np.float32)
        raw_metrics = evaluate_variant(raw, truth, weight)
        scalar_metrics = evaluate_variant(scalar, truth, weight)
        scalar_h_metrics = evaluate_variant(scalar_h, truth, weight)
        fold_rows.append(
            {
                "fold": fold,
                "beta": beta,
                "pair_count": int(len(left)),
                "pair_distance_mean": float(np.mean(distance)),
                "raw": raw_metrics,
                "scalar_phase": scalar_metrics,
                "scalar_phase_h_steering": scalar_h_metrics,
                "scalar_real_cosine_gain": scalar_metrics["real_complex_cosine"]
                - raw_metrics["real_complex_cosine"],
                "scalar_h_real_cosine_gain": scalar_h_metrics["real_complex_cosine"]
                - raw_metrics["real_complex_cosine"],
            }
        )
        print(
            f"v327 fold={fold} raw={raw_metrics['real_complex_cosine']:.4f} "
            f"scalar={scalar_metrics['real_complex_cosine']:.4f} "
            f"scalar_h={scalar_h_metrics['real_complex_cosine']:.4f}",
            flush=True,
        )

    scalar_gain = float(np.mean([row["scalar_real_cosine_gain"] for row in fold_rows]))
    scalar_h_gain = float(
        np.mean([row["scalar_h_real_cosine_gain"] for row in fold_rows])
    )
    scalar_wins = int(sum(row["scalar_real_cosine_gain"] > 0 for row in fold_rows))
    passed = scalar_gain >= 0.15 and scalar_wins >= 4
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The v325 path-length carrier phase transports the complete official-H "
            "2x16x8x4 path gain tensor, not only its array-center scalar."
        ),
        "fold_rows": fold_rows,
        "summary": {
            "scalar_real_cosine_gain": scalar_gain,
            "scalar_h_real_cosine_gain": scalar_h_gain,
            "scalar_fold_wins": scalar_wins,
        },
        "decision_rule": {
            "scalar_real_cosine_gain": ">=0.15",
            "scalar_fold_wins": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
