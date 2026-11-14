#!/usr/bin/env python3
"""Test whether official matched paths calibrate point-cloud geometry delays."""

from __future__ import annotations

import json

import numpy as np

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 332
VERSION = "v332_patch_delay_residual_field"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
SHUFFLE_REPEATS = 20


def weighted_mae(error: np.ndarray, weight: np.ndarray) -> float:
    return float(
        np.sum(np.abs(error).astype(np.float64) * weight.astype(np.float64))
        / np.maximum(np.sum(weight, dtype=np.float64), 1e-30)
    )


def main() -> None:
    gain_data = v325.recover_center_gains()
    rows = v325.matched_rows(gain_data)
    component = np.load(v297.COMPONENT_CACHE)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    observed_delay = component["delay"][rows["point"], rows["component"]].astype(np.float64)
    geometry_delay = (
        float(calibration["intercept_bins"])
        + float(calibration["slope_bins_per_meter"]) * rows["length"].astype(np.float64)
    )
    residual = observed_delay - geometry_delay
    train_xy = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    valid_points = np.unique(rows["point"])
    folds = v315.disjoint_folds(valid_points, train_xy)
    rng = np.random.default_rng(332)
    fold_rows = []

    for fold, query_points in enumerate(folds):
        fit_points = np.setdiff1d(valid_points, query_points)
        left, right, distance = v325.nearest_patch_pairs(
            rows, query_points, fit_points, train_xy
        )
        weight = np.sqrt(rows["mass"][left] * rows["mass"][right]).astype(np.float32)
        baseline_mae = weighted_mae(residual[left], weight)
        prediction_error = residual[right] - residual[left]
        predicted_mae = weighted_mae(prediction_error, weight)
        shuffle_mae = []
        for _ in range(SHUFFLE_REPEATS):
            shuffled = residual[right][rng.permutation(len(right))]
            shuffle_mae.append(weighted_mae(shuffled - residual[left], weight))
        reduction = (baseline_mae - predicted_mae) / max(baseline_mae, 1e-30)
        fold_rows.append(
            {
                "fold": fold,
                "pair_count": int(len(left)),
                "pair_distance": summarize(distance),
                "geometry_zero_residual_mae_bins": baseline_mae,
                "nearest_patch_residual_mae_bins": predicted_mae,
                "relative_mae_reduction": reduction,
                "shuffle_residual_mae_bins": summarize(np.asarray(shuffle_mae)),
            }
        )
        print(
            f"v332 fold={fold} baseline={baseline_mae:.4f} predicted={predicted_mae:.4f} "
            f"reduction={reduction:+.3%}",
            flush=True,
        )

    reduction = float(np.mean([row["relative_mae_reduction"] for row in fold_rows]))
    fold_wins = int(sum(row["relative_mae_reduction"] >= 0.30 for row in fold_rows))
    passed = reduction >= 0.30 and fold_wins >= 4
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The signed official-H minus point-cloud geometry delay is a reusable "
            "patch-specific spatial field, not independent candidate error."
        ),
        "parameters": {
            "pairing": "same patch nearest fit match within 25m",
            "shuffle_repeats": SHUFFLE_REPEATS,
        },
        "residual_bins": summarize(residual),
        "fold_rows": fold_rows,
        "summary": {
            "relative_mae_reduction": reduction,
            "fold_wins_at_30pct": fold_wins,
        },
        "decision_rule": {
            "relative_mae_reduction": ">=0.30",
            "fold_wins_at_30pct": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "decision": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
