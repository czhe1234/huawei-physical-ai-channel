#!/usr/bin/env python3
"""Confirm full-path energy identity on mutually exclusive isolation-matched folds."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v273_planar_barrier_kernel as v273
import analyze_v314_highdepth_path_sequence_reuse as v314
from project_paths import PATHS


VERSION_NUMBER = 315
VERSION = "v315_full_sequence_confirmatory"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
FOLD_COUNT = 5
STRATA = 20


def disjoint_folds(valid_ids: np.ndarray, train_xy: np.ndarray) -> list[np.ndarray]:
    nearest = cKDTree(train_xy[valid_ids]).query(train_xy[valid_ids], k=2)[0][:, 1]
    edges = np.quantile(nearest, np.linspace(0.0, 1.0, STRATA + 1))
    stratum = np.clip(np.searchsorted(edges[1:-1], nearest, side="right"), 0, STRATA - 1)
    rng = np.random.default_rng(315)
    folds: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    for value in range(STRATA):
        ids = valid_ids[stratum == value].copy()
        rng.shuffle(ids)
        offset = value % FOLD_COUNT
        for index, point_id in enumerate(ids):
            folds[(index + offset) % FOLD_COUNT].append(int(point_id))
    return [np.sort(np.asarray(ids, dtype=np.int64)) for ids in folds]


def group_means(key: np.ndarray, value: np.ndarray) -> dict[str, float]:
    return {
        str(item): float(np.mean(value[key == item]))
        for item in np.unique(key)
    }


def evaluate_fold(
    point_id: np.ndarray,
    mass: np.ndarray,
    first_key: np.ndarray,
    full_key: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
) -> dict[str, float]:
    fit = np.isin(point_id, fit_ids)
    query = np.isin(point_id, query_ids)
    fit_log = np.log(np.maximum(mass[fit], 1e-5))
    truth = np.log(np.maximum(mass[query], 1e-5))
    global_mean = float(np.mean(fit_log))
    first_mean = group_means(first_key[fit], fit_log)
    full_mean = group_means(full_key[fit], fit_log)
    first_prediction = np.asarray(
        [first_mean.get(str(key), global_mean) for key in first_key[query]]
    )
    full_seen = np.asarray([str(key) in full_mean for key in full_key[query]])
    full_prediction = np.asarray(
        [
            full_mean.get(
                str(full), first_mean.get(str(first), global_mean)
            )
            for first, full in zip(first_key[query], full_key[query])
        ]
    )
    first_rmse = float(np.sqrt(np.mean((first_prediction - truth) ** 2)))
    full_rmse = float(np.sqrt(np.mean((full_prediction - truth) ** 2)))
    query_mass = mass[query]
    coverage = float(
        np.sum(query_mass[full_seen]) / np.maximum(np.sum(query_mass), 1e-30)
    )
    return {
        "fit_match_count": int(np.sum(fit)),
        "query_match_count": int(np.sum(query)),
        "full_mass_coverage": coverage,
        "first1_log_rmse": first_rmse,
        "hierarchical_full_log_rmse": full_rmse,
        "relative_rmse_reduction": (first_rmse - full_rmse) / max(first_rmse, 1e-30),
    }


def main() -> None:
    data = np.load(v314.SEQUENCE_CACHE)
    point_id = data["point_id"].astype(np.int64)
    mass = data["mass"].astype(np.float32)
    first_key = data["key_first1"]
    full_key = data["key_full"]
    target = np.load(v273.TARGET_CACHE)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    folds = disjoint_folds(valid_ids, train_pos[:, :2])
    rows = []
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        row = evaluate_fold(
            point_id, mass, first_key, full_key, fit_ids, query_ids
        )
        row["fold"] = fold
        row["fit_ue_count"] = int(len(fit_ids))
        row["query_ue_count"] = int(len(query_ids))
        rows.append(row)
    coverage = float(np.mean([row["full_mass_coverage"] for row in rows]))
    reduction = float(np.mean([row["relative_rmse_reduction"] for row in rows]))
    wins = int(sum(row["relative_rmse_reduction"] >= 0.05 for row in rows))
    passed = coverage >= 0.50 and reduction >= 0.05 and wins >= 4
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A complete ordered interaction/patch chain predicts high-depth path energy "
            "better than first-wall identity alone."
        ),
        "fold_design": (
            "Five mutually exclusive folds stratified by leave-one-out nearest-neighbor "
            "distance; every valid UE appears in exactly one query fold."
        ),
        "fold_rows": rows,
        "summary": {
            "full_mass_coverage": coverage,
            "relative_rmse_reduction": reduction,
            "fold_wins_at_5pct": wins,
        },
        "decision_rule": {
            "full_mass_coverage": ">=0.50",
            "relative_rmse_reduction": ">=0.05",
            "fold_wins_at_5pct": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
