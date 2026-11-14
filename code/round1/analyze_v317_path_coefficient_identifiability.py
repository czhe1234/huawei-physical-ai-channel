#!/usr/bin/env python3
"""Test whether default-concrete path coefficients explain official path mass."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import analyze_v273_planar_barrier_kernel as v273
import analyze_v293_interaction_physical_path_law as v293
import analyze_v312_highdepth_path_identity_match as v312
import analyze_v314_highdepth_path_sequence_reuse as v314
import analyze_v315_full_sequence_confirmatory as v315
import build_v316_pointcloud_path_coefficients as v316
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 317
VERSION = "v317_path_coefficient_identifiability"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
SHUFFLE_REPEATS = 20


def coefficient_features(
    coefficient: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    power = np.abs(coefficient.astype(np.complex128)) ** 2
    available = valid & (power > 0.0)
    log_power = np.log10(np.maximum(power, 1e-30))
    relative = np.zeros_like(log_power)
    rank = np.zeros_like(log_power)
    for point_id in range(len(power)):
        ids = np.flatnonzero(available[point_id])
        if len(ids) == 0:
            continue
        relative[point_id, ids] = log_power[point_id, ids] - np.max(log_power[point_id, ids])
        order = np.argsort(log_power[point_id, ids])
        rank[point_id, ids[order]] = (np.arange(len(ids)) + 0.5) / len(ids)
    return np.stack(
        [
            np.clip(log_power, -30.0, -5.0),
            np.clip(relative, -20.0, 0.0),
            rank,
            available.astype(np.float64),
        ],
        axis=2,
    ).astype(np.float32)


def model(numeric_dim: int):
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(range(numeric_dim))),
            (
                "identity",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                [numeric_dim],
            ),
        ]
    )
    return make_pipeline(transform, RidgeCV(alphas=np.asarray([0.1, 1.0, 10.0, 100.0])))


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    return {
        "rmse": rmse,
        "pearson": float(pearsonr(prediction, truth).statistic),
        "spearman": float(spearmanr(prediction, truth).statistic),
    }


def main() -> None:
    sequence = np.load(v314.SEQUENCE_CACHE)
    point_id = sequence["point_id"].astype(np.int64)
    candidate_id = sequence["candidate_id"].astype(np.int64)
    mass = sequence["mass"].astype(np.float32)
    target = np.log(np.maximum(mass, 1e-5)).astype(np.float64)
    first_patch = sequence["patch_id"][:, 0].astype(np.int64)
    first_interaction = sequence["interaction"][:, 0].astype(np.int64)
    identity = (first_interaction * 2000 + first_patch).astype(np.float64)

    path_data = np.load(v312.PATH_CACHE)
    interaction_data = np.load(v293.FEATURE_CACHE)
    geometry = interaction_data["train_path_features"][point_id, candidate_id].astype(np.float64)
    coefficient_data = np.load(v316.OUTPUT)
    all_coefficient = coefficient_data["train_coefficient"]
    all_coefficient_feature = coefficient_features(
        all_coefficient, path_data["train_valid"].astype(bool)
    )
    coefficient = all_coefficient_feature[point_id, candidate_id].astype(np.float64)
    geometry_x = np.concatenate([geometry, identity[:, None]], axis=1)
    augmented_x = np.concatenate([geometry, coefficient, identity[:, None]], axis=1)

    target_data = np.load(v273.TARGET_CACHE)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    rows = []
    all_truth = []
    all_base = []
    all_augmented = []
    all_log_power = []
    for fold, query_ids in enumerate(folds):
        query = np.isin(point_id, query_ids)
        fit = ~query
        geometry_model = model(geometry.shape[1])
        geometry_model.fit(geometry_x[fit], target[fit])
        base_prediction = geometry_model.predict(geometry_x[query])
        coefficient_model = model(geometry.shape[1] + coefficient.shape[1])
        coefficient_model.fit(augmented_x[fit], target[fit])
        augmented_prediction = coefficient_model.predict(augmented_x[query])
        base_metric = metrics(base_prediction, target[query])
        augmented_metric = metrics(augmented_prediction, target[query])
        rows.append(
            {
                "fold": fold,
                "fit_match_count": int(np.sum(fit)),
                "query_match_count": int(np.sum(query)),
                "geometry_firstwall": base_metric,
                "plus_default_concrete_coefficient": augmented_metric,
                "relative_rmse_reduction": (
                    base_metric["rmse"] - augmented_metric["rmse"]
                )
                / max(base_metric["rmse"], 1e-30),
            }
        )
        all_truth.append(target[query])
        all_base.append(base_prediction)
        all_augmented.append(augmented_prediction)
        all_log_power.append(coefficient[query, 0])

    truth = np.concatenate(all_truth)
    base_prediction = np.concatenate(all_base)
    augmented_prediction = np.concatenate(all_augmented)
    log_power = np.concatenate(all_log_power)
    pooled_base = metrics(base_prediction, truth)
    pooled_augmented = metrics(augmented_prediction, truth)
    reduction = (pooled_base["rmse"] - pooled_augmented["rmse"]) / pooled_base["rmse"]
    residual_correlation = float(pearsonr(log_power, truth - base_prediction).statistic)

    depth = path_data["train_interaction_count"][point_id, candidate_id].astype(np.int64)
    rng = np.random.default_rng(317)
    shuffled_rmse = []
    shuffled_residual_correlation = []
    for repeat in range(SHUFFLE_REPEATS):
        shuffled = coefficient.copy()
        for value in np.unique(depth):
            ids = np.flatnonzero(depth == value)
            shuffled[ids] = shuffled[rng.permutation(ids)]
        shuffled_x = np.concatenate([geometry, shuffled, identity[:, None]], axis=1)
        repeat_prediction = np.empty_like(target)
        for query_ids in folds:
            query = np.isin(point_id, query_ids)
            fit = ~query
            current = model(geometry.shape[1] + coefficient.shape[1])
            current.fit(shuffled_x[fit], target[fit])
            repeat_prediction[query] = current.predict(shuffled_x[query])
        shuffled_rmse.append(float(np.sqrt(np.mean((repeat_prediction - target) ** 2))))
        shuffled_residual_correlation.append(
            float(pearsonr(shuffled[:, 0], target - base_prediction).statistic)
        )
        print(f"v317 shuffle {repeat + 1}/{SHUFFLE_REPEATS}", flush=True)

    fold_wins = int(sum(row["relative_rmse_reduction"] > 0.0 for row in rows))
    passed = reduction >= 0.10 and fold_wins >= 4
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Default-concrete Sionna path coefficients explain official matched component "
            "mass beyond interaction geometry and first-wall identity."
        ),
        "match_count": int(len(target)),
        "coefficient_available_fraction": float(np.mean(coefficient[:, 3] > 0.5)),
        "fold_rows": rows,
        "pooled": {
            "geometry_firstwall": pooled_base,
            "plus_default_concrete_coefficient": pooled_augmented,
            "relative_rmse_reduction": float(reduction),
            "fold_wins": fold_wins,
            "log_power_vs_geometry_residual_pearson": residual_correlation,
        },
        "depth_stratified_shuffle": {
            "rmse": summarize(np.asarray(shuffled_rmse)),
            "log_power_vs_geometry_residual_pearson": summarize(
                np.asarray(shuffled_residual_correlation)
            ),
        },
        "decision_rule": {
            "relative_rmse_reduction": ">=0.10",
            "fold_wins": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
