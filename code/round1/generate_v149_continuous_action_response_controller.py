#!/usr/bin/env python3
"""Generate v149 with an EM-constrained continuous action controller."""

from __future__ import annotations

import json
import zipfile
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from analyze_post_v143_hidden_information import normalize_without_floor, support_score
from analyze_v145_gated_local_h_field import local_h_predictions, stable_features
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    digest,
    output_comparison,
)
from generate_v146_continuous_diffusion_gate import summarize_gain
from generate_v147_continuous_local_action_gate import (
    adjusted_support,
    apply_complex_local_field,
    local_descriptors,
)
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v149_continuous_action_response_controller"
OUT_DIR = PATHS.cache_dir(149)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(149)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v149_Round1_Test_Channel.npy"
DECISION_OUT = OUT_DIR / "v149_continuous_action_response_decisions.npz"
REPORT = REPORT_DIR / "v149_continuous_action_response_controller_generate.json"
ZIP_OUT = (
    PATHS.round1_versions
    / "v149_continuous_action_response_controller_candidate_pending.zip"
)

RANDOM_STATE = 14901
K = 8
MAX_DISTANCE = 20.0
STRENGTH_GRID = np.linspace(0.05, 0.60, 12, dtype=np.float32)
POSITIVE_THRESHOLD = 0.00025
SEEDS = (14901, 14931)
INNER_SEEDS = (14903, 14937)
REGIMES = ("clear", "light", "medium", "heavy")

CONFIGS = tuple(
    (probability, penalty, minimum_lcb, uncertainty_quantile)
    for probability in (0.60, 0.70, 0.80)
    for penalty in (0.75, 1.25, 1.75)
    for minimum_lcb in (0.0, 0.0010, 0.0020)
    for uncertainty_quantile in (1.00, 0.85, 0.75)
)


def config_name(config: tuple[float, float, float, float]) -> str:
    probability, penalty, minimum_lcb, uncertainty_quantile = config
    return (
        f"p{probability:.2f}_lcb{penalty:.2f}_"
        f"min{minimum_lcb:.4f}_uq{uncertainty_quantile:.2f}"
    )


def make_action_features(
    point_features: np.ndarray, strengths: np.ndarray = STRENGTH_GRID
) -> np.ndarray:
    repeated = np.repeat(point_features, len(strengths), axis=0)
    tiled = np.tile(strengths, len(point_features)).astype(np.float32)
    action_terms = np.column_stack(
        [tiled, tiled * tiled, np.sqrt(tiled)]
    ).astype(np.float32)
    values = np.concatenate([repeated, action_terms], axis=1)
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def dense_action_gains(
    current: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    maximum_distance: np.ndarray,
) -> np.ndarray:
    base_score = support_score(current, truth)
    gains = np.zeros((len(current), len(STRENGTH_GRID)), dtype=np.float32)
    active = maximum_distance <= MAX_DISTANCE
    for index, strength in enumerate(STRENGTH_GRID):
        adjusted = adjusted_support(
            current,
            prediction,
            np.full(len(current), strength, dtype=np.float32),
            active,
        )
        gains[:, index] = support_score(adjusted, truth) - base_score
    return gains


def fit_surface_ensemble(
    features: np.ndarray,
    gains: np.ndarray,
    positive: np.ndarray,
    train_index: np.ndarray,
    evaluation_features: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    benefit_predictions = []
    probability_predictions = []
    train_action_features = make_action_features(features[train_index])
    train_gain = gains[train_index].reshape(-1).astype(np.float32)
    evaluation_action_features = make_action_features(evaluation_features)
    for seed in seeds:
        regressor = ExtraTreesRegressor(
            n_estimators=96,
            max_features=0.60,
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=-1,
        )
        classifier = ExtraTreesClassifier(
            n_estimators=96,
            max_features=0.60,
            min_samples_leaf=16,
            class_weight="balanced_subsample",
            random_state=seed + 1000,
            n_jobs=-1,
        )
        regressor.fit(train_action_features, train_gain)
        classifier.fit(features[train_index], positive[train_index])
        benefit_predictions.append(
            regressor.predict(evaluation_action_features)
            .reshape(len(evaluation_features), len(STRENGTH_GRID))
            .astype(np.float32)
        )
        probability = classifier.predict_proba(evaluation_features)
        if 1 in classifier.classes_:
            column = int(np.flatnonzero(classifier.classes_ == 1)[0])
            probability_predictions.append(probability[:, column].astype(np.float32))
        else:
            probability_predictions.append(
                np.zeros(len(evaluation_features), dtype=np.float32)
            )
    benefits = np.stack(benefit_predictions, axis=0)
    probabilities = np.stack(probability_predictions, axis=0)
    return (
        benefits.mean(axis=0).astype(np.float32),
        benefits.std(axis=0).astype(np.float32),
        probabilities.mean(axis=0).astype(np.float32),
        probabilities.std(axis=0).astype(np.float32),
    )


def choose_strengths(
    benefit: np.ndarray,
    uncertainty: np.ndarray,
    probability: np.ndarray,
    maximum_distance: np.ndarray,
    config: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    probability_threshold, penalty, minimum_lcb, uncertainty_quantile = config
    lower_bound = benefit - penalty * uncertainty
    best_index = np.argmax(lower_bound, axis=1)
    rows = np.arange(len(best_index))
    best_lower_bound = lower_bound[rows, best_index].astype(np.float32)
    best_uncertainty = uncertainty[rows, best_index].astype(np.float32)
    within_range = maximum_distance <= MAX_DISTANCE
    if uncertainty_quantile >= 0.999 or not np.any(within_range):
        uncertainty_threshold = float("inf")
    else:
        uncertainty_threshold = float(
            np.quantile(best_uncertainty[within_range], uncertainty_quantile)
        )
    active = (
        within_range
        & (probability >= probability_threshold)
        & (best_lower_bound > minimum_lcb)
        & (best_uncertainty <= uncertainty_threshold)
    )
    selected_index = np.where(active, best_index + 1, 0).astype(np.int16)
    strength = np.where(active, STRENGTH_GRID[best_index], 0.0).astype(np.float32)
    return (
        strength,
        selected_index,
        best_lower_bound,
        best_uncertainty,
        uncertainty_threshold,
    )


def realized_gain(selected_index: np.ndarray, gains: np.ndarray) -> np.ndarray:
    output = np.zeros(len(selected_index), dtype=np.float32)
    active = selected_index > 0
    output[active] = gains[
        np.flatnonzero(active), selected_index[active].astype(np.int64) - 1
    ]
    return output


def select_config(
    benefit: np.ndarray,
    uncertainty: np.ndarray,
    probability: np.ndarray,
    gains: np.ndarray,
    valid: np.ndarray,
    maximum_distance: np.ndarray,
) -> tuple[tuple[float, float, float, float], dict[str, object]]:
    scored = []
    best_config = CONFIGS[0]
    best_key = (-np.inf, -np.inf, 0.0)
    for config in CONFIGS:
        _, selected, _, _, uncertainty_threshold = choose_strengths(
            benefit, uncertainty, probability, maximum_distance, config
        )
        gain = realized_gain(selected, gains)
        active = valid & (selected > 0)
        full_gain = float(np.mean(gain[valid])) if np.any(valid) else 0.0
        active_gain = float(np.mean(gain[active])) if np.any(active) else 0.0
        active_fraction = float(np.mean(active[valid])) if np.any(valid) else 0.0
        positive_fraction = (
            float(np.mean(gain[active] > 0.0)) if np.any(active) else 0.0
        )
        row = {
            "config": config_name(config),
            "full_gain_mean": full_gain,
            "active_gain_mean": active_gain,
            "active_fraction": active_fraction,
            "active_positive_fraction": positive_fraction,
            "uncertainty_threshold": uncertainty_threshold,
        }
        scored.append(row)
        key = (full_gain, active_gain, -active_fraction)
        if key > best_key:
            best_key = key
            best_config = config
    scored.sort(key=lambda item: item["full_gain_mean"], reverse=True)
    return best_config, {"searched_configs": len(CONFIGS), "top_configs": scored[:12]}


def nested_predictions(
    positions_train: np.ndarray,
    positions_test: np.ndarray,
    current: np.ndarray,
    truth: np.ndarray,
    h_target: np.ndarray,
    base_features_train: np.ndarray,
    base_features_test: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions_train[:, :2])
    selected_valid = np.flatnonzero(valid)
    outer = GroupKFold(n_splits=5)
    oof_benefit = np.zeros((len(current), len(STRENGTH_GRID)), dtype=np.float32)
    oof_uncertainty = np.zeros_like(oof_benefit)
    oof_probability = np.zeros(len(current), dtype=np.float32)
    oof_probability_uncertainty = np.zeros(len(current), dtype=np.float32)
    oof_strength = np.zeros(len(current), dtype=np.float32)
    oof_selected = np.zeros(len(current), dtype=np.int16)
    oof_lower_bound = np.zeros(len(current), dtype=np.float32)
    oof_selected_uncertainty = np.zeros(len(current), dtype=np.float32)
    oof_maximum_distance = np.zeros(len(current), dtype=np.float32)
    oof_realized_gain = np.zeros(len(current), dtype=np.float32)
    oof_action_gains = np.zeros_like(oof_benefit)
    oof_fold = np.full(len(current), -1, dtype=np.int16)
    test_benefit_folds = []
    test_uncertainty_folds = []
    test_probability_folds = []
    test_probability_uncertainty_folds = []
    test_prediction_folds = []
    test_dispersion_folds = []
    test_distance_folds = []
    selected_configs = []
    fold_reports = []

    for outer_fold, (outer_train_local, outer_validation_local) in enumerate(
        outer.split(selected_valid, groups=groups[selected_valid])
    ):
        outer_train = selected_valid[outer_train_local]
        outer_validation = selected_valid[outer_validation_local]
        inner_prediction = np.zeros((len(current), h_target.shape[1]), dtype=np.float32)
        inner_dispersion = np.zeros(len(current), dtype=np.float32)
        inner_distance = np.zeros(len(current), dtype=np.float32)
        inner = GroupKFold(n_splits=4)
        for inner_train_local, inner_validation_local in inner.split(
            outer_train, groups=groups[outer_train]
        ):
            inner_train = outer_train[inner_train_local]
            inner_validation = outer_train[inner_validation_local]
            prediction, dispersion, distance = local_h_predictions(
                positions_train, h_target, inner_train, inner_validation
            )
            inner_prediction[inner_validation] = prediction[K]
            inner_dispersion[inner_validation] = dispersion[K]
            inner_distance[inner_validation] = distance[K]

        inner_gains = dense_action_gains(
            current[outer_train],
            truth[outer_train],
            inner_prediction[outer_train],
            inner_distance[outer_train],
        )
        inner_positive = (
            inner_gains.max(axis=1) > POSITIVE_THRESHOLD
        ).astype(np.int8)
        inner_features = local_descriptors(
            base_features_train[outer_train],
            inner_prediction[outer_train],
            inner_dispersion[outer_train],
            inner_distance[outer_train],
        )

        local_count = len(outer_train)
        config_benefit = np.zeros((local_count, len(STRENGTH_GRID)), dtype=np.float32)
        config_uncertainty = np.zeros_like(config_benefit)
        config_probability = np.zeros(local_count, dtype=np.float32)
        config_probability_uncertainty = np.zeros(local_count, dtype=np.float32)
        for train_local, validation_local in inner.split(
            np.arange(local_count), groups=groups[outer_train]
        ):
            prediction = fit_surface_ensemble(
                inner_features,
                inner_gains,
                inner_positive,
                train_local,
                inner_features[validation_local],
                INNER_SEEDS,
            )
            config_benefit[validation_local] = prediction[0]
            config_uncertainty[validation_local] = prediction[1]
            config_probability[validation_local] = prediction[2]
            config_probability_uncertainty[validation_local] = prediction[3]

        fold_config, config_report = select_config(
            config_benefit,
            config_uncertainty,
            config_probability,
            inner_gains,
            np.ones(local_count, dtype=bool),
            inner_distance[outer_train],
        )
        selected_configs.append(fold_config)

        validation_prediction, validation_dispersion, validation_distance = (
            local_h_predictions(
                positions_train, h_target, outer_train, outer_validation
            )
        )
        validation_features = local_descriptors(
            base_features_train[outer_validation],
            validation_prediction[K],
            validation_dispersion[K],
            validation_distance[K],
        )
        train_index = np.arange(local_count, dtype=np.int64)
        validation_model = fit_surface_ensemble(
            inner_features,
            inner_gains,
            inner_positive,
            train_index,
            validation_features,
            SEEDS,
        )
        (
            validation_strength,
            validation_selected,
            validation_lower_bound,
            validation_selected_uncertainty,
            validation_uncertainty_threshold,
        ) = choose_strengths(
            validation_model[0],
            validation_model[1],
            validation_model[2],
            validation_distance[K],
            fold_config,
        )
        validation_gains = dense_action_gains(
            current[outer_validation],
            truth[outer_validation],
            validation_prediction[K],
            validation_distance[K],
        )
        validation_realized = realized_gain(validation_selected, validation_gains)

        oof_benefit[outer_validation] = validation_model[0]
        oof_uncertainty[outer_validation] = validation_model[1]
        oof_probability[outer_validation] = validation_model[2]
        oof_probability_uncertainty[outer_validation] = validation_model[3]
        oof_strength[outer_validation] = validation_strength
        oof_selected[outer_validation] = validation_selected
        oof_lower_bound[outer_validation] = validation_lower_bound
        oof_selected_uncertainty[outer_validation] = validation_selected_uncertainty
        oof_maximum_distance[outer_validation] = validation_distance[K]
        oof_realized_gain[outer_validation] = validation_realized
        oof_action_gains[outer_validation] = validation_gains
        oof_fold[outer_validation] = outer_fold

        test_prediction, test_dispersion, test_distance = local_h_predictions(
            np.concatenate([positions_train, positions_test], axis=0),
            np.concatenate(
                [
                    h_target,
                    np.zeros(
                        (len(positions_test), h_target.shape[1]), dtype=np.float32
                    ),
                ],
                axis=0,
            ),
            outer_train,
            np.arange(len(positions_train), len(positions_train) + len(positions_test)),
        )
        test_features = local_descriptors(
            base_features_test,
            test_prediction[K],
            test_dispersion[K],
            test_distance[K],
        )
        test_model = fit_surface_ensemble(
            inner_features,
            inner_gains,
            inner_positive,
            train_index,
            test_features,
            SEEDS,
        )
        test_benefit_folds.append(test_model[0])
        test_uncertainty_folds.append(test_model[1])
        test_probability_folds.append(test_model[2])
        test_probability_uncertainty_folds.append(test_model[3])
        test_prediction_folds.append(test_prediction[K])
        test_dispersion_folds.append(test_dispersion[K])
        test_distance_folds.append(test_distance[K])

        fold_reports.append(
            {
                "fold": outer_fold,
                "selected_config": config_name(fold_config),
                "train_count": int(len(outer_train)),
                "validation_count": int(len(outer_validation)),
                "validation_gain": summarize_gain(
                    validation_realized,
                    np.ones(len(validation_realized), dtype=bool),
                ),
                "active_fraction": float(np.mean(validation_selected > 0)),
                "active_gain": (
                    summarize_gain(
                        validation_realized,
                        validation_selected > 0,
                    )
                    if np.any(validation_selected > 0)
                    else {}
                ),
                "selected_strength": summarize(validation_strength),
                "continuous_risk": summarize(validation_model[2]),
                "selected_lower_bound": summarize(validation_lower_bound),
                "selected_uncertainty": summarize(validation_selected_uncertainty),
                "uncertainty_threshold": validation_uncertainty_threshold,
                "inner_selection": config_report,
            }
        )

    config_counts = Counter(config_name(config) for config in selected_configs)
    modal_name = config_counts.most_common(1)[0][0]
    modal_config = next(config for config in CONFIGS if config_name(config) == modal_name)

    test_benefit_stack = np.stack(test_benefit_folds, axis=0)
    test_benefit = test_benefit_stack.mean(axis=0).astype(np.float32)
    test_uncertainty = np.sqrt(
        np.maximum(
            np.mean(np.stack(test_uncertainty_folds, axis=0) ** 2, axis=0)
            + test_benefit_stack.var(axis=0),
            0.0,
        )
    ).astype(np.float32)
    test_probability_stack = np.stack(test_probability_folds, axis=0)
    test_probability = test_probability_stack.mean(axis=0).astype(np.float32)
    test_probability_uncertainty = np.sqrt(
        np.maximum(
            np.mean(np.stack(test_probability_uncertainty_folds, axis=0) ** 2, axis=0)
            + test_probability_stack.var(axis=0),
            0.0,
        )
    ).astype(np.float32)
    test_prediction = np.stack(test_prediction_folds, axis=0).mean(axis=0).astype(
        np.float32
    )
    test_dispersion = np.stack(test_dispersion_folds, axis=0).mean(axis=0).astype(
        np.float32
    )
    test_distance = np.stack(test_distance_folds, axis=0).mean(axis=0).astype(
        np.float32
    )
    (
        test_strength,
        test_selected,
        test_lower_bound,
        test_selected_uncertainty,
        test_uncertainty_threshold,
    ) = choose_strengths(
        test_benefit,
        test_uncertainty,
        test_probability,
        test_distance,
        modal_config,
    )

    return {
        "groups": groups,
        "oof_benefit": oof_benefit,
        "oof_uncertainty": oof_uncertainty,
        "oof_probability": oof_probability,
        "oof_probability_uncertainty": oof_probability_uncertainty,
        "oof_strength": oof_strength,
        "oof_selected": oof_selected,
        "oof_lower_bound": oof_lower_bound,
        "oof_selected_uncertainty": oof_selected_uncertainty,
        "oof_maximum_distance": oof_maximum_distance,
        "oof_realized_gain": oof_realized_gain,
        "oof_action_gains": oof_action_gains,
        "oof_fold": oof_fold,
        "test_benefit": test_benefit,
        "test_uncertainty": test_uncertainty,
        "test_probability": test_probability,
        "test_probability_uncertainty": test_probability_uncertainty,
        "test_prediction": test_prediction,
        "test_dispersion": test_dispersion,
        "test_distance": test_distance,
        "test_strength": test_strength,
        "test_selected": test_selected,
        "test_lower_bound": test_lower_bound,
        "test_selected_uncertainty": test_selected_uncertainty,
        "test_uncertainty_threshold": test_uncertainty_threshold,
        "modal_config": modal_config,
        "config_counts": dict(config_counts),
        "fold_reports": fold_reports,
    }


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(
        np.float32
    )
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(
        np.float32
    )
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    residual_probability = np.load(
        PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
    ).astype(np.float32)
    source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    risk = build_risk(
        regimes_train,
        residual_probability,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    current, _ = apply_risk_shift(source, risk)
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    valid = truth_raw.sum(axis=(1, 2, 3)) > 0
    truth = normalize_without_floor(truth_raw)
    current_h = current.sum(axis=(2, 3))
    truth_h = truth.sum(axis=(2, 3))
    h_target = np.clip(
        np.log(np.maximum(truth_h, 1e-10))
        - np.log(np.maximum(current_h, 1e-10)),
        -3.0,
        3.0,
    ).astype(np.float32)
    base_features_train = stable_features(
        "train",
        positions_train,
        blocked_train,
        current,
        geodesic,
        pointcloud,
    )
    test_target = normalize_support(
        np.load(PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"),
        1e-12,
    )
    base_features_test = stable_features(
        "test",
        positions_test,
        blocked_test,
        test_target,
        geodesic,
        pointcloud,
    )
    nested = nested_predictions(
        positions_train,
        positions_test,
        current,
        truth,
        h_target,
        base_features_train,
        base_features_test,
        valid,
    )

    active_oof = valid & (nested["oof_selected"] > 0)
    active_test = nested["test_selected"] > 0
    fold_gain_means = [
        row["validation_gain"]["mean"] for row in nested["fold_reports"]
    ]
    application = apply_complex_local_field(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        OUTPUT,
        positions_test,
        nested["test_prediction"],
        nested["test_strength"],
    )
    output = np.load(OUTPUT, mmap_mode="r")
    baseline = np.load(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", mmap_mode="r"
    )
    comparison = output_comparison(output, baseline, regimes_test)
    if output.shape != baseline.shape or output.dtype != np.complex64:
        raise ValueError(
            f"invalid output layout: shape={output.shape} dtype={output.dtype}"
        )
    if not comparison["finite"]:
        raise ValueError("v149 output contains non-finite values")

    np.savez_compressed(
        DECISION_OUT,
        strengths=STRENGTH_GRID,
        valid=valid,
        train_oof_benefit=nested["oof_benefit"],
        train_oof_uncertainty=nested["oof_uncertainty"],
        train_oof_probability=nested["oof_probability"],
        train_oof_probability_uncertainty=nested["oof_probability_uncertainty"],
        train_oof_strength=nested["oof_strength"],
        train_oof_selected=nested["oof_selected"],
        train_oof_lower_bound=nested["oof_lower_bound"],
        train_oof_selected_uncertainty=nested["oof_selected_uncertainty"],
        train_oof_maximum_distance=nested["oof_maximum_distance"],
        train_oof_realized_gain=nested["oof_realized_gain"],
        train_oof_action_gains=nested["oof_action_gains"],
        train_oof_fold=nested["oof_fold"],
        test_benefit=nested["test_benefit"],
        test_uncertainty=nested["test_uncertainty"],
        test_probability=nested["test_probability"],
        test_probability_uncertainty=nested["test_probability_uncertainty"],
        test_prediction=nested["test_prediction"],
        test_dispersion=nested["test_dispersion"],
        test_distance=nested["test_distance"],
        test_strength=nested["test_strength"],
        test_selected=nested["test_selected"],
        test_lower_bound=nested["test_lower_bound"],
        test_selected_uncertainty=nested["test_selected_uncertainty"],
    )

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352",
        "primary_hypothesis": (
            "learn an EM-constrained continuous action policy: independent "
            "risk probability R(x), action-response benefit B(x,a), and "
            "benefit uncertainty U(x,a) choose local H residual strength"
        ),
        "physical_interpretation": (
            "local H residual direction is treated as a low-dimensional "
            "angle-support correction; the controller estimates how strongly "
            "to apply it under NLOS multipath uncertainty while preserving "
            "complex phase and total power"
        ),
        "only_primary_change": (
            "replace v145 fixed strength with a nested spatial OOF policy "
            "that searches a dense continuous strength grid by maximizing "
            "expected_gain - lambda * uncertainty"
        ),
        "model": {
            "base_feature_dimension": int(base_features_train.shape[1]),
            "decision_feature_dimension": int(
                base_features_train.shape[1] + 16 + 7 + 3
            ),
            "local_neighbors": K,
            "maximum_distance": MAX_DISTANCE,
            "strength_grid": {
                "count": int(len(STRENGTH_GRID)),
                "min": float(STRENGTH_GRID.min()),
                "max": float(STRENGTH_GRID.max()),
                "values": [float(value) for value in STRENGTH_GRID],
            },
            "positive_threshold": POSITIVE_THRESHOLD,
            "outputs": [
                "R(x): positive-action probability",
                "B(x,a): expected full-support gain over continuous strength",
                "U(x,a): ensemble gain uncertainty over continuous strength",
            ],
            "selection": "argmax_a B(x,a)-lambda*U(x,a)",
            "hard_regime_gate_for_new_layer": False,
            "conserves_total_power": True,
            "preserves_complex_phase_skeleton": True,
        },
        "nested_spatial_oof": {
            "modal_config": config_name(nested["modal_config"]),
            "config_counts": nested["config_counts"],
            "folds": nested["fold_reports"],
            "fold_gain_means": fold_gain_means,
            "mean_of_fold_means": float(np.mean(fold_gain_means)),
            "all_folds_positive": bool(np.all(np.asarray(fold_gain_means) > 0)),
            "full_gain": summarize_gain(nested["oof_realized_gain"], valid),
            "active_fraction": float(np.mean(active_oof[valid])),
            "active_gain": (
                summarize_gain(nested["oof_realized_gain"], active_oof)
                if np.any(active_oof)
                else {}
            ),
            "selected_strength": summarize(nested["oof_strength"][valid]),
            "by_regime": {
                regime: {
                    "count": int(np.sum(valid & (regimes_train == regime))),
                    "gain": summarize_gain(
                        nested["oof_realized_gain"],
                        valid & (regimes_train == regime),
                    ),
                    "active_fraction": float(
                        np.mean(active_oof[valid & (regimes_train == regime)])
                    ),
                }
                for regime in REGIMES
            },
        },
        "test_decision": {
            "continuous_risk_probability": summarize(nested["test_probability"]),
            "probability_uncertainty": summarize(
                nested["test_probability_uncertainty"]
            ),
            "maximum_expected_benefit": summarize(
                np.max(nested["test_benefit"], axis=1)
            ),
            "selected_lower_bound": summarize(nested["test_lower_bound"]),
            "selected_uncertainty": summarize(nested["test_selected_uncertainty"]),
            "uncertainty_threshold": nested["test_uncertainty_threshold"],
            "active_fraction": float(np.mean(active_test)),
            "selected_strength": summarize(nested["test_strength"]),
            "active_by_legacy_regime_diagnostic_only": {
                regime: float(np.mean(active_test[regimes_test == regime]))
                for regime in REGIMES
            },
            "local_prediction": summarize(nested["test_prediction"]),
            "local_dispersion": summarize(nested["test_dispersion"]),
            "maximum_neighbor_distance": summarize(nested["test_distance"]),
        },
        "application": application,
        "comparison_to_v140": comparison,
        "excluded": {
            "v144_v_correction": True,
            "fixed_v145_strength_everywhere": True,
            "fixed_v148_point15_gate": True,
            "new_hard_clear_light_medium_heavy_gate": True,
            "complex_phase_regression": True,
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "decisions": PATHS.relative(DECISION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
        "integrity": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": comparison["finite"],
            "output_bytes": int(output.nbytes),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source_path, name in (
            (__file__, "generate_v149_continuous_action_response_controller.py"),
            (
                PATHS.code / "generate_v147_continuous_local_action_gate.py",
                "generate_v147_continuous_local_action_gate.py",
            ),
            (
                PATHS.code / "generate_v146_continuous_diffusion_gate.py",
                "generate_v146_continuous_diffusion_gate.py",
            ),
            (
                PATHS.code / "analyze_v145_gated_local_h_field.py",
                "analyze_v145_gated_local_h_field.py",
            ),
            (
                PATHS.code / "analyze_post_v143_hidden_information.py",
                "analyze_post_v143_hidden_information.py",
            ),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
            ),
            (
                PATHS.code / "generate_v143_transductive_geodesic_risk.py",
                "generate_v143_transductive_geodesic_risk.py",
            ),
            (
                PATHS.code / "generate_v112_kernel_moe_path_support.py",
                "generate_v112_kernel_moe_path_support.py",
            ),
            (
                PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                "generate_v98_path_ensemble_graph_attention_joint_ad.py",
            ),
            (
                PATHS.code / "generate_v87_easy_field_projection.py",
                "generate_v87_easy_field_projection.py",
            ),
            (PATHS.code / "powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ):
            archive.write(source_path, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    report["integrity"].update(
        {
            "zip_crc": True,
            "zip_members": members,
            "sha256": digest(ZIP_OUT, "sha256"),
            "md5": digest(ZIP_OUT, "md5"),
        }
    )
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {ZIP_OUT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
