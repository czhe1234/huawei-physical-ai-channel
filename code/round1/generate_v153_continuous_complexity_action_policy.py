#!/usr/bin/env python3
"""Audit v153 continuous complexity vector action policy."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from analyze_continuous_propagation_complexity import (
    channel_labels,
    make_geometry_features,
)
from analyze_post_v143_hidden_information import normalize_without_floor, support_score
from analyze_v145_gated_local_h_field import local_h_predictions
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, support_features
from generate_v143_transductive_geodesic_risk import apply_risk_shift
from generate_v87_easy_field_projection import summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v153_continuous_complexity_action_policy"
OUT_DIR = PATHS.cache_dir(153)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(153)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DECISION_OUT = OUT_DIR / "v153_continuous_complexity_action_policy_oof.npz"
REPORT = REPORT_DIR / "v153_continuous_complexity_action_policy_generate.json"
ZIP_OUT = PATHS.round1_versions / "v153_continuous_complexity_action_policy_candidate_pending.zip"

RANDOM_STATE = 15303
LOCAL_K = 8
MAX_DISTANCE = 20.0
TREE_COUNT = 96
POSITIVE_THRESHOLD = 0.00025
REGIMES = ("clear", "light", "medium", "heavy")
COMPLEXITY_LABELS = (
    "log10_total_power",
    "h_spread",
    "delay_rms_spread",
    "local_h_coherence_20m",
    "local_delay_coherence_20m",
    "joint_top1",
)
ACTIONS = (
    ("local_h", 0.10),
    ("local_h", 0.15),
    ("local_h", 0.25),
    ("h_diff", 0.02),
    ("h_diff", 0.05),
    ("d_diff", 0.02),
    ("d_diff", 0.05),
    ("local_d", 0.10),
    ("local_d", 0.15),
)
CONFIGS = tuple(
    (probability, penalty, minimum)
    for probability in (0.65, 0.75, 0.85)
    for penalty in (0.50, 1.00, 1.50)
    for minimum in (0.0, 0.00025, 0.00050)
)


def summarize_gain(values: np.ndarray) -> dict[str, object]:
    return {**summarize(values), "positive_fraction": float(np.mean(values > 0.0))}


def config_name(config: tuple[float, float, float]) -> str:
    probability, penalty, minimum = config
    return f"p{probability:.2f}_lcb{penalty:.2f}_min{minimum:.5f}"


def action_name(index: int) -> str:
    action_type, strength = ACTIONS[index]
    return f"{action_type}_{strength:.2f}"


def local_axis_predictions(
    positions: np.ndarray,
    axis_target: np.ndarray,
    train_index: np.ndarray,
    query_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=LOCAL_K, n_jobs=-1)
    model.fit(positions[train_index, :2])
    distance, local = model.kneighbors(positions[query_index, :2])
    neighbors = train_index[local]
    inverse = 1.0 / np.maximum(distance, 0.25)
    weight = inverse / np.maximum(inverse.sum(axis=1, keepdims=True), 1e-12)
    prediction = np.sum(weight[..., None] * axis_target[neighbors], axis=1)
    return prediction.astype(np.float32), distance[:, LOCAL_K - 1].astype(np.float32)


def diffuse_axis(source: np.ndarray, coefficient: float, axis: int) -> np.ndarray:
    z = np.log(np.maximum(source, 1e-12)).astype(np.float32)
    laplacian = np.roll(z, -1, axis=axis) + np.roll(z, 1, axis=axis) - 2.0 * z
    raw = np.exp(z + coefficient * laplacian).astype(np.float32)
    raw /= np.maximum(raw.sum(axis=(1, 2, 3), keepdims=True), TINY)
    ratio = np.clip(raw / np.maximum(source, 1e-12), 0.50**2, 2.00**2)
    output = source * ratio
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output.astype(np.float32)


def apply_h_local(
    source: np.ndarray,
    prediction: np.ndarray,
    distance: np.ndarray,
    strength: float,
) -> np.ndarray:
    gate = (distance <= MAX_DISTANCE).astype(np.float32)
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0) * (strength * gate)[:, None]
    ).astype(np.float32)
    output = source * ratio[:, :, None, None]
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output.astype(np.float32)


def apply_d_local(
    source: np.ndarray,
    prediction: np.ndarray,
    distance: np.ndarray,
    strength: float,
) -> np.ndarray:
    gate = (distance <= MAX_DISTANCE).astype(np.float32)
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0) * (strength * gate)[:, None]
    ).astype(np.float32)
    output = source * ratio[:, None, None, :]
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output.astype(np.float32)


def fit_complexity(
    geometry_features: np.ndarray,
    labels: dict[str, np.ndarray],
    train_index: np.ndarray,
    query_index: np.ndarray,
) -> np.ndarray:
    predictions = []
    for label_index, name in enumerate(COMPLEXITY_LABELS):
        model = ExtraTreesRegressor(
            n_estimators=TREE_COUNT,
            max_features=0.65,
            min_samples_leaf=10,
            random_state=RANDOM_STATE + label_index,
            n_jobs=-1,
        )
        model.fit(geometry_features[train_index], labels[name][train_index])
        predictions.append(model.predict(geometry_features[query_index]).astype(np.float32))
    return np.column_stack(predictions).astype(np.float32)


def build_policy_features(
    geometry_features: np.ndarray,
    source_features: np.ndarray,
    index: np.ndarray,
    complexity: np.ndarray,
    h_prediction: np.ndarray,
    h_distance: np.ndarray,
    d_prediction: np.ndarray,
    d_distance: np.ndarray,
) -> np.ndarray:
    h_stats = np.column_stack(
        [
            np.mean(h_prediction, axis=1),
            np.mean(np.abs(h_prediction), axis=1),
            np.max(np.abs(h_prediction), axis=1),
            np.std(h_prediction, axis=1),
            h_distance,
            (h_distance <= MAX_DISTANCE).astype(np.float32),
        ]
    )
    d_stats = np.column_stack(
        [
            np.mean(d_prediction, axis=1),
            np.mean(np.abs(d_prediction), axis=1),
            np.max(np.abs(d_prediction), axis=1),
            np.std(d_prediction, axis=1),
            d_distance,
            (d_distance <= MAX_DISTANCE).astype(np.float32),
        ]
    )
    values = np.concatenate(
        [
            geometry_features[index],
            complexity,
            source_features[index],
            h_stats.astype(np.float32),
            d_stats.astype(np.float32),
        ],
        axis=1,
    )
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def action_gains(
    source: np.ndarray,
    truth: np.ndarray,
    query_index: np.ndarray,
    h_prediction: np.ndarray,
    h_distance: np.ndarray,
    d_prediction: np.ndarray,
    d_distance: np.ndarray,
) -> np.ndarray:
    base = support_score(source[query_index], truth[query_index])
    gains = np.zeros((len(query_index), len(ACTIONS)), dtype=np.float32)
    for action_index, (action_type, strength) in enumerate(ACTIONS):
        if action_type == "local_h":
            adjusted = apply_h_local(
                source[query_index], h_prediction, h_distance, strength
            )
        elif action_type == "local_d":
            adjusted = apply_d_local(
                source[query_index], d_prediction, d_distance, strength
            )
        elif action_type == "h_diff":
            adjusted = diffuse_axis(source[query_index], strength, axis=1)
        elif action_type == "d_diff":
            adjusted = diffuse_axis(source[query_index], strength, axis=3)
        else:
            raise ValueError(f"unknown action type {action_type}")
        gains[:, action_index] = support_score(adjusted, truth[query_index]) - base
    return gains


def fit_ensemble(
    features: np.ndarray,
    gains: np.ndarray,
    positive: np.ndarray,
    train_index: np.ndarray,
    evaluation_features: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    benefit_predictions = []
    probability_predictions = []
    for seed in seeds:
        regressor = ExtraTreesRegressor(
            n_estimators=TREE_COUNT,
            max_features=0.65,
            min_samples_leaf=8,
            random_state=seed,
            n_jobs=-1,
        )
        classifier = ExtraTreesClassifier(
            n_estimators=TREE_COUNT,
            max_features=0.65,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=seed + 1000,
            n_jobs=-1,
        )
        regressor.fit(features[train_index], gains[train_index])
        classifier.fit(features[train_index], positive[train_index])
        benefit_predictions.append(
            regressor.predict(evaluation_features).astype(np.float32)
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


def complexity_thresholds(complexity: np.ndarray) -> dict[str, float]:
    return {
        "local_h_coherence": float(np.quantile(complexity[:, 3], 0.60)),
        "local_delay_coherence": float(np.quantile(complexity[:, 4], 0.60)),
        "h_spread_high": float(np.quantile(complexity[:, 1], 0.62)),
        "h_spread_very_high": float(np.quantile(complexity[:, 1], 0.82)),
        "h_spread_low": float(np.quantile(complexity[:, 1], 0.50)),
        "delay_spread_high": float(np.quantile(complexity[:, 2], 0.62)),
        "joint_top1_low": float(np.quantile(complexity[:, 5], 0.50)),
    }


def action_validity(
    complexity: np.ndarray,
    h_distance: np.ndarray,
    d_distance: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    h_spread = complexity[:, 1]
    delay_spread = complexity[:, 2]
    local_h = complexity[:, 3]
    local_d = complexity[:, 4]
    joint_top1 = complexity[:, 5]
    valid = np.zeros((len(complexity), len(ACTIONS)), dtype=bool)
    for action_index, (action_type, _) in enumerate(ACTIONS):
        if action_type == "local_h":
            valid[:, action_index] = (
                (local_h >= thresholds["local_h_coherence"])
                & (h_distance <= MAX_DISTANCE)
                & (h_spread < thresholds["h_spread_very_high"])
            )
        elif action_type == "h_diff":
            valid[:, action_index] = (
                (h_spread >= thresholds["h_spread_high"])
                & (joint_top1 <= thresholds["joint_top1_low"])
            )
        elif action_type == "d_diff":
            valid[:, action_index] = (
                (delay_spread >= thresholds["delay_spread_high"])
                & (h_spread < thresholds["h_spread_low"])
            )
        elif action_type == "local_d":
            valid[:, action_index] = (
                (local_d >= thresholds["local_delay_coherence"])
                & (d_distance <= MAX_DISTANCE)
                & (delay_spread >= thresholds["delay_spread_high"])
                & (h_spread < thresholds["h_spread_very_high"])
            )
    return valid


def choose_actions(
    benefit: np.ndarray,
    uncertainty: np.ndarray,
    probability: np.ndarray,
    config: tuple[float, float, float],
    valid_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability_threshold, uncertainty_penalty, minimum_benefit = config
    lower_bound = benefit - uncertainty_penalty * uncertainty
    lower_bound = np.where(valid_actions, lower_bound, -1e9)
    best_index = np.argmax(lower_bound, axis=1)
    best_lower_bound = lower_bound[np.arange(len(best_index)), best_index]
    active = (
        (probability >= probability_threshold)
        & (best_lower_bound > minimum_benefit)
        & (best_lower_bound > -1e8)
    )
    selected = np.where(active, best_index + 1, 0).astype(np.int16)
    return selected, best_lower_bound.astype(np.float32)


def realized_gain(selected: np.ndarray, gains: np.ndarray) -> np.ndarray:
    output = np.zeros(len(selected), dtype=np.float32)
    active = selected > 0
    if np.any(active):
        output[active] = gains[
            np.flatnonzero(active), selected[active].astype(np.int64) - 1
        ]
    return output


def counter_to_json(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def nested_oof(
    positions: np.ndarray,
    geometry_features: np.ndarray,
    source_features: np.ndarray,
    source: np.ndarray,
    truth: np.ndarray,
    h_target: np.ndarray,
    d_target: np.ndarray,
    labels: dict[str, np.ndarray],
    valid: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, object]:
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions[:, :2])
    selected_valid = np.flatnonzero(valid)
    outer = GroupKFold(n_splits=5)
    oof_gain = np.zeros(len(positions), dtype=np.float32)
    oof_selected = np.zeros(len(positions), dtype=np.int16)
    fold_reports = []

    for fold, (outer_train_local, outer_validation_local) in enumerate(
        outer.split(selected_valid, groups=groups[selected_valid])
    ):
        outer_train = selected_valid[outer_train_local]
        outer_validation = selected_valid[outer_validation_local]
        inner_complexity = np.zeros((len(positions), len(COMPLEXITY_LABELS)), dtype=np.float32)
        inner_h_prediction = np.zeros((len(positions), 16), dtype=np.float32)
        inner_h_distance = np.zeros(len(positions), dtype=np.float32)
        inner_d_prediction = np.zeros((len(positions), 192), dtype=np.float32)
        inner_d_distance = np.zeros(len(positions), dtype=np.float32)

        inner = GroupKFold(n_splits=4)
        for inner_train_local, inner_validation_local in inner.split(
            outer_train, groups=groups[outer_train]
        ):
            inner_train = outer_train[inner_train_local]
            inner_validation = outer_train[inner_validation_local]
            h_prediction, _, h_distance = local_h_predictions(
                positions, h_target, inner_train, inner_validation
            )
            d_prediction, d_distance = local_axis_predictions(
                positions, d_target, inner_train, inner_validation
            )
            inner_h_prediction[inner_validation] = h_prediction[LOCAL_K]
            inner_h_distance[inner_validation] = h_distance[LOCAL_K]
            inner_d_prediction[inner_validation] = d_prediction
            inner_d_distance[inner_validation] = d_distance
            inner_complexity[inner_validation] = fit_complexity(
                geometry_features, labels, inner_train, inner_validation
            )

        train_features = build_policy_features(
            geometry_features,
            source_features,
            outer_train,
            inner_complexity[outer_train],
            inner_h_prediction[outer_train],
            inner_h_distance[outer_train],
            inner_d_prediction[outer_train],
            inner_d_distance[outer_train],
        )
        train_gains = action_gains(
            source,
            truth,
            outer_train,
            inner_h_prediction[outer_train],
            inner_h_distance[outer_train],
            inner_d_prediction[outer_train],
            inner_d_distance[outer_train],
        )
        positive = (train_gains.max(axis=1) > POSITIVE_THRESHOLD).astype(np.int8)
        thresholds = complexity_thresholds(inner_complexity[outer_train])
        train_valid_actions = action_validity(
            inner_complexity[outer_train],
            inner_h_distance[outer_train],
            inner_d_distance[outer_train],
            thresholds,
        )

        inner_benefit = np.zeros_like(train_gains, dtype=np.float32)
        inner_uncertainty = np.zeros_like(train_gains, dtype=np.float32)
        inner_probability = np.zeros(len(outer_train), dtype=np.float32)
        for local_train, local_validation in GroupKFold(n_splits=4).split(
            np.arange(len(outer_train)), groups=groups[outer_train]
        ):
            prediction = fit_ensemble(
                train_features,
                train_gains,
                positive,
                local_train,
                train_features[local_validation],
                (RANDOM_STATE + 10 * fold + 1, RANDOM_STATE + 10 * fold + 2),
            )
            inner_benefit[local_validation] = prediction[0]
            inner_uncertainty[local_validation] = prediction[1]
            inner_probability[local_validation] = prediction[2]

        best_config = CONFIGS[0]
        best_key = (-1e9, -1e9)
        config_reports = {}
        for config in CONFIGS:
            selected, lower_bound = choose_actions(
                inner_benefit,
                inner_uncertainty,
                inner_probability,
                config,
                train_valid_actions,
            )
            gain = realized_gain(selected, train_gains)
            active = float(np.mean(selected > 0))
            mean = float(gain.mean())
            config_reports[config_name(config)] = {
                "full_gain_mean": mean,
                "active_fraction": active,
                "lower_bound": summarize(lower_bound),
            }
            if active > 0.45:
                continue
            key = (mean, -abs(active - 0.18))
            if key > best_key:
                best_key = key
                best_config = config

        h_prediction, _, h_distance = local_h_predictions(
            positions, h_target, outer_train, outer_validation
        )
        d_prediction, d_distance = local_axis_predictions(
            positions, d_target, outer_train, outer_validation
        )
        validation_complexity = fit_complexity(
            geometry_features, labels, outer_train, outer_validation
        )
        validation_features = build_policy_features(
            geometry_features,
            source_features,
            outer_validation,
            validation_complexity,
            h_prediction[LOCAL_K],
            h_distance[LOCAL_K],
            d_prediction,
            d_distance,
        )
        validation_prediction = fit_ensemble(
            train_features,
            train_gains,
            positive,
            np.arange(len(outer_train)),
            validation_features,
            (
                RANDOM_STATE + 100 + fold,
                RANDOM_STATE + 200 + fold,
                RANDOM_STATE + 300 + fold,
            ),
        )
        validation_valid_actions = action_validity(
            validation_complexity,
            h_distance[LOCAL_K],
            d_distance,
            thresholds,
        )
        validation_selected, validation_lower_bound = choose_actions(
            validation_prediction[0],
            validation_prediction[1],
            validation_prediction[2],
            best_config,
            validation_valid_actions,
        )
        validation_gains = action_gains(
            source,
            truth,
            outer_validation,
            h_prediction[LOCAL_K],
            h_distance[LOCAL_K],
            d_prediction,
            d_distance,
        )
        validation_gain = realized_gain(validation_selected, validation_gains)
        oof_gain[outer_validation] = validation_gain
        oof_selected[outer_validation] = validation_selected
        fold_reports.append(
            {
                "fold": fold,
                "selected_config": config_name(best_config),
                "train_count": int(len(outer_train)),
                "validation_count": int(len(outer_validation)),
                "validation_gain": summarize_gain(validation_gain),
                "active_fraction": float(np.mean(validation_selected > 0)),
                "action_counts": counter_to_json(Counter(validation_selected)),
                "thresholds": thresholds,
                "inner_config_reports": config_reports,
                "validation_lower_bound": summarize(validation_lower_bound),
            }
        )

    selected_gain = oof_gain[valid]
    selected_actions = oof_selected[valid]
    by_regime = {}
    for regime in REGIMES:
        mask = valid & (regimes == regime)
        by_regime[regime] = {
            "count": int(mask.sum()),
            "gain": summarize_gain(oof_gain[mask]),
            "active_fraction": float(np.mean(oof_selected[mask] > 0)),
        }
    return {
        "oof_gain": oof_gain,
        "oof_selected": oof_selected,
        "fold_reports": fold_reports,
        "summary": {
            "full_gain": summarize_gain(selected_gain),
            "active_fraction": float(np.mean(selected_actions > 0)),
            "fold_means": [
                row["validation_gain"]["mean"] for row in fold_reports
            ],
            "all_folds_positive": bool(
                np.all(
                    [
                        row["validation_gain"]["mean"] > 0.0
                        for row in fold_reports
                    ]
                )
            ),
            "action_counts": counter_to_json(Counter(selected_actions)),
            "by_regime": by_regime,
        },
    }


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(
        np.float32
    )
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    pointcloud = np.load(
        PATHS.shared_features / "official_pointcloud_scene_cache.npz"
    )
    probability_train = np.load(
        PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
    ).astype(np.float32)
    source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    train_risk = build_risk(
        regimes_train,
        probability_train,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    train_current, _ = apply_risk_shift(source, train_risk)
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    valid = truth_raw.sum(axis=(1, 2, 3)) > 0
    truth = normalize_without_floor(truth_raw)
    truth_h = truth.sum(axis=(2, 3))
    truth_d = truth.sum(axis=(1, 2))
    current_h = train_current.sum(axis=(2, 3))
    current_d = train_current.sum(axis=(1, 2))
    h_target = np.clip(
        np.log(np.maximum(truth_h, 1e-10))
        - np.log(np.maximum(current_h, 1e-10)),
        -3.0,
        3.0,
    ).astype(np.float32)
    d_target = np.clip(
        np.log(np.maximum(truth_d, 1e-10))
        - np.log(np.maximum(current_d, 1e-10)),
        -3.0,
        3.0,
    ).astype(np.float32)

    labels_valid, _ = channel_labels(truth_raw, positions_train, valid)
    labels = {}
    for name, values in labels_valid.items():
        expanded = np.zeros(len(positions_train), dtype=np.float32)
        expanded[valid] = values
        labels[name] = expanded
    geometry_features, _ = make_geometry_features(
        positions_train,
        blocked_train,
        geodesic["train_features"].astype(np.float32),
        pointcloud["train_features"].astype(np.float32),
    )
    source_signature = support_features(train_current).astype(np.float32)

    nested = nested_oof(
        positions_train,
        geometry_features,
        source_signature,
        train_current,
        truth,
        h_target,
        d_target,
        labels,
        valid,
        regimes_train,
    )
    accepted_local = bool(
        nested["summary"]["full_gain"]["mean"] > 0.0
        and nested["summary"]["all_folds_positive"]
    )
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    np.savez_compressed(
        DECISION_OUT,
        oof_gain=nested["oof_gain"].astype(np.float32),
        oof_selected=nested["oof_selected"].astype(np.int16),
        valid=valid.astype(bool),
        actions=np.asarray([action_name(i) for i in range(len(ACTIONS))]),
    )
    report = {
        "method": VERSION,
        "status": "candidate_pending" if accepted_local else "rejected_local",
        "base": "v140 online 0.501352; exact v140 train analogue used for local audit",
        "primary_hypothesis": (
            "use the continuous complexity vector from section 150.8 to choose "
            "among physically matched actions: local H residual, bounded H "
            "diffusion, delay diffusion, local delay residual, or fallback"
        ),
        "only_primary_change": (
            "replace hard heavy buckets and single-action local-H gating with "
            "a complexity-valid action family and learned benefit/uncertainty "
            "selection"
        ),
        "model": {
            "complexity_labels": list(COMPLEXITY_LABELS),
            "actions": [action_name(i) for i in range(len(ACTIONS))],
            "positive_threshold": POSITIVE_THRESHOLD,
            "tree_count": TREE_COUNT,
            "selection_configs": [config_name(config) for config in CONFIGS],
            "action_validity": {
                "local_h": "local_h_coherence high, H spread not very high, XY neighbor <=20m",
                "h_diff": "H spread high and joint top1 low",
                "d_diff": "delay RMS high and H spread low",
                "local_d": "local_delay_coherence high, delay RMS high, H spread not very high, neighbor <=20m",
                "fallback": "implicit no-action when no valid action has positive LCB",
            },
        },
        "nested_spatial_oof": {
            "outer_folds": 5,
            "inner_folds": 4,
            "fold_reports": nested["fold_reports"],
            **nested["summary"],
        },
        "outputs": {
            "decisions": PATHS.relative(DECISION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
        "integrity": {
            "zip_created": accepted_local,
            "zip_exists": ZIP_OUT.exists(),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
