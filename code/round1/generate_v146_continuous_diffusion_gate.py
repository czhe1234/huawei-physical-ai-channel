#!/usr/bin/env python3
"""Generate v146 with a learned continuous benefit/uncertainty diffusion gate."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import (
    build_risk,
    entropy01,
    row_cos,
    support_features,
)
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    digest,
    output_comparison,
)
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import (
    ad_transform,
    beta_subcarriers,
    inverse_ad,
)
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


VERSION = "v146_continuous_diffusion_gate"
OUT_DIR = PATHS.cache_dir(146)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(146)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v146_Round1_Test_Channel.npy"
DECISION_OUT = OUT_DIR / "v146_continuous_diffusion_decisions.npz"
REPORT = REPORT_DIR / "v146_continuous_diffusion_gate_generate.json"
ZIP_OUT = PATHS.round1_versions / "v146_continuous_diffusion_gate_candidate_pending.zip"

RANDOM_STATE = 14607
ACTIONS = np.asarray([0.02, 0.05, 0.10, 0.20, 0.35], dtype=np.float32)
ORACLE_POSITIVE_THRESHOLD = 0.001
GAIN_CLIP = (0.50, 2.00)
REGIMES = ("clear", "light", "medium", "heavy")
SEEDS = (14607, 14631, 14657)
INNER_SEEDS = (14607, 14631)
CONFIGS = tuple(
    (probability, penalty, minimum)
    for probability in (0.50, 0.60, 0.70)
    for penalty in (0.50, 1.00, 1.50)
    for minimum in (0.0000, 0.00025, 0.00050)
)


def config_name(config: tuple[float, float, float]) -> str:
    probability, penalty, minimum = config
    return f"p{probability:.2f}_lcb{penalty:.2f}_min{minimum:.5f}"


def summarize_gain(gain: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    values = gain[valid]
    return {
        **summarize(values),
        "positive_fraction": float(np.mean(values > 0)),
    }


def diffuse_support(source: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    output = np.empty_like(source, dtype=np.float32)
    for start in range(0, len(source), 32):
        end = min(start + 32, len(source))
        current = source[start:end].astype(np.float32)
        z = np.log(np.maximum(current, 1e-12)).astype(np.float32)
        laplacian = (
            np.roll(z, -1, axis=1)
            + np.roll(z, 1, axis=1)
            - 2.0 * z
        )
        raw = np.exp(
            z + coefficient[start:end, None, None, None] * laplacian
        ).astype(np.float32)
        raw /= np.maximum(raw.sum(axis=(1, 2, 3), keepdims=True), TINY)
        power_ratio = np.clip(
            raw / np.maximum(current, 1e-12),
            GAIN_CLIP[0] ** 2,
            GAIN_CLIP[1] ** 2,
        )
        adjusted = current * power_ratio
        adjusted /= np.maximum(
            adjusted.sum(axis=(1, 2, 3), keepdims=True), TINY
        )
        output[start:end] = adjusted.astype(np.float32)
    return output


def action_gains(
    source: np.ndarray, truth: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    base_score = row_cos(source, truth)
    gains = np.zeros((len(source), len(ACTIONS)), dtype=np.float32)
    for action_index, action in enumerate(ACTIONS):
        coefficient = np.full(len(source), action, dtype=np.float32)
        score = row_cos(diffuse_support(source, coefficient), truth)
        gains[:, action_index] = score - base_score
    gains[~valid] = 0.0
    return base_score, gains


def clipped01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def decision_features(
    positions: np.ndarray,
    blocked: np.ndarray,
    geodesic: np.ndarray,
    geodesic_names: list[str],
    pointcloud: np.ndarray,
    probability: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    index = {name: i for i, name in enumerate(geodesic_names)}
    vector = positions - BS_POS[None, :]
    distance = np.linalg.norm(vector, axis=1)
    azimuth = np.arctan2(vector[:, 1], vector[:, 0])
    residual_entropy = entropy01(probability)
    residual_top = probability.max(axis=1)
    unsaturated = np.column_stack(
        [
            clipped01((geodesic[:, index["geo_ratio"]] - 1.13) / 0.34),
            clipped01((geodesic[:, index["geo_excess"]] - 14.0) / 52.0),
            clipped01((5.2 - geodesic[:, index["mean_clearance"]]) / 5.2),
            clipped01((geodesic[:, index["turn_rate"]] - 0.14) / 0.26),
            clipped01((residual_entropy - 0.22) / 0.40),
            clipped01((0.62 - residual_top) / 0.36),
        ]
    ).astype(np.float32)
    basic = np.column_stack(
        [
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            distance,
            np.log1p(distance),
            np.sin(azimuth),
            np.cos(azimuth),
            blocked,
            blocked / np.maximum(distance, 1.0),
            residual_entropy,
            residual_top,
        ]
    ).astype(np.float32)
    values = np.concatenate(
        [
            basic,
            unsaturated,
            geodesic.astype(np.float32),
            pointcloud.astype(np.float32),
            support_features(source),
        ],
        axis=1,
    )
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


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
            n_estimators=240,
            max_features=0.65,
            min_samples_leaf=8,
            random_state=seed,
            n_jobs=-1,
        )
        classifier = ExtraTreesClassifier(
            n_estimators=240,
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


def choose_actions(
    benefit: np.ndarray,
    uncertainty: np.ndarray,
    probability: np.ndarray,
    config: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    probability_threshold, uncertainty_penalty, minimum_benefit = config
    lower_bound = benefit - uncertainty_penalty * uncertainty
    action_index = np.argmax(lower_bound, axis=1)
    best_lower_bound = lower_bound[np.arange(len(lower_bound)), action_index]
    active = (
        (probability >= probability_threshold)
        & (best_lower_bound > minimum_benefit)
    )
    selected = np.where(active, action_index + 1, 0).astype(np.int16)
    return selected, best_lower_bound.astype(np.float32)


def realized_gain(selected: np.ndarray, gains: np.ndarray) -> np.ndarray:
    output = np.zeros(len(selected), dtype=np.float32)
    active = selected > 0
    output[active] = gains[
        np.flatnonzero(active), selected[active].astype(np.int64) - 1
    ]
    return output


def select_config(
    benefit: np.ndarray,
    uncertainty: np.ndarray,
    probability: np.ndarray,
    gains: np.ndarray,
    valid: np.ndarray,
) -> tuple[tuple[float, float, float], dict[str, object]]:
    reports: dict[str, object] = {}
    best_config = CONFIGS[0]
    best_value = -np.inf
    for config in CONFIGS:
        selected, lower_bound = choose_actions(
            benefit, uncertainty, probability, config
        )
        gain = realized_gain(selected, gains)
        active = valid & (selected > 0)
        mean_gain = float(np.mean(gain[valid]))
        reports[config_name(config)] = {
            "full_gain_mean": mean_gain,
            "active_fraction": float(np.mean(active[valid])),
            "active_gain_mean": float(np.mean(gain[active])) if np.any(active) else 0.0,
            "active_positive_fraction": (
                float(np.mean(gain[active] > 0)) if np.any(active) else 0.0
            ),
            "lower_bound": summarize(lower_bound[valid]),
        }
        if mean_gain > best_value:
            best_value = mean_gain
            best_config = config
    return best_config, reports


def nested_spatial_oof(
    features: np.ndarray,
    gains: np.ndarray,
    positive: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    test_features: np.ndarray,
) -> dict[str, object]:
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions[:, :2])
    selected_valid = np.flatnonzero(valid)
    outer = GroupKFold(n_splits=5)
    oof_benefit = np.zeros_like(gains, dtype=np.float32)
    oof_uncertainty = np.zeros_like(gains, dtype=np.float32)
    oof_probability = np.zeros(len(features), dtype=np.float32)
    oof_selected = np.zeros(len(features), dtype=np.int16)
    test_benefit_folds = []
    test_uncertainty_folds = []
    test_probability_folds = []
    test_probability_uncertainty_folds = []
    fold_reports = []
    selected_configs = []

    for fold, (outer_train_local, outer_validation_local) in enumerate(
        outer.split(selected_valid, groups=groups[selected_valid])
    ):
        outer_train = selected_valid[outer_train_local]
        outer_validation = selected_valid[outer_validation_local]
        inner_benefit = np.zeros_like(gains, dtype=np.float32)
        inner_uncertainty = np.zeros_like(gains, dtype=np.float32)
        inner_probability = np.zeros(len(features), dtype=np.float32)
        inner = GroupKFold(n_splits=4)
        for inner_train_local, inner_validation_local in inner.split(
            outer_train, groups=groups[outer_train]
        ):
            inner_train = outer_train[inner_train_local]
            inner_validation = outer_train[inner_validation_local]
            prediction = fit_ensemble(
                features,
                gains,
                positive,
                inner_train,
                features[inner_validation],
                INNER_SEEDS,
            )
            inner_benefit[inner_validation] = prediction[0]
            inner_uncertainty[inner_validation] = prediction[1]
            inner_probability[inner_validation] = prediction[2]
        fold_config, config_reports = select_config(
            inner_benefit,
            inner_uncertainty,
            inner_probability,
            gains,
            np.isin(np.arange(len(features)), outer_train),
        )
        selected_configs.append(fold_config)

        validation_prediction = fit_ensemble(
            features,
            gains,
            positive,
            outer_train,
            features[outer_validation],
            SEEDS,
        )
        oof_benefit[outer_validation] = validation_prediction[0]
        oof_uncertainty[outer_validation] = validation_prediction[1]
        oof_probability[outer_validation] = validation_prediction[2]
        fold_selected, _ = choose_actions(
            validation_prediction[0],
            validation_prediction[1],
            validation_prediction[2],
            fold_config,
        )
        oof_selected[outer_validation] = fold_selected

        test_prediction = fit_ensemble(
            features,
            gains,
            positive,
            outer_train,
            test_features,
            SEEDS,
        )
        test_benefit_folds.append(test_prediction[0])
        test_uncertainty_folds.append(test_prediction[1])
        test_probability_folds.append(test_prediction[2])
        test_probability_uncertainty_folds.append(test_prediction[3])

        validation_gain = realized_gain(
            fold_selected, gains[outer_validation]
        )
        fold_reports.append(
            {
                "fold": fold,
                "selected_config": config_name(fold_config),
                "train_count": int(len(outer_train)),
                "validation_count": int(len(outer_validation)),
                "validation_gain": summarize_gain(
                    validation_gain,
                    np.ones(len(validation_gain), dtype=bool),
                ),
                "active_fraction": float(
                    np.mean(fold_selected > 0)
                ),
                "inner_config_reports": config_reports,
            }
        )

    config_counts = Counter(config_name(value) for value in selected_configs)
    modal_name = config_counts.most_common(1)[0][0]
    modal_config = next(value for value in CONFIGS if config_name(value) == modal_name)

    test_benefit_stack = np.stack(test_benefit_folds, axis=0)
    test_benefit = test_benefit_stack.mean(axis=0).astype(np.float32)
    within_variance = np.mean(
        np.stack(test_uncertainty_folds, axis=0) ** 2, axis=0
    )
    between_variance = test_benefit_stack.var(axis=0)
    test_uncertainty = np.sqrt(
        np.maximum(within_variance + between_variance, 0.0)
    ).astype(np.float32)
    test_probability_stack = np.stack(test_probability_folds, axis=0)
    test_probability = test_probability_stack.mean(axis=0).astype(np.float32)
    test_probability_uncertainty = np.sqrt(
        np.maximum(
            np.mean(
                np.stack(test_probability_uncertainty_folds, axis=0) ** 2,
                axis=0,
            )
            + test_probability_stack.var(axis=0),
            0.0,
        )
    ).astype(np.float32)
    test_selected, test_lower_bound = choose_actions(
        test_benefit,
        test_uncertainty,
        test_probability,
        modal_config,
    )
    oof_gain = realized_gain(oof_selected, gains)
    active = valid & (oof_selected > 0)
    return {
        "groups": groups,
        "oof_benefit": oof_benefit,
        "oof_uncertainty": oof_uncertainty,
        "oof_probability": oof_probability,
        "oof_selected": oof_selected,
        "oof_gain": oof_gain,
        "test_benefit": test_benefit,
        "test_uncertainty": test_uncertainty,
        "test_probability": test_probability,
        "test_probability_uncertainty": test_probability_uncertainty,
        "test_selected": test_selected,
        "test_lower_bound": test_lower_bound,
        "modal_config": modal_config,
        "config_counts": dict(config_counts),
        "fold_reports": fold_reports,
        "oof_report": {
            "full_gain": summarize_gain(oof_gain, valid),
            "active_fraction": float(np.mean(active[valid])),
            "active_gain": summarize_gain(oof_gain, active) if np.any(active) else {},
            "by_regime": {},
        },
    }


def apply_complex_diffusion(
    base_path,
    output_path,
    positions: np.ndarray,
    coefficient: np.ndarray,
) -> dict[str, object]:
    base = np.load(base_path, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=base.shape
    )
    beta = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(positions - BS_POS[None, :], axis=1)
    target_cosines = []
    power_ratios = []
    gain_minimum = []
    gain_maximum = []
    for start in range(0, len(base), 10):
        end = min(start + 10, len(base))
        current = np.asarray(base[start:end], dtype=np.complex64)
        derotated = current * np.exp(
            1j * distance[start:end, None, None, None] * beta
        ).astype(np.complex64)
        ad0 = ad_transform(derotated)
        power0 = (np.abs(ad0) ** 2).astype(np.float32)
        collapsed0 = power0.sum(axis=(1, 4))
        total0 = np.maximum(
            collapsed0.sum(axis=(1, 2, 3)), TINY
        ).astype(np.float32)
        support0 = collapsed0 / total0[:, None, None, None]
        support1 = diffuse_support(support0, coefficient[start:end])
        gain = np.sqrt(
            support1 / np.maximum(support0, 1e-12)
        ).astype(np.float32)
        gain = np.clip(gain, GAIN_CLIP[0], GAIN_CLIP[1])
        ad1 = ad0 * gain[:, None, :, :, None, :]
        total1 = np.maximum(
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5)), TINY
        ).astype(np.float32)
        ad1 *= np.sqrt(total0 / total1)[:, None, None, None, None, None]
        reconstructed = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        reconstructed *= np.exp(
            -1j * distance[start:end, None, None, None] * beta
        ).astype(np.complex64)
        output[start:end] = reconstructed.astype(np.complex64)
        output.flush()

        collapsed1 = (np.abs(ad1) ** 2).sum(axis=(1, 4))
        support_after = collapsed1 / np.maximum(
            collapsed1.sum(axis=(1, 2, 3), keepdims=True), TINY
        )
        target_cosines.extend(row_cos(support_after, support1).tolist())
        power_ratios.extend((total1 / total0).tolist())
        gain_minimum.extend(gain.reshape(end - start, -1).min(axis=1).tolist())
        gain_maximum.extend(gain.reshape(end - start, -1).max(axis=1).tolist())
    return {
        "target_cosine_after": summarize(np.asarray(target_cosines)),
        "power_ratio_before_final_normalization": summarize(
            np.asarray(power_ratios)
        ),
        "gain_min": summarize(np.asarray(gain_minimum)),
        "gain_max": summarize(np.asarray(gain_maximum)),
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
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    pointcloud = np.load(
        PATHS.shared_features / "official_pointcloud_scene_cache.npz"
    )
    probability_train = np.load(
        PATHS.shared_features / "train_v129_testlike_residual_probs.npy"
    ).astype(np.float32)
    probability_test = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_probs.npy"
    ).astype(np.float32)

    train_v129 = normalize_support(
        np.load(PATHS.shared_features / "train_v129_testlike_target.npy"),
        1e-12,
    )
    train_risk = build_risk(
        regimes_train,
        probability_train,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    train_v140, _ = apply_risk_shift(train_v129, train_risk)
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    valid = truth_raw.sum(axis=(1, 2, 3)) > 0
    truth = normalize_support(truth_raw, 1e-12)
    base_score, gains = action_gains(train_v140, truth, valid)
    positive = (gains.max(axis=1) > ORACLE_POSITIVE_THRESHOLD).astype(np.int8)

    test_v140_target = normalize_support(
        np.load(
            PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
        ),
        1e-12,
    )
    features_train = decision_features(
        positions_train,
        blocked_train,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
        pointcloud["train_features"].astype(np.float32),
        probability_train,
        train_v140,
    )
    features_test = decision_features(
        positions_test,
        blocked_test,
        geodesic["test_features"].astype(np.float32),
        geodesic_names,
        pointcloud["test_features"].astype(np.float32),
        probability_test,
        test_v140_target,
    )
    nested = nested_spatial_oof(
        features_train,
        gains,
        positive,
        positions_train,
        valid,
        features_test,
    )
    for regime in REGIMES:
        mask = valid & (regimes_train == regime)
        nested["oof_report"]["by_regime"][regime] = {
            "count": int(mask.sum()),
            "gain": summarize_gain(nested["oof_gain"], mask),
            "active_fraction": float(
                np.mean(nested["oof_selected"][mask] > 0)
            ),
        }

    selected_test = nested["test_selected"]
    coefficient_test = np.zeros(len(selected_test), dtype=np.float32)
    active_test = selected_test > 0
    coefficient_test[active_test] = ACTIONS[
        selected_test[active_test].astype(np.int64) - 1
    ]
    application_report = apply_complex_diffusion(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        OUTPUT,
        positions_test,
        coefficient_test,
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
        raise ValueError("v146 output contains non-finite values")

    np.savez_compressed(
        DECISION_OUT,
        actions=ACTIONS,
        valid=valid,
        train_base_score=base_score,
        train_action_gains=gains,
        train_oof_benefit=nested["oof_benefit"],
        train_oof_uncertainty=nested["oof_uncertainty"],
        train_oof_probability=nested["oof_probability"],
        train_oof_selected=nested["oof_selected"],
        test_benefit=nested["test_benefit"],
        test_uncertainty=nested["test_uncertainty"],
        test_probability=nested["test_probability"],
        test_probability_uncertainty=nested[
            "test_probability_uncertainty"
        ],
        test_lower_bound=nested["test_lower_bound"],
        test_selected=selected_test,
        test_coefficient=coefficient_test,
    )

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352",
        "primary_hypothesis": (
            "replace the added hard-regime decision with three continuous "
            "outputs: probability of positive diffusion, expected benefit "
            "for each action, and ensemble uncertainty"
        ),
        "only_primary_change": (
            "apply confidence-bounded pure H log-support diffusion directly "
            "to the v140 complex output; v145 local residual field is excluded"
        ),
        "data_limit": (
            "train v129/v140 analogues retain the section-140 source-domain "
            "gap; five spatial fold models are averaged instead of full fitting"
        ),
        "model": {
            "feature_dimension": int(features_train.shape[1]),
            "features": (
                "continuous position/occlusion, unsaturated geodesic terms, "
                "point-cloud geometry, residual entropy/top1, source support signature"
            ),
            "actions": ACTIONS.tolist(),
            "oracle_positive_threshold": ORACLE_POSITIVE_THRESHOLD,
            "outputs": [
                "positive-action probability",
                "expected gain per diffusion action",
                "ensemble gain uncertainty per action",
            ],
            "selection": "maximize expected_gain-lambda*uncertainty",
            "hard_regime_gate_for_new_layer": False,
            "gain_clip": list(GAIN_CLIP),
        },
        "oracle": {
            "base_score": summarize(base_score[valid]),
            "best_gain": summarize_gain(gains.max(axis=1), valid),
            "positive_label_fraction": float(np.mean(positive[valid] > 0)),
            "best_action_counts": {
                f"{float(action):.2f}": int(
                    np.sum(valid & (np.argmax(gains, axis=1) == index))
                )
                for index, action in enumerate(ACTIONS)
            },
        },
        "nested_spatial_oof": {
            "modal_config": config_name(nested["modal_config"]),
            "config_counts": nested["config_counts"],
            "fold_reports": nested["fold_reports"],
            **nested["oof_report"],
        },
        "test_decision": {
            "continuous_risk_probability": summarize(
                nested["test_probability"]
            ),
            "probability_uncertainty": summarize(
                nested["test_probability_uncertainty"]
            ),
            "selected_benefit": summarize(
                np.max(nested["test_benefit"], axis=1)
            ),
            "selected_uncertainty": summarize(
                nested["test_uncertainty"][
                    np.arange(len(selected_test)),
                    np.maximum(selected_test.astype(np.int64) - 1, 0),
                ]
            ),
            "lower_bound": summarize(nested["test_lower_bound"]),
            "active_fraction": float(np.mean(active_test)),
            "action_counts": {
                "0.00": int(np.sum(selected_test == 0)),
                **{
                    f"{float(action):.2f}": int(np.sum(selected_test == index + 1))
                    for index, action in enumerate(ACTIONS)
                },
            },
            "active_by_legacy_regime_diagnostic_only": {
                regime: float(np.mean(active_test[regimes_test == regime]))
                for regime in REGIMES
            },
        },
        "application": application_report,
        "comparison_to_v140": comparison,
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

    with zipfile.ZipFile(
        ZIP_OUT, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source_path, name in (
            (__file__, "generate_v146_continuous_diffusion_gate.py"),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
            ),
            (
                PATHS.code / "generate_v143_transductive_geodesic_risk.py",
                "generate_v143_transductive_geodesic_risk.py",
            ),
            (
                PATHS.code
                / "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                "generate_v98_path_ensemble_graph_attention_joint_ad.py",
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
