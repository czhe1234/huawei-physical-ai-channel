#!/usr/bin/env python3
"""Generate v147 with continuous benefit/uncertainty gating of a local H field."""

from __future__ import annotations

import json
import zipfile

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from analyze_post_v143_hidden_information import normalize_without_floor, support_score
from analyze_v145_gated_local_h_field import local_h_predictions, stable_features
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    digest,
    output_comparison,
)
from generate_v146_continuous_diffusion_gate import (
    fit_ensemble,
    summarize_gain,
)
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import (
    ad_transform,
    beta_subcarriers,
    inverse_ad,
)
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


VERSION = "v147_continuous_local_action_gate"
OUT_DIR = PATHS.cache_dir(147)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(147)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v147_Round1_Test_Channel.npy"
DECISION_OUT = OUT_DIR / "v147_continuous_local_action_decisions.npz"
REPORT = REPORT_DIR / "v147_continuous_local_action_gate_generate.json"
ZIP_OUT = PATHS.round1_versions / "v147_continuous_local_action_gate_candidate_pending.zip"

RANDOM_STATE = 14701
K = 8
STRENGTHS = np.asarray([0.15, 0.25, 0.40, 0.60], dtype=np.float32)
POSITIVE_THRESHOLD = 0.00025
PROBABILITY_THRESHOLD = 0.55
UNCERTAINTY_PENALTY = 1.00
MINIMUM_LCB = 0.0
MAX_DISTANCE = 20.0
GAIN_CLIP = (0.45, 2.25)
SEEDS = (14701, 14731, 14759)
REGIMES = ("clear", "light", "medium", "heavy")


def local_descriptors(
    base_features: np.ndarray,
    prediction: np.ndarray,
    dispersion: np.ndarray,
    maximum_distance: np.ndarray,
) -> np.ndarray:
    summary = np.column_stack(
        [
            np.mean(prediction, axis=1),
            np.mean(np.abs(prediction), axis=1),
            np.max(np.abs(prediction), axis=1),
            np.std(prediction, axis=1),
            dispersion,
            maximum_distance,
            (maximum_distance <= MAX_DISTANCE).astype(np.float32),
        ]
    ).astype(np.float32)
    values = np.concatenate([base_features, prediction, summary], axis=1)
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def adjusted_support(
    current: np.ndarray,
    prediction: np.ndarray,
    strength: np.ndarray,
    active: np.ndarray | None = None,
) -> np.ndarray:
    if active is None:
        active = np.ones(len(current), dtype=bool)
    effective = np.where(active, strength, 0.0).astype(np.float32)
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0) * effective[:, None]
    ).astype(np.float32)
    output = current * ratio[:, :, None, None]
    output /= np.maximum(
        output.sum(axis=(1, 2, 3), keepdims=True), TINY
    )
    return output.astype(np.float32)


def action_gains(
    current: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    maximum_distance: np.ndarray,
) -> np.ndarray:
    base = support_score(current, truth)
    gains = np.zeros((len(current), len(STRENGTHS)), dtype=np.float32)
    within_range = maximum_distance <= MAX_DISTANCE
    for index, strength in enumerate(STRENGTHS):
        adjusted = adjusted_support(
            current,
            prediction,
            np.full(len(current), strength, dtype=np.float32),
            within_range,
        )
        gains[:, index] = support_score(adjusted, truth) - base
    return gains


def choose_actions(
    benefit: np.ndarray,
    uncertainty: np.ndarray,
    probability: np.ndarray,
    maximum_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower_bound = benefit - UNCERTAINTY_PENALTY * uncertainty
    best_index = np.argmax(lower_bound, axis=1)
    best_lower_bound = lower_bound[np.arange(len(lower_bound)), best_index]
    active = (
        (probability >= PROBABILITY_THRESHOLD)
        & (best_lower_bound > MINIMUM_LCB)
        & (maximum_distance <= MAX_DISTANCE)
    )
    selected = np.where(active, best_index + 1, 0).astype(np.int16)
    return selected, best_lower_bound.astype(np.float32)


def realized_gain(selected: np.ndarray, gains: np.ndarray) -> np.ndarray:
    output = np.zeros(len(selected), dtype=np.float32)
    active = selected > 0
    output[active] = gains[
        np.flatnonzero(active), selected[active].astype(np.int64) - 1
    ]
    return output


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
    oof_benefit = np.zeros((len(current), len(STRENGTHS)), dtype=np.float32)
    oof_uncertainty = np.zeros_like(oof_benefit)
    oof_probability = np.zeros(len(current), dtype=np.float32)
    oof_selected = np.zeros(len(current), dtype=np.int16)
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
        inner_gains = action_gains(
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
        local_train_index = np.arange(len(outer_train), dtype=np.int64)
        validation_model = fit_ensemble(
            inner_features,
            inner_gains,
            inner_positive,
            local_train_index,
            validation_features,
            SEEDS,
        )
        validation_selected, validation_lower_bound = choose_actions(
            validation_model[0],
            validation_model[1],
            validation_model[2],
            validation_distance[K],
        )
        validation_gains = action_gains(
            current[outer_validation],
            truth[outer_validation],
            validation_prediction[K],
            validation_distance[K],
        )
        validation_realized = realized_gain(
            validation_selected, validation_gains
        )
        oof_benefit[outer_validation] = validation_model[0]
        oof_uncertainty[outer_validation] = validation_model[1]
        oof_probability[outer_validation] = validation_model[2]
        oof_selected[outer_validation] = validation_selected
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
                        (len(positions_test), h_target.shape[1]),
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            ),
            outer_train,
            np.arange(
                len(positions_train),
                len(positions_train) + len(positions_test),
            ),
        )
        test_features = local_descriptors(
            base_features_test,
            test_prediction[K],
            test_dispersion[K],
            test_distance[K],
        )
        test_model = fit_ensemble(
            inner_features,
            inner_gains,
            inner_positive,
            local_train_index,
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
                "train_count": int(len(outer_train)),
                "validation_count": int(len(outer_validation)),
                "gain": summarize_gain(
                    validation_realized,
                    np.ones(len(validation_realized), dtype=bool),
                ),
                "active_fraction": float(
                    np.mean(validation_selected > 0)
                ),
                "continuous_risk": summarize(validation_model[2]),
                "lower_bound": summarize(validation_lower_bound),
            }
        )

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
            np.mean(
                np.stack(test_probability_uncertainty_folds, axis=0) ** 2,
                axis=0,
            )
            + test_probability_stack.var(axis=0),
            0.0,
        )
    ).astype(np.float32)
    test_prediction = np.stack(test_prediction_folds, axis=0).mean(axis=0)
    test_dispersion = np.stack(test_dispersion_folds, axis=0).mean(axis=0)
    test_distance = np.stack(test_distance_folds, axis=0).mean(axis=0)
    test_selected, test_lower_bound = choose_actions(
        test_benefit,
        test_uncertainty,
        test_probability,
        test_distance,
    )
    return {
        "oof_benefit": oof_benefit,
        "oof_uncertainty": oof_uncertainty,
        "oof_probability": oof_probability,
        "oof_selected": oof_selected,
        "oof_maximum_distance": oof_maximum_distance,
        "oof_realized_gain": oof_realized_gain,
        "oof_action_gains": oof_action_gains,
        "oof_fold": oof_fold,
        "test_benefit": test_benefit,
        "test_uncertainty": test_uncertainty,
        "test_probability": test_probability,
        "test_probability_uncertainty": test_probability_uncertainty,
        "test_prediction": test_prediction.astype(np.float32),
        "test_dispersion": test_dispersion.astype(np.float32),
        "test_distance": test_distance.astype(np.float32),
        "test_selected": test_selected,
        "test_lower_bound": test_lower_bound,
        "fold_reports": fold_reports,
    }


def apply_complex_local_field(
    base_path,
    output_path,
    positions: np.ndarray,
    prediction: np.ndarray,
    strength: np.ndarray,
) -> dict[str, object]:
    base = np.load(base_path, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=base.shape
    )
    beta = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(positions - BS_POS[None, :], axis=1)
    target_cosine = []
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
        support1 = adjusted_support(
            support0,
            prediction[start:end],
            strength[start:end],
        )
        power_ratio = np.clip(
            support1 / np.maximum(support0, 1e-12),
            GAIN_CLIP[0] ** 2,
            GAIN_CLIP[1] ** 2,
        )
        support1 = support0 * power_ratio
        support1 /= np.maximum(
            support1.sum(axis=(1, 2, 3), keepdims=True), TINY
        )
        gain = np.sqrt(
            support1 / np.maximum(support0, 1e-12)
        ).astype(np.float32)
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
        target_cosine.extend(row_cos(support_after, support1).tolist())
        gain_minimum.extend(gain.reshape(end - start, -1).min(axis=1).tolist())
        gain_maximum.extend(gain.reshape(end - start, -1).max(axis=1).tolist())
    return {
        "target_cosine_after": summarize(np.asarray(target_cosine)),
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
    pointcloud = np.load(
        PATHS.shared_features / "official_pointcloud_scene_cache.npz"
    )
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
        np.load(
            PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
        ),
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

    active_oof = nested["oof_selected"] > 0
    fold_gain_means = [row["gain"]["mean"] for row in nested["fold_reports"]]

    test_strength = np.zeros(len(positions_test), dtype=np.float32)
    active_test = nested["test_selected"] > 0
    test_strength[active_test] = STRENGTHS[
        nested["test_selected"][active_test].astype(np.int64) - 1
    ]
    application = apply_complex_local_field(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy",
        OUTPUT,
        positions_test,
        nested["test_prediction"],
        test_strength,
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
        raise ValueError("v147 output contains non-finite values")

    np.savez_compressed(
        DECISION_OUT,
        strengths=STRENGTHS,
        train_oof_benefit=nested["oof_benefit"],
        train_oof_uncertainty=nested["oof_uncertainty"],
        train_oof_probability=nested["oof_probability"],
        train_oof_selected=nested["oof_selected"],
        train_oof_maximum_distance=nested["oof_maximum_distance"],
        train_oof_realized_gain=nested["oof_realized_gain"],
        train_oof_action_gains=nested["oof_action_gains"],
        train_oof_fold=nested["oof_fold"],
        test_benefit=nested["test_benefit"],
        test_uncertainty=nested["test_uncertainty"],
        test_probability=nested["test_probability"],
        test_probability_uncertainty=nested[
            "test_probability_uncertainty"
        ],
        test_prediction=nested["test_prediction"],
        test_dispersion=nested["test_dispersion"],
        test_distance=nested["test_distance"],
        test_lower_bound=nested["test_lower_bound"],
        test_selected=nested["test_selected"],
        test_strength=test_strength,
    )

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352",
        "primary_hypothesis": (
            "the positive local H residual action can be made problem-specific "
            "by learning continuous action probability, benefit, and uncertainty"
        ),
        "only_primary_change": (
            "confidence-bounded selection among local H field strengths; no "
            "v145 fixed all-covered action and no new hard regime gate"
        ),
        "model": {
            "base_feature_dimension": int(base_features_train.shape[1]),
            "decision_feature_dimension": int(
                base_features_train.shape[1] + 16 + 7
            ),
            "local_neighbors": K,
            "maximum_distance": MAX_DISTANCE,
            "actions": STRENGTHS.tolist(),
            "positive_threshold": POSITIVE_THRESHOLD,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "uncertainty_penalty": UNCERTAINTY_PENALTY,
            "minimum_lcb": MINIMUM_LCB,
            "outputs": [
                "positive-action probability",
                "expected full-support gain per strength",
                "ensemble gain uncertainty per strength",
            ],
            "hard_regime_gate_for_new_layer": False,
            "five_fold_models_used_at_test": True,
        },
        "nested_spatial_oof": {
            "folds": nested["fold_reports"],
            "fold_gain_means": fold_gain_means,
            "mean_of_fold_means": float(np.mean(fold_gain_means)),
            "all_folds_positive": bool(np.all(np.asarray(fold_gain_means) > 0)),
            "full_gain": summarize_gain(nested["oof_realized_gain"], valid),
            "active_fraction": float(np.mean(active_oof[valid])),
            "active_gain": (
                summarize_gain(
                    nested["oof_realized_gain"], valid & active_oof
                )
                if np.any(valid & active_oof)
                else {}
            ),
            "by_regime": {
                regime: {
                    "count": int(np.sum(valid & (regimes_train == regime))),
                    "gain": summarize_gain(
                        nested["oof_realized_gain"],
                        valid & (regimes_train == regime),
                    ),
                    "active_fraction": float(
                        np.mean(
                            active_oof[valid & (regimes_train == regime)]
                        )
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
            "active_fraction": float(np.mean(active_test)),
            "action_counts": {
                "0.00": int(np.sum(nested["test_selected"] == 0)),
                **{
                    f"{float(value):.2f}": int(
                        np.sum(nested["test_selected"] == index + 1)
                    )
                    for index, value in enumerate(STRENGTHS)
                },
            },
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
            "v145_fixed_strength_everywhere": True,
            "v144_v_correction": True,
            "new_fixed_h_shift": True,
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
    with zipfile.ZipFile(
        ZIP_OUT, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source_path, name in (
            (__file__, "generate_v147_continuous_local_action_gate.py"),
            (
                PATHS.code / "analyze_v145_gated_local_h_field.py",
                "analyze_v145_gated_local_h_field.py",
            ),
            (
                PATHS.code / "generate_v146_continuous_diffusion_gate.py",
                "generate_v146_continuous_diffusion_gate.py",
            ),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
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
