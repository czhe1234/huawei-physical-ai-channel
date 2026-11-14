#!/usr/bin/env python3
"""Generate v151 with geometry-learned continuous complexity gating."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from analyze_continuous_propagation_complexity import (
    channel_labels,
    make_geometry_features,
)
from analyze_post_v143_hidden_information import normalize_without_floor, support_score
from analyze_v145_gated_local_h_field import local_h_predictions
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    digest,
    output_comparison,
)
from generate_v145_local_h_residual_field import local_h_field
from generate_v147_continuous_local_action_gate import adjusted_support
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v151_complexity_gated_local_h_field"
OUT_DIR = PATHS.cache_dir(151)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(151)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v151_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v151_complexity_gated_local_h_field_target.npy"
FIELD_OUT = OUT_DIR / "v151_complexity_gated_local_h_field.npz"
REPORT = REPORT_DIR / "v151_complexity_gated_local_h_field_generate.json"
ZIP_OUT = PATHS.round1_versions / "v151_complexity_gated_local_h_field_candidate_pending.zip"

RANDOM_STATE = 151
K = 8
MAX_DISTANCE = 20.0
CONFIGS = tuple(
    (strength, h_quantile, c_quantile)
    for strength in (0.15, 0.25, 0.35)
    for h_quantile in (0.40, 0.50, 0.60, 0.70)
    for c_quantile in (0.30, 0.40, 0.50, 0.60, 0.70)
)
STRENGTHS = np.asarray([0.15, 0.25, 0.35], dtype=np.float32)
REGIMES = ("clear", "light", "medium", "heavy")


def config_name(config: tuple[float, float, float]) -> str:
    strength, h_quantile, c_quantile = config
    return f"s{strength:.2f}_hq{h_quantile:.2f}_cq{c_quantile:.2f}"


def fit_complexity(
    features: np.ndarray,
    target: np.ndarray,
    train_index: np.ndarray,
    evaluation_features: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=220,
        max_features=0.65,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features[train_index], target[train_index])
    return model.predict(evaluation_features).astype(np.float32)


def action_gain(
    current: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    distance: np.ndarray,
    strength: np.ndarray,
) -> np.ndarray:
    base = support_score(current, truth)
    adjusted = adjusted_support(
        current,
        prediction,
        strength.astype(np.float32),
        distance <= MAX_DISTANCE,
    )
    return (support_score(adjusted, truth) - base).astype(np.float32)


def action_gain_matrix(
    current: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    gains = np.zeros((len(current), len(STRENGTHS)), dtype=np.float32)
    for index, strength in enumerate(STRENGTHS):
        gains[:, index] = action_gain(
            current,
            truth,
            prediction,
            distance,
            np.full(len(current), strength, dtype=np.float32),
        )
    return gains


def realized_from_matrix(
    gains: np.ndarray,
    active: np.ndarray,
    strength: float,
) -> np.ndarray:
    output = np.zeros(len(active), dtype=np.float32)
    index = int(np.flatnonzero(np.isclose(STRENGTHS, strength))[0])
    output[active] = gains[active, index]
    return output


def select_by_config(
    h_pred: np.ndarray,
    c_pred: np.ndarray,
    distance: np.ndarray,
    blocked: np.ndarray,
    config: tuple[float, float, float],
    threshold_source: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    strength, h_quantile, c_quantile = config
    source_h = h_pred if threshold_source is None else threshold_source[0]
    source_c = c_pred if threshold_source is None else threshold_source[1]
    h_threshold = float(np.quantile(source_h, h_quantile))
    c_threshold = float(np.quantile(source_c, c_quantile))
    active = (
        (blocked > 0.0)
        & (distance <= MAX_DISTANCE)
        & (h_pred >= h_threshold)
        & (c_pred >= c_threshold)
    )
    strength_vector = np.where(active, strength, 0.0).astype(np.float32)
    return strength_vector, active, h_threshold, c_threshold


def summarize_gain(gain: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    values = gain[mask]
    if len(values) == 0:
        return {"count": 0, "mean": 0.0, "positive_fraction": 0.0}
    return {**summarize(values), "positive_fraction": float(np.mean(values > 0.0))}


def nested_oof(
    positions_train: np.ndarray,
    positions_test: np.ndarray,
    features_train: np.ndarray,
    features_test: np.ndarray,
    current: np.ndarray,
    truth: np.ndarray,
    h_target: np.ndarray,
    complexity_h: np.ndarray,
    complexity_c: np.ndarray,
    blocked_train: np.ndarray,
    blocked_test: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    groups = KMeans(n_clusters=25, random_state=RANDOM_STATE, n_init=20).fit_predict(
        positions_train[:, :2]
    )
    selected_valid = np.flatnonzero(valid)
    outer = GroupKFold(n_splits=5)
    oof_gain = np.zeros(len(positions_train), dtype=np.float32)
    oof_strength = np.zeros(len(positions_train), dtype=np.float32)
    oof_h_pred = np.zeros(len(positions_train), dtype=np.float32)
    oof_c_pred = np.zeros(len(positions_train), dtype=np.float32)
    oof_distance = np.zeros(len(positions_train), dtype=np.float32)
    test_prediction_folds = []
    test_h_folds = []
    test_c_folds = []
    test_distance_folds = []
    fold_reports = []
    selected_configs = []

    for fold, (outer_train_local, outer_validation_local) in enumerate(
        outer.split(selected_valid, groups=groups[selected_valid])
    ):
        outer_train = selected_valid[outer_train_local]
        outer_validation = selected_valid[outer_validation_local]
        inner = GroupKFold(n_splits=4)
        inner_prediction = np.zeros((len(positions_train), h_target.shape[1]), dtype=np.float32)
        inner_distance = np.zeros(len(positions_train), dtype=np.float32)
        inner_h_pred = np.zeros(len(positions_train), dtype=np.float32)
        inner_c_pred = np.zeros(len(positions_train), dtype=np.float32)
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
            inner.split(outer_train, groups=groups[outer_train])
        ):
            inner_train = outer_train[inner_train_local]
            inner_validation = outer_train[inner_validation_local]
            prediction, _, distance = local_h_predictions(
                positions_train, h_target, inner_train, inner_validation
            )
            inner_prediction[inner_validation] = prediction[K]
            inner_distance[inner_validation] = distance[K]
            inner_h_pred[inner_validation] = fit_complexity(
                features_train,
                complexity_h,
                inner_train,
                features_train[inner_validation],
                RANDOM_STATE + 10 + inner_fold,
            )
            inner_c_pred[inner_validation] = fit_complexity(
                features_train,
                complexity_c,
                inner_train,
                features_train[inner_validation],
                RANDOM_STATE + 20 + inner_fold,
            )

        inner_gain_matrix = action_gain_matrix(
            current[outer_train],
            truth[outer_train],
            inner_prediction[outer_train],
            inner_distance[outer_train],
        )
        best_config = CONFIGS[0]
        best_value = -np.inf
        config_rows = []
        for config in CONFIGS:
            strength, active, h_threshold, c_threshold = select_by_config(
                inner_h_pred[outer_train],
                inner_c_pred[outer_train],
                inner_distance[outer_train],
                blocked_train[outer_train],
                config,
            )
            gain = realized_from_matrix(inner_gain_matrix, active, config[0])
            full_gain = float(np.mean(gain))
            active_gain = float(np.mean(gain[active])) if np.any(active) else 0.0
            active_fraction = float(np.mean(active))
            row = {
                "config": config_name(config),
                "full_gain_mean": full_gain,
                "active_gain_mean": active_gain,
                "active_fraction": active_fraction,
                "positive_fraction": float(np.mean(gain[active] > 0.0)) if np.any(active) else 0.0,
                "h_threshold": h_threshold,
                "c_threshold": c_threshold,
            }
            config_rows.append(row)
            key = (full_gain, active_gain, -active_fraction)
            if key > (best_value, -np.inf, -np.inf):
                best_value = full_gain
                best_config = config
        config_rows.sort(key=lambda item: item["full_gain_mean"], reverse=True)
        selected_configs.append(best_config)

        validation_prediction, _, validation_distance = local_h_predictions(
            positions_train, h_target, outer_train, outer_validation
        )
        validation_h = fit_complexity(
            features_train,
            complexity_h,
            outer_train,
            features_train[outer_validation],
            RANDOM_STATE + 100 + fold,
        )
        validation_c = fit_complexity(
            features_train,
            complexity_c,
            outer_train,
            features_train[outer_validation],
            RANDOM_STATE + 200 + fold,
        )
        threshold_source = (inner_h_pred[outer_train], inner_c_pred[outer_train])
        strength, active, h_threshold, c_threshold = select_by_config(
            validation_h,
            validation_c,
            validation_distance[K],
            blocked_train[outer_validation],
            best_config,
            threshold_source=threshold_source,
        )
        validation_gain_matrix = action_gain_matrix(
            current[outer_validation],
            truth[outer_validation],
            validation_prediction[K],
            validation_distance[K],
        )
        gain = realized_from_matrix(validation_gain_matrix, active, best_config[0])
        oof_gain[outer_validation] = gain
        oof_strength[outer_validation] = strength
        oof_h_pred[outer_validation] = validation_h
        oof_c_pred[outer_validation] = validation_c
        oof_distance[outer_validation] = validation_distance[K]

        combined_pos = np.concatenate([positions_train, positions_test], axis=0)
        combined_target = np.concatenate(
            [
                h_target,
                np.zeros((len(positions_test), h_target.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        test_prediction, _, test_distance = local_h_predictions(
            combined_pos,
            combined_target,
            outer_train,
            np.arange(len(positions_train), len(positions_train) + len(positions_test)),
        )
        test_h = fit_complexity(
            features_train,
            complexity_h,
            outer_train,
            features_test,
            RANDOM_STATE + 300 + fold,
        )
        test_c = fit_complexity(
            features_train,
            complexity_c,
            outer_train,
            features_test,
            RANDOM_STATE + 400 + fold,
        )
        test_prediction_folds.append(test_prediction[K])
        test_h_folds.append(test_h)
        test_c_folds.append(test_c)
        test_distance_folds.append(test_distance[K])

        fold_reports.append(
            {
                "fold": fold,
                "selected_config": config_name(best_config),
                "validation_gain": summarize_gain(gain, np.ones(len(gain), dtype=bool)),
                "active_fraction": float(np.mean(active)),
                "active_gain": summarize_gain(gain, active),
                "h_threshold": h_threshold,
                "c_threshold": c_threshold,
                "top_inner_configs": config_rows[:10],
            }
        )

    config_counts = Counter(config_name(config) for config in selected_configs)
    modal_name = config_counts.most_common(1)[0][0]
    modal_config = next(config for config in CONFIGS if config_name(config) == modal_name)
    test_prediction = np.mean(np.stack(test_prediction_folds, axis=0), axis=0).astype(np.float32)
    test_h = np.mean(np.stack(test_h_folds, axis=0), axis=0).astype(np.float32)
    test_c = np.mean(np.stack(test_c_folds, axis=0), axis=0).astype(np.float32)
    test_distance = np.mean(np.stack(test_distance_folds, axis=0), axis=0).astype(np.float32)
    threshold_source = (oof_h_pred[valid], oof_c_pred[valid])
    test_strength, test_active, test_h_threshold, test_c_threshold = select_by_config(
        test_h,
        test_c,
        test_distance,
        blocked_test,
        modal_config,
        threshold_source=threshold_source,
    )
    return {
        "oof_gain": oof_gain,
        "oof_strength": oof_strength,
        "oof_h_pred": oof_h_pred,
        "oof_c_pred": oof_c_pred,
        "oof_distance": oof_distance,
        "fold_reports": fold_reports,
        "config_counts": dict(config_counts),
        "modal_config": modal_config,
        "test_prediction": test_prediction,
        "test_h_pred": test_h,
        "test_c_pred": test_c,
        "test_distance": test_distance,
        "test_strength": test_strength,
        "test_active": test_active,
        "test_h_threshold": test_h_threshold,
        "test_c_threshold": test_c_threshold,
    }


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
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
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(np.float32)
    valid = truth_raw.sum(axis=(1, 2, 3)) > 0
    truth = normalize_without_floor(truth_raw)
    train_current_h = train_current.sum(axis=(2, 3))
    truth_h = truth.sum(axis=(2, 3))
    h_target = np.clip(
        np.log(np.maximum(truth_h, 1e-10))
        - np.log(np.maximum(train_current_h, 1e-10)),
        -3.0,
        3.0,
    ).astype(np.float32)
    complexity_labels_valid, _ = channel_labels(truth_raw, positions_train, valid)
    complexity_h = np.zeros(len(positions_train), dtype=np.float32)
    complexity_c = np.zeros(len(positions_train), dtype=np.float32)
    complexity_h[valid] = complexity_labels_valid["h_spread"]
    complexity_c[valid] = complexity_labels_valid["local_h_coherence_20m"]
    features_train_all, _ = make_geometry_features(
        positions_train,
        blocked_train,
        geodesic["train_features"].astype(np.float32),
        pointcloud["train_features"].astype(np.float32),
    )
    features_test, _ = make_geometry_features(
        positions_test,
        blocked_test,
        geodesic["test_features"].astype(np.float32),
        pointcloud["test_features"].astype(np.float32),
    )
    features_train = features_train_all
    nested = nested_oof(
        positions_train,
        positions_test,
        features_train,
        features_test,
        train_current,
        truth,
        h_target,
        complexity_h,
        complexity_c,
        blocked_train,
        blocked_test,
        valid,
    )

    v140_target = normalize_support(
        np.load(PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"),
        1e-12,
    )
    prediction, dispersion, v145_gate, neighbors, graph_values = local_h_field(
        positions_train,
        positions_test,
        h_target,
        valid,
    )
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0) * nested["test_strength"][:, None]
    ).astype(np.float32)
    final_target = (v140_target * ratio[:, :, None, None]).astype(np.float32)
    final_target /= np.maximum(final_target.sum(axis=(1, 2, 3), keepdims=True), TINY)
    np.save(TARGET_OUT, final_target)
    np.savez_compressed(
        FIELD_OUT,
        prediction=prediction,
        dispersion=dispersion,
        v145_gate=v145_gate,
        selected_strength=nested["test_strength"],
        active=nested["test_active"],
        h_spread_pred=nested["test_h_pred"],
        local_h_coherence_pred=nested["test_c_pred"],
        h_threshold=nested["test_h_threshold"],
        c_threshold=nested["test_c_threshold"],
        neighbor_distances=graph_values[:, :K],
        neighbor_weights=graph_values[:, K:],
        neighbor_indices=neighbors,
    )

    weights = np.load(
        PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_weights.npy"
    ).astype(np.float32)
    args = argparse.Namespace(
        base=str(PATHS.cache_dir(129) / "v129_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.028,medium:0.470,heavy:0.780",
        min_gate=0.12,
        max_gate=2.95,
        conf_center=0.22,
        conf_width=0.38,
        entropy_low=0.02,
        entropy_high=0.90,
        gain_clip="0.34,3.20",
        top_gain_clip="0.48,2.35",
        top_components=5632,
        target_floor=1e-12,
        project_batch=20,
    )
    projection_report = projection(
        final_target, weights, regimes_test, positions_test.astype(np.float64), args
    )
    output = np.load(OUTPUT, mmap_mode="r")
    baseline = np.load(PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", mmap_mode="r")
    comparison = output_comparison(output, baseline, regimes_test)
    if output.shape != baseline.shape or output.dtype != np.complex64:
        raise ValueError(f"invalid output layout: shape={output.shape} dtype={output.dtype}")
    if not comparison["finite"]:
        raise ValueError("v151 output contains non-finite values")

    active_oof = valid & (nested["oof_strength"] > 0)
    active_test = nested["test_strength"] > 0
    v140_h = v140_target.sum(axis=(2, 3))
    final_h = final_target.sum(axis=(2, 3))
    report_args = dict(vars(args))
    report_args["base"] = PATHS.relative(args.base)
    report_args["output"] = PATHS.relative(args.output)
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352; v145 local-H field action",
        "primary_hypothesis": (
            "replace the crude heavy split with geometry-learned continuous "
            "h_spread and local_h_coherence gates for the local H residual field"
        ),
        "only_primary_change": (
            "v145 local H field strength is applied only when geometry-predicted "
            "H angular spread and local H coherence exceed learned quantile gates"
        ),
        "model": {
            "complexity_targets": ["h_spread", "local_h_coherence_20m"],
            "candidate_configs": len(CONFIGS),
            "maximum_distance": MAX_DISTANCE,
            "modal_config": config_name(nested["modal_config"]),
            "config_counts": nested["config_counts"],
            "hard_heavy_gate": False,
            "blocked_clear_suppression": True,
        },
        "nested_spatial_oof": {
            "folds": nested["fold_reports"],
            "full_gain": summarize_gain(nested["oof_gain"], valid),
            "active_fraction": float(np.mean(active_oof[valid])),
            "active_gain": summarize_gain(nested["oof_gain"], active_oof),
            "all_fold_means": [
                row["validation_gain"]["mean"] for row in nested["fold_reports"]
            ],
            "all_folds_positive": bool(
                np.all([row["validation_gain"]["mean"] > 0 for row in nested["fold_reports"]])
            ),
            "by_regime": {
                regime: {
                    "count": int(np.sum(valid & (regimes_train == regime))),
                    "gain": summarize_gain(
                        nested["oof_gain"], valid & (regimes_train == regime)
                    ),
                    "active_fraction": float(
                        np.mean(active_oof[valid & (regimes_train == regime)])
                    ),
                }
                for regime in REGIMES
            },
        },
        "test_decision": {
            "active_fraction": float(np.mean(active_test)),
            "selected_strength": summarize(nested["test_strength"]),
            "h_spread_pred": summarize(nested["test_h_pred"]),
            "local_h_coherence_pred": summarize(nested["test_c_pred"]),
            "h_threshold": nested["test_h_threshold"],
            "c_threshold": nested["test_c_threshold"],
            "active_by_regime": {
                regime: float(np.mean(active_test[regimes_test == regime]))
                for regime in REGIMES
            },
        },
        "target_diagnostics": {
            "target_to_v140_target": summarize(row_cos(final_target, v140_target)),
            "h_marginal_l1_change": summarize(np.sum(np.abs(final_h - v140_h), axis=1)),
            "h_marginal_max_ratio": summarize(
                np.max(final_h / np.maximum(v140_h, 1e-12), axis=1)
            ),
        },
        "projection": projection_report,
        "comparison_to_v140": comparison,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "field": PATHS.relative(FIELD_OUT),
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
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source_path, name in (
            (__file__, "generate_v151_complexity_gated_local_h_field.py"),
            (
                PATHS.code / "analyze_continuous_propagation_complexity.py",
                "analyze_continuous_propagation_complexity.py",
            ),
            (
                PATHS.code / "generate_v145_local_h_residual_field.py",
                "generate_v145_local_h_residual_field.py",
            ),
            (
                PATHS.code / "generate_v147_continuous_local_action_gate.py",
                "generate_v147_continuous_local_action_gate.py",
            ),
            (
                PATHS.code / "analyze_v145_gated_local_h_field.py",
                "analyze_v145_gated_local_h_field.py",
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
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ZIP_OUT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
