#!/usr/bin/env python3
"""Analyze why the v149/v150 action controller did not transfer online."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from analyze_post_v143_hidden_information import normalize_without_floor
from analyze_v145_gated_local_h_field import stable_features
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk
from generate_v143_transductive_geodesic_risk import apply_risk_shift
from generate_v149_continuous_action_response_controller import (
    choose_strengths,
    config_name,
)
from generate_v87_easy_field_projection import summarize
from powerfield_common import BS_POS, regime_of
from project_paths import PATHS


OUT_DIR = PATHS.version_report_dir(150)
REPORT = OUT_DIR / "v150_failure_data_signal_analysis.json"
BIN_CSV = OUT_DIR / "v150_failure_score_bins.csv"
TOP_SHIFT_CSV = OUT_DIR / "v150_failure_top_shift_features.csv"

MODAL_CONFIG = (0.60, 0.75, 0.0, 0.85)
REGIMES = ("clear", "light", "medium", "heavy")


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return summarize(values)


def psi(train: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if len(train) == 0 or len(test) == 0:
        return 0.0
    edges = np.quantile(train, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_hist, _ = np.histogram(train, edges)
    test_hist, _ = np.histogram(test, edges)
    train_rate = np.maximum(train_hist / max(train_hist.sum(), 1), 1e-6)
    test_rate = np.maximum(test_hist / max(test_hist.sum(), 1), 1e-6)
    return float(np.sum((test_rate - train_rate) * np.log(test_rate / train_rate)))


def top_univariate_shift(
    names: list[str],
    train: np.ndarray,
    test: np.ndarray,
    top_k: int = 40,
) -> list[dict[str, object]]:
    rows = []
    for index, name in enumerate(names):
        train_values = train[:, index]
        test_values = test[:, index]
        ks = float(ks_2samp(train_values, test_values).statistic)
        train_std = float(np.std(train_values))
        mean_shift = float(
            (np.mean(test_values) - np.mean(train_values))
            / max(train_std, 1e-6)
        )
        rows.append(
            {
                "feature": name,
                "ks": ks,
                "psi": psi(train_values, test_values),
                "train_mean": float(np.mean(train_values)),
                "test_mean": float(np.mean(test_values)),
                "mean_shift_train_std": mean_shift,
            }
        )
    rows.sort(key=lambda item: (item["ks"], abs(item["mean_shift_train_std"])), reverse=True)
    return rows[:top_k]


def domain_auc(features_train: np.ndarray, features_test: np.ndarray) -> dict[str, object]:
    x = np.vstack([features_train, features_test]).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(len(features_train), dtype=np.int8),
            np.ones(len(features_test), dtype=np.int8),
        ]
    )
    clf = ExtraTreesClassifier(
        n_estimators=260,
        max_features=0.55,
        min_samples_leaf=8,
        random_state=15017,
        n_jobs=-1,
        class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=15017)
    proba = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "auc": float(roc_auc_score(y, proba)),
        "train_score": finite_summary(proba[y == 0]),
        "test_score": finite_summary(proba[y == 1]),
    }


def bin_table(
    name: str,
    train_metric: np.ndarray,
    test_metric: np.ndarray,
    gain: np.ndarray,
    train_active: np.ndarray,
    test_active: np.ndarray,
    valid: np.ndarray,
) -> list[dict[str, object]]:
    train_values = train_metric[valid]
    test_values = test_metric
    edges = np.quantile(train_values, np.linspace(0.0, 1.0, 11))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return []
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_bin = np.digitize(train_metric, edges[1:-1], right=True)
    test_bin = np.digitize(test_values, edges[1:-1], right=True)
    rows = []
    for bin_index in range(len(edges) - 1):
        train_mask = valid & (train_bin == bin_index)
        test_mask = test_bin == bin_index
        rows.append(
            {
                "metric": name,
                "bin": int(bin_index),
                "low": float(edges[bin_index]),
                "high": float(edges[bin_index + 1]),
                "train_count": int(np.sum(train_mask)),
                "train_gain_mean": (
                    float(np.mean(gain[train_mask])) if np.any(train_mask) else 0.0
                ),
                "train_positive_fraction": (
                    float(np.mean(gain[train_mask] > 0.0))
                    if np.any(train_mask)
                    else 0.0
                ),
                "train_active_fraction": (
                    float(np.mean(train_active[train_mask]))
                    if np.any(train_mask)
                    else 0.0
                ),
                "test_count": int(np.sum(test_mask)),
                "test_fraction": float(np.mean(test_mask)),
                "test_active_fraction": (
                    float(np.mean(test_active[test_mask]))
                    if np.any(test_mask)
                    else 0.0
                ),
            }
        )
    return rows


def by_regime(
    regimes: np.ndarray,
    gain: np.ndarray | None,
    active: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, object]:
    output = {}
    if valid is None:
        valid = np.ones(len(regimes), dtype=bool)
    for regime in REGIMES:
        mask = valid & (regimes == regime)
        row = {
            "count": int(np.sum(mask)),
            "active_fraction": float(np.mean(active[mask])) if np.any(mask) else 0.0,
        }
        if gain is not None:
            row["gain"] = finite_summary(gain[mask])
            row["positive_fraction"] = (
                float(np.mean(gain[mask] > 0.0)) if np.any(mask) else 0.0
            )
        output[regime] = row
    return output


def strength_counts(strength: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.round(strength.astype(np.float64), 4), return_counts=True)
    return {f"{value:.4f}": int(count) for value, count in zip(values, counts)}


def build_stable_feature_domain() -> tuple[np.ndarray, np.ndarray]:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
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
        regime_of(blocked_train),
        residual_probability,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    current, _ = apply_risk_shift(source, risk)
    test_target = normalize_support(
        np.load(PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"),
        1e-12,
    )
    return (
        stable_features(
            "train", positions_train, blocked_train, current, geodesic, pointcloud
        ),
        stable_features(
            "test", positions_test, blocked_test, test_target, geodesic, pointcloud
        ),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    decisions = np.load(
        PATHS.cache_dir(149) / "v149_continuous_action_response_decisions.npz"
    )
    v149_report = json.loads(
        (
            PATHS.version_report_dir(149)
            / "v149_continuous_action_response_controller_generate.json"
        ).read_text(encoding="utf-8")
    )
    v150_report = json.loads(
        (
            PATHS.version_report_dir(150)
            / "v150_los_safe_continuous_action_controller_generate.json"
        ).read_text(encoding="utf-8")
    )

    valid = decisions["valid"].astype(bool)
    train_probability = decisions["train_oof_probability"].astype(np.float32)
    train_benefit = decisions["train_oof_benefit"].astype(np.float32)
    train_uncertainty = decisions["train_oof_uncertainty"].astype(np.float32)
    train_lower_bound = decisions["train_oof_lower_bound"].astype(np.float32)
    train_selected_uncertainty = decisions[
        "train_oof_selected_uncertainty"
    ].astype(np.float32)
    train_distance = decisions["train_oof_maximum_distance"].astype(np.float32)
    train_gain = decisions["train_oof_realized_gain"].astype(np.float32)
    train_active_fold_config = decisions["train_oof_selected"].astype(np.int16) > 0
    train_strength_fold_config = decisions["train_oof_strength"].astype(np.float32)
    test_probability = decisions["test_probability"].astype(np.float32)
    test_benefit = decisions["test_benefit"].astype(np.float32)
    test_uncertainty = decisions["test_uncertainty"].astype(np.float32)
    test_lower_bound = decisions["test_lower_bound"].astype(np.float32)
    test_selected_uncertainty = decisions["test_selected_uncertainty"].astype(np.float32)
    test_distance = decisions["test_distance"].astype(np.float32)
    test_strength_v149 = decisions["test_strength"].astype(np.float32)
    test_strength_v150 = test_strength_v149.copy()
    test_strength_v150[blocked_test <= 0.0] = 0.0
    test_active_v149 = test_strength_v149 > 0.0
    test_active_v150 = test_strength_v150 > 0.0
    train_max_benefit = np.max(train_benefit, axis=1)
    test_max_benefit = np.max(test_benefit, axis=1)

    modal_train = choose_strengths(
        train_benefit, train_uncertainty, train_probability, train_distance, MODAL_CONFIG
    )
    modal_test = choose_strengths(
        test_benefit, test_uncertainty, test_probability, test_distance, MODAL_CONFIG
    )
    train_modal_strength, train_modal_selected = modal_train[0], modal_train[1]
    train_modal_active = train_modal_selected > 0
    test_modal_strength, test_modal_selected = modal_test[0], modal_test[1]
    test_modal_active = test_modal_selected > 0
    train_modal_gain = np.zeros(len(train_gain), dtype=np.float32)
    action_gains = decisions["train_oof_action_gains"].astype(np.float32)
    modal_active_rows = train_modal_selected > 0
    train_modal_gain[modal_active_rows] = action_gains[
        np.flatnonzero(modal_active_rows),
        train_modal_selected[modal_active_rows].astype(np.int64) - 1,
    ]

    gate_train_threshold = float(np.quantile(modal_train[3][valid], MODAL_CONFIG[3]))
    gate_test_threshold = float(np.quantile(modal_test[3], MODAL_CONFIG[3]))
    gate_diagnostics = {
        "modal_config": config_name(MODAL_CONFIG),
        "train_uncertainty_threshold": gate_train_threshold,
        "test_uncertainty_threshold": gate_test_threshold,
        "train_pass_probability": float(np.mean(train_probability[valid] >= MODAL_CONFIG[0])),
        "test_pass_probability": float(np.mean(test_probability >= MODAL_CONFIG[0])),
        "train_pass_lcb": float(np.mean(modal_train[2][valid] > MODAL_CONFIG[2])),
        "test_pass_lcb": float(np.mean(modal_test[2] > MODAL_CONFIG[2])),
        "train_pass_uncertainty": float(np.mean(modal_train[3][valid] <= gate_train_threshold)),
        "test_pass_uncertainty": float(np.mean(modal_test[3] <= gate_test_threshold)),
        "train_modal_active_fraction": float(np.mean(train_modal_active[valid])),
        "test_modal_active_fraction": float(np.mean(test_modal_active)),
        "test_v150_active_fraction_after_los": float(np.mean(test_active_v150)),
        "train_modal_gain": finite_summary(train_modal_gain[valid]),
        "train_fold_config_gain": finite_summary(train_gain[valid]),
    }

    meta_train = np.column_stack(
        [
            train_probability[valid],
            train_max_benefit[valid],
            train_lower_bound[valid],
            train_selected_uncertainty[valid],
            train_distance[valid],
            train_strength_fold_config[valid],
        ]
    ).astype(np.float32)
    meta_test = np.column_stack(
        [
            test_probability,
            test_max_benefit,
            test_lower_bound,
            test_selected_uncertainty,
            test_distance,
            test_strength_v150,
        ]
    ).astype(np.float32)
    meta_names = [
        "probability",
        "max_benefit",
        "selected_lower_bound",
        "selected_uncertainty",
        "neighbor_distance",
        "selected_strength",
    ]
    meta_shift = top_univariate_shift(meta_names, meta_train, meta_test, len(meta_names))
    meta_domain_auc = domain_auc(meta_train, meta_test)

    physical_basic_train = np.column_stack(
        [
            positions_train[:, 0],
            positions_train[:, 1],
            positions_train[:, 2],
            np.linalg.norm(positions_train - BS_POS[None, :], axis=1),
            blocked_train,
        ]
    ).astype(np.float32)
    physical_basic_test = np.column_stack(
        [
            positions_test[:, 0],
            positions_test[:, 1],
            positions_test[:, 2],
            np.linalg.norm(positions_test - BS_POS[None, :], axis=1),
            blocked_test,
        ]
    ).astype(np.float32)
    geodesic_names = [f"geodesic:{name}" for name in geodesic["feature_names"]]
    pointcloud_names = [
        f"pointcloud:{index}"
        for index in range(pointcloud["train_features"].shape[1])
    ]
    physical_names = [
        "x",
        "y",
        "z",
        "bs_distance",
        "blocked",
        *geodesic_names,
        *pointcloud_names,
    ]
    physical_train = np.concatenate(
        [
            physical_basic_train[valid],
            geodesic["train_features"].astype(np.float32)[valid],
            pointcloud["train_features"].astype(np.float32)[valid],
        ],
        axis=1,
    )
    physical_test = np.concatenate(
        [
            physical_basic_test,
            geodesic["test_features"].astype(np.float32),
            pointcloud["test_features"].astype(np.float32),
        ],
        axis=1,
    )
    physical_shift = top_univariate_shift(physical_names, physical_train, physical_test)
    physical_domain_auc = domain_auc(physical_train, physical_test)

    stable_train, stable_test = build_stable_feature_domain()
    stable_domain_auc = domain_auc(stable_train[valid], stable_test)

    bin_rows = []
    for metric_name, train_metric, test_metric in (
        ("probability", train_probability, test_probability),
        ("max_benefit", train_max_benefit, test_max_benefit),
        ("selected_lower_bound", train_lower_bound, test_lower_bound),
        (
            "negative_selected_uncertainty",
            -train_selected_uncertainty,
            -test_selected_uncertainty,
        ),
    ):
        bin_rows.extend(
            bin_table(
                metric_name,
                train_metric,
                test_metric,
                train_gain,
                train_active_fold_config,
                test_active_v150,
                valid,
            )
        )
    with BIN_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bin_rows)
    with TOP_SHIFT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(physical_shift[0].keys()))
        writer.writeheader()
        writer.writerows(physical_shift)

    report = {
        "method": "v150_failure_data_signal_analysis",
        "status": "analysis_only_no_candidate",
        "online_context": {
            "v140_best": 0.501352,
            "v150_online": v150_report.get("online_score", 0.50127),
            "v150_delta_to_v140": v150_report.get("online_delta_to_v140", -0.000082),
            "v149_status": v149_report["status"],
            "v150_status": v150_report["status"],
        },
        "main_findings": {
            "fold_oof_active_fraction": float(np.mean(train_active_fold_config[valid])),
            "test_v149_active_fraction": float(np.mean(test_active_v149)),
            "test_v150_active_fraction": float(np.mean(test_active_v150)),
            "train_fold_config_gain_mean": float(np.mean(train_gain[valid])),
            "train_fold_config_active_gain_mean": (
                float(np.mean(train_gain[valid & train_active_fold_config]))
                if np.any(valid & train_active_fold_config)
                else 0.0
            ),
            "train_fold_config_active_positive_fraction": (
                float(np.mean(train_gain[valid & train_active_fold_config] > 0.0))
                if np.any(valid & train_active_fold_config)
                else 0.0
            ),
            "meta_domain_auc": meta_domain_auc["auc"],
            "physical_domain_auc": physical_domain_auc["auc"],
            "stable_feature_domain_auc": stable_domain_auc["auc"],
        },
        "gate_diagnostics": gate_diagnostics,
        "regime_diagnostics": {
            "train_fold_config": by_regime(
                regimes_train, train_gain, train_active_fold_config, valid
            ),
            "train_modal_config": by_regime(
                regimes_train, train_modal_gain, train_modal_active, valid
            ),
            "test_v149": by_regime(regimes_test, None, test_active_v149),
            "test_v150": by_regime(regimes_test, None, test_active_v150),
        },
        "score_calibration_bins_csv": PATHS.relative(BIN_CSV),
        "top_physical_shift_csv": PATHS.relative(TOP_SHIFT_CSV),
        "meta_shift_top": meta_shift,
        "physical_shift_top": physical_shift[:20],
        "domain_auc": {
            "meta_decision_outputs": meta_domain_auc,
            "physical_geodesic_pointcloud": physical_domain_auc,
            "stable_423_features": stable_domain_auc,
        },
        "strength_counts": {
            "train_fold_config": strength_counts(train_strength_fold_config[valid]),
            "train_modal_config": strength_counts(train_modal_strength[valid]),
            "test_v149": strength_counts(test_strength_v149),
            "test_v150": strength_counts(test_strength_v150),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "score_bins_csv": PATHS.relative(BIN_CSV),
            "top_shift_csv": PATHS.relative(TOP_SHIFT_CSV),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
