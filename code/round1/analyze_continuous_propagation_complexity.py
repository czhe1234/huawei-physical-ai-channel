#!/usr/bin/env python3
"""Learnability audit for continuous propagation-complexity labels."""

from __future__ import annotations

import csv
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from generate_v87_easy_field_projection import summarize
from powerfield_common import BS_POS, regime_of
from project_paths import PATHS


OUT_DIR = PATHS.analysis / "results" / "official_data_multidim"
REPORT = OUT_DIR / "continuous_complexity_learnability_analysis.json"
PREDICTABILITY_CSV = OUT_DIR / "continuous_complexity_predictability.csv"
IMPORTANCE_CSV = OUT_DIR / "continuous_complexity_feature_importance.csv"
CLUSTER_CSV = OUT_DIR / "continuous_complexity_train_clusters.csv"
TEST_PRED_CSV = OUT_DIR / "continuous_complexity_test_predictions_by_regime.csv"

REGIMES = ("clear", "light", "medium", "heavy")
TINY = 1e-12
RANDOM_STATE = 15151


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return summarize(values)


def entropy01(probability: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    p = np.maximum(probability, TINY)
    raw = -np.sum(p * np.log(p), axis=axis)
    if isinstance(axis, tuple):
        size = np.prod([probability.shape[item] for item in axis])
    else:
        size = probability.shape[axis]
    return (raw / np.log(size)).astype(np.float32)


def make_geometry_features(
    positions: np.ndarray,
    blocked: np.ndarray,
    geodesic_features: np.ndarray,
    pointcloud_features: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    vector = positions - BS_POS[None, :]
    distance = np.linalg.norm(vector, axis=1)
    azimuth = np.arctan2(vector[:, 1], vector[:, 0])
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
        ]
    ).astype(np.float32)
    names = [
        "x",
        "y",
        "z",
        "bs_distance",
        "log1p_bs_distance",
        "sin_azimuth",
        "cos_azimuth",
        "blocked",
        "blocked_over_distance",
    ]
    features = np.concatenate(
        [
            basic,
            geodesic_features.astype(np.float32),
            pointcloud_features.astype(np.float32),
        ],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0), names


def channel_labels(
    support_raw: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    total_power_all = support_raw.sum(axis=(1, 2, 3)).astype(np.float32)
    support = support_raw[valid].astype(np.float32)
    support /= np.maximum(support.sum(axis=(1, 2, 3), keepdims=True), TINY)
    h = support.sum(axis=(2, 3))
    v = support.sum(axis=(1, 3))
    delay = support.sum(axis=(1, 2))
    flat = support.reshape(len(support), -1)
    sorted_flat = np.sort(flat, axis=1)[:, ::-1]
    h_entropy = entropy01(h, axis=1)
    v_entropy = entropy01(v, axis=1)
    delay_entropy = entropy01(delay, axis=1)
    joint_entropy = entropy01(flat, axis=1)
    top1 = sorted_flat[:, 0].astype(np.float32)
    top8 = sorted_flat[:, :8].sum(axis=1).astype(np.float32)
    delay_bins = np.arange(delay.shape[1], dtype=np.float32)
    delay_mean = (delay * delay_bins[None, :]).sum(axis=1)
    delay_rms = np.sqrt(
        np.maximum((delay * (delay_bins[None, :] - delay_mean[:, None]) ** 2).sum(axis=1), 0.0)
    ).astype(np.float32)
    h_angles = np.exp(2j * np.pi * np.arange(h.shape[1]) / h.shape[1])
    h_resultant = np.abs((h * h_angles[None, :]).sum(axis=1))
    h_spread = (1.0 - h_resultant).astype(np.float32)

    tree = cKDTree(positions[valid, :2])
    nn_dist, nn_index = tree.query(positions[valid, :2], k=9)
    h_norm = np.maximum(np.linalg.norm(h, axis=1), TINY)
    d_norm = np.maximum(np.linalg.norm(delay, axis=1), TINY)
    h_coherence = np.zeros(len(h), dtype=np.float32)
    delay_coherence = np.zeros(len(h), dtype=np.float32)
    local_log_power_std = np.zeros(len(h), dtype=np.float32)
    log_power = np.log10(np.maximum(total_power_all[valid], TINY)).astype(np.float32)
    for i in range(len(h)):
        neighbors = nn_index[i, 1:]
        within = nn_dist[i, 1:] <= 20.0
        if not np.any(within):
            neighbors = nn_index[i, 1:4]
        else:
            neighbors = neighbors[within]
        h_cos = (h[neighbors] @ h[i]) / np.maximum(h_norm[neighbors] * h_norm[i], TINY)
        d_cos = (delay[neighbors] @ delay[i]) / np.maximum(d_norm[neighbors] * d_norm[i], TINY)
        h_coherence[i] = float(np.mean(h_cos))
        delay_coherence[i] = float(np.mean(d_cos))
        local_log_power_std[i] = float(np.std(log_power[neighbors]))

    labels = {
        "log10_total_power": log_power,
        "h_entropy": h_entropy,
        "h_spread": h_spread,
        "v_entropy": v_entropy,
        "delay_entropy": delay_entropy,
        "delay_rms_spread": delay_rms,
        "joint_entropy": joint_entropy,
        "joint_top1": top1,
        "joint_top8": top8,
        "local_h_coherence_20m": h_coherence,
        "local_delay_coherence_20m": delay_coherence,
        "local_log_power_std_20m": local_log_power_std,
    }
    parts = {"h": h, "v": v, "delay": delay}
    return labels, parts


def regime_baseline_r2(values: np.ndarray, regimes: np.ndarray) -> float:
    prediction = np.zeros_like(values, dtype=np.float32)
    for regime in REGIMES:
        mask = regimes == regime
        if np.any(mask):
            prediction[mask] = float(np.mean(values[mask]))
    return float(r2_score(values, prediction))


def blocked_correlation(values: np.ndarray, blocked: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(blocked, values).statistic),
        "spearman": float(spearmanr(blocked, values).statistic),
    }


def oof_predict_targets(
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    positions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], dict[str, list[dict[str, object]]]]:
    groups = KMeans(n_clusters=25, random_state=RANDOM_STATE, n_init=20).fit_predict(
        positions[:, :2]
    )
    splitter = GroupKFold(n_splits=5)
    predictions = {name: np.zeros(len(features), dtype=np.float32) for name in labels}
    metrics: dict[str, dict[str, float]] = {}
    importances: dict[str, list[dict[str, object]]] = {}
    for name, values in labels.items():
        fold_predictions = np.zeros(len(features), dtype=np.float32)
        fold_id = 0
        for train_index, validation_index in splitter.split(features, groups=groups):
            model = ExtraTreesRegressor(
                n_estimators=360,
                max_features=0.65,
                min_samples_leaf=10,
                random_state=RANDOM_STATE + fold_id,
                n_jobs=-1,
            )
            model.fit(features[train_index], values[train_index])
            fold_predictions[validation_index] = model.predict(features[validation_index]).astype(
                np.float32
            )
            fold_id += 1
        predictions[name] = fold_predictions
        high_label = values >= np.quantile(values, 0.70)
        metrics[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "oof_r2": float(r2_score(values, fold_predictions)),
            "oof_pearson": float(pearsonr(values, fold_predictions).statistic),
            "oof_spearman": float(spearmanr(values, fold_predictions).statistic),
            "top30_auc": float(roc_auc_score(high_label.astype(np.int8), fold_predictions)),
        }
        full_model = ExtraTreesRegressor(
            n_estimators=420,
            max_features=0.65,
            min_samples_leaf=8,
            random_state=RANDOM_STATE + 100,
            n_jobs=-1,
        )
        full_model.fit(features, values)
        order = np.argsort(full_model.feature_importances_)[::-1][:15]
        importances[name] = [
            {
                "feature_index": int(index),
                "importance": float(full_model.feature_importances_[index]),
            }
            for index in order
        ]
    return predictions, metrics, importances


def train_test_predictions(
    features_train: np.ndarray,
    features_test: np.ndarray,
    labels: dict[str, np.ndarray],
    regimes_test: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    output = {}
    rows = []
    for name, values in labels.items():
        model = ExtraTreesRegressor(
            n_estimators=480,
            max_features=0.65,
            min_samples_leaf=8,
            random_state=RANDOM_STATE + 200,
            n_jobs=-1,
        )
        model.fit(features_train, values)
        pred_test = model.predict(features_test).astype(np.float32)
        output[name] = {
            "train_truth": finite_summary(values),
            "test_predicted": finite_summary(pred_test),
        }
        for regime in REGIMES:
            mask = regimes_test == regime
            rows.append(
                {
                    "target": name,
                    "regime": regime,
                    "count": int(np.sum(mask)),
                    "test_pred_mean": float(np.mean(pred_test[mask])) if np.any(mask) else 0.0,
                    "test_pred_median": float(np.median(pred_test[mask])) if np.any(mask) else 0.0,
                }
            )
    return output, rows


def cluster_channel_types(
    labels: dict[str, np.ndarray],
    regimes: np.ndarray,
) -> list[dict[str, object]]:
    cluster_targets = [
        "log10_total_power",
        "h_spread",
        "delay_rms_spread",
        "joint_top1",
        "local_h_coherence_20m",
    ]
    matrix = np.column_stack([labels[name] for name in cluster_targets]).astype(np.float32)
    matrix = StandardScaler().fit_transform(matrix)
    clusters = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=50).fit_predict(matrix)
    rows = []
    for cluster in sorted(np.unique(clusters)):
        mask = clusters == cluster
        row: dict[str, object] = {"cluster": int(cluster), "count": int(np.sum(mask))}
        for target in cluster_targets:
            row[f"{target}_mean"] = float(np.mean(labels[target][mask]))
            row[f"{target}_median"] = float(np.median(labels[target][mask]))
        for regime in REGIMES:
            row[f"regime_{regime}_fraction"] = float(np.mean(regimes[mask] == regime))
        rows.append(row)
    rows.sort(key=lambda row: row["h_spread_mean"], reverse=True)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    positions_train_all = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    support_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train_all = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_train_all = regime_of(blocked_train_all)
    regimes_test = regime_of(blocked_test)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    total_power = support_raw.sum(axis=(1, 2, 3))
    valid = total_power > 0
    labels, _ = channel_labels(support_raw, positions_train_all, valid)
    features_train_all, base_feature_names = make_geometry_features(
        positions_train_all,
        blocked_train_all,
        geodesic["train_features"].astype(np.float32),
        pointcloud["train_features"].astype(np.float32),
    )
    features_test, _ = make_geometry_features(
        positions_test,
        blocked_test,
        geodesic["test_features"].astype(np.float32),
        pointcloud["test_features"].astype(np.float32),
    )
    geodesic_names = [f"geodesic:{value}" for value in geodesic["feature_names"]]
    pointcloud_names = [
        f"pointcloud:{index}" for index in range(pointcloud["train_features"].shape[1])
    ]
    feature_names = [*base_feature_names, *geodesic_names, *pointcloud_names]
    features_train = features_train_all[valid]
    positions_train = positions_train_all[valid]
    blocked_train = blocked_train_all[valid]
    regimes_train = regimes_train_all[valid]
    oof_predictions, predictability, importances = oof_predict_targets(
        features_train, labels, positions_train
    )
    for target, rows in importances.items():
        for row in rows:
            row["feature"] = feature_names[row["feature_index"]]
    prediction_rows = []
    for target, stats in predictability.items():
        row = {
            "target": target,
            **stats,
            "regime_baseline_r2": regime_baseline_r2(labels[target], regimes_train),
            "blocked_pearson": blocked_correlation(labels[target], blocked_train)["pearson"],
            "blocked_spearman": blocked_correlation(labels[target], blocked_train)["spearman"],
        }
        prediction_rows.append(row)
    prediction_rows.sort(key=lambda row: row["oof_spearman"], reverse=True)
    with PREDICTABILITY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)
    importance_rows = []
    for target, rows in importances.items():
        for rank, row in enumerate(rows, start=1):
            importance_rows.append({"target": target, "rank": rank, **row})
    with IMPORTANCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(importance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(importance_rows)
    clusters = cluster_channel_types(labels, regimes_train)
    with CLUSTER_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clusters[0].keys()))
        writer.writeheader()
        writer.writerows(clusters)
    test_predictions, test_rows = train_test_predictions(
        features_train, features_test, labels, regimes_test
    )
    with TEST_PRED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)

    report = {
        "method": "continuous_propagation_complexity_learnability",
        "status": "analysis_only_no_candidate",
        "goal": (
            "replace the crude clear/light/medium/heavy blocked split with "
            "learnable continuous propagation-complexity variables"
        ),
        "valid_points": int(np.sum(valid)),
        "targets": {
            target: {
                "truth": finite_summary(values),
                "predictability": predictability[target],
                "regime_baseline_r2": regime_baseline_r2(values, regimes_train),
                "blocked_correlation": blocked_correlation(values, blocked_train),
                "top_features": importances[target][:8],
            }
            for target, values in labels.items()
        },
        "ranked_predictability": prediction_rows,
        "test_predicted_complexity": test_predictions,
        "channel_type_clusters": clusters,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "predictability_csv": PATHS.relative(PREDICTABILITY_CSV),
            "feature_importance_csv": PATHS.relative(IMPORTANCE_CSV),
            "cluster_csv": PATHS.relative(CLUSTER_CSV),
            "test_prediction_csv": PATHS.relative(TEST_PRED_CSV),
        },
        "interpretation_template": {
            "blocked_four_regimes_are_scientifically_insufficient": True,
            "preferred_form": (
                "continuous variables: visibility/power loss, H angular spread, "
                "delay spread, sparsity/top-k, and local coherence"
            ),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
