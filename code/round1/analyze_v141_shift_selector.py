#!/usr/bin/env python3
"""Test whether continuous shift gain is predictable under spatial OOF."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold

from generate_v140_balanced_hminus2_overdrive import (
    build_risk,
    point_features,
    support_features,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(141)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v141_shift_selector_analysis.json"
PREDICTIONS = PATHS.cache_dir(141) / "v141_shift_gain_predictions.npy"
PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1).astype(np.float64)
    bb = b.reshape(b.shape[0], -1).astype(np.float64)
    return (
        np.sum(aa * bb, axis=1)
        / (np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1)) + 1e-30)
    ).astype(np.float32)


def shifted_log_blend(
    support: np.ndarray, h1_weight: float, h2_weight: float, batch: int = 100
) -> np.ndarray:
    out = np.empty_like(support, dtype=np.float32)
    floor = 1e-12
    for start in range(0, len(support), batch):
        end = min(start + batch, len(support))
        current = np.asarray(support[start:end], dtype=np.float32)
        log_current = np.log(np.maximum(current, floor))
        shifted1 = np.roll(current, -1, axis=1)
        mixed = np.exp(
            (1.0 - h1_weight) * log_current
            + h1_weight * np.log(np.maximum(shifted1, floor))
        ).astype(np.float32)
        if h2_weight > 0:
            shifted2 = np.roll(current, -2, axis=1)
            mixed = np.exp(
                (1.0 - h2_weight) * np.log(np.maximum(mixed, floor))
                + h2_weight * np.log(np.maximum(shifted2, floor))
            ).astype(np.float32)
        mixed /= np.maximum(mixed.sum(axis=(1, 2, 3), keepdims=True), TINY)
        out[start:end] = mixed
    return out


def make_features(
    support: np.ndarray,
    pos: np.ndarray,
    blocked: np.ndarray,
    scene: np.ndarray,
    geo: np.ndarray,
    template_probs: np.ndarray,
) -> np.ndarray:
    features = np.concatenate(
        [
            point_features(pos, blocked),
            scene.astype(np.float32),
            geo.astype(np.float32),
            template_probs.astype(np.float32),
            support_features(support),
        ],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=600,
        max_features=0.65,
        min_samples_leaf=4,
        random_state=14107,
        n_jobs=-1,
    )


def selection_curve(pred: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    order = np.argsort(pred)[::-1]
    result: dict[str, object] = {}
    for fraction in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        count = max(1, int(round(len(order) * fraction)))
        selected = order[:count]
        result[f"{fraction:.2f}"] = {
            "count": int(count),
            "pred_mean": float(np.mean(pred[selected])),
            "actual_mean": float(np.mean(truth[selected])),
            "actual_positive_fraction": float(np.mean(truth[selected] > 0)),
            "population_delta": float(np.sum(truth[selected]) / len(truth)),
        }
    return result


def cross_validated(
    features: np.ndarray,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, object]]:
    prediction = np.empty(len(target), dtype=np.float32)
    for train, valid in splits:
        estimator = model()
        estimator.fit(features[train], target[train])
        prediction[valid] = estimator.predict(features[valid]).astype(np.float32)
    pearson = float(np.corrcoef(prediction, target)[0, 1])
    spearman = float(spearmanr(prediction, target).statistic)
    return prediction, {
        "pearson": pearson,
        "spearman": spearman,
        "prediction_mean": float(np.mean(prediction)),
        "target_mean": float(np.mean(target)),
        "selection": selection_curve(prediction, target),
    }


def main() -> None:
    truth = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(np.float32)
    truth /= np.maximum(truth.sum(axis=(1, 2, 3), keepdims=True), TINY)
    train_support = np.load(
        PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"
    ).astype(np.float32)
    train_support /= np.maximum(
        train_support.sum(axis=(1, 2, 3), keepdims=True), TINY
    )
    test_support = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_target.npy"
    ).astype(np.float32)
    test_support /= np.maximum(test_support.sum(axis=(1, 2, 3), keepdims=True), TINY)

    base_cos = row_cos(train_support, truth)
    h1_support = shifted_log_blend(train_support, 0.165, 0.0)
    h12_support = shifted_log_blend(train_support, 0.165, 0.060)
    h1_gain = row_cos(h1_support, truth) - base_cos
    h12_gain = row_cos(h12_support, truth) - base_cos
    h2_increment = h12_gain - h1_gain

    scene = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    geo = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    heights = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_blocked = heights["train_blocked"].astype(np.float32)
    test_blocked = heights["test_blocked"].astype(np.float32)
    train_probs = np.load(
        PATHS.shared_features / "train_v125eq_crossfit_template_probs.npy"
    ).astype(np.float32)
    test_probs = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_probs.npy"
    ).astype(np.float32)

    train_features = make_features(
        train_support,
        train_pos,
        train_blocked,
        scene["train_features"],
        geo["train_features"],
        train_probs,
    )
    test_features = make_features(
        test_support,
        test_pos,
        test_blocked,
        scene["test_features"],
        geo["test_features"],
        test_probs,
    )

    spatial_groups = KMeans(n_clusters=25, random_state=14107, n_init=20).fit_predict(
        train_pos[:, :2]
    )
    group_splits = list(GroupKFold(n_splits=5).split(train_features, groups=spatial_groups))
    random_splits = list(
        KFold(n_splits=5, shuffle=True, random_state=14107).split(train_features)
    )

    targets = {
        "h1_gain": h1_gain,
        "h12_gain": h12_gain,
        "h2_increment": h2_increment,
    }
    diagnostics: dict[str, object] = {}
    test_predictions = []
    for name, target in targets.items():
        spatial_prediction, spatial_report = cross_validated(
            train_features, target, group_splits
        )
        random_prediction, random_report = cross_validated(
            train_features, target, random_splits
        )
        estimator = model()
        estimator.fit(train_features, target)
        test_prediction = estimator.predict(test_features).astype(np.float32)
        test_predictions.append(test_prediction)
        diagnostics[name] = {
            "target": {
                "mean": float(np.mean(target)),
                "positive_fraction": float(np.mean(target > 0)),
                "p90": float(np.quantile(target, 0.90)),
                "max": float(np.max(target)),
            },
            "spatial_oof": spatial_report,
            "random_oof": random_report,
            "test_prediction": {
                "mean": float(np.mean(test_prediction)),
                "p10": float(np.quantile(test_prediction, 0.10)),
                "median": float(np.median(test_prediction)),
                "p90": float(np.quantile(test_prediction, 0.90)),
                "max": float(np.max(test_prediction)),
            },
            "spatial_oof_prediction": spatial_prediction.tolist(),
            "random_oof_prediction": random_prediction.tolist(),
        }

    names = [str(value) for value in geo["feature_names"]]
    train_risk = build_risk(
        regime_of(train_blocked), train_probs, geo["train_features"], names
    )
    test_risk = build_risk(
        regime_of(test_blocked), test_probs, geo["test_features"], names
    )
    prediction_matrix = np.column_stack([*test_predictions, test_risk]).astype(np.float32)
    np.save(PREDICTIONS, prediction_matrix)

    report = {
        "hypothesis": (
            "regress continuous cosine gain from small H log-shifts instead of "
            "classifying sparse full-bin oracle labels"
        ),
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "train_risk": {
            "mean": float(np.mean(train_risk)),
            "p90": float(np.quantile(train_risk, 0.90)),
        },
        "test_risk": {
            "mean": float(np.mean(test_risk)),
            "p90": float(np.quantile(test_risk, 0.90)),
        },
        "diagnostics": diagnostics,
        "prediction_columns": ["h1_gain", "h12_gain", "h2_increment", "risk"],
        "predictions": PATHS.relative(PREDICTIONS),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {REPORT}")
    for name, values in diagnostics.items():
        print(
            name,
            "spatial_pearson=",
            values["spatial_oof"]["pearson"],
            "top10_actual=",
            values["spatial_oof"]["selection"]["0.10"]["actual_mean"],
            "test_p90=",
            values["test_prediction"]["p90"],
        )


if __name__ == "__main__":
    main()
