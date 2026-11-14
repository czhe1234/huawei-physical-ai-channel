#!/usr/bin/env python3
"""Test spatial-OOF learnability of post-v140 marginal residual structure."""

from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from analyze_post_v143_hidden_information import (
    marginal_oracle,
    normalize_without_floor,
    summarize,
    support_score,
)
from analyze_v143_matched_residual_probability import basic_features
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, support_features
from generate_v143_transductive_geodesic_risk import apply_risk_shift
from powerfield_common import TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(143)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v143_marginal_residual_learnability_analysis.json"
RANDOM_STATE = 14403
STRENGTHS = (0.10, 0.25, 0.50, 1.00)


def marginal(x: np.ndarray, keep_axes: tuple[int, ...]) -> np.ndarray:
    sum_axes = tuple(axis for axis in (1, 2, 3) if axis not in keep_axes)
    return x.sum(axis=sum_axes)


def apply_log_ratio(
    base: np.ndarray,
    prediction: np.ndarray,
    keep_axes: tuple[int, ...],
    strength: float,
) -> np.ndarray:
    shape = [len(base), 1, 1, 1]
    for output_axis, source_axis in enumerate(keep_axes, start=1):
        shape[source_axis] = prediction.shape[output_axis]
    ratio = np.exp(np.clip(prediction, -3.0, 3.0) * float(strength)).reshape(shape)
    output = (base * ratio).astype(np.float32)
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output


def classification_oof(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    selected = np.flatnonzero(valid)
    prediction = np.full(len(labels), -1, dtype=np.int32)
    splitter = GroupKFold(n_splits=5)
    for train_local, val_local in splitter.split(
        x[selected], labels[selected], groups[selected]
    ):
        train_index = selected[train_local]
        val_index = selected[val_local]
        model = ExtraTreesClassifier(
            n_estimators=360,
            max_features="sqrt",
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train_index], labels[train_index])
        prediction[val_index] = model.predict(x[val_index])
    truth = labels[selected]
    pred = prediction[selected]
    classes = np.unique(labels[valid])
    return {
        "accuracy": float(np.mean(pred == truth)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "classes": [int(value) for value in classes],
        "counts": {str(int(value)): int(np.sum(truth == value)) for value in classes},
        "confusion": confusion_matrix(truth, pred, labels=classes).tolist(),
    }


def main() -> None:
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = height["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    scene = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    pointcloud_names = [str(value) for value in scene["feature_names"]]
    residual_prob = np.load(
        PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
    ).astype(np.float32)

    source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"), 1e-12
    )
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    truth_power = truth_raw.sum(axis=(1, 2, 3))
    valid = truth_power > 0
    truth = normalize_without_floor(truth_raw)
    risk = build_risk(
        regimes, residual_prob, geodesic["train_features"], geodesic_names
    )
    current, _ = apply_risk_shift(source, risk)
    current_score = support_score(current, truth)

    basic, basic_names = basic_features(pos_train, blocked)
    features = np.concatenate(
        [
            basic,
            geodesic["train_features"],
            scene["train_features"],
            residual_prob,
            support_features(current),
        ],
        axis=1,
    )
    features = np.nan_to_num(
        features, nan=0.0, posinf=8.0, neginf=-8.0
    ).astype(np.float32)
    feature_names = (
        basic_names
        + pointcloud_names[:0]
        + geodesic_names
        + pointcloud_names
        + [f"residual_prob_{index:02d}" for index in range(96)]
        + [
            f"support_signature_{index:03d}"
            for index in range(features.shape[1] - len(basic_names) - len(geodesic_names) - len(pointcloud_names) - 96)
        ]
    )
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(pos_train[:, :2])
    selected = np.flatnonzero(valid)
    splitter = GroupKFold(n_splits=5)

    axes = {
        "H": (1,),
        "V": (2,),
        "D": (3,),
        "HV": (1, 2),
    }
    learnability: dict[str, object] = {}
    oracle_gains: dict[str, np.ndarray] = {}
    for name, keep_axes in axes.items():
        base_marginal = marginal(current, keep_axes)
        truth_marginal = marginal(truth, keep_axes)
        log_ratio = np.log(np.maximum(truth_marginal, 1e-10)) - np.log(
            np.maximum(base_marginal, 1e-10)
        )
        log_ratio = np.clip(log_ratio, -3.0, 3.0).astype(np.float32)
        prediction = np.zeros_like(log_ratio, dtype=np.float32)
        fold_reports = []
        for fold, (train_local, val_local) in enumerate(
            splitter.split(features[selected], groups=groups[selected])
        ):
            train_index = selected[train_local]
            val_index = selected[val_local]
            model = ExtraTreesRegressor(
                n_estimators=280,
                max_features="sqrt",
                min_samples_leaf=5,
                random_state=RANDOM_STATE + fold,
                n_jobs=-1,
            )
            model.fit(
                features[train_index], log_ratio[train_index].reshape(len(train_index), -1)
            )
            prediction[val_index] = model.predict(features[val_index]).reshape(
                (len(val_index),) + log_ratio.shape[1:]
            )
            fold_reports.append(
                {
                    "fold": fold,
                    "train_count": int(len(train_index)),
                    "validation_count": int(len(val_index)),
                }
            )

        oracle_target = marginal_oracle(current, truth, keep_axes)
        oracle_gain = support_score(oracle_target, truth) - current_score
        oracle_gains[name] = oracle_gain
        strength_reports = {}
        for strength in STRENGTHS:
            adjusted = apply_log_ratio(current, prediction, keep_axes, strength)
            score = support_score(adjusted, truth)
            gain = score - current_score
            strength_reports[f"{strength:.2f}"] = {
                "gain": summarize(gain[valid]),
                "positive_fraction": float(np.mean(gain[valid] > 0)),
                "top_bad_20_gain": summarize(
                    gain[
                        selected[
                            np.argsort(current_score[selected])[
                                : max(1, int(0.20 * len(selected)))
                            ]
                        ]
                    ]
                ),
            }
        learnability[name] = {
            "target_shape": list(log_ratio.shape[1:]),
            "oracle_gain": summarize(oracle_gain[valid]),
            "prediction_log_ratio": summarize(prediction[valid]),
            "strengths": strength_reports,
            "folds": fold_reports,
        }

    best_axis = np.argmax(
        np.stack([oracle_gains["H"], oracle_gains["V"], oracle_gains["D"]], axis=1),
        axis=1,
    ).astype(np.int32)
    axis_classification = classification_oof(features, best_axis, groups, valid)

    powers = np.asarray([0.70, 0.82, 0.92, 1.00, 1.08, 1.18, 1.32], dtype=np.float32)
    temperature_scores = []
    for power in powers:
        target = np.maximum(current, 1e-12) ** power
        target /= np.maximum(target.sum(axis=(1, 2, 3), keepdims=True), TINY)
        temperature_scores.append(support_score(target, truth))
    best_power_index = np.argmax(np.stack(temperature_scores, axis=1), axis=1)
    temperature_class = np.where(
        powers[best_power_index] < 0.96,
        0,
        np.where(powers[best_power_index] > 1.04, 2, 1),
    ).astype(np.int32)
    temperature_classification = classification_oof(
        features, temperature_class, groups, valid
    )

    report = {
        "method": "v143_marginal_residual_learnability_analysis",
        "hypothesis": (
            "the post-v140 residual is marginal shape/allocation rather than an "
            "integer shift; test whether H/V/D/HV log-ratios are spatially learnable"
        ),
        "valid_points": int(valid.sum()),
        "feature_dim": int(features.shape[1]),
        "feature_names_count": len(feature_names),
        "current_support_score": summarize(current_score[valid]),
        "marginal_learnability": learnability,
        "best_axis_classification": {
            "class_map": {"0": "H", "1": "V", "2": "D"},
            **axis_classification,
        },
        "temperature_classification": {
            "class_map": {"0": "smooth", "1": "neutral", "2": "sharpen"},
            **temperature_classification,
        },
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {REPORT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
