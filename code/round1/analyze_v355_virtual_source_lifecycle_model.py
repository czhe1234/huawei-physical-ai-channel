#!/usr/bin/env python3
"""Strict source lifecycle classifier/regressor for virtual-source transport."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v354_nested_virtual_source_transport as v354
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 355
VERSION = "v355_virtual_source_lifecycle_model"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v355_virtual_source_lifecycle_oof.npz"

NEGATIVE_RATIO = 8
CLASSIFIER_ITERATIONS = 140
REGRESSOR_ITERATIONS = 120
MIN_MEAN_DELTA = 0.01
MIN_FOLD_WINS = 4
LOCAL_K = (8, 16, 32)


def local_boundary_features(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    bank_target: np.ndarray,
    train_xy: np.ndarray,
) -> np.ndarray:
    max_k = min(max(LOCAL_K), len(bank_ids))
    distance, local = cKDTree(train_xy[bank_ids]).query(
        train_xy[query_ids], k=max_k
    )
    if max_k == 1:
        distance = distance[:, None]
        local = local[:, None]
    neighbor_target = bank_target[local]
    channels: list[np.ndarray] = []
    for requested in LOCAL_K:
        take = min(requested, max_k)
        value = neighbor_target[:, :take]
        dist = distance[:, :take]
        inverse = 1.0 / np.maximum(dist, 0.25) ** 2
        inverse /= inverse.sum(axis=1, keepdims=True)
        channels.extend(
            [
                np.mean(value > 0.0, axis=1),
                np.mean(value, axis=1),
                np.max(value, axis=1),
                np.sum(inverse[:, :, None] * value, axis=1),
            ]
        )

    query_xy = train_xy[query_ids]
    source_count = bank_target.shape[1]
    positive_distance = np.full((len(query_ids), source_count, 3), 1000.0, dtype=np.float32)
    negative_distance = np.full_like(positive_distance, 1000.0)
    for source in range(source_count):
        for present, output in (
            (bank_target[:, source] > 0.0, positive_distance),
            (bank_target[:, source] <= 0.0, negative_distance),
        ):
            count = int(np.sum(present))
            if count == 0:
                continue
            take = min(3, count)
            value = cKDTree(train_xy[bank_ids[present]]).query(query_xy, k=take)[0]
            if take == 1:
                value = value[:, None]
            output[:, source, :take] = value
            if take < 3:
                output[:, source, take:] = value[:, -1, None]
    channels.extend(
        [
            positive_distance[:, :, 0] / 300.0,
            positive_distance[:, :, 2] / 300.0,
            negative_distance[:, :, 0] / 300.0,
            negative_distance[:, :, 2] / 300.0,
            positive_distance[:, :, 0]
            / np.maximum(
                positive_distance[:, :, 0] + negative_distance[:, :, 0], 1e-6
            ),
            (negative_distance[:, :, 0] - positive_distance[:, :, 0]) / 300.0,
        ]
    )
    return np.stack(channels, axis=2).astype(np.float32)


def pair_features(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    bank_target: np.ndarray,
    centers: np.ndarray,
    train_pos: np.ndarray,
    scene_feature: np.ndarray,
    signature: np.ndarray,
    scene: dict[str, np.ndarray],
) -> np.ndarray:
    support = v311.support_statistics(
        bank_ids,
        bank_target,
        train_pos,
        scene_feature,
        signature,
        centers,
    )
    physical, _geometry_dim = v311.pair_features(
        query_ids,
        centers,
        support,
        train_pos,
        scene_feature,
        signature,
        scene,
    )
    boundary = local_boundary_features(
        query_ids, bank_ids, bank_target, train_pos[:, :2]
    )
    return np.nan_to_num(
        np.concatenate([physical, boundary], axis=2),
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    ).astype(np.float32)


def crossfit_training_features(
    fit_ids: np.ndarray,
    fit_target: np.ndarray,
    centers: np.ndarray,
    train_pos: np.ndarray,
    scene_feature: np.ndarray,
    signature: np.ndarray,
    scene: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    row_lookup = np.full(len(train_pos), -1, dtype=np.int64)
    row_lookup[fit_ids] = np.arange(len(fit_ids), dtype=np.int64)
    subfolds = v315.disjoint_folds(fit_ids, train_pos[:, :2])
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for query_ids in subfolds:
        bank_ids = np.setdiff1d(fit_ids, query_ids)
        bank_rows = row_lookup[bank_ids]
        query_rows = row_lookup[query_ids]
        if np.any(bank_rows < 0) or np.any(query_rows < 0):
            raise AssertionError("crossfit lifecycle row lookup failed")
        feature_parts.append(
            pair_features(
                query_ids,
                bank_ids,
                fit_target[bank_rows],
                centers,
                train_pos,
                scene_feature,
                signature,
                scene,
            )
        )
        target_parts.append(fit_target[query_rows])
    return np.concatenate(feature_parts), np.concatenate(target_parts)


def fit_models(
    feature: np.ndarray, target: np.ndarray, seed: int
) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor, dict[str, object]]:
    x = feature.reshape(-1, feature.shape[2])
    y = target.reshape(-1)
    positive = np.flatnonzero(y > 0.0)
    negative = np.flatnonzero(y <= 0.0)
    rng = np.random.default_rng(seed)
    negative_count = min(len(negative), max(len(positive) * NEGATIVE_RATIO, 4096))
    selected_negative = rng.choice(negative, size=negative_count, replace=False)
    selected = np.concatenate([positive, selected_negative])
    rng.shuffle(selected)
    y_presence = (y[selected] > 0.0).astype(np.int32)
    positive_weight = negative_count / max(len(positive), 1)
    sample_weight = np.where(y_presence > 0, positive_weight, 1.0).astype(np.float32)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=CLASSIFIER_ITERATIONS,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=0.4,
        early_stopping=False,
        random_state=seed,
    )
    classifier.fit(x[selected], y_presence, sample_weight=sample_weight)

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=REGRESSOR_ITERATIONS,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.5,
        early_stopping=False,
        random_state=seed + 10_000,
    )
    regressor.fit(
        x[positive],
        np.log(np.maximum(y[positive], 1e-5)),
        sample_weight=np.sqrt(np.maximum(y[positive], 1e-5)),
    )
    return classifier, regressor, {
        "pair_count": int(len(y)),
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
        "sampled_negative_count": int(negative_count),
        "feature_dim": int(x.shape[1]),
    }


def predict_mass(
    classifier: HistGradientBoostingClassifier,
    regressor: HistGradientBoostingRegressor,
    feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = feature.shape[:2]
    x = feature.reshape(-1, feature.shape[2])
    probability = classifier.predict_proba(x)[:, 1]
    conditional_mass = np.exp(regressor.predict(x))
    mass = probability * conditional_mass
    return (
        np.clip(mass.reshape(shape), 0.0, 0.75).astype(np.float32),
        probability.reshape(shape).astype(np.float32),
        conditional_mass.reshape(shape).astype(np.float32),
    )


def main() -> None:
    for path in (
        v354.TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v354.TRAIN_CANDIDATES,
        v273.SCENE_CACHE,
        v273.FEATURE_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = np.load(v354.TARGET_CACHE)
    truth_sh = v354.normalize(target["sh"].astype(np.float32))
    truth_d = v354.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    observation, observation_report = v354.build_physical_observations()
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    scene_data = np.load(v273.SCENE_CACHE)
    scene = {key: scene_data[key] for key in scene_data.files}
    scene_feature = scene["train_features"].astype(np.float32)
    signature = np.load(v273.FEATURE_CACHE)["train_signature"].astype(np.float32)

    fold_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_anchor: list[np.ndarray] = []
    oof_prediction: list[np.ndarray] = []
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = v311.discover_sources(observation, fit_ids)
        fit_target, fit_assignment = v311.assign_targets(
            observation, fit_ids, centers, train_pos
        )
        outer_target, outer_assignment = v311.assign_targets(
            observation, outer_ids, centers, train_pos
        )
        train_feature, train_target = crossfit_training_features(
            fit_ids,
            fit_target,
            centers,
            train_pos,
            scene_feature,
            signature,
            scene,
        )
        classifier, regressor, training = fit_models(
            train_feature, train_target, VERSION_NUMBER + fold
        )
        outer_feature = pair_features(
            outer_ids,
            fit_ids,
            fit_target,
            centers,
            train_pos,
            scene_feature,
            signature,
            scene,
        )
        predicted_mass, probability, conditional_mass = predict_mass(
            classifier, regressor, outer_feature
        )
        outer_neighbor = v354.neighbors(
            outer_ids, fit_ids, train_pos[:, :2], False
        )
        if np.intersect1d(np.unique(outer_neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered v355 KNN current")
        anchor = v354.anchor_from_neighbors(truth_sh, outer_neighbor)
        prediction, diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            outer_neighbor,
            anchor,
            fit_target,
            centers,
            train_pos[:, :2],
            query_mass_override=predicted_mass,
        )
        base_metric = v354.metrics(
            anchor, truth_sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = v354.metrics(
            prediction, truth_sh[outer_ids], truth_d[outer_ids]
        )
        delta = {
            key: float(model_metric[key] - base_metric[key])
            for key in ("sh", "d", "weighted")
        }
        lifecycle_metrics = v311.evaluate_prediction(predicted_mass, outer_target)
        knn_mass = v311.knn_prediction(
            outer_ids, fit_ids, fit_target, train_pos
        )
        knn_metrics = v311.evaluate_prediction(knn_mass, outer_target)
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "source_discovery": source_report,
                "fit_assignment": fit_assignment,
                "outer_assignment": outer_assignment,
                "training": training,
                "lifecycle_metrics": lifecycle_metrics,
                "knn_mass_metrics": knn_metrics,
                "probability": summarize(probability),
                "conditional_mass": summarize(conditional_mass),
                "predicted_total_mass": summarize(predicted_mass.sum(axis=1)),
                "true_total_mass": summarize(outer_target.sum(axis=1)),
                "baseline": base_metric,
                "prediction": model_metric,
                "delta": delta,
                "transport": {
                    key: summarize(value) for key, value in diagnostics.items()
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_lifecycle_support": 0,
                    "outer_in_knn_bank": 0,
                },
            }
        )
        oof_ids.append(outer_ids)
        oof_anchor.append(anchor)
        oof_prediction.append(prediction)
        print(
            f"v355 fold={fold} sources={len(centers)} "
            f"ap={lifecycle_metrics['pair_ap']:.4f} "
            f"knn_ap={knn_metrics['pair_ap']:.4f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )

    mean_delta = {
        key: float(np.mean([row["delta"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    wins = int(sum(row["delta"]["weighted"] > 0.0 for row in fold_rows))
    lifecycle_ap_gain = float(
        np.mean(
            [
                row["lifecycle_metrics"]["pair_ap"]
                - row["knn_mass_metrics"]["pair_ap"]
                for row in fold_rows
            ]
        )
    )
    passed = bool(mean_delta["weighted"] >= MIN_MEAN_DELTA and wins >= MIN_FOLD_WINS)
    np.savez_compressed(
        OOF_OUT,
        ids=np.concatenate(oof_ids),
        anchor=np.concatenate(oof_anchor),
        prediction=np.concatenate(oof_prediction),
    )
    report = {
        "method": VERSION,
        "status": "strict_oof_breakthrough" if passed else "analysis_only_no_candidate",
        "round": 2,
        "primary_hypothesis": (
            "Virtual-source birth/death is a source-specific spatial boundary problem; "
            "cross-fit positive/negative support distances and point-cloud route features "
            "should predict expected source mass better than Euclidean KNN averaging."
        ),
        "compliance": {
            "uses": [
                "official outer-fit H virtual-source targets",
                "official train coordinates and Round1_Map point-cloud features",
                "cross-fit lifecycle features within every outer-fit bank",
            ],
            "does_not_use": [
                "outer H in source discovery, support features, KNN, or model fitting",
                "v207/v212/synthetic truth or teacher",
                "local/online labels or restored scene RT channels",
            ],
        },
        "parameters": {
            "model": "shared histogram gradient boosting presence + positive log-mass",
            "negative_ratio": NEGATIVE_RATIO,
            "classifier_iterations": CLASSIFIER_ITERATIONS,
            "regressor_iterations": REGRESSOR_ITERATIONS,
            "local_k": list(LOCAL_K),
            "transport": "frozen v354 subtract/transport/add",
            "parameter_search": "none",
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_delta": mean_delta,
            "fold_wins": wins,
            "lifecycle_pair_ap_gain_vs_knn": lifecycle_ap_gain,
            "passed": passed,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
