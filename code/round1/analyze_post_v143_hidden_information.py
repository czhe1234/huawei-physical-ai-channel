#!/usr/bin/env python3
"""Analyze hidden information after the near-tie v143 online result."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos, support_features
from generate_v143_transductive_geodesic_risk import (
    diffuse_risk,
    graph_average,
    official_graph_embedding,
    robust_standardize,
    apply_risk_shift,
)
from powerfield_common import TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(143)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v143_post_online_hidden_information_analysis.json"
TEST_CSV = REPORT_DIR / "v143_hidden_information_test_priority.csv"
RANDOM_STATE = 14371


def summarize(x: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(x, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return 0.0, 0.0
    return float(pearsonr(a[valid], b[valid]).statistic), float(
        spearmanr(a[valid], b[valid]).statistic
    )


def normalize_without_floor(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    return (
        values / np.maximum(values.sum(axis=(1, 2, 3), keepdims=True), TINY)
    ).astype(np.float32)


def support_score(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    joint = row_cos(pred, truth)
    beam = row_cos(pred.sum(axis=3), truth.sum(axis=3))
    delay = row_cos(pred.sum(axis=(1, 2)), truth.sum(axis=(1, 2)))
    return (0.50 * joint + 0.25 * beam + 0.25 * delay).astype(np.float32)


def marginal_oracle(
    base: np.ndarray, truth: np.ndarray, keep_axes: tuple[int, ...]
) -> np.ndarray:
    all_axes = (1, 2, 3)
    sum_axes = tuple(axis for axis in all_axes if axis not in keep_axes)
    base_marginal = base.sum(axis=sum_axes, keepdims=True)
    truth_marginal = truth.sum(axis=sum_axes, keepdims=True)
    ratio = truth_marginal / np.maximum(base_marginal, 1e-12)
    output = (base * ratio).astype(np.float32)
    output /= np.maximum(output.sum(axis=all_axes, keepdims=True), TINY)
    return output


def temperature_oracle(
    base: np.ndarray, truth: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    powers = np.array([0.70, 0.82, 0.92, 1.00, 1.08, 1.18, 1.32], dtype=np.float32)
    scores = []
    for power in powers:
        target = np.maximum(base, 1e-12) ** power
        target /= np.maximum(target.sum(axis=(1, 2, 3), keepdims=True), TINY)
        scores.append(support_score(target, truth))
    score_grid = np.stack(scores, axis=1)
    best = np.argmax(score_grid, axis=1)
    base_score = score_grid[:, int(np.where(powers == 1.0)[0][0])]
    gain = score_grid[np.arange(len(base)), best] - base_score
    counts = {
        f"{float(power):.2f}": int(np.sum((best == index) & valid))
        for index, power in enumerate(powers)
    }
    return powers[best], gain.astype(np.float32), counts


def quantile_groups(
    key: np.ndarray,
    values: dict[str, np.ndarray],
    valid: np.ndarray,
    bins: int = 5,
) -> list[dict[str, object]]:
    indices = np.flatnonzero(valid)
    order = indices[np.argsort(key[indices])]
    result = []
    for number, selected in enumerate(np.array_split(order, bins)):
        row: dict[str, object] = {
            "bin": number,
            "count": int(len(selected)),
            "key": summarize(key[selected]),
        }
        for name, value in values.items():
            row[name] = summarize(value[selected])
        result.append(row)
    return result


def category_groups(
    categories: np.ndarray,
    values: dict[str, np.ndarray],
    valid: np.ndarray,
) -> dict[str, object]:
    result = {}
    for category in np.unique(categories):
        selected = valid & (categories == category)
        result[str(category)] = {
            "count": int(selected.sum()),
            **{name: summarize(value[selected]) for name, value in values.items()},
        }
    return result


def regression_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    selected = np.flatnonzero(valid)
    prediction = np.full(len(y), np.nan, dtype=np.float32)
    splitter = GroupKFold(n_splits=5)
    for train_local, val_local in splitter.split(
        x[selected], y[selected], groups[selected]
    ):
        train_index = selected[train_local]
        val_index = selected[val_local]
        model = ExtraTreesRegressor(
            n_estimators=320,
            max_features="sqrt",
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train_index], y[train_index])
        prediction[val_index] = model.predict(x[val_index]).astype(np.float32)

    p = prediction[selected]
    t = y[selected]
    pearson, spearman = safe_corr(p, t)
    order = selected[np.argsort(prediction[selected])[::-1]]
    top10 = order[: max(1, int(math.ceil(0.10 * len(order))))]
    top20 = order[: max(1, int(math.ceil(0.20 * len(order))))]
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(mean_absolute_error(t, p)),
        "r2": float(r2_score(t, p)),
        "prediction": summarize(p),
        "target": summarize(t),
        "top10_actual": summarize(y[top10]),
        "top20_actual": summarize(y[top20]),
    }


def domain_analysis(
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    group_ranges: dict[str, tuple[int, int]],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate(
        [np.zeros(len(x_train), dtype=np.int32), np.ones(len(x_test), dtype=np.int32)]
    )
    prediction = np.zeros(len(x), dtype=np.float32)
    importance = np.zeros(x.shape[1], dtype=np.float64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_index, val_index in splitter.split(x, y):
        model = ExtraTreesClassifier(
            n_estimators=420,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train_index], y[train_index])
        prediction[val_index] = model.predict_proba(x[val_index])[:, 1]
        importance += model.feature_importances_
    importance /= splitter.get_n_splits()
    ranked = np.argsort(importance)[::-1][:20]
    grouped = {
        name: float(importance[start:end].sum())
        for name, (start, end) in group_ranges.items()
    }
    return (
        {
            "auc": float(roc_auc_score(y, prediction)),
            "train_propensity": summarize(prediction[: len(x_train)]),
            "test_propensity": summarize(prediction[len(x_train) :]),
            "group_importance": grouped,
            "top_features": [
                {
                    "name": feature_names[index],
                    "importance": float(importance[index]),
                }
                for index in ranked
            ],
        },
        prediction[: len(x_train)],
        prediction[len(x_train) :],
    )


def standardized_mean_difference(
    train: np.ndarray, test: np.ndarray, names: list[str]
) -> list[dict[str, float | str]]:
    mean_train = train.mean(axis=0)
    mean_test = test.mean(axis=0)
    variance = 0.5 * (train.var(axis=0) + test.var(axis=0))
    smd = (mean_test - mean_train) / np.sqrt(np.maximum(variance, 1e-12))
    order = np.argsort(np.abs(smd))[::-1][:25]
    return [
        {
            "name": names[index],
            "smd": float(smd[index]),
            "train_mean": float(mean_train[index]),
            "test_mean": float(mean_test[index]),
        }
        for index in order
    ]


def main() -> None:
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes_train = regime_of(height["train_blocked"].astype(np.float32))
    regimes_test = regime_of(height["test_blocked"].astype(np.float32))
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    pointcloud_names = [str(value) for value in pointcloud["feature_names"]]
    probs_train = np.load(
        PATHS.shared_features / "train_v125eq_crossfit_template_probs.npy"
    ).astype(np.float32)
    probs_test = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_probs.npy"
    ).astype(np.float32)

    source_train = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"), 1e-12
    )
    source_test = normalize_support(
        np.load(
            PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_target.npy"
        ),
        1e-12,
    )
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    truth_power = truth_raw.sum(axis=(1, 2, 3))
    valid = truth_power > 0
    truth = normalize_without_floor(truth_raw)

    raw_train = build_risk(
        regimes_train, probs_train, geodesic["train_features"], geodesic_names
    )
    raw_test = build_risk(
        regimes_test, probs_test, geodesic["test_features"], geodesic_names
    )
    graph_train_embedding = official_graph_embedding(
        pos_train,
        geodesic["train_features"],
        pointcloud["train_features"],
        probs_train,
    )
    graph_test_embedding = official_graph_embedding(
        pos_test,
        geodesic["test_features"],
        pointcloud["test_features"],
        probs_test,
    )
    graph_train_embedding, graph_test_embedding = robust_standardize(
        graph_train_embedding, graph_test_embedding
    )
    average_train, disagreement_train, _, _ = graph_average(
        graph_train_embedding, raw_train, 24
    )
    average_test, disagreement_test, _, _ = graph_average(
        graph_test_embedding, raw_test, 10
    )
    graph_train = diffuse_risk(raw_train, average_train, regimes_train)
    graph_test = diffuse_risk(raw_test, average_test, regimes_test)

    v140_train_target, _ = apply_risk_shift(source_train, raw_train)
    v143_train_target, _ = apply_risk_shift(source_train, graph_train)
    base_score = support_score(source_train, truth)
    v140_score = support_score(v140_train_target, truth)
    v143_score = support_score(v143_train_target, truth)
    raw_gain = v140_score - base_score
    graph_gain = v143_score - base_score
    graph_advantage = v143_score - v140_score
    current_badness = 1.0 - v140_score

    oracle_targets = {
        "H": marginal_oracle(v140_train_target, truth, (1,)),
        "V": marginal_oracle(v140_train_target, truth, (2,)),
        "D": marginal_oracle(v140_train_target, truth, (3,)),
        "HV": marginal_oracle(v140_train_target, truth, (1, 2)),
        "HD": marginal_oracle(v140_train_target, truth, (1, 3)),
        "VD": marginal_oracle(v140_train_target, truth, (2, 3)),
    }
    oracle_gain = {
        name: support_score(target, truth) - v140_score
        for name, target in oracle_targets.items()
    }
    oracle_gain["full_support"] = support_score(truth, truth) - v140_score
    best_axis_name = np.array(["H", "V", "D"])[
        np.argmax(np.stack([oracle_gain["H"], oracle_gain["V"], oracle_gain["D"]], axis=1), axis=1)
    ]
    best_axis_gain = np.max(
        np.stack([oracle_gain["H"], oracle_gain["V"], oracle_gain["D"]], axis=1),
        axis=1,
    ).astype(np.float32)
    best_temperature, temperature_gain, temperature_counts = temperature_oracle(
        v140_train_target, truth, valid
    )

    distance = np.linalg.norm(pos_train[:, :2] - np.array([50.0, 0.0]), axis=1)
    azimuth = np.arctan2(pos_train[:, 1], pos_train[:, 0] - 50.0)
    basic_train = np.column_stack(
        [
            pos_train[:, 0],
            pos_train[:, 1],
            distance,
            np.sin(azimuth),
            np.cos(azimuth),
            height["train_blocked"],
        ]
    ).astype(np.float32)
    distance_test = np.linalg.norm(pos_test[:, :2] - np.array([50.0, 0.0]), axis=1)
    azimuth_test = np.arctan2(pos_test[:, 1], pos_test[:, 0] - 50.0)
    basic_test = np.column_stack(
        [
            pos_test[:, 0],
            pos_test[:, 1],
            distance_test,
            np.sin(azimuth_test),
            np.cos(azimuth_test),
            height["test_blocked"],
        ]
    ).astype(np.float32)
    basic_names = ["x", "y", "distance_bs", "sin_az", "cos_az", "blocked"]
    template_names = [f"template_prob_{index:02d}" for index in range(probs_train.shape[1])]
    signature_train = support_features(source_train)
    signature_test = support_features(source_test)
    signature_names = [f"support_signature_{index:03d}" for index in range(signature_train.shape[1])]

    feature_sets = {
        "basic_geodesic": np.concatenate(
            [basic_train, geodesic["train_features"]], axis=1
        ),
        "official": np.concatenate(
            [basic_train, geodesic["train_features"], pointcloud["train_features"]],
            axis=1,
        ),
        "official_template": np.concatenate(
            [
                basic_train,
                geodesic["train_features"],
                pointcloud["train_features"],
                probs_train,
            ],
            axis=1,
        ),
        "full": np.concatenate(
            [
                basic_train,
                geodesic["train_features"],
                pointcloud["train_features"],
                probs_train,
                signature_train,
            ],
            axis=1,
        ),
    }
    feature_sets = {
        name: np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
            np.float32
        )
        for name, values in feature_sets.items()
    }
    spatial_groups = KMeans(n_clusters=25, random_state=RANDOM_STATE, n_init=20).fit_predict(
        pos_train[:, :2]
    )
    labels = {
        "graph_advantage": graph_advantage,
        "current_badness": current_badness,
        "best_axis_oracle_gain": best_axis_gain,
    }
    predictability = {
        label_name: {
            feature_name: regression_oof(
                features, label, spatial_groups, valid
            )
            for feature_name, features in feature_sets.items()
        }
        for label_name, label in labels.items()
    }

    domain_train = np.concatenate(
        [
            basic_train,
            geodesic["train_features"],
            pointcloud["train_features"],
            probs_train,
        ],
        axis=1,
    ).astype(np.float32)
    domain_test = np.concatenate(
        [
            basic_test,
            geodesic["test_features"],
            pointcloud["test_features"],
            probs_test,
        ],
        axis=1,
    ).astype(np.float32)
    domain_train = np.nan_to_num(domain_train, nan=0.0, posinf=8.0, neginf=-8.0)
    domain_test = np.nan_to_num(domain_test, nan=0.0, posinf=8.0, neginf=-8.0)
    domain_names = basic_names + geodesic_names + pointcloud_names + template_names
    boundaries = {
        "basic": (0, len(basic_names)),
        "geodesic": (len(basic_names), len(basic_names) + len(geodesic_names)),
        "pointcloud": (
            len(basic_names) + len(geodesic_names),
            len(basic_names) + len(geodesic_names) + len(pointcloud_names),
        ),
        "template": (
            len(basic_names) + len(geodesic_names) + len(pointcloud_names),
            len(domain_names),
        ),
    }
    domain_report, train_propensity, test_propensity = domain_analysis(
        domain_train, domain_test, domain_names, boundaries
    )
    propensity_bins = quantile_groups(
        train_propensity,
        {
            "graph_advantage": graph_advantage,
            "raw_gain": raw_gain,
            "graph_gain": graph_gain,
            "current_badness": current_badness,
            "best_axis_gain": best_axis_gain,
        },
        valid,
    )

    graph_values = {
        "graph_advantage": graph_advantage,
        "raw_gain": raw_gain,
        "graph_gain": graph_gain,
        "current_badness": current_badness,
        "best_axis_gain": best_axis_gain,
    }
    graph_analysis = {
        "by_risk_delta_quintile": quantile_groups(
            graph_train - raw_train, graph_values, valid
        ),
        "by_disagreement_quintile": quantile_groups(
            disagreement_train, graph_values, valid
        ),
        "by_raw_risk_quintile": quantile_groups(raw_train, graph_values, valid),
        "by_regime": category_groups(regimes_train, graph_values, valid),
        "risk_delta_graph_advantage_corr": safe_corr(
            graph_train - raw_train, graph_advantage
        ),
        "disagreement_graph_advantage_corr": safe_corr(
            disagreement_train, graph_advantage
        ),
        "changed_activation": {
            "train_added": int(np.sum((raw_train <= 0.64) & (graph_train > 0.64))),
            "train_removed": int(np.sum((raw_train > 0.64) & (graph_train <= 0.64))),
            "test_added": int(np.sum((raw_test <= 0.64) & (graph_test > 0.64))),
            "test_removed": int(np.sum((raw_test > 0.64) & (graph_test <= 0.64))),
        },
    }

    oracle_report = {
        "current_support_score": summarize(v140_score[valid]),
        "gain": {name: summarize(value[valid]) for name, value in oracle_gain.items()},
        "positive_fraction": {
            name: float(np.mean(value[valid] > 1e-8))
            for name, value in oracle_gain.items()
        },
        "best_single_axis_counts": {
            name: int(np.sum((best_axis_name == name) & valid)) for name in ("H", "V", "D")
        },
        "best_single_axis_gain": summarize(best_axis_gain[valid]),
        "temperature_best_counts": temperature_counts,
        "temperature_gain": summarize(temperature_gain[valid]),
        "temperature_by_regime": category_groups(
            regimes_train,
            {
                "best_power": best_temperature,
                "gain": temperature_gain,
            },
            valid,
        ),
    }

    test_risk_delta = graph_test - raw_test
    priority = (
        np.abs(test_risk_delta)
        * (0.4 + disagreement_test)
        * (0.25 + test_propensity)
    )
    order = np.argsort(priority)[::-1]
    with TEST_CSV.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "x",
                "y",
                "regime",
                "raw_risk",
                "graph_risk",
                "risk_delta",
                "graph_disagreement",
                "test_propensity",
                "priority",
            ]
        )
        for index in order:
            writer.writerow(
                [
                    int(index),
                    float(pos_test[index, 0]),
                    float(pos_test[index, 1]),
                    str(regimes_test[index]),
                    float(raw_test[index]),
                    float(graph_test[index]),
                    float(test_risk_delta[index]),
                    float(disagreement_test[index]),
                    float(test_propensity[index]),
                    float(priority[index]),
                ]
            )

    report = {
        "method": "post_v143_hidden_information_analysis",
        "online_evidence": {
            "v140": 0.501352,
            "v142": 0.500329,
            "v143": 0.501271,
            "v143_minus_v140": -0.000081,
            "interpretation": (
                "fixed graph diffusion nearly preserves v140 but does not beat it; "
                "look for conditional rather than global graph value"
            ),
        },
        "valid_train_points": int(valid.sum()),
        "excluded_exact_zero_points": int((~valid).sum()),
        "graph_analysis": graph_analysis,
        "support_oracles": oracle_report,
        "spatial_oof_predictability": predictability,
        "domain_shift": {
            **domain_report,
            "top_standardized_mean_differences": standardized_mean_difference(
                domain_train, domain_test, domain_names
            ),
            "train_propensity_quintiles": propensity_bins,
        },
        "test_priority": {
            "csv": PATHS.relative(TEST_CSV),
            "top_indices": [int(index) for index in order[:40]],
            "risk_delta": summarize(test_risk_delta),
            "graph_disagreement": summarize(disagreement_test),
        },
        "conclusions": [
            "fixed graph diffusion has near-zero online value and graph advantage is not spatially predictable",
            "v143 train diagnostics mixed v125 propagation-template probabilities with v129 residual-cluster probabilities",
            "post-v140 residual capacity is largest in HV and other axis marginals, not integer shifts",
            "official pointcloud, matched residual probabilities, and current support enrich predictable badness and best-axis gain",
            "the next primary hypothesis should be low-strength continuous marginal-ratio correction, led by V and D",
        ],
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {REPORT}")
    print(json.dumps(report["graph_analysis"], indent=2))
    print(json.dumps(report["support_oracles"], indent=2))
    print(json.dumps(report["spatial_oof_predictability"], indent=2))
    print(json.dumps(report["domain_shift"], indent=2))


if __name__ == "__main__":
    main()
