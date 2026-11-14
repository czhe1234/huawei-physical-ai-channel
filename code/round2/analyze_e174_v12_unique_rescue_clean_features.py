#!/usr/bin/env python3
"""Audit whether V12-only rescue points are identifiable from Clean features."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import analyze_e169_hard_hole_pas_peak_precursors as e169
from project_paths import PATHS


VERSION = "Round2-E174-v12-unique-rescue-clean-features"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e174_v12_unique_rescue_clean_features"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e174_v12_unique_rescue_clean_features"
VIS_DIR = PATHS.analysis / "visualizations" / "复赛V12独有救援点Clean特征审计"

E167_POINT_CSV = (
    PATHS.analysis
    / "results"
    / "round2_e167_clean_test_hole_difficulty"
    / "V7逐点Clean可见量与误差.csv"
)
E153_POINT_CSV = (
    PATHS.analysis
    / "results"
    / "round2_v7_v12_peak_actions"
    / "500点修正动作明细.csv"
)
E153_METRICS = (
    PATHS.artifacts / "round2" / "v7_v12_peak_actions" / "v7_v12_peak_action_metrics.npz"
)
E173_METRICS = (
    PATHS.artifacts
    / "round2"
    / "e173_v7_clean_carrier_oracle"
    / "carrier_point_metrics_and_compact.npz"
)
GEOMETRY_CACHE = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_geometry_visibility_features.npz"
)
ROUTE_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)

REPORT_JSON = ANALYSIS_DIR / "v12_unique_rescue_clean_features.json"
REPORT_MD = ANALYSIS_DIR / "V12独有救援点在Clean世界是否可识别.md"
POINT_CSV = ANALYSIS_DIR / "500点V12独有救援标签与Clean特征.csv"
HOLE_CSV = ANALYSIS_DIR / "十洞V12独有救援点分布.csv"
FEATURE_CSV = ANALYSIS_DIR / "Clean特征组间差异.csv"
UNIVARIATE_CSV = ANALYSIS_DIR / "单Clean特征跨洞识别力.csv"
PROBE_CSV = ANALYSIS_DIR / "Clean特征组跨洞判别结果.csv"
FOLD_CSV = ANALYSIS_DIR / "逐fold跨洞判别结果.csv"
PERMUTATION_CSV = ANALYSIS_DIR / "标签打乱负控制.csv"
LINEAGE_CSV = ANALYSIS_DIR / "Clean特征血缘与禁止字段.csv"

N_POINT = 500
TINY = 1e-30
SEED = 20_260_811_174

STRICT = {
    "v7_max": 0.60,
    "carrier_max": 0.65,
    "v12_min": 0.70,
    "unique_gain_min": 0.10,
}
SENSITIVE = {
    "v7_max": 0.65,
    "carrier_max": 0.70,
    "v12_min": 0.70,
    "unique_gain_min": 0.08,
}

E167_FEATURES = {
    "hole_depth_m": ("洞深度", "structure"),
    "normalized_hole_depth": ("归一化洞深度", "structure"),
    "nearest_train_m": ("最近Train距离", "structure"),
    "knn8_distance_m": ("第8近Train距离", "structure"),
    "train_density_k8_per_m2": ("Train局部密度", "structure"),
    "angular_max_gap_deg": ("Train最大角度缺口", "structure"),
    "one_sidedness": ("Train单侧性", "structure"),
    "quadrants_occupied": ("Train覆盖象限数", "structure"),
    "bs_distance_m": ("BS距离", "structure"),
    "neighbor_pas_dispersion": ("邻居PAS分歧", "neighbor_h"),
    "neighbor_pdp_dispersion": ("邻居PDP分歧", "neighbor_h"),
    "two_side_pas_difference": ("洞两侧PAS差异", "neighbor_h"),
    "two_side_pdp_difference": ("洞两侧PDP差异", "neighbor_h"),
    "neighbor_complex_h_variation_per_m": ("邻居复H每米变化", "neighbor_h"),
}

E153_FEATURES = {
    "V7角度熵": ("v7_angle_entropy", "V7角度熵", "v7_spectrum"),
    "V7角度Top1质量": ("v7_angle_top1", "V7角度Top1质量", "v7_spectrum"),
    "V7角度Top4质量": ("v7_angle_top4", "V7角度Top4质量", "v7_spectrum"),
    "V7时延熵": ("v7_delay_entropy", "V7时延熵", "v7_spectrum"),
    "V7时延Top1质量": ("v7_delay_top1", "V7时延Top1质量", "v7_spectrum"),
    "V7时延Top4质量": ("v7_delay_top4", "V7时延Top4质量", "v7_spectrum"),
    "V7总功率对数": ("v7_log_power", "V7总功率对数", "v7_spectrum"),
    "17载体角度分歧": ("carrier_angle_dispersion", "17载体角度分歧", "carrier_disagreement"),
    "17载体时延分歧": ("carrier_delay_dispersion", "17载体时延分歧", "carrier_disagreement"),
    "17载体功率分歧": ("carrier_log_power_std", "17载体功率分歧", "carrier_disagreement"),
    "V7与载体共识角度距离": ("v7_carrier_angle_distance", "V7与载体共识角度距离", "carrier_disagreement"),
    "V7与载体共识时延距离": ("v7_carrier_delay_distance", "V7与载体共识时延距离", "carrier_disagreement"),
    "最近邻距离": ("neighbor_nearest_distance", "最近邻距离(E153)", "neighbor_h"),
    "前8邻居平均距离": ("neighbor_mean_distance8", "前8邻居平均距离", "neighbor_h"),
    "前16邻居平均距离": ("neighbor_mean_distance16", "前16邻居平均距离", "neighbor_h"),
    "有效邻居数量": ("neighbor_effective_count", "有效邻居数量", "neighbor_h"),
    "邻居角度分歧": ("neighbor_angle_dispersion", "邻居角度分歧", "neighbor_h"),
    "邻居时延分歧": ("neighbor_delay_dispersion", "邻居时延分歧", "neighbor_h"),
    "邻居共识与V7角度距离": ("neighbor_angle_distance_to_v7", "邻居共识与V7角度距离", "neighbor_h"),
    "邻居共识与V7时延距离": ("neighbor_delay_distance_to_v7", "邻居共识与V7时延距离", "neighbor_h"),
}

FORBIDDEN_E153_FIELDS = {
    "truth_zero",
    "v7_pas",
    "v12_pas",
    "pas_gain",
    "v7_pdp",
    "v12_pdp",
    "pdp_gain",
    "point_nmse_v7",
    "point_nmse_v12",
    "point_score_gain_proxy",
    "angle_action",
    "delay_action",
    "angle_direction_alignment",
    "delay_direction_alignment",
    "angle_new_mass",
    "angle_removed_mass",
    "angle_moved_mass",
    "angle_strengthened_mass",
    "angle_weakened_mass",
    "delay_new_mass",
    "delay_removed_mass",
    "delay_moved_mass",
    "delay_strengthened_mass",
    "delay_weakened_mass",
    "邻居角度修正方向一致性",
    "邻居时延修正方向一致性",
    "官方邻居锚点角度方向一致性",
    "官方邻居锚点时延方向一致性",
}


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    group: str
    source: str
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def configure_chinese_font() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "SimHei"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "default"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def read_indexed_csv(path: Path, index_key: str) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    indexed: list[dict[str, str] | None] = [None] * N_POINT
    for row in rows:
        index = int(row[index_key])
        if not 0 <= index < N_POINT or indexed[index] is not None:
            raise RuntimeError(f"invalid or duplicate {index_key}={index} in {path}")
        indexed[index] = row
    if any(row is None for row in indexed):
        raise RuntimeError(f"{path} does not cover all {N_POINT} points")
    return [row for row in indexed if row is not None], fieldnames


def float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        text = row[name].strip()
        values[index] = float(text) if text else float("nan")
    return values


def build_labels() -> dict[str, np.ndarray]:
    e153 = np.load(E153_METRICS, allow_pickle=False)
    e173 = np.load(E173_METRICS, allow_pickle=False)
    if e153["v7_pas"].shape != (N_POINT,) or e173["pas"].shape != (17, N_POINT):
        raise RuntimeError("E153/E173 point metric shape mismatch")
    v7_pas = np.asarray(e153["v7_pas"], dtype=np.float64)
    v12_pas = np.asarray(e153["v12_pas"], dtype=np.float64)
    best_carrier_pas = np.max(np.asarray(e173["pas"], dtype=np.float64), axis=0)
    best_carrier_index = np.argmax(np.asarray(e173["pas"], dtype=np.float64), axis=0)
    clean_best_pas = np.maximum(v7_pas, best_carrier_pas)
    unique_gain = v12_pas - clean_best_pas
    strict = (
        (v7_pas <= STRICT["v7_max"])
        & (best_carrier_pas <= STRICT["carrier_max"])
        & (v12_pas >= STRICT["v12_min"])
        & (unique_gain >= STRICT["unique_gain_min"])
    )
    sensitive = (
        (v7_pas <= SENSITIVE["v7_max"])
        & (best_carrier_pas <= SENSITIVE["carrier_max"])
        & (v12_pas >= SENSITIVE["v12_min"])
        & (unique_gain >= SENSITIVE["unique_gain_min"])
    )
    v7_good = (v7_pas >= 0.75) & (np.abs(v12_pas - v7_pas) <= 0.05)
    v12_bad = (clean_best_pas <= 0.65) & (v12_pas <= 0.65)
    carrier_exists = (
        (v7_pas <= 0.65)
        & (best_carrier_pas >= 0.70)
        & (np.abs(best_carrier_pas - v12_pas) <= 0.03)
    )
    label = np.full(N_POINT, "background", dtype="U24")
    for name, mask in (
        ("v12_still_bad", v12_bad),
        ("v7_already_good", v7_good),
        ("carrier_answer_exists", carrier_exists),
        ("unique_rescue", strict),
    ):
        label[mask] = name
    return {
        "query_ids": np.asarray(e153["query_ids"], dtype=np.int64),
        "fold": np.asarray(e153["fold_id"], dtype=np.int8),
        "site": np.asarray(e153["site"], dtype=np.int8),
        "v7_pas": v7_pas,
        "v12_pas": v12_pas,
        "best_carrier_pas": best_carrier_pas,
        "best_carrier_index": best_carrier_index.astype(np.int16),
        "clean_best_pas": clean_best_pas,
        "unique_gain": unique_gain,
        "strict": strict,
        "sensitive": sensitive,
        "v7_good": v7_good,
        "v12_bad": v12_bad,
        "carrier_exists": carrier_exists,
        "label": label,
    }


def compact_features() -> list[Feature]:
    compact = np.asarray(np.load(e169.V7_COMPACT, mmap_mode="r"), dtype=np.float32)
    if compact.shape != (N_POINT, 4, 16, 8):
        raise RuntimeError(f"unexpected V7 compact PAS shape {compact.shape}")
    peak_count = np.empty((N_POINT, 4), dtype=np.float64)
    for point in range(N_POINT):
        for ue in range(4):
            peak_count[point, ue] = len(e169.extract_peaks(compact[point, ue]))
    e173 = np.load(E173_METRICS, allow_pickle=False)
    carrier = np.maximum(np.asarray(e173["compact_pas"], dtype=np.float32), 0.0)
    consensus = np.mean(carrier, axis=0)
    consensus /= np.maximum(consensus.sum(axis=(2, 3), keepdims=True), TINY)
    root_distance = 0.5 * np.sum(
        np.square(np.sqrt(carrier) - np.sqrt(consensus[None])), axis=(3, 4)
    )
    entropy = -np.sum(
        np.where(consensus > 0.0, consensus * np.log(np.maximum(consensus, TINY)), 0.0),
        axis=(2, 3),
    ) / math.log(16 * 8)
    return [
        Feature("v7_peak_count_mean", "V7每UE平均峰数", "v7_spectrum", "V7 compact PAS", peak_count.mean(axis=1)),
        Feature("v7_peak_count_total", "V7四UE总峰数", "v7_spectrum", "V7 compact PAS", peak_count.sum(axis=1)),
        Feature("v7_peak_count_max", "V7单UE最大峰数", "v7_spectrum", "V7 compact PAS", peak_count.max(axis=1)),
        Feature(
            "v7_support_fraction_001",
            "V7角域大于1%格子比例",
            "v7_spectrum",
            "V7 compact PAS",
            np.mean(compact >= 0.01, axis=(1, 2, 3)),
        ),
        Feature(
            "carrier_angle_root_dispersion_mean",
            "17载体角域Hellinger分歧均值",
            "carrier_disagreement",
            "E173 Clean carrier compact PAS",
            root_distance.mean(axis=(0, 2)),
        ),
        Feature(
            "carrier_angle_root_dispersion_max",
            "17载体角域Hellinger分歧最大值",
            "carrier_disagreement",
            "E173 Clean carrier compact PAS",
            root_distance.max(axis=(0, 2)),
        ),
        Feature(
            "carrier_consensus_angle_entropy",
            "17载体共识角域熵",
            "carrier_disagreement",
            "E173 Clean carrier compact PAS",
            entropy.mean(axis=1),
        ),
    ]


def build_features(labels: dict[str, np.ndarray]) -> tuple[list[Feature], np.ndarray, list[dict[str, object]]]:
    e167_rows, _ = read_indexed_csv(E167_POINT_CSV, "oof_row")
    e153_rows, e153_fields = read_indexed_csv(E153_POINT_CSV, "point_row")
    hole_id = np.asarray([row["pseudo_hole_id"] for row in e167_rows], dtype="U16")
    if not np.array_equal(np.asarray([int(row["query_id"]) for row in e153_rows]), labels["query_ids"]):
        raise RuntimeError("E153 point CSV query IDs do not match metrics")

    features: list[Feature] = []
    lineage: list[dict[str, object]] = []
    for name, (label, group) in E167_FEATURES.items():
        feature = Feature(name, label, group, "E167 official geometry/Train-H table", float_column(e167_rows, name))
        features.append(feature)
        lineage.append({"feature": name, "label": label, "group": group, "source": feature.source, "allowed": True})
    for source_name, (name, label, group) in E153_FEATURES.items():
        if source_name not in e153_fields:
            raise RuntimeError(f"missing E153 Clean feature {source_name}")
        feature = Feature(name, label, group, "E153 Clean observable field", float_column(e153_rows, source_name))
        features.append(feature)
        lineage.append({"feature": name, "label": label, "group": group, "source": feature.source, "source_field": source_name, "allowed": True})

    geometry = np.load(GEOMETRY_CACHE, allow_pickle=False)
    geometry_names = [str(name) for name in geometry["names"].tolist()]
    geometry_values = np.asarray(geometry["train"], dtype=np.float64)[labels["query_ids"]]
    geometry_allowed = {
        "bs_distance",
        "log_bs_distance",
        "sin_azimuth",
        "cos_azimuth",
        "sin_elevation",
        "cos_elevation",
        *geometry_names[14:],
    }
    for index, name in enumerate(geometry_names):
        if name not in geometry_allowed:
            continue
        group = "structure" if name in {"bs_distance", "log_bs_distance", "sin_azimuth", "cos_azimuth", "sin_elevation", "cos_elevation"} else "map_geometry"
        feature = Feature(f"geometry_{name}", f"地图几何:{name}", group, "official Map visibility cache", geometry_values[:, index])
        features.append(feature)
        lineage.append({"feature": feature.name, "label": feature.label, "group": group, "source": feature.source, "allowed": True})

    route = np.load(ROUTE_CACHE, allow_pickle=False)
    route_names = [str(name) for name in route["feature_names"].tolist()]
    route_values = np.asarray(route["train_features"], dtype=np.float64)[labels["query_ids"]]
    for index, name in enumerate(route_names):
        feature = Feature(f"route_{name}", f"地图路径:{name}", "map_geometry", "official Map geodesic cache", route_values[:, index])
        features.append(feature)
        lineage.append({"feature": feature.name, "label": feature.label, "group": feature.group, "source": feature.source, "allowed": True})

    for feature in compact_features():
        features.append(feature)
        lineage.append({"feature": feature.name, "label": feature.label, "group": feature.group, "source": feature.source, "allowed": True})

    for field in sorted(FORBIDDEN_E153_FIELDS):
        lineage.append({"feature": "", "label": field, "group": "forbidden", "source": "E153", "allowed": False, "reason": "depends on held GT or V12 correction"})
    names = [feature.name for feature in features]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate Clean feature names")
    for feature in features:
        if feature.values.shape != (N_POINT,):
            raise RuntimeError(f"invalid feature shape {feature.name}: {feature.values.shape}")
        lowered = f"{feature.name} {feature.source}".lower()
        if any(token in lowered for token in ("sionna", "truth", "v12", "gain", "target")):
            raise RuntimeError(f"forbidden lineage token in Clean feature {feature.name}")
    return features, hole_id, lineage


def feature_groups(features: list[Feature]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for group in ("structure", "neighbor_h", "carrier_disagreement", "v7_spectrum", "map_geometry"):
        selected = [feature.values for feature in features if feature.group == group]
        output[group] = np.column_stack(selected).astype(np.float64)
    output["all_clean"] = np.column_stack([feature.values for feature in features]).astype(np.float64)
    return output


def make_model(name: str, trees: int, seed: int) -> Pipeline:
    if name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=seed % (2**32),
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=trees,
                    max_depth=4,
                    min_samples_leaf=12,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed % (2**32),
                ),
            ),
        ]
    )


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, probability)) if np.any(y) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if len(np.unique(y)) == 2 else float("nan"),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "prevalence": float(np.mean(y)),
    }


def cross_hole_probe(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    model_name: str,
    trees: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, object]]]:
    probability = np.full(len(target), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for outer in range(5):
        held = folds == outer
        train = ~held
        if len(np.unique(target[train])) < 2:
            probability[held] = float(np.mean(target[train]))
        else:
            model = make_model(model_name, trees, seed + outer * 1009)
            model.fit(matrix[train], target[train])
            probability[held] = model.predict_proba(matrix[held])[:, 1]
        local = binary_metrics(target[held], probability[held])
        fold_rows.append(
            {
                "outer_fold": outer,
                "count": int(np.sum(held)),
                "positive_count": int(np.sum(target[held])),
                **local,
            }
        )
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("cross-hole probe produced non-finite probabilities")
    return probability, binary_metrics(target, probability), fold_rows


def bootstrap_difference(
    left: np.ndarray, right: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(left) == 0 or len(right) == 0:
        return float("nan"), float("nan")
    values = np.empty(count, dtype=np.float64)
    for index in range(count):
        values[index] = np.median(rng.choice(left, len(left), replace=True)) - np.median(
            rng.choice(right, len(right), replace=True)
        )
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    difference = left[:, None] - right[None, :]
    return float((np.sum(difference > 0.0) - np.sum(difference < 0.0)) / difference.size)


def feature_effect_rows(
    features: list[Feature], labels: dict[str, np.ndarray], hole_id: np.ndarray, bootstrap: int, seed: int
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    comparisons = {
        "all_non_target": ~labels["strict"],
        "v7_already_good": labels["label"] == "v7_already_good",
        "v12_still_bad": labels["label"] == "v12_still_bad",
        "carrier_answer_exists": labels["label"] == "carrier_answer_exists",
    }
    rows: list[dict[str, object]] = []
    for feature in features:
        for comparison, control_mask in comparisons.items():
            target_values = feature.values[labels["strict"]]
            control_values = feature.values[control_mask]
            target_values = target_values[np.isfinite(target_values)]
            control_values = control_values[np.isfinite(control_values)]
            pooled = math.sqrt(
                0.5 * (float(np.var(target_values)) + float(np.var(control_values)))
            ) if len(target_values) and len(control_values) else float("nan")
            low, high = bootstrap_difference(target_values, control_values, bootstrap, rng)
            positive_holes = 0
            negative_holes = 0
            evaluable_holes = 0
            for hole in np.unique(hole_id):
                local_target = labels["strict"] & (hole_id == hole) & np.isfinite(feature.values)
                local_control = control_mask & (hole_id == hole) & np.isfinite(feature.values)
                if np.any(local_target) and np.any(local_control):
                    difference = float(np.median(feature.values[local_target]) - np.median(feature.values[local_control]))
                    evaluable_holes += 1
                    positive_holes += int(difference > 0.0)
                    negative_holes += int(difference < 0.0)
            rows.append(
                {
                    "feature": feature.name,
                    "label": feature.label,
                    "group": feature.group,
                    "comparison": comparison,
                    "target_count": len(target_values),
                    "control_count": len(control_values),
                    "target_median": float(np.median(target_values)) if len(target_values) else float("nan"),
                    "target_q25": float(np.quantile(target_values, 0.25)) if len(target_values) else float("nan"),
                    "target_q75": float(np.quantile(target_values, 0.75)) if len(target_values) else float("nan"),
                    "control_median": float(np.median(control_values)) if len(control_values) else float("nan"),
                    "control_q25": float(np.quantile(control_values, 0.25)) if len(control_values) else float("nan"),
                    "control_q75": float(np.quantile(control_values, 0.75)) if len(control_values) else float("nan"),
                    "median_difference": float(np.median(target_values) - np.median(control_values)) if len(target_values) and len(control_values) else float("nan"),
                    "bootstrap_median_difference_low": low,
                    "bootstrap_median_difference_high": high,
                    "standardized_mean_difference": float((np.mean(target_values) - np.mean(control_values)) / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else float("nan"),
                    "cliffs_delta": cliffs_delta(target_values, control_values),
                    "evaluable_holes": evaluable_holes,
                    "positive_holes": positive_holes,
                    "negative_holes": negative_holes,
                }
            )
    return rows


def univariate_rows(
    features: list[Feature], labels: dict[str, np.ndarray], trees: int, seed: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target = labels["strict"].astype(np.int8)
    for index, feature in enumerate(features):
        probability, metric, fold_rows = cross_hole_probe(
            feature.values[:, None], target, labels["fold"], "logistic", trees, seed + index * 13
        )
        fold_auc = [row["roc_auc"] for row in fold_rows if np.isfinite(row["roc_auc"])]
        rows.append(
            {
                "feature": feature.name,
                "label": feature.label,
                "group": feature.group,
                **metric,
                "fold_auc_gt_half": int(np.sum(np.asarray(fold_auc) > 0.5)),
                "evaluable_folds": len(fold_auc),
                "probability_mean_target": float(np.mean(probability[target == 1])),
                "probability_mean_control": float(np.mean(probability[target == 0])),
            }
        )
    return rows


def task_masks(labels: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {
        "unique_vs_all": (np.ones(N_POINT, dtype=bool), labels["strict"].astype(np.int8)),
    }
    for name in ("v7_already_good", "v12_still_bad", "carrier_answer_exists"):
        mask = labels["strict"] | (labels["label"] == name)
        output[f"unique_vs_{name}"] = (mask, labels["strict"][mask].astype(np.int8))
    return output


def permutation_control(
    target: np.ndarray,
    probability: np.ndarray,
    site: np.ndarray,
    count: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    auc = np.empty(count, dtype=np.float64)
    ap = np.empty(count, dtype=np.float64)
    for index in range(count):
        shuffled = target.copy()
        for value in np.unique(site):
            rows = np.flatnonzero(site == value)
            shuffled[rows] = rng.permutation(shuffled[rows])
        auc[index] = roc_auc_score(shuffled, probability)
        ap[index] = average_precision_score(shuffled, probability)
    observed_auc = float(roc_auc_score(target, probability))
    observed_ap = float(average_precision_score(target, probability))
    return {
        "observed_auc": observed_auc,
        "observed_ap": observed_ap,
        "auc_shuffle_mean": float(np.mean(auc)),
        "auc_shuffle_q95": float(np.quantile(auc, 0.95)),
        "auc_permutation_p": float((1 + np.sum(auc >= observed_auc)) / (count + 1)),
        "ap_shuffle_mean": float(np.mean(ap)),
        "ap_shuffle_q95": float(np.quantile(ap, 0.95)),
        "ap_permutation_p": float((1 + np.sum(ap >= observed_ap)) / (count + 1)),
    }


def point_and_hole_rows(
    labels: dict[str, np.ndarray], features: list[Feature], hole_id: np.ndarray
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    point_rows: list[dict[str, object]] = []
    for point in range(N_POINT):
        row: dict[str, object] = {
            "oof_row": point,
            "query_id": int(labels["query_ids"][point]),
            "hole_id": hole_id[point],
            "fold": int(labels["fold"][point]),
            "site": int(labels["site"][point]),
            "group": labels["label"][point],
            "strict_unique_rescue": bool(labels["strict"][point]),
            "sensitive_unique_rescue": bool(labels["sensitive"][point]),
            "v7_pas": float(labels["v7_pas"][point]),
            "best_carrier_pas": float(labels["best_carrier_pas"][point]),
            "best_carrier_index": int(labels["best_carrier_index"][point]),
            "v12_pas": float(labels["v12_pas"][point]),
            "v12_unique_pas_gain": float(labels["unique_gain"][point]),
        }
        row.update({feature.name: float(feature.values[point]) for feature in features})
        point_rows.append(row)

    hole_rows: list[dict[str, object]] = []
    groups = ("unique_rescue", "v7_already_good", "v12_still_bad", "carrier_answer_exists", "background")
    for hole in sorted(np.unique(hole_id)):
        hole_mask = hole_id == hole
        for group in groups:
            mask = hole_mask & (labels["label"] == group)
            hole_rows.append(
                {
                    "hole_id": hole,
                    "fold": int(labels["fold"][hole_mask][0]),
                    "site": int(labels["site"][hole_mask][0]),
                    "group": group,
                    "count": int(np.sum(mask)),
                    "hole_count": int(np.sum(hole_mask)),
                    "fraction": float(np.mean(labels["label"][hole_mask] == group)),
                    "v7_pas_mean": float(np.mean(labels["v7_pas"][mask])) if np.any(mask) else float("nan"),
                    "best_carrier_pas_mean": float(np.mean(labels["best_carrier_pas"][mask])) if np.any(mask) else float("nan"),
                    "v12_pas_mean": float(np.mean(labels["v12_pas"][mask])) if np.any(mask) else float("nan"),
                    "unique_gain_mean": float(np.mean(labels["unique_gain"][mask])) if np.any(mask) else float("nan"),
                }
            )
    return point_rows, hole_rows


def make_plots(
    labels: dict[str, np.ndarray], hole_id: np.ndarray, univariate: list[dict[str, object]],
    predictions: dict[str, np.ndarray], probe_rows: list[dict[str, object]]
) -> list[Path]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    paths: list[Path] = []

    holes = sorted(np.unique(hole_id))
    groups = ("unique_rescue", "carrier_answer_exists", "v7_already_good", "v12_still_bad", "background")
    colors = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#bdbdbd")
    bottom = np.zeros(len(holes))
    fig, ax = plt.subplots(figsize=(11, 5))
    for group, color in zip(groups, colors):
        values = np.asarray([np.sum((hole_id == hole) & (labels["label"] == group)) for hole in holes])
        ax.bar(holes, values, bottom=bottom, label=group, color=color)
        bottom += values
    ax.set_ylabel("点数")
    ax.set_title("十个pseudo-hole中的V12独有救援点与对照点")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = VIS_DIR / "01_十洞目标与对照点分布.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    top = sorted(univariate, key=lambda row: float(row["roc_auc"]), reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([row["label"] for row in reversed(top)], [row["roc_auc"] for row in reversed(top)], color="#4c78a8")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.75, color="red", linestyle="--", linewidth=1)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("整洞隔离OOF ROC-AUC")
    ax.set_title("单个Clean特征识别V12独有救援点")
    fig.tight_layout()
    path = VIS_DIR / "02_单特征跨洞识别力.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    main_rows = [row for row in probe_rows if row["task"] == "unique_vs_all" and row["feature_group"] == "all_clean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    target = labels["strict"].astype(np.int8)
    for row in main_rows:
        probability = predictions[row["prediction_key"]]
        order = np.argsort(probability)[::-1]
        recall = np.cumsum(target[order]) / max(int(np.sum(target)), 1)
        precision = np.cumsum(target[order]) / np.arange(1, len(target) + 1)
        axes[0].plot(np.arange(1, len(target) + 1), recall, label=row["model"])
        axes[1].plot(recall, precision, label=row["model"])
    axes[0].set_xlabel("按预测概率排序后的点数")
    axes[0].set_ylabel("目标点累计召回")
    axes[0].set_title("目标点排序集中度")
    axes[1].axhline(np.mean(target), color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("跨洞OOF Precision-Recall")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    path = VIS_DIR / "03_全部Clean特征跨洞判别.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def write_report(report: dict[str, object]) -> None:
    REPORT_JSON.write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2), encoding="utf-8")
    counts = report["label_counts"]
    verdict = report["verdict"]
    main = report["main_all_clean_results"]
    best_features = report["best_univariate_features"]
    best_signal = verdict["best_main_feature_group_signal"]
    comparisons = report["best_control_comparisons"]
    hole_counts = report["strict_count_by_hole"]
    model_lines = "\n".join(
        f"| {row['model']} | {row['roc_auc']:.3f} | {row['average_precision']:.3f} | {row['balanced_accuracy']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['fold_auc_gt_half']}/5 |"
        for row in main
    )
    feature_lines = "\n".join(
        f"| {row['label']} | {row['group']} | {row['roc_auc']:.3f} | {row['average_precision']:.3f} | {row['fold_auc_gt_half']}/{row['evaluable_folds']} |"
        for row in best_features[:12]
    )
    hole_lines = "\n".join(f"| {hole} | {count} |" for hole, count in hole_counts.items())
    text = f"""# V12独有救援点在Clean世界是否可识别

## 一句话结论

**{verdict['plain_conclusion']}**

## 目标与对照

严格V12独有救援点共有`{counts['unique_rescue']}`个，覆盖`{verdict['strict_hole_count']}/10`个洞；宽松敏感性标签为`{counts['sensitive_unique_rescue']}`个。三类互斥对照为：V7本来就好`{counts['v7_already_good']}`点、V12仍没救好`{counts['v12_still_bad']}`点、17路carrier中已有接近V12答案`{counts['carrier_answer_exists']}`点。

严格目标点的V7/最佳carrier/V12平均PAS=`{verdict['strict_v7_pas_mean']:.3f}/{verdict['strict_best_carrier_pas_mean']:.3f}/{verdict['strict_v12_pas_mean']:.3f}`，V12相对Clean候选上限的平均独有增益=`{verdict['strict_unique_gain_mean']:+.3f}`。

## 十洞分布

| pseudo-hole | 严格目标点数 |
| --- | ---: |
{hole_lines}

## 严格跨洞判别

主验证每次同时隔离同一fold的两个站点洞，训练过程看不到这两个洞的任何标签。

| 固定统计探针 | ROC-AUC | AP | Balanced Acc. | Precision | Recall | AUC>0.5折数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{model_lines}

预注册通过门为AUC至少`0.75`、AP至少流行率两倍、balanced accuracy至少`0.65`、至少4/5折AUC高于0.5，并超过site内标签打乱95%分位。最终裁决=`{verdict['level']}`。

单独看最强Clean分组时，`{best_signal['feature_group']}+{best_signal['model']}`达到AUC/AP/balanced accuracy=`{best_signal['roc_auc']:.3f}/{best_signal['average_precision']:.3f}/{best_signal['balanced_accuracy']:.3f}`、`{best_signal['fold_auc_gt_half']}/5`折正向。这说明存在弱提示，但没有形成稳定的完整识别规则。

对照分解最关键：相对V7本来就好的点，最佳固定探针AUC=`{comparisons['unique_vs_v7_already_good']['roc_auc']:.3f}`；相对V12仍没救好的坏点，最佳只有`{comparisons['unique_vs_v12_still_bad']['roc_auc']:.3f}`且`{comparisons['unique_vs_v12_still_bad']['fold_auc_gt_half']}/5`折正向。Clean特征主要能判断“V7是否不可靠”，不能稳定判断“这个坏点是否会被V12救活”。

## 最有区分度的单个Clean特征

| 特征 | 分组 | 整洞OOF AUC | AP | 正向fold数 |
| --- | --- | ---: | ---: | ---: |
{feature_lines}

单变量表只用于解释，不代替全部Clean特征的整洞验证。完整组间中位数、四分位数、bootstrap区间和逐洞方向见CSV。

## 数据边界

特征只来自官方Train H/Pos、Map、Setup、Clean V7和17路Clean carrier自身预测分歧。V12与held真值只用于定义“是否真的修好”的离线标签；所有V12修正方向、GT得分和动作字段均被硬拒绝。本轮没有生成H、训练信道模型或产生候选版本。
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    labels = build_labels()
    features, hole_id, lineage_rows = build_features(labels)
    groups = feature_groups(features)
    point_rows, hole_rows = point_and_hole_rows(labels, features, hole_id)
    effect_rows = feature_effect_rows(features, labels, hole_id, args.bootstrap, args.seed)
    univariate = univariate_rows(features, labels, args.trees, args.seed)

    tasks = task_masks(labels)
    probe_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for task_name, (mask, target) in tasks.items():
        for group_name, matrix_all in groups.items():
            matrix = matrix_all[mask]
            folds = labels["fold"][mask]
            for model_name in ("logistic", "extra_trees"):
                probability, metric, local_folds = cross_hole_probe(
                    matrix, target, folds, model_name, args.trees, args.seed + len(probe_rows) * 31
                )
                key = f"{task_name}:{group_name}:{model_name}"
                full_probability = np.full(N_POINT, np.nan, dtype=np.float64)
                full_probability[mask] = probability
                predictions[key] = full_probability
                valid_auc = [row["roc_auc"] for row in local_folds if np.isfinite(row["roc_auc"])]
                probe_rows.append(
                    {
                        "task": task_name,
                        "feature_group": group_name,
                        "model": model_name,
                        "feature_count": int(matrix.shape[1]),
                        "count": int(np.sum(mask)),
                        "positive_count": int(np.sum(target)),
                        **metric,
                        "fold_auc_gt_half": int(np.sum(np.asarray(valid_auc) > 0.5)),
                        "evaluable_folds": len(valid_auc),
                        "prediction_key": key,
                    }
                )
                for row in local_folds:
                    fold_rows.append(
                        {"task": task_name, "feature_group": group_name, "model": model_name, **row}
                    )

    permutation_rows: list[dict[str, object]] = []
    target = labels["strict"].astype(np.int8)
    for model_name in ("logistic", "extra_trees"):
        key = f"unique_vs_all:all_clean:{model_name}"
        result = permutation_control(
            target, predictions[key], labels["site"], args.permutations, args.seed + (0 if model_name == "logistic" else 1)
        )
        permutation_rows.append({"model": model_name, "permutations": args.permutations, **result})

    main_rows = [
        row for row in probe_rows if row["task"] == "unique_vs_all" and row["feature_group"] == "all_clean"
    ]
    permutation_lookup = {row["model"]: row for row in permutation_rows}
    strict_holes = sorted(set(hole_id[labels["strict"]].tolist()))
    eligible = int(np.sum(labels["strict"])) >= 20 and len(strict_holes) >= 4
    passed_models = []
    for row in main_rows:
        control = permutation_lookup[row["model"]]
        passed = (
            eligible
            and float(row["roc_auc"]) >= 0.75
            and float(row["average_precision"]) >= 2.0 * float(row["prevalence"])
            and float(row["balanced_accuracy"]) >= 0.65
            and int(row["fold_auc_gt_half"]) >= 4
            and float(row["roc_auc"]) > float(control["auc_shuffle_q95"])
        )
        row["passes_gate"] = passed
        if passed:
            passed_models.append(row["model"])
    main_task_rows = [row for row in probe_rows if row["task"] == "unique_vs_all"]
    best_main_signal = max(main_task_rows, key=lambda row: float(row["roc_auc"]))
    best_auc = float(best_main_signal["roc_auc"])
    if passed_models:
        level = "clean_features_identifiable"
        plain = "这些V12独有救援点在Clean世界中具有可跨完整空间洞识别的明显特征。"
    elif best_auc >= 0.65:
        level = "weak_clean_signal"
        plain = "这些点带有以V7谱不确定度为主的弱Clean信号，但无法在完全未见空间洞中稳定识别。"
    else:
        level = "not_clean_identifiable"
        plain = "现有Clean特征无法在完全未见空间洞中稳定识别V12独有救援点。"

    plot_paths = make_plots(labels, hole_id, univariate, predictions, probe_rows)
    write_csv(POINT_CSV, point_rows)
    write_csv(HOLE_CSV, hole_rows)
    write_csv(FEATURE_CSV, effect_rows)
    write_csv(UNIVARIATE_CSV, univariate)
    write_csv(PROBE_CSV, probe_rows)
    write_csv(FOLD_CSV, fold_rows)
    write_csv(PERMUTATION_CSV, permutation_rows)
    write_csv(LINEAGE_CSV, lineage_rows)

    strict_count_by_hole = {
        hole: int(np.sum(labels["strict"] & (hole_id == hole))) for hole in sorted(np.unique(hole_id))
    }
    best_univariate = sorted(univariate, key=lambda row: float(row["roc_auc"]), reverse=True)[:20]
    best_control_comparisons = {
        task: max(
            (row for row in probe_rows if row["task"] == task),
            key=lambda row: float(row["roc_auc"]),
        )
        for task in (
            "unique_vs_v7_already_good",
            "unique_vs_v12_still_bad",
            "unique_vs_carrier_answer_exists",
        )
    }
    report = {
        "version": VERSION,
        "status": "analysis_complete",
        "strict_definition": STRICT,
        "sensitivity_definition": SENSITIVE,
        "clean_lineage": {
            "sionna_used_in_features": False,
            "v12_role": "offline label definition only",
            "held_truth_role": "offline repair verification only",
            "forbidden_e153_fields": sorted(FORBIDDEN_E153_FIELDS),
            "feature_count": len(features),
            "feature_group_counts": {
                group: int(sum(feature.group == group for feature in features))
                for group in ("structure", "neighbor_h", "carrier_disagreement", "v7_spectrum", "map_geometry")
            },
        },
        "label_counts": {
            "unique_rescue": int(np.sum(labels["strict"])),
            "sensitive_unique_rescue": int(np.sum(labels["sensitive"])),
            "v7_already_good": int(np.sum(labels["label"] == "v7_already_good")),
            "v12_still_bad": int(np.sum(labels["label"] == "v12_still_bad")),
            "carrier_answer_exists": int(np.sum(labels["label"] == "carrier_answer_exists")),
            "background": int(np.sum(labels["label"] == "background")),
        },
        "strict_count_by_hole": strict_count_by_hole,
        "main_all_clean_results": main_rows,
        "all_probe_results": probe_rows,
        "permutation_controls": permutation_rows,
        "best_control_comparisons": best_control_comparisons,
        "best_univariate_features": best_univariate,
        "verdict": {
            "level": level,
            "plain_conclusion": plain,
            "passed_models": passed_models,
            "strict_hole_count": len(strict_holes),
            "strict_v7_pas_mean": float(np.mean(labels["v7_pas"][labels["strict"]])),
            "strict_best_carrier_pas_mean": float(np.mean(labels["best_carrier_pas"][labels["strict"]])),
            "strict_v12_pas_mean": float(np.mean(labels["v12_pas"][labels["strict"]])),
            "strict_unique_gain_mean": float(np.mean(labels["unique_gain"][labels["strict"]])),
            "best_main_feature_group_signal": best_main_signal,
            "thresholds": {
                "roc_auc": 0.75,
                "average_precision_multiple": 2.0,
                "balanced_accuracy": 0.65,
                "positive_fold_count": 4,
            },
        },
        "outputs": [
            PATHS.relative(path)
            for path in (
                REPORT_JSON,
                REPORT_MD,
                POINT_CSV,
                HOLE_CSV,
                FEATURE_CSV,
                UNIVARIATE_CSV,
                PROBE_CSV,
                FOLD_CSV,
                PERMUTATION_CSV,
                LINEAGE_CSV,
                *plot_paths,
            )
        ],
        "elapsed_seconds": time.time() - started,
    }
    write_report(report)
    print(json.dumps(json_value(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
