#!/usr/bin/env python3
"""Audit E180 action utility, regret, and Clean cross-hole selectability."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import resource
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import analyze_e174_v12_unique_rescue_clean_features as e174
import analyze_e180_clean_point_action_oracle as e180
from common.official_metrics import PointMetricStatistics, aggregate_statistics
from project_paths import PATHS


VERSION = "Round2-E181-clean-action-utility"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e181_clean_action_utility"
REPORT_JSON = ANALYSIS_DIR / "clean_action_utility.json"
REPORT_MD = ANALYSIS_DIR / "13个Clean动作效用后悔值与可选择性.md"
POINT_WIDE_CSV = ANALYSIS_DIR / "fixed500_action_utility_wide.csv"
POINT_LONG_CSV = ANALYSIS_DIR / "fixed500_action_utility_long.csv"
ACTION_CSV = ANALYSIS_DIR / "per_action_utility_summary.csv"
ACTION_HOLE_CSV = ANALYSIS_DIR / "per_action_hole_summary.csv"
ORACLE_CSV = ANALYSIS_DIR / "topk_margin_sign_oracles.csv"
CRITICAL_CSV = ANALYSIS_DIR / "critical_point_cumulative_curve.csv"
MODEL_CSV = ANALYSIS_DIR / "gain_regression_oof_scores.csv"
MODEL_FOLD_CSV = ANALYSIS_DIR / "gain_regression_fold_scores.csv"
MODEL_HOLE_CSV = ANALYSIS_DIR / "gain_regression_hole_scores.csv"
MODEL_ACTION_CSV = ANALYSIS_DIR / "gain_regression_per_action.csv"
FEATURE_CSV = ANALYSIS_DIR / "clean_feature_gain_relations.csv"
SELECTION_CSV = ANALYSIS_DIR / "oof_model_action_selection.csv"
MARGIN_PLOT = ANALYSIS_DIR / "best_action_margin_distribution.png"
CRITICAL_PLOT = ANALYSIS_DIR / "critical_point_score_curve.png"
ACTION_HOLE_PLOT = ANALYSIS_DIR / "action_hole_gain_heatmap.png"

E177_POINT_CSV = (
    PATHS.analysis
    / "results"
    / "round2_e177_v7_failure_reliability"
    / "fixed500_v7_failure_risk.csv"
)
E179_POINT_CSV = (
    PATHS.analysis
    / "results"
    / "round2_e179_v7_regime_oracle"
    / "fixed500_regime_oracle_points.csv"
)

N_POINT = 500
TINY = 1e-30
TIE_TOLERANCE = 1e-12
SEED = 20_260_811_181
EXPECTED_V7_SCORE = 0.6444563457343335
EXPECTED_E180_SCORE = 0.6772011759697127
TARGET_SCORE = 0.6775208075
SUCCESS_SCORE = 0.660000
NEAR_TIE_THRESHOLDS = (0.001, 0.005, 0.010, 0.020)
LARGE_MARGIN_THRESHOLDS = (0.010, 0.020, 0.050)
HARD_HOLES = ("P_S1F1", "P_S1F2", "P_S1F3", "P_S1F4")


@dataclass(frozen=True)
class CleanFeature:
    name: str
    group: str
    values: np.ndarray
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=384)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def read_indexed_csv(path: Path, index_key: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed: list[dict[str, str] | None] = [None] * N_POINT
    for row in rows:
        index = int(row[index_key])
        if not 0 <= index < N_POINT or indexed[index] is not None:
            raise RuntimeError(f"invalid or duplicate {index_key}={index} in {path}")
        indexed[index] = row
    if any(row is None for row in indexed):
        raise RuntimeError(f"{path} does not cover fixed500")
    return [row for row in indexed if row is not None]


def selected_statistics(
    selection: np.ndarray,
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
) -> PointMetricStatistics:
    rows = np.arange(len(selection), dtype=np.int64)
    return PointMetricStatistics(
        pas=np.asarray(pas[selection, rows], dtype=np.float64),
        pdp=np.asarray(pdp[selection, rows], dtype=np.float64),
        squared_error=np.asarray(error[selection, rows], dtype=np.float64),
        truth_energy=np.asarray(energy, dtype=np.float64),
    )


def mixed_statistics(
    weights: np.ndarray,
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
) -> PointMetricStatistics:
    if weights.shape != pas.shape:
        raise ValueError(f"weight shape mismatch: {weights.shape} vs {pas.shape}")
    return PointMetricStatistics(
        pas=np.sum(weights * pas, axis=0, dtype=np.float64),
        pdp=np.sum(weights * pdp, axis=0, dtype=np.float64),
        squared_error=np.sum(weights * error, axis=0, dtype=np.float64),
        truth_energy=np.asarray(energy, dtype=np.float64),
    )


def result_row(
    name: str,
    values: PointMetricStatistics,
    baseline_score: float,
    category: str,
    **extra: object,
) -> dict[str, object]:
    result = aggregate_statistics(values)
    return {
        "category": category,
        "name": name,
        **result,
        "delta_vs_v7": float(result["score"]) - baseline_score,
        "retained_e180_gain": (
            (float(result["score"]) - baseline_score)
            / max(EXPECTED_E180_SCORE - baseline_score, TINY)
        ),
        **extra,
    }


def load_actions() -> tuple[
    list[e180.ActionSpec],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    metadata = np.load(e180.OOF_META, allow_pickle=False)
    query_ids = np.asarray(metadata["query_ids"], dtype=np.int64)
    fold_id = np.asarray(metadata["fold_id"], dtype=np.int8)
    e168 = np.load(e180.E168_METRICS, allow_pickle=False)
    site = np.asarray(e168["site"], dtype=np.int8)
    if not np.array_equal(query_ids, np.asarray(e168["query_ids"], dtype=np.int64)):
        raise RuntimeError("canonical fixed500 order mismatch")

    actions = e180.build_actions(fold_id, site)
    primary = [action for action in actions if action.primary]
    action_ids = [action.action_id for action in primary]
    expected = [
        "V7",
        "V4",
        "E121_path",
        "E122_path",
        "E142_raw",
        "E142_projected",
        "E161_knn",
        "E162_knn",
        "carrier_00_nearest",
        "carrier_07_raw_idw_k24",
        "carrier_08_radial_idw_k24",
        "carrier_11_radial_affine_k24_r0.1",
        "carrier_16_affine_k24_steer_axis90_v8pi",
    ]
    if action_ids != expected:
        raise RuntimeError(f"E180 compact action identity changed: {action_ids}")

    cached = e180.load_cached_metrics(actions)
    if set(action_ids) - set(cached):
        raise RuntimeError(f"missing primary metrics: {sorted(set(action_ids) - set(cached))}")
    pas = np.stack([cached[action_id].pas for action_id in action_ids])
    pdp = np.stack([cached[action_id].pdp for action_id in action_ids])
    error = np.stack([cached[action_id].squared_error for action_id in action_ids])
    energy = np.asarray(cached["V7"].truth_energy, dtype=np.float64)
    for action_id in action_ids:
        if not np.allclose(cached[action_id].truth_energy, energy, rtol=1e-6, atol=1e-8):
            raise RuntimeError(f"truth energy mismatch for {action_id}")
    point_nmse = error / np.maximum(energy[None], TINY)
    utility = 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + point_nmse)
    return primary, action_ids, query_ids, fold_id, site, pas, pdp, error, energy, utility


def load_clean_features(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> tuple[list[CleanFeature], np.ndarray]:
    # E174's builder has an explicit whitelist and rejects all V12/GT-derived fields.
    base, hole_id, _ = e174.build_features({"query_ids": query_ids})
    features = [
        CleanFeature(item.name, item.group, np.asarray(item.values, dtype=np.float64), item.source)
        for item in base
    ]
    e177_rows = read_indexed_csv(E177_POINT_CSV, "oof_row")
    e179_rows = read_indexed_csv(E179_POINT_CSV, "oof_row")
    for index, row in enumerate(e177_rows):
        if int(row["query_id"]) != int(query_ids[index]):
            raise RuntimeError("E177 query order mismatch")
        if int(row["broad_fold"]) != int(fold_id[index]):
            raise RuntimeError("E177 fold order mismatch")
    for index, row in enumerate(e179_rows):
        if int(row["query_id"]) != int(query_ids[index]):
            raise RuntimeError("E179 query order mismatch")

    features.extend(
        [
            CleanFeature(
                "e177_failure_risk",
                "reliability_risk",
                np.asarray([float(row["risk_extra_trees"]) for row in e177_rows]),
                "E177 strict whole-hole OOF Clean reliability probe",
            ),
            CleanFeature(
                "v7_map_risk",
                "reliability_risk",
                np.asarray([float(row["v7_physical_risk"]) for row in e179_rows]),
                "V7 fold-external official Map risk",
            ),
            CleanFeature(
                "site",
                "site",
                np.asarray(site, dtype=np.float64),
                "official position site partition",
            ),
        ]
    )
    names = [feature.name for feature in features]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate Clean feature name")
    for feature in features:
        if feature.values.shape != (N_POINT,):
            raise RuntimeError(f"bad Clean feature shape: {feature.name}")
        lowered = f"{feature.name} {feature.source}".lower()
        forbidden = ("truth", "sionna", "v12", "point_score", "gt_", "target", "gain")
        if any(token in lowered for token in forbidden):
            raise RuntimeError(f"forbidden Clean feature lineage: {feature.name}/{feature.source}")
    return features, np.asarray(hole_id, dtype="U16")


def rank_actions(utility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.empty_like(utility, dtype=np.int16)
    rank = np.empty_like(utility, dtype=np.int16)
    for point in range(N_POINT):
        remaining = list(range(len(utility)))
        ranked: list[int] = []
        while remaining:
            maximum = max(float(utility[action, point]) for action in remaining)
            tied = [
                action
                for action in remaining
                if float(utility[action, point]) >= maximum - TIE_TOLERANCE
            ]
            ranked.extend(tied)
            tied_set = set(tied)
            remaining = [action for action in remaining if action not in tied_set]
        order[:, point] = np.asarray(ranked, dtype=np.int16)
        rank[order[:, point], point] = np.arange(1, len(order) + 1, dtype=np.int16)
    return order, rank


def utility_tables(
    action_ids: list[str],
    actions: list[e180.ActionSpec],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    hole_id: np.ndarray,
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    utility: np.ndarray,
    order: np.ndarray,
    rank: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]:
    best_utility = utility[order[0], np.arange(N_POINT)]
    gain = utility - utility[0]
    regret = np.maximum(best_utility[None] - utility, 0.0)
    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    action_by_id = {action.action_id: action for action in actions}
    for point in range(N_POINT):
        best = int(order[0, point])
        second = int(order[1, point])
        third = int(order[2, point])
        near_counts = {
            f"actions_within_{threshold:.3f}": int(
                np.sum(regret[:, point] <= threshold + TIE_TOLERANCE)
            )
            for threshold in NEAR_TIE_THRESHOLDS
        }
        row: dict[str, object] = {
            "oof_row": point,
            "query_id": int(query_ids[point]),
            "broad_fold": int(fold_id[point]),
            "site": int(site[point]),
            "hole_id": str(hole_id[point]),
            "best_action": action_ids[best],
            "second_action": action_ids[second],
            "third_action": action_ids[third],
            "best_utility": float(utility[best, point]),
            "best_vs_second_margin": float(max(utility[best, point] - utility[second, point], 0.0)),
            "best_vs_third_margin": float(max(utility[best, point] - utility[third, point], 0.0)),
            "best_vs_v7_gain": float(gain[best, point]),
            **near_counts,
        }
        for index, action_id in enumerate(action_ids):
            row[f"utility__{action_id}"] = float(utility[index, point])
            row[f"gain__{action_id}"] = float(gain[index, point])
            row[f"regret__{action_id}"] = float(regret[index, point])
            row[f"rank__{action_id}"] = int(rank[index, point])
            point_nmse = float(error[index, point] / max(energy[point], TINY))
            long_rows.append(
                {
                    "oof_row": point,
                    "query_id": int(query_ids[point]),
                    "broad_fold": int(fold_id[point]),
                    "site": int(site[point]),
                    "hole_id": str(hole_id[point]),
                    "action_index": index,
                    "action_id": action_id,
                    "family": action_by_id[action_id].family,
                    "pas": float(pas[index, point]),
                    "pdp": float(pdp[index, point]),
                    "point_nmse": point_nmse,
                    "utility": float(utility[index, point]),
                    "gain_vs_v7": float(gain[index, point]),
                    "regret_vs_best": float(regret[index, point]),
                    "rank": int(rank[index, point]),
                    "positive_gain": bool(gain[index, point] > TIE_TOLERANCE),
                    "is_best": bool(index == best),
                }
            )
        wide_rows.append(row)
    return wide_rows, long_rows, gain, regret


def action_summaries(
    actions: list[e180.ActionSpec],
    action_ids: list[str],
    hole_id: np.ndarray,
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    gain: np.ndarray,
    regret: np.ndarray,
    rank: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline_score = float(aggregate_statistics(selected_statistics(np.zeros(N_POINT, dtype=np.int16), pas, pdp, error, energy))["score"])
    action_rows: list[dict[str, object]] = []
    hole_rows: list[dict[str, object]] = []
    holes = sorted(np.unique(hole_id).tolist())
    for index, action in enumerate(actions):
        always = selected_statistics(np.full(N_POINT, index, dtype=np.int16), pas, pdp, error, energy)
        result = aggregate_statistics(always)
        positive = gain[index] > TIE_TOLERANCE
        nonbest = rank[index] != 1
        best_counts = []
        for hole in holes:
            mask = hole_id == hole
            best_count = int(np.sum((rank[index] == 1) & mask))
            best_counts.append((best_count / max(int(np.sum(mask)), 1), hole))
            hole_rows.append(
                {
                    "action_id": action.action_id,
                    "family": action.family,
                    "hole_id": hole,
                    "count": int(np.sum(mask)),
                    "best_count": best_count,
                    "best_fraction": best_count / max(int(np.sum(mask)), 1),
                    "mean_gain_vs_v7": float(np.mean(gain[index, mask])),
                    "positive_gain_fraction": float(np.mean(positive[mask])),
                    "mean_regret": float(np.mean(regret[index, mask])),
                }
            )
        preferred = sorted(best_counts, reverse=True)[:3]
        action_rows.append(
            {
                "action_index": index,
                "action_id": action.action_id,
                "family": action.family,
                "best_count": int(np.sum(rank[index] == 1)),
                "best_fraction": float(np.mean(rank[index] == 1)),
                "mean_gain_vs_v7": float(np.mean(gain[index])),
                "positive_gain_count": int(np.sum(positive)),
                "positive_gain_fraction": float(np.mean(positive)),
                "mean_positive_gain": float(np.mean(gain[index, positive])) if np.any(positive) else 0.0,
                "mean_gain_when_best": float(np.mean(gain[index, rank[index] == 1])),
                "mean_wrong_selection_regret": float(np.mean(regret[index, nonbest])) if np.any(nonbest) else 0.0,
                "p90_wrong_selection_regret": float(np.quantile(regret[index, nonbest], 0.90)) if np.any(nonbest) else 0.0,
                "always_action_pas": result["pas"],
                "always_action_pdp": result["pdp"],
                "always_action_nmse": result["nmse"],
                "always_action_score": result["score"],
                "always_action_delta_vs_v7": float(result["score"]) - baseline_score,
                "preferred_holes": ";".join(f"{hole}:{fraction:.3f}" for fraction, hole in preferred),
            }
        )
    return action_rows, hole_rows


def oracle_diagnostics(
    action_ids: list[str],
    fold_id: np.ndarray,
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    utility: np.ndarray,
    gain: np.ndarray,
    order: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    baseline_values = selected_statistics(np.zeros(N_POINT, dtype=np.int16), pas, pdp, error, energy)
    baseline_score = float(aggregate_statistics(baseline_values)["score"])
    best_selection = np.asarray(order[0], dtype=np.int16)
    best_values = selected_statistics(best_selection, pas, pdp, error, energy)
    best_score = float(aggregate_statistics(best_values)["score"])
    if abs(baseline_score - EXPECTED_V7_SCORE) > 5e-7:
        raise RuntimeError(f"V7 identity failed: {baseline_score}")
    if abs(best_score - EXPECTED_E180_SCORE) > 5e-7:
        raise RuntimeError(f"E180 identity failed: {best_score}")

    rows: list[dict[str, object]] = [
        result_row("V7", baseline_values, baseline_score, "reference"),
        result_row("exact_best_action", best_values, baseline_score, "oracle"),
    ]
    for rank_index, label in ((1, "rank_2_floor"), (2, "rank_3_floor")):
        selection = np.asarray(order[rank_index], dtype=np.int16)
        rows.append(
            result_row(
                label,
                selected_statistics(selection, pas, pdp, error, energy),
                baseline_score,
                "topk_floor",
            )
        )
    for count in (2, 3):
        weights = np.zeros_like(pas, dtype=np.float64)
        point = np.arange(N_POINT)
        for rank_index in range(count):
            weights[order[rank_index], point] += 1.0 / count
        rows.append(
            result_row(
                f"top_{count}_uniform_metric_expectation",
                mixed_statistics(weights, pas, pdp, error, energy),
                baseline_score,
                "topk_uniform",
                note="Expected sufficient statistics under uniform choice inside the GT Top-k set",
            )
        )

    best = order[0]
    point = np.arange(N_POINT)
    best_gain = gain[best, point]
    best_second = np.maximum(utility[best, point] - utility[order[1], point], 0.0)
    for threshold in LARGE_MARGIN_THRESHOLDS:
        mask = (best_gain > TIE_TOLERANCE) & (best_second >= threshold)
        selection = np.zeros(N_POINT, dtype=np.int16)
        selection[mask] = best[mask]
        rows.append(
            result_row(
                f"large_margin_best_second_ge_{threshold:.3f}",
                selected_statistics(selection, pas, pdp, error, energy),
                baseline_score,
                "large_margin_oracle",
                switched_points=int(np.sum(mask)),
            )
        )

    # Binary sign oracle: V7 versus one action.
    for index, action_id in enumerate(action_ids):
        selection = np.zeros(N_POINT, dtype=np.int16)
        if index != 0:
            selection[gain[index] > TIE_TOLERANCE] = index
        rows.append(
            result_row(
                f"binary_sign__{action_id}",
                selected_statistics(selection, pas, pdp, error, energy),
                baseline_score,
                "binary_sign_oracle",
                switched_points=int(np.sum(selection != 0)),
            )
        )

    # Combined sign-only oracle: held sign is known, held gain magnitude is not.
    combined = np.zeros(N_POINT, dtype=np.int16)
    priorities: dict[str, list[str]] = {}
    for outer in range(5):
        train = fold_id != outer
        held = fold_id == outer
        mean_positive = np.full(len(action_ids), -np.inf, dtype=np.float64)
        for action in range(1, len(action_ids)):
            positive = train & (gain[action] > TIE_TOLERANCE)
            if np.any(positive):
                mean_positive[action] = float(np.mean(gain[action, positive]))
        priority = np.argsort(-mean_positive, kind="stable")
        priorities[str(outer)] = [action_ids[index] for index in priority if index != 0]
        for local_point in np.flatnonzero(held):
            for action in priority:
                if action != 0 and gain[action, local_point] > TIE_TOLERANCE:
                    combined[local_point] = int(action)
                    break
    rows.append(
        result_row(
            "combined_sign_only_fold_external_priority",
            selected_statistics(combined, pas, pdp, error, energy),
            baseline_score,
            "combined_sign_oracle",
            switched_points=int(np.sum(combined != 0)),
        )
    )

    critical_rows: list[dict[str, object]] = []
    orders = {
        "best_v7_gain": np.argsort(-best_gain, kind="stable"),
        "best_second_margin": np.argsort(-best_second, kind="stable"),
    }
    thresholds: dict[str, dict[str, int | float]] = {}
    for ordering_name, point_order in orders.items():
        reached: dict[str, int | float] = {}
        selection = np.zeros(N_POINT, dtype=np.int16)
        for count, selected_point in enumerate(point_order, start=1):
            if best_gain[selected_point] > TIE_TOLERANCE:
                selection[selected_point] = best[selected_point]
            result = aggregate_statistics(selected_statistics(selection, pas, pdp, error, energy))
            retained = (float(result["score"]) - baseline_score) / max(best_score - baseline_score, TINY)
            critical_rows.append(
                {
                    "ordering": ordering_name,
                    "prefix_count": count,
                    "new_point": int(selected_point),
                    "new_point_best_gain": float(best_gain[selected_point]),
                    "new_point_best_second_margin": float(best_second[selected_point]),
                    "switched_count": int(np.sum(selection != 0)),
                    **result,
                    "delta_vs_v7": float(result["score"]) - baseline_score,
                    "retained_e180_gain": retained,
                }
            )
            for fraction in (0.50, 0.80, 0.90):
                key = f"{fraction:.2f}"
                if key not in reached and retained >= fraction:
                    reached[key] = count
        thresholds[ordering_name] = reached

    diagnostics = {
        "baseline_score": baseline_score,
        "exact_score": best_score,
        "exact_delta": best_score - baseline_score,
        "fold_external_sign_priorities": priorities,
        "critical_prefix_counts": thresholds,
    }
    return rows, critical_rows, diagnostics


def feature_matrix(features: list[CleanFeature], groups: set[str] | None = None) -> tuple[np.ndarray, list[str]]:
    selected = [feature for feature in features if groups is None or feature.group in groups]
    return np.column_stack([feature.values for feature in selected]).astype(np.float64), [feature.name for feature in selected]


def expanded_rows(context: np.ndarray, action_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_index = np.tile(np.arange(N_POINT, dtype=np.int64), action_count)
    action_index = np.repeat(np.arange(action_count, dtype=np.int16), N_POINT)
    repeated = context[point_index]
    one_hot = np.eye(action_count, dtype=np.float64)[action_index]
    return repeated, one_hot, point_index


def ridge_design_fit_predict(
    context: np.ndarray,
    gain: np.ndarray,
    fold_id: np.ndarray,
    outer: int,
) -> tuple[np.ndarray, np.ndarray]:
    action_count = gain.shape[0]
    train_point = fold_id != outer
    held_point = fold_id == outer
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = StandardScaler()
    train_context = scaler.fit_transform(imputer.fit_transform(context[train_point]))
    held_context = scaler.transform(imputer.transform(context[held_point]))
    train_actions = np.repeat(np.arange(action_count), int(np.sum(train_point)))
    held_actions = np.repeat(np.arange(action_count), int(np.sum(held_point)))
    train_repeated = np.tile(train_context, (action_count, 1))
    held_repeated = np.tile(held_context, (action_count, 1))
    train_one_hot = np.eye(action_count)[train_actions]
    held_one_hot = np.eye(action_count)[held_actions]
    train_design = np.concatenate(
        [train_one_hot, (train_repeated[:, None, :] * train_one_hot[:, :, None]).reshape(len(train_actions), -1)],
        axis=1,
    )
    held_design = np.concatenate(
        [held_one_hot, (held_repeated[:, None, :] * held_one_hot[:, :, None]).reshape(len(held_actions), -1)],
        axis=1,
    )
    target = gain[:, train_point].reshape(-1)
    model = Ridge(alpha=10.0, fit_intercept=True)
    model.fit(train_design, target)
    prediction = model.predict(held_design).reshape(action_count, int(np.sum(held_point)))
    prediction[0] = 0.0
    coefficients = np.abs(model.coef_[action_count:]).reshape(action_count, -1).mean(axis=0)
    return prediction, coefficients


def tree_fit_predict(
    context: np.ndarray,
    gain: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    outer: int,
    trees: int,
    seed: int,
    permute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    action_count = gain.shape[0]
    train_point = fold_id != outer
    held_point = fold_id == outer
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_context = imputer.fit_transform(context[train_point])
    held_context = imputer.transform(context[held_point])
    train_actions = np.repeat(np.arange(action_count), int(np.sum(train_point)))
    held_actions = np.repeat(np.arange(action_count), int(np.sum(held_point)))
    train_design = np.concatenate(
        [np.tile(train_context, (action_count, 1)), np.eye(action_count)[train_actions]], axis=1
    )
    held_design = np.concatenate(
        [np.tile(held_context, (action_count, 1)), np.eye(action_count)[held_actions]], axis=1
    )
    target_matrix = np.array(gain[:, train_point], copy=True)
    if permute:
        rng = np.random.default_rng(seed + outer * 1009)
        train_site = site[train_point]
        for action in range(action_count):
            for site_value in (0, 1):
                rows = np.flatnonzero(train_site == site_value)
                target_matrix[action, rows] = target_matrix[action, rows][rng.permutation(len(rows))]
    target = target_matrix.reshape(-1)
    model = ExtraTreesRegressor(
        n_estimators=trees,
        max_depth=8,
        min_samples_leaf=12,
        max_features="sqrt",
        n_jobs=-1,
        random_state=(seed + outer * 1009) % (2**32),
    )
    model.fit(train_design, target)
    prediction = model.predict(held_design).reshape(action_count, int(np.sum(held_point)))
    prediction[0] = 0.0
    context_importance = model.feature_importances_[: train_context.shape[1]]
    return prediction, context_importance


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    exclude_first_action_from_sign: bool = True,
) -> dict[str, float]:
    truth_flat = truth.reshape(-1)
    prediction_flat = prediction.reshape(-1)
    sign_rows = np.ones_like(truth, dtype=bool)
    if exclude_first_action_from_sign:
        sign_rows[0] = False
    y = truth[sign_rows]
    p = prediction[sign_rows]
    sign = y > TIE_TOLERANCE
    pearson = (
        float(np.corrcoef(truth_flat, prediction_flat)[0, 1])
        if float(np.std(truth_flat)) > 1e-15 and float(np.std(prediction_flat)) > 1e-15
        else float("nan")
    )
    spearman = (
        float(spearmanr(truth_flat, prediction_flat).statistic)
        if float(np.std(truth_flat)) > 1e-15 and float(np.std(prediction_flat)) > 1e-15
        else float("nan")
    )
    return {
        "rmse": math.sqrt(mean_squared_error(truth_flat, prediction_flat)),
        "mae": mean_absolute_error(truth_flat, prediction_flat),
        "r2": r2_score(truth_flat, prediction_flat),
        "pearson": pearson,
        "spearman": spearman,
        "sign_auc": float(roc_auc_score(sign, p)) if len(np.unique(sign)) == 2 else float("nan"),
        "sign_ap": float(average_precision_score(sign, p)) if np.any(sign) else float("nan"),
        "sign_accuracy": float(np.mean((p > 0.0) == sign)),
        "positive_prevalence": float(np.mean(sign)),
    }


def evaluate_models(
    features: list[CleanFeature],
    action_ids: list[str],
    fold_id: np.ndarray,
    site: np.ndarray,
    hole_id: np.ndarray,
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    gain: np.ndarray,
    oracle_order: np.ndarray,
    trees: int,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, np.ndarray],
    list[dict[str, object]],
]:
    baseline_values = selected_statistics(np.zeros(N_POINT, dtype=np.int16), pas, pdp, error, energy)
    baseline_score = float(aggregate_statistics(baseline_values)["score"])
    groups = {
        "risk_only": {"reliability_risk", "site"},
        "structure": {"structure", "site"},
        "neighbor_h": {"neighbor_h", "site"},
        "v7_spectrum": {"v7_spectrum", "site"},
        "carrier_disagreement": {"carrier_disagreement", "site"},
        "map_geometry": {"map_geometry", "site"},
        "all_clean": None,
    }
    predictions: dict[str, np.ndarray] = {}
    importances: dict[str, tuple[np.ndarray, list[str], int]] = {}
    fold_rows: list[dict[str, object]] = []

    # Action-prior uses no point context and is a required baseline.
    prior = np.zeros_like(gain)
    for outer in range(5):
        train = fold_id != outer
        held = fold_id == outer
        prior[:, held] = np.mean(gain[:, train], axis=1)[:, None]
    prior[0] = 0.0
    predictions["action_prior"] = prior

    all_context, all_names = feature_matrix(features)
    for model_name in ("ridge_all_clean", "extra_trees_all_clean", "extra_trees_permuted"):
        output = np.full_like(gain, np.nan)
        importance_sum: np.ndarray | None = None
        importance_names: list[str] = all_names
        for outer in range(5):
            held = fold_id == outer
            if model_name == "ridge_all_clean":
                local, importance = ridge_design_fit_predict(all_context, gain, fold_id, outer)
            else:
                local, importance = tree_fit_predict(
                    all_context,
                    gain,
                    fold_id,
                    site,
                    outer,
                    trees,
                    seed + (100_000 if model_name == "extra_trees_permuted" else 0),
                    permute=model_name == "extra_trees_permuted",
                )
                # SimpleImputer may add missingness indicators. Keep only named raw features.
                importance = importance[: len(all_names)]
            output[:, held] = local
            fold_metric = regression_metrics(gain[:, held], local)
            selection = np.argmax(local, axis=0).astype(np.int16)
            selected = selected_statistics(selection, pas[:, held], pdp[:, held], error[:, held], energy[held])
            fold_rows.append(
                {
                    "model": model_name,
                    "outer_fold": outer,
                    "point_count": int(np.sum(held)),
                    **fold_metric,
                    **{f"official_{key}": value for key, value in aggregate_statistics(selected).items()},
                    "delta_vs_v7": float(aggregate_statistics(selected)["score"])
                    - float(aggregate_statistics(PointMetricStatistics(pas[0, held], pdp[0, held], error[0, held], energy[held]))["score"]),
                }
            )
            importance_sum = importance if importance_sum is None else importance_sum + importance
        if not np.all(np.isfinite(output)):
            raise RuntimeError(f"non-finite OOF prediction: {model_name}")
        predictions[model_name] = output
        importances[model_name] = (importance_sum / 5.0, importance_names, len(importance_names))

    # Fixed group probes use ExtraTrees only; no result-dependent model selection.
    for group_name, selected_groups in groups.items():
        if group_name == "all_clean":
            continue
        context, names = feature_matrix(features, selected_groups)
        output = np.full_like(gain, np.nan)
        importance_sum: np.ndarray | None = None
        for outer in range(5):
            held = fold_id == outer
            local, importance = tree_fit_predict(
                context, gain, fold_id, site, outer, trees, seed + 20_000, permute=False
            )
            output[:, held] = local
            importance = importance[: len(names)]
            importance_sum = importance if importance_sum is None else importance_sum + importance
        predictions[f"extra_trees_{group_name}"] = output
        importances[f"extra_trees_{group_name}"] = (importance_sum / 5.0, names, len(names))

    model_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    hole_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for model_name, prediction in predictions.items():
        metrics = regression_metrics(gain, prediction)
        selection = np.argmax(prediction, axis=0).astype(np.int16)
        values = selected_statistics(selection, pas, pdp, error, energy)
        result = aggregate_statistics(values)
        oracle_best = oracle_order[0]
        model_rows.append(
            {
                "model": model_name,
                **metrics,
                **result,
                "delta_vs_v7": float(result["score"]) - baseline_score,
                "gap_vs_0_66": float(result["score"]) - SUCCESS_SCORE,
                "gap_vs_e180": float(result["score"]) - EXPECTED_E180_SCORE,
                "gap_vs_current_target": float(result["score"]) - TARGET_SCORE,
                "exact_best_action_rate_diagnostic": float(np.mean(selection == oracle_best)),
                "top2_action_rate_diagnostic": float(
                    np.mean([selection[point] in oracle_order[:2, point] for point in range(N_POINT)])
                ),
                "top3_action_rate_diagnostic": float(
                    np.mean([selection[point] in oracle_order[:3, point] for point in range(N_POINT)])
                ),
                "changed_from_v7_count": int(np.sum(selection != 0)),
                "positive_fold_count": int(
                    sum(row["delta_vs_v7"] > 0.0 for row in fold_rows if row["model"] == model_name)
                ) if any(row["model"] == model_name for row in fold_rows) else None,
            }
        )
        for point in range(N_POINT):
            selection_rows.append(
                {
                    "model": model_name,
                    "oof_row": point,
                    "broad_fold": int(fold_id[point]),
                    "site": int(site[point]),
                    "hole_id": str(hole_id[point]),
                    "selected_action": action_ids[selection[point]],
                    "predicted_gain": float(prediction[selection[point], point]),
                    "true_gain": float(gain[selection[point], point]),
                    "true_selected_action_positive": bool(gain[selection[point], point] > TIE_TOLERANCE),
                }
            )
        for action, action_id in enumerate(action_ids):
            local = regression_metrics(
                gain[action : action + 1],
                prediction[action : action + 1],
                exclude_first_action_from_sign=False,
            )
            action_rows.append(
                {
                    "model": model_name,
                    "action_id": action_id,
                    **local,
                    "selected_count": int(np.sum(selection == action)),
                    "mean_predicted_gain": float(np.mean(prediction[action])),
                    "mean_true_gain": float(np.mean(gain[action])),
                }
            )
        for hole in sorted(np.unique(hole_id).tolist()):
            mask = hole_id == hole
            selected = PointMetricStatistics(
                values.pas[mask], values.pdp[mask], values.squared_error[mask], values.truth_energy[mask]
            )
            baseline_hole = PointMetricStatistics(pas[0, mask], pdp[0, mask], error[0, mask], energy[mask])
            local_result = aggregate_statistics(selected)
            hole_rows.append(
                {
                    "model": model_name,
                    "hole_id": hole,
                    **local_result,
                    "delta_vs_v7": float(local_result["score"])
                    - float(aggregate_statistics(baseline_hole)["score"]),
                }
            )

    feature_rows: list[dict[str, object]] = []
    for model_name, (importance, names, _) in importances.items():
        for name, value in zip(names, importance):
            feature_rows.append(
                {
                    "analysis": "model_importance",
                    "model": model_name,
                    "feature": name,
                    "importance": float(value),
                }
            )
    for feature in features:
        for action, action_id in enumerate(action_ids[1:], start=1):
            valid = np.isfinite(feature.values)
            correlation = (
                spearmanr(feature.values[valid], gain[action, valid]).statistic
                if float(np.std(feature.values[valid])) > 1e-15
                and float(np.std(gain[action, valid])) > 1e-15
                else float("nan")
            )
            fold_signs = []
            for outer in range(5):
                mask = valid & (fold_id == outer)
                value = (
                    spearmanr(feature.values[mask], gain[action, mask]).statistic
                    if float(np.std(feature.values[mask])) > 1e-15
                    and float(np.std(gain[action, mask])) > 1e-15
                    else float("nan")
                )
                if math.isfinite(float(value)):
                    fold_signs.append(float(value))
            feature_rows.append(
                {
                    "analysis": "univariate_gain_relation",
                    "feature": feature.name,
                    "feature_group": feature.group,
                    "action_id": action_id,
                    "spearman": float(correlation),
                    "positive_folds": int(np.sum(np.asarray(fold_signs) > 0.0)),
                    "negative_folds": int(np.sum(np.asarray(fold_signs) < 0.0)),
                    "mean_abs_fold_spearman": float(np.mean(np.abs(fold_signs))) if fold_signs else float("nan"),
                }
            )
    return model_rows, fold_rows, hole_rows, action_rows, feature_rows, predictions, selection_rows


def make_plots(
    action_ids: list[str],
    hole_id: np.ndarray,
    gain: np.ndarray,
    utility: np.ndarray,
    order: np.ndarray,
    critical_rows: list[dict[str, object]],
    baseline_score: float,
) -> None:
    point = np.arange(N_POINT)
    best_second = np.maximum(utility[order[0], point] - utility[order[1], point], 0.0)
    best_third = np.maximum(utility[order[0], point] - utility[order[2], point], 0.0)
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    axis.hist(best_second, bins=40, alpha=0.75, label="best - second")
    axis.hist(best_third, bins=40, alpha=0.55, label="best - third")
    for threshold in NEAR_TIE_THRESHOLDS:
        axis.axvline(threshold, color="#111827", alpha=0.20, linewidth=1)
    axis.set_xlabel("Point utility margin")
    axis.set_ylabel("Fixed500 point count")
    axis.set_title("E181 best-action ambiguity")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(MARGIN_PLOT, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for ordering, color in (("best_v7_gain", "#2563eb"), ("best_second_margin", "#dc2626")):
        selected = [row for row in critical_rows if row["ordering"] == ordering]
        axis.plot(
            [int(row["prefix_count"]) for row in selected],
            [float(row["score"]) for row in selected],
            label=ordering,
            color=color,
        )
    axis.axhline(baseline_score, color="#111827", linestyle="--", label="V7")
    axis.axhline(EXPECTED_E180_SCORE, color="#059669", linestyle="--", label="E180 oracle")
    axis.set_xlabel("GT-ranked points allowed to switch")
    axis.set_ylabel("Complete official score")
    axis.set_title("E181 concentration of E180 oracle gain")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(CRITICAL_PLOT, dpi=180)
    plt.close(figure)

    holes = sorted(np.unique(hole_id).tolist())
    matrix = np.empty((len(action_ids), len(holes)), dtype=np.float64)
    for action in range(len(action_ids)):
        for hole_index, hole in enumerate(holes):
            matrix[action, hole_index] = float(np.mean(gain[action, hole_id == hole]))
    figure, axis = plt.subplots(figsize=(12.0, 7.0))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    axis.set_xticks(np.arange(len(holes)), holes, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(action_ids)), action_ids)
    axis.set_title("Mean point-utility gain versus V7")
    figure.colorbar(image, ax=axis, label="Mean gain")
    figure.tight_layout()
    figure.savefig(ACTION_HOLE_PLOT, dpi=180)
    plt.close(figure)


def render_report(report: dict[str, object]) -> str:
    ambiguity = report["ambiguity"]
    critical = report["critical_points"]
    oracle = report["oracles"]
    models = report["models"]
    displayed_models = {
        "action_prior",
        "ridge_all_clean",
        "extra_trees_all_clean",
        "extra_trees_permuted",
    }
    model_lines = "\n".join(
        f"| {row['model']} | {row['score']:.6f} | {row['delta_vs_v7']:+.6f} | "
        f"{row['rmse']:.5f} | {row['sign_auc']:.3f} | {row['positive_fold_count'] if row['positive_fold_count'] is not None else '-'} |"
        for row in models
        if row["model"] in displayed_models
    )
    oracle_names = {
        "rank_2_floor",
        "rank_3_floor",
        "top_2_uniform_metric_expectation",
        "top_3_uniform_metric_expectation",
        "combined_sign_only_fold_external_priority",
        "large_margin_best_second_ge_0.010",
        "large_margin_best_second_ge_0.020",
        "large_margin_best_second_ge_0.050",
    }
    oracle_lines = "\n".join(
        f"| {row['name']} | {row['score']:.6f} | {row['delta_vs_v7']:+.6f} | {row['retained_e180_gain']:.1%} |"
        for row in oracle
        if row["name"] in oracle_names
    )
    conclusion = report["verdict"]
    return f"""# 13个Clean动作效用、后悔值与可选择性

## 最关键结论

E180的精确最佳动作oracle仍为`{report['identity']['e180_score']:.9f}`，相对V7提高`{report['identity']['e180_delta']:+.9f}`。本轮结论是：**{conclusion['plain_language']}**

这是一轮`diagnostic_only`整洞OOF分析，不是可提交模型，也没有生成测试H或版本。

## 动作是否近似并列

- `best-second`间隔中位数=`{ambiguity['best_second_median']:.6f}`，p90=`{ambiguity['best_second_p90']:.6f}`。
- `best-third`间隔中位数=`{ambiguity['best_third_median']:.6f}`。
- 距最佳效用`0.005`内平均有`{ambiguity['mean_actions_within_0_005']:.2f}`个动作，至少2个动作近似并列的点有`{ambiguity['points_with_top2_within_0_005']}/500`。
- 按`best-V7`收益排序，达到E180增益50%/80%/90%分别需要`{critical['gain_order_50']}/{critical['gain_order_80']}/{critical['gain_order_90']}`个点。

## Top-k与sign oracle

| 诊断 | 官方总分 | 相对V7 | 保留E180增益 |
| --- | ---: | ---: | ---: |
{oracle_lines}

## 跨洞收益预测

| 模型/对照 | 官方总分 | 相对V7 | 收益RMSE | sign AUC | 正向fold数 |
| --- | ---: | ---: | ---: | ---: | ---: |
{model_lines}

模型目标不是13类accuracy，而是`Clean context + action -> gain versus V7`。所有held预测都同时隔离同一ordinal下两个站点的完整pseudo-hole；V7已知收益固定为0。

## 图

![最佳动作间隔](best_action_margin_distribution.png)

![关键点累计总分](critical_point_score_curve.png)

![动作逐洞平均收益](action_hole_gain_heatmap.png)

## 边界

输入只包含E177 OOF风险、V7 map risk、洞结构、邻居官方Train H分歧、V7自身谱状态、17个Clean carrier分歧、site和官方Map几何。没有使用Sionna、V8-V12修正、held GT特征、坐标/hole ID或测试答案。GT只用于效用标签和本轮oracle评价。
"""


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    actions, action_ids, query_ids, fold_id, site, pas, pdp, error, energy, utility = load_actions()
    features, hole_id = load_clean_features(query_ids, fold_id, site)
    order, rank = rank_actions(utility)
    wide_rows, long_rows, gain, regret = utility_tables(
        action_ids,
        actions,
        query_ids,
        fold_id,
        site,
        hole_id,
        pas,
        pdp,
        error,
        energy,
        utility,
        order,
        rank,
    )
    action_rows, action_hole_rows = action_summaries(
        actions, action_ids, hole_id, pas, pdp, error, energy, gain, regret, rank
    )
    oracle_rows, critical_rows, oracle_diagnostics_values = oracle_diagnostics(
        action_ids, fold_id, pas, pdp, error, energy, utility, gain, order
    )
    (
        model_rows,
        fold_rows,
        model_hole_rows,
        model_action_rows,
        feature_rows,
        predictions,
        selection_rows,
    ) = evaluate_models(
        features,
        action_ids,
        fold_id,
        site,
        hole_id,
        pas,
        pdp,
        error,
        energy,
        gain,
        order,
        args.trees,
        args.seed,
    )

    baseline_score = float(oracle_diagnostics_values["baseline_score"])
    exact_score = float(oracle_diagnostics_values["exact_score"])
    point = np.arange(N_POINT)
    best_second = np.maximum(utility[order[0], point] - utility[order[1], point], 0.0)
    best_third = np.maximum(utility[order[0], point] - utility[order[2], point], 0.0)
    near_counts = {
        threshold: np.sum(regret <= threshold + TIE_TOLERANCE, axis=0)
        for threshold in NEAR_TIE_THRESHOLDS
    }
    oracle_by_name = {str(row["name"]): row for row in oracle_rows}
    rank2_retained = float(oracle_by_name["rank_2_floor"]["retained_e180_gain"])
    top2_retained = float(oracle_by_name["top_2_uniform_metric_expectation"]["retained_e180_gain"])
    critical80 = int(oracle_diagnostics_values["critical_prefix_counts"]["best_v7_gain"].get("0.80", N_POINT + 1))
    avoid_bad_sufficient = max(rank2_retained, top2_retained) >= 0.80 and critical80 <= 150

    principal_models = [
        row for row in model_rows if row["model"] in {"ridge_all_clean", "extra_trees_all_clean"}
    ]
    prior_score = next(float(row["score"]) for row in model_rows if row["model"] == "action_prior")
    permuted_score = next(float(row["score"]) for row in model_rows if row["model"] == "extra_trees_permuted")
    learned_pass = [
        row
        for row in principal_models
        if float(row["score"]) >= SUCCESS_SCORE
        and int(row["positive_fold_count"] or 0) >= 4
        and float(row["score"]) >= prior_score + 0.002
        and float(row["score"]) >= permuted_score + 0.002
    ]
    if avoid_bad_sufficient and learned_pass:
        plain = "多数收益不要求精确猜中唯一第一名，而且Clean收益回归已经能跨洞兑现到0.66以上"
        status = "diagnostic_pass_avoid_bad_and_gain_predictable"
    elif avoid_bad_sufficient:
        plain = "动作本身存在较大容错，但现有Clean上下文仍不能稳定预测该选哪个动作"
        status = "diagnostic_action_tolerant_but_gain_not_predictable"
    elif learned_pass:
        plain = "动作排序需要较精确，但固定Clean收益回归仍通过了跨洞0.66门"
        status = "diagnostic_precise_ranking_but_gain_predictable"
    else:
        sign_retained = float(
            oracle_by_name["combined_sign_only_fold_external_priority"]["retained_e180_gain"]
        )
        plain = (
            f"不要求500点全部精确猜中唯一第一名，但要在约{critical80}个关键点上可靠避开负动作；"
            f"任取Top-2只保留{top2_retained:.1%}增益，完美sign也只保留{sign_retained:.1%}，"
            "而现有Clean收益回归尚未学会这个选择"
        )
        status = "diagnostic_precise_ranking_and_gain_not_predictable"

    report: dict[str, object] = {
        "experiment": VERSION,
        "status": status,
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_correction_used": False,
        "held_gt_used_only_for_action_utility_labels_and_oracles": True,
        "identity": {
            "v7_score": baseline_score,
            "e180_score": exact_score,
            "e180_delta": exact_score - baseline_score,
            "expected_v7_score": EXPECTED_V7_SCORE,
            "expected_e180_score": EXPECTED_E180_SCORE,
        },
        "ambiguity": {
            "best_second_mean": float(np.mean(best_second)),
            "best_second_median": float(np.median(best_second)),
            "best_second_p90": float(np.quantile(best_second, 0.90)),
            "best_third_mean": float(np.mean(best_third)),
            "best_third_median": float(np.median(best_third)),
            "near_tie_thresholds": {
                f"{threshold:.3f}": {
                    "mean_actions": float(np.mean(near_counts[threshold])),
                    "points_with_at_least_2": int(np.sum(near_counts[threshold] >= 2)),
                    "points_with_at_least_3": int(np.sum(near_counts[threshold] >= 3)),
                }
                for threshold in NEAR_TIE_THRESHOLDS
            },
            "mean_actions_within_0_005": float(np.mean(near_counts[0.005])),
            "points_with_top2_within_0_005": int(np.sum(near_counts[0.005] >= 2)),
        },
        "critical_points": {
            "gain_order_50": int(oracle_diagnostics_values["critical_prefix_counts"]["best_v7_gain"].get("0.50", N_POINT + 1)),
            "gain_order_80": critical80,
            "gain_order_90": int(oracle_diagnostics_values["critical_prefix_counts"]["best_v7_gain"].get("0.90", N_POINT + 1)),
            "all_curves": oracle_diagnostics_values["critical_prefix_counts"],
        },
        "oracles": oracle_rows,
        "actions": action_rows,
        "models": model_rows,
        "verdict": {
            "avoid_bad_is_sufficient": avoid_bad_sufficient,
            "clean_gain_predictability_passed": bool(learned_pass),
            "passing_models": [row["model"] for row in learned_pass],
            "plain_language": plain,
        },
        "feature_count": len(features),
        "feature_group_counts": {
            group: int(sum(feature.group == group for feature in features))
            for group in sorted({feature.group for feature in features})
        },
        "performance": {
            "wall_seconds": time.perf_counter() - started,
            "device": "CPU",
            "gpu_used": False,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "measured_bottleneck": "fixed whole-hole ExtraTrees gain regressions and CSV output",
            "cached_point_metrics_reused": True,
        },
        "files": {
            "point_wide": PATHS.relative(POINT_WIDE_CSV),
            "point_long": PATHS.relative(POINT_LONG_CSV),
            "action_summary": PATHS.relative(ACTION_CSV),
            "action_hole": PATHS.relative(ACTION_HOLE_CSV),
            "oracles": PATHS.relative(ORACLE_CSV),
            "critical_curve": PATHS.relative(CRITICAL_CSV),
            "models": PATHS.relative(MODEL_CSV),
            "model_folds": PATHS.relative(MODEL_FOLD_CSV),
            "model_holes": PATHS.relative(MODEL_HOLE_CSV),
            "model_actions": PATHS.relative(MODEL_ACTION_CSV),
            "feature_relations": PATHS.relative(FEATURE_CSV),
            "selection": PATHS.relative(SELECTION_CSV),
        },
    }

    write_csv(POINT_WIDE_CSV, wide_rows)
    write_csv(POINT_LONG_CSV, long_rows)
    write_csv(ACTION_CSV, action_rows)
    write_csv(ACTION_HOLE_CSV, action_hole_rows)
    write_csv(ORACLE_CSV, oracle_rows)
    write_csv(CRITICAL_CSV, critical_rows)
    write_csv(MODEL_CSV, model_rows)
    write_csv(MODEL_FOLD_CSV, fold_rows)
    write_csv(MODEL_HOLE_CSV, model_hole_rows)
    write_csv(MODEL_ACTION_CSV, model_action_rows)
    write_csv(FEATURE_CSV, feature_rows)
    write_csv(SELECTION_CSV, selection_rows)
    make_plots(action_ids, hole_id, gain, utility, order, critical_rows, baseline_score)
    report["performance"]["wall_seconds"] = time.perf_counter() - started
    REPORT_JSON.write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(json_value(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
