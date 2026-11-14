#!/usr/bin/env python3
"""Explain every loss between true decomposed paths and V7 Top-15 coverage."""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e131_v7_path_selection_evidence as e131
import analyze_global_local_path_lifecycle as e79
import train_e115_complete_path_state_operator as e115
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import component_energy
from run_v6_regime_component_lifecycle import REGIME_CACHE


VERSION = "Round2-E132-v7-path-coverage-gap"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_path_coverage_gap"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_path_coverage_gap"
VISUAL_DIR = PATHS.analysis / "visualizations" / "复赛V7路径覆盖缺口"
REPORT = ANALYSIS_DIR / "v7_path_coverage_gap.json"
METRICS = ARTIFACT_DIR / "v7_path_coverage_gap_metrics.npz"
E131_METRICS = (
    PATHS.artifacts
    / "round2"
    / "v7_path_selection_evidence"
    / "v7_path_selection_evidence_metrics.npz"
)
V7_SUPPORT = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "oof_stacked_current_ad_f16.npy"
)

CATEGORY_NAMES = (
    "below_threshold",
    "no_near_candidate",
    "identity_collision",
    "top15_capacity",
    "v7_ranking",
    "captured",
)
CATEGORY_LABELS = (
    "阈值以下弱路径",
    "附近没有候选",
    "候选身份冲突",
    "Top-15容量损失",
    "V7排序损失",
    "成功覆盖",
)
CATEGORY_COLORS = (
    "#9ca3af",
    "#dc2626",
    "#7c3aed",
    "#f59e0b",
    "#2563eb",
    "#16a34a",
)
MISS_NAMES = ("neither", "base_only", "v7_only", "both")
MISS_LABELS = ("两者都未识别", "仅原先验识别", "仅V7识别", "两者均识别但融合落选")
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def detailed_assignment(
    candidate_coordinate: np.ndarray,
    candidate_mask: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    truth_power: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    count, candidate_count, _ = candidate_coordinate.shape
    truth_count = truth_coordinate.shape[1]
    truth_energy = component_energy(truth_coefficient)
    truth_fraction_power = np.divide(
        truth_energy,
        np.maximum(truth_power[:, None], TINY),
        out=np.zeros_like(truth_energy),
    )
    total_component_energy = truth_energy.sum(axis=1, dtype=np.float64)
    truth_fraction = np.divide(
        truth_energy,
        np.maximum(total_component_energy[:, None], TINY),
        out=np.zeros_like(truth_energy),
    )
    valid_point = total_component_energy > TINY
    valid_truth = truth_fraction_power >= e115.Config().minimum_fraction

    candidate_to_truth = np.full((count, candidate_count), -1, dtype=np.int16)
    truth_to_candidate = np.full((count, truth_count), -1, dtype=np.int16)
    nearest_candidate = np.full((count, truth_count), -1, dtype=np.int16)
    nearest_cost = np.full((count, truth_count), 1e6, dtype=np.float32)
    nearest_axis_delta = np.zeros((count, truth_count, 3), dtype=np.float32)
    target_mass_power = np.zeros((count, candidate_count), dtype=np.float32)
    target_mass_component = np.zeros((count, candidate_count), dtype=np.float32)
    matched_cost: list[np.ndarray] = []
    bandwidth = float(e115.Config().soft_bandwidth)
    limit = float(e115.Config().soft_cost_limit)

    for row in np.flatnonzero(valid_point):
        candidate_ids = np.flatnonzero(candidate_mask[row])
        truth_ids = np.flatnonzero(valid_truth[row])
        if len(candidate_ids) == 0 or len(truth_ids) == 0:
            continue
        candidate = candidate_coordinate[row, candidate_ids]
        truth = truth_coordinate[row, truth_ids]
        cost = e79.support_cost(candidate, truth)
        local_nearest = np.argmin(cost, axis=0)
        nearest_candidate[row, truth_ids] = candidate_ids[local_nearest]
        nearest_cost[row, truth_ids] = cost[local_nearest, np.arange(len(truth_ids))]
        for axis, period in enumerate(e115.PERIODS):
            nearest_axis_delta[row, truth_ids, axis] = np.abs(
                e79.periodic_delta(
                    candidate[local_nearest, axis],
                    truth[:, axis],
                    float(period),
                )
            )

        utility = truth_fraction_power[row, truth_ids][None] * np.exp(
            -0.5 * cost / bandwidth
        )
        utility[cost > limit] = 0.0
        candidate_local, truth_local = linear_sum_assignment(-utility)
        accepted = utility[candidate_local, truth_local] > 0.0
        candidate_local = candidate_local[accepted]
        truth_local = truth_local[accepted]
        if len(candidate_local) == 0:
            continue
        assigned_candidate = candidate_ids[candidate_local]
        assigned_truth = truth_ids[truth_local]
        candidate_to_truth[row, assigned_candidate] = assigned_truth
        truth_to_candidate[row, assigned_truth] = assigned_candidate
        target_mass_power[row, assigned_candidate] = truth_fraction_power[
            row, assigned_truth
        ]
        target_mass_component[row, assigned_candidate] = truth_fraction[
            row, assigned_truth
        ]
        matched_cost.append(cost[candidate_local, truth_local])

    matched_cost_array = (
        np.concatenate(matched_cost) if matched_cost else np.empty(0, np.float32)
    )
    return {
        "truth_energy": truth_energy,
        "truth_fraction": truth_fraction.astype(np.float32),
        "truth_fraction_power": truth_fraction_power.astype(np.float32),
        "total_component_energy": total_component_energy,
        "valid_point": valid_point,
        "valid_truth": valid_truth,
        "candidate_to_truth": candidate_to_truth,
        "truth_to_candidate": truth_to_candidate,
        "nearest_candidate": nearest_candidate,
        "nearest_cost": nearest_cost,
        "nearest_axis_delta": nearest_axis_delta,
        "target_mass_power": target_mass_power,
        "target_mass_component": target_mass_component,
        "matched_cost_mean": float(np.mean(matched_cost_array))
        if len(matched_cost_array)
        else 0.0,
        "matched_cost_p90": float(np.quantile(matched_cost_array, 0.90))
        if len(matched_cost_array)
        else 0.0,
    }


def sum_truth_fraction(
    truth_fraction: np.ndarray, selection: np.ndarray
) -> np.ndarray:
    return np.sum(truth_fraction * selection, axis=1, dtype=np.float64)


def selected_truth_mask(
    selected_candidate: np.ndarray,
    candidate_to_truth: np.ndarray,
    truth_count: int,
) -> np.ndarray:
    output = np.zeros((len(selected_candidate), truth_count), dtype=bool)
    for row in range(len(output)):
        candidates = np.flatnonzero(
            selected_candidate[row] & (candidate_to_truth[row] >= 0)
        )
        if len(candidates):
            output[row, candidate_to_truth[row, candidates]] = True
    return output


def top_k_truth_mask(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros_like(valid)
    for row in range(len(score)):
        ids = np.flatnonzero(valid[row])
        take = min(e131.TOP_K, len(ids))
        if take:
            local = np.argpartition(score[row, ids], -take)[-take:]
            output[row, ids[local]] = True
    return output


def build_waterfall(
    assignment: dict[str, np.ndarray | float | int],
    candidate_mask: np.ndarray,
    local_selected: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    truth_fraction = np.asarray(assignment["truth_fraction"], dtype=np.float64)
    valid_truth = np.asarray(assignment["valid_truth"], dtype=bool)
    truth_to_candidate = np.asarray(assignment["truth_to_candidate"], dtype=np.int16)
    candidate_to_truth = np.asarray(assignment["candidate_to_truth"], dtype=np.int16)
    nearest_cost = np.asarray(assignment["nearest_cost"], dtype=np.float32)
    limit = float(e115.Config().soft_cost_limit)

    matched_truth = truth_to_candidate >= 0
    below_threshold = ~valid_truth
    no_near = valid_truth & (nearest_cost > limit)
    collision = valid_truth & (nearest_cost <= limit) & ~matched_truth
    target_component = np.asarray(
        assignment["target_mass_component"], dtype=np.float32
    )
    oracle_selected = e131.top_k_mask(target_component, candidate_mask)
    oracle_truth = selected_truth_mask(
        oracle_selected, candidate_to_truth, truth_fraction.shape[1]
    )
    actual_truth = selected_truth_mask(
        local_selected, candidate_to_truth, truth_fraction.shape[1]
    )

    matched_mass = sum_truth_fraction(truth_fraction, matched_truth)
    oracle_mass = sum_truth_fraction(truth_fraction, oracle_truth)
    actual_mass = sum_truth_fraction(truth_fraction, actual_truth)
    categories = np.column_stack(
        (
            sum_truth_fraction(truth_fraction, below_threshold),
            sum_truth_fraction(truth_fraction, no_near),
            sum_truth_fraction(truth_fraction, collision),
            matched_mass - oracle_mass,
            oracle_mass - actual_mass,
            actual_mass,
        )
    )
    categories[np.abs(categories) < 1e-12] = 0.0
    if np.min(categories) < -1e-7:
        raise RuntimeError(f"negative waterfall category: {np.min(categories)}")
    categories = np.maximum(categories, 0.0)
    valid_point = np.asarray(assignment["valid_point"], dtype=bool)
    maximum_error = float(np.max(np.abs(categories[valid_point].sum(axis=1) - 1.0)))
    if maximum_error > 2e-6:
        raise RuntimeError(f"waterfall does not sum to one: {maximum_error}")
    return categories.astype(np.float32), {
        "matched_truth": matched_truth,
        "below_threshold_truth": below_threshold,
        "no_near_truth": no_near,
        "collision_truth": collision,
        "oracle_selected_candidate": oracle_selected,
        "oracle_selected_truth": oracle_truth,
        "actual_selected_truth": actual_truth,
    }


def summarize_categories(
    categories: np.ndarray,
    valid: np.ndarray,
    component_energy_total: np.ndarray,
) -> dict[str, object]:
    point_balanced = np.mean(categories[valid], axis=0, dtype=np.float64)
    weight = component_energy_total[valid].astype(np.float64)
    energy_weighted = np.sum(
        categories[valid] * weight[:, None], axis=0, dtype=np.float64
    ) / max(float(weight.sum()), TINY)
    return {
        "category_names": list(CATEGORY_NAMES),
        "category_labels": list(CATEGORY_LABELS),
        "point_balanced_fraction": {
            name: float(value) for name, value in zip(CATEGORY_NAMES, point_balanced)
        },
        "global_component_energy_weighted_fraction": {
            name: float(value) for name, value in zip(CATEGORY_NAMES, energy_weighted)
        },
        "point_balanced_sum": float(point_balanced.sum()),
        "energy_weighted_sum": float(energy_weighted.sum()),
    }


def grouped_categories(
    categories: np.ndarray,
    valid: np.ndarray,
    group: np.ndarray,
    labels: list[str],
) -> list[dict[str, object]]:
    output = []
    for group_id, label in enumerate(labels):
        rows = valid & (group == group_id)
        value = (
            np.mean(categories[rows], axis=0, dtype=np.float64)
            if np.any(rows)
            else np.zeros(len(CATEGORY_NAMES), dtype=np.float64)
        )
        output.append(
            {
                "group": int(group_id),
                "label": label,
                "count": int(rows.sum()),
                "fractions": {
                    name: float(item) for name, item in zip(CATEGORY_NAMES, value)
                },
            }
        )
    return output


def counterfactual_ranking(
    e131_metrics: np.lib.npyio.NpzFile,
    assignment: dict[str, np.ndarray | float | int],
    truth_coordinate: np.ndarray,
    support: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    valid_truth = np.asarray(assignment["valid_truth"], dtype=bool)
    truth_fraction = np.asarray(assignment["truth_fraction"], dtype=np.float32)
    candidate_to_truth = np.asarray(assignment["candidate_to_truth"], dtype=np.int16)
    target_component = np.asarray(
        assignment["target_mass_component"], dtype=np.float32
    )
    truth_evidence = e131.patch_energy(
        support, truth_coordinate, valid_truth
    ).astype(np.float32)
    candidate_truth_evidence = np.zeros_like(target_component)
    for row in range(len(candidate_truth_evidence)):
        ids = np.flatnonzero(candidate_to_truth[row] >= 0)
        if len(ids):
            candidate_truth_evidence[row, ids] = truth_evidence[
                row, candidate_to_truth[row, ids]
            ]

    base_rank = e131.row_ranks(e131_metrics["base_mass"], candidate_mask)
    candidate_evidence_rank = e131.row_ranks(
        e131_metrics["v7_local_energy"], candidate_mask
    )
    truth_evidence_rank = e131.row_ranks(candidate_truth_evidence, candidate_mask)
    selections = {
        "base_prior": e131_metrics["baseline_selected"].astype(bool),
        "v7_candidate_coordinate_only": e131.top_k_mask(
            candidate_evidence_rank, candidate_mask
        ),
        "base_plus_v7_candidate_coordinate": e131_metrics[
            "local_selected"
        ].astype(bool),
        "base_plus_v7_true_coordinate": e131.top_k_mask(
            0.5 * (base_rank + truth_evidence_rank), candidate_mask
        ),
        "candidate_mass_oracle_top15": e131.top_k_mask(
            target_component, candidate_mask
        ),
    }
    candidate_summary = {}
    matched_total = target_component.sum(axis=1, dtype=np.float64)
    valid_matched = matched_total > TINY
    for name, selected in selections.items():
        selected_mass = (target_component * selected).sum(axis=1, dtype=np.float64)
        recall = np.divide(
            selected_mass,
            np.maximum(matched_total, TINY),
            out=np.zeros_like(selected_mass),
        )
        candidate_summary[name] = {
            "matched_mass_recall_point_balanced": float(
                np.mean(recall[valid_matched])
            ),
            "all_query_component_mass_captured": float(
                np.mean(selected_mass[np.asarray(assignment["valid_point"], bool)])
            ),
        }

    truth_oracle = top_k_truth_mask(truth_fraction, valid_truth)
    truth_v7 = top_k_truth_mask(truth_evidence, valid_truth)
    true_oracle_coverage = sum_truth_fraction(truth_fraction, truth_oracle)
    true_v7_coverage = sum_truth_fraction(truth_fraction, truth_v7)
    valid_point = np.asarray(assignment["valid_point"], dtype=bool)
    field_summary = {
        "true_path_mass_oracle_top15": float(
            np.mean(true_oracle_coverage[valid_point])
        ),
        "v7_at_exact_true_coordinates_top15": float(
            np.mean(true_v7_coverage[valid_point])
        ),
        "v7_field_ranking_gap": float(
            np.mean((true_oracle_coverage - true_v7_coverage)[valid_point])
        ),
    }
    actual = candidate_summary["base_plus_v7_candidate_coordinate"]
    exact = candidate_summary["base_plus_v7_true_coordinate"]
    oracle = candidate_summary["candidate_mass_oracle_top15"]
    interpretation = {
        "candidate_coordinate_sampling_effect": float(
            exact["matched_mass_recall_point_balanced"]
            - actual["matched_mass_recall_point_balanced"]
        ),
        "remaining_ranking_gap_with_true_coordinates": float(
            oracle["matched_mass_recall_point_balanced"]
            - exact["matched_mass_recall_point_balanced"]
        ),
    }
    return (
        {
            "candidate_selector_ladder": candidate_summary,
            "v7_field_at_true_path_locations": field_summary,
            "counterfactual_differences": interpretation,
        },
        {
            "truth_v7_energy": truth_evidence,
            "candidate_truth_coordinate_energy": candidate_truth_evidence,
            "truth_oracle_selected": truth_oracle,
            "truth_v7_selected": truth_v7,
            **{f"selection_{name}": value for name, value in selections.items()},
        },
    )


def ranking_miss_breakdown(
    assignment: dict[str, np.ndarray | float | int],
    candidate_mask: np.ndarray,
    e131_metrics: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    target = np.asarray(assignment["target_mass_component"], dtype=np.float64)
    matched = target > 0.0
    actual = e131_metrics["local_selected"].astype(bool)
    missed = matched & ~actual
    base_only = e131_metrics["baseline_selected"].astype(bool)
    v7_only = e131.top_k_mask(
        e131.row_ranks(e131_metrics["v7_local_energy"], candidate_mask),
        candidate_mask,
    )
    miss_masks = {
        "neither": missed & ~base_only & ~v7_only,
        "base_only": missed & base_only & ~v7_only,
        "v7_only": missed & ~base_only & v7_only,
        "both": missed & base_only & v7_only,
    }
    total_missed = float(np.sum(target * missed, dtype=np.float64))
    return {
        "denominator": "matched true component mass missed by fused Top-15",
        "total_missed_component_mass": total_missed,
        "fractions": {
            name: float(np.sum(target * mask, dtype=np.float64))
            / max(total_missed, TINY)
            for name, mask in miss_masks.items()
        },
        "labels": {name: label for name, label in zip(MISS_NAMES, MISS_LABELS)},
    }


def coordinate_diagnostics(
    assignment: dict[str, np.ndarray | float | int],
    e131_metrics: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    candidate_to_truth = np.asarray(assignment["candidate_to_truth"], dtype=np.int16)
    nearest_axis = np.asarray(assignment["nearest_axis_delta"], dtype=np.float64)
    truth_fraction = np.asarray(assignment["truth_fraction"], dtype=np.float64)
    selected = e131_metrics["local_selected"].astype(bool)
    selected_truth = selected_truth_mask(
        selected, candidate_to_truth, truth_fraction.shape[1]
    )
    matched_truth = np.asarray(assignment["truth_to_candidate"], dtype=np.int16) >= 0
    missed_truth = matched_truth & ~selected_truth
    no_near = np.asarray(assignment["valid_truth"], dtype=bool) & (
        np.asarray(assignment["nearest_cost"], dtype=np.float32)
        > float(e115.Config().soft_cost_limit)
    )

    def weighted_axis(mask: np.ndarray) -> list[float]:
        weight = truth_fraction * mask
        total = float(weight.sum())
        if total <= TINY:
            return [0.0, 0.0, 0.0]
        return [
            float(np.sum(weight * nearest_axis[..., axis], dtype=np.float64) / total)
            for axis in range(3)
        ]

    no_near_contribution = np.stack(
        (
            np.square(nearest_axis[..., 0] / 1.5),
            np.square(nearest_axis[..., 1]),
            np.square(nearest_axis[..., 2] / 2.0),
        ),
        axis=-1,
    )
    dominant_axis = np.argmax(no_near_contribution, axis=-1)
    no_near_weight = truth_fraction * no_near
    dominant_mass = np.asarray(
        [float(no_near_weight[dominant_axis == axis].sum()) for axis in range(3)]
    )
    dominant_mass /= max(float(dominant_mass.sum()), TINY)
    return {
        "axis_order": ["horizontal_bin", "vertical_bin", "delay_bin"],
        "selected_matched_axis_error": weighted_axis(selected_truth),
        "missed_matched_axis_error": weighted_axis(missed_truth),
        "no_near_axis_error": weighted_axis(no_near),
        "no_near_dominant_cost_axis_fraction": dominant_mass.tolist(),
    }


def path_count_statistics(
    assignment: dict[str, np.ndarray | float | int], valid: np.ndarray
) -> dict[str, object]:
    true_count = np.asarray(assignment["valid_truth"], bool).sum(axis=1)
    matched_count = (np.asarray(assignment["truth_to_candidate"], np.int16) >= 0).sum(
        axis=1
    )
    return {
        "true_path_count_mean": float(np.mean(true_count[valid])),
        "true_path_count_median": float(np.median(true_count[valid])),
        "true_path_count_p90": float(np.quantile(true_count[valid], 0.90)),
        "points_with_more_than_15_true_paths": int(np.sum(valid & (true_count > 15))),
        "matched_path_count_mean": float(np.mean(matched_count[valid])),
        "matched_path_count_median": float(np.median(matched_count[valid])),
        "points_with_more_than_15_matched_paths": int(
            np.sum(valid & (matched_count > 15))
        ),
        "points_with_no_matched_path": int(np.sum(valid & (matched_count == 0))),
    }


def draw_waterfall(summary: dict[str, object]) -> None:
    values = np.asarray(
        [summary["point_balanced_fraction"][name] for name in CATEGORY_NAMES]
    )
    fig, ax = plt.subplots(figsize=(12, 3.4), constrained_layout=True)
    left = 0.0
    for value, label, color in zip(values, CATEGORY_LABELS, CATEGORY_COLORS):
        ax.barh([0], [value], left=left, color=color, height=0.56, label=label)
        if value >= 0.025:
            ax.text(
                left + value / 2.0,
                0,
                f"{value * 100:.1f}%",
                ha="center",
                va="center",
                color="white" if color not in ("#9ca3af", "#f59e0b") else "black",
                fontsize=10,
                fontweight="bold",
            )
        left += value
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("真实分解路径能量占比")
    ax.set_title("V7 Top-15路径覆盖损失的完整分解")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(VISUAL_DIR / "01_路径能量覆盖损失瀑布图.png", dpi=180)
    plt.close(fig)


def draw_regime(regime_summary: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in regime_summary]
    matrix = np.asarray(
        [
            [row["fractions"][name] for name in CATEGORY_NAMES]
            for row in regime_summary
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    bottom = np.zeros(len(labels), dtype=np.float64)
    for index, (name, label, color) in enumerate(
        zip(CATEGORY_NAMES, CATEGORY_LABELS, CATEGORY_COLORS)
    ):
        ax.bar(labels, matrix[:, index], bottom=bottom, color=color, label=label)
        bottom += matrix[:, index]
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("真实分解路径能量占比")
    ax.set_xlabel("传播难度分区")
    ax.set_title("不同传播区域的路径覆盖损失构成")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(VISUAL_DIR / "02_不同传播区域的覆盖损失构成.png", dpi=180)
    plt.close(fig)


def draw_selector_ladder(counterfactual: dict[str, object]) -> None:
    rows = counterfactual["candidate_selector_ladder"]
    names = (
        "base_prior",
        "v7_candidate_coordinate_only",
        "base_plus_v7_candidate_coordinate",
        "base_plus_v7_true_coordinate",
        "candidate_mass_oracle_top15",
    )
    labels = ("原质量先验", "仅V7局部能量", "原先验+V7", "真实坐标采样V7", "候选内Top-15上限")
    values = [rows[name]["matched_mass_recall_point_balanced"] for name in names]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=("#64748b", "#0ea5e9", "#2563eb", "#7c3aed", "#16a34a"))
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("可匹配路径中的能量召回")
    ax.set_title("候选路径选择的信息上限与实际差距")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(VISUAL_DIR / "03_候选路径选择能力阶梯图.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parse_args()
    started = time.time()
    configure_chinese_font()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    e131_metrics = np.load(E131_METRICS, allow_pickle=False)
    query_ids = e131_metrics["query_ids"].astype(np.int64)
    candidate_fold = e131_metrics["candidate_fold"].astype(np.int8)
    query_site = e131_metrics["query_site"].astype(np.int8)
    candidate_mask = e131_metrics["candidate_mask"].astype(bool)
    candidate_cache = e131.gather_candidate_cache(query_ids, candidate_fold)
    candidate_coordinate = candidate_cache["coordinate"]

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"][query_ids].astype(np.float32)
    truth_coefficient = component["coefficients"][query_ids].astype(np.complex64)
    truth_power = component["point_power"][query_ids].astype(np.float64)
    residual_fraction = component["residual_fraction"][query_ids].astype(np.float64)
    assignment = detailed_assignment(
        candidate_coordinate,
        candidate_mask,
        truth_coordinate,
        truth_coefficient,
        truth_power,
    )
    target_difference = float(
        np.max(
            np.abs(
                np.asarray(assignment["target_mass_power"], dtype=np.float32)
                - e131_metrics["target_mass"].astype(np.float32)
            )
        )
    )
    if target_difference > 2e-6:
        raise RuntimeError(f"E131 assignment replay mismatch: {target_difference}")

    categories, masks = build_waterfall(
        assignment,
        candidate_mask,
        e131_metrics["local_selected"].astype(bool),
    )
    valid = np.asarray(assignment["valid_point"], dtype=bool)
    summary = summarize_categories(
        categories,
        valid,
        np.asarray(assignment["total_component_energy"], dtype=np.float64),
    )
    regime_cache = np.load(REGIME_CACHE, allow_pickle=False)
    regime = regime_cache["oof_regime"].astype(np.int8)
    regime_names = regime_cache["regime_names"].astype(str).tolist()
    by_regime = grouped_categories(categories, valid, regime, regime_names)
    by_site = grouped_categories(
        categories, valid, query_site, ["基站0", "基站1"]
    )
    support = np.load(V7_SUPPORT, mmap_mode="r")
    counterfactual, counterfactual_arrays = counterfactual_ranking(
        e131_metrics,
        assignment,
        truth_coordinate,
        support,
        candidate_mask,
    )
    miss_breakdown = ranking_miss_breakdown(
        assignment, candidate_mask, e131_metrics
    )
    coordinate = coordinate_diagnostics(assignment, e131_metrics)
    counts = path_count_statistics(assignment, valid)

    point_balanced = summary["point_balanced_fraction"]
    loss_names = CATEGORY_NAMES[:-1]
    dominant_loss = max(loss_names, key=lambda name: point_balanced[name])
    captured = float(point_balanced["captured"])
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "objective": (
            "Partition 100% of held decomposed-path coefficient energy into mutually "
            "exclusive candidate-generation, Top-15, ranking, and captured terms."
        ),
        "protocol": {
            "query_count": int(len(query_ids)),
            "nonzero_component_points": int(valid.sum()),
            "official_h_used_only_as_held_diagnostic_truth": True,
            "v7_is_strict_oof_official_h_only": True,
            "network_training": False,
            "parameter_scan": False,
            "top_k": e131.TOP_K,
            "candidate_match_cost_limit": float(e115.Config().soft_cost_limit),
            "candidate_match_bandwidth": float(e115.Config().soft_bandwidth),
            "e131_assignment_replay_max_abs_difference": target_difference,
        },
        "definitions": {
            "denominator": (
                "Within each nonzero point, the squared magnitudes of all 24 true "
                "decomposed-path coefficient vectors are normalized to one."
            ),
            "below_threshold": "component below the fixed E124 1e-5 power threshold",
            "no_near_candidate": "no candidate within the fixed periodic H/V/delay cost limit",
            "identity_collision": "a nearby candidate exists but one-to-one assignment gives it to another true path",
            "top15_capacity": "mass matched to candidates but outside the true-mass-optimal Top-15",
            "v7_ranking": "gap between candidate Top-15 oracle and the E131 V7-fused Top-15",
            "captured": "true path mass assigned to a candidate selected by E131",
        },
        "coverage_waterfall": summary,
        "by_regime": by_regime,
        "by_site": by_site,
        "path_counts": counts,
        "candidate_match": {
            "matched_cost_mean": assignment["matched_cost_mean"],
            "matched_cost_p90": assignment["matched_cost_p90"],
        },
        "ranking_counterfactuals": counterfactual,
        "ranking_miss_breakdown": miss_breakdown,
        "coordinate_diagnostics": coordinate,
        "component_representation": {
            "explained_fraction_nonzero_mean": float(
                np.mean(1.0 - residual_fraction[valid])
            ),
            "explained_fraction_nonzero_median": float(
                np.median(1.0 - residual_fraction[valid])
            ),
            "residual_is_not_added_to_coefficient_mass_waterfall": True,
        },
        "decision": {
            "dominant_point_balanced_loss": dominant_loss,
            "dominant_loss_fraction": float(point_balanced[dominant_loss]),
            "captured_fraction_of_all_decomposed_path_mass": captured,
            "next_step_must_target": dominant_loss,
        },
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "visualizations": [
                PATHS.relative(VISUAL_DIR / "01_路径能量覆盖损失瀑布图.png"),
                PATHS.relative(VISUAL_DIR / "02_不同传播区域的覆盖损失构成.png"),
                PATHS.relative(VISUAL_DIR / "03_候选路径选择能力阶梯图.png"),
            ],
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        candidate_fold=candidate_fold,
        query_site=query_site,
        regime=regime,
        valid_point=valid,
        category_names=np.asarray(CATEGORY_NAMES, dtype="U32"),
        category_fraction=categories,
        truth_fraction=np.asarray(assignment["truth_fraction"], dtype=np.float32),
        valid_truth=np.asarray(assignment["valid_truth"], dtype=bool),
        candidate_to_truth=np.asarray(assignment["candidate_to_truth"], dtype=np.int16),
        truth_to_candidate=np.asarray(assignment["truth_to_candidate"], dtype=np.int16),
        nearest_cost=np.asarray(assignment["nearest_cost"], dtype=np.float32),
        nearest_axis_delta=np.asarray(
            assignment["nearest_axis_delta"], dtype=np.float32
        ),
        target_mass_component=np.asarray(
            assignment["target_mass_component"], dtype=np.float32
        ),
        oracle_selected_candidate=masks["oracle_selected_candidate"],
        actual_selected_truth=masks["actual_selected_truth"],
        **counterfactual_arrays,
    )
    draw_waterfall(summary)
    draw_regime(by_regime)
    draw_selector_ladder(counterfactual)
    print(
        json.dumps(
            {
                "coverage_waterfall": summary["point_balanced_fraction"],
                "path_counts": counts,
                "ranking_counterfactuals": counterfactual,
                "ranking_miss_breakdown": miss_breakdown,
                "coordinate_diagnostics": coordinate,
                "decision": payload["decision"],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
