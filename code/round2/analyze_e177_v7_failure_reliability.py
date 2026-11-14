#!/usr/bin/env python3
"""Test whether Clean observables identify severe V7 failures across holes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import analyze_e174_v12_unique_rescue_clean_features as e174
from project_paths import PATHS


VERSION = "Round2-E177-v7-failure-reliability"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e177_v7_failure_reliability"
REPORT_JSON = ANALYSIS_DIR / "v7_failure_reliability.json"
REPORT_MD = ANALYSIS_DIR / "V7严重失败点Clean识别报告.md"
POINT_CSV = ANALYSIS_DIR / "fixed500_v7_failure_risk.csv"
PROBE_CSV = ANALYSIS_DIR / "clean_feature_group_oof_metrics.csv"
FOLD_CSV = ANALYSIS_DIR / "broad_fold_oof_metrics.csv"
HOLE_CSV = ANALYSIS_DIR / "pseudo_hole_failure_metrics.csv"
FEATURE_CSV = ANALYSIS_DIR / "clean_feature_univariate_metrics.csv"
PERMUTATION_CSV = ANALYSIS_DIR / "reliability_core_permutation_control.csv"
RISK_PLOT = ANALYSIS_DIR / "v7_score_vs_clean_risk.png"
HOLE_PLOT = ANALYSIS_DIR / "hard_hole_auc_ap.png"

E153_METRICS = (
    PATHS.artifacts
    / "round2"
    / "v7_v12_peak_actions"
    / "v7_v12_peak_action_metrics.npz"
)

N_POINT = 500
SEVERE_COUNT = 100
SENSITIVE_COUNT = 50
HARD_HOLES = ("P_S1F1", "P_S1F2", "P_S1F3", "P_S1F4")
SEED = 20_260_811_177


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refresh-controls", action="store_true")
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


def ranked_low_mask(value: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((np.arange(len(value), dtype=np.int64), value))
    output = np.zeros(len(value), dtype=bool)
    output[order[:count]] = True
    return output


def ranked_high_mask(value: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((np.arange(len(value), dtype=np.int64), -value))
    output = np.zeros(len(value), dtype=bool)
    output[order[:count]] = True
    return output


def top_fraction_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    fraction: float = 0.20,
) -> dict[str, float | int]:
    count = max(1, int(round(len(truth) * fraction)))
    selected = np.argsort(probability, kind="stable")[-count:]
    true_positive = int(np.sum(truth[selected]))
    prevalence = float(np.mean(truth))
    precision = true_positive / count
    recall = true_positive / max(int(np.sum(truth)), 1)
    return {
        "top_fraction": fraction,
        "top_count": count,
        "top_true_positive": true_positive,
        "precision_at_top_fraction": precision,
        "recall_at_top_fraction": recall,
        "lift_at_top_fraction": precision / max(prevalence, 1e-12),
    }


def load_labels() -> dict[str, np.ndarray | float]:
    metrics = np.load(E153_METRICS, allow_pickle=False)
    allowed = (
        "query_ids",
        "fold_id",
        "site",
        "v7_pas",
        "v7_pdp",
        "v7_error",
        "truth_energy",
    )
    values = {name: np.asarray(metrics[name]) for name in allowed}
    if any(len(value) != N_POINT for value in values.values()):
        raise RuntimeError("E153 fixed500 metric shape mismatch")
    point_nmse = np.asarray(values["v7_error"], dtype=np.float64) / np.maximum(
        np.asarray(values["truth_energy"], dtype=np.float64), 1e-30
    )
    point_score = (
        0.4 * np.asarray(values["v7_pas"], dtype=np.float64)
        + 0.4 * np.asarray(values["v7_pdp"], dtype=np.float64)
        + 0.2 / (1.0 + point_nmse)
    )
    severe = ranked_low_mask(point_score, SEVERE_COUNT)
    sensitive = ranked_low_mask(point_score, SENSITIVE_COUNT)
    pas_bad = np.asarray(values["v7_pas"], dtype=np.float64) <= 0.50
    nmse_bad = ranked_high_mask(point_nmse, SEVERE_COUNT)
    return {
        **values,
        "point_nmse": point_nmse,
        "point_score": point_score,
        "severity": 1.0 - point_score,
        "severe_failure": severe,
        "severe_threshold": float(np.max(point_score[severe])),
        "bottom10_failure": sensitive,
        "pas_bad": pas_bad,
        "nmse_bad": nmse_bad,
    }


def feature_matrices(
    features: list[e174.Feature],
) -> tuple[dict[str, np.ndarray], list[str]]:
    groups = e174.feature_groups(features)
    core_names = [
        feature.name
        for feature in features
        if feature.group in {"v7_spectrum", "neighbor_h", "carrier_disagreement"}
    ]
    core_values = [
        feature.values
        for feature in features
        if feature.group in {"v7_spectrum", "neighbor_h", "carrier_disagreement"}
    ]
    groups["reliability_core"] = np.column_stack(core_values).astype(np.float64)
    return groups, core_names


def evaluate_probe(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    model_name: str,
    trees: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    probability, metrics, fold_rows = e174.cross_hole_probe(
        matrix, target, folds, model_name, trees, seed
    )
    severity_correlation = spearmanr(probability, target.astype(np.float64)).statistic
    summary: dict[str, object] = {
        **metrics,
        **top_fraction_metrics(target, probability),
        "positive_folds": int(
            np.sum(
                [
                    float(row["roc_auc"]) > 0.5
                    for row in fold_rows
                    if math.isfinite(float(row["roc_auc"]))
                ]
            )
        ),
        "risk_label_spearman": float(severity_correlation),
    }
    return probability, summary, fold_rows


def subset_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    local_target = target[mask]
    local_probability = probability[mask]
    result = e174.binary_metrics(local_target, local_probability)
    result.update(top_fraction_metrics(local_target, local_probability))
    result["count"] = int(np.sum(mask))
    result["positive_count"] = int(np.sum(local_target))
    return result


def nonzero_control(
    labels: dict[str, np.ndarray | float],
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> dict[str, object]:
    truth_energy = np.asarray(labels["truth_energy"], dtype=np.float64)
    nonzero = truth_energy > 1e-30
    return {
        "zero_truth_count": int(np.sum(~nonzero)),
        "zero_truth_severe_count": int(np.sum(target & ~nonzero)),
        "nonzero_count": int(np.sum(nonzero)),
        "nonzero_severe_count": int(np.sum(target & nonzero)),
        "models": {
            model_name: subset_metrics(target, probability, nonzero)
            for model_name, probability in probabilities.items()
        },
    }


def permute_within_fold_site(
    target: np.ndarray,
    folds: np.ndarray,
    site: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = target.copy()
    for fold in range(5):
        for site_value in np.unique(site):
            rows = np.flatnonzero((folds == fold) & (site == site_value))
            output[rows] = target[rows][rng.permutation(len(rows))]
    return output


def univariate_rows(
    features: list[e174.Feature],
    target: np.ndarray,
    folds: np.ndarray,
    trees: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, feature in enumerate(features):
        matrix = np.asarray(feature.values, dtype=np.float64)[:, None]
        probability, metrics, fold_values = e174.cross_hole_probe(
            matrix, target, folds, "logistic", trees, seed + index * 101
        )
        finite = np.isfinite(feature.values)
        scale = float(np.nanstd(feature.values))
        effect = (
            float(
                np.nanmean(feature.values[target & finite])
                - np.nanmean(feature.values[~target & finite])
            )
            / max(scale, 1e-12)
        )
        rows.append(
            {
                "feature": feature.name,
                "label": feature.label,
                "group": feature.group,
                "effect_severe_minus_other_std": effect,
                **metrics,
                "positive_folds": int(
                    np.sum(
                        [
                            float(row["roc_auc"]) > 0.5
                            for row in fold_values
                            if math.isfinite(float(row["roc_auc"]))
                        ]
                    )
                ),
                "risk_mean": float(np.mean(probability)),
            }
        )
    return sorted(rows, key=lambda row: float(row["roc_auc"]), reverse=True)


def make_plots(
    labels: dict[str, np.ndarray | float],
    hole_id: np.ndarray,
    core_probabilities: dict[str, np.ndarray],
    hole_rows: list[dict[str, object]],
) -> None:
    point_score = np.asarray(labels["point_score"], dtype=np.float64)
    severe = np.asarray(labels["severe_failure"], dtype=bool)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for axis, model_name in zip(axes, ("logistic", "extra_trees")):
        risk = core_probabilities[model_name]
        axis.scatter(
            point_score[~severe],
            risk[~severe],
            s=14,
            alpha=0.45,
            color="#64748b",
            label="Other",
        )
        axis.scatter(
            point_score[severe],
            risk[severe],
            s=20,
            alpha=0.75,
            color="#dc2626",
            label="Severe failure",
        )
        axis.axvline(float(labels["severe_threshold"]), color="#111827", linestyle="--")
        axis.set_xlabel("V7 point score (GT label only)")
        axis.set_ylabel("OOF Clean risk")
        axis.set_title(model_name.replace("_", " ").title())
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle("Round2 E177: Clean V7 failure risk")
    figure.tight_layout()
    figure.savefig(RISK_PLOT, dpi=180)
    plt.close(figure)

    selected = [row for row in hole_rows if row["hole_id"] in HARD_HOLES]
    x = np.arange(len(HARD_HOLES), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for axis, metric in zip(axes, ("roc_auc", "average_precision")):
        for offset, model_name, color in (
            (-width / 2, "logistic", "#0f766e"),
            (width / 2, "extra_trees", "#2563eb"),
        ):
            values = [
                next(
                    float(row[metric])
                    for row in selected
                    if row["hole_id"] == hole and row["model"] == model_name
                )
                for hole in HARD_HOLES
            ]
            axis.bar(x + offset, values, width=width, label=model_name, color=color)
        axis.axhline(0.5 if metric == "roc_auc" else 0.2, color="#111827", linestyle="--")
        axis.set_xticks(x, HARD_HOLES)
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Round2 E177: four hard pseudo-holes")
    figure.tight_layout()
    figure.savefig(HOLE_PLOT, dpi=180)
    plt.close(figure)


def write_report(report: dict[str, object]) -> None:
    decision = report["decision"]
    lines = [
        "# V7严重失败点Clean识别报告",
        "",
        "## 结论",
        "",
        str(decision["plain_language_conclusion"]),
        "",
        "## 主标签",
        "",
        f"- 严重失败点：{report['labels']['severe_count']}/500。",
        f"- V7逐点分阈值：{report['labels']['severe_threshold']:.9f}。",
        "- GT只定义标签，不进入任何Clean特征。",
        "",
        "## reliability_core整洞OOF",
        "",
        "| 模型 | AUC | AP | Balanced accuracy | Top20% precision | Top20% recall | AUC>0.5折数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in ("logistic", "extra_trees"):
        value = report["primary"][model_name]
        lines.append(
            f"| {model_name} | {value['roc_auc']:.6f} | {value['average_precision']:.6f} | "
            f"{value['balanced_accuracy']:.6f} | {value['precision_at_top_fraction']:.6f} | "
            f"{value['recall_at_top_fraction']:.6f} | {value['positive_folds']}/5 |"
        )
    lines.extend(
        [
            "",
            "## 四个困难洞",
            "",
            "详见`pseudo_hole_failure_metrics.csv`和`hard_hole_auc_ap.png`。",
            "",
            "## 边界",
            "",
            "本实验只判断哪里危险，不预测应该如何修正，也不产生H或版本。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    query_ids = np.asarray(labels["query_ids"], dtype=np.int64)
    folds = np.asarray(labels["fold_id"], dtype=np.int8)
    site = np.asarray(labels["site"], dtype=np.int8)
    target = np.asarray(labels["severe_failure"], dtype=bool)
    if int(np.sum(target)) != SEVERE_COUNT:
        raise RuntimeError("severe failure count changed")

    if args.refresh_controls:
        if not REPORT_JSON.is_file() or not POINT_CSV.is_file():
            raise FileNotFoundError("refresh requires the completed E177 report and point CSV")
        with POINT_CSV.open(encoding="utf-8-sig", newline="") as handle:
            point_rows = list(csv.DictReader(handle))
        if len(point_rows) != N_POINT:
            raise RuntimeError("E177 point CSV row count changed")
        probabilities = {
            "logistic": np.asarray(
                [float(row["risk_logistic"]) for row in point_rows], dtype=np.float64
            ),
            "extra_trees": np.asarray(
                [float(row["risk_extra_trees"]) for row in point_rows], dtype=np.float64
            ),
        }
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        report["nonzero_control"] = nonzero_control(labels, target, probabilities)
        REPORT_JSON.write_text(
            json.dumps(e174.json_value(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_report(e174.json_value(report))
        print(json.dumps(report["nonzero_control"], ensure_ascii=False, indent=2))
        return

    features, hole_id, lineage = e174.build_features({"query_ids": query_ids})
    groups, core_names = feature_matrices(features)
    probe_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    probabilities: dict[tuple[str, str], np.ndarray] = {}
    for group_name in (
        "reliability_core",
        "v7_spectrum",
        "neighbor_h",
        "carrier_disagreement",
        "structure",
        "map_geometry",
        "all_clean",
    ):
        for model_name in ("logistic", "extra_trees"):
            probability, summary, local_folds = evaluate_probe(
                groups[group_name],
                target,
                folds,
                model_name,
                args.trees,
                args.seed + len(probe_rows) * 1009,
            )
            summary["risk_actual_severity_spearman"] = float(
                spearmanr(
                    probability,
                    np.asarray(labels["severity"], dtype=np.float64),
                ).statistic
            )
            probabilities[(group_name, model_name)] = probability
            probe_rows.append(
                {
                    "label": "severe_failure_bottom20",
                    "feature_group": group_name,
                    "model": model_name,
                    "feature_count": int(groups[group_name].shape[1]),
                    **summary,
                }
            )
            for row in local_folds:
                fold_rows.append(
                    {
                        "label": "severe_failure_bottom20",
                        "feature_group": group_name,
                        "model": model_name,
                        **row,
                    }
                )

    sensitivity_labels = {
        "bottom10_failure": np.asarray(labels["bottom10_failure"], dtype=bool),
        "v7_pas_le_0_50": np.asarray(labels["pas_bad"], dtype=bool),
        "top20_point_nmse": np.asarray(labels["nmse_bad"], dtype=bool),
    }
    sensitivity: list[dict[str, object]] = []
    for label_name, local_target in sensitivity_labels.items():
        for model_name in ("logistic", "extra_trees"):
            _, summary, local_folds = evaluate_probe(
                groups["reliability_core"],
                local_target,
                folds,
                model_name,
                args.trees,
                args.seed + 50000 + len(sensitivity) * 1009,
            )
            sensitivity.append(
                {
                    "label": label_name,
                    "model": model_name,
                    "positive_count": int(np.sum(local_target)),
                    **summary,
                    "folds": local_folds,
                }
            )

    core_probabilities = {
        model_name: probabilities[("reliability_core", model_name)]
        for model_name in ("logistic", "extra_trees")
    }
    hole_rows: list[dict[str, object]] = []
    for hole in sorted(np.unique(hole_id).tolist()):
        mask = hole_id == hole
        for model_name, probability in core_probabilities.items():
            hole_rows.append(
                {
                    "hole_id": hole,
                    "is_hard_hole": hole in HARD_HOLES,
                    "model": model_name,
                    **subset_metrics(target, probability, mask),
                    "mean_v7_point_score": float(
                        np.mean(np.asarray(labels["point_score"])[mask])
                    ),
                    "mean_v7_pas": float(
                        np.mean(np.asarray(labels["v7_pas"])[mask])
                    ),
                }
            )

    hard_mask = np.isin(hole_id, HARD_HOLES)
    hard_summary = {
        model_name: subset_metrics(target, probability, hard_mask)
        for model_name, probability in core_probabilities.items()
    }
    hard_positive_holes = {
        model_name: int(
            np.sum(
                [
                    float(row["roc_auc"]) > 0.5
                    for row in hole_rows
                    if row["is_hard_hole"] and row["model"] == model_name
                    and math.isfinite(float(row["roc_auc"]))
                ]
            )
        )
        for model_name in core_probabilities
    }

    rng = np.random.default_rng(args.seed + 90000)
    permutation_rows: list[dict[str, object]] = []
    for permutation in range(args.permutations):
        permuted = permute_within_fold_site(target, folds, site, rng)
        for model_name in ("logistic", "extra_trees"):
            _, metrics, _ = e174.cross_hole_probe(
                groups["reliability_core"],
                permuted,
                folds,
                model_name,
                args.trees,
                args.seed + 100000 + permutation * 101 + (0 if model_name == "logistic" else 1),
            )
            permutation_rows.append(
                {
                    "permutation": permutation,
                    "model": model_name,
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                }
            )

    primary: dict[str, dict[str, object]] = {}
    for model_name in ("logistic", "extra_trees"):
        row = next(
            value
            for value in probe_rows
            if value["feature_group"] == "reliability_core"
            and value["model"] == model_name
        )
        null_auc = np.asarray(
            [
                value["roc_auc"]
                for value in permutation_rows
                if value["model"] == model_name
            ],
            dtype=np.float64,
        )
        threshold = float(np.quantile(null_auc, 0.95))
        pass_gate = bool(
            float(row["roc_auc"]) >= 0.75
            and float(row["average_precision"]) >= 0.40
            and int(row["positive_folds"]) >= 4
            and float(hard_summary[model_name]["roc_auc"]) >= 0.70
            and hard_positive_holes[model_name] >= 3
            and float(row["roc_auc"]) > threshold
        )
        primary[model_name] = {
            **row,
            "hard_holes_combined": hard_summary[model_name],
            "hard_holes_auc_above_half": hard_positive_holes[model_name],
            "permutation_auc_p95": threshold,
            "permutation_p_value": float(
                (1 + np.sum(null_auc >= float(row["roc_auc"])))
                / (len(null_auc) + 1)
            ),
            "passed": pass_gate,
        }

    feature_rows = univariate_rows(
        features, target, folds, args.trees, args.seed + 200000
    )
    point_rows: list[dict[str, object]] = []
    for index in range(N_POINT):
        point_rows.append(
            {
                "oof_row": index,
                "query_id": int(query_ids[index]),
                "broad_fold": int(folds[index]),
                "site": int(site[index]),
                "hole_id": str(hole_id[index]),
                "v7_pas_label_only": float(np.asarray(labels["v7_pas"])[index]),
                "v7_pdp_label_only": float(np.asarray(labels["v7_pdp"])[index]),
                "v7_point_nmse_label_only": float(
                    np.asarray(labels["point_nmse"])[index]
                ),
                "v7_point_score_label_only": float(
                    np.asarray(labels["point_score"])[index]
                ),
                "severe_failure": bool(target[index]),
                "risk_logistic": float(core_probabilities["logistic"][index]),
                "risk_extra_trees": float(core_probabilities["extra_trees"][index]),
            }
        )

    passed = bool(any(value["passed"] for value in primary.values()))
    conclusion = (
        "Clean信息可以跨未见空间洞识别V7严重失败点，但这只说明能够知道哪里危险，"
        "不说明知道应向哪个方向修正。"
        if passed
        else "现有Clean不确定性特征不足以稳定跨未见空间洞识别V7严重失败点；"
        "它们至多能把容易点与复杂点分开，不能可靠回答哪里一定危险。"
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete_gate_passed" if passed else "analysis_complete_gate_failed",
        "diagnostic_only": True,
        "official_inputs_only": True,
        "sionna_used": False,
        "v12_features_used": False,
        "labels": {
            "definition": "0.4*V7_PAS + 0.4*V7_PDP + 0.2/(1+point_NMSE)",
            "severe_count": int(np.sum(target)),
            "severe_prevalence": float(np.mean(target)),
            "severe_threshold": float(labels["severe_threshold"]),
            "zero_truth_points_retained": int(
                np.sum(np.asarray(labels["truth_energy"]) <= 1e-30)
            ),
        },
        "features": {
            "total_clean_features": len(features),
            "reliability_core_features": len(core_names),
            "reliability_core_names": core_names,
            "lineage_rows": len(lineage),
        },
        "primary": primary,
        "hard_holes": {
            "hole_ids": list(HARD_HOLES),
            "combined": hard_summary,
            "auc_above_half_count": hard_positive_holes,
        },
        "nonzero_control": nonzero_control(labels, target, core_probabilities),
        "feature_group_results": probe_rows,
        "sensitivity_results": sensitivity,
        "top_univariate_features": feature_rows[:20],
        "decision": {
            "passed": passed,
            "clean_can_identify_where_v7_is_dangerous": passed,
            "clean_can_identify_correction_direction": False,
            "plain_language_conclusion": conclusion,
        },
        "outputs": {
            "report_md": PATHS.relative(REPORT_MD),
            "point_csv": PATHS.relative(POINT_CSV),
            "probe_csv": PATHS.relative(PROBE_CSV),
            "fold_csv": PATHS.relative(FOLD_CSV),
            "hole_csv": PATHS.relative(HOLE_CSV),
            "feature_csv": PATHS.relative(FEATURE_CSV),
            "permutation_csv": PATHS.relative(PERMUTATION_CSV),
            "risk_plot": PATHS.relative(RISK_PLOT),
            "hole_plot": PATHS.relative(HOLE_PLOT),
        },
        "seconds": float(time.time() - started),
        "device": "CPU",
        "measured_bottleneck": "fixed ExtraTrees cross-hole probes and permutation controls",
    }

    write_csv(POINT_CSV, point_rows)
    write_csv(PROBE_CSV, probe_rows)
    write_csv(FOLD_CSV, fold_rows)
    write_csv(HOLE_CSV, hole_rows)
    write_csv(FEATURE_CSV, feature_rows)
    write_csv(PERMUTATION_CSV, permutation_rows)
    make_plots(labels, hole_id, core_probabilities, hole_rows)
    REPORT_JSON.write_text(
        json.dumps(e174.json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(e174.json_value(payload))
    print(json.dumps(e174.json_value(payload), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
