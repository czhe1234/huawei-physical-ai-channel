#!/usr/bin/env python3
"""Measure the fixed500 counterfactual ceiling of V7's four regime choices."""

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
import pandas as pd
import torch
from sklearn.metrics import mutual_info_score

from data import load_positions
from official_metrics import (
    OfficialMetricAccumulator,
    PointMetricStatistics,
    aggregate_statistics,
)
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, statistics
from run_v6_regime_component_lifecycle import (
    PROFILE_ALPHA,
    REGIME_CACHE,
    REGIME_NAMES,
    TARGET_PATHS,
    apply_candidate,
    candidate_list,
    point_metrics,
)
from analyze_v7_stacked_lifecycle_overlay import (
    METRICS as V7_METRICS,
    OOF_CURRENT,
    OOF_SCALED,
    SELECTION as V7_SELECTION,
)


VERSION = "Round2-E179-v7-regime-oracle"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e179_v7_regime_oracle"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e179_v7_regime_oracle"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
E178_POINT = (
    PATHS.analysis
    / "results"
    / "round2_e178_v7_regime_e177_reliability"
    / "fixed500_v7_regime_e177_risk.csv"
)

BASELINE_OUTPUT = ARTIFACT_DIR / "fixed500_v7_original_regime_channel.npy"
FORCED_OUTPUTS = tuple(
    ARTIFACT_DIR / f"fixed500_v7_forced_{name}_channel.npy" for name in REGIME_NAMES
)
ORACLE_OUTPUT = ARTIFACT_DIR / "fixed500_v7_gt_regime_oracle_channel.npy"

REPORT_JSON = ANALYSIS_DIR / "v7_regime_oracle.json"
POINT_CSV = ANALYSIS_DIR / "fixed500_regime_oracle_points.csv"
DECISION_CSV = ANALYSIS_DIR / "fold_site_forced_regime_decisions.csv"
HOLE_CSV = ANALYSIS_DIR / "pseudo_hole_regime_oracle.csv"
SITE_CSV = ANALYSIS_DIR / "site_regime_oracle.csv"
BOUNDARY_CSV = ANALYSIS_DIR / "light_medium_boundary_oracle.csv"
BIN2D_CSV = ANALYSIS_DIR / "map_e177_risk_best_regime_bins.csv"
SCATTER_PLOT = ANALYSIS_DIR / "map_risk_e177_risk_gt_best_regime.png"
CONFUSION_PLOT = ANALYSIS_DIR / "original_vs_gt_best_regime.png"
HOLE_PLOT = ANALYSIS_DIR / "pseudo_hole_oracle_gain.png"

BASELINE_SCORE = 0.6444563457343335
TIE_TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_fold_site_regime_actions(
    metrics: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    names = metrics["candidate_names"].astype(str)
    fold_id = metrics["fold_id"].astype(np.int8)
    site = metrics["site"].astype(np.int8)
    original_regime = metrics["regime"].astype(np.int8)
    allowed = np.flatnonzero(
        (names == "identity") | np.char.startswith(names, "lifecycle_shift_")
    )
    choices = np.zeros((5, 2, 4), dtype=np.int16)
    rows: list[dict[str, object]] = []
    for held_fold in range(5):
        for site_value in range(2):
            for regime_value in range(4):
                fit = (
                    (fold_id != held_fold)
                    & (site == site_value)
                    & (original_regime == regime_value)
                )
                if int(np.sum(fit)) < 8:
                    choice = 0
                    fit_score = float(
                        statistics(
                            metrics["pas"][0],
                            metrics["pdp"][0],
                            metrics["error"][0],
                            metrics["energy"][0],
                            fit,
                        )["score"]
                    )
                else:
                    scored = []
                    for candidate_index in allowed:
                        score = float(
                            statistics(
                                metrics["pas"][candidate_index],
                                metrics["pdp"][candidate_index],
                                metrics["error"][candidate_index],
                                metrics["energy"][candidate_index],
                                fit,
                            )["score"]
                        )
                        scored.append((score, -int(candidate_index), int(candidate_index)))
                    fit_score, _, choice = max(scored)
                choices[held_fold, site_value, regime_value] = choice
                rows.append(
                    {
                        "held_fold": held_fold,
                        "site": site_value,
                        "forced_regime": REGIME_NAMES[regime_value],
                        "fit_count": int(np.sum(fit)),
                        "candidate_index": int(choice),
                        "candidate": str(names[choice]),
                        "fit_score": fit_score,
                    }
                )

    historical = json.loads(V7_SELECTION.read_text(encoding="utf-8"))["winner"]
    for row in historical["nested_decisions"]:
        fold = int(row["held_fold"])
        site_value = int(row["site"])
        regime_value = REGIME_NAMES.index(str(row["regime"]))
        actual = str(names[choices[fold, site_value, regime_value]])
        if actual != str(row["candidate"]):
            raise RuntimeError(
                f"nested decision mismatch fold={fold} site={site_value} "
                f"regime={regime_value}: {actual} != {row['candidate']}"
            )
    return choices, rows


def prediction_for_choices(
    base: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    choice: np.ndarray,
    regime: np.ndarray,
    site: np.ndarray,
    candidates: list[object],
) -> torch.Tensor:
    output = base.clone()
    for candidate_index in np.unique(choice):
        if int(candidate_index) == 0:
            continue
        rows = np.flatnonzero(choice == candidate_index)
        candidate = candidates[int(candidate_index) - 1]
        if candidate.target != "lifecycle_shift":
            raise RuntimeError(f"unexpected candidate target {candidate.target}")
        output[rows] = apply_candidate(
            base[rows].clone(),
            current[rows],
            target[rows],
            regime[rows],
            site[rows],
            candidate,
        )
    return output


def empty_metric_arrays(count: int) -> dict[str, np.ndarray]:
    return {
        key: np.empty(count, dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }


def assign_metric_block(
    output: dict[str, np.ndarray],
    start: int,
    end: int,
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for key, value in zip(("pas", "pdp", "error", "energy"), values):
        output[key][start:end] = value


def official_score_channel(
    channel: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> tuple[PointMetricStatistics, dict[str, float | int]]:
    accumulator = OfficialMetricAccumulator()
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        accumulator.add(
            np.asarray(channel[start:end], dtype=np.complex64),
            np.asarray(truth[query_ids[start:end]], dtype=np.complex64),
        )
    values = accumulator.statistics()
    return values, aggregate_statistics(values)


def local_point_scores(values: PointMetricStatistics) -> np.ndarray:
    nmse = values.squared_error / np.maximum(values.truth_energy, 1e-30)
    return 0.4 * values.pas + 0.4 * values.pdp + 0.2 / (1.0 + nmse)


def selected_statistics(
    baseline: PointMetricStatistics,
    oracle: PointMetricStatistics,
    selected: np.ndarray,
) -> dict[str, float | int]:
    return aggregate_statistics(
        PointMetricStatistics(
            pas=np.where(selected, oracle.pas, baseline.pas),
            pdp=np.where(selected, oracle.pdp, baseline.pdp),
            squared_error=np.where(
                selected, oracle.squared_error, baseline.squared_error
            ),
            truth_energy=baseline.truth_energy,
        )
    )


def categorical_information(
    best: np.ndarray, map_risk: np.ndarray, e177_risk: np.ndarray
) -> tuple[dict[str, float], list[dict[str, object]]]:
    map_bin = pd.qcut(map_risk, 4, labels=False, duplicates="drop").astype(int)
    e177_bin = pd.qcut(e177_risk, 4, labels=False, duplicates="drop").astype(int)
    joint_bin = map_bin * 4 + e177_bin
    conditional = 0.0
    rows: list[dict[str, object]] = []
    for left in range(4):
        mask = map_bin == left
        conditional += float(np.mean(mask)) * mutual_info_score(
            best[mask], e177_bin[mask]
        )
        for right in range(4):
            local = mask & (e177_bin == right)
            counts = np.bincount(best[local], minlength=4)
            rows.append(
                {
                    "map_risk_quartile": left + 1,
                    "e177_risk_quartile": right + 1,
                    "count": int(np.sum(local)),
                    **{
                        f"best_{REGIME_NAMES[index]}": int(counts[index])
                        for index in range(4)
                    },
                    "dominant_regime": REGIME_NAMES[int(np.argmax(counts))]
                    if np.sum(counts)
                    else "none",
                    "purity": float(np.max(counts) / np.sum(counts))
                    if np.sum(counts)
                    else float("nan"),
                }
            )
    return (
        {
            "mi_best_map_quartile": float(mutual_info_score(best, map_bin)),
            "mi_best_e177_quartile": float(mutual_info_score(best, e177_bin)),
            "mi_best_joint_quartile": float(mutual_info_score(best, joint_bin)),
            "conditional_mi_best_e177_given_map": conditional,
        },
        rows,
    )


def draw_plots(points: pd.DataFrame, hole_rows: list[dict[str, object]]) -> None:
    colors = ("#2a9d8f", "#4c78a8", "#f2b134", "#d1495b")
    fig, axis = plt.subplots(figsize=(9, 6), dpi=160)
    for regime_value, name in enumerate(REGIME_NAMES):
        rows = points["gt_best_regime_index"] == regime_value
        axis.scatter(
            points.loc[rows, "v7_physical_risk"],
            points.loc[rows, "risk_extra_trees"],
            s=22,
            alpha=0.66,
            color=colors[regime_value],
            edgecolors="none",
            label=name,
        )
    axis.set_xlabel("V7 continuous map risk")
    axis.set_ylabel("E177 OOF failure risk")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.22)
    axis.legend(title="GT best regime", ncol=2)
    fig.tight_layout()
    fig.savefig(SCATTER_PLOT)
    plt.close(fig)

    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(
        confusion,
        (
            points["v7_regime_index"].to_numpy(dtype=np.int64),
            points["gt_best_regime_index"].to_numpy(dtype=np.int64),
        ),
        1,
    )
    fig, axis = plt.subplots(figsize=(6, 5), dpi=160)
    image = axis.imshow(confusion, cmap="Blues")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center")
    axis.set_xticks(range(4), REGIME_NAMES)
    axis.set_yticks(range(4), REGIME_NAMES)
    axis.set_xlabel("GT best regime")
    axis.set_ylabel("Original V7 regime")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT)
    plt.close(fig)

    holes = [str(row["hole_id"]) for row in hole_rows]
    delta = [float(row["delta_score"]) for row in hole_rows]
    fig, axis = plt.subplots(figsize=(10, 5), dpi=160)
    axis.bar(holes, delta, color=["#4c78a8" if value >= 0 else "#d1495b" for value in delta])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Oracle score delta vs V7")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(HOLE_PLOT)
    plt.close(fig)


def json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    device_index = torch.cuda.current_device() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_index)

    read_started = time.time()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime_cache = np.load(REGIME_CACHE, allow_pickle=False)
    original_regime = regime_cache["oof_regime"].astype(np.int8)
    map_risk = regime_cache["oof_risk"].astype(np.float64)
    metrics_npz = np.load(V7_METRICS, allow_pickle=False)
    metrics = {key: np.asarray(metrics_npz[key]) for key in metrics_npz.files}
    choices, decision_rows = select_fold_site_regime_actions(metrics)
    candidates = candidate_list()
    base = np.load(OOF_SCALED, mmap_mode="r")
    current = np.load(OOF_CURRENT, mmap_mode="r")
    target = np.load(TARGET_PATHS["lifecycle_shift"][0], mmap_mode="r")
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_context = pd.read_csv(E178_POINT, encoding="utf-8-sig")
    if not np.array_equal(point_context["query_id"].to_numpy(), query_ids):
        raise RuntimeError("E178 point order does not match V7 OOF order")
    read_seconds = time.time() - read_started

    count = len(query_ids)
    output_shape = tuple(base.shape)
    baseline_output = np.lib.format.open_memmap(
        BASELINE_OUTPUT, mode="w+", dtype=np.complex64, shape=output_shape
    )
    forced_outputs = [
        np.lib.format.open_memmap(path, mode="w+", dtype=np.complex64, shape=output_shape)
        for path in FORCED_OUTPUTS
    ]
    gpu_metrics = {
        "baseline": empty_metric_arrays(count),
        **{name: empty_metric_arrays(count) for name in REGIME_NAMES},
    }

    projection_started = time.time()
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        local_fold = fold_id[start:end]
        local_site = site[start:end]
        local_original = original_regime[start:end]
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        current_tensor = torch.as_tensor(
            np.asarray(current[start:end], dtype=np.float32), device=device
        )
        target_tensor = torch.as_tensor(
            np.asarray(target[start:end], dtype=np.float32), device=device
        )
        truth_tensor = torch.as_tensor(
            np.array(truth[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        baseline_choice = choices[local_fold, local_site, local_original]
        with torch.inference_mode():
            prediction = prediction_for_choices(
                base_tensor,
                current_tensor,
                target_tensor,
                baseline_choice,
                local_original,
                local_site,
                candidates,
            )
            baseline_output[start:end] = prediction.cpu().numpy().astype(np.complex64)
            assign_metric_block(
                gpu_metrics["baseline"],
                start,
                end,
                point_metrics(prediction, truth_tensor),
            )
            for forced_regime, name in enumerate(REGIME_NAMES):
                forced = np.full(end - start, forced_regime, dtype=np.int8)
                forced_choice = choices[local_fold, local_site, forced]
                prediction = prediction_for_choices(
                    base_tensor,
                    current_tensor,
                    target_tensor,
                    forced_choice,
                    forced,
                    local_site,
                    candidates,
                )
                forced_outputs[forced_regime][start:end] = (
                    prediction.cpu().numpy().astype(np.complex64)
                )
                assign_metric_block(
                    gpu_metrics[name],
                    start,
                    end,
                    point_metrics(prediction, truth_tensor),
                )
        baseline_output.flush()
        for output in forced_outputs:
            output.flush()
        print(f"E179 counterfactual projection {end}/{count}", flush=True)
    projection_seconds = time.time() - projection_started

    scoring_started = time.time()
    baseline_values, baseline_result = official_score_channel(
        baseline_output, truth, query_ids, args.batch_size
    )
    forced_values: list[PointMetricStatistics] = []
    forced_results: list[dict[str, float | int]] = []
    for name, output in zip(REGIME_NAMES, forced_outputs):
        values, result = official_score_channel(output, truth, query_ids, args.batch_size)
        forced_values.append(values)
        forced_results.append(result)
        print(f"E179 official score forced {name}: {result['score']:.9f}", flush=True)
    scoring_seconds = time.time() - scoring_started

    if abs(float(baseline_result["score"]) - BASELINE_SCORE) > 2e-7:
        raise RuntimeError(
            f"V7 identity failed: {baseline_result['score']} != {BASELINE_SCORE}"
        )
    for key in ("pas", "pdp", "error", "energy"):
        official = {
            "pas": baseline_values.pas,
            "pdp": baseline_values.pdp,
            "error": baseline_values.squared_error,
            "energy": baseline_values.truth_energy,
        }[key]
        if not np.allclose(official, gpu_metrics["baseline"][key], rtol=2e-5, atol=2e-7):
            raise RuntimeError(f"GPU/official baseline {key} mismatch")

    local_scores = np.stack([local_point_scores(value) for value in forced_values])
    maximum = np.max(local_scores, axis=0)
    best_regime = np.empty(count, dtype=np.int8)
    original_is_optimal = np.zeros(count, dtype=bool)
    optimal_count = np.empty(count, dtype=np.int8)
    for point in range(count):
        optimal = np.flatnonzero(local_scores[:, point] >= maximum[point] - TIE_TOLERANCE)
        optimal_count[point] = len(optimal)
        original_is_optimal[point] = original_regime[point] in optimal
        if original_is_optimal[point]:
            best_regime[point] = original_regime[point]
        else:
            best_regime[point] = min(
                optimal,
                key=lambda value: (abs(int(value) - int(original_regime[point])), int(value)),
            )

    oracle_output = np.lib.format.open_memmap(
        ORACLE_OUTPUT, mode="w+", dtype=np.complex64, shape=output_shape
    )
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        block = np.empty((end - start, *output_shape[1:]), dtype=np.complex64)
        for regime_value in range(4):
            rows = np.flatnonzero(best_regime[start:end] == regime_value)
            if len(rows):
                block[rows] = forced_outputs[regime_value][start:end][rows]
        oracle_output[start:end] = block
        oracle_output.flush()
    oracle_values, oracle_result = official_score_channel(
        oracle_output, truth, query_ids, args.batch_size
    )

    # Verify that forcing every point to its original regime exactly reconstructs V7.
    original_replay = PointMetricStatistics(
        pas=np.choose(original_regime, [value.pas for value in forced_values]),
        pdp=np.choose(original_regime, [value.pdp for value in forced_values]),
        squared_error=np.choose(
            original_regime, [value.squared_error for value in forced_values]
        ),
        truth_energy=baseline_values.truth_energy,
    )
    original_replay_result = aggregate_statistics(original_replay)
    if abs(float(original_replay_result["score"]) - BASELINE_SCORE) > 2e-7:
        raise RuntimeError("forced-original point replay does not reconstruct V7")

    point_context = point_context.copy()
    point_context["v7_physical_risk"] = map_risk
    point_context["v7_regime_index"] = original_regime
    point_context["v7_regime"] = [REGIME_NAMES[value] for value in original_regime]
    point_context["gt_best_regime_index"] = best_regime
    point_context["gt_best_regime"] = [REGIME_NAMES[value] for value in best_regime]
    point_context["original_is_gt_optimal"] = original_is_optimal
    point_context["gt_optimal_regime_count"] = optimal_count
    point_context["regime_distance"] = np.abs(best_regime - original_regime)
    point_context["gt_best_local_score_gain"] = (
        local_scores[best_regime, np.arange(count)]
        - local_scores[original_regime, np.arange(count)]
    )
    for regime_value, name in enumerate(REGIME_NAMES):
        point_context[f"local_score_forced_{name}"] = local_scores[regime_value]
    point_context.to_csv(POINT_CSV, index=False, encoding="utf-8-sig")

    hole_rows: list[dict[str, object]] = []
    for hole_id in sorted(point_context["hole_id"].unique()):
        mask = point_context["hole_id"].to_numpy() == hole_id
        base_hole = aggregate_statistics(baseline_values, mask)
        oracle_hole = aggregate_statistics(oracle_values, mask)
        global_hole_only = selected_statistics(baseline_values, oracle_values, mask)
        hole_rows.append(
            {
                "hole_id": str(hole_id),
                "site": int(point_context.loc[mask, "site"].iloc[0]),
                "count": int(np.sum(mask)),
                "original_optimal_rate": float(np.mean(original_is_optimal[mask])),
                "adjacent_better_rate": float(
                    np.mean((np.abs(best_regime - original_regime) == 1)[mask])
                ),
                "far_better_rate": float(
                    np.mean((np.abs(best_regime - original_regime) >= 2)[mask])
                ),
                **{
                    f"best_{name}": int(np.sum(best_regime[mask] == value))
                    for value, name in enumerate(REGIME_NAMES)
                },
                "v7_pas": float(base_hole["pas"]),
                "v7_pdp": float(base_hole["pdp"]),
                "v7_nmse": float(base_hole["nmse"]),
                "v7_score": float(base_hole["score"]),
                "oracle_pas": float(oracle_hole["pas"]),
                "oracle_pdp": float(oracle_hole["pdp"]),
                "oracle_nmse": float(oracle_hole["nmse"]),
                "oracle_score": float(oracle_hole["score"]),
                "delta_score": float(oracle_hole["score"] - base_hole["score"]),
                "global_gain_if_only_this_hole": float(
                    global_hole_only["score"] - baseline_result["score"]
                ),
            }
        )
    write_csv(HOLE_CSV, hole_rows)

    site_rows: list[dict[str, object]] = []
    for site_value in range(2):
        mask = site == site_value
        base_site = aggregate_statistics(baseline_values, mask)
        oracle_site = aggregate_statistics(oracle_values, mask)
        site_rows.append(
            {
                "site": site_value,
                "count": int(np.sum(mask)),
                "original_optimal_rate": float(np.mean(original_is_optimal[mask])),
                "adjacent_better_rate": float(
                    np.mean((np.abs(best_regime - original_regime) == 1)[mask])
                ),
                "far_better_rate": float(
                    np.mean((np.abs(best_regime - original_regime) >= 2)[mask])
                ),
                **{
                    f"best_{name}": int(np.sum(best_regime[mask] == value))
                    for value, name in enumerate(REGIME_NAMES)
                },
                "v7_score": float(base_site["score"]),
                "oracle_score": float(oracle_site["score"]),
                "delta_score": float(oracle_site["score"] - base_site["score"]),
            }
        )
    write_csv(SITE_CSV, site_rows)

    light_medium_scale = 0.5 * (
        point_context["light_medium_threshold"]
        - point_context["clear_light_threshold"]
        + point_context["medium_heavy_threshold"]
        - point_context["light_medium_threshold"]
    )
    boundary_margin = (
        point_context["v7_physical_risk"]
        - point_context["light_medium_threshold"]
    ) / light_medium_scale
    boundary_rows: list[dict[str, object]] = []
    for radius in (0.10, 0.25, 0.50):
        mask = np.abs(boundary_margin.to_numpy()) <= radius
        boundary_rows.append(
            {
                "normalized_radius": radius,
                "count": int(np.sum(mask)),
                "original_optimal_rate": float(np.mean(original_is_optimal[mask])),
                "adjacent_better_rate": float(
                    np.mean((np.abs(best_regime - original_regime) == 1)[mask])
                ),
                "far_better_rate": float(
                    np.mean((np.abs(best_regime - original_regime) >= 2)[mask])
                ),
                **{
                    f"best_{name}": int(np.sum(best_regime[mask] == value))
                    for value, name in enumerate(REGIME_NAMES)
                },
            }
        )
    write_csv(BOUNDARY_CSV, boundary_rows)

    information, bin_rows = categorical_information(
        best_regime,
        map_risk,
        point_context["risk_extra_trees"].to_numpy(dtype=np.float64),
    )
    write_csv(BIN2D_CSV, bin_rows)
    write_csv(DECISION_CSV, decision_rows)
    draw_plots(point_context, hole_rows)

    fold_rows = []
    for fold in range(5):
        mask = fold_id == fold
        base_fold = aggregate_statistics(baseline_values, mask)
        oracle_fold = aggregate_statistics(oracle_values, mask)
        fold_rows.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "baseline_score": float(base_fold["score"]),
                "oracle_score": float(oracle_fold["score"]),
                "delta_score": float(oracle_fold["score"] - base_fold["score"]),
            }
        )

    oracle_delta = float(oracle_result["score"] - baseline_result["score"])
    single_hole_gains = np.asarray(
        [float(row["global_gain_if_only_this_hole"]) for row in hole_rows]
    )
    max_hole_share = float(np.max(single_hole_gains) / max(oracle_delta, 1e-30))
    if oracle_delta < 0.002:
        decision = "stop_regime_gating_no_oracle_space"
    elif (
        oracle_delta >= 0.010
        and sum(float(row["delta_score"]) > 0.0 for row in fold_rows) >= 4
        and max_hole_share <= 0.50
    ):
        decision = "meaningful_oracle_space"
    else:
        decision = "limited_oracle_space_not_ready_for_gate"

    timing = {
        "data_and_selection_seconds": read_seconds,
        "gpu_projection_and_write_seconds": projection_seconds,
        "official_cpu_scoring_seconds": scoring_seconds,
        "total_seconds": float(time.time() - started),
    }
    payload = {
        "experiment": VERSION,
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "official_inputs_only": True,
        "sionna_used": False,
        "baseline_identity": {
            "expected_score": BASELINE_SCORE,
            "official": baseline_result,
            "forced_original_replay": original_replay_result,
        },
        "forced_all_points": {
            name: result for name, result in zip(REGIME_NAMES, forced_results)
        },
        "point_oracle": {
            "official": oracle_result,
            "delta_score": oracle_delta,
            "original_is_optimal_count": int(np.sum(original_is_optimal)),
            "original_is_optimal_rate": float(np.mean(original_is_optimal)),
            "adjacent_better_count": int(
                np.sum(np.abs(best_regime - original_regime) == 1)
            ),
            "adjacent_better_rate": float(
                np.mean(np.abs(best_regime - original_regime) == 1)
            ),
            "far_better_count": int(
                np.sum(np.abs(best_regime - original_regime) >= 2)
            ),
            "far_better_rate": float(
                np.mean(np.abs(best_regime - original_regime) >= 2)
            ),
            "tie_count": int(np.sum(optimal_count > 1)),
            "best_counts": {
                name: int(np.sum(best_regime == value))
                for value, name in enumerate(REGIME_NAMES)
            },
            "positive_folds": int(
                sum(float(row["delta_score"]) > 0.0 for row in fold_rows)
            ),
            "max_single_hole_global_gain_share": max_hole_share,
        },
        "folds": fold_rows,
        "sites": site_rows,
        "holes": hole_rows,
        "light_medium_boundary": boundary_rows,
        "two_continuous_variable_information": information,
        "decision": decision,
        "timing": timing,
        "device": {
            "selected": str(device),
            "name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index))
            if device.type == "cuda"
            else 0,
        },
        "measured_bottleneck": max(
            (
                "data_and_selection_seconds",
                "gpu_projection_and_write_seconds",
                "official_cpu_scoring_seconds",
            ),
            key=timing.get,
        ),
        "outputs": {
            "baseline_channel": PATHS.relative(BASELINE_OUTPUT),
            "forced_channels": [PATHS.relative(path) for path in FORCED_OUTPUTS],
            "oracle_channel": PATHS.relative(ORACLE_OUTPUT),
            "point_csv": PATHS.relative(POINT_CSV),
            "decision_csv": PATHS.relative(DECISION_CSV),
            "hole_csv": PATHS.relative(HOLE_CSV),
            "site_csv": PATHS.relative(SITE_CSV),
            "boundary_csv": PATHS.relative(BOUNDARY_CSV),
            "bin2d_csv": PATHS.relative(BIN2D_CSV),
            "scatter_plot": PATHS.relative(SCATTER_PLOT),
            "confusion_plot": PATHS.relative(CONFUSION_PLOT),
            "hole_plot": PATHS.relative(HOLE_PLOT),
        },
    }
    REPORT_JSON.write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_value(payload), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
