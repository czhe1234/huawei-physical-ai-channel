#!/usr/bin/env python3
"""Test one-to-one V7 peak alignment in path coverage and actual channel rollout."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import time

import numpy as np
from scipy.ndimage import convolve, maximum_filter
from scipy.optimize import linear_sum_assignment
import torch

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e123_complete_path_variable_system as e123
import analyze_e131_v7_path_selection_evidence as e131
import analyze_e132_v7_path_coverage_gap as e132
import analyze_global_local_path_lifecycle as e79
import train_e115_complete_path_state_operator as e115
from project_paths import PATHS
from run_v6_regime_component_lifecycle import REGIME_CACHE


VERSION = "Round2-E133-v7-deformable-peak-alignment"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_deformable_peak_alignment"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_deformable_peak_alignment"
REPORT = ANALYSIS_DIR / "v7_deformable_peak_alignment.json"
METRICS = ARTIFACT_DIR / "v7_deformable_peak_alignment_metrics.npz"

PERIODS = np.asarray((e115.MH, e115.MV, e115.S_SC), dtype=np.float32)
PATCH_SIZE = tuple(2 * value + 1 for value in e131.PATCH_RADIUS)
ALIGNMENT_COST_LIMIT = float(e115.Config().soft_cost_limit)
ALIGNMENT_BANDWIDTH = float(e115.Config().soft_bandwidth)
MINIMUM_ALL_PATH_CAPTURE = 0.78
MINIMUM_ACTUAL_SCORE_GAIN = 0.002
MINIMUM_POSITIVE_FOLDS = 4
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def smoothed_field(support: np.ndarray) -> np.ndarray:
    kernel = np.ones(PATCH_SIZE, dtype=np.float32)
    return convolve(np.asarray(support, dtype=np.float32), kernel, mode="wrap")


def align_to_unique_peaks(
    support: np.ndarray,
    coordinate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    aligned_coordinate = np.array(coordinate, dtype=np.float32, copy=True)
    aligned_energy = np.zeros(mask.shape, dtype=np.float32)
    original_energy = np.zeros(mask.shape, dtype=np.float32)
    snapped = np.zeros(mask.shape, dtype=bool)
    assignment_utility_gain = np.zeros(mask.shape, dtype=np.float32)
    displacement = np.zeros((*mask.shape, 3), dtype=np.float32)
    peak_count = np.zeros(len(mask), dtype=np.int16)

    for row in range(len(mask)):
        field = smoothed_field(support[row])
        local_maximum = maximum_filter(field, size=PATCH_SIZE, mode="wrap")
        peak_coordinate = np.argwhere((field == local_maximum) & (field > 0.0))
        peak_count[row] = min(len(peak_coordinate), np.iinfo(np.int16).max)

        candidate_ids = np.flatnonzero(mask[row])
        if len(candidate_ids) == 0:
            continue
        candidate_coordinate = coordinate[row, candidate_ids]
        center = np.rint(candidate_coordinate).astype(np.int64)
        center %= PERIODS.astype(np.int64)
        fallback = field[center[:, 0], center[:, 1], center[:, 2]].astype(np.float64)
        original_energy[row, candidate_ids] = fallback.astype(np.float32)
        aligned_energy[row, candidate_ids] = fallback.astype(np.float32)
        if len(peak_coordinate) == 0:
            continue

        peak_coordinate_float = peak_coordinate.astype(np.float32)
        peak_energy = field[
            peak_coordinate[:, 0], peak_coordinate[:, 1], peak_coordinate[:, 2]
        ].astype(np.float64)
        scale = max(float(np.max(peak_energy)), float(np.max(fallback)), TINY)
        cost = e79.support_cost(candidate_coordinate, peak_coordinate_float)
        allowed = cost <= ALIGNMENT_COST_LIMIT
        peak_utility = (peak_energy[None] / scale) * np.exp(
            -0.5 * cost / ALIGNMENT_BANDWIDTH
        )
        peak_utility[~allowed] = -1e6

        candidate_count = len(candidate_ids)
        utility = np.full(
            (candidate_count, len(peak_coordinate) + candidate_count),
            -1e6,
            dtype=np.float64,
        )
        utility[:, : len(peak_coordinate)] = peak_utility
        utility[np.arange(candidate_count), len(peak_coordinate) + np.arange(candidate_count)] = (
            fallback / scale
        )
        candidate_local, selected_column = linear_sum_assignment(-utility)
        if len(candidate_local) != candidate_count:
            raise RuntimeError("peak assignment did not cover every valid candidate")

        for local_id, column in zip(candidate_local, selected_column, strict=True):
            if column >= len(peak_coordinate):
                continue
            fallback_utility = fallback[local_id] / scale
            selected_utility = utility[local_id, column]
            if selected_utility + 1e-12 < fallback_utility:
                raise RuntimeError("global assignment selected a worse private fallback")
            candidate_id = candidate_ids[local_id]
            peak = peak_coordinate_float[column]
            delta = np.asarray(
                [
                    e79.periodic_delta(
                        peak[axis],
                        candidate_coordinate[local_id, axis],
                        float(PERIODS[axis]),
                    )
                    for axis in range(3)
                ],
                dtype=np.float32,
            )
            aligned_coordinate[row, candidate_id] = np.mod(
                candidate_coordinate[local_id] + delta, PERIODS
            )
            aligned_energy[row, candidate_id] = np.float32(peak_energy[column])
            snapped[row, candidate_id] = bool(np.any(np.abs(delta) > 1e-7))
            assignment_utility_gain[row, candidate_id] = np.float32(
                selected_utility - fallback_utility
            )
            displacement[row, candidate_id] = delta

    return {
        "coordinate": aligned_coordinate,
        "energy": aligned_energy,
        "original_energy": original_energy,
        "snapped": snapped,
        "utility_gain": assignment_utility_gain,
        "displacement": displacement,
        "peak_count": peak_count,
    }


def selected_capture(
    selection: np.ndarray,
    candidate_to_truth: np.ndarray,
    truth_fraction: np.ndarray,
) -> np.ndarray:
    selected_truth = e132.selected_truth_mask(
        selection, candidate_to_truth, truth_fraction.shape[1]
    )
    return e132.sum_truth_fraction(truth_fraction, selected_truth)


def group_values(
    value: np.ndarray,
    valid: np.ndarray,
    group: np.ndarray,
    labels: list[str] | None = None,
) -> list[dict[str, float | int | str]]:
    output = []
    for group_id in np.unique(group):
        rows = valid & (group == group_id)
        item: dict[str, float | int | str] = {
            "group": int(group_id),
            "count": int(rows.sum()),
            "value": float(np.mean(value[rows])) if np.any(rows) else 0.0,
        }
        if labels is not None:
            item["label"] = labels[int(group_id)]
        output.append(item)
    return output


def group_gains(
    candidate: list[dict[str, float | int | str]],
    baseline: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    output = []
    for left, right in zip(candidate, baseline, strict=True):
        item: dict[str, float | int | str] = {
            "group": int(left["group"]),
            "count": int(left["count"]),
            "gain": float(left["value"]) - float(right["value"]),
        }
        if "label" in left:
            item["label"] = str(left["label"])
        output.append(item)
    return output


def selected_mass(base_mass: np.ndarray, selection: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = e123.normalize_mass(base_mass * selection, mask)
    empty = output.sum(axis=1) <= 0.0
    output[empty] = base_mass[empty]
    return output


def fold_score_gains(
    candidate: dict[str, object], baseline: dict[str, object]
) -> list[dict[str, float | int]]:
    return [
        {
            "fold": int(left["fold"]),
            "gain": float(left["score"]) - float(right["score"]),
        }
        for left, right in zip(
            candidate["folds"], baseline["folds"], strict=True
        )
    ]


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    e131_metrics = np.load(e131.METRICS, allow_pickle=False)
    e132_metrics = np.load(e132.METRICS, allow_pickle=False)
    query_ids = e131_metrics["query_ids"].astype(np.int64)
    candidate_fold = e131_metrics["candidate_fold"].astype(np.int8)
    query_site = e131_metrics["query_site"].astype(np.int8)
    candidate_mask = e131_metrics["candidate_mask"].astype(bool)
    candidate_cache = e131.gather_candidate_cache(query_ids, candidate_fold)
    candidate_coordinate = candidate_cache["coordinate"]
    support = np.load(e131.V7_SUPPORT, mmap_mode="r")
    alignment = align_to_unique_peaks(
        support, candidate_coordinate, candidate_mask
    )
    energy_replay_error = float(
        np.max(
            np.abs(
                alignment["original_energy"]
                - e131_metrics["v7_local_energy"].astype(np.float32)
            )
        )
    )
    energy_scale = max(
        float(np.max(np.abs(e131_metrics["v7_local_energy"]))), 1.0
    )
    if energy_replay_error > 2e-5 * energy_scale:
        raise RuntimeError(f"E131 patch energy replay mismatch: {energy_replay_error}")

    base_rank = e131.row_ranks(e131_metrics["base_mass"], candidate_mask)
    aligned_rank = e131.row_ranks(alignment["energy"], candidate_mask)
    aligned_score = 0.5 * (base_rank + aligned_rank)
    current_selection = e131_metrics["local_selected"].astype(bool)
    aligned_selection = e131.top_k_mask(aligned_score, candidate_mask)
    ranking, matched_recall = e131.ranking_metrics(
        aligned_score,
        aligned_selection,
        e131_metrics["target_mass"].astype(np.float32),
        candidate_mask,
        candidate_fold,
        query_site,
    )

    truth_fraction = e132_metrics["truth_fraction"].astype(np.float32)
    candidate_to_truth = e132_metrics["candidate_to_truth"].astype(np.int16)
    valid_point = e132_metrics["valid_point"].astype(bool)
    current_capture = selected_capture(
        current_selection, candidate_to_truth, truth_fraction
    )
    aligned_capture = selected_capture(
        aligned_selection, candidate_to_truth, truth_fraction
    )
    true_coordinate_capture = selected_capture(
        e132_metrics["selection_base_plus_v7_true_coordinate"].astype(bool),
        candidate_to_truth,
        truth_fraction,
    )
    current_capture_mean = float(np.mean(current_capture[valid_point]))
    if abs(current_capture_mean - 0.7530630593) > 2e-6:
        raise RuntimeError(f"E132 current coverage replay mismatch: {current_capture_mean}")
    aligned_capture_mean = float(np.mean(aligned_capture[valid_point]))

    regime_cache = np.load(REGIME_CACHE, allow_pickle=False)
    regime = regime_cache["oof_regime"].astype(np.int8)
    regime_names = regime_cache["regime_names"].astype(str).tolist()
    current_by_fold = group_values(
        current_capture, valid_point, candidate_fold
    )
    aligned_by_fold = group_values(
        aligned_capture, valid_point, candidate_fold
    )
    current_by_site = group_values(
        current_capture, valid_point, query_site, ["基站0", "基站1"]
    )
    aligned_by_site = group_values(
        aligned_capture, valid_point, query_site, ["基站0", "基站1"]
    )
    current_by_regime = group_values(
        current_capture, valid_point, regime, regime_names
    )
    aligned_by_regime = group_values(
        aligned_capture, valid_point, regime, regime_names
    )
    coverage_fold_gains = group_gains(aligned_by_fold, current_by_fold)
    coverage_site_gains = group_gains(aligned_by_site, current_by_site)
    coverage_regime_gains = group_gains(aligned_by_regime, current_by_regime)

    full_cache = e123.collect_cache_arrays(query_ids, candidate_fold)
    if not np.allclose(
        full_cache["coordinate"].astype(np.float32), candidate_coordinate, atol=0.0
    ):
        raise RuntimeError("full candidate cache coordinate mismatch")
    base_mass = e123.normalize_mass(
        np.exp(
            np.clip(full_cache["base_log_mass"].astype(np.float32), -30.0, 20.0)
        ),
        candidate_mask,
    )
    base_port = e123.normalize_port(
        np.exp(
            np.clip(full_cache["base_log_port"].astype(np.float32), -20.0, 10.0)
        )
    )
    base_phase = e123.phase_from_parts(full_cache["base_phase"])
    base_power = np.exp(
        np.clip(full_cache["base_log_power"].astype(np.float64), -80.0, 80.0)
    )
    current_mass = selected_mass(base_mass, current_selection, candidate_mask)
    aligned_mass = selected_mass(base_mass, aligned_selection, candidate_mask)

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    true_coordinate = component["coordinates"][query_ids].astype(np.float32)
    true_coefficient = component["coefficients"][query_ids].astype(np.complex64)
    true_power = component["point_power"][query_ids].astype(np.float64)
    nonzero = true_power > TINY
    truth_components = (true_coordinate, true_coefficient)
    states: OrderedDict[str, e123.PathState] = OrderedDict(
        (
            (
                "unmodified_base",
                e123.PathState(
                    candidate_coordinate, base_mass, base_port, base_phase, base_power
                ),
            ),
            (
                "current_selection_current_coordinate",
                e123.PathState(
                    candidate_coordinate,
                    current_mass,
                    base_port,
                    base_phase,
                    base_power,
                ),
            ),
            (
                "current_selection_aligned_coordinate",
                e123.PathState(
                    alignment["coordinate"],
                    current_mass,
                    base_port,
                    base_phase,
                    base_power,
                ),
            ),
            (
                "aligned_selection_current_coordinate",
                e123.PathState(
                    candidate_coordinate,
                    aligned_mass,
                    base_port,
                    base_phase,
                    base_power,
                ),
            ),
            (
                "aligned_selection_aligned_coordinate",
                e123.PathState(
                    alignment["coordinate"],
                    aligned_mass,
                    base_port,
                    base_phase,
                    base_power,
                ),
            ),
        )
    )
    actual_rollout = {}
    for name, state in states.items():
        actual_rollout[name] = e123.evaluate_state(
            state,
            query_ids,
            candidate_fold,
            nonzero,
            truth_components,
            device,
            args.batch_size,
        )
        print(
            f"{name} score={actual_rollout[name]['all']['score']:.6f}",
            flush=True,
        )

    actual_baseline = actual_rollout["current_selection_current_coordinate"]
    actual_primary = actual_rollout["aligned_selection_aligned_coordinate"]
    actual_selection_only = actual_rollout["aligned_selection_current_coordinate"]
    actual_coordinate_only = actual_rollout["current_selection_aligned_coordinate"]
    primary_fold_gains = fold_score_gains(actual_primary, actual_baseline)
    selection_fold_gains = fold_score_gains(actual_selection_only, actual_baseline)
    coordinate_fold_gains = fold_score_gains(actual_coordinate_only, actual_baseline)

    snapped = alignment["snapped"] & candidate_mask
    snapped_displacement = np.abs(alignment["displacement"][snapped])
    displacement_summary = {
        "axis_order": ["horizontal_bin", "vertical_bin", "delay_bin"],
        "mean_abs": (
            np.mean(snapped_displacement, axis=0).tolist()
            if len(snapped_displacement)
            else [0.0, 0.0, 0.0]
        ),
        "p90_abs": (
            np.quantile(snapped_displacement, 0.90, axis=0).tolist()
            if len(snapped_displacement)
            else [0.0, 0.0, 0.0]
        ),
    }

    coverage_passed = bool(
        aligned_capture_mean >= MINIMUM_ALL_PATH_CAPTURE
        and sum(float(row["gain"]) > 0.0 for row in coverage_fold_gains)
        >= MINIMUM_POSITIVE_FOLDS
        and all(float(row["gain"]) > 0.0 for row in coverage_site_gains)
    )
    actual_score_gain = float(
        actual_primary["all"]["score"] - actual_baseline["all"]["score"]
    )
    actual_passed = bool(
        actual_score_gain >= MINIMUM_ACTUAL_SCORE_GAIN
        and sum(float(row["gain"]) > 0.0 for row in primary_fold_gains)
        >= MINIMUM_POSITIVE_FOLDS
    )
    if not coverage_passed:
        next_action = "stop_local_peak_alignment"
    elif not actual_passed:
        next_action = "retain_alignment_evidence_but_redesign_joint_mass_state_interface"
    else:
        next_action = "train_or_deploy_aligned_path_state_operator"

    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "objective": (
            "Determine whether one-to-one V7 local-peak alignment both recovers held "
            "path energy and improves an actual predicted-state full-channel rollout."
        ),
        "protocol": {
            "query_count": int(len(query_ids)),
            "nonzero_component_points": int(valid_point.sum()),
            "network_training": False,
            "parameter_scan": False,
            "candidate_count_unchanged": True,
            "top_k": e131.TOP_K,
            "patch_size_h_v_delay": list(PATCH_SIZE),
            "peak_nms_size_h_v_delay": list(PATCH_SIZE),
            "alignment_cost_limit": ALIGNMENT_COST_LIMIT,
            "alignment_bandwidth": ALIGNMENT_BANDWIDTH,
            "one_peak_at_most_one_candidate": True,
            "private_no_action_fallback_per_candidate": True,
            "official_h_used_only_for_held_scoring": True,
        },
        "alignment": {
            "e131_patch_energy_replay_max_abs_error": energy_replay_error,
            "peak_count_mean": float(np.mean(alignment["peak_count"])),
            "peak_count_median": float(np.median(alignment["peak_count"])),
            "snapped_candidate_count": int(snapped.sum()),
            "snapped_candidate_fraction_of_valid": float(
                snapped.sum() / max(int(candidate_mask.sum()), 1)
            ),
            "snapped_points": int(np.sum(np.any(snapped, axis=1))),
            "changed_top15_candidate_count": int(
                np.sum(current_selection ^ aligned_selection)
            ),
            "changed_top15_point_count": int(
                np.sum(np.any(current_selection ^ aligned_selection, axis=1))
            ),
            "displacement": displacement_summary,
            "utility_gain_mean_snapped": float(
                np.mean(alignment["utility_gain"][snapped])
            )
            if np.any(snapped)
            else 0.0,
        },
        "coverage": {
            "current_all_path_capture": current_capture_mean,
            "aligned_all_path_capture": aligned_capture_mean,
            "gain": float(aligned_capture_mean - current_capture_mean),
            "true_coordinate_counterfactual_capture": float(
                np.mean(true_coordinate_capture[valid_point])
            ),
            "matched_path_mass_recall": ranking,
            "fold_gains": coverage_fold_gains,
            "site_gains": coverage_site_gains,
            "regime_gains": coverage_regime_gains,
        },
        "actual_predicted_state_rollout": actual_rollout,
        "actual_attribution": {
            "baseline": "current_selection_current_coordinate",
            "coordinate_only_score_gain": float(
                actual_coordinate_only["all"]["score"]
                - actual_baseline["all"]["score"]
            ),
            "selection_only_score_gain": float(
                actual_selection_only["all"]["score"]
                - actual_baseline["all"]["score"]
            ),
            "joint_score_gain": actual_score_gain,
            "coordinate_only_fold_gains": coordinate_fold_gains,
            "selection_only_fold_gains": selection_fold_gains,
            "joint_fold_gains": primary_fold_gains,
        },
        "gate": {
            "minimum_all_path_capture": MINIMUM_ALL_PATH_CAPTURE,
            "minimum_actual_score_gain": MINIMUM_ACTUAL_SCORE_GAIN,
            "minimum_positive_folds": MINIMUM_POSITIVE_FOLDS,
            "coverage_passed": coverage_passed,
            "actual_rollout_passed": actual_passed,
            "passed": bool(coverage_passed and actual_passed),
        },
        "decision": {
            "next_action": next_action,
            "candidate_or_zip_allowed": bool(coverage_passed and actual_passed),
        },
        "outputs": {"metrics": PATHS.relative(METRICS)},
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
        candidate_mask=candidate_mask,
        aligned_coordinate=alignment["coordinate"].astype(np.float32),
        original_v7_energy=alignment["original_energy"].astype(np.float32),
        aligned_v7_energy=alignment["energy"].astype(np.float32),
        snapped=alignment["snapped"],
        displacement=alignment["displacement"].astype(np.float32),
        utility_gain=alignment["utility_gain"].astype(np.float32),
        peak_count=alignment["peak_count"],
        current_selection=current_selection,
        aligned_selection=aligned_selection,
        current_all_path_capture=current_capture.astype(np.float32),
        aligned_all_path_capture=aligned_capture.astype(np.float32),
        aligned_matched_mass_recall=matched_recall.astype(np.float32),
    )
    print(
        json.dumps(
            {
                "alignment": payload["alignment"],
                "coverage": payload["coverage"],
                "actual_attribution": payload["actual_attribution"],
                "gate": payload["gate"],
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
