#!/usr/bin/env python3
"""Test direct Sionna path evidence in the fixed candidate-path selector."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e123_complete_path_variable_system as e123
import analyze_e124_one_to_one_path_identity_ceiling as e124
import analyze_e131_v7_path_selection_evidence as e131
import analyze_e133_v7_deformable_peak_alignment as e133
import analyze_path_candidate_information_audit as e84
import optimize_e98_global_array_angle_mapping as e98
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E134-sionna-candidate-selection-evidence"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_candidate_selection_evidence"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sionna_candidate_selection_evidence"
REPORT = ANALYSIS_DIR / "sionna_candidate_selection_evidence.json"
METRICS = ARTIFACT_DIR / "sionna_candidate_selection_evidence_metrics.npz"

MATCH_BANDWIDTH = 0.25
MATCH_COST_LIMIT = 1.0
MINIMUM_RECALL_GAIN = 0.02
MINIMUM_ACTUAL_SCORE_GAIN = 0.002
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def corrected_sionna_coordinates(
    tau: np.ndarray,
    angles: np.ndarray,
    site: np.ndarray,
    fold: np.ndarray,
) -> np.ndarray:
    parameter_cache = np.load(e98.PARAMETERS, allow_pickle=False)
    latent_by_fold = parameter_cache["latent"].astype(np.float64)
    output = np.zeros((len(tau), tau.shape[1], 3), dtype=np.float32)
    theta = np.asarray(angles[..., 0], dtype=np.float64)
    phi = np.asarray(angles[..., 1], dtype=np.float64)
    delay = np.remainder(np.asarray(tau, dtype=np.float64) * 30_000.0 * 192.0, 192.0)

    for fold_value in range(5):
        rows = np.flatnonzero(fold == fold_value)
        if len(rows) == 0:
            continue
        bounded = np.tanh(latent_by_fold[fold_value])
        yaw = np.radians(np.asarray(e98.CURRENT_YAW_DEG, dtype=np.float64))
        yaw += math.radians(30.0) * bounded[0:2]
        tilt = np.radians(np.asarray(e98.CURRENT_TILT_DEG, dtype=np.float64))
        tilt += math.radians(18.0) * bounded[2:4]
        horizontal_spacing = 0.50 + 0.18 * bounded[4]
        vertical_spacing = 4.00 + 1.20 * bounded[5]
        horizontal_offset = 4.0 * bounded[6:8]
        vertical_offset = 2.0 + 4.0 * bounded[8:10]

        local_site = site[rows]
        local_yaw = yaw[local_site][:, None]
        local_tilt = tilt[local_site][:, None]
        local_theta = theta[rows]
        local_phi = phi[rows]
        horizontal = (
            16.0
            * horizontal_spacing
            * np.sin(local_theta)
            * np.sin(local_phi - local_yaw)
            + horizontal_offset[local_site][:, None]
        ) % 16.0
        vertical = (
            8.0
            * vertical_spacing
            * (
                np.sin(local_tilt)
                * np.sin(local_theta)
                * np.cos(local_phi - local_yaw)
                - np.cos(local_tilt) * np.cos(local_theta)
            )
            + vertical_offset[local_site][:, None]
        ) % 8.0
        output[rows, :, 0] = horizontal.astype(np.float32)
        output[rows, :, 1] = vertical.astype(np.float32)
        output[rows, :, 2] = delay[rows].astype(np.float32)
    return output


def candidate_path_cost(
    candidate: np.ndarray,
    path: np.ndarray,
    axes: tuple[int, ...],
) -> np.ndarray:
    cost = np.zeros((len(candidate), len(path)), dtype=np.float64)
    for axis in axes:
        delta = e84.periodic_delta(
            candidate[:, None, axis],
            path[None, :, axis],
            float(e84.PERIODS[axis]),
        )
        cost += np.square(delta / float(e84.SUPPORT_SCALES[axis]))
    return cost


def transfer_sionna_evidence(
    candidate_coordinate: np.ndarray,
    candidate_mask: np.ndarray,
    path_coordinate: np.ndarray,
    path_mask: np.ndarray,
    path_mass: np.ndarray,
    axes: tuple[int, ...],
    one_to_one: bool,
) -> tuple[np.ndarray, dict[str, float | int]]:
    evidence = np.zeros(candidate_mask.shape, dtype=np.float32)
    matched = 0
    matched_points = 0
    transferred_mass = 0.0
    total_mass = 0.0
    minimum_costs: list[np.ndarray] = []

    for row in range(len(candidate_coordinate)):
        candidate_ids = np.flatnonzero(candidate_mask[row])
        path_ids = np.flatnonzero(path_mask[row] & (path_mass[row] > 0.0))
        if len(candidate_ids) == 0 or len(path_ids) == 0:
            continue
        local_mass = path_mass[row, path_ids].astype(np.float64)
        total_mass += float(local_mass.sum())
        cost = candidate_path_cost(
            candidate_coordinate[row, candidate_ids],
            path_coordinate[row, path_ids],
            axes,
        )
        minimum_costs.append(np.min(cost, axis=0))
        kernel = np.exp(-0.5 * cost / MATCH_BANDWIDTH)
        kernel[cost > MATCH_COST_LIMIT] = 0.0

        if one_to_one:
            utility = kernel * local_mass[None, :]
            candidate_local, path_local = linear_sum_assignment(-utility)
            valid = utility[candidate_local, path_local] > 0.0
            candidate_local = candidate_local[valid]
            path_local = path_local[valid]
            evidence[row, candidate_ids[candidate_local]] = utility[
                candidate_local, path_local
            ].astype(np.float32)
            matched += int(len(candidate_local))
            if len(candidate_local):
                matched_points += 1
                transferred_mass += float(local_mass[path_local].sum())
        else:
            denominator = kernel.sum(axis=0, keepdims=True)
            responsibility = np.divide(
                kernel,
                np.maximum(denominator, TINY),
                out=np.zeros_like(kernel),
                where=denominator > 0.0,
            )
            local_evidence = responsibility @ local_mass
            evidence[row, candidate_ids] = local_evidence.astype(np.float32)
            supported = denominator[0] > 0.0
            matched += int(np.sum(local_evidence > 0.0))
            if np.any(local_evidence > 0.0):
                matched_points += 1
            transferred_mass += float(local_mass[supported].sum())

    flat_minimum = (
        np.concatenate(minimum_costs) if minimum_costs else np.empty(0, dtype=np.float64)
    )
    summary = {
        "axes": list(axes),
        "assignment": "one_to_one" if one_to_one else "soft_responsibility",
        "positive_candidate_pairs": int(matched),
        "points_with_evidence": int(matched_points),
        "transferred_sionna_mass_fraction": float(
            transferred_mass / max(total_mass, TINY)
        ),
        "path_nearest_cost_median": (
            float(np.median(flat_minimum)) if len(flat_minimum) else math.nan
        ),
        "path_nearest_cost_p90": (
            float(np.quantile(flat_minimum, 0.90)) if len(flat_minimum) else math.nan
        ),
    }
    return evidence, summary


def evaluate_actual_states(
    selected: dict[str, np.ndarray],
    query_ids: np.ndarray,
    candidate_fold: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_coordinate: np.ndarray,
    true_coordinate: np.ndarray,
    true_coefficient: np.ndarray,
    true_power: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    full_cache = e123.collect_cache_arrays(query_ids, candidate_fold)
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
    nonzero = true_power > TINY
    truth_components = (true_coordinate, true_coefficient)
    output = OrderedDict()
    for name, selection in selected.items():
        state = e123.PathState(
            candidate_coordinate,
            e133.selected_mass(base_mass, selection, candidate_mask),
            base_port,
            base_phase,
            base_power,
        )
        output[name] = e123.evaluate_state(
            state,
            query_ids,
            candidate_fold,
            nonzero,
            truth_components,
            device,
            batch_size,
        )
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    e131_metrics = np.load(e131.METRICS, allow_pickle=False)
    query_ids = e131_metrics["query_ids"].astype(np.int64)
    candidate_fold = e131_metrics["candidate_fold"].astype(np.int8)
    query_site = e131_metrics["query_site"].astype(np.int8)
    candidate_mask = e131_metrics["candidate_mask"].astype(bool)
    target_mass = e131_metrics["target_mass"].astype(np.float32)
    candidate_cache = e131.gather_candidate_cache(query_ids, candidate_fold)
    candidate_coordinate = candidate_cache["coordinate"]

    path_names = ("tau", "angles", "mask", "presence", "energy_fraction")
    path_cache = {
        name: np.asarray(
            np.load(e84.PATH_CACHE / f"{name}.npy", mmap_mode="r")[query_ids]
        )
        for name in path_names
    }
    _, _, _, train_site, _ = load_positions()
    raw_coordinate = e84.sionna_coordinates(
        path_cache["tau"],
        path_cache["angles"],
        train_site[query_ids],
        json.loads(e84.SIONNA_STATE.read_text(encoding="utf-8"))["configuration"],
        vertical_alignment=2.0,
    )
    corrected_coordinate = corrected_sionna_coordinates(
        path_cache["tau"],
        path_cache["angles"],
        query_site,
        candidate_fold,
    )
    path_mask = path_cache["mask"].astype(bool)
    path_mass = np.maximum(path_cache["energy_fraction"].astype(np.float32), 0.0)
    path_mass *= np.sqrt(
        np.maximum(path_cache["presence"].astype(np.float32), 1.0) / 24.0
    )
    path_mass *= path_mask

    evidence = OrderedDict()
    evidence_summary = {}
    evidence["raw_full_one_to_one"], evidence_summary["raw_full_one_to_one"] = (
        transfer_sionna_evidence(
            candidate_coordinate,
            candidate_mask,
            raw_coordinate,
            path_mask,
            path_mass,
            (0, 1, 2),
            True,
        )
    )
    (
        evidence["corrected_full_one_to_one"],
        evidence_summary["corrected_full_one_to_one"],
    ) = transfer_sionna_evidence(
        candidate_coordinate,
        candidate_mask,
        corrected_coordinate,
        path_mask,
        path_mass,
        (0, 1, 2),
        True,
    )
    evidence["corrected_full_soft"], evidence_summary["corrected_full_soft"] = (
        transfer_sionna_evidence(
            candidate_coordinate,
            candidate_mask,
            corrected_coordinate,
            path_mask,
            path_mass,
            (0, 1, 2),
            False,
        )
    )
    evidence["corrected_hd_one_to_one"], evidence_summary[
        "corrected_hd_one_to_one"
    ] = transfer_sionna_evidence(
        candidate_coordinate,
        candidate_mask,
        corrected_coordinate,
        path_mask,
        path_mass,
        (0, 2),
        True,
    )
    evidence["delay_one_to_one"], evidence_summary["delay_one_to_one"] = (
        transfer_sionna_evidence(
            candidate_coordinate,
            candidate_mask,
            corrected_coordinate,
            path_mask,
            path_mass,
            (2,),
            True,
        )
    )

    base_rank = e131.row_ranks(e131_metrics["base_mass"], candidate_mask)
    v7_rank = e131.row_ranks(e131_metrics["v7_local_energy"], candidate_mask)
    score = OrderedDict()
    score["base_plus_v7"] = 0.5 * (base_rank + v7_rank)
    for name, value in evidence.items():
        sionna_rank = e131.row_ranks(value, candidate_mask)
        score[f"base_v7_plus_{name}"] = (base_rank + v7_rank + sionna_rank) / 3.0
    selected = OrderedDict(
        (name, e131.top_k_mask(value, candidate_mask))
        for name, value in score.items()
    )
    if not np.array_equal(
        selected["base_plus_v7"], e131_metrics["local_selected"].astype(bool)
    ):
        raise RuntimeError("E131 base+V7 selection replay mismatch")

    ranking = OrderedDict()
    point_recall = {}
    for name, value in score.items():
        ranking[name], point_recall[name] = e131.ranking_metrics(
            value,
            selected[name],
            target_mass,
            candidate_mask,
            candidate_fold,
            query_site,
        )

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    true_coordinate = component["coordinates"][query_ids].astype(np.float32)
    true_coefficient = component["coefficients"][query_ids].astype(np.complex64)
    true_power = component["point_power"][query_ids].astype(np.float64)
    targets, assignment = e124.build_one_to_one_targets(
        candidate_coordinate,
        candidate_mask,
        true_coordinate,
        true_coefficient,
        true_power,
    )
    oracle = e131.selection_oracle_replay(
        selected,
        targets,
        query_ids,
        candidate_fold,
        true_coordinate,
        true_coefficient,
        true_power,
        device,
        args.batch_size,
    )
    actual = evaluate_actual_states(
        selected,
        query_ids,
        candidate_fold,
        candidate_mask,
        candidate_coordinate,
        true_coordinate,
        true_coefficient,
        true_power,
        device,
        args.batch_size,
    )

    primary = "base_v7_plus_corrected_full_one_to_one"
    baseline = "base_plus_v7"
    recall_gain = float(
        ranking[primary]["true_mass_recall"]
        - ranking[baseline]["true_mass_recall"]
    )
    fold_gain = e131.group_gain(
        ranking[primary]["fold_true_mass_recall"],
        ranking[baseline]["fold_true_mass_recall"],
    )
    site_gain = e131.group_gain(
        ranking[primary]["site_true_mass_recall"],
        ranking[baseline]["site_true_mass_recall"],
    )
    actual_gain = float(
        actual[primary]["all"]["score"] - actual[baseline]["all"]["score"]
    )
    actual_fold_gain = e133.fold_score_gains(actual[primary], actual[baseline])
    information_passed = bool(
        recall_gain >= MINIMUM_RECALL_GAIN
        and sum(float(row["gain"]) > 0.0 for row in fold_gain) >= 4
        and all(float(row["gain"]) > 0.0 for row in site_gain)
        and float(oracle[primary]["all"]["pas"])
        > float(oracle[baseline]["all"]["pas"])
        and float(oracle[primary]["all"]["pdp"])
        > float(oracle[baseline]["all"]["pdp"])
    )
    actual_passed = bool(
        actual_gain >= MINIMUM_ACTUAL_SCORE_GAIN
        and sum(float(row["gain"]) > 0.0 for row in actual_fold_gain) >= 4
        and float(actual[primary]["all"]["pas"])
        >= float(actual[baseline]["all"]["pas"])
        and float(actual[primary]["all"]["pdp"])
        >= float(actual[baseline]["all"]["pdp"])
    )
    gate = {
        "primary_variant": primary,
        "baseline_variant": baseline,
        "minimum_true_mass_recall_gain": MINIMUM_RECALL_GAIN,
        "minimum_actual_full_score_gain": MINIMUM_ACTUAL_SCORE_GAIN,
        "true_mass_recall_gain": recall_gain,
        "fold_true_mass_recall_gains": fold_gain,
        "site_true_mass_recall_gains": site_gain,
        "actual_full_score_gain": actual_gain,
        "actual_fold_score_gains": actual_fold_gain,
        "candidate_selection_information_passed": information_passed,
        "actual_predicted_state_rollout_passed": actual_passed,
        "passed": bool(information_passed and actual_passed),
    }
    if information_passed and actual_passed:
        next_action = "train_joint_selector_with_sionna_candidate_features"
    elif information_passed:
        next_action = "retain_sionna_identity_evidence_but_replace_independent_hard_selection"
    else:
        next_action = "stop_direct_sionna_candidate_selection"

    payload = {
        "experiment": VERSION,
        "status": "information_and_rollout_audit_complete",
        "primary_hypothesis": (
            "At the same query location, E98-aligned multi-seed Sionna paths provide "
            "candidate-level identity and mass evidence beyond the official-only V7 field."
        ),
        "protocol": {
            "canonical_fixed500": True,
            "query_count": int(len(query_ids)),
            "test_sionna_role": "diagnostic direct candidate evidence only",
            "official_held_h_role": "one-to-one labels and final scoring only",
            "e98_mapping_is_strict_oof": True,
            "top_k": e131.TOP_K,
            "match_bandwidth": MATCH_BANDWIDTH,
            "match_cost_limit": MATCH_COST_LIMIT,
            "fusion": "equal-rank base mass, V7 local energy, and Sionna evidence",
            "parameter_scan": False,
            "network_training": False,
        },
        "assignment": assignment,
        "sionna_evidence": evidence_summary,
        "ranking": ranking,
        "selection_isolated_true_state_replay": oracle,
        "actual_predicted_state_rollout": actual,
        "gate": gate,
        "decision": {
            "next_action": next_action,
            "candidate_or_zip_allowed": False,
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
        candidate_mask=candidate_mask,
        target_mass=target_mass,
        raw_sionna_coordinate=raw_coordinate,
        corrected_sionna_coordinate=corrected_coordinate,
        sionna_path_mask=path_mask,
        sionna_path_mass=path_mass,
        **{f"evidence_{name}": value for name, value in evidence.items()},
        **{f"selected_{name}": value for name, value in selected.items()},
        **{f"point_recall_{name}": value for name, value in point_recall.items()},
    )
    print(
        json.dumps(
            {
                "ranking": {
                    name: {
                        "true_mass_recall": value["true_mass_recall"],
                        "existence_recall": value["existence_recall"],
                        "point_average_precision": value["point_average_precision"],
                    }
                    for name, value in ranking.items()
                },
                "selection_isolated_score": {
                    name: value["all"] for name, value in oracle.items()
                },
                "actual_score": {name: value["all"] for name, value in actual.items()},
                "gate": gate,
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
