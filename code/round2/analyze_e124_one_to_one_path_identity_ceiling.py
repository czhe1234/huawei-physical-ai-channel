#!/usr/bin/env python3
"""Audit one-to-one candidate/true-path identity supervision for Round 2."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e121_standalone_unique_path_carrier as e121
import analyze_e123_complete_path_variable_system as e123
import analyze_global_local_path_lifecycle as e79
import train_e115_complete_path_state_operator as e115
from project_paths import PATHS
from run_path_component_transport import component_energy


VERSION = "Round2-E124-one-to-one-path-identity-ceiling"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "one_to_one_path_identity_ceiling"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_one_to_one_path_identity_ceiling"
REPORT = ANALYSIS_DIR / "one_to_one_path_identity_ceiling.json"
METRICS = ARTIFACT_DIR / "one_to_one_path_identity_ceiling_metrics.npz"
E123_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_complete_path_variable_system"
    / "complete_path_variable_system.json"
)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def normalize_port(value: np.ndarray) -> np.ndarray:
    energy = np.abs(value) ** 2
    total = energy.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        energy,
        np.maximum(total, TINY),
        out=np.full_like(energy, 1.0 / e115.PORTS, dtype=np.float64),
    ).astype(np.float32)


def unit_complex(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    return np.divide(
        value,
        np.maximum(magnitude, 1e-12),
        out=np.ones_like(value, dtype=np.complex64),
    ).astype(np.complex64)


def build_one_to_one_targets(
    base_coordinate: np.ndarray,
    mask: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    truth_power: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    count, candidates, _ = base_coordinate.shape
    target_mass = np.zeros((count, candidates), dtype=np.float32)
    target_coordinate = np.array(base_coordinate, copy=True)
    target_port = np.full(
        (count, candidates, e115.PORTS), 1.0 / e115.PORTS, dtype=np.float32
    )
    target_phase = np.ones(
        (count, candidates, e115.PORTS), dtype=np.complex64
    )
    matched_count = np.zeros(count, dtype=np.int16)
    matched_fraction = np.zeros(count, dtype=np.float64)
    total_fraction = np.zeros(count, dtype=np.float64)
    conflict_fraction = np.zeros(count, dtype=np.float64)
    matched_cost_parts: list[np.ndarray] = []
    bandwidth = float(e115.Config().soft_bandwidth)
    limit = float(e115.Config().soft_cost_limit)

    for row in range(count):
        candidate_ids = np.flatnonzero(mask[row])
        energy = component_energy(truth_coefficient[row])
        fraction = energy / max(float(truth_power[row]), TINY)
        truth_ids = np.flatnonzero(fraction >= 1e-5)
        total_fraction[row] = float(np.sum(fraction[truth_ids]))
        if len(candidate_ids) == 0 or len(truth_ids) == 0:
            continue
        candidate = base_coordinate[row, candidate_ids]
        truth = truth_coordinate[row, truth_ids]
        cost = e79.support_cost(candidate, truth)
        utility = fraction[truth_ids][None] * np.exp(-0.5 * cost / bandwidth)
        utility[cost > limit] = 0.0
        candidate_local, truth_local = linear_sum_assignment(-utility)
        accepted = utility[candidate_local, truth_local] > 0.0
        candidate_local = candidate_local[accepted]
        truth_local = truth_local[accepted]
        if len(candidate_local) == 0:
            continue
        assigned_candidate = candidate_ids[candidate_local]
        assigned_truth = truth_ids[truth_local]
        coefficient = truth_coefficient[row, assigned_truth]
        target_mass[row, assigned_candidate] = fraction[assigned_truth]
        target_coordinate[row, assigned_candidate] = truth_coordinate[
            row, assigned_truth
        ]
        target_port[row, assigned_candidate] = normalize_port(coefficient)
        target_phase[row, assigned_candidate] = unit_complex(coefficient)
        matched_count[row] = len(assigned_candidate)
        matched_fraction[row] = float(np.sum(fraction[assigned_truth]))
        matched_cost_parts.append(cost[candidate_local, truth_local])

        nearest = np.argmin(cost, axis=0)
        unique, nearest_count = np.unique(nearest, return_counts=True)
        conflicting_candidates = unique[nearest_count > 1]
        if len(conflicting_candidates):
            conflict = np.isin(nearest, conflicting_candidates)
            conflict_fraction[row] = float(np.sum(fraction[truth_ids][conflict]))

    normalized_mass = e123.normalize_mass(target_mass, mask)
    matched_cost = (
        np.concatenate(matched_cost_parts)
        if matched_cost_parts
        else np.empty(0, dtype=np.float64)
    )
    nonzero = truth_power > TINY
    audit = {
        "objective": "maximize mass-weighted coordinate-kernel utility",
        "one_candidate_at_most_one_truth_path": True,
        "one_truth_path_at_most_one_candidate": True,
        "cost_limit": limit,
        "bandwidth": bandwidth,
        "matched_path_count_nonzero_mean": float(np.mean(matched_count[nonzero])),
        "matched_path_count_nonzero_median": float(np.median(matched_count[nonzero])),
        "matched_coefficient_mass_fraction_nonzero_mean": float(
            np.mean(matched_fraction[nonzero])
        ),
        "matched_fraction_of_available_true_mass": float(
            np.sum(matched_fraction) / max(float(np.sum(total_fraction)), TINY)
        ),
        "nearest_candidate_identity_conflict_mass_nonzero_mean": float(
            np.mean(conflict_fraction[nonzero])
        ),
        "matched_cost_mean": float(np.mean(matched_cost)) if len(matched_cost) else 0.0,
        "matched_cost_p90": float(np.quantile(matched_cost, 0.90))
        if len(matched_cost)
        else 0.0,
    }
    return (
        {
            "mass_absolute": target_mass,
            "mass_normalized": normalized_mass,
            "coordinate": target_coordinate,
            "port": target_port,
            "phase": target_phase,
            "matched_count": matched_count,
            "matched_fraction": matched_fraction,
            "conflict_fraction": conflict_fraction,
        },
        audit,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    endpoint = np.load(e121.ENDPOINT, allow_pickle=False)
    query_ids = endpoint["query_ids"].astype(np.int64)
    full_fold = endpoint["full_fold"].astype(np.int8)
    cache = e123.collect_cache_arrays(query_ids, full_fold)
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    true_coordinate = component["coordinates"][query_ids].astype(np.float32)
    true_coefficient = component["coefficients"][query_ids].astype(np.complex64)
    true_power = component["point_power"][query_ids].astype(np.float64)
    nonzero = true_power > TINY
    base_coordinate = cache["coordinate"].astype(np.float32)
    mask = cache["mask"].astype(bool)
    base_power = np.exp(
        np.clip(cache["base_log_power"].astype(np.float64), -80.0, 80.0)
    )

    targets, assignment = build_one_to_one_targets(
        base_coordinate,
        mask,
        true_coordinate,
        true_coefficient,
        true_power,
    )
    states = {
        "one_to_one_normalized_base_power": e123.PathState(
            targets["coordinate"],
            targets["mass_normalized"],
            targets["port"],
            targets["phase"],
            base_power,
        ),
        "one_to_one_normalized_true_power": e123.PathState(
            targets["coordinate"],
            targets["mass_normalized"],
            targets["port"],
            targets["phase"],
            true_power,
        ),
        "one_to_one_absolute": e123.PathState(
            targets["coordinate"],
            targets["mass_absolute"],
            targets["port"],
            targets["phase"],
            true_power,
            enforce_power=False,
        ),
        "one_to_one_absolute_plus_true_residual": e123.PathState(
            targets["coordinate"],
            targets["mass_absolute"],
            targets["port"],
            targets["phase"],
            true_power,
            enforce_power=False,
            add_true_residual=True,
        ),
    }
    true_components = (true_coordinate, true_coefficient)
    results = {
        name: e123.evaluate_state(
            state,
            query_ids,
            full_fold,
            nonzero,
            true_components,
            device,
            args.batch_size,
        )
        for name, state in states.items()
    }
    e123_report = json.loads(E123_REPORT.read_text(encoding="utf-8"))
    soft_absolute = float(
        e123_report["variant_results"]["absolute_resolved_state_oracle"]["all"][
            "score"
        ]
    )
    soft_residual = float(
        e123_report["variant_results"][
            "absolute_resolved_plus_true_residual"
        ]["all"]["score"]
    )
    soft_normalized = float(
        e123_report["variant_results"]["all_path_state_true_power"]["all"][
            "score"
        ]
    )
    hard_absolute = float(results["one_to_one_absolute"]["all"]["score"])
    hard_residual = float(
        results["one_to_one_absolute_plus_true_residual"]["all"]["score"]
    )
    hard_normalized = float(
        results["one_to_one_normalized_true_power"]["all"]["score"]
    )
    gate = {
        "minimum_absolute_gain_over_soft_assignment": 0.02,
        "minimum_residual_gain_over_soft_assignment": 0.02,
        "minimum_normalized_gain_over_soft_assignment": 0.01,
        "absolute_gain_over_soft_assignment": hard_absolute - soft_absolute,
        "residual_gain_over_soft_assignment": hard_residual - soft_residual,
        "normalized_gain_over_soft_assignment": hard_normalized - soft_normalized,
    }
    gate["passed"] = bool(
        gate["absolute_gain_over_soft_assignment"]
        >= gate["minimum_absolute_gain_over_soft_assignment"]
        and gate["residual_gain_over_soft_assignment"]
        >= gate["minimum_residual_gain_over_soft_assignment"]
        and gate["normalized_gain_over_soft_assignment"]
        >= gate["minimum_normalized_gain_over_soft_assignment"]
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "One-to-one path identity targets remove the phase and coefficient mixing "
            "introduced by soft many-to-many assignment and materially raise the "
            "candidate-constrained complete-channel ceiling."
        ),
        "strict_protocol": {
            "query_count": int(len(query_ids)),
            "candidate_donors_exclude_outer_and_inner": True,
            "held_truth_used_only_for_assignment_oracle_and_scoring": True,
            "v12_or_sionna_used_as_input": False,
        },
        "assignment": assignment,
        "results": results,
        "soft_assignment_references": {
            "normalized_true_power": soft_normalized,
            "absolute": soft_absolute,
            "absolute_plus_true_residual": soft_residual,
        },
        "gate": gate,
        "decision": {
            "use_one_to_one_identity_supervision": bool(gate["passed"]),
            "next_experiment_if_passed": (
                "Train a hierarchical identity-first decoder with Hungarian targets, "
                "teacher-forced state heads, and rollout evaluation."
            ),
            "next_experiment_if_failed": (
                "Retain soft targets and redesign the candidate family before training."
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    names = tuple(results)
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        full_fold=full_fold,
        variant_names=np.asarray(names, dtype="U64"),
        variant_pas=np.asarray(
            [results[name]["all"]["pas"] for name in names], dtype=np.float64
        ),
        variant_pdp=np.asarray(
            [results[name]["all"]["pdp"] for name in names], dtype=np.float64
        ),
        variant_nmse=np.asarray(
            [results[name]["all"]["nmse"] for name in names], dtype=np.float64
        ),
        variant_score=np.asarray(
            [results[name]["all"]["score"] for name in names], dtype=np.float64
        ),
        matched_count=targets["matched_count"],
        matched_fraction=targets["matched_fraction"],
        conflict_fraction=targets["conflict_fraction"],
    )
    print(
        json.dumps(
            {
                "assignment": assignment,
                "results": {name: value["all"] for name, value in results.items()},
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
