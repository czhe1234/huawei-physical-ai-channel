#!/usr/bin/env python3
"""Test whether the official-only V7 field improves candidate-path ranking."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e123_complete_path_variable_system as e123
import analyze_e124_one_to_one_path_identity_ceiling as e124
import train_e115_complete_path_state_operator as e115
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META


VERSION = "Round2-E131-v7-path-selection-evidence"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_path_selection_evidence"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_path_selection_evidence"
REPORT = ANALYSIS_DIR / "v7_path_selection_evidence.json"
METRICS = ARTIFACT_DIR / "v7_path_selection_evidence_metrics.npz"
V7_SUPPORT = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "oof_stacked_current_ad_f16.npy"
)

TOP_K = 15
PATCH_RADIUS = (1, 1, 2)
NEIGHBOR_COUNT = 4
NEIGHBOR_SCALE_M = 12.0
MINIMUM_LOCAL_RECALL_GAIN = 0.02
MINIMUM_CONTINUITY_RECALL_GAIN = 0.005
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def gather_candidate_cache(
    query_ids: np.ndarray, query_fold: np.ndarray
) -> dict[str, np.ndarray]:
    cache = e123.collect_cache_arrays(query_ids, query_fold)
    return {
        "coordinate": cache["coordinate"].astype(np.float32),
        "mask": cache["mask"].astype(bool),
        "base_mass": np.exp(
            np.clip(cache["base_log_mass"].astype(np.float32), -30.0, 20.0)
        ),
    }


def patch_energy(
    support: np.ndarray,
    coordinate: np.ndarray,
    mask: np.ndarray,
    source_rows: np.ndarray | None = None,
) -> np.ndarray:
    count, candidate_count, _ = coordinate.shape
    if source_rows is None:
        source_rows = np.arange(count, dtype=np.int64)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if source_rows.shape != (count,):
        raise ValueError("source_rows must contain one support row per query")

    center = np.rint(coordinate).astype(np.int64)
    rows = source_rows[:, None]
    output = np.zeros((count, candidate_count), dtype=np.float32)
    for horizontal in range(-PATCH_RADIUS[0], PATCH_RADIUS[0] + 1):
        h_index = (center[..., 0] + horizontal) % e115.MH
        for vertical in range(-PATCH_RADIUS[1], PATCH_RADIUS[1] + 1):
            v_index = (center[..., 1] + vertical) % e115.MV
            for delay in range(-PATCH_RADIUS[2], PATCH_RADIUS[2] + 1):
                d_index = (center[..., 2] + delay) % e115.S_SC
                output += np.asarray(
                    support[rows, h_index, v_index, d_index], dtype=np.float32
                )
    output *= mask
    return output


def nearest_query_neighbors(
    positions: np.ndarray,
    query_fold: np.ndarray,
    query_site: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(positions)
    neighbor_ids = np.full((count, NEIGHBOR_COUNT), -1, dtype=np.int64)
    neighbor_distance = np.full(
        (count, NEIGHBOR_COUNT), np.inf, dtype=np.float64
    )
    for fold in range(5):
        for site in np.unique(query_site):
            rows = np.flatnonzero((query_fold == fold) & (query_site == site))
            if len(rows) <= 1:
                continue
            delta = positions[rows, None, :2] - positions[None, rows, :2]
            distance = np.linalg.norm(delta, axis=2)
            np.fill_diagonal(distance, np.inf)
            take = min(NEIGHBOR_COUNT, len(rows) - 1)
            local = np.argpartition(distance, take - 1, axis=1)[:, :take]
            local_distance = np.take_along_axis(distance, local, axis=1)
            order = np.argsort(local_distance, axis=1)
            local = np.take_along_axis(local, order, axis=1)
            local_distance = np.take_along_axis(local_distance, order, axis=1)
            neighbor_ids[rows, :take] = rows[local]
            neighbor_distance[rows, :take] = local_distance
    if np.any(neighbor_ids[:, 0] < 0):
        raise RuntimeError("at least one query has no same-fold, same-site neighbor")
    return neighbor_ids, neighbor_distance


def continuity_energy(
    support: np.ndarray,
    coordinate: np.ndarray,
    mask: np.ndarray,
    local_energy: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weighted = np.zeros_like(local_energy, dtype=np.float64)
    total_weight = np.zeros(len(local_energy), dtype=np.float64)
    for index in range(NEIGHBOR_COUNT):
        valid = neighbor_ids[:, index] >= 0
        source = np.where(valid, neighbor_ids[:, index], np.arange(len(valid)))
        value = patch_energy(support, coordinate, mask, source)
        weight = np.where(
            valid,
            np.exp(-neighbor_distance[:, index] / NEIGHBOR_SCALE_M),
            0.0,
        )
        weighted += value * weight[:, None]
        total_weight += weight
    neighbor_energy = np.divide(
        weighted,
        np.maximum(total_weight[:, None], TINY),
        out=np.zeros_like(weighted),
    ).astype(np.float32)
    continuity = np.sqrt(
        np.maximum(local_energy, 0.0) * np.maximum(neighbor_energy, 0.0)
    ).astype(np.float32)
    continuity *= mask
    return neighbor_energy, continuity


def row_ranks(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(value, dtype=np.float32)
    for row in range(len(value)):
        ids = np.flatnonzero(mask[row])
        if len(ids):
            output[row, ids] = (
                rankdata(value[row, ids], method="average") / float(len(ids))
            ).astype(np.float32)
    return output


def top_k_mask(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.zeros_like(mask)
    for row in range(len(score)):
        ids = np.flatnonzero(mask[row])
        take = min(TOP_K, len(ids))
        if take:
            local = np.argpartition(score[row, ids], -take)[-take:]
            selected[row, ids[local]] = True
    return selected


def mean_by_group(
    value: np.ndarray,
    valid: np.ndarray,
    group: np.ndarray,
) -> list[dict[str, float | int]]:
    output = []
    for group_id in np.unique(group):
        rows = valid & (group == group_id)
        output.append(
            {
                "group": int(group_id),
                "count": int(rows.sum()),
                "value": float(np.mean(value[rows])) if np.any(rows) else 0.0,
            }
        )
    return output


def ranking_metrics(
    score: np.ndarray,
    selected: np.ndarray,
    target_mass: np.ndarray,
    mask: np.ndarray,
    query_fold: np.ndarray,
    query_site: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    target_total = target_mass.sum(axis=1, dtype=np.float64)
    valid = target_total > TINY
    selected_mass = (target_mass * selected).sum(axis=1, dtype=np.float64)
    mass_recall = np.divide(
        selected_mass,
        np.maximum(target_total, TINY),
        out=np.zeros_like(selected_mass),
    )
    positive = target_mass > 0.0
    positive_count = positive.sum(axis=1)
    selected_positive = (positive & selected).sum(axis=1)
    existence_recall = np.divide(
        selected_positive,
        np.maximum(positive_count, 1),
        out=np.zeros_like(selected_positive, dtype=np.float64),
    )
    selected_count = selected.sum(axis=1)
    precision = np.divide(
        selected_positive,
        np.maximum(selected_count, 1),
        out=np.zeros_like(selected_positive, dtype=np.float64),
    )
    point_ap = np.zeros(len(score), dtype=np.float64)
    for row in np.flatnonzero(valid):
        ids = np.flatnonzero(mask[row])
        labels = positive[row, ids]
        if np.any(labels) and np.any(~labels):
            point_ap[row] = average_precision_score(labels, score[row, ids])
    return (
        {
            "nonzero_count": int(valid.sum()),
            "top_k": TOP_K,
            "true_mass_recall": float(np.mean(mass_recall[valid])),
            "existence_recall": float(np.mean(existence_recall[valid])),
            "selection_precision": float(np.mean(precision[valid])),
            "point_average_precision": float(np.mean(point_ap[valid])),
            "fold_true_mass_recall": mean_by_group(
                mass_recall, valid, query_fold
            ),
            "site_true_mass_recall": mean_by_group(
                mass_recall, valid, query_site
            ),
        },
        mass_recall,
    )


def group_gain(
    candidate: list[dict[str, float | int]],
    baseline: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    if len(candidate) != len(baseline):
        raise ValueError("group summaries are not aligned")
    return [
        {
            "group": int(left["group"]),
            "gain": float(left["value"]) - float(right["value"]),
        }
        for left, right in zip(candidate, baseline, strict=True)
    ]


def selection_oracle_replay(
    selected: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    query_ids: np.ndarray,
    query_fold: np.ndarray,
    true_coordinate: np.ndarray,
    true_coefficient: np.ndarray,
    true_power: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    nonzero = true_power > TINY
    output = {}
    for name, selection in selected.items():
        state = e123.PathState(
            coordinate=targets["coordinate"],
            mass=targets["mass_absolute"] * selection,
            port=targets["port"],
            phase=targets["phase"],
            power=true_power,
            enforce_power=False,
        )
        output[name] = e123.evaluate_state(
            state,
            query_ids,
            query_fold,
            nonzero,
            (true_coordinate, true_coefficient),
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

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    v7_fold = metadata["fold_id"].astype(np.int8)
    full_fold = np.load(e115.FULL_FOLD).astype(np.int8)
    candidate_fold = full_fold[query_ids]

    setup, train_positions, _, train_site, _ = load_positions()
    del setup
    query_positions = train_positions[query_ids]
    query_site = train_site[query_ids].astype(np.int8)
    cache = gather_candidate_cache(query_ids, candidate_fold)
    coordinate = cache["coordinate"]
    mask = cache["mask"]
    base_mass = cache["base_mass"] * mask

    support = np.load(V7_SUPPORT, mmap_mode="r")
    expected_shape = (len(query_ids), e115.MH, e115.MV, e115.S_SC)
    if support.shape != expected_shape:
        raise RuntimeError(f"unexpected V7 support shape: {support.shape}")
    local_energy = patch_energy(support, coordinate, mask)
    neighbor_ids, neighbor_distance = nearest_query_neighbors(
        query_positions, v7_fold, query_site
    )
    neighbor_energy, continuity = continuity_energy(
        support,
        coordinate,
        mask,
        local_energy,
        neighbor_ids,
        neighbor_distance,
    )

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    true_coordinate = component["coordinates"][query_ids].astype(np.float32)
    true_coefficient = component["coefficients"][query_ids].astype(np.complex64)
    true_power = component["point_power"][query_ids].astype(np.float64)
    targets, assignment = e124.build_one_to_one_targets(
        coordinate,
        mask,
        true_coordinate,
        true_coefficient,
        true_power,
    )
    target_mass = targets["mass_absolute"]

    base_rank = row_ranks(base_mass, mask)
    local_rank = row_ranks(local_energy, mask)
    continuity_rank = row_ranks(continuity, mask)
    score = {
        "baseline": base_rank,
        "base_plus_v7_local": 0.5 * (base_rank + local_rank),
        "base_plus_v7_continuity": (
            base_rank + local_rank + continuity_rank
        )
        / 3.0,
    }
    selected = {name: top_k_mask(value, mask) for name, value in score.items()}
    ranking = {}
    point_recall = {}
    for name in score:
        ranking[name], point_recall[name] = ranking_metrics(
            score[name],
            selected[name],
            target_mass,
            mask,
            candidate_fold,
            query_site,
        )

    oracle = selection_oracle_replay(
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
    baseline = ranking["baseline"]
    local = ranking["base_plus_v7_local"]
    continuous = ranking["base_plus_v7_continuity"]
    local_fold_gain = group_gain(
        local["fold_true_mass_recall"], baseline["fold_true_mass_recall"]
    )
    local_site_gain = group_gain(
        local["site_true_mass_recall"], baseline["site_true_mass_recall"]
    )
    continuity_fold_gain = group_gain(
        continuous["fold_true_mass_recall"], local["fold_true_mass_recall"]
    )
    local_recall_gain = float(local["true_mass_recall"]) - float(
        baseline["true_mass_recall"]
    )
    continuity_recall_gain = float(continuous["true_mass_recall"]) - float(
        local["true_mass_recall"]
    )
    baseline_official = oracle["baseline"]["all"]
    local_official = oracle["base_plus_v7_local"]["all"]
    continuous_official = oracle["base_plus_v7_continuity"]["all"]
    local_passed = bool(
        local_recall_gain >= MINIMUM_LOCAL_RECALL_GAIN
        and sum(float(row["gain"]) > 0.0 for row in local_fold_gain) >= 4
        and all(float(row["gain"]) > 0.0 for row in local_site_gain)
        and float(local_official["pas"]) > float(baseline_official["pas"])
        and float(local_official["pdp"]) > float(baseline_official["pdp"])
    )
    continuity_passed = bool(
        continuity_recall_gain >= MINIMUM_CONTINUITY_RECALL_GAIN
        and sum(float(row["gain"]) > 0.0 for row in continuity_fold_gain) >= 3
        and float(continuous_official["pas"]) >= float(local_official["pas"])
        and float(continuous_official["pdp"]) >= float(local_official["pdp"])
    )
    gate = {
        "minimum_local_true_mass_recall_gain": MINIMUM_LOCAL_RECALL_GAIN,
        "minimum_continuity_true_mass_recall_gain": (
            MINIMUM_CONTINUITY_RECALL_GAIN
        ),
        "local_true_mass_recall_gain": local_recall_gain,
        "continuity_true_mass_recall_gain": continuity_recall_gain,
        "local_fold_gains": local_fold_gain,
        "local_site_gains": local_site_gain,
        "continuity_fold_gains": continuity_fold_gain,
        "local_information_passed": local_passed,
        "continuity_information_passed": continuity_passed,
        "passed": local_passed,
    }
    if not local_passed:
        next_action = "stop_v7_candidate_evidence_branch"
    elif continuity_passed:
        next_action = "train_selector_with_local_and_continuity_evidence"
    else:
        next_action = "train_selector_with_local_v7_evidence_only"

    payload = {
        "experiment": VERSION,
        "status": "information_audit_complete",
        "primary_hypothesis": (
            "The official-only V7 angle-delay field contains enough hidden-query "
            "evidence to rank candidate paths more accurately than the existing "
            "candidate mass prior, before any larger selector is trained."
        ),
        "protocol": {
            "query_count": int(len(query_ids)),
            "canonical_fixed500": True,
            "v7_is_strict_oof_official_h_only": True,
            "v7_and_candidate_system_use_separate_strict_oof_partitions": True,
            "v7_fold_role": "same-model spatial continuity neighbors",
            "candidate_fold_role": "candidate cache selection and fold reporting",
            "held_h_used_only_for_one_to_one_labels_and_scoring": True,
            "network_training": False,
            "parameter_scan": False,
            "top_k": TOP_K,
            "patch_radius_h_v_delay": list(PATCH_RADIUS),
            "same_fold_same_site_neighbors": NEIGHBOR_COUNT,
            "neighbor_scale_m": NEIGHBOR_SCALE_M,
            "fusion": "equal-weight within-query percentile ranks",
        },
        "assignment": assignment,
        "ranking": ranking,
        "selection_isolated_true_state_replay": oracle,
        "gate": gate,
        "decision": {
            "next_action": next_action,
            "large_training_authorized": bool(local_passed),
            "use_continuity_if_training": bool(continuity_passed),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        v7_fold=v7_fold,
        candidate_fold=candidate_fold,
        query_site=query_site,
        candidate_mask=mask,
        target_mass=target_mass.astype(np.float32),
        base_mass=base_mass.astype(np.float32),
        v7_local_energy=local_energy,
        v7_neighbor_energy=neighbor_energy,
        v7_continuity_energy=continuity,
        baseline_selected=selected["baseline"],
        local_selected=selected["base_plus_v7_local"],
        continuity_selected=selected["base_plus_v7_continuity"],
        baseline_point_mass_recall=point_recall["baseline"],
        local_point_mass_recall=point_recall["base_plus_v7_local"],
        continuity_point_mass_recall=point_recall[
            "base_plus_v7_continuity"
        ],
        neighbor_ids=neighbor_ids.astype(np.int32),
        neighbor_distance=neighbor_distance.astype(np.float32),
    )
    print(
        json.dumps(
            {
                "ranking": {
                    name: {
                        "true_mass_recall": value["true_mass_recall"],
                        "existence_recall": value["existence_recall"],
                        "point_average_precision": value[
                            "point_average_precision"
                        ],
                    }
                    for name, value in ranking.items()
                },
                "selection_isolated_official": {
                    name: value["all"] for name, value in oracle.items()
                },
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
