#!/usr/bin/env python3
"""Audit local and Sionna path-candidate information against official components."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import time

import numpy as np
from sklearn.metrics import average_precision_score

import analyze_global_local_path_lifecycle as e79
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E84-path-candidate-information-audit"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "path_candidate_information_audit"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_path_candidate_information_audit"
REPORT = ANALYSIS_DIR / "path_candidate_information_audit.json"
PER_POINT = ARTIFACT_DIR / "per_point_candidate_information.npz"

COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
SIONNA_STATE = (
    PATHS.artifacts
    / "round2"
    / "sionna_multiseed"
    / "primary_d16_base_v4lambda_1m"
    / "full"
    / "state.json"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FULL_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
FULL_REGIME = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_regime_train_n4000.npy"
)

MH = 16.0
MV = 8.0
S_SC = 192.0
PERIODS = np.asarray((MH, MV, S_SC), dtype=np.float32)
SUPPORT_SCALES = np.asarray((1.5, 1.0, 2.0), dtype=np.float32)
TIGHT_COST = 0.25
SOFT_BANDWIDTH = 0.25
TINY = 1e-30
REGIME_NAMES = ("clear", "light", "medium", "heavy")


@dataclass(frozen=True)
class Config:
    budgets: tuple[int, ...] = (24, 48, 96, 128)
    thresholds: tuple[float, ...] = (0.25, 1.0, 4.0, 9.0)
    shuffles: int = 12
    random_state: int = 2026080794


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffles", type=int, default=12)
    return parser.parse_args()


def periodic_delta(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    return (left - right + period / 2.0) % period - period / 2.0


def support_cost(
    candidates: np.ndarray,
    truth: np.ndarray,
    axes: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    cost = np.zeros((len(candidates), len(truth)), dtype=np.float32)
    for axis in axes:
        delta = periodic_delta(
            candidates[:, None, axis], truth[None, :, axis], float(PERIODS[axis])
        )
        cost += np.square(delta / SUPPORT_SCALES[axis], dtype=np.float32)
    return cost


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> list[float]:
    if len(values) == 0 or float(np.sum(weights)) <= 0.0:
        return [math.nan for _ in quantiles]
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    cumulative /= cumulative[-1]
    return [
        float(sorted_values[min(np.searchsorted(cumulative, q), len(values) - 1)])
        for q in quantiles
    ]


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def rank_percentile(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(score, dtype=np.float32)
    for row in range(len(score)):
        valid = np.flatnonzero(mask[row])
        if len(valid) == 0:
            continue
        order = valid[np.argsort(-score[row, valid], kind="stable")]
        output[row, order] = 1.0 - np.arange(len(order), dtype=np.float32) / max(
            len(order) - 1, 1
        )
    return output


def sionna_coordinates(
    tau: np.ndarray,
    angles: np.ndarray,
    site: np.ndarray,
    configuration: dict[str, object],
    vertical_alignment: float = 2.0,
) -> np.ndarray:
    yaw_values = np.radians(
        np.asarray(configuration["site_yaw_deg"], dtype=np.float32)
    )
    tilt_values = np.radians(
        np.asarray(configuration["site_tilt_deg"], dtype=np.float32)
    )
    yaw = yaw_values[site][:, None]
    tilt = tilt_values[site][:, None]
    theta_t = np.asarray(angles[..., 0], dtype=np.float32)
    phi_t = np.asarray(angles[..., 1], dtype=np.float32)
    horizontal_spacing = float(configuration["tx_horizontal_spacing"])
    vertical_spacing = float(configuration["tx_vertical_spacing"])
    horizontal = (
        MH * horizontal_spacing * np.sin(theta_t) * np.sin(phi_t - yaw)
    ) % MH
    vertical = (
        MV
        * vertical_spacing
        * (
            np.sin(tilt) * np.sin(theta_t) * np.cos(phi_t - yaw)
            - np.cos(tilt) * np.cos(theta_t)
        )
        + vertical_alignment
    ) % MV
    delay = (np.asarray(tau, dtype=np.float32) * 30_000.0 * S_SC) % S_SC
    return np.stack((horizontal, vertical, delay), axis=2).astype(np.float32)


def soft_target_mass(
    cost: np.ndarray, truth_mass: np.ndarray, threshold: float = TIGHT_COST
) -> np.ndarray:
    if cost.size == 0:
        return np.zeros(cost.shape[0], dtype=np.float64)
    kernel = np.exp(-0.5 * cost / SOFT_BANDWIDTH, dtype=np.float64)
    kernel[cost > threshold] = 0.0
    column_total = kernel.sum(axis=0, keepdims=True)
    responsibility = np.divide(
        kernel,
        np.maximum(column_total, TINY),
        out=np.zeros_like(kernel),
        where=column_total > 0.0,
    )
    return responsibility @ truth_mass


def evaluate_dense_candidates(
    name: str,
    coordinates: np.ndarray,
    mask: np.ndarray,
    rankings: dict[str, np.ndarray],
    truth_coordinates: np.ndarray,
    truth_mass: np.ndarray,
    truth_valid: np.ndarray,
    budgets: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    accumulators = {
        (ranking, budget, threshold): [0.0, 0.0, []]
        for ranking in rankings
        for budget in budgets
        for threshold in thresholds
    }
    dimension_mass = {"delay": 0.0, "horizontal_delay": 0.0, "full": 0.0}
    total_truth_mass = 0.0
    supported_truth_mass = 0.0
    ambiguity_weighted = 0.0
    candidate_positive = 0
    candidate_total = 0
    nearest_costs: list[np.ndarray] = []
    nearest_weights: list[np.ndarray] = []
    oracle_mass = {budget: 0.0 for budget in budgets}
    raw_scores: dict[str, list[np.ndarray]] = {ranking: [] for ranking in rankings}
    labels: list[np.ndarray] = []
    positive_log_prediction: dict[str, list[np.ndarray]] = {
        ranking: [] for ranking in rankings
    }
    positive_log_target: list[np.ndarray] = []
    point_truth_mass = np.zeros(len(coordinates), dtype=np.float64)
    point_tight_coverage = np.full(len(coordinates), np.nan, dtype=np.float32)
    point_ambiguity = np.full(len(coordinates), np.nan, dtype=np.float32)
    point_energy_cosine = {
        ranking: np.full(len(coordinates), np.nan, dtype=np.float32)
        for ranking in rankings
    }
    point_oracle24 = np.full(len(coordinates), np.nan, dtype=np.float32)

    for row in range(len(coordinates)):
        truth_ids = np.flatnonzero(truth_valid[row])
        candidate_ids = np.flatnonzero(mask[row])
        if len(truth_ids) == 0:
            continue
        local_truth = truth_coordinates[row, truth_ids]
        local_mass = truth_mass[row, truth_ids].astype(np.float64)
        mass_total = float(np.sum(local_mass))
        if mass_total <= 0.0:
            continue
        point_truth_mass[row] = mass_total
        total_truth_mass += mass_total
        if len(candidate_ids) == 0:
            point_tight_coverage[row] = 0.0
            point_ambiguity[row] = 0.0
            point_oracle24[row] = 0.0
            for ranking in rankings:
                point_energy_cosine[ranking][row] = 0.0
            for key in accumulators:
                accumulators[key][1] += mass_total
                accumulators[key][2].append(0.0)
            continue
        local_candidates = coordinates[row, candidate_ids]
        cost = support_cost(local_candidates, local_truth)
        minimum = np.min(cost, axis=0)
        nearest_costs.append(minimum)
        nearest_weights.append(local_mass)
        tight_covered = minimum <= TIGHT_COST
        tight_mass = float(np.sum(local_mass[tight_covered]))
        supported_truth_mass += tight_mass
        point_tight_coverage[row] = tight_mass / mass_total
        ambiguity = np.sum(cost <= TIGHT_COST, axis=0)
        if np.any(tight_covered):
            local_ambiguity = float(
                np.sum(local_mass[tight_covered] * ambiguity[tight_covered])
                / max(tight_mass, TINY)
            )
            point_ambiguity[row] = local_ambiguity
            ambiguity_weighted += local_ambiguity * tight_mass
        candidate_near_truth = np.min(cost, axis=1) <= TIGHT_COST
        candidate_positive += int(np.sum(candidate_near_truth))
        candidate_total += len(candidate_ids)

        for key, axes in (
            ("delay", (2,)),
            ("horizontal_delay", (0, 2)),
            ("full", (0, 1, 2)),
        ):
            marginal_cost = support_cost(local_candidates, local_truth, axes)
            covered = np.min(marginal_cost, axis=0) <= TIGHT_COST
            dimension_mass[key] += float(np.sum(local_mass[covered]))

        target = soft_target_mass(cost, local_mass)
        target_total = float(np.sum(target))
        labels.append(target > 0.0)
        positive = target > 0.0
        if np.any(positive):
            normalized_target = target / max(target_total, TINY)
            positive_log_target.append(np.log(np.maximum(normalized_target[positive], 1e-12)))
        for ranking, score in rankings.items():
            local_score = np.maximum(score[row, candidate_ids].astype(np.float64), 0.0)
            raw_scores[ranking].append(local_score)
            point_energy_cosine[ranking][row] = safe_cosine(local_score, target)
            if np.any(positive):
                normalized_score = local_score / max(float(np.sum(local_score)), TINY)
                positive_log_prediction[ranking].append(
                    np.log(np.maximum(normalized_score[positive], 1e-12))
                )
            order = np.argsort(-score[row, candidate_ids], kind="stable")
            for budget in budgets:
                selected = order[: min(budget, len(order))]
                selected_cost = cost[selected]
                for threshold in thresholds:
                    covered = np.min(selected_cost, axis=0) <= threshold
                    covered_mass = float(np.sum(local_mass[covered]))
                    bucket = accumulators[(ranking, budget, threshold)]
                    bucket[0] += covered_mass
                    bucket[1] += mass_total
                    bucket[2].append(covered_mass / mass_total)
        target_order = np.argsort(-target, kind="stable")
        for budget in budgets:
            retained = float(np.sum(target[target_order[: min(budget, len(target_order))]]))
            oracle_mass[budget] += retained
            if budget == 24:
                point_oracle24[row] = retained / mass_total

    coverage_rows = []
    for (ranking, budget, threshold), (covered, total, points) in accumulators.items():
        coverage_rows.append(
            {
                "ranking": ranking,
                "budget": budget,
                "threshold": threshold,
                "mass_coverage": covered / max(total, TINY),
                "mean_point_coverage": float(np.mean(points)) if points else math.nan,
            }
        )
    coverage_rows.sort(
        key=lambda row: (row["ranking"], row["budget"], row["threshold"])
    )
    flat_labels = np.concatenate(labels) if labels else np.empty(0, dtype=bool)
    calibration = {}
    for ranking in rankings:
        flat_score = (
            np.concatenate(raw_scores[ranking])
            if raw_scores[ranking]
            else np.empty(0, dtype=np.float64)
        )
        log_prediction = (
            np.concatenate(positive_log_prediction[ranking])
            if positive_log_prediction[ranking]
            else np.empty(0, dtype=np.float64)
        )
        log_target = (
            np.concatenate(positive_log_target)
            if positive_log_target
            else np.empty(0, dtype=np.float64)
        )
        correlation = (
            float(np.corrcoef(log_prediction, log_target)[0, 1])
            if len(log_prediction) >= 2
            and np.std(log_prediction) > 0.0
            and np.std(log_target) > 0.0
            else math.nan
        )
        valid_point_cosine = point_energy_cosine[ranking][
            np.isfinite(point_energy_cosine[ranking])
        ]
        calibration[ranking] = {
            "candidate_average_precision": (
                float(average_precision_score(flat_labels, flat_score))
                if len(flat_labels) and np.any(flat_labels)
                else math.nan
            ),
            "mean_point_target_cosine": (
                float(np.mean(valid_point_cosine))
                if len(valid_point_cosine)
                else math.nan
            ),
            "positive_log_mass_pearson": correlation,
        }
    nearest_value = np.concatenate(nearest_costs) if nearest_costs else np.empty(0)
    nearest_weight = np.concatenate(nearest_weights) if nearest_weights else np.empty(0)
    report = {
        "name": name,
        "point_count": len(coordinates),
        "nonzero_truth_points": int(np.sum(point_truth_mass > 0.0)),
        "total_truth_mass": total_truth_mass,
        "coverage": coverage_rows,
        "all_candidate_tight_mass_coverage": supported_truth_mass
        / max(total_truth_mass, TINY),
        "dimension_tight_mass_coverage": {
            key: value / max(total_truth_mass, TINY)
            for key, value in dimension_mass.items()
        },
        "mass_weighted_candidate_ambiguity": ambiguity_weighted
        / max(supported_truth_mass, TINY),
        "candidate_tight_precision": candidate_positive / max(candidate_total, 1),
        "nearest_cost_weighted_quantiles": dict(
            zip(
                ("p50", "p75", "p90", "p95", "p99"),
                weighted_quantile(
                    nearest_value,
                    nearest_weight,
                    (0.50, 0.75, 0.90, 0.95, 0.99),
                ),
            )
        ),
        "oracle_soft_mass_retention": {
            str(budget): value / max(total_truth_mass, TINY)
            for budget, value in oracle_mass.items()
        },
        "energy_calibration": calibration,
    }
    per_point = {
        "truth_mass": point_truth_mass,
        "tight_coverage": point_tight_coverage,
        "ambiguity": point_ambiguity,
        "oracle24_mass_retention": point_oracle24,
        **{
            f"{ranking}_energy_cosine": value
            for ranking, value in point_energy_cosine.items()
        },
    }
    return report, per_point


def tight_mass_coverage(
    coordinates: np.ndarray,
    mask: np.ndarray,
    truth_coordinates: np.ndarray,
    truth_mass: np.ndarray,
    truth_valid: np.ndarray,
    axes: tuple[int, ...] = (0, 1, 2),
) -> float:
    covered_mass = 0.0
    total_mass = 0.0
    for row in range(len(coordinates)):
        truth_ids = np.flatnonzero(truth_valid[row])
        candidate_ids = np.flatnonzero(mask[row])
        if len(truth_ids) == 0:
            continue
        local_mass = truth_mass[row, truth_ids].astype(np.float64)
        total_mass += float(np.sum(local_mass))
        if len(candidate_ids) == 0:
            continue
        cost = support_cost(
            coordinates[row, candidate_ids], truth_coordinates[row, truth_ids], axes
        )
        covered_mass += float(np.sum(local_mass[np.min(cost, axis=0) <= TIGHT_COST]))
    return covered_mass / max(total_mass, TINY)


def shuffled_controls(
    coordinates: np.ndarray,
    mask: np.ndarray,
    site: np.ndarray,
    truth_coordinates: np.ndarray,
    truth_mass: np.ndarray,
    truth_valid: np.ndarray,
    count: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = np.arange(len(coordinates))
    values = {"delay": [], "horizontal_delay": [], "full": []}
    for _ in range(count):
        permutation = np.array(rows, copy=True)
        for site_value in np.unique(site):
            local = np.flatnonzero(site == site_value)
            permutation[local] = rng.permutation(local)
        for name, axes in (
            ("delay", (2,)),
            ("horizontal_delay", (0, 2)),
            ("full", (0, 1, 2)),
        ):
            values[name].append(
                tight_mass_coverage(
                    coordinates[permutation],
                    mask[permutation],
                    truth_coordinates,
                    truth_mass,
                    truth_valid,
                    axes,
                )
            )
    return {
        name: {
            "mean": float(np.mean(local)),
            "std": float(np.std(local)),
            "values": local,
        }
        for name, local in values.items()
    }


def load_oof_local_candidates(
    fold_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = e79.Config().maximum_candidates
    coordinates = np.zeros((len(fold_id), maximum, 3), dtype=np.float32)
    mask = np.zeros((len(fold_id), maximum), dtype=bool)
    score = np.zeros((len(fold_id), maximum), dtype=np.float32)
    for outer in range(5):
        path = e79.candidate_cache_path(outer, e79.Config())
        cache = np.load(path, allow_pickle=False)
        offsets = cache["offsets"].astype(np.int64)
        held = np.flatnonzero(fold_id == outer)
        for row in held:
            start, end = int(offsets[row]), int(offsets[row + 1])
            count = min(end - start, maximum)
            if count <= 0:
                continue
            coordinates[row, :count] = cache["coordinates"][start : start + count]
            fraction = cache["fractions"][start : start + count]
            presence = cache["presences"][start : start + count]
            score[row, :count] = fraction * np.sqrt(np.maximum(presence, 1e-8))
            mask[row, :count] = True
    return coordinates, mask, score


def sequence_hashes(cache: dict[str, np.ndarray]) -> np.ndarray:
    source = np.asarray(cache["source"], dtype=np.int64)
    depth = np.asarray(cache["interaction_count"], dtype=np.int64)
    interactions = np.asarray(cache["interactions"], dtype=np.uint64)
    objects = np.asarray(cache["objects"], dtype=np.uint64)
    primitives = np.asarray(cache["primitives"], dtype=np.uint64)
    result = np.full(source.shape, np.uint64(1469598103934665603), dtype=np.uint64)
    result ^= np.asarray(source + 2, dtype=np.uint64)
    result *= np.uint64(1099511628211)
    result ^= np.asarray(depth, dtype=np.uint64)
    result *= np.uint64(1099511628211)
    for step in range(interactions.shape[2]):
        active = step < depth
        value = (
            interactions[..., step]
            ^ (objects[..., step] * np.uint64(0x9E3779B185EBCA87))
            ^ (primitives[..., step] * np.uint64(0xC2B2AE3D27D4EB4F))
            ^ np.uint64(step + 1)
        )
        updated = (result ^ value) * np.uint64(1099511628211)
        result = np.where(active, updated, result)
    return result


def sequence_reuse_audit(
    cache: dict[str, np.ndarray], fold_id: np.ndarray
) -> dict[str, object]:
    mask = np.asarray(cache["mask"], dtype=bool)
    hashes = sequence_hashes(cache)
    point = np.broadcast_to(np.arange(len(mask))[:, None], mask.shape)
    valid_hash = hashes[mask]
    valid_point = point[mask].astype(np.int32)
    pair = np.empty(len(valid_hash), dtype=[("hash", "<u8"), ("point", "<i4")])
    pair["hash"] = valid_hash
    pair["point"] = valid_point
    unique_pair = np.unique(pair)
    unique_hash, point_count = np.unique(unique_pair["hash"], return_counts=True)
    occurrence = point_count[np.searchsorted(unique_hash, valid_hash)]
    energy = np.maximum(np.asarray(cache["energy_fraction"], dtype=np.float64)[mask], 0.0)
    path_rows = []
    for minimum in (2, 5, 20, 100):
        selected = occurrence >= minimum
        path_rows.append(
            {
                "minimum_distinct_points": minimum,
                "path_fraction": float(np.mean(selected)),
                "energy_weighted_fraction": float(
                    np.sum(energy[selected]) / max(float(np.sum(energy)), TINY)
                ),
            }
        )
    crossfold = []
    for fold in range(5):
        fit_hash = np.unique(hashes[(fold_id != fold)[:, None] & mask])
        held_mask = (fold_id == fold)[:, None] & mask
        held_hash = hashes[held_mask]
        held_energy = np.maximum(
            np.asarray(cache["energy_fraction"], dtype=np.float64)[held_mask], 0.0
        )
        seen = np.isin(held_hash, fit_hash, assume_unique=False)
        crossfold.append(
            {
                "fold": fold,
                "path_fraction": float(np.mean(seen)) if len(seen) else math.nan,
                "energy_weighted_fraction": float(
                    np.sum(held_energy[seen])
                    / max(float(np.sum(held_energy)), TINY)
                ),
            }
        )
    return {
        "valid_paths": int(len(valid_hash)),
        "unique_ordered_sequences": int(len(unique_hash)),
        "reuse_thresholds": path_rows,
        "crossfold_seen": crossfold,
        "crossfold_energy_weighted_mean": float(
            np.mean([row["energy_weighted_fraction"] for row in crossfold])
        ),
    }


def grouped_point_coverage(
    coverage: np.ndarray,
    truth_mass: np.ndarray,
    groups: np.ndarray,
    names: list[str],
) -> list[dict[str, object]]:
    output = []
    valid = np.isfinite(coverage) & (truth_mass > 0.0)
    for value, name in enumerate(names):
        selected = valid & (groups == value)
        output.append(
            {
                "group": name,
                "points": int(np.sum(selected)),
                "mean_point_coverage": (
                    float(np.mean(coverage[selected])) if np.any(selected) else math.nan
                ),
                "mass_weighted_coverage": (
                    float(np.sum(coverage[selected] * truth_mass[selected]))
                    / max(float(np.sum(truth_mass[selected])), TINY)
                    if np.any(selected)
                    else math.nan
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(shuffles=args.shuffles)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    component = np.load(COMPONENTS, allow_pickle=False)
    truth_coordinates = component["coordinates"].astype(np.float32)
    truth_coefficients = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    truth_energy = np.sum(np.abs(truth_coefficients) ** 2, axis=2, dtype=np.float64)
    truth_mass = np.divide(
        truth_energy,
        np.maximum(point_power[:, None], TINY),
        out=np.zeros_like(truth_energy),
        where=point_power[:, None] > TINY,
    )
    truth_valid = (point_power[:, None] > TINY) & (truth_mass >= 1e-5)

    path_names = (
        "tau",
        "angles",
        "mask",
        "source",
        "presence",
        "energy_fraction",
        "interaction_count",
        "interactions",
        "objects",
        "primitives",
    )
    path_cache = {
        name: np.load(PATH_CACHE / f"{name}.npy", mmap_mode="r")
        for name in path_names
    }
    setup, _, _, train_site, _ = load_positions()
    del setup
    state = json.loads(SIONNA_STATE.read_text(encoding="utf-8"))
    sionna_coordinate = sionna_coordinates(
        path_cache["tau"],
        path_cache["angles"],
        train_site,
        state["configuration"],
        vertical_alignment=2.0,
    )
    sionna_unaligned = np.array(sionna_coordinate, copy=True)
    sionna_unaligned[..., 1] = (sionna_unaligned[..., 1] - 2.0) % MV
    sionna_mask = np.asarray(path_cache["mask"], dtype=bool)
    raw_energy = np.maximum(
        np.asarray(path_cache["energy_fraction"], dtype=np.float32), 0.0
    )
    stability = raw_energy * np.sqrt(
        np.maximum(np.asarray(path_cache["presence"], dtype=np.float32), 1.0) / 24.0
    )
    cache_order = np.broadcast_to(
        np.linspace(1.0, 0.0, sionna_mask.shape[1], dtype=np.float32),
        sionna_mask.shape,
    )
    full_report, full_point = evaluate_dense_candidates(
        "sionna_full_4000",
        sionna_coordinate,
        sionna_mask,
        {"cache_order": cache_order, "energy": raw_energy, "stability": stability},
        truth_coordinates,
        truth_mass,
        truth_valid,
        config.budgets,
        config.thresholds,
    )
    unaligned_coverage = tight_mass_coverage(
        sionna_unaligned,
        sionna_mask,
        truth_coordinates,
        truth_mass,
        truth_valid,
    )
    shuffle = shuffled_controls(
        sionna_coordinate,
        sionna_mask,
        train_site,
        truth_coordinates,
        truth_mass,
        truth_valid,
        config.shuffles,
        config.random_state,
    )

    source_depth = {}
    source = np.asarray(path_cache["source"], dtype=np.int8)
    depth = np.asarray(path_cache["interaction_count"], dtype=np.uint8)
    for key, selected in {
        "base": sionna_mask & (source == 0),
        "scatter": sionna_mask & (source == 1),
        "depth0": sionna_mask & (depth == 0),
        "depth1": sionna_mask & (depth == 1),
        "depth2": sionna_mask & (depth == 2),
        "depth3plus": sionna_mask & (depth >= 3),
    }.items():
        source_depth[key] = tight_mass_coverage(
            sionna_coordinate,
            selected,
            truth_coordinates,
            truth_mass,
            truth_valid,
        )

    full_fold = np.load(FULL_FOLD).astype(np.int8)
    full_regime = np.load(FULL_REGIME).astype(np.int8)
    sequence = sequence_reuse_audit(path_cache, full_fold)
    full_groups = {
        "site": grouped_point_coverage(
            full_point["tight_coverage"],
            full_point["truth_mass"],
            train_site,
            ["site0", "site1"],
        ),
        "regime": grouped_point_coverage(
            full_point["tight_coverage"],
            full_point["truth_mass"],
            full_regime,
            list(REGIME_NAMES),
        ),
        "matched_spatial_fold": grouped_point_coverage(
            full_point["tight_coverage"],
            full_point["truth_mass"],
            full_fold,
            [f"fold{fold}" for fold in range(5)],
        ),
    }

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    local_coordinate, local_mask, local_score = load_oof_local_candidates(fold_id)
    local_report, local_point = evaluate_dense_candidates(
        "local_track_strict_500",
        local_coordinate,
        local_mask,
        {"lifecycle": local_score},
        truth_coordinates[query_ids],
        truth_mass[query_ids],
        truth_valid[query_ids],
        (24, 48, 96, 176),
        config.thresholds,
    )
    strict_sionna_report, strict_sionna_point = evaluate_dense_candidates(
        "sionna_strict_500",
        sionna_coordinate[query_ids],
        sionna_mask[query_ids],
        {"energy": raw_energy[query_ids], "stability": stability[query_ids]},
        truth_coordinates[query_ids],
        truth_mass[query_ids],
        truth_valid[query_ids],
        config.budgets,
        config.thresholds,
    )
    local_rank = rank_percentile(local_score, local_mask)
    sionna_rank = rank_percentile(stability[query_ids], sionna_mask[query_ids])
    union_coordinate = np.concatenate(
        (local_coordinate, sionna_coordinate[query_ids]), axis=1
    )
    union_mask = np.concatenate((local_mask, sionna_mask[query_ids]), axis=1)
    union_score = np.concatenate((local_rank, sionna_rank), axis=1)
    union_report, union_point = evaluate_dense_candidates(
        "local_sionna_union_strict_500",
        union_coordinate,
        union_mask,
        {"source_rank": union_score},
        truth_coordinates[query_ids],
        truth_mass[query_ids],
        truth_valid[query_ids],
        (24, 48, 96, 128, union_coordinate.shape[1]),
        config.thresholds,
    )
    strict_regime = np.load(
        PATHS.artifacts
        / "round2"
        / "regime_lifecycle_support"
        / "regime_assignments.npz",
        allow_pickle=False,
    )["oof_regime"].astype(np.int8)
    strict_groups = {}
    for candidate_name, point in (
        ("local", local_point),
        ("sionna", strict_sionna_point),
        ("union", union_point),
    ):
        strict_groups[candidate_name] = {
            "fold": grouped_point_coverage(
                point["tight_coverage"],
                point["truth_mass"],
                fold_id,
                [f"fold{fold}" for fold in range(5)],
            ),
            "regime": grouped_point_coverage(
                point["tight_coverage"],
                point["truth_mass"],
                strict_regime,
                list(REGIME_NAMES),
            ),
        }

    sionna_shuffle_full = float(shuffle["full"]["mean"])
    sionna_shuffle_delay = float(shuffle["delay"]["mean"])
    sionna_full = float(full_report["all_candidate_tight_mass_coverage"])
    sionna_delay = float(full_report["dimension_tight_mass_coverage"]["delay"])
    local_strict = float(local_report["all_candidate_tight_mass_coverage"])
    union_strict = float(union_report["all_candidate_tight_mass_coverage"])
    sionna_strict = float(strict_sionna_report["all_candidate_tight_mass_coverage"])
    union_delta = union_strict - max(local_strict, sionna_strict)
    full_margin = sionna_full - sionna_shuffle_full
    delay_margin = sionna_delay - sionna_shuffle_delay
    sequence_reuse = float(sequence["crossfold_energy_weighted_mean"])
    if union_delta >= 0.005 and full_margin >= 0.02 and sequence_reuse >= 0.50:
        next_branch = "surface_interaction_energy_model"
    elif delay_margin >= 0.05:
        next_branch = "path_support_alignment_model"
    else:
        next_branch = "candidate_generation_revision"
    decision = {
        "frozen_thresholds": {
            "minimum_union_gain": 0.005,
            "minimum_position_specific_full_margin": 0.02,
            "minimum_crossfold_sequence_energy_reuse": 0.50,
            "fallback_minimum_delay_margin": 0.05,
        },
        "strict_local_coverage": local_strict,
        "strict_sionna_coverage": sionna_strict,
        "strict_union_coverage": union_strict,
        "strict_union_gain_over_best_source": union_delta,
        "sionna_full_margin_over_shuffle": full_margin,
        "sionna_delay_margin_over_shuffle": delay_margin,
        "crossfold_sequence_energy_reuse": sequence_reuse,
        "next_branch": next_branch,
    }

    np.savez_compressed(
        PER_POINT,
        full_point_ids=np.arange(len(train_site), dtype=np.int32),
        full_site=train_site.astype(np.int8),
        full_fold=full_fold,
        full_regime=full_regime,
        full_truth_mass=full_point["truth_mass"],
        full_sionna_tight_coverage=full_point["tight_coverage"],
        full_sionna_ambiguity=full_point["ambiguity"],
        full_sionna_energy_cosine=full_point["energy_energy_cosine"],
        strict_query_ids=query_ids.astype(np.int32),
        strict_fold=fold_id,
        strict_regime=strict_regime,
        strict_truth_mass=local_point["truth_mass"],
        strict_local_tight_coverage=local_point["tight_coverage"],
        strict_sionna_tight_coverage=strict_sionna_point["tight_coverage"],
        strict_union_tight_coverage=union_point["tight_coverage"],
        strict_local_ambiguity=local_point["ambiguity"],
        strict_sionna_ambiguity=strict_sionna_point["ambiguity"],
        strict_union_ambiguity=union_point["ambiguity"],
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Explicit Sionna interaction paths add position-specific support beyond "
            "local official-H tracks, and their ordered surface identities are reusable "
            "enough to justify a path-level sim-to-official model."
        ),
        "configuration": {
            "budgets": config.budgets,
            "thresholds": config.thresholds,
            "tight_cost_interpretation": {
                "horizontal_only_bins": 0.75,
                "vertical_only_bins": 0.50,
                "delay_only_bins": 1.00,
            },
            "vertical_alignment_bins": 2.0,
            "shuffles": config.shuffles,
            "random_state": config.random_state,
        },
        "official_component_audit": {
            "mean_explained_fraction": float(np.mean(np.sum(truth_mass, axis=1))),
            "nonzero_points": int(np.sum(point_power > TINY)),
            "valid_truth_components": int(np.sum(truth_valid)),
        },
        "sionna_full_4000": full_report,
        "sionna_unaligned_full_tight_coverage": unaligned_coverage,
        "sionna_same_site_shuffle": shuffle,
        "sionna_source_and_depth_tight_coverage": source_depth,
        "ordered_sequence_reuse": sequence,
        "full_grouped_coverage": full_groups,
        "strict_500": {
            "local": local_report,
            "sionna": strict_sionna_report,
            "union": union_report,
            "grouped": strict_groups,
        },
        "decision": decision,
        "per_point_cache": PATHS.relative(PER_POINT),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sionna_full": {
                    "tight": sionna_full,
                    "delay": sionna_delay,
                    "shuffle_tight": sionna_shuffle_full,
                    "shuffle_delay": sionna_shuffle_delay,
                },
                "strict_500": {
                    "local": local_strict,
                    "sionna": sionna_strict,
                    "union": union_strict,
                },
                "sequence_reuse": sequence_reuse,
                "decision": decision,
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
