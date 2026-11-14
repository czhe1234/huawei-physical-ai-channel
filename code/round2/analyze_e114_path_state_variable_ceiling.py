#!/usr/bin/env python3
"""Audit the causal ceiling of complete path-state prediction for Round 2."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_global_local_path_lifecycle as e79
import train_global_local_path_set_selector as e82
from common.official_metrics import point_metric_statistics
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import component_energy
from run_path_track_spectral_field import valid_components


VERSION = "Round2-E114-path-state-variable-ceiling"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "path_state_variable_ceiling"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_path_state_variable_ceiling"
REPORT = ANALYSIS_DIR / "path_state_variable_ceiling.json"
METRICS = ARTIFACT_DIR / "path_state_variable_ceiling_metrics.npz"

COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

PERIODS = np.asarray((e79.MH, e79.MV, e79.S_SC), dtype=np.float32)
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    donor_count: int = 32
    donor_components: int = 20
    raw_candidates: int = 176
    output_paths: int = 24
    minimum_fraction: float = 1e-5
    spatial_scale_m: float = 12.0
    soft_cost_limit: float = e82.SOFT_TUBE_COST_LIMIT
    soft_bandwidth: float = e82.SOFT_TUBE_BANDWIDTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def empty_variant(count: int, paths: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((count, paths, 3), dtype=np.float32),
        np.zeros((count, paths, e79.PORTS), dtype=np.complex64),
    )


def normalize_rows(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    return np.divide(
        output,
        np.maximum(output.sum(axis=1, keepdims=True), TINY),
        out=np.zeros_like(output),
    ).astype(np.float32)


def soft_targets(
    coordinate: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    point_power: float,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    count = len(coordinate)
    mass = np.zeros(count, dtype=np.float32)
    delta = np.zeros((count, 3), dtype=np.float32)
    port = np.full((count, e79.PORTS), 1.0 / e79.PORTS, dtype=np.float32)
    if count == 0 or point_power <= TINY:
        return mass, delta, port, 0.0, 0.0

    energy = np.sum(np.abs(truth_coefficient) ** 2, axis=1, dtype=np.float64)
    fraction = energy / max(float(point_power), TINY)
    valid = fraction >= config.minimum_fraction
    if not np.any(valid):
        return mass, delta, port, 0.0, 0.0
    truth_coordinate = truth_coordinate[valid]
    truth_coefficient = truth_coefficient[valid]
    fraction = fraction[valid]
    cost = e79.support_cost(coordinate, truth_coordinate)
    kernel = np.exp(-0.5 * cost / config.soft_bandwidth)
    kernel[cost > config.soft_cost_limit] = 0.0
    kernel /= np.maximum(kernel.sum(axis=0, keepdims=True), TINY)
    contribution = kernel * fraction[None]
    mass = contribution.sum(axis=1).astype(np.float32)
    local_total = contribution.sum(axis=1, keepdims=True)
    responsibility = np.divide(
        contribution,
        np.maximum(local_total, TINY),
        out=np.zeros_like(contribution),
    )
    coordinate_delta = np.stack(
        tuple(
            e79.periodic_delta(
                truth_coordinate[None, :, axis],
                coordinate[:, None, axis],
                float(PERIODS[axis]),
            )
            for axis in range(3)
        ),
        axis=2,
    )
    delta = np.einsum("ct,ctd->cd", responsibility, coordinate_delta).astype(
        np.float32
    )
    truth_port = normalize_rows(np.abs(truth_coefficient) ** 2)
    predicted_port = responsibility @ truth_port
    positive = mass > 0.0
    port[positive] = predicted_port[positive].astype(np.float32)
    covered = np.any(cost <= config.soft_cost_limit, axis=0)
    return (
        mass,
        delta,
        port,
        float(np.sum(fraction[covered], dtype=np.float64)),
        float(np.sum(fraction, dtype=np.float64)),
    )


def assign_variant(
    target: tuple[np.ndarray, np.ndarray],
    row: int,
    coordinate: np.ndarray,
    mass: np.ndarray,
    port: np.ndarray,
    output_paths: int,
) -> None:
    if len(mass) == 0:
        return
    selected = np.argsort(-mass, kind="stable")[:output_paths]
    count = len(selected)
    profile = normalize_rows(port[selected])
    target[0][row, :count] = coordinate[selected]
    target[1][row, :count] = np.sqrt(
        np.maximum(mass[selected, None] * profile, 0.0)
    ).astype(np.complex64)


def track_factorial_variants(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    point_power: np.ndarray,
    config: Config,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    names = (
        "track_base",
        "mass",
        "coordinate",
        "port",
        "mass_coordinate",
        "mass_port",
        "coordinate_port",
        "complete_state_oracle",
        "true_24_path_ceiling",
    )
    variants = {
        name: empty_variant(len(query_ids), config.output_paths) for name in names
    }
    supported = 0.0
    total = 0.0
    for outer in range(5):
        cache = e82.load_candidate_cache(outer)
        offsets = cache["offsets"].astype(np.int64)
        rows = np.flatnonzero(fold_id == outer)
        for row in rows:
            point_id = int(query_ids[row])
            normalized_truth = truth_coefficient[point_id, : config.output_paths] / np.sqrt(
                max(float(point_power[point_id]), TINY)
            )
            variants["true_24_path_ceiling"][0][row] = truth_coordinate[
                point_id, : config.output_paths
            ]
            variants["true_24_path_ceiling"][1][row] = normalized_truth
            start = int(offsets[row])
            end = int(offsets[row + 1])
            if end <= start:
                continue
            coordinate = cache["coordinates"][start:end].astype(np.float32)
            donor_port = normalize_rows(cache["profiles"][start:end])
            base_mass = np.maximum(cache["fractions"][start:end], 1e-12) * np.sqrt(
                np.maximum(cache["presences"][start:end], 1e-8)
            )
            base_mass /= max(float(np.sum(base_mass)), TINY)
            target_mass, target_delta, target_port, local_supported, local_total = (
                soft_targets(
                    coordinate,
                    truth_coordinate[point_id, : config.output_paths],
                    truth_coefficient[point_id, : config.output_paths],
                    float(point_power[point_id]),
                    config,
                )
            )
            supported += local_supported
            total += local_total
            corrected = (coordinate + target_delta) % PERIODS
            state = {
                "track_base": (coordinate, base_mass, donor_port),
                "mass": (coordinate, target_mass, donor_port),
                "coordinate": (corrected, base_mass, donor_port),
                "port": (coordinate, base_mass, target_port),
                "mass_coordinate": (corrected, target_mass, donor_port),
                "mass_port": (coordinate, target_mass, target_port),
                "coordinate_port": (corrected, base_mass, target_port),
                "complete_state_oracle": (corrected, target_mass, target_port),
            }
            for name, values in state.items():
                assign_variant(variants[name], row, *values, config.output_paths)
    return variants, {"tight_mass_coverage": supported / max(total, TINY)}


def raw_token_variants(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    point_power: np.ndarray,
    config: Config,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    variants = {
        "raw_token_base": empty_variant(len(query_ids), config.raw_candidates),
        "raw_token_complete_state_oracle": empty_variant(
            len(query_ids), config.raw_candidates
        ),
    }
    valid, fraction = valid_components(
        component_energy(truth_coefficient),
        point_power,
        config.donor_components,
        config.minimum_fraction,
    )
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    supported = 0.0
    total = 0.0
    candidate_counts: list[int] = []
    for outer in range(5):
        outer_fit = folds[f"fold{outer}_fit"].astype(np.int64)
        outer_rows = np.flatnonzero(fold_id == outer)
        for site_value in np.unique(site[query_ids[outer_rows]]):
            rows = outer_rows[site[query_ids[outer_rows]] == site_value]
            bank = outer_fit[
                (site[outer_fit] == site_value) & (point_power[outer_fit] > TINY)
            ]
            take = min(config.donor_count, len(bank))
            distance, local = cKDTree(positions[bank, :2]).query(
                positions[query_ids[rows], :2], k=take, workers=-1
            )
            distance = np.asarray(distance)
            local = np.asarray(local)
            if take == 1:
                distance = distance[:, None]
                local = local[:, None]
            for output_row, donor_distance, donor_local in zip(rows, distance, local):
                point_id = int(query_ids[output_row])
                donors = bank[donor_local.astype(np.int64)]
                point_weight = np.exp(
                    -0.5 * np.square(donor_distance / config.spatial_scale_m)
                ) / np.sqrt(np.maximum(donor_distance, 1.0))
                point_weight /= max(float(np.sum(point_weight)), TINY)
                donor_ids = np.repeat(donors, config.donor_components)
                component_ids = np.tile(
                    np.arange(config.donor_components), len(donors)
                )
                token_weight = np.repeat(point_weight, config.donor_components)
                keep = valid[donor_ids, component_ids]
                donor_ids = donor_ids[keep]
                component_ids = component_ids[keep]
                token_weight = token_weight[keep]
                score = fraction[donor_ids, component_ids] * np.sqrt(
                    np.maximum(token_weight, 1e-12)
                )
                selected = np.argsort(-score, kind="stable")[
                    : config.raw_candidates
                ]
                donor_ids = donor_ids[selected]
                component_ids = component_ids[selected]
                score = score[selected]
                coordinate = truth_coordinate[donor_ids, component_ids]
                donor_port = normalize_rows(
                    np.abs(truth_coefficient[donor_ids, component_ids]) ** 2
                )
                base_mass = score / max(float(np.sum(score)), TINY)
                target_mass, target_delta, target_port, local_supported, local_total = (
                    soft_targets(
                        coordinate,
                        truth_coordinate[point_id, : config.output_paths],
                        truth_coefficient[point_id, : config.output_paths],
                        float(point_power[point_id]),
                        config,
                    )
                )
                supported += local_supported
                total += local_total
                candidate_counts.append(len(selected))
                assign_variant(
                    variants["raw_token_base"],
                    output_row,
                    coordinate,
                    base_mass,
                    donor_port,
                    config.raw_candidates,
                )
                assign_variant(
                    variants["raw_token_complete_state_oracle"],
                    output_row,
                    (coordinate + target_delta) % PERIODS,
                    target_mass,
                    target_port,
                    config.raw_candidates,
                )
    return variants, {
        "tight_mass_coverage": supported / max(total, TINY),
        "candidate_count_mean": float(np.mean(candidate_counts)),
        "candidate_count_min": int(np.min(candidate_counts, initial=0)),
        "candidate_count_max": int(np.max(candidate_counts, initial=0)),
    }


def endpoint_metrics(
    prediction: np.ndarray, truth: np.ndarray, fold_id: np.ndarray
) -> dict[str, object]:
    values = point_metric_statistics(prediction, truth)
    combined = 0.5 * (values.pas + values.pdp)
    return {
        "pas": float(np.mean(values.pas)),
        "pdp": float(np.mean(values.pdp)),
        "pas_pdp_mean": float(np.mean(combined)),
        "fold_pas_pdp_mean": [
            float(np.mean(combined[fold_id == fold])) for fold in range(5)
        ],
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config()
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    component = np.load(COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    _, positions, _, site, _ = load_positions()

    track_variants, track_audit = track_factorial_variants(
        query_ids,
        fold_id,
        truth_coordinate,
        truth_coefficient,
        point_power,
        config,
    )
    raw_variants, raw_audit = raw_token_variants(
        query_ids,
        fold_id,
        positions,
        site,
        truth_coordinate,
        truth_coefficient,
        point_power,
        config,
    )
    all_variants = {**track_variants, **raw_variants}
    results = e79.evaluate_variants(
        all_variants, query_ids, fold_id, device, args.batch_size
    )

    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth_h = np.asarray(train_h[query_ids], dtype=np.complex64)
    v12_h = np.load(V12_CHANNEL, mmap_mode="r")
    v12_endpoint = endpoint_metrics(v12_h, truth_h, fold_id)
    base = float(results["track_base"]["pas_pdp_mean"])
    complete = float(results["complete_state_oracle"]["pas_pdp_mean"])
    raw_complete = float(
        results["raw_token_complete_state_oracle"]["pas_pdp_mean"]
    )
    factorial = {
        name: {
            "pas": float(results[name]["pas"]["mean"]),
            "pdp": float(results[name]["pdp"]["mean"]),
            "pas_pdp_mean": float(results[name]["pas_pdp_mean"]),
            "delta_vs_track_base": float(results[name]["pas_pdp_mean"] - base),
            "fold_pas_pdp_mean": [
                float(0.5 * (left + right))
                for left, right in zip(
                    results[name]["pas"]["fold"], results[name]["pdp"]["fold"]
                )
            ],
        }
        for name in track_variants
    }
    raw = {
        name: {
            "pas": float(results[name]["pas"]["mean"]),
            "pdp": float(results[name]["pdp"]["mean"]),
            "pas_pdp_mean": float(results[name]["pas_pdp_mean"]),
        }
        for name in raw_variants
    }
    gate = {
        "minimum_raw_token_tight_mass_coverage": 0.95,
        "minimum_complete_state_margin_over_v12_endpoint": 0.10,
        "raw_token_tight_mass_coverage": float(raw_audit["tight_mass_coverage"]),
        "track_complete_state_margin_over_v12_endpoint": float(
            complete - v12_endpoint["pas_pdp_mean"]
        ),
        "raw_complete_state_margin_over_v12_endpoint": float(
            raw_complete - v12_endpoint["pas_pdp_mean"]
        ),
    }
    gate["passed"] = bool(
        gate["raw_token_tight_mass_coverage"]
        >= gate["minimum_raw_token_tight_mass_coverage"]
        and gate["raw_complete_state_margin_over_v12_endpoint"]
        >= gate["minimum_complete_state_margin_over_v12_endpoint"]
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The official-neighbor path bank already contains the hidden support, and "
            "the remaining ceiling is controlled by per-path energy, continuous "
            "delay-angle correction, and polarization/UE state rather than by a "
            "low-dimensional final-spectrum operator."
        ),
        "official_inputs_only_raw_branch": True,
        "config": asdict(config),
        "v12_endpoint": v12_endpoint,
        "track_candidate_audit": track_audit,
        "track_factorial": factorial,
        "raw_neighbor_token_audit": raw_audit,
        "raw_neighbor_token_results": raw,
        "deployment_gate": gate,
        "causal_conclusion": {
            "largest_single_missing_state": "path_mass_and_competition",
            "mass_only_gain": float(factorial["mass"]["delta_vs_track_base"]),
            "coordinate_only_gain": float(
                factorial["coordinate"]["delta_vs_track_base"]
            ),
            "port_only_gain": float(factorial["port"]["delta_vs_track_base"]),
            "mass_port_score": float(factorial["mass_port"]["pas_pdp_mean"]),
            "mass_coordinate_score": float(
                factorial["mass_coordinate"]["pas_pdp_mean"]
            ),
            "complete_state_score": complete,
            "true_24_path_score": float(
                factorial["true_24_path_ceiling"]["pas_pdp_mean"]
            ),
            "next_experiment": (
                "Train a full path-state operator over uncompressed official-neighbor "
                "complex path tokens, candidate competition, map scene tokens, and "
                "query-query lifecycle evidence."
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        variant_names=np.asarray(tuple(all_variants), dtype="U64"),
        variant_pas=np.asarray(
            [results[name]["pas"]["mean"] for name in all_variants],
            dtype=np.float64,
        ),
        variant_pdp=np.asarray(
            [results[name]["pdp"]["mean"] for name in all_variants],
            dtype=np.float64,
        ),
    )
    print(
        json.dumps(
            {
                "v12_endpoint": v12_endpoint,
                "factorial": {
                    name: row["pas_pdp_mean"] for name, row in factorial.items()
                },
                "raw": raw,
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
