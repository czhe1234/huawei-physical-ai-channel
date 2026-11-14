#!/usr/bin/env python3
"""Estimate path-conditioned residual wavevectors after wall dephasing."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from analyze_v4_map_virtual_source_phase import (
    COMPONENT_CACHE,
    Config as MapConfig,
    build_wall_bank,
    image_sources,
    split_phase_score,
)
from analyze_v5_explicit_virtual_path_identity import (
    ASSIGNMENT_CACHE,
    FOLDS,
    OOF_META,
    Config as IdentityConfig,
    load_fold_parameters,
    phase_cosine,
    summarize,
    unit_vector,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import TINY


VERSION = "Round2-E35-wall-residual-wavevector"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_wall_residual_wavevector"
REPORT = ANALYSIS_DIR / "wall_residual_wavevector.json"


@dataclass(frozen=True)
class Config:
    paths: int = 16
    donor_count: int = 24
    minimum_observations: int = 6
    maximum_donor_distance_m: float = 55.0
    distance_scale_floor_m: float = 5.0
    residual_radii_rad_per_m: tuple[float, ...] = (
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
    )
    residual_directions: int = 72
    all_score_weight: float = 0.45
    split_score_weight: float = 0.55
    control_rotation_degrees: float = 37.0
    required_cosine_gain: float = 0.05
    required_positive_folds: int = 4


def residual_wavevector_grid(config: Config) -> np.ndarray:
    rows = [np.zeros(2, dtype=np.float64)]
    angle = np.arange(config.residual_directions, dtype=np.float64)
    angle *= 2.0 * math.pi / config.residual_directions
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    for radius in config.residual_radii_rad_per_m:
        if radius > 0.0:
            rows.extend(radius * direction)
    return np.asarray(rows, dtype=np.float64)


def rotate_vectors(value: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    return value @ rotation.T


def transport_with_wavevector(
    dephased: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    wavevector: np.ndarray,
    beta: float,
    query_length: float,
) -> np.ndarray:
    phase = np.exp(1j * (offsets @ wavevector))
    latent = np.sum(weights[:, None] * dephased * phase[:, None], axis=0)
    return unit_vector(latent) * np.exp(-1j * beta * query_length)


def fit_residual_wavevector(
    dephased: np.ndarray,
    offsets: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, float, float]:
    phase = np.exp(1j * (offsets @ grid.T))
    all_score, split_score = split_phase_score(
        dephased,
        phase,
        weights,
        positions,
    )
    score = (
        config.all_score_weight * all_score
        + config.split_score_weight * split_score
    )
    selected = int(np.argmax(score))
    return grid[selected], float(score[selected]), float(score[0])


def weighted_summary(rows: list[dict[str, float]], key: str) -> dict[str, float | int]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    weights = np.asarray([row["energy"] for row in rows], dtype=np.float64)
    return summarize(values, weights)


def evaluate(config: Config) -> dict[str, object]:
    setup, positions, _, site, _ = load_positions()
    component_cache = np.load(COMPONENT_CACHE)
    coefficients = component_cache["coefficients"].astype(np.complex64)
    component_energy = np.sum(
        np.abs(coefficients[:, : config.paths]) ** 2, axis=2, dtype=np.float64
    )
    assignments = np.load(ASSIGNMENT_CACHE)
    folds = np.load(FOLDS)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    fold_parameters = load_fold_parameters()
    wall_bank = build_wall_bank(MapConfig())
    sources = {
        site_value: image_sources(wall_bank, setup.bs_positions[site_value])
        for site_value in range(2)
    }
    grid = residual_wavevector_grid(config)
    rotated_grid = rotate_vectors(grid, config.control_rotation_degrees)
    rows: list[dict[str, float]] = []
    held_energy = 0.0
    evaluable_energy = 0.0

    for fold in range(5):
        fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
        held_ids = query_ids[fold_id == fold]
        for site_value in range(2):
            fit_site = fit_ids[site[fit_ids] == site_value]
            held_site = held_ids[site[held_ids] == site_value]
            beta = float(fold_parameters[(fold, site_value)]["beta"])
            identity_bank: dict[int, list[tuple[int, int]]] = {}
            for point_id in fit_site:
                for component, identity in enumerate(
                    assignments["path_id"][fold, point_id]
                ):
                    if identity >= 0:
                        identity_bank.setdefault(int(identity), []).append(
                            (int(point_id), int(component))
                        )

            for query_id in held_site:
                held_energy += float(np.sum(component_energy[query_id]))
                for component, identity in enumerate(
                    assignments["path_id"][fold, query_id]
                ):
                    if identity < 0:
                        continue
                    energy = float(component_energy[query_id, component])
                    if energy <= TINY:
                        continue
                    observations = identity_bank.get(int(identity), [])
                    if len(observations) < config.minimum_observations:
                        continue
                    ids = np.asarray([value[0] for value in observations], dtype=np.int64)
                    parts = np.asarray([value[1] for value in observations], dtype=np.int64)
                    distance = np.linalg.norm(
                        positions[ids, :2] - positions[query_id, None, :2], axis=1
                    )
                    keep = distance <= config.maximum_donor_distance_m
                    ids = ids[keep]
                    parts = parts[keep]
                    distance = distance[keep]
                    if len(ids) < config.minimum_observations:
                        continue
                    order = np.argsort(distance)[: config.donor_count]
                    ids = ids[order]
                    parts = parts[order]
                    distance = distance[order]
                    scale = max(
                        float(np.median(distance)), config.distance_scale_floor_m
                    )
                    weights = np.exp(-0.5 * (distance / (1.5 * scale)) ** 2)
                    weights /= max(float(np.sum(weights)), TINY)

                    observed = coefficients[ids, parts]
                    observed /= np.maximum(
                        np.linalg.norm(observed, axis=1, keepdims=True), 1e-20
                    )
                    donor_length = assignments["path_length"][fold, ids, parts]
                    query_length = float(
                        assignments["path_length"][fold, query_id, component]
                    )
                    dephased = observed * np.exp(1j * beta * donor_length)[:, None]
                    offsets = positions[ids, :2] - positions[query_id, None, :2]
                    selected, fit_score, nominal_score = fit_residual_wavevector(
                        dephased,
                        offsets,
                        positions[ids, :2],
                        weights,
                        grid,
                        config,
                    )
                    selected_index = int(
                        np.argmin(np.sum((grid - selected[None]) ** 2, axis=1))
                    )
                    prediction = transport_with_wavevector(
                        dephased,
                        offsets,
                        weights,
                        selected,
                        beta,
                        query_length,
                    )
                    nominal = transport_with_wavevector(
                        dephased,
                        offsets,
                        weights,
                        grid[0],
                        beta,
                        query_length,
                    )
                    rotated = transport_with_wavevector(
                        dephased,
                        offsets,
                        weights,
                        rotated_grid[selected_index],
                        beta,
                        query_length,
                    )
                    opposite = transport_with_wavevector(
                        dephased,
                        offsets,
                        weights,
                        -selected,
                        beta,
                        query_length,
                    )
                    truth = coefficients[query_id, component]
                    rows.append(
                        {
                            "fold": float(fold),
                            "site": float(site_value),
                            "identity": float(identity),
                            "energy": energy,
                            "nominal": phase_cosine(nominal, truth),
                            "residual_wavevector": phase_cosine(prediction, truth),
                            "rotated_control": phase_cosine(rotated, truth),
                            "opposite_control": phase_cosine(opposite, truth),
                            "wavevector_norm": float(np.linalg.norm(selected)),
                            "fit_score": fit_score,
                            "nominal_fit_score": nominal_score,
                            "nearest_distance_m": float(distance[0]),
                            "observation_count": float(len(ids)),
                        }
                    )
                    evaluable_energy += energy
        print(f"wall residual wavevector fold={fold} rows={len(rows)}", flush=True)

    variants = (
        "nominal",
        "residual_wavevector",
        "rotated_control",
        "opposite_control",
    )
    overall = {name: weighted_summary(rows, name) for name in variants}
    fold_rows = []
    fold_gain = []
    for fold in range(5):
        selected_rows = [row for row in rows if int(row["fold"]) == fold]
        summary = {name: weighted_summary(selected_rows, name) for name in variants}
        gain = float(
            summary["residual_wavevector"]["weighted_mean"]
            - summary["nominal"]["weighted_mean"]
        )
        fold_gain.append(gain)
        fold_rows.append({"fold": fold, "gain_vs_nominal": gain, **summary})

    gain = float(
        overall["residual_wavevector"]["weighted_mean"]
        - overall["nominal"]["weighted_mean"]
    )
    best_control = max(
        float(overall["rotated_control"]["weighted_mean"]),
        float(overall["opposite_control"]["weighted_mean"]),
    )
    positive_folds = int(np.sum(np.asarray(fold_gain) > 0.0))
    passed = bool(
        gain >= config.required_cosine_gain
        and positive_folds >= config.required_positive_folds
        and float(overall["residual_wavevector"]["weighted_mean"]) > best_control
    )
    wavevector_norm = np.asarray(
        [row["wavevector_norm"] for row in rows], dtype=np.float64
    )
    row_energy = np.asarray([row["energy"] for row in rows], dtype=np.float64)
    fit_gain = np.asarray(
        [row["fit_score"] - row["nominal_fit_score"] for row in rows],
        dtype=np.float64,
    )
    diagnostics = {
        "row_count": len(rows),
        "evaluable_energy_fraction": float(evaluable_energy / max(held_energy, TINY)),
        "unique_wall_families": int(len({int(row["identity"]) for row in rows})),
        "wavevector_norm": summarize(wavevector_norm, row_energy),
        "fit_score_gain": summarize(fit_gain, row_energy),
    }
    return {
        "overall": overall,
        "folds": fold_rows,
        "gain_vs_nominal": gain,
        "margin_over_best_control": float(
            overall["residual_wavevector"]["weighted_mean"] - best_control
        ),
        "positive_fold_count": positive_folds,
        "stage_passed": passed,
        "stage_rule": {
            "minimum_weighted_cosine_gain": config.required_cosine_gain,
            "minimum_positive_folds": config.required_positive_folds,
            "must_beat_both_controls": True,
        },
        "diagnostics": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    started = time.time()
    config = Config()
    evaluation = evaluate(config)
    report = {
        "experiment": VERSION,
        "status": "stage_passed" if evaluation["stage_passed"] else "analysis_only",
        "primary_hypothesis": (
            "After removing the nominal wall path phase, a fit-only local residual "
            "wavevector captures effective higher-order propagation and predicts held "
            "path carrier phase better than a constant latent or physical controls."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "held_path_identity_used_for_evaluation_only": True,
        "config": asdict(config),
        "evaluation": evaluation,
        "seconds": float(time.time() - started),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
