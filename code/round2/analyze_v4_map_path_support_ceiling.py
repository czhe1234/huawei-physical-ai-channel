#!/usr/bin/env python3
"""Measure the PAS/PDP ceiling of official-map LOS and one-bounce supports."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from analyze_v4_map_virtual_source_phase import (
    METERS_PER_DELAY_BIN,
    Config as WallConfig,
    build_wall_bank,
    canonicalize_normals,
    image_sources,
)
from analyze_v4_path_track_spectral_targets import (
    MH,
    MV,
    S_SC,
    cosine,
    render_noncoherent,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import circular_delta, component_energy


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_map_path_support_ceiling"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_map_path_support_ceiling"
REPORT = ANALYSIS_DIR / "map_path_support_ceiling.json"
CANDIDATE_CACHE = ARTIFACT_DIR / "oof_map_path_candidates.npz"

COMPONENT_CACHE = (
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
FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
TRAIN_PAS = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_pas_link_support_f16.npy"
)
TRAIN_PDP = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_pdp_link_support_f16.npy"
)


@dataclass(frozen=True)
class Config:
    true_paths: int = 16
    candidate_counts: tuple[int, ...] = (16, 32, 64, 128, 256)
    horizontal_scale: float = 1.5
    vertical_scale: float = 1.0
    delay_scale: float = 2.0
    maximum_excess_length_m: float = 240.0
    area_rank_weight_m: float = 0.35
    horizontal_control_shift: float = 3.0
    seed: int = 20_260_805


def circular_abs(value: np.ndarray, target: np.ndarray, period: float) -> np.ndarray:
    return np.abs(circular_delta(value, target, period))


def calibrate_array(
    fit_ids: np.ndarray,
    site_value: int,
    positions: np.ndarray,
    site: np.ndarray,
    bs_position: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    point_power: np.ndarray,
) -> tuple[float, float, int]:
    ids = fit_ids[(site[fit_ids] == site_value) & (point_power[fit_ids] > 1e-30)]
    coordinate = coordinates[ids]
    energy = component_energy(coefficients[ids])
    direct = positions[ids] - bs_position[None]
    length = np.linalg.norm(direct, axis=1)
    delay_error = circular_abs(coordinate[:, :, 2], length[:, None] / METERS_PER_DELAY_BIN, S_SC)
    provisional = np.argmin(
        delay_error - 0.02 * np.log(np.maximum(energy, 1e-30)), axis=1
    )
    keep = delay_error[np.arange(len(ids)), provisional] <= 0.40
    ids = ids[keep]
    coordinate = coordinate[keep]
    energy = energy[keep]
    provisional = provisional[keep]
    direct = direct[keep]
    length = length[keep]
    selected_energy = energy[np.arange(len(ids)), provisional]
    azimuth = np.arctan2(direct[:, 1], direct[:, 0])
    observed_h = coordinate[np.arange(len(ids)), provisional, 0]
    weights = np.sqrt(np.maximum(selected_energy, 1e-30))
    horizontal_grid = np.linspace(-math.pi, math.pi, 1441)
    horizontal_error = np.empty(len(horizontal_grid), dtype=np.float64)
    for row, orientation in enumerate(horizontal_grid):
        expected = np.mod(8.0 * np.sin(azimuth - orientation), MH)
        horizontal_error[row] = np.average(
            circular_abs(observed_h, expected, MH), weights=weights
        )
    horizontal = float(horizontal_grid[int(np.argmin(horizontal_error))])
    expected_h = np.mod(8.0 * np.sin(azimuth - horizontal), MH)
    score = (
        delay_error[keep] / 0.25
        + circular_abs(coordinate[:, :, 0], expected_h[:, None], MH) / 0.80
        - 0.15
        * np.log(
            np.maximum(
                energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-30),
                1e-8,
            )
        )
    )
    selected = np.argmin(score, axis=1)
    observed_v = coordinate[np.arange(len(ids)), selected, 1]
    elevation = np.arctan2(
        direct[:, 2], np.maximum(np.linalg.norm(direct[:, :2], axis=1), 1e-9)
    )
    selected_weight = np.sqrt(np.maximum(energy[np.arange(len(ids)), selected], 1e-30))
    vertical_grid = np.linspace(-math.pi / 2.0, math.pi / 2.0, 1441)
    vertical_error = np.empty(len(vertical_grid), dtype=np.float64)
    for row, orientation in enumerate(vertical_grid):
        expected = np.mod(4.0 * np.sin(elevation - orientation), MV)
        vertical_error[row] = np.average(
            circular_abs(observed_v, expected, MV), weights=selected_weight
        )
    vertical = float(vertical_grid[int(np.argmin(vertical_error))])
    return horizontal, vertical, int(len(ids))


def wall_candidates(
    position: np.ndarray,
    bs_position: np.ndarray,
    sources: np.ndarray,
    wall_normal: np.ndarray,
    wall_offset: np.ndarray,
    tangent_min: np.ndarray,
    tangent_max: np.ndarray,
    z_min: np.ndarray,
    z_max: np.ndarray,
    area: np.ndarray,
    horizontal_orientation: float,
    vertical_orientation: float,
    maximum_candidates: int,
    config: Config,
) -> np.ndarray:
    delta = position[None] - sources
    length = np.linalg.norm(delta, axis=1)
    denominator = np.sum(delta[:, :2] * wall_normal, axis=1)
    numerator = wall_offset - np.sum(sources[:, :2] * wall_normal, axis=1)
    parameter = numerator / np.where(np.abs(denominator) > 1e-12, denominator, np.nan)
    reflection = sources + parameter[:, None] * delta
    tangent = np.column_stack([-wall_normal[:, 1], wall_normal[:, 0]])
    tangent_coordinate = np.sum(reflection[:, :2] * tangent, axis=1)
    valid = (
        np.isfinite(parameter)
        & (parameter > 0.0)
        & (parameter < 1.0)
        & (tangent_coordinate >= tangent_min - 1.0)
        & (tangent_coordinate <= tangent_max + 1.0)
        & (reflection[:, 2] >= z_min - 1.0)
        & (reflection[:, 2] <= z_max + 1.0)
    )
    direct_length = float(np.linalg.norm(position - bs_position))
    valid &= length <= direct_length + config.maximum_excess_length_m
    ids = np.flatnonzero(valid)
    if len(ids):
        priority = length[ids] - config.area_rank_weight_m * np.log1p(area[ids])
        ids = ids[np.argsort(priority)[: max(maximum_candidates - 1, 0)]]
        departure = reflection[ids] - bs_position[None]
        azimuth = np.arctan2(departure[:, 1], departure[:, 0])
        elevation = np.arctan2(
            departure[:, 2], np.maximum(np.linalg.norm(departure[:, :2], axis=1), 1e-9)
        )
        reflected = np.column_stack(
            [
                np.mod(8.0 * np.sin(azimuth - horizontal_orientation), MH),
                np.mod(4.0 * np.sin(elevation - vertical_orientation), MV),
                np.mod(length[ids] / METERS_PER_DELAY_BIN, S_SC),
            ]
        )
    else:
        reflected = np.empty((0, 3), dtype=np.float64)
    direct = position - bs_position
    direct_azimuth = math.atan2(direct[1], direct[0])
    direct_elevation = math.atan2(direct[2], max(float(np.linalg.norm(direct[:2])), 1e-9))
    los = np.asarray(
        [
            np.mod(8.0 * math.sin(direct_azimuth - horizontal_orientation), MH),
            np.mod(4.0 * math.sin(direct_elevation - vertical_orientation), MV),
            np.mod(direct_length / METERS_PER_DELAY_BIN, S_SC),
        ],
        dtype=np.float64,
    )
    return np.vstack([los[None], reflected]).astype(np.float32)


def match_candidates(
    candidates: np.ndarray,
    truth_coordinates: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    h = circular_abs(candidates[:, None, 0], truth_coordinates[None, :, 0], MH)
    v = circular_abs(candidates[:, None, 1], truth_coordinates[None, :, 1], MV)
    d = circular_abs(candidates[:, None, 2], truth_coordinates[None, :, 2], S_SC)
    cost = (
        (h / config.horizontal_scale) ** 2
        + (v / config.vertical_scale) ** 2
        + (d / config.delay_scale) ** 2
    )
    candidate_rows, truth_columns = linear_sum_assignment(cost)
    order = np.argsort(truth_columns)
    return candidate_rows[order], truth_columns[order]


def summarize(values: np.ndarray, fold_id: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [
            float(np.mean(values[fold_id == fold])) for fold in np.unique(fold_id)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (COMPONENT_CACHE, OOF_META, FOLD_CACHE, TRAIN_PAS, TRAIN_PDP):
        if not path.is_file():
            raise FileNotFoundError(path)
    setup, positions, _, site, _ = load_positions()
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    folds = np.load(FOLD_CACHE)
    component = np.load(COMPONENT_CACHE)
    coordinates = component["coordinates"].astype(np.float32)
    coefficients = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    wall_config = WallConfig()
    wall = build_wall_bank(wall_config)
    wall_normal = canonicalize_normals(wall.normal)
    sources = {
        int(site_value): image_sources(wall, setup.bs_positions[int(site_value)])
        for site_value in np.unique(site)
    }
    calibration: dict[tuple[int, int], tuple[float, float]] = {}
    calibration_rows: list[dict[str, object]] = []
    for fold in np.unique(fold_id):
        fit_ids = folds[f"fold{int(fold)}_fit"].astype(np.int64)
        for site_value in np.unique(site):
            horizontal, vertical, count = calibrate_array(
                fit_ids,
                int(site_value),
                positions,
                site,
                setup.bs_positions[int(site_value)],
                coordinates,
                coefficients,
                point_power,
            )
            calibration[(int(fold), int(site_value))] = (horizontal, vertical)
            calibration_rows.append(
                {
                    "fold": int(fold),
                    "site": int(site_value),
                    "horizontal_orientation_degrees": float(np.degrees(horizontal)),
                    "vertical_orientation_degrees": float(np.degrees(vertical)),
                    "los_calibration_count": count,
                }
            )

    max_candidates = max(config.candidate_counts)
    candidate_values = np.zeros((len(query_ids), max_candidates, 3), dtype=np.float32)
    candidate_valid = np.zeros((len(query_ids), max_candidates), dtype=bool)
    for row, (query_id, fold) in enumerate(zip(query_ids, fold_id)):
        site_value = int(site[query_id])
        horizontal, vertical = calibration[(int(fold), site_value)]
        candidates = wall_candidates(
            positions[query_id],
            setup.bs_positions[site_value],
            sources[site_value],
            wall_normal,
            wall.offset,
            wall.tangent_min,
            wall.tangent_max,
            wall.z_min,
            wall.z_max,
            wall.area,
            horizontal,
            vertical,
            max_candidates,
            config,
        )
        count = min(len(candidates), max_candidates)
        candidate_values[row, :count] = candidates[:count]
        candidate_valid[row, :count] = True
        if (row + 1) % 50 == 0:
            print(f"map candidates {row + 1}/{len(query_ids)}", flush=True)

    truth_coordinates = coordinates[query_ids, : config.true_paths]
    truth_coefficients = coefficients[query_ids, : config.true_paths]
    truth_energy = component_energy(truth_coefficients)
    truth_energy /= np.maximum(truth_energy.sum(axis=1, keepdims=True), 1e-30)
    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    match_diagnostics: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(config.seed)
    point_permutation = rng.permutation(len(query_ids))
    for candidate_count in config.candidate_counts:
        matched = np.zeros((len(query_ids), config.true_paths, 3), dtype=np.float32)
        matched_coefficients = np.zeros(
            (len(query_ids), config.true_paths, coefficients.shape[2]), dtype=np.complex64
        )
        match_cost = np.empty((len(query_ids), config.true_paths), dtype=np.float32)
        for row in range(len(query_ids)):
            candidates = candidate_values[row, :candidate_count]
            valid = candidate_valid[row, :candidate_count]
            candidates = candidates[valid]
            if len(candidates) < config.true_paths:
                padding = np.repeat(candidates[:1], config.true_paths - len(candidates), axis=0)
                candidates = np.concatenate([candidates, padding], axis=0)
            candidate_rows, truth_columns = match_candidates(
                candidates, truth_coordinates[row], config
            )
            matched[row, truth_columns] = candidates[candidate_rows]
            matched_coefficients[row, truth_columns] = truth_coefficients[row, truth_columns]
            delta_h = circular_abs(
                matched[row, :, 0], truth_coordinates[row, :, 0], MH
            )
            delta_v = circular_abs(
                matched[row, :, 1], truth_coordinates[row, :, 1], MV
            )
            delta_d = circular_abs(
                matched[row, :, 2], truth_coordinates[row, :, 2], S_SC
            )
            match_cost[row] = np.sqrt(
                (delta_h / config.horizontal_scale) ** 2
                + (delta_v / config.vertical_scale) ** 2
                + (delta_d / config.delay_scale) ** 2
            )
        name = f"map_top{candidate_count}"
        cases[name] = (matched, matched_coefficients)
        shifted = matched.copy()
        shifted[:, :, 0] = np.mod(
            shifted[:, :, 0] + config.horizontal_control_shift, MH
        )
        cases[f"{name}_hshift_control"] = (shifted, matched_coefficients)
        delay_control = matched.copy()
        delay_control[:, :, 2] = matched[point_permutation, :, 2]
        cases[f"{name}_delay_shuffle_control"] = (delay_control, matched_coefficients)
        match_diagnostics[name] = {
            "energy_weighted_support_cost": float(
                np.mean(np.sum(match_cost * truth_energy, axis=1))
            ),
            "energy_fraction_cost_below_1": float(
                np.mean(np.sum((match_cost < 1.0) * truth_energy, axis=1))
            ),
            "energy_fraction_cost_below_2": float(
                np.mean(np.sum((match_cost < 2.0) * truth_energy, axis=1))
            ),
        }

    truth_pas = np.load(TRAIN_PAS, mmap_mode="r")
    truth_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    metrics = {
        name: {
            "pas": np.empty(len(query_ids), dtype=np.float64),
            "pdp": np.empty(len(query_ids), dtype=np.float64),
        }
        for name in cases
    }
    device = torch.device(args.device)
    for start in range(0, len(query_ids), args.batch_size):
        end = min(start + args.batch_size, len(query_ids))
        ids = query_ids[start:end]
        block_pas = np.asarray(truth_pas[ids], dtype=np.float32)
        block_pdp = np.asarray(truth_pdp[ids], dtype=np.float32)
        for name, (case_coordinates, case_coefficients) in cases.items():
            _, pas, pdp = render_noncoherent(
                case_coordinates[start:end], case_coefficients[start:end], device
            )
            metrics[name]["pas"][start:end] = cosine(
                pas, block_pas, 1
            ).mean(axis=(1, 2), dtype=np.float64)
            metrics[name]["pdp"][start:end] = cosine(
                pdp, block_pdp, 3
            ).mean(axis=(1, 2), dtype=np.float64)
        print(f"map support evaluation {end}/{len(query_ids)}", flush=True)

    result = {
        name: {
            "pas": summarize(value["pas"], fold_id),
            "pdp": summarize(value["pdp"], fold_id),
        }
        for name, value in metrics.items()
    }
    np.savez_compressed(
        CANDIDATE_CACHE,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        candidates=candidate_values.astype(np.float16),
        valid=candidate_valid,
    )
    payload = {
        "experiment": "Round2-E19-map-path-support-ceiling",
        "status": "analysis_only",
        "primary_hypothesis": (
            "Official-map LOS and finite-wall one-bounce paths add deployable delay-angle "
            "candidate locations beyond local spectral interpolation."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "calibration": calibration_rows,
        "candidate_count": {
            "mean": float(np.mean(candidate_valid.sum(axis=1))),
            "min": int(np.min(candidate_valid.sum(axis=1))),
            "max": int(np.max(candidate_valid.sum(axis=1))),
        },
        "match_diagnostics": match_diagnostics,
        "results": result,
        "outputs": {
            "candidates": PATHS.relative(CANDIDATE_CACHE),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
