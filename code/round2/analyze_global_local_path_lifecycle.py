#!/usr/bin/env python3
"""Learn path birth/death selection from global map and local path tracks.

The experiment keeps the existing fit-only path graph as a deliberately broad
candidate generator.  A cross-fitted model then selects paths jointly from
candidate persistence, energy, port profile, query geometry, and official-map
features.  Every outer fold removes the complete spatial hole before either
candidate construction or model fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from analyze_v4_path_track_spectral_targets import (
    MH,
    MV,
    N_UE,
    S_SC,
    TRAIN_PAS,
    TRAIN_PDP,
    cosine,
    periodic_delta,
    render_noncoherent,
)
from data import load_positions
from project_paths import PATHS
from run_path_track_spectral_field import (
    Config as TrackConfig,
    build_fold_tracks,
    component_energy,
    predict_track_candidates,
    valid_components,
)


VERSION = "Round2-E79-global-local-path-lifecycle"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "global_local_path_lifecycle"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_global_local_path_lifecycle"
REPORT = ANALYSIS_DIR / "global_local_path_lifecycle.json"
PREDICTIONS = ARTIFACT_DIR / "oof_path_lifecycle_predictions.npz"

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
    PATHS.artifacts
    / "round2"
    / "foundation"
    / "round2_spatial_hole_folds.npz"
)
GEODESIC_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)

TINY = 1e-30
PORTS = 2 * N_UE
MAP_FEATURE_COUNT = 28
POINT_CONTEXT_DIM = 18
LOCAL_FEATURE_DIM = 30 + POINT_CONTEXT_DIM


@dataclass(frozen=True)
class Config:
    track_components: int = 20
    output_paths: int = 24
    donor_count: int = 96
    maximum_candidates: int = 176
    match_cost_limit: float = 4.0
    minimum_truth_fraction: float = 1e-5
    trees: int = 480
    classifier_leaf: int = 6
    regressor_leaf: int = 5
    max_features: float = 0.72
    random_state: int = 2026080791


@dataclass(frozen=True)
class RenderSpec:
    gamma: float
    top_k: int
    support_scale: float
    port_scale: float

    @property
    def name(self) -> str:
        return (
            f"g{self.gamma:.2f}_k{self.top_k}_"
            f"s{self.support_scale:.2f}_p{self.port_scale:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trees", type=int, default=480)
    parser.add_argument("--force-candidates", action="store_true")
    return parser.parse_args()


def rank_fraction(value: np.ndarray, descending: bool = True) -> np.ndarray:
    order = np.argsort(-value if descending else value, kind="stable")
    rank = np.empty(len(value), dtype=np.float32)
    rank[order] = np.arange(len(value), dtype=np.float32)
    return rank / max(len(value) - 1, 1)


def support_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    horizontal = periodic_delta(left[:, None, 0], right[None, :, 0], MH)
    vertical = periodic_delta(left[:, None, 1], right[None, :, 1], MV)
    delay = periodic_delta(left[:, None, 2], right[None, :, 2], S_SC)
    return (horizontal / 1.5) ** 2 + vertical**2 + (delay / 2.0) ** 2


def query_context(
    point_id: int,
    bank: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
) -> np.ndarray:
    local_bank = bank[site[bank] == site[point_id]]
    take = min(64, len(local_bank))
    distance, local = cKDTree(positions[local_bank, :2]).query(
        positions[point_id, :2], k=take, workers=-1
    )
    distance = np.atleast_1d(distance).astype(np.float64)
    donors = local_bank[np.atleast_1d(local).astype(np.int64)]
    valid = point_power[donors] > TINY
    log_power = np.log10(np.maximum(point_power[donors], 1e-30))
    weight = np.exp(-distance / 12.0)
    weight /= max(float(weight.sum()), TINY)
    positive_weight = weight * valid
    positive_total = max(float(positive_weight.sum()), TINY)
    power_mean = float(np.sum(positive_weight * log_power) / positive_total)
    power_std = float(
        np.sqrt(
            np.sum(positive_weight * np.square(log_power - power_mean))
            / positive_total
        )
    )
    relative = positions[point_id, :2] - bs_positions[int(site[point_id]), :2]
    radial = max(float(np.linalg.norm(relative)), 1e-9)
    angle = math.atan2(float(relative[1]), float(relative[0]))
    offsets = positions[donors, :2] - positions[point_id, :2]
    covariance = offsets.T @ (weight[:, None] * offsets)
    eigenvalues = np.linalg.eigvalsh(covariance)
    isotropy = float(eigenvalues[0] / max(eigenvalues[-1], 1e-9))
    quadrant = (
        (offsets[:, 0] >= 0).astype(np.int8)
        + 2 * (offsets[:, 1] >= 0).astype(np.int8)
    )
    coverage = float(len(np.unique(quadrant))) / 4.0
    quantiles = np.quantile(distance, (0.0, 0.25, 0.50, 0.90)) / 30.0
    zero_rates = [
        float(np.mean(~valid[: min(limit, len(valid))])) for limit in (8, 24, 64)
    ]
    return np.asarray(
        [
            relative[0] / 150.0,
            relative[1] / 150.0,
            radial / 200.0,
            math.sin(angle),
            math.cos(angle),
            float(site[point_id] == 0),
            float(site[point_id] == 1),
            *quantiles.tolist(),
            *zero_rates,
            power_mean / 20.0,
            power_std / 10.0,
            isotropy,
            coverage,
        ],
        dtype=np.float32,
    )


def candidate_features(
    candidates: list[tuple[np.ndarray, np.ndarray, float, float]],
    point_context: np.ndarray,
    maximum_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not candidates:
        return (
            np.empty((0, LOCAL_FEATURE_DIM), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, PORTS), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    coordinate = np.asarray([row[0] for row in candidates], dtype=np.float32)
    profile = np.asarray([row[1] for row in candidates], dtype=np.float32)
    fraction = np.asarray([row[2] for row in candidates], dtype=np.float32)
    presence = np.asarray([row[3] for row in candidates], dtype=np.float32)
    score = fraction * np.sqrt(np.maximum(presence, 1e-8))
    keep = np.argsort(-score, kind="stable")[:maximum_candidates]
    coordinate = coordinate[keep]
    profile = profile[keep]
    fraction = fraction[keep]
    presence = presence[keep]
    score = score[keep]
    profile = np.maximum(profile, 0.0)
    profile /= np.maximum(profile.sum(axis=1, keepdims=True), TINY)

    pair_cost = support_cost(coordinate, coordinate)
    np.fill_diagonal(pair_cost, np.inf)
    nearest = np.min(pair_cost, axis=1) if len(coordinate) > 1 else np.full(1, 99.0)
    density = (
        np.exp(-np.minimum(pair_cost, 30.0)).sum(axis=1)
        if len(coordinate) > 1
        else np.zeros(1)
    )
    count1 = np.sum(pair_cost <= 1.0, axis=1)
    count4 = np.sum(pair_cost <= 4.0, axis=1)
    phase = np.column_stack(
        [
            np.sin(2.0 * np.pi * coordinate[:, 0] / MH),
            np.cos(2.0 * np.pi * coordinate[:, 0] / MH),
            np.sin(2.0 * np.pi * coordinate[:, 1] / MV),
            np.cos(2.0 * np.pi * coordinate[:, 1] / MV),
            np.sin(2.0 * np.pi * coordinate[:, 2] / S_SC),
            np.cos(2.0 * np.pi * coordinate[:, 2] / S_SC),
        ]
    ).astype(np.float32)
    normalized_coordinate = coordinate / np.asarray([MH, MV, S_SC], dtype=np.float32)
    repeated = np.broadcast_to(point_context, (len(coordinate), len(point_context)))
    features = np.column_stack(
        [
            phase,
            normalized_coordinate,
            np.sqrt(profile),
            np.log(np.maximum(fraction, 1e-10)),
            np.log(np.maximum(presence, 1e-8)),
            np.log(np.maximum(score, 1e-12)),
            fraction,
            presence,
            rank_fraction(score),
            rank_fraction(fraction),
            rank_fraction(presence),
            np.minimum(nearest, 30.0) / 30.0,
            density / 12.0,
            np.minimum(count1, 12) / 12.0,
            np.minimum(count4, 24) / 24.0,
            np.full(len(coordinate), len(coordinate) / maximum_candidates),
            repeated,
        ]
    ).astype(np.float32)
    return features, coordinate, profile, fraction, presence


def candidate_targets(
    coordinate: np.ndarray,
    true_coordinate: np.ndarray,
    true_coefficient: np.ndarray,
    point_power: float,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(coordinate)
    mass = np.zeros(count, dtype=np.float32)
    delta = np.zeros((count, 3), dtype=np.float32)
    port = np.full((count, PORTS), 1.0 / PORTS, dtype=np.float32)
    match_cost = np.full(count, np.inf, dtype=np.float32)
    if count == 0 or point_power <= TINY:
        return mass, delta, port, match_cost
    truth_energy = np.sum(np.abs(true_coefficient) ** 2, axis=1, dtype=np.float64)
    truth_fraction = truth_energy / max(point_power, TINY)
    valid = truth_fraction >= config.minimum_truth_fraction
    if not np.any(valid):
        return mass, delta, port, match_cost
    truth_ids = np.flatnonzero(valid)
    cost = support_cost(coordinate, true_coordinate[truth_ids])
    candidate_rows, truth_columns = linear_sum_assignment(cost)
    selected_cost = cost[candidate_rows, truth_columns]
    accepted = selected_cost <= config.match_cost_limit
    candidate_rows = candidate_rows[accepted]
    truth_ids = truth_ids[truth_columns[accepted]]
    match_cost[candidate_rows] = selected_cost[accepted].astype(np.float32)
    mass[candidate_rows] = truth_fraction[truth_ids].astype(np.float32)
    delta[candidate_rows, 0] = periodic_delta(
        true_coordinate[truth_ids, 0], coordinate[candidate_rows, 0], MH
    )
    delta[candidate_rows, 1] = periodic_delta(
        true_coordinate[truth_ids, 1], coordinate[candidate_rows, 1], MV
    )
    delta[candidate_rows, 2] = periodic_delta(
        true_coordinate[truth_ids, 2], coordinate[candidate_rows, 2], S_SC
    )
    local_port = np.abs(true_coefficient[truth_ids]) ** 2
    local_port /= np.maximum(local_port.sum(axis=1, keepdims=True), TINY)
    port[candidate_rows] = local_port.astype(np.float32)
    return mass, delta, port, match_cost


def candidate_cache_path(outer: int, config: Config) -> object:
    return ARTIFACT_DIR / (
        f"outer{outer}_candidate_pool_n{config.donor_count}_"
        f"c{config.maximum_candidates}.npz"
    )


def build_candidate_cache(
    outer: int,
    config: Config,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    point_power: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    force: bool,
) -> dict[str, np.ndarray]:
    path = candidate_cache_path(outer, config)
    if path.is_file() and not force:
        cached = np.load(path, allow_pickle=False)
        return {key: cached[key] for key in cached.files}
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    outer_fit = folds[f"fold{outer}_fit"].astype(np.int64)
    valid, fraction = valid_components(
        component_energy(coefficients),
        point_power,
        config.track_components,
        1e-5,
    )
    track_config = TrackConfig(
        component_count=config.track_components,
        output_paths=config.output_paths,
        donor_count=config.donor_count,
    )
    offsets = np.zeros(len(query_ids) + 1, dtype=np.int64)
    feature_parts: list[np.ndarray] = []
    coordinate_parts: list[np.ndarray] = []
    profile_parts: list[np.ndarray] = []
    fraction_parts: list[np.ndarray] = []
    presence_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    port_parts: list[np.ndarray] = []
    cost_parts: list[np.ndarray] = []
    cursor = 0
    for context_fold in range(5):
        rows = np.flatnonzero(fold_id == context_fold)
        if context_fold == outer:
            bank = outer_fit
        else:
            inner_fit = folds[f"fold{context_fold}_fit"].astype(np.int64)
            bank = np.intersect1d(outer_fit, inner_fit, assume_unique=True)
        roots, diagnostics = build_fold_tracks(
            bank,
            positions,
            site,
            coordinates,
            coefficients,
            valid,
            fraction,
            track_config,
        )
        for row in rows:
            point_id = int(query_ids[row])
            local_context = query_context(
                point_id,
                bank,
                positions,
                site,
                bs_positions,
                point_power,
            )
            nonzero_bank = bank[
                (site[bank] == site[point_id]) & (point_power[bank] > TINY)
            ]
            take = min(config.donor_count, len(nonzero_bank))
            if take:
                distance, local = cKDTree(positions[nonzero_bank, :2]).query(
                    positions[point_id, :2], k=take, workers=-1
                )
                distance = np.atleast_1d(distance)
                donors = nonzero_bank[np.atleast_1d(local).astype(np.int64)]
                candidates, _ = predict_track_candidates(
                    positions[point_id],
                    donors,
                    distance,
                    positions,
                    coordinates,
                    coefficients,
                    valid,
                    fraction,
                    roots,
                    track_config,
                )
            else:
                candidates = []
            features, local_coordinate, profile, local_fraction, presence = (
                candidate_features(
                    candidates, local_context, config.maximum_candidates
                )
            )
            mass, delta, target_port, match_cost = candidate_targets(
                local_coordinate,
                coordinates[point_id, :24],
                coefficients[point_id, :24],
                float(point_power[point_id]),
                config,
            )
            feature_parts.append(features)
            coordinate_parts.append(local_coordinate)
            profile_parts.append(profile)
            fraction_parts.append(local_fraction)
            presence_parts.append(presence)
            mass_parts.append(mass)
            delta_parts.append(delta)
            port_parts.append(target_port)
            cost_parts.append(match_cost)
            cursor += len(features)
            offsets[row + 1] = cursor
        print(
            f"outer={outer} context_fold={context_fold} rows={len(rows)} "
            f"tracks={diagnostics['track_count']} candidates={cursor}",
            flush=True,
        )
    # Rows were generated by fold rather than row order. Repack them into canonical
    # row order so offsets remain a simple point-to-candidate lookup.
    generated_rows = np.concatenate(
        [np.flatnonzero(fold_id == context_fold) for context_fold in range(5)]
    )
    by_row = {int(row): index for index, row in enumerate(generated_rows)}
    ordered_features = [feature_parts[by_row[row]] for row in range(len(query_ids))]
    ordered_coordinates = [coordinate_parts[by_row[row]] for row in range(len(query_ids))]
    ordered_profiles = [profile_parts[by_row[row]] for row in range(len(query_ids))]
    ordered_fraction = [fraction_parts[by_row[row]] for row in range(len(query_ids))]
    ordered_presence = [presence_parts[by_row[row]] for row in range(len(query_ids))]
    ordered_mass = [mass_parts[by_row[row]] for row in range(len(query_ids))]
    ordered_delta = [delta_parts[by_row[row]] for row in range(len(query_ids))]
    ordered_port = [port_parts[by_row[row]] for row in range(len(query_ids))]
    ordered_cost = [cost_parts[by_row[row]] for row in range(len(query_ids))]
    counts = np.asarray([len(value) for value in ordered_features], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def concatenate(parts: list[np.ndarray], shape: tuple[int, ...], dtype: object) -> np.ndarray:
        if int(offsets[-1]) == 0:
            return np.empty((0, *shape), dtype=dtype)
        return np.concatenate(parts, axis=0).astype(dtype, copy=False)

    payload = {
        "offsets": offsets,
        "local_features": concatenate(
            ordered_features,
            (LOCAL_FEATURE_DIM,),
            np.float32,
        ),
        "coordinates": concatenate(ordered_coordinates, (3,), np.float32),
        "profiles": concatenate(ordered_profiles, (PORTS,), np.float32),
        "fractions": concatenate(ordered_fraction, (), np.float32),
        "presences": concatenate(ordered_presence, (), np.float32),
        "target_mass": concatenate(ordered_mass, (), np.float32),
        "target_delta": concatenate(ordered_delta, (3,), np.float32),
        "target_port": concatenate(ordered_port, (PORTS,), np.float32),
        "match_cost": concatenate(ordered_cost, (), np.float32),
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id.astype(np.int8),
        "outer": np.int8(outer),
    }
    np.savez_compressed(path, **payload)
    return payload


def row_ids(offsets: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(offsets) - 1, dtype=np.int32), np.diff(offsets))


def expanded_features(
    cache: dict[str, np.ndarray],
    geodesic: np.ndarray,
    query_ids: np.ndarray,
    include_map: bool,
) -> np.ndarray:
    local = np.asarray(cache["local_features"], dtype=np.float32)
    if not include_map:
        return local
    rows = row_ids(cache["offsets"])
    global_features = np.asarray(geodesic[query_ids[rows]], dtype=np.float32)
    global_features = np.sign(global_features) * np.log1p(np.abs(global_features))
    return np.concatenate((local, global_features), axis=1).astype(np.float32)


def fit_models(
    features: np.ndarray,
    mass: np.ndarray,
    delta: np.ndarray,
    port: np.ndarray,
    config: Config,
    seed: int,
) -> tuple[ExtraTreesClassifier, ExtraTreesRegressor]:
    positive = mass > 0.0
    if np.sum(positive) < 100 or np.sum(~positive) < 100:
        raise RuntimeError("candidate labels do not contain both classes")
    classifier = ExtraTreesClassifier(
        n_estimators=config.trees,
        min_samples_leaf=config.classifier_leaf,
        max_features=config.max_features,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    classifier.fit(features, positive)
    target = np.concatenate(
        (
            np.log(np.maximum(mass[positive], 1e-10))[:, None],
            delta[positive],
            np.sqrt(np.maximum(port[positive], 0.0)),
        ),
        axis=1,
    )
    regressor = ExtraTreesRegressor(
        n_estimators=config.trees,
        min_samples_leaf=config.regressor_leaf,
        max_features=config.max_features,
        n_jobs=-1,
        random_state=seed + 1,
    )
    regressor.fit(features[positive], target)
    return classifier, regressor


def render_specs() -> tuple[RenderSpec, ...]:
    return tuple(
        RenderSpec(gamma, top_k, support_scale, port_scale)
        for gamma in (0.50, 0.75, 1.00)
        for top_k in (12, 16, 24)
        for support_scale in (0.50, 1.00)
        for port_scale in (0.0, 0.50)
    )


def predict_candidate_set(
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    probability: np.ndarray,
    regression: np.ndarray,
    spec: RenderSpec,
) -> tuple[np.ndarray, np.ndarray]:
    output_coordinate = np.zeros((len(rows), 24, 3), dtype=np.float32)
    output_coefficient = np.zeros((len(rows), 24, PORTS), dtype=np.complex64)
    offsets = cache["offsets"]
    coordinate = cache["coordinates"]
    donor_profile = cache["profiles"]
    for output_row, point_row in enumerate(rows):
        start = int(offsets[point_row])
        end = int(offsets[point_row + 1])
        if end <= start:
            continue
        local_probability = np.clip(probability[start:end], 1e-6, 1.0)
        local_regression = regression[start:end]
        mass = np.exp(np.clip(local_regression[:, 0], -14.0, 2.0))
        score = np.power(local_probability, spec.gamma) * mass
        selected = np.argsort(-score, kind="stable")[: spec.top_k]
        count = min(len(selected), 24)
        selected = selected[:count]
        predicted_coordinate = np.array(coordinate[start:end][selected], copy=True)
        predicted_coordinate += spec.support_scale * local_regression[selected, 1:4]
        predicted_coordinate %= np.asarray([MH, MV, S_SC], dtype=np.float32)
        donor_amplitude = np.sqrt(
            np.maximum(donor_profile[start:end][selected], 0.0)
        )
        predicted_amplitude = np.maximum(local_regression[selected, 4:], 0.0)
        amplitude = (
            (1.0 - spec.port_scale) * donor_amplitude
            + spec.port_scale * predicted_amplitude
        )
        profile = np.square(amplitude)
        profile /= np.maximum(profile.sum(axis=1, keepdims=True), TINY)
        output_coordinate[output_row, :count] = predicted_coordinate
        output_coefficient[output_row, :count] = np.sqrt(
            np.maximum(score[selected, None] * profile, 0.0)
        ).astype(np.complex64)
    return output_coordinate, output_coefficient


def summarize(values: np.ndarray, fold_id: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [
            float(np.mean(values[fold_id == fold])) for fold in range(5)
        ],
    }


def evaluate_variants(
    variants: dict[str, tuple[np.ndarray, np.ndarray]],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    truth_pas = np.load(TRAIN_PAS, mmap_mode="r")
    truth_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    point_power = np.load(COMPONENTS)["point_power"][query_ids]
    nonzero = point_power > TINY
    output: dict[str, object] = {}
    for name, (coordinates, coefficients) in variants.items():
        pas = np.zeros(len(query_ids), dtype=np.float64)
        pdp = np.zeros(len(query_ids), dtype=np.float64)
        for start in range(0, len(query_ids), batch_size):
            end = min(start + batch_size, len(query_ids))
            _, predicted_pas, predicted_pdp = render_noncoherent(
                coordinates[start:end], coefficients[start:end], device
            )
            ids = query_ids[start:end]
            pas[start:end] = cosine(
                predicted_pas,
                np.asarray(truth_pas[ids], dtype=np.float32),
                1,
            ).mean(axis=(1, 2), dtype=np.float64)
            pdp[start:end] = cosine(
                predicted_pdp,
                np.asarray(truth_pdp[ids], dtype=np.float32),
                3,
            ).mean(axis=(1, 2), dtype=np.float64)
        output[name] = {
            "pas": summarize(pas, fold_id),
            "pdp": summarize(pdp, fold_id),
            "pas_pdp_mean": float(0.5 * (np.mean(pas) + np.mean(pdp))),
            "nonzero_pas": float(np.mean(pas[nonzero])),
            "nonzero_pdp": float(np.mean(pdp[nonzero])),
            "nonzero_pas_pdp_mean": float(
                0.5 * (np.mean(pas[nonzero]) + np.mean(pdp[nonzero]))
            ),
        }
        print(
            f"evaluate {name} mean={output[name]['pas_pdp_mean']:.6f}",
            flush=True,
        )
    return output


def model_diagnostics(
    target_mass: np.ndarray,
    probability: np.ndarray,
    fold_id: np.ndarray,
    candidate_rows: np.ndarray,
) -> dict[str, object]:
    target = target_mass > 0.0
    auc = roc_auc_score(target, probability) if len(np.unique(target)) == 2 else math.nan
    ap = average_precision_score(target, probability) if np.any(target) else math.nan
    fold_rows = []
    for fold in range(5):
        selected = fold_id[candidate_rows] == fold
        local_target = target[selected]
        local_probability = probability[selected]
        fold_rows.append(
            {
                "fold": fold,
                "count": int(np.sum(selected)),
                "positive_fraction": float(np.mean(local_target)),
                "auc": float(roc_auc_score(local_target, local_probability))
                if len(np.unique(local_target)) == 2
                else math.nan,
                "average_precision": float(
                    average_precision_score(local_target, local_probability)
                )
                if np.any(local_target)
                else math.nan,
            }
        )
    return {
        "count": int(len(target)),
        "positive_fraction": float(np.mean(target)),
        "auc": float(auc),
        "average_precision": float(ap),
        "folds": fold_rows,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(trees=args.trees)
    setup, positions, _, site, _ = load_positions()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    component = np.load(COMPONENTS, allow_pickle=False)
    coordinates = component["coordinates"].astype(np.float32)
    coefficients = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    geodesic = np.load(GEODESIC_CACHE, allow_pickle=False)["train_features"].astype(
        np.float32
    )

    specs = render_specs()
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group in ("local", "global_local"):
        for spec in specs:
            variants[f"{group}_{spec.name}"] = (
                np.zeros((len(query_ids), 24, 3), dtype=np.float32),
                np.zeros((len(query_ids), 24, PORTS), dtype=np.complex64),
            )

    held_probability = {
        group: [] for group in ("local", "global_local")
    }
    held_mass: list[np.ndarray] = []
    held_fold: list[np.ndarray] = []
    held_candidate_rows: list[np.ndarray] = []
    pool_diagnostics = []
    for outer in range(5):
        cache = build_candidate_cache(
            outer,
            config,
            positions,
            site,
            setup.bs_positions,
            coordinates,
            coefficients,
            point_power,
            query_ids,
            fold_id,
            args.force_candidates,
        )
        candidate_point_rows = row_ids(cache["offsets"])
        train_candidates = fold_id[candidate_point_rows] != outer
        held_candidates = fold_id[candidate_point_rows] == outer
        held_rows = np.flatnonzero(fold_id == outer)
        held_mass.append(cache["target_mass"][held_candidates])
        held_fold.append(np.full(np.sum(held_candidates), outer, dtype=np.int8))
        held_candidate_rows.append(candidate_point_rows[held_candidates])
        pool_diagnostics.append(
            {
                "outer": outer,
                "train_candidates": int(np.sum(train_candidates)),
                "held_candidates": int(np.sum(held_candidates)),
                "held_positive_fraction": float(
                    np.mean(cache["target_mass"][held_candidates] > 0.0)
                ),
                "held_pool_median": float(
                    np.median(np.diff(cache["offsets"])[held_rows])
                ),
            }
        )
        for group, include_map in (("local", False), ("global_local", True)):
            feature = expanded_features(cache, geodesic, query_ids, include_map)
            classifier, regressor = fit_models(
                feature[train_candidates],
                cache["target_mass"][train_candidates],
                cache["target_delta"][train_candidates],
                cache["target_port"][train_candidates],
                config,
                config.random_state + 1000 * outer + (100 if include_map else 0),
            )
            probability = classifier.predict_proba(feature)[:, 1].astype(np.float32)
            regression = regressor.predict(feature).astype(np.float32)
            held_probability[group].append(probability[held_candidates])
            for spec in specs:
                local_coordinate, local_coefficient = predict_candidate_set(
                    cache, held_rows, probability, regression, spec
                )
                target_coordinate, target_coefficient = variants[
                    f"{group}_{spec.name}"
                ]
                target_coordinate[held_rows] = local_coordinate
                target_coefficient[held_rows] = local_coefficient
            print(
                f"outer={outer} group={group} train={np.sum(train_candidates)} "
                f"held={np.sum(held_candidates)}",
                flush=True,
            )

    np.savez_compressed(
        PREDICTIONS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        **{
            f"{name}_{part}": value[index]
            for name, value in variants.items()
            for index, part in enumerate(("coordinates", "coefficients"))
        },
    )
    results = evaluate_variants(
        variants, query_ids, fold_id, torch.device(args.device), args.batch_size
    )
    ranking = sorted(
        (
            {"name": name, **row}
            for name, row in results.items()
        ),
        key=lambda row: float(row["pas_pdp_mean"]),
        reverse=True,
    )
    target_mass = np.concatenate(held_mass)
    candidate_folds = np.concatenate(held_fold)
    candidate_rows = np.concatenate(held_candidate_rows)
    selection = {
        group: model_diagnostics(
            target_mass,
            np.concatenate(held_probability[group]),
            fold_id,
            candidate_rows,
        )
        for group in ("local", "global_local")
    }
    best_local = next(row for row in ranking if row["name"].startswith("local_"))
    best_global = next(
        row for row in ranking if row["name"].startswith("global_local_")
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A broad fit-only path pool already contains the missing paths; "
            "global map state plus local lifecycle evidence can learn which paths "
            "survive inside an unseen spatial hole."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": "fixed 500-point five-fold spatial holes; full outer hole removed",
        "config": asdict(config),
        "candidate_pool": pool_diagnostics,
        "selection_metrics": selection,
        "best_local": best_local,
        "best_global_local": best_global,
        "global_map_margin": float(
            best_global["pas_pdp_mean"] - best_local["pas_pdp_mean"]
        ),
        "top_candidates": ranking[:20],
        "prediction_cache": PATHS.relative(PREDICTIONS),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selection_metrics": selection,
                "best_local": best_local,
                "best_global_local": best_global,
                "global_map_margin": payload["global_map_margin"],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
