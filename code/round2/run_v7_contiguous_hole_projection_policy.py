#!/usr/bin/env python3
"""Learn V4 projection actions from target-shaped contiguous pseudo holes."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance_matrix
from sklearn.ensemble import ExtraTreesRegressor

from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import solve_linear_rbf_weights
from run_v3_anchored_neural_rbf import (
    AnchoredWeightCorrector,
    Config as V3Config,
    DESCRIPTORS,
    GEOMETRY,
)
from run_v4_structured_complex_field import (
    FOLD_CACHE,
    TrainConfig,
    channel_features,
    context_features,
)
from run_v7_augmented_projection_policy import (
    BASE_CANDIDATE_BY_SITE,
    V3_WEIGHTS,
    _actual_subset,
    _build_pseudo_base,
    _build_pseudo_targets,
    _choose_actions,
    _fit_group_calibration,
    _generate_pseudo_metrics,
    _labels,
    _point_features,
    _predicted_utility,
    _score_selection,
    _site_scales,
    greedy_candidates,
)


VERSION = "Round2-E50-contiguous-hole-projection-policy"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "contiguous_hole_projection_policy"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_contiguous_hole_projection_policy"
REPORT = ANALYSIS_DIR / "contiguous_hole_projection_policy.json"

V4_METRICS = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "calibrated_exact_projection_metrics.npz"
)
GEODESIC = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)

NEIGHBORS = 64
MAPPING_NEIGHBORS = 16
MIN_CENTER_SEPARATION_M = 18.0
GEO_COLUMNS = np.asarray(
    (0, 2, 3, 4, 10, 11, 12, 13, 15, 17, 22, 23, 25, 26, 27),
    dtype=np.int64,
)
TINY = 1e-30


@dataclass(frozen=True)
class AnalogHole:
    site: int
    center_xy: np.ndarray
    query_ids: np.ndarray
    guard_m: float
    mapping_median_m: float
    mapping_p90_m: float
    nearest_median_m: float
    nearest_p90_m: float
    feature_distance: float
    score: float


def _fold_paths(fold: int) -> dict[str, Path]:
    prefix = ARTIFACT_DIR / f"fold{fold}"
    return {
        "context": prefix.with_name(prefix.name + "_contiguous_context.npz"),
        "base": prefix.with_name(prefix.name + "_pseudo_base.npy"),
        "ad": prefix.with_name(prefix.name + "_pseudo_ad_f16.npy"),
        "raw_pas": prefix.with_name(prefix.name + "_pseudo_raw_pas_f16.npy"),
        "raw_pdp": prefix.with_name(prefix.name + "_pseudo_raw_pdp_f16.npy"),
        "cal_pas": prefix.with_name(prefix.name + "_pseudo_cal_pas_f16.npy"),
        "cal_pdp": prefix.with_name(prefix.name + "_pseudo_cal_pdp_f16.npy"),
        "metrics": prefix.with_name(prefix.name + "_pseudo_projection_metrics.npz"),
        "calibration": prefix.with_name(prefix.name + "_group_calibration.npz"),
        "selection": prefix.with_name(prefix.name + "_candidate_indices.json"),
    }


def _summary_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.mean(features, axis=0, dtype=np.float64),
        np.std(features, axis=0, dtype=np.float64),
    )


def _assigned_template(
    centered_template: np.ndarray,
    center_xy: np.ndarray,
    bank_ids: np.ndarray,
    positions: np.ndarray,
    tree: cKDTree,
) -> tuple[np.ndarray, np.ndarray] | None:
    desired = center_xy[None, :] + centered_template
    count = min(MAPPING_NEIGHBORS, len(bank_ids))
    while True:
        _, local = tree.query(desired, k=count, workers=-1)
        local = np.asarray(local, dtype=np.int64).reshape(len(desired), -1)
        candidates = np.unique(local)
        if len(candidates) >= len(desired) or count >= min(128, len(bank_ids)):
            break
        count = min(count * 2, 128, len(bank_ids))
    if len(candidates) < len(desired):
        return None
    cost = distance_matrix(desired, positions[bank_ids[candidates], :2])
    template_rows, candidate_columns = linear_sum_assignment(cost)
    if len(template_rows) != len(desired):
        return None
    assigned = np.empty(len(desired), dtype=np.int64)
    assigned[template_rows] = bank_ids[candidates[candidate_columns]]
    mapping = np.empty(len(desired), dtype=np.float64)
    mapping[template_rows] = cost[template_rows, candidate_columns]
    return assigned, mapping


def _guarded_bank(
    fit_valid_ids: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    guard_m: float,
) -> np.ndarray:
    query_tree = cKDTree(positions[query_ids, :2])
    distance = query_tree.query(positions[fit_valid_ids, :2], k=1, workers=-1)[0]
    keep = distance > max(guard_m, 1e-6)
    return fit_valid_ids[keep]


def _match_guard(
    fit_valid_ids: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    target_median: float,
    target_p90: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for guard_m in np.linspace(0.0, 12.0, 49):
        bank = _guarded_bank(fit_valid_ids, query_ids, positions, float(guard_m))
        if len(bank) < max(NEIGHBORS + 32, 128):
            continue
        nearest = cKDTree(positions[bank, :2]).query(
            positions[query_ids, :2], k=1, workers=-1
        )[0]
        median = float(np.median(nearest))
        p90 = float(np.quantile(nearest, 0.90))
        objective = abs(math.log((median + 0.25) / (target_median + 0.25)))
        objective += 0.35 * abs(math.log((p90 + 0.25) / (target_p90 + 0.25)))
        if best is None or objective < best[0]:
            best = (objective, float(guard_m), bank, nearest)
    if best is None:
        raise RuntimeError("could not construct a contiguous guarded donor bank")
    return best[1], best[2], best[3]


def _candidate_centers(
    bank_ids: np.ndarray,
    target_features: np.ndarray,
    selection_features: np.ndarray,
    positions: np.ndarray,
    seed: int,
) -> np.ndarray:
    bank_feature = selection_features[bank_ids]
    mean = np.mean(bank_feature, axis=0, dtype=np.float64)
    std = np.maximum(np.std(bank_feature, axis=0, dtype=np.float64), 1e-4)
    target_mean = np.mean(target_features, axis=0, dtype=np.float64)
    distance = np.mean(
        np.square((bank_feature - target_mean[None, :]) / std[None, :]), axis=1
    )
    ranked = bank_ids[np.argsort(distance)[: min(320, len(bank_ids))]]
    rng = np.random.default_rng(seed)
    random_count = min(96, len(bank_ids))
    random_ids = rng.choice(bank_ids, size=random_count, replace=False)
    candidates = np.unique(np.concatenate([ranked, random_ids]))
    order = np.argsort(
        np.mean(
            np.square(
                (selection_features[candidates] - target_mean[None, :]) / std[None, :]
            ),
            axis=1,
        )
    )
    selected: list[int] = []
    for point in candidates[order]:
        if not selected:
            selected.append(int(point))
            continue
        distance_xy = np.linalg.norm(
            positions[np.asarray(selected), :2] - positions[int(point), :2], axis=1
        )
        if float(np.min(distance_xy)) >= 5.0:
            selected.append(int(point))
        if len(selected) >= 180:
            break
    return np.asarray(selected, dtype=np.int64)


def _analog_candidates(
    fold: int,
    site_value: int,
    target_ids: np.ndarray,
    fit_ids: np.ndarray,
    valid: np.ndarray,
    positions: np.ndarray,
    selection_features: np.ndarray,
) -> list[AnalogHole]:
    bank_ids = fit_ids[load_positions()[3][fit_ids] == site_value]
    fit_valid_ids = bank_ids[valid[bank_ids]]
    bank_tree = cKDTree(positions[bank_ids, :2])
    target_xy = positions[target_ids, :2]
    centered = target_xy - np.mean(target_xy, axis=0, keepdims=True)
    target_feature = selection_features[target_ids]
    feature_mean, feature_std = _summary_features(selection_features[fit_ids])
    feature_std = np.maximum(feature_std, 1e-4)
    target_mean, target_std = _summary_features(target_feature)
    target_nearest = cKDTree(positions[fit_valid_ids, :2]).query(
        target_xy, k=1, workers=-1
    )[0]
    target_median = float(np.median(target_nearest))
    target_p90 = float(np.quantile(target_nearest, 0.90))
    centers = _candidate_centers(
        bank_ids,
        target_feature,
        selection_features,
        positions,
        20_260_806 + fold * 1009 + site_value * 97,
    )
    output: list[AnalogHole] = []
    for center_id in centers:
        mapped = _assigned_template(
            centered,
            positions[center_id, :2],
            bank_ids,
            positions,
            bank_tree,
        )
        if mapped is None:
            continue
        query_ids, mapping = mapped
        if len(np.unique(query_ids)) != len(target_ids):
            continue
        guard_m, _, nearest = _match_guard(
            fit_valid_ids,
            query_ids,
            positions,
            target_median,
            target_p90,
        )
        local_feature = selection_features[query_ids]
        local_mean, local_std = _summary_features(local_feature)
        mean_distance = float(
            np.sqrt(np.mean(np.square((local_mean - target_mean) / feature_std)))
        )
        std_distance = float(
            np.sqrt(np.mean(np.square((local_std - target_std) / feature_std)))
        )
        mapping_median = float(np.median(mapping))
        mapping_p90 = float(np.quantile(mapping, 0.90))
        nearest_median = float(np.median(nearest))
        nearest_p90 = float(np.quantile(nearest, 0.90))
        nearest_mismatch = abs(
            math.log((nearest_median + 0.25) / (target_median + 0.25))
        ) + 0.35 * abs(math.log((nearest_p90 + 0.25) / (target_p90 + 0.25)))
        score = (
            mean_distance
            + 0.35 * std_distance
            + 0.10 * mapping_median
            + 0.04 * mapping_p90
            + 0.45 * nearest_mismatch
        )
        output.append(
            AnalogHole(
                site=site_value,
                center_xy=np.asarray(positions[center_id, :2], dtype=np.float64),
                query_ids=query_ids,
                guard_m=guard_m,
                mapping_median_m=mapping_median,
                mapping_p90_m=mapping_p90,
                nearest_median_m=nearest_median,
                nearest_p90_m=nearest_p90,
                feature_distance=mean_distance,
                score=score,
            )
        )
    return sorted(output, key=lambda value: value.score)


def _select_diverse_holes(candidates: list[AnalogHole], count: int) -> list[AnalogHole]:
    selected: list[AnalogHole] = []
    for minimum_separation in (MIN_CENTER_SEPARATION_M, 10.0, 0.0):
        for candidate in candidates:
            if any(candidate is value for value in selected):
                continue
            if selected and minimum_separation > 0.0:
                center_distance = np.linalg.norm(
                    np.stack([value.center_xy for value in selected])
                    - candidate.center_xy[None, :],
                    axis=1,
                )
                if float(np.min(center_distance)) < minimum_separation:
                    continue
            overlap = [
                np.mean(np.isin(candidate.query_ids, value.query_ids))
                for value in selected
            ]
            if overlap and max(overlap) > 0.20:
                continue
            selected.append(candidate)
            if len(selected) >= count:
                return selected
    if len(selected) < count:
        raise RuntimeError(f"only selected {len(selected)} analog holes, need {count}")
    return selected


@torch.no_grad()
def _correct_weights(
    fold: int,
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    distances: np.ndarray,
    base_weights: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / f"fold{fold}_final.pt",
        map_location=device,
        weights_only=False,
    )
    config = V3Config(**checkpoint["config"])
    if config.neighbors != donor_ids.shape[1]:
        raise ValueError("contiguous-hole donor count does not match V3 checkpoint")
    geometry_raw = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)[:, 3:]
    geometry_numpy = (
        (geometry_raw - checkpoint["feature_mean"][None, :])
        / checkpoint["feature_std"][None, :]
    ).astype(np.float32)
    descriptors = np.load(DESCRIPTORS, allow_pickle=False)
    spectra_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    _, positions, _, _, _ = load_positions()
    geometry = torch.as_tensor(geometry_numpy, device=device)
    position = torch.as_tensor(positions[:, :2], dtype=torch.float32, device=device)
    spectra = torch.as_tensor(spectra_numpy, device=device)
    model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    pas_weights = np.empty_like(base_weights, dtype=np.float32)
    pdp_weights = np.empty_like(base_weights, dtype=np.float32)
    for start in range(0, len(query_ids), 128):
        end = min(start + 128, len(query_ids))
        query = torch.as_tensor(query_ids[start:end], dtype=torch.long, device=device)
        donor = torch.as_tensor(donor_ids[start:end], dtype=torch.long, device=device)
        _, _, diagnostic = model(
            geometry[query],
            position[query],
            geometry[donor],
            position[donor],
            spectra[donor],
            torch.as_tensor(distances[start:end], device=device),
            torch.as_tensor(base_weights[start:end], device=device),
        )
        pas_weights[start:end] = diagnostic["pas_weights"].cpu().numpy()
        pdp_weights[start:end] = diagnostic["pdp_weights"].cpu().numpy()
    return pas_weights, pdp_weights


def _hole_feature_block(
    query_ids: np.ndarray,
    positions: np.ndarray,
    geodesic: np.ndarray,
    distances: np.ndarray,
    fit_geo_mean: np.ndarray,
    fit_geo_std: np.ndarray,
) -> np.ndarray:
    xy = np.asarray(positions[query_ids, :2], dtype=np.float64)
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    vectors = vectors[:, np.argsort(values)[::-1]]
    if vectors[0, 0] < 0.0:
        vectors[:, 0] *= -1.0
    if np.linalg.det(vectors) < 0.0:
        vectors[:, 1] *= -1.0
    principal = centered @ vectors
    scale = np.maximum(np.quantile(np.abs(principal), 0.90, axis=0), 1.0)
    local = principal / scale[None, :]
    radial = np.linalg.norm(local, axis=1, keepdims=True)
    nearest = distances[:, 0]
    shape = np.asarray(
        [
            len(query_ids) / 100.0,
            scale[0] / 30.0,
            scale[1] / 30.0,
            np.median(nearest) / 10.0,
            np.quantile(nearest, 0.90) / 15.0,
            np.mean(nearest) / 10.0,
        ],
        dtype=np.float32,
    )
    selected_geo = geodesic[query_ids][:, GEO_COLUMNS]
    geo_mean = np.mean(selected_geo, axis=0, dtype=np.float64)
    geo_std = np.std(selected_geo, axis=0, dtype=np.float64)
    point_geo = (selected_geo - fit_geo_mean[None, :]) / fit_geo_std[None, :]
    mean_geo = (geo_mean - fit_geo_mean) / fit_geo_std
    std_geo = geo_std / fit_geo_std
    return np.concatenate(
        [
            local.astype(np.float32),
            radial.astype(np.float32),
            np.repeat(shape[None, :], len(query_ids), axis=0),
            point_geo.astype(np.float32),
            np.repeat(mean_geo[None, :], len(query_ids), axis=0).astype(np.float32),
            np.repeat(std_geo[None, :], len(query_ids), axis=0).astype(np.float32),
            (point_geo - mean_geo[None, :]).astype(np.float32),
        ],
        axis=1,
    )


def _build_context(
    fold: int,
    sample_limit: int,
    actual: dict[str, np.ndarray],
    device: torch.device,
    output_path: Path,
) -> dict[str, np.ndarray]:
    if output_path.is_file():
        cached = np.load(output_path, allow_pickle=False)
        if int(cached["sample_limit"]) == sample_limit:
            return {key: cached[key] for key in cached.files}

    _, positions, _, site, _ = load_positions()
    descriptors = np.load(DESCRIPTORS, allow_pickle=False)
    valid = descriptors["valid"].astype(bool)
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    geometry = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    geodesic = np.load(GEODESIC, allow_pickle=False)["train_features"].astype(np.float32)
    selection_features = np.concatenate([geometry[:, 3:], geodesic], axis=1)
    held = actual["fold_id"].astype(np.int8) == fold
    target_total = int(np.sum(held))
    repeats = max(4, int(math.ceil(sample_limit / max(target_total, 1))))

    holes: list[AnalogHole] = []
    template_sizes: dict[int, int] = {}
    for site_value in (0, 1):
        rows = np.flatnonzero(held & (actual["site"].astype(np.int8) == site_value))
        target_ids = actual["query_ids"][rows].astype(np.int64)
        template_sizes[site_value] = len(target_ids)
        candidates = _analog_candidates(
            fold,
            site_value,
            target_ids,
            fit_ids,
            valid,
            positions,
            selection_features,
        )
        selected = _select_diverse_holes(candidates, repeats)
        holes.extend(selected)
        print(
            f"E50 analog fold={fold} site={site_value} candidates={len(candidates)} "
            f"selected={len(selected)} best={selected[0].score:.4f} "
            f"worst={selected[-1].score:.4f}",
            flush=True,
        )

    query_parts: list[np.ndarray] = []
    donor_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    hole_parts: list[np.ndarray] = []
    template_parts: list[np.ndarray] = []
    quality_parts: list[np.ndarray] = []
    hole_feature_parts: list[np.ndarray] = []
    fit_selected_geo = geodesic[fit_ids][:, GEO_COLUMNS]
    fit_geo_mean = np.mean(fit_selected_geo, axis=0, dtype=np.float64)
    fit_geo_std = np.maximum(
        np.std(fit_selected_geo, axis=0, dtype=np.float64), 1e-4
    )

    hole_rows: list[dict[str, object]] = []
    for hole_id, hole in enumerate(holes):
        fit_valid_ids = fit_ids[(site[fit_ids] == hole.site) & valid[fit_ids]]
        donor_bank = _guarded_bank(
            fit_valid_ids, hole.query_ids, positions, hole.guard_m
        )
        tree = cKDTree(positions[donor_bank, :2])
        distances, local = tree.query(
            positions[hole.query_ids, :2], k=NEIGHBORS, workers=-1
        )
        donor_ids = donor_bank[np.asarray(local, dtype=np.int64)]
        base_weights = np.stack(
            [
                solve_linear_rbf_weights(
                    positions[ids, :2], positions[query_id, :2], 10.0
                )
                for query_id, ids in zip(hole.query_ids, donor_ids)
            ]
        ).astype(np.float32)
        query_parts.append(hole.query_ids)
        donor_parts.append(donor_ids.astype(np.int32))
        distance_parts.append(np.asarray(distances, dtype=np.float32))
        weight_parts.append(base_weights)
        hole_parts.append(np.full(len(hole.query_ids), hole_id, dtype=np.int16))
        template_parts.append(
            np.arange(len(hole.query_ids), dtype=np.int16)
        )
        quality_parts.append(
            np.full(len(hole.query_ids), hole.score, dtype=np.float32)
        )
        hole_feature_parts.append(
            _hole_feature_block(
                hole.query_ids,
                positions,
                geodesic,
                np.asarray(distances, dtype=np.float32),
                fit_geo_mean,
                fit_geo_std,
            )
        )
        hole_rows.append(
            {
                "hole_id": hole_id,
                "site": hole.site,
                "query_count": int(len(hole.query_ids)),
                "center_xy": hole.center_xy.tolist(),
                "guard_m": hole.guard_m,
                "mapping_median_m": hole.mapping_median_m,
                "mapping_p90_m": hole.mapping_p90_m,
                "nearest_median_m": hole.nearest_median_m,
                "nearest_p90_m": hole.nearest_p90_m,
                "feature_distance": hole.feature_distance,
                "score": hole.score,
            }
        )

    query_ids = np.concatenate(query_parts).astype(np.int64)
    donor_ids = np.concatenate(donor_parts).astype(np.int64)
    distances = np.concatenate(distance_parts).astype(np.float32)
    base_weights = np.concatenate(weight_parts).astype(np.float32)
    pas_weights, pdp_weights = _correct_weights(
        fold, query_ids, donor_ids, distances, base_weights, device
    )
    context = {
        "query_ids": query_ids.astype(np.int32),
        "site": site[query_ids].astype(np.int8),
        "donor_ids": donor_ids.astype(np.int32),
        "distances": distances,
        "base_weights": base_weights,
        "pas_weights": pas_weights,
        "pdp_weights": pdp_weights,
        "hole_id": np.concatenate(hole_parts),
        "template_index": np.concatenate(template_parts),
        "analog_quality": np.concatenate(quality_parts),
        "hole_features": np.concatenate(hole_feature_parts).astype(np.float32),
        "fit_geo_mean": fit_geo_mean.astype(np.float32),
        "fit_geo_std": fit_geo_std.astype(np.float32),
        "sample_limit": np.int32(sample_limit),
        "repeats": np.int16(repeats),
        "template_size_site0": np.int16(template_sizes[0]),
        "template_size_site1": np.int16(template_sizes[1]),
        "hole_rows_json": np.asarray(json.dumps(hole_rows, ensure_ascii=False)),
    }
    np.savez_compressed(output_path, **context)
    return context


def _base_features(
    fold: int,
    context: dict[str, np.ndarray],
    base_path: Path,
    device: torch.device,
) -> np.ndarray:
    base = _build_pseudo_base(fold, context, base_path)
    channel_parts: list[np.ndarray] = []
    for start in range(0, len(base), 8):
        end = min(start + 8, len(base))
        tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        channel_parts.append(channel_features(tensor).cpu().numpy())
    geometry = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    return np.concatenate(
        [
            geometry[context["query_ids"].astype(np.int64)],
            np.concatenate(channel_parts).astype(np.float32),
            context_features(
                context["distances"],
                context["base_weights"],
                context["pas_weights"],
                context["pdp_weights"],
            ),
        ],
        axis=1,
    ).astype(np.float32)


def _actual_hole_features(
    fold: int,
    actual: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, positions, _, site, _ = load_positions()
    geodesic = np.load(GEODESIC, allow_pickle=False)["train_features"].astype(np.float32)
    weights = np.load(V3_WEIGHTS, allow_pickle=False)
    if not np.array_equal(weights["query_ids"], actual["query_ids"]):
        raise RuntimeError("V3 OOF query order mismatch")
    donor_ids = weights["donor_ids"].astype(np.int64)
    distances = np.linalg.norm(
        positions[donor_ids, :2]
        - positions[actual["query_ids"].astype(np.int64), None, :2],
        axis=2,
    ).astype(np.float32)
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    selected_geo = geodesic[fit_ids][:, GEO_COLUMNS]
    fit_geo_mean = np.mean(selected_geo, axis=0, dtype=np.float64)
    fit_geo_std = np.maximum(np.std(selected_geo, axis=0, dtype=np.float64), 1e-4)
    held = actual["fold_id"].astype(np.int8) == fold
    held_rows = np.flatnonzero(held)
    feature = np.empty(
        (len(held_rows), 3 + 6 + 4 * len(GEO_COLUMNS)), dtype=np.float32
    )
    template_index = np.empty(len(held_rows), dtype=np.int16)
    group = np.empty(len(held_rows), dtype=np.int8)
    cursor = 0
    for site_value in (0, 1):
        rows = np.flatnonzero(held & (actual["site"].astype(np.int8) == site_value))
        query_ids = actual["query_ids"][rows].astype(np.int64)
        block = _hole_feature_block(
            query_ids,
            positions,
            geodesic,
            distances[rows],
            fit_geo_mean,
            fit_geo_std,
        )
        count = len(rows)
        feature[cursor : cursor + count] = block
        template_index[cursor : cursor + count] = np.arange(count, dtype=np.int16)
        group[cursor : cursor + count] = site_value
        cursor += count
    order = np.concatenate(
        [
            np.flatnonzero(held & (actual["site"].astype(np.int8) == site_value))
            for site_value in (0, 1)
        ]
    )
    return order, feature, template_index, group


def _retrieval_prediction(
    labels: np.ndarray,
    site: np.ndarray,
    template_index: np.ndarray,
    quality: np.ndarray,
    target_site: np.ndarray,
    target_template: np.ndarray,
    excluded_group: np.ndarray | None = None,
    group: np.ndarray | None = None,
) -> np.ndarray:
    output = np.empty(
        (len(target_site), labels.shape[1], labels.shape[2]), dtype=np.float32
    )
    for row, (site_value, template_value) in enumerate(
        zip(target_site, target_template)
    ):
        mask = (site == site_value) & (template_index == template_value)
        if excluded_group is not None and group is not None:
            mask &= group != excluded_group[row]
        bank = np.flatnonzero(mask)
        if len(bank) == 0:
            mask = site == site_value
            if excluded_group is not None and group is not None:
                mask &= group != excluded_group[row]
            bank = np.flatnonzero(mask)
        local_quality = quality[bank].astype(np.float64)
        local_quality -= np.min(local_quality)
        weight = np.exp(-local_quality / max(np.median(local_quality) + 0.25, 0.25))
        weight /= np.sum(weight)
        output[row] = np.tensordot(
            weight.astype(np.float32), labels[bank], axes=(0, 0)
        )
    return output


def _grouped_choice(
    utility: np.ndarray,
    base_local: np.ndarray,
    group: np.ndarray,
    threshold: float,
    fraction: float,
    components: np.ndarray,
    pas_floor: float,
    pdp_floor: float,
    agreement: np.ndarray | None,
) -> np.ndarray:
    output = base_local.copy()
    for value in np.unique(group):
        rows = np.flatnonzero(group == value)
        output[rows] = _choose_actions(
            utility[rows],
            base_local[rows],
            threshold,
            fraction,
            components=components[rows],
            pas_floor=pas_floor,
            pdp_floor=pdp_floor,
            agreement_utility=None if agreement is None else agreement[rows],
        )
    return output


def _cross_hole_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    leaf: int,
    seed: int,
) -> np.ndarray:
    output = np.empty_like(labels, dtype=np.float32)
    for held_group in np.unique(groups):
        train = groups != held_group
        held = groups == held_group
        model = ExtraTreesRegressor(
            n_estimators=320,
            min_samples_leaf=leaf,
            max_features=0.72,
            n_jobs=-1,
            random_state=seed + int(held_group) * 101,
        )
        model.fit(features[train], labels[train].reshape(np.sum(train), -1))
        output[held] = model.predict(features[held]).reshape(
            np.sum(held), labels.shape[1], labels.shape[2]
        )
    return output


def _select_control(
    pseudo: dict[str, np.ndarray],
    labels: np.ndarray,
    errors: np.ndarray,
    base_local: np.ndarray,
    groups: np.ndarray,
    model_predictions: dict[int, np.ndarray],
    retrieval: np.ndarray,
    baseline_nmse: float,
) -> tuple[dict[str, float | int | bool], dict[str, object]]:
    baseline = _score_selection(
        pseudo, errors, base_local, np.ones(len(base_local), dtype=bool)
    )
    best: tuple[float, dict[str, float | int | bool], dict[str, object]] | None = None
    for leaf, model_prediction in model_predictions.items():
        for model_weight in (0.25, 0.50, 0.75, 1.00):
            prediction = model_weight * model_prediction + (1.0 - model_weight) * retrieval
            utility = _predicted_utility(prediction, baseline_nmse)
            model_utility = _predicted_utility(model_prediction, baseline_nmse)
            retrieval_utility = _predicted_utility(retrieval, baseline_nmse)
            for threshold in (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005):
                for fraction in (0.10, 0.20, 0.35, 0.50, 0.75):
                    for pas_floor in (-0.002, 0.0, 0.001):
                        for pdp_floor in (-0.002, 0.0):
                            for require_agreement in (False, True):
                                agreement = None
                                if require_agreement:
                                    agreement = np.minimum(model_utility, retrieval_utility)
                                selected = _grouped_choice(
                                    utility,
                                    base_local,
                                    groups,
                                    threshold,
                                    fraction,
                                    prediction,
                                    pas_floor,
                                    pdp_floor,
                                    agreement,
                                )
                                score = _score_selection(
                                    pseudo,
                                    errors,
                                    selected,
                                    np.ones(len(selected), dtype=bool),
                                )
                                hole_delta = []
                                for group_value in np.unique(groups):
                                    mask = groups == group_value
                                    base_score = _score_selection(
                                        pseudo, errors, base_local, mask
                                    )
                                    new_score = _score_selection(
                                        pseudo, errors, selected, mask
                                    )
                                    hole_delta.append(
                                        float(new_score["score"] - base_score["score"])
                                    )
                                hole_delta_array = np.asarray(hole_delta, dtype=np.float64)
                                mean_delta = float(score["score"] - baseline["score"])
                                positive = int(np.sum(hole_delta_array > -1e-6))
                                robust = mean_delta - 0.20 * float(np.std(hole_delta_array))
                                if positive < math.ceil(0.70 * len(hole_delta_array)):
                                    robust -= 0.02
                                control = {
                                    "leaf": leaf,
                                    "model_weight": model_weight,
                                    "threshold": threshold,
                                    "fraction": fraction,
                                    "pas_floor": pas_floor,
                                    "pdp_floor": pdp_floor,
                                    "require_agreement": require_agreement,
                                }
                                diagnostic = {
                                    "score": score,
                                    "delta": mean_delta,
                                    "robust": robust,
                                    "positive_holes": positive,
                                    "hole_count": int(len(hole_delta_array)),
                                    "hole_delta": hole_delta,
                                    "active": int(np.sum(selected != base_local)),
                                }
                                if best is None or robust > best[0]:
                                    best = (robust, control, diagnostic)
    if best is None:
        raise RuntimeError("no contiguous-hole control selected")
    return best[1], best[2]


def evaluate_fold(
    fold: int,
    candidate_indices: np.ndarray,
    pseudo: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    pseudo_base_features: np.ndarray,
    actual: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, object]:
    fold_id = actual["fold_id"].astype(np.int8)
    outer_train = fold_id != fold
    held = fold_id == fold
    scales = _site_scales(
        {**actual, "candidate_names": np.asarray(candidate_indices)},
        np.arange(len(candidate_indices)),
        outer_train,
    )
    pseudo_labels, pseudo_errors, pseudo_base_local, _ = _labels(
        pseudo, candidate_indices, scales
    )
    actual_labels, actual_errors, actual_base_local, _ = _labels(
        actual, candidate_indices, scales
    )
    pseudo_observable = np.stack(
        [pseudo["pas_target"], pseudo["pdp_target"]], axis=2
    )
    pseudo_x = _point_features(
        pseudo_base_features,
        context["query_ids"].astype(np.int64),
        pseudo_observable,
        pseudo["site"].astype(np.int8),
        pseudo_base_local,
    )
    pseudo_x = np.concatenate(
        [pseudo_x, context["hole_features"].astype(np.float32)], axis=1
    )
    oof_features_path = (
        PATHS.artifacts / "round2" / "augmented_projection_policy" / "oof_point_features.npy"
    )
    if not oof_features_path.is_file():
        raise FileNotFoundError(oof_features_path)
    oof_base_features = np.load(oof_features_path).astype(np.float32)
    actual_observable = np.stack(
        [actual["pas_target"], actual["pdp_target"]], axis=2
    )
    actual_x = _point_features(
        oof_base_features,
        actual["query_ids"].astype(np.int64),
        actual_observable,
        actual["site"].astype(np.int8),
        actual_base_local,
    )
    held_order, held_hole_features, held_template, held_group = _actual_hole_features(
        fold, actual
    )
    if not np.array_equal(np.sort(held_order), np.flatnonzero(held)):
        raise RuntimeError("held contiguous-hole rows do not cover the outer fold")
    held_x = np.concatenate(
        [actual_x[held_order], held_hole_features], axis=1
    )

    groups = context["hole_id"].astype(np.int16)
    retrieval_oof = _retrieval_prediction(
        pseudo_labels,
        context["site"].astype(np.int8),
        context["template_index"].astype(np.int16),
        context["analog_quality"].astype(np.float32),
        context["site"].astype(np.int8),
        context["template_index"].astype(np.int16),
        excluded_group=groups,
        group=groups,
    )
    model_predictions = {
        leaf: _cross_hole_predictions(
            pseudo_x,
            pseudo_labels,
            groups,
            leaf,
            20_270_806 + fold * 1009 + leaf,
        )
        for leaf in (4, 8, 16)
    }
    outer_baseline = _score_selection(
        actual, actual_errors, actual_base_local, outer_train
    )
    control, pseudo_cv = _select_control(
        pseudo,
        pseudo_labels,
        pseudo_errors,
        pseudo_base_local,
        groups,
        model_predictions,
        retrieval_oof,
        float(outer_baseline["nmse"]),
    )

    final_prediction_parts = []
    for seed_offset in (0, 1):
        model = ExtraTreesRegressor(
            n_estimators=640,
            min_samples_leaf=int(control["leaf"]),
            max_features=0.72,
            n_jobs=-1,
            random_state=20_280_806 + fold * 1009 + seed_offset * 7919,
        )
        model.fit(pseudo_x, pseudo_labels.reshape(len(pseudo_x), -1))
        final_prediction_parts.append(
            model.predict(held_x).reshape(
                len(held_x), len(candidate_indices), 3
            )
        )
    model_prediction = np.mean(final_prediction_parts, axis=0).astype(np.float32)
    retrieval_prediction = _retrieval_prediction(
        pseudo_labels,
        context["site"].astype(np.int8),
        context["template_index"].astype(np.int16),
        context["analog_quality"].astype(np.float32),
        held_group,
        held_template,
    )
    prediction = float(control["model_weight"]) * model_prediction + (
        1.0 - float(control["model_weight"])
    ) * retrieval_prediction
    utility = _predicted_utility(prediction, float(outer_baseline["nmse"]))
    agreement = None
    if bool(control["require_agreement"]):
        agreement = np.minimum(
            _predicted_utility(model_prediction, float(outer_baseline["nmse"])),
            _predicted_utility(retrieval_prediction, float(outer_baseline["nmse"])),
        )
    held_base_local = actual_base_local[held_order]
    held_selected = _grouped_choice(
        utility,
        held_base_local,
        held_group,
        float(control["threshold"]),
        float(control["fraction"]),
        prediction,
        float(control["pas_floor"]),
        float(control["pdp_floor"]),
        agreement,
    )
    selected = actual_base_local.copy()
    selected[held_order] = held_selected
    held_baseline = _score_selection(actual, actual_errors, actual_base_local, held)
    held_score = _score_selection(actual, actual_errors, selected, held)
    oracle_selected = actual_base_local.copy()
    truth_utility = _predicted_utility(
        actual_labels[held_order], float(outer_baseline["nmse"])
    )
    oracle_selected[held_order] = np.argmax(truth_utility, axis=1)
    oracle = _score_selection(actual, actual_errors, oracle_selected, held)
    return {
        "fold": fold,
        "candidate_count": int(len(candidate_indices)),
        "pseudo_count": int(len(pseudo_x)),
        "pseudo_hole_count": int(len(np.unique(groups))),
        "feature_count": int(pseudo_x.shape[1]),
        "selected_control": control,
        "pseudo_cross_hole": pseudo_cv,
        "held_baseline": held_baseline,
        "held": held_score,
        "held_delta": float(held_score["score"] - held_baseline["score"]),
        "held_active": int(np.sum(held_selected != held_base_local)),
        "held_oracle": oracle,
        "held_oracle_delta": float(oracle["score"] - held_baseline["score"]),
        "selected_local": held_selected.tolist(),
        "held_order": held_order.tolist(),
    }


def run_fold(
    fold: int,
    device: torch.device,
    batch_size: int,
    candidate_count: int,
    sample_limit: int,
) -> dict[str, object]:
    started = time.time()
    paths = _fold_paths(fold)
    loaded = np.load(V4_METRICS, allow_pickle=False)
    full_actual = {key: loaded[key] for key in loaded.files}
    fit_mask = full_actual["fold_id"].astype(np.int8) != fold
    candidate_indices = greedy_candidates(full_actual, fit_mask, candidate_count)
    paths["selection"].write_text(
        json.dumps(
            {
                "fold": fold,
                "candidate_indices": candidate_indices.tolist(),
                "candidate_names": full_actual["candidate_names"][candidate_indices].tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    context = _build_context(fold, sample_limit, full_actual, device, paths["context"])
    pseudo_base_features = _base_features(fold, context, paths["base"], device)
    models = _fit_group_calibration(fold, device, paths["calibration"])
    _build_pseudo_targets(fold, context, models, paths)
    pseudo = _generate_pseudo_metrics(
        fold, candidate_indices, context, paths, device, batch_size
    )
    actual = _actual_subset(full_actual, candidate_indices)
    actual["candidate_names"] = full_actual["candidate_names"][candidate_indices]
    result = evaluate_fold(
        fold,
        candidate_indices,
        pseudo,
        context,
        pseudo_base_features,
        actual,
        device,
    )
    result["candidate_indices"] = candidate_indices.tolist()
    result["candidate_names"] = full_actual["candidate_names"][candidate_indices].tolist()
    result["context"] = PATHS.relative(paths["context"])
    result["metrics"] = PATHS.relative(paths["metrics"])
    result["seconds"] = float(time.time() - started)
    return result


def _aggregate(records: list[dict[str, object]]) -> dict[str, object] | None:
    if len(records) != 5:
        return None
    count = np.asarray(
        [int(record["held_baseline"]["count"]) for record in records], dtype=np.float64
    )
    delta = np.asarray([float(record["held_delta"]) for record in records])
    return {
        "weighted_delta": float(np.sum(count * delta) / np.sum(count)),
        "positive_folds": int(np.sum(delta > 0.0)),
        "fold_delta": delta.tolist(),
        "active": int(np.sum([int(record["held_active"]) for record in records])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "evaluate"), default="pilot")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--sample-limit", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    folds = (
        [args.fold]
        if args.phase == "pilot"
        else [int(value) for value in args.folds.split(",") if value.strip()]
    )
    records = [
        run_fold(
            fold,
            device,
            args.batch_size,
            args.candidate_count,
            args.sample_limit,
        )
        for fold in folds
    ]
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "Projection utility becomes transferable when every pseudo label is rebuilt "
            "from a complete translated target-shaped hole with a shared observed boundary, "
            "and the policy retains each point's relative position inside that hole."
        ),
        "parameters": {
            "folds": folds,
            "candidate_count": args.candidate_count,
            "sample_limit": args.sample_limit,
            "neighbors": NEIGHBORS,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "records": records,
        "aggregate": _aggregate(records),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
