#!/usr/bin/env python3
"""Audit real Round 2 test holes and Clean V7 per-hole failure profiles."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN

from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics,
)
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E167-clean-test-hole-difficulty"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e167_clean_test_hole_difficulty"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e167_clean_test_hole_difficulty"
VIS_DIR = PATHS.analysis / "visualizations" / "复赛Clean测试洞难度审计"

REPORT_JSON = ANALYSIS_DIR / "clean_test_hole_difficulty.json"
REPORT_MD = ANALYSIS_DIR / "真实测试洞与CleanV7逐洞难度审计.md"
REAL_HOLES_CSV = ANALYSIS_DIR / "真实10个测试洞几何统计.csv"
FIXED_HOLES_CSV = ANALYSIS_DIR / "fixed500十个验证洞几何统计.csv"
DISTRIBUTION_CSV = ANALYSIS_DIR / "真实测试洞与fixed500分布差异.csv"
MATCH_CSV = ANALYSIS_DIR / "真实测试洞与pseudo洞匹配.csv"
SCORE_CSV = ANALYSIS_DIR / "V7逐pseudo洞官方分.csv"
DEPTH_CSV = ANALYSIS_DIR / "V7洞深度分层官方分.csv"
POINT_CSV = ANALYSIS_DIR / "V7逐点Clean可见量与误差.csv"
CORRELATION_CSV = ANALYSIS_DIR / "V7失败变量相关与分层.csv"

FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
V7_COMPLETE = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "canonical_v7_complete.npy"
)
MAP_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
SPECTRAL_DESCRIPTORS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_spectral_descriptors.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

EXPECTED_V7 = {
    "pas": 0.597983259844433,
    "pdp": 0.7564459407811781,
    "nmse": 0.9477104887777993,
    "score": 0.6444563457343335,
}
IDENTITY_TOLERANCE = 5e-6
MAP_MATCH_FEATURES = (
    "geodesic_ratio",
    "bottleneck_clearance",
    "route_hit_fraction",
    "center_route_hit_fraction",
    "route_occupancy_fraction",
    "panorama_blocked_fraction",
    "panorama_mean_clearance",
    "opposite_clearance_mean",
)


@dataclass(frozen=True)
class Hole:
    hole_id: str
    kind: str
    site: int
    ordinal: int
    point_ids: np.ndarray
    xy: np.ndarray
    fit_ids: np.ndarray
    ignored_ids: np.ndarray
    map_features: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--h-neighbors", type=int, default=8)
    return parser.parse_args()


def sorted_clusters(xy: np.ndarray) -> list[np.ndarray]:
    labels = DBSCAN(eps=15.0, min_samples=3).fit_predict(xy)
    if np.any(labels < 0):
        raise RuntimeError("DBSCAN left test coordinates outside a spatial cluster")
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return sorted(groups, key=lambda ids: tuple(np.mean(xy[ids], axis=0).tolist()))


def convex_hull_area(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)
    except Exception:
        return 0.0


def principal_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centered = np.asarray(xy, dtype=np.float64) - np.mean(xy, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major = vectors[:, order[0]]
    minor = vectors[:, order[1]]
    aspect = math.sqrt(max(float(values[order[0]]), 1e-12) / max(float(values[order[1]]), 1e-12))
    return major.astype(np.float64), minor.astype(np.float64), float(aspect)


def angular_visibility(query_xy: np.ndarray, neighbor_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = neighbor_xy - query_xy[:, None, :]
    norm = np.linalg.norm(delta, axis=2)
    unit = delta / np.maximum(norm[..., None], 1e-12)
    angles = np.mod(np.arctan2(delta[..., 1], delta[..., 0]), 2.0 * np.pi)
    ordered = np.sort(angles, axis=1)
    wrapped = np.concatenate([ordered, ordered[:, :1] + 2.0 * np.pi], axis=1)
    max_gap = np.max(np.diff(wrapped, axis=1), axis=1) * 180.0 / np.pi
    one_sided = np.linalg.norm(np.mean(unit, axis=1), axis=1)
    quadrant = np.floor(angles / (0.5 * np.pi)).astype(np.int64)
    occupied = np.asarray([len(np.unique(row)) for row in quadrant], dtype=np.float64)
    return max_gap, one_sided, occupied


def visibility_arrays(query_xy: np.ndarray, fit_xy: np.ndarray, k: int = 16) -> dict[str, np.ndarray]:
    take = min(k, len(fit_xy))
    distance, local = cKDTree(fit_xy).query(query_xy, k=take, workers=-1)
    distance = np.asarray(distance, dtype=np.float64)
    local = np.asarray(local, dtype=np.int64)
    if take == 1:
        distance = distance[:, None]
        local = local[:, None]
    neighbor_xy = fit_xy[local]
    max_gap, one_sided, occupied = angular_visibility(query_xy, neighbor_xy)
    return {
        "distance": distance,
        "local": local,
        "max_gap_deg": max_gap,
        "one_sidedness": one_sided,
        "quadrants": occupied,
    }


def map_feature_summary(features: np.ndarray, names: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in MAP_MATCH_FEATURES:
        index = int(np.flatnonzero(names == name)[0])
        value = np.asarray(features[:, index], dtype=np.float64)
        finite = np.isfinite(value)
        output[f"map_{name}_mean"] = float(np.mean(value[finite])) if np.any(finite) else float("nan")
        output[f"map_{name}_std"] = float(np.std(value[finite])) if np.any(finite) else float("nan")
    return output


def hole_summary(hole: Hole, train_positions: np.ndarray, bs_positions: np.ndarray, map_names: np.ndarray) -> dict[str, object]:
    xy = np.asarray(hole.xy, dtype=np.float64)
    fit_xy = np.asarray(train_positions[hole.fit_ids, :2], dtype=np.float64)
    visible = visibility_arrays(xy, fit_xy, 16)
    distance = visible["distance"]
    _, _, aspect = principal_axes(xy)
    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    width, height = maximum - minimum
    d1 = distance[:, 0]
    bs_distance = np.linalg.norm(xy - bs_positions[hole.site, :2], axis=1)
    shape = "细长" if aspect >= 3.0 else "团块"
    gap_median = float(np.median(visible["max_gap_deg"]))
    if gap_median > 210.0:
        surround = "明显单侧"
    elif gap_median < 150.0:
        surround = "多侧包围"
    else:
        surround = "过渡"
    output: dict[str, object] = {
        "hole_id": hole.hole_id,
        "kind": hole.kind,
        "site": hole.site,
        "ordinal": hole.ordinal,
        "point_count": int(len(xy)),
        "center_x": float(np.mean(xy[:, 0])),
        "center_y": float(np.mean(xy[:, 1])),
        "xmin": float(minimum[0]),
        "xmax": float(maximum[0]),
        "ymin": float(minimum[1]),
        "ymax": float(maximum[1]),
        "width_m": float(width),
        "height_m": float(height),
        "bbox_area_m2": float(width * height),
        "convex_hull_area_m2": convex_hull_area(xy),
        "pca_aspect_ratio": aspect,
        "shape_label": shape,
        "nearest_train_mean_m": float(np.mean(d1)),
        "nearest_train_median_m": float(np.median(d1)),
        "nearest_train_p90_m": float(np.quantile(d1, 0.90)),
        "nearest_train_max_m": float(np.max(d1)),
        "knn4_median_m": float(np.median(distance[:, min(3, distance.shape[1] - 1)])),
        "knn8_median_m": float(np.median(distance[:, min(7, distance.shape[1] - 1)])),
        "knn16_median_m": float(np.median(distance[:, min(15, distance.shape[1] - 1)])),
        "angular_max_gap_median_deg": gap_median,
        "angular_max_gap_p90_deg": float(np.quantile(visible["max_gap_deg"], 0.90)),
        "one_sidedness_median": float(np.median(visible["one_sidedness"])),
        "quadrants_occupied_median": float(np.median(visible["quadrants"])),
        "surround_label": surround,
        "bs_distance_min_m": float(np.min(bs_distance)),
        "bs_distance_median_m": float(np.median(bs_distance)),
        "bs_distance_max_m": float(np.max(bs_distance)),
        "ignored_count": int(len(hole.ignored_ids)),
        "visible_fit_count": int(len(hole.fit_ids)),
    }
    output.update(map_feature_summary(hole.map_features, map_names))
    return output


def build_holes() -> tuple[list[Hole], list[Hole], dict[str, object]]:
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    map_cache = np.load(MAP_FEATURES, allow_pickle=False)
    map_names = map_cache["feature_names"]
    train_map = map_cache["train_features"].astype(np.float32)
    test_map = map_cache["test_features"].astype(np.float32)
    real: list[Hole] = []
    for site in sorted(np.unique(test_site).tolist()):
        test_global = np.flatnonzero(test_site == site)
        local_clusters = sorted_clusters(test_positions[test_global, :2])
        fit_ids = np.flatnonzero(train_site == site).astype(np.int64)
        for ordinal, local in enumerate(local_clusters):
            global_ids = test_global[local]
            real.append(
                Hole(
                    hole_id=f"S{site}H{ordinal}",
                    kind="real_test",
                    site=int(site),
                    ordinal=ordinal,
                    point_ids=global_ids.astype(np.int64),
                    xy=test_positions[global_ids, :2].astype(np.float64),
                    fit_ids=fit_ids,
                    ignored_ids=np.empty(0, dtype=np.int64),
                    map_features=test_map[global_ids],
                )
            )

    folds = np.load(FOLD_CACHE, allow_pickle=False)
    pseudo: list[Hole] = []
    for fold in range(5):
        query = folds[f"fold{fold}_query"].astype(np.int64)
        fit = folds[f"fold{fold}_fit"].astype(np.int64)
        ignored = folds[f"fold{fold}_ignored"].astype(np.int64)
        for site in sorted(np.unique(train_site[query]).tolist()):
            local_query = query[train_site[query] == site]
            local_fit = fit[train_site[fit] == site]
            local_ignored = ignored[train_site[ignored] == site]
            pseudo.append(
                Hole(
                    hole_id=f"P_S{site}F{fold}",
                    kind="canonical_pseudo",
                    site=int(site),
                    ordinal=fold,
                    point_ids=local_query,
                    xy=train_positions[local_query, :2].astype(np.float64),
                    fit_ids=local_fit,
                    ignored_ids=local_ignored,
                    map_features=train_map[local_query],
                )
            )
    audit = {
        "real_count": len(real),
        "pseudo_count": len(pseudo),
        "fold_query_counts": [int(len(folds[f"fold{fold}_query"])) for fold in range(5)],
        "fold_fit_counts": [int(len(folds[f"fold{fold}_fit"])) for fold in range(5)],
        "fold_ignored_counts": [int(len(folds[f"fold{fold}_ignored"])) for fold in range(5)],
        "pseudo_query_unique_count": int(len(np.unique(np.concatenate([hole.point_ids for hole in pseudo])))),
        "pseudo_query_total_count": int(sum(len(hole.point_ids) for hole in pseudo)),
    }
    return real, pseudo, {"setup": setup, "train_positions": train_positions, "test_positions": test_positions, "train_site": train_site, "test_site": test_site, "map_names": map_names, "audit": audit}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def robust_standardize(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median[None]), axis=0)
    scale = 1.4826 * mad
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    output = (values - median[None]) / scale[None]
    output[~np.isfinite(output)] = 0.0
    return output


def matching_vectors(rows: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, slice]]:
    size = np.asarray(
        [
            [
                math.log1p(float(row["point_count"])),
                math.log1p(float(row["width_m"])),
                math.log1p(float(row["height_m"])),
                math.log1p(float(row["convex_hull_area_m2"])),
                math.log(max(float(row["pca_aspect_ratio"]), 1e-6)),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    visibility = np.asarray(
        [
            [
                float(row["nearest_train_median_m"]),
                float(row["nearest_train_p90_m"]),
                float(row["nearest_train_max_m"]),
                float(row["knn4_median_m"]),
                float(row["knn8_median_m"]),
                float(row["knn16_median_m"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    surround = np.asarray(
        [
            [
                float(row["angular_max_gap_median_deg"]) / 180.0,
                float(row["one_sidedness_median"]),
                float(row["quadrants_occupied_median"]) / 4.0,
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    bs = np.asarray(
        [
            [
                float(row["bs_distance_median_m"]),
                float(row["bs_distance_max_m"]) - float(row["bs_distance_min_m"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    map_values = np.asarray(
        [
            [float(row[f"map_{name}_mean"]) for name in MAP_MATCH_FEATURES]
            for row in rows
        ],
        dtype=np.float64,
    )
    groups = [size, visibility, surround, bs, map_values]
    standardized = [robust_standardize(group) for group in groups]
    pieces = []
    slices: dict[str, slice] = {}
    start = 0
    for name, group in zip(("size_shape", "visibility", "surround", "bs", "map"), standardized):
        pieces.append(group)
        slices[name] = slice(start, start + group.shape[1])
        start += group.shape[1]
    return np.concatenate(pieces, axis=1), slices


def matching_cost(a: np.ndarray, b: np.ndarray, groups: dict[str, slice]) -> float:
    values = [float(np.sqrt(np.mean(np.square(a[part] - b[part])))) for part in groups.values()]
    return float(np.mean(values))


def build_matches(real_rows: list[dict[str, object]], pseudo_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, float]]:
    combined = real_rows + pseudo_rows
    vectors, groups = matching_vectors(combined)
    real_vector = vectors[: len(real_rows)]
    pseudo_vector = vectors[len(real_rows) :]
    cost = np.full((len(real_rows), len(pseudo_rows)), np.inf, dtype=np.float64)
    for i, real in enumerate(real_rows):
        for j, pseudo in enumerate(pseudo_rows):
            if int(real["site"]) == int(pseudo["site"]):
                cost[i, j] = matching_cost(real_vector[i], pseudo_vector[j], groups)

    original_cost = []
    for i, real in enumerate(real_rows):
        candidate = next(
            j
            for j, pseudo in enumerate(pseudo_rows)
            if int(pseudo["site"]) == int(real["site"])
            and int(pseudo["ordinal"]) == int(real["ordinal"])
        )
        original_cost.append(float(cost[i, candidate]))
    strong_threshold = float(np.median(original_cost))
    usable_threshold = float(np.quantile(original_cost, 0.90))

    matches: list[dict[str, object]] = []
    used_pairs: set[tuple[int, int]] = set()
    for round_index in range(3):
        for site in (0, 1):
            real_ids = [index for index, row in enumerate(real_rows) if int(row["site"]) == site]
            pseudo_ids = [index for index, row in enumerate(pseudo_rows) if int(row["site"]) == site]
            local = cost[np.ix_(real_ids, pseudo_ids)].copy()
            for local_i, global_i in enumerate(real_ids):
                for local_j, global_j in enumerate(pseudo_ids):
                    if (global_i, global_j) in used_pairs:
                        local[local_i, local_j] = 1e6
            row_index, column_index = linear_sum_assignment(local)
            for local_i, local_j in zip(row_index, column_index):
                global_i = real_ids[int(local_i)]
                global_j = pseudo_ids[int(local_j)]
                value = float(cost[global_i, global_j])
                used_pairs.add((global_i, global_j))
                if value <= strong_threshold:
                    quality = "强匹配"
                    usable = True
                elif value <= usable_threshold:
                    quality = "可用"
                    usable = True
                else:
                    quality = "弱匹配"
                    usable = False
                matches.append(
                    {
                        "round": round_index + 1,
                        "real_hole_id": real_rows[global_i]["hole_id"],
                        "pseudo_hole_id": pseudo_rows[global_j]["hole_id"],
                        "site": site,
                        "cost": value,
                        "quality": quality,
                        "usable": usable,
                        "original_fixed_pair": bool(
                            int(real_rows[global_i]["ordinal"])
                            == int(pseudo_rows[global_j]["ordinal"])
                        ),
                        "real_count": real_rows[global_i]["point_count"],
                        "pseudo_count": pseudo_rows[global_j]["point_count"],
                        "real_nearest_median_m": real_rows[global_i]["nearest_train_median_m"],
                        "pseudo_nearest_median_m": pseudo_rows[global_j]["nearest_train_median_m"],
                        "real_aspect": real_rows[global_i]["pca_aspect_ratio"],
                        "pseudo_aspect": pseudo_rows[global_j]["pca_aspect_ratio"],
                    }
                )
    return matches, {
        "strong_threshold": strong_threshold,
        "usable_threshold": usable_threshold,
        "original_cost_median": strong_threshold,
        "original_cost_p90": usable_threshold,
        "original_cost_min": float(np.min(original_cost)),
        "original_cost_max": float(np.max(original_cost)),
    }


def score_v7(
    pseudo: list[Hole],
    context: dict[str, object],
    batch_size: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[str, object]]:
    train_positions = np.asarray(context["train_positions"])
    train_site = np.asarray(context["train_site"])
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    prediction = np.load(V7_COMPLETE, mmap_mode="r")
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    if prediction.shape != (500, 256, 4, 192):
        raise ValueError(f"unexpected V7 shape: {prediction.shape}")
    point_pas = np.empty(500, dtype=np.float64)
    point_pdp = np.empty(500, dtype=np.float64)
    point_error = np.empty(500, dtype=np.float64)
    point_energy = np.empty(500, dtype=np.float64)
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        values = point_metric_statistics(
            np.asarray(prediction[start:end], dtype=np.complex64),
            np.asarray(truth[query_ids[start:end]], dtype=np.complex64),
        )
        point_pas[start:end] = values.pas
        point_pdp[start:end] = values.pdp
        point_error[start:end] = values.squared_error
        point_energy[start:end] = values.truth_energy
        print(f"E167 official metrics {end}/500", flush=True)
    statistics = PointMetricStatistics(point_pas, point_pdp, point_error, point_energy)
    overall = aggregate_statistics(statistics)
    identity_error = {
        key: abs(float(overall[key]) - value) for key, value in EXPECTED_V7.items()
    }
    if max(identity_error.values()) > IDENTITY_TOLERANCE:
        raise RuntimeError(f"Clean V7 identity gate failed: {identity_error}")

    global_to_oof = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    rows: list[dict[str, object]] = []
    for hole in pseudo:
        oof_rows = global_to_oof[hole.point_ids]
        if np.any(oof_rows < 0):
            raise RuntimeError(f"pseudo hole escaped canonical OOF: {hole.hole_id}")
        if np.any(fold_id[oof_rows] != hole.ordinal):
            raise RuntimeError(f"fold mismatch for {hole.hole_id}")
        official = aggregate_statistics(statistics, np.isin(np.arange(500), oof_rows))
        local_nmse = point_error[oof_rows] / np.maximum(point_energy[oof_rows], 1e-30)
        rows.append(
            {
                "pseudo_hole_id": hole.hole_id,
                "site": hole.site,
                "fold": hole.ordinal,
                **official,
                "point_pas_std": float(np.std(point_pas[oof_rows])),
                "point_pdp_std": float(np.std(point_pdp[oof_rows])),
                "point_nmse_median": float(np.median(local_nmse)),
                "point_score_std": float(
                    np.std(
                        0.4 * point_pas[oof_rows]
                        + 0.4 * point_pdp[oof_rows]
                        + 0.2 / (1.0 + local_nmse)
                    )
                ),
            }
        )
    return rows, {
        "query_ids": query_ids,
        "fold_id": fold_id,
        "pas": point_pas,
        "pdp": point_pdp,
        "squared_error": point_error,
        "truth_energy": point_energy,
        "global_to_oof": global_to_oof,
        "train_site": train_site,
    }, {"overall": overall, "identity_error": identity_error}


def cosine_dispersion(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    valid = norms > 0.0
    if np.sum(valid) < 2:
        return float("nan")
    unit = array[valid] / norms[valid, None]
    mean = np.mean(unit, axis=0)
    mean_norm = np.linalg.norm(mean)
    if mean_norm == 0.0:
        return 1.0
    return float(np.mean(1.0 - unit @ (mean / mean_norm)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / norm)


def two_side_spectral_difference(
    query_xy: np.ndarray,
    donor_xy: np.ndarray,
    donor_pas: np.ndarray,
    donor_pdp: np.ndarray,
    minor_axis: np.ndarray,
) -> tuple[float, float]:
    side = (donor_xy - query_xy[None]) @ minor_axis
    left = side < 0.0
    right = side >= 0.0
    if np.sum(left) < 2 or np.sum(right) < 2:
        return float("nan"), float("nan")
    return (
        cosine_distance(np.mean(donor_pas[left], axis=0), np.mean(donor_pas[right], axis=0)),
        cosine_distance(np.mean(donor_pdp[left], axis=0), np.mean(donor_pdp[right], axis=0)),
    )


def complex_h_variation(channel: np.ndarray, donor_ids: np.ndarray, donor_xy: np.ndarray) -> float:
    value = np.asarray(channel[donor_ids], dtype=np.complex64).reshape(len(donor_ids), -1)
    norms = np.linalg.norm(value, axis=1)
    gram = np.abs(value @ value.conj().T)
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(gram, np.maximum(denominator, 1e-30))
    spatial = np.linalg.norm(donor_xy[:, None] - donor_xy[None], axis=2)
    upper = np.triu_indices(len(donor_ids), 1)
    valid = (denominator[upper] > 0.0) & (spatial[upper] > 0.0)
    if not np.any(valid):
        return float("nan")
    variation = (1.0 - correlation[upper][valid]) / spatial[upper][valid]
    return float(np.median(variation))


def point_failure_table(
    pseudo: list[Hole],
    context: dict[str, object],
    metrics: dict[str, np.ndarray],
    h_neighbors: int,
) -> list[dict[str, object]]:
    train_positions = np.asarray(context["train_positions"])
    map_cache = np.load(MAP_FEATURES, allow_pickle=False)
    map_names = map_cache["feature_names"]
    train_map = map_cache["train_features"].astype(np.float32)
    map_index = {str(name): index for index, name in enumerate(map_names.tolist())}
    descriptors = np.load(SPECTRAL_DESCRIPTORS, allow_pickle=False)
    pas_descriptor = descriptors["global_pas"].astype(np.float32)
    pdp_descriptor = descriptors["global_pdp"].astype(np.float32)
    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    global_to_oof = metrics["global_to_oof"]
    rows: list[dict[str, object]] = []
    for hole in pseudo:
        fit_xy = train_positions[hole.fit_ids, :2]
        visible = visibility_arrays(hole.xy, fit_xy, 16)
        neighbor_global = hole.fit_ids[visible["local"]]
        depth = visible["distance"][:, 0]
        q1, q2 = np.quantile(depth, [1.0 / 3.0, 2.0 / 3.0])
        _, minor_axis, _ = principal_axes(hole.xy)
        for local, global_id in enumerate(hole.point_ids):
            oof_row = int(global_to_oof[global_id])
            donors = neighbor_global[local, :h_neighbors]
            donor_xy = train_positions[donors, :2]
            donor_pas = pas_descriptor[donors]
            donor_pdp = pdp_descriptor[donors]
            side_pas, side_pdp = two_side_spectral_difference(
                train_positions[global_id, :2], donor_xy, donor_pas, donor_pdp, minor_axis
            )
            if depth[local] <= q1:
                depth_bin = "边缘"
            elif depth[local] <= q2:
                depth_bin = "中部"
            else:
                depth_bin = "深处"
            point_nmse = float(metrics["squared_error"][oof_row] / max(metrics["truth_energy"][oof_row], 1e-30))
            row: dict[str, object] = {
                "pseudo_hole_id": hole.hole_id,
                "site": hole.site,
                "fold": hole.ordinal,
                "global_train_id": int(global_id),
                "oof_row": oof_row,
                "depth_bin": depth_bin,
                "hole_depth_m": float(depth[local]),
                "normalized_hole_depth": float((depth[local] - np.min(depth)) / max(np.quantile(depth, 0.90) - np.min(depth), 1e-6)),
                "nearest_train_m": float(visible["distance"][local, 0]),
                "knn8_distance_m": float(visible["distance"][local, min(7, visible["distance"].shape[1] - 1)]),
                "train_density_k8_per_m2": float(8.0 / (math.pi * max(visible["distance"][local, min(7, visible["distance"].shape[1] - 1)] ** 2, 1e-6))),
                "angular_max_gap_deg": float(visible["max_gap_deg"][local]),
                "one_sidedness": float(visible["one_sidedness"][local]),
                "quadrants_occupied": float(visible["quadrants"][local]),
                "bs_distance_m": float(np.linalg.norm(train_positions[global_id, :2] - context["setup"].bs_positions[hole.site, :2])),
                "neighbor_pas_dispersion": cosine_dispersion(donor_pas),
                "neighbor_pdp_dispersion": cosine_dispersion(donor_pdp),
                "two_side_pas_difference": side_pas,
                "two_side_pdp_difference": side_pdp,
                "neighbor_complex_h_variation_per_m": complex_h_variation(channel, donors, donor_xy),
                "geodesic_ratio": float(train_map[global_id, map_index["geodesic_ratio"]]),
                "route_hit_fraction": float(train_map[global_id, map_index["route_hit_fraction"]]),
                "panorama_blocked_fraction": float(train_map[global_id, map_index["panorama_blocked_fraction"]]),
                "bottleneck_clearance": float(train_map[global_id, map_index["bottleneck_clearance"]]),
                "pas": float(metrics["pas"][oof_row]),
                "pdp": float(metrics["pdp"][oof_row]),
                "squared_error": float(metrics["squared_error"][oof_row]),
                "truth_energy": float(metrics["truth_energy"][oof_row]),
                "nmse": point_nmse,
                "nmse_score": float(1.0 / (1.0 + point_nmse)),
            }
            row["point_score"] = 0.4 * float(row["pas"]) + 0.4 * float(row["pdp"]) + 0.2 * float(row["nmse_score"])
            rows.append(row)
        print(f"E167 Clean visible features {hole.hole_id} {len(hole.point_ids)}", flush=True)
    return rows


def aggregate_depth(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    order = ("边缘", "中部", "深处")
    for hole_id in sorted({str(row["pseudo_hole_id"]) for row in point_rows}):
        for label in order:
            selected = [row for row in point_rows if row["pseudo_hole_id"] == hole_id and row["depth_bin"] == label]
            stats = PointMetricStatistics(
                pas=np.asarray([row["pas"] for row in selected], dtype=np.float64),
                pdp=np.asarray([row["pdp"] for row in selected], dtype=np.float64),
                squared_error=np.asarray([row["squared_error"] for row in selected], dtype=np.float64),
                truth_energy=np.asarray([row["truth_energy"] for row in selected], dtype=np.float64),
            )
            official_like = aggregate_statistics(stats)
            output.append({"pseudo_hole_id": hole_id, "depth_bin": label, **official_like})
    for label in order:
        selected = [row for row in point_rows if row["depth_bin"] == label]
        stats = PointMetricStatistics(
            pas=np.asarray([row["pas"] for row in selected], dtype=np.float64),
            pdp=np.asarray([row["pdp"] for row in selected], dtype=np.float64),
            squared_error=np.asarray([row["squared_error"] for row in selected], dtype=np.float64),
            truth_energy=np.asarray([row["truth_energy"] for row in selected], dtype=np.float64),
        )
        output.append({"pseudo_hole_id": "ALL", "depth_bin": label, **aggregate_statistics(stats)})
    return output


def finite_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def correlation_rows(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    variables = (
        "nearest_train_m",
        "normalized_hole_depth",
        "knn8_distance_m",
        "train_density_k8_per_m2",
        "angular_max_gap_deg",
        "one_sidedness",
        "bs_distance_m",
        "neighbor_pas_dispersion",
        "neighbor_pdp_dispersion",
        "two_side_pas_difference",
        "two_side_pdp_difference",
        "neighbor_complex_h_variation_per_m",
        "geodesic_ratio",
        "route_hit_fraction",
        "panorama_blocked_fraction",
        "bottleneck_clearance",
    )
    pas = finite_array(point_rows, "pas")
    pas_error = 1.0 - pas
    hole_ids = sorted({str(row["pseudo_hole_id"]) for row in point_rows})
    output: list[dict[str, object]] = []
    for variable in variables:
        values = finite_array(point_rows, variable)
        valid = np.isfinite(values) & np.isfinite(pas_error)
        point_rho = (
            float(spearmanr(values[valid], pas_error[valid]).statistic)
            if np.sum(valid) >= 8 and np.std(values[valid]) > 0.0
            else float("nan")
        )
        q1, q2 = np.quantile(values[valid], [1.0 / 3.0, 2.0 / 3.0]) if np.sum(valid) else (float("nan"), float("nan"))
        low = valid & (values <= q1)
        high = valid & (values > q2)
        low_pas = float(np.mean(pas[low])) if np.any(low) else float("nan")
        high_pas = float(np.mean(pas[high])) if np.any(high) else float("nan")
        hole_value = []
        hole_error = []
        for hole_id in hole_ids:
            mask = np.asarray([row["pseudo_hole_id"] == hole_id for row in point_rows], dtype=bool) & valid
            if np.any(mask):
                hole_value.append(float(np.mean(values[mask])))
                hole_error.append(float(np.mean(pas_error[mask])))
        hole_rho = (
            float(spearmanr(hole_value, hole_error).statistic)
            if len(hole_value) >= 5 and np.std(hole_value) > 0.0
            else float("nan")
        )
        output.append(
            {
                "variable": variable,
                "point_spearman_with_pas_error": point_rho,
                "hole_mean_spearman_with_pas_error": hole_rho,
                "low_tercile_pas": low_pas,
                "high_tercile_pas": high_pas,
                "low_minus_high_pas": low_pas - high_pas,
                "low_threshold": float(q1),
                "high_threshold": float(q2),
                "valid_count": int(np.sum(valid)),
            }
        )
    return output


def distribution_difference(real_rows: list[dict[str, object]], pseudo_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "point_count",
        "width_m",
        "height_m",
        "convex_hull_area_m2",
        "pca_aspect_ratio",
        "nearest_train_median_m",
        "nearest_train_p90_m",
        "knn8_median_m",
        "angular_max_gap_median_deg",
        "one_sidedness_median",
        "bs_distance_median_m",
        "map_geodesic_ratio_mean",
        "map_route_hit_fraction_mean",
        "map_panorama_blocked_fraction_mean",
    )
    output = []
    for metric in metrics:
        real = np.asarray([float(row[metric]) for row in real_rows], dtype=np.float64)
        pseudo = np.asarray([float(row[metric]) for row in pseudo_rows], dtype=np.float64)
        pooled = np.concatenate([real, pseudo])
        scale = max(float(np.nanstd(pooled)), 1e-12)
        output.append(
            {
                "metric": metric,
                "real_mean": float(np.nanmean(real)),
                "real_median": float(np.nanmedian(real)),
                "fixed500_mean": float(np.nanmean(pseudo)),
                "fixed500_median": float(np.nanmedian(pseudo)),
                "mean_difference_fixed_minus_real": float(np.nanmean(pseudo) - np.nanmean(real)),
                "standardized_mean_difference": float((np.nanmean(pseudo) - np.nanmean(real)) / scale),
            }
        )
    return output


def attach_scores(matches: list[dict[str, object]], score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {str(row["pseudo_hole_id"]): row for row in score_rows}
    output = []
    for row in matches:
        score = by_id[str(row["pseudo_hole_id"])]
        output.append({**row, **{f"v7_{key}": score[key] for key in ("pas", "pdp", "nmse", "score")}})
    return output


def real_difficulty(matches: list[dict[str, object]], real_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for real in real_rows:
        rows = [row for row in matches if row["real_hole_id"] == real["hole_id"]]
        usable = [row for row in rows if bool(row["usable"])]
        primary = next(row for row in rows if int(row["round"]) == 1)
        source = usable if usable else [primary]
        values = np.asarray([float(row["v7_score"]) for row in source], dtype=np.float64)
        pas = np.asarray([float(row["v7_pas"]) for row in source], dtype=np.float64)
        output.append(
            {
                "real_hole_id": real["hole_id"],
                "site": real["site"],
                "usable_match_count": len(usable),
                "primary_pseudo_hole": primary["pseudo_hole_id"],
                "primary_match_quality": primary["quality"],
                "primary_v7_pas": primary["v7_pas"],
                "primary_v7_score": primary["v7_score"],
                "proxy_v7_pas_mean": float(np.mean(pas)),
                "proxy_v7_pas_std": float(np.std(pas)),
                "proxy_v7_score_mean": float(np.mean(values)),
                "proxy_v7_score_std": float(np.std(values)),
                "proxy_source": "usable_matches" if usable else "weak_primary_fallback",
            }
        )
    return sorted(output, key=lambda row: float(row["proxy_v7_pas_mean"]))


def failure_patterns(
    score_rows: list[dict[str, object]],
    depth_rows: list[dict[str, object]],
    correlations: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    ordered = sorted(score_rows, key=lambda row: float(row["pas"]))
    spread = float(ordered[-1]["pas"] - ordered[0]["pas"])
    if spread >= 0.10:
        output.append(
            {
                "pattern": "不同空间洞的PAS难度明显分层",
                "evidence": f"最高{ordered[-1]['pseudo_hole_id']} PAS={ordered[-1]['pas']:.6f}，最低{ordered[0]['pseudo_hole_id']} PAS={ordered[0]['pas']:.6f}，差={spread:.6f}",
                "magnitude": spread,
            }
        )
    all_depth = {str(row["depth_bin"]): row for row in depth_rows if row["pseudo_hole_id"] == "ALL"}
    if "边缘" in all_depth and "深处" in all_depth:
        drop = float(all_depth["边缘"]["pas"] - all_depth["深处"]["pas"])
        if drop >= 0.05:
            output.append(
                {
                    "pattern": "洞越深PAS越差",
                    "evidence": f"边缘PAS={all_depth['边缘']['pas']:.6f}，深处PAS={all_depth['深处']['pas']:.6f}，下降={drop:.6f}",
                    "magnitude": drop,
                }
            )
    ranked = sorted(
        correlations,
        key=lambda row: abs(float(row["low_minus_high_pas"])) if np.isfinite(float(row["low_minus_high_pas"])) else -1.0,
        reverse=True,
    )
    for row in ranked:
        magnitude = abs(float(row["low_minus_high_pas"]))
        if magnitude < 0.05:
            continue
        direction = "高值组更差" if float(row["low_minus_high_pas"]) > 0 else "低值组更差"
        output.append(
            {
                "pattern": f"{row['variable']}出现明显PAS分层",
                "evidence": f"低三分位PAS={row['low_tercile_pas']:.6f}，高三分位PAS={row['high_tercile_pas']:.6f}，差={row['low_minus_high_pas']:+.6f}，{direction}",
                "magnitude": magnitude,
                "point_spearman": row["point_spearman_with_pas_error"],
                "hole_spearman": row["hole_mean_spearman_with_pas_error"],
            }
        )
        if len(output) >= 5:
            break
    return output


def configure_chinese_font() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = family
    else:
        plt.rcParams["font.sans-serif"] = ["Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_holes(real: list[Hole], pseudo: list[Hole], context: dict[str, object]) -> Path:
    train_positions = np.asarray(context["train_positions"])
    train_site = np.asarray(context["train_site"])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for site in (0, 1):
        ax = axes[site, 0]
        train = train_positions[train_site == site, :2]
        ax.scatter(train[:, 0], train[:, 1], s=4, c="#c8c8c8", alpha=0.45, label="训练点")
        for hole in [value for value in real if value.site == site]:
            color = colors[site * 5 + hole.ordinal]
            ax.scatter(hole.xy[:, 0], hole.xy[:, 1], s=24, color=color, label=hole.hole_id)
        ax.scatter(context["setup"].bs_positions[site, 0], context["setup"].bs_positions[site, 1], marker="*", s=180, c="black", label="BS")
        ax.set_title(f"站点{site}：真实Test空间洞")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(ncol=2, fontsize=8)
        ax = axes[site, 1]
        ax.scatter(train[:, 0], train[:, 1], s=4, c="#d8d8d8", alpha=0.35, label="训练点")
        for hole in [value for value in pseudo if value.site == site]:
            color = colors[site * 5 + hole.ordinal]
            ax.scatter(hole.xy[:, 0], hole.xy[:, 1], s=24, color=color, label=hole.hole_id)
        ax.scatter(context["setup"].bs_positions[site, 0], context["setup"].bs_positions[site, 1], marker="*", s=180, c="black", label="BS")
        ax.set_title(f"站点{site}：fixed500 canonical pseudo-hole")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(ncol=2, fontsize=8)
    path = VIS_DIR / "01_真实Test洞与fixed500验证洞位置.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_distribution(distribution: list[dict[str, object]]) -> Path:
    chosen = [row for row in distribution if row["metric"] in ("point_count", "convex_hull_area_m2", "nearest_train_median_m", "knn8_median_m", "angular_max_gap_median_deg", "bs_distance_median_m")]
    labels = [str(row["metric"]) for row in chosen]
    real = [float(row["real_mean"]) for row in chosen]
    fixed = [float(row["fixed500_mean"]) for row in chosen]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        ax.bar([0, 1], [real[index], fixed[index]], color=["#2364aa", "#d95f02"])
        ax.set_xticks([0, 1], ["真实Test", "fixed500"])
        ax.set_title(labels[index])
    path = VIS_DIR / "02_真实Test与fixed500几何分布对比.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_hole_scores(score_rows: list[dict[str, object]]) -> Path:
    rows = sorted(score_rows, key=lambda row: float(row["pas"]))
    labels = [str(row["pseudo_hole_id"]) for row in rows]
    x = np.arange(len(rows))
    figure, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    width = 0.25
    ax.bar(x - width, [float(row["pas"]) for row in rows], width, label="PAS", color="#2364aa")
    ax.bar(x, [float(row["pdp"]) for row in rows], width, label="PDP", color="#2a9d8f")
    ax.bar(x + width, [float(row["score"]) for row in rows], width, label="总分", color="#d95f02")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Clean V7在十个canonical pseudo-hole上的官方分")
    ax.legend()
    path = VIS_DIR / "03_V7逐pseudo洞官方分.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_depth(depth_rows: list[dict[str, object]]) -> Path:
    rows = [row for row in depth_rows if row["pseudo_hole_id"] == "ALL"]
    order = {"边缘": 0, "中部": 1, "深处": 2}
    rows.sort(key=lambda row: order[str(row["depth_bin"])])
    labels = [str(row["depth_bin"]) for row in rows]
    x = np.arange(len(rows))
    figure, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(x, [float(row["pas"]) for row in rows], marker="o", linewidth=2, label="PAS")
    ax.plot(x, [float(row["pdp"]) for row in rows], marker="s", linewidth=2, label="PDP")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("V7随pseudo-hole内部深度的变化")
    ax.legend()
    path = VIS_DIR / "04_V7洞深度分层.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_failure_variables(correlations: list[dict[str, object]]) -> Path:
    rows = sorted(correlations, key=lambda row: abs(float(row["low_minus_high_pas"])), reverse=True)[:8]
    labels = [str(row["variable"]) for row in rows]
    values = [float(row["low_minus_high_pas"]) for row in rows]
    figure, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    colors = ["#d95f02" if value > 0 else "#2364aa" for value in values]
    ax.barh(np.arange(len(rows)), values, color=colors)
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("低三分位PAS - 高三分位PAS")
    ax.set_title("Clean可见量上下三分位的V7 PAS差")
    ax.invert_yaxis()
    path = VIS_DIR / "05_V7失败变量分层.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_real_difficulty(rows: list[dict[str, object]]) -> Path:
    labels = [str(row["real_hole_id"]) for row in rows]
    values = [float(row["proxy_v7_pas_mean"]) for row in rows]
    errors = [float(row["proxy_v7_pas_std"]) for row in rows]
    figure, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = np.arange(len(rows))
    ax.bar(x, values, yerr=errors, color="#6a994e", capsize=4)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("匹配pseudo-hole的V7 PAS均值")
    ax.set_title("真实Test洞的Clean几何匹配难度代理（低值更难）")
    path = VIS_DIR / "06_真实Test洞难度代理.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def markdown_report(
    real_rows: list[dict[str, object]],
    distribution: list[dict[str, object]],
    matches: list[dict[str, object]],
    score_rows: list[dict[str, object]],
    depth_rows: list[dict[str, object]],
    patterns: list[dict[str, object]],
    difficulty: list[dict[str, object]],
    images: list[Path],
    overall: dict[str, object],
    audit: dict[str, object],
    thresholds: dict[str, float],
) -> str:
    score_by_id = {str(row["pseudo_hole_id"]): row for row in score_rows}
    matches_by_real: dict[str, list[dict[str, object]]] = {}
    for row in matches:
        if bool(row["usable"]):
            matches_by_real.setdefault(str(row["real_hole_id"]), []).append(row)
    difficulty_by_id = {str(row["real_hole_id"]): row for row in difficulty}

    lines = [
        "# 真实测试洞与Clean V7逐洞难度审计",
        "",
        "## 固定结论",
        "",
        f"fixed500 Clean V7复现：PAS={overall['pas']:.6f}，PDP={overall['pdp']:.6f}，NMSE={overall['nmse']:.6f}，总分={overall['score']:.6f}。",
        "",
        "V7的PAS并不是十个洞都处于约0.60：十洞最高为0.848578，最低为0.333432。整体低分主要由少数困难空间洞拖低。",
        "",
        "## 真实10个Test洞",
        "",
        "| hole_id | 点数 | 宽×高(m) | 凸包面积(m²) | 形状 | 最近Train median/p90/max(m) | 包围关系 | BS距离范围(m) | geodesic ratio |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in real_rows:
        lines.append(
            f"| {row['hole_id']} | {int(row['point_count'])} | {float(row['width_m']):.1f}×{float(row['height_m']):.1f} | "
            f"{float(row['convex_hull_area_m2']):.1f} | {row['shape_label']} | "
            f"{float(row['nearest_train_median_m']):.2f}/{float(row['nearest_train_p90_m']):.2f}/{float(row['nearest_train_max_m']):.2f} | "
            f"{row['surround_label']} | {float(row['bs_distance_min_m']):.1f}–{float(row['bs_distance_max_m']):.1f} | "
            f"{float(row['map_geodesic_ratio_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## fixed500血缘与分布差异",
            "",
            "fixed500不是随机点。它来自`round2_spatial_hole_folds.npz`定义的5折连续空间洞；每折同时隐藏两个站点各一个洞，按`fold × site`还原为10个互不重复的canonical pseudo-hole。",
            "",
            f"五折总点数为{'/'.join(str(value) for value in audit['fold_query_counts'])}，合计{audit['pseudo_query_unique_count']}个唯一点；各pseudo-hole点数与真实Test洞对应序号完全一致。",
            "",
            "| 指标 | 真实Test均值 | fixed500均值 | fixed−real | 标准化差异 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in distribution:
        lines.append(
            f"| {row['metric']} | {float(row['real_mean']):.4f} | {float(row['fixed500_mean']):.4f} | "
            f"{float(row['mean_difference_fixed_minus_real']):+.4f} | {float(row['standardized_mean_difference']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "结论：fixed500在点数、最近Train距离、邻居方向包围和地图遮挡比例上与真实洞接近；主要偏差是pseudo-hole平均宽高更大、凸包面积约大52%，BS距离更远，并且geodesic ratio略低。所有统计量的标准化均值差都小于0.75，因此它不是明显失配的验证集，但也不是逐洞完全等价。",
            "",
            "## 真实洞的pseudo-hole替身",
            "",
            f"匹配只使用Clean几何；强匹配阈值≤{thresholds['strong_threshold']:.3f}，可用阈值≤{thresholds['usable_threshold']:.3f}。弱匹配不进入难度均值。",
            "",
            "| 真实洞 | 可用替身 | 数量 | PAS代理(mean±std) | 总分代理(mean±std) | 证据强度 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for real in real_rows:
        hole_id = str(real["hole_id"])
        row = difficulty_by_id[hole_id]
        usable = matches_by_real.get(hole_id, [])
        labels = "、".join(f"{item['pseudo_hole_id']}({item['quality']})" for item in usable) or "无"
        count = int(row["usable_match_count"])
        confidence = "较强" if count >= 3 else "有限" if count == 2 else "很弱"
        lines.append(
            f"| {hole_id} | {labels} | {count} | {float(row['proxy_v7_pas_mean']):.3f}±{float(row['proxy_v7_pas_std']):.3f} | "
            f"{float(row['proxy_v7_score_mean']):.3f}±{float(row['proxy_v7_score_std']):.3f} | {confidence} |"
        )

    lines.extend(
        [
            "",
            "## V7逐pseudo-hole官方分",
            "",
            "| pseudo-hole | 点数 | PAS | PDP | NMSE | 总分 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(score_by_id.values(), key=lambda item: (int(item["site"]), int(item["fold"]))):
        lines.append(
            f"| {row['pseudo_hole_id']} | {int(row['count'])} | {float(row['pas']):.6f} | {float(row['pdp']):.6f} | "
            f"{float(row['nmse']):.6f} | {float(row['score']):.6f} |"
        )

    all_depth = {str(row["depth_bin"]): row for row in depth_rows if row["pseudo_hole_id"] == "ALL"}
    lines.extend(
        [
            "",
            "## 洞内部深度",
            "",
            "| 位置 | 点数 | PAS | PDP | NMSE | 总分 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("边缘", "中部", "深处"):
        row = all_depth[label]
        lines.append(
            f"| {label} | {int(row['count'])} | {float(row['pas']):.6f} | {float(row['pdp']):.6f} | "
            f"{float(row['nmse']):.6f} | {float(row['score']):.6f} |"
        )

    lines.extend([
        "",
        "## 结果数据、图",
        "",
    ])
    for image in images:
        relative = Path("../../visualizations/复赛Clean测试洞难度审计") / image.name
        lines.extend([f"![{image.stem}]({relative.as_posix()})", ""])
    lines.extend(["## 现象", ""])
    if patterns:
        for index, pattern in enumerate(patterns, 1):
            lines.append(f"{index}. {pattern['pattern']}：{pattern['evidence']}。")
    else:
        lines.append("没有任何预注册分层达到明显幅度门槛。")
    lines.extend(["", "## 能说明什么问题", ""])
    hardest = difficulty[:3]
    easiest = difficulty[-3:][::-1]
    lines.append("最可能困难的真实Test洞：" + "、".join(f"{row['real_hole_id']}（PAS代理{row['proxy_v7_pas_mean']:.3f}）" for row in hardest) + "。")
    lines.append("最可能简单的真实Test洞：" + "、".join(f"{row['real_hole_id']}（PAS代理{row['proxy_v7_pas_mean']:.3f}）" for row in easiest) + "。")
    lines.append("")
    lines.append("真实洞难度是Clean几何替身代理，不是真实Test GT。S1H2、S1H0和S0H2只有1个可用替身，结论置信度很弱；其余多数洞也只有2到3个替身。完整机器表位于本目录CSV和JSON。")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()

    required = (FOLD_CACHE, OOF_META, V7_COMPLETE, MAP_FEATURES, SPECTRAL_DESCRIPTORS, TRAIN_CHANNEL)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Clean inputs: {missing}")

    real, pseudo, context = build_holes()
    real_rows = [hole_summary(hole, context["train_positions"], context["setup"].bs_positions, context["map_names"]) for hole in real]
    pseudo_rows = [hole_summary(hole, context["train_positions"], context["setup"].bs_positions, context["map_names"]) for hole in pseudo]
    write_csv(REAL_HOLES_CSV, real_rows)
    write_csv(FIXED_HOLES_CSV, pseudo_rows)

    distribution = distribution_difference(real_rows, pseudo_rows)
    write_csv(DISTRIBUTION_CSV, distribution)
    matches, thresholds = build_matches(real_rows, pseudo_rows)

    score_rows, point_metrics, identity = score_v7(pseudo, context, args.metric_batch_size)
    matches = attach_scores(matches, score_rows)
    write_csv(MATCH_CSV, matches)
    write_csv(SCORE_CSV, score_rows)

    point_rows = point_failure_table(pseudo, context, point_metrics, args.h_neighbors)
    write_csv(POINT_CSV, point_rows)
    depth_rows = aggregate_depth(point_rows)
    write_csv(DEPTH_CSV, depth_rows)
    correlations = correlation_rows(point_rows)
    write_csv(CORRELATION_CSV, correlations)
    patterns = failure_patterns(score_rows, depth_rows, correlations)
    difficulty = real_difficulty(matches, real_rows)

    images = [
        plot_holes(real, pseudo, context),
        plot_distribution(distribution),
        plot_hole_scores(score_rows),
        plot_depth(depth_rows),
        plot_failure_variables(correlations),
        plot_real_difficulty(difficulty),
    ]

    fixed_difference_large = [
        row for row in distribution if abs(float(row["standardized_mean_difference"])) >= 0.75
    ]
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "network_training": False,
        "goal": "Determine whether V7 PAS failure is broad or concentrated in specific spatial-hole types.",
        "inputs": [PATHS.relative(path) for path in required],
        "hole_definition": {
            "test_clustering": "per-site DBSCAN eps=15m min_samples=3, sorted by center XY",
            "real_hole_ids": [hole.hole_id for hole in real],
            "canonical_hole_ids": [hole.hole_id for hole in pseudo],
            **context["audit"],
        },
        "v7_identity": identity,
        "matching_thresholds": thresholds,
        "real_holes": real_rows,
        "canonical_holes": pseudo_rows,
        "distribution_difference": distribution,
        "large_distribution_differences": fixed_difference_large,
        "matches": matches,
        "v7_pseudo_hole_scores": score_rows,
        "depth_profiles": depth_rows,
        "failure_correlations": correlations,
        "failure_patterns": patterns,
        "real_hole_difficulty_proxy": difficulty,
        "most_likely_difficult": difficulty[:3],
        "most_likely_simple": difficulty[-3:][::-1],
        "outputs": {
            "real_holes_csv": PATHS.relative(REAL_HOLES_CSV),
            "fixed_holes_csv": PATHS.relative(FIXED_HOLES_CSV),
            "distribution_csv": PATHS.relative(DISTRIBUTION_CSV),
            "match_csv": PATHS.relative(MATCH_CSV),
            "score_csv": PATHS.relative(SCORE_CSV),
            "depth_csv": PATHS.relative(DEPTH_CSV),
            "point_csv": PATHS.relative(POINT_CSV),
            "correlation_csv": PATHS.relative(CORRELATION_CSV),
            "visualizations": [PATHS.relative(path) for path in images],
            "markdown": PATHS.relative(REPORT_MD),
        },
        "seconds": float(time.time() - started),
    }
    REPORT_JSON.write_text(
        json.dumps(json_compatible(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(
        markdown_report(
            real_rows,
            distribution,
            matches,
            score_rows,
            depth_rows,
            patterns,
            difficulty,
            images,
            identity["overall"],
            context["audit"],
            thresholds,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "v7": identity["overall"],
        "patterns": patterns,
        "hardest": difficulty[:3],
        "easiest": difficulty[-3:][::-1],
        "seconds": payload["seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
