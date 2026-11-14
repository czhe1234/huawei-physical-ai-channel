#!/usr/bin/env python3
"""Build Clean pseudo-hole families specialized to the 10 real Test holes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

import analyze_e167_clean_test_hole_difficulty as e167
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E183-test-hole-specialization-library"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e183_test_hole_specialization"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e183_test_hole_specialization"

LIBRARY_NPZ = ARTIFACT_DIR / "matched_pseudo_hole_library.npz"
REPORT_JSON = ANALYSIS_DIR / "matching_library_gate.json"
SUMMARY_CSV = ANALYSIS_DIR / "真实Test洞替身库门控汇总.csv"
INVENTORY_CSV = ANALYSIS_DIR / "逐替身洞匹配清单.csv"
WITHIN_OVERLAP_CSV = ANALYSIS_DIR / "同目标洞替身重叠诊断.csv"
CROSS_OVERLAP_CSV = ANALYSIS_DIR / "跨目标洞替身重叠诊断.csv"

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
CANONICAL_MATCH_CSV = (
    PATHS.analysis
    / "results"
    / "round2_e167_clean_test_hole_difficulty"
    / "真实测试洞与pseudo洞匹配.csv"
)

SCALES = (0.85, 1.00, 1.15)
ROTATIONS_DEG = (-15.0, 0.0, 15.0)
MIN_COUNT_FRACTION = 0.65
MAX_COUNT_FRACTION = 1.35
MIN_POINT_COUNT = 12
MAX_SELECTED = 30
MIN_PASS_COUNT = 20
MIN_ABSOLUTE_COUNT = 12
MAX_PAIR_JACCARD = 0.35
MAX_P90_JACCARD = 0.25
PRESCREEN_LIMIT = 3000
EXTERNAL_POINT_K = 16
EXTERNAL_CENTER_K = 32
SEARCH_K = 192
TINY = 1e-30

MAP_FIELDS = e167.MAP_MATCH_FEATURES

GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "size_shape": (
        "log_point_count",
        "log_width_m",
        "log_height_m",
        "log_convex_hull_area_m2",
        "log_pca_aspect_ratio",
    ),
    "visibility": (
        "nearest_external_mean_m",
        "nearest_external_median_m",
        "nearest_external_p90_m",
        "nearest_external_max_m",
        "knn4_external_median_m",
        "knn8_external_median_m",
        "knn16_external_median_m",
    ),
    "surround": (
        "angular_max_gap_median_turn",
        "angular_max_gap_p90_turn",
        "one_sidedness_median",
        "quadrants_occupied_fraction",
    ),
    "bs": (
        "bs_distance_median_m",
        "bs_distance_range_m",
    ),
    "map": tuple(
        f"map_{name}_{stat}" for name in MAP_FIELDS for stat in ("mean", "std")
    ),
    "external_spectrum": (
        "external_pas_dispersion",
        "external_pdp_dispersion",
        "two_side_pas_difference",
        "two_side_pdp_difference",
        "external_center_distance_mean_m",
        "external_center_distance_max_m",
    ),
}
FEATURE_NAMES = tuple(name for fields in GROUP_FIELDS.values() for name in fields)


@dataclass(frozen=True)
class CandidateOccurrence:
    point_ids: tuple[int, ...]
    center_global_id: int
    center_x: float
    center_y: float
    scale: float
    rotation_deg: float
    quick_cost: float


@dataclass(frozen=True)
class SearchContext:
    site: int
    global_ids: np.ndarray
    xy: np.ndarray
    tree: cKDTree
    local_by_global: np.ndarray
    neighbor_global: np.ndarray
    neighbor_distance: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-centers", type=int, default=None)
    parser.add_argument("--prescreen-limit", type=int, default=PRESCREEN_LIMIT)
    parser.add_argument("--max-selected", type=int, default=MAX_SELECTED)
    return parser.parse_args()


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
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def normalized_rows(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = array.sum(axis=1, keepdims=True)
    output = np.zeros_like(array)
    np.divide(array, np.maximum(total, TINY), out=output, where=total > 0.0)
    return output


def build_search_contexts(
    train_positions: np.ndarray, train_site: np.ndarray
) -> dict[int, SearchContext]:
    contexts: dict[int, SearchContext] = {}
    for site in sorted(np.unique(train_site).tolist()):
        global_ids = np.flatnonzero(train_site == site).astype(np.int64)
        xy = np.asarray(train_positions[global_ids, :2], dtype=np.float64)
        tree = cKDTree(xy)
        take = min(SEARCH_K, len(global_ids))
        distance, local = tree.query(xy, k=take, workers=-1)
        local_by_global = np.full(len(train_positions), -1, dtype=np.int64)
        local_by_global[global_ids] = np.arange(len(global_ids), dtype=np.int64)
        contexts[int(site)] = SearchContext(
            site=int(site),
            global_ids=global_ids,
            xy=xy,
            tree=tree,
            local_by_global=local_by_global,
            neighbor_global=global_ids[np.asarray(local, dtype=np.int64)],
            neighbor_distance=np.asarray(distance, dtype=np.float64),
        )
    return contexts


def first_external_neighbors(
    neighbor_global: np.ndarray,
    neighbor_distance: np.ndarray,
    excluded_mask: np.ndarray,
    take: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = ~excluded_mask[neighbor_global]
    order = np.cumsum(valid, axis=1)
    selected = valid & (order <= take)
    counts = np.sum(selected, axis=1)
    if np.any(counts < take):
        raise RuntimeError(
            f"SEARCH_K={neighbor_global.shape[1]} did not expose {take} external neighbors"
        )
    return (
        neighbor_global[selected].reshape(len(neighbor_global), take),
        neighbor_distance[selected].reshape(len(neighbor_global), take),
    )


def external_point_neighbors(
    xy: np.ndarray,
    point_ids: np.ndarray | None,
    excluded_ids: np.ndarray,
    context: SearchContext,
    take: int,
) -> tuple[np.ndarray, np.ndarray]:
    excluded_mask = np.zeros(len(context.local_by_global), dtype=bool)
    excluded_mask[excluded_ids] = True
    if point_ids is None:
        search_take = min(SEARCH_K, len(context.global_ids))
        distance, local = context.tree.query(xy, k=search_take, workers=-1)
        neighbor_global = context.global_ids[np.asarray(local, dtype=np.int64)]
        return first_external_neighbors(
            neighbor_global, np.asarray(distance, dtype=np.float64), excluded_mask, take
        )
    local_rows = context.local_by_global[point_ids]
    if np.any(local_rows < 0):
        raise RuntimeError("candidate point belongs to the wrong site")
    return first_external_neighbors(
        context.neighbor_global[local_rows],
        context.neighbor_distance[local_rows],
        excluded_mask,
        take,
    )


def external_center_neighbors(
    center: np.ndarray,
    excluded_ids: np.ndarray,
    context: SearchContext,
    take: int,
) -> tuple[np.ndarray, np.ndarray]:
    search_take = min(SEARCH_K, len(context.global_ids))
    distance, local = context.tree.query(center, k=search_take, workers=-1)
    neighbor_global = context.global_ids[np.atleast_1d(local).astype(np.int64)][None]
    neighbor_distance = np.atleast_1d(distance).astype(np.float64)[None]
    excluded_mask = np.zeros(len(context.local_by_global), dtype=bool)
    excluded_mask[excluded_ids] = True
    ids, distances = first_external_neighbors(
        neighbor_global, neighbor_distance, excluded_mask, take
    )
    return ids[0], distances[0]


def finite_mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def build_profile(
    *,
    xy: np.ndarray,
    point_ids: np.ndarray | None,
    excluded_ids: np.ndarray,
    site: int,
    map_features: np.ndarray,
    setup: object,
    map_names: np.ndarray,
    train_positions: np.ndarray,
    train_pas: np.ndarray,
    train_pdp: np.ndarray,
    search: SearchContext,
) -> tuple[dict[str, float], dict[str, int]]:
    xy = np.asarray(xy, dtype=np.float64)
    excluded_ids = np.unique(np.asarray(excluded_ids, dtype=np.int64))
    neighbor_ids, neighbor_distance = external_point_neighbors(
        xy, point_ids, excluded_ids, search, EXTERNAL_POINT_K
    )
    if np.intersect1d(neighbor_ids, excluded_ids).size:
        raise AssertionError("candidate-hole IDs leaked into point-neighbor features")
    neighbor_xy = train_positions[neighbor_ids, :2]
    max_gap, one_sided, occupied = e167.angular_visibility(xy, neighbor_xy)

    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    width, height = maximum - minimum
    _, minor_axis, aspect = e167.principal_axes(xy)
    center = np.mean(xy, axis=0)
    bs_distance = np.linalg.norm(xy - setup.bs_positions[site, :2], axis=1)

    donor_ids, donor_distance = external_center_neighbors(
        center, excluded_ids, search, EXTERNAL_CENTER_K
    )
    if np.intersect1d(donor_ids, excluded_ids).size:
        raise AssertionError("candidate-hole IDs leaked into spectral matching features")
    donor_xy = train_positions[donor_ids, :2]
    donor_pas = train_pas[donor_ids]
    donor_pdp = train_pdp[donor_ids]
    side_pas, side_pdp = e167.two_side_spectral_difference(
        center, donor_xy, donor_pas, donor_pdp, minor_axis
    )

    output: dict[str, float] = {
        "log_point_count": math.log1p(len(xy)),
        "log_width_m": math.log1p(float(width)),
        "log_height_m": math.log1p(float(height)),
        "log_convex_hull_area_m2": math.log1p(e167.convex_hull_area(xy)),
        "log_pca_aspect_ratio": math.log(max(aspect, 1e-6)),
        "nearest_external_mean_m": float(np.mean(neighbor_distance[:, 0])),
        "nearest_external_median_m": float(np.median(neighbor_distance[:, 0])),
        "nearest_external_p90_m": float(np.quantile(neighbor_distance[:, 0], 0.90)),
        "nearest_external_max_m": float(np.max(neighbor_distance[:, 0])),
        "knn4_external_median_m": float(np.median(neighbor_distance[:, 3])),
        "knn8_external_median_m": float(np.median(neighbor_distance[:, 7])),
        "knn16_external_median_m": float(np.median(neighbor_distance[:, 15])),
        "angular_max_gap_median_turn": float(np.median(max_gap) / 180.0),
        "angular_max_gap_p90_turn": float(np.quantile(max_gap, 0.90) / 180.0),
        "one_sidedness_median": float(np.median(one_sided)),
        "quadrants_occupied_fraction": float(np.median(occupied) / 4.0),
        "bs_distance_median_m": float(np.median(bs_distance)),
        "bs_distance_range_m": float(np.ptp(bs_distance)),
        "external_pas_dispersion": e167.cosine_dispersion(donor_pas),
        "external_pdp_dispersion": e167.cosine_dispersion(donor_pdp),
        "two_side_pas_difference": side_pas,
        "two_side_pdp_difference": side_pdp,
        "external_center_distance_mean_m": float(np.mean(donor_distance)),
        "external_center_distance_max_m": float(np.max(donor_distance)),
    }
    map_name_to_index = {
        str(name): index for index, name in enumerate(map_names.tolist())
    }
    for name in MAP_FIELDS:
        mean, std = finite_mean_std(map_features[:, map_name_to_index[name]])
        output[f"map_{name}_mean"] = mean
        output[f"map_{name}_std"] = std
    missing = [name for name in FEATURE_NAMES if name not in output]
    if missing:
        raise RuntimeError(f"profile omitted matching features: {missing}")
    audit = {
        "excluded_point_count": int(len(excluded_ids)),
        "point_neighbor_count": int(neighbor_ids.size),
        "spectral_donor_count": int(len(donor_ids)),
        "internal_point_neighbor_hits": 0,
        "internal_spectral_donor_hits": 0,
    }
    return output, audit


def transformed_template_hull(xy: np.ndarray, scale: float, rotation_deg: float) -> ConvexHull:
    center = np.mean(xy, axis=0, keepdims=True)
    relative = np.asarray(xy, dtype=np.float64) - center
    theta = math.radians(rotation_deg)
    rotation = np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    transformed = scale * (relative @ rotation.T)
    return ConvexHull(transformed)


def generate_candidates(
    target: e167.Hole,
    target_profile: dict[str, float],
    search: SearchContext,
    setup: object,
    max_centers: int | None,
) -> tuple[list[CandidateOccurrence], dict[str, int]]:
    target_count = len(target.point_ids)
    minimum_count = max(MIN_POINT_COUNT, math.ceil(MIN_COUNT_FRACTION * target_count))
    maximum_count = math.floor(MAX_COUNT_FRACTION * target_count)
    centers = np.arange(len(search.global_ids), dtype=np.int64)
    if max_centers is not None:
        centers = centers[: max(0, max_centers)]
    target_bs = target_profile["bs_distance_median_m"]
    target_width = math.expm1(target_profile["log_width_m"])
    target_height = math.expm1(target_profile["log_height_m"])
    retained: dict[tuple[int, ...], CandidateOccurrence] = {}
    audit = {"raw_outlines": 0, "count_rejected": 0, "duplicates": 0}

    for scale in SCALES:
        for rotation_deg in ROTATIONS_DEG:
            hull = transformed_template_hull(target.xy, scale, rotation_deg)
            radius = float(np.max(np.linalg.norm(hull.points[hull.vertices], axis=1))) + 1e-8
            neighborhoods = search.tree.query_ball_point(
                search.xy[centers], radius, workers=-1
            )
            for center_local, local_ids in zip(centers.tolist(), neighborhoods):
                audit["raw_outlines"] += 1
                if not minimum_count <= len(local_ids) <= maximum_count * 3:
                    audit["count_rejected"] += 1
                    continue
                local_array = np.asarray(local_ids, dtype=np.int64)
                relative = search.xy[local_array] - search.xy[center_local]
                inside = np.all(
                    relative @ hull.equations[:, :-1].T + hull.equations[:, -1]
                    <= 1e-8,
                    axis=1,
                )
                selected_local = local_array[inside]
                if not minimum_count <= len(selected_local) <= maximum_count:
                    audit["count_rejected"] += 1
                    continue
                point_ids = tuple(sorted(search.global_ids[selected_local].tolist()))
                selected_xy = search.xy[selected_local]
                width, height = np.ptp(selected_xy, axis=0)
                bs_distance = np.linalg.norm(
                    selected_xy - setup.bs_positions[target.site, :2], axis=1
                )
                quick_cost = (
                    abs(math.log(len(point_ids) / target_count))
                    + 0.35 * abs(math.log(max(width, 1e-6) / max(target_width, 1e-6)))
                    + 0.35 * abs(math.log(max(height, 1e-6) / max(target_height, 1e-6)))
                    + 0.01 * abs(float(np.median(bs_distance)) - target_bs)
                )
                occurrence = CandidateOccurrence(
                    point_ids=point_ids,
                    center_global_id=int(search.global_ids[center_local]),
                    center_x=float(search.xy[center_local, 0]),
                    center_y=float(search.xy[center_local, 1]),
                    scale=float(scale),
                    rotation_deg=float(rotation_deg),
                    quick_cost=float(quick_cost),
                )
                previous = retained.get(point_ids)
                if previous is None or occurrence.quick_cost < previous.quick_cost:
                    if previous is not None:
                        audit["duplicates"] += 1
                    retained[point_ids] = occurrence
                else:
                    audit["duplicates"] += 1
    return sorted(retained.values(), key=lambda item: item.quick_cost), audit


def robust_feature_matrix(
    profiles: list[dict[str, float]],
) -> tuple[np.ndarray, dict[str, slice], dict[str, list[float]]]:
    raw = np.asarray(
        [[profile[name] for name in FEATURE_NAMES] for profile in profiles], dtype=np.float64
    )
    median = np.nanmedian(raw, axis=0)
    raw = np.where(np.isfinite(raw), raw, median[None])
    mad = np.median(np.abs(raw - median[None]), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(raw, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    standardized = (raw - median[None]) / scale[None]
    groups: dict[str, slice] = {}
    start = 0
    for group, fields in GROUP_FIELDS.items():
        groups[group] = slice(start, start + len(fields))
        start += len(fields)
    return standardized, groups, {"median": median.tolist(), "scale": scale.tolist()}


def grouped_cost(
    left: np.ndarray, right: np.ndarray, groups: dict[str, slice]
) -> tuple[float, dict[str, float]]:
    components = {
        name: float(np.sqrt(np.mean(np.square(left[part] - right[part]))))
        for name, part in groups.items()
    }
    return float(np.mean(list(components.values()))), components


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    a = set(left)
    b = set(right)
    return float(len(a & b) / max(len(a | b), 1))


def greedy_diverse_selection(
    ranked: list[tuple[float, CandidateOccurrence, dict[str, float], np.ndarray]],
    max_selected: int,
) -> list[tuple[float, CandidateOccurrence, dict[str, float], np.ndarray]]:
    selected: list[tuple[float, CandidateOccurrence, dict[str, float], np.ndarray]] = []
    for item in ranked:
        if all(jaccard(item[1].point_ids, other[1].point_ids) <= MAX_PAIR_JACCARD for other in selected):
            selected.append(item)
            if len(selected) >= max_selected:
                break
    return selected


def overlap_statistics(
    selections: dict[str, list[tuple[float, CandidateOccurrence, dict[str, float], np.ndarray]]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, float]]]:
    within_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    for target, items in selections.items():
        values: list[float] = []
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                value = jaccard(items[left][1].point_ids, items[right][1].point_ids)
                values.append(value)
                within_rows.append(
                    {
                        "target_hole_id": target,
                        "left_rank": left + 1,
                        "right_rank": right + 1,
                        "jaccard": value,
                    }
                )
        array = np.asarray(values if values else [0.0], dtype=np.float64)
        summary[target] = {
            "jaccard_median": float(np.median(array)),
            "jaccard_p90": float(np.quantile(array, 0.90)),
            "jaccard_max": float(np.max(array)),
        }

    targets = sorted(selections)
    for left_index, left_target in enumerate(targets):
        for right_target in targets[left_index + 1 :]:
            if left_target[:2] != right_target[:2]:
                continue
            values = [
                jaccard(left[1].point_ids, right[1].point_ids)
                for left in selections[left_target]
                for right in selections[right_target]
            ]
            array = np.asarray(values if values else [0.0], dtype=np.float64)
            cross_rows.append(
                {
                    "left_target_hole_id": left_target,
                    "right_target_hole_id": right_target,
                    "pair_count": len(values),
                    "jaccard_median": float(np.median(array)),
                    "jaccard_p90": float(np.quantile(array, 0.90)),
                    "jaccard_max": float(np.max(array)),
                    "overlap_gt_0_35_count": int(np.sum(array > MAX_PAIR_JACCARD)),
                }
            )
    return within_rows, cross_rows, summary


def archived_canonical_costs() -> dict[str, float]:
    output: dict[str, float] = {}
    with CANONICAL_MATCH_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["original_fixed_pair"].strip().lower() == "true":
                output[row["real_hole_id"]] = float(row["cost"])
    return output


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    setup, train_positions, _, train_site, _ = load_positions()
    real_holes, canonical_holes, context = e167.build_holes()
    if len(real_holes) != 10 or len(canonical_holes) != 10:
        raise RuntimeError(
            f"expected 10 real and 10 canonical holes, got {len(real_holes)}/{len(canonical_holes)}"
        )

    map_cache = np.load(MAP_FEATURES, allow_pickle=False)
    map_names = np.asarray(map_cache["feature_names"])
    train_map = np.asarray(map_cache["train_features"], dtype=np.float32)
    descriptors = np.load(SPECTRAL_DESCRIPTORS, allow_pickle=False)
    train_pas = normalized_rows(descriptors["global_pas"])
    train_pdp = normalized_rows(descriptors["global_pdp"])
    searches = build_search_contexts(train_positions, train_site)

    target_profiles: dict[str, dict[str, float]] = {}
    canonical_profiles: dict[str, dict[str, float]] = {}
    leak_audits: list[dict[str, int]] = []
    canonical_by_key = {
        (hole.site, hole.ordinal): hole for hole in canonical_holes
    }
    for target in real_holes:
        target_profile, audit = build_profile(
            xy=target.xy,
            point_ids=None,
            excluded_ids=np.empty(0, dtype=np.int64),
            site=target.site,
            map_features=target.map_features,
            setup=setup,
            map_names=map_names,
            train_positions=train_positions,
            train_pas=train_pas,
            train_pdp=train_pdp,
            search=searches[target.site],
        )
        target_profiles[target.hole_id] = target_profile
        leak_audits.append(audit)

        canonical = canonical_by_key[(target.site, target.ordinal)]
        excluded = np.concatenate([canonical.point_ids, canonical.ignored_ids])
        canonical_profile, audit = build_profile(
            xy=canonical.xy,
            point_ids=canonical.point_ids,
            excluded_ids=excluded,
            site=canonical.site,
            map_features=canonical.map_features,
            setup=setup,
            map_names=map_names,
            train_positions=train_positions,
            train_pas=train_pas,
            train_pdp=train_pdp,
            search=searches[canonical.site],
        )
        canonical_profiles[target.hole_id] = canonical_profile
        leak_audits.append(audit)

    pools: dict[str, list[CandidateOccurrence]] = {}
    generation_audits: dict[str, dict[str, int]] = {}
    unique_occurrences: dict[tuple[int, ...], CandidateOccurrence] = {}
    for target in real_holes:
        candidates, audit = generate_candidates(
            target,
            target_profiles[target.hole_id],
            searches[target.site],
            setup,
            args.max_centers,
        )
        candidates = candidates[: args.prescreen_limit]
        pools[target.hole_id] = candidates
        generation_audits[target.hole_id] = {
            **audit,
            "unique_count": int(audit["raw_outlines"] - audit["count_rejected"] - audit["duplicates"]),
            "prescreen_count": len(candidates),
        }
        for candidate in candidates:
            unique_occurrences.setdefault(candidate.point_ids, candidate)
        print(
            f"E183 candidates {target.hole_id}: unique="
            f"{generation_audits[target.hole_id]['unique_count']} "
            f"prescreen={len(candidates)}",
            flush=True,
        )

    if args.dry_run:
        print(
            json.dumps(
                json_compatible(
                    {
                        "status": "dry_run_passed",
                        "real_hole_count": len(real_holes),
                        "canonical_hole_count": len(canonical_holes),
                        "candidate_generation": generation_audits,
                        "matching_feature_names": FEATURE_NAMES,
                        "matching_reads_complete_train_h": False,
                        "matching_uses_candidate_internal_descriptors": False,
                        "internal_neighbor_hits": int(
                            sum(
                                item["internal_point_neighbor_hits"]
                                + item["internal_spectral_donor_hits"]
                                for item in leak_audits
                            )
                        ),
                    }
                ),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

    candidate_profiles: dict[tuple[int, ...], dict[str, float]] = {}
    candidate_audits: dict[tuple[int, ...], dict[str, int]] = {}
    for index, (key, candidate) in enumerate(unique_occurrences.items(), start=1):
        ids = np.asarray(key, dtype=np.int64)
        site = int(train_site[ids[0]])
        profile, audit = build_profile(
            xy=train_positions[ids, :2],
            point_ids=ids,
            excluded_ids=ids,
            site=site,
            map_features=train_map[ids],
            setup=setup,
            map_names=map_names,
            train_positions=train_positions,
            train_pas=train_pas,
            train_pdp=train_pdp,
            search=searches[site],
        )
        candidate_profiles[key] = profile
        candidate_audits[key] = audit
        if index % 500 == 0 or index == len(unique_occurrences):
            print(f"E183 profiles: {index}/{len(unique_occurrences)}", flush=True)

    ordered_target_ids = [hole.hole_id for hole in real_holes]
    ordered_candidate_keys = list(candidate_profiles)
    profiles = (
        [target_profiles[hole_id] for hole_id in ordered_target_ids]
        + [canonical_profiles[hole_id] for hole_id in ordered_target_ids]
        + [candidate_profiles[key] for key in ordered_candidate_keys]
    )
    standardized, groups, scaler = robust_feature_matrix(profiles)
    target_vectors = standardized[:10]
    canonical_vectors = standardized[10:20]
    candidate_vectors = standardized[20:]
    candidate_vector_by_key = {
        key: candidate_vectors[index] for index, key in enumerate(ordered_candidate_keys)
    }
    canonical_costs: dict[str, float] = {}
    canonical_components: dict[str, dict[str, float]] = {}
    for index, hole_id in enumerate(ordered_target_ids):
        cost, components = grouped_cost(
            target_vectors[index], canonical_vectors[index], groups
        )
        canonical_costs[hole_id] = cost
        canonical_components[hole_id] = components
    canonical_median = float(np.median(list(canonical_costs.values())))
    archived_costs = archived_canonical_costs()

    selections: dict[
        str, list[tuple[float, CandidateOccurrence, dict[str, float], np.ndarray]]
    ] = {}
    for target_index, hole_id in enumerate(ordered_target_ids):
        ranked = []
        for candidate in pools[hole_id]:
            vector = candidate_vector_by_key[candidate.point_ids]
            cost, components = grouped_cost(target_vectors[target_index], vector, groups)
            ranked.append((cost, candidate, components, vector))
        ranked.sort(key=lambda item: item[0])
        selections[hole_id] = greedy_diverse_selection(ranked, args.max_selected)

    within_rows, cross_rows, overlap_summary = overlap_statistics(selections)
    summary_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    selected_ids_flat: list[int] = []
    selected_offsets = [0]
    selected_target_index: list[int] = []
    selected_rank: list[int] = []
    selected_match_cost: list[float] = []
    selected_center_id: list[int] = []
    selected_scale: list[float] = []
    selected_rotation: list[float] = []
    selected_features: list[np.ndarray] = []
    gate_passes: list[bool] = []

    for target_index, hole_id in enumerate(ordered_target_ids):
        selected = selections[hole_id]
        costs = np.asarray([item[0] for item in selected], dtype=np.float64)
        top20 = costs[: min(20, len(costs))]
        median_top20 = float(np.median(top20)) if len(top20) else float("inf")
        cost_threshold = max(canonical_costs[hole_id], canonical_median)
        overlap = overlap_summary[hole_id]
        count_gate = len(selected) >= MIN_PASS_COUNT
        absolute_gate = len(selected) >= MIN_ABSOLUTE_COUNT
        cost_gate = len(top20) >= 20 and median_top20 <= cost_threshold
        overlap_gate = (
            overlap["jaccard_p90"] <= MAX_P90_JACCARD
            and overlap["jaccard_max"] <= MAX_PAIR_JACCARD
        )
        passed = count_gate and cost_gate and overlap_gate
        gate_passes.append(passed)
        if not absolute_gate:
            status = "fail_fewer_than_12"
        elif not count_gate:
            status = "weak_12_to_19"
        elif not cost_gate:
            status = "weak_match_cost"
        elif not overlap_gate:
            status = "weak_overlap"
        else:
            status = "pass"
        summary_rows.append(
            {
                "target_hole_id": hole_id,
                "site": int(hole_id[1]),
                "target_point_count": len(real_holes[target_index].point_ids),
                **generation_audits[hole_id],
                "selected_count": len(selected),
                "best_match_cost": float(costs[0]) if len(costs) else float("nan"),
                "top20_match_cost_median": median_top20,
                "top20_match_cost_p90": float(np.quantile(top20, 0.90)) if len(top20) else float("nan"),
                "e183_canonical_cost": canonical_costs[hole_id],
                "canonical_cost_median": canonical_median,
                "cost_gate_threshold": cost_threshold,
                "archived_e167_canonical_cost": archived_costs.get(hole_id, float("nan")),
                **overlap,
                "count_gate": count_gate,
                "cost_gate": cost_gate,
                "overlap_gate": overlap_gate,
                "gate_status": status,
            }
        )
        for rank, (cost, candidate, components, vector) in enumerate(selected, start=1):
            ids = list(candidate.point_ids)
            inventory_rows.append(
                {
                    "pseudo_hole_id": f"E183_{hole_id}_{rank:02d}",
                    "target_hole_id": hole_id,
                    "site": int(hole_id[1]),
                    "rank": rank,
                    "point_count": len(ids),
                    "center_global_id": candidate.center_global_id,
                    "center_x": candidate.center_x,
                    "center_y": candidate.center_y,
                    "scale": candidate.scale,
                    "rotation_deg": candidate.rotation_deg,
                    "quick_cost": candidate.quick_cost,
                    "match_cost": cost,
                    **{f"cost_{name}": value for name, value in components.items()},
                    "point_ids": " ".join(str(value) for value in ids),
                }
            )
            selected_ids_flat.extend(ids)
            selected_offsets.append(len(selected_ids_flat))
            selected_target_index.append(target_index)
            selected_rank.append(rank)
            selected_match_cost.append(cost)
            selected_center_id.append(candidate.center_global_id)
            selected_scale.append(candidate.scale)
            selected_rotation.append(candidate.rotation_deg)
            selected_features.append(vector)

    internal_hits = int(
        sum(
            audit["internal_point_neighbor_hits"] + audit["internal_spectral_donor_hits"]
            for audit in [*leak_audits, *candidate_audits.values()]
        )
    )
    if internal_hits:
        raise AssertionError(f"internal matching-feature leakage detected: {internal_hits}")
    overall_pass = bool(all(gate_passes))
    elapsed = time.perf_counter() - started
    peak_memory_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LIBRARY_NPZ,
        point_ids_flat=np.asarray(selected_ids_flat, dtype=np.int32),
        point_offsets=np.asarray(selected_offsets, dtype=np.int64),
        target_index=np.asarray(selected_target_index, dtype=np.int8),
        rank=np.asarray(selected_rank, dtype=np.int8),
        match_cost=np.asarray(selected_match_cost, dtype=np.float32),
        center_global_id=np.asarray(selected_center_id, dtype=np.int32),
        scale=np.asarray(selected_scale, dtype=np.float32),
        rotation_deg=np.asarray(selected_rotation, dtype=np.float32),
        standardized_features=np.asarray(selected_features, dtype=np.float32),
        target_hole_ids=np.asarray(ordered_target_ids, dtype="U8"),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U64"),
        target_features=target_vectors.astype(np.float32),
        canonical_features=canonical_vectors.astype(np.float32),
        canonical_cost=np.asarray(
            [canonical_costs[hole_id] for hole_id in ordered_target_ids], dtype=np.float32
        ),
    )
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(INVENTORY_CSV, inventory_rows)
    write_csv(WITHIN_OVERLAP_CSV, within_rows)
    write_csv(CROSS_OVERLAP_CSV, cross_rows)

    report = {
        "experiment": VERSION,
        "status": "matching_library_gate_passed" if overall_pass else "matching_library_gate_failed",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "model_training": False,
        "complete_train_h_read": False,
        "candidate_internal_h_or_descriptors_used": False,
        "goal": "Build deployment-matched Clean pseudo-hole families for each real Test hole.",
        "contract": {
            "minimum_selected_per_target": MIN_PASS_COUNT,
            "immediate_failure_below": MIN_ABSOLUTE_COUNT,
            "maximum_selected_per_target": args.max_selected,
            "top20_cost_rule": "median <= max(target E183 canonical cost, median E183 canonical cost)",
            "within_family_jaccard_p90_max": MAX_P90_JACCARD,
            "within_family_jaccard_max": MAX_PAIR_JACCARD,
        },
        "template_transforms": {
            "scales": SCALES,
            "rotations_deg": ROTATIONS_DEG,
            "minimum_count_fraction": MIN_COUNT_FRACTION,
            "maximum_count_fraction": MAX_COUNT_FRACTION,
        },
        "matching_features": {
            "groups": GROUP_FIELDS,
            "feature_names": FEATURE_NAMES,
            "scaler": scaler,
            "candidate_internal_points_excluded_from_all_neighbor_features": True,
            "internal_neighbor_hits": internal_hits,
        },
        "canonical_cost_components": canonical_components,
        "canonical_cost_median": canonical_median,
        "targets": summary_rows,
        "cross_target_overlap": cross_rows,
        "overall_pass": overall_pass,
        "selected_pseudo_hole_count": len(inventory_rows),
        "compute": {
            "wall_seconds": elapsed,
            "selected_gpu": "none_cpu_matching_gate",
            "peak_memory_gib": peak_memory_gib,
            "measured_bottleneck": "candidate profile construction and external-neighbor summaries",
        },
        "outputs": {
            "library_npz": str(LIBRARY_NPZ.relative_to(PATHS.root)),
            "summary_csv": str(SUMMARY_CSV.relative_to(PATHS.root)),
            "inventory_csv": str(INVENTORY_CSV.relative_to(PATHS.root)),
            "within_overlap_csv": str(WITHIN_OVERLAP_CSV.relative_to(PATHS.root)),
            "cross_overlap_csv": str(CROSS_OVERLAP_CSV.relative_to(PATHS.root)),
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(
            json_compatible(report), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_compatible(report), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
