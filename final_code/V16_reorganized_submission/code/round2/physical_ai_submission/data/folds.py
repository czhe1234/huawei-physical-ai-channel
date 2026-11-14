from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans


@dataclass(frozen=True)
class SpatialHoleFold:
    fold: int
    fit_indices: np.ndarray
    query_indices: np.ndarray
    ignored_indices: np.ndarray
    query_site: np.ndarray
    rows: tuple[dict[str, float | int | list[float]], ...]


def _load_external_folds(
    train_positions: np.ndarray,
    train_site: np.ndarray,
) -> list[SpatialHoleFold] | None:
    configured = os.environ.get("HW_SPATIAL_FOLD_CACHE")
    if not configured:
        return None
    source = Path(configured).expanduser().resolve()
    cached = np.load(source, allow_pickle=False)
    folds: list[SpatialHoleFold] = []
    all_ids = np.arange(len(train_positions), dtype=np.int64)
    for fold in range(5):
        key = f"fold{fold }_query"
        if key not in cached:
            raise RuntimeError(f"missing {key } in external fold cache: {source }")
        query = np.unique(np.asarray(cached[key], dtype=np.int64))
        if np.any((query < 0) | (query >= len(train_positions))):
            raise RuntimeError(f"out-of-range query IDs in {source }: fold {fold }")
        fit_key = f"fold{fold }_fit"
        if fit_key in cached:
            fit = np.unique(np.asarray(cached[fit_key], dtype=np.int64))
        else:
            fit = all_ids[~np.isin(all_ids, query)]
        ignored_key = f"fold{fold }_ignored"
        ignored = (
            np.unique(np.asarray(cached[ignored_key], dtype=np.int64))
            if ignored_key in cached
            else np.empty(0, dtype=np.int64)
        )
        if np.intersect1d(query, fit).size or np.intersect1d(query, ignored).size:
            raise RuntimeError(
                f"overlapping external fold sets in {source }: fold {fold }"
            )
        rows = tuple(
            {
                "site": int(site_value),
                "query_count": int(np.sum(train_site[query] == site_value)),
                "fit_count": int(np.sum(train_site[fit] == site_value)),
                "ignored_guard_count": int(np.sum(train_site[ignored] == site_value)),
                "guard_m": 0.0,
            }
            for site_value in sorted(np.unique(train_site).tolist())
        )
        folds.append(
            SpatialHoleFold(
                fold=fold,
                fit_indices=fit,
                query_indices=query,
                ignored_indices=ignored,
                query_site=np.asarray(train_site[query], dtype=np.int8),
                rows=rows,
            )
        )
    covered = np.concatenate([fold.query_indices for fold in folds])
    if len(np.unique(covered)) != len(covered):
        raise RuntimeError(f"external fold queries overlap: {source }")
    return folds


def _sorted_test_clusters(test_xy: np.ndarray) -> list[np.ndarray]:
    labels = DBSCAN(eps=15.0, min_samples=3).fit_predict(test_xy)
    if np.any(labels < 0):
        raise RuntimeError("DBSCAN left test coordinates outside a spatial cluster")
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return sorted(
        clusters,
        key=lambda ids: tuple(np.mean(test_xy[ids], axis=0).tolist()),
    )


def _nearest_distance(query_xy: np.ndarray, fit_xy: np.ndarray) -> np.ndarray:
    return np.asarray(
        cKDTree(fit_xy).query(query_xy, k=1, workers=-1)[0], dtype=np.float64
    )


def _guarded_fit_mask(
    site_xy: np.ndarray,
    query_local: np.ndarray,
    guard_m: float,
) -> np.ndarray:
    distance_to_query = cKDTree(site_xy[query_local]).query(site_xy, k=1, workers=-1)[0]
    mask = distance_to_query > guard_m
    mask[query_local] = False
    return mask


def _match_guard_distance(
    site_xy: np.ndarray,
    query_local: np.ndarray,
    target_median_m: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    low = 0.0
    high = max(2.0, target_median_m * 2.0)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(24):
        guard = 0.5 * (low + high)
        fit_mask = _guarded_fit_mask(site_xy, query_local, guard)
        if int(np.sum(fit_mask)) < 256:
            high = guard
            continue
        distances = _nearest_distance(site_xy[query_local], site_xy[fit_mask])
        best = (guard, fit_mask, distances)
        if float(np.median(distances)) < target_median_m:
            low = guard
        else:
            high = guard
    if best is None:
        raise RuntimeError("could not construct a guarded spatial fold")
    return best


def build_spatial_hole_folds(
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    seed: int = 20_260_805,
) -> list[SpatialHoleFold]:
    pass

    external = _load_external_folds(train_positions, train_site)
    if external is not None:
        return external

    fold_parts: list[list[tuple[int, np.ndarray, np.ndarray, dict[str, object]]]] = [
        [] for _ in range(5)
    ]
    for site in sorted(np.unique(train_site).tolist()):
        site_global = np.flatnonzero(train_site == site)
        site_xy = np.asarray(train_positions[site_global, :2], dtype=np.float64)
        test_global = np.flatnonzero(test_site == site)
        test_xy = np.asarray(test_positions[test_global, :2], dtype=np.float64)
        test_clusters = _sorted_test_clusters(test_xy)
        if len(test_clusters) != 5:
            raise RuntimeError(
                f"site {site } has {len (test_clusters )} test clusters, expected 5"
            )

        broad = KMeans(n_clusters=5, random_state=seed + int(site), n_init=32).fit(
            site_xy
        )
        broad_ids = sorted(
            range(5), key=lambda label: tuple(broad.cluster_centers_[label].tolist())
        )
        for fold, (broad_id, test_cluster) in enumerate(zip(broad_ids, test_clusters)):
            members = np.flatnonzero(broad.labels_ == broad_id)
            center = broad.cluster_centers_[broad_id]
            target_distances = _nearest_distance(test_xy[test_cluster], site_xy)
            target_median = float(np.median(target_distances))
            template_xy = test_xy[test_cluster]
            centered_template = template_xy - np.mean(
                template_xy, axis=0, keepdims=True
            )
            mappings: list[tuple[float, float, np.ndarray, np.ndarray]] = []
            for template_scale in (0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
                translated_template = (
                    center[None, :] + template_scale * centered_template
                )
                assignment_cost = distance_matrix(translated_template, site_xy[members])
                template_rows, member_columns = linear_sum_assignment(assignment_cost)
                if len(template_rows) != len(test_cluster):
                    continue
                candidate_query = np.sort(members[member_columns])
                natural_fit = _guarded_fit_mask(site_xy, candidate_query, 0.0)
                natural_distance = _nearest_distance(
                    site_xy[candidate_query], site_xy[natural_fit]
                )
                mappings.append(
                    (
                        template_scale,
                        float(np.median(natural_distance)),
                        candidate_query,
                        assignment_cost[template_rows, member_columns],
                    )
                )
            if not mappings:
                raise RuntimeError("could not map the complete test-hole template")
            feasible = [mapping for mapping in mappings if mapping[1] <= target_median]
            if feasible:
                template_scale, natural_median, query_local, mapping_distance = max(
                    feasible, key=lambda mapping: mapping[1]
                )
            else:
                template_scale, natural_median, query_local, mapping_distance = min(
                    mappings, key=lambda mapping: mapping[1]
                )
            guard, fit_local_mask, achieved = _match_guard_distance(
                site_xy, query_local, target_median
            )
            fit_local = np.flatnonzero(fit_local_mask)
            ignored_local = np.flatnonzero(~fit_local_mask)
            ignored_local = ignored_local[~np.isin(ignored_local, query_local)]
            query_global = site_global[query_local]
            fit_global = site_global[fit_local]
            ignored_global = site_global[ignored_local]

            row: dict[str, object] = {
                "site": int(site),
                "query_count": int(len(query_global)),
                "fit_count": int(len(fit_global)),
                "ignored_guard_count": int(len(ignored_global)),
                "guard_m": float(guard),
                "target_test_nearest_median_m": target_median,
                "validation_nearest_median_m": float(np.median(achieved)),
                "validation_nearest_p90_m": float(np.quantile(achieved, 0.90)),
                "template_scale": float(template_scale),
                "natural_validation_nearest_median_m": float(natural_median),
                "template_mapping_median_m": float(np.median(mapping_distance)),
                "template_mapping_p90_m": float(np.quantile(mapping_distance, 0.90)),
                "query_center_xy": np.mean(site_xy[query_local], axis=0).tolist(),
                "test_center_xy": np.mean(test_xy[test_cluster], axis=0).tolist(),
            }
            fold_parts[fold].append(
                (
                    int(site),
                    fit_global,
                    query_global,
                    {"ignored": ignored_global, **row},
                )
            )

    folds: list[SpatialHoleFold] = []
    all_indices = np.arange(len(train_positions), dtype=np.int64)
    for fold, parts in enumerate(fold_parts):
        query = np.sort(np.concatenate([part[2] for part in parts])).astype(np.int64)
        ignored = np.sort(
            np.concatenate(
                [np.asarray(part[3]["ignored"], dtype=np.int64) for part in parts]
            )
        )
        removed = np.zeros(len(train_positions), dtype=bool)
        removed[query] = True
        removed[ignored] = True
        fit = all_indices[~removed]
        rows = []
        for _, _, _, raw_row in parts:
            rows.append(
                {key: value for key, value in raw_row.items() if key != "ignored"}
            )
        folds.append(
            SpatialHoleFold(
                fold=fold,
                fit_indices=fit,
                query_indices=query,
                ignored_indices=ignored,
                query_site=np.asarray(train_site[query], dtype=np.int8),
                rows=tuple(rows),
            )
        )
    return folds
