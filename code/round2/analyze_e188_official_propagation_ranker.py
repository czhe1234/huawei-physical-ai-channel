#!/usr/bin/env python3
"""Learn a Clean ranking of propagation-similar official Train points."""

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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor
import torch

import analyze_v4_route_topology_retrieval as e11
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics_torch,
)
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E188-official-propagation-ranker"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e188_official_propagation_ranker"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e188_official_propagation_ranker"
)

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
BROAD_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
DESCRIPTOR_PATH = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_spectral_descriptors.npz"
)
PAS_PATH = (
    PATHS.artifacts
    / "round2"
    / "e169_hard_hole_pas_peak_precursors"
    / "official_train_compact_pas.npy"
)
AD_PATH = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_ad_support_f16.npy"
)
GEOMETRY_PATH = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
ROUTE_PATH = (
    PATHS.artifacts
    / "round2"
    / "route_topology"
    / "round2_route_topology_features.npz"
)
COMPONENT_PATH = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
E184_METRICS = (
    PATHS.artifacts
    / "round2"
    / "e184_clean_propagation_similarity_retrieval"
    / "retrieval_metrics.npz"
)

LOCAL_HOLE_NEIGHBORS = 8
BOUNDARY_COUNTS = (8, 24, 64)
MAX_BOUNDARY_COUNT = max(BOUNDARY_COUNTS)
RANKED_CONTEXT_COUNT = 320
CANDIDATE_COUNT = 256
TRAIN_CANDIDATE_COUNT = 32
TOP_K = 5
LABEL_WEIGHTS = (0.35, 0.35, 0.30)

TREE_COUNT = 192
TREE_MAX_DEPTH = 14
TREE_MIN_SAMPLES_LEAF = 6
TREE_MAX_FEATURES = 0.65

SEED = 20_260_812_188
TINY = 1e-30


@dataclass(frozen=True)
class QuerySpec:
    query_id: int
    site: int
    removed_ids: np.ndarray
    candidate_ids: np.ndarray
    ranked_context_ids: np.ndarray


@dataclass(frozen=True)
class SimilarityBank:
    ids_by_site: dict[int, np.ndarray]
    local_index: np.ndarray
    pas: dict[int, np.ndarray]
    pdp: dict[int, np.ndarray]
    angle_delay: dict[int, np.ndarray]

    def pair_values(
        self,
        site: int,
        left_ids: np.ndarray,
        right_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left = self.local_index[np.asarray(left_ids, dtype=np.int64)]
        right = self.local_index[np.asarray(right_ids, dtype=np.int64)]
        if np.any(left < 0) or np.any(right < 0):
            raise RuntimeError("pair lookup escaped the site similarity bank")
        return (
            self.pas[site][np.ix_(left, right)],
            self.pdp[site][np.ix_(left, right)],
            self.angle_delay[site][np.ix_(left, right)],
        )

    def truth_components(
        self, query_id: int, candidate_ids: np.ndarray, site: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_local = int(self.local_index[query_id])
        candidate_local = self.local_index[np.asarray(candidate_ids, dtype=np.int64)]
        if query_local < 0 or np.any(candidate_local < 0):
            raise RuntimeError("truth lookup escaped the site similarity bank")
        return (
            self.pas[site][query_local, candidate_local],
            self.pdp[site][query_local, candidate_local],
            self.angle_delay[site][query_local, candidate_local],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "full"), default="screen")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--matrix-batch-size", type=int, default=256)
    parser.add_argument("--metric-batch-size", type=int, default=16)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def normalize_groups(value: np.ndarray, groups: int) -> np.ndarray:
    shaped = np.maximum(np.asarray(value, dtype=np.float32), 0.0).reshape(
        len(value), groups, -1
    )
    norm = np.sqrt(
        np.sum(shaped * shaped, axis=2, keepdims=True, dtype=np.float64)
    )
    return np.divide(
        shaped,
        np.maximum(norm, TINY),
        out=np.zeros_like(shaped),
        where=norm > 0.0,
    )


@torch.inference_mode()
def grouped_similarity_matrix(
    value: np.ndarray,
    groups: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    source = torch.as_tensor(
        normalize_groups(value, groups), dtype=torch.float32, device=device
    )
    output = np.empty((len(source), len(source)), dtype=np.float32)
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        similarity = torch.einsum("bgd,kgd->bkg", source[start:end], source)
        output[start:end] = (
            similarity.mean(dim=2, dtype=torch.float64).float().cpu().numpy()
        )
    return output


@torch.inference_mode()
def hellinger_similarity_matrix(
    value: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    source = torch.sqrt(
        torch.as_tensor(np.asarray(value, dtype=np.float32), device=device).clamp_min(
            0.0
        )
    ).reshape(len(value), -1)
    source = source / torch.linalg.vector_norm(source, dim=1, keepdim=True).clamp_min(
        TINY
    )
    output = np.empty((len(source), len(source)), dtype=np.float32)
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        output[start:end] = (source[start:end] @ source.T).cpu().numpy()
    return output


def build_similarity_bank(
    train_site: np.ndarray,
    compact_pas: np.ndarray,
    global_pdp: np.ndarray,
    angle_delay: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[SimilarityBank, dict[str, float]]:
    started = time.perf_counter()
    ids_by_site: dict[int, np.ndarray] = {}
    pas: dict[int, np.ndarray] = {}
    pdp: dict[int, np.ndarray] = {}
    ad: dict[int, np.ndarray] = {}
    local_index = np.full(len(train_site), -1, dtype=np.int32)
    for site in (0, 1):
        ids = np.flatnonzero(train_site == site).astype(np.int64)
        ids_by_site[site] = ids
        local_index[ids] = np.arange(len(ids), dtype=np.int32)
        pas[site] = grouped_similarity_matrix(
            np.asarray(compact_pas[ids], dtype=np.float32), 4, device, batch_size
        )
        pdp[site] = grouped_similarity_matrix(
            np.asarray(global_pdp[ids], dtype=np.float32), 1, device, batch_size
        )
        ad[site] = hellinger_similarity_matrix(
            np.asarray(angle_delay[ids], dtype=np.float32), device, batch_size
        )
        print(f"E188 similarity site={site} count={len(ids)}", flush=True)
    torch.cuda.synchronize(device)
    return (
        SimilarityBank(ids_by_site, local_index, pas, pdp, ad),
        {"wall_seconds": time.perf_counter() - started},
    )


def make_query_specs(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    positions: np.ndarray,
    train_site: np.ndarray,
    pseudo_hole: bool,
    outer_forbidden: np.ndarray,
) -> dict[int, QuerySpec]:
    output: dict[int, QuerySpec] = {}
    for site in (0, 1):
        rows = np.asarray(
            [int(value) for value in query_ids if int(train_site[value]) == site],
            dtype=np.int64,
        )
        if len(rows) == 0:
            continue
        bank = fit_ids[train_site[fit_ids] == site]
        count = min(len(bank), RANKED_CONTEXT_COUNT + LOCAL_HOLE_NEIGHBORS + 16)
        distance, local = cKDTree(positions[bank, :2]).query(
            positions[rows, :2], k=count, workers=-1
        )
        distance = np.atleast_2d(distance)
        local = np.atleast_2d(local)
        raw_ids = bank[local]
        for row_index, query_id in enumerate(rows):
            ids = raw_ids[row_index]
            if pseudo_hole:
                neighbors = ids[ids != query_id][:LOCAL_HOLE_NEIGHBORS]
                if len(neighbors) != LOCAL_HOLE_NEIGHBORS:
                    raise RuntimeError(f"query {query_id} lacks local-hole neighbors")
                removed = np.concatenate(
                    [np.asarray([query_id], dtype=np.int64), neighbors]
                )
            else:
                removed = np.asarray(outer_forbidden, dtype=np.int64)
            keep = ~np.isin(ids, removed)
            available = ids[keep]
            if len(available) < CANDIDATE_COUNT:
                raise RuntimeError(
                    f"query {query_id} has only {len(available)} legal candidates"
                )
            output[int(query_id)] = QuerySpec(
                query_id=int(query_id),
                site=site,
                removed_ids=np.asarray(removed, dtype=np.int64),
                candidate_ids=available[:CANDIDATE_COUNT].astype(np.int64),
                ranked_context_ids=available[:RANKED_CONTEXT_COUNT].astype(np.int64),
            )
    if len(output) != len(query_ids):
        raise RuntimeError("query-spec construction lost rows")
    return output


def derangement(ids: np.ndarray, site: np.ndarray, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    output: dict[int, int] = {}
    for site_value in (0, 1):
        local = np.asarray(ids[site[ids] == site_value], dtype=np.int64)
        if len(local) == 0:
            continue
        if len(local) == 1:
            output[int(local[0])] = int(local[0])
            continue
        order = rng.permutation(local)
        shifted = np.roll(order, 1)
        for left, right in zip(order, shifted):
            output[int(left)] = int(right)
    return output


def relative_bs_feature(points: np.ndarray, bs_xy: np.ndarray) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64)[:, :2] - bs_xy[None]
    radius = np.linalg.norm(delta, axis=1)
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    return np.column_stack(
        [delta[:, 0], delta[:, 1], radius, np.cos(angle), np.sin(angle)]
    ).astype(np.float32)


def row_percentile(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.float32)
    rank[order] = np.arange(len(values), dtype=np.float32)
    return rank / max(len(values) - 1, 1)


def context_boundary_ids(
    context: QuerySpec,
    target: QuerySpec,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    forbidden = np.unique(
        np.concatenate(
            [target.removed_ids, np.asarray([target.query_id], dtype=np.int64)]
        )
    )
    keep = ~np.isin(context.ranked_context_ids, forbidden)
    ids = context.ranked_context_ids[keep][:MAX_BOUNDARY_COUNT]
    if len(ids) != MAX_BOUNDARY_COUNT:
        raise RuntimeError(
            f"context {context.query_id} for target {target.query_id} has "
            f"only {len(ids)} safe boundary points"
        )
    distance = np.linalg.norm(
        positions[ids, :2] - positions[context.query_id, None, :2], axis=1
    ).astype(np.float32)
    order = np.argsort(distance, kind="stable")
    return ids[order], distance[order]


def boundary_weights(distance: np.ndarray) -> np.ndarray:
    scale = max(float(np.median(distance)), 1.0)
    weight = np.exp(-np.asarray(distance, dtype=np.float64) / scale)
    return (weight / max(float(np.sum(weight)), TINY)).astype(np.float32)


def append_block(
    blocks: list[np.ndarray],
    names: list[str],
    prefix: str,
    value: np.ndarray,
) -> None:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    blocks.append(array)
    if array.shape[1] == 1:
        names.append(prefix)
    else:
        names.extend(f"{prefix}_{index}" for index in range(array.shape[1]))


def pair_features(
    target: QuerySpec,
    context: QuerySpec,
    candidate_ids: np.ndarray,
    setup,
    positions: np.ndarray,
    geometry: np.ndarray,
    route: dict[str, np.ndarray],
    power: np.ndarray,
    pas_entropy: np.ndarray,
    pdp_entropy: np.ndarray,
    valid: np.ndarray,
    similarities: SimilarityBank,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    boundary_ids, boundary_distance = context_boundary_ids(
        context, target, positions
    )
    pas_matrix, pdp_matrix, ad_matrix = similarities.pair_values(
        target.site, candidates, boundary_ids
    )
    view_matrices = {
        "pas": pas_matrix,
        "pdp": pdp_matrix,
        "ad": ad_matrix,
    }

    blocks: list[np.ndarray] = []
    names: list[str] = []
    candidate_count = len(candidates)
    weighted_view: dict[tuple[str, int], np.ndarray] = {}
    for view_name, matrix in view_matrices.items():
        for count in BOUNDARY_COUNTS:
            local = matrix[:, :count]
            weight = boundary_weights(boundary_distance[:count])
            weighted = local @ weight
            weighted_view[(view_name, count)] = weighted.astype(np.float32)
            append_block(blocks, names, f"{view_name}_weighted_k{count}", weighted)
            append_block(blocks, names, f"{view_name}_mean_k{count}", local.mean(1))
            append_block(blocks, names, f"{view_name}_max_k{count}", local.max(1))
            append_block(blocks, names, f"{view_name}_std_k{count}", local.std(1))

    for count in BOUNDARY_COUNTS:
        joint = (
            LABEL_WEIGHTS[0] * weighted_view[("pas", count)]
            + LABEL_WEIGHTS[1] * weighted_view[("pdp", count)]
            + LABEL_WEIGHTS[2] * weighted_view[("ad", count)]
        )
        robust = np.minimum(
            np.minimum(
                weighted_view[("pas", count)], weighted_view[("pdp", count)]
            ),
            weighted_view[("ad", count)],
        )
        append_block(blocks, names, f"joint_weighted_k{count}", joint)
        append_block(blocks, names, f"robust_weighted_k{count}", robust)

    # The shuffled control swaps only the surrounding H state. Query geometry stays fixed.
    query_position = positions[target.query_id, :2]
    candidate_position = positions[candidates, :2]
    spatial_delta = candidate_position - query_position[None]
    spatial_distance = np.linalg.norm(spatial_delta, axis=1).astype(np.float32)
    route_distance = e11.route_distance(
        route,
        np.asarray([target.query_id], dtype=np.int64),
        candidates[None],
    )[0].astype(np.float32)
    query_relative = relative_bs_feature(
        positions[[target.query_id]], setup.bs_positions[target.site, :2]
    )[0]
    candidate_relative = relative_bs_feature(
        positions[candidates], setup.bs_positions[target.site, :2]
    )
    relative_delta = candidate_relative - query_relative[None]
    geometry_distance = np.sqrt(
        np.mean(relative_delta * relative_delta, axis=1, dtype=np.float64)
    ).astype(np.float32)
    pas_distance = 1.0 - weighted_view[("pas", 24)]
    pdp_distance = 1.0 - weighted_view[("pdp", 24)]
    spectral_distance = 0.5 * (pas_distance + pdp_distance)
    clean_score = 0.25 * (
        row_percentile(pas_distance)
        + row_percentile(pdp_distance)
        + row_percentile(route_distance)
        + row_percentile(geometry_distance)
    )
    append_block(blocks, names, "spatial_dx", spatial_delta[:, 0])
    append_block(blocks, names, "spatial_dy", spatial_delta[:, 1])
    append_block(blocks, names, "spatial_distance", spatial_distance)
    append_block(blocks, names, "route_distance", route_distance)
    append_block(blocks, names, "bs_geometry_distance", geometry_distance)
    append_block(blocks, names, "pas_distance", pas_distance)
    append_block(blocks, names, "pdp_distance", pdp_distance)
    append_block(blocks, names, "spectral_distance", spectral_distance)
    append_block(blocks, names, "clean_score", clean_score)
    append_block(blocks, names, "spatial_rank", row_percentile(spatial_distance))
    append_block(blocks, names, "route_rank", row_percentile(route_distance))
    append_block(blocks, names, "geometry_rank", row_percentile(geometry_distance))
    append_block(blocks, names, "pas_rank", row_percentile(pas_distance))
    append_block(blocks, names, "pdp_rank", row_percentile(pdp_distance))
    append_block(blocks, names, "spectral_rank", row_percentile(spectral_distance))

    query_map = geometry[target.query_id]
    candidate_map = geometry[candidates]
    append_block(
        blocks,
        names,
        "query_map",
        np.broadcast_to(query_map[None], (candidate_count, len(query_map))),
    )
    append_block(blocks, names, "candidate_map", candidate_map)
    append_block(blocks, names, "map_abs_delta", np.abs(candidate_map - query_map))
    append_block(
        blocks,
        names,
        "query_bs",
        np.broadcast_to(query_relative[None], (candidate_count, len(query_relative))),
    )
    append_block(blocks, names, "candidate_bs", candidate_relative)
    append_block(blocks, names, "bs_delta", relative_delta)

    log_power = np.log1p(np.maximum(power, 0.0)).astype(np.float32)
    boundary_weight = boundary_weights(boundary_distance[:24])
    boundary_log_power = float(log_power[boundary_ids[:24]] @ boundary_weight)
    candidate_scalar = np.column_stack(
        [
            log_power[candidates],
            pas_entropy[candidates],
            pdp_entropy[candidates],
            valid[candidates].astype(np.float32),
            log_power[candidates] - boundary_log_power,
        ]
    ).astype(np.float32)
    append_block(blocks, names, "candidate_state", candidate_scalar)

    context_scalars: list[float] = [
        float(target.site),
        float(boundary_distance[0]),
        float(boundary_distance[7]),
        float(boundary_distance[23]),
        float(boundary_distance[63]),
        float(np.mean(boundary_distance[:8])),
        float(np.std(boundary_distance[:8])),
        boundary_log_power,
        float(np.std(log_power[boundary_ids[:24]])),
        float(np.mean(valid[boundary_ids[:24]])),
        float(np.mean(pas_entropy[boundary_ids[:24]])),
        float(np.std(pas_entropy[boundary_ids[:24]])),
        float(np.mean(pdp_entropy[boundary_ids[:24]])),
        float(np.std(pdp_entropy[boundary_ids[:24]])),
    ]
    context_names = [
        "site",
        "boundary_d1",
        "boundary_d8",
        "boundary_d24",
        "boundary_d64",
        "boundary_d8_mean",
        "boundary_d8_std",
        "boundary_log_power_mean",
        "boundary_log_power_std",
        "boundary_valid_fraction",
        "boundary_pas_entropy_mean",
        "boundary_pas_entropy_std",
        "boundary_pdp_entropy_mean",
        "boundary_pdp_entropy_std",
    ]
    first8 = boundary_ids[:8]
    for view_name, matrix_bank in (
        ("pas", similarities.pas[target.site]),
        ("pdp", similarities.pdp[target.site]),
        ("ad", similarities.angle_delay[target.site]),
    ):
        local = similarities.local_index[first8]
        pair = matrix_bank[np.ix_(local, local)]
        values = pair[np.triu_indices(len(local), k=1)]
        context_scalars.extend(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.min(values)),
                float(np.max(values)),
            ]
        )
        context_names.extend(
            [
                f"boundary_{view_name}_pair_mean",
                f"boundary_{view_name}_pair_std",
                f"boundary_{view_name}_pair_min",
                f"boundary_{view_name}_pair_max",
            ]
        )
    append_block(
        blocks,
        names,
        "context",
        np.broadcast_to(
            np.asarray(context_scalars, dtype=np.float32)[None],
            (candidate_count, len(context_scalars)),
        ),
    )
    names[-len(context_names) :] = context_names

    feature = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    feature = np.nan_to_num(feature, nan=0.0, posinf=1e6, neginf=-1e6)
    return feature, names, {
        "clean_score": clean_score,
        "spatial_distance": spatial_distance,
    }


def candidate_labels(
    query_id: int,
    site: int,
    candidate_ids: np.ndarray,
    similarities: SimilarityBank,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pas, pdp, ad = similarities.truth_components(query_id, candidate_ids, site)
    label = LABEL_WEIGHTS[0] * pas + LABEL_WEIGHTS[1] * pdp + LABEL_WEIGHTS[2] * ad
    return label.astype(np.float32), {
        "pas": pas.astype(np.float32),
        "pdp": pdp.astype(np.float32),
        "angle_delay": ad.astype(np.float32),
    }


def training_candidate_indices(
    query_id: int,
    label: np.ndarray,
    clean_score: np.ndarray,
) -> np.ndarray:
    selected: list[int] = []

    def add(values: np.ndarray) -> None:
        for value in values:
            index = int(value)
            if index not in selected:
                selected.append(index)

    add(np.argsort(-label, kind="stable")[:8])
    add(np.argsort(clean_score, kind="stable")[:8])
    add(np.arange(min(8, len(label)), dtype=np.int64))
    rng = np.random.default_rng(SEED + int(query_id) * 1009)
    add(rng.permutation(len(label)))
    return np.asarray(selected[:TRAIN_CANDIDATE_COUNT], dtype=np.int64)


def build_training_data(
    train_ids: np.ndarray,
    specs: dict[int, QuerySpec],
    shuffled_context: dict[int, int],
    setup,
    positions: np.ndarray,
    geometry: np.ndarray,
    route: dict[str, np.ndarray],
    power: np.ndarray,
    pas_entropy: np.ndarray,
    pdp_entropy: np.ndarray,
    valid: np.ndarray,
    similarities: SimilarityBank,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    started = time.perf_counter()
    correct_rows: list[np.ndarray] = []
    shuffled_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    feature_names: list[str] | None = None
    oracle_label: list[float] = []
    clean_label: list[float] = []
    for row, query_id_value in enumerate(train_ids):
        query_id = int(query_id_value)
        target = specs[query_id]
        full_feature, names, auxiliary = pair_features(
            target,
            target,
            target.candidate_ids,
            setup,
            positions,
            geometry,
            route,
            power,
            pas_entropy,
            pdp_entropy,
            valid,
            similarities,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("feature-name drift during training construction")
        label, _ = candidate_labels(
            query_id, target.site, target.candidate_ids, similarities
        )
        selected = training_candidate_indices(
            query_id, label, auxiliary["clean_score"]
        )
        control_context = specs[shuffled_context[query_id]]
        control_feature, control_names, _ = pair_features(
            target,
            control_context,
            target.candidate_ids[selected],
            setup,
            positions,
            geometry,
            route,
            power,
            pas_entropy,
            pdp_entropy,
            valid,
            similarities,
        )
        if control_names != feature_names:
            raise RuntimeError("control feature-name drift")
        correct_rows.append(full_feature[selected])
        shuffled_rows.append(control_feature)
        targets.append(label[selected])
        oracle_label.append(float(np.max(label)))
        clean_label.append(
            float(label[int(np.argmin(auxiliary["clean_score"]))])
        )
        if (row + 1) % 250 == 0 or row + 1 == len(train_ids):
            print(
                f"E188 training features {row + 1}/{len(train_ids)}",
                flush=True,
            )
    if feature_names is None:
        raise RuntimeError("no training features were built")
    return (
        np.concatenate(correct_rows, axis=0),
        np.concatenate(shuffled_rows, axis=0),
        np.concatenate(targets, axis=0),
        feature_names,
        {
            "wall_seconds": time.perf_counter() - started,
            "query_count": int(len(train_ids)),
            "pair_count": int(len(train_ids) * TRAIN_CANDIDATE_COUNT),
            "feature_count": int(len(feature_names)),
            "mean_oracle_label": float(np.mean(oracle_label)),
            "mean_clean_formula_label": float(np.mean(clean_label)),
        },
    )


def rank_eval_queries(
    eval_ids: np.ndarray,
    specs: dict[int, QuerySpec],
    control_context: dict[int, int],
    correct_model: ExtraTreesRegressor,
    shuffled_model: ExtraTreesRegressor,
    feature_names: list[str],
    setup,
    positions: np.ndarray,
    geometry: np.ndarray,
    route: dict[str, np.ndarray],
    power: np.ndarray,
    pas_entropy: np.ndarray,
    pdp_entropy: np.ndarray,
    valid: np.ndarray,
    similarities: SimilarityBank,
) -> dict[str, object]:
    selected = np.empty((len(eval_ids), TOP_K), dtype=np.int32)
    selected_score = np.empty((len(eval_ids), TOP_K), dtype=np.float32)
    selected_label = np.empty((len(eval_ids), TOP_K), dtype=np.float32)
    control_selected = np.empty_like(selected)
    control_score = np.empty_like(selected_score)
    control_label = np.empty_like(selected_label)
    label_best = np.empty(len(eval_ids), dtype=np.int32)
    ndcg = np.empty(len(eval_ids), dtype=np.float32)
    top5_oracle = np.empty(len(eval_ids), dtype=bool)
    components: dict[str, np.ndarray] = {
        "pas": np.empty((len(eval_ids), TOP_K), dtype=np.float32),
        "pdp": np.empty((len(eval_ids), TOP_K), dtype=np.float32),
        "angle_delay": np.empty((len(eval_ids), TOP_K), dtype=np.float32),
    }
    control_components = {key: np.empty_like(value) for key, value in components.items()}

    for row, query_id_value in enumerate(eval_ids):
        query_id = int(query_id_value)
        target = specs[query_id]
        feature, names, _ = pair_features(
            target,
            target,
            target.candidate_ids,
            setup,
            positions,
            geometry,
            route,
            power,
            pas_entropy,
            pdp_entropy,
            valid,
            similarities,
        )
        if names != feature_names:
            raise RuntimeError("evaluation feature-name drift")
        score = correct_model.predict(feature)
        top = np.argsort(-score, kind="stable")[:TOP_K]
        label, truth = candidate_labels(
            query_id, target.site, target.candidate_ids, similarities
        )
        selected[row] = target.candidate_ids[top]
        selected_score[row] = score[top]
        selected_label[row] = label[top]
        for key in components:
            components[key][row] = truth[key][top]

        control_spec = specs[control_context[query_id]]
        control_feature, control_names, _ = pair_features(
            target,
            control_spec,
            target.candidate_ids,
            setup,
            positions,
            geometry,
            route,
            power,
            pas_entropy,
            pdp_entropy,
            valid,
            similarities,
        )
        if control_names != feature_names:
            raise RuntimeError("evaluation control feature-name drift")
        shuffled_score = shuffled_model.predict(control_feature)
        shuffled_top = np.argsort(-shuffled_score, kind="stable")[:TOP_K]
        control_selected[row] = target.candidate_ids[shuffled_top]
        control_score[row] = shuffled_score[shuffled_top]
        control_label[row] = label[shuffled_top]
        for key in control_components:
            control_components[key][row] = truth[key][shuffled_top]

        best = int(np.argmax(label))
        label_best[row] = int(target.candidate_ids[best])
        top5_oracle[row] = best in set(int(value) for value in top)
        ideal = np.sort(label)[::-1][:TOP_K]
        observed = label[top]
        discount = 1.0 / np.log2(np.arange(2, TOP_K + 2))
        ideal_dcg = float(np.sum(ideal * discount))
        ndcg[row] = float(np.sum(observed * discount) / max(ideal_dcg, TINY))
    return {
        "selected_ids": selected,
        "selected_score": selected_score,
        "selected_label": selected_label,
        "components": components,
        "control_selected_ids": control_selected,
        "control_score": control_score,
        "control_label": control_label,
        "control_components": control_components,
        "label_best_ids": label_best,
        "top5_oracle_recall": float(np.mean(top5_oracle)),
        "ndcg_at_5": float(np.mean(ndcg)),
    }


@torch.inference_mode()
def official_selected_statistics(
    query_ids: np.ndarray,
    selected_ids: np.ndarray,
    train_channel: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> PointMetricStatistics:
    pas: list[np.ndarray] = []
    pdp: list[np.ndarray] = []
    error: list[np.ndarray] = []
    energy: list[np.ndarray] = []
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        prediction = torch.as_tensor(
            np.array(train_channel[selected_ids[start:end]], dtype=np.complex64),
            device=device,
        )
        truth = torch.as_tensor(
            np.array(train_channel[query_ids[start:end]], dtype=np.complex64),
            device=device,
        )
        values = point_metric_statistics_torch(prediction, truth)
        pas.append(values.pas)
        pdp.append(values.pdp)
        error.append(values.squared_error)
        energy.append(values.truth_energy)
    return PointMetricStatistics(
        pas=np.concatenate(pas),
        pdp=np.concatenate(pdp),
        squared_error=np.concatenate(error),
        truth_energy=np.concatenate(energy),
    )


def circular_delta(destination: np.ndarray, source: np.ndarray, period: float) -> np.ndarray:
    return (destination - source + period / 2.0) % period - period / 2.0


def component_common_mass(
    left_coordinate: np.ndarray,
    left_energy: np.ndarray,
    right_coordinate: np.ndarray,
    right_energy: np.ndarray,
) -> float:
    left_total = float(np.sum(left_energy))
    right_total = float(np.sum(right_energy))
    if left_total <= TINY or right_total <= TINY:
        return float(left_total <= TINY and right_total <= TINY)
    delta_h = circular_delta(
        right_coordinate[None, :, 0], left_coordinate[:, None, 0], 16.0
    )
    delta_v = circular_delta(
        right_coordinate[None, :, 1], left_coordinate[:, None, 1], 8.0
    )
    delta_d = circular_delta(
        right_coordinate[None, :, 2], left_coordinate[:, None, 2], 192.0
    )
    cost = (delta_h / 1.5) ** 2 + (delta_v / 1.0) ** 2 + (delta_d / 2.0) ** 2
    rows, columns = linear_sum_assignment(cost)
    accepted = (
        (np.abs(delta_h[rows, columns]) <= 2.0)
        & (np.abs(delta_v[rows, columns]) <= 1.5)
        & (np.abs(delta_d[rows, columns]) <= 4.0)
        & (cost[rows, columns] <= 8.0)
    )
    left_share = left_energy / max(left_total, TINY)
    right_share = right_energy / max(right_total, TINY)
    return float(
        np.sum(
            np.minimum(
                left_share[rows[accepted]], right_share[columns[accepted]]
            )
        )
    )


def path_common_values(
    query_ids: np.ndarray,
    selected_ids: np.ndarray,
    coordinates: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(query_ids), dtype=np.float32)
    for row, (query, selected) in enumerate(zip(query_ids, selected_ids)):
        output[row] = component_common_mass(
            coordinates[int(query)],
            energy[int(query)],
            coordinates[int(selected)],
            energy[int(selected)],
        )
    return output


def load_e184_selection(
    eval_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metrics = np.load(E184_METRICS, allow_pickle=False)
    query_ids = metrics["query_ids"].astype(np.int64)
    lookup = {int(value): row for row, value in enumerate(query_ids)}
    rows = np.asarray([lookup[int(value)] for value in eval_ids], dtype=np.int64)
    methods = metrics["method_names"].astype(str).tolist()
    selected = metrics["selected_ids"].astype(np.int64)
    output = {
        "e184_clean": selected[methods.index("clean_full"), rows, 0],
        "e184_oracle_local256": selected[
            methods.index("oracle_spectral_local256"), rows, 0
        ],
    }
    return output, {
        "rows": rows,
        "methods": methods,
        "truth_nonzero": metrics["truth_nonzero"][rows].astype(bool),
    }


def method_summary(
    name: str,
    query_ids: np.ndarray,
    selected_ids: np.ndarray,
    official: PointMetricStatistics,
    similarities: SimilarityBank,
    train_site: np.ndarray,
    component_coordinates: np.ndarray,
    component_energy: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pas = np.empty(len(query_ids), dtype=np.float32)
    pdp = np.empty(len(query_ids), dtype=np.float32)
    ad = np.empty(len(query_ids), dtype=np.float32)
    for site in (0, 1):
        rows = np.flatnonzero(train_site[query_ids] == site)
        if len(rows) == 0:
            continue
        local_pas, local_pdp, local_ad = similarities.truth_components(
            int(query_ids[rows[0]]), selected_ids[rows[:1]], site
        )
        del local_pas, local_pdp, local_ad
        query_local = similarities.local_index[query_ids[rows]]
        selected_local = similarities.local_index[selected_ids[rows]]
        pas[rows] = similarities.pas[site][query_local, selected_local]
        pdp[rows] = similarities.pdp[site][query_local, selected_local]
        ad[rows] = similarities.angle_delay[site][query_local, selected_local]
    path_common = path_common_values(
        query_ids,
        selected_ids,
        component_coordinates,
        component_energy,
    )
    aggregate = aggregate_statistics(official)
    row = {
        "method": name,
        "query_count": int(len(query_ids)),
        "compact_pas_top1": float(np.mean(pas)),
        "compact_pdp_top1": float(np.mean(pdp)),
        "compact_joint_top1": float(np.mean(0.5 * (pas + pdp))),
        "angle_delay_top1": float(np.mean(ad)),
        "path_common_mass_top1": float(np.mean(path_common)),
        "official_pas_top1": float(np.mean(official.pas)),
        "official_pdp_top1": float(np.mean(official.pdp)),
        "official_joint_top1": float(np.mean(0.5 * (official.pas + official.pdp))),
        "direct_copy_nmse": float(aggregate["nmse"]),
        "direct_copy_score": float(aggregate["score"]),
    }
    return row, pas, pdp, ad, path_common


def fit_model(features: np.ndarray, target: np.ndarray, seed: int) -> ExtraTreesRegressor:
    sklearn_seed = int(seed % (2**32 - 1))
    model = ExtraTreesRegressor(
        n_estimators=TREE_COUNT,
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        max_features=TREE_MAX_FEATURES,
        n_jobs=-1,
        random_state=sklearn_seed,
    )
    model.fit(features, target)
    return model


def device_report(device: torch.device) -> dict[str, object]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(device),
        "cuda": True,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def run_outer(
    outer: int,
    eval_site: int | None,
    setup,
    positions: np.ndarray,
    train_site: np.ndarray,
    query_ids: np.ndarray,
    canonical_fold: np.ndarray,
    broad_fold: np.ndarray,
    folds,
    geometry: np.ndarray,
    route: dict[str, np.ndarray],
    descriptors,
    similarities: SimilarityBank,
    train_channel: np.ndarray,
    component_coordinates: np.ndarray,
    component_energy: np.ndarray,
    device: torch.device,
    metric_batch_size: int,
) -> dict[str, object]:
    outer_started = time.perf_counter()
    fit_ids = folds[f"fold{outer}_fit"].astype(np.int64)
    outer_forbidden = np.concatenate(
        [
            folds[f"fold{outer}_query"].astype(np.int64),
            folds[f"fold{outer}_ignored"].astype(np.int64),
        ]
    )
    valid = descriptors["valid"].astype(bool)
    train_ids = fit_ids[(broad_fold[fit_ids] != outer) & valid[fit_ids]]
    eval_mask = canonical_fold == outer
    if eval_site is not None:
        eval_mask &= train_site[query_ids] == eval_site
    eval_ids = query_ids[eval_mask]
    eval_ids = eval_ids[valid[eval_ids]]
    if len(eval_ids) == 0:
        raise RuntimeError(f"outer {outer} has no nonzero evaluation queries")

    train_specs = make_query_specs(
        train_ids,
        fit_ids,
        positions,
        train_site,
        pseudo_hole=True,
        outer_forbidden=outer_forbidden,
    )
    eval_specs = make_query_specs(
        eval_ids,
        fit_ids,
        positions,
        train_site,
        pseudo_hole=False,
        outer_forbidden=outer_forbidden,
    )
    train_context_map = derangement(
        train_ids, train_site, SEED + outer * 101
    )
    eval_context_map = derangement(
        eval_ids, train_site, SEED + outer * 101 + 1
    )

    correct_x, shuffled_x, target, feature_names, training_report = (
        build_training_data(
            train_ids,
            train_specs,
            train_context_map,
            setup,
            positions,
            geometry,
            route,
            descriptors["total_power"].astype(np.float64),
            descriptors["pas_entropy"].astype(np.float32),
            descriptors["pdp_entropy"].astype(np.float32),
            valid,
            similarities,
        )
    )
    fit_started = time.perf_counter()
    correct_model = fit_model(correct_x, target, SEED + outer * 1009)
    correct_fit_seconds = time.perf_counter() - fit_started
    fit_started = time.perf_counter()
    shuffled_model = fit_model(shuffled_x, target, SEED + outer * 1009 + 1)
    shuffled_fit_seconds = time.perf_counter() - fit_started
    ranking = rank_eval_queries(
        eval_ids,
        eval_specs,
        eval_context_map,
        correct_model,
        shuffled_model,
        feature_names,
        setup,
        positions,
        geometry,
        route,
        descriptors["total_power"].astype(np.float64),
        descriptors["pas_entropy"].astype(np.float32),
        descriptors["pdp_entropy"].astype(np.float32),
        valid,
        similarities,
    )
    e184_selected, e184_meta = load_e184_selection(eval_ids)
    if not bool(np.all(e184_meta["truth_nonzero"])):
        raise RuntimeError("nonzero screen unexpectedly includes a zero target")
    method_ids = {
        "learned_ranker": np.asarray(ranking["selected_ids"])[:, 0],
        "shuffled_context_control": np.asarray(
            ranking["control_selected_ids"]
        )[:, 0],
        **e184_selected,
    }
    method_rows: list[dict[str, object]] = []
    method_statistics: dict[str, PointMetricStatistics] = {}
    method_point_values: dict[str, dict[str, np.ndarray]] = {}
    for name, selected_ids in method_ids.items():
        official = official_selected_statistics(
            eval_ids,
            np.asarray(selected_ids, dtype=np.int64),
            train_channel,
            device,
            metric_batch_size,
        )
        summary, pas, pdp, ad, path_common = method_summary(
            name,
            eval_ids,
            np.asarray(selected_ids, dtype=np.int64),
            official,
            similarities,
            train_site,
            component_coordinates,
            component_energy,
        )
        method_rows.append(summary)
        method_statistics[name] = official
        method_point_values[name] = {
            "compact_pas": pas,
            "compact_pdp": pdp,
            "angle_delay": ad,
            "path_common": path_common,
        }
        print(
            f"E188 outer={outer} evaluated {name} "
            f"joint={summary['official_joint_top1']:.6f}",
            flush=True,
        )

    summary_by_name = {str(row["method"]): row for row in method_rows}
    learned_joint = float(summary_by_name["learned_ranker"]["official_joint_top1"])
    shuffled_joint = float(
        summary_by_name["shuffled_context_control"]["official_joint_top1"]
    )
    clean_joint = float(summary_by_name["e184_clean"]["official_joint_top1"])
    oracle_joint = float(
        summary_by_name["e184_oracle_local256"]["official_joint_top1"]
    )
    gates = {
        "oracle_capacity_required": 0.05,
        "oracle_capacity": oracle_joint - clean_joint,
        "oracle_capacity_passed": oracle_joint - clean_joint >= 0.05,
        "learned_not_below_e184_clean": learned_joint >= clean_joint,
        "shuffle_margin_required": 0.05,
        "shuffle_margin": learned_joint - shuffled_joint,
        "shuffle_margin_passed": learned_joint - shuffled_joint >= 0.05,
    }
    gates["screen_passed"] = bool(
        gates["oracle_capacity_passed"]
        and gates["learned_not_below_e184_clean"]
        and gates["shuffle_margin_passed"]
    )

    point_rows: list[dict[str, object]] = []
    for row, query_id in enumerate(eval_ids):
        item: dict[str, object] = {
            "query_id": int(query_id),
            "query_no": int(query_id) + 1,
            "site": int(train_site[query_id]),
            "outer": outer,
            "x": float(positions[query_id, 0]),
            "y": float(positions[query_id, 1]),
            "learned_top5_ids": " ".join(
                str(int(value)) for value in np.asarray(ranking["selected_ids"])[row]
            ),
            "learned_top5_nos": " ".join(
                str(int(value) + 1)
                for value in np.asarray(ranking["selected_ids"])[row]
            ),
            "learned_top5_scores": " ".join(
                f"{float(value):.7f}"
                for value in np.asarray(ranking["selected_score"])[row]
            ),
            "learned_top5_labels": " ".join(
                f"{float(value):.7f}"
                for value in np.asarray(ranking["selected_label"])[row]
            ),
            "label_oracle_id": int(np.asarray(ranking["label_best_ids"])[row]),
            "e184_clean_id": int(method_ids["e184_clean"][row]),
            "e184_oracle_id": int(method_ids["e184_oracle_local256"][row]),
        }
        for name in method_ids:
            stats = method_statistics[name]
            values = method_point_values[name]
            item[f"{name}_selected_id"] = int(method_ids[name][row])
            item[f"{name}_official_pas"] = float(stats.pas[row])
            item[f"{name}_official_pdp"] = float(stats.pdp[row])
            item[f"{name}_official_joint"] = float(
                0.5 * (stats.pas[row] + stats.pdp[row])
            )
            item[f"{name}_angle_delay"] = float(values["angle_delay"][row])
            item[f"{name}_path_common"] = float(values["path_common"][row])
        point_rows.append(item)

    return {
        "outer": outer,
        "eval_site": eval_site,
        "train_query_count": int(len(train_ids)),
        "eval_query_count": int(len(eval_ids)),
        "feature_names": feature_names,
        "training": training_report,
        "fit_seconds": {
            "correct": correct_fit_seconds,
            "shuffled": shuffled_fit_seconds,
        },
        "ranking": {
            "top5_oracle_recall": ranking["top5_oracle_recall"],
            "ndcg_at_5": ranking["ndcg_at_5"],
        },
        "method_rows": method_rows,
        "gates": gates,
        "point_rows": point_rows,
        "prediction_arrays": {
            "query_ids": eval_ids.astype(np.int32),
            "learned_selected_ids": np.asarray(ranking["selected_ids"]).astype(
                np.int32
            ),
            "learned_selected_scores": np.asarray(ranking["selected_score"]).astype(
                np.float32
            ),
            "learned_selected_labels": np.asarray(ranking["selected_label"]).astype(
                np.float32
            ),
            "control_selected_ids": np.asarray(
                ranking["control_selected_ids"]
            ).astype(np.int32),
            "label_best_ids": np.asarray(ranking["label_best_ids"]).astype(np.int32),
            "e184_clean_ids": method_ids["e184_clean"].astype(np.int32),
            "e184_oracle_ids": method_ids["e184_oracle_local256"].astype(np.int32),
        },
        "wall_seconds": time.perf_counter() - outer_started,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E188 requires one CUDA GPU for official angle-delay matrices")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    torch.backends.cuda.matmul.allow_tf32 = False

    required = (
        TRAIN_CHANNEL,
        FOLD_CACHE,
        OOF_META,
        BROAD_FOLD,
        DESCRIPTOR_PATH,
        PAS_PATH,
        AD_PATH,
        GEOMETRY_PATH,
        ROUTE_PATH,
        COMPONENT_PATH,
        E184_METRICS,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    setup, positions, _, train_site, _ = load_positions()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    broad_fold = np.load(BROAD_FOLD, allow_pickle=False).astype(np.int8)
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    descriptors = np.load(DESCRIPTOR_PATH, allow_pickle=False)
    compact_pas = np.load(PAS_PATH, mmap_mode="r")
    angle_delay = np.load(AD_PATH, mmap_mode="r")
    geometry_cache = np.load(GEOMETRY_PATH, allow_pickle=False)
    geometry = geometry_cache["train"].astype(np.float32)
    route = e11.route_signature(np.load(ROUTE_PATH, allow_pickle=False), "train")
    components = np.load(COMPONENT_PATH, allow_pickle=False)
    component_coordinates = components["coordinates"].astype(np.float32)
    component_energy = np.sum(
        np.abs(components["coefficients"].astype(np.complex64)) ** 2,
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    if not (
        len(positions)
        == len(train_site)
        == len(broad_fold)
        == len(compact_pas)
        == len(angle_delay)
        == len(geometry)
        == 4000
    ):
        raise RuntimeError("E188 official Train identity mismatch")
    if len(query_ids) != 500 or len(np.unique(query_ids)) != 500:
        raise RuntimeError("E188 canonical fixed500 identity mismatch")

    similarities, similarity_timing = build_similarity_bank(
        train_site,
        compact_pas,
        descriptors["global_pdp"].astype(np.float32),
        angle_delay,
        device,
        args.matrix_batch_size,
    )
    outer_specs = [(0, 0)] if args.mode == "screen" else [(outer, None) for outer in range(5)]
    outer_results: list[dict[str, object]] = []
    for outer, eval_site in outer_specs:
        outer_results.append(
            run_outer(
                outer,
                eval_site,
                setup,
                positions,
                train_site,
                query_ids,
                canonical_fold,
                broad_fold,
                folds,
                geometry,
                route,
                descriptors,
                similarities,
                train_channel,
                component_coordinates,
                component_energy,
                device,
                args.metric_batch_size,
            )
        )

    method_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for result in outer_results:
        method_rows.extend(result.pop("method_rows"))
        point_rows.extend(result.pop("point_rows"))
        prediction_arrays = result.pop("prediction_arrays")
        prefix = f"outer{int(result['outer'])}"
        for key, value in prediction_arrays.items():
            arrays[f"{prefix}_{key}"] = value
    prediction_path = ARTIFACT_DIR / f"{args.mode}_predictions.npz"
    np.savez_compressed(prediction_path, **arrays)
    method_path = ANALYSIS_DIR / f"{args.mode}_method_summary.csv"
    point_path = ANALYSIS_DIR / f"{args.mode}_point_rankings.csv"
    write_csv(method_path, method_rows)
    write_csv(point_path, point_rows)

    wall_seconds = time.perf_counter() - started
    report = {
        "experiment": VERSION,
        "status": "screen_complete" if args.mode == "screen" else "diagnostic_complete",
        "mode": args.mode,
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v14_used": False,
        "e185_e186_used": False,
        "protocol": {
            "canonical_evaluation": "complete spatial holes",
            "training_query": "official Train point with itself and nearest 8 removed",
            "candidate_pool": "nearest 256 same-site points in the outer legal fit set",
            "query_truth_usage": "training label and post-ranking evaluation only",
            "zero_query_policy": "reported separately; excluded from ordinary-state ranker",
        },
        "label": {
            "compact_pas_weight": LABEL_WEIGHTS[0],
            "global_pdp_weight": LABEL_WEIGHTS[1],
            "angle_delay_weight": LABEL_WEIGHTS[2],
        },
        "model": {
            "class": "ExtraTreesRegressor",
            "n_estimators": TREE_COUNT,
            "max_depth": TREE_MAX_DEPTH,
            "min_samples_leaf": TREE_MIN_SAMPLES_LEAF,
            "max_features": TREE_MAX_FEATURES,
            "train_candidates_per_query": TRAIN_CANDIDATE_COUNT,
        },
        "similarity_timing": similarity_timing,
        "outer_results": outer_results,
        "outputs": {
            "predictions": PATHS.relative(prediction_path),
            "method_summary": PATHS.relative(method_path),
            "point_rankings": PATHS.relative(point_path),
        },
        "performance": {
            "wall_seconds": wall_seconds,
            "device": device_report(device),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024.0 * 1024.0),
            "peak_torch_memory_gib": torch.cuda.max_memory_allocated(device)
            / (1024.0**3),
            "measured_bottleneck": "CPU pair-feature construction and ExtraTrees fitting",
        },
    }
    report_path = ANALYSIS_DIR / f"{args.mode}_report.json"
    report_path.write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_value(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
