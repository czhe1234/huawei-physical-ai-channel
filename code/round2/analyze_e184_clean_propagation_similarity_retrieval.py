#!/usr/bin/env python3
"""Audit whether Clean observables retrieve propagation-similar train points."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

import analyze_v4_route_topology_retrieval as e11
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
)
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E184-clean-propagation-similarity-retrieval"
ARTIFACT_DIR = (
    PATHS.artifacts / "round2" / "e184_clean_propagation_similarity_retrieval"
)
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e184_clean_propagation_similarity_retrieval"
)
RANKING_PATH = ARTIFACT_DIR / "deployment_visible_rankings.npz"
METRIC_PATH = ARTIFACT_DIR / "retrieval_metrics.npz"
REPORT_PATH = ANALYSIS_DIR / "clean_propagation_similarity_retrieval.json"
METHOD_CSV = ANALYSIS_DIR / "method_summary.csv"
HOLE_CSV = ANALYSIS_DIR / "per_hole_summary.csv"
POINT_CSV = ANALYSIS_DIR / "point_top1_retrieval.csv"
ORACLE_CSV = ANALYSIS_DIR / "oracle_spatial_rank_recall.csv"

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
V7_COMPLETE = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "canonical_v7_complete.npy"
)
TRAIN_DESCRIPTORS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_spectral_descriptors.npz"
)
TRAIN_COMPACT_PAS = (
    PATHS.artifacts
    / "round2"
    / "e169_hard_hole_pas_peak_precursors"
    / "official_train_compact_pas.npy"
)
TRAIN_AD = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_ad_support_f16.npy"
)
ROUTE_CACHE = (
    PATHS.artifacts
    / "round2"
    / "route_topology"
    / "round2_route_topology_features.npz"
)
COMPONENT_CACHE = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)

DEPLOYMENT_METHODS = (
    "nearest",
    "geometry",
    "route",
    "spectral",
    "clean_full",
    "clean_local256",
    "clean_shuffled",
)
ORACLE_METHODS = (
    "oracle_spectral_full",
    "oracle_spectral_local256",
    "oracle_ad_full",
    "oracle_ad_local256",
)
ALL_METHODS = DEPLOYMENT_METHODS + ORACLE_METHODS
TOP_K = 8
LOCAL_POOL = 256
TINY = 1e-30
SEED = 20_260_812_184


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channel-load-batch-size", type=int, default=128)
    parser.add_argument("--metric-batch-size", type=int, default=128)
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


def normalize_l2(value: np.ndarray, axis: int) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    norm = np.sqrt(np.sum(array * array, axis=axis, keepdims=True, dtype=np.float64))
    return np.divide(
        array,
        np.maximum(norm, TINY),
        out=np.zeros_like(array),
        where=norm > 0.0,
    )


def grouped_cosine_matrix_numpy(
    query: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    left = np.asarray(query, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]:
        raise ValueError(f"grouped cosine mismatch: {left.shape} vs {right.shape}")
    left_norm = normalize_l2(left, 2)
    right_norm = normalize_l2(right, 2)
    similarity = np.einsum("qgd,kgd->qkg", left_norm, right_norm, optimize=True)
    left_zero = np.sum(left * left, axis=2) <= TINY
    right_zero = np.sum(right * right, axis=2) <= TINY
    both_zero = left_zero[:, None, :] & right_zero[None, :, :]
    similarity = np.where(both_zero, 1.0, similarity)
    return np.mean(similarity, axis=2, dtype=np.float64).astype(np.float32)


@torch.inference_mode()
def grouped_cosine_matrix(
    query: np.ndarray,
    candidate: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    left = torch.as_tensor(np.asarray(query, dtype=np.float32), device=device)
    right = torch.as_tensor(np.asarray(candidate, dtype=np.float32), device=device)
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]:
        raise ValueError(
            f"grouped cosine mismatch: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    left = left.clamp_min(0.0)
    right = right.clamp_min(0.0)
    left_energy = (left * left).sum(dim=2, dtype=torch.float64)
    right_energy = (right * right).sum(dim=2, dtype=torch.float64)
    left_norm = (
        left / torch.sqrt(left_energy).clamp_min(TINY).unsqueeze(2)
    ).to(torch.float32)
    right_norm = (
        right / torch.sqrt(right_energy).clamp_min(TINY).unsqueeze(2)
    ).to(torch.float32)
    similarity = torch.einsum("qgd,kgd->qkg", left_norm, right_norm)
    both_zero = (left_energy[:, None] <= TINY) & (right_energy[None] <= TINY)
    similarity = torch.where(both_zero, torch.ones_like(similarity), similarity)
    return similarity.mean(dim=2, dtype=torch.float64).to(torch.float32).cpu().numpy()


def validate_grouped_cosine(device: torch.device) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    query = rng.random((7, 4, 128), dtype=np.float32)
    candidate = rng.random((13, 4, 128), dtype=np.float32)
    query[0, 0] = 0.0
    candidate[0, 0] = 0.0
    reference = grouped_cosine_matrix_numpy(query, candidate)
    accelerated = grouped_cosine_matrix(query, candidate, device)
    max_abs = float(np.max(np.abs(reference - accelerated)))
    top4_equal = bool(
        np.array_equal(
            np.argsort(-reference, axis=1, kind="stable")[:, :4],
            np.argsort(-accelerated, axis=1, kind="stable")[:, :4],
        )
    )
    if max_abs > 2e-7 or not top4_equal:
        raise RuntimeError(
            f"GPU grouped cosine identity failed: max_abs={max_abs:.3e} "
            f"top4_equal={top4_equal}"
        )
    return {"max_abs": max_abs, "top4_equal": top4_equal}


def grouped_cosine_pairs(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    left = np.asarray(query, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError(f"pair cosine mismatch: {left.shape} vs {right.shape}")
    left_norm = normalize_l2(left, 2)
    right_norm = normalize_l2(right, 2)
    similarity = np.sum(left_norm * right_norm, axis=2, dtype=np.float64)
    both_zero = (np.sum(left * left, axis=2) <= TINY) & (
        np.sum(right * right, axis=2) <= TINY
    )
    similarity = np.where(both_zero, 1.0, similarity)
    return np.mean(similarity, axis=1, dtype=np.float64).astype(np.float32)


def row_percentile(distance: np.ndarray) -> np.ndarray:
    value = np.asarray(distance, dtype=np.float64)
    order = np.argsort(value, axis=1, kind="stable")
    rank = np.empty_like(order, dtype=np.float32)
    positions = np.arange(value.shape[1], dtype=np.float32)
    rank[np.arange(len(value))[:, None], order] = positions[None]
    return rank / max(value.shape[1] - 1, 1)


def top_ids(score: np.ndarray, candidate_ids: np.ndarray, count: int = TOP_K) -> np.ndarray:
    take = min(count, score.shape[1])
    local = np.argpartition(score, take - 1, axis=1)[:, :take]
    local_score = np.take_along_axis(score, local, axis=1)
    order = np.argsort(local_score, axis=1, kind="stable")
    local = np.take_along_axis(local, order, axis=1)
    ids = np.broadcast_to(np.asarray(candidate_ids, dtype=np.int64)[None], score.shape)
    return np.take_along_axis(ids, local, axis=1)


def geometry_distance(
    query_positions: np.ndarray,
    candidate_positions: np.ndarray,
    bs_xy: np.ndarray,
) -> np.ndarray:
    query_delta = np.asarray(query_positions, dtype=np.float64) - bs_xy[None]
    candidate_delta = np.asarray(candidate_positions, dtype=np.float64) - bs_xy[None]

    def features(delta: np.ndarray) -> np.ndarray:
        radius = np.linalg.norm(delta, axis=1)
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        return np.column_stack(
            [delta[:, 0], delta[:, 1], radius, np.cos(angle), np.sin(angle)]
        )

    candidate_feature = features(candidate_delta)
    query_feature = features(query_delta)
    mean = np.mean(candidate_feature, axis=0)
    scale = np.maximum(np.std(candidate_feature, axis=0), 1e-4)
    candidate_feature = (candidate_feature - mean) / scale
    query_feature = (query_feature - mean) / scale
    difference = query_feature[:, None] - candidate_feature[None]
    return np.sqrt(np.mean(difference * difference, axis=2)).astype(np.float32)


@torch.inference_mode()
def v7_visible_spectra(
    channel: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    compact_pas = np.empty((len(channel), 4, 16, 8), dtype=np.float32)
    global_pdp = np.empty((len(channel), 192), dtype=np.float32)
    io_seconds = 0.0
    gpu_seconds = 0.0
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        io_started = time.perf_counter()
        host = np.array(channel[start:end], dtype=np.complex64, copy=True)
        io_seconds += time.perf_counter() - io_started
        gpu_started = time.perf_counter()
        value = torch.as_tensor(host, device=device)
        shaped = value.reshape(len(value), 2, 16, 8, 4, 192)
        angle = torch.fft.fft2(shaped, dim=(2, 3)).abs().square()
        angle = angle.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2)
        angle_total = angle.sum(dim=(2, 3), keepdim=True)
        angle = torch.where(
            angle_total > 0.0,
            angle / angle_total.clamp_min(TINY),
            torch.zeros_like(angle),
        )
        delay = torch.fft.ifft(shaped, dim=-1).abs().square().mean(dim=(1, 2, 3, 4))
        delay_total = delay.sum(dim=1, keepdim=True)
        delay = torch.where(
            delay_total > 0.0,
            delay / delay_total.clamp_min(TINY),
            torch.zeros_like(delay),
        )
        compact_pas[start:end] = angle.float().cpu().numpy()
        global_pdp[start:end] = delay.float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gpu_seconds += time.perf_counter() - gpu_started
    return compact_pas, global_pdp, {
        "io_seconds": io_seconds,
        "gpu_seconds": gpu_seconds,
    }


def build_deployment_rankings(
    setup,
    positions: np.ndarray,
    train_site: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    query_pas: np.ndarray,
    query_pdp: np.ndarray,
    train_pas: np.ndarray,
    train_pdp: np.ndarray,
    route: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, float]]:
    started = time.perf_counter()
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    selected = {
        name: np.full((len(query_ids), TOP_K), -1, dtype=np.int32)
        for name in DEPLOYMENT_METHODS
    }
    pools: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in query_ids]
    spatial_orders: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in query_ids]
    rng = np.random.default_rng(SEED)

    query_site = train_site[query_ids]
    for outer in range(5):
        outer_rows = np.flatnonzero(fold_id == outer)
        fit_all = folds[f"fold{outer}_fit"].astype(np.int64)
        for site_value in (0, 1):
            rows = outer_rows[query_site[outer_rows] == site_value]
            if len(rows) == 0:
                continue
            targets = query_ids[rows]
            fit = fit_all[train_site[fit_all] == site_value]
            candidate_ids = np.broadcast_to(fit[None], (len(rows), len(fit)))
            spatial = np.linalg.norm(
                positions[targets, None, :2] - positions[fit, :2][None], axis=2
            )
            spatial_order = np.argsort(spatial, axis=1, kind="stable")
            local_count = min(LOCAL_POOL, len(fit))
            local_mask = np.zeros_like(spatial, dtype=bool)
            local_mask[
                np.arange(len(rows))[:, None], spatial_order[:, :local_count]
            ] = True

            candidate_pas = np.asarray(train_pas[fit], dtype=np.float32).reshape(
                len(fit), 4, -1
            )
            candidate_pdp = np.asarray(train_pdp[fit], dtype=np.float32).reshape(
                len(fit), 1, -1
            )
            pas_distance = 1.0 - grouped_cosine_matrix(
                query_pas[rows].reshape(len(rows), 4, -1),
                candidate_pas,
                device,
            )
            pdp_distance = 1.0 - grouped_cosine_matrix(
                query_pdp[rows].reshape(len(rows), 1, -1),
                candidate_pdp,
                device,
            )
            route_distance = e11.route_distance(route, targets, candidate_ids)
            geometry = geometry_distance(
                positions[targets, :2],
                positions[fit, :2],
                setup.bs_positions[site_value, :2],
            )

            pas_rank = row_percentile(pas_distance)
            pdp_rank = row_percentile(pdp_distance)
            route_rank = row_percentile(route_distance)
            geometry_rank = row_percentile(geometry)
            spectral_score = 0.5 * (pas_rank + pdp_rank)
            clean_score = 0.25 * (
                pas_rank + pdp_rank + route_rank + geometry_rank
            )

            permutation = rng.permutation(len(fit))
            shuffled_pas_distance = pas_distance[:, permutation]
            shuffled_pdp_distance = pdp_distance[:, permutation]
            shuffled_route_distance = route_distance[:, permutation]
            shuffled_score = 0.25 * (
                row_percentile(shuffled_pas_distance)
                + row_percentile(shuffled_pdp_distance)
                + row_percentile(shuffled_route_distance)
                + geometry_rank
            )

            score_by_method = {
                "nearest": spatial,
                "geometry": geometry,
                "route": route_distance,
                "spectral": spectral_score,
                "clean_full": clean_score,
                "clean_local256": np.where(local_mask, clean_score, np.inf),
                "clean_shuffled": shuffled_score,
            }
            for name, score in score_by_method.items():
                selected[name][rows] = top_ids(score, fit)
            for local_row, output_row in enumerate(rows):
                pools[output_row] = fit.copy()
                spatial_orders[output_row] = fit[spatial_order[local_row]].copy()
            print(
                f"E184 ranking outer={outer} site={site_value} "
                f"queries={len(rows)} candidates={len(fit)}",
                flush=True,
            )

    if any(np.any(value < 0) for value in selected.values()):
        raise RuntimeError("deployment ranking did not cover every query")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RANKING_PATH,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id.astype(np.int8),
        method_names=np.asarray(DEPLOYMENT_METHODS),
        selected_ids=np.stack([selected[name] for name in DEPLOYMENT_METHODS]),
        candidate_counts=np.asarray([len(value) for value in pools], dtype=np.int16),
    )
    return selected, pools, spatial_orders, {
        "wall_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def build_oracle_rankings(
    selected: dict[str, np.ndarray],
    pools: list[np.ndarray],
    spatial_orders: list[np.ndarray],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    train_site: np.ndarray,
    truth_pas: np.ndarray,
    truth_pdp: np.ndarray,
    train_pas: np.ndarray,
    train_pdp: np.ndarray,
    train_ad: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    started = time.perf_counter()
    for name in ORACLE_METHODS:
        selected[name] = np.full((len(query_ids), TOP_K), -1, dtype=np.int32)
    query_site = train_site[query_ids]

    for outer in range(5):
        outer_rows = np.flatnonzero(fold_id == outer)
        for site_value in (0, 1):
            rows = outer_rows[query_site[outer_rows] == site_value]
            if len(rows) == 0:
                continue
            fit = pools[rows[0]]
            if any(not np.array_equal(pools[row], fit) for row in rows):
                raise RuntimeError("candidate pool drift inside fold/site group")
            candidate_pas = np.asarray(train_pas[fit], dtype=np.float32).reshape(
                len(fit), 4, -1
            )
            candidate_pdp = np.asarray(train_pdp[fit], dtype=np.float32).reshape(
                len(fit), 1, -1
            )
            pas_similarity = grouped_cosine_matrix(
                truth_pas[rows].reshape(len(rows), 4, -1),
                candidate_pas,
                device,
            )
            pdp_similarity = grouped_cosine_matrix(
                truth_pdp[rows].reshape(len(rows), 1, -1),
                candidate_pdp,
                device,
            )
            spectral_score = -0.5 * (pas_similarity + pdp_similarity)
            selected["oracle_spectral_full"][rows] = top_ids(spectral_score, fit)

            local_ids = np.stack([spatial_orders[row][:LOCAL_POOL] for row in rows])
            local_lookup = {int(value): index for index, value in enumerate(fit)}
            local_columns = np.asarray(
                [[local_lookup[int(value)] for value in values] for values in local_ids],
                dtype=np.int64,
            )
            local_spectral = np.take_along_axis(spectral_score, local_columns, axis=1)
            local_selected = np.argpartition(local_spectral, TOP_K - 1, axis=1)[
                :, :TOP_K
            ]
            local_value = np.take_along_axis(local_spectral, local_selected, axis=1)
            local_order = np.argsort(local_value, axis=1, kind="stable")
            local_selected = np.take_along_axis(local_selected, local_order, axis=1)
            selected["oracle_spectral_local256"][rows] = np.take_along_axis(
                local_ids, local_selected, axis=1
            )

            query_ad = torch.sqrt(
                torch.as_tensor(
                    np.asarray(train_ad[query_ids[rows]], dtype=np.float32),
                    device=device,
                ).clamp_min(0.0)
            ).reshape(len(rows), -1)
            candidate_ad = torch.sqrt(
                torch.as_tensor(
                    np.asarray(train_ad[fit], dtype=np.float32), device=device
                ).clamp_min(0.0)
            ).reshape(len(fit), -1)
            ad_similarity = query_ad @ candidate_ad.T
            full_top = torch.topk(ad_similarity, k=TOP_K, dim=1).indices.cpu().numpy()
            selected["oracle_ad_full"][rows] = fit[full_top]
            local_columns_t = torch.as_tensor(local_columns, device=device)
            local_ad = torch.gather(ad_similarity, 1, local_columns_t)
            local_top = torch.topk(local_ad, k=TOP_K, dim=1).indices.cpu().numpy()
            selected["oracle_ad_local256"][rows] = np.take_along_axis(
                local_ids, local_top, axis=1
            )
            print(
                f"E184 oracle outer={outer} site={site_value} "
                f"queries={len(rows)} candidates={len(fit)}",
                flush=True,
            )
    return {"wall_seconds": time.perf_counter() - started}


def selected_compact_metrics(
    selected_ids: np.ndarray,
    truth_pas: np.ndarray,
    truth_pdp: np.ndarray,
    train_pas: np.ndarray,
    train_pdp: np.ndarray,
    train_ad: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, top_k = selected_ids.shape
    pas = np.empty((count, top_k), dtype=np.float32)
    pdp = np.empty((count, top_k), dtype=np.float32)
    ad = np.empty((count, top_k), dtype=np.float32)
    truth_pas_flat = truth_pas.reshape(count, 4, -1)
    truth_pdp_flat = truth_pdp.reshape(count, 1, -1)
    for rank in range(top_k):
        candidate_pas = np.asarray(
            train_pas[selected_ids[:, rank]], dtype=np.float32
        ).reshape(count, 4, -1)
        candidate_pdp = np.asarray(
            train_pdp[selected_ids[:, rank]], dtype=np.float32
        ).reshape(count, 1, -1)
        pas[:, rank] = grouped_cosine_pairs(truth_pas_flat, candidate_pas)
        pdp[:, rank] = grouped_cosine_pairs(truth_pdp_flat, candidate_pdp)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        query_root = np.sqrt(
            np.maximum(np.asarray(train_ad[query_ids[start:end]], dtype=np.float32), 0.0)
        )
        candidate_root = np.sqrt(
            np.maximum(
                np.asarray(train_ad[selected_ids[start:end]], dtype=np.float32), 0.0
            )
        )
        ad[start:end] = np.sum(
            candidate_root * query_root[:, None], axis=(2, 3, 4), dtype=np.float64
        ).astype(np.float32)
    return pas, pdp, ad


def circular_delta(destination: np.ndarray, source: np.ndarray, period: float) -> np.ndarray:
    return (destination - source + period / 2.0) % period - period / 2.0


def component_common_mass(
    left_coordinate: np.ndarray,
    left_energy: np.ndarray,
    right_coordinate: np.ndarray,
    right_energy: np.ndarray,
) -> float:
    if float(np.sum(left_energy)) <= TINY or float(np.sum(right_energy)) <= TINY:
        return float(float(np.sum(left_energy)) <= TINY and float(np.sum(right_energy)) <= TINY)
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
    rows = rows[accepted]
    columns = columns[accepted]
    left_share = left_energy / max(float(np.sum(left_energy)), TINY)
    right_share = right_energy / max(float(np.sum(right_energy)), TINY)
    return float(np.sum(np.minimum(left_share[rows], right_share[columns])))


def selected_component_metrics(
    selected_ids: np.ndarray,
    query_ids: np.ndarray,
    coordinates: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    output = np.empty(selected_ids.shape, dtype=np.float32)
    for row, query_id in enumerate(query_ids):
        for rank, candidate_id in enumerate(selected_ids[row]):
            output[row, rank] = component_common_mass(
                coordinates[query_id],
                energy[query_id],
                coordinates[candidate_id],
                energy[candidate_id],
            )
    return output


@torch.inference_mode()
def load_official_channel_cache(
    train_channel: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    if device.type != "cuda":
        raise RuntimeError("E184 official H cache requires a CUDA device")
    channel = torch.empty(train_channel.shape, dtype=torch.complex64, device=device)
    pas = torch.empty(train_channel.shape, dtype=torch.float32, device=device)
    pdp = torch.empty(train_channel.shape, dtype=torch.float32, device=device)
    io_seconds = 0.0
    transfer_compute_seconds = 0.0
    for start in range(0, len(train_channel), batch_size):
        end = min(start + batch_size, len(train_channel))
        io_started = time.perf_counter()
        host = np.array(train_channel[start:end], dtype=np.complex64, copy=True)
        io_seconds += time.perf_counter() - io_started
        compute_started = time.perf_counter()
        channel[start:end].copy_(torch.from_numpy(host))
        value = channel[start:end]
        shaped = value.reshape(len(value), 2, 16, 8, 4, 192)
        pas[start:end] = (
            torch.fft.fft2(shaped, dim=(2, 3)).abs().square().reshape_as(value)
        )
        pdp[start:end] = torch.fft.ifft(value, dim=-1).abs().square()
        torch.cuda.synchronize(device)
        transfer_compute_seconds += time.perf_counter() - compute_started
        print(f"E184 official cache rows={end}/{len(train_channel)}", flush=True)
    return channel, pas, pdp, {
        "io_seconds": io_seconds,
        "transfer_and_spectrum_seconds": transfer_compute_seconds,
    }


@torch.inference_mode()
def official_selected_metrics_all(
    selected: dict[str, np.ndarray],
    query_ids: np.ndarray,
    train_channel: torch.Tensor,
    train_pas: torch.Tensor,
    train_pdp: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, PointMetricStatistics], np.ndarray, np.ndarray]:
    method_names = tuple(selected)
    selected_ids = np.stack([selected[name] for name in method_names])
    top_k = selected_ids.shape[2]
    pas = np.empty((len(method_names), len(query_ids), top_k), dtype=np.float64)
    pdp = np.empty_like(pas)
    error = np.empty((len(method_names), len(query_ids)), dtype=np.float64)
    energy = np.empty_like(error)

    def cosine(value: torch.Tensor, reference: torch.Tensor, axis: int) -> torch.Tensor:
        numerator = (value * reference).sum(dim=axis, dtype=torch.float64)
        value_norm = (value * value).sum(dim=axis, dtype=torch.float64)
        reference_norm = (reference * reference).sum(dim=axis, dtype=torch.float64)
        denominator = torch.sqrt(value_norm * reference_norm)
        denominator = torch.where(
            denominator == 0.0,
            torch.full_like(denominator, 1e-12),
            denominator,
        )
        return (numerator / denominator).to(torch.float32)

    for rank in range(top_k):
        for start in range(0, len(query_ids), batch_size):
            end = min(start + batch_size, len(query_ids))
            candidate_ids = torch.as_tensor(
                selected_ids[:, start:end, rank], dtype=torch.long, device=device
            )
            target_ids = torch.as_tensor(
                query_ids[start:end], dtype=torch.long, device=device
            )
            candidate_pas = train_pas[candidate_ids]
            target_pas = train_pas[target_ids].unsqueeze(0)
            batch_pas = cosine(candidate_pas, target_pas, 2).mean(
                dim=(2, 3), dtype=torch.float64
            )
            candidate_pdp = train_pdp[candidate_ids]
            target_pdp = train_pdp[target_ids].unsqueeze(0)
            batch_pdp = cosine(candidate_pdp, target_pdp, 4).mean(
                dim=(2, 3), dtype=torch.float64
            )
            pas[:, start:end, rank] = batch_pas.cpu().numpy()
            pdp[:, start:end, rank] = batch_pdp.cpu().numpy()
            if rank == 0:
                candidate_channel = train_channel[candidate_ids]
                target_channel = train_channel[target_ids].unsqueeze(0)
                batch_error = torch.abs(candidate_channel - target_channel).square().sum(
                    dim=(2, 3, 4), dtype=torch.float64
                )
                batch_energy = torch.abs(target_channel).square().sum(
                    dim=(2, 3, 4), dtype=torch.float64
                )
                error[:, start:end] = batch_error.cpu().numpy()
                energy[:, start:end] = (
                    batch_energy.expand_as(batch_error).cpu().numpy()
                )
        print(f"E184 official metrics rank={rank + 1}/{top_k}", flush=True)
    statistics = {
        name: PointMetricStatistics(
            pas=pas[index, :, 0],
            pdp=pdp[index, :, 0],
            squared_error=error[index],
            truth_energy=energy[index],
        )
        for index, name in enumerate(method_names)
    }
    return statistics, pas, pdp


def spatial_distance(
    selected_ids: np.ndarray, query_ids: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(
        positions[selected_ids, :2] - positions[query_ids, None, :2], axis=2
    ).astype(np.float32)


def spatial_rank(
    selected_ids: np.ndarray, spatial_orders: list[np.ndarray]
) -> np.ndarray:
    output = np.empty(len(selected_ids), dtype=np.int32)
    for row, candidate_id in enumerate(selected_ids[:, 0]):
        location = np.flatnonzero(spatial_orders[row] == candidate_id)
        if len(location) != 1:
            raise RuntimeError("oracle candidate missing from spatial order")
        output[row] = int(location[0]) + 1
    return output


def device_report(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"device": str(device), "cuda": False}
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(device),
        "cuda": True,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E184 requires CUDA for the registered GPU-resident replay")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    cosine_identity = validate_grouped_cosine(device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        TRAIN_CHANNEL,
        FOLD_CACHE,
        OOF_META,
        V7_COMPLETE,
        TRAIN_DESCRIPTORS,
        TRAIN_COMPACT_PAS,
        TRAIN_AD,
        ROUTE_CACHE,
        COMPONENT_CACHE,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    setup, positions, _, train_site, _ = load_positions()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    if len(query_ids) != 500 or len(np.unique(query_ids)) != 500:
        raise RuntimeError("canonical fixed500 identity mismatch")
    query_site = train_site[query_ids]
    hole_id = np.asarray(
        [f"P_S{int(site)}F{int(fold)}" for site, fold in zip(query_site, fold_id)]
    )

    descriptors = np.load(TRAIN_DESCRIPTORS, allow_pickle=False)
    train_pas = np.load(TRAIN_COMPACT_PAS, mmap_mode="r")
    train_pdp = descriptors["global_pdp"].astype(np.float32)
    train_power = descriptors["total_power"].astype(np.float64)
    train_ad = np.load(TRAIN_AD, mmap_mode="r")
    route = e11.route_signature(np.load(ROUTE_CACHE, allow_pickle=False), "train")
    v7 = np.load(V7_COMPLETE, mmap_mode="r")
    query_pas, query_pdp, v7_timing = v7_visible_spectra(
        v7, device, args.batch_size
    )

    selected, pools, spatial_orders, ranking_timing = build_deployment_rankings(
        setup,
        positions,
        train_site,
        query_ids,
        fold_id,
        query_pas,
        query_pdp,
        train_pas,
        train_pdp,
        route,
        device,
    )

    # Ground-truth arrays are first indexed only after deployment-visible rankings exist.
    truth_pas = np.asarray(train_pas[query_ids], dtype=np.float32)
    truth_pdp = np.asarray(train_pdp[query_ids], dtype=np.float32)
    oracle_timing = build_oracle_rankings(
        selected,
        pools,
        spatial_orders,
        query_ids,
        fold_id,
        train_site,
        truth_pas,
        truth_pdp,
        train_pas,
        train_pdp,
        train_ad,
        device,
    )

    components = np.load(COMPONENT_CACHE, allow_pickle=False)
    component_coordinates = components["coordinates"].astype(np.float32)
    component_energy = np.sum(
        np.abs(components["coefficients"].astype(np.complex64)) ** 2,
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    train_channel_mmap = np.load(TRAIN_CHANNEL, mmap_mode="r")
    (
        train_channel,
        train_channel_pas,
        train_channel_pdp,
        channel_cache_timing,
    ) = load_official_channel_cache(
        train_channel_mmap,
        device,
        args.channel_load_batch_size,
    )
    official_by_method, official_selected_pas, official_selected_pdp = (
        official_selected_metrics_all(
        selected,
        query_ids,
        train_channel,
        train_channel_pas,
        train_channel_pdp,
        device,
        args.metric_batch_size,
        )
    )
    nonzero = train_power[query_ids] > TINY
    method_metrics: dict[str, dict[str, np.ndarray | PointMetricStatistics]] = {}
    method_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    hole_rows: list[dict[str, object]] = []

    for method_index, name in enumerate(ALL_METHODS):
        compact_pas, compact_pdp, ad = selected_compact_metrics(
            selected[name],
            truth_pas,
            truth_pdp,
            train_pas,
            train_pdp,
            train_ad,
            query_ids,
            args.batch_size,
        )
        path_common = selected_component_metrics(
            selected[name], query_ids, component_coordinates, component_energy
        )
        official = official_by_method[name]
        official_top8_joint = 0.5 * (
            official_selected_pas[method_index]
            + official_selected_pdp[method_index]
        )
        distance = spatial_distance(selected[name], query_ids, positions)
        official_joint = 0.5 * (official.pas + official.pdp)
        compact_joint = 0.5 * (compact_pas + compact_pdp)
        selected_zero = train_power[selected[name][:, 0]] <= TINY
        method_metrics[name] = {
            "compact_pas": compact_pas,
            "compact_pdp": compact_pdp,
            "compact_joint": compact_joint,
            "ad": ad,
            "path_common": path_common,
            "official": official,
            "official_joint": official_joint,
            "distance": distance,
            "selected_zero": selected_zero,
        }
        aggregate = aggregate_statistics(official)
        method_rows.append(
            {
                "method": name,
                "deployment_visible": name in DEPLOYMENT_METHODS,
                "nonzero_count": int(np.sum(nonzero)),
                "compact_pas_top1": float(np.mean(compact_pas[nonzero, 0])),
                "compact_pdp_top1": float(np.mean(compact_pdp[nonzero, 0])),
                "compact_joint_top1": float(np.mean(compact_joint[nonzero, 0])),
                "compact_joint_top8_mean": float(np.mean(compact_joint[nonzero])),
                "angle_delay_top1": float(np.mean(ad[nonzero, 0])),
                "angle_delay_top8_mean": float(np.mean(ad[nonzero])),
                "path_common_mass_top1": float(np.mean(path_common[nonzero, 0])),
                "path_common_mass_top8_mean": float(np.mean(path_common[nonzero])),
                "official_pas_top1": float(np.mean(official.pas[nonzero])),
                "official_pdp_top1": float(np.mean(official.pdp[nonzero])),
                "official_joint_top1": float(np.mean(official_joint[nonzero])),
                "official_joint_top8_mean": float(
                    np.mean(official_top8_joint[nonzero])
                ),
                "official_nmse_all500": float(aggregate["nmse"]),
                "official_score_all500": float(aggregate["score"]),
                "distance_top1_mean_m": float(np.mean(distance[:, 0])),
                "distance_top1_median_m": float(np.median(distance[:, 0])),
                "distance_top1_p90_m": float(np.quantile(distance[:, 0], 0.90)),
                "zero_target_count": int(np.sum(~nonzero)),
                "zero_target_zero_donor_count": int(np.sum(selected_zero & ~nonzero)),
                "nonzero_target_zero_donor_count": int(np.sum(selected_zero & nonzero)),
            }
        )
        for row in range(len(query_ids)):
            point_rows.append(
                {
                    "oof_row": row,
                    "global_train_id": int(query_ids[row]),
                    "hole_id": str(hole_id[row]),
                    "site": int(query_site[row]),
                    "fold": int(fold_id[row]),
                    "truth_nonzero": bool(nonzero[row]),
                    "method": name,
                    "selected_train_id": int(selected[name][row, 0]),
                    "distance_m": float(distance[row, 0]),
                    "compact_pas": float(compact_pas[row, 0]),
                    "compact_pdp": float(compact_pdp[row, 0]),
                    "angle_delay": float(ad[row, 0]),
                    "path_common_mass": float(path_common[row, 0]),
                    "official_pas": float(official.pas[row]),
                    "official_pdp": float(official.pdp[row]),
                    "official_joint": float(official_joint[row]),
                }
            )
        print(f"E184 evaluated method={name}", flush=True)

    nearest_joint = np.asarray(method_metrics["nearest"]["official_joint"])
    for local_hole in sorted(np.unique(hole_id).tolist()):
        mask = (hole_id == local_hole) & nonzero
        for name in ALL_METHODS:
            official = method_metrics[name]["official"]
            if not isinstance(official, PointMetricStatistics):
                raise TypeError("official metric payload type mismatch")
            joint = np.asarray(method_metrics[name]["official_joint"])
            hole_rows.append(
                {
                    "hole_id": local_hole,
                    "method": name,
                    "nonzero_count": int(np.sum(mask)),
                    "official_pas": float(np.mean(official.pas[mask])),
                    "official_pdp": float(np.mean(official.pdp[mask])),
                    "official_joint": float(np.mean(joint[mask])),
                    "delta_vs_nearest": float(
                        np.mean(joint[mask] - nearest_joint[mask])
                    ),
                }
            )

    oracle_rows: list[dict[str, object]] = []
    oracle_ranks: dict[str, np.ndarray] = {}
    for name in ORACLE_METHODS:
        rank = spatial_rank(selected[name], spatial_orders)
        oracle_ranks[name] = rank
        for cutoff in (1, 8, 32, 64, 128, 256):
            oracle_rows.append(
                {
                    "method": name,
                    "spatial_cutoff": cutoff,
                    "recall": float(np.mean(rank <= cutoff)),
                }
            )

    np.savez_compressed(
        METRIC_PATH,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        hole_id=hole_id,
        method_names=np.asarray(ALL_METHODS),
        selected_ids=np.stack([selected[name] for name in ALL_METHODS]),
        compact_pas=np.stack(
            [np.asarray(method_metrics[name]["compact_pas"]) for name in ALL_METHODS]
        ),
        compact_pdp=np.stack(
            [np.asarray(method_metrics[name]["compact_pdp"]) for name in ALL_METHODS]
        ),
        angle_delay=np.stack(
            [np.asarray(method_metrics[name]["ad"]) for name in ALL_METHODS]
        ),
        path_common_mass=np.stack(
            [np.asarray(method_metrics[name]["path_common"]) for name in ALL_METHODS]
        ),
        official_pas=np.stack(
            [
                method_metrics[name]["official"].pas
                for name in ALL_METHODS
                if isinstance(method_metrics[name]["official"], PointMetricStatistics)
            ]
        ),
        official_pdp=np.stack(
            [
                method_metrics[name]["official"].pdp
                for name in ALL_METHODS
                if isinstance(method_metrics[name]["official"], PointMetricStatistics)
            ]
        ),
        distance=np.stack(
            [np.asarray(method_metrics[name]["distance"]) for name in ALL_METHODS]
        ),
        truth_nonzero=nonzero,
    )

    write_csv(METHOD_CSV, method_rows)
    write_csv(HOLE_CSV, hole_rows)
    write_csv(POINT_CSV, point_rows)
    write_csv(ORACLE_CSV, oracle_rows)

    method_summary = {row["method"]: row for row in method_rows}
    nearest_value = float(method_summary["nearest"]["official_joint_top1"])
    clean_value = float(method_summary["clean_full"]["official_joint_top1"])
    shuffled_value = float(
        method_summary["clean_shuffled"]["official_joint_top1"]
    )
    oracle_value = float(
        method_summary["oracle_spectral_full"]["official_joint_top1"]
    )
    positive_holes = sum(
        1
        for row in hole_rows
        if row["method"] == "clean_full" and float(row["delta_vs_nearest"]) > 0.0
    )
    compact_direction = (
        float(method_summary["clean_full"]["compact_joint_top1"])
        > float(method_summary["nearest"]["compact_joint_top1"])
    )
    path_direction = (
        float(method_summary["clean_full"]["path_common_mass_top1"])
        > float(method_summary["nearest"]["path_common_mass_top1"])
    )
    oracle_gate = oracle_value - nearest_value >= 0.05
    clean_gate = (
        clean_value - nearest_value >= 0.02
        and positive_holes >= 7
        and clean_value - shuffled_value >= 0.01
        and compact_direction
        and path_direction
    )
    if not oracle_gate:
        conclusion = "candidate_library_lacks_large_similarity_oracle"
    elif not clean_gate:
        conclusion = "similar_points_exist_but_clean_retrieval_fails"
    else:
        conclusion = "clean_retrieval_supported"

    performance = {
        "wall_seconds": time.perf_counter() - started,
        "v7_visible_spectra": v7_timing,
        "deployment_ranking": ranking_timing,
        "oracle_ranking": oracle_timing,
        "official_channel_cache": channel_cache_timing,
        "device": device_report(device),
        "grouped_cosine_identity": cosine_identity,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024.0 * 1024.0),
        "peak_torch_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (1024.0**3)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "single-pass official H load plus CPU path matching",
    }
    report = {
        "experiment": VERSION,
        "status": "diagnostic_complete",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "query_count": len(query_ids),
        "nonzero_query_count": int(np.sum(nonzero)),
        "zero_query_count": int(np.sum(~nonzero)),
        "strict_protocol": {
            "query": "canonical fixed500 complete spatial holes",
            "candidate_pool": "same-site outer-fit official Train points only",
            "deployment_rankings_saved_before_gt_indexing": True,
            "full_pool_and_local256_reported": True,
            "target_gt_used_for": "oracle and post-ranking evaluation only",
        },
        "method_summary": method_rows,
        "oracle_spatial_rank": {
            name: {
                "median": float(np.median(rank)),
                "p90": float(np.quantile(rank, 0.90)),
                "recall_at_8": float(np.mean(rank <= 8)),
                "recall_at_32": float(np.mean(rank <= 32)),
                "recall_at_64": float(np.mean(rank <= 64)),
                "recall_at_128": float(np.mean(rank <= 128)),
                "recall_at_256": float(np.mean(rank <= 256)),
            }
            for name, rank in oracle_ranks.items()
        },
        "gates": {
            "oracle_gain_required": 0.05,
            "oracle_gain": oracle_value - nearest_value,
            "oracle_gate_passed": oracle_gate,
            "clean_gain_required": 0.02,
            "clean_gain": clean_value - nearest_value,
            "positive_holes_required": 7,
            "positive_holes": positive_holes,
            "shuffled_margin_required": 0.01,
            "shuffled_margin": clean_value - shuffled_value,
            "compact_direction_positive": compact_direction,
            "path_direction_positive": path_direction,
            "clean_gate_passed": clean_gate,
            "conclusion": conclusion,
        },
        "outputs": {
            "deployment_rankings": PATHS.relative(RANKING_PATH),
            "metrics": PATHS.relative(METRIC_PATH),
            "method_csv": PATHS.relative(METHOD_CSV),
            "hole_csv": PATHS.relative(HOLE_CSV),
            "point_csv": PATHS.relative(POINT_CSV),
            "oracle_csv": PATHS.relative(ORACLE_CSV),
        },
        "performance": performance,
    }
    REPORT_PATH.write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_value(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
