#!/usr/bin/env python3
"""Find nonlocal propagation-state groups using only official Round 2 Train H."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E187-official-train-propagation-groups"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e187_official_train_propagation_groups"
)
REPORT_PATH = ANALYSIS_DIR / "report.json"
REPORT_MD = ANALYSIS_DIR / "官方训练H传播相似点分析.md"
PAIR_CSV = ANALYSIS_DIR / "robust_nonlocal_pairs.csv"
GROUP_CSV = ANALYSIS_DIR / "robust_train_groups.csv"
BEST_CSV = ANALYSIS_DIR / "all_train_best_nonlocal.csv"
ZERO_CSV = ANALYSIS_DIR / "zero_channel_points.csv"
FIGURE_PATH = ANALYSIS_DIR / "官方Train传播组空间分布.png"

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
COMPONENT_PATH = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
FOLD_PATH = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)

MIN_DISTANCE_M = 30.0
VIEW_P95 = 0.95
ROBUST_P99 = 0.99
MUTUAL_TOP_K = 8
COMPONENT_BACKGROUND_COUNT = 20_000
COMPONENT_P95 = 0.95
CONTROL_DISTANCE_HALF_WIDTH_M = 2.5
SEED = 20_260_812_187
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=256)
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


@torch.inference_mode()
def grouped_cosine_matrix(
    values: np.ndarray,
    groups: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    source = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
    source = source.reshape(len(source), groups, -1).clamp_min(0.0)
    energy = (source * source).sum(dim=2, keepdim=True, dtype=torch.float64)
    source = (source / torch.sqrt(energy).clamp_min(TINY)).to(torch.float32)
    output = np.empty((len(source), len(source)), dtype=np.float32)
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        similarity = torch.einsum("bgd,kgd->bkg", source[start:end], source)
        output[start:end] = (
            similarity.mean(dim=2, dtype=torch.float64).to(torch.float32).cpu().numpy()
        )
    return output


@torch.inference_mode()
def hellinger_matrix(
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    source = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
    source = torch.sqrt(source.clamp_min(0.0)).reshape(len(source), -1)
    source = source / torch.linalg.vector_norm(source, dim=1, keepdim=True).clamp_min(
        TINY
    )
    output = np.empty((len(source), len(source)), dtype=np.float32)
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        output[start:end] = (source[start:end] @ source.T).cpu().numpy()
    return output


def empirical_percentile_matrix(
    similarity: np.ndarray, background: np.ndarray
) -> np.ndarray:
    ordered = np.sort(np.asarray(background, dtype=np.float32))
    rank = np.searchsorted(ordered, similarity.reshape(-1), side="right")
    return (rank.reshape(similarity.shape) / max(len(ordered), 1)).astype(np.float32)


def upper_nonlocal_mask(
    positions: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(
        positions[:, None, :2] - positions[None, :, :2], axis=2
    ).astype(np.float32)
    mask = (
        np.triu(np.ones((len(positions), len(positions)), dtype=bool), k=1)
        & (distance >= MIN_DISTANCE_M)
        & valid[:, None]
        & valid[None, :]
    )
    return distance, mask


def mutual_top_k_mask(score: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    ranked = np.where(eligible | eligible.T, score, -np.inf)
    np.fill_diagonal(ranked, -np.inf)
    take = min(MUTUAL_TOP_K, max(len(ranked) - 1, 1))
    top = np.argpartition(ranked, -take, axis=1)[:, -take:]
    membership = np.zeros_like(ranked, dtype=bool)
    membership[np.arange(len(ranked))[:, None], top] = True
    return membership & membership.T


def component_score(
    left: int,
    right: int,
    coordinates: np.ndarray,
    energy: np.ndarray,
) -> float:
    return component_common_mass(
        coordinates[left], energy[left], coordinates[right], energy[right]
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
    rows = rows[accepted]
    columns = columns[accepted]
    left_share = left_energy / max(left_total, TINY)
    right_share = right_energy / max(right_total, TINY)
    return float(np.sum(np.minimum(left_share[rows], right_share[columns])))


def component_background(
    pairs: np.ndarray,
    coordinates: np.ndarray,
    energy: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    count = min(COMPONENT_BACKGROUND_COUNT, len(pairs))
    selected = rng.choice(len(pairs), size=count, replace=False)
    output = np.empty(count, dtype=np.float32)
    for row, pair_index in enumerate(selected):
        left, right = pairs[pair_index]
        output[row] = component_score(
            int(left), int(right), coordinates, energy
        )
    return output


def connected_components(
    point_count: int, edges: list[tuple[int, int]]
) -> list[list[int]]:
    parent = np.arange(point_count, dtype=np.int32)
    size = np.ones(point_count, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in edges:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    members: dict[int, list[int]] = {}
    touched = {value for edge in edges for value in edge}
    for value in sorted(touched):
        members.setdefault(find(value), []).append(value)
    return sorted(
        (group for group in members.values() if len(group) >= 3),
        key=lambda group: (-len(group), group[0]),
    )


def build_distance_control(
    edges: list[tuple[int, int]],
    site_ids: np.ndarray,
    site_positions: dict[int, np.ndarray],
    site_valid: dict[int, np.ndarray],
    final_edge_set: set[tuple[int, int]],
    coordinates: np.ndarray,
    energy: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    control = np.empty(len(edges), dtype=np.float32)
    mismatch = np.empty(len(edges), dtype=np.float32)
    for row, (left, right) in enumerate(edges):
        site = int(site_ids[left])
        ids = site_valid[site]
        positions = site_positions[site]
        left_local = int(np.flatnonzero(ids == left)[0])
        target_distance = float(
            np.linalg.norm(positions[left_local, :2] - positions[np.flatnonzero(ids == right)[0], :2])
        )
        distances = np.linalg.norm(
            positions[:, :2] - positions[left_local, :2][None], axis=1
        )
        allowed = (
            (ids != left)
            & (ids != right)
            & (distances >= MIN_DISTANCE_M)
            & (np.abs(distances - target_distance) <= CONTROL_DISTANCE_HALF_WIDTH_M)
        )
        candidates = ids[allowed]
        if len(candidates) == 0:
            order = np.argsort(np.abs(distances - target_distance), kind="stable")
            candidates = np.asarray(
                [
                    int(ids[index])
                    for index in order
                    if ids[index] not in (left, right)
                    and distances[index] >= MIN_DISTANCE_M
                ][:16],
                dtype=np.int64,
            )
        shuffled = candidates[rng.permutation(len(candidates))]
        chosen = int(shuffled[0])
        for candidate in shuffled:
            key = tuple(sorted((left, int(candidate))))
            if key not in final_edge_set:
                chosen = int(candidate)
                break
        chosen_local = int(np.flatnonzero(ids == chosen)[0])
        mismatch[row] = abs(float(distances[chosen_local]) - target_distance)
        control[row] = component_score(left, chosen, coordinates, energy)
    return control, mismatch


def plot_groups(
    positions: np.ndarray,
    site_ids: np.ndarray,
    zero: np.ndarray,
    groups: list[list[int]],
    edges: list[tuple[int, int]],
    group_by_point: dict[int, str],
) -> None:
    _, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    colors = plt.get_cmap("tab20")
    for axis, site in zip(axes, (0, 1)):
        site_mask = site_ids == site
        axis.scatter(
            positions[site_mask, 0],
            positions[site_mask, 1],
            s=7,
            c="#b8b8b8",
            alpha=0.35,
            linewidths=0,
            label="all Train",
        )
        site_zero = site_mask & zero
        axis.scatter(
            positions[site_zero, 0],
            positions[site_zero, 1],
            s=20,
            marker="x",
            c="#111111",
            alpha=0.65,
            label="zero H",
        )
        site_groups = [group for group in groups if int(site_ids[group[0]]) == site]
        for group_index, group in enumerate(site_groups):
            color = colors(group_index % 20)
            points = np.asarray(group, dtype=np.int64)
            group_id = group_by_point[int(points[0])]
            for left, right in edges:
                if left in group_by_point and right in group_by_point:
                    if group_by_point[left] == group_id and group_by_point[right] == group_id:
                        axis.plot(
                            positions[[left, right], 0],
                            positions[[left, right], 1],
                            color=color,
                            alpha=0.26,
                            linewidth=0.7,
                            zorder=1,
                        )
            axis.scatter(
                positions[points, 0],
                positions[points, 1],
                s=30,
                facecolors=[color],
                edgecolors="#222222",
                linewidths=0.45,
                label=f"{group_id} ({len(points)})",
                zorder=2,
            )
            center = np.mean(positions[points, :2], axis=0)
            axis.text(center[0], center[1], group_id, fontsize=8, weight="bold")
        axis.set_title(f"Site {site}: official-H nonlocal propagation groups")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(alpha=0.18)
        axis.set_aspect("equal", adjustable="datalim")
        if len(site_groups) <= 12:
            axis.legend(fontsize=7, loc="best")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    phase_started = started
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E187 requires one CUDA GPU for the angle-delay matrix")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    torch.backends.cuda.matmul.allow_tf32 = False

    required = (DESCRIPTOR_PATH, PAS_PATH, AD_PATH, COMPONENT_PATH, FOLD_PATH)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    setup, positions, _, train_site, _ = load_positions()
    fold_id = np.load(FOLD_PATH).astype(np.int8)
    descriptors = np.load(DESCRIPTOR_PATH, allow_pickle=False)
    pas = np.load(PAS_PATH, mmap_mode="r")
    pdp = descriptors["global_pdp"].astype(np.float32)
    power = descriptors["total_power"].astype(np.float64)
    valid = descriptors["valid"].astype(bool) & (power > TINY)
    ad = np.load(AD_PATH, mmap_mode="r")
    components = np.load(COMPONENT_PATH, allow_pickle=False)
    component_coordinates = components["coordinates"].astype(np.float32)
    component_energy = np.sum(
        np.abs(components["coefficients"].astype(np.complex64)) ** 2,
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    if not (
        len(positions)
        == len(train_site)
        == len(fold_id)
        == len(pas)
        == len(pdp)
        == len(ad)
        == len(component_coordinates)
        == 4000
    ):
        raise RuntimeError("E187 canonical Train identity mismatch")
    if not np.array_equal(valid, np.asarray(ad).reshape(4000, -1).sum(axis=1) > 0):
        raise RuntimeError("official spectral valid masks disagree")
    load_seconds = time.perf_counter() - phase_started

    rng = np.random.default_rng(SEED)
    site_results: dict[int, dict[str, object]] = {}
    candidate_records: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    component_background_values: dict[int, np.ndarray] = {}
    site_positions: dict[int, np.ndarray] = {}
    site_valid_ids: dict[int, np.ndarray] = {}
    matrix_seconds = 0.0
    component_seconds = 0.0

    for site in (0, 1):
        site_started = time.perf_counter()
        ids = np.flatnonzero(train_site == site).astype(np.int64)
        local_valid = valid[ids]
        site_positions[site] = positions[ids[local_valid]]
        site_valid_ids[site] = ids[local_valid]
        distance, upper_mask = upper_nonlocal_mask(positions[ids], local_valid)
        upper_left, upper_right = np.nonzero(upper_mask)
        background_pairs = np.column_stack([ids[upper_left], ids[upper_right]])

        pas_similarity = grouped_cosine_matrix(
            np.asarray(pas[ids], dtype=np.float32), 4, device, args.batch_size
        )
        pdp_similarity = grouped_cosine_matrix(
            np.asarray(pdp[ids], dtype=np.float32), 1, device, args.batch_size
        )
        ad_similarity = hellinger_matrix(
            np.asarray(ad[ids], dtype=np.float32), device, args.batch_size
        )
        torch.cuda.synchronize(device)
        matrix_seconds += time.perf_counter() - site_started

        pas_background = pas_similarity[upper_mask]
        pdp_background = pdp_similarity[upper_mask]
        ad_background = ad_similarity[upper_mask]
        thresholds = {
            "pas_p95": float(np.quantile(pas_background, VIEW_P95)),
            "pdp_p95": float(np.quantile(pdp_background, VIEW_P95)),
            "angle_delay_p95": float(np.quantile(ad_background, VIEW_P95)),
        }
        pas_percentile = empirical_percentile_matrix(pas_similarity, pas_background)
        pdp_percentile = empirical_percentile_matrix(pdp_similarity, pdp_background)
        ad_percentile = empirical_percentile_matrix(ad_similarity, ad_background)
        robust_percentile = np.minimum(
            np.minimum(pas_percentile, pdp_percentile), ad_percentile
        )
        robust_background = robust_percentile[upper_mask]
        thresholds["robust_percentile_p99"] = float(
            np.quantile(robust_background, ROBUST_P99)
        )

        symmetric_eligible = upper_mask | upper_mask.T
        mutual = mutual_top_k_mask(robust_percentile, upper_mask)
        spectral_gate = (
            symmetric_eligible
            & (pas_similarity >= thresholds["pas_p95"])
            & (pdp_similarity >= thresholds["pdp_p95"])
            & (ad_similarity >= thresholds["angle_delay_p95"])
            & (robust_percentile >= thresholds["robust_percentile_p99"])
            & mutual
        )
        spectral_gate = np.triu(spectral_gate, k=1)
        spectral_left, spectral_right = np.nonzero(spectral_gate)

        component_started = time.perf_counter()
        component_background_values[site] = component_background(
            background_pairs,
            component_coordinates,
            component_energy,
            rng,
        )
        component_threshold = float(
            np.quantile(component_background_values[site], COMPONENT_P95)
        )
        component_seconds += time.perf_counter() - component_started
        thresholds["component_common_mass_p95"] = component_threshold

        for left_local, right_local in zip(spectral_left, spectral_right):
            left = int(ids[left_local])
            right = int(ids[right_local])
            common_mass = component_score(
                left, right, component_coordinates, component_energy
            )
            if common_mass < component_threshold:
                continue
            candidate_records.append(
                {
                    "left_index": left,
                    "left_no": left + 1,
                    "right_index": right,
                    "right_no": right + 1,
                    "site": site,
                    "left_fold": int(fold_id[left]),
                    "right_fold": int(fold_id[right]),
                    "cross_fold": bool(fold_id[left] != fold_id[right]),
                    "left_x": float(positions[left, 0]),
                    "left_y": float(positions[left, 1]),
                    "right_x": float(positions[right, 0]),
                    "right_y": float(positions[right, 1]),
                    "distance_m": float(distance[left_local, right_local]),
                    "pas_similarity": float(pas_similarity[left_local, right_local]),
                    "pdp_similarity": float(pdp_similarity[left_local, right_local]),
                    "angle_delay_similarity": float(
                        ad_similarity[left_local, right_local]
                    ),
                    "pas_background_percentile": float(
                        pas_percentile[left_local, right_local]
                    ),
                    "pdp_background_percentile": float(
                        pdp_percentile[left_local, right_local]
                    ),
                    "angle_delay_background_percentile": float(
                        ad_percentile[left_local, right_local]
                    ),
                    "robust_background_percentile": float(
                        robust_percentile[left_local, right_local]
                    ),
                    "component_common_mass": float(common_mass),
                }
            )

        robust_rank = np.where(symmetric_eligible, robust_percentile, -np.inf)
        np.fill_diagonal(robust_rank, -np.inf)
        best_local = np.argmax(robust_rank, axis=1)
        final_site_edges = {
            tuple(sorted((int(row["left_index"]), int(row["right_index"]))))
            for row in candidate_records
            if int(row["site"]) == site
        }
        for local_index, partner_local in enumerate(best_local):
            point = int(ids[local_index])
            if not local_valid[local_index]:
                continue
            partner = int(ids[partner_local])
            edge_key = tuple(sorted((point, partner)))
            best_rows.append(
                {
                    "point_index": point,
                    "point_no": point + 1,
                    "site": site,
                    "fold": int(fold_id[point]),
                    "x": float(positions[point, 0]),
                    "y": float(positions[point, 1]),
                    "best_point_index": partner,
                    "best_point_no": partner + 1,
                    "best_fold": int(fold_id[partner]),
                    "best_x": float(positions[partner, 0]),
                    "best_y": float(positions[partner, 1]),
                    "distance_m": float(distance[local_index, partner_local]),
                    "pas_similarity": float(pas_similarity[local_index, partner_local]),
                    "pdp_similarity": float(pdp_similarity[local_index, partner_local]),
                    "angle_delay_similarity": float(
                        ad_similarity[local_index, partner_local]
                    ),
                    "robust_background_percentile": float(
                        robust_percentile[local_index, partner_local]
                    ),
                    "is_final_robust_edge": edge_key in final_site_edges,
                }
            )

        site_results[site] = {
            "point_count": int(len(ids)),
            "nonzero_count": int(np.sum(local_valid)),
            "zero_count": int(np.sum(~local_valid)),
            "background_pair_count": int(np.sum(upper_mask)),
            "spectral_candidate_edge_count": int(len(spectral_left)),
            "thresholds": thresholds,
            "matrix_seconds": float(time.perf_counter() - site_started),
        }
        del pas_similarity, pdp_similarity, ad_similarity
        del pas_percentile, pdp_percentile, ad_percentile, robust_percentile
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_records.sort(
        key=lambda row: (
            -float(row["robust_background_percentile"]),
            -float(row["component_common_mass"]),
            int(row["left_index"]),
            int(row["right_index"]),
        )
    )
    final_edges = [
        (int(row["left_index"]), int(row["right_index"]))
        for row in candidate_records
    ]
    final_edge_set = {tuple(sorted(edge)) for edge in final_edges}
    control, control_mismatch = build_distance_control(
        final_edges,
        train_site,
        site_positions,
        site_valid_ids,
        final_edge_set,
        component_coordinates,
        component_energy,
        rng,
    )
    for row, control_value, mismatch in zip(
        candidate_records, control, control_mismatch
    ):
        row["distance_matched_control_common_mass"] = float(control_value)
        row["control_distance_abs_error_m"] = float(mismatch)

    groups = connected_components(4000, final_edges)
    group_by_point: dict[int, str] = {}
    group_rows: list[dict[str, object]] = []
    edge_by_point: dict[int, list[dict[str, object]]] = {}
    for row in candidate_records:
        edge_by_point.setdefault(int(row["left_index"]), []).append(row)
        edge_by_point.setdefault(int(row["right_index"]), []).append(row)
    group_summaries: list[dict[str, object]] = []
    for group_number, members in enumerate(groups, start=1):
        group_id = f"G{group_number:02d}"
        for member in members:
            group_by_point[member] = group_id
        member_set = set(members)
        group_edges = [
            row
            for row in candidate_records
            if int(row["left_index"]) in member_set
            and int(row["right_index"]) in member_set
        ]
        folds = sorted({int(fold_id[member]) for member in members})
        summary = {
            "group_id": group_id,
            "site": int(train_site[members[0]]),
            "member_count": len(members),
            "members_zero_based": members,
            "members_one_based": [member + 1 for member in members],
            "folds": folds,
            "edge_count": len(group_edges),
            "cross_fold_edge_count": int(
                sum(bool(row["cross_fold"]) for row in group_edges)
            ),
            "minimum_robust_percentile": float(
                min(float(row["robust_background_percentile"]) for row in group_edges)
            ),
            "mean_component_common_mass": float(
                np.mean([float(row["component_common_mass"]) for row in group_edges])
            ),
        }
        group_summaries.append(summary)
        degrees = {
            member: sum(
                int(row["left_index"]) == member
                or int(row["right_index"]) == member
                for row in group_edges
            )
            for member in members
        }
        for member_rank, member in enumerate(members, start=1):
            strongest = max(
                edge_by_point[member],
                key=lambda row: (
                    float(row["robust_background_percentile"]),
                    float(row["component_common_mass"]),
                ),
            )
            partner = (
                int(strongest["right_index"])
                if int(strongest["left_index"]) == member
                else int(strongest["left_index"])
            )
            group_rows.append(
                {
                    "group_id": group_id,
                    "group_size": len(members),
                    "group_edge_count": len(group_edges),
                    "member_rank": member_rank,
                    "train_index": member,
                    "train_no": member + 1,
                    "site": int(train_site[member]),
                    "fold": int(fold_id[member]),
                    "x": float(positions[member, 0]),
                    "y": float(positions[member, 1]),
                    "degree": degrees[member],
                    "strongest_partner_index": partner,
                    "strongest_partner_no": partner + 1,
                    "strongest_robust_percentile": float(
                        strongest["robust_background_percentile"]
                    ),
                    "strongest_component_common_mass": float(
                        strongest["component_common_mass"]
                    ),
                }
            )

    grouped_members = set(group_by_point)
    grouped_edges = [
        row
        for row in candidate_records
        if int(row["left_index"]) in grouped_members
        and int(row["right_index"]) in grouped_members
    ]
    cross_fold_count = int(sum(bool(row["cross_fold"]) for row in candidate_records))
    edge_component = np.asarray(
        [float(row["component_common_mass"]) for row in candidate_records],
        dtype=np.float32,
    )
    component_margin = float(np.median(edge_component) - np.median(control)) if len(control) else 0.0
    gate = {
        "minimum_robust_edges": 20,
        "minimum_groups_size_at_least_3": 3,
        "minimum_cross_fold_edge_fraction": 0.5,
        "minimum_component_control_median_margin": 0.10,
        "robust_edge_count": len(candidate_records),
        "group_count": len(groups),
        "cross_fold_edge_fraction": cross_fold_count / max(len(candidate_records), 1),
        "component_control_median_margin": component_margin,
    }
    gate["passed"] = bool(
        gate["robust_edge_count"] >= gate["minimum_robust_edges"]
        and gate["group_count"] >= gate["minimum_groups_size_at_least_3"]
        and gate["cross_fold_edge_fraction"]
        >= gate["minimum_cross_fold_edge_fraction"]
        and gate["component_control_median_margin"]
        >= gate["minimum_component_control_median_margin"]
    )
    gate["conclusion"] = (
        "official_train_h_nonlocal_propagation_groups_exist"
        if gate["passed"]
        else "registered_official_h_grouping_gate_failed"
    )

    zero_rows = [
        {
            "train_index": int(point),
            "train_no": int(point) + 1,
            "site": int(train_site[point]),
            "fold": int(fold_id[point]),
            "x": float(positions[point, 0]),
            "y": float(positions[point, 1]),
        }
        for point in np.flatnonzero(~valid)
    ]
    write_csv(PAIR_CSV, candidate_records)
    write_csv(GROUP_CSV, group_rows)
    write_csv(BEST_CSV, best_rows)
    write_csv(ZERO_CSV, zero_rows)
    plot_groups(
        positions,
        train_site,
        ~valid,
        groups,
        final_edges,
        group_by_point,
    )

    total_seconds = time.perf_counter() - started
    device_properties = torch.cuda.get_device_properties(device_index)
    performance = {
        "selected_gpu": str(device),
        "gpu_name": device_properties.name,
        "load_seconds": load_seconds,
        "similarity_matrix_seconds": matrix_seconds,
        "component_background_seconds": component_seconds,
        "total_seconds": total_seconds,
        "peak_torch_memory_gib": torch.cuda.max_memory_allocated(device_index) / 2**30,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    performance["measured_bottleneck"] = max(
        (
            ("load", load_seconds),
            ("similarity_matrices", matrix_seconds),
            ("component_background", component_seconds),
            ("other_output_and_controls", total_seconds - load_seconds - matrix_seconds - component_seconds),
        ),
        key=lambda item: item[1],
    )[0]

    report = {
        "version": VERSION,
        "input_boundary": {
            "official_train_h_derived_only": True,
            "sionna_used": False,
            "v12_or_test_truth_used": False,
            "train_point_count": 4000,
            "nonzero_point_count": int(np.sum(valid)),
            "zero_point_count": int(np.sum(~valid)),
        },
        "definition": {
            "minimum_xy_distance_m": MIN_DISTANCE_M,
            "views": [
                "official compact PAS grouped by UE",
                "official global PDP",
                "official joint angle-delay Hellinger similarity",
                "official continuous 24-component common mass",
            ],
            "view_background_percentile": VIEW_P95,
            "robust_background_percentile": ROBUST_P99,
            "mutual_top_k": MUTUAL_TOP_K,
            "component_background_count_per_site": COMPONENT_BACKGROUND_COUNT,
            "component_background_percentile": COMPONENT_P95,
        },
        "sites": site_results,
        "graph": {
            "robust_edge_count": len(candidate_records),
            "edge_endpoint_count": len({value for edge in final_edges for value in edge}),
            "cross_fold_edge_count": cross_fold_count,
            "cross_fold_edge_fraction": cross_fold_count / max(len(candidate_records), 1),
            "group_count_size_at_least_3": len(groups),
            "group_member_count": len(grouped_members),
            "grouped_edge_count": len(grouped_edges),
            "groups": group_summaries,
        },
        "component_control": {
            "edge_median": float(np.median(edge_component)) if len(edge_component) else 0.0,
            "distance_matched_control_median": float(np.median(control)) if len(control) else 0.0,
            "median_margin": component_margin,
            "distance_abs_error_median": float(np.median(control_mismatch)) if len(control_mismatch) else 0.0,
            "distance_abs_error_p90": float(np.quantile(control_mismatch, 0.9)) if len(control_mismatch) else 0.0,
        },
        "gate": gate,
        "performance": performance,
        "outputs": {
            "pairs_csv": PATHS.relative(PAIR_CSV),
            "groups_csv": PATHS.relative(GROUP_CSV),
            "all_best_csv": PATHS.relative(BEST_CSV),
            "zero_points_csv": PATHS.relative(ZERO_CSV),
            "report_markdown": PATHS.relative(REPORT_MD),
            "figure": PATHS.relative(FIGURE_PATH),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    strongest = candidate_records[:20]
    lines = [
        "# 官方训练H传播相似点分析",
        "",
        "## 直接结论",
        "",
        (
            f"仅使用4000个官方Train H及坐标，共找到 **{len(candidate_records)}** 条"
            f"相距至少`{MIN_DISTANCE_M:.0f}m`的稳健传播相似边，形成 **{len(groups)}** 个"
            f"至少3点的传播组，覆盖 **{len(grouped_members)}** 个非零训练点。"
        ),
        (
            f"跨五折空间区域的边占`{cross_fold_count / max(len(candidate_records), 1):.1%}`；"
            f"24连续路径公共质量中位数相对同站距离匹配打乱控制高`{component_margin:+.6f}`。"
        ),
        f"预注册门结论：`{gate['conclusion']}`，整体通过=`{str(gate['passed']).lower()}`。",
        "",
        "这里没有使用Sionna、V12、Test H或E185分组。相似关系完全由官方Train H重新定义。",
        "",
        "## 传播组",
        "",
        "| 组 | 站点 | 点数 | Train点号（1-based） | 空间折 | 边数 | 最弱稳健百分位 | 路径公共质量均值 |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for group in group_summaries:
        members = ", ".join(str(value) for value in group["members_one_based"])
        folds = ", ".join(str(value) for value in group["folds"])
        lines.append(
            f"| {group['group_id']} | {group['site']} | {group['member_count']} | {members} | "
            f"{folds} | {group['edge_count']} | {group['minimum_robust_percentile']:.6f} | "
            f"{group['mean_component_common_mass']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 最强非局部点对",
            "",
            "| A点 | B点 | 站点 | 折A->折B | 距离m | PAS | PDP | angle-delay | 稳健背景百分位 | 路径公共质量 |",
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in strongest:
        lines.append(
            f"| {row['left_no']} | {row['right_no']} | {row['site']} | "
            f"{row['left_fold']}->{row['right_fold']} | {row['distance_m']:.2f} | "
            f"{row['pas_similarity']:.6f} | {row['pdp_similarity']:.6f} | "
            f"{row['angle_delay_similarity']:.6f} | {row['robust_background_percentile']:.6f} | "
            f"{row['component_common_mass']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `robust_nonlocal_pairs.csv`：全部严格点对及四种官方H相似度。",
            "- `robust_train_groups.csv`：三点以上传播组成员。",
            "- `all_train_best_nonlocal.csv`：每个非零Train点最像的30米外Train点。",
            "- `zero_channel_points.csv`：262个全零覆盖状态点。",
            "- `官方Train传播组空间分布.png`：传播组空间分布。",
            "",
            "本分析只证明官方Train H中是否存在可重复传播状态，不训练模型、不生成Test H或提交包。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_value(report["graph"]), ensure_ascii=False, indent=2))
    print(json.dumps(json_value(report["component_control"]), ensure_ascii=False, indent=2))
    print(json.dumps(json_value(gate), ensure_ascii=False, indent=2))
    print(json.dumps(json_value(performance), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
