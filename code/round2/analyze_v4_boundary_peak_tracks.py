#!/usr/bin/env python3
"""Track boundary-anchored PAS/PDP peaks through complete Round 2 holes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from data import load_positions
from project_paths import PATHS
from run_v3_grouped_link_refinement import build_grouped_descriptors
from analyze_v4_group_graph_tv import boundary_ids, normalize_amplitude, power_cosine


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_boundary_peak_tracks"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_boundary_peak_tracks"
REPORT = ANALYSIS_DIR / "boundary_peak_tracks.json"
POINT_CACHE = ARTIFACT_DIR / "boundary_peak_track_scores.npz"

FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
OOF_WEIGHTS = (
    PATHS.artifacts / "round2" / "rbf_spectral_field" / "round2_rbf_oof_weights.npz"
)
DESCRIPTORS = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_spectral_descriptors.npz"
)

PEAKS = 6
PEAK_NMS_RADIUS = 2
PEAK_WINDOW_RADIUS = 2
SPATIAL_NEIGHBORS = 8
MATCH_THRESHOLDS = (2.0, 4.0, 6.0)
MIN_BOUNDARY_COUNTS = (2, 4, 8)
MAX_SHIFTS = (2.0, 4.0)
ALPHAS = (0.25, 0.50, 1.00)


@dataclass(frozen=True)
class Config:
    match_threshold: float
    min_boundary: int
    max_shift: float
    alpha: float

    @property
    def name(self) -> str:
        return (
            f"m{self.match_threshold:g}_b{self.min_boundary}_"
            f"x{self.max_shift:g}_a{self.alpha:g}"
        )


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.rank = np.zeros(count, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def v2_anchor(
    amplitude: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(donor_ids), amplitude.shape[1], amplitude.shape[2]), dtype=np.float32)
    for row, (ids, values) in enumerate(zip(donor_ids, weights)):
        output[row] = np.maximum(
            np.einsum(
                "k,kgd->gd",
                values.astype(np.float32),
                np.asarray(amplitude[ids], dtype=np.float32),
            ),
            0.0,
        )
    return normalize_amplitude(output)


def extract_peaks(power: np.ndarray, circular: bool) -> tuple[np.ndarray, np.ndarray]:
    nodes, length = power.shape
    bins = np.full((nodes, PEAKS), -1, dtype=np.int16)
    mass = np.zeros((nodes, PEAKS), dtype=np.float32)
    for node in range(nodes):
        work = np.asarray(power[node], dtype=np.float32).copy()
        for peak in range(PEAKS):
            index = int(np.argmax(work))
            value = float(work[index])
            if value <= 0.0:
                break
            bins[node, peak] = index
            mass[node, peak] = value
            offsets = np.arange(-PEAK_NMS_RADIUS, PEAK_NMS_RADIUS + 1)
            if circular:
                work[(index + offsets) % length] = 0.0
            else:
                selected = index + offsets
                selected = selected[(selected >= 0) & (selected < length)]
                work[selected] = 0.0
    return bins, mass


def bin_distance(left: np.ndarray, right: np.ndarray, length: int, circular: bool) -> np.ndarray:
    distance = np.abs(left[:, None] - right[None, :]).astype(np.float64)
    if circular:
        distance = np.minimum(distance, length - distance)
    return distance


def spatial_edges(xy: np.ndarray) -> list[tuple[int, int]]:
    take = min(SPATIAL_NEIGHBORS + 1, len(xy))
    neighbor = cKDTree(xy).query(xy, k=take, workers=-1)[1][:, 1:]
    edges = {
        (min(source, int(target)), max(source, int(target)))
        for source in range(len(xy))
        for target in neighbor[source]
        if int(target) != source
    }
    return sorted(edges)


def track_components(
    bins: np.ndarray,
    mass: np.ndarray,
    edges: list[tuple[int, int]],
    length: int,
    circular: bool,
    threshold: float,
) -> dict[int, list[tuple[int, int]]]:
    union = UnionFind(bins.size)
    for left, right in edges:
        left_valid = np.flatnonzero(bins[left] >= 0)
        right_valid = np.flatnonzero(bins[right] >= 0)
        if not len(left_valid) or not len(right_valid):
            continue
        cost = bin_distance(
            bins[left, left_valid], bins[right, right_valid], length, circular
        )
        left_match, right_match = linear_sum_assignment(cost)
        for local_left, local_right in zip(left_match, right_match):
            if cost[local_left, local_right] <= threshold:
                union.union(
                    left * PEAKS + int(left_valid[local_left]),
                    right * PEAKS + int(right_valid[local_right]),
                )
    components: dict[int, list[tuple[int, int]]] = {}
    for node in range(len(bins)):
        for peak in range(PEAKS):
            if bins[node, peak] < 0 or mass[node, peak] <= 0.0:
                continue
            root = union.find(node * PEAKS + peak)
            components.setdefault(root, []).append((node, peak))
    return components


def unwrap_bins(values: np.ndarray, weights: np.ndarray, length: int) -> np.ndarray:
    angle = values * (2.0 * np.pi / length)
    reference = np.angle(np.sum(weights * np.exp(1j * angle))) * length / (2.0 * np.pi)
    reference %= length
    return reference + (values - reference + length / 2.0) % length - length / 2.0


def robust_plane(
    xy: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    center = np.mean(xy, axis=0)
    scale = max(float(np.std(xy)), 1.0)
    design = np.column_stack(
        [np.ones(len(xy)), (xy[:, 0] - center[0]) / scale, (xy[:, 1] - center[1]) / scale]
    )
    fit_weights = np.maximum(weights, 1e-5).copy()
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(3):
        system = design.T @ (fit_weights[:, None] * design) + np.diag([1e-5, 0.03, 0.03])
        right = design.T @ (fit_weights * values)
        coefficients = np.linalg.solve(system, right)
        residual = values - design @ coefficients
        scale_residual = max(float(np.median(np.abs(residual))) * 1.4826, 0.25)
        fit_weights = np.maximum(weights, 1e-5) / np.maximum(
            1.0, np.abs(residual) / (2.5 * scale_residual)
        )
    return design @ coefficients


def fractional_deposit(
    output: np.ndarray,
    index: float,
    value: float,
    circular: bool,
) -> None:
    length = len(output)
    low = math_floor = int(np.floor(index))
    fraction = float(index - math_floor)
    high = low + 1
    if circular:
        output[low % length] += value * (1.0 - fraction)
        output[high % length] += value * fraction
    else:
        low = min(max(low, 0), length - 1)
        high = min(max(high, 0), length - 1)
        output[low] += value * (1.0 - fraction)
        output[high] += value * fraction


def apply_tracks(
    base_amplitude: np.ndarray,
    boundary_amplitude: np.ndarray,
    boundary_xy: np.ndarray,
    query_xy: np.ndarray,
    circular: bool,
    match_threshold: float,
    min_boundary: int,
    max_shift: float,
) -> tuple[np.ndarray, dict[str, float]]:
    boundary_count = len(boundary_amplitude)
    all_amplitude = np.concatenate([boundary_amplitude, base_amplitude], axis=0)
    all_power = np.square(all_amplitude)
    xy = np.concatenate([boundary_xy, query_xy], axis=0)
    length = all_power.shape[1]
    bins, mass = extract_peaks(all_power, circular)
    components = track_components(
        bins,
        mass,
        spatial_edges(xy),
        length,
        circular,
        match_threshold,
    )
    shifts: dict[tuple[int, int], float] = {}
    accepted = 0
    for members in components.values():
        best_by_node: dict[int, tuple[int, float]] = {}
        for node, peak in members:
            if node not in best_by_node or mass[node, peak] > best_by_node[node][1]:
                best_by_node[node] = (peak, float(mass[node, peak]))
        selected = [(node, peak) for node, (peak, _) in best_by_node.items()]
        boundary_members = sum(node < boundary_count for node, _ in selected)
        query_members = sum(node >= boundary_count for node, _ in selected)
        if boundary_members < min_boundary or query_members < 2:
            continue
        node_ids = np.asarray([node for node, _ in selected], dtype=np.int64)
        peak_ids = np.asarray([peak for _, peak in selected], dtype=np.int64)
        values = bins[node_ids, peak_ids].astype(np.float64)
        fit_values = unwrap_bins(values, mass[node_ids, peak_ids], length) if circular else values
        fit_weight = mass[node_ids, peak_ids].astype(np.float64)
        fit_weight *= np.where(node_ids < boundary_count, 3.0, 0.75)
        fitted = robust_plane(xy[node_ids], fit_values, fit_weight)
        for local, (node, peak) in enumerate(selected):
            if node < boundary_count:
                continue
            original = float(values[local])
            target = float(fitted[local])
            if circular:
                delta = (target - original + length / 2.0) % length - length / 2.0
            else:
                delta = target - original
            shifts[(node - boundary_count, peak)] = float(
                np.clip(delta, -max_shift, max_shift)
            )
        accepted += 1

    output_power = np.square(base_amplitude).copy()
    shifted_mass = 0.0
    for query in range(len(base_amplitude)):
        occupied = np.zeros(length, dtype=bool)
        ordered = sorted(
            [
                (peak, delta)
                for (node, peak), delta in shifts.items()
                if node == query and abs(delta) > 1e-4
            ],
            key=lambda row: float(mass[boundary_count + query, row[0]]),
            reverse=True,
        )
        for peak, delta in ordered:
            center = int(bins[boundary_count + query, peak])
            offsets = np.arange(-PEAK_WINDOW_RADIUS, PEAK_WINDOW_RADIUS + 1)
            indices = (center + offsets) % length if circular else center + offsets
            if not circular:
                indices = indices[(indices >= 0) & (indices < length)]
            indices = np.asarray(indices, dtype=np.int64)
            indices = indices[~occupied[indices]]
            if not len(indices):
                continue
            values = output_power[query, indices].copy()
            output_power[query, indices] = 0.0
            occupied[indices] = True
            for index, value in zip(indices, values):
                fractional_deposit(
                    output_power[query], float(index) + delta, float(value), circular
                )
            shifted_mass += float(np.sum(values))
    target = normalize_amplitude(np.sqrt(np.maximum(output_power, 0.0)))
    return target, {
        "accepted_tracks": float(accepted),
        "shifted_mass_per_query": float(shifted_mass / max(len(base_amplitude), 1)),
        "shift_count": float(len(shifts)),
    }


def selected_scores(score: np.ndarray, choice: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask)
    groups = score.shape[2]
    return score[
        choice[:, None], rows[None, :], np.arange(groups)[:, None]
    ].T


def nested_summary(
    score: np.ndarray,
    baseline: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    names: list[str],
) -> dict[str, object]:
    nested = np.empty_like(baseline)
    choices: list[dict[str, object]] = []
    for held in np.unique(fold_id):
        for site_value in np.unique(site[fold_id == held]):
            train = (fold_id != held) & (site == site_value)
            test = (fold_id == held) & (site == site_value)
            choice = np.argmax(score[:, train].mean(axis=1), axis=0)
            nested[test] = selected_scores(score, choice, test)
            choices.append(
                {
                    "held_fold": int(held),
                    "site": int(site_value),
                    "configs": [names[int(index)] for index in choice],
                }
            )
    posterior = int(np.argmax(score.mean(axis=(1, 2))))

    def result(value: np.ndarray) -> dict[str, object]:
        return {
            "mean": float(np.mean(value)),
            "delta": float(np.mean(value) - np.mean(baseline)),
            "fold_delta": [
                float(
                    np.mean(value[fold_id == fold])
                    - np.mean(baseline[fold_id == fold])
                )
                for fold in np.unique(fold_id)
            ],
        }

    return {
        "baseline": float(np.mean(baseline)),
        "posterior_best": {"config": names[posterior], **result(score[posterior])},
        "nested_site_group": {**result(nested), "choices": choices},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="0,1,2,3,4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _, positions, _, site, _ = load_positions()
    valid = np.load(DESCRIPTORS)["valid"].astype(bool)
    folds = np.load(FOLD_CACHE)
    weights = np.load(OOF_WEIGHTS)
    query_ids = weights["query_ids"].astype(np.int64)
    fold_id = weights["fold_id"].astype(np.int8)
    donor_ids = weights["neighbor_ids"].astype(np.int64)
    base_weights = weights["weights"].astype(np.float32)
    query_site = site[query_ids]
    pas_raw, pdp_raw = build_grouped_descriptors()
    pas_amplitude = normalize_amplitude(np.asarray(pas_raw, dtype=np.float32))
    pdp_amplitude = normalize_amplitude(np.asarray(pdp_raw, dtype=np.float32))
    base_pas = v2_anchor(pas_amplitude, donor_ids, base_weights)
    base_pdp = v2_anchor(pdp_amplitude, donor_ids, base_weights)
    configs = [
        Config(match, boundary, shift, alpha)
        for match in MATCH_THRESHOLDS
        for boundary in MIN_BOUNDARY_COUNTS
        for shift in MAX_SHIFTS
        for alpha in ALPHAS
    ]
    pas_prediction = np.repeat(base_pas[None], len(configs), axis=0)
    pdp_prediction = np.repeat(base_pdp[None], len(configs), axis=0)
    diagnostics: list[dict[str, object]] = []
    selected_folds = {int(value) for value in args.folds.split(",") if value.strip()}

    for held in np.unique(fold_id):
        if int(held) not in selected_folds:
            continue
        held_rows = np.flatnonzero(fold_id == held)
        fit_all = folds[f"fold{int(held)}_fit"].astype(np.int64)
        for site_value in np.unique(query_site[held_rows]):
            output_rows = held_rows[query_site[held_rows] == site_value]
            query = query_ids[output_rows]
            fit_site = fit_all[site[fit_all] == site_value]
            boundary = boundary_ids(fit_site, query, positions, valid)
            for metric_name, amplitude, base, output, circular in (
                ("pas", pas_amplitude, base_pas, pas_prediction, True),
                ("pdp", pdp_amplitude, base_pdp, pdp_prediction, False),
            ):
                target_cache: dict[tuple[float, int, float], tuple[np.ndarray, dict[str, float]]] = {}
                for match in MATCH_THRESHOLDS:
                    for minimum in MIN_BOUNDARY_COUNTS:
                        for shift in MAX_SHIFTS:
                            group_targets = []
                            group_stats = []
                            for group in range(amplitude.shape[1]):
                                target, stats = apply_tracks(
                                    base[output_rows, group],
                                    amplitude[boundary, group],
                                    positions[boundary, :2],
                                    positions[query, :2],
                                    circular,
                                    match,
                                    minimum,
                                    shift,
                                )
                                group_targets.append(target)
                                group_stats.append(stats)
                            target_cache[(match, minimum, shift)] = (
                                np.stack(group_targets, axis=1),
                                {
                                    key: float(np.mean([row[key] for row in group_stats]))
                                    for key in group_stats[0]
                                },
                            )
                for config_index, config in enumerate(configs):
                    target, stats = target_cache[
                        (config.match_threshold, config.min_boundary, config.max_shift)
                    ]
                    blended = (1.0 - config.alpha) * base[output_rows] + config.alpha * target
                    output[config_index, output_rows] = normalize_amplitude(blended)
                    if config.alpha == 1.0:
                        diagnostics.append(
                            {
                                "fold": int(held),
                                "site": int(site_value),
                                "metric": metric_name,
                                "config": config.name,
                                **stats,
                            }
                        )
            print(
                f"boundary tracks fold={int(held)} site={int(site_value)} "
                f"query={len(query)} boundary={len(boundary)}",
                flush=True,
            )

    pas_truth = pas_amplitude[query_ids]
    pdp_truth = pdp_amplitude[query_ids]
    base_pas_score = power_cosine(base_pas, pas_truth)
    base_pdp_score = power_cosine(base_pdp, pdp_truth)
    pas_score = np.stack(
        [power_cosine(value, pas_truth) for value in pas_prediction]
    )
    pdp_score = np.stack(
        [power_cosine(value, pdp_truth) for value in pdp_prediction]
    )
    names = [config.name for config in configs]
    np.savez_compressed(
        POINT_CACHE,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=query_site.astype(np.int8),
        config_names=np.asarray(names),
        base_pas=base_pas_score,
        base_pdp=base_pdp_score,
        pas=pas_score,
        pdp=pdp_score,
    )
    report = {
        "experiment": "Round2-E16-boundary-peak-tracks",
        "status": "analysis_only",
        "primary_hypothesis": (
            "Boundary truth and query-query peak continuity can identify persistent moving "
            "paths without blurring path birth/death events."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "test_test_used": True,
        "parameters": {
            "peaks": PEAKS,
            "peak_nms_radius": PEAK_NMS_RADIUS,
            "peak_window_radius": PEAK_WINDOW_RADIUS,
            "spatial_neighbors": SPATIAL_NEIGHBORS,
            "configs": [asdict(config) for config in configs],
        },
        "pas": nested_summary(
            pas_score, base_pas_score, fold_id, query_site, names
        ),
        "pdp": nested_summary(
            pdp_score, base_pdp_score, fold_id, query_site, names
        ),
        "diagnostics": diagnostics,
        "outputs": {
            "point_cache": PATHS.relative(POINT_CACHE),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
