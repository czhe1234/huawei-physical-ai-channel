#!/usr/bin/env python3
"""Audit BS-anchored geometric symmetry and train-region analogs for Round 2."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import resource
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from project_paths import PATHS


EXPERIMENT = "Round2-E166-official-geometry-symmetry-audit"
RESULT_DIR = PATHS.analysis / "results" / "round2_e166_geometry_symmetry"
FIGURE_DIR = PATHS.analysis / "visualizations" / "复赛官方场景几何对称性"
INITIAL_AUDIT_DIR = PATHS.analysis / "results" / "round2_initial_data_audit"
REPORT_PATH = RESULT_DIR / "geometry_symmetry_audit.json"
GLOBAL_CSV = RESULT_DIR / "global_transform_scores.csv"
LOCAL_CANDIDATE_CSV = RESULT_DIR / "test_hole_transform_scores.csv"
LOCAL_BEST_CSV = RESULT_DIR / "test_hole_best_analogs.csv"


@dataclass(frozen=True)
class Config:
    angle_step_deg: int = 2
    same_site_identity_exclusion_deg: float = 15.0
    peak_neighborhood_deg: float = 12.0
    wall_sample_step_m: float = 1.0
    wall_grid_m: float = 1.0
    vertical_normal_z_max: float = 0.25
    minimum_face_vertical_span_m: float = 0.75
    minimum_face_xy_span_m: float = 0.50
    global_support_radius_m: float = 30.0
    local_radii_m: tuple[float, ...] = (30.0, 50.0, 80.0)
    corridor_half_width_m: float = 8.0
    support_distance_m: float = 3.0
    support_coverage_min: float = 0.80
    support_median_max_m: float = 2.5
    local_f2_min: float = 0.45
    local_f5_min: float = 0.70
    corridor_f3_min: float = 0.50
    local_peak_margin_min: float = 0.03
    local_stability_max_m: float = 12.0
    systematic_local_count_min: int = 7
    global_domain_coverage_min: float = 0.75
    global_f2_min: float = 0.50
    global_f5_min: float = 0.75
    global_peak_margin_min: float = 0.03
    global_aligned_hole_count_min: int = 4


@dataclass(frozen=True)
class WallSet:
    xy: np.ndarray
    height: np.ndarray
    tangent: np.ndarray
    orientation_strength: np.ndarray

    def subset(self, selection: np.ndarray) -> "WallSet":
        return WallSet(
            xy=np.asarray(self.xy[selection], dtype=np.float64),
            height=np.asarray(self.height[selection], dtype=np.float64),
            tangent=np.asarray(self.tangent[selection], dtype=np.float64),
            orientation_strength=np.asarray(
                self.orientation_strength[selection], dtype=np.float64
            ),
        )

    def __len__(self) -> int:
        return int(len(self.xy))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle-step-deg", type=int, default=Config.angle_step_deg)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def infer_site_ids(
    train_pos: np.ndarray, test_pos: np.ndarray, bs_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    ordered_y = np.sort(np.asarray(train_pos[:, 1], dtype=np.float64))
    gap_index = int(np.argmax(np.diff(ordered_y)))
    boundary = float((ordered_y[gap_index] + ordered_y[gap_index + 1]) / 2.0)
    site_order = np.argsort(bs_positions[:, 1])
    low_site, high_site = int(site_order[0]), int(site_order[1])

    def assign(position: np.ndarray) -> np.ndarray:
        return np.where(position[:, 1] < boundary, low_site, high_site).astype(np.int8)

    return assign(train_pos), assign(test_pos), boundary


def load_cluster_ids(test_pos: np.ndarray, test_site: np.ndarray) -> np.ndarray:
    path = INITIAL_AUDIT_DIR / "round2_test_point_summary.csv"
    cluster = np.full(len(test_pos), -1, dtype=np.int16)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            if index < 0 or index >= len(test_pos):
                raise ValueError(f"test point index out of range in {path}: {index}")
            if int(row["site"]) != int(test_site[index]):
                raise ValueError(f"site mismatch for test point {index}")
            if not np.allclose(
                test_pos[index, :2],
                [float(row["x"]), float(row["y"])],
                atol=1e-8,
                rtol=0.0,
            ):
                raise ValueError(f"position mismatch for test point {index}")
            cluster[index] = int(row["cluster"])
    if np.any(cluster < 0) or len(np.unique(cluster)) != 10:
        raise ValueError("frozen initial-audit cluster labels are incomplete")
    return cluster


def load_inputs() -> dict[str, Any]:
    train_pos = np.load(PATHS.round2 / "Round2_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round2 / "Round2_Test_Pos.npy").astype(np.float64)
    setup = json.loads(
        (PATHS.round2 / "Round2_Setup.json").read_text(encoding="utf-8")
    )
    bs_positions = np.asarray(setup["X"], dtype=np.float64)
    train_site, test_site, boundary = infer_site_ids(
        train_pos, test_pos, bs_positions
    )
    test_cluster = load_cluster_ids(test_pos, test_site)
    return {
        "train_pos": train_pos,
        "test_pos": test_pos,
        "setup": setup,
        "bs_positions": bs_positions,
        "train_site": train_site,
        "test_site": test_site,
        "test_cluster": test_cluster,
        "site_boundary_y": boundary,
    }


def extract_wall_network(mesh: trimesh.Trimesh, config: Config) -> WallSet:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    vertical = np.abs(normals[:, 2]) <= config.vertical_normal_z_max
    triangles = triangles[vertical]
    if len(triangles) == 0:
        raise RuntimeError("official mesh produced no near-vertical faces")

    pair = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    delta = (
        triangles[:, pair[:, 1], :2] - triangles[:, pair[:, 0], :2]
    )
    length = np.linalg.norm(delta, axis=2)
    longest = np.argmax(length, axis=1)
    row = np.arange(len(triangles), dtype=np.int64)
    left = pair[longest, 0]
    right = pair[longest, 1]
    start = triangles[row, left, :2]
    end = triangles[row, right, :2]
    xy_span = length[row, longest]
    z_min = np.min(triangles[:, :, 2], axis=1)
    z_max = np.max(triangles[:, :, 2], axis=1)
    z_span = z_max - z_min
    keep = (
        (xy_span >= config.minimum_face_xy_span_m)
        & (z_span >= config.minimum_face_vertical_span_m)
    )
    start = start[keep]
    end = end[keep]
    xy_span = xy_span[keep]
    z_min = z_min[keep]
    z_max = z_max[keep]
    if len(start) == 0:
        raise RuntimeError("vertical-face filters removed the complete wall network")

    sampled_xy: list[np.ndarray] = []
    sampled_z_min: list[np.ndarray] = []
    sampled_z_max: list[np.ndarray] = []
    sampled_cos2: list[np.ndarray] = []
    sampled_sin2: list[np.ndarray] = []
    for face_index in range(len(start)):
        count = max(
            2,
            int(math.ceil(xy_span[face_index] / config.wall_sample_step_m)) + 1,
        )
        fraction = np.linspace(0.0, 1.0, count, dtype=np.float64)
        points = (
            start[face_index][None]
            + fraction[:, None] * (end[face_index] - start[face_index])[None]
        )
        direction = end[face_index] - start[face_index]
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        angle = math.atan2(float(direction[1]), float(direction[0]))
        sampled_xy.append(points)
        sampled_z_min.append(np.full(count, z_min[face_index], dtype=np.float64))
        sampled_z_max.append(np.full(count, z_max[face_index], dtype=np.float64))
        sampled_cos2.append(np.full(count, math.cos(2.0 * angle), dtype=np.float64))
        sampled_sin2.append(np.full(count, math.sin(2.0 * angle), dtype=np.float64))

    xy = np.concatenate(sampled_xy, axis=0)
    raw_z_min = np.concatenate(sampled_z_min)
    raw_z_max = np.concatenate(sampled_z_max)
    raw_cos2 = np.concatenate(sampled_cos2)
    raw_sin2 = np.concatenate(sampled_sin2)
    key = np.rint(xy / config.wall_grid_m).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    count = int(np.max(inverse)) + 1
    frequency = np.bincount(inverse, minlength=count).astype(np.float64)
    aggregate_xy = np.column_stack(
        [
            np.bincount(inverse, weights=xy[:, axis], minlength=count)
            / np.maximum(frequency, 1.0)
            for axis in range(2)
        ]
    )
    aggregate_z_min = np.full(count, np.inf, dtype=np.float64)
    aggregate_z_max = np.full(count, -np.inf, dtype=np.float64)
    np.minimum.at(aggregate_z_min, inverse, raw_z_min)
    np.maximum.at(aggregate_z_max, inverse, raw_z_max)
    mean_cos2 = (
        np.bincount(inverse, weights=raw_cos2, minlength=count)
        / np.maximum(frequency, 1.0)
    )
    mean_sin2 = (
        np.bincount(inverse, weights=raw_sin2, minlength=count)
        / np.maximum(frequency, 1.0)
    )
    strength = np.sqrt(mean_cos2**2 + mean_sin2**2)
    angle = 0.5 * np.arctan2(mean_sin2, mean_cos2)
    tangent = np.column_stack((np.cos(angle), np.sin(angle)))
    height = aggregate_z_max - aggregate_z_min
    finite = (
        np.all(np.isfinite(aggregate_xy), axis=1)
        & np.isfinite(height)
        & np.all(np.isfinite(tangent), axis=1)
        & (height >= config.minimum_face_vertical_span_m)
    )
    return WallSet(
        xy=aggregate_xy[finite],
        height=height[finite],
        tangent=tangent[finite],
        orientation_strength=np.clip(strength[finite], 0.0, 1.0),
    )


def angle_matrix(angle_deg: float, mirrored: bool) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    if not mirrored:
        return rotation
    reflection = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    return rotation @ reflection


def transform_xy(
    xy: np.ndarray,
    matrix: np.ndarray,
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
) -> np.ndarray:
    return (np.asarray(xy) - source_anchor[None]) @ matrix.T + target_anchor[None]


def inverse_transform_xy(
    xy: np.ndarray,
    matrix: np.ndarray,
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
) -> np.ndarray:
    return (np.asarray(xy) - target_anchor[None]) @ matrix + source_anchor[None]


def angular_distance(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def nontrivial_transform(
    source_site: int,
    target_site: int,
    angle_deg: float,
    mirrored: bool,
    config: Config,
) -> bool:
    if source_site != target_site or mirrored:
        return True
    return angular_distance(angle_deg, 0.0) >= config.same_site_identity_exclusion_deg


def harmonic_mean(first: float, second: float) -> float:
    total = first + second
    return 0.0 if total <= 0.0 else float(2.0 * first * second / total)


def fraction_within(distance: np.ndarray, threshold: float) -> float:
    if len(distance) == 0:
        return 1.0
    return float(np.mean(np.asarray(distance) <= threshold))


def empty_geometry_metrics() -> dict[str, float]:
    return {
        "f2": 0.0,
        "f3": 0.0,
        "f5": 0.0,
        "orientation": 0.0,
        "height_mae_m": float("inf"),
        "height_score": 0.0,
    }


def compare_wall_sets(
    source: WallSet,
    target: WallSet,
    matrix: np.ndarray,
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
    source_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    source_tree: cKDTree | None = None,
    target_tree: cKDTree | None = None,
) -> dict[str, float]:
    if len(source) == 0 and len(target) == 0:
        return {
            "f2": 1.0,
            "f3": 1.0,
            "f5": 1.0,
            "orientation": 1.0,
            "height_mae_m": 0.0,
            "height_score": 1.0,
        }
    if len(source) == 0 or len(target) == 0:
        return empty_geometry_metrics()

    source_eval = (
        np.ones(len(source), dtype=bool)
        if source_mask is None
        else np.asarray(source_mask, dtype=bool)
    )
    target_eval = (
        np.ones(len(target), dtype=bool)
        if target_mask is None
        else np.asarray(target_mask, dtype=bool)
    )
    if not np.any(source_eval) or not np.any(target_eval):
        return empty_geometry_metrics()

    source_tree = source_tree or cKDTree(source.xy)
    target_tree = target_tree or cKDTree(target.xy)
    transformed_source = transform_xy(
        source.xy, matrix, source_anchor, target_anchor
    )
    inverse_target = inverse_transform_xy(
        target.xy, matrix, source_anchor, target_anchor
    )
    forward_distance, forward_index = target_tree.query(transformed_source, k=1)
    reverse_distance, reverse_index = source_tree.query(inverse_target, k=1)
    forward_distance = np.asarray(forward_distance, dtype=np.float64)[source_eval]
    reverse_distance = np.asarray(reverse_distance, dtype=np.float64)[target_eval]
    forward_index = np.asarray(forward_index, dtype=np.int64)[source_eval]
    reverse_index = np.asarray(reverse_index, dtype=np.int64)[target_eval]

    result: dict[str, float] = {}
    for threshold in (2.0, 3.0, 5.0):
        precision = fraction_within(forward_distance, threshold)
        recall = fraction_within(reverse_distance, threshold)
        result[f"f{int(threshold)}"] = harmonic_mean(precision, recall)

    transformed_tangent = source.tangent @ matrix.T
    source_tangent_eval = transformed_tangent[source_eval]
    target_tangent_eval = target.tangent[target_eval]
    source_strength_eval = source.orientation_strength[source_eval]
    target_strength_eval = target.orientation_strength[target_eval]

    orientation_value: list[np.ndarray] = []
    orientation_weight: list[np.ndarray] = []
    height_error: list[np.ndarray] = []
    forward_match = forward_distance <= 2.0
    if np.any(forward_match):
        target_match = forward_index[forward_match]
        orientation_value.append(
            np.abs(
                np.sum(
                    source_tangent_eval[forward_match]
                    * target.tangent[target_match],
                    axis=1,
                )
            )
        )
        orientation_weight.append(
            np.sqrt(
                source_strength_eval[forward_match]
                * target.orientation_strength[target_match]
            )
        )
        height_error.append(
            np.abs(
                source.height[source_eval][forward_match]
                - target.height[target_match]
            )
        )
    reverse_match = reverse_distance <= 2.0
    if np.any(reverse_match):
        source_match = reverse_index[reverse_match]
        orientation_value.append(
            np.abs(
                np.sum(
                    target_tangent_eval[reverse_match]
                    * transformed_tangent[source_match],
                    axis=1,
                )
            )
        )
        orientation_weight.append(
            np.sqrt(
                target_strength_eval[reverse_match]
                * source.orientation_strength[source_match]
            )
        )
        height_error.append(
            np.abs(
                target.height[target_eval][reverse_match]
                - source.height[source_match]
            )
        )

    if orientation_value:
        value = np.concatenate(orientation_value)
        weight = np.concatenate(orientation_weight)
        result["orientation"] = float(
            np.sum(value * weight) / max(float(np.sum(weight)), 1e-12)
        )
        all_height_error = np.concatenate(height_error)
        result["height_mae_m"] = float(np.median(all_height_error))
        result["height_score"] = float(
            math.exp(-result["height_mae_m"] / 8.0)
        )
    else:
        result["orientation"] = 0.0
        result["height_mae_m"] = float("inf")
        result["height_score"] = 0.0
    return result


def site_wall_sets(
    wall: WallSet,
    positions_by_site: list[np.ndarray],
    config: Config,
) -> list[WallSet]:
    output: list[WallSet] = []
    for positions in positions_by_site:
        distance = cKDTree(positions).query(wall.xy, k=1)[0]
        output.append(wall.subset(distance <= config.global_support_radius_m))
    return output


def search_global_transforms(
    site_walls: list[WallSet],
    positions_by_site: list[np.ndarray],
    bs_positions: np.ndarray,
    config: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    position_trees = [cKDTree(position) for position in positions_by_site]
    wall_trees = [cKDTree(item.xy) for item in site_walls]
    angles = range(0, 360, config.angle_step_deg)

    for source_site in range(len(site_walls)):
        for target_site in range(len(site_walls)):
            pair_rows: list[dict[str, Any]] = []
            source = site_walls[source_site]
            target = site_walls[target_site]
            source_anchor = bs_positions[source_site, :2]
            target_anchor = bs_positions[target_site, :2]
            for mirrored in (False, True):
                for angle_deg in angles:
                    matrix = angle_matrix(angle_deg, mirrored)
                    transformed_source = transform_xy(
                        source.xy, matrix, source_anchor, target_anchor
                    )
                    inverse_target = inverse_transform_xy(
                        target.xy, matrix, source_anchor, target_anchor
                    )
                    source_valid = (
                        position_trees[target_site].query(transformed_source, k=1)[0]
                        <= config.global_support_radius_m
                    )
                    target_valid = (
                        position_trees[source_site].query(inverse_target, k=1)[0]
                        <= config.global_support_radius_m
                    )
                    forward_coverage = float(np.mean(source_valid))
                    reverse_coverage = float(np.mean(target_valid))
                    domain_coverage = float(
                        math.sqrt(forward_coverage * reverse_coverage)
                    )
                    metric = compare_wall_sets(
                        source,
                        target,
                        matrix,
                        source_anchor,
                        target_anchor,
                        source_mask=source_valid,
                        target_mask=target_valid,
                        source_tree=wall_trees[source_site],
                        target_tree=wall_trees[target_site],
                    )
                    score = domain_coverage * (
                        0.45 * metric["f2"]
                        + 0.25 * metric["f5"]
                        + 0.20 * metric["orientation"]
                        + 0.10 * metric["height_score"]
                    )
                    pair_rows.append(
                        {
                            "source_site": source_site,
                            "target_site": target_site,
                            "angle_deg": int(angle_deg),
                            "mirrored": bool(mirrored),
                            "nontrivial": nontrivial_transform(
                                source_site,
                                target_site,
                                angle_deg,
                                mirrored,
                                config,
                            ),
                            "forward_domain_coverage": forward_coverage,
                            "reverse_domain_coverage": reverse_coverage,
                            "domain_coverage": domain_coverage,
                            "wall_f2": metric["f2"],
                            "wall_f5": metric["f5"],
                            "orientation": metric["orientation"],
                            "height_mae_m": metric["height_mae_m"],
                            "height_score": metric["height_score"],
                            "score": float(score),
                            "pair_best": False,
                            "control_p99": None,
                            "peak_margin": None,
                        }
                    )
            eligible = [row for row in pair_rows if row["nontrivial"]]
            if not eligible:
                raise RuntimeError(f"no nontrivial transforms for {source_site}->{target_site}")
            best = max(eligible, key=lambda row: float(row["score"]))
            control = [
                row["score"]
                for row in eligible
                if not (
                    row["mirrored"] == best["mirrored"]
                    and angular_distance(row["angle_deg"], best["angle_deg"])
                    <= config.peak_neighborhood_deg
                )
            ]
            control_p99 = float(np.quantile(control, 0.99)) if control else float(best["score"])
            peak_margin = float(best["score"] - control_p99)
            best["pair_best"] = True
            best["control_p99"] = control_p99
            best["peak_margin"] = peak_margin
            best["base_geometry_pass"] = bool(
                best["domain_coverage"] >= config.global_domain_coverage_min
                and best["wall_f2"] >= config.global_f2_min
                and best["wall_f5"] >= config.global_f5_min
                and peak_margin >= config.global_peak_margin_min
            )
            best_rows.append(dict(best))
            rows.extend(pair_rows)
            print(
                "global",
                f"{source_site}->{target_site}",
                f"angle={best['angle_deg']}",
                f"mirror={int(best['mirrored'])}",
                f"F2={best['wall_f2']:.3f}",
                f"F5={best['wall_f5']:.3f}",
                f"score={best['score']:.3f}",
                f"margin={peak_margin:.3f}",
                flush=True,
            )
    return rows, best_rows


def point_segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    denominator = float(np.dot(vector, vector))
    if denominator <= 1e-12:
        return np.linalg.norm(points - start[None], axis=1)
    fraction = ((points - start[None]) @ vector) / denominator
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None] + fraction[:, None] * vector[None]
    return np.linalg.norm(points - closest, axis=1)


def local_radius_score(metric: dict[str, float]) -> float:
    return float(
        0.55 * metric["f2"]
        + 0.25 * metric["f5"]
        + 0.10 * metric["orientation"]
        + 0.10 * metric["height_score"]
    )


def search_local_analogs(
    wall: WallSet,
    train_pos: np.ndarray,
    train_site: np.ndarray,
    test_pos: np.ndarray,
    test_site: np.ndarray,
    test_cluster: np.ndarray,
    bs_positions: np.ndarray,
    config: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wall_tree = cKDTree(wall.xy)
    train_xy_by_site = [
        train_pos[train_site == site, :2] for site in range(len(bs_positions))
    ]
    train_trees = [cKDTree(item) for item in train_xy_by_site]
    all_rows: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    angles = range(0, 360, config.angle_step_deg)

    for cluster_id in sorted(np.unique(test_cluster).astype(int).tolist()):
        cluster_mask = test_cluster == cluster_id
        cluster_points = test_pos[cluster_mask, :2]
        source_site = int(np.unique(test_site[cluster_mask]).item())
        center = np.mean(cluster_points, axis=0)
        source_anchor = bs_positions[source_site, :2]
        source_patch: dict[int, WallSet] = {}
        source_patch_tree: dict[int, cKDTree | None] = {}
        for radius in config.local_radii_m:
            indices = wall_tree.query_ball_point(center, radius)
            patch = wall.subset(np.asarray(indices, dtype=np.int64))
            radius_key = int(radius)
            source_patch[radius_key] = patch
            source_patch_tree[radius_key] = cKDTree(patch.xy) if len(patch) else None
        source_corridor = wall.subset(
            point_segment_distance(wall.xy, source_anchor, center)
            <= config.corridor_half_width_m
        )
        source_corridor_tree = (
            cKDTree(source_corridor.xy) if len(source_corridor) else None
        )

        cluster_rows: list[dict[str, Any]] = []
        for target_site in range(len(bs_positions)):
            target_anchor = bs_positions[target_site, :2]
            target_train = train_xy_by_site[target_site]
            target_train_tree = train_trees[target_site]
            for mirrored in (False, True):
                for angle_deg in angles:
                    matrix = angle_matrix(angle_deg, mirrored)
                    mapped_points = transform_xy(
                        cluster_points, matrix, source_anchor, target_anchor
                    )
                    mapped_center = transform_xy(
                        center[None], matrix, source_anchor, target_anchor
                    )[0]
                    train_distance = target_train_tree.query(mapped_points, k=1)[0]
                    support_coverage = float(
                        np.mean(train_distance <= config.support_distance_m)
                    )
                    support_median = float(np.median(train_distance))
                    nontrivial = nontrivial_transform(
                        source_site,
                        target_site,
                        angle_deg,
                        mirrored,
                        config,
                    )
                    eligible = bool(
                        nontrivial
                        and support_coverage >= config.support_coverage_min
                        and support_median <= config.support_median_max_m
                    )
                    row: dict[str, Any] = {
                        "cluster": cluster_id,
                        "source_site": source_site,
                        "target_site": target_site,
                        "angle_deg": int(angle_deg),
                        "mirrored": bool(mirrored),
                        "nontrivial": nontrivial,
                        "eligible": eligible,
                        "mapped_center_x": float(mapped_center[0]),
                        "mapped_center_y": float(mapped_center[1]),
                        "support_coverage3": support_coverage,
                        "support_median_m": support_median,
                        "score": None,
                    }
                    if eligible:
                        radius_metrics: dict[int, dict[str, float]] = {}
                        for radius in config.local_radii_m:
                            radius_key = int(radius)
                            target_indices = wall_tree.query_ball_point(
                                mapped_center, radius
                            )
                            target_patch = wall.subset(
                                np.asarray(target_indices, dtype=np.int64)
                            )
                            metric = compare_wall_sets(
                                source_patch[radius_key],
                                target_patch,
                                matrix,
                                source_anchor,
                                target_anchor,
                                source_tree=source_patch_tree[radius_key],
                                target_tree=(
                                    cKDTree(target_patch.xy) if len(target_patch) else None
                                ),
                            )
                            radius_metrics[radius_key] = metric
                            for name, value in metric.items():
                                row[f"r{radius_key}_{name}"] = value
                            row[f"r{radius_key}_score"] = local_radius_score(metric)

                        target_corridor = wall.subset(
                            point_segment_distance(
                                wall.xy, target_anchor, mapped_center
                            )
                            <= config.corridor_half_width_m
                        )
                        corridor_metric = compare_wall_sets(
                            source_corridor,
                            target_corridor,
                            matrix,
                            source_anchor,
                            target_anchor,
                            source_tree=source_corridor_tree,
                            target_tree=(
                                cKDTree(target_corridor.xy)
                                if len(target_corridor)
                                else None
                            ),
                        )
                        for name, value in corridor_metric.items():
                            row[f"corridor_{name}"] = value
                        row["score"] = float(
                            0.15 * radius_metrics[30]["f2"]
                            + 0.30 * radius_metrics[50]["f2"]
                            + 0.10 * radius_metrics[80]["f2"]
                            + 0.15 * radius_metrics[50]["f5"]
                            + 0.10 * corridor_metric["f3"]
                            + 0.10 * radius_metrics[50]["orientation"]
                            + 0.10 * radius_metrics[50]["height_score"]
                        )
                    cluster_rows.append(row)

        eligible_rows = [
            row
            for row in cluster_rows
            if row["eligible"] and row["score"] is not None
        ]
        if eligible_rows:
            best = max(eligible_rows, key=lambda row: float(row["score"]))
            control = [
                float(row["score"])
                for row in eligible_rows
                if not (
                    row["target_site"] == best["target_site"]
                    and row["mirrored"] == best["mirrored"]
                    and angular_distance(row["angle_deg"], best["angle_deg"])
                    <= config.peak_neighborhood_deg
                )
            ]
            control_p99 = (
                float(np.quantile(control, 0.99))
                if control
                else float(best["score"])
            )
            peak_margin = float(best["score"] - control_p99)
            radius_winners: dict[str, dict[str, Any]] = {}
            winner_centers: list[np.ndarray] = []
            for radius in config.local_radii_m:
                key = f"r{int(radius)}_score"
                winner = max(eligible_rows, key=lambda row: float(row[key]))
                radius_winners[str(int(radius))] = {
                    "target_site": int(winner["target_site"]),
                    "angle_deg": int(winner["angle_deg"]),
                    "mirrored": bool(winner["mirrored"]),
                    "mapped_center": [
                        float(winner["mapped_center_x"]),
                        float(winner["mapped_center_y"]),
                    ],
                    "score": float(winner[key]),
                }
                winner_centers.append(
                    np.asarray(
                        [winner["mapped_center_x"], winner["mapped_center_y"]],
                        dtype=np.float64,
                    )
                )
            stability = 0.0
            for left in range(len(winner_centers)):
                for right in range(left + 1, len(winner_centers)):
                    stability = max(
                        stability,
                        float(np.linalg.norm(winner_centers[left] - winner_centers[right])),
                    )
            robust = bool(
                best["r50_f2"] >= config.local_f2_min
                and best["r50_f5"] >= config.local_f5_min
                and best["corridor_f3"] >= config.corridor_f3_min
                and peak_margin >= config.local_peak_margin_min
                and stability <= config.local_stability_max_m
            )
            best_record = {
                **dict(best),
                "control_p99": control_p99,
                "peak_margin": peak_margin,
                "radius_winners": radius_winners,
                "radius_winner_center_max_distance_m": stability,
                "robust_geometry_analog": robust,
                "eligible_transform_count": len(eligible_rows),
            }
        else:
            best_record = {
                "cluster": cluster_id,
                "source_site": source_site,
                "eligible_transform_count": 0,
                "robust_geometry_analog": False,
                "reason": "no_nontrivial_transform_met_training_support_gate",
            }
        best_records.append(best_record)
        all_rows.extend(cluster_rows)
        print(
            "local",
            f"cluster={cluster_id}",
            f"eligible={best_record['eligible_transform_count']}",
            (
                f"target={best_record.get('target_site')} "
                f"angle={best_record.get('angle_deg')} "
                f"mirror={int(bool(best_record.get('mirrored', False)))} "
                f"F2={best_record.get('r50_f2', float('nan')):.3f} "
                f"F5={best_record.get('r50_f5', float('nan')):.3f} "
                f"margin={best_record.get('peak_margin', float('nan')):.3f} "
                f"robust={int(best_record['robust_geometry_analog'])}"
            ),
            flush=True,
        )
    return all_rows, best_records


def finalize_global_decision(
    global_best: list[dict[str, Any]],
    local_best: list[dict[str, Any]],
    config: Config,
) -> bool:
    any_pass = False
    for global_row in global_best:
        aligned = [
            int(local["cluster"])
            for local in local_best
            if local.get("robust_geometry_analog", False)
            and int(local["source_site"]) == int(global_row["source_site"])
            and int(local["target_site"]) == int(global_row["target_site"])
            and bool(local["mirrored"]) == bool(global_row["mirrored"])
            and int(local["angle_deg"]) == int(global_row["angle_deg"])
        ]
        global_row["aligned_robust_holes"] = aligned
        global_row["aligned_robust_hole_count"] = len(aligned)
        global_row["global_symmetry_pass"] = bool(
            global_row.get("base_geometry_pass", False)
            and len(aligned) >= config.global_aligned_hole_count_min
        )
        any_pass = any_pass or global_row["global_symmetry_pass"]
    return any_pass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (dict, list, tuple)):
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(
    wall: WallSet,
    train_pos: np.ndarray,
    train_site: np.ndarray,
    test_pos: np.ndarray,
    test_cluster: np.ndarray,
    bs_positions: np.ndarray,
) -> Path:
    path = FIGURE_DIR / "01_官方墙网与十个测试洞.png"
    figure, axis = plt.subplots(figsize=(10.5, 13.0))
    color_value = np.clip(wall.height, 0.0, np.quantile(wall.height, 0.98))
    axis.scatter(
        wall.xy[:, 0],
        wall.xy[:, 1],
        c=color_value,
        cmap="Greys",
        s=2.0,
        alpha=0.45,
        linewidths=0,
        label="官方PLY竖直墙网",
    )
    site_colors = ("#4C78A8", "#F58518")
    for site in range(len(bs_positions)):
        local = train_site == site
        axis.scatter(
            train_pos[local, 0],
            train_pos[local, 1],
            s=5,
            color=site_colors[site],
            alpha=0.16,
            linewidths=0,
            label=f"基站{site}训练点",
        )
    cluster_colors = plt.get_cmap("tab10")
    for color_index, cluster_id in enumerate(sorted(np.unique(test_cluster))):
        local = test_cluster == cluster_id
        axis.scatter(
            test_pos[local, 0],
            test_pos[local, 1],
            s=20,
            color=cluster_colors(color_index),
            edgecolors="white",
            linewidths=0.35,
            alpha=0.90,
        )
        center = np.mean(test_pos[local, :2], axis=0)
        axis.text(
            center[0],
            center[1],
            str(int(cluster_id)),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
        )
    axis.scatter(
        bs_positions[:, 0],
        bs_positions[:, 1],
        marker="*",
        s=270,
        color="#F2C14E",
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
        label="基站",
    )
    position = np.concatenate([train_pos[:, :2], test_pos[:, :2]], axis=0)
    lower = np.min(position, axis=0) - 45.0
    upper = np.max(position, axis=0) + 45.0
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X（米）")
    axis.set_ylabel("Y（米）")
    axis.set_title("官方三维地图竖直墙网、训练点与十个测试空间洞")
    axis.grid(alpha=0.16)
    axis.legend(loc="upper left", fontsize=8, frameon=True)
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_global_overlays(
    site_walls: list[WallSet],
    global_best: list[dict[str, Any]],
    positions_by_site: list[np.ndarray],
    train_pos: np.ndarray,
    train_site: np.ndarray,
    test_pos: np.ndarray,
    test_site: np.ndarray,
    bs_positions: np.ndarray,
) -> Path:
    path = FIGURE_DIR / "02_四组基站锚定最佳全局变换叠加.png"
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 13.5))
    for axis, record in zip(axes.flat, global_best, strict=True):
        source_site = int(record["source_site"])
        target_site = int(record["target_site"])
        matrix = angle_matrix(record["angle_deg"], record["mirrored"])
        transformed = transform_xy(
            site_walls[source_site].xy,
            matrix,
            bs_positions[source_site, :2],
            bs_positions[target_site, :2],
        )
        axis.scatter(
            site_walls[target_site].xy[:, 0],
            site_walls[target_site].xy[:, 1],
            s=3,
            color="#4D4D4D",
            alpha=0.38,
            linewidths=0,
            label="目标区域墙网",
        )
        axis.scatter(
            transformed[:, 0],
            transformed[:, 1],
            s=3,
            color="#D1495B",
            alpha=0.40,
            linewidths=0,
            label="变换后的源墙网",
        )
        local_train = train_site == target_site
        axis.scatter(
            train_pos[local_train, 0],
            train_pos[local_train, 1],
            s=4,
            color="#2A9D8F",
            alpha=0.12,
            linewidths=0,
            label="目标站训练点",
        )
        source_test = test_site == source_site
        mapped_test = transform_xy(
            test_pos[source_test, :2],
            matrix,
            bs_positions[source_site, :2],
            bs_positions[target_site, :2],
        )
        axis.scatter(
            mapped_test[:, 0],
            mapped_test[:, 1],
            s=10,
            facecolors="none",
            edgecolors="#0077B6",
            linewidths=0.55,
            alpha=0.75,
            label="映射后的源测试洞",
        )
        axis.scatter(
            bs_positions[target_site, 0],
            bs_positions[target_site, 1],
            marker="*",
            s=170,
            color="#F2C14E",
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )
        target_position = positions_by_site[target_site]
        lower = np.min(target_position, axis=0) - 38.0
        upper = np.max(target_position, axis=0) + 38.0
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.14)
        axis.set_title(
            f"站{source_site} -> 站{target_site} | "
            f"角度{record['angle_deg']}° | 镜像{int(record['mirrored'])}\n"
            f"F2={record['wall_f2']:.3f}, F5={record['wall_f5']:.3f}, "
            f"覆盖={record['domain_coverage']:.3f}, margin={record['peak_margin']:.3f}"
        )
    axes[0, 0].legend(loc="upper left", fontsize=7, frameon=True)
    figure.suptitle("以基站为锚点的四组最佳全局旋转/镜像叠加", fontsize=16)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_global_angle_scan(
    global_rows: list[dict[str, Any]], global_best: list[dict[str, Any]]
) -> Path:
    path = FIGURE_DIR / "03_全局对称角度扫描曲线.png"
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), sharex=True, sharey=True)
    pair_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for axis, pair, best in zip(axes.flat, pair_order, global_best, strict=True):
        source_site, target_site = pair
        pair_rows = [
            row
            for row in global_rows
            if row["source_site"] == source_site
            and row["target_site"] == target_site
        ]
        for mirrored, color, label in (
            (False, "#4C78A8", "旋转"),
            (True, "#D1495B", "镜像+旋转"),
        ):
            local = sorted(
                [row for row in pair_rows if row["mirrored"] == mirrored],
                key=lambda row: row["angle_deg"],
            )
            axis.plot(
                [row["angle_deg"] for row in local],
                [row["score"] for row in local],
                color=color,
                linewidth=1.5,
                label=label,
            )
        axis.axvline(best["angle_deg"], color="black", linewidth=1.0, linestyle="--")
        axis.axhline(
            best["control_p99"], color="#666666", linewidth=1.0, linestyle=":"
        )
        axis.set_title(
            f"站{source_site} -> 站{target_site} | "
            f"赢家{best['angle_deg']}° / 镜像{int(best['mirrored'])}"
        )
        axis.set_xlim(0, 358)
        axis.grid(alpha=0.20)
        axis.set_xlabel("角度（度）")
        axis.set_ylabel("全局几何综合分")
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.suptitle("全局BS锚定刚体变换扫描：尖峰还是宽泛偶合", fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_local_overlays(
    wall: WallSet,
    local_best: list[dict[str, Any]],
    train_pos: np.ndarray,
    train_site: np.ndarray,
    test_pos: np.ndarray,
    test_cluster: np.ndarray,
    bs_positions: np.ndarray,
) -> Path:
    path = FIGURE_DIR / "04_十个测试洞最佳训练区几何叠加.png"
    wall_tree = cKDTree(wall.xy)
    figure, axes = plt.subplots(5, 2, figsize=(14.0, 28.0))
    for axis, record in zip(axes.flat, local_best, strict=True):
        cluster_id = int(record["cluster"])
        if record.get("eligible_transform_count", 0) == 0:
            axis.text(0.5, 0.5, "没有通过训练覆盖门的非恒等变换", ha="center", va="center")
            axis.set_title(f"测试洞 {cluster_id}")
            axis.set_axis_off()
            continue
        source_site = int(record["source_site"])
        target_site = int(record["target_site"])
        matrix = angle_matrix(record["angle_deg"], record["mirrored"])
        source_points = test_pos[test_cluster == cluster_id, :2]
        source_center = np.mean(source_points, axis=0)
        mapped_points = transform_xy(
            source_points,
            matrix,
            bs_positions[source_site, :2],
            bs_positions[target_site, :2],
        )
        mapped_center = np.asarray(
            [record["mapped_center_x"], record["mapped_center_y"]], dtype=np.float64
        )
        source_indices = wall_tree.query_ball_point(source_center, 50.0)
        target_indices = wall_tree.query_ball_point(mapped_center, 50.0)
        source_wall = wall.subset(np.asarray(source_indices, dtype=np.int64))
        target_wall = wall.subset(np.asarray(target_indices, dtype=np.int64))
        transformed_source_wall = transform_xy(
            source_wall.xy,
            matrix,
            bs_positions[source_site, :2],
            bs_positions[target_site, :2],
        )
        target_train = train_pos[train_site == target_site, :2]
        nearby_train = (
            np.linalg.norm(target_train - mapped_center[None], axis=1) <= 55.0
        )
        axis.scatter(
            target_wall.xy[:, 0],
            target_wall.xy[:, 1],
            s=8,
            color="#4D4D4D",
            alpha=0.50,
            linewidths=0,
            label="目标训练区墙网",
        )
        axis.scatter(
            transformed_source_wall[:, 0],
            transformed_source_wall[:, 1],
            s=8,
            color="#D1495B",
            alpha=0.50,
            linewidths=0,
            label="变换后的测试洞墙网",
        )
        axis.scatter(
            target_train[nearby_train, 0],
            target_train[nearby_train, 1],
            s=8,
            color="#2A9D8F",
            alpha=0.35,
            linewidths=0,
            label="目标站训练点",
        )
        axis.scatter(
            mapped_points[:, 0],
            mapped_points[:, 1],
            s=20,
            facecolors="#0077B6",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.90,
            label="映射后的测试点",
        )
        axis.set_xlim(mapped_center[0] - 55.0, mapped_center[0] + 55.0)
        axis.set_ylim(mapped_center[1] - 55.0, mapped_center[1] + 55.0)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.16)
        status = "通过稳健门" if record["robust_geometry_analog"] else "未通过稳健门"
        axis.set_title(
            f"洞{cluster_id} -> 站{target_site} | {record['angle_deg']}° | "
            f"镜像{int(record['mirrored'])} | {status}\n"
            f"F2/F5={record['r50_f2']:.3f}/{record['r50_f5']:.3f}, "
            f"走廊F3={record['corridor_f3']:.3f}, "
            f"训练覆盖={record['support_coverage3']:.2f}, margin={record['peak_margin']:.3f}",
            fontsize=10,
        )
    axes[0, 0].legend(loc="upper left", fontsize=7, frameon=True)
    figure.suptitle("十个测试洞各自最佳的BS锚定训练区几何对应", fontsize=16)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    figure.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_local_thresholds(local_best: list[dict[str, Any]], config: Config) -> Path:
    path = FIGURE_DIR / "05_逐洞几何类比指标与事前门槛.png"
    labels = [str(int(record["cluster"])) for record in local_best]
    robust_color = [
        "#2A9D8F" if record.get("robust_geometry_analog", False) else "#D1495B"
        for record in local_best
    ]

    def values(key: str, default: float = 0.0) -> list[float]:
        return [float(record.get(key, default)) for record in local_best]

    panels = (
        ("support_coverage3", config.support_coverage_min, "训练点3米覆盖率"),
        ("r50_f2", config.local_f2_min, "50米墙网F2"),
        ("r50_f5", config.local_f5_min, "50米墙网F5"),
        ("corridor_f3", config.corridor_f3_min, "BS-洞走廊F3"),
        ("peak_margin", config.local_peak_margin_min, "独特峰值margin"),
        (
            "radius_winner_center_max_distance_m",
            config.local_stability_max_m,
            "30/50/80米赢家中心最大偏移",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.5))
    for axis, (key, threshold, title) in zip(axes.flat, panels, strict=True):
        data = values(key)
        axis.bar(labels, data, color=robust_color, alpha=0.88)
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1.1)
        axis.set_title(title)
        axis.set_xlabel("测试洞编号")
        axis.grid(axis="y", alpha=0.20)
        if key != "radius_winner_center_max_distance_m":
            axis.set_ylim(bottom=0.0)
    figure.suptitle("绿色表示通过全部事前门槛，红色表示至少一项失败", fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def relative_path(path: Path) -> str:
    return PATHS.relative(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    config = Config(angle_step_deg=args.angle_step_deg)
    if 360 % config.angle_step_deg != 0:
        raise ValueError("angle step must divide 360 exactly")
    configure_matplotlib()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timing: dict[str, float] = {}

    phase = time.perf_counter()
    inputs = load_inputs()
    mesh = trimesh.load(
        PATHS.round2 / "Round2_Map.ply", process=False, force="mesh"
    )
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(mesh).__name__}")
    wall = extract_wall_network(mesh, config)
    timing["load_and_wall_extraction_seconds"] = time.perf_counter() - phase

    train_pos = inputs["train_pos"]
    test_pos = inputs["test_pos"]
    train_site = inputs["train_site"]
    test_site = inputs["test_site"]
    test_cluster = inputs["test_cluster"]
    bs_positions = inputs["bs_positions"]
    positions_by_site = [
        np.concatenate(
            [
                train_pos[train_site == site, :2],
                test_pos[test_site == site, :2],
            ],
            axis=0,
        )
        for site in range(len(bs_positions))
    ]
    site_walls = site_wall_sets(wall, positions_by_site, config)
    print(
        f"wall network: {len(wall)} samples; site supports "
        + "/".join(str(len(item)) for item in site_walls),
        flush=True,
    )

    phase = time.perf_counter()
    global_rows, global_best = search_global_transforms(
        site_walls, positions_by_site, bs_positions, config
    )
    timing["global_search_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    local_rows, local_best = search_local_analogs(
        wall,
        train_pos,
        train_site,
        test_pos,
        test_site,
        test_cluster,
        bs_positions,
        config,
    )
    timing["local_search_seconds"] = time.perf_counter() - phase

    global_pass = finalize_global_decision(global_best, local_best, config)
    robust_count = int(
        sum(bool(record.get("robust_geometry_analog", False)) for record in local_best)
    )
    systematic_local_pass = robust_count >= config.systematic_local_count_min
    if global_pass and systematic_local_pass:
        conclusion = "global_and_systematic_local_bs_anchored_geometry_symmetry_found"
    elif global_pass:
        conclusion = "global_symmetry_found_without_systematic_independent_local_analogs"
    elif systematic_local_pass:
        conclusion = "systematic_local_analogs_found_without_one_global_symmetry"
    elif robust_count:
        conclusion = "only_isolated_local_geometry_analogs_found"
    else:
        conclusion = "no_actionable_bs_anchored_geometry_symmetry"

    phase = time.perf_counter()
    write_csv(GLOBAL_CSV, global_rows)
    write_csv(LOCAL_CANDIDATE_CSV, local_rows)
    write_csv(LOCAL_BEST_CSV, local_best)
    figure_paths = [
        plot_overview(
            wall,
            train_pos,
            train_site,
            test_pos,
            test_cluster,
            bs_positions,
        ),
        plot_global_overlays(
            site_walls,
            global_best,
            positions_by_site,
            train_pos,
            train_site,
            test_pos,
            test_site,
            bs_positions,
        ),
        plot_global_angle_scan(global_rows, global_best),
        plot_local_overlays(
            wall,
            local_best,
            train_pos,
            train_site,
            test_pos,
            test_cluster,
            bs_positions,
        ),
        plot_local_thresholds(local_best, config),
    ]
    timing["output_and_plot_seconds"] = time.perf_counter() - phase
    total_seconds = time.perf_counter() - started
    timing["total_seconds"] = total_seconds
    measured_bottleneck = max(
        (key for key in timing if key != "total_seconds"), key=timing.get
    )
    peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

    report = {
        "experiment": EXPERIMENT,
        "status": (
            "analysis_complete_geometry_symmetry_found"
            if global_pass or systematic_local_pass
            else "analysis_complete_geometry_symmetry_gate_failed"
        ),
        "type": "diagnostic_only",
        "official_inputs_only": True,
        "train_channel_values_read": False,
        "sionna_used": False,
        "v12_used": False,
        "model_trained": False,
        "submission_generated": False,
        "final_score_goal": {
            "fixed_500_target": 0.6775208075,
            "online_target": 0.70,
            "official_score_computed_this_experiment": False,
        },
        "config": asdict(config),
        "inputs": {
            "map": relative_path(PATHS.round2 / "Round2_Map.ply"),
            "train_positions": relative_path(
                PATHS.round2 / "Round2_Train_Pos.npy"
            ),
            "test_positions": relative_path(PATHS.round2 / "Round2_Test_Pos.npy"),
            "setup": relative_path(PATHS.round2 / "Round2_Setup.json"),
            "frozen_cluster_labels": relative_path(
                INITIAL_AUDIT_DIR / "round2_test_point_summary.csv"
            ),
        },
        "wall_network": {
            "sample_count": len(wall),
            "site_support_sample_count": [len(item) for item in site_walls],
            "height_median_m": float(np.median(wall.height)),
            "height_p90_m": float(np.quantile(wall.height, 0.90)),
            "orientation_strength_median": float(
                np.median(wall.orientation_strength)
            ),
        },
        "global_search": {
            "transform_count": len(global_rows),
            "best_by_site_pair": global_best,
            "global_symmetry_pass": global_pass,
        },
        "local_analogs": {
            "transform_count": len(local_rows),
            "best_by_test_cluster": local_best,
            "robust_geometry_analog_count": robust_count,
            "systematic_local_analog_pass": systematic_local_pass,
        },
        "decision": {
            "conclusion": conclusion,
            "global_symmetry_pass": global_pass,
            "systematic_local_analog_pass": systematic_local_pass,
            "robust_local_count": robust_count,
            "interpretation_if_failed": (
                "The official sampling layout can support shape-matched pseudo holes, "
                "but BS-anchored wall position/orientation/height and corridor geometry "
                "do not provide a systematic symmetry for copying propagation state."
            ),
        },
        "performance": {
            **timing,
            "selected_gpu": "none_cpu_only",
            "peak_rss_mb": peak_rss_mb,
            "measured_bottleneck": measured_bottleneck,
        },
        "outputs": {
            "report": relative_path(REPORT_PATH),
            "global_scores_csv": relative_path(GLOBAL_CSV),
            "local_candidates_csv": relative_path(LOCAL_CANDIDATE_CSV),
            "local_best_csv": relative_path(LOCAL_BEST_CSV),
            "figures": [relative_path(path) for path in figure_paths],
        },
    }
    REPORT_PATH.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report["decision"]), ensure_ascii=False, indent=2))
    print(f"report: {relative_path(REPORT_PATH)}", flush=True)


if __name__ == "__main__":
    main()
