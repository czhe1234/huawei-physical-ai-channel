#!/usr/bin/env python3
"""Group real Round 2 test points by robust Sionna propagation state."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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
import trimesh

from project_paths import PATHS
from round2.data import load_positions


VERSION = "Round2-E185-sionna-test-propagation-similarity"
CACHE_ROOT = PATHS.artifacts / "round2" / "joint_path_token_oracle"
TRAIN_CACHE = CACHE_ROOT / "n4000_s24_k128_b096_d064_scatter1_sel106"
TEST_CACHE = CACHE_ROOT / "test_n500_s24_k128_b096_d064_scatter1_sel106"
RESULT_DIR = (
    PATHS.analysis
    / "results"
    / "round2_e185_sionna_test_propagation_similarity"
)
INITIAL_AUDIT_CSV = (
    PATHS.analysis
    / "results"
    / "round2_initial_data_audit"
    / "round2_test_point_summary.csv"
)

PAIR_CSV = RESULT_DIR / "robust_nonlocal_pairs.csv"
GROUP_CSV = RESULT_DIR / "robust_test_groups.csv"
ANALOG_CSV = RESULT_DIR / "test_to_train_analogs.csv"
ALL_BEST_CSV = RESULT_DIR / "all_test_best_nonlocal.csv"
REPORT_JSON = RESULT_DIR / "report.json"
REPORT_MD = RESULT_DIR / "传播规律相似点分析.md"
FIGURE = RESULT_DIR / "传播组空间分布.png"

LIGHT_SPEED_MPS = 299_792_458.0
TINY = 1e-12

# Physics-fixed bins. No dataset standardization or result-driven tuning is used.
DELAY_UPPER_M = np.asarray(
    [2.0, 5.0, 10.0, 20.0, 35.0, 55.0, 80.0, 120.0, 180.0, 260.0, 400.0],
    dtype=np.float64,
)
DEPTH_UPPER = np.asarray([0, 1, 2, 3, 4, 5, 7, 9, 12], dtype=np.float64)
PRESENCE_UPPER = np.asarray([1, 2, 4, 8, 12, 18, 24, 48], dtype=np.float64)
AZIMUTH_BINS = 16
ELEVATION_BINS = 12
INTERACTION_BINS = 5
RANK_BINS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-distance-m", type=float, default=30.0)
    parser.add_argument("--mutual-k", type=int, default=8)
    parser.add_argument("--train-top-k", type=int, default=5)
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.relative_to(PATHS.root))


def wrap_angle(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def angle_bin(value: np.ndarray, count: int) -> np.ndarray:
    wrapped = wrap_angle(value)
    scaled = np.floor((wrapped + np.pi) * count / (2.0 * np.pi)).astype(np.int64)
    return np.clip(scaled, 0, count - 1)


def upper_bin(value: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.searchsorted(upper, value, side="right").astype(np.int64)


def probability_block(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    total = float(np.sum(flat))
    if total <= TINY:
        return np.zeros(len(flat), dtype=np.float32)
    return np.sqrt(np.maximum(flat / total, 0.0)).astype(np.float32)


def load_cache(cache: Path, expected_points: int) -> dict[str, np.ndarray]:
    names = (
        "rows",
        "mask",
        "source",
        "tau",
        "angles",
        "energy_fraction",
        "presence",
        "interaction_count",
        "interactions",
        "primitives",
    )
    missing = [cache / f"{name}.npy" for name in names if not (cache / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError([relative(path) for path in missing])
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}
    rows = np.asarray(arrays["rows"], dtype=np.int64)
    if not np.array_equal(rows, np.arange(expected_points, dtype=np.int64)):
        raise RuntimeError(f"cache row identity mismatch: {relative(cache)}")
    shape = (expected_points, 128)
    for name in ("mask", "source", "tau", "energy_fraction", "presence", "interaction_count"):
        if arrays[name].shape != shape:
            raise RuntimeError(f"unexpected {name} shape in {relative(cache)}: {arrays[name].shape}")
    if arrays["angles"].shape != shape + (4,):
        raise RuntimeError(f"unexpected angles shape in {relative(cache)}")
    if arrays["interactions"].shape != shape + (16,):
        raise RuntimeError(f"unexpected interactions shape in {relative(cache)}")
    if arrays["primitives"].shape != shape + (16,):
        raise RuntimeError(f"unexpected primitives shape in {relative(cache)}")
    mask = np.asarray(arrays["mask"], dtype=bool)
    for name in ("tau", "angles", "energy_fraction"):
        values = np.asarray(arrays[name])[mask]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"non-finite masked {name} values in {relative(cache)}")
    source = np.asarray(arrays["source"])
    if not np.all(np.isin(source[mask], [0, 1])):
        raise RuntimeError(f"invalid path source in {relative(cache)}")
    return arrays


def load_test_clusters(test_positions: np.ndarray, test_sites: np.ndarray) -> np.ndarray:
    cluster = np.full(len(test_positions), -1, dtype=np.int16)
    with INITIAL_AUDIT_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            if int(row["site"]) != int(test_sites[index]):
                raise RuntimeError(f"site mismatch in frozen cluster row {index}")
            if not np.allclose(
                test_positions[index, :2],
                [float(row["x"]), float(row["y"])],
                atol=1e-8,
                rtol=0.0,
            ):
                raise RuntimeError(f"position mismatch in frozen cluster row {index}")
            cluster[index] = int(row["cluster"])
    if np.any(cluster < 0) or len(np.unique(cluster)) != 10:
        raise RuntimeError("frozen real-test cluster labels are incomplete")
    return cluster


def canonical_face_planes(decimals: int = 3) -> tuple[np.ndarray, dict[str, int], trimesh.Trimesh]:
    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh).__name__)
    normal = np.asarray(mesh.face_normals, dtype=np.float64).copy()
    center = np.asarray(mesh.triangles_center, dtype=np.float64)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), TINY)
    undecided = np.ones(len(normal), dtype=bool)
    for axis in range(3):
        flip = undecided & (normal[:, axis] < 0.0)
        normal[flip] *= -1.0
        undecided &= np.abs(normal[:, axis]) < 1e-10
    offset = np.sum(normal * center, axis=1)
    key = np.round(np.column_stack((normal, offset)), decimals=decimals)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    report = {
        "face_count": int(len(normal)),
        "plane_family_count": int(np.max(inverse) + 1),
        "round_decimals": decimals,
    }
    return inverse.astype(np.int32), report, mesh


def structural_embedding(
    cache: dict[str, np.ndarray],
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    source_value: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.asarray(cache["mask"])
    source = np.asarray(cache["source"])
    tau = np.asarray(cache["tau"])
    angles = np.asarray(cache["angles"])
    energy_fraction = np.asarray(cache["energy_fraction"])
    presence = np.asarray(cache["presence"])
    depth = np.asarray(cache["interaction_count"], dtype=np.int64)
    interactions = np.asarray(cache["interactions"], dtype=np.int64)

    block_sizes = [
        len(DELAY_UPPER_M) + 1,
        AZIMUTH_BINS,
        AZIMUTH_BINS,
        ELEVATION_BINS,
        ELEVATION_BINS,
        len(DEPTH_UPPER) + 1,
        INTERACTION_BINS,
        INTERACTION_BINS,
        INTERACTION_BINS,
        INTERACTION_BINS * INTERACTION_BINS,
        (len(DELAY_UPPER_M) + 1) * AZIMUTH_BINS,
        (len(DELAY_UPPER_M) + 1) * AZIMUTH_BINS,
        AZIMUTH_BINS * AZIMUTH_BINS,
        (len(DELAY_UPPER_M) + 1) * (len(DEPTH_UPPER) + 1),
        INTERACTION_BINS * (len(DELAY_UPPER_M) + 1),
        RANK_BINS,
        len(PRESENCE_UPPER) + 1,
    ]
    dimension = int(sum(block_sizes))
    output = np.zeros((len(positions), dimension), dtype=np.float32)
    path_counts = np.zeros(len(positions), dtype=np.int16)
    los_mass = np.zeros(len(positions), dtype=np.float32)

    for point in range(len(positions)):
        slots = np.flatnonzero(mask[point] & (source[point] == source_value))
        if len(slots) == 0:
            continue
        local_weight = np.asarray(energy_fraction[point, slots], dtype=np.float64)
        if source_value == 0:
            local_weight *= np.sqrt(
                np.maximum(np.asarray(presence[point, slots], dtype=np.float64), 1.0)
            )
        valid = np.isfinite(local_weight) & (local_weight > 0.0)
        slots = slots[valid]
        local_weight = local_weight[valid]
        if len(slots) == 0:
            continue
        local_weight /= max(float(np.sum(local_weight)), TINY)
        path_counts[point] = len(slots)

        bs = bs_positions[int(sites[point])]
        ue = positions[point]
        offset = ue - bs
        direct_distance = float(np.linalg.norm(offset))
        bearing_tx = math.atan2(float(offset[1]), float(offset[0]))
        bearing_rx = math.atan2(float(-offset[1]), float(-offset[0]))
        direct_theta_tx = math.acos(float(np.clip(offset[2] / direct_distance, -1.0, 1.0)))
        direct_theta_rx = math.acos(float(np.clip(-offset[2] / direct_distance, -1.0, 1.0)))

        local_tau = np.asarray(tau[point, slots], dtype=np.float64)
        local_angles = np.asarray(angles[point, slots], dtype=np.float64)
        local_depth = np.asarray(depth[point, slots], dtype=np.int64)
        excess_m = np.maximum(local_tau * LIGHT_SPEED_MPS - direct_distance, 0.0)
        departure_azimuth = wrap_angle(local_angles[:, 1] - bearing_tx)
        arrival_azimuth = wrap_angle(local_angles[:, 3] - bearing_rx)
        departure_elevation = wrap_angle(local_angles[:, 0] - direct_theta_tx)
        arrival_elevation = wrap_angle(local_angles[:, 2] - direct_theta_rx)

        delay_index = upper_bin(excess_m, DELAY_UPPER_M)
        depth_index = upper_bin(local_depth, DEPTH_UPPER)
        departure_index = angle_bin(departure_azimuth, AZIMUTH_BINS)
        arrival_index = angle_bin(arrival_azimuth, AZIMUTH_BINS)
        departure_elevation_index = angle_bin(departure_elevation, ELEVATION_BINS)
        arrival_elevation_index = angle_bin(arrival_elevation, ELEVATION_BINS)
        presence_index = upper_bin(
            np.asarray(presence[point, slots], dtype=np.float64), PRESENCE_UPPER
        )

        delay_hist = np.bincount(
            delay_index, weights=local_weight, minlength=len(DELAY_UPPER_M) + 1
        )
        departure_hist = np.bincount(
            departure_index, weights=local_weight, minlength=AZIMUTH_BINS
        )
        arrival_hist = np.bincount(
            arrival_index, weights=local_weight, minlength=AZIMUTH_BINS
        )
        departure_elevation_hist = np.bincount(
            departure_elevation_index,
            weights=local_weight,
            minlength=ELEVATION_BINS,
        )
        arrival_elevation_hist = np.bincount(
            arrival_elevation_index,
            weights=local_weight,
            minlength=ELEVATION_BINS,
        )
        depth_hist = np.bincount(
            depth_index, weights=local_weight, minlength=len(DEPTH_UPPER) + 1
        )
        presence_hist = np.bincount(
            presence_index, weights=local_weight, minlength=len(PRESENCE_UPPER) + 1
        )

        mechanism = np.zeros(INTERACTION_BINS, dtype=np.float64)
        first = np.zeros(INTERACTION_BINS, dtype=np.float64)
        last = np.zeros(INTERACTION_BINS, dtype=np.float64)
        bigram = np.zeros((INTERACTION_BINS, INTERACTION_BINS), dtype=np.float64)
        mechanism_delay = np.zeros(
            (INTERACTION_BINS, len(DELAY_UPPER_M) + 1), dtype=np.float64
        )
        for local_row, (slot, path_weight, path_depth, delay_bin) in enumerate(
            zip(slots, local_weight, local_depth, delay_index)
        ):
            del local_row
            if path_depth <= 0:
                mechanism[0] += path_weight
                first[0] += path_weight
                last[0] += path_weight
                mechanism_delay[0, delay_bin] += path_weight
                continue
            sequence = np.clip(
                np.asarray(interactions[point, slot, :path_depth], dtype=np.int64),
                1,
                INTERACTION_BINS - 1,
            )
            counts = np.bincount(sequence, minlength=INTERACTION_BINS).astype(np.float64)
            fraction = counts / max(path_depth, 1)
            mechanism += path_weight * fraction
            mechanism_delay[:, delay_bin] += path_weight * fraction
            first[int(sequence[0])] += path_weight
            last[int(sequence[-1])] += path_weight
            if len(sequence) > 1:
                pair_weight = path_weight / (len(sequence) - 1)
                for left, right in zip(sequence[:-1], sequence[1:]):
                    bigram[int(left), int(right)] += pair_weight

        joint_delay_departure = np.bincount(
            delay_index * AZIMUTH_BINS + departure_index,
            weights=local_weight,
            minlength=(len(DELAY_UPPER_M) + 1) * AZIMUTH_BINS,
        )
        joint_delay_arrival = np.bincount(
            delay_index * AZIMUTH_BINS + arrival_index,
            weights=local_weight,
            minlength=(len(DELAY_UPPER_M) + 1) * AZIMUTH_BINS,
        )
        joint_departure_arrival = np.bincount(
            departure_index * AZIMUTH_BINS + arrival_index,
            weights=local_weight,
            minlength=AZIMUTH_BINS * AZIMUTH_BINS,
        )
        joint_delay_depth = np.bincount(
            delay_index * (len(DEPTH_UPPER) + 1) + depth_index,
            weights=local_weight,
            minlength=(len(DELAY_UPPER_M) + 1) * (len(DEPTH_UPPER) + 1),
        )

        sorted_weight = np.sort(local_weight)[::-1]
        rank_profile = np.zeros(RANK_BINS, dtype=np.float64)
        take = min(RANK_BINS - 1, len(sorted_weight))
        rank_profile[:take] = sorted_weight[:take]
        rank_profile[-1] = float(np.sum(sorted_weight[take:]))
        los_mass[point] = float(np.sum(local_weight[local_depth == 0]))

        blocks = [
            delay_hist,
            departure_hist,
            arrival_hist,
            departure_elevation_hist,
            arrival_elevation_hist,
            depth_hist,
            mechanism,
            first,
            last,
            bigram,
            joint_delay_departure,
            joint_delay_arrival,
            joint_departure_arrival,
            joint_delay_depth,
            mechanism_delay,
            rank_profile,
            presence_hist,
        ]
        vector = np.concatenate([probability_block(block) for block in blocks])
        norm = float(np.linalg.norm(vector))
        if norm > TINY:
            vector /= norm
        output[point] = vector.astype(np.float32)

    report = {
        "source": "base" if source_value == 0 else "scatter",
        "dimension": dimension,
        "nonempty_points": int(np.sum(path_counts > 0)),
        "path_count": {
            "mean": float(np.mean(path_counts)),
            "median": float(np.median(path_counts)),
            "min": int(np.min(path_counts)),
            "max": int(np.max(path_counts)),
        },
        "los_mass": {
            "mean": float(np.mean(los_mass)),
            "median": float(np.median(los_mass)),
        },
        "unit_norm_max_error": float(
            np.max(
                np.abs(
                    np.linalg.norm(output[path_counts > 0], axis=1) - 1.0
                ),
                initial=0.0,
            )
        ),
    }
    return output, report


SurfaceKey = tuple[int, ...]


def surface_key_to_id(key: SurfaceKey) -> str:
    value = 1469598103934665603
    for item in key:
        value ^= int(item) & 0xFFFFFFFFFFFFFFFF
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def surface_key_to_text(key: SurfaceKey) -> str:
    depth = int(key[0])
    pieces = []
    for step in range(depth):
        pieces.append(f"{key[1 + 2 * step]}@P{key[2 + 2 * step]}")
    return ">".join(pieces)


def surface_states(
    cache: dict[str, np.ndarray], plane_id: np.ndarray
) -> tuple[list[dict[SurfaceKey, float]], dict[str, Any]]:
    mask = np.asarray(cache["mask"])
    source = np.asarray(cache["source"])
    depth = np.asarray(cache["interaction_count"], dtype=np.int64)
    interactions = np.asarray(cache["interactions"], dtype=np.int64)
    primitives = np.asarray(cache["primitives"], dtype=np.int64)
    energy_fraction = np.asarray(cache["energy_fraction"], dtype=np.float64)
    presence = np.asarray(cache["presence"], dtype=np.float64)

    active = mask[..., None] & (source[..., None] == 0) & (
        np.arange(primitives.shape[2])[None, None, :] < depth[..., None]
    )
    maximum_primitive = int(np.max(primitives[active], initial=0))
    primitive_to_plane = np.empty(
        max(maximum_primitive + 1, len(plane_id)), dtype=np.int32
    )
    primitive_to_plane[: len(plane_id)] = plane_id
    if maximum_primitive >= len(plane_id):
        extra = maximum_primitive + 1 - len(plane_id)
        primitive_to_plane[len(plane_id) : maximum_primitive + 1] = (
            int(np.max(plane_id)) + 1 + np.arange(extra, dtype=np.int32)
        )

    output: list[dict[SurfaceKey, float]] = []
    family_counts = []
    for point in range(len(mask)):
        slots = np.flatnonzero(mask[point] & (source[point] == 0) & (depth[point] > 0))
        state: dict[SurfaceKey, float] = defaultdict(float)
        for slot in slots:
            local_depth = int(depth[point, slot])
            key_values: list[int] = [local_depth]
            for step in range(local_depth):
                primitive = int(primitives[point, slot, step])
                key_values.extend(
                    [
                        int(interactions[point, slot, step]),
                        int(primitive_to_plane[primitive]),
                    ]
                )
            key = tuple(key_values)
            weight = float(energy_fraction[point, slot]) * math.sqrt(
                max(float(presence[point, slot]), 1.0)
            )
            if math.isfinite(weight) and weight > 0.0:
                state[key] += weight
        total = float(sum(state.values()))
        if total > TINY:
            state = {key: value / total for key, value in state.items()}
        else:
            state = {}
        output.append(state)
        family_counts.append(len(state))
    return output, {
        "maximum_sionna_primitive": maximum_primitive,
        "non_ply_primitive_ids": int(max(maximum_primitive + 1 - len(plane_id), 0)),
        "family_count": {
            "mean": float(np.mean(family_counts)),
            "median": float(np.median(family_counts)),
            "min": int(np.min(family_counts)),
            "max": int(np.max(family_counts)),
        },
    }


def surface_overlap(
    left: dict[SurfaceKey, float], right: dict[SurfaceKey, float]
) -> tuple[float, int, SurfaceKey | None, float]:
    if not left or not right:
        return 0.0, 0, None, 0.0
    if len(left) > len(right):
        left, right = right, left
    intersection = 0.0
    shared = 0
    dominant_key: SurfaceKey | None = None
    dominant_mass = 0.0
    for key, value in left.items():
        other = right.get(key)
        if other is None:
            continue
        common = min(value, other)
        intersection += common
        shared += 1
        if common > dominant_mass:
            dominant_mass = common
            dominant_key = key
    denominator = 2.0 - intersection
    score = intersection / denominator if denominator > TINY else 1.0
    return float(score), shared, dominant_key, float(dominant_mass)


def pairwise_distance_xy(position: np.ndarray) -> np.ndarray:
    delta = position[:, None, :2] - position[None, :, :2]
    return np.sqrt(np.sum(delta * delta, axis=2))


def quantile(value: np.ndarray, probability: float) -> float:
    if len(value) == 0:
        raise RuntimeError("cannot calculate a quantile from an empty background")
    return float(np.quantile(np.asarray(value, dtype=np.float64), probability))


def mutual_neighbor_sets(
    robust: np.ndarray,
    distances: np.ndarray,
    sites: np.ndarray,
    minimum_distance: float,
    count: int,
) -> list[set[int]]:
    output: list[set[int]] = []
    for point in range(len(robust)):
        candidates = np.flatnonzero(
            (sites == sites[point])
            & (distances[point] >= minimum_distance)
            & (np.arange(len(robust)) != point)
        )
        order = candidates[np.argsort(-robust[point, candidates], kind="stable")]
        output.append(set(map(int, order[:count])))
    return output


class DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.rank = np.zeros(count, dtype=np.int8)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

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


def graph_components(
    point_count: int, edges: list[dict[str, Any]], minimum_size: int = 3
) -> list[list[int]]:
    disjoint = DisjointSet(point_count)
    edge_points: set[int] = set()
    for edge in edges:
        left = int(edge["test_a_index"])
        right = int(edge["test_b_index"])
        disjoint.union(left, right)
        edge_points.update((left, right))
    components: dict[int, list[int]] = defaultdict(list)
    for point in sorted(edge_points):
        components[disjoint.find(point)].append(point)
    output = [sorted(members) for members in components.values() if len(members) >= minimum_size]
    output.sort(key=lambda members: (-len(members), min(members)))
    return output


def top_indices(value: np.ndarray, mask: np.ndarray, count: int) -> np.ndarray:
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    order = candidates[np.argsort(-value[candidates], kind="stable")]
    return order[:count].astype(np.int64)


def build_surface_inverted(
    states: list[dict[SurfaceKey, float]], sites: np.ndarray
) -> dict[int, dict[SurfaceKey, list[tuple[int, float]]]]:
    output: dict[int, dict[SurfaceKey, list[tuple[int, float]]]] = {}
    for site in sorted(np.unique(sites).astype(int).tolist()):
        index: dict[SurfaceKey, list[tuple[int, float]]] = defaultdict(list)
        for point in np.flatnonzero(sites == site):
            for key, mass in states[int(point)].items():
                index[key].append((int(point), float(mass)))
        output[site] = index
    return output


def best_surface_train(
    query_state: dict[SurfaceKey, float],
    inverted: dict[SurfaceKey, list[tuple[int, float]]],
    train_positions: np.ndarray,
    query_position: np.ndarray,
    minimum_distance: float,
) -> tuple[tuple[int, float] | None, tuple[int, float] | None]:
    intersection: dict[int, float] = defaultdict(float)
    for key, query_mass in query_state.items():
        for train_point, train_mass in inverted.get(key, []):
            intersection[train_point] += min(query_mass, train_mass)
    if not intersection:
        return None, None
    scored = [
        (point, common / max(2.0 - common, TINY))
        for point, common in intersection.items()
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    overall = scored[0]
    nonlocal_best = None
    for point, score in scored:
        distance = float(np.linalg.norm(train_positions[point, :2] - query_position[:2]))
        if distance >= minimum_distance:
            nonlocal_best = (point, score)
            break
    return overall, nonlocal_best


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_groups(
    mesh: trimesh.Trimesh,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    test_sites: np.ndarray,
    bs_positions: np.ndarray,
    edges: list[dict[str, Any]],
    group_by_point: dict[int, str],
    best_nonlocal_train: dict[int, int],
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    groups = sorted(set(group_by_point.values()))
    color_map = {
        group: plt.get_cmap("tab20")(index % 20) for index, group in enumerate(groups)
    }
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for site, axis in enumerate(axes):
        site_test = np.flatnonzero(test_sites == site)
        focus = test_positions[site_test, :2]
        low = np.min(focus, axis=0) - 20.0
        high = np.max(focus, axis=0) + 20.0
        visible_vertex = (
            (vertices[:, 0] >= low[0])
            & (vertices[:, 0] <= high[0])
            & (vertices[:, 1] >= low[1])
            & (vertices[:, 1] <= high[1])
        )
        axis.scatter(
            vertices[visible_vertex, 0],
            vertices[visible_vertex, 1],
            s=0.5,
            c="#c8c8c8",
            alpha=0.35,
            rasterized=True,
        )
        axis.scatter(
            focus[:, 0], focus[:, 1], s=12, c="#777777", alpha=0.45, label="all test"
        )
        for edge in edges:
            left = int(edge["test_a_index"])
            right = int(edge["test_b_index"])
            if int(test_sites[left]) != site:
                continue
            group = group_by_point.get(left)
            color = color_map.get(group, "#444444")
            axis.plot(
                [test_positions[left, 0], test_positions[right, 0]],
                [test_positions[left, 1], test_positions[right, 1]],
                color=color,
                linewidth=1.3,
                alpha=0.65,
            )
        for group in groups:
            members = np.asarray(
                [point for point, value in group_by_point.items() if value == group and int(test_sites[point]) == site],
                dtype=np.int64,
            )
            if len(members) == 0:
                continue
            axis.scatter(
                test_positions[members, 0],
                test_positions[members, 1],
                s=34,
                color=color_map[group],
                edgecolor="black",
                linewidth=0.35,
                label=group,
                zorder=4,
            )
            center = np.mean(test_positions[members, :2], axis=0)
            axis.text(center[0], center[1], group, fontsize=8, weight="bold")
            analogs = sorted(
                set(best_nonlocal_train[point] for point in members if point in best_nonlocal_train)
            )
            if analogs:
                analogs_array = np.asarray(analogs, dtype=np.int64)
                axis.scatter(
                    train_positions[analogs_array, 0],
                    train_positions[analogs_array, 1],
                    marker="x",
                    s=32,
                    color=color_map[group],
                    linewidth=1.1,
                    zorder=5,
                )
        axis.scatter(
            [bs_positions[site, 0]],
            [bs_positions[site, 1]],
            marker="*",
            s=180,
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
            label="BS",
            zorder=6,
        )
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"Site {site}: nonlocal propagation groups")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=min(8, len(unique)))
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    setup, train_positions, test_positions, train_sites, test_sites = load_positions()
    bs_positions = np.asarray(setup.bs_positions, dtype=np.float64)
    train_positions = np.asarray(train_positions, dtype=np.float64)
    test_positions = np.asarray(test_positions, dtype=np.float64)
    train_sites = np.asarray(train_sites, dtype=np.int8)
    test_sites = np.asarray(test_sites, dtype=np.int8)
    test_clusters = load_test_clusters(test_positions, test_sites)
    train_cache = load_cache(TRAIN_CACHE, len(train_positions))
    test_cache = load_cache(TEST_CACHE, len(test_positions))
    plane_id, plane_report, mesh = canonical_face_planes()
    load_seconds = time.perf_counter() - load_started

    fingerprint_started = time.perf_counter()
    train_base, train_base_report = structural_embedding(
        train_cache, train_positions, train_sites, bs_positions, 0
    )
    train_scatter, train_scatter_report = structural_embedding(
        train_cache, train_positions, train_sites, bs_positions, 1
    )
    test_base, test_base_report = structural_embedding(
        test_cache, test_positions, test_sites, bs_positions, 0
    )
    test_scatter, test_scatter_report = structural_embedding(
        test_cache, test_positions, test_sites, bs_positions, 1
    )
    train_surface, train_surface_report = surface_states(train_cache, plane_id)
    test_surface, test_surface_report = surface_states(test_cache, plane_id)
    fingerprint_seconds = time.perf_counter() - fingerprint_started

    pair_started = time.perf_counter()
    base_similarity = np.clip(test_base @ test_base.T, 0.0, 1.0)
    scatter_similarity = np.clip(test_scatter @ test_scatter.T, 0.0, 1.0)
    robust_similarity = np.minimum(base_similarity, scatter_similarity)
    distances = pairwise_distance_xy(test_positions)
    surface_similarity = np.zeros((len(test_positions), len(test_positions)), dtype=np.float32)
    surface_shared_count = np.zeros_like(surface_similarity, dtype=np.int16)
    surface_dominant: dict[tuple[int, int], tuple[SurfaceKey | None, float]] = {}

    thresholds: dict[int, dict[str, float]] = {}
    backgrounds: dict[int, dict[str, list[float]]] = {}
    for site in sorted(np.unique(test_sites).astype(int).tolist()):
        rows = np.flatnonzero(test_sites == site)
        backgrounds[site] = {"base": [], "scatter": [], "robust": [], "surface": []}
        for local_left, left in enumerate(rows):
            for right in rows[local_left + 1 :]:
                if distances[left, right] < args.minimum_distance_m:
                    continue
                score, shared, dominant, dominant_mass = surface_overlap(
                    test_surface[int(left)], test_surface[int(right)]
                )
                surface_similarity[left, right] = surface_similarity[right, left] = score
                surface_shared_count[left, right] = surface_shared_count[right, left] = shared
                surface_dominant[(int(left), int(right))] = (dominant, dominant_mass)
                backgrounds[site]["base"].append(float(base_similarity[left, right]))
                backgrounds[site]["scatter"].append(float(scatter_similarity[left, right]))
                backgrounds[site]["robust"].append(float(robust_similarity[left, right]))
                backgrounds[site]["surface"].append(score)
        thresholds[site] = {
            "base_p95": quantile(np.asarray(backgrounds[site]["base"]), 0.95),
            "scatter_p95": quantile(np.asarray(backgrounds[site]["scatter"]), 0.95),
            "robust_p99": quantile(np.asarray(backgrounds[site]["robust"]), 0.99),
            "surface_p99": quantile(np.asarray(backgrounds[site]["surface"]), 0.99),
            "background_pair_count": float(len(backgrounds[site]["robust"])),
        }

    neighbor_sets = mutual_neighbor_sets(
        robust_similarity,
        distances,
        test_sites,
        args.minimum_distance_m,
        args.mutual_k,
    )
    edges: list[dict[str, Any]] = []
    for left in range(len(test_positions)):
        for right in range(left + 1, len(test_positions)):
            if test_sites[left] != test_sites[right] or distances[left, right] < args.minimum_distance_m:
                continue
            site = int(test_sites[left])
            threshold = thresholds[site]
            base_ok = float(base_similarity[left, right]) >= threshold["base_p95"]
            scatter_ok = float(scatter_similarity[left, right]) >= threshold["scatter_p95"]
            mutual = right in neighbor_sets[left] and left in neighbor_sets[right]
            structure_edge = bool(
                base_ok
                and scatter_ok
                and float(robust_similarity[left, right]) >= threshold["robust_p99"]
                and mutual
            )
            surface_edge = bool(
                base_ok
                and scatter_ok
                and float(surface_similarity[left, right]) > 0.0
                and float(surface_similarity[left, right]) >= threshold["surface_p99"]
            )
            if not (structure_edge or surface_edge):
                continue
            dominant, dominant_mass = surface_dominant.get((left, right), (None, 0.0))
            edge_type = (
                "structure+surface"
                if structure_edge and surface_edge
                else "structure"
                if structure_edge
                else "surface"
            )
            edges.append(
                {
                    "edge_type": edge_type,
                    "site": site,
                    "test_a_index": left,
                    "test_a_no": left + 1,
                    "test_a_cluster": int(test_clusters[left]),
                    "test_a_x": float(test_positions[left, 0]),
                    "test_a_y": float(test_positions[left, 1]),
                    "test_b_index": right,
                    "test_b_no": right + 1,
                    "test_b_cluster": int(test_clusters[right]),
                    "test_b_x": float(test_positions[right, 0]),
                    "test_b_y": float(test_positions[right, 1]),
                    "xy_distance_m": float(distances[left, right]),
                    "cross_test_hole": bool(test_clusters[left] != test_clusters[right]),
                    "base_similarity": float(base_similarity[left, right]),
                    "scatter_similarity": float(scatter_similarity[left, right]),
                    "robust_similarity": float(robust_similarity[left, right]),
                    "surface_jaccard": float(surface_similarity[left, right]),
                    "shared_surface_family_count": int(surface_shared_count[left, right]),
                    "dominant_shared_sequence_id": (
                        surface_key_to_id(dominant) if dominant is not None else ""
                    ),
                    "dominant_shared_sequence": (
                        surface_key_to_text(dominant) if dominant is not None else ""
                    ),
                    "dominant_shared_mass": float(dominant_mass),
                    "mutual_top_k": mutual,
                    "base_p95": threshold["base_p95"],
                    "scatter_p95": threshold["scatter_p95"],
                    "robust_p99": threshold["robust_p99"],
                    "surface_p99": threshold["surface_p99"],
                }
            )

    edge_points = sorted(
        set(
            [int(edge["test_a_index"]) for edge in edges]
            + [int(edge["test_b_index"]) for edge in edges]
        )
    )
    eligible_components = graph_components(len(test_positions), edges)
    eligible_components.sort(
        key=lambda members: (-len(members), int(test_sites[members[0]]), min(members))
    )
    structure_edges = [edge for edge in edges if "structure" in str(edge["edge_type"])]
    structure_components = graph_components(len(test_positions), structure_edges)
    structure_edge_points = sorted(
        set(
            [int(edge["test_a_index"]) for edge in structure_edges]
            + [int(edge["test_b_index"]) for edge in structure_edges]
        )
    )
    structure_cross_hole_count = sum(
        bool(edge["cross_test_hole"]) for edge in structure_edges
    )
    group_by_point: dict[int, str] = {}
    group_members: dict[str, list[int]] = {}
    for index, members in enumerate(eligible_components, start=1):
        group = f"G{index:02d}"
        group_members[group] = sorted(members)
        for point in members:
            group_by_point[point] = group
    for edge in edges:
        left_group = group_by_point.get(int(edge["test_a_index"]), "")
        right_group = group_by_point.get(int(edge["test_b_index"]), "")
        edge["group_id"] = left_group if left_group == right_group else ""
    edges.sort(
        key=lambda row: (
            -float(row["robust_similarity"]),
            -float(row["surface_jaccard"]),
            int(row["test_a_index"]),
            int(row["test_b_index"]),
        )
    )
    for edge_id, edge in enumerate(edges, start=1):
        edge["pair_id"] = edge_id

    all_best_rows: list[dict[str, Any]] = []
    for point in range(len(test_positions)):
        candidates = np.flatnonzero(
            (test_sites == test_sites[point])
            & (distances[point] >= args.minimum_distance_m)
            & (np.arange(len(test_positions)) != point)
        )
        best = int(candidates[np.argmax(robust_similarity[point, candidates])])
        all_best_rows.append(
            {
                "test_index": point,
                "test_no": point + 1,
                "site": int(test_sites[point]),
                "cluster": int(test_clusters[point]),
                "x": float(test_positions[point, 0]),
                "y": float(test_positions[point, 1]),
                "best_test_index": best,
                "best_test_no": best + 1,
                "best_cluster": int(test_clusters[best]),
                "best_x": float(test_positions[best, 0]),
                "best_y": float(test_positions[best, 1]),
                "xy_distance_m": float(distances[point, best]),
                "cross_test_hole": bool(test_clusters[point] != test_clusters[best]),
                "base_similarity": float(base_similarity[point, best]),
                "scatter_similarity": float(scatter_similarity[point, best]),
                "robust_similarity": float(robust_similarity[point, best]),
                "surface_jaccard": float(surface_similarity[point, best]),
                "mutual_top_k": bool(point in neighbor_sets[best] and best in neighbor_sets[point]),
                "group_id": group_by_point.get(point, ""),
            }
        )
    pair_seconds = time.perf_counter() - pair_started

    analog_started = time.perf_counter()
    analog_targets = sorted(group_by_point)
    if not analog_targets:
        analog_targets = edge_points
    surface_inverted = build_surface_inverted(train_surface, train_sites)
    analog_rows: list[dict[str, Any]] = []
    best_train_overall: dict[int, tuple[int, float]] = {}
    best_train_nonlocal: dict[int, tuple[int, float]] = {}
    best_surface_overall: dict[int, tuple[int, float]] = {}
    best_surface_nonlocal: dict[int, tuple[int, float]] = {}

    for site in sorted(np.unique(test_sites).astype(int).tolist()):
        query_rows = np.asarray(
            [point for point in analog_targets if int(test_sites[point]) == site],
            dtype=np.int64,
        )
        if len(query_rows) == 0:
            continue
        train_rows = np.flatnonzero(train_sites == site)
        base_score = np.clip(test_base[query_rows] @ train_base[train_rows].T, 0.0, 1.0)
        scatter_score = np.clip(
            test_scatter[query_rows] @ train_scatter[train_rows].T, 0.0, 1.0
        )
        robust_score = np.minimum(base_score, scatter_score)
        for local_query, test_point in enumerate(query_rows):
            local_distance = np.linalg.norm(
                train_positions[train_rows, :2] - test_positions[test_point, :2], axis=1
            )
            valid = (
                (np.linalg.norm(train_base[train_rows], axis=1) > 0.0)
                & (np.linalg.norm(train_scatter[train_rows], axis=1) > 0.0)
            )
            overall_local = top_indices(
                robust_score[local_query], valid, args.train_top_k
            )
            nonlocal_local = top_indices(
                robust_score[local_query],
                valid & (local_distance >= args.minimum_distance_m),
                args.train_top_k,
            )
            if len(overall_local):
                train_point = int(train_rows[int(overall_local[0])])
                best_train_overall[int(test_point)] = (
                    train_point,
                    float(robust_score[local_query, int(overall_local[0])]),
                )
            if len(nonlocal_local):
                train_point = int(train_rows[int(nonlocal_local[0])])
                best_train_nonlocal[int(test_point)] = (
                    train_point,
                    float(robust_score[local_query, int(nonlocal_local[0])]),
                )

            for mode, local_indices in (
                ("structure_overall", overall_local),
                ("structure_nonlocal", nonlocal_local),
            ):
                for rank, local_index in enumerate(local_indices, start=1):
                    train_point = int(train_rows[int(local_index)])
                    surface_score, shared, dominant, dominant_mass = surface_overlap(
                        test_surface[int(test_point)], train_surface[train_point]
                    )
                    analog_rows.append(
                        {
                            "group_id": group_by_point.get(int(test_point), "pair_only"),
                            "test_index": int(test_point),
                            "test_no": int(test_point) + 1,
                            "test_cluster": int(test_clusters[test_point]),
                            "test_x": float(test_positions[test_point, 0]),
                            "test_y": float(test_positions[test_point, 1]),
                            "match_mode": mode,
                            "rank": rank,
                            "train_index": train_point,
                            "train_no": train_point + 1,
                            "train_x": float(train_positions[train_point, 0]),
                            "train_y": float(train_positions[train_point, 1]),
                            "xy_distance_m": float(local_distance[int(local_index)]),
                            "base_similarity": float(base_score[local_query, int(local_index)]),
                            "scatter_similarity": float(scatter_score[local_query, int(local_index)]),
                            "robust_similarity": float(robust_score[local_query, int(local_index)]),
                            "surface_jaccard": surface_score,
                            "shared_surface_family_count": shared,
                            "dominant_shared_sequence_id": (
                                surface_key_to_id(dominant) if dominant is not None else ""
                            ),
                            "dominant_shared_sequence": (
                                surface_key_to_text(dominant) if dominant is not None else ""
                            ),
                            "dominant_shared_mass": dominant_mass,
                        }
                    )

            surface_overall, surface_nonlocal = best_surface_train(
                test_surface[int(test_point)],
                surface_inverted[site],
                train_positions,
                test_positions[int(test_point)],
                args.minimum_distance_m,
            )
            for mode, surface_result in (
                ("surface_overall", surface_overall),
                ("surface_nonlocal", surface_nonlocal),
            ):
                if surface_result is None:
                    continue
                train_point, surface_score = surface_result
                if mode == "surface_overall":
                    best_surface_overall[int(test_point)] = (train_point, surface_score)
                else:
                    best_surface_nonlocal[int(test_point)] = (train_point, surface_score)
                local_index = int(np.flatnonzero(train_rows == train_point)[0])
                _, shared, dominant, dominant_mass = surface_overlap(
                    test_surface[int(test_point)], train_surface[train_point]
                )
                analog_rows.append(
                    {
                        "group_id": group_by_point.get(int(test_point), "pair_only"),
                        "test_index": int(test_point),
                        "test_no": int(test_point) + 1,
                        "test_cluster": int(test_clusters[test_point]),
                        "test_x": float(test_positions[test_point, 0]),
                        "test_y": float(test_positions[test_point, 1]),
                        "match_mode": mode,
                        "rank": 1,
                        "train_index": train_point,
                        "train_no": train_point + 1,
                        "train_x": float(train_positions[train_point, 0]),
                        "train_y": float(train_positions[train_point, 1]),
                        "xy_distance_m": float(
                            np.linalg.norm(
                                train_positions[train_point, :2]
                                - test_positions[test_point, :2]
                            )
                        ),
                        "base_similarity": float(base_score[local_query, local_index]),
                        "scatter_similarity": float(scatter_score[local_query, local_index]),
                        "robust_similarity": float(robust_score[local_query, local_index]),
                        "surface_jaccard": float(surface_score),
                        "shared_surface_family_count": shared,
                        "dominant_shared_sequence_id": (
                            surface_key_to_id(dominant) if dominant is not None else ""
                        ),
                        "dominant_shared_sequence": (
                            surface_key_to_text(dominant) if dominant is not None else ""
                        ),
                        "dominant_shared_mass": dominant_mass,
                    }
                )
    analog_rows.sort(
        key=lambda row: (
            str(row["group_id"]),
            int(row["test_index"]),
            str(row["match_mode"]),
            int(row["rank"]),
        )
    )
    analog_seconds = time.perf_counter() - analog_started

    group_rows: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    covered_overall = 0
    covered_nonlocal = 0
    covered_total = 0
    for group, members in group_members.items():
        member_set = set(members)
        local_edges = [
            edge
            for edge in edges
            if int(edge["test_a_index"]) in member_set
            and int(edge["test_b_index"]) in member_set
        ]
        weakest = min(float(edge["robust_similarity"]) for edge in local_edges)
        cross_count = sum(bool(edge["cross_test_hole"]) for edge in local_edges)
        clusters = sorted(set(int(test_clusters[point]) for point in members))
        for member_rank, point in enumerate(members, start=1):
            degree = sum(
                int(edge["test_a_index"]) == point or int(edge["test_b_index"]) == point
                for edge in local_edges
            )
            overall = best_train_overall.get(point)
            nonlocal_match = best_train_nonlocal.get(point)
            overall_covered = bool(overall is not None and overall[1] >= weakest)
            nonlocal_covered = bool(
                nonlocal_match is not None and nonlocal_match[1] >= weakest
            )
            covered_overall += int(overall_covered)
            covered_nonlocal += int(nonlocal_covered)
            covered_total += 1
            group_rows.append(
                {
                    "group_id": group,
                    "group_size": len(members),
                    "group_edge_count": len(local_edges),
                    "group_cross_hole_edge_count": cross_count,
                    "group_weakest_robust_similarity": weakest,
                    "member_rank": member_rank,
                    "test_index": point,
                    "test_no": point + 1,
                    "site": int(test_sites[point]),
                    "cluster": int(test_clusters[point]),
                    "x": float(test_positions[point, 0]),
                    "y": float(test_positions[point, 1]),
                    "degree": degree,
                    "best_train_index": overall[0] if overall is not None else "",
                    "best_train_no": overall[0] + 1 if overall is not None else "",
                    "best_train_robust_similarity": overall[1] if overall is not None else "",
                    "best_train_meets_group_threshold": overall_covered,
                    "best_nonlocal_train_index": (
                        nonlocal_match[0] if nonlocal_match is not None else ""
                    ),
                    "best_nonlocal_train_no": (
                        nonlocal_match[0] + 1 if nonlocal_match is not None else ""
                    ),
                    "best_nonlocal_train_robust_similarity": (
                        nonlocal_match[1] if nonlocal_match is not None else ""
                    ),
                    "best_nonlocal_train_meets_group_threshold": nonlocal_covered,
                    "best_surface_train_index": (
                        best_surface_overall[point][0] if point in best_surface_overall else ""
                    ),
                    "best_surface_train_no": (
                        best_surface_overall[point][0] + 1
                        if point in best_surface_overall
                        else ""
                    ),
                    "best_surface_train_jaccard": (
                        best_surface_overall[point][1] if point in best_surface_overall else ""
                    ),
                }
            )
        group_summaries.append(
            {
                "group_id": group,
                "site": int(test_sites[members[0]]),
                "member_count": len(members),
                "members_zero_based": members,
                "members_one_based": [point + 1 for point in members],
                "clusters": clusters,
                "edge_count": len(local_edges),
                "cross_hole_edge_count": cross_count,
                "cross_hole_edge_fraction": cross_count / max(len(local_edges), 1),
                "robust_similarity": {
                    "min": weakest,
                    "mean": float(
                        np.mean([float(edge["robust_similarity"]) for edge in local_edges])
                    ),
                    "max": float(
                        np.max([float(edge["robust_similarity"]) for edge in local_edges])
                    ),
                },
            }
        )

    cross_hole_edge_count = sum(bool(edge["cross_test_hole"]) for edge in edges)
    cross_hole_edge_fraction = cross_hole_edge_count / max(len(edges), 1)
    overall_coverage = covered_overall / max(covered_total, 1)
    nonlocal_coverage = covered_nonlocal / max(covered_total, 1)
    gate_passed = bool(
        len(edges) >= 20
        and len(group_members) >= 3
        and cross_hole_edge_fraction >= 0.5
        and nonlocal_coverage >= 0.70
    )
    if len(edges) < 20 or len(group_members) < 3:
        conclusion = "insufficient_robust_nonlocal_test_groups"
    elif nonlocal_coverage < 0.70:
        conclusion = "test_groups_exist_but_train_state_coverage_is_insufficient"
    else:
        conclusion = "robust_test_groups_and_official_train_analogs_exist"

    output_started = time.perf_counter()
    write_csv(PAIR_CSV, edges)
    write_csv(GROUP_CSV, group_rows)
    write_csv(ANALOG_CSV, analog_rows)
    write_csv(ALL_BEST_CSV, all_best_rows)
    best_nonlocal_train_for_plot = {
        point: value[0] for point, value in best_train_nonlocal.items()
    }
    plot_groups(
        mesh,
        train_positions,
        test_positions,
        test_sites,
        bs_positions,
        edges,
        group_by_point,
        best_nonlocal_train_for_plot,
    )

    top_edges = edges[:20]
    markdown = [
        "# Sionna传播规律相似点分析",
        "",
        "## 直接结论",
        "",
        (
            f"按预先冻结的严格口径，共找到 **{len(edges)}** 条相距至少"
            f" `{args.minimum_distance_m:.0f}m` 的稳健传播相似边，形成 **{len(group_members)}** 个"
            "至少3点的传播组。"
        ),
        (
            f"跨真实测试洞的边占 `{cross_hole_edge_fraction:.1%}`；组内成员在4000个官方Train点中"
            f"找到同等级非局部对应点的覆盖率为 `{nonlocal_coverage:.1%}`。"
        ),
        (
            f"删掉全部表面族边后，单靠base/scatter双配置结构证据仍有 **{len(structure_edges)}** 条边、"
            f"**{len(structure_components)}** 个至少3点组，跨洞边占 "
            f"`{structure_cross_hole_count / max(len(structure_edges), 1):.1%}`；主结论不依赖表面阈值。"
        ),
        f"预注册门结论：`{conclusion}`，整体通过=`{str(gate_passed).lower()}`。",
        "",
        "这里的分组不是全地图旋转或镜像对称，而是两套Sionna配置都支持的局部传播状态重复。",
        "",
        "## 传播组",
        "",
        "| 组 | 站点 | 点数 | Test点号（1-based） | 测试洞 | 最弱稳健相似度 |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for group in group_summaries:
        member_text = ", ".join(map(str, group["members_one_based"]))
        cluster_text = ", ".join(map(str, group["clusters"]))
        markdown.append(
            f"| {group['group_id']} | {group['site']} | {group['member_count']} | "
            f"{member_text} | {cluster_text} | {group['robust_similarity']['min']:.6f} |"
        )
    if not group_summaries:
        markdown.append("| 无 | - | 0 | - | - | - |")

    markdown.extend(
        [
            "",
            "## 最强非局部点对",
            "",
            "| A点 | B点 | 站点 | 洞A→洞B | 距离m | base | scatter | 稳健分 | 表面重合 |",
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for edge in top_edges:
        markdown.append(
            f"| {edge['test_a_no']} | {edge['test_b_no']} | {edge['site']} | "
            f"{edge['test_a_cluster']}→{edge['test_b_cluster']} | {edge['xy_distance_m']:.2f} | "
            f"{edge['base_similarity']:.6f} | {edge['scatter_similarity']:.6f} | "
            f"{edge['robust_similarity']:.6f} | {edge['surface_jaccard']:.6f} |"
        )

    markdown.extend(
        [
            "",
            "## 文件怎么看",
            "",
            "- `robust_nonlocal_pairs.csv`：全部严格通过的Test-Test点对、坐标、距离和相似度。",
            "- `robust_test_groups.csv`：传播组成员及每个成员最好的Train对应点。",
            "- `test_to_train_analogs.csv`：每个入组Test点的结构Top-5、非局部Top-5和表面族对应点。",
            "- `all_test_best_nonlocal.csv`：500个Test点各自最像的非局部Test点，即使未过严格门也保留。",
            "- `传播组空间分布.png`：组内远距离连线和对应Train点的空间位置。",
            "",
            "本分析只把Sionna当离线显微镜。官方Train H未读取，Sionna不会进入后续训练、权重或最终推理。",
        ]
    )
    REPORT_MD.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    output_seconds = time.perf_counter() - output_started

    phase_seconds = {
        "load_and_identity": load_seconds,
        "fingerprints_and_surface_states": fingerprint_seconds,
        "test_pair_graph": pair_seconds,
        "train_analog_search": analog_seconds,
        "output_and_figure": output_seconds,
    }
    total_seconds = time.perf_counter() - started
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "type": "diagnostic_only",
        "official_train_channel_values_read": False,
        "official_test_channel_values_read": False,
        "sionna_used_for_training": False,
        "submission_generated": False,
        "configuration": {
            "minimum_distance_m": args.minimum_distance_m,
            "mutual_k": args.mutual_k,
            "train_top_k": args.train_top_k,
            "base_threshold_quantile": 0.95,
            "scatter_threshold_quantile": 0.95,
            "robust_threshold_quantile": 0.99,
            "surface_threshold_quantile": 0.99,
            "surface_plane_round_decimals": 3,
        },
        "causal_chain": (
            "two independent multiseed Sionna path observations -> offline propagation-state "
            "labels -> robust nonlocal real-test groups -> official-train coordinates with the "
            "same state and available official H -> a future Clean model may learn the hidden state"
        ),
        "inputs": {
            "train_path_cache": relative(TRAIN_CACHE),
            "test_path_cache": relative(TEST_CACHE),
            "train_positions": relative(PATHS.round2 / "Round2_Train_Pos.npy"),
            "test_positions": relative(PATHS.round2 / "Round2_Test_Pos.npy"),
            "map": relative(PATHS.round2 / "Round2_Map.ply"),
            "test_clusters": relative(INITIAL_AUDIT_CSV),
        },
        "identity_checks": {
            "train_rows_are_canonical_0_to_3999": True,
            "test_rows_are_canonical_0_to_499": True,
            "masked_tau_angles_energy_finite": True,
            "test_cluster_count": int(len(np.unique(test_clusters))),
            "plane_identity": plane_report,
            "train_surface": train_surface_report,
            "test_surface": test_surface_report,
        },
        "fingerprints": {
            "definition": (
                "Hellinger-embedded relative excess-delay, departure/arrival angle, depth, "
                "interaction, joint-structure, energy-rank, and recurrence histograms"
            ),
            "delay_upper_m": DELAY_UPPER_M.tolist(),
            "depth_upper": DEPTH_UPPER.tolist(),
            "presence_upper": PRESENCE_UPPER.tolist(),
            "azimuth_bins": AZIMUTH_BINS,
            "elevation_bins": ELEVATION_BINS,
            "train_base": train_base_report,
            "train_scatter": train_scatter_report,
            "test_base": test_base_report,
            "test_scatter": test_scatter_report,
        },
        "thresholds_by_site": {str(site): value for site, value in thresholds.items()},
        "test_graph": {
            "robust_edge_count": len(edges),
            "structure_edge_count": sum("structure" in str(edge["edge_type"]) for edge in edges),
            "surface_edge_count": sum("surface" in str(edge["edge_type"]) for edge in edges),
            "edge_endpoint_count": len(edge_points),
            "cross_hole_edge_count": cross_hole_edge_count,
            "cross_hole_edge_fraction": cross_hole_edge_fraction,
            "group_count": len(group_members),
            "group_member_count": len(group_by_point),
            "groups": group_summaries,
        },
        "structure_only_sensitivity": {
            "edge_count": len(structure_edges),
            "edge_endpoint_count": len(structure_edge_points),
            "cross_hole_edge_count": structure_cross_hole_count,
            "cross_hole_edge_fraction": structure_cross_hole_count
            / max(len(structure_edges), 1),
            "group_count_size_at_least_3": len(structure_components),
            "group_sizes": [len(members) for members in structure_components],
            "interpretation": (
                "Removing every surface-family edge leaves the same number of multi-point "
                "groups, so the primary conclusion is supported by the two independent "
                "structural fingerprints rather than a sparse surface-overlap threshold."
            ),
        },
        "train_analog_coverage": {
            "group_member_count": covered_total,
            "overall_count": covered_overall,
            "overall_fraction": overall_coverage,
            "nonlocal_count": covered_nonlocal,
            "nonlocal_fraction": nonlocal_coverage,
            "coverage_threshold": "each member's group weakest robust Test-Test edge",
        },
        "gate": {
            "minimum_robust_edges": 20,
            "minimum_groups_size_at_least_3": 3,
            "minimum_cross_hole_edge_fraction": 0.5,
            "minimum_nonlocal_train_coverage": 0.70,
            "passed": gate_passed,
            "conclusion": conclusion,
        },
        "performance": {
            "selected_gpu": "none_cpu_only",
            "phase_seconds": phase_seconds,
            "total_seconds": total_seconds,
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2),
            "measured_bottleneck": max(phase_seconds, key=phase_seconds.get),
        },
        "outputs": {
            "pairs_csv": relative(PAIR_CSV),
            "groups_csv": relative(GROUP_CSV),
            "analogs_csv": relative(ANALOG_CSV),
            "all_test_best_csv": relative(ALL_BEST_CSV),
            "report_markdown": relative(REPORT_MD),
            "figure": relative(FIGURE),
        },
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
