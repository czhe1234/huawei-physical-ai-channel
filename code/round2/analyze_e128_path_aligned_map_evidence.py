#!/usr/bin/env python3
"""Audit candidate-aligned official-map path and visibility evidence."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import trimesh

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e123_complete_path_variable_system as e123
import analyze_e124_one_to_one_path_identity_ceiling as e124
import analyze_e127_candidate_lifecycle_information as e127
import analyze_global_local_path_lifecycle as e79
import analyze_v4_map_path_support_ceiling as e19
import analyze_v4_map_virtual_source_phase as e8
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
import train_e126_conditional_path_state_heads as e126
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E128-path-aligned-map-evidence"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "path_aligned_map_evidence"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_path_aligned_map_evidence"
MAP_CACHE = ARTIFACT_DIR / "outer0_fit_only_map_path_bank.npz"
METRICS = ARTIFACT_DIR / "path_aligned_map_outer0_metrics.npz"
REPORT = ANALYSIS_DIR / "path_aligned_map_outer0_audit.json"

OUTER = 0
INNER = 1
MAP_PATHS = 64
MAXIMUM_EXCESS_LENGTH_M = 240.0
AREA_RANK_WEIGHT_M = 0.35
RAY_BATCH_SIZE = 4096
TINY = 1e-30

META_NAMES = (
    "is_los",
    "log_wall_area",
    "excess_length_m",
    "wall_normal_x",
    "wall_normal_y",
    "reflection_x_scaled",
    "reflection_y_scaled",
    "reflection_to_query_x_scaled",
    "reflection_to_query_y_scaled",
    "first_leg_fraction",
    "incidence_cosine",
    "departure_cosine",
    "blocker_count",
    "first_leg_blockers",
    "second_leg_blockers",
    "path_visible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_808_128)
    parser.add_argument("--force-map-cache", action="store_true")
    return parser.parse_args()


def periodic_abs(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    return np.abs(np.remainder(left - right + period / 2.0, period) - period / 2.0)


def unique_distance_count(values: np.ndarray, tolerance: float = 0.35) -> int:
    if len(values) == 0:
        return 0
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return int(1 + np.sum(np.diff(ordered) > tolerance))


def segment_blocker_counts(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    vectors = targets - origins
    lengths = np.linalg.norm(vectors, axis=1)
    directions = np.divide(
        vectors,
        np.maximum(lengths[:, None], 1e-12),
        out=np.zeros_like(vectors),
    )
    output = np.zeros(len(origins), dtype=np.int16)
    for start in range(0, len(origins), batch_size):
        stop = min(start + batch_size, len(origins))
        locations, ray_ids, _ = mesh.ray.intersects_location(
            origins[start:stop], directions[start:stop], multiple_hits=True
        )
        if len(ray_ids):
            ray_ids = ray_ids.astype(np.int64)
            distance = np.linalg.norm(
                locations - origins[start:stop][ray_ids], axis=1
            )
            for ray in np.unique(ray_ids):
                local_distance = distance[ray_ids == ray]
                valid = (local_distance > 0.50) & (
                    local_distance < lengths[start + ray] - 0.75
                )
                output[start + ray] = unique_distance_count(local_distance[valid])
        print(f"E128 visibility rays {stop}/{len(origins)}", flush=True)
    return output


def point_map_paths(
    position: np.ndarray,
    bs_position: np.ndarray,
    sources: np.ndarray,
    wall: e8.WallBank,
    horizontal_orientation: float,
    vertical_orientation: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, int]]]:
    normal = e8.canonicalize_normals(wall.normal)
    delta = position[None] - sources
    length = np.linalg.norm(delta, axis=1)
    denominator = np.sum(delta[:, :2] * normal, axis=1)
    numerator = wall.offset - np.sum(sources[:, :2] * normal, axis=1)
    parameter = numerator / np.where(
        np.abs(denominator) > 1e-12, denominator, np.nan
    )
    reflection = sources + parameter[:, None] * delta
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    tangent_coordinate = np.sum(reflection[:, :2] * tangent, axis=1)
    direct_length = float(np.linalg.norm(position - bs_position))
    valid = (
        np.isfinite(parameter)
        & (parameter > 0.0)
        & (parameter < 1.0)
        & (tangent_coordinate >= wall.tangent_min - 1.0)
        & (tangent_coordinate <= wall.tangent_max + 1.0)
        & (reflection[:, 2] >= wall.z_min - 1.0)
        & (reflection[:, 2] <= wall.z_max + 1.0)
        & (length <= direct_length + MAXIMUM_EXCESS_LENGTH_M)
    )
    wall_ids = np.flatnonzero(valid)
    if len(wall_ids):
        priority = length[wall_ids] - AREA_RANK_WEIGHT_M * np.log1p(
            wall.area[wall_ids]
        )
        wall_ids = wall_ids[np.argsort(priority)[: MAP_PATHS - 1]]

    coordinates = np.zeros((1 + len(wall_ids), 3), dtype=np.float32)
    metadata = np.zeros((1 + len(wall_ids), len(META_NAMES)), dtype=np.float32)
    segments: list[tuple[np.ndarray, np.ndarray, int]] = []

    direct = position - bs_position
    direct_azimuth = math.atan2(float(direct[1]), float(direct[0]))
    direct_elevation = math.atan2(
        float(direct[2]), max(float(np.linalg.norm(direct[:2])), 1e-9)
    )
    coordinates[0] = (
        np.mod(8.0 * math.sin(direct_azimuth - horizontal_orientation), e79.MH),
        np.mod(4.0 * math.sin(direct_elevation - vertical_orientation), e79.MV),
        np.mod(direct_length / e8.METERS_PER_DELAY_BIN, e79.S_SC),
    )
    metadata[0, 0] = 1.0
    metadata[0, 5:7] = position[:2] / 200.0
    metadata[0, 9] = 1.0
    segments.append((bs_position.copy(), position.copy(), 0))

    if len(wall_ids):
        selected_reflection = reflection[wall_ids]
        departure = selected_reflection - bs_position[None]
        azimuth = np.arctan2(departure[:, 1], departure[:, 0])
        elevation = np.arctan2(
            departure[:, 2],
            np.maximum(np.linalg.norm(departure[:, :2], axis=1), 1e-9),
        )
        coordinates[1:, 0] = np.mod(
            8.0 * np.sin(azimuth - horizontal_orientation), e79.MH
        )
        coordinates[1:, 1] = np.mod(
            4.0 * np.sin(elevation - vertical_orientation), e79.MV
        )
        coordinates[1:, 2] = np.mod(
            length[wall_ids] / e8.METERS_PER_DELAY_BIN, e79.S_SC
        )

        first_length = np.linalg.norm(departure, axis=1)
        incoming = departure / np.maximum(first_length[:, None], 1e-12)
        outgoing_vector = position[None] - selected_reflection
        outgoing_length = np.linalg.norm(outgoing_vector, axis=1)
        outgoing = outgoing_vector / np.maximum(outgoing_length[:, None], 1e-12)
        wall_normal_3d = np.column_stack(
            (normal[wall_ids], np.zeros(len(wall_ids), dtype=np.float64))
        )
        metadata[1:, 1] = np.log1p(wall.area[wall_ids]).astype(np.float32)
        metadata[1:, 2] = (length[wall_ids] - direct_length).astype(np.float32)
        metadata[1:, 3:5] = normal[wall_ids].astype(np.float32)
        metadata[1:, 5:7] = (selected_reflection[:, :2] / 200.0).astype(
            np.float32
        )
        metadata[1:, 7:9] = (
            (selected_reflection[:, :2] - position[None, :2]) / 100.0
        ).astype(np.float32)
        metadata[1:, 9] = np.divide(
            first_length, np.maximum(length[wall_ids], 1e-12)
        ).astype(np.float32)
        metadata[1:, 10] = np.abs(
            np.sum(incoming * wall_normal_3d, axis=1)
        ).astype(np.float32)
        metadata[1:, 11] = np.abs(
            np.sum(outgoing * wall_normal_3d, axis=1)
        ).astype(np.float32)
        for local, point in enumerate(selected_reflection, start=1):
            segments.append((bs_position.copy(), point.copy(), 2 * local - 1))
            segments.append((point.copy(), position.copy(), 2 * local))
    return coordinates, metadata, segments


def build_map_cache(
    fit_rows: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    component: np.lib.npyio.NpzFile,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if MAP_CACHE.is_file() and not force:
        cached = np.load(MAP_CACHE, allow_pickle=False)
        if (
            cached["coordinates"].shape == (len(positions), MAP_PATHS, 3)
            and cached["metadata"].shape
            == (len(positions), MAP_PATHS, len(META_NAMES))
        ):
            diagnostics = json.loads(str(cached["diagnostics_json"]))
            return (
                cached["coordinates"].astype(np.float32),
                cached["metadata"].astype(np.float32),
                cached["valid"].astype(bool),
                diagnostics,
            )

    wall = e8.build_wall_bank(e8.Config())
    sources = {
        int(site_value): e8.image_sources(wall, bs_positions[int(site_value)])
        for site_value in np.unique(site)
    }
    coordinates = component["coordinates"].astype(np.float32)
    coefficients = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    calibration: dict[int, tuple[float, float, int]] = {}
    for site_value in np.unique(site):
        calibration[int(site_value)] = e19.calibrate_array(
            fit_rows,
            int(site_value),
            positions,
            site,
            bs_positions[int(site_value)],
            coordinates,
            coefficients,
            point_power,
        )

    map_coordinate = np.zeros((len(positions), MAP_PATHS, 3), dtype=np.float32)
    map_metadata = np.zeros(
        (len(positions), MAP_PATHS, len(META_NAMES)), dtype=np.float32
    )
    map_valid = np.zeros((len(positions), MAP_PATHS), dtype=bool)
    ray_origins: list[np.ndarray] = []
    ray_targets: list[np.ndarray] = []
    ray_point: list[int] = []
    ray_path: list[int] = []
    ray_leg: list[int] = []
    for point_id in range(len(positions)):
        site_value = int(site[point_id])
        horizontal, vertical, _ = calibration[site_value]
        local_coordinate, local_metadata, segments = point_map_paths(
            positions[point_id],
            bs_positions[site_value],
            sources[site_value],
            wall,
            horizontal,
            vertical,
        )
        count = min(len(local_coordinate), MAP_PATHS)
        map_coordinate[point_id, :count] = local_coordinate[:count]
        map_metadata[point_id, :count] = local_metadata[:count]
        map_valid[point_id, :count] = True
        for origin, target, encoded_leg in segments:
            path_id = 0 if encoded_leg == 0 else (encoded_leg + 1) // 2
            leg = 1 if encoded_leg == 0 or encoded_leg % 2 == 1 else 2
            if path_id >= count:
                continue
            ray_origins.append(origin)
            ray_targets.append(target)
            ray_point.append(point_id)
            ray_path.append(path_id)
            ray_leg.append(leg)
        if (point_id + 1) % 500 == 0:
            print(f"E128 map paths {point_id + 1}/{len(positions)}", flush=True)

    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh).__name__)
    blocker = segment_blocker_counts(
        mesh,
        np.asarray(ray_origins, dtype=np.float64),
        np.asarray(ray_targets, dtype=np.float64),
        RAY_BATCH_SIZE,
    )
    first = np.zeros((len(positions), MAP_PATHS), dtype=np.int16)
    second = np.zeros_like(first)
    for point_id, path_id, leg, value in zip(
        ray_point, ray_path, ray_leg, blocker
    ):
        if leg == 1:
            first[point_id, path_id] = value
        else:
            second[point_id, path_id] = value
    total = first + second
    map_metadata[..., 12] = total
    map_metadata[..., 13] = first
    map_metadata[..., 14] = second
    map_metadata[..., 15] = (total == 0) & map_valid

    diagnostics = {
        "calibration": {
            str(site_value): {
                "horizontal_orientation_degrees": float(np.degrees(values[0])),
                "vertical_orientation_degrees": float(np.degrees(values[1])),
                "fit_los_count": int(values[2]),
            }
            for site_value, values in calibration.items()
        },
        "wall_count": int(len(wall.offset)),
        "path_count_mean": float(np.mean(map_valid.sum(axis=1))),
        "path_count_min": int(np.min(map_valid.sum(axis=1))),
        "path_count_max": int(np.max(map_valid.sum(axis=1))),
        "visible_path_fraction": float(
            np.sum((total == 0) & map_valid) / np.maximum(np.sum(map_valid), 1)
        ),
        "los_clear_fraction": float(np.mean(map_metadata[:, 0, 15])),
        "reflection_visible_fraction": float(
            np.sum(map_metadata[:, 1:, 15])
            / np.maximum(np.sum(map_valid[:, 1:]), 1)
        ),
        "segment_count": int(len(blocker)),
    }
    np.savez_compressed(
        MAP_CACHE,
        coordinates=map_coordinate,
        metadata=map_metadata,
        valid=map_valid,
        metadata_names=np.asarray(META_NAMES),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    return map_coordinate, map_metadata, map_valid, diagnostics


def map_aligned_features(
    candidate_coordinate: np.ndarray,
    candidate_mask: np.ndarray,
    map_coordinate: np.ndarray,
    map_metadata: np.ndarray,
    map_valid: np.ndarray,
    source_rows: np.ndarray | None = None,
) -> np.ndarray:
    count, candidates, _ = candidate_coordinate.shape
    metadata_dim = map_metadata.shape[2]
    feature_dim = 22 + 2 * metadata_dim
    output = np.zeros((count, candidates, feature_dim), dtype=np.float32)
    for row in range(count):
        source = row if source_rows is None else int(source_rows[row])
        valid_candidate = np.flatnonzero(candidate_mask[row])
        valid_map = np.flatnonzero(map_valid[source])
        if len(valid_candidate) == 0 or len(valid_map) == 0:
            continue
        candidate = candidate_coordinate[row, valid_candidate]
        paths = map_coordinate[source, valid_map]
        metadata = map_metadata[source, valid_map]
        dh = periodic_abs(candidate[:, None, 0], paths[None, :, 0], e79.MH) / 1.5
        dv = periodic_abs(candidate[:, None, 1], paths[None, :, 1], e79.MV)
        dd = periodic_abs(candidate[:, None, 2], paths[None, :, 2], e79.S_SC) / 2.0
        full = np.square(dh) + np.square(dv) + np.square(dd)
        horizontal_delay = np.square(dh) + np.square(dd)
        delay = np.square(dd)
        nearest_hd = np.argmin(horizontal_delay, axis=1)
        minimum_full = np.min(full, axis=1)
        minimum_hd = horizontal_delay[np.arange(len(candidate)), nearest_hd]
        minimum_delay = np.min(delay, axis=1)
        visible = metadata[:, 15] > 0.5
        if np.any(visible):
            visible_ids = np.flatnonzero(visible)
            visible_full = full[:, visible_ids]
            visible_hd = horizontal_delay[:, visible_ids]
            visible_delay = delay[:, visible_ids]
            nearest_visible_local = np.argmin(visible_hd, axis=1)
            nearest_visible = visible_ids[nearest_visible_local]
            minimum_visible_full = np.min(visible_full, axis=1)
            minimum_visible_hd = visible_hd[
                np.arange(len(candidate)), nearest_visible_local
            ]
            minimum_visible_delay = np.min(visible_delay, axis=1)
            nearest_visible_metadata = metadata[nearest_visible]
        else:
            minimum_visible_full = np.full(len(candidate), 50.0)
            minimum_visible_hd = np.full(len(candidate), 50.0)
            minimum_visible_delay = np.full(len(candidate), 50.0)
            nearest_visible_metadata = np.zeros(
                (len(candidate), metadata_dim), dtype=np.float32
            )
        values = np.column_stack(
            (
                np.exp(-0.5 * minimum_full),
                np.exp(-0.5 * minimum_hd),
                np.exp(-0.5 * minimum_delay),
                np.log1p(np.minimum(minimum_full, 50.0)),
                np.log1p(np.minimum(minimum_hd, 50.0)),
                np.log1p(np.minimum(minimum_delay, 50.0)),
                dh[np.arange(len(candidate)), nearest_hd],
                dv[np.arange(len(candidate)), nearest_hd],
                dd[np.arange(len(candidate)), nearest_hd],
                np.mean(full < 1.0, axis=1),
                np.mean(horizontal_delay < 1.0, axis=1),
                np.mean(delay < 0.25, axis=1),
                np.exp(-0.5 * minimum_visible_full),
                np.exp(-0.5 * minimum_visible_hd),
                np.exp(-0.5 * minimum_visible_delay),
                np.log1p(np.minimum(minimum_visible_full, 50.0)),
                np.log1p(np.minimum(minimum_visible_hd, 50.0)),
                np.log1p(np.minimum(minimum_visible_delay, 50.0)),
                np.full(len(candidate), np.mean(visible), dtype=np.float32),
                np.full(len(candidate), np.sum(visible) / MAP_PATHS, dtype=np.float32),
                np.minimum(minimum_visible_full - minimum_full, 50.0),
                np.minimum(minimum_visible_hd - minimum_hd, 50.0),
                metadata[nearest_hd],
                nearest_visible_metadata,
            )
        ).astype(np.float32)
        output[row, valid_candidate] = values
    return output


def candidate_shuffle(
    features: np.ndarray, mask: np.ndarray, seed: int
) -> tuple[np.ndarray, int]:
    output = np.zeros_like(features)
    rng = np.random.default_rng(seed)
    fixed = 0
    for row in range(len(features)):
        ids = np.flatnonzero(mask[row])
        if len(ids) <= 1:
            output[row, ids] = features[row, ids]
            fixed += len(ids)
            continue
        shift = int(rng.integers(1, len(ids)))
        source = np.roll(ids, shift)
        output[row, ids] = features[row, source]
        fixed += int(np.sum(ids == source))
    return output, fixed


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    directory = e118.cache_dir(e123.cache_tag(OUTER))
    store = e118.ClusterTensorStore(directory, device)
    identity_model, normalization, _ = e126.load_identity_core(store, device)
    identity_logits, identity_mass = e127.identity_predictions(
        identity_model, store, normalization, args.batch_size
    )
    del identity_model

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    coordinate = np.asarray(
        np.load(directory / "coordinate.npy", mmap_mode="r"), np.float32
    )
    mask = np.asarray(np.load(directory / "mask.npy", mmap_mode="r"), bool)
    base_log_mass = np.asarray(
        np.load(directory / "base_log_mass.npy", mmap_mode="r"), np.float32
    )
    member_count = np.asarray(
        np.load(directory / "member_count.npy", mmap_mode="r"), np.int16
    )
    targets, assignment = e124.build_one_to_one_targets(
        coordinate,
        mask,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
    )
    target_absolute = targets["mass_absolute"].astype(np.float32)
    target_total = target_absolute.sum(axis=1, keepdims=True)
    target_mass = np.divide(
        target_absolute,
        np.maximum(target_total, TINY),
        out=np.zeros_like(target_absolute),
    )
    target_active = target_absolute > 0.0
    hard_mass = torch.as_tensor(target_absolute, device=device)

    setup, positions, _, site, _ = load_positions()
    fold_id = np.load(e115.FULL_FOLD).astype(np.int8)
    fit_rows = np.flatnonzero((fold_id != OUTER) & (fold_id != INNER))
    validation_rows = np.flatnonzero(fold_id == INNER)
    held_rows = np.flatnonzero(fold_id == OUTER)
    neighbors, shuffled_neighbors, neighbor_distance = e127.build_neighbors(
        positions, site, fold_id, 20_260_808_127
    )
    identity_probability = 1.0 / (
        1.0 + np.exp(-np.clip(identity_logits, -30.0, 30.0))
    )
    lifecycle = e127.lifecycle_features(
        coordinate,
        mask,
        identity_mass,
        identity_probability,
        neighbors,
        neighbor_distance,
    )
    unary = e127.unary_features(
        identity_logits, identity_mass, base_log_mass, member_count, mask
    )
    lifecycle_features = np.concatenate((unary, lifecycle), axis=2)

    map_coordinate, map_metadata, map_valid, map_diagnostics = build_map_cache(
        fit_rows,
        positions,
        site,
        setup.bs_positions,
        component,
        args.force_map_cache,
    )
    correct_map = map_aligned_features(
        coordinate, mask, map_coordinate, map_metadata, map_valid
    )
    candidate_shuffled_map, candidate_shuffle_fixed = candidate_shuffle(
        correct_map, mask, args.seed
    )
    query_shuffle_rows = shuffled_neighbors[:, 0]
    query_shuffled_map = map_aligned_features(
        coordinate,
        mask,
        map_coordinate,
        map_metadata,
        map_valid,
        query_shuffle_rows,
    )

    feature_sets = {
        "lifecycle_reference": lifecycle_features,
        "correct_map": np.concatenate((lifecycle_features, correct_map), axis=2),
        "candidate_shuffled_map": np.concatenate(
            (lifecycle_features, candidate_shuffled_map), axis=2
        ),
        "query_shuffled_map": np.concatenate(
            (lifecycle_features, query_shuffled_map), axis=2
        ),
    }
    probes = {
        name: e127.fit_probe(features, target_active, mask, fit_rows)
        for name, features in feature_sets.items()
    }
    logits = {
        name: e127.probe_scores(probes[name], features, mask)
        for name, features in feature_sets.items()
    }
    selected_alpha: dict[str, float] = {}
    selection_grids: dict[str, list[dict[str, object]]] = {}
    for name, probe_logit in logits.items():
        alpha, grid = e127.select_alpha(
            name,
            probe_logit,
            identity_logits,
            mask,
            store,
            validation_rows,
            normalization,
            hard_mass,
            args.batch_size,
        )
        selected_alpha[name] = alpha
        selection_grids[name] = grid

    held_metrics = {
        "identity_reference": e127.evaluate_score(
            identity_logits,
            store,
            held_rows,
            normalization,
            hard_mass,
            args.batch_size,
            True,
        )
    }
    for name, probe_logit in logits.items():
        held_metrics[name] = e127.evaluate_score(
            e127.blend_score(
                identity_logits, probe_logit, mask, selected_alpha[name]
            ),
            store,
            held_rows,
            normalization,
            hard_mass,
            args.batch_size,
            True,
        )

    held_probe_ap = {
        "identity_raw": e127.candidate_ap(
            identity_logits, target_active, mask, held_rows
        ),
        **{
            name: e127.candidate_ap(value, target_active, mask, held_rows)
            for name, value in logits.items()
        },
        "correct_map_hd_kernel": e127.candidate_ap(
            correct_map[..., 1], target_active, mask, held_rows
        ),
        "candidate_shuffled_map_hd_kernel": e127.candidate_ap(
            candidate_shuffled_map[..., 1], target_active, mask, held_rows
        ),
        "query_shuffled_map_hd_kernel": e127.candidate_ap(
            query_shuffled_map[..., 1], target_active, mask, held_rows
        ),
        "correct_visible_map_hd_kernel": e127.candidate_ap(
            correct_map[..., 13], target_active, mask, held_rows
        ),
        "candidate_shuffled_visible_map_hd_kernel": e127.candidate_ap(
            candidate_shuffled_map[..., 13], target_active, mask, held_rows
        ),
        "query_shuffled_visible_map_hd_kernel": e127.candidate_ap(
            query_shuffled_map[..., 13], target_active, mask, held_rows
        ),
    }

    identity = held_metrics["identity_reference"]
    lifecycle_result = held_metrics["lifecycle_reference"]
    correct = held_metrics["correct_map"]
    candidate_control = held_metrics["candidate_shuffled_map"]
    query_control = held_metrics["query_shuffled_map"]
    ap_candidate_margin = held_probe_ap["correct_map"] - held_probe_ap[
        "candidate_shuffled_map"
    ]
    ap_query_margin = held_probe_ap["correct_map"] - held_probe_ap[
        "query_shuffled_map"
    ]
    top16_gain = (
        correct["top16_target_mass_recall"]
        - identity["top16_target_mass_recall"]
    )
    full_gain = correct["full_official"]["score"] - identity["full_official"][
        "score"
    ]
    full_candidate_margin = (
        correct["full_official"]["score"]
        - candidate_control["full_official"]["score"]
    )
    full_query_margin = (
        correct["full_official"]["score"]
        - query_control["full_official"]["score"]
    )
    gate = {
        "minimum_ap_margin_over_each_control": 0.01,
        "require_positive_top16_gain_over_identity": True,
        "require_positive_full_gain_over_identity": True,
        "require_positive_full_margin_over_each_control": True,
        "correct_vs_candidate_shuffle_ap_margin": float(ap_candidate_margin),
        "correct_vs_query_shuffle_ap_margin": float(ap_query_margin),
        "top16_gain_over_identity": float(top16_gain),
        "full_gain_over_identity": float(full_gain),
        "full_margin_over_candidate_shuffle": float(full_candidate_margin),
        "full_margin_over_query_shuffle": float(full_query_margin),
    }
    gate["passed"] = bool(
        min(ap_candidate_margin, ap_query_margin) >= 0.01
        and top16_gain > 0.0
        and full_gain > 0.0
        and min(full_candidate_margin, full_query_margin) > 0.0
    )

    identity_replay_error = abs(
        float(identity["full_official"]["score"]) - 0.4948253852127809
    )
    lifecycle_replay_error = abs(
        float(lifecycle_result["full_official"]["score"])
        - 0.49371129213740533
    )
    if identity_replay_error > 2e-6:
        raise RuntimeError(
            f"E125 identity replay drifted by {identity_replay_error:.3e}"
        )
    if selected_alpha["lifecycle_reference"] != 0.25:
        raise RuntimeError(
            "E127 lifecycle alpha did not replay: "
            f"{selected_alpha['lifecycle_reference']}"
        )
    if lifecycle_replay_error > 2e-6:
        raise RuntimeError(
            f"E127 lifecycle replay drifted by {lifecycle_replay_error:.3e}"
        )

    np.savez_compressed(
        METRICS,
        fold_id=fold_id,
        map_valid=map_valid,
        map_metadata=map_metadata.astype(np.float16),
        correct_map_features=correct_map.astype(np.float16),
        candidate_shuffled_map_features=candidate_shuffled_map.astype(np.float16),
        query_shuffled_map_features=query_shuffled_map.astype(np.float16),
        identity_logits=identity_logits.astype(np.float32),
        **{f"{name}_logit": value.astype(np.float32) for name, value in logits.items()},
        target_active=target_active,
        target_mass=target_mass,
    )
    report = {
        "experiment": VERSION,
        "status": "information_audit_complete",
        "primary_hypothesis": (
            "Candidate-aligned official-map LOS/reflection identity and segment "
            "visibility convert E127 existence ranking into correct path mass."
        ),
        "strict_protocol": {
            "outer": OUTER,
            "inner": INNER,
            "fit_count": int(len(fit_rows)),
            "validation_count": int(len(validation_rows)),
            "held_count": int(len(held_rows)),
            "held_used_for_probe_or_alpha_selection": False,
            "array_calibration_uses_fit_rows_only": True,
            "candidate_donor_excludes_outer_and_inner": True,
            "official_h_and_map_only": True,
            "sionna_or_v12_used": False,
        },
        "fixed_design": {
            "map_paths": MAP_PATHS,
            "maximum_excess_length_m": MAXIMUM_EXCESS_LENGTH_M,
            "area_rank_weight_m": AREA_RANK_WEIGHT_M,
            "map_cost_scales_h_v_d": [1.5, 1.0, 2.0],
            "lifecycle_neighbors": e127.NEIGHBORS,
            "lifecycle_spatial_scale_m": e127.SPATIAL_SCALE_M,
            "lifecycle_match_bandwidth": e127.MATCH_BANDWIDTH,
            "alphas_selected_on_inner": list(e127.ALPHAS),
            "candidate_shuffle_fixed_pairs": int(candidate_shuffle_fixed),
            "query_shuffle": (
                "first E127 same-site same-fold random row, excluding query and its "
                "eight correct spatial neighbors"
            ),
            "map_metadata_names": list(META_NAMES),
            "map_feature_dim": int(correct_map.shape[2]),
            "probe": "same standardized class-balanced logistic regression as E127",
        },
        "map_diagnostics": map_diagnostics,
        "assignment": assignment,
        "selected_alpha": selected_alpha,
        "inner_selection_grids": selection_grids,
        "held_probe_ap": held_probe_ap,
        "held_metrics": held_metrics,
        "identity_replay_absolute_error": float(identity_replay_error),
        "lifecycle_replay_absolute_error": float(lifecycle_replay_error),
        "gate": gate,
        "decision": {
            "train_path_aligned_map_network": bool(gate["passed"]),
            "next_if_passed": (
                "Add the verified map-path state to the identity-first candidate "
                "data-association network before reopening downstream states."
            ),
            "next_if_failed": (
                "Stop official one-bounce map features and use Sionna continuous "
                "trajectories only as an offline variable-discovery laboratory."
            ),
        },
        "outputs": {
            "map_cache": PATHS.relative(MAP_CACHE),
            "map_cache_bytes": int(MAP_CACHE.stat().st_size),
            "metrics": PATHS.relative(METRICS),
            "metrics_bytes": int(METRICS.stat().st_size),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "map_diagnostics": map_diagnostics,
                "selected_alpha": selected_alpha,
                "held_probe_ap": held_probe_ap,
                "held_metrics": held_metrics,
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
