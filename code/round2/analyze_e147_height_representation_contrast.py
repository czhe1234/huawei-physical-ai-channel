#!/usr/bin/env python3
"""Contrast planar, single-height, and multi-height path representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time

import numpy as np

import analyze_e143_effective_surface_identity as e143
import analyze_e146_map_propagation_graph_coverage as e146
import analyze_v4_map_virtual_source_phase as map_phase
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import MV, component_energy


VERSION = "Round2-E147-height-representation-contrast"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "height_representation_contrast"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_height_representation_contrast"
METADATA_CACHE = ARTIFACT_DIR / "train_candidate_interaction_geometry.npz"
METRICS = ARTIFACT_DIR / "height_representation_metrics.npz"
REPORT = ANALYSIS_DIR / "height_representation_contrast.json"


@dataclass(frozen=True)
class Config:
    planar_height_m: float = 1.5
    height_margin_m: float = 0.20
    height_quantiles: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    minimum_distinct_v_bin: float = 0.05
    required_gain: float = 0.02
    required_winner_positive_folds: int = 4
    required_control_margin: float = 0.15
    required_control_positive_folds: int = 5
    required_structured_over_shuffle: float = 0.01
    required_multiheight_nearest_full: float = 0.88
    seed: int = 20_260_809_147


def empty_candidate_arrays(count: int, budget: int) -> dict[str, np.ndarray]:
    shape = (count, budget)
    return {
        "theta": np.zeros(shape, dtype=np.float32),
        "phi": np.zeros(shape, dtype=np.float32),
        "delay": np.zeros(shape, dtype=np.float32),
        "path_length": np.zeros(shape, dtype=np.float32),
        "score": np.full(shape, np.inf, dtype=np.float32),
        "path_type": np.full(shape, -1, dtype=np.int8),
        "depth": np.zeros(shape, dtype=np.int8),
        "mask": np.zeros(shape, dtype=bool),
    }


def record_candidate(
    arrays: dict[str, np.ndarray],
    first_xy: np.ndarray,
    first_z: np.ndarray,
    z_min: np.ndarray,
    z_max: np.ndarray,
    row: int,
    slot: int,
    direction: np.ndarray,
    path_length: float,
    path_type: int,
    depth: int,
    score: float,
    bs: np.ndarray,
    lower: float,
    upper: float,
) -> None:
    e146.append_candidate(
        arrays,
        row,
        slot,
        direction,
        path_length,
        path_type,
        depth,
        score,
    )
    first_xy[row, slot] = bs[:2] + direction[:2]
    first_z[row, slot] = bs[2] + direction[2]
    z_min[row, slot] = lower
    z_max[row, slot] = upper


def metadata_config(config: Config, base_config: e146.Config) -> dict[str, object]:
    return {
        "experiment": VERSION,
        "candidate_config": e146.candidate_config_payload(base_config),
        "height_margin_m": config.height_margin_m,
    }


def validate_reconstruction(
    rebuilt: dict[str, np.ndarray], original: dict[str, np.ndarray]
) -> dict[str, float | bool]:
    for key in ("mask", "path_type", "depth"):
        if not np.array_equal(rebuilt[key], original[key]):
            mismatch = int(np.sum(rebuilt[key] != original[key]))
            raise RuntimeError(f"candidate reconstruction mismatch for {key}: {mismatch}")
    valid = rebuilt["mask"]
    maximum_error: dict[str, float] = {}
    for key in ("theta", "phi", "delay", "path_length", "score"):
        difference = np.abs(
            rebuilt[key][valid].astype(np.float64)
            - original[key][valid].astype(np.float64)
        )
        maximum_error[key] = float(np.max(difference)) if len(difference) else 0.0
        if maximum_error[key] > 2e-5:
            raise RuntimeError(
                f"candidate reconstruction mismatch for {key}: {maximum_error[key]}"
            )
    return {"exact_discrete": True, **maximum_error}


def build_interaction_metadata(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    original: dict[str, np.ndarray],
    config: Config,
    base_config: e146.Config,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool | str]]:
    expected = metadata_config(config, base_config)
    if METADATA_CACHE.is_file():
        cached = np.load(METADATA_CACHE, allow_pickle=False)
        if json.loads(str(cached["config_json"])) == expected:
            metadata = {
                key: cached[key]
                for key in ("first_xy", "first_z", "z_min", "z_max")
            }
            return metadata, {
                "reused": True,
                "exact_discrete": True,
                "theta": 0.0,
                "phi": 0.0,
                "delay": 0.0,
                "path_length": 0.0,
                "score": 0.0,
            }

    mesh = e146.load_mesh()
    wall = map_phase.build_wall_bank(map_phase.Config())
    graph, _ = e146.build_graph(mesh, bs_positions, base_config, force=False)
    route_bank = e146.build_route_bank(graph, bs_positions, base_config)
    query_corner, query_visible, _ = e146.query_visible_corners(
        mesh, positions, graph, base_config
    )
    rebuilt = empty_candidate_arrays(len(positions), base_config.candidate_budget)
    shape = rebuilt["mask"].shape
    first_xy = np.full((*shape, 2), np.nan, dtype=np.float32)
    first_z = np.full(shape, np.nan, dtype=np.float32)
    z_min = np.full(shape, np.nan, dtype=np.float32)
    z_max = np.full(shape, np.nan, dtype=np.float32)
    sources = {
        site_value: map_phase.image_sources(wall, bs_positions[site_value])
        for site_value in range(len(bs_positions))
    }
    corner_start = 1 + base_config.ground_quota + base_config.reflection_quota

    for row, ue in enumerate(positions):
        site_value = int(site[row])
        bs = bs_positions[site_value]
        direct_vector = ue - bs
        direct_length = float(np.linalg.norm(direct_vector))
        record_candidate(
            rebuilt,
            first_xy,
            first_z,
            z_min,
            z_max,
            row,
            0,
            direct_vector,
            direct_length,
            e146.TYPE_DIRECT,
            0,
            0.0,
            bs,
            float(ue[2]),
            float(ue[2]),
        )
        for local, height in enumerate((0.0, -0.5)[: base_config.ground_quota]):
            direction, length = e146.ground_candidate(ue, bs, height)
            record_candidate(
                rebuilt,
                first_xy,
                first_z,
                z_min,
                z_max,
                row,
                1 + local,
                direction,
                length,
                e146.TYPE_GROUND,
                1,
                length - direct_length,
                bs,
                height,
                height,
            )

        reflected = e146.reflection_candidates(
            ue, bs, wall, sources[site_value], base_config
        )
        reflection_start = 1 + base_config.ground_quota
        for local, (score, direction, length, wall_id) in enumerate(reflected):
            record_candidate(
                rebuilt,
                first_xy,
                first_z,
                z_min,
                z_max,
                row,
                reflection_start + local,
                direction,
                length,
                e146.TYPE_REFLECTION,
                1,
                score,
                bs,
                float(wall.z_min[wall_id]),
                float(wall.z_max[wall_id]),
            )

        graph_candidates: list[
            tuple[float, np.ndarray, float, int, tuple[int, ...]]
        ] = []
        for endpoint, visible in zip(query_corner[row], query_visible[row]):
            if not visible:
                continue
            endpoint = int(endpoint)
            for depth in range(1, base_config.maximum_corner_depth + 1):
                for _, route in route_bank[site_value][depth][endpoint]:
                    length, direction, turn = e146.route_geometry(route, ue, bs, graph)
                    excess = length - direct_length
                    if (
                        excess < -1e-5
                        or excess > base_config.maximum_excess_length_m
                    ):
                        continue
                    score = (
                        excess
                        + base_config.interaction_penalty_m * depth
                        + base_config.turn_penalty_m * turn
                        - 0.02
                        * float(np.sum(graph.node_strength[np.asarray(route)]))
                    )
                    graph_candidates.append((score, direction, length, depth, route))
        graph_candidates.sort(key=lambda item: item[0])
        selected_graph: list[
            tuple[float, np.ndarray, float, int, tuple[int, ...]]
        ] = []
        for candidate in graph_candidates:
            direction = candidate[1]
            unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
            if any(
                abs(candidate[2] - kept[2]) < 0.10
                and np.dot(
                    unit,
                    kept[1] / max(float(np.linalg.norm(kept[1])), 1e-12),
                )
                > 0.9995
                for kept in selected_graph
            ):
                continue
            selected_graph.append(candidate)
            if len(selected_graph) >= base_config.corner_quota:
                break
        for local, (score, direction, length, depth, route) in enumerate(
            selected_graph
        ):
            first_node = int(route[0])
            record_candidate(
                rebuilt,
                first_xy,
                first_z,
                z_min,
                z_max,
                row,
                corner_start + local,
                direction,
                length,
                e146.TYPE_CORNER_1 + depth - 1,
                depth,
                score,
                bs,
                float(graph.node_z_min[first_node]),
                float(graph.node_z_max[first_node]),
            )
        if (row + 1) % 500 == 0:
            print(f"E147 metadata {row + 1}/{len(positions)}", flush=True)

    validation = validate_reconstruction(rebuilt, original)
    metadata = {
        "first_xy": first_xy,
        "first_z": first_z,
        "z_min": z_min,
        "z_max": z_max,
    }
    np.savez_compressed(
        METADATA_CACHE,
        **metadata,
        config_json=np.asarray(json.dumps(expected, sort_keys=True)),
    )
    return metadata, {"reused": False, **validation}


def vertical_bin(
    first_xy: np.ndarray,
    height: np.ndarray,
    bs: np.ndarray,
    vertical_axis: np.ndarray,
) -> np.ndarray:
    direction = np.column_stack((first_xy - bs[None, :2], height - bs[2]))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    value = direction @ vertical_axis[:3] + vertical_axis[3]
    return np.mod(value, MV).astype(np.float32)


def planar_coordinates(
    current: np.ndarray,
    base: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    site: np.ndarray,
    bs_positions: np.ndarray,
    vertical_axis: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, dict[str, float]]:
    output = np.array(current, copy=True)
    interaction = base["mask"] & (base["path_type"] >= e146.TYPE_REFLECTION)
    in_range = []
    for row in range(len(output)):
        slots = np.flatnonzero(interaction[row])
        if len(slots) == 0:
            continue
        heights = np.full(len(slots), config.planar_height_m, dtype=np.float64)
        output[row, slots, 1] = vertical_bin(
            metadata["first_xy"][row, slots].astype(np.float64),
            heights,
            bs_positions[int(site[row])],
            vertical_axis[row].astype(np.float64),
        )
        in_range.extend(
            (
                (heights >= metadata["z_min"][row, slots])
                & (heights <= metadata["z_max"][row, slots])
            ).tolist()
        )
    return output, {
        "interaction_count": int(np.sum(interaction)),
        "planar_height_inside_surface_fraction": float(np.mean(in_range)),
        "maximum_h_change": 0.0,
        "maximum_delay_change": 0.0,
    }


def build_multiheight_coordinates(
    current: np.ndarray,
    base: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    site: np.ndarray,
    bs_positions: np.ndarray,
    vertical_axis: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    coordinate = np.array(current, copy=True)
    mask = np.array(base["mask"], copy=True)
    path_type = np.array(base["path_type"], copy=True)
    added = np.zeros(len(mask), dtype=np.int16)
    parent_count = np.zeros(len(mask), dtype=np.int16)
    height_span = []

    for row in range(len(mask)):
        free = list(np.flatnonzero(~mask[row]))
        parents = np.flatnonzero(
            mask[row] & (path_type[row] >= e146.TYPE_REFLECTION)
        )
        if not free or len(parents) == 0:
            continue
        parents = parents[np.argsort(base["score"][row, parents])]
        proposals: dict[int, list[float]] = {}
        for parent in parents:
            low = float(metadata["z_min"][row, parent])
            high = float(metadata["z_max"][row, parent])
            interior_low = min(low + config.height_margin_m, high)
            interior_high = max(high - config.height_margin_m, interior_low)
            height_span.append(max(interior_high - interior_low, 0.0))
            values = [config.planar_height_m]
            values.extend(
                interior_low + q * (interior_high - interior_low)
                for q in config.height_quantiles
            )
            candidate_v = vertical_bin(
                np.repeat(
                    metadata["first_xy"][row, parent][None].astype(np.float64),
                    len(values),
                    axis=0,
                ),
                np.asarray(values, dtype=np.float64),
                bs_positions[int(site[row])],
                vertical_axis[row].astype(np.float64),
            )
            unique_v: list[float] = []
            current_v = float(coordinate[row, parent, 1])
            for value in candidate_v:
                value = float(value)
                existing = [current_v, *unique_v]
                distance = [
                    abs(float(e146.circular_delta(value, old, MV)))
                    for old in existing
                ]
                if min(distance) >= config.minimum_distinct_v_bin:
                    unique_v.append(value)
            proposals[int(parent)] = unique_v

        round_index = 0
        while free:
            progressed = False
            for parent in parents:
                values = proposals[int(parent)]
                if round_index >= len(values) or not free:
                    continue
                slot = free.pop(0)
                coordinate[row, slot] = coordinate[row, parent]
                coordinate[row, slot, 1] = values[round_index]
                mask[row, slot] = True
                path_type[row, slot] = path_type[row, parent]
                added[row] += 1
                progressed = True
            if not progressed:
                break
            round_index += 1
        parent_count[row] = int(
            np.sum([len(values) > 0 for values in proposals.values()])
        )

    return coordinate, mask, path_type, {
        "candidate_count_mean": float(np.mean(mask.sum(axis=1))),
        "candidate_count_median": float(np.median(mask.sum(axis=1))),
        "candidate_count_min": int(np.min(mask.sum(axis=1))),
        "candidate_count_max": int(np.max(mask.sum(axis=1))),
        "points_at_budget_fraction": float(np.mean(mask.sum(axis=1) == mask.shape[1])),
        "added_count_mean": float(np.mean(added)),
        "parents_with_alternative_mean": float(np.mean(parent_count)),
        "available_height_span_median": float(np.median(height_span)),
        "maximum_h_change": 0.0,
        "maximum_delay_change": 0.0,
    }


def permute_interactions(
    coordinate: np.ndarray,
    mask: np.ndarray,
    path_type: np.ndarray,
    source: np.ndarray,
    base_config: e146.Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_coordinate = np.array(coordinate, copy=True)
    output_mask = np.array(mask, copy=True)
    output_type = np.array(path_type, copy=True)
    start = 1 + base_config.ground_quota
    output_coordinate[:, start:] = coordinate[source, start:]
    output_mask[:, start:] = mask[source, start:]
    output_type[:, start:] = path_type[source, start:]
    return output_coordinate, output_mask, output_type


def shuffle_v_within_point(
    coordinate: np.ndarray,
    mask: np.ndarray,
    path_type: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.array(coordinate, copy=True)
    for row in range(len(output)):
        slots = np.flatnonzero(
            mask[row] & (path_type[row] >= e146.TYPE_REFLECTION)
        )
        if len(slots) > 1:
            output[row, slots, 1] = output[row, rng.permutation(slots), 1]
    return output


def evaluate_variant(
    coordinate: np.ndarray,
    mask: np.ndarray,
    path_type: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_energy: np.ndarray,
    truth_power: np.ndarray,
    fold: np.ndarray,
    permutations: list[np.ndarray],
    base_config: e146.Config,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    actual = e146.evaluate_coverage(
        coordinate,
        mask,
        path_type,
        truth_coordinate,
        truth_energy,
        truth_power,
        base_config,
    )
    control_metrics = []
    for source in permutations:
        control_coordinate, control_mask, control_type = permute_interactions(
            coordinate, mask, path_type, source, base_config
        )
        control_metrics.append(
            e146.evaluate_coverage(
                control_coordinate,
                control_mask,
                control_type,
                truth_coordinate,
                truth_energy,
                truth_power,
                base_config,
            )
        )
    control_point = np.mean(
        np.stack([item["point_coverage"] for item in control_metrics]), axis=0
    )
    actual_fold = e146.fold_summary(
        np.asarray(actual["point_coverage"]), fold, truth_energy
    )
    nearest_fold = e146.fold_summary(
        np.asarray(actual["nearest_full"]), fold, truth_energy
    )
    control_fold = e146.fold_summary(control_point, fold, truth_energy)
    control_global = float(
        np.mean([item["global_energy_coverage"] for item in control_metrics])
    )
    fold_margin = np.asarray(actual_fold) - np.asarray(control_fold)
    report = {
        "candidate_count_mean": float(np.mean(mask.sum(axis=1))),
        "candidate_count_median": float(np.median(mask.sum(axis=1))),
        "global_energy_coverage": float(actual["global_energy_coverage"]),
        "global_quality_coverage": float(actual["global_quality_coverage"]),
        "nearest_full_coverage": float(actual["global_nearest_full"]),
        "nearest_h_delay_coverage": float(actual["global_nearest_hd"]),
        "nearest_delay_coverage": float(actual["global_nearest_d"]),
        "fold_energy_coverage": actual_fold,
        "fold_nearest_full_coverage": nearest_fold,
        "control_energy_coverage_mean": control_global,
        "control_fold_energy_coverage": control_fold,
        "control_margin": float(actual["global_energy_coverage"]) - control_global,
        "control_fold_margin": fold_margin.tolist(),
        "control_positive_folds": int(np.sum(fold_margin > 0.0)),
    }
    arrays = {
        "point_coverage": np.asarray(actual["point_coverage"]),
        "point_quality": np.asarray(actual["point_quality"]),
        "nearest_full": np.asarray(actual["nearest_full"]),
        "control_point_coverage": control_point.astype(np.float32),
    }
    return report, arrays


def fold_delta(
    candidate: dict[str, object], baseline: dict[str, object], key: str
) -> list[float]:
    return (
        np.asarray(candidate[key], dtype=np.float64)
        - np.asarray(baseline[key], dtype=np.float64)
    ).tolist()


def decide(
    variants: dict[str, dict[str, object]],
    shuffled: dict[str, object],
    config: Config,
) -> dict[str, object]:
    current = variants["current_single_height"]
    planar = variants["fixed_plane_z1p5"]
    multi = variants["surface_range_multiheight"]
    planar_energy_gain = float(planar["global_energy_coverage"]) - float(
        current["global_energy_coverage"]
    )
    planar_nearest_gain = float(planar["nearest_full_coverage"]) - float(
        current["nearest_full_coverage"]
    )
    multi_energy_gain = float(multi["global_energy_coverage"]) - float(
        current["global_energy_coverage"]
    )
    multi_nearest_gain = float(multi["nearest_full_coverage"]) - float(
        current["nearest_full_coverage"]
    )
    structured_gain = float(multi["global_energy_coverage"]) - float(
        shuffled["global_energy_coverage"]
    )
    multi_fold_gain = np.asarray(multi["fold_energy_coverage"]) - np.asarray(
        current["fold_energy_coverage"]
    )
    planar_fold_gain = np.asarray(planar["fold_energy_coverage"]) - np.asarray(
        current["fold_energy_coverage"]
    )
    causal_pass = bool(
        float(multi["control_margin"]) >= config.required_control_margin
        and int(multi["control_positive_folds"])
        >= config.required_control_positive_folds
    )
    multi_gain_pass = bool(
        multi_energy_gain >= config.required_gain
        and multi_nearest_gain >= config.required_gain
        and int(np.sum(multi_fold_gain > 0.0))
        >= config.required_winner_positive_folds
    )
    planar_gain_pass = bool(
        planar_energy_gain >= config.required_gain
        and planar_nearest_gain >= config.required_gain
        and int(np.sum(planar_fold_gain > 0.0))
        >= config.required_winner_positive_folds
    )
    planar_captures_multi = bool(
        multi_energy_gain > 0.0
        and multi_nearest_gain > 0.0
        and planar_energy_gain >= 0.80 * multi_energy_gain
        and planar_nearest_gain >= 0.80 * multi_nearest_gain
        and multi_energy_gain - planar_energy_gain < 0.01
        and multi_nearest_gain - planar_nearest_gain < 0.01
    )
    structured_height_pass = bool(
        structured_gain >= config.required_structured_over_shuffle
    )
    absolute_pass = bool(
        float(multi["nearest_full_coverage"])
        >= config.required_multiheight_nearest_full
    )

    if planar_gain_pass and planar_captures_multi:
        conclusion = "fixed_plane_supported"
        next_action = "treat wall/corner propagation as a two-dimensional path-state problem"
    elif multi_gain_pass and causal_pass and structured_height_pass:
        conclusion = (
            "structured_multiheight_validated"
            if absolute_pass
            else "structured_multiheight_direction_only"
        )
        next_action = (
            "design the propagation network with continuous interaction-height packets"
            if absolute_pass
            else "improve structured V coverage before training a propagation network"
        )
    elif multi_gain_pass and causal_pass:
        conclusion = "vertical_ambiguity_not_physical_height"
        next_action = (
            "model V as a latent multi-modal path state instead of a deterministic ray height"
        )
    else:
        conclusion = "height_representation_not_decisive"
        next_action = "return to candidate identity or array-factor decomposition"

    return {
        "conclusion": conclusion,
        "next_action": next_action,
        "planar_energy_gain": planar_energy_gain,
        "planar_nearest_full_gain": planar_nearest_gain,
        "planar_positive_energy_folds": int(np.sum(planar_fold_gain > 0.0)),
        "multiheight_energy_gain": multi_energy_gain,
        "multiheight_nearest_full_gain": multi_nearest_gain,
        "multiheight_positive_energy_folds": int(np.sum(multi_fold_gain > 0.0)),
        "structured_multiheight_over_v_shuffle": structured_gain,
        "causal_control_pass": causal_pass,
        "multiheight_gain_pass": multi_gain_pass,
        "planar_gain_pass": planar_gain_pass,
        "planar_captures_at_least_80pct_multiheight_gain": planar_captures_multi,
        "structured_height_pass": structured_height_pass,
        "multiheight_absolute_nearest_full_pass": absolute_pass,
        "train_propagation_network": bool(
            conclusion == "structured_multiheight_validated"
        ),
    }


def main() -> None:
    started = time.time()
    config = Config()
    base_config = e146.Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (e146.CANDIDATE_CACHE, e146.METRICS, e146.COMPONENTS, e146.FULL_FOLD)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    setup, positions, _, site, _ = load_positions()
    original_cache = np.load(e146.CANDIDATE_CACHE, allow_pickle=False)
    original = {
        key: original_cache[key]
        for key in original_cache.files
        if key != "config_json"
    }
    metadata, reconstruction = build_interaction_metadata(
        positions,
        site,
        setup.bs_positions,
        original,
        config,
        base_config,
    )
    component = np.load(e146.COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_energy = component_energy(
        component["coefficients"].astype(np.complex64)
    ).astype(np.float64)
    truth_power = component["point_power"].astype(np.float64)
    fold = np.load(e146.FULL_FOLD, allow_pickle=False).astype(np.int8)
    e146_metrics = np.load(e146.METRICS, allow_pickle=False)
    horizontal_axis = e146_metrics["horizontal_array_parameter"].astype(np.float64)
    vertical_axis = e146_metrics["vertical_array_parameter"].astype(np.float64)
    current_coordinate = e146.map_coordinates_linear_array(
        original, horizontal_axis, vertical_axis
    )
    planar_coordinate, planar_diagnostics = planar_coordinates(
        current_coordinate,
        original,
        metadata,
        site,
        setup.bs_positions,
        vertical_axis,
        config,
    )
    multi_coordinate, multi_mask, multi_type, multi_diagnostics = (
        build_multiheight_coordinates(
            current_coordinate,
            original,
            metadata,
            site,
            setup.bs_positions,
            vertical_axis,
            config,
        )
    )
    permutations, control_diagnostics = e146.build_control_permutations(
        positions, site, fold, setup.bs_positions, base_config
    )

    definitions = {
        "current_single_height": (
            current_coordinate,
            original["mask"],
            original["path_type"],
        ),
        "fixed_plane_z1p5": (
            planar_coordinate,
            original["mask"],
            original["path_type"],
        ),
        "surface_range_multiheight": (
            multi_coordinate,
            multi_mask,
            multi_type,
        ),
    }
    variants: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {"fold": fold, "site": site.astype(np.int8)}
    for name, (coordinate, mask, path_type) in definitions.items():
        print(f"E147 evaluate {name}", flush=True)
        variants[name], local_arrays = evaluate_variant(
            coordinate,
            mask,
            path_type,
            truth_coordinate,
            truth_energy,
            truth_power,
            fold,
            permutations,
            base_config,
        )
        for key, value in local_arrays.items():
            arrays[f"{name}_{key}"] = value

    shuffled_coordinate = shuffle_v_within_point(
        multi_coordinate, multi_mask, multi_type, config.seed
    )
    shuffled_raw = e146.evaluate_coverage(
        shuffled_coordinate,
        multi_mask,
        multi_type,
        truth_coordinate,
        truth_energy,
        truth_power,
        base_config,
    )
    shuffled = {
        "global_energy_coverage": float(shuffled_raw["global_energy_coverage"]),
        "global_quality_coverage": float(shuffled_raw["global_quality_coverage"]),
        "nearest_full_coverage": float(shuffled_raw["global_nearest_full"]),
        "fold_energy_coverage": e146.fold_summary(
            np.asarray(shuffled_raw["point_coverage"]), fold, truth_energy
        ),
        "same_candidate_count": True,
        "same_per_point_v_marginal": True,
        "only_parent_h_delay_to_v_association_destroyed": True,
    }

    current = variants["current_single_height"]
    for name in ("fixed_plane_z1p5", "surface_range_multiheight"):
        variants[name]["delta_from_current"] = {
            "global_energy_coverage": float(variants[name]["global_energy_coverage"])
            - float(current["global_energy_coverage"]),
            "nearest_full_coverage": float(variants[name]["nearest_full_coverage"])
            - float(current["nearest_full_coverage"]),
            "fold_energy_coverage": fold_delta(
                variants[name], current, "fold_energy_coverage"
            ),
            "fold_nearest_full_coverage": fold_delta(
                variants[name], current, "fold_nearest_full_coverage"
            ),
        }

    old_report = json.loads(e146.REPORT.read_text(encoding="utf-8"))
    baseline_reproduction = {
        "expected_energy_coverage": float(
            old_report["actual"]["global_energy_coverage"]
        ),
        "observed_energy_coverage": float(current["global_energy_coverage"]),
        "absolute_error": abs(
            float(old_report["actual"]["global_energy_coverage"])
            - float(current["global_energy_coverage"])
        ),
        "expected_nearest_full": float(
            old_report["actual"]["nearest_full_coverage"]
        ),
        "observed_nearest_full": float(current["nearest_full_coverage"]),
    }
    if baseline_reproduction["absolute_error"] > 1e-8:
        raise RuntimeError(
            "E146 baseline reproduction failed: "
            f"{baseline_reproduction['absolute_error']}"
        )

    decision = decide(variants, shuffled, config)
    np.savez_compressed(METRICS, **arrays)
    payload = {
        "experiment": VERSION,
        "status": "analysis_only_information_gate",
        "final_goal": "Round 2 online score above 0.70 from official inputs",
        "single_experimental_goal": (
            "Determine whether wall/corner V should be represented by a fixed plane, "
            "one interpolated height, or a bounded multi-height state."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "v12_used": False,
        "network_trained": False,
        "submission_generated": False,
        "config": asdict(config),
        "fixed_controls": {
            "candidate_budget": base_config.candidate_budget,
            "horizontal_coordinate_unchanged": True,
            "delay_coordinate_unchanged": True,
            "direct_and_ground_candidates_unchanged": True,
            "matching": "same E146 mass-weighted one-to-one Hungarian",
            "spatial_mismatch_control": control_diagnostics,
        },
        "candidate_reconstruction": reconstruction,
        "baseline_reproduction": baseline_reproduction,
        "planar_diagnostics": planar_diagnostics,
        "multiheight_diagnostics": multi_diagnostics,
        "variants": variants,
        "multiheight_v_shuffle_control": shuffled,
        "decision": decision,
        "outputs": {
            "metadata": PATHS.relative(METADATA_CACHE),
            "metrics": PATHS.relative(METRICS),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
