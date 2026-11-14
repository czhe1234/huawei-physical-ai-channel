#!/usr/bin/env python3
"""Test globally shared surface-sequence laws for physical path energy.

The v522 delay-angle candidates already cover most official support. This
experiment keeps those destinations fixed and predicts only a scalar energy
correction per path. The new information is the ordered interaction sequence:
official point-cloud wall patches, interaction vertices, segment geometry, and
incidence directions. A zero-action alpha remains available at every model
selection step.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
import torch
import torch.nn as nn

import analyze_v283_joint_geometry_path_renderer as v283
import analyze_v313_highdepth_matched_path_renderer as v313
import analyze_v545_path_block_family_oracle as v545
import generate_v523_end_to_end_path_pdp_residual as v523
import generate_v543_disagreement_conditioned_cell_fusion as v543
import train_v518_path_pol_vertical_correction as v518
import train_v570_global_path_family_graph_renderer as v570
from project_paths import PATHS


VERSION_NUMBER = 572
VERSION = "v572_surface_sequence_path_gain_renderer"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
FOLD_DIR = CACHE_DIR / "fold_results"
SURFACE_CACHE = CACHE_DIR / "v572_aligned_surface_sequences_2500.npz"
ORACLE_CACHE = CACHE_DIR / "v572_nonnegative_path_energy_oracle.npz"
OOF_CACHE = CACHE_DIR / "v572_surface_sequence_oof.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

SOURCE_DATA = PATHS.cache_dir(522) / "v522_source_path_coefficients_2500.npz"
PATH_ROOT = PATHS.cache_dir(475) / "shards"
OOF_SOURCE = v545.OOF_SOURCE
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TRAIN_POWER = v545.TRAIN_POWER
SCENE_CACHE = v283.SCENE_CACHE

POINT_COUNT = 2000
ALL_POINT_COUNT = 2500
PATH_LIMIT = 32
MAX_STEPS = 16
HEIGHT = 192
WIDTH = 16
SUBCARRIER_SPACING = 30_000.0
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
REGIME_NAMES = ("clear", "light", "medium", "heavy")
FLOOR = 1e-12
SEED = 572_001
ALPHA_GRID = np.linspace(0.0, 1.0, 11, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--sequence-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--max-log-gain", type=float, default=4.0)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--oracle-ridge", type=float, default=1e-3)
    parser.add_argument("--oracle-aux-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-surface-cache", action="store_true")
    parser.add_argument("--force-fold", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def require_inputs() -> None:
    required = (SOURCE_DATA, OOF_SOURCE, TRAIN_POS, TRAIN_POWER, SCENE_CACHE)
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    shards = sorted(PATH_ROOT.glob("shard_*/path_targets.npz"))
    if len(shards) != 25:
        missing.append(f"expected 25 v475 shards, found {len(shards)}")
    if missing:
        raise FileNotFoundError(f"missing v572 inputs: {missing}")


def raw_candidate_indices(data: np.lib.npyio.NpzFile, row: int) -> np.ndarray:
    valid = data["valid"][row].astype(bool)
    original = np.flatnonzero(valid)
    if not len(original):
        return original
    tau = data["tau"][row, valid].astype(np.float64)
    phi = data["phi_t"][row, valid].astype(np.float64)
    depth = data["interaction_count"][row, valid].astype(np.float64)
    delay_center = np.mod(HEIGHT * SUBCARRIER_SPACING * tau, HEIGHT)
    h_center = np.mod(
        8.0 * np.sin(phi - math.radians(-30.0)),
        WIDTH,
    )
    key_delay = np.rint(delay_center * 8.0).astype(np.int32)
    key_h = np.rint(h_center * 4.0).astype(np.int32)
    order = np.lexsort((tau, depth))
    selected: list[int] = []
    seen: set[tuple[int, int]] = set()
    for index in order:
        key = (int(key_delay[index]), int(key_h[index]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(int(index))
    return original[np.asarray(selected, dtype=np.int64)]


def match_selected_paths(
    source: dict[str, np.ndarray],
    point_id: int,
    raw: np.lib.npyio.NpzFile,
    row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output = np.full(PATH_LIMIT, -1, dtype=np.int16)
    delay_error = np.full(PATH_LIMIT, np.inf, dtype=np.float32)
    angle_error = np.full(PATH_LIMIT, np.inf, dtype=np.float32)
    depth_error = np.full(PATH_LIMIT, np.inf, dtype=np.float32)
    raw_indices = raw_candidate_indices(raw, row)
    if not len(raw_indices):
        return output, delay_error, angle_error, depth_error

    tau = raw["tau"][row, raw_indices].astype(np.float64)
    phi = raw["phi_t"][row, raw_indices].astype(np.float64)
    depth = raw["interaction_count"][row, raw_indices].astype(np.float64)
    h_center = np.mod(8.0 * np.sin(phi - math.radians(-30.0)), WIDTH)
    for path in np.flatnonzero(source["valid"][point_id]):
        target_tau = float(source["tau"][point_id, path])
        target_h = float(source["h_center"][point_id, path])
        target_depth = float(source["depth"][point_id, path])
        local_delay = np.abs(tau - target_tau) * HEIGHT * SUBCARRIER_SPACING
        local_angle = np.minimum(
            np.abs(h_center - target_h),
            WIDTH - np.abs(h_center - target_h),
        )
        local_depth = np.abs(depth - target_depth)
        cost = local_delay + local_angle + 0.1 * local_depth
        choice = int(np.argmin(cost))
        delay_error[path] = float(local_delay[choice])
        angle_error[path] = float(local_angle[choice])
        depth_error[path] = float(local_depth[choice])
        if (
            local_delay[choice] <= 1e-4
            and local_angle[choice] <= 1e-4
            and local_depth[choice] <= 0.1
        ):
            output[path] = int(raw_indices[choice])
    return output, delay_error, angle_error, depth_error


def safe_unit(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(length, 1e-9), length[..., 0]


def numeric_step_features(
    positions: np.ndarray,
    vertices: np.ndarray,
    valid: np.ndarray,
    patch_id: np.ndarray,
    depth: np.ndarray,
    patches: np.ndarray,
) -> np.ndarray:
    point_count, path_count, step_count = valid.shape
    rx = positions[:, None, None, :].astype(np.float64)
    vertex = vertices.astype(np.float64)
    previous = np.empty_like(vertex)
    previous[:, :, 0] = BS_POSITION
    previous[:, :, 1:] = vertex[:, :, :-1]
    following = np.empty_like(vertex)
    following[:, :, :-1] = vertex[:, :, 1:]
    following[:] = np.where(
        (np.arange(step_count)[None, None, :, None] + 1 >= depth[:, :, None, None]),
        rx,
        following,
    )
    incoming = vertex - previous
    outgoing = following - vertex
    incoming_unit, incoming_length = safe_unit(incoming)
    outgoing_unit, outgoing_length = safe_unit(outgoing)

    safe_patch = np.clip(patch_id - 1, 0, len(patches) - 1)
    patch = patches[safe_patch].astype(np.float64)
    patch_valid = (patch_id > 0) & (patch_id <= len(patches))
    patch *= patch_valid[..., None]
    normal = np.zeros(vertex.shape, dtype=np.float64)
    normal[..., :2] = patch[..., :2]
    tangent = np.zeros_like(normal)
    tangent[..., 0] = -normal[..., 1]
    tangent[..., 1] = normal[..., 0]
    perpendicular = (
        np.sum(normal[..., :2] * vertex[..., :2], axis=-1) - patch[..., 2]
    )
    tangent_coordinate = np.sum(tangent[..., :2] * vertex[..., :2], axis=-1)
    tangent_mid = 0.5 * (patch[..., 3] + patch[..., 4])
    tangent_span = np.maximum(patch[..., 4] - patch[..., 3], 1.0)
    z_mid = 0.5 * (patch[..., 5] + patch[..., 6])
    z_span = np.maximum(patch[..., 6] - patch[..., 5], 1.0)
    incidence_in = np.abs(np.sum(incoming_unit * normal, axis=-1))
    incidence_out = np.abs(np.sum(outgoing_unit * normal, axis=-1))
    turning = np.sum(incoming_unit * outgoing_unit, axis=-1)

    segment_length = np.where(valid, incoming_length, 0.0)
    total_length = np.maximum(segment_length.sum(axis=2, keepdims=True), 1e-6)
    cumulative = np.cumsum(segment_length, axis=2) / total_length
    direct = np.linalg.norm(positions.astype(np.float64) - BS_POSITION, axis=1)
    direct = direct[:, None, None]
    depth_safe = np.maximum(depth[:, :, None], 1)
    step_index = np.arange(step_count, dtype=np.float64)[None, None, :]
    depth_step = np.broadcast_to(depth[:, :, None], valid.shape)
    total_step = np.broadcast_to(total_length, valid.shape)
    direct_step = np.broadcast_to(direct, valid.shape)

    features = np.concatenate(
        [
            vertex / np.asarray([500.0, 500.0, 50.0]),
            (vertex - rx) / np.asarray([500.0, 500.0, 50.0]),
            (vertex - BS_POSITION) / np.asarray([500.0, 500.0, 50.0]),
            incoming_unit,
            outgoing_unit,
            (incoming_length / 500.0)[..., None],
            (outgoing_length / 500.0)[..., None],
            incidence_in[..., None],
            incidence_out[..., None],
            turning[..., None],
            (perpendicular / 8.0)[..., None],
            ((tangent_coordinate - tangent_mid) / tangent_span)[..., None],
            ((vertex[..., 2] - z_mid) / z_span)[..., None],
            patch[..., 0:2],
            (patch[..., 2:3] / 500.0),
            ((patch[..., 4:5] - patch[..., 3:4]) / 250.0),
            ((patch[..., 5:6] + patch[..., 6:7]) / 100.0),
            ((patch[..., 6:7] - patch[..., 5:6]) / 50.0),
            (np.log1p(np.maximum(patch[..., 7:8], 0.0)) / 10.0),
            patch[..., 8:10],
            (patch[..., 10:12] / 180.0),
            (cumulative[..., None]),
            ((step_index + 1.0) / depth_safe)[..., None],
            (depth_step / MAX_STEPS)[..., None],
            (total_step / 1000.0)[..., None],
            (
                (total_step - direct_step)
                / np.maximum(direct_step, 1.0)
            )[..., None],
            patch_valid[..., None].astype(np.float64),
        ],
        axis=-1,
    ).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
    features *= valid[..., None]
    if features.shape[:3] != (point_count, path_count, step_count):
        raise ValueError(f"unexpected step feature shape: {features.shape}")
    return features


def build_surface_cache(source: dict[str, np.ndarray], force: bool) -> dict[str, object]:
    if SURFACE_CACHE.is_file() and not force:
        with np.load(SURFACE_CACHE) as loaded:
            return json.loads(str(loaded["summary_json"].item()))

    positions = source["positions"].astype(np.float32)
    aligned_index = np.full((ALL_POINT_COUNT, PATH_LIMIT), -1, dtype=np.int16)
    delay_error = np.full((ALL_POINT_COUNT, PATH_LIMIT), np.inf, dtype=np.float32)
    angle_error = np.full_like(delay_error, np.inf)
    depth_error = np.full_like(delay_error, np.inf)
    step_valid = np.zeros(
        (ALL_POINT_COUNT, PATH_LIMIT, MAX_STEPS), dtype=bool
    )
    interaction = np.zeros(
        (ALL_POINT_COUNT, PATH_LIMIT, MAX_STEPS), dtype=np.uint8
    )
    object_token = np.zeros_like(interaction, dtype=np.uint8)
    vertices = np.zeros(
        (ALL_POINT_COUNT, PATH_LIMIT, MAX_STEPS, 3), dtype=np.float32
    )
    depth = np.zeros((ALL_POINT_COUNT, PATH_LIMIT), dtype=np.int16)
    object_values: set[int] = set()
    started = time.monotonic()

    for shard_index, path in enumerate(sorted(PATH_ROOT.glob("shard_*/path_targets.npz"))):
        with np.load(path.parent / "metadata.npz") as metadata, np.load(path) as raw:
            point_ids = metadata["position_id"].astype(np.int64)
            for row, point_id in enumerate(point_ids):
                matched, local_delay, local_angle, local_depth = match_selected_paths(
                    source, int(point_id), raw, row
                )
                aligned_index[point_id] = matched
                delay_error[point_id] = local_delay
                angle_error[point_id] = local_angle
                depth_error[point_id] = local_depth
                for target_path in np.flatnonzero(matched >= 0):
                    raw_path = int(matched[target_path])
                    count = min(
                        int(raw["interaction_count"][row, raw_path]),
                        MAX_STEPS,
                    )
                    depth[point_id, target_path] = count
                    if count <= 0:
                        continue
                    step_valid[point_id, target_path, :count] = True
                    local_interaction = raw["interactions"][
                        row, raw_path, :count
                    ].astype(np.uint8)
                    local_object = raw["objects"][row, raw_path, :count].astype(np.int64)
                    interaction[point_id, target_path, :count] = local_interaction
                    vertices[point_id, target_path, :count] = raw["vertices"][
                        row, raw_path, :count
                    ]
                    object_values.update(int(value) for value in local_object)
        print(
            f"v572 align shard={shard_index + 1}/25 "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    object_lookup = {value: index + 1 for index, value in enumerate(sorted(object_values))}
    for path in sorted(PATH_ROOT.glob("shard_*/path_targets.npz")):
        with np.load(path.parent / "metadata.npz") as metadata, np.load(path) as raw:
            point_ids = metadata["position_id"].astype(np.int64)
            for row, point_id in enumerate(point_ids):
                for target_path in np.flatnonzero(aligned_index[point_id] >= 0):
                    raw_path = int(aligned_index[point_id, target_path])
                    count = int(depth[point_id, target_path])
                    if count:
                        values = raw["objects"][row, raw_path, :count].astype(np.int64)
                        object_token[point_id, target_path, :count] = np.asarray(
                            [object_lookup[int(value)] for value in values],
                            dtype=np.uint8,
                        )

    scene = np.load(SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    patch_id = np.zeros_like(interaction, dtype=np.int16)
    for start in range(0, ALL_POINT_COUNT, 100):
        stop = min(start + 100, ALL_POINT_COUNT)
        local_patch = v313.map_first_patch(
            vertices[start:stop],
            step_valid[start:stop],
            patches,
        ).astype(np.int16)
        ground = (
            step_valid[start:stop]
            & (vertices[start:stop, ..., 2] <= 0.4)
            & (interaction[start:stop] == 1)
        )
        local_patch[ground] = len(patches) + 1
        patch_id[start:stop] = local_patch
        print(f"v572 wall mapping {stop}/{ALL_POINT_COUNT}", flush=True)

    step_numeric_parts = []
    for start in range(0, ALL_POINT_COUNT, 100):
        stop = min(start + 100, ALL_POINT_COUNT)
        step_numeric_parts.append(
            numeric_step_features(
                positions[start:stop],
                vertices[start:stop],
                step_valid[start:stop],
                patch_id[start:stop],
                depth[start:stop],
                patches,
            ).astype(np.float16)
        )
    step_numeric = np.concatenate(step_numeric_parts, axis=0)

    selected_valid = source["valid"].astype(bool)
    aligned = selected_valid & (aligned_index >= 0)
    finite_delay = delay_error[aligned]
    finite_angle = angle_error[aligned]
    finite_depth = depth_error[aligned]
    summary: dict[str, object] = {
        "selected_paths": int(np.sum(selected_valid)),
        "aligned_paths": int(np.sum(aligned)),
        "alignment_fraction": float(np.mean(aligned[selected_valid])),
        "unmatched_points": int(np.sum(np.any(selected_valid & ~aligned, axis=1))),
        "valid_steps": int(np.sum(step_valid)),
        "wall_steps": int(np.sum((patch_id > 0) & (patch_id <= len(patches)))),
        "ground_steps": int(np.sum(patch_id == len(patches) + 1)),
        "wall_patch_count": int(len(patches)),
        "used_wall_patches": int(
            len(np.unique(patch_id[(patch_id > 0) & (patch_id <= len(patches))]))
        ),
        "object_lookup": {str(key): value for key, value in object_lookup.items()},
        "step_numeric_dim": int(step_numeric.shape[-1]),
        "alignment_error": {
            "delay_bin_max": float(np.max(finite_delay)) if len(finite_delay) else None,
            "h_bin_max": float(np.max(finite_angle)) if len(finite_angle) else None,
            "depth_max": float(np.max(finite_depth)) if len(finite_depth) else None,
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SURFACE_CACHE,
        aligned_index=aligned_index,
        step_valid=step_valid,
        interaction=interaction,
        object_token=object_token,
        patch_id=patch_id,
        step_numeric=step_numeric,
        depth=depth,
        summary_json=np.asarray(json.dumps(summary, ensure_ascii=False)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def exact_spectral_kernels(
    tau: np.ndarray,
    h_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency_index = np.arange(HEIGHT, dtype=np.float64) - HEIGHT // 2
    horizontal_index = np.arange(WIDTH, dtype=np.float64)
    delay = np.empty(tau.shape + (HEIGHT,), dtype=np.float16)
    angle = np.empty(h_center.shape + (WIDTH,), dtype=np.float16)
    for start in range(0, len(tau), 100):
        stop = min(start + 100, len(tau))
        frequency_phase = np.exp(
            -2j
            * np.pi
            * frequency_index[None, None, :]
            * SUBCARRIER_SPACING
            * tau[start:stop, :, None]
        )
        delay_power = np.abs(np.fft.fft(frequency_phase, axis=2)) ** 2
        delay_power /= np.maximum(delay_power.sum(axis=2, keepdims=True), FLOOR)
        horizontal_phase = np.exp(
            2j
            * np.pi
            * horizontal_index[None, None, :]
            * h_center[start:stop, :, None]
            / WIDTH
        )
        angle_power = np.abs(np.fft.fft(horizontal_phase, axis=2)) ** 2
        angle_power /= np.maximum(angle_power.sum(axis=2, keepdims=True), FLOOR)
        delay[start:stop] = delay_power.astype(np.float16)
        angle[start:stop] = angle_power.astype(np.float16)
    return delay, angle


def build_path_energy_oracle(
    data: dict[str, np.ndarray],
    ridge: float,
) -> dict[str, object]:
    if ORACLE_CACHE.is_file():
        with np.load(ORACLE_CACHE) as loaded:
            saved_ridge = float(loaded["ridge"])
            if math.isclose(saved_ridge, ridge, rel_tol=0.0, abs_tol=1e-12):
                data["oracle_weight"] = loaded["oracle_weight"].astype(np.float32)
                data["oracle_score"] = loaded["oracle_score"].astype(np.float32)
                data["source_physical_score"] = loaded[
                    "source_physical_score"
                ].astype(np.float32)
                return json.loads(str(loaded["summary_json"].item()))

    oracle_weight = np.zeros((POINT_COUNT, PATH_LIMIT), dtype=np.float32)
    source_prediction = np.zeros((POINT_COUNT, HEIGHT * WIDTH), dtype=np.float32)
    oracle_prediction = np.zeros_like(source_prediction)
    started = time.monotonic()
    for point_id in range(POINT_COUNT):
        valid = data["path_valid"][point_id]
        count = int(np.sum(valid))
        if count == 0:
            continue
        delay = data["delay_kernel"][point_id, valid].astype(np.float64)
        angle = data["angle_kernel"][point_id, valid].astype(np.float64)
        matrix = np.einsum("kd,kh->dhk", delay, angle).reshape(
            HEIGHT * WIDTH, count
        )
        truth = data["truth"][point_id].astype(np.float64)
        source = data["source_energy"][point_id, valid].astype(np.float64)
        source /= max(float(np.sum(source)), FLOOR)
        source_local = matrix @ source
        source_local /= max(float(np.sum(source_local)), FLOOR)
        source_prediction[point_id] = source_local.astype(np.float32)
        if ridge > 0.0:
            augmented_matrix = np.vstack(
                [matrix, math.sqrt(ridge) * np.eye(count, dtype=np.float64)]
            )
            augmented_target = np.concatenate(
                [truth, math.sqrt(ridge) * source]
            )
        else:
            augmented_matrix = matrix
            augmented_target = truth
        weight, _ = nnls(augmented_matrix, augmented_target, maxiter=500)
        if float(np.sum(weight)) <= FLOOR:
            weight = source
        weight /= max(float(np.sum(weight)), FLOOR)
        local = np.zeros(PATH_LIMIT, dtype=np.float32)
        local[valid] = weight.astype(np.float32)
        oracle_weight[point_id] = local
        prediction = matrix @ weight
        prediction /= max(float(np.sum(prediction)), FLOOR)
        oracle_prediction[point_id] = prediction.astype(np.float32)
        if (point_id + 1) % 250 == 0:
            print(
                f"v572 path-energy oracle {point_id + 1}/{POINT_COUNT} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    source_metrics = v543.spectrum_scores(data["truth"], source_prediction)
    oracle_metrics = v543.spectrum_scores(data["truth"], oracle_prediction)
    source_score = source_metrics["weighted"].astype(np.float32)
    oracle_score = oracle_metrics["weighted"].astype(np.float32)
    valid_points = data["point_valid"]
    summary = {
        "ridge": ridge,
        "source_physical": {
            key: float(np.mean(value[valid_points]))
            for key, value in source_metrics.items()
        },
        "nonnegative_oracle": {
            key: float(np.mean(value[valid_points]))
            for key, value in oracle_metrics.items()
        },
        "weighted_gain": float(
            np.mean(oracle_score[valid_points] - source_score[valid_points])
        ),
        "positive_points": int(
            np.sum(oracle_score[valid_points] > source_score[valid_points])
        ),
        "scored_points": int(np.sum(valid_points)),
    }
    data["oracle_weight"] = oracle_weight
    data["oracle_score"] = oracle_score
    data["source_physical_score"] = source_score
    np.savez_compressed(
        ORACLE_CACHE,
        ridge=np.asarray(ridge, dtype=np.float64),
        oracle_weight=oracle_weight,
        oracle_score=oracle_score,
        source_physical_score=source_score,
        summary_json=np.asarray(json.dumps(summary, ensure_ascii=False)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def load_data(
    oracle_ridge: float = 1e-3,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    with np.load(SOURCE_DATA) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    surface_summary = build_surface_cache(source, force=False)
    with np.load(SURFACE_CACHE) as loaded:
        surface = {name: np.asarray(loaded[name]) for name in loaded.files if name != "summary_json"}
    with np.load(OOF_SOURCE) as loaded:
        truth = loaded["truth"].astype(np.float32)
        base = loaded["calibrated"].astype(np.float32)
        regimes = loaded["regimes"].astype("U8")
    positions = np.load(TRAIN_POS).astype(np.float32)
    if not np.allclose(source["positions"][:POINT_COUNT], positions, atol=1e-5):
        raise ValueError("v522 train positions do not match official train positions")
    power = np.load(TRAIN_POWER).astype(np.float64)
    point_valid = power != 0.0
    regime_code = np.asarray(
        [REGIME_NAMES.index(str(name)) for name in regimes], dtype=np.int8
    )
    base_features = v523.source_base_features(source, np.pad(regime_code, (0, 500)))
    base_features = v518.input_features(
        base_features,
        source["source_coefficients"].astype(np.complex128),
    )
    source_energy = np.sum(
        np.abs(source["source_coefficients"]) ** 2,
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    source_energy *= source["valid"]
    delay_kernel, angle_kernel = exact_spectral_kernels(
        source["tau"].astype(np.float64),
        source["h_center"].astype(np.float64),
    )
    data = {
        "truth": truth,
        "base": base,
        "regimes": regimes,
        "regime_code": regime_code,
        "positions": positions,
        "point_valid": point_valid,
        "folds": v570.contiguous_fold_codes(positions),
        "path_valid": source["valid"][:POINT_COUNT].astype(bool),
        "base_features": base_features[:POINT_COUNT].astype(np.float32),
        "source_energy": source_energy[:POINT_COUNT],
        "delay_kernel": delay_kernel[:POINT_COUNT],
        "angle_kernel": angle_kernel[:POINT_COUNT],
        "step_valid": surface["step_valid"][:POINT_COUNT].astype(bool),
        "interaction": surface["interaction"][:POINT_COUNT].astype(np.int64),
        "object_token": surface["object_token"][:POINT_COUNT].astype(np.int64),
        "patch_id": surface["patch_id"][:POINT_COUNT].astype(np.int64),
        "step_numeric": surface["step_numeric"][:POINT_COUNT].astype(np.float32),
    }
    oracle_summary = build_path_energy_oracle(data, oracle_ridge)
    return data, surface_summary, oracle_summary


def standardizers(
    data: dict[str, np.ndarray], fit_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_mask = data["path_valid"][fit_ids]
    base = data["base_features"][fit_ids][path_mask]
    base_mean = base.mean(axis=0).astype(np.float32)
    base_std = np.maximum(base.std(axis=0), 1e-4).astype(np.float32)
    step_mask = data["step_valid"][fit_ids]
    step = data["step_numeric"][fit_ids][step_mask]
    if len(step):
        step_mean = step.mean(axis=0).astype(np.float32)
        step_std = np.maximum(step.std(axis=0), 1e-4).astype(np.float32)
    else:
        step_mean = np.zeros(data["step_numeric"].shape[-1], dtype=np.float32)
        step_std = np.ones_like(step_mean)
    return base_mean, base_std, step_mean, step_std


def path_shuffle_map(
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    mapping = np.arange(POINT_COUNT * PATH_LIMIT, dtype=np.int64).reshape(
        POINT_COUNT, PATH_LIMIT
    )
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int]] = []
    for point_id in ids:
        for path in np.flatnonzero(data["path_valid"][point_id]):
            rows.append((int(point_id), int(path)))
    if not rows:
        return mapping
    point = np.asarray([value[0] for value in rows], dtype=np.int64)
    path = np.asarray([value[1] for value in rows], dtype=np.int64)
    depth = np.rint(
        data["base_features"][point, path, 15] * 16.0
    ).astype(np.int16)
    log_energy = np.log(np.maximum(data["source_energy"][point, path], FLOOR))
    edges = np.quantile(log_energy, np.linspace(0.0, 1.0, 6))
    energy_bin = np.searchsorted(edges[1:-1], log_energy, side="right")
    regime = data["regime_code"][point]
    flat = point * PATH_LIMIT + path
    for regime_value in range(4):
        for depth_value in np.unique(depth):
            for energy_value in range(5):
                group = np.flatnonzero(
                    (regime == regime_value)
                    & (depth == depth_value)
                    & (energy_bin == energy_value)
                )
                if len(group) <= 1:
                    continue
                permuted = group[rng.permutation(len(group))]
                mapping[point[group], path[group]] = flat[permuted]
    return mapping


class PathEnergyRenderer(nn.Module):
    def __init__(
        self,
        base_dim: int,
        step_dim: int,
        hidden_dim: int,
        sequence_dim: int,
        patch_count: int,
        object_count: int,
        max_log_gain: float,
        use_surface: bool,
    ) -> None:
        super().__init__()
        self.use_surface = use_surface
        self.max_log_gain = max_log_gain
        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if use_surface:
            self.interaction_embedding = nn.Embedding(8, 12, padding_idx=0)
            self.object_embedding = nn.Embedding(object_count + 1, 10, padding_idx=0)
            self.patch_embedding = nn.Embedding(patch_count + 1, 20, padding_idx=0)
            nn.init.zeros_(self.patch_embedding.weight)
            self.step_encoder = nn.Sequential(
                nn.Linear(step_dim + 45, sequence_dim),
                nn.LayerNorm(sequence_dim),
                nn.GELU(),
                nn.Linear(sequence_dim, sequence_dim),
            )
            sequence_layer = nn.TransformerEncoderLayer(
                d_model=sequence_dim,
                nhead=4,
                dim_feedforward=4 * sequence_dim,
                dropout=0.06,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(sequence_layer, num_layers=2)
            self.surface_projection = nn.Sequential(
                nn.Linear(2 * sequence_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.combine = nn.Linear(2 * hidden_dim, hidden_dim)
        path_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=4 * hidden_dim,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(path_layer, num_layers=3)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_surface(
        self,
        step_numeric: torch.Tensor,
        interaction: torch.Tensor,
        object_token: torch.Tensor,
        patch_id: torch.Tensor,
        step_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, paths, steps, _ = step_numeric.shape
        position = torch.arange(steps, device=step_numeric.device, dtype=torch.float32)
        phase = 2.0 * math.pi * position / max(steps, 1)
        position_feature = torch.stack(
            [
                position / max(steps - 1, 1),
                torch.sin(phase),
                torch.cos(phase),
            ],
            dim=1,
        )
        position_feature = position_feature[None, None].expand(batch, paths, -1, -1)
        token = torch.cat(
            [
                step_numeric,
                self.interaction_embedding(interaction),
                self.object_embedding(object_token),
                self.patch_embedding(patch_id),
                position_feature,
            ],
            dim=-1,
        )
        token = self.step_encoder(token).reshape(batch * paths, steps, -1)
        mask = step_valid.reshape(batch * paths, steps)
        safe_mask = mask.clone()
        empty = ~torch.any(safe_mask, dim=1)
        if torch.any(empty):
            safe_mask[empty, 0] = True
        token = self.sequence_encoder(token, src_key_padding_mask=~safe_mask)
        weight = mask.to(token.dtype)[..., None]
        mean = torch.sum(token * weight, dim=1) / torch.clamp(
            torch.sum(weight, dim=1), min=1.0
        )
        maximum = torch.max(
            torch.where(mask[..., None], token, torch.full_like(token, -1e4)),
            dim=1,
        ).values
        maximum[empty] = 0.0
        return self.surface_projection(torch.cat([mean, maximum], dim=1)).reshape(
            batch, paths, -1
        )

    def forward(
        self,
        base_features: torch.Tensor,
        path_valid: torch.Tensor,
        step_numeric: torch.Tensor,
        interaction: torch.Tensor,
        object_token: torch.Tensor,
        patch_id: torch.Tensor,
        step_valid: torch.Tensor,
    ) -> torch.Tensor:
        token = self.base_encoder(base_features)
        if self.use_surface:
            surface = self.encode_surface(
                step_numeric,
                interaction,
                object_token,
                patch_id,
                step_valid,
            )
            token = self.combine(torch.cat([token, surface], dim=-1))
        safe_valid = path_valid.clone()
        empty = ~torch.any(safe_valid, dim=1)
        if torch.any(empty):
            safe_valid[empty, 0] = True
        token = self.path_encoder(token, src_key_padding_mask=~safe_valid)
        log_gain = self.max_log_gain * torch.tanh(self.output(token)[..., 0])
        return log_gain * path_valid.to(log_gain.dtype)


def gather_surface(
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    path_map: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path_map is None:
        return (
            data["step_numeric"][ids],
            data["interaction"][ids],
            data["object_token"][ids],
            data["patch_id"][ids],
            data["step_valid"][ids],
        )
    flat = path_map[ids]
    point = flat // PATH_LIMIT
    path = flat % PATH_LIMIT
    return (
        data["step_numeric"][point, path],
        data["interaction"][point, path],
        data["object_token"][point, path],
        data["patch_id"][point, path],
        data["step_valid"][point, path],
    )


def batch_tensors(
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    path_map: np.ndarray | None = None,
) -> tuple[torch.Tensor, ...]:
    base_mean, base_std, step_mean, step_std = normalization
    base = (
        (data["base_features"][ids] - base_mean[None, None])
        / base_std[None, None]
    ).astype(np.float32)
    base *= data["path_valid"][ids, :, None]
    numeric, interaction, object_token, patch_id, step_valid = gather_surface(
        ids, data, path_map
    )
    numeric = (
        (numeric - step_mean[None, None, None])
        / step_std[None, None, None]
    ).astype(np.float32)
    numeric *= step_valid[..., None]
    return (
        torch.from_numpy(base).to(device),
        torch.from_numpy(data["path_valid"][ids]).to(device),
        torch.from_numpy(numeric).to(device),
        torch.from_numpy(interaction).to(device),
        torch.from_numpy(object_token).to(device),
        torch.from_numpy(patch_id).to(device),
        torch.from_numpy(step_valid).to(device),
        torch.from_numpy(data["source_energy"][ids]).to(device),
        torch.from_numpy(data["delay_kernel"][ids].astype(np.float32)).to(device),
        torch.from_numpy(data["angle_kernel"][ids].astype(np.float32)).to(device),
        torch.from_numpy(data["base"][ids]).to(device),
        torch.from_numpy(data["truth"][ids]).to(device),
        torch.from_numpy(data["oracle_weight"][ids]).to(device),
    )


def render_spectrum(
    source_energy: torch.Tensor,
    delay_kernel: torch.Tensor,
    angle_kernel: torch.Tensor,
    log_gain: torch.Tensor,
    path_valid: torch.Tensor,
) -> torch.Tensor:
    weight = source_energy * torch.exp(log_gain) * path_valid.to(source_energy.dtype)
    spectrum = torch.einsum(
        "bk,bkd,bkh->bdh",
        weight,
        delay_kernel,
        angle_kernel,
    ).reshape(len(weight), HEIGHT * WIDTH)
    return spectrum / torch.clamp(spectrum.sum(dim=1, keepdim=True), min=1e-20)


def torch_log_blend(
    base: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    delta = torch.clamp(
        torch.log(torch.clamp(target, min=FLOOR))
        - torch.log(torch.clamp(base, min=FLOOR)),
        min=-3.0,
        max=3.0,
    )
    output = base * torch.exp(float(alpha) * delta)
    return output / torch.clamp(output.sum(dim=1, keepdim=True), min=1e-20)


def torch_scores(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    truth_grid = truth.reshape(-1, HEIGHT, WIDTH)
    pred_grid = prediction.reshape(-1, HEIGHT, WIDTH)

    def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = left.reshape(len(left), -1)
        right = right.reshape(len(right), -1)
        numerator = torch.sum(left * right, dim=1)
        denominator = torch.sqrt(
            torch.sum(left * left, dim=1) * torch.sum(right * right, dim=1)
        )
        return numerator / torch.clamp(denominator, min=1e-20)

    joint = cosine(truth_grid, pred_grid)
    truth_delay = truth_grid.sum(dim=2)
    pred_delay = pred_grid.sum(dim=2)
    truth_angle = truth_grid.sum(dim=1)
    pred_angle = pred_grid.sum(dim=1)
    delay = cosine(truth_delay, pred_delay)
    angle = cosine(truth_angle, pred_angle)
    weighted = 0.50 * joint + 0.25 * delay + 0.25 * angle
    return joint, delay, angle, weighted


def path_weight_loss(
    source_energy: torch.Tensor,
    log_gain: torch.Tensor,
    path_valid: torch.Tensor,
    oracle_weight: torch.Tensor,
) -> torch.Tensor:
    logits = torch.log(torch.clamp(source_energy, min=FLOOR)) + log_gain
    logits = torch.where(path_valid, logits, torch.full_like(logits, -1e4))
    log_probability = torch.log_softmax(logits, dim=1)
    target = 0.98 * oracle_weight
    source_fraction = source_energy / torch.clamp(
        source_energy.sum(dim=1, keepdim=True), min=FLOOR
    )
    target = target + 0.02 * source_fraction
    target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=FLOOR)
    return torch.mean(-torch.sum(target * log_probability, dim=1))


@torch.no_grad()
def evaluate_model(
    model: PathEnergyRenderer,
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    batch_size: int,
    path_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    physical = np.zeros((len(ids), HEIGHT * WIDTH), dtype=np.float32)
    abs_gain = []
    for start in range(0, len(ids), batch_size):
        local_ids = ids[start : start + batch_size]
        tensors = batch_tensors(
            local_ids, data, normalization, device, path_map=path_map
        )
        log_gain = model(*tensors[:7])
        prediction = render_spectrum(
            tensors[7], tensors[8], tensors[9], log_gain, tensors[1]
        )
        physical[start : start + len(local_ids)] = prediction.cpu().numpy()
        local_valid = tensors[1]
        abs_gain.extend(torch.abs(log_gain[local_valid]).cpu().numpy().tolist())
    diagnostics = {
        "mean_abs_log_gain": float(np.mean(abs_gain)) if abs_gain else 0.0,
        "p90_abs_log_gain": float(np.quantile(abs_gain, 0.90)) if abs_gain else 0.0,
    }
    return data["base"][ids], physical, diagnostics


def select_alpha(
    truth: np.ndarray,
    base: np.ndarray,
    physical: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    best_alpha = 0.0
    best_score = float(np.mean(v543.spectrum_scores(truth, base)["weighted"]))
    grid: dict[str, float] = {"0.0": best_score}
    for alpha in ALPHA_GRID[1:]:
        prediction = v543.log_blend(base, physical, float(alpha))
        score = float(np.mean(v543.spectrum_scores(truth, prediction)["weighted"]))
        grid[f"{float(alpha):.1f}"] = score
        if score > best_score + 1e-6:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, best_score, grid


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def make_model(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    surface_summary: dict[str, object],
    use_surface: bool,
    device: torch.device,
) -> PathEnergyRenderer:
    object_count = max(
        int(value) for value in surface_summary["object_lookup"].values()
    )
    return PathEnergyRenderer(
        base_dim=data["base_features"].shape[-1],
        step_dim=data["step_numeric"].shape[-1],
        hidden_dim=args.hidden_dim,
        sequence_dim=args.sequence_dim,
        patch_count=int(surface_summary["wall_patch_count"]) + 1,
        object_count=object_count,
        max_log_gain=args.max_log_gain,
        use_surface=use_surface,
    ).to(device)


def train_model(
    train_ids: np.ndarray,
    validation_ids: np.ndarray | None,
    data: dict[str, np.ndarray],
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    surface_summary: dict[str, object],
    device: torch.device,
    use_surface: bool,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[PathEnergyRenderer, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(data, args, surface_summary, use_surface, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(seed)
    epochs = fixed_epochs if fixed_epochs is not None else args.epochs
    best_score = float("-inf")
    best_epoch = 0
    best_alpha = 0.0
    best_state = state_dict_cpu(model)
    history = []
    pretrain_history = []
    stale = 0
    started = time.monotonic()

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        order = rng.permutation(train_ids)
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            tensors = batch_tensors(ids, data, normalization, device)
            log_gain = model(*tensors[:7])
            loss = path_weight_loss(
                tensors[7],
                log_gain,
                tensors[1],
                tensors[12],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(ids)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.pretrain_epochs:
            row = {
                "epoch": epoch,
                "path_weight_loss": running / len(train_ids),
            }
            pretrain_history.append(row)
            print(
                f"v572 {'surface' if use_surface else 'base'} "
                f"pretrain={epoch:03d} path_loss={row['path_weight_loss']:.6f}",
                flush=True,
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if validation_ids is not None and args.pretrain_epochs > 0:
        base, physical, diagnostics = evaluate_model(
            model,
            validation_ids,
            data,
            normalization,
            device,
            args.batch_size,
        )
        alpha, validation_score, _ = select_alpha(
            data["truth"][validation_ids], base, physical
        )
        best_score = validation_score
        best_epoch = 0
        best_alpha = alpha
        best_state = state_dict_cpu(model)
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "validation_score": validation_score,
                "validation_alpha": alpha,
                **diagnostics,
            }
        )
        print(
            f"v572 {'surface' if use_surface else 'base'} after_pretrain "
            f"validation={validation_score:.6f} alpha={alpha:.2f}",
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_ids)
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            tensors = batch_tensors(ids, data, normalization, device)
            log_gain = model(*tensors[:7])
            physical = render_spectrum(
                tensors[7], tensors[8], tensors[9], log_gain, tensors[1]
            )
            prediction = torch_log_blend(tensors[10], physical, args.train_alpha)
            _, _, _, score = torch_scores(tensors[11], prediction)
            valid_weight = tensors[7] / torch.clamp(
                tensors[7].sum(dim=1, keepdim=True), min=1e-20
            )
            regularization = torch.mean(
                torch.sum(valid_weight * log_gain * log_gain, dim=1)
            )
            auxiliary = path_weight_loss(
                tensors[7],
                log_gain,
                tensors[1],
                tensors[12],
            )
            loss = (
                1.0
                - torch.mean(score)
                + 2e-4 * regularization
                + args.oracle_aux_weight * auxiliary
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(ids)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            if validation_ids is not None:
                base, physical, diagnostics = evaluate_model(
                    model,
                    validation_ids,
                    data,
                    normalization,
                    device,
                    args.batch_size,
                )
                alpha, validation_score, _ = select_alpha(
                    data["truth"][validation_ids], base, physical
                )
            else:
                diagnostics = {"mean_abs_log_gain": 0.0, "p90_abs_log_gain": 0.0}
                alpha = 1.0
                validation_score = float("nan")
            row = {
                "epoch": epoch,
                "train_loss": running / len(train_ids),
                "validation_score": validation_score,
                "validation_alpha": alpha,
                **diagnostics,
            }
            history.append(row)
            print(
                f"v572 {'surface' if use_surface else 'base'} epoch={epoch:03d} "
                f"loss={row['train_loss']:.6f} validation={validation_score:.6f} "
                f"alpha={alpha:.2f}",
                flush=True,
            )
            if validation_ids is not None:
                if validation_score > best_score + 1e-5:
                    best_score = validation_score
                    best_epoch = epoch
                    best_alpha = alpha
                    best_state = state_dict_cpu(model)
                    stale = 0
                else:
                    stale += 5
                    if stale >= args.patience:
                        break

    if validation_ids is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
        best_alpha = 1.0
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_score": (
            float(best_score) if validation_ids is not None else None
        ),
        "best_alpha": float(best_alpha),
        "elapsed_seconds": float(time.monotonic() - started),
        "pretrain_history": pretrain_history,
        "history": history,
    }


def metric_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    return {
        key: float(np.mean(value))
        for key, value in v543.spectrum_scores(truth, prediction).items()
    }


def run_variant(
    fold: int,
    name: str,
    use_surface: bool,
    data: dict[str, np.ndarray],
    surface_summary: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object], dict[str, torch.Tensor]]:
    valid = data["point_valid"]
    outer = data["folds"] == fold
    fit_mask = valid & ~outer
    query_mask = valid & outer
    tune_fold = (fold + 1) % 5
    inner_validation_mask = fit_mask & (data["folds"] == tune_fold)
    inner_train_mask = fit_mask & ~inner_validation_mask
    inner_train_ids = np.flatnonzero(inner_train_mask)
    inner_validation_ids = np.flatnonzero(inner_validation_mask)
    fit_ids = np.flatnonzero(fit_mask)
    query_ids = np.flatnonzero(query_mask)
    print(
        f"v572 {name} fold={fold} inner_train={len(inner_train_ids)} "
        f"inner_validation={len(inner_validation_ids)} outer={len(query_ids)}",
        flush=True,
    )

    inner_norm = standardizers(data, inner_train_ids)
    selected_model, selection = train_model(
        inner_train_ids,
        inner_validation_ids,
        data,
        inner_norm,
        args,
        surface_summary,
        device,
        use_surface,
        args.seed + 1000 * fold + (100 if use_surface else 0),
    )
    selected_epoch = int(selection["best_epoch"])
    selected_alpha = float(selection["best_alpha"])
    del selected_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final_norm = standardizers(data, fit_ids)
    model, final_training = train_model(
        fit_ids,
        None,
        data,
        final_norm,
        args,
        surface_summary,
        device,
        use_surface,
        args.seed + 1000 * fold + (101 if use_surface else 1),
        fixed_epochs=selected_epoch,
    )
    base, physical, diagnostics = evaluate_model(
        model, query_ids, data, final_norm, device, args.batch_size
    )
    prediction = v543.log_blend(base, physical, selected_alpha)
    shuffled_prediction = None
    shuffled_diagnostics = None
    if use_surface:
        shuffled_map = path_shuffle_map(
            query_ids,
            data,
            args.seed + 30_000 + fold,
        )
        _, shuffled_physical, shuffled_diagnostics = evaluate_model(
            model,
            query_ids,
            data,
            final_norm,
            device,
            args.batch_size,
            path_map=shuffled_map,
        )
        shuffled_prediction = v543.log_blend(
            base, shuffled_physical, selected_alpha
        )
    details: dict[str, object] = {
        "name": name,
        "use_surface": use_surface,
        "selected_epoch": selected_epoch,
        "selected_alpha": selected_alpha,
        "selection": selection,
        "final_training": final_training,
        "diagnostics": diagnostics,
        "baseline": metric_summary(data["truth"][query_ids], base),
        "prediction": metric_summary(data["truth"][query_ids], prediction),
    }
    details["gain"] = {
        key: float(details["prediction"][key] - details["baseline"][key])
        for key in details["prediction"]
    }
    if shuffled_prediction is not None:
        details["shuffled_surface"] = {
            "metrics": metric_summary(data["truth"][query_ids], shuffled_prediction),
            "diagnostics": shuffled_diagnostics,
        }
        details["surface_margin_vs_shuffled"] = float(
            details["prediction"]["weighted"]
            - details["shuffled_surface"]["metrics"]["weighted"]
        )
    state = {
        "state_dict": state_dict_cpu(model),
        "base_mean": torch.from_numpy(final_norm[0]),
        "base_std": torch.from_numpy(final_norm[1]),
        "step_mean": torch.from_numpy(final_norm[2]),
        "step_std": torch.from_numpy(final_norm[3]),
    }
    return prediction, details, state


def run_fold(
    fold: int,
    data: dict[str, np.ndarray],
    surface_summary: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    fold_path = FOLD_DIR / f"fold_{fold}.npz"
    if fold_path.is_file() and not args.force_fold:
        with np.load(fold_path) as loaded:
            return json.loads(str(loaded["details"].item()))

    base_prediction, base_details, base_state = run_variant(
        fold,
        "original_path_features",
        False,
        data,
        surface_summary,
        args,
        device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    surface_prediction, surface_details, surface_state = run_variant(
        fold,
        "surface_sequence_features",
        True,
        data,
        surface_summary,
        args,
        device,
    )
    query_ids = np.flatnonzero(data["point_valid"] & (data["folds"] == fold))
    surface_gain = float(surface_details["gain"]["weighted"])
    base_gain = float(base_details["gain"]["weighted"])
    margin_base = float(
        surface_details["prediction"]["weighted"]
        - base_details["prediction"]["weighted"]
    )
    margin_shuffled = float(surface_details.get("surface_margin_vs_shuffled", 0.0))
    details = {
        "fold": fold,
        "query_points": int(len(query_ids)),
        "original": base_details,
        "surface": surface_details,
        "causal_comparison": {
            "surface_gain_vs_v543": surface_gain,
            "original_gain_vs_v543": base_gain,
            "surface_margin_vs_original": margin_base,
            "surface_margin_vs_shuffled": margin_shuffled,
            "hard_region_success": bool(
                surface_gain >= 0.005
                and margin_base >= 0.003
                and margin_shuffled >= 0.003
            ),
        },
    }
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": VERSION,
            "fold": fold,
            "original": base_state,
            "surface": surface_state,
            "original_details": base_details,
            "surface_details": surface_details,
        },
        CHECKPOINT_DIR / f"fold_{fold}.pt",
    )
    np.savez_compressed(
        fold_path,
        query_ids=query_ids.astype(np.int32),
        original_prediction=base_prediction.astype(np.float32),
        surface_prediction=surface_prediction.astype(np.float32),
        details=np.asarray(json.dumps(details, ensure_ascii=False)),
    )
    print(json.dumps(details["causal_comparison"], indent=2), flush=True)
    return details


def finalize(
    data: dict[str, np.ndarray],
    surface_summary: dict[str, object],
    oracle_summary: dict[str, object],
    rows: list[dict[str, object]],
    selected_folds: list[int],
    args: argparse.Namespace,
) -> None:
    complete = len(selected_folds) == 5
    original = data["base"].copy()
    surface = data["base"].copy()
    for fold in selected_folds:
        with np.load(FOLD_DIR / f"fold_{fold}.npz") as loaded:
            ids = loaded["query_ids"].astype(np.int64)
            original[ids] = loaded["original_prediction"]
            surface[ids] = loaded["surface_prediction"]
    evaluated = data["point_valid"] & np.isin(data["folds"], selected_folds)
    baseline_metrics = metric_summary(data["truth"][evaluated], data["base"][evaluated])
    original_metrics = metric_summary(data["truth"][evaluated], original[evaluated])
    surface_metrics = metric_summary(data["truth"][evaluated], surface[evaluated])
    fold_surface_gain = np.asarray(
        [row["causal_comparison"]["surface_gain_vs_v543"] for row in rows],
        dtype=np.float64,
    )
    fold_original_margin = np.asarray(
        [row["causal_comparison"]["surface_margin_vs_original"] for row in rows],
        dtype=np.float64,
    )
    fold_shuffled_margin = np.asarray(
        [row["causal_comparison"]["surface_margin_vs_shuffled"] for row in rows],
        dtype=np.float64,
    )
    hard_region_success = bool(rows[0]["causal_comparison"]["hard_region_success"])
    expand_supported = bool(
        complete
        and float(surface_metrics["weighted"] - baseline_metrics["weighted"]) >= 0.005
        and int(np.sum(fold_surface_gain > 0.0)) >= 4
        and float(np.mean(fold_original_margin)) >= 0.003
        and float(np.mean(fold_shuffled_margin)) >= 0.003
    )
    np.savez_compressed(
        OOF_CACHE,
        evaluated=evaluated,
        folds=data["folds"],
        regimes=data["regimes"],
        original=original.astype(np.float32),
        surface=surface.astype(np.float32),
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Fixed Sionna delay-angle candidates are already sufficient; ordered official-"
            "pointcloud surface sequences provide a globally reusable law for path energy "
            "that transfers better than path identity or neighboring residual copies."
        ),
        "new_information": [
            "ordered v475 interaction type sequence aligned to each v522 selected path",
            "official point-cloud wall patch identity and continuous patch geometry",
            "reflection/refraction vertices, segment lengths, incidence and departure directions",
        ],
        "surface_alignment": surface_summary,
        "path_energy_oracle": oracle_summary,
        "split": {
            "type": "five contiguous XY regions with one nested contiguous tuning region",
            "evaluated_folds": selected_folds,
            "zero_power_points_excluded_from_scoring": int(
                np.sum(~data["point_valid"])
            ),
        },
        "model": {
            "base_features": "v526-equivalent v522 path and source-coefficient features",
            "surface_encoder": "two-layer ordered step Transformer plus global patch embeddings",
            "path_encoder": "three-layer set Transformer",
            "output": "one bounded scalar log-energy correction per fixed path",
            "candidate_positions_changed": False,
            "phase_changed": False,
            "alpha_grid_includes_exact_v543_fallback": True,
            "epochs": args.epochs,
            "pretrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size,
            "train_alpha": args.train_alpha,
        },
        "folds": rows,
        "summary": {
            "baseline": baseline_metrics,
            "original": original_metrics,
            "surface": surface_metrics,
            "surface_gain_vs_v543": {
                key: float(surface_metrics[key] - baseline_metrics[key])
                for key in surface_metrics
            },
            "surface_margin_vs_original": {
                key: float(surface_metrics[key] - original_metrics[key])
                for key in surface_metrics
            },
        },
        "decision": {
            "hard_region_success": hard_region_success,
            "expand_five_fold_supported": expand_supported,
            "hard_region_gate": (
                "surface gain vs v543 >= 0.005, margin vs original >= 0.003, "
                "and margin vs shuffled surface >= 0.003"
            ),
            "five_fold_gate": (
                "mean gain >= 0.005, >=4/5 positive regions, mean surface margins "
                "vs original and shuffled each >= 0.003"
            ),
            "submission_generated": False,
        },
        "outputs": {
            "surface_cache": PATHS.relative(SURFACE_CACHE),
            "path_energy_oracle": PATHS.relative(ORACLE_CACHE),
            "oof": PATHS.relative(OOF_CACHE),
            "fold_results": PATHS.relative(FOLD_DIR),
            "checkpoints": PATHS.relative(CHECKPOINT_DIR),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = json_safe(report)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(json.dumps(report["decision"], indent=2), flush=True)


def main() -> None:
    args = parse_args()
    require_inputs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_DATA) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    surface_summary = build_surface_cache(source, args.force_surface_cache)
    if args.build_only:
        return
    data, surface_summary, oracle_summary = load_data(args.oracle_ridge)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selected_folds = list(range(5)) if args.all_folds else [args.fold]
    rows = [
        run_fold(fold, data, surface_summary, args, device)
        for fold in selected_folds
    ]
    finalize(
        data,
        surface_summary,
        oracle_summary,
        rows,
        selected_folds,
        args,
    )


if __name__ == "__main__":
    main()
