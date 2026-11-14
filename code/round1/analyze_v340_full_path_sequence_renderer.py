#!/usr/bin/env python3
"""Add full ordered interaction sequences to the v313 official path renderer."""

from __future__ import annotations

import math

import numpy as np

import analyze_v313_highdepth_matched_path_renderer as v313
from project_paths import PATHS


VERSION_NUMBER = 340
VERSION = "v340_full_path_sequence_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS_OUT = OUT_DIR / "v340_joint_targets.npz"
OUTPUT = OUT_DIR / "v340_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"

MAX_DEPTH = 16
PATCH_HARMONICS = 8
POSITION_MODES = 4
PATCH_MAP_CHUNK = 20_000


def map_sequence_patches(
    vertices: np.ndarray,
    valid_step: np.ndarray,
    patches: np.ndarray,
) -> np.ndarray:
    flat_vertices = vertices.reshape(-1, 3)
    flat_valid = valid_step.reshape(-1)
    output = np.zeros(len(flat_vertices), dtype=np.int64)
    active = np.flatnonzero(flat_valid)
    for start in range(0, len(active), PATCH_MAP_CHUNK):
        ids = active[start : start + PATCH_MAP_CHUNK]
        output[ids] = v313.map_first_patch(
            flat_vertices[ids], np.ones(len(ids), dtype=bool), patches
        )
    return output.reshape(valid_step.shape)


def sequence_features(
    path_cache: dict[str, np.ndarray],
    prefix: str,
    patches: np.ndarray,
) -> np.ndarray:
    valid_path = (
        path_cache[f"{prefix}_valid"][:, : v313.PATH_LIMIT].astype(bool)
        & (path_cache[f"{prefix}_interaction_count"][:, : v313.PATH_LIMIT] >= 2)
    )
    depth = path_cache[f"{prefix}_interaction_count"][:, : v313.PATH_LIMIT].astype(np.int64)
    interaction = path_cache[f"{prefix}_interactions"][:, : v313.PATH_LIMIT].astype(np.int64)
    vertices = np.nan_to_num(
        path_cache[f"{prefix}_vertices"][:, : v313.PATH_LIMIT].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    step = np.arange(MAX_DEPTH, dtype=np.int64)
    valid_step = valid_path[:, :, None] & (step[None, None, :] < depth[:, :, None])
    patch_id = map_sequence_patches(vertices, valid_step, patches)
    ground_id = len(patches) + 1
    ground = valid_step & (vertices[..., 2] <= 0.4) & (interaction == 1)
    patch_id[ground] = ground_id
    patch_id[~valid_step] = 0

    patch_scale = float(len(patches) + 2)
    patch_angle = 2.0 * math.pi * patch_id.astype(np.float32) / patch_scale
    patch_encoding = []
    for order in range(1, PATCH_HARMONICS + 1):
        patch_encoding.append(np.sin(order * patch_angle))
        patch_encoding.append(np.cos(order * patch_angle))
    patch_encoding = np.stack(patch_encoding, axis=3).astype(np.float32)
    patch_encoding *= valid_step[..., None]

    position = (np.arange(MAX_DEPTH, dtype=np.float32) + 0.5) / MAX_DEPTH
    position_basis = np.stack(
        [np.cos(math.pi * mode * position) for mode in range(POSITION_MODES)],
        axis=0,
    ).astype(np.float32)
    ordered_patch = np.einsum(
        "npse,ms->npme", patch_encoding, position_basis, optimize=True
    ).reshape(len(valid_path), v313.PATH_LIMIT, -1)
    ordered_patch /= np.maximum(np.sqrt(depth[..., None].astype(np.float32)), 1.0)

    interaction_values = (0, 1, 2, 8)
    ordered_interaction = np.concatenate(
        [
            ((interaction == value) & valid_step).astype(np.float32)
            for value in interaction_values
        ],
        axis=2,
    )

    centered_vertices = vertices.copy()
    centered_vertices[..., 0] = (centered_vertices[..., 0] - 150.0) / 150.0
    centered_vertices[..., 1] = (centered_vertices[..., 1] + 50.0) / 180.0
    centered_vertices[..., 2] = centered_vertices[..., 2] / 40.0
    centered_vertices *= valid_step[..., None]
    vertex_modes = np.einsum(
        "npsd,ms->npmd", centered_vertices, position_basis, optimize=True
    ).reshape(len(valid_path), v313.PATH_LIMIT, -1)
    vertex_modes /= np.maximum(depth[..., None].astype(np.float32), 1.0)

    previous = np.zeros_like(vertices)
    previous[:, :, 0] = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
    previous[:, :, 1:] = vertices[:, :, :-1]
    segment_length = np.linalg.norm(vertices - previous, axis=3).astype(np.float32) / 250.0
    segment_length *= valid_step

    depth_onehot = (
        np.arange(1, MAX_DEPTH + 1, dtype=np.int64)[None, None, :] == depth[:, :, None]
    ).astype(np.float32)
    output = np.concatenate(
        [ordered_patch, ordered_interaction, vertex_modes, segment_length, depth_onehot],
        axis=2,
    ).astype(np.float32)
    output[~valid_path] = 0.0
    return np.nan_to_num(output, nan=0.0, posinf=4.0, neginf=-4.0)


_original_highdepth_geometry = v313.highdepth_geometry


def highdepth_geometry_with_sequence(
    path_cache: dict[str, np.ndarray],
    interaction_cache: dict[str, np.ndarray],
    prefix: str,
    pos: np.ndarray,
    patches: np.ndarray,
) -> dict[str, np.ndarray]:
    output = _original_highdepth_geometry(
        path_cache, interaction_cache, prefix, pos, patches
    )
    sequence = sequence_features(path_cache, prefix, patches)
    output["numeric"] = np.concatenate([output["numeric"], sequence], axis=2)
    return output


def main() -> None:
    v313.VERSION_NUMBER = VERSION_NUMBER
    v313.VERSION = VERSION
    v313.OUT_DIR = OUT_DIR
    v313.REPORT_DIR = REPORT_DIR
    v313.CHECKPOINT_DIR = CHECKPOINT_DIR
    v313.TARGETS_OUT = TARGETS_OUT
    v313.OUTPUT = OUTPUT
    v313.REPORT = REPORT
    v313.highdepth_geometry = highdepth_geometry_with_sequence
    v313.main()


if __name__ == "__main__":
    main()
