#!/usr/bin/env python3
"""v270: segment-level visibility audit for official wall reflections."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

import analyze_v254_route_topology_spectral_field as v254
import analyze_v259_v251_path_atom_oracle as v259
import analyze_v264_patch_scatterer_transformer as v264
from generate_v87_easy_field_projection import S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 270
VERSION = "v270_reflection_visibility_audit"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
VISIBILITY_CACHE = OUT_DIR / "v270_reflection_visibility.npz"

TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
TEST_CANDIDATES = PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
NNLS_CACHE = PATHS.cache_dir(264) / "v264_train_scatterer_nnls.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
REFLECTIONS = 32
VAL_COUNT = 320
VAL_SEED = 254
INTERSECTION_EPS = 0.015
PATCH_TOLERANCE = 0.45
BATCH = 256


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def segment_occlusion(
    starts: np.ndarray,
    ends: np.ndarray,
    own_patch: np.ndarray,
    patches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = patches[:, :2].astype(np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    tangent = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    c = patches[:, 2].astype(np.float64)
    t_min = patches[:, 3].astype(np.float64) - PATCH_TOLERANCE
    t_max = patches[:, 4].astype(np.float64) + PATCH_TOLERANCE
    z_min = patches[:, 5].astype(np.float64) - PATCH_TOLERANCE
    z_max = patches[:, 6].astype(np.float64) + PATCH_TOLERANCE
    count = np.zeros(len(starts), dtype=np.int16)
    nearest = np.ones(len(starts), dtype=np.float32)
    strength = np.zeros(len(starts), dtype=np.float32)
    patch_strength = (
        np.log1p(np.maximum(patches[:, 7], 0.0))
        / np.maximum(1.0 + patches[:, 9], 1e-3)
    ).astype(np.float64)
    patch_strength /= max(float(np.max(patch_strength)), 1e-12)
    for start in range(0, len(starts), BATCH):
        end = min(start + BATCH, len(starts))
        a = starts[start:end].astype(np.float64)
        direction = ends[start:end].astype(np.float64) - a
        denom = direction[:, :2] @ normals.T
        numer = c[None, :] - a[:, :2] @ normals.T
        tau = numer / np.where(np.abs(denom) < 1e-10, np.nan, denom)
        x = a[:, 0, None] + tau * direction[:, 0, None]
        y = a[:, 1, None] + tau * direction[:, 1, None]
        z = a[:, 2, None] + tau * direction[:, 2, None]
        tangential = x * tangent[None, :, 0] + y * tangent[None, :, 1]
        hit = (
            (tau > INTERSECTION_EPS)
            & (tau < 1.0 - INTERSECTION_EPS)
            & (tangential >= t_min[None, :])
            & (tangential <= t_max[None, :])
            & (z >= z_min[None, :])
            & (z <= z_max[None, :])
            & np.isfinite(tau)
        )
        own = own_patch[start:end]
        local = np.arange(end - start)
        valid_own = (own >= 0) & (own < len(patches))
        hit[local[valid_own], own[valid_own]] = False
        count[start:end] = hit.sum(axis=1).astype(np.int16)
        nearest[start:end] = np.min(np.where(hit, tau, 1.0), axis=1).astype(np.float32)
        strength[start:end] = np.max(
            np.where(hit, patch_strength[None, :], 0.0), axis=1
        ).astype(np.float32)
    return count, nearest, strength


def split_visibility(
    positions: np.ndarray,
    candidates: dict[str, np.ndarray],
    patches: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    patch_id = candidates["patch_id"][:, :REFLECTIONS].astype(np.int64)
    reflection = candidates["reflection_point"][:, :REFLECTIONS].astype(np.float64)
    valid = candidates["score"][:, :REFLECTIONS] > 0.0
    count = len(positions)
    flat_reflection = reflection.reshape(-1, 3)
    flat_patch = patch_id.reshape(-1)
    flat_valid = valid.reshape(-1)
    bs_start = np.repeat(BS[None, :], count * REFLECTIONS, axis=0)
    ue_end = np.repeat(positions[:, None, :], REFLECTIONS, axis=1).reshape(-1, 3)
    first_count = np.zeros(count * REFLECTIONS, dtype=np.int16)
    first_nearest = np.ones(count * REFLECTIONS, dtype=np.float32)
    first_strength = np.zeros(count * REFLECTIONS, dtype=np.float32)
    second_count = np.zeros_like(first_count)
    second_nearest = np.ones_like(first_nearest)
    second_strength = np.zeros_like(first_strength)
    active = np.flatnonzero(flat_valid)
    first = segment_occlusion(
        bs_start[active], flat_reflection[active], flat_patch[active], patches
    )
    second = segment_occlusion(
        flat_reflection[active], ue_end[active], flat_patch[active], patches
    )
    first_count[active], first_nearest[active], first_strength[active] = first
    second_count[active], second_nearest[active], second_strength[active] = second
    los_count, los_nearest, los_strength = segment_occlusion(
        np.repeat(BS[None, :], count, axis=0),
        positions,
        np.full(count, -1, dtype=np.int64),
        patches,
    )
    print(
        f"v270 {prefix} valid={int(np.sum(valid))} visible="
        f"{int(np.sum(valid & ((first_count + second_count).reshape(count, REFLECTIONS) == 0)))}",
        flush=True,
    )
    return {
        "first_count": first_count.reshape(count, REFLECTIONS),
        "second_count": second_count.reshape(count, REFLECTIONS),
        "first_nearest": first_nearest.reshape(count, REFLECTIONS),
        "second_nearest": second_nearest.reshape(count, REFLECTIONS),
        "first_strength": first_strength.reshape(count, REFLECTIONS),
        "second_strength": second_strength.reshape(count, REFLECTIONS),
        "los_count": los_count,
        "los_nearest": los_nearest,
        "los_strength": los_strength,
    }


def filtered_oracle(
    val_ids: np.ndarray,
    target_d: np.ndarray,
    positions: np.ndarray,
    candidates: dict[str, np.ndarray],
    blocker_count: np.ndarray,
    max_blockers: int,
) -> tuple[np.ndarray, np.ndarray]:
    reconstruction = np.zeros((len(val_ids), S_SC), dtype=np.float32)
    used = np.zeros(len(val_ids), dtype=np.int32)
    direct = np.linalg.norm(positions - BS[None, :], axis=1)
    direct_delay = direct / v264.LIGHT_SPEED * S_SC * v264.SUBCARRIER_SPACING
    reflection_delay = (
        candidates["path_length"][:, :REFLECTIONS]
        / v264.LIGHT_SPEED
        * S_SC
        * v264.SUBCARRIER_SPACING
    )
    for row, point_id in enumerate(val_ids):
        visible = (
            (candidates["score"][point_id, :REFLECTIONS] > 0.0)
            & (blocker_count[point_id] <= max_blockers)
        )
        delay = np.concatenate(
            [
                np.asarray([direct_delay[point_id]], dtype=np.float32),
                reflection_delay[point_id, visible].astype(np.float32),
            ]
        )
        interaction = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.ones(int(np.sum(visible)), dtype=np.int64)]
        )
        atoms = v259.point_atoms(delay, interaction)
        reconstruction[row], _path_mass = v259.solve(target_d[point_id], atoms)
        used[row] = len(delay)
    return reconstruction, used


def main() -> None:
    for path in (TRAIN_CANDIDATES, TEST_CANDIDATES, SCENE_CACHE, NNLS_CACHE, TARGET_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    patches = np.load(SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(TRAIN_CANDIDATES))
    test_candidates = dict(np.load(TEST_CANDIDATES))
    if VISIBILITY_CACHE.is_file():
        visibility = dict(np.load(VISIBILITY_CACHE))
    else:
        train_visibility = split_visibility(train_pos, train_candidates, patches, "train")
        test_visibility = split_visibility(test_pos, test_candidates, patches, "test")
        visibility = {
            **{f"train_{key}": value for key, value in train_visibility.items()},
            **{f"test_{key}": value for key, value in test_visibility.items()},
        }
        np.savez_compressed(VISIBILITY_CACHE, **visibility)
    train_blocker_count = visibility["train_first_count"] + visibility["train_second_count"]
    test_blocker_count = visibility["test_first_count"] + visibility["test_second_count"]
    train_valid_path = train_candidates["score"][:, :REFLECTIONS] > 0.0
    test_valid_path = test_candidates["score"][:, :REFLECTIONS] > 0.0

    nnls_weights = np.load(NNLS_CACHE)["weights"].astype(np.float32)
    path_mass = nnls_weights[:, : v264.PATHS_PER_POINT * len(v264.SHIFTS)].reshape(
        len(nnls_weights), v264.PATHS_PER_POINT, len(v264.SHIFTS)
    ).sum(axis=2)[:, 1:]
    active = path_mass[train_valid_path] > 1e-3
    blocker_flat = train_blocker_count[train_valid_path]
    strength_flat = np.maximum(
        visibility["train_first_strength"], visibility["train_second_strength"]
    )[train_valid_path]
    auc_count = float(roc_auc_score(active, -blocker_flat)) if len(np.unique(active)) == 2 else None
    auc_strength = float(roc_auc_score(active, -strength_flat)) if len(np.unique(active)) == 2 else None

    targets = np.load(TARGET_CACHE)
    target_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    matched = v254.choose_seeded_matched_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    oracle_results = {}
    for max_blockers in (0, 1, 2, 99):
        reconstruction, used = filtered_oracle(
            val_ids,
            target_d,
            train_pos,
            train_candidates,
            train_blocker_count,
            max_blockers,
        )
        oracle_results[f"max_blockers_{max_blockers}"] = {
            "cosine": summarize(row_cos(reconstruction, target_d[val_ids])),
            "used_paths": summarize(used),
        }

    visible_train = train_valid_path & (train_blocker_count == 0)
    visible_test = test_valid_path & (test_blocker_count == 0)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Explicit BS-to-reflector and reflector-to-UE occlusion removes false one-bounce "
            "candidates that NNLS can ignore but unseen-point models cannot identify reliably."
        ),
        "compliance_boundary": {
            "uses": [
                "official point-cloud wall patches",
                "official train/test coordinates",
                "analytic reflection points and segment-patch intersections",
                "official train H NNLS weights and PDP for analysis-only purity/oracle checks",
            ],
            "does_not_use": [
                "Sionna or synthetic labels",
                "teacher output",
                "local replay truth",
                "hidden labels",
                "raw RT H",
            ],
        },
        "parameters": {
            "patch_count": int(len(patches)),
            "reflection_count": REFLECTIONS,
            "intersection_eps": INTERSECTION_EPS,
            "patch_tolerance": PATCH_TOLERANCE,
        },
        "visibility": {
            "train_valid_paths": int(np.sum(train_valid_path)),
            "train_visible_paths": int(np.sum(visible_train)),
            "train_visible_fraction": float(np.sum(visible_train) / np.maximum(np.sum(train_valid_path), 1)),
            "test_valid_paths": int(np.sum(test_valid_path)),
            "test_visible_paths": int(np.sum(visible_test)),
            "test_visible_fraction": float(np.sum(visible_test) / np.maximum(np.sum(test_valid_path), 1)),
            "train_blocker_count": summarize(train_blocker_count[train_valid_path]),
            "test_blocker_count": summarize(test_blocker_count[test_valid_path]),
            "train_los_blocker_count": summarize(visibility["train_los_count"]),
            "test_los_blocker_count": summarize(visibility["test_los_count"]),
        },
        "nnls_purity": {
            "active_threshold": 1e-3,
            "active_rate_visible": float(np.mean(path_mass[visible_train] > 1e-3)),
            "active_rate_blocked": float(np.mean(path_mass[train_valid_path & ~visible_train] > 1e-3)),
            "mean_mass_visible": float(np.mean(path_mass[visible_train])),
            "mean_mass_blocked": float(np.mean(path_mass[train_valid_path & ~visible_train])),
            "auc_negative_blocker_count": auc_count,
            "auc_negative_blocker_strength": auc_strength,
        },
        "pdp_oracle": oracle_results,
        "outputs": {
            "visibility": PATHS.relative(VISIBILITY_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
