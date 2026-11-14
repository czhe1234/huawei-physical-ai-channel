#!/usr/bin/env python3
"""v271: full-patch two-bounce candidate construction and PDP oracle audit."""

from __future__ import annotations

import json

import numpy as np

import analyze_v254_route_topology_spectral_field as v254
import analyze_v259_v251_path_atom_oracle as v259
import analyze_v264_patch_scatterer_transformer as v264
from generate_v87_easy_field_projection import S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 271
VERSION = "v271_fullpatch_two_bounce_oracle"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
TWO_BOUNCE_CACHE = OUT_DIR / "v271_two_bounce_candidates.npz"

TRAIN_ONE = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
TEST_ONE = PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
GLOBAL_PATCHES = 64
LOCAL_PATCHES = 32
TOP_TWO = 32
VAL_COUNT = 320
VAL_SEED = 254


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def mirror_xy(point: np.ndarray, normal: np.ndarray, c: np.ndarray) -> np.ndarray:
    signed = np.sum(point * normal, axis=1) - c
    return point - 2.0 * signed[:, None] * normal


def inside_patch(
    point: np.ndarray,
    patch_ids: np.ndarray,
    patches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normal = patches[patch_ids, :2].astype(np.float64)
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    tangential = np.sum(point[:, :2] * tangent, axis=1)
    t0 = patches[patch_ids, 3].astype(np.float64)
    t1 = patches[patch_ids, 4].astype(np.float64)
    z0 = patches[patch_ids, 5].astype(np.float64)
    z1 = patches[patch_ids, 6].astype(np.float64)
    inside = (
        (tangential >= t0 - 0.75)
        & (tangential <= t1 + 0.75)
        & (point[:, 2] >= z0 - 0.75)
        & (point[:, 2] <= z1 + 0.75)
    )
    width = np.maximum(t1 - t0, 1e-3)
    height = np.maximum(z1 - z0, 1e-3)
    t_margin = np.minimum(tangential - t0, t1 - tangential) / width
    z_margin = np.minimum(point[:, 2] - z0, z1 - point[:, 2]) / height
    return inside, np.minimum(t_margin, z_margin)


def point_two_bounce(
    ue: np.ndarray,
    local_patch_ids: np.ndarray,
    patches: np.ndarray,
) -> dict[str, np.ndarray]:
    ids = np.unique(
        np.concatenate(
            [
                np.arange(min(GLOBAL_PATCHES, len(patches)), dtype=np.int64),
                local_patch_ids[local_patch_ids >= 0].astype(np.int64),
            ]
        )
    )
    p1 = np.repeat(ids, len(ids))
    p2 = np.tile(ids, len(ids))
    keep = p1 != p2
    p1 = p1[keep]
    p2 = p2[keep]
    n1 = patches[p1, :2].astype(np.float64)
    n2 = patches[p2, :2].astype(np.float64)
    c1 = patches[p1, 2].astype(np.float64)
    c2 = patches[p2, 2].astype(np.float64)
    same_surface = (
        np.abs(np.sum(n1 * n2, axis=1)) > 0.995
    ) & (np.abs(c1 - c2) < 1.5)

    bs_xy = np.repeat(BS[None, :2], len(p1), axis=0)
    img1_xy = mirror_xy(bs_xy, n1, c1)
    img12_xy = mirror_xy(img1_xy, n2, c2)
    img1 = np.concatenate([img1_xy, np.full((len(p1), 1), BS[2])], axis=1)
    img12 = np.concatenate([img12_xy, np.full((len(p1), 1), BS[2])], axis=1)

    ray2 = ue[None, :] - img12
    denom2 = np.sum(ray2[:, :2] * n2, axis=1)
    numer2 = c2 - np.sum(img12[:, :2] * n2, axis=1)
    tau2 = numer2 / np.where(np.abs(denom2) < 1e-10, np.nan, denom2)
    r2 = img12 + tau2[:, None] * ray2
    inside2, margin2 = inside_patch(r2, p2, patches)

    ray1 = r2 - img1
    denom1 = np.sum(ray1[:, :2] * n1, axis=1)
    numer1 = c1 - np.sum(img1[:, :2] * n1, axis=1)
    tau1 = numer1 / np.where(np.abs(denom1) < 1e-10, np.nan, denom1)
    r1 = img1 + tau1[:, None] * ray1
    inside1, margin1 = inside_patch(r1, p1, patches)
    valid = (
        ~same_surface
        & np.isfinite(tau1)
        & np.isfinite(tau2)
        & (tau1 > 1e-4)
        & (tau1 < 1.0 - 1e-4)
        & (tau2 > 1e-4)
        & (tau2 < 1.0 - 1e-4)
        & inside1
        & inside2
    )

    d0 = np.linalg.norm(r1 - BS[None, :], axis=1)
    d1 = np.linalg.norm(r2 - r1, axis=1)
    d2 = np.linalg.norm(ue[None, :] - r2, axis=1)
    path_length = d0 + d1 + d2
    direct = float(np.linalg.norm(ue - BS))
    excess = path_length - direct
    q1 = np.log1p(np.maximum(patches[p1, 7], 0.0)) / (1.0 + patches[p1, 9])
    q2 = np.log1p(np.maximum(patches[p2, 7], 0.0)) / (1.0 + patches[p2, 9])
    quality = (q1 + q2) / max(float(np.max(np.log1p(patches[:, 7]))), 1e-6)
    margin = np.minimum(margin1, margin2)
    score = (
        3.0 * valid.astype(np.float64)
        + 0.18 * quality
        + 0.20 * np.clip(margin, -1.0, 1.0)
        - 0.009 * np.maximum(excess, 0.0)
    )
    score = np.where(valid, score, -np.inf)
    finite = np.flatnonzero(np.isfinite(score))
    if len(finite) == 0:
        return {
            "patch1": np.empty(0, dtype=np.int32),
            "patch2": np.empty(0, dtype=np.int32),
            "score": np.empty(0, dtype=np.float32),
            "path_length": np.empty(0, dtype=np.float32),
            "excess_length": np.empty(0, dtype=np.float32),
            "reflection1": np.empty((0, 3), dtype=np.float32),
            "reflection2": np.empty((0, 3), dtype=np.float32),
        }
    order = finite[np.argsort(score[finite])[::-1][:TOP_TWO]]
    return {
        "patch1": p1[order].astype(np.int32),
        "patch2": p2[order].astype(np.int32),
        "score": score[order].astype(np.float32),
        "path_length": path_length[order].astype(np.float32),
        "excess_length": excess[order].astype(np.float32),
        "reflection1": r1[order].astype(np.float32),
        "reflection2": r2[order].astype(np.float32),
    }


def build_split(
    positions: np.ndarray,
    one_candidates: dict[str, np.ndarray],
    patches: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    count = len(positions)
    out = {
        "valid": np.zeros((count, TOP_TWO), dtype=bool),
        "patch1": np.full((count, TOP_TWO), -1, dtype=np.int32),
        "patch2": np.full((count, TOP_TWO), -1, dtype=np.int32),
        "score": np.full((count, TOP_TWO), -np.inf, dtype=np.float32),
        "path_length": np.zeros((count, TOP_TWO), dtype=np.float32),
        "excess_length": np.zeros((count, TOP_TWO), dtype=np.float32),
        "reflection1": np.zeros((count, TOP_TWO, 3), dtype=np.float32),
        "reflection2": np.zeros((count, TOP_TWO, 3), dtype=np.float32),
    }
    for index, ue in enumerate(positions):
        local_ids = one_candidates["patch_id"][index, :LOCAL_PATCHES]
        result = point_two_bounce(ue.astype(np.float64), local_ids, patches)
        n = len(result["score"])
        if n:
            out["valid"][index, :n] = True
            for key in out:
                if key == "valid":
                    continue
                out[key][index, :n] = result[key]
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(
                f"v271 {prefix} {index + 1}/{count} mean_paths="
                f"{float(np.mean(out['valid'][: index + 1].sum(axis=1))):.2f}",
                flush=True,
            )
    return out


def solve_variant(
    val_ids: np.ndarray,
    target_d: np.ndarray,
    positions: np.ndarray,
    one: dict[str, np.ndarray],
    two: dict[str, np.ndarray],
    use_one: bool,
    use_two: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reconstruction = np.zeros((len(val_ids), S_SC), dtype=np.float32)
    used = np.zeros(len(val_ids), dtype=np.int32)
    direct_delay = (
        np.linalg.norm(positions - BS[None, :], axis=1)
        / v264.LIGHT_SPEED
        * S_SC
        * v264.SUBCARRIER_SPACING
    )
    one_delay = (
        one["path_length"][:, :LOCAL_PATCHES]
        / v264.LIGHT_SPEED
        * S_SC
        * v264.SUBCARRIER_SPACING
    )
    two_delay = (
        two["path_length"] / v264.LIGHT_SPEED * S_SC * v264.SUBCARRIER_SPACING
    )
    for row, point_id in enumerate(val_ids):
        delay = [float(direct_delay[point_id])]
        interaction = [0]
        if use_one:
            mask = one["score"][point_id, :LOCAL_PATCHES] > 0.0
            delay.extend(one_delay[point_id, mask].tolist())
            interaction.extend([1] * int(np.sum(mask)))
        if use_two:
            mask = two["valid"][point_id]
            delay.extend(two_delay[point_id, mask].tolist())
            interaction.extend([2] * int(np.sum(mask)))
        atoms = v259.point_atoms(
            np.asarray(delay, dtype=np.float32), np.asarray(interaction, dtype=np.int64)
        )
        reconstruction[row], _mass = v259.solve(target_d[point_id], atoms)
        used[row] = len(delay)
    return reconstruction, used


def main() -> None:
    for path in (TRAIN_ONE, TEST_ONE, SCENE_CACHE, TARGET_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    patches = np.load(SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_one = dict(np.load(TRAIN_ONE))
    test_one = dict(np.load(TEST_ONE))
    if TWO_BOUNCE_CACHE.is_file():
        cache = np.load(TWO_BOUNCE_CACHE)
        train_two = {key.removeprefix("train_"): cache[key] for key in cache.files if key.startswith("train_")}
        test_two = {key.removeprefix("test_"): cache[key] for key in cache.files if key.startswith("test_")}
    else:
        train_two = build_split(train_pos, train_one, patches, "train")
        test_two = build_split(test_pos, test_one, patches, "test")
        np.savez_compressed(
            TWO_BOUNCE_CACHE,
            **{f"train_{key}": value for key, value in train_two.items()},
            **{f"test_{key}": value for key, value in test_two.items()},
        )

    targets = np.load(TARGET_CACHE)
    target_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    matched = v254.choose_seeded_matched_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    results = {}
    for name, use_one, use_two in (
        ("one_only", True, False),
        ("two_only", False, True),
        ("one_plus_two", True, True),
    ):
        reconstruction, used = solve_variant(
            val_ids, target_d, train_pos, train_one, train_two, use_one, use_two
        )
        results[name] = {
            "cosine": summarize(row_cos(reconstruction, target_d[val_ids])),
            "used_paths": summarize(used),
        }

    valid_two_train = train_two["valid"]
    valid_two_test = test_two["valid"]
    pair_ids = np.stack(
        [train_two["patch1"][valid_two_train], train_two["patch2"][valid_two_train]], axis=1
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Analytic ordered wall-patch pairs recover multibounce delays missing from the "
            "one-bounce scatterer set and raise the official PDP representation oracle."
        ),
        "compliance_boundary": {
            "uses": [
                "official point-cloud wall patches",
                "official train/test coordinates",
                "analytic two-reflection image geometry",
                "official train H PDP for analysis-only NNLS oracle",
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
            "global_patches": GLOBAL_PATCHES,
            "local_patches": LOCAL_PATCHES,
            "top_two": TOP_TWO,
            "patch_count": int(len(patches)),
        },
        "candidate_diagnostics": {
            "train_two_count": summarize(valid_two_train.sum(axis=1)),
            "test_two_count": summarize(valid_two_test.sum(axis=1)),
            "train_unique_pairs": int(len(np.unique(pair_ids, axis=0))) if len(pair_ids) else 0,
            "train_excess": summarize(train_two["excess_length"][valid_two_train]),
        },
        "pdp_oracle": results,
        "outputs": {
            "candidates": PATHS.relative(TWO_BOUNCE_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
