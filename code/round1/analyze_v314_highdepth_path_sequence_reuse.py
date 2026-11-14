#!/usr/bin/env python3
"""Audit cross-UE reuse of full high-depth interaction sequences."""

from __future__ import annotations

import json

import numpy as np

import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v312_highdepth_path_identity_match as v312
import analyze_v313_highdepth_matched_path_renderer as v313
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 314
VERSION = "v314_highdepth_path_sequence_reuse"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SEQUENCE_CACHE = OUT_DIR / "v314_matched_path_sequences.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

OUTER_SEEDS = (314, 1314, 2314, 3314, 4314)
OUTER_COUNT = 320
MAX_DEPTH = 16


def sequence_keys(
    interaction: np.ndarray, patch_id: np.ndarray, depth: np.ndarray
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, limit in (("first1", 1), ("first2", 2), ("first3", 3), ("full", MAX_DEPTH)):
        keys = []
        for row in range(len(depth)):
            take = min(int(depth[row]), limit)
            key = tuple(
                (int(interaction[row, step]), int(patch_id[row, step]))
                for step in range(take)
            )
            keys.append(repr(key))
        output[name] = np.asarray(keys)
    return output


def group_summary(key: np.ndarray, point_id: np.ndarray, mass: np.ndarray) -> dict[str, object]:
    unique, inverse = np.unique(key, return_inverse=True)
    occurrence = np.bincount(inverse)
    ue_count = np.asarray(
        [len(np.unique(point_id[inverse == group])) for group in range(len(unique))]
    )
    mass_coverage = {}
    total = float(np.sum(mass, dtype=np.float64))
    for threshold in (2, 3, 5, 10):
        reusable = ue_count[inverse] >= threshold
        mass_coverage[str(threshold)] = float(np.sum(mass[reusable]) / max(total, 1e-30))
    return {
        "unique_sequence_count": int(len(unique)),
        "occurrence": summarize(occurrence),
        "unique_ue": summarize(ue_count),
        "mass_coverage_by_min_unique_ue": mass_coverage,
    }


def fold_prediction(
    key: np.ndarray,
    point_id: np.ndarray,
    mass: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
) -> dict[str, float]:
    fit = np.isin(point_id, fit_ids)
    query = np.isin(point_id, query_ids)
    if not np.any(fit) or not np.any(query):
        return {"coverage": 0.0, "log_rmse": 0.0, "count": 0}
    fit_log = np.log(np.maximum(mass[fit], 1e-5))
    global_mean = float(np.mean(fit_log))
    means: dict[str, float] = {}
    for value in np.unique(key[fit]):
        means[str(value)] = float(np.mean(fit_log[key[fit] == value]))
    query_key = key[query]
    prediction = np.asarray([means.get(str(value), global_mean) for value in query_key])
    truth = np.log(np.maximum(mass[query], 1e-5))
    seen = np.asarray([str(value) in means for value in query_key])
    query_mass = mass[query]
    return {
        "coverage": float(np.sum(query_mass[seen]) / np.maximum(np.sum(query_mass), 1e-30)),
        "log_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "count": int(np.sum(query)),
    }


def main() -> None:
    match = np.load(v312.MATCH_CACHE)
    good = (
        (match["length_error_bins"] <= v312.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v312.GOOD_H_ERROR)
        & (match["candidate_id"] < v313.PATH_LIMIT)
    )
    point_id = match["point_id"][good].astype(np.int64)
    candidate_id = match["candidate_id"][good].astype(np.int64)
    mass = match["mass"][good].astype(np.float32)
    path_data = np.load(v312.PATH_CACHE)
    depth = path_data["train_interaction_count"][point_id, candidate_id].astype(np.int64)
    interaction = path_data["train_interactions"][point_id, candidate_id].astype(np.int64)
    vertices = np.nan_to_num(
        path_data["train_vertices"][point_id, candidate_id].astype(np.float32), nan=0.0
    )
    valid_step = np.arange(MAX_DEPTH)[None, :] < depth[:, None]
    scene = np.load(v306.SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    patch_id = v313.map_first_patch(vertices, valid_step, patches)
    ground_id = len(patches) + 1
    ground = valid_step & (vertices[:, :, 2] <= 0.4) & (interaction == 1)
    patch_id[ground] = ground_id
    keys = sequence_keys(interaction, patch_id, depth)
    np.savez_compressed(
        SEQUENCE_CACHE,
        point_id=point_id,
        candidate_id=candidate_id,
        mass=mass,
        depth=depth,
        interaction=interaction,
        patch_id=patch_id,
        **{f"key_{name}": value for name, value in keys.items()},
    )

    target = np.load(v273.TARGET_CACHE)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    summaries = {name: group_summary(key, point_id, mass) for name, key in keys.items()}
    fold_rows = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        level = {
            name: fold_prediction(key, point_id, mass, fit_ids, outer_ids)
            for name, key in keys.items()
        }
        fold_rows.append({"fold": fold, "seed": seed, "levels": level})
    pooled = {}
    for name in keys:
        pooled[name] = {
            "coverage": float(np.mean([row["levels"][name]["coverage"] for row in fold_rows])),
            "log_rmse": float(np.mean([row["levels"][name]["log_rmse"] for row in fold_rows])),
        }
    first1_rmse = pooled["first1"]["log_rmse"]
    passed = (
        pooled["first2"]["coverage"] >= 0.60
        and pooled["first3"]["coverage"] >= 0.40
        and pooled["first2"]["log_rmse"] <= 0.97 * first1_rmse
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The ordered interaction/official-wall sequence supplies reusable multi-hop "
            "path identity beyond v313's first-wall embedding."
        ),
        "matched_path_count": int(len(point_id)),
        "matched_mass": float(np.sum(mass, dtype=np.float64)),
        "ground_patch_id": ground_id,
        "sequence_summaries": summaries,
        "fold_rows": fold_rows,
        "pooled_spatial_oof": pooled,
        "decision_rule": {
            "first2_coverage": ">=0.60",
            "first3_coverage": ">=0.40",
            "first2_log_rmse_vs_first1": "at least 3% lower",
            "passed": bool(passed),
        },
        "outputs": {
            "sequences": PATHS.relative(SEQUENCE_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
