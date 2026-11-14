#!/usr/bin/env python3
"""v263: full official wall-patch specular scatterer NNLS oracle audit."""

from __future__ import annotations

import json

import numpy as np

import analyze_v254_route_topology_spectral_field as v254
import analyze_v259_v251_path_atom_oracle as v259
import build_official_pointcloud_scene as pointcloud_scene
from generate_v87_easy_field_projection import S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 263
VERSION = "v263_full_wall_scatterer_oracle"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
CANDIDATES = OUT_DIR / "v263_full_wall_reflection_candidates.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
V239_NNLS = PATHS.cache_dir(239) / "v239_train_nnls_path_labels.npz"

VAL_COUNT = 320
VAL_SEED = 254
PATH_LIMITS = (8, 16, 32, 64, 128)
SUBCARRIER_SPACING = 30000.0
LIGHT_SPEED = 299792458.0


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def main() -> None:
    for path in (TARGET_CACHE, SCENE_CACHE, V239_NNLS):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos_3d = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    train_pos = train_pos_3d[:, :2]
    scene = np.load(SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    if CANDIDATES.is_file():
        candidate = np.load(CANDIDATES)
        patch_id = candidate["patch_id"]
        score = candidate["score"]
        path_length = candidate["path_length"]
    else:
        built = pointcloud_scene.build_reflection_candidates(
            train_pos_3d, patches, top_k=max(PATH_LIMITS), batch=64
        )
        patch_id = built["patch_id"]
        score = built["score"]
        path_length = built["path_length"]
        np.savez_compressed(CANDIDATES, **built)

    target_data = np.load(TARGET_CACHE)
    target_d = target_data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    matched = v254.choose_seeded_matched_indices(
        train_pos, test_pos, VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    v239_reconstruction = np.load(V239_NNLS)["reconstruction"].astype(np.float32)
    direct_length = np.linalg.norm(
        train_pos_3d - np.asarray([50.0, 0.0, 25.0], dtype=np.float64)[None, :],
        axis=1,
    )
    direct_delay = direct_length / LIGHT_SPEED * S_SC * SUBCARRIER_SPACING
    reflection_delay = path_length / LIGHT_SPEED * S_SC * SUBCARRIER_SPACING

    results: dict[str, object] = {
        "v239_same_points": {
            "cosine": summarize(row_cos(v239_reconstruction[val_ids], target_d[val_ids]))
        }
    }
    for limit in PATH_LIMITS:
        reconstruction = np.zeros((len(val_ids), S_SC), dtype=np.float32)
        path_mass = np.zeros(len(val_ids), dtype=np.float32)
        used_paths = np.zeros(len(val_ids), dtype=np.int32)
        unique_patches: set[int] = set()
        for row, point_id in enumerate(val_ids):
            valid_reflection = np.flatnonzero(score[point_id, :limit] > 0.0)
            delays = [float(direct_delay[point_id])]
            interactions = [0]
            if len(valid_reflection):
                delays.extend(reflection_delay[point_id, valid_reflection].tolist())
                interactions.extend([1] * len(valid_reflection))
                unique_patches.update(
                    int(value) for value in patch_id[point_id, valid_reflection]
                )
            atoms = v259.point_atoms(
                np.asarray(delays, dtype=np.float32),
                np.asarray(interactions, dtype=np.int64),
            )
            reconstruction[row], path_mass[row] = v259.solve(target_d[point_id], atoms)
            used_paths[row] = len(delays)
        cosine = row_cos(reconstruction, target_d[val_ids])
        results[f"top_{limit}"] = {
            "cosine": summarize(cosine),
            "path_mass": summarize(path_mass),
            "used_paths": summarize(used_paths),
            "unique_patch_count": int(len(unique_patches)),
        }
        print(f"v263 top {limit}: mean cosine {float(np.mean(cosine)):.9f}", flush=True)

    valid_score = score > 0.0
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "All stable official point-cloud wall patches provide missing persistent scatterer "
            "identities and a more expressive specular delay set than the top8/15 path cache."
        ),
        "compliance_boundary": {
            "uses": [
                "official point-cloud wall patches",
                "analytic image-method reflection geometry",
                "official train H validation PDP for an analysis-only NNLS oracle",
            ],
            "does_not_use": [
                "Sionna or synthetic channel labels",
                "teacher output",
                "local replay truth",
                "hidden labels",
                "raw RT H",
            ],
        },
        "parameters": {
            "wall_patch_count": int(len(patches)),
            "path_limits": list(PATH_LIMITS),
            "validation_seed": VAL_SEED,
            "validation_count": int(len(val_ids)),
            "subcarrier_spacing": SUBCARRIER_SPACING,
        },
        "candidate_diagnostics": {
            "valid_reflections_per_point": summarize(valid_score.sum(axis=1)),
            "positive_score": summarize(score[valid_score]),
            "delay_bin": summarize(reflection_delay[valid_score]),
        },
        "results": results,
        "local_gate": None,
        "outputs": {
            "candidates": PATHS.relative(CANDIDATES),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
