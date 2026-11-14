#!/usr/bin/env python3
"""Evaluate the official-H PDP atom oracle of v272 geometry-only paths."""

from __future__ import annotations

import json

import numpy as np

import analyze_v254_route_topology_spectral_field as v254
import analyze_v259_v251_path_atom_oracle as v259
from generate_v87_easy_field_projection import S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 272
VERSION = "v272_extruded_pointcloud_scene"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / "v272_extruded_mesh_path_oracle.json"
PATH_CACHE = OUT_DIR / "v272_extruded_sionna_paths.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
V263_REPORT = PATHS.version_report_dir(263) / "v263_full_wall_scatterer_oracle.json"
BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
VAL_COUNT = 320
VAL_SEED = 254


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def solve_limit(
    val_ids: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
    cache: np.lib.npyio.NpzFile,
    limit: int,
    force_direct: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reconstruction = np.zeros((len(val_ids), S_SC), dtype=np.float32)
    used = np.zeros(len(val_ids), dtype=np.int32)
    valid = cache["train_valid"]
    delay = cache["train_delay_bin_physical"]
    interaction = cache["train_interaction_count"]
    direct_delay = np.linalg.norm(positions - BS[None, :], axis=1) / 299_792_458.0 * S_SC * 30_000.0
    for row, point_id in enumerate(val_ids):
        ids = np.flatnonzero(valid[point_id])[:limit]
        values = delay[point_id, ids].astype(np.float32).tolist()
        types = interaction[point_id, ids].astype(np.int64).tolist()
        if force_direct and not np.any(np.asarray(types) == 0):
            values.insert(0, float(direct_delay[point_id]))
            types.insert(0, 0)
        if len(values) == 0:
            values = [float(direct_delay[point_id])]
            types = [0]
        atoms = v259.point_atoms(np.asarray(values, dtype=np.float32), np.asarray(types, dtype=np.int64))
        reconstruction[row], _mass = v259.solve(target[point_id], atoms)
        used[row] = len(values)
    return reconstruction, used


def main() -> None:
    for path in (PATH_CACHE, TARGET_CACHE, V263_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = np.load(PATH_CACHE)
    targets = np.load(TARGET_CACHE)
    target_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    matched = v254.choose_seeded_matched_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    results = {}
    for force_direct in (False, True):
        for limit in (16, 32, 64, 128):
            reconstruction, used = solve_limit(
                val_ids, target_d, train_pos, cache, limit, force_direct
            )
            results[f"{'forced_direct' if force_direct else 'solver_only'}_top{limit}"] = {
                "cosine": summarize(row_cos(reconstruction, target_d[val_ids])),
                "used_paths": summarize(used),
            }
    with V263_REPORT.open("r", encoding="utf-8") as handle:
        v263 = json.load(handle)
    train_count = cache["train_valid"].sum(axis=1)
    test_count = cache["test_valid"].sum(axis=1)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A closed extruded mesh reconstructed from official point-cloud components yields "
            "more complete and useful multibounce path geometry than disjoint wall patches."
        ),
        "compliance_boundary": {
            "uses": [
                "official point-cloud component/height grid",
                "official train/test coordinates",
                "Sionna PathSolver geometry and visibility only",
                "official train H PDP for analysis-only NNLS oracle",
            ],
            "does_not_use": [
                "Paths.cfr or RT MIMO H",
                "path coefficients",
                "restored Etoile mesh/material labels",
                "synthetic channel labels",
                "teacher/local truth/hidden labels",
            ],
        },
        "coverage": {
            "train_count": summarize(train_count),
            "test_count": summarize(test_count),
            "train_zero": int(np.sum(train_count == 0)),
            "test_zero": int(np.sum(test_count == 0)),
        },
        "pdp_oracle": results,
        "v263_top32_reference": v263["results"]["top_32"]["cosine"],
        "outputs": {
            "paths": PATHS.relative(PATH_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
