#!/usr/bin/env python3
"""PDP atom oracle for high-depth official-point-cloud geometry paths."""

from __future__ import annotations

import json

import numpy as np

import analyze_v254_route_topology_spectral_field as v254
import analyze_v259_v251_path_atom_oracle as v259
from generate_v87_easy_field_projection import S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 291
VERSION = "v291_highdepth_pointcloud_path_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
LOW_DEPTH_CACHE = PATHS.cache_dir(251) / "v251_pointcloud_sionna_paths.npz"
ANALYTIC_CACHE = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"

VAL_COUNT = 320
VAL_SEED = 254
LIMITS = (32, 64, 128)
LIGHT_SPEED = 299_792_458.0
SUBCARRIER_SPACING = 30_000.0


def analytic_atoms(
    point_id: int,
    train_pos: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> np.ndarray:
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    direct_length = float(np.linalg.norm(train_pos[point_id] - bs))
    valid = np.flatnonzero(candidates["score"][point_id, :32] > 0.0)
    lengths = [direct_length]
    interactions = [0]
    if len(valid):
        lengths.extend(candidates["path_length"][point_id, valid].tolist())
        interactions.extend([1] * len(valid))
    delay = (
        np.asarray(lengths, dtype=np.float32)
        / LIGHT_SPEED
        * S_SC
        * SUBCARRIER_SPACING
    )
    return v259.point_atoms(
        delay, np.asarray(interactions, dtype=np.int64)
    )


def evaluate(
    target_d: np.ndarray,
    val_ids: np.ndarray,
    train_pos: np.ndarray,
    paths: np.lib.npyio.NpzFile,
    limit: int,
    analytic: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    delay = paths["train_delay_bin_physical"].astype(np.float32)
    depth = paths["train_interaction_count"].astype(np.int64)
    valid = paths["train_valid"].astype(bool)
    reconstruction = np.zeros((len(val_ids), S_SC), dtype=np.float32)
    path_mass = np.zeros(len(val_ids), dtype=np.float32)
    used = np.zeros(len(val_ids), dtype=np.int32)
    maximum_depth = np.zeros(len(val_ids), dtype=np.int32)
    for row, point_id in enumerate(val_ids):
        ids = np.flatnonzero(valid[point_id, :limit])
        if len(ids):
            atoms = v259.point_atoms(delay[point_id, ids], depth[point_id, ids])
            maximum_depth[row] = int(np.max(depth[point_id, ids]))
        else:
            atoms = v259.fixed_atoms(0.0)
        if analytic is not None:
            atoms = np.concatenate(
                [analytic_atoms(int(point_id), train_pos, analytic), atoms],
                axis=0,
            )
        reconstruction[row], path_mass[row] = v259.solve(target_d[point_id], atoms)
        used[row] = len(ids)
    cosine = v259.row_cos(reconstruction, target_d[val_ids])
    return {
        "cosine": summarize(cosine),
        "path_mass": summarize(path_mass),
        "used_paths": summarize(used),
        "max_depth": summarize(maximum_depth),
        "zero_path_count": int(np.sum(used == 0)),
    }


def main() -> None:
    for path in (TARGET_CACHE, PATH_CACHE, LOW_DEPTH_CACHE, ANALYTIC_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    target_data = np.load(TARGET_CACHE)
    target_d = target_data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    matched = v254.choose_seeded_matched_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    high = np.load(PATH_CACHE)
    low = np.load(LOW_DEPTH_CACHE)
    analytic = dict(np.load(ANALYTIC_CACHE))

    results: dict[str, object] = {}
    for limit in LIMITS:
        results[f"low_depth_{limit}"] = evaluate(
            target_d, val_ids, train_pos, low, limit, None
        )
        results[f"high_depth_{limit}"] = evaluate(
            target_d, val_ids, train_pos, high, limit, None
        )
        results[f"analytic_plus_high_{limit}"] = evaluate(
            target_d, val_ids, train_pos, high, limit, analytic
        )
        print(
            f"v291 limit={limit} "
            f"low={results[f'low_depth_{limit}']['cosine']['mean']:.9f} "
            f"high={results[f'high_depth_{limit}']['cosine']['mean']:.9f} "
            f"union={results[f'analytic_plus_high_{limit}']['cosine']['mean']:.9f}",
            flush=True,
        )

    high_depth = high["train_interaction_count"][high["train_valid"]]
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The depth-16 high-budget official-point-cloud solver exposes high-order path "
            "multiplicity and delays absent from the depth-4 geometry cache."
        ),
        "compliance_boundary": {
            "uses": [
                "official point-cloud wall mesh",
                "official train/test coordinates",
                "Sionna geometry and visibility only",
                "official train PDP for an analysis-only NNLS oracle",
            ],
            "does_not_use": [
                "Paths.cfr or path coefficients",
                "synthetic/Sionna channel labels",
                "teacher output",
                "local replay truth or hidden labels",
            ],
        },
        "parameters": {
            "validation_seed": VAL_SEED,
            "validation_count": int(len(val_ids)),
            "limits": list(LIMITS),
        },
        "high_depth_diagnostics": {
            "interaction_depth": summarize(high_depth),
            "train_path_count": summarize(high["train_valid"].sum(axis=1)),
            "test_path_count": summarize(high["test_valid"].sum(axis=1)),
        },
        "results": results,
        "outputs": {
            "paths": PATHS.relative(PATH_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
