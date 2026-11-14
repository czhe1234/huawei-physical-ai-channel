#!/usr/bin/env python3
"""v259: NNLS oracle audit for the 128 v251 geometry paths.

This isolates path-set expressiveness from scalar Sionna field quality and from
geometry-to-weight learning. It uses the same atom widths, offsets, ridge, and
sum penalty as v239, but substitutes the physically visible v251 path delays.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import nnls

import analyze_v239_nnls_path_energy_decoder as v239
import analyze_v254_route_topology_spectral_field as v254
from generate_v87_easy_field_projection import S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 259
VERSION = "v259_v251_path_atom_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
PATH_CACHE = PATHS.cache_dir(251) / "v251_pointcloud_sionna_paths.npz"
V239_NNLS = PATHS.cache_dir(239) / "v239_train_nnls_path_labels.npz"

VAL_COUNT = 320
VAL_SEED = 254
PATH_LIMITS = (16, 32, 64, 128)
SHIFTS = np.asarray([-0.85, 0.0, 0.85], dtype=np.float32)
RIDGE = 0.022
SUM_PENALTY = 0.55


def normalize_d(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 1e-10)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-20)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def fixed_atoms(max_delay: float) -> np.ndarray:
    x = np.arange(S_SC, dtype=np.float32)
    early0 = np.exp(-0.5 * (x / 0.55) ** 2)
    early1 = np.exp(-0.5 * ((x - 1.0) / 0.75) ** 2)
    early2 = np.exp(-0.5 * ((x - 2.0) / 1.05) ** 2)
    broad = np.exp(-0.5 * ((x - 10.0) / 4.5) ** 2)
    delta = np.maximum(x - max_delay, 0.0)
    causal = (x >= max_delay).astype(np.float32)
    tail8 = np.exp(-delta / 8.0) * causal
    tail22 = np.exp(-delta / 22.0) * causal
    atoms = np.stack([early0, early1, early2, broad, tail8, tail22], axis=0)
    atoms /= np.maximum(atoms.sum(axis=1, keepdims=True), 1e-30)
    return atoms.astype(np.float32)


def point_atoms(delay: np.ndarray, interaction_count: np.ndarray) -> np.ndarray:
    bins = np.arange(S_SC, dtype=np.float32)[None, None, :]
    center = delay[:, None, None] + SHIFTS[None, :, None]
    sigma = np.where(interaction_count == 0, 0.62, 0.96).astype(np.float32)
    atoms = np.exp(-0.5 * ((bins - center) / sigma[:, None, None]) ** 2)
    atoms /= np.maximum(atoms.sum(axis=2, keepdims=True), 1e-30)
    path_atoms = atoms.reshape(-1, S_SC)
    return np.concatenate([path_atoms, fixed_atoms(float(np.max(delay)))], axis=0)


def solve(target: np.ndarray, atoms: np.ndarray) -> tuple[np.ndarray, float]:
    atom_count = len(atoms)
    eye = math.sqrt(RIDGE) * np.eye(atom_count, dtype=np.float64)
    sum_row = math.sqrt(SUM_PENALTY) * np.ones((1, atom_count), dtype=np.float64)
    matrix = np.vstack([atoms.T.astype(np.float64), eye, sum_row])
    rhs = np.concatenate(
        [
            target.astype(np.float64),
            np.zeros(atom_count, dtype=np.float64),
            [math.sqrt(SUM_PENALTY)],
        ]
    )
    weight, _ = nnls(matrix, rhs, maxiter=atom_count * 10)
    if weight.sum() <= 1e-20:
        weight[-1] = 1.0
    weight /= weight.sum()
    reconstruction = normalize_d((weight[None] @ atoms).astype(np.float32))[0]
    path_mass = float(weight[:-6].sum())
    return reconstruction, path_mass


def main() -> None:
    for path in (TARGET_CACHE, PATH_CACHE, V239_NNLS):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    target_data = np.load(TARGET_CACHE)
    target_d = target_data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    matched = v254.choose_seeded_matched_indices(
        train_pos, test_pos, VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    paths = np.load(PATH_CACHE)
    delay = paths["train_delay_bin_physical"].astype(np.float32)
    interaction_count = paths["train_interaction_count"].astype(np.int64)
    path_valid = paths["train_valid"].astype(bool)
    v239_data = np.load(V239_NNLS)
    v239_reconstruction = v239_data["reconstruction"].astype(np.float32)

    results: dict[str, object] = {
        "v239_same_points": {
            "cosine": summarize(row_cos(v239_reconstruction[val_ids], target_d[val_ids]))
        }
    }
    for limit in PATH_LIMITS:
        reconstruction = np.zeros((len(val_ids), S_SC), dtype=np.float32)
        path_mass = np.zeros(len(val_ids), dtype=np.float32)
        used_paths = np.zeros(len(val_ids), dtype=np.int32)
        for row, point_id in enumerate(val_ids):
            ids = np.flatnonzero(path_valid[point_id, :limit])
            if len(ids) == 0:
                atoms = fixed_atoms(0.0)
            else:
                atoms = point_atoms(
                    delay[point_id, ids], interaction_count[point_id, ids]
                )
            reconstruction[row], path_mass[row] = solve(target_d[point_id], atoms)
            used_paths[row] = len(ids)
        cosine = row_cos(reconstruction, target_d[val_ids])
        results[f"top_{limit}"] = {
            "cosine": summarize(cosine),
            "path_mass": summarize(path_mass),
            "used_paths": summarize(used_paths),
            "zero_path_count": int(np.sum(used_paths == 0)),
        }
        print(f"v259 top {limit}: mean cosine {float(np.mean(cosine)):.9f}", flush=True)

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The 128 physically visible v251 paths provide a materially more expressive delay "
            "atom set than the 15 compressed v239 paths when scalar field quality is removed."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H PDP",
                "official train/test coordinates",
                "v251 geometry-only paths reconstructed from official point cloud",
                "validation truth for an analysis-only NNLS oracle",
            ],
            "does_not_use": [
                "Sionna CFR or MIMO RT H",
                "synthetic channel labels",
                "teacher output",
                "local replay truth",
                "hidden labels",
            ],
        },
        "parameters": {
            "validation_seed": VAL_SEED,
            "validation_count": int(len(val_ids)),
            "path_limits": list(PATH_LIMITS),
            "shifts": SHIFTS.tolist(),
            "ridge": RIDGE,
            "sum_penalty": SUM_PENALTY,
        },
        "results": results,
        "local_gate": None,
        "outputs": {"report": PATHS.relative(REPORT), "prediction": None, "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
