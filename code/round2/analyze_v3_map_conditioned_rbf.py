#!/usr/bin/env python3
"""Screen official-map-conditioned RBF neighborhoods for Round 2 V3."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from analyze_v3_spectral_interpolators import (
    KernelConfig,
    cosine,
    predict_spectrum,
    solve_weights,
)
from data import load_positions
from project_paths import PATHS


ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_spectral_transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_spectral_transport"
REPORT = ANALYSIS_DIR / "map_conditioned_rbf_screen.json"
WEIGHTS = ARTIFACT_DIR / "map_conditioned_rbf_weights.npz"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
GEOMETRY = PRE_DIR / "round2_geometry_visibility_features.npz"
OOF_META = PRE_DIR / "round2_pre_sionna_oof_foundation_meta.npz"
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
V2_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)

RBF = KernelConfig("linear_const_v2", "negative_linear", 64, 10.0)


@dataclass(frozen=True)
class MapConfig:
    name: str
    feature_group: str
    penalty_m: float
    candidate_neighbors: int = 256


FEATURE_GROUPS = {
    "visibility": np.asarray([14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]),
    "surface": np.arange(27, 37),
    "visibility_surface": np.r_[14:27, 27:37],
    "propagation": np.r_[6:12, 14:27, 27:37],
}

CONFIGS = tuple(
    MapConfig(f"{group}_p{penalty:g}", group, penalty)
    for group in FEATURE_GROUPS
    for penalty in (1.0, 2.0, 4.0, 8.0, 12.0, 20.0)
)


def robust_scale(values: np.ndarray) -> np.ndarray:
    lower = np.quantile(values, 0.10, axis=0)
    upper = np.quantile(values, 0.90, axis=0)
    return np.maximum(upper - lower, 1e-3).astype(np.float32)


def build_weights(
    config: MapConfig,
    positions: np.ndarray,
    geometry: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids = np.empty((len(query_ids), RBF.neighbors), dtype=np.int32)
    weights = np.empty((len(query_ids), RBF.neighbors), dtype=np.float32)
    folds = np.load(FOLD_CACHE)
    columns = FEATURE_GROUPS[config.feature_group]
    for held_fold in np.unique(fold_id):
        fold_rows = np.flatnonzero(fold_id == held_fold)
        fit_ids = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
        for site_value in np.unique(site[query_ids[fold_rows]]):
            rows = fold_rows[site[query_ids[fold_rows]] == site_value]
            bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
            count = min(config.candidate_neighbors, len(bank))
            distance, local = cKDTree(positions[bank, :2]).query(
                positions[query_ids[rows], :2], k=count, workers=-1
            )
            candidate = bank[local]
            scale = robust_scale(geometry[bank][:, columns])
            for local_row, output_row in enumerate(rows):
                ids = candidate[local_row]
                delta = (
                    geometry[ids][:, columns]
                    - geometry[query_ids[output_row], columns][None, :]
                ) / scale[None, :]
                feature_distance = np.sqrt(np.mean(delta * delta, axis=1))
                effective = distance[local_row] + config.penalty_m * feature_distance
                selected = ids[np.argsort(effective)[: RBF.neighbors]]
                donor_ids[output_row] = selected
                weights[output_row] = solve_weights(
                    positions[selected, :2], positions[query_ids[output_row], :2], RBF
                )
    return donor_ids, weights


def summarize(values: np.ndarray, fold: np.ndarray, site: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [float(np.mean(values[fold == item])) for item in np.unique(fold)],
        "site": {
            str(int(item)): float(np.mean(values[site == item]))
            for item in np.unique(site)
        },
    }


def main() -> None:
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    descriptors = np.load(DESCRIPTORS)
    global_pas = descriptors["global_pas"].astype(np.float32)
    global_pdp = descriptors["global_pdp"].astype(np.float32)
    valid = descriptors["valid"].astype(bool)
    geometry = np.load(GEOMETRY)["train"].astype(np.float32)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, positions, _, site, _ = load_positions()
    query_site = site[query_ids]
    truth_pas = global_pas[query_ids]
    truth_pdp = global_pdp[query_ids]

    baseline_cache = np.load(V2_WEIGHTS)
    base_ids = baseline_cache["neighbor_ids"].astype(np.int64)
    base_weights = baseline_cache["weights"].astype(np.float32)
    base_pas = cosine(predict_spectrum(global_pas, base_ids, base_weights), truth_pas)
    base_pdp = cosine(predict_spectrum(global_pdp, base_ids, base_weights), truth_pdp)
    baseline_combined = 0.5 * (base_pas + base_pdp)

    rows: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id,
    }
    for config in CONFIGS:
        donor_ids, weights = build_weights(
            config, positions, geometry, site, valid, query_ids, fold_id
        )
        pas = cosine(predict_spectrum(global_pas, donor_ids, weights), truth_pas)
        pdp = cosine(predict_spectrum(global_pdp, donor_ids, weights), truth_pdp)
        combined = 0.5 * (pas + pdp)
        row = {
            "config": asdict(config),
            "pas": summarize(pas, fold_id, query_site),
            "pdp": summarize(pdp, fold_id, query_site),
            "combined": summarize(combined, fold_id, query_site),
            "delta_vs_v2": {
                "pas": float(np.mean(pas - base_pas)),
                "pdp": float(np.mean(pdp - base_pdp)),
                "combined": float(np.mean(combined - baseline_combined)),
                "fold": [
                    float(np.mean((combined - baseline_combined)[fold_id == item]))
                    for item in np.unique(fold_id)
                ],
            },
        }
        rows.append(row)
        saved[f"{config.name}_donor_ids"] = donor_ids
        saved[f"{config.name}_weights"] = weights
        print(
            f"{config.name}: PAS={row['pas']['mean']:.6f} "
            f"PDP={row['pdp']['mean']:.6f} delta={row['delta_vs_v2']['combined']:+.6f}",
            flush=True,
        )

    ranking = sorted(rows, key=lambda row: float(row["combined"]["mean"]), reverse=True)
    payload = {
        "experiment": "Round2-E4-V3-map-conditioned-RBF-screen",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes",
        "baseline": {
            "pas": summarize(base_pas, fold_id, query_site),
            "pdp": summarize(base_pdp, fold_id, query_site),
            "combined": summarize(baseline_combined, fold_id, query_site),
        },
        "results": rows,
        "ranking": [
            {
                "name": row["config"]["name"],
                "pas": row["pas"]["mean"],
                "pdp": row["pdp"]["mean"],
                "combined": row["combined"]["mean"],
                "delta_vs_v2": row["delta_vs_v2"],
            }
            for row in ranking
        ],
        "seconds": float(time.time() - started),
    }
    np.savez_compressed(WEIGHTS, **saved)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["ranking"][:8], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
