#!/usr/bin/env python3
"""Test geometry-local complex carrier stacking on strict Round 2 holes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np

from analyze_v7_complex_carrier_stacking import GRAM_CACHE, _solve_weights
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META


VERSION = "Round2-E47-local-complex-carrier-stacking"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "complex_carrier_stacking"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_complex_carrier_stacking"
GEODESIC = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
REPORT = ANALYSIS_DIR / "local_complex_carrier_stacking.json"


@dataclass(frozen=True)
class LocalConfig:
    mode: str
    neighbors: int
    ridge: float
    sum_penalty: float

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_k{self.neighbors}_ridge{self.ridge:g}"
            f"_sum{self.sum_penalty:g}"
        )


def configs() -> list[LocalConfig]:
    return [
        LocalConfig(mode, neighbors, ridge, penalty)
        for mode in ("xy", "geometry", "combined")
        for neighbors in (16, 32, 64, 128)
        for ridge in (0.1, 1.0)
        for penalty in (0.0, 0.1)
    ]


def standardized(value: np.ndarray, rows: np.ndarray) -> np.ndarray:
    mean = np.mean(value[rows], axis=0, dtype=np.float64)
    scale = np.std(value[rows], axis=0, dtype=np.float64)
    return ((value - mean) / np.maximum(scale, 1e-4)).astype(np.float32)


def feature_distance(
    config: LocalConfig,
    xy: np.ndarray,
    geometry: np.ndarray,
    fit_rows: np.ndarray,
    query_row: int,
) -> np.ndarray:
    xy_distance = np.linalg.norm(xy[fit_rows] - xy[query_row], axis=1)
    geometry_distance = np.sqrt(
        np.mean(
            np.square(geometry[fit_rows] - geometry[query_row]),
            axis=1,
            dtype=np.float64,
        )
    )
    if config.mode == "xy":
        return xy_distance
    if config.mode == "geometry":
        return geometry_distance
    if config.mode == "combined":
        xy_scale = max(float(np.median(xy_distance)), 1e-6)
        geometry_scale = max(float(np.median(geometry_distance)), 1e-6)
        return np.sqrt(
            np.square(xy_distance / xy_scale)
            + np.square(geometry_distance / geometry_scale)
        )
    raise ValueError(config.mode)


def local_error(
    cache: dict[str, np.ndarray],
    config: LocalConfig,
    fit_rows: np.ndarray,
    query_rows: np.ndarray,
    site: np.ndarray,
    xy_raw: np.ndarray,
    geometry_raw: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    gram = cache["gram"]
    cross = cache["cross"]
    energy = cache["energy"]
    xy = standardized(xy_raw, fit_rows)
    geometry = standardized(geometry_raw, fit_rows)
    errors = np.empty(len(query_rows), dtype=np.float64)
    diagnostics: list[dict[str, float]] = []
    solve_config = type(
        "SolveConfig",
        (),
        {"ridge": config.ridge, "sum_penalty": config.sum_penalty},
    )()
    for destination, row in enumerate(query_rows):
        available = fit_rows[site[fit_rows] == site[row]]
        distance = feature_distance(config, xy, geometry, available, int(row))
        take = min(config.neighbors, len(available))
        order = np.argpartition(distance, take - 1)[:take]
        selected = available[order]
        selected_distance = distance[order]
        local_scale = max(float(np.max(selected_distance)), 1e-6)
        local_weight = np.exp(-np.square(selected_distance / local_scale))
        local_weight /= max(float(np.sum(local_weight)), 1e-30)
        total = 0.0
        norms = []
        for port in range(8):
            local_gram = np.sum(
                local_weight[:, None, None] * gram[selected, port],
                axis=0,
                dtype=np.complex128,
            )
            local_cross = np.sum(
                local_weight[:, None] * cross[selected, port],
                axis=0,
                dtype=np.complex128,
            )
            weight = _solve_weights(local_gram, local_cross, solve_config)
            value = (
                np.vdot(weight, gram[row, port].astype(np.complex128) @ weight).real
                - 2.0
                * np.vdot(weight, cross[row, port].astype(np.complex128)).real
                + energy[row, port]
            )
            total += float(value)
            norms.append(float(np.linalg.norm(weight)))
        errors[destination] = max(total, 0.0)
        diagnostics.append(
            {
                "distance_max": float(np.max(selected_distance)),
                "distance_median": float(np.median(selected_distance)),
                "weight_norm_mean": float(np.mean(norms)),
            }
        )
    return errors, diagnostics


def evaluate_all(
    cache: dict[str, np.ndarray],
    site: np.ndarray,
    xy: np.ndarray,
    geometry: np.ndarray,
) -> list[dict[str, object]]:
    fold_id = cache["fold_id"].astype(np.int8)
    rows = np.arange(len(fold_id), dtype=np.int64)
    records: list[dict[str, object]] = []
    for config in configs():
        total_error = 0.0
        total_energy = 0.0
        fold_nmse = []
        diagnostics = []
        for fold in np.unique(fold_id):
            fit = rows[fold_id != fold]
            held = rows[fold_id == fold]
            error, detail = local_error(
                cache, config, fit, held, site, xy, geometry
            )
            energy = float(cache["energy"][held].sum())
            total_error += float(np.sum(error))
            total_energy += energy
            fold_nmse.append(float(np.sum(error) / max(energy, 1e-30)))
            diagnostics.extend(detail)
        nmse = total_error / max(total_energy, 1e-30)
        records.append(
            {
                "config": {
                    "name": config.name,
                    "mode": config.mode,
                    "neighbors": config.neighbors,
                    "ridge": config.ridge,
                    "sum_penalty": config.sum_penalty,
                },
                "nmse": float(nmse),
                "nmse_score": float(1.0 / (1.0 + nmse)),
                "fold_nmse": fold_nmse,
                "distance_max_mean": float(
                    np.mean([value["distance_max"] for value in diagnostics])
                ),
                "weight_norm_mean": float(
                    np.mean([value["weight_norm_mean"] for value in diagnostics])
                ),
            }
        )
        print(f"local stack {config.name} nmse={nmse:.6f}", flush=True)
    return sorted(records, key=lambda row: float(row["nmse"]))


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = np.load(GRAM_CACHE, allow_pickle=False)
    cache = {key: cache_file[key] for key in cache_file.files}
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    _, train_positions, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    xy = train_positions[query_ids, :2].astype(np.float32)
    geometry_all = np.load(GEODESIC, allow_pickle=False)["train_features"].astype(
        np.float32
    )
    geometry = geometry_all[query_ids]
    ranking = evaluate_all(cache, site, xy, geometry)
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "Carrier reliability varies with location and propagation state; locally "
            "fitted complex stacks should retain the bank's pointwise complementarity "
            "better than one site-wide coefficient vector."
        ),
        "global_site_reference_nmse": 0.9223085540567044,
        "ranking": ranking,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
