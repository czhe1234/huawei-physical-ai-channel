#!/usr/bin/env python3
"""Screen spatial spectral interpolators on the fixed Round 2 hole folds."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree, distance_matrix

from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_spectral_transport"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_spectral_transport"
REPORT = ANALYSIS_DIR / "interpolator_screen.json"
WEIGHTS = ARTIFACT_DIR / "interpolator_screen_weights.npz"

DESCRIPTORS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_spectral_descriptors.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
V2_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)


@dataclass(frozen=True)
class KernelConfig:
    name: str
    kernel: str
    neighbors: int
    smoothing: float
    length_scale: float = 1.0
    affine: bool = False


CONFIGS = (
    KernelConfig("linear_const_v2", "negative_linear", 64, 10.0),
    KernelConfig("linear_affine_s03", "negative_linear", 64, 0.3, affine=True),
    KernelConfig("linear_affine_s1", "negative_linear", 64, 1.0, affine=True),
    KernelConfig("linear_affine_s3", "negative_linear", 64, 3.0, affine=True),
    KernelConfig("linear_affine_s10", "negative_linear", 64, 10.0, affine=True),
    KernelConfig("linear_affine_s30", "negative_linear", 64, 30.0, affine=True),
    KernelConfig("tps_affine_s1", "thin_plate", 64, 1.0, affine=True),
    KernelConfig("tps_affine_s10", "thin_plate", 64, 10.0, affine=True),
    KernelConfig("tps_affine_s100", "thin_plate", 64, 100.0, affine=True),
    KernelConfig("cubic_affine_s10", "cubic", 64, 10.0, affine=True),
    KernelConfig("cubic_affine_s100", "cubic", 64, 100.0, affine=True),
    KernelConfig("cubic_affine_s1000", "cubic", 64, 1000.0, affine=True),
    KernelConfig("matern_l08_n001", "matern32", 64, 0.01, 8.0),
    KernelConfig("matern_l12_n001", "matern32", 64, 0.01, 12.0),
    KernelConfig("matern_l20_n001", "matern32", 64, 0.01, 20.0),
    KernelConfig("matern_l32_n001", "matern32", 64, 0.01, 32.0),
    KernelConfig("matern_l20_n005", "matern32", 64, 0.05, 20.0),
    KernelConfig("matern_l32_n005", "matern32", 64, 0.05, 32.0),
)


def kernel_value(distance: np.ndarray, config: KernelConfig) -> np.ndarray:
    radius = np.asarray(distance, dtype=np.float64)
    if config.kernel == "negative_linear":
        return -radius
    if config.kernel == "thin_plate":
        safe = np.maximum(radius, 1e-12)
        value = radius * radius * np.log(safe)
        return np.where(radius > 0.0, value, 0.0)
    if config.kernel == "cubic":
        return radius**3
    if config.kernel == "matern32":
        scaled = math.sqrt(3.0) * radius / config.length_scale
        return (1.0 + scaled) * np.exp(-scaled)
    raise ValueError(f"unknown kernel: {config.kernel}")


def solve_weights(
    donor_xy: np.ndarray,
    query_xy: np.ndarray,
    config: KernelConfig,
) -> np.ndarray:
    count = len(donor_xy)
    offset = np.asarray(donor_xy, dtype=np.float64) - np.asarray(
        query_xy, dtype=np.float64
    )[None, :]
    local_scale = max(float(np.median(np.linalg.norm(offset, axis=1))), 1.0)
    if config.affine:
        polynomial = np.column_stack(
            [np.ones(count), offset[:, 0] / local_scale, offset[:, 1] / local_scale]
        )
        query_polynomial = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        polynomial = np.ones((count, 1), dtype=np.float64)
        query_polynomial = np.ones(1, dtype=np.float64)

    kernel = kernel_value(distance_matrix(donor_xy, donor_xy), config)
    kernel[np.diag_indices(count)] += config.smoothing
    right_kernel = kernel_value(np.linalg.norm(offset, axis=1), config)
    order = polynomial.shape[1]
    system = np.zeros((count + order, count + order), dtype=np.float64)
    system[:count, :count] = kernel
    system[:count, count:] = polynomial
    system[count:, :count] = polynomial.T
    right = np.concatenate([right_kernel, query_polynomial])
    try:
        result = np.linalg.solve(system, right)[:count]
    except np.linalg.LinAlgError:
        result = np.linalg.lstsq(system, right, rcond=1e-10)[0][:count]
    return result.astype(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=1, dtype=np.float64)
        * np.sum(right * right, axis=1, dtype=np.float64)
    )
    return np.divide(
        numerator,
        np.maximum(denominator, 1e-30),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def predict_spectrum(
    cache: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    donor = np.sqrt(np.maximum(cache[donor_ids], 0.0)).astype(np.float32)
    amplitude = np.maximum(
        np.einsum("qk,qkd->qd", weights, donor, optimize=True), 0.0
    )
    power = amplitude * amplitude
    total = np.sum(power, axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        power,
        np.maximum(total, 1e-20),
        out=np.zeros_like(power),
        where=total > 0.0,
    )


def build_oof_weights(
    config: KernelConfig,
    positions: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids = np.empty((len(query_ids), config.neighbors), dtype=np.int32)
    weights = np.empty((len(query_ids), config.neighbors), dtype=np.float32)
    folds = np.load(FOLD_CACHE)
    for held_fold in np.unique(fold_id):
        fold_rows = np.flatnonzero(fold_id == held_fold)
        fit_ids = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
        for site_value in np.unique(site[query_ids[fold_rows]]):
            rows = fold_rows[site[query_ids[fold_rows]] == site_value]
            bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
            _, local = cKDTree(positions[bank, :2]).query(
                positions[query_ids[rows], :2],
                k=config.neighbors,
                workers=-1,
            )
            selected = bank[local]
            donor_ids[rows] = selected
            for local_row, output_row in enumerate(rows):
                weights[output_row] = solve_weights(
                    positions[selected[local_row], :2],
                    positions[query_ids[output_row], :2],
                    config,
                )
    return donor_ids, weights


def summarize(
    values: np.ndarray,
    fold_id: np.ndarray,
    query_site: np.ndarray,
) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [
            float(np.mean(values[fold_id == fold])) for fold in np.unique(fold_id)
        ],
        "site": {
            str(int(site_value)): float(np.mean(values[query_site == site_value]))
            for site_value in np.unique(query_site)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional config names; the default evaluates the complete screen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    descriptors = np.load(DESCRIPTORS)
    global_pas = descriptors["global_pas"].astype(np.float32)
    global_pdp = descriptors["global_pdp"].astype(np.float32)
    valid = descriptors["valid"].astype(bool)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, positions, _, site, _ = load_positions()
    query_site = site[query_ids]

    requested = set(args.configs or ())
    configs = [config for config in CONFIGS if not requested or config.name in requested]
    if requested - {config.name for config in configs}:
        raise ValueError(f"unknown configs: {sorted(requested - {c.name for c in configs})}")

    truth_pas = global_pas[query_ids]
    truth_pdp = global_pdp[query_ids]
    reports: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id,
    }

    for config in configs:
        if config.name == "linear_const_v2" and V2_WEIGHTS.is_file():
            cached = np.load(V2_WEIGHTS)
            donor_ids = cached["neighbor_ids"].astype(np.int32)
            weights = cached["weights"].astype(np.float32)
        else:
            donor_ids, weights = build_oof_weights(
                config, positions, site, valid, query_ids, fold_id
            )
        predicted_pas = predict_spectrum(global_pas, donor_ids, weights)
        predicted_pdp = predict_spectrum(global_pdp, donor_ids, weights)
        pas = cosine(predicted_pas, truth_pas)
        pdp = cosine(predicted_pdp, truth_pdp)
        combined = 0.5 * (pas + pdp)
        row = {
            "config": asdict(config),
            "pas": summarize(pas, fold_id, query_site),
            "pdp": summarize(pdp, fold_id, query_site),
            "combined": summarize(combined, fold_id, query_site),
            "weight_l1_mean": float(np.mean(np.sum(np.abs(weights), axis=1))),
            "weight_l1_p95": float(np.quantile(np.sum(np.abs(weights), axis=1), 0.95)),
            "weight_max_abs": float(np.max(np.abs(weights))),
        }
        reports.append(row)
        saved[f"{config.name}_donor_ids"] = donor_ids
        saved[f"{config.name}_weights"] = weights
        print(
            f"{config.name}: PAS={row['pas']['mean']:.6f} "
            f"PDP={row['pdp']['mean']:.6f} combined={row['combined']['mean']:.6f}",
            flush=True,
        )

    baseline = next(row for row in reports if row["config"]["name"] == "linear_const_v2")
    for row in reports:
        row["delta_vs_v2"] = {
            key: float(row[key]["mean"] - baseline[key]["mean"])
            for key in ("pas", "pdp", "combined")
        }
    ranking = sorted(
        reports, key=lambda row: float(row["combined"]["mean"]), reverse=True
    )
    payload = {
        "experiment": "Round2-E4-V3-interpolator-screen",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes",
        "results": reports,
        "ranking": [
            {
                "name": row["config"]["name"],
                "pas": row["pas"]["mean"],
                "pdp": row["pdp"]["mean"],
                "combined": row["combined"]["mean"],
                "delta_combined_vs_v2": row["delta_vs_v2"]["combined"],
            }
            for row in ranking
        ],
        "seconds": float(time.time() - started),
    }
    np.savez_compressed(WEIGHTS, **saved)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["ranking"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
