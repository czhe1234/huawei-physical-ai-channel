#!/usr/bin/env python3
"""Predict grouped PAS/PDP with mode-specific nonstationary spatial fields.

The experiment keeps the V3/V4 grouped spectral anchor and changes one thing:
each latent spectral mode may use a different spatial covariance model.  Model
selection and correction strength are estimated only from the other complete
spatial holes inside each outer fold.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import FOLD_CACHE
from run_v3_grouped_link_refinement import GROUPED_PAS, GROUPED_PDP


VERSION = "Round2-E51-mode-adaptive-spectral-field"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "mode_adaptive_spectral_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_mode_adaptive_spectral_field"
REPORT = ANALYSIS_DIR / "mode_adaptive_spectral_field.json"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
OOF_META = PRE_DIR / "round2_pre_sionna_oof_foundation_meta.npz"
V3_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "v3_anchored_neural_rbf"
    / "oof_epoch6_corrected_weights.npz"
)
GEODESIC = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)

TINY = 1e-30
NEIGHBORS = 96
SPATIAL_LENGTHS = (3.0, 5.0, 8.0, 12.0, 18.0, 27.0, 40.0)
GEOMETRY_CONFIGS = ((8.0, 0.20), (12.0, 0.20), (12.0, 0.55), (18.0, 0.55))
PLANE_LENGTHS = (6.0, 12.0, 24.0)
BLEND_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
GEO_COLUMNS = np.asarray(
    (0, 2, 3, 4, 10, 11, 12, 13, 15, 17, 22, 23, 25, 26, 27),
    dtype=np.int64,
)


@dataclass(frozen=True)
class Basis:
    mean: np.ndarray
    vectors: np.ndarray
    singular_values: np.ndarray


def _normalize_groups(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    norm = np.sqrt(np.sum(output * output, axis=-1, keepdims=True, dtype=np.float64))
    return np.divide(
        output,
        np.maximum(norm, TINY),
        out=np.zeros_like(output),
        where=norm > 0.0,
    )


def _grouped_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_power = np.square(np.asarray(prediction, dtype=np.float32))
    truth_power = np.square(np.asarray(truth, dtype=np.float32))
    numerator = np.sum(pred_power * truth_power, axis=-1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(pred_power * pred_power, axis=-1, dtype=np.float64)
        * np.sum(truth_power * truth_power, axis=-1, dtype=np.float64)
    )
    cosine = np.divide(
        numerator,
        np.maximum(denominator, TINY),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return cosine.mean(axis=1, dtype=np.float64)


def _fit_basis(
    value: np.ndarray,
    fit_ids: np.ndarray,
    components: int,
    device: torch.device,
    seed: int,
) -> Basis:
    torch.manual_seed(seed)
    matrix = torch.as_tensor(
        np.asarray(value[fit_ids], dtype=np.float32).reshape(len(fit_ids), -1),
        device=device,
    )
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    count = min(components, len(fit_ids) - 1, centered.shape[1])
    _, singular_values, vectors = torch.pca_lowrank(
        centered,
        q=count,
        center=False,
        niter=4,
    )
    return Basis(
        mean=mean.cpu().numpy().astype(np.float32),
        vectors=vectors.cpu().numpy().astype(np.float32),
        singular_values=singular_values.cpu().numpy().astype(np.float32),
    )


def _encode(value: np.ndarray, ids: np.ndarray, basis: Basis) -> np.ndarray:
    flat = np.asarray(value[ids], dtype=np.float32).reshape(len(ids), -1)
    return (flat - basis.mean[None]) @ basis.vectors


def _decode(coefficients: np.ndarray, basis: Basis, shape: tuple[int, int]) -> np.ndarray:
    flat = basis.mean[None] + coefficients @ basis.vectors.T
    return _normalize_groups(flat.reshape(len(coefficients), *shape))


def _filter_anchor_weights(
    query_rows: np.ndarray,
    allowed: np.ndarray,
    metric: str,
    cached: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids = cached["donor_ids"][query_rows].astype(np.int64)
    key = "pas_weights" if metric == "pas" else "pdp_weights"
    weights = cached[key][query_rows].astype(np.float64)
    weights *= allowed[donor_ids]
    for row in range(len(weights)):
        total = float(np.sum(weights[row]))
        if abs(total) < 0.10:
            weights[row] = np.maximum(weights[row], 0.0)
            total = float(np.sum(weights[row]))
        if abs(total) < 1e-10:
            candidates = np.flatnonzero(allowed[donor_ids[row]])
            if len(candidates) == 0:
                raise RuntimeError("anchor has no fit-only donor")
            weights[row] = 0.0
            weights[row, candidates[0]] = 1.0
        else:
            weights[row] /= total
    return donor_ids, weights.astype(np.float32)


def _nearest_donors(
    query_ids: np.ndarray,
    allowed: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids = np.empty((len(query_ids), NEIGHBORS), dtype=np.int64)
    distances = np.empty((len(query_ids), NEIGHBORS), dtype=np.float32)
    for site_value in np.unique(site[query_ids]):
        rows = np.flatnonzero(site[query_ids] == site_value)
        bank = np.flatnonzero(allowed & (site == site_value))
        if len(bank) < NEIGHBORS:
            raise RuntimeError(f"site {site_value} has only {len(bank)} donors")
        local_distance, local_ids = cKDTree(positions[bank, :2]).query(
            positions[query_ids[rows], :2],
            k=NEIGHBORS,
            workers=-1,
        )
        donor_ids[rows] = bank[np.asarray(local_ids, dtype=np.int64)]
        distances[rows] = np.asarray(local_distance, dtype=np.float32)
    return donor_ids, distances


def _positive_kernel(distance: np.ndarray, length: float) -> np.ndarray:
    weights = np.exp(-0.5 * np.square(distance / length, dtype=np.float64))
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), TINY)
    return weights.astype(np.float32)


def _geometry_kernel(
    distance: np.ndarray,
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    geometry: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    length: float,
    strength: float,
) -> np.ndarray:
    query = (geometry[query_ids][:, GEO_COLUMNS] - mean) / scale
    donor = (geometry[donor_ids][:, :, GEO_COLUMNS] - mean) / scale
    geometry_distance = np.mean(np.square(donor - query[:, None]), axis=2)
    log_weight = -0.5 * np.square(distance / length, dtype=np.float64)
    log_weight -= strength * geometry_distance
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    weights = np.exp(log_weight)
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), TINY)
    return weights.astype(np.float32)


def _local_plane_weights(
    query_xy: np.ndarray,
    donor_xy: np.ndarray,
    distance: np.ndarray,
    length: float,
) -> np.ndarray:
    output = np.empty(distance.shape, dtype=np.float32)
    for row in range(len(query_xy)):
        delta = (donor_xy[row] - query_xy[row, None]) / max(length, 1e-6)
        design = np.column_stack(
            [np.ones(len(delta), dtype=np.float64), delta.astype(np.float64)]
        )
        kernel = np.exp(-0.5 * np.square(distance[row] / length, dtype=np.float64))
        normal = design.T @ (kernel[:, None] * design)
        normal[1, 1] += 0.08
        normal[2, 2] += 0.08
        right = design.T * kernel[None]
        weights = np.linalg.solve(normal, right)[0]
        total = float(np.sum(weights))
        output[row] = (weights / max(abs(total), 1e-12)).astype(np.float32)
    return output


def _candidate_predictions(
    metric: str,
    query_rows: np.ndarray,
    query_ids: np.ndarray,
    allowed: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    geometry: np.ndarray,
    coefficient_table: np.ndarray,
    cached_weights: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, list[str]]:
    predictions: list[np.ndarray] = []
    names: list[str] = []

    anchor_ids, anchor_weights = _filter_anchor_weights(
        query_rows, allowed, metric, cached_weights
    )
    predictions.append(
        np.einsum(
            "qk,qkc->qc",
            anchor_weights,
            coefficient_table[anchor_ids],
            optimize=True,
        )
    )
    names.append("v3_anchor")

    donor_ids, distance = _nearest_donors(query_ids, allowed, positions, site)
    donor_coefficients = coefficient_table[donor_ids]
    for length in SPATIAL_LENGTHS:
        weights = _positive_kernel(distance, length)
        predictions.append(
            np.einsum("qk,qkc->qc", weights, donor_coefficients, optimize=True)
        )
        names.append(f"gaussian_l{length:g}")

    fit_geometry = geometry[allowed][:, GEO_COLUMNS]
    geometry_mean = np.mean(fit_geometry, axis=0, dtype=np.float64)
    geometry_scale = np.maximum(
        np.std(fit_geometry, axis=0, dtype=np.float64), 1e-3
    )
    for length, strength in GEOMETRY_CONFIGS:
        weights = _geometry_kernel(
            distance,
            query_ids,
            donor_ids,
            geometry,
            geometry_mean,
            geometry_scale,
            length,
            strength,
        )
        predictions.append(
            np.einsum("qk,qkc->qc", weights, donor_coefficients, optimize=True)
        )
        names.append(f"geometric_l{length:g}_g{strength:g}")

    donor_xy = positions[donor_ids, :2]
    query_xy = positions[query_ids, :2]
    for length in PLANE_LENGTHS:
        weights = _local_plane_weights(query_xy, donor_xy, distance, length)
        predictions.append(
            np.einsum("qk,qkc->qc", weights, donor_coefficients, optimize=True)
        )
        names.append(f"local_plane_l{length:g}")

    return np.stack(predictions, axis=0).astype(np.float32), names


def _mode_policy(
    calibration_predictions: np.ndarray,
    calibration_truth: np.ndarray,
    held_predictions: np.ndarray,
    singular_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    base_calibration = calibration_predictions[0]
    base_held = held_predictions[0]
    components = calibration_truth.shape[1]
    selected = np.zeros(components, dtype=np.int16)
    slopes = np.zeros(components, dtype=np.float32)
    base_mse = np.mean(
        np.square(base_calibration - calibration_truth, dtype=np.float64), axis=0
    )
    selected_mse = base_mse.copy()

    for component in range(components):
        best_index = 0
        best_slope = 0.0
        best_mse = float(base_mse[component])
        target_residual = (
            calibration_truth[:, component] - base_calibration[:, component]
        ).astype(np.float64)
        for candidate in range(1, len(calibration_predictions)):
            proposal = (
                calibration_predictions[candidate, :, component]
                - base_calibration[:, component]
            ).astype(np.float64)
            denominator = float(np.dot(proposal, proposal))
            if denominator <= 1e-12:
                continue
            slope = float(np.clip(np.dot(proposal, target_residual) / denominator, 0.0, 1.25))
            # Shrink mode-specific decisions with limited calibration support.
            slope *= len(calibration_truth) / (len(calibration_truth) + 24.0)
            corrected = base_calibration[:, component] + slope * proposal
            mse = float(np.mean(np.square(corrected - calibration_truth[:, component])))
            if mse < best_mse * (1.0 - 1e-4):
                best_index = candidate
                best_slope = slope
                best_mse = mse
        selected[component] = best_index
        slopes[component] = best_slope
        selected_mse[component] = best_mse

    output = base_held.copy()
    for component in range(components):
        candidate = int(selected[component])
        if candidate == 0 or slopes[component] <= 0.0:
            continue
        output[:, component] += slopes[component] * (
            held_predictions[candidate, :, component] - base_held[:, component]
        )

    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    weighted_gain = float(
        np.sum(energy * (base_mse - selected_mse))
        / max(float(np.sum(energy * base_mse)), TINY)
    )
    return output, {
        "active_modes": int(np.count_nonzero(selected)),
        "mean_slope": float(np.mean(slopes[selected != 0])) if np.any(selected) else 0.0,
        "weighted_coefficient_mse_reduction": weighted_gain,
        "selected": selected.tolist(),
        "slopes": slopes.tolist(),
    }


def _blend(anchor: np.ndarray, proposal: np.ndarray, alpha: float) -> np.ndarray:
    return _normalize_groups((1.0 - alpha) * anchor + alpha * proposal)


def _metric_fold(
    outer_fold: int,
    metric: str,
    amplitude: np.ndarray,
    components: int,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    geometry: np.ndarray,
    folds: np.lib.npyio.NpzFile,
    cached_weights: np.lib.npyio.NpzFile,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    held_rows_all = np.flatnonzero(fold_id == outer_fold)
    output = np.asarray(amplitude[query_ids[held_rows_all]], dtype=np.float32).copy()
    site_records: list[dict[str, object]] = []

    outer_fit = np.zeros(len(amplitude), dtype=bool)
    outer_fit[folds[f"fold{outer_fold}_fit"].astype(np.int64)] = True
    outer_fit &= valid

    for site_value in np.unique(site[query_ids[held_rows_all]]):
        held_local_rows = np.flatnonzero(
            site[query_ids[held_rows_all]] == site_value
        )
        held_rows = held_rows_all[held_local_rows]
        held_ids = query_ids[held_rows]
        calibration_rows = np.flatnonzero(
            (fold_id != outer_fold)
            & (site[query_ids] == site_value)
            & outer_fit[query_ids]
        )
        calibration_ids = query_ids[calibration_rows]
        basis_fit_ids = np.flatnonzero(outer_fit & (site == site_value))
        basis = _fit_basis(
            amplitude,
            basis_fit_ids,
            components,
            device,
            20_260_806 + outer_fold * 1009 + int(site_value) * 97 + (0 if metric == "pas" else 41),
        )
        coefficient_table = _encode(
            amplitude,
            np.arange(len(amplitude), dtype=np.int64),
            basis,
        )

        calibration_prediction_parts: list[np.ndarray] = []
        calibration_truth_parts: list[np.ndarray] = []
        names: list[str] | None = None
        for calibration_fold in np.unique(fold_id[calibration_rows]):
            rows = calibration_rows[fold_id[calibration_rows] == calibration_fold]
            allowed = outer_fit.copy()
            allowed[folds[f"fold{int(calibration_fold)}_query"].astype(np.int64)] = False
            prediction, local_names = _candidate_predictions(
                metric,
                rows,
                query_ids[rows],
                allowed,
                positions,
                site,
                geometry,
                coefficient_table,
                cached_weights,
            )
            if names is None:
                names = local_names
            elif names != local_names:
                raise RuntimeError("candidate ordering changed")
            calibration_prediction_parts.append(prediction)
            calibration_truth_parts.append(coefficient_table[query_ids[rows]])
        calibration_predictions = np.concatenate(calibration_prediction_parts, axis=1)
        calibration_truth = np.concatenate(calibration_truth_parts, axis=0)

        held_predictions, held_names = _candidate_predictions(
            metric,
            held_rows,
            held_ids,
            outer_fit,
            positions,
            site,
            geometry,
            coefficient_table,
            cached_weights,
        )
        if names != held_names:
            raise RuntimeError("held candidate ordering changed")

        calibration_policy_coefficients, policy = _mode_policy(
            calibration_predictions,
            calibration_truth,
            calibration_predictions,
            basis.singular_values,
        )
        held_policy_coefficients, _ = _mode_policy(
            calibration_predictions,
            calibration_truth,
            held_predictions,
            basis.singular_values,
        )
        calibration_anchor = _decode(
            calibration_predictions[0], basis, amplitude.shape[1:]
        )
        calibration_proposal = _decode(
            calibration_policy_coefficients, basis, amplitude.shape[1:]
        )
        calibration_truth_amplitude = np.asarray(
            amplitude[calibration_ids], dtype=np.float32
        )
        alpha_scores = []
        for alpha in BLEND_GRID:
            alpha_scores.append(
                float(
                    np.mean(
                        _grouped_cosine(
                            _blend(calibration_anchor, calibration_proposal, alpha),
                            calibration_truth_amplitude,
                        )
                    )
                )
            )
        alpha_index = int(np.argmax(alpha_scores))
        alpha = BLEND_GRID[alpha_index]

        held_anchor = _decode(held_predictions[0], basis, amplitude.shape[1:])
        held_proposal = _decode(
            held_policy_coefficients, basis, amplitude.shape[1:]
        )
        held_output = _blend(held_anchor, held_proposal, alpha)
        output[held_local_rows] = held_output
        held_truth = np.asarray(amplitude[held_ids], dtype=np.float32)
        reconstruction = _decode(
            coefficient_table[held_ids], basis, amplitude.shape[1:]
        )
        site_records.append(
            {
                "site": int(site_value),
                "held_count": int(len(held_ids)),
                "calibration_count": int(len(calibration_ids)),
                "components": int(len(basis.singular_values)),
                "candidate_names": names,
                "alpha": float(alpha),
                "alpha_scores": alpha_scores,
                "anchor": float(np.mean(_grouped_cosine(held_anchor, held_truth))),
                "proposal": float(np.mean(_grouped_cosine(held_proposal, held_truth))),
                "selected": float(np.mean(_grouped_cosine(held_output, held_truth))),
                "pca_reconstruction": float(
                    np.mean(_grouped_cosine(reconstruction, held_truth))
                ),
                "policy": policy,
            }
        )

    truth = np.asarray(amplitude[query_ids[held_rows_all]], dtype=np.float32)
    return output, {
        "fold": int(outer_fold),
        "metric": metric,
        "score": float(np.mean(_grouped_cosine(output, truth))),
        "sites": site_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="1")
    parser.add_argument("--components-pas", type=int, default=96)
    parser.add_argument("--components-pdp", type=int, default=48)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (GROUPED_PAS, GROUPED_PDP, DESCRIPTORS, OOF_META, V3_WEIGHTS, FOLD_CACHE, GEODESIC)
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    _, train_positions, _, train_site, _ = load_positions()
    descriptor = np.load(DESCRIPTORS)
    valid = descriptor["valid"].astype(bool)
    meta = np.load(OOF_META)
    query_ids = meta["query_ids"].astype(np.int64)
    fold_id = meta["fold_id"].astype(np.int8)
    cached_weights = np.load(V3_WEIGHTS)
    if not np.array_equal(cached_weights["query_ids"], query_ids):
        raise RuntimeError("V3 OOF query ordering differs")
    folds = np.load(FOLD_CACHE)
    geometry_cache = np.load(GEODESIC)
    geometry = geometry_cache["train_features"].astype(np.float32)
    pas = np.load(GROUPED_PAS, mmap_mode="r")
    pdp = np.load(GROUPED_PDP, mmap_mode="r")
    device = torch.device(args.device)
    selected_folds = [int(value) for value in args.folds.split(",") if value.strip()]

    records: list[dict[str, object]] = []
    for outer_fold in selected_folds:
        fold_started = time.time()
        pas_output, pas_record = _metric_fold(
            outer_fold,
            "pas",
            pas,
            args.components_pas,
            query_ids,
            fold_id,
            train_positions,
            train_site,
            valid,
            geometry,
            folds,
            cached_weights,
            device,
        )
        pdp_output, pdp_record = _metric_fold(
            outer_fold,
            "pdp",
            pdp,
            args.components_pdp,
            query_ids,
            fold_id,
            train_positions,
            train_site,
            valid,
            geometry,
            folds,
            cached_weights,
            device,
        )
        held_rows = np.flatnonzero(fold_id == outer_fold)
        held_ids = query_ids[held_rows]
        pas_truth = np.asarray(pas[held_ids], dtype=np.float32)
        pdp_truth = np.asarray(pdp[held_ids], dtype=np.float32)
        pas_score = _grouped_cosine(pas_output, pas_truth)
        pdp_score = _grouped_cosine(pdp_output, pdp_truth)
        np.savez_compressed(
            ARTIFACT_DIR / f"fold{outer_fold}_mode_adaptive_targets.npz",
            query_ids=held_ids,
            pas=pas_output.astype(np.float16),
            pdp=pdp_output.astype(np.float16),
            pas_score=pas_score,
            pdp_score=pdp_score,
        )
        records.append(
            {
                "fold": int(outer_fold),
                "pas": pas_record,
                "pdp": pdp_record,
                "combined": float(np.mean(0.5 * (pas_score + pdp_score))),
                "seconds": float(time.time() - fold_started),
            }
        )
        REPORT.write_text(
            json.dumps(
                {
                    "experiment": VERSION,
                    "status": "running",
                    "official_inputs_only": True,
                    "sionna_used": False,
                    "primary_hypothesis": (
                        "Different latent PAS/PDP modes have different spatial covariance; "
                        "mode-wise covariance selection should improve complete-hole extrapolation "
                        "without replacing the V3/V4 spectral anchor."
                    ),
                    "parameters": vars(args),
                    "records": records,
                    "seconds": float(time.time() - started),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"fold={outer_fold} PAS={pas_record['score']:.6f} "
            f"PDP={pdp_record['score']:.6f} combined={records[-1]['combined']:.6f}",
            flush=True,
        )

    REPORT.write_text(
        json.dumps(
            {
                "experiment": VERSION,
                "status": "analysis_only",
                "official_inputs_only": True,
                "sionna_used": False,
                "primary_hypothesis": (
                    "Different latent PAS/PDP modes have different spatial covariance; "
                    "mode-wise covariance selection should improve complete-hole extrapolation "
                    "without replacing the V3/V4 spectral anchor."
                ),
                "parameters": vars(args),
                "records": records,
                "seconds": float(time.time() - started),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
