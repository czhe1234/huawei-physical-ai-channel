#!/usr/bin/env python3
"""Replace the E85 global angle residual with the strict E98 mapping."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

import analyze_e100_locked_energy_angle_correction as e100
import analyze_e86_surface_endpoint_v11_blend as e86
import optimize_e98_global_array_angle_mapping as e98
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E101-deduplicated-angle-residual"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "deduplicated_angle_residual"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_deduplicated_angle_residual"
REPORT = ANALYSIS_DIR / "deduplicated_angle_residual.json"
ENDPOINT = ARTIFACT_DIR / "strict_oof_deduplicated_angle_endpoint.npz"
RENDERED_PAS = ARTIFACT_DIR / "strict_oof_deduplicated_angle_pas_f16.npy"
RENDERED_PDP = ARTIFACT_DIR / "strict_oof_deduplicated_angle_pdp_f16.npy"
OOF_CHANNEL = ARTIFACT_DIR / "strict_oof_deduplicated_angle_v11_blend_channel.npy"
PERIODS = np.asarray((16.0, 8.0, 192.0), dtype=np.float32)
TINY = 1e-30


def build_mapping_delta(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    base_coordinate: np.ndarray,
    mapping_data: e98.MappingData,
    latent: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    corrected_base = np.array(base_coordinate, copy=True)
    for outer in range(5):
        rows = np.flatnonzero(fold_id == outer)
        global_rows = query_ids[rows]
        horizontal, vertical = mapping_data.candidate_angles(
            torch.as_tensor(global_rows, device=device),
            torch.as_tensor(latent[outer], device=device),
        )
        corrected_base[rows, :, 0] = torch.remainder(horizontal, 16.0).cpu().numpy()
        corrected_base[rows, :, 1] = torch.remainder(vertical, 8.0).cpu().numpy()

    mapping_delta = np.zeros_like(corrected_base)
    for axis, period in enumerate(PERIODS[:2]):
        mapping_delta[..., axis] = e100.periodic_delta(
            corrected_base[..., axis], base_coordinate[..., axis], float(period)
        )
    return corrected_base, mapping_delta


def weighted_projection(
    residual: np.ndarray,
    mapping_delta: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    dot = float(np.sum(weight * residual * mapping_delta, dtype=np.float64))
    residual_norm = float(np.sum(weight * residual * residual, dtype=np.float64))
    mapping_norm = float(
        np.sum(weight * mapping_delta * mapping_delta, dtype=np.float64)
    )
    raw = dot / max(mapping_norm, TINY)
    return {
        "raw_coefficient": float(raw),
        "coefficient": float(np.clip(raw, 0.0, 1.0)),
        "weighted_cosine": float(
            dot / max(math.sqrt(residual_norm * mapping_norm), TINY)
        ),
        "residual_rms_bins": float(math.sqrt(residual_norm / max(len(weight), 1))),
        "mapping_delta_rms_bins": float(
            math.sqrt(mapping_norm / max(len(weight), 1))
        ),
    }


def fit_cross_hole_coefficients(
    old_residual: np.ndarray,
    mapping_delta: np.ndarray,
    mass: np.ndarray,
    site: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    point_weight = mass / np.maximum(mass.sum(axis=1, keepdims=True), TINY)
    coefficients = np.zeros((5, 2, 2), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    for outer in range(5):
        for site_id in range(2):
            fit = (fold_id != outer) & (site == site_id)
            for axis, axis_name in ((0, "horizontal"), (1, "vertical")):
                metrics = weighted_projection(
                    old_residual[fit, :, axis],
                    mapping_delta[fit, :, axis],
                    point_weight[fit],
                )
                coefficients[outer, site_id, axis] = metrics["coefficient"]
                fold_rows.append(
                    {
                        "held_fold": outer,
                        "site": site_id,
                        "axis": axis_name,
                        "fit_points": int(np.sum(fit)),
                        **metrics,
                        "remaining_mapping_fraction": float(
                            1.0 - metrics["coefficient"]
                        ),
                    }
                )

    global_rows: list[dict[str, object]] = []
    for site_id in range(2):
        fit = site == site_id
        for axis, axis_name in ((0, "horizontal"), (1, "vertical")):
            metrics = weighted_projection(
                old_residual[fit, :, axis],
                mapping_delta[fit, :, axis],
                point_weight[fit],
            )
            global_rows.append(
                {
                    "site": site_id,
                    "axis": axis_name,
                    "points": int(np.sum(fit)),
                    **metrics,
                    "remaining_mapping_fraction": float(
                        1.0 - metrics["coefficient"]
                    ),
                }
            )
    return coefficients, fold_rows, global_rows


def apply_deduplicated_mapping(
    old_coordinate: np.ndarray,
    mapping_delta: np.ndarray,
    coefficients: np.ndarray,
    site: np.ndarray,
    fold_id: np.ndarray,
) -> np.ndarray:
    coordinate = np.array(old_coordinate, copy=True)
    for outer in range(5):
        for site_id in range(2):
            rows = (fold_id == outer) & (site == site_id)
            for axis, period in enumerate(PERIODS[:2]):
                remaining = 1.0 - float(coefficients[outer, site_id, axis])
                coordinate[rows, :, axis] = np.mod(
                    old_coordinate[rows, :, axis]
                    + remaining * mapping_delta[rows, :, axis],
                    period,
                )
    return coordinate


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    required = (
        e100.OLD_ENDPOINT,
        e100.OLD_RENDERED_PDP,
        e100.MAPPING_PARAMETERS,
        e100.V12_CHANNEL,
        e100.OOF_CHANNEL,
        e85.BASE_COORDINATE,
        e86.V11_CHANNEL,
        e86.TRAIN_CHANNEL,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    store = e85.DataStore()
    query_ids = store.strict_ids
    fold_id = store.strict_fold
    site = np.asarray(store.site[query_ids], dtype=np.int64)
    endpoint = np.load(e100.OLD_ENDPOINT, allow_pickle=False)
    if not np.array_equal(endpoint["query_ids"].astype(np.int64), query_ids):
        raise ValueError("old endpoint query order does not match canonical 500 points")
    mass = endpoint["mass"].astype(np.float32)
    old_coordinate = endpoint["coordinate"].astype(np.float32)
    port = endpoint["port"].astype(np.float32)
    base_coordinate_all = np.load(e85.BASE_COORDINATE, mmap_mode="r")
    base_coordinate = np.asarray(base_coordinate_all[query_ids], dtype=np.float32)
    mapping = np.load(e100.MAPPING_PARAMETERS, allow_pickle=False)
    latent = mapping["refit_latent"].astype(np.float32)
    mapping_data = e98.MappingData(device)

    corrected_base, mapping_delta = build_mapping_delta(
        query_ids,
        fold_id,
        base_coordinate,
        mapping_data,
        latent,
        device,
    )
    old_residual = np.zeros_like(old_coordinate)
    for axis, period in enumerate(PERIODS[:2]):
        old_residual[..., axis] = e100.periodic_delta(
            old_coordinate[..., axis], base_coordinate[..., axis], float(period)
        )
    old_residual[..., 2] = e100.periodic_delta(
        old_coordinate[..., 2], base_coordinate[..., 2], float(PERIODS[2])
    )
    coefficients, coefficient_rows, deployment_rows = fit_cross_hole_coefficients(
        old_residual, mapping_delta, mass, site, fold_id
    )
    coordinate = apply_deduplicated_mapping(
        old_coordinate, mapping_delta, coefficients, site, fold_id
    )
    if not np.isfinite(coordinate).all():
        raise FloatingPointError("deduplicated coordinates are not finite")
    delay_max_abs = float(
        np.max(
            np.abs(
                e100.periodic_delta(
                    coordinate[..., 2], old_coordinate[..., 2], float(PERIODS[2])
                )
            )
        )
    )

    pas, pdp = e100.endpoint_scores(
        store, query_ids, mass, coordinate, port, device
    )
    old_pas = endpoint["pas"].astype(np.float64)
    old_pdp = endpoint["pdp"].astype(np.float64)
    np.savez_compressed(
        ENDPOINT,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=site.astype(np.int8),
        mass=mass,
        coordinate=coordinate,
        port=port,
        pas=pas,
        pdp=pdp,
        mapping_delta=mapping_delta.astype(np.float32),
        old_residual=old_residual.astype(np.float32),
        cross_hole_coefficients=coefficients,
    )
    endpoint_pas_np, endpoint_pdp_np = e86.render_cache(
        str(ENDPOINT),
        str(RENDERED_PAS),
        str(RENDERED_PDP),
        device,
        batch_size=8,
        force=True,
    )
    old_rendered_pdp = np.load(e100.OLD_RENDERED_PDP, mmap_mode="r")
    pdp_lock_max_abs = float(
        np.max(
            np.abs(
                np.asarray(endpoint_pdp_np, dtype=np.float32)
                - np.asarray(old_rendered_pdp, dtype=np.float32)
            )
        )
    )

    base_np = np.load(e86.V11_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(
        np.load(e86.TRAIN_CHANNEL, mmap_mode="r")[query_ids], dtype=np.complex64
    )
    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    truth = torch.as_tensor(truth_np, device=device)
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    endpoint_pas = torch.as_tensor(
        np.asarray(endpoint_pas_np, dtype=np.float32), device=device
    )
    endpoint_pdp = torch.as_tensor(
        np.asarray(endpoint_pdp_np, dtype=np.float32), device=device
    )
    spec = e86.Candidate(
        endpoint="deduplicated_angle_residual",
        mode="sqrt",
        pas_beta=0.20,
        pdp_beta=0.20,
        order="pas_pdp",
        cycles=4,
    )
    with torch.inference_mode():
        prediction = e86.make_candidate(
            base, base_pas, base_pdp, endpoint_pas, endpoint_pdp, spec
        )
        selected_metrics = point_metrics(prediction, truth, truth_pas, truth_pdp)
        v11_metrics = point_metrics(base, truth, truth_pas, truth_pdp)
    np.save(OOF_CHANNEL, prediction.cpu().numpy().astype(np.complex64))
    v12 = torch.as_tensor(
        np.array(
            np.load(e100.V12_CHANNEL, mmap_mode="r"),
            dtype=np.complex64,
            copy=True,
        ),
        device=device,
    )
    e100_full = torch.as_tensor(
        np.array(np.load(e100.OOF_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    with torch.inference_mode():
        v12_metrics = point_metrics(v12, truth, truth_pas, truth_pdp)
        e100_metrics = point_metrics(e100_full, truth, truth_pas, truth_pdp)
    versus_v11 = e100.metric_comparison(selected_metrics, v11_metrics, fold_id)
    versus_v12 = e100.metric_comparison(selected_metrics, v12_metrics, fold_id)
    versus_e100 = e100.metric_comparison(selected_metrics, e100_metrics, fold_id)

    endpoint_old = e100.summarize_endpoint(old_pas, old_pdp, fold_id)
    endpoint_new = e100.summarize_endpoint(pas, pdp, fold_id)
    endpoint_gain = float(
        endpoint_new["pas_pdp_mean"] - endpoint_old["pas_pdp_mean"]
    )
    decision = {
        "minimum_endpoint_gain": 0.0,
        "minimum_score_gain_vs_v12": 0.0005,
        "minimum_positive_folds_vs_v12": 4,
        "minimum_worst_fold_delta_vs_v12": -0.001,
        "endpoint_gain": endpoint_gain,
        "score_gain_vs_v12": versus_v12["delta"],
        "positive_folds_vs_v12": versus_v12["positive_folds"],
        "worst_fold_delta_vs_v12": versus_v12["worst_fold_delta"],
    }
    decision["passed"] = bool(
        endpoint_gain >= decision["minimum_endpoint_gain"]
        and versus_v12["delta"] >= decision["minimum_score_gain_vs_v12"]
        and versus_v12["positive_folds"]
        >= decision["minimum_positive_folds_vs_v12"]
        and versus_v12["worst_fold_delta"]
        >= decision["minimum_worst_fold_delta_vs_v12"]
    )
    output = np.load(OOF_CHANNEL, mmap_mode="r")
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "Replacing the E85 residual component parallel to the E98 mapping with "
            "the corrected global base improves V12 without changing energy or delay."
        ),
        "validation": (
            "Canonical 500 contiguous holes. Each held hole uses site/axis projection "
            "coefficients estimated from the other four holes without channel labels."
        ),
        "action": {
            "formula": (
                "corrected_base + (old_residual - projection(old_residual, "
                "mapping_delta))"
            ),
            "coefficient_granularity": "held-fold x site x axis",
            "coefficient_source": "old E85 output and held-excluded E98 mapping only",
            "coefficient_clamp": [0.0, 1.0],
            "strength_scan": False,
            "mass_locked": True,
            "port_locked": True,
            "delay_locked": True,
            "delay_max_abs_difference_bins": delay_max_abs,
            "pdp_render_max_abs_difference": pdp_lock_max_abs,
        },
        "projection": {
            "cross_hole": coefficient_rows,
            "all_points_deployment_equivalent": deployment_rows,
            "cross_hole_coefficients": coefficients.tolist(),
        },
        "endpoint": {
            "old_e85": endpoint_old,
            "deduplicated": endpoint_new,
            "gain": endpoint_gain,
        },
        "fixed_policy": {
            **spec.__dict__,
            "selection": "pre-registered analytic de-duplication; no score scan",
        },
        "complete_score": {
            "versus_v11": versus_v11,
            "versus_v12": versus_v12,
            "versus_e100": versus_e100,
        },
        "decision": {**decision, "generate_v13": decision["passed"]},
        "outputs": {
            "endpoint": PATHS.relative(ENDPOINT),
            "rendered_pas": PATHS.relative(RENDERED_PAS),
            "rendered_pdp": PATHS.relative(RENDERED_PDP),
            "oof_channel": PATHS.relative(OOF_CHANNEL),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(
                np.isfinite(output.real).all() and np.isfinite(output.imag).all()
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
