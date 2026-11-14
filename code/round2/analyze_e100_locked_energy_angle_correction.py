#!/usr/bin/env python3
"""Apply the strict E98 H/V mapping delta to the frozen E85 path endpoint."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

import analyze_e86_surface_endpoint_v11_blend as e86
import optimize_e98_global_array_angle_mapping as e98
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E100-locked-energy-angle-correction"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "locked_energy_angle_correction"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_locked_energy_angle_correction"
REPORT = ANALYSIS_DIR / "locked_energy_angle_correction.json"
ENDPOINT = ARTIFACT_DIR / "strict_oof_locked_energy_corrected_endpoint.npz"
RENDERED_PAS = ARTIFACT_DIR / "strict_oof_locked_energy_rendered_pas_f16.npy"
RENDERED_PDP = ARTIFACT_DIR / "strict_oof_locked_energy_rendered_pdp_f16.npy"
OOF_CHANNEL = ARTIFACT_DIR / "strict_oof_locked_energy_v11_blend_channel.npy"
OLD_ENDPOINT = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "strict_oof_geometry_sequence_path_endpoint.npz"
)
OLD_RENDERED_PDP = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_geometry_sequence_rendered_pdp_f16.npy"
)
MAPPING_PARAMETERS = (
    PATHS.artifacts
    / "round2"
    / "corrected_array_surface_model"
    / "strict_mapping_parameters.npz"
)
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)
PERIODS = np.asarray((16.0, 8.0, 192.0), dtype=np.float32)
TINY = 1e-30


def periodic_delta(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    return (left - right + period / 2.0) % period - period / 2.0


def summarize_endpoint(
    pas: np.ndarray, pdp: np.ndarray, fold_id: np.ndarray
) -> dict[str, object]:
    return e85.summarize_prediction(
        pas,
        pdp,
        np.arange(len(pas), dtype=np.int64),
        fold_id,
    )


def metric_comparison(
    selected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, object]:
    selected_summary = statistics(*selected)
    baseline_summary = statistics(*baseline)
    folds = []
    for outer in range(5):
        mask = fold_id == outer
        local = statistics(*selected, mask)
        anchor = statistics(*baseline, mask)
        folds.append(
            {
                "fold": outer,
                "selected": local,
                "baseline": anchor,
                "delta": float(local["score"] - anchor["score"]),
            }
        )
    return {
        "selected": selected_summary,
        "baseline": baseline_summary,
        "delta": float(selected_summary["score"] - baseline_summary["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


@torch.inference_mode()
def endpoint_scores(
    store: e85.DataStore,
    query_ids: np.ndarray,
    mass: np.ndarray,
    coordinate: np.ndarray,
    port: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    pas = np.zeros(len(query_ids), dtype=np.float64)
    pdp = np.zeros(len(query_ids), dtype=np.float64)
    mean = np.zeros(store.base.shape[-1], dtype=np.float32)
    std = np.ones_like(mean)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        rows = query_ids[start:end]
        _, target_pas, target_pdp = store.batch(rows, mean, std, device)
        output = {
            "mass": torch.as_tensor(mass[start:end], device=device),
            "coordinate": torch.as_tensor(coordinate[start:end], device=device),
            "port": torch.as_tensor(port[start:end], device=device),
            "score_residual": torch.zeros(end - start, e85.PATHS_PER_POINT, device=device),
            "coordinate_residual": torch.zeros(
                end - start, e85.PATHS_PER_POINT, 3, device=device
            ),
            "port_residual": torch.zeros(
                end - start, e85.PATHS_PER_POINT, e85.MP * e85.N_UE, device=device
            ),
        }
        _, values = e85.objective(output, target_pas, target_pdp, e85.Config())
        pas[start:end] = values["pas"].cpu().numpy()
        pdp[start:end] = values["pdp"].cpu().numpy()
    return pas, pdp


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    required = (
        OLD_ENDPOINT,
        OLD_RENDERED_PDP,
        MAPPING_PARAMETERS,
        V12_CHANNEL,
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
    endpoint = np.load(OLD_ENDPOINT, allow_pickle=False)
    if not np.array_equal(endpoint["query_ids"].astype(np.int64), query_ids):
        raise ValueError("old endpoint query order does not match canonical 500 points")
    mass = endpoint["mass"].astype(np.float32)
    old_coordinate = endpoint["coordinate"].astype(np.float32)
    port = endpoint["port"].astype(np.float32)
    base_coordinate_all = np.load(e85.BASE_COORDINATE, mmap_mode="r")
    base_coordinate = np.asarray(base_coordinate_all[query_ids], dtype=np.float32)
    mapping = np.load(MAPPING_PARAMETERS, allow_pickle=False)
    latent = mapping["refit_latent"].astype(np.float32)
    mapping_data = e98.MappingData(device)

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
        mapping_delta[..., axis] = periodic_delta(
            corrected_base[..., axis], base_coordinate[..., axis], float(period)
        )
    corrected_coordinate = np.array(old_coordinate, copy=True)
    corrected_coordinate[..., :2] = np.mod(
        old_coordinate[..., :2] + mapping_delta[..., :2], PERIODS[:2]
    )
    if not np.isfinite(corrected_coordinate).all():
        raise FloatingPointError("corrected endpoint coordinates are not finite")

    pas, pdp = endpoint_scores(
        store, query_ids, mass, corrected_coordinate, port, device
    )
    old_pas = endpoint["pas"].astype(np.float64)
    old_pdp = endpoint["pdp"].astype(np.float64)
    np.savez_compressed(
        ENDPOINT,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        mass=mass,
        coordinate=corrected_coordinate,
        port=port,
        pas=pas,
        pdp=pdp,
        mapping_delta=mapping_delta.astype(np.float32),
    )
    endpoint_pas_np, endpoint_pdp_np = e86.render_cache(
        str(ENDPOINT),
        str(RENDERED_PAS),
        str(RENDERED_PDP),
        device,
        batch_size=8,
        force=True,
    )
    old_rendered_pdp = np.load(OLD_RENDERED_PDP, mmap_mode="r")
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
        endpoint="locked_energy_corrected_angle",
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
        np.array(np.load(V12_CHANNEL, mmap_mode="r"), dtype=np.complex64, copy=True),
        device=device,
    )
    with torch.inference_mode():
        v12_metrics = point_metrics(v12, truth, truth_pas, truth_pdp)
    versus_v11 = metric_comparison(selected_metrics, v11_metrics, fold_id)
    versus_v12 = metric_comparison(selected_metrics, v12_metrics, fold_id)

    weight = mass / np.maximum(mass.sum(axis=1, keepdims=True), TINY)
    old_residual = np.zeros_like(old_coordinate)
    for axis, period in enumerate(PERIODS[:2]):
        old_residual[..., axis] = periodic_delta(
            old_coordinate[..., axis], base_coordinate[..., axis], float(period)
        )
    overlap = {}
    for axis, name in ((0, "horizontal"), (1, "vertical")):
        left = old_residual[..., axis]
        right = mapping_delta[..., axis]
        dot = float(np.sum(weight * left * right, dtype=np.float64))
        left_norm = float(np.sum(weight * left * left, dtype=np.float64))
        right_norm = float(np.sum(weight * right * right, dtype=np.float64))
        overlap[name] = {
            "weighted_cosine": float(
                dot / max(math.sqrt(left_norm * right_norm), TINY)
            ),
            "old_residual_rms_bins": float(math.sqrt(left_norm / len(weight))),
            "mapping_delta_rms_bins": float(math.sqrt(right_norm / len(weight))),
            "projection_coefficient_old_on_mapping": float(
                dot / max(right_norm, TINY)
            ),
        }

    endpoint_old = summarize_endpoint(old_pas, old_pdp, fold_id)
    endpoint_new = summarize_endpoint(pas, pdp, fold_id)
    endpoint_gain = float(
        endpoint_new["pas_pdp_mean"] - endpoint_old["pas_pdp_mean"]
    )
    decision = {
        "minimum_endpoint_gain": 0.001,
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
            "The E98 H/V mapping delta improves V12 when the E85 path mass, port "
            "allocation, delay coordinates, and learned local residuals are frozen."
        ),
        "validation": (
            "Canonical 500-point contiguous holes using each held hole's E99 refit "
            "mapping, which excludes that hole's official labels."
        ),
        "action": {
            "mapping_strength": 1.0,
            "changed_axes": ["horizontal", "vertical"],
            "mass_locked": True,
            "port_locked": True,
            "delay_locked": True,
            "old_coordinate_residual_locked": True,
            "pdp_render_max_abs_difference": pdp_lock_max_abs,
        },
        "endpoint": {
            "old_e85": endpoint_old,
            "corrected": endpoint_new,
            "gain": endpoint_gain,
        },
        "residual_overlap": overlap,
        "fixed_policy": {
            **spec.__dict__,
            "selection": "pre-registered full mapping delta; no strength scan",
        },
        "complete_score": {
            "versus_v11": versus_v11,
            "versus_v12": versus_v12,
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
