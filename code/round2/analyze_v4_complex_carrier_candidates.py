#!/usr/bin/env python3
"""Test NMSE-oriented complex carriers under the frozen V3 spectral targets."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import torch

from complex_baselines import BaselineConfig, predict_one, query_site_neighbors
from data import load_positions
from evaluate_v3_anchored_projection import target_batch
from official_metrics import PointMetricStatistics, aggregate_statistics, point_metric_statistics
from pre_sionna_spectral import project_channel
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, OOF_BASE, OOF_META, optimal_scale, scaled_error


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_structured_complex_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_structured_complex_field"
REPORT = ANALYSIS_DIR / "complex_carrier_candidates.json"
ALTERNATIVE_BASE = ARTIFACT_DIR / "oof_nmse_oriented_complex_carrier.npy"
ALTERNATIVE_PROJECTED = ARTIFACT_DIR / "oof_nmse_oriented_v3_projection.npy"

FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
FOUNDATION_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_foundation_benchmark"
    / "round2_foundation_benchmark.json"
)
V3_METRICS = (
    PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / "oof_projection_metrics.npz"
)
V3_AD = (
    PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / "oof_ad_pas_f16.npy"
)
V3_PAS = (
    PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / "oof_pas_neural_f16.npy"
)
V3_PDP = (
    PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / "oof_pdp_neural_f16.npy"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"


def carrier_configs() -> dict[int, BaselineConfig]:
    # Existing V3 targets recover PAS/PDP after carrier construction, so the carrier
    # is chosen for complex coherence instead of its standalone spectral score.
    return {
        0: BaselineConfig(
            name="affine_k24_steer_axis0_v8pi",
            method="radial_affine",
            k=24,
            ridge=0.30,
            steering_alpha=1.0,
            horizontal_axis_deg=0.0,
            vertical_scale=8.0 * np.pi,
        ),
        1: BaselineConfig(
            name="radial_idw_k24",
            method="radial_idw",
            k=24,
        ),
    }


def valid_memmap(path, shape: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return value.shape == shape and value.dtype == np.complex64


def build_alternative_base(batch_message: int = 10) -> dict[str, object]:
    setup, train_positions, _, train_site, _ = load_positions()
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    output_shape = (len(query_ids),) + tuple(train_channel.shape[1:])
    if valid_memmap(ALTERNATIVE_BASE, output_shape):
        return {"reused": True, "shape": list(output_shape)}

    report = json.loads(FOUNDATION_REPORT.read_text(encoding="utf-8"))
    carrier_by_fold = report["carrier_fit_only_by_fold"]
    folds = np.load(FOLD_CACHE)
    configs = carrier_configs()
    row_for_query = np.full(len(train_positions), -1, dtype=np.int64)
    row_for_query[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    output = np.lib.format.open_memmap(
        ALTERNATIVE_BASE,
        mode="w+",
        dtype=np.complex64,
        shape=output_shape,
    )

    for fold in np.unique(fold_id):
        fit_ids = folds[f"fold{int(fold)}_fit"].astype(np.int64)
        held_ids = query_ids[fold_id == fold]
        distances, neighbor_ids = query_site_neighbors(
            train_positions,
            train_site,
            fit_ids,
            held_ids,
            max_k=max(config.k for config in configs.values()),
        )
        for local_row, query_id in enumerate(held_ids):
            site = int(train_site[query_id])
            carrier = carrier_by_fold[str(int(fold))][str(site)]
            config = replace(
                configs[site],
                beta=float(carrier["beta"]),
                gamma=float(carrier["gamma"]),
            )
            ids = neighbor_ids[local_row, : config.k]
            prediction, _ = predict_one(
                np.asarray(train_channel[ids], dtype=np.complex64),
                train_positions[ids],
                train_positions[query_id],
                distances[local_row, : config.k],
                setup.bs_positions[site],
                config,
            )
            output[row_for_query[query_id]] = prediction
            if (local_row + 1) % batch_message == 0 or local_row + 1 == len(held_ids):
                print(
                    f"alternative carrier fold={int(fold)} "
                    f"queries={local_row + 1}/{len(held_ids)}",
                    flush=True,
                )
    output.flush()
    return {"reused": False, "shape": list(output_shape)}


def aggregate(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    return aggregate_statistics(
        PointMetricStatistics(pas, pdp, error, energy),
        mask,
    )


def project_and_score(
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids]
    metrics = np.load(V3_METRICS)
    selected_action = metrics["selected_action"].astype(np.int16)

    base = np.load(ALTERNATIVE_BASE, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    ad_cache = np.load(V3_AD, mmap_mode="r")
    pas_cache = np.load(V3_PAS, mmap_mode="r")
    pdp_cache = np.load(V3_PDP, mmap_mode="r")
    output_shape = tuple(base.shape)
    output = np.lib.format.open_memmap(
        ALTERNATIVE_PROJECTED,
        mode="w+",
        dtype=np.complex64,
        shape=output_shape,
    )
    count = len(query_ids)
    pas_values = np.empty(count, dtype=np.float64)
    pdp_values = np.empty(count, dtype=np.float64)
    predicted_energy = np.empty(count, dtype=np.float64)
    truth_energy = np.empty(count, dtype=np.float64)
    cross = np.empty(count, dtype=np.float64)

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        ad, pas, pdp = target_batch(
            ad_cache, pas_cache, pdp_cache, start, end, device
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        prediction = np.empty_like(np.asarray(base[start:end]))
        local_actions = selected_action[start:end]
        for action_index in np.unique(local_actions):
            rows = np.flatnonzero(local_actions == action_index)
            action = ACTIONS[int(action_index)]
            with torch.inference_mode():
                projected = project_channel(
                    base_tensor[rows],
                    ad[rows],
                    pas[rows],
                    pdp[rows],
                    action.alpha_ad,
                    action.alpha_pas,
                    action.alpha_pdp,
                )
            prediction[rows] = projected.cpu().numpy()
        output[start:end] = prediction.astype(np.complex64)
        truth = np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64)
        point = point_metric_statistics(prediction, truth)
        pas_values[start:end] = point.pas
        pdp_values[start:end] = point.pdp
        truth_energy[start:end] = point.truth_energy
        predicted_energy[start:end] = np.sum(
            np.abs(prediction) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        cross[start:end] = np.real(
            np.sum(
                prediction * np.conj(truth),
                axis=(1, 2, 3),
                dtype=np.complex128,
            )
        )
        print(f"alternative V3 projection {end}/{count}", flush=True)
    output.flush()

    selected_scale = np.empty(count, dtype=np.float32)
    nested_error = np.empty(count, dtype=np.float64)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            fit = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            scale = optimal_scale(predicted_energy, cross, fit)
            selected_scale[held] = scale
            nested_error[held] = scaled_error(
                predicted_energy[held],
                truth_energy[held],
                cross[held],
                scale,
            )
            decisions.append(
                {
                    "held_fold": int(held_fold),
                    "site": int(site_value),
                    "scale": float(scale),
                    "held_count": int(np.sum(held)),
                }
            )

    overall = aggregate(pas_values, pdp_values, nested_error, truth_energy)
    fold_scores = [
        {
            "fold": int(fold),
            **aggregate(
                pas_values,
                pdp_values,
                nested_error,
                truth_energy,
                fold_id == fold,
            ),
        }
        for fold in np.unique(fold_id)
    ]
    site_scores = {
        str(int(site_value)): aggregate(
            pas_values,
            pdp_values,
            nested_error,
            truth_energy,
            site == site_value,
        )
        for site_value in np.unique(site)
    }
    np.savez_compressed(
        ARTIFACT_DIR / "oof_nmse_oriented_projection_metrics.npz",
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=site,
        selected_action=selected_action,
        selected_scale=selected_scale,
        pas=pas_values,
        pdp=pdp_values,
        error=nested_error,
        truth_energy=truth_energy,
        predicted_energy=predicted_energy,
        cross=cross,
    )
    return {
        "overall": overall,
        "folds": fold_scores,
        "sites": site_scores,
        "nested_scale_decisions": decisions,
        "projected_output": PATHS.relative(ALTERNATIVE_PROJECTED),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    base = build_alternative_base()
    result = project_and_score(torch.device(args.device), args.batch_size)
    payload = {
        "experiment": "Round2-E6-V4-complex-carrier-audit",
        "status": "analysis_only",
        "primary_hypothesis": (
            "The V3 spectral targets should be projected on a carrier selected for "
            "complex coherence/NMSE rather than standalone PAS/PDP score."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "carrier_configs": {
            str(site): config.to_dict() for site, config in carrier_configs().items()
        },
        "base_generation": base,
        "v3_reference": {
            "pas": 0.5811764771365105,
            "pdp": 0.7490299601831307,
            "nmse": 0.9526311808008533,
            "score": 0.6345084718235374,
        },
        "result": result,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
