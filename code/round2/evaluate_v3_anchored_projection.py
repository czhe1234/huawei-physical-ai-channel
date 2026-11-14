#!/usr/bin/env python3
"""Evaluate the frozen V3 neural spectral targets in the full channel pipeline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from data import load_positions
from official_metrics import PointMetricStatistics, aggregate_statistics, point_metric_statistics
from pre_sionna_spectral import project_channel
from project_paths import PATHS
from run_rbf_spectral_field import (
    ACTIONS,
    OOF_BASE,
    OOF_META,
    optimal_scale,
    scaled_error,
    statistics,
)


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_anchored_neural_rbf"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_anchored_neural_rbf"
REPORT = ANALYSIS_DIR / "anchored_projection_scores.json"
METRICS = ARTIFACT_DIR / "oof_projection_metrics.npz"
SELECTION = ARTIFACT_DIR / "projection_selection.json"

TARGET_AD = ARTIFACT_DIR / "oof_ad_pas_f16.npy"
TARGET_PAS = ARTIFACT_DIR / "oof_pas_neural_f16.npy"
TARGET_PDP = ARTIFACT_DIR / "oof_pdp_neural_f16.npy"

PAS_GLOBAL_SHRINK = 0.10
PDP_GLOBAL_SHRINK = 0.55


def normalize(value: torch.Tensor, dim: int) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dim, keepdim=True)
    return torch.where(total > 1e-20, value / total.clamp_min(1e-20), torch.zeros_like(value))


def amplitude_blend(
    base: torch.Tensor,
    shared: torch.Tensor,
    alpha: float,
    dim: int,
) -> torch.Tensor:
    amplitude = (1.0 - alpha) * torch.sqrt(base.clamp_min(0.0))
    amplitude = amplitude + alpha * torch.sqrt(shared.clamp_min(0.0))
    return normalize(amplitude.square(), dim)


def target_batch(
    ad_cache: np.ndarray,
    pas_cache: np.ndarray,
    pdp_cache: np.ndarray,
    start: int,
    end: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ad = torch.as_tensor(np.asarray(ad_cache[start:end], dtype=np.float32), device=device)
    pas = torch.as_tensor(np.asarray(pas_cache[start:end], dtype=np.float32), device=device)
    pdp = torch.as_tensor(np.asarray(pdp_cache[start:end], dtype=np.float32), device=device)

    shared_pas = pas.mean(dim=(2, 3), keepdim=True).expand_as(pas)
    shared_pdp = pdp.mean(dim=(1, 2), keepdim=True).expand_as(pdp)
    pas = amplitude_blend(pas, shared_pas, PAS_GLOBAL_SHRINK, 1)
    pdp = amplitude_blend(pdp, shared_pdp, PDP_GLOBAL_SHRINK, 3)
    return ad, pas, pdp


def evaluate(device: torch.device, batch_size: int) -> dict[str, object]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]

    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    base = np.load(OOF_BASE, mmap_mode="r")
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")

    count = len(query_ids)
    shape = (len(ACTIONS), count)
    pas_values = np.empty(shape, dtype=np.float64)
    pdp_values = np.empty(shape, dtype=np.float64)
    error_values = np.empty(shape, dtype=np.float64)
    truth_energy_values = np.empty(shape, dtype=np.float64)
    predicted_energy = np.empty(shape, dtype=np.float64)
    cross = np.empty(shape, dtype=np.float64)

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        ad, pas, pdp = target_batch(
            ad_cache, pas_cache, pdp_cache, start, end, device
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True)
        for action_index, action in enumerate(ACTIONS):
            with torch.inference_mode():
                prediction = project_channel(
                    base_tensor,
                    ad,
                    pas,
                    pdp,
                    action.alpha_ad,
                    action.alpha_pas,
                    action.alpha_pdp,
                ).cpu().numpy()
            metric = point_metric_statistics(prediction, truth)
            pas_values[action_index, start:end] = metric.pas
            pdp_values[action_index, start:end] = metric.pdp
            error_values[action_index, start:end] = metric.squared_error
            truth_energy_values[action_index, start:end] = metric.truth_energy
            predicted_energy[action_index, start:end] = np.sum(
                np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            cross[action_index, start:end] = np.real(
                np.sum(
                    prediction * np.conj(truth),
                    axis=(1, 2, 3),
                    dtype=np.complex128,
                )
            )
        print(f"V3 projection OOF {end}/{count}", flush=True)

    selected_action = np.empty(count, dtype=np.int16)
    selected_scale = np.empty(count, dtype=np.float32)
    nested_error = np.empty(count, dtype=np.float64)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(query_site):
            train = (fold_id != held_fold) & (query_site == site_value)
            held = (fold_id == held_fold) & (query_site == site_value)
            candidates: list[tuple[float, int, float]] = []
            for action_index in range(len(ACTIONS)):
                scale = optimal_scale(
                    predicted_energy[action_index], cross[action_index], train
                )
                candidate_error = scaled_error(
                    predicted_energy[action_index],
                    truth_energy_values[action_index],
                    cross[action_index],
                    scale,
                )
                score = float(
                    statistics(
                        pas_values[action_index],
                        pdp_values[action_index],
                        candidate_error,
                        truth_energy_values[action_index],
                        train,
                    )["score"]
                )
                candidates.append((score, action_index, scale))
            train_score, action_index, scale = max(candidates)
            selected_action[held] = action_index
            selected_scale[held] = scale
            nested_error[held] = scaled_error(
                predicted_energy[action_index, held],
                truth_energy_values[action_index, held],
                cross[action_index, held],
                scale,
            )
            decisions.append(
                {
                    "held_fold": int(held_fold),
                    "site": int(site_value),
                    "action": ACTIONS[action_index].name,
                    "scale": float(scale),
                    "training_score": train_score,
                    "held_count": int(np.sum(held)),
                }
            )

    selected_pas = np.take_along_axis(
        pas_values, selected_action[None, :], axis=0
    )[0]
    selected_pdp = np.take_along_axis(
        pdp_values, selected_action[None, :], axis=0
    )[0]
    selected_truth_energy = np.take_along_axis(
        truth_energy_values, selected_action[None, :], axis=0
    )[0]
    nested = statistics(
        selected_pas, selected_pdp, nested_error, selected_truth_energy
    )
    fold_scores = []
    for held_fold in np.unique(fold_id):
        fold_scores.append(
            {
                "fold": int(held_fold),
                **statistics(
                    selected_pas,
                    selected_pdp,
                    nested_error,
                    selected_truth_energy,
                    fold_id == held_fold,
                ),
            }
        )

    final_policy: list[dict[str, object]] = []
    for site_value in np.unique(query_site):
        mask = query_site == site_value
        candidates = []
        for action_index in range(len(ACTIONS)):
            scale = optimal_scale(
                predicted_energy[action_index], cross[action_index], mask
            )
            candidate_error = scaled_error(
                predicted_energy[action_index],
                truth_energy_values[action_index],
                cross[action_index],
                scale,
            )
            score = float(
                statistics(
                    pas_values[action_index],
                    pdp_values[action_index],
                    candidate_error,
                    truth_energy_values[action_index],
                    mask,
                )["score"]
            )
            candidates.append((score, action_index, scale))
        score, action_index, scale = max(candidates)
        final_policy.append(
            {
                "site": int(site_value),
                "action_index": int(action_index),
                "action": ACTIONS[action_index].name,
                "scale": float(scale),
                "oof_site_score": score,
            }
        )

    raw_actions = []
    calibrated_actions = []
    for action_index, action in enumerate(ACTIONS):
        raw_actions.append(
            {
                "action": asdict(action),
                "all": statistics(
                    pas_values[action_index],
                    pdp_values[action_index],
                    error_values[action_index],
                    truth_energy_values[action_index],
                ),
            }
        )
        calibrated_error = error_values[action_index].copy()
        scale_by_site: dict[str, float] = {}
        for site_value in np.unique(query_site):
            mask = query_site == site_value
            scale = optimal_scale(
                predicted_energy[action_index], cross[action_index], mask
            )
            scale_by_site[str(int(site_value))] = scale
            calibrated_error[mask] = scaled_error(
                predicted_energy[action_index, mask],
                truth_energy_values[action_index, mask],
                cross[action_index, mask],
                scale,
            )
        calibrated_actions.append(
            {
                "action": asdict(action),
                "scale_by_site": scale_by_site,
                "all": statistics(
                    pas_values[action_index],
                    pdp_values[action_index],
                    calibrated_error,
                    truth_energy_values[action_index],
                ),
            }
        )

    np.savez_compressed(
        METRICS,
        action_names=np.asarray([action.name for action in ACTIONS]),
        pas=pas_values,
        pdp=pdp_values,
        error=error_values,
        truth_energy=truth_energy_values,
        predicted_energy=predicted_energy,
        cross=cross,
        query_ids=query_ids,
        fold_id=fold_id,
        site=query_site,
        selected_action=selected_action,
        selected_scale=selected_scale,
    )
    payload = {
        "target_policy": {
            "ad_weights": "pas",
            "pas_global_shrink": PAS_GLOBAL_SHRINK,
            "pdp_global_shrink": PDP_GLOBAL_SHRINK,
        },
        "raw_actions": raw_actions,
        "calibrated_actions": calibrated_actions,
        "nested_selected": nested,
        "nested_fold_scores": fold_scores,
        "nested_decisions": decisions,
        "final_policy": final_policy,
    }
    SELECTION.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "Round2-E4-V3-anchored-neural-projection",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": evaluate(torch.device(args.device), args.batch_size),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
