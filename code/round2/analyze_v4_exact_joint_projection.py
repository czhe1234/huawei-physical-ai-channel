#!/usr/bin/env python3
"""Recover the V3 spectral target gap with exact alternating projections."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from data import load_positions
from evaluate_v3_anchored_projection import (
    METRICS as V3_METRICS,
    TARGET_AD,
    TARGET_PAS,
    TARGET_PDP,
    target_batch,
)
from pre_sionna_spectral import project_angle_delay
from project_paths import PATHS
from run_rbf_spectral_field import (
    ACTIONS,
    OOF_BASE,
    OOF_META,
    optimal_scale,
    scaled_error,
    statistics,
)


VERSION = "Round2-E21-exact-multidomain-projection"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_exact_joint_projection"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_exact_joint_projection"
METRICS = ARTIFACT_DIR / "exact_joint_projection_metrics.npz"
REPORT = ANALYSIS_DIR / "exact_joint_projection.json"

CHECKPOINTS = (1, 2, 4, 8, 16)
ORDERS = ("pas_pdp", "pdp_pas")
TRIPLE_ORDERS = ("ad_pas_pdp", "ad_pdp_pas", "pas_ad_pdp", "pdp_ad_pas")
TINY = 1e-30


def normalized(
    value: torch.Tensor,
    dims: tuple[int, ...],
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    result = value / total.clamp_min(TINY)
    if fallback is not None:
        result = torch.where(total > TINY, result, fallback)
    return result


def preserve_point_power(
    channel: torch.Tensor, reference_power: torch.Tensor
) -> torch.Tensor:
    current = channel.abs().square().sum(dim=(1, 2, 3))
    scale = torch.sqrt(reference_power / current.clamp_min(TINY))
    return channel * scale[:, None, None, None]


def pas_distribution(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    spectrum = torch.fft.fft2(shaped, dim=(2, 3))
    power = spectrum.abs().square()
    return normalized(power, (1, 2, 3))


def pdp_distribution(channel: torch.Tensor) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1)
    return normalized(delay.abs().square(), (3,))


def ad_distribution(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    transformed = torch.fft.ifft(
        torch.fft.fft2(shaped, dim=(2, 3)), dim=-1
    )
    power = transformed.abs().square().sum(dim=(1, 4))
    return normalized(power, (1, 2, 3))


def fixed_blend(
    current: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    dims: tuple[int, ...],
) -> torch.Tensor:
    target = normalized(target, dims, fallback=current)
    return normalized((1.0 - alpha) * current + alpha * target, dims)


def exact_pas_projection(
    channel: torch.Tensor, desired: torch.Tensor
) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    spectrum = torch.fft.fft2(shaped, dim=(2, 3))
    magnitude = spectrum.abs()
    slice_energy = magnitude.square().sum(dim=(1, 2, 3), keepdim=True)
    desired_magnitude = torch.sqrt((desired * slice_energy).clamp_min(0.0))
    unit = spectrum / magnitude.clamp_min(TINY)
    unit = torch.where(
        magnitude > TINY,
        unit,
        torch.ones((), dtype=spectrum.dtype, device=spectrum.device),
    )
    restored = torch.fft.ifft2(desired_magnitude * unit, dim=(2, 3))
    return restored.reshape_as(channel)


def exact_pdp_projection(
    channel: torch.Tensor, desired: torch.Tensor
) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1)
    magnitude = delay.abs()
    link_energy = magnitude.square().sum(dim=3, keepdim=True)
    desired_magnitude = torch.sqrt((desired * link_energy).clamp_min(0.0))
    unit = delay / magnitude.clamp_min(TINY)
    unit = torch.where(
        magnitude > TINY,
        unit,
        torch.ones((), dtype=delay.dtype, device=delay.device),
    )
    return torch.fft.fft(desired_magnitude * unit, dim=-1)


def exact_ad_projection(
    channel: torch.Tensor, desired: torch.Tensor
) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    transformed = torch.fft.ifft(
        torch.fft.fft2(shaped, dim=(2, 3)), dim=-1
    )
    power = transformed.abs().square()
    collapsed = power.sum(dim=(1, 4))
    total = collapsed.sum(dim=(1, 2, 3), keepdim=True)
    gain = torch.sqrt(
        (desired * total).clamp_min(0.0) / collapsed.clamp_min(TINY)
    )
    transformed = transformed * gain[:, None, :, :, None, :]
    restored = torch.fft.ifft2(
        torch.fft.fft(transformed, dim=-1), dim=(2, 3)
    )
    return restored.reshape_as(channel)


def initialize_trajectory(
    base: torch.Tensor,
    ad_target: torch.Tensor,
    pas_target: torch.Tensor,
    pdp_target: torch.Tensor,
    action_index: int,
    order: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action = ACTIONS[action_index]
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    candidate = project_angle_delay(base, ad_target, action.alpha_ad)
    target_pas = pas_target.reshape(len(base), 2, 16, 8, 4, 192)

    if order == "pas_pdp":
        desired_pas = fixed_blend(
            pas_distribution(candidate), target_pas, action.alpha_pas, (1, 2, 3)
        )
        candidate = exact_pas_projection(candidate, desired_pas)
        desired_pdp = fixed_blend(
            pdp_distribution(candidate), pdp_target, action.alpha_pdp, (3,)
        )
        candidate = exact_pdp_projection(candidate, desired_pdp)
    elif order == "pdp_pas":
        desired_pdp = fixed_blend(
            pdp_distribution(candidate), pdp_target, action.alpha_pdp, (3,)
        )
        candidate = exact_pdp_projection(candidate, desired_pdp)
        desired_pas = fixed_blend(
            pas_distribution(candidate), target_pas, action.alpha_pas, (1, 2, 3)
        )
        candidate = exact_pas_projection(candidate, desired_pas)
    else:
        raise ValueError(order)

    return (
        preserve_point_power(candidate, reference_power),
        desired_pas,
        desired_pdp,
        reference_power,
    )


def continue_trajectory(
    candidate: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
    reference_power: torch.Tensor,
    order: str,
) -> torch.Tensor:
    if order == "pas_pdp":
        candidate = exact_pas_projection(candidate, desired_pas)
        candidate = exact_pdp_projection(candidate, desired_pdp)
    elif order == "pdp_pas":
        candidate = exact_pdp_projection(candidate, desired_pdp)
        candidate = exact_pas_projection(candidate, desired_pas)
    else:
        raise ValueError(order)
    return preserve_point_power(candidate, reference_power)


def apply_domain(
    candidate: torch.Tensor,
    domain: str,
    desired_ad: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
) -> torch.Tensor:
    if domain == "ad":
        return exact_ad_projection(candidate, desired_ad)
    if domain == "pas":
        return exact_pas_projection(candidate, desired_pas)
    if domain == "pdp":
        return exact_pdp_projection(candidate, desired_pdp)
    raise ValueError(domain)


def initialize_triple_trajectory(
    base: torch.Tensor,
    ad_target: torch.Tensor,
    pas_target: torch.Tensor,
    pdp_target: torch.Tensor,
    action_index: int,
    order: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action = ACTIONS[action_index]
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    candidate = base
    target_pas = pas_target.reshape(len(base), 2, 16, 8, 4, 192)
    desired_ad = ad_distribution(candidate)
    desired_pas = pas_distribution(candidate)
    desired_pdp = pdp_distribution(candidate)
    for domain in order.split("_"):
        if domain == "ad":
            desired_ad = fixed_blend(
                ad_distribution(candidate), ad_target, action.alpha_ad, (1, 2, 3)
            )
        elif domain == "pas":
            desired_pas = fixed_blend(
                pas_distribution(candidate), target_pas, action.alpha_pas, (1, 2, 3)
            )
        elif domain == "pdp":
            desired_pdp = fixed_blend(
                pdp_distribution(candidate), pdp_target, action.alpha_pdp, (3,)
            )
        candidate = apply_domain(
            candidate, domain, desired_ad, desired_pas, desired_pdp
        )
    return (
        preserve_point_power(candidate, reference_power),
        desired_ad,
        desired_pas,
        desired_pdp,
        reference_power,
    )


def continue_triple_trajectory(
    candidate: torch.Tensor,
    desired_ad: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
    reference_power: torch.Tensor,
    order: str,
) -> torch.Tensor:
    for domain in order.split("_"):
        candidate = apply_domain(
            candidate, domain, desired_ad, desired_pas, desired_pdp
        )
    return preserve_point_power(candidate, reference_power)


def cosine(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    dim: int | tuple[int, ...],
) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=dim, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=dim, dtype=torch.float64)
        * truth.square().sum(dim=dim, dtype=torch.float64)
    )
    denominator = torch.where(
        denominator == 0.0,
        torch.full_like(denominator, 1e-12),
        denominator,
    )
    return numerator / denominator


def truth_views(truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shaped = truth.reshape(len(truth), 2, 16, 8, 4, 192)
    pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
    return pas, pdp


def metric_components(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
) -> dict[str, np.ndarray]:
    shaped = prediction.reshape(len(prediction), 2, 16, 8, 4, 192)
    prediction_pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    prediction_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    pas = cosine(prediction_pas, truth_pas, (1, 2, 3)).mean(
        dim=(1, 2), dtype=torch.float64
    )
    pdp = cosine(prediction_pdp, truth_pdp, 3).mean(
        dim=(1, 2), dtype=torch.float64
    )
    pas_target = cosine(prediction_pas, desired_pas, (1, 2, 3)).mean(
        dim=(1, 2), dtype=torch.float64
    )
    pdp_target = cosine(prediction_pdp, desired_pdp, 3).mean(
        dim=(1, 2), dtype=torch.float64
    )
    squared_error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    truth_energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    predicted_energy = prediction.abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    cross = (prediction * truth.conj()).sum(
        dim=(1, 2, 3), dtype=torch.complex128
    ).real
    return {
        "pas": pas.cpu().numpy(),
        "pdp": pdp.cpu().numpy(),
        "pas_target": pas_target.cpu().numpy(),
        "pdp_target": pdp_target.cpu().numpy(),
        "error": squared_error.cpu().numpy(),
        "truth_energy": truth_energy.cpu().numpy(),
        "predicted_energy": predicted_energy.cpu().numpy(),
        "cross": cross.cpu().numpy(),
    }


def new_candidate_names(checkpoints: tuple[int, ...]) -> list[str]:
    dual = [
        f"exact_{order}_{action.name}_c{cycle:02d}"
        for action in ACTIONS
        for order in ORDERS
        for cycle in checkpoints
    ]
    triple = [
        f"exact_{order}_{action.name}_c{cycle:02d}"
        for action in ACTIONS
        if action.alpha_ad > 0.0
        for order in TRIPLE_ORDERS
        for cycle in checkpoints
    ]
    return dual + triple


def generate_metrics(
    device: torch.device,
    batch_size: int,
    checkpoints: tuple[int, ...],
) -> dict[str, np.ndarray]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    base = np.load(OOF_BASE, mmap_mode="r")
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    names = new_candidate_names(checkpoints)
    output = {
        key: np.empty((len(names), len(query_ids)), dtype=np.float64)
        for key in (
            "pas",
            "pdp",
            "pas_target",
            "pdp_target",
            "error",
            "truth_energy",
            "predicted_energy",
            "cross",
        )
    }
    name_to_index = {name: index for index, name in enumerate(names)}
    max_cycles = max(checkpoints)

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        ad, pas, pdp = target_batch(
            ad_cache, pas_cache, pdp_cache, start, end, device
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        with torch.inference_mode():
            truth_pas, truth_pdp = truth_views(truth)
            for action_index, action in enumerate(ACTIONS):
                for order in ORDERS:
                    candidate, desired_pas, desired_pdp, reference_power = (
                        initialize_trajectory(
                            base_tensor,
                            ad,
                            pas,
                            pdp,
                            action_index,
                            order,
                        )
                    )
                    for cycle in range(1, max_cycles + 1):
                        if cycle > 1:
                            candidate = continue_trajectory(
                                candidate,
                                desired_pas,
                                desired_pdp,
                                reference_power,
                                order,
                            )
                        if cycle not in checkpoints:
                            continue
                        name = f"exact_{order}_{action.name}_c{cycle:02d}"
                        candidate_index = name_to_index[name]
                        values = metric_components(
                            candidate,
                            truth,
                            truth_pas,
                            truth_pdp,
                            desired_pas,
                            desired_pdp,
                        )
                        for key, value in values.items():
                            output[key][candidate_index, start:end] = value
                if action.alpha_ad <= 0.0:
                    continue
                for order in TRIPLE_ORDERS:
                    (
                        candidate,
                        desired_ad,
                        desired_pas,
                        desired_pdp,
                        reference_power,
                    ) = initialize_triple_trajectory(
                        base_tensor,
                        ad,
                        pas,
                        pdp,
                        action_index,
                        order,
                    )
                    for cycle in range(1, max_cycles + 1):
                        if cycle > 1:
                            candidate = continue_triple_trajectory(
                                candidate,
                                desired_ad,
                                desired_pas,
                                desired_pdp,
                                reference_power,
                                order,
                            )
                        if cycle not in checkpoints:
                            continue
                        name = f"exact_{order}_{action.name}_c{cycle:02d}"
                        candidate_index = name_to_index[name]
                        values = metric_components(
                            candidate,
                            truth,
                            truth_pas,
                            truth_pdp,
                            desired_pas,
                            desired_pdp,
                        )
                        for key, value in values.items():
                            output[key][candidate_index, start:end] = value
        print(f"exact joint projection {end}/{len(query_ids)}", flush=True)

    output["candidate_names"] = np.asarray(names)
    output["query_ids"] = query_ids
    output["fold_id"] = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    output["site"] = train_site[query_ids].astype(np.int8)
    return output


def combine_with_v3(
    generated: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    legacy = np.load(V3_METRICS)
    if not np.array_equal(legacy["query_ids"], generated["query_ids"]):
        raise RuntimeError("V3 and exact-projection query order differs")
    combined: dict[str, np.ndarray] = {
        "candidate_names": np.concatenate(
            [
                np.asarray([f"legacy_{name}" for name in legacy["action_names"]]),
                generated["candidate_names"],
            ]
        ),
        "query_ids": generated["query_ids"],
        "fold_id": generated["fold_id"],
        "site": generated["site"],
    }
    for key in ("pas", "pdp", "error", "truth_energy", "predicted_energy", "cross"):
        combined[key] = np.concatenate([legacy[key], generated[key]], axis=0)
    return combined


def select_nested(combined: dict[str, np.ndarray]) -> dict[str, object]:
    names = combined["candidate_names"]
    fold_id = combined["fold_id"]
    site = combined["site"]
    count = len(fold_id)
    selected_candidate = np.empty(count, dtype=np.int16)
    selected_scale = np.empty(count, dtype=np.float32)
    selected_error = np.empty(count, dtype=np.float64)
    decisions: list[dict[str, object]] = []

    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            train = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            rows: list[tuple[float, int, float]] = []
            for candidate_index in range(len(names)):
                scale = optimal_scale(
                    combined["predicted_energy"][candidate_index],
                    combined["cross"][candidate_index],
                    train,
                )
                error = scaled_error(
                    combined["predicted_energy"][candidate_index],
                    combined["truth_energy"][candidate_index],
                    combined["cross"][candidate_index],
                    scale,
                )
                score = float(
                    statistics(
                        combined["pas"][candidate_index],
                        combined["pdp"][candidate_index],
                        error,
                        combined["truth_energy"][candidate_index],
                        train,
                    )["score"]
                )
                rows.append((score, candidate_index, scale))
            training_score, candidate_index, scale = max(rows)
            selected_candidate[held] = candidate_index
            selected_scale[held] = scale
            selected_error[held] = scaled_error(
                combined["predicted_energy"][candidate_index, held],
                combined["truth_energy"][candidate_index, held],
                combined["cross"][candidate_index, held],
                scale,
            )
            decisions.append(
                {
                    "held_fold": int(held_fold),
                    "site": int(site_value),
                    "candidate": str(names[candidate_index]),
                    "scale": float(scale),
                    "training_score": training_score,
                    "held_count": int(np.sum(held)),
                }
            )

    row = np.arange(count)
    selected_pas = combined["pas"][selected_candidate, row]
    selected_pdp = combined["pdp"][selected_candidate, row]
    selected_truth_energy = combined["truth_energy"][selected_candidate, row]
    overall = statistics(
        selected_pas, selected_pdp, selected_error, selected_truth_energy
    )
    folds = [
        {
            "fold": int(fold),
            **statistics(
                selected_pas,
                selected_pdp,
                selected_error,
                selected_truth_energy,
                fold_id == fold,
            ),
        }
        for fold in np.unique(fold_id)
    ]

    final_policy: list[dict[str, object]] = []
    for site_value in np.unique(site):
        mask = site == site_value
        candidates: list[tuple[float, int, float]] = []
        for candidate_index in range(len(names)):
            scale = optimal_scale(
                combined["predicted_energy"][candidate_index],
                combined["cross"][candidate_index],
                mask,
            )
            error = scaled_error(
                combined["predicted_energy"][candidate_index],
                combined["truth_energy"][candidate_index],
                combined["cross"][candidate_index],
                scale,
            )
            score = float(
                statistics(
                    combined["pas"][candidate_index],
                    combined["pdp"][candidate_index],
                    error,
                    combined["truth_energy"][candidate_index],
                    mask,
                )["score"]
            )
            candidates.append((score, candidate_index, scale))
        score, candidate_index, scale = max(candidates)
        final_policy.append(
            {
                "site": int(site_value),
                "candidate_index": int(candidate_index),
                "candidate": str(names[candidate_index]),
                "scale": float(scale),
                "oof_site_score": score,
            }
        )
    return {
        "overall": overall,
        "folds": folds,
        "decisions": decisions,
        "final_policy": final_policy,
        "selected_candidate": selected_candidate,
        "selected_scale": selected_scale,
    }


def v3_nested_baseline(combined: dict[str, np.ndarray]) -> dict[str, object]:
    legacy = np.load(V3_METRICS)
    selected = legacy["selected_action"].astype(np.int64)
    row = np.arange(len(selected))
    scale = legacy["selected_scale"].astype(np.float64)
    error = scaled_error(
        legacy["predicted_energy"][selected, row],
        legacy["truth_energy"][selected, row],
        legacy["cross"][selected, row],
        scale,
    )
    pas = legacy["pas"][selected, row]
    pdp = legacy["pdp"][selected, row]
    energy = legacy["truth_energy"][selected, row]
    fold_id = combined["fold_id"]
    return {
        "overall": statistics(pas, pdp, error, energy),
        "folds": [
            {
                "fold": int(fold),
                **statistics(pas, pdp, error, energy, fold_id == fold),
            }
            for fold in np.unique(fold_id)
        ],
    }


def summarize_candidates(
    generated: dict[str, np.ndarray], combined: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = len(combined["candidate_names"]) - len(generated["candidate_names"])
    for local_index, name in enumerate(generated["candidate_names"]):
        candidate_index = offset + local_index
        calibrated_error = combined["error"][candidate_index].copy()
        scales: dict[str, float] = {}
        for site_value in np.unique(combined["site"]):
            mask = combined["site"] == site_value
            scale = optimal_scale(
                combined["predicted_energy"][candidate_index],
                combined["cross"][candidate_index],
                mask,
            )
            calibrated_error[mask] = scaled_error(
                combined["predicted_energy"][candidate_index, mask],
                combined["truth_energy"][candidate_index, mask],
                combined["cross"][candidate_index, mask],
                scale,
            )
            scales[str(int(site_value))] = float(scale)
        row = {
            "candidate": str(name),
            "scale_by_site": scales,
            "score": statistics(
                combined["pas"][candidate_index],
                combined["pdp"][candidate_index],
                calibrated_error,
                combined["truth_energy"][candidate_index],
            ),
            "target_realization": {
                "pas": float(np.mean(generated["pas_target"][local_index])),
                "pdp": float(np.mean(generated["pdp_target"][local_index])),
            },
        }
        rows.append(row)
    return sorted(rows, key=lambda row: float(row["score"]["score"]), reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-cycles", type=int, choices=CHECKPOINTS, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = tuple(value for value in CHECKPOINTS if value <= args.max_cycles)
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    generated = generate_metrics(torch.device(args.device), args.batch_size, checkpoints)
    np.savez_compressed(METRICS, **generated)
    combined = combine_with_v3(generated)
    selected = select_nested(combined)
    baseline = v3_nested_baseline(combined)
    fold_deltas = [
        {
            "fold": int(new["fold"]),
            "score": float(new["score"] - old["score"]),
            "pas": float(new["pas"] - old["pas"]),
            "pdp": float(new["pdp"] - old["pdp"]),
            "nmse": float(new["nmse"] - old["nmse"]),
        }
        for new, old in zip(selected["folds"], baseline["folds"])
    ]
    candidate_rows = summarize_candidates(generated, combined)
    overall_delta = float(
        selected["overall"]["score"] - baseline["overall"]["score"]
    )
    passed = (
        selected["overall"]["score"] >= 0.645
        and sum(row["score"] > 0.0 for row in fold_deltas) >= 4
    )
    report = {
        "experiment": VERSION,
        "status": "candidate_signal_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "V3 predicts better PAS/PDP targets than its one-pass clipped projection realizes; "
            "fixed-target alternating magnitude projections can recover that implementation loss "
            "without introducing a new information source."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "parameters": {
            "actions": [asdict(action) for action in ACTIONS],
            "orders": list(ORDERS),
            "triple_orders": list(TRIPLE_ORDERS),
            "checkpoints": list(checkpoints),
            "batch_size": args.batch_size,
            "selection": "outer-fold and site nested selection with legacy V3 actions retained",
        },
        "v3_nested_baseline": baseline,
        "nested_selected": {
            "overall": selected["overall"],
            "folds": selected["folds"],
            "decisions": selected["decisions"],
            "final_policy": selected["final_policy"],
        },
        "delta_vs_v3": {
            "overall_score": overall_delta,
            "folds": fold_deltas,
        },
        "top_candidates": candidate_rows[:20],
        "decision": {
            "passed": passed,
            "threshold": "score>=0.645 and at least 4/5 fold deltas positive",
        },
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "v3": baseline["overall"],
                "selected": selected["overall"],
                "delta": report["delta_vs_v3"],
                "final_policy": selected["final_policy"],
                "top_candidates": candidate_rows[:5],
                "passed": passed,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
