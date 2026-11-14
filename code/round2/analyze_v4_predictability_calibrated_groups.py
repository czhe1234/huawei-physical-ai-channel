#!/usr/bin/env python3
"""Calibrate V3 grouped spectra by each latent mode's OOF predictability."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from data import load_positions
from evaluate_v3_anchored_projection import (
    PAS_GLOBAL_SHRINK,
    PDP_GLOBAL_SHRINK,
    TARGET_PAS,
    TARGET_PDP,
)
from project_paths import PATHS
from run_rbf_spectral_field import FOLD_CACHE, OOF_META
from run_v3_grouped_link_refinement import GROUPED_PAS, GROUPED_PDP


VERSION = "Round2-E22-predictability-calibrated-groups"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_predictability_calibrated_groups"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_predictability_calibrated_groups"
REPORT = ANALYSIS_DIR / "predictability_calibrated_groups.json"
SCORES = ARTIFACT_DIR / "target_scores.npz"
PAS_GROUP_VARIANTS = ARTIFACT_DIR / "oof_pas_group_variants_f16.npy"
PDP_GROUP_VARIANTS = ARTIFACT_DIR / "oof_pdp_group_variants_f16.npy"
OOF_CALIBRATED_PAS = ARTIFACT_DIR / "oof_calibrated_pas_f16.npy"
OOF_CALIBRATED_PDP = ARTIFACT_DIR / "oof_calibrated_pdp_f16.npy"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_PAS = PRE_DIR / "round2_train_pas_link_support_f16.npy"
TRAIN_PDP = PRE_DIR / "round2_train_pdp_link_support_f16.npy"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
V3_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "v3_anchored_neural_rbf"
    / "oof_epoch6_corrected_weights.npz"
)

CALIBRATION_STRENGTHS = (0.0, 0.5, 1.0)
GROUP_BLEND_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
PAS_COMPONENTS = 128
PDP_COMPONENTS = 64
TINY = 1e-30


def normalize_power(value: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = value.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, TINY),
        out=np.zeros_like(value),
        where=total > 0.0,
    )


def normalize_group_amplitude(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    norm = np.sqrt(np.sum(value * value, axis=-1, keepdims=True, dtype=np.float64))
    return np.divide(
        value,
        np.maximum(norm, TINY),
        out=np.zeros_like(value),
        where=norm > 0.0,
    )


def grouped_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_power = np.square(np.asarray(prediction, dtype=np.float32))
    truth_power = np.square(np.asarray(truth, dtype=np.float32))
    numerator = np.sum(pred_power * truth_power, axis=-1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(pred_power * pred_power, axis=-1, dtype=np.float64)
        * np.sum(truth_power * truth_power, axis=-1, dtype=np.float64)
    )
    values = np.divide(
        numerator,
        np.maximum(denominator, TINY),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return values.mean(axis=1, dtype=np.float64)


def linkwise_cosine(
    prediction: np.ndarray,
    truth: np.ndarray,
    axis: int,
    average_axes: tuple[int, ...],
) -> np.ndarray:
    numerator = np.sum(prediction * truth, axis=axis, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(prediction * prediction, axis=axis, dtype=np.float64)
        * np.sum(truth * truth, axis=axis, dtype=np.float64)
    )
    values = np.divide(
        numerator,
        np.maximum(denominator, TINY),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return values.mean(axis=average_axes, dtype=np.float64)


def predict_grouped(
    amplitude: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    output = np.empty(
        (len(donor_ids), amplitude.shape[1], amplitude.shape[2]), dtype=np.float32
    )
    for row, (ids, raw_weights) in enumerate(zip(donor_ids, weights)):
        selected_weights = np.asarray(raw_weights, dtype=np.float64).copy()
        if allowed is not None:
            selected_weights *= allowed[ids]
            total = float(np.sum(selected_weights))
            if abs(total) < 0.10:
                selected_weights = np.maximum(selected_weights, 0.0)
                total = float(np.sum(selected_weights))
            if abs(total) < 1e-8:
                nearest = int(np.flatnonzero(allowed[ids])[0])
                selected_weights[:] = 0.0
                selected_weights[nearest] = 1.0
            else:
                selected_weights /= total
        output[row] = np.maximum(
            np.tensordot(
                selected_weights.astype(np.float32),
                np.asarray(amplitude[ids], dtype=np.float32),
                axes=(0, 0),
            ),
            0.0,
        )
    return normalize_group_amplitude(output)


def pca_basis(
    fit_amplitude: np.ndarray,
    components: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    value = torch.as_tensor(fit_amplitude, dtype=torch.float32, device=device)
    mean = value.mean(dim=0)
    centered = value - mean
    count = min(components, len(value) - 1, value.shape[1])
    _, _, vectors = torch.pca_lowrank(
        centered, q=count, center=False, niter=4
    )
    return mean.cpu().numpy(), vectors.cpu().numpy()


def encode(value: np.ndarray, mean: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float32) - mean[None, :]) @ vectors


def calibrate_metric(
    amplitude: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    components: int,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    query_site: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    folds: np.lib.npyio.NpzFile,
    device: torch.device,
    metric_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    flat = np.asarray(amplitude, dtype=np.float32).reshape(len(amplitude), -1)
    output = np.empty(
        (
            len(CALIBRATION_STRENGTHS),
            len(query_ids),
            amplitude.shape[1],
            amplitude.shape[2],
        ),
        dtype=np.float32,
    )
    records: list[dict[str, object]] = []

    for held_fold in np.unique(fold_id):
        fit_ids = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
        allowed = np.zeros(len(amplitude), dtype=bool)
        allowed[fit_ids] = True
        allowed &= valid
        for site_value in np.unique(query_site):
            held_rows = np.flatnonzero(
                (fold_id == held_fold) & (query_site == site_value)
            )
            calibration_rows = np.flatnonzero(
                (fold_id != held_fold)
                & (query_site == site_value)
                & allowed[query_ids]
            )
            fit_site_ids = np.flatnonzero(allowed & (site == site_value))
            mean, vectors = pca_basis(
                flat[fit_site_ids],
                components,
                device,
                metric_seed + int(held_fold) * 101 + int(site_value) * 1009,
            )

            calibration_prediction = predict_grouped(
                amplitude,
                donor_ids[calibration_rows],
                weights[calibration_rows],
                allowed,
            )
            held_prediction = predict_grouped(
                amplitude,
                donor_ids[held_rows],
                weights[held_rows],
                allowed,
            )
            calibration_pred_flat = calibration_prediction.reshape(
                len(calibration_rows), -1
            )
            held_pred_flat = held_prediction.reshape(len(held_rows), -1)
            pred_coefficients = encode(calibration_pred_flat, mean, vectors)
            truth_coefficients = encode(
                flat[query_ids[calibration_rows]], mean, vectors
            )
            denominator = np.sum(
                pred_coefficients * pred_coefficients, axis=0, dtype=np.float64
            )
            slope = np.divide(
                np.sum(
                    pred_coefficients * truth_coefficients,
                    axis=0,
                    dtype=np.float64,
                ),
                np.maximum(denominator, 1e-12),
            )
            slope = np.clip(slope, 0.0, 1.25).astype(np.float32)
            held_coefficients = encode(held_pred_flat, mean, vectors)
            for variant, strength in enumerate(CALIBRATION_STRENGTHS):
                gain = 1.0 + strength * (slope - 1.0)
                correction = ((held_coefficients * gain) - held_coefficients) @ vectors.T
                corrected = held_pred_flat + correction
                output[variant, held_rows] = normalize_group_amplitude(
                    corrected.reshape(
                        len(held_rows), amplitude.shape[1], amplitude.shape[2]
                    )
                )
            records.append(
                {
                    "fold": int(held_fold),
                    "site": int(site_value),
                    "fit_count": int(len(fit_site_ids)),
                    "calibration_count": int(len(calibration_rows)),
                    "held_count": int(len(held_rows)),
                    "components": int(vectors.shape[1]),
                    "slope_mean": float(np.mean(slope)),
                    "slope_zero_fraction": float(np.mean(slope == 0.0)),
                    "slope_one_or_more_fraction": float(np.mean(slope >= 1.0)),
                }
            )
    return output, {"fold_site": records}


def globally_shrunk_targets(
    pas: np.ndarray, pdp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pas = normalize_power(pas, 1)
    pdp = normalize_power(pdp, 3)
    shared_pas = np.broadcast_to(
        pas.mean(axis=(2, 3), keepdims=True, dtype=np.float64).astype(np.float32),
        pas.shape,
    )
    shared_pdp = np.broadcast_to(
        pdp.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32),
        pdp.shape,
    )
    pas_amplitude = (
        (1.0 - PAS_GLOBAL_SHRINK) * np.sqrt(np.maximum(pas, 0.0))
        + PAS_GLOBAL_SHRINK * np.sqrt(np.maximum(shared_pas, 0.0))
    )
    pdp_amplitude = (
        (1.0 - PDP_GLOBAL_SHRINK) * np.sqrt(np.maximum(pdp, 0.0))
        + PDP_GLOBAL_SHRINK * np.sqrt(np.maximum(shared_pdp, 0.0))
    )
    return (
        normalize_power(pas_amplitude * pas_amplitude, 1),
        normalize_power(pdp_amplitude * pdp_amplitude, 3),
    )


def blend_group_pas(
    target: np.ndarray, grouped: np.ndarray, alpha: float
) -> np.ndarray:
    if alpha <= 0.0:
        return target
    batch = len(target)
    amplitude = np.sqrt(np.maximum(target, 0.0)).reshape(
        batch, 256, 4, 8, 24
    )
    group_amplitude = np.transpose(grouped, (0, 2, 1)).reshape(
        batch, 256, 4, 8, 1
    )
    blended = (1.0 - alpha) * amplitude + alpha * group_amplitude
    power = blended * blended
    return normalize_power(power, 1).reshape(batch, 256, 4, 192)


def blend_group_pdp(
    target: np.ndarray, grouped: np.ndarray, alpha: float
) -> np.ndarray:
    if alpha <= 0.0:
        return target
    batch = len(target)
    amplitude = np.sqrt(np.maximum(target, 0.0)).reshape(
        batch, 2, 16, 8, 4, 192
    )
    group_amplitude = grouped.reshape(batch, 2, 4, 192)[:, :, None, None]
    blended = (1.0 - alpha) * amplitude + alpha * group_amplitude
    power = blended * blended
    return normalize_power(power, 5).reshape(batch, 256, 4, 192)


def evaluate_targets(
    pas_variants: np.ndarray,
    pdp_variants: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    target_pas = np.load(TARGET_PAS, mmap_mode="r")
    target_pdp = np.load(TARGET_PDP, mmap_mode="r")
    names = [
        f"cal{calibration:g}_blend{blend:g}"
        for calibration in CALIBRATION_STRENGTHS
        for blend in GROUP_BLEND_STRENGTHS
    ]
    pas_scores = np.empty((len(names), len(query_ids)), dtype=np.float64)
    pdp_scores = np.empty((len(names), len(query_ids)), dtype=np.float64)

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        pas, pdp = globally_shrunk_targets(
            np.asarray(target_pas[start:end], dtype=np.float32),
            np.asarray(target_pdp[start:end], dtype=np.float32),
        )
        truth_pas = np.asarray(train_pas[query_ids[start:end]], dtype=np.float32)
        truth_pdp = np.asarray(train_pdp[query_ids[start:end]], dtype=np.float32)
        index = 0
        for calibration_index in range(len(CALIBRATION_STRENGTHS)):
            for blend in GROUP_BLEND_STRENGTHS:
                candidate_pas = blend_group_pas(
                    pas,
                    pas_variants[calibration_index, start:end],
                    blend,
                )
                candidate_pdp = blend_group_pdp(
                    pdp,
                    pdp_variants[calibration_index, start:end],
                    blend,
                )
                pas_scores[index, start:end] = linkwise_cosine(
                    candidate_pas, truth_pas, 1, (1, 2)
                )
                pdp_scores[index, start:end] = linkwise_cosine(
                    candidate_pdp, truth_pdp, 3, (1, 2)
                )
                index += 1
        print(f"group target evaluation {end}/{len(query_ids)}", flush=True)
    return pas_scores, pdp_scores, names


def summarize(values: np.ndarray, fold_id: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [
            float(np.mean(values[fold_id == fold])) for fold in np.unique(fold_id)
        ],
    }


def nested_selection(
    scores: np.ndarray,
    names: list[str],
    fold_id: np.ndarray,
) -> dict[str, object]:
    selected = np.empty(len(fold_id), dtype=np.int16)
    choices: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        train = fold_id != held_fold
        held = fold_id == held_fold
        means = np.mean(scores[:, train], axis=1)
        best = int(np.argmax(means))
        selected[held] = best
        choices.append(
            {
                "held_fold": int(held_fold),
                "candidate": names[best],
                "training_score": float(means[best]),
                "held_score": float(np.mean(scores[best, held])),
            }
        )
    values = scores[selected, np.arange(len(fold_id))]
    posterior = int(np.argmax(np.mean(scores, axis=1)))
    return {
        "nested": summarize(values, fold_id),
        "choices": choices,
        "posterior_best": {
            "candidate": names[posterior],
            "score": summarize(scores[posterior], fold_id),
        },
        "selected_index": selected,
        "posterior_index": posterior,
    }


def parse_candidate(name: str) -> tuple[int, float]:
    calibration_text, blend_text = name.split("_")
    calibration = float(calibration_text.removeprefix("cal"))
    blend = float(blend_text.removeprefix("blend"))
    calibration_index = CALIBRATION_STRENGTHS.index(calibration)
    return calibration_index, blend


def write_selected_targets(
    pas_variants: np.ndarray,
    pdp_variants: np.ndarray,
    pas_candidate: str,
    pdp_candidate: str,
    batch_size: int,
) -> None:
    pas_calibration, pas_blend = parse_candidate(pas_candidate)
    pdp_calibration, pdp_blend = parse_candidate(pdp_candidate)
    source_pas = np.load(TARGET_PAS, mmap_mode="r")
    source_pdp = np.load(TARGET_PDP, mmap_mode="r")
    output_pas = np.lib.format.open_memmap(
        OOF_CALIBRATED_PAS,
        mode="w+",
        dtype=np.float16,
        shape=source_pas.shape,
    )
    output_pdp = np.lib.format.open_memmap(
        OOF_CALIBRATED_PDP,
        mode="w+",
        dtype=np.float16,
        shape=source_pdp.shape,
    )
    for start in range(0, len(source_pas), batch_size):
        end = min(start + batch_size, len(source_pas))
        pas, pdp = globally_shrunk_targets(
            np.asarray(source_pas[start:end], dtype=np.float32),
            np.asarray(source_pdp[start:end], dtype=np.float32),
        )
        output_pas[start:end] = blend_group_pas(
            pas,
            pas_variants[pas_calibration, start:end],
            pas_blend,
        ).astype(np.float16)
        output_pdp[start:end] = blend_group_pdp(
            pdp,
            pdp_variants[pdp_calibration, start:end],
            pdp_blend,
        ).astype(np.float16)
        output_pas.flush()
        output_pdp.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, site, _ = load_positions()
    query_site = site[query_ids]
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    folds = np.load(FOLD_CACHE)
    weights = np.load(V3_WEIGHTS)
    device = torch.device(args.device)

    pas_amplitude = np.load(GROUPED_PAS, mmap_mode="r")
    pdp_amplitude = np.load(GROUPED_PDP, mmap_mode="r")
    pas_variants, pas_diagnostics = calibrate_metric(
        pas_amplitude,
        weights["donor_ids"].astype(np.int64),
        weights["pas_weights"].astype(np.float32),
        PAS_COMPONENTS,
        query_ids,
        fold_id,
        query_site,
        site,
        valid,
        folds,
        device,
        20_260_805,
    )
    pdp_variants, pdp_diagnostics = calibrate_metric(
        pdp_amplitude,
        weights["donor_ids"].astype(np.int64),
        weights["pdp_weights"].astype(np.float32),
        PDP_COMPONENTS,
        query_ids,
        fold_id,
        query_site,
        site,
        valid,
        folds,
        device,
        20_270_805,
    )
    pas_store = np.lib.format.open_memmap(
        PAS_GROUP_VARIANTS,
        mode="w+",
        dtype=np.float16,
        shape=pas_variants.shape,
    )
    pdp_store = np.lib.format.open_memmap(
        PDP_GROUP_VARIANTS,
        mode="w+",
        dtype=np.float16,
        shape=pdp_variants.shape,
    )
    pas_store[:] = pas_variants.astype(np.float16)
    pdp_store[:] = pdp_variants.astype(np.float16)
    pas_store.flush()
    pdp_store.flush()

    pas_group_scores = [
        grouped_cosine(pas_variants[index], np.asarray(pas_amplitude[query_ids]))
        for index in range(len(CALIBRATION_STRENGTHS))
    ]
    pdp_group_scores = [
        grouped_cosine(pdp_variants[index], np.asarray(pdp_amplitude[query_ids]))
        for index in range(len(CALIBRATION_STRENGTHS))
    ]
    pas_scores, pdp_scores, names = evaluate_targets(
        pas_variants, pdp_variants, query_ids, args.batch_size
    )
    pas_selection = nested_selection(pas_scores, names, fold_id)
    pdp_selection = nested_selection(pdp_scores, names, fold_id)
    combined_scores = 0.5 * (pas_scores + pdp_scores)
    combined_selection = nested_selection(combined_scores, names, fold_id)
    pas_candidate = pas_selection["posterior_best"]["candidate"]
    pdp_candidate = pdp_selection["posterior_best"]["candidate"]
    write_selected_targets(
        pas_variants,
        pdp_variants,
        pas_candidate,
        pdp_candidate,
        args.batch_size,
    )
    np.savez_compressed(
        SCORES,
        names=np.asarray(names),
        pas=pas_scores,
        pdp=pdp_scores,
        combined=combined_scores,
        query_ids=query_ids,
        fold_id=fold_id,
        site=query_site,
    )
    baseline_index = names.index("cal0_blend0")
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "V3 applies one donor mixture to all official links. Grouped spectral modes "
            "whose OOF coefficients are not spatially predictable should be shrunk before "
            "they are injected into full linkwise PAS/PDP targets."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes; PCA and reliability slopes are fit inside each outer fold",
        "parameters": {
            "calibration_strengths": list(CALIBRATION_STRENGTHS),
            "group_blend_strengths": list(GROUP_BLEND_STRENGTHS),
            "pas_components": PAS_COMPONENTS,
            "pdp_components": PDP_COMPONENTS,
        },
        "group_scores": {
            "pas": {
                str(value): summarize(pas_group_scores[index], fold_id)
                for index, value in enumerate(CALIBRATION_STRENGTHS)
            },
            "pdp": {
                str(value): summarize(pdp_group_scores[index], fold_id)
                for index, value in enumerate(CALIBRATION_STRENGTHS)
            },
        },
        "full_target_scores": {
            "baseline": {
                "pas": summarize(pas_scores[baseline_index], fold_id),
                "pdp": summarize(pdp_scores[baseline_index], fold_id),
                "combined": summarize(combined_scores[baseline_index], fold_id),
            },
            "pas_selection": {
                key: value
                for key, value in pas_selection.items()
                if key not in ("selected_index", "posterior_index")
            },
            "pdp_selection": {
                key: value
                for key, value in pdp_selection.items()
                if key not in ("selected_index", "posterior_index")
            },
            "combined_selection": {
                key: value
                for key, value in combined_selection.items()
                if key not in ("selected_index", "posterior_index")
            },
        },
        "diagnostics": {
            "pas": pas_diagnostics,
            "pdp": pdp_diagnostics,
        },
        "outputs": {
            "pas_group_variants": PATHS.relative(PAS_GROUP_VARIANTS),
            "pdp_group_variants": PATHS.relative(PDP_GROUP_VARIANTS),
            "scores": PATHS.relative(SCORES),
            "selected_pas_target": PATHS.relative(OOF_CALIBRATED_PAS),
            "selected_pdp_target": PATHS.relative(OOF_CALIBRATED_PDP),
            "selected_pas_policy": pas_candidate,
            "selected_pdp_policy": pdp_candidate,
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "group_scores": payload["group_scores"],
                "full_target_scores": payload["full_target_scores"],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
