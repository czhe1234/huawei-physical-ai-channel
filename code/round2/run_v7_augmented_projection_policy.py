#!/usr/bin/env python3
"""Learn V4 projection actions from fold-isolated pseudo spatial holes."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from analyze_v4_exact_joint_projection import (
    continue_trajectory,
    initialize_trajectory,
    metric_components,
)
from analyze_v4_predictability_calibrated_groups import (
    blend_group_pas,
    blend_group_pdp,
    encode,
    globally_shrunk_targets,
    normalize_group_amplitude,
    pca_basis,
    predict_grouped,
)
from build_v3_anchored_neural_rbf import spectral_target
from complex_baselines import predict_one
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, OOF_META, optimal_scale, scaled_error, statistics
from run_v3_grouped_link_refinement import GROUPED_PAS, GROUPED_PDP
from run_v4_calibrated_exact_projection import (
    PAS_GROUP_BLEND,
    PDP_CALIBRATION_STRENGTH,
    PDP_GROUP_BLEND,
)
from run_v4_structured_complex_field import (
    DESCRIPTORS,
    FOLD_CACHE,
    FOUNDATION_REPORT,
    GEOMETRY,
    POINT_POWER,
    TRAIN_CHANNEL,
    TrainConfig,
    build_oof_features,
    context_features,
    current_carrier_configs,
    pseudo_dataset_path,
    pseudo_v3_weights,
)


VERSION = "Round2-E49-augmented-projection-policy"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "augmented_projection_policy"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_augmented_projection_policy"
REPORT = ANALYSIS_DIR / "augmented_projection_policy.json"

V4_METRICS = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "calibrated_exact_projection_metrics.npz"
)
V3_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "v3_anchored_neural_rbf"
    / "oof_epoch6_corrected_weights.npz"
)
TRAIN_AD = PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_ad_support_f16.npy"
TRAIN_PAS = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_pas_link_support_f16.npy"
)
TRAIN_PDP = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_pdp_link_support_f16.npy"
)
GEODESIC = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)

BASE_CANDIDATE_BY_SITE = {0: 159, 1: 274}
PAS_COMPONENTS = 128
PDP_COMPONENTS = 64
PAS_CALIBRATION_STRENGTH = 1.0
TINY = 1e-30

PATTERN = re.compile(
    r"^exact_(raw|pascal|pdpcal|bothcal)_(pas_pdp|pdp_pas)_"
    r"(ad\d+_pas\d+_pdp\d+)_c(\d+)$"
)


@dataclass(frozen=True)
class ParsedCandidate:
    index: int
    name: str
    mode: str
    order: str
    action_index: int
    cycles: int


def _candidate(index: int, names: np.ndarray) -> ParsedCandidate:
    name = str(names[index])
    match = PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"cannot parse projection candidate: {name}")
    mode, order, action_name, cycles = match.groups()
    action_index = next(
        action_index
        for action_index, action in enumerate(ACTIONS)
        if action.name == action_name
    )
    return ParsedCandidate(index, name, mode, order, action_index, int(cycles))


def _site_scales(
    arrays: dict[str, np.ndarray], candidate_indices: np.ndarray, fit_mask: np.ndarray
) -> np.ndarray:
    output = np.empty((len(candidate_indices), 2), dtype=np.float64)
    site = arrays["site"].astype(np.int8)
    for site_value in (0, 1):
        mask = fit_mask & (site == site_value)
        for local, candidate_index in enumerate(candidate_indices):
            output[local, site_value] = optimal_scale(
                arrays["predicted_energy"][candidate_index],
                arrays["cross"][candidate_index],
                mask,
            )
    return output


def _scaled_candidate_errors(
    arrays: dict[str, np.ndarray], candidate_indices: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    site = arrays["site"].astype(np.int8)
    count = arrays["pas"].shape[1]
    output = np.empty((len(candidate_indices), count), dtype=np.float64)
    for local, candidate_index in enumerate(candidate_indices):
        point_scale = scales[local, site]
        output[local] = scaled_error(
            arrays["predicted_energy"][candidate_index],
            arrays["truth_energy"][candidate_index],
            arrays["cross"][candidate_index],
            point_scale,
        )
    return output


def _point_utility(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    truth_energy: np.ndarray,
) -> np.ndarray:
    local_nmse = np.divide(
        error,
        np.maximum(truth_energy, TINY),
        out=np.full_like(error, np.inf, dtype=np.float64),
        where=truth_energy > 0.0,
    )
    return 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + local_nmse)


def greedy_candidates(
    arrays: dict[str, np.ndarray], fit_mask: np.ndarray, candidate_count: int
) -> np.ndarray:
    all_indices = np.arange(len(arrays["candidate_names"]), dtype=np.int64)
    scales = _site_scales(arrays, all_indices, fit_mask)
    errors = _scaled_candidate_errors(arrays, all_indices, scales)
    utility = _point_utility(
        arrays["pas"], arrays["pdp"], errors, arrays["truth_energy"]
    )
    rows = np.flatnonzero(fit_mask)
    selected = [BASE_CANDIDATE_BY_SITE[0], BASE_CANDIDATE_BY_SITE[1]]
    covered = np.max(utility[selected][:, rows], axis=0)
    while len(selected) < candidate_count:
        score = np.mean(
            np.maximum(utility[:, rows], covered[None, :]), axis=1, dtype=np.float64
        )
        score[selected] = -np.inf
        winner = int(np.argmax(score))
        selected.append(winner)
        covered = np.maximum(covered, utility[winner, rows])
    return np.asarray(selected, dtype=np.int64)


def _fold_paths(fold: int) -> dict[str, Path]:
    prefix = ARTIFACT_DIR / f"fold{fold}"
    return {
        "base": prefix.with_name(prefix.name + "_pseudo_base.npy"),
        "ad": prefix.with_name(prefix.name + "_pseudo_ad_f16.npy"),
        "raw_pas": prefix.with_name(prefix.name + "_pseudo_raw_pas_f16.npy"),
        "raw_pdp": prefix.with_name(prefix.name + "_pseudo_raw_pdp_f16.npy"),
        "cal_pas": prefix.with_name(prefix.name + "_pseudo_cal_pas_f16.npy"),
        "cal_pdp": prefix.with_name(prefix.name + "_pseudo_cal_pdp_f16.npy"),
        "metrics": prefix.with_name(prefix.name + "_pseudo_projection_metrics.npz"),
        "control_grid": prefix.with_name(prefix.name + "_control_grid.npz"),
        "calibration": prefix.with_name(prefix.name + "_group_calibration.npz"),
        "selection": prefix.with_name(prefix.name + "_candidate_indices.json"),
    }


def _pseudo_context(
    fold: int, device: torch.device, sample_limit: int | None
) -> dict[str, np.ndarray]:
    config = TrainConfig()
    cached = np.load(pseudo_dataset_path(fold, config), allow_pickle=False)
    query_ids = cached["query_ids"].astype(np.int64)
    guard_index = cached["guard_index"].astype(np.int8)
    features = cached["features"].astype(np.float32)
    if sample_limit is not None and sample_limit < len(query_ids):
        order = np.argsort(
            (query_ids.astype(np.int64) * 2_654_435_761 + fold * 104_729)
            % 2_147_483_647
        )[:sample_limit]
        query_ids = query_ids[order]
        guard_index = guard_index[order]
        features = features[order]

    _, positions, _, site, _ = load_positions()
    descriptors = np.load(DESCRIPTORS, allow_pickle=False)
    valid = descriptors["valid"].astype(bool)
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    donor_ids, distances, base_weights, pas_weights, pdp_weights = pseudo_v3_weights(
        fold,
        query_ids,
        guard_index,
        fit_ids,
        positions,
        site,
        valid,
        device,
    )
    return {
        "query_ids": query_ids,
        "guard_index": guard_index,
        "features": features,
        "fit_ids": fit_ids,
        "donor_ids": donor_ids,
        "distances": distances,
        "base_weights": base_weights,
        "pas_weights": pas_weights,
        "pdp_weights": pdp_weights,
    }


def _build_pseudo_base(
    fold: int, context: dict[str, np.ndarray], output_path: Path
) -> np.ndarray:
    expected_shape = (len(context["query_ids"]), 256, 4, 192)
    if output_path.is_file():
        cached = np.load(output_path, mmap_mode="r")
        if cached.shape == expected_shape and cached.dtype == np.complex64:
            return cached

    setup, positions, _, site, _ = load_positions()
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    foundation = json.loads(FOUNDATION_REPORT.read_text(encoding="utf-8"))
    carrier = foundation["carrier_fit_only_by_fold"][str(fold)]
    configs = current_carrier_configs()
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=expected_shape
    )
    for row, query_id in enumerate(context["query_ids"]):
        site_value = int(site[query_id])
        local_carrier = carrier[str(site_value)]
        config = replace(
            configs[site_value],
            beta=float(local_carrier["beta"]),
            gamma=float(local_carrier["gamma"]),
        )
        ids = context["donor_ids"][row, : config.k]
        prediction, _ = predict_one(
            np.asarray(train_channel[ids], dtype=np.complex64),
            positions[ids],
            positions[query_id],
            context["distances"][row, : config.k],
            setup.bs_positions[site_value],
            config,
        )
        output[row] = prediction
        if (row + 1) % 100 == 0 or row + 1 == len(output):
            output.flush()
            print(f"E49 pseudo base fold={fold} {row + 1}/{len(output)}", flush=True)
    return np.load(output_path, mmap_mode="r")


def _fit_group_calibration(
    fold: int, device: torch.device, output_path: Path
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if output_path.is_file():
        cached = np.load(output_path, allow_pickle=False)
        return {
            key.removesuffix("_mean"): (
                cached[key],
                cached[key.replace("_mean", "_vectors")],
                cached[key.replace("_mean", "_slope")],
            )
            for key in cached.files
            if key.endswith("_mean")
        }

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, site, _ = load_positions()
    query_site = site[query_ids]
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    descriptors = np.load(DESCRIPTORS, allow_pickle=False)
    valid = descriptors["valid"].astype(bool)
    allowed = np.zeros(len(site), dtype=bool)
    allowed[fit_ids] = True
    allowed &= valid
    weights = np.load(V3_WEIGHTS, allow_pickle=False)
    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for metric, amplitude_path, weight_key, components, seed in (
        ("pas", GROUPED_PAS, "pas_weights", PAS_COMPONENTS, 20_260_805),
        ("pdp", GROUPED_PDP, "pdp_weights", PDP_COMPONENTS, 20_270_805),
    ):
        amplitude = np.load(amplitude_path, mmap_mode="r")
        flat = np.asarray(amplitude, dtype=np.float32).reshape(len(amplitude), -1)
        for site_value in (0, 1):
            fit_site = np.flatnonzero(allowed & (site == site_value))
            calibration_rows = np.flatnonzero(
                (fold_id != fold)
                & (query_site == site_value)
                & allowed[query_ids]
            )
            mean, vectors = pca_basis(
                flat[fit_site],
                components,
                device,
                seed + fold * 101 + site_value * 1009,
            )
            prediction = predict_grouped(
                amplitude,
                weights["donor_ids"][calibration_rows].astype(np.int64),
                weights[weight_key][calibration_rows].astype(np.float32),
                allowed,
            )
            predicted_coefficients = encode(
                prediction.reshape(len(prediction), -1), mean, vectors
            )
            truth_coefficients = encode(
                flat[query_ids[calibration_rows]], mean, vectors
            )
            denominator = np.sum(
                predicted_coefficients * predicted_coefficients,
                axis=0,
                dtype=np.float64,
            )
            slope = np.divide(
                np.sum(
                    predicted_coefficients * truth_coefficients,
                    axis=0,
                    dtype=np.float64,
                ),
                np.maximum(denominator, 1e-12),
            )
            models[f"{metric}_site{site_value}"] = (
                mean.astype(np.float32),
                vectors.astype(np.float32),
                np.clip(slope, 0.0, 1.25).astype(np.float32),
            )
            print(
                f"E49 calibration fold={fold} metric={metric} site={site_value} "
                f"rows={len(calibration_rows)}",
                flush=True,
            )
    payload: dict[str, np.ndarray] = {}
    for key, (mean, vectors, slope) in models.items():
        payload[f"{key}_mean"] = mean
        payload[f"{key}_vectors"] = vectors
        payload[f"{key}_slope"] = slope
    np.savez_compressed(output_path, **payload)
    return models


def _apply_group_model(
    prediction: np.ndarray,
    query_site: np.ndarray,
    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    metric: str,
    strength: float,
) -> np.ndarray:
    output = np.empty_like(prediction, dtype=np.float32)
    shape = prediction.shape[1:]
    for site_value in (0, 1):
        rows = np.flatnonzero(query_site == site_value)
        mean, vectors, slope = models[f"{metric}_site{site_value}"]
        flat = prediction[rows].reshape(len(rows), -1)
        coefficients = encode(flat, mean, vectors)
        gain = 1.0 + strength * (slope - 1.0)
        corrected = flat + ((coefficients * gain) - coefficients) @ vectors.T
        output[rows] = normalize_group_amplitude(
            corrected.reshape(len(rows), *shape)
        )
    return output


def _build_pseudo_targets(
    fold: int,
    context: dict[str, np.ndarray],
    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    paths: dict[str, Path],
) -> None:
    count = len(context["query_ids"])
    expected = {
        "ad": (count, 16, 8, 192),
        "raw_pas": (count, 256, 4, 192),
        "raw_pdp": (count, 256, 4, 192),
        "cal_pas": (count, 256, 4, 192),
        "cal_pdp": (count, 256, 4, 192),
    }
    if all(
        paths[key].is_file()
        and np.load(paths[key], mmap_mode="r").shape == shape
        for key, shape in expected.items()
    ):
        return

    train_ad = np.load(TRAIN_AD, mmap_mode="r")
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    grouped_pas = np.load(GROUPED_PAS, mmap_mode="r")
    grouped_pdp = np.load(GROUPED_PDP, mmap_mode="r")
    _, _, _, site, _ = load_positions()
    query_site = site[context["query_ids"]].astype(np.int8)

    grouped_pas_prediction = predict_grouped(
        grouped_pas, context["donor_ids"], context["pas_weights"]
    )
    grouped_pdp_prediction = predict_grouped(
        grouped_pdp, context["donor_ids"], context["pdp_weights"]
    )
    calibrated_group_pas = _apply_group_model(
        grouped_pas_prediction, query_site, models, "pas", PAS_CALIBRATION_STRENGTH
    )
    calibrated_group_pdp = _apply_group_model(
        grouped_pdp_prediction,
        query_site,
        models,
        "pdp",
        PDP_CALIBRATION_STRENGTH,
    )

    stores = {
        key: np.lib.format.open_memmap(
            paths[key], mode="w+", dtype=np.float16, shape=shape
        )
        for key, shape in expected.items()
    }
    for row in range(count):
        ids = context["donor_ids"][row]
        ad = spectral_target(train_ad, ids, context["pas_weights"][row], (0, 1, 2))
        pas = spectral_target(train_pas, ids, context["pas_weights"][row], 0)
        pdp = spectral_target(train_pdp, ids, context["pdp_weights"][row], 2)
        raw_pas, raw_pdp = globally_shrunk_targets(pas[None], pdp[None])
        stores["ad"][row] = ad.astype(np.float16)
        stores["raw_pas"][row] = raw_pas[0].astype(np.float16)
        stores["raw_pdp"][row] = raw_pdp[0].astype(np.float16)
        stores["cal_pas"][row] = blend_group_pas(
            raw_pas, calibrated_group_pas[row : row + 1], PAS_GROUP_BLEND
        )[0].astype(np.float16)
        stores["cal_pdp"][row] = blend_group_pdp(
            raw_pdp, calibrated_group_pdp[row : row + 1], PDP_GROUP_BLEND
        )[0].astype(np.float16)
        if (row + 1) % 50 == 0 or row + 1 == count:
            for store in stores.values():
                store.flush()
            print(f"E49 pseudo targets fold={fold} {row + 1}/{count}", flush=True)


def _mode_targets(
    mode: str,
    raw_pas: torch.Tensor,
    raw_pdp: torch.Tensor,
    calibrated_pas: torch.Tensor,
    calibrated_pdp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas = calibrated_pas if mode in ("pascal", "bothcal") else raw_pas
    pdp = calibrated_pdp if mode in ("pdpcal", "bothcal") else raw_pdp
    return pas, pdp


def _generate_pseudo_metrics(
    fold: int,
    candidate_indices: np.ndarray,
    context: dict[str, np.ndarray],
    paths: dict[str, Path],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if paths["metrics"].is_file():
        cached = np.load(paths["metrics"], allow_pickle=False)
        if (
            np.array_equal(cached["candidate_indices"], candidate_indices)
            and np.array_equal(cached["query_ids"], context["query_ids"])
        ):
            return {key: cached[key] for key in cached.files}

    actual = np.load(V4_METRICS, allow_pickle=False)
    names = actual["candidate_names"]
    parsed = [_candidate(int(index), names) for index in candidate_indices]
    grouped: dict[tuple[str, str, int], list[tuple[int, ParsedCandidate]]] = {}
    for local, candidate in enumerate(parsed):
        grouped.setdefault(
            (candidate.mode, candidate.order, candidate.action_index), []
        ).append((local, candidate))

    count = len(context["query_ids"])
    output = {
        key: np.empty((len(parsed), count), dtype=np.float64)
        for key in (
            "pas",
            "pdp",
            "pas_target",
            "pdp_target",
            "predicted_energy",
            "truth_energy",
            "cross",
        )
    }
    base = np.load(paths["base"], mmap_mode="r")
    ad_cache = np.load(paths["ad"], mmap_mode="r")
    raw_pas_cache = np.load(paths["raw_pas"], mmap_mode="r")
    raw_pdp_cache = np.load(paths["raw_pdp"], mmap_mode="r")
    cal_pas_cache = np.load(paths["cal_pas"], mmap_mode="r")
    cal_pdp_cache = np.load(paths["cal_pdp"], mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(
                truth_all[context["query_ids"][start:end]],
                dtype=np.complex64,
                copy=True,
            ),
            device=device,
        )
        ad = torch.as_tensor(
            np.asarray(ad_cache[start:end], dtype=np.float32), device=device
        )
        raw_pas = torch.as_tensor(
            np.asarray(raw_pas_cache[start:end], dtype=np.float32), device=device
        )
        raw_pdp = torch.as_tensor(
            np.asarray(raw_pdp_cache[start:end], dtype=np.float32), device=device
        )
        cal_pas = torch.as_tensor(
            np.asarray(cal_pas_cache[start:end], dtype=np.float32), device=device
        )
        cal_pdp = torch.as_tensor(
            np.asarray(cal_pdp_cache[start:end], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            shaped_truth = truth.reshape(len(truth), 2, 16, 8, 4, 192)
            truth_pas = torch.fft.fft2(shaped_truth, dim=(2, 3)).abs().square().float()
            truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
            for (mode, order, action_index), members in grouped.items():
                pas, pdp = _mode_targets(mode, raw_pas, raw_pdp, cal_pas, cal_pdp)
                candidate, desired_pas, desired_pdp, reference_power = initialize_trajectory(
                    base_tensor, ad, pas, pdp, action_index, order
                )
                wanted = {member.cycles: local for local, member in members}
                max_cycle = max(wanted)
                for cycle in range(1, max_cycle + 1):
                    if cycle > 1:
                        candidate = continue_trajectory(
                            candidate,
                            desired_pas,
                            desired_pdp,
                            reference_power,
                            order,
                        )
                    if cycle not in wanted:
                        continue
                    local = wanted[cycle]
                    values = metric_components(
                        candidate,
                        truth,
                        truth_pas,
                        truth_pdp,
                        desired_pas,
                        desired_pdp,
                    )
                    for key in output:
                        output[key][local, start:end] = values[key]
        print(f"E49 pseudo metrics fold={fold} {end}/{count}", flush=True)

    output.update(
        {
            "candidate_indices": candidate_indices.astype(np.int16),
            "candidate_names": names[candidate_indices],
            "query_ids": context["query_ids"].astype(np.int32),
            "site": load_positions()[3][context["query_ids"]].astype(np.int8),
        }
    )
    np.savez_compressed(paths["metrics"], **output)
    return output


def _point_features(
    base_features: np.ndarray,
    query_ids: np.ndarray,
    observables: np.ndarray,
    site: np.ndarray,
    base_local: np.ndarray,
) -> np.ndarray:
    geodesic = np.load(GEODESIC, allow_pickle=False)["train_features"].astype(np.float32)
    base_observable = observables[base_local, np.arange(len(query_ids))]
    candidate_observable = observables.transpose(1, 0, 2)
    relation = np.concatenate(
        [
            candidate_observable,
            candidate_observable - base_observable[:, None, :],
            np.abs(candidate_observable - base_observable[:, None, :]),
        ],
        axis=2,
    )
    return np.concatenate(
        [
            base_features.astype(np.float32),
            geodesic[query_ids],
            np.eye(2, dtype=np.float32)[site],
            relation.reshape(len(query_ids), -1).astype(np.float32),
        ],
        axis=1,
    )


def _labels(
    arrays: dict[str, np.ndarray],
    candidate_indices: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    site = arrays["site"].astype(np.int8)
    local_by_index = {int(index): local for local, index in enumerate(candidate_indices)}
    base_local = np.asarray(
        [local_by_index[BASE_CANDIDATE_BY_SITE[int(value)]] for value in site],
        dtype=np.int64,
    )
    errors = np.empty((len(candidate_indices), len(site)), dtype=np.float64)
    for local in range(len(candidate_indices)):
        point_scale = scales[local, site]
        errors[local] = scaled_error(
            arrays["predicted_energy"][local],
            arrays["truth_energy"][local],
            arrays["cross"][local],
            point_scale,
        )
    rows = np.arange(len(site))
    base_pas = arrays["pas"][base_local, rows]
    base_pdp = arrays["pdp"][base_local, rows]
    base_error = errors[base_local, rows]
    mean_energy = float(np.mean(arrays["truth_energy"][0]))
    components = np.stack(
        [
            arrays["pas"] - base_pas[None, :],
            arrays["pdp"] - base_pdp[None, :],
            (errors - base_error[None, :]) / max(mean_energy, TINY),
        ],
        axis=2,
    ).transpose(1, 0, 2)
    return components.astype(np.float32), errors, base_local, np.full(len(site), mean_energy)


def _actual_subset(
    actual: dict[str, np.ndarray], candidate_indices: np.ndarray
) -> dict[str, np.ndarray]:
    output = {
        key: actual[key][candidate_indices]
        for key in (
            "pas",
            "pdp",
            "pas_target",
            "pdp_target",
            "predicted_energy",
            "truth_energy",
            "cross",
        )
    }
    output.update(
        {
            "site": actual["site"],
            "query_ids": actual["query_ids"],
            "fold_id": actual["fold_id"],
        }
    )
    return output


def _score_selection(
    arrays: dict[str, np.ndarray],
    errors: np.ndarray,
    selected: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    rows = np.arange(len(selected))
    return statistics(
        arrays["pas"][selected, rows],
        arrays["pdp"][selected, rows],
        errors[selected, rows],
        arrays["truth_energy"][selected, rows],
        mask,
    )


def _predicted_utility(
    predicted_components: np.ndarray, baseline_nmse: float
) -> np.ndarray:
    return (
        0.4 * predicted_components[:, :, 0]
        + 0.4 * predicted_components[:, :, 1]
        - 0.2
        * predicted_components[:, :, 2]
        / max((1.0 + baseline_nmse) ** 2, 1e-12)
    )


def _choose_actions(
    utility: np.ndarray,
    base_local: np.ndarray,
    threshold: float,
    fraction: float,
    components: np.ndarray | None = None,
    pas_floor: float = -np.inf,
    pdp_floor: float = -np.inf,
    agreement_utility: np.ndarray | None = None,
) -> np.ndarray:
    value = utility.copy()
    value[np.arange(len(value)), base_local] = 0.0
    selected = np.argmax(value, axis=1)
    confidence = value[np.arange(len(value)), selected]
    active = confidence > threshold
    if components is not None:
        rows = np.arange(len(value))
        active &= components[rows, selected, 0] >= pas_floor
        active &= components[rows, selected, 1] >= pdp_floor
    if agreement_utility is not None:
        active &= agreement_utility[np.arange(len(value)), selected] > 0.0
    if fraction < 1.0:
        keep = int(np.floor(fraction * len(value)))
        if keep <= 0:
            active[:] = False
        elif np.sum(active) > keep:
            cutoff = np.argpartition(confidence, -keep)[-keep:]
            limited = np.zeros(len(value), dtype=bool)
            limited[cutoff] = True
            active &= limited
    return np.where(active, selected, base_local).astype(np.int64)


def evaluate_fold(
    fold: int,
    candidate_indices: np.ndarray,
    pseudo: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
) -> dict[str, object]:
    fold_id = actual["fold_id"].astype(np.int8)
    outer_train = fold_id != fold
    held = fold_id == fold
    scales = _site_scales(
        {**actual, "candidate_names": np.asarray(candidate_indices)},
        np.arange(len(candidate_indices)),
        outer_train,
    )
    pseudo_components, _, pseudo_base_local, _ = _labels(
        pseudo, candidate_indices, scales
    )
    actual_components, actual_errors, actual_base_local, _ = _labels(
        actual, candidate_indices, scales
    )

    pseudo_observable = np.stack(
        [pseudo["pas_target"], pseudo["pdp_target"]], axis=2
    )
    actual_observable = np.stack(
        [actual["pas_target"], actual["pdp_target"]], axis=2
    )
    pseudo_x = _point_features(
        context["features"],
        context["query_ids"],
        pseudo_observable,
        pseudo["site"].astype(np.int8),
        pseudo_base_local,
    )
    oof_features_path = ARTIFACT_DIR / "oof_point_features.npy"
    if oof_features_path.is_file():
        oof_base_features = np.load(oof_features_path)
    else:
        _, oof_base_features = build_oof_features(torch.device("cuda:0"), 8)
        np.save(oof_features_path, oof_base_features.astype(np.float32))
    actual_x = _point_features(
        oof_base_features,
        actual["query_ids"].astype(np.int64),
        actual_observable,
        actual["site"].astype(np.int8),
        actual_base_local,
    )

    pseudo_model = ExtraTreesRegressor(
        n_estimators=640,
        min_samples_leaf=10,
        max_features=0.72,
        n_jobs=-1,
        random_state=20_260_806 + fold,
    )
    pseudo_model.fit(pseudo_x, pseudo_components.reshape(len(pseudo_x), -1))
    pseudo_prediction = pseudo_model.predict(actual_x).reshape(actual_components.shape)

    held_predictions: dict[int, np.ndarray] = {}
    train_predictions: dict[int, np.ndarray] = {}
    for leaf in (8, 16, 32):
        cross_prediction = np.zeros_like(actual_components)
        for inner in np.unique(fold_id[outer_train]):
            fit = outer_train & (fold_id != inner)
            valid = outer_train & (fold_id == inner)
            model = ExtraTreesRegressor(
                n_estimators=480,
                min_samples_leaf=leaf,
                max_features=0.68,
                n_jobs=-1,
                random_state=20_261_000 + fold * 101 + int(inner),
            )
            model.fit(actual_x[fit], actual_components[fit].reshape(np.sum(fit), -1))
            cross_prediction[valid] = model.predict(actual_x[valid]).reshape(
                np.sum(valid), len(candidate_indices), 3
            )
        final_model = ExtraTreesRegressor(
            n_estimators=640,
            min_samples_leaf=leaf,
            max_features=0.68,
            n_jobs=-1,
            random_state=20_262_000 + fold * 101 + leaf,
        )
        final_model.fit(
            actual_x[outer_train],
            actual_components[outer_train].reshape(np.sum(outer_train), -1),
        )
        held_predictions[leaf] = final_model.predict(actual_x[held]).reshape(
            np.sum(held), len(candidate_indices), 3
        )
        train_predictions[leaf] = cross_prediction

    baseline = _score_selection(
        actual, actual_errors, actual_base_local, outer_train
    )
    baseline_nmse = float(baseline["nmse"])
    held_baseline = _score_selection(actual, actual_errors, actual_base_local, held)
    held_rows = np.flatnonzero(held)
    grid_parameters: list[tuple[float, float, float, int, float, float, bool]] = []
    grid_held_delta: list[float] = []
    grid_held_active: list[int] = []
    best: tuple[float, dict[str, float | int | bool], dict[str, object]] | None = None
    for leaf, actual_prediction in train_predictions.items():
        for pseudo_weight in (0.50, 0.75, 1.00):
            blended = (
                pseudo_weight * pseudo_prediction
                + (1.0 - pseudo_weight) * actual_prediction
            )
            utility = _predicted_utility(blended, baseline_nmse)
            pseudo_utility = _predicted_utility(pseudo_prediction, baseline_nmse)
            actual_utility = _predicted_utility(actual_prediction, baseline_nmse)
            for threshold in (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005):
                for fraction in (0.05, 0.10, 0.20, 0.35, 0.50):
                    for pas_floor in (-0.002, 0.0, 0.001):
                        for pdp_floor in (-0.002, 0.0):
                            for require_agreement in (False, True):
                                agreement = None
                                if require_agreement:
                                    agreement = np.minimum(
                                        pseudo_utility, actual_utility
                                    )
                                selected = _choose_actions(
                                    utility,
                                    actual_base_local,
                                    threshold,
                                    fraction,
                                    components=blended,
                                    pas_floor=pas_floor,
                                    pdp_floor=pdp_floor,
                                    agreement_utility=agreement,
                                )
                                score = _score_selection(
                                    actual, actual_errors, selected, outer_train
                                )
                                control = {
                                    "pseudo_weight": pseudo_weight,
                                    "threshold": threshold,
                                    "fraction": fraction,
                                    "actual_leaf": leaf,
                                    "pas_floor": pas_floor,
                                    "pdp_floor": pdp_floor,
                                    "require_agreement": require_agreement,
                                }
                                row = {
                                    "score": score,
                                    "active": int(
                                        np.sum(
                                            selected[outer_train]
                                            != actual_base_local[outer_train]
                                        )
                                    ),
                                }
                                if best is None or float(score["score"]) > best[0]:
                                    best = (float(score["score"]), control, row)
                                held_blended = (
                                    pseudo_weight * pseudo_prediction[held]
                                    + (1.0 - pseudo_weight) * held_predictions[leaf]
                                )
                                held_utility_grid = _predicted_utility(
                                    held_blended, baseline_nmse
                                )
                                held_agreement_grid = None
                                if require_agreement:
                                    held_agreement_grid = np.minimum(
                                        _predicted_utility(
                                            pseudo_prediction[held], baseline_nmse
                                        ),
                                        _predicted_utility(
                                            held_predictions[leaf], baseline_nmse
                                        ),
                                    )
                                held_selected_grid = _choose_actions(
                                    held_utility_grid,
                                    actual_base_local[held],
                                    threshold,
                                    fraction,
                                    components=held_blended,
                                    pas_floor=pas_floor,
                                    pdp_floor=pdp_floor,
                                    agreement_utility=held_agreement_grid,
                                )
                                full_selected_grid = actual_base_local.copy()
                                full_selected_grid[held_rows] = held_selected_grid
                                held_score_grid = _score_selection(
                                    actual, actual_errors, full_selected_grid, held
                                )
                                grid_parameters.append(
                                    (
                                        pseudo_weight,
                                        threshold,
                                        fraction,
                                        leaf,
                                        pas_floor,
                                        pdp_floor,
                                        require_agreement,
                                    )
                                )
                                grid_held_delta.append(
                                    float(
                                        held_score_grid["score"]
                                        - held_baseline["score"]
                                    )
                                )
                                grid_held_active.append(
                                    int(
                                        np.sum(
                                            held_selected_grid
                                            != actual_base_local[held]
                                        )
                                    )
                                )
    if best is None:
        raise RuntimeError("no E49 control selected")

    control = best[1]
    pseudo_weight = float(control["pseudo_weight"])
    threshold = float(control["threshold"])
    fraction = float(control["fraction"])
    leaf = int(control["actual_leaf"])
    held_prediction = (
        pseudo_weight * pseudo_prediction[held]
        + (1.0 - pseudo_weight) * held_predictions[leaf]
    )
    held_utility = _predicted_utility(held_prediction, baseline_nmse)
    held_pseudo_utility = _predicted_utility(
        pseudo_prediction[held], baseline_nmse
    )
    held_actual_utility = _predicted_utility(
        held_predictions[leaf], baseline_nmse
    )
    held_agreement = None
    if bool(control["require_agreement"]):
        held_agreement = np.minimum(held_pseudo_utility, held_actual_utility)
    held_selected_local = _choose_actions(
        held_utility,
        actual_base_local[held],
        threshold,
        fraction,
        components=held_prediction,
        pas_floor=float(control["pas_floor"]),
        pdp_floor=float(control["pdp_floor"]),
        agreement_utility=held_agreement,
    )
    selected = actual_base_local.copy()
    selected[held_rows] = held_selected_local
    held_score = _score_selection(actual, actual_errors, selected, held)
    grid_path = _fold_paths(fold)["control_grid"]
    np.savez_compressed(
        grid_path,
        parameters=np.asarray(grid_parameters, dtype=np.float64),
        held_delta=np.asarray(grid_held_delta, dtype=np.float64),
        held_active=np.asarray(grid_held_active, dtype=np.int16),
        held_count=np.int16(np.sum(held)),
    )

    oracle_selected = actual_base_local.copy()
    held_truth_utility = _predicted_utility(actual_components[held], baseline_nmse)
    oracle_selected[held_rows] = np.argmax(held_truth_utility, axis=1)
    oracle = _score_selection(actual, actual_errors, oracle_selected, held)
    return {
        "fold": fold,
        "candidate_count": int(len(candidate_indices)),
        "candidates": [str(value) for value in actual["candidate_names"]],
        "pseudo_count": int(len(pseudo_x)),
        "feature_count": int(pseudo_x.shape[1]),
        "selected_control": control,
        "control_fit": best[2],
        "held_baseline": held_baseline,
        "held": held_score,
        "held_delta": float(held_score["score"] - held_baseline["score"]),
        "held_active": int(
            np.sum(held_selected_local != actual_base_local[held])
        ),
        "held_oracle": oracle,
        "held_oracle_delta": float(oracle["score"] - held_baseline["score"]),
        "control_grid": PATHS.relative(grid_path),
    }


def aggregate_shared_controls(records: list[dict[str, object]]) -> dict[str, object]:
    grids = [
        np.load(PATHS.root / str(record["control_grid"]), allow_pickle=False)
        for record in records
    ]
    parameters = grids[0]["parameters"]
    if any(not np.array_equal(grid["parameters"], parameters) for grid in grids[1:]):
        raise RuntimeError("E49 control grids are not aligned across folds")
    delta = np.stack([grid["held_delta"] for grid in grids], axis=0)
    count = np.asarray([int(grid["held_count"]) for grid in grids], dtype=np.float64)
    nested_delta = np.empty(len(grids), dtype=np.float64)
    nested_index = np.empty(len(grids), dtype=np.int32)
    for held_fold in range(len(grids)):
        fit = np.arange(len(grids)) != held_fold
        mean = np.sum(delta[fit] * count[fit, None], axis=0) / np.sum(count[fit])
        positive = np.sum(delta[fit] > 0.0, axis=0)
        robust = mean - 0.35 * np.std(delta[fit], axis=0)
        robust = np.where(positive >= 3, robust, -np.inf)
        winner = int(np.argmax(robust))
        nested_index[held_fold] = winner
        nested_delta[held_fold] = delta[held_fold, winner]
    full_mean = np.sum(delta * count[:, None], axis=0) / np.sum(count)
    full_positive = np.sum(delta > 0.0, axis=0)
    full_robust = full_mean - 0.35 * np.std(delta, axis=0)
    full_robust = np.where(full_positive >= 4, full_robust, -np.inf)
    full_winner = int(np.argmax(full_robust))

    def decode(index: int) -> dict[str, float | int | bool]:
        row = parameters[index]
        return {
            "pseudo_weight": float(row[0]),
            "threshold": float(row[1]),
            "fraction": float(row[2]),
            "actual_leaf": int(row[3]),
            "pas_floor": float(row[4]),
            "pdp_floor": float(row[5]),
            "require_agreement": bool(row[6]),
        }

    return {
        "nested_fold_delta": nested_delta.tolist(),
        "nested_weighted_delta": float(np.sum(nested_delta * count) / np.sum(count)),
        "nested_positive_folds": int(np.sum(nested_delta > 0.0)),
        "nested_controls": [decode(int(index)) for index in nested_index],
        "final_control": decode(full_winner),
        "final_fold_delta": delta[:, full_winner].tolist(),
        "final_weighted_delta": float(full_mean[full_winner]),
        "final_positive_folds": int(full_positive[full_winner]),
    }


def run_fold(
    fold: int,
    device: torch.device,
    batch_size: int,
    candidate_count: int,
    sample_limit: int | None,
) -> dict[str, object]:
    started = time.time()
    paths = _fold_paths(fold)
    loaded = np.load(V4_METRICS, allow_pickle=False)
    full_actual = {key: loaded[key] for key in loaded.files}
    fit_mask = full_actual["fold_id"].astype(np.int8) != fold
    candidate_indices = greedy_candidates(full_actual, fit_mask, candidate_count)
    paths["selection"].write_text(
        json.dumps(
            {
                "fold": fold,
                "candidate_indices": candidate_indices.tolist(),
                "candidate_names": full_actual["candidate_names"][candidate_indices].tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    context = _pseudo_context(fold, device, sample_limit)
    _build_pseudo_base(fold, context, paths["base"])
    models = _fit_group_calibration(fold, device, paths["calibration"])
    _build_pseudo_targets(fold, context, models, paths)
    pseudo = _generate_pseudo_metrics(
        fold, candidate_indices, context, paths, device, batch_size
    )
    actual = _actual_subset(full_actual, candidate_indices)
    actual["candidate_names"] = full_actual["candidate_names"][candidate_indices]
    result = evaluate_fold(fold, candidate_indices, pseudo, context, actual)
    result["seconds"] = float(time.time() - started)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "evaluate"), default="pilot")
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--sample-limit", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    folds = (
        [args.fold]
        if args.phase == "pilot"
        else [int(value) for value in args.folds.split(",") if value.strip()]
    )
    records = [
        run_fold(
            fold,
            device,
            args.batch_size,
            args.candidate_count,
            args.sample_limit,
        )
        for fold in folds
    ]
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "Projection-action utility becomes learnable when the 500 real spatial-hole "
            "labels are augmented with thousands of fold-isolated pseudo-hole labels built "
            "from official H, map geometry and the same V4 projection mechanism."
        ),
        "parameters": {
            "folds": folds,
            "candidate_count": args.candidate_count,
            "sample_limit": args.sample_limit,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "records": records,
        "shared_control": aggregate_shared_controls(records)
        if len(records) == 5
        else None,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
