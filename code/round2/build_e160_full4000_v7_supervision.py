#!/usr/bin/env python3
"""Build the strict V7-domain supervision foundation for Round2-E160."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from analyze_v4_predictability_calibrated_groups import (
    GROUPED_PAS,
    GROUPED_PDP,
    PAS_COMPONENTS,
    PDP_COMPONENTS,
    blend_group_pas,
    blend_group_pdp,
    encode,
    globally_shrunk_targets,
    normalize_group_amplitude,
    pca_basis,
    predict_grouped,
)
from analyze_v7_complex_carrier_stacking import StackConfig, _solve_weights
from analyze_v4_exact_joint_projection import continue_trajectory, initialize_trajectory
from data import load_positions
from official_metrics import (
    OfficialMetricAccumulator,
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics,
)
from project_paths import PATHS
from run_foundation_benchmark import load_or_compute_power
from run_rbf_spectral_field import ACTIONS, OOF_META, optimal_scale, scaled_error
from run_v6_regime_component_lifecycle import (
    REGIME_CACHE,
    REGIME_NAMES,
    TARGET_PATHS,
    _candidate_from_name,
    apply_candidate,
)
from analyze_v7_stacked_lifecycle_overlay import (
    OOF_CURRENT,
    OOF_SCALED,
    SELECTION as LIFECYCLE_SELECTION,
)
from build_v7_complex_carrier_bank import carrier_configs
from build_v3_anchored_neural_rbf import TRAIN_AD, TRAIN_PAS, TRAIN_PDP, spectral_target
from carrier_calibration import estimate_carrier
from run_v3_anchored_neural_rbf import (
    DESCRIPTORS,
    GEOMETRY,
    PAS_SIZE,
    AnchoredWeightCorrector,
    Config as V3Config,
    power_cosine,
    set_seed,
    standardize_geometry,
)
from run_v4_calibrated_exact_projection import (
    EXACT_PATTERN,
    PAS_CALIBRATION_STRENGTH,
    PAS_GROUP_BLEND,
    PDP_CALIBRATION_STRENGTH,
    PDP_GROUP_BLEND,
    SELECTION as V4_SELECTION,
    apply_calibration,
    calibration_slope,
    mode_targets,
)
from train_e150_v7_pathwise_carrier_operator import (
    batched_carrier_predictions,
    strict_neighbors,
)
from train_full_path_pdp_link_gate import FOLD_IDS


VERSION = "Round2-E160-full4000-v7-supervision"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e160_full4000_v7_supervision"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e160_full4000_v7_supervision"
CANONICAL_COMPLETE = ARTIFACT_DIR / "canonical_v7_complete.npy"
FULL_RAW_STACK = ARTIFACT_DIR / "full4000_v7_raw_stack.npy"
FULL_STACK_WEIGHTS = ARTIFACT_DIR / "full4000_v7_stack_weights.npz"
FULL_V3_WEIGHTS = ARTIFACT_DIR / "full4000_v3_corrected_weights.npz"
FULL_TARGET_AD = ARTIFACT_DIR / "full4000_v7_target_ad_f16.npy"
FULL_TARGET_PAS = ARTIFACT_DIR / "full4000_v7_target_pas_f16.npy"
FULL_TARGET_PDP = ARTIFACT_DIR / "full4000_v7_target_pdp_f16.npy"
FULL_CALIBRATED_PAS = ARTIFACT_DIR / "full4000_v7_calibrated_pas_f16.npy"
FULL_CALIBRATED_PDP = ARTIFACT_DIR / "full4000_v7_calibrated_pdp_f16.npy"
FULL_CORE = ARTIFACT_DIR / "full4000_v7_core.npy"
REPORT = ANALYSIS_DIR / "full4000_v7_supervision.json"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
STACK_SELECTION = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_stacking"
    / "complex_stack_selection.json"
)

EXPECTED_CORE_SCORE = 0.6443140570336366
EXPECTED_COMPLETE_SCORE = 0.6444563457343335
IDENTITY_TOLERANCE = 5e-6
V3_EPOCHS = 6
V3_NEIGHBORS = 64
V3_SMOOTHING = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("identity", "carriers", "core", "clean_carriers", "clean_core"),
        default="identity",
        help="Run the identity gate or build the strict full-4000 raw V7 stack.",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--v3-batch-size", type=int, default=128)
    parser.add_argument("--rbf-batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def score_memmap(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> dict[str, float | int]:
    accumulator = OfficialMetricAccumulator()
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        accumulator.add(
            np.asarray(prediction[start:end], dtype=np.complex64),
            np.asarray(truth[query_ids[start:end]], dtype=np.complex64),
        )
    return accumulator.result()


def nested_lifecycle_policy(
    selection: dict[str, object],
) -> tuple[str, dict[tuple[int, int, int], str]]:
    winner = selection.get("winner")
    if not isinstance(winner, dict):
        raise RuntimeError("V7 lifecycle selection has no winner")
    stage = str(winner["stage"])
    policy: dict[tuple[int, int, int], str] = {}
    for row in winner["nested_decisions"]:
        key = (
            int(row["held_fold"]),
            int(row["site"]),
            REGIME_NAMES.index(str(row["regime"])),
        )
        policy[key] = str(row["candidate"])
    return stage, policy


def apply_nested_lifecycle(
    device: torch.device,
    batch_size: int,
    core: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    regime: np.ndarray,
) -> dict[str, object]:
    selection = json.loads(LIFECYCLE_SELECTION.read_text(encoding="utf-8"))
    stage, policy = nested_lifecycle_policy(selection)
    current = np.load(OOF_CURRENT, mmap_mode="r")
    target = np.load(TARGET_PATHS[stage][0], mmap_mode="r")
    output = np.lib.format.open_memmap(
        CANONICAL_COMPLETE,
        mode="w+",
        dtype=np.complex64,
        shape=core.shape,
    )

    action_counts: dict[str, int] = {}
    for start in range(0, len(core), batch_size):
        end = min(start + batch_size, len(core))
        block = np.array(core[start:end], dtype=np.complex64, copy=True)
        block_fold = fold_id[start:end]
        block_site = site[start:end]
        block_regime = regime[start:end]
        current_tensor = torch.as_tensor(
            np.asarray(current[start:end], dtype=np.float32), device=device
        )
        target_tensor = torch.as_tensor(
            np.asarray(target[start:end], dtype=np.float32), device=device
        )
        for fold_value in np.unique(block_fold):
            for site_value in np.unique(block_site):
                for regime_value in np.unique(block_regime):
                    rows = np.flatnonzero(
                        (block_fold == fold_value)
                        & (block_site == site_value)
                        & (block_regime == regime_value)
                    )
                    if len(rows) == 0:
                        continue
                    name = policy.get(
                        (int(fold_value), int(site_value), int(regime_value)),
                        "identity",
                    )
                    action_counts[name] = action_counts.get(name, 0) + int(len(rows))
                    if name == "identity":
                        continue
                    with torch.inference_mode():
                        prediction = apply_candidate(
                            torch.as_tensor(block[rows], device=device),
                            current_tensor[rows],
                            target_tensor[rows],
                            block_regime[rows],
                            block_site[rows],
                            _candidate_from_name(name),
                        )
                    block[rows] = prediction.cpu().numpy().astype(np.complex64)
        output[start:end] = block
        print(f"E160 canonical lifecycle {end}/{len(core)}", flush=True)
    output.flush()

    finite = True
    for start in range(0, len(output), batch_size):
        value = np.asarray(output[start : start + batch_size])
        finite &= bool(np.isfinite(value.real).all() and np.isfinite(value.imag).all())
    if not finite:
        raise ValueError("E160 canonical lifecycle output contains non-finite values")
    return {
        "path": PATHS.relative(CANONICAL_COMPLETE),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "stage": stage,
        "action_counts": action_counts,
    }


def gpu_report(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"device": str(device), "cuda": False}
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(device),
        "cuda": True,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def identity_gate(device: torch.device, batch_size: int) -> dict[str, object]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    core = np.load(OOF_SCALED, mmap_mode="r")

    core_started = time.time()
    core_score = score_memmap(core, truth, query_ids, batch_size)
    core_seconds = time.time() - core_started

    lifecycle_started = time.time()
    output = apply_nested_lifecycle(device, batch_size, core, fold_id, site, regime)
    lifecycle_seconds = time.time() - lifecycle_started
    complete = np.load(CANONICAL_COMPLETE, mmap_mode="r")
    complete_started = time.time()
    complete_score = score_memmap(complete, truth, query_ids, batch_size)
    complete_score_seconds = time.time() - complete_started

    core_error = abs(float(core_score["score"]) - EXPECTED_CORE_SCORE)
    complete_error = abs(float(complete_score["score"]) - EXPECTED_COMPLETE_SCORE)
    passed = core_error <= IDENTITY_TOLERANCE and complete_error <= IDENTITY_TOLERANCE
    return {
        "passed": passed,
        "tolerance": IDENTITY_TOLERANCE,
        "expected": {
            "core_score": EXPECTED_CORE_SCORE,
            "complete_score": EXPECTED_COMPLETE_SCORE,
        },
        "core": {
            "official": core_score,
            "absolute_score_error": core_error,
            "seconds": core_seconds,
            "path": PATHS.relative(OOF_SCALED),
        },
        "complete": {
            "official": complete_score,
            "absolute_score_error": complete_error,
            "generation_seconds": lifecycle_seconds,
            "score_seconds": complete_score_seconds,
            **output,
        },
    }


def stack_config() -> StackConfig:
    selection = json.loads(STACK_SELECTION.read_text(encoding="utf-8"))
    record = selection["posterior_best"]["config"]
    config = StackConfig(
        scope=str(record["scope"]),
        ports=str(record["ports"]),
        ridge=float(record["ridge"]),
        sum_penalty=float(record["sum_penalty"]),
    )
    if config.scope != "site" or config.ports != "port8":
        raise RuntimeError(f"E160 requires the frozen V7 site/port8 stack, got {config}")
    return config


def configured_carriers(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    bs_positions: np.ndarray,
    fit_ids: np.ndarray,
    point_power: np.ndarray,
    fold: int,
) -> tuple[dict[int, list], dict[str, object]]:
    seed = 20_260_805 + fold * 101
    calibration = estimate_carrier(
        train_channel,
        train_positions,
        train_site,
        bs_positions,
        fit_ids,
        point_power,
        max_pairs_per_site=500,
        seed=seed,
    )
    base_beta = np.asarray(
        [calibration[int(site_value)].beta for site_value in sorted(calibration)],
        dtype=np.float64,
    )
    selected_beta = {
        int(site_value): float(calibration[int(site_value)].beta)
        for site_value in calibration
    }
    selected_gamma = {
        int(site_value): float(calibration[int(site_value)].gamma)
        for site_value in calibration
    }
    robustness: dict[str, object] = {
        "triggered": False,
        "trigger_beta_spread": float(np.ptp(base_beta)),
        "threshold": 5.0,
    }
    if float(np.ptp(base_beta)) > 5.0:
        repeated = [calibration]
        for retry in range(1, 7):
            repeated.append(
                estimate_carrier(
                    train_channel,
                    train_positions,
                    train_site,
                    bs_positions,
                    fit_ids,
                    point_power,
                    max_pairs_per_site=500,
                    seed=seed + retry * 10_007,
                )
            )
        samples: dict[str, object] = {}
        for site_value in sorted(calibration):
            site_value = int(site_value)
            beta_values = np.asarray(
                [value[site_value].beta for value in repeated], dtype=np.float64
            )
            gamma_values = np.asarray(
                [value[site_value].gamma for value in repeated], dtype=np.float64
            )
            median_beta = float(np.median(beta_values))
            if abs(selected_beta[site_value] - median_beta) > 2.0:
                selected_beta[site_value] = median_beta
                selected_gamma[site_value] = float(np.median(gamma_values))
            samples[str(site_value)] = {
                "beta": beta_values.tolist(),
                "gamma": gamma_values.tolist(),
                "selected_beta": selected_beta[site_value],
                "selected_gamma": selected_gamma[site_value],
            }
        robustness = {
            **robustness,
            "triggered": True,
            "resample_count": len(repeated),
            "samples": samples,
        }
    templates = carrier_configs()
    configs = {
        int(site_value): [
            replace(
                template,
                beta=selected_beta[int(site_value)],
                gamma=selected_gamma[int(site_value)],
            )
            for template in templates
        ]
        for site_value in np.unique(train_site[fit_ids])
    }
    report = {
        "sites": {
            str(site_value): {
                **calibration[int(site_value)].to_dict(),
                "selected_beta": selected_beta[int(site_value)],
                "selected_gamma": selected_gamma[int(site_value)],
            }
            for site_value in sorted(configs)
        },
        "robustness": robustness,
    }
    return configs, report


def carrier_port_view(carriers: torch.Tensor) -> torch.Tensor:
    batch, count = carriers.shape[:2]
    return (
        carriers.reshape(batch, count, 2, 16, 8, 4, 192)
        .permute(0, 2, 5, 1, 3, 4, 6)
        .reshape(batch, 8, count, -1)
    )


def truth_port_view(channel: torch.Tensor) -> torch.Tensor:
    batch = len(channel)
    return (
        channel.reshape(batch, 2, 16, 8, 4, 192)
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(batch, 8, -1)
    )


def accumulate_stack_statistics(
    device_channel: torch.Tensor,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    fit_ids: np.ndarray,
    configs_by_site: dict[int, list],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    distances, neighbors = strict_neighbors(positions, site, fit_ids, fit_ids, 32)
    allowed = np.zeros(len(site), dtype=bool)
    allowed[fit_ids] = True
    if not bool(allowed[neighbors].all()):
        raise RuntimeError("strict stack-fit neighbors escaped the outer fit set")
    carrier_count = len(next(iter(configs_by_site.values())))
    gram = np.zeros((2, 8, carrier_count, carrier_count), dtype=np.complex128)
    cross = np.zeros((2, 8, carrier_count), dtype=np.complex128)
    started = time.time()
    for start in range(0, len(fit_ids), batch_size):
        end = min(start + batch_size, len(fit_ids))
        query = fit_ids[start:end]
        prediction = batched_carrier_predictions(
            device_channel,
            neighbors[start:end],
            positions[neighbors[start:end]],
            positions[query],
            distances[start:end],
            bs_positions,
            site[query],
            configs_by_site,
        )
        candidate = carrier_port_view(prediction)
        query_tensor = torch.as_tensor(query, device=device_channel.device)
        truth = truth_port_view(device_channel[query_tensor])
        for site_value in np.unique(site[query]):
            local = torch.as_tensor(
                np.flatnonzero(site[query] == site_value),
                dtype=torch.long,
                device=device_channel.device,
            )
            local_candidate = candidate[local]
            local_truth = truth[local]
            batch_gram = torch.einsum(
                "bgcd,bged->gce", local_candidate.conj(), local_candidate
            )
            batch_cross = torch.einsum(
                "bgcd,bgd->gc", local_candidate.conj(), local_truth
            )
            gram[int(site_value)] += batch_gram.cpu().numpy().astype(np.complex128)
            cross[int(site_value)] += batch_cross.cpu().numpy().astype(np.complex128)
        del prediction, candidate, truth
        if end % (batch_size * 10) == 0 or end == len(fit_ids):
            print(f"E160 stack statistics {end}/{len(fit_ids)}", flush=True)
    return gram, cross, {
        "query_count": int(len(fit_ids)),
        "neighbor_count": 32,
        "seconds": float(time.time() - started),
    }


def solve_site_port_weights(
    gram: np.ndarray,
    cross: np.ndarray,
    config: StackConfig,
) -> dict[int, np.ndarray]:
    weights: dict[int, np.ndarray] = {}
    for site_value in range(2):
        weights[site_value] = np.stack(
            [
                _solve_weights(gram[site_value, port], cross[site_value, port], config)
                for port in range(8)
            ]
        ).astype(np.complex64)
    return weights


def combine_carriers(
    carriers: torch.Tensor,
    query_site: np.ndarray,
    weights: dict[int, np.ndarray],
) -> torch.Tensor:
    candidate = carrier_port_view(carriers)
    local_weights = np.stack([weights[int(value)] for value in query_site]).astype(
        np.complex64
    )
    combined = torch.einsum(
        "bgcd,bgc->bgd",
        candidate,
        torch.as_tensor(local_weights, device=carriers.device),
    )
    batch = len(carriers)
    return (
        combined.reshape(batch, 2, 4, 16, 8, 192)
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(batch, 256, 4, 192)
        .to(torch.complex64)
    )


def build_full_raw_stack(
    device: torch.device,
    batch_size: int,
    force: bool,
    require_identity: bool = True,
) -> dict[str, object]:
    if require_identity:
        if not REPORT.is_file():
            raise RuntimeError("E160 identity report is missing")
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        if not bool(previous.get("identity_gate", {}).get("passed")):
            raise RuntimeError("E160 identity gate has not passed")
    if FULL_RAW_STACK.is_file() and not force:
        cached = np.load(FULL_RAW_STACK, mmap_mode="r")
        expected = np.load(TRAIN_CHANNEL, mmap_mode="r").shape
        if cached.shape == expected and cached.dtype == np.complex64:
            raise FileExistsError(
                f"{FULL_RAW_STACK} already exists; pass --force to rebuild"
            )

    setup, positions, _, site, _ = load_positions()
    fold_id = np.load(FOLD_IDS, allow_pickle=False).astype(np.int8)
    if fold_id.shape != (len(positions),) or sorted(np.unique(fold_id)) != list(range(5)):
        raise ValueError("invalid full-4000 broad fold assignment")
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = load_or_compute_power(train_channel, 16)
    load_started = time.time()
    device_channel = torch.tensor(
        np.asarray(train_channel), dtype=torch.complex64, device=device
    )
    load_seconds = time.time() - load_started
    output = np.lib.format.open_memmap(
        FULL_RAW_STACK,
        mode="w+",
        dtype=np.complex64,
        shape=train_channel.shape,
    )
    config = stack_config()
    arrays: dict[str, np.ndarray] = {"fold_id": fold_id}
    folds: list[dict[str, object]] = []
    total_started = time.time()

    for outer in range(5):
        fold_started = time.time()
        fit_ids = np.flatnonzero(fold_id != outer).astype(np.int64)
        held_ids = np.flatnonzero(fold_id == outer).astype(np.int64)
        configs_by_site, calibration = configured_carriers(
            train_channel,
            positions,
            site,
            setup.bs_positions,
            fit_ids,
            point_power,
            outer,
        )
        gram, cross, statistics_report = accumulate_stack_statistics(
            device_channel,
            positions,
            site,
            setup.bs_positions,
            fit_ids,
            configs_by_site,
            batch_size,
        )
        weights = solve_site_port_weights(gram, cross, config)
        for site_value, value in weights.items():
            arrays[f"fold{outer}_site{site_value}"] = value

        distances, neighbors = strict_neighbors(
            positions, site, fit_ids, held_ids, 32
        )
        if np.any(fold_id[neighbors] == outer):
            raise RuntimeError(f"outer {outer} held carrier used a held-fold donor")
        prediction_started = time.time()
        for start in range(0, len(held_ids), batch_size):
            end = min(start + batch_size, len(held_ids))
            query = held_ids[start:end]
            carriers = batched_carrier_predictions(
                device_channel,
                neighbors[start:end],
                positions[neighbors[start:end]],
                positions[query],
                distances[start:end],
                setup.bs_positions,
                site[query],
                configs_by_site,
            )
            output[query] = combine_carriers(
                carriers, site[query], weights
            ).cpu().numpy()
            if end % (batch_size * 5) == 0 or end == len(held_ids):
                print(
                    f"E160 raw stack outer={outer} {end}/{len(held_ids)}",
                    flush=True,
                )
            del carriers
        output.flush()
        folds.append(
            {
                "fold": outer,
                "fit_count": int(len(fit_ids)),
                "held_count": int(len(held_ids)),
                "carrier_calibration": calibration,
                "stack_statistics": statistics_report,
                "prediction_seconds": float(time.time() - prediction_started),
                "weight_norm": {
                    str(site_value): float(np.linalg.norm(value))
                    for site_value, value in weights.items()
                },
                "seconds": float(time.time() - fold_started),
            }
        )
        print(f"E160 raw stack outer={outer} complete", flush=True)

    np.savez_compressed(FULL_STACK_WEIGHTS, **arrays)
    del device_channel
    if device.type == "cuda":
        torch.cuda.empty_cache()
    finite = True
    for start in range(0, len(output), max(batch_size, 8)):
        value = np.asarray(output[start : start + max(batch_size, 8)])
        finite &= bool(np.isfinite(value.real).all() and np.isfinite(value.imag).all())
    if not finite:
        raise ValueError("E160 full raw stack contains non-finite values")
    score_started = time.time()
    official = score_memmap(
        np.load(FULL_RAW_STACK, mmap_mode="r"),
        train_channel,
        np.arange(len(train_channel), dtype=np.int64),
        max(batch_size, 8),
    )
    return {
        "path": PATHS.relative(FULL_RAW_STACK),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "weights": PATHS.relative(FULL_STACK_WEIGHTS),
        "official": official,
        "fold_counts": np.bincount(fold_id, minlength=5).astype(int).tolist(),
        "gpu_load_seconds": float(load_seconds),
        "score_seconds": float(time.time() - score_started),
        "folds": folds,
        "seconds": float(time.time() - total_started),
    }


def context_path(outer: int, kind: str) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_{kind}_v3_context.npz"


def solve_rbf_batch(
    donor_xy: np.ndarray,
    query_xy: np.ndarray,
    smoothing: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    count = donor_xy.shape[-2]
    flat_donor = donor_xy.reshape(-1, count, 2)
    flat_query = np.broadcast_to(
        query_xy[:, None, :], (*donor_xy.shape[:2], 2)
    ).reshape(-1, 2)
    output = np.empty((len(flat_donor), count), dtype=np.float32)
    for start in range(0, len(flat_donor), batch_size):
        end = min(start + batch_size, len(flat_donor))
        donor = torch.as_tensor(
            flat_donor[start:end], dtype=torch.float64, device=device
        )
        query = torch.as_tensor(
            flat_query[start:end], dtype=torch.float64, device=device
        )
        pairwise = torch.cdist(donor, donor)
        system = torch.zeros(
            (len(donor), count + 1, count + 1),
            dtype=torch.float64,
            device=device,
        )
        system[:, :count, :count] = -pairwise
        diagonal = torch.arange(count, device=device)
        system[:, diagonal, diagonal] += smoothing
        system[:, :count, count] = 1.0
        system[:, count, :count] = 1.0
        right = torch.empty(
            (len(donor), count + 1), dtype=torch.float64, device=device
        )
        right[:, :count] = -torch.linalg.vector_norm(
            donor - query[:, None], dim=2
        )
        right[:, count] = 1.0
        output[start:end] = torch.linalg.solve(system, right)[
            :, :count
        ].float().cpu().numpy()
    return output.reshape(*donor_xy.shape[:2], count)


def build_v3_context(
    outer: int,
    kind: str,
    positions: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    guards: tuple[float, ...],
    device: torch.device,
    rbf_batch_size: int,
    force: bool,
) -> dict[str, np.ndarray]:
    path = context_path(outer, kind)
    if path.is_file() and not force:
        cached = np.load(path, allow_pickle=False)
        if np.array_equal(cached["query_ids"], query_ids) and np.allclose(
            cached["guards"], guards
        ):
            return {key: cached[key] for key in cached.files}

    neighbor_count = V3_NEIGHBORS
    candidate_count = max(320, neighbor_count + 1)
    donor_ids = np.empty(
        (len(query_ids), len(guards), neighbor_count), dtype=np.int32
    )
    distances = np.empty_like(donor_ids, dtype=np.float32)
    fit_membership = np.zeros(len(positions), dtype=bool)
    fit_membership[fit_ids] = True
    for site_value in np.unique(site[query_ids]):
        rows = np.flatnonzero(site[query_ids] == site_value)
        bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
        count = min(candidate_count, len(bank))
        raw_distance, local = cKDTree(positions[bank, :2]).query(
            positions[query_ids[rows], :2], k=count, workers=-1
        )
        raw_ids = bank[local]
        for local_row, destination in enumerate(rows):
            query_id = int(query_ids[destination])
            for guard_index, guard in enumerate(guards):
                keep = raw_distance[local_row] >= guard
                if fit_membership[query_id]:
                    keep &= raw_ids[local_row] != query_id
                chosen = raw_ids[local_row, keep][:neighbor_count]
                chosen_distance = raw_distance[local_row, keep][:neighbor_count]
                if len(chosen) != neighbor_count:
                    raise RuntimeError(
                        f"outer={outer} query={query_id} guard={guard} "
                        f"has only {len(chosen)} V3 donors"
                    )
                donor_ids[destination, guard_index] = chosen
                distances[destination, guard_index] = chosen_distance
    if not bool(fit_membership[donor_ids].all()):
        raise RuntimeError(f"outer {outer} V3 context escaped the fit set")
    donor_xy = positions[donor_ids, :2].astype(np.float64)
    weights = solve_rbf_batch(
        donor_xy,
        positions[query_ids, :2].astype(np.float64),
        V3_SMOOTHING,
        device,
        rbf_batch_size,
    )
    np.savez_compressed(
        path,
        query_ids=query_ids.astype(np.int32),
        donor_ids=donor_ids,
        weights=weights.astype(np.float32),
        distances=distances,
        guards=np.asarray(guards, dtype=np.float32),
    )
    return {
        "query_ids": query_ids.astype(np.int32),
        "donor_ids": donor_ids,
        "weights": weights.astype(np.float32),
        "distances": distances,
        "guards": np.asarray(guards, dtype=np.float32),
    }


def train_v3_outer(
    outer: int,
    fit_ids: np.ndarray,
    train_context: dict[str, np.ndarray],
    geometry_numpy: np.ndarray,
    positions_numpy: np.ndarray,
    spectra_numpy: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[AnchoredWeightCorrector, dict[str, object], dict[str, torch.Tensor]]:
    config = V3Config(epochs=V3_EPOCHS, batch_size=batch_size)
    set_seed(config.seed + outer * 1009 + 160_000)
    standardized, feature_mean, feature_std = standardize_geometry(
        geometry_numpy, fit_ids
    )
    geometry = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    positions = torch.as_tensor(
        positions_numpy[:, :2], dtype=torch.float32, device=device
    )
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    context = {
        key: torch.as_tensor(value, device=device)
        for key, value in train_context.items()
        if key in ("query_ids", "donor_ids", "weights", "distances")
    }
    model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    rng = np.random.default_rng(config.seed + outer * 1009 + 160_000)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, V3_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_context["query_ids"]))
        losses: list[float] = []
        pas_values: list[float] = []
        pdp_values: list[float] = []
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guard = torch.as_tensor(
                rng.integers(0, len(config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            query = context["query_ids"][rows]
            donors = context["donor_ids"][rows, guard]
            base = context["weights"][rows, guard]
            distance = context["distances"][rows, guard]
            pas, pdp, diagnostics = model(
                geometry[query],
                positions[query],
                geometry[donors],
                positions[donors],
                spectra[donors],
                distance,
                base,
            )
            pas_cosine = power_cosine(pas, spectra[query, :PAS_SIZE])
            pdp_cosine = power_cosine(pdp, spectra[query, PAS_SIZE:])
            correction = diagnostics["correction"]
            regularization = config.correction_regularization * (
                correction.square().sum(dim=1).mean()
                * config.neighbors
                / max(config.correction_l1_bound**2, 1e-8)
            )
            loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean())
            loss = loss + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pas_values.append(float(pas_cosine.mean().detach().cpu()))
            pdp_values.append(float(pdp_cosine.mean().detach().cpu()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "pas": float(np.mean(pas_values)),
            "pdp": float(np.mean(pdp_values)),
        }
        history.append(row)
        print(f"E160 V3 outer={outer} epoch={epoch}/{V3_EPOCHS} {row}", flush=True)
    checkpoint = ARTIFACT_DIR / f"outer{outer}_v3_epoch6.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.__dict__,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "epoch": V3_EPOCHS,
        },
        checkpoint,
    )
    return model, {
        "checkpoint": PATHS.relative(checkpoint),
        "fit_count": int(len(fit_ids)),
        "train_count": int(len(train_context["query_ids"])),
        "history": history,
    }, {"geometry": geometry, "positions": positions, "spectra": spectra}


@torch.inference_mode()
def predict_v3_weights(
    model: AnchoredWeightCorrector,
    context: dict[str, np.ndarray],
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    donor_ids = context["donor_ids"][:, 0].astype(np.int64)
    base_weights = context["weights"][:, 0].astype(np.float32)
    distances = context["distances"][:, 0].astype(np.float32)
    query_ids = context["query_ids"].astype(np.int64)
    pas = np.empty_like(base_weights)
    pdp = np.empty_like(base_weights)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        query = torch.as_tensor(query_ids[start:end], device=device)
        donors = torch.as_tensor(donor_ids[start:end], device=device)
        _, _, diagnostics = model(
            tensors["geometry"][query],
            tensors["positions"][query],
            tensors["geometry"][donors],
            tensors["positions"][donors],
            tensors["spectra"][donors],
            torch.as_tensor(distances[start:end], device=device),
            torch.as_tensor(base_weights[start:end], device=device),
        )
        pas[start:end] = diagnostics["pas_weights"].cpu().numpy()
        pdp[start:end] = diagnostics["pdp_weights"].cpu().numpy()
    return pas, pdp


def fit_group_models(
    amplitude: np.ndarray,
    prediction: np.ndarray,
    fit_mask: np.ndarray,
    site: np.ndarray,
    components: int,
    device: torch.device,
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    flat = np.asarray(amplitude, dtype=np.float32).reshape(len(amplitude), -1)
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for site_value in np.unique(site):
        rows = np.flatnonzero(fit_mask & (site == site_value))
        mean, vectors = pca_basis(flat[rows], components, device, seed + int(site_value) * 1009)
        predicted = encode(prediction[rows].reshape(len(rows), -1), mean, vectors)
        truth = encode(flat[rows], mean, vectors)
        denominator = np.sum(predicted * predicted, axis=0, dtype=np.float64)
        slope = np.divide(
            np.sum(predicted * truth, axis=0, dtype=np.float64),
            np.maximum(denominator, 1e-12),
        )
        models[int(site_value)] = (
            mean.astype(np.float32),
            vectors.astype(np.float32),
            np.clip(slope, 0.0, 1.25).astype(np.float32),
        )
    return models


def target_arrays(
    donor_ids: np.ndarray,
    pas_weights: np.ndarray,
    pdp_weights: np.ndarray,
    grouped_pas: np.ndarray,
    grouped_pdp: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def interpolate(
        cache: np.ndarray,
        weights: np.ndarray,
        axes: int | tuple[int, ...],
    ) -> np.ndarray:
        selected = donor_ids[rows]
        unique, inverse = np.unique(selected, return_inverse=True)
        amplitude = np.sqrt(
            np.maximum(np.asarray(cache[unique], dtype=np.float32), 0.0)
        )
        inverse = inverse.reshape(selected.shape)
        values = []
        for local, row in enumerate(rows):
            mixed = np.maximum(
                np.tensordot(
                    weights[row].astype(np.float32),
                    amplitude[inverse[local]],
                    axes=(0, 0),
                ),
                0.0,
            )
            power = mixed * mixed
            total = power.sum(axis=axes, keepdims=True, dtype=np.float64)
            values.append(
                np.divide(
                    power,
                    np.maximum(total, 1e-20),
                    out=np.zeros_like(power),
                    where=total > 0.0,
                )
            )
        return np.stack(values)

    ad = interpolate(TRAIN_AD_CACHE, pas_weights, (0, 1, 2))
    pas = interpolate(TRAIN_PAS_CACHE, pas_weights, 0)
    pdp = interpolate(TRAIN_PDP_CACHE, pdp_weights, 2)
    raw_pas, raw_pdp = globally_shrunk_targets(pas, pdp)
    calibrated_pas = blend_group_pas(raw_pas, grouped_pas[rows], PAS_GROUP_BLEND)
    calibrated_pdp = blend_group_pdp(raw_pdp, grouped_pdp[rows], PDP_GROUP_BLEND)
    return ad, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp


def v4_candidate_names() -> list[str]:
    selection = json.loads(V4_SELECTION.read_text(encoding="utf-8"))
    return ["identity", *sorted({str(row["candidate"]) for row in selection["final_policy"]})]


def project_candidate(
    base: torch.Tensor,
    ad: torch.Tensor,
    raw_pas: torch.Tensor,
    raw_pdp: torch.Tensor,
    calibrated_pas: torch.Tensor,
    calibrated_pdp: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if name == "identity":
        return base.clone()
    match = EXACT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"unsupported E160 V4 candidate: {name}")
    mode, order, action_name, cycles_text = match.groups()
    action_index = next(
        index for index, action in enumerate(ACTIONS) if action.name == action_name
    )
    pas, pdp = mode_targets(
        mode, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp
    )
    prediction, desired_pas, desired_pdp, reference_power = initialize_trajectory(
        base, ad, pas, pdp, action_index, order
    )
    for _ in range(2, int(cycles_text) + 1):
        prediction = continue_trajectory(
            prediction, desired_pas, desired_pdp, reference_power, order
        )
    return prediction


TRAIN_AD_CACHE = np.load(TRAIN_AD, mmap_mode="r")
TRAIN_PAS_CACHE = np.load(TRAIN_PAS, mmap_mode="r")
TRAIN_PDP_CACHE = np.load(TRAIN_PDP, mmap_mode="r")


def select_scale_rows(
    fit_ids: np.ndarray, site: np.ndarray, maximum_per_site: int = 128
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for site_value in np.unique(site[fit_ids]):
        rows = fit_ids[site[fit_ids] == site_value]
        if len(rows) > maximum_per_site:
            index = np.linspace(0, len(rows) - 1, maximum_per_site).round().astype(int)
            rows = rows[index]
        selected.append(rows)
    return np.sort(np.concatenate(selected)).astype(np.int64)


def select_projection_policy(
    device_channel: torch.Tensor,
    train_channel: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    fit_ids: np.ndarray,
    sample_ids: np.ndarray,
    configs_by_site: dict[int, list],
    stack_weights: dict[int, np.ndarray],
    donor_ids: np.ndarray,
    pas_weights: np.ndarray,
    pdp_weights: np.ndarray,
    calibrated_group_pas: np.ndarray,
    calibrated_group_pdp: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    names = v4_candidate_names()
    distances, neighbors = strict_neighbors(
        positions, site, fit_ids, sample_ids, 32
    )
    metrics = {
        name: {
            key: np.empty(len(sample_ids), dtype=np.float64)
            for key in ("pas", "pdp", "truth_energy", "predicted_energy", "cross")
        }
        for name in names
    }
    for start in range(0, len(sample_ids), batch_size):
        end = min(start + batch_size, len(sample_ids))
        query = sample_ids[start:end]
        carriers = batched_carrier_predictions(
            device_channel,
            neighbors[start:end],
            positions[neighbors[start:end]],
            positions[query],
            distances[start:end],
            bs_positions,
            site[query],
            configs_by_site,
        )
        base = combine_carriers(carriers, site[query], stack_weights)
        ad, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp = target_arrays(
            donor_ids,
            pas_weights,
            pdp_weights,
            calibrated_group_pas,
            calibrated_group_pdp,
            query,
        )
        truth = np.asarray(train_channel[query], dtype=np.complex64)
        for name in names:
            with torch.inference_mode():
                prediction = project_candidate(
                    base,
                    torch.as_tensor(ad, device=device),
                    torch.as_tensor(raw_pas, device=device),
                    torch.as_tensor(raw_pdp, device=device),
                    torch.as_tensor(calibrated_pas, device=device),
                    torch.as_tensor(calibrated_pdp, device=device),
                    name,
                ).cpu().numpy()
            point = point_metric_statistics(prediction, truth)
            metrics[name]["pas"][start:end] = point.pas
            metrics[name]["pdp"][start:end] = point.pdp
            metrics[name]["truth_energy"][start:end] = point.truth_energy
            metrics[name]["predicted_energy"][start:end] = np.sum(
                np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            metrics[name]["cross"][start:end] = np.real(
                np.sum(
                    prediction * np.conj(truth),
                    axis=(1, 2, 3),
                    dtype=np.complex128,
                )
            )
        print(f"E160 policy sample {end}/{len(sample_ids)}", flush=True)

    policy: dict[int, dict[str, object]] = {}
    report: dict[str, object] = {"sample_count": int(len(sample_ids)), "sites": {}}
    for site_value in np.unique(site[sample_ids]):
        mask = site[sample_ids] == site_value
        candidates: list[dict[str, object]] = []
        for name in names:
            values = metrics[name]
            scale = optimal_scale(values["predicted_energy"], values["cross"], mask)
            error = scaled_error(
                values["predicted_energy"],
                values["truth_energy"],
                values["cross"],
                scale,
            )
            score = aggregate_statistics(
                PointMetricStatistics(
                    values["pas"], values["pdp"], error, values["truth_energy"]
                ),
                mask,
            )
            candidates.append({"candidate": name, "scale": scale, "official": score})
        winner = max(candidates, key=lambda row: float(row["official"]["score"]))
        policy[int(site_value)] = winner
        report["sites"][str(int(site_value))] = {
            "winner": winner,
            "candidates": candidates,
        }
    return policy, report


def build_full_v7_core(
    device: torch.device,
    projection_batch_size: int,
    v3_batch_size: int,
    rbf_batch_size: int,
    force: bool,
) -> dict[str, object]:
    if not FULL_RAW_STACK.is_file() or not FULL_STACK_WEIGHTS.is_file():
        raise FileNotFoundError("E160 full raw stack must be built before the core")
    setup, positions, _, site, _ = load_positions()
    fold_id = np.load(FOLD_IDS, allow_pickle=False).astype(np.int8)
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    raw_stack = np.load(FULL_RAW_STACK, mmap_mode="r")
    descriptors = np.load(DESCRIPTORS, allow_pickle=False)
    valid = descriptors["valid"].astype(bool)
    spectra_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    geometry_numpy = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    grouped_pas_amplitude = np.load(GROUPED_PAS, mmap_mode="r")
    grouped_pdp_amplitude = np.load(GROUPED_PDP, mmap_mode="r")
    point_power = load_or_compute_power(train_channel, 16)
    stack_arrays = np.load(FULL_STACK_WEIGHTS, allow_pickle=False)
    all_ids = np.arange(len(positions), dtype=np.int64)

    outputs = [
        (FULL_TARGET_AD, np.float16, (len(positions), 16, 8, 192)),
        (FULL_TARGET_PAS, np.float16, train_channel.shape),
        (FULL_TARGET_PDP, np.float16, train_channel.shape),
        (FULL_CALIBRATED_PAS, np.float16, train_channel.shape),
        (FULL_CALIBRATED_PDP, np.float16, train_channel.shape),
        (FULL_CORE, np.complex64, train_channel.shape),
    ]
    if not force:
        existing = [path for path, _, _ in outputs if path.is_file()]
        if existing:
            raise FileExistsError(f"E160 core outputs already exist: {existing}")
    ad_output = np.lib.format.open_memmap(
        FULL_TARGET_AD, mode="w+", dtype=np.float16, shape=outputs[0][2]
    )
    pas_output = np.lib.format.open_memmap(
        FULL_TARGET_PAS, mode="w+", dtype=np.float16, shape=train_channel.shape
    )
    pdp_output = np.lib.format.open_memmap(
        FULL_TARGET_PDP, mode="w+", dtype=np.float16, shape=train_channel.shape
    )
    calibrated_pas_output = np.lib.format.open_memmap(
        FULL_CALIBRATED_PAS, mode="w+", dtype=np.float16, shape=train_channel.shape
    )
    calibrated_pdp_output = np.lib.format.open_memmap(
        FULL_CALIBRATED_PDP, mode="w+", dtype=np.float16, shape=train_channel.shape
    )
    core_output = np.lib.format.open_memmap(
        FULL_CORE, mode="w+", dtype=np.complex64, shape=train_channel.shape
    )
    global_donor = np.empty((len(positions), V3_NEIGHBORS), dtype=np.int32)
    global_base = np.empty((len(positions), V3_NEIGHBORS), dtype=np.float32)
    global_pas = np.empty_like(global_base)
    global_pdp = np.empty_like(global_base)
    folds: list[dict[str, object]] = []
    device_channel = torch.tensor(
        np.asarray(train_channel), dtype=torch.complex64, device=device
    )
    started = time.time()

    for outer in range(5):
        fold_started = time.time()
        fit_ids = np.flatnonzero(fold_id != outer).astype(np.int64)
        held_ids = np.flatnonzero(fold_id == outer).astype(np.int64)
        train_ids = fit_ids[valid[fit_ids]]
        config = V3Config(epochs=V3_EPOCHS, batch_size=v3_batch_size)
        train_context = build_v3_context(
            outer,
            "train",
            positions,
            site,
            valid,
            fit_ids,
            train_ids,
            config.guard_radii_m,
            device,
            rbf_batch_size,
            force,
        )
        deploy_context = build_v3_context(
            outer,
            "deploy",
            positions,
            site,
            valid,
            fit_ids,
            all_ids,
            (0.0,),
            device,
            rbf_batch_size,
            force,
        )
        model, training_report, tensors = train_v3_outer(
            outer,
            fit_ids,
            train_context,
            geometry_numpy,
            positions,
            spectra_numpy,
            device,
            v3_batch_size,
        )
        pas_weights, pdp_weights = predict_v3_weights(
            model, deploy_context, tensors, device, v3_batch_size
        )
        donor_ids = deploy_context["donor_ids"][:, 0].astype(np.int64)
        base_weights = deploy_context["weights"][:, 0].astype(np.float32)
        grouped_pas_prediction = predict_grouped(
            grouped_pas_amplitude, donor_ids, pas_weights
        )
        grouped_pdp_prediction = predict_grouped(
            grouped_pdp_amplitude, donor_ids, pdp_weights
        )
        fit_mask = (fold_id != outer) & valid
        pas_models = fit_group_models(
            grouped_pas_amplitude,
            grouped_pas_prediction,
            fit_mask,
            site,
            PAS_COMPONENTS,
            device,
            20_260_805 + outer * 101,
        )
        pdp_models = fit_group_models(
            grouped_pdp_amplitude,
            grouped_pdp_prediction,
            fit_mask,
            site,
            PDP_COMPONENTS,
            device,
            20_270_805 + outer * 101,
        )
        calibrated_group_pas = apply_calibration(
            grouped_pas_prediction, site, pas_models, PAS_CALIBRATION_STRENGTH
        )
        calibrated_group_pdp = apply_calibration(
            grouped_pdp_prediction, site, pdp_models, PDP_CALIBRATION_STRENGTH
        )

        for start in range(0, len(held_ids), projection_batch_size):
            rows = held_ids[start : start + projection_batch_size]
            ad, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp = target_arrays(
                donor_ids,
                pas_weights,
                pdp_weights,
                calibrated_group_pas,
                calibrated_group_pdp,
                rows,
            )
            ad_output[rows] = ad.astype(np.float16)
            pas_output[rows] = raw_pas.astype(np.float16)
            pdp_output[rows] = raw_pdp.astype(np.float16)
            calibrated_pas_output[rows] = calibrated_pas.astype(np.float16)
            calibrated_pdp_output[rows] = calibrated_pdp.astype(np.float16)
            if (start + len(rows)) % (projection_batch_size * 10) == 0 or (
                start + len(rows) == len(held_ids)
            ):
                print(
                    f"E160 targets outer={outer} {start + len(rows)}/{len(held_ids)}",
                    flush=True,
                )
        for output in (
            ad_output,
            pas_output,
            pdp_output,
            calibrated_pas_output,
            calibrated_pdp_output,
        ):
            output.flush()

        configs_by_site, calibration = configured_carriers(
            train_channel,
            positions,
            site,
            setup.bs_positions,
            fit_ids,
            point_power,
            outer,
        )
        stack_weights = {
            site_value: stack_arrays[f"fold{outer}_site{site_value}"].astype(
                np.complex64
            )
            for site_value in (0, 1)
        }
        sample_ids = select_scale_rows(fit_ids, site)
        policy, policy_report = select_projection_policy(
            device_channel,
            train_channel,
            positions,
            site,
            setup.bs_positions,
            fit_ids,
            sample_ids,
            configs_by_site,
            stack_weights,
            donor_ids,
            pas_weights,
            pdp_weights,
            calibrated_group_pas,
            calibrated_group_pdp,
            device,
            projection_batch_size,
        )

        for start in range(0, len(held_ids), projection_batch_size):
            rows = held_ids[start : start + projection_batch_size]
            base = torch.as_tensor(
                np.asarray(raw_stack[rows], dtype=np.complex64), device=device
            )
            ad = torch.as_tensor(
                np.asarray(ad_output[rows], dtype=np.float32), device=device
            )
            raw_pas = torch.as_tensor(
                np.asarray(pas_output[rows], dtype=np.float32), device=device
            )
            raw_pdp = torch.as_tensor(
                np.asarray(pdp_output[rows], dtype=np.float32), device=device
            )
            calibrated_pas = torch.as_tensor(
                np.asarray(calibrated_pas_output[rows], dtype=np.float32), device=device
            )
            calibrated_pdp = torch.as_tensor(
                np.asarray(calibrated_pdp_output[rows], dtype=np.float32), device=device
            )
            block = np.empty((len(rows), *train_channel.shape[1:]), dtype=np.complex64)
            for site_value in np.unique(site[rows]):
                local = np.flatnonzero(site[rows] == site_value)
                decision = policy[int(site_value)]
                with torch.inference_mode():
                    prediction = project_candidate(
                        base[local],
                        ad[local],
                        raw_pas[local],
                        raw_pdp[local],
                        calibrated_pas[local],
                        calibrated_pdp[local],
                        str(decision["candidate"]),
                    )
                block[local] = (
                    prediction.cpu().numpy() * np.float32(decision["scale"])
                ).astype(np.complex64)
            core_output[rows] = block
            if (start + len(rows)) % (projection_batch_size * 10) == 0 or (
                start + len(rows) == len(held_ids)
            ):
                print(
                    f"E160 core outer={outer} {start + len(rows)}/{len(held_ids)}",
                    flush=True,
                )
        core_output.flush()

        global_donor[held_ids] = donor_ids[held_ids].astype(np.int32)
        global_base[held_ids] = base_weights[held_ids]
        global_pas[held_ids] = pas_weights[held_ids]
        global_pdp[held_ids] = pdp_weights[held_ids]
        folds.append(
            {
                "fold": outer,
                "fit_count": int(len(fit_ids)),
                "held_count": int(len(held_ids)),
                "training": training_report,
                "carrier_calibration": calibration,
                "projection_policy": policy_report,
                "seconds": float(time.time() - fold_started),
            }
        )
        del model, tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    np.savez_compressed(
        FULL_V3_WEIGHTS,
        fold_id=fold_id,
        donor_ids=global_donor,
        base_weights=global_base,
        pas_weights=global_pas,
        pdp_weights=global_pdp,
        epoch=np.int32(V3_EPOCHS),
    )
    finite = True
    for output in (
        ad_output,
        pas_output,
        pdp_output,
        calibrated_pas_output,
        calibrated_pdp_output,
        core_output,
    ):
        for start in range(0, len(output), 8):
            value = np.asarray(output[start : start + 8])
            finite &= bool(np.isfinite(value).all())
    if not finite:
        raise ValueError("E160 full core outputs contain non-finite values")
    official = score_memmap(core_output, train_channel, all_ids, 8)
    return {
        "path": PATHS.relative(FULL_CORE),
        "shape": list(core_output.shape),
        "dtype": str(core_output.dtype),
        "finite": finite,
        "official": official,
        "weights": PATHS.relative(FULL_V3_WEIGHTS),
        "targets": {
            "ad": PATHS.relative(FULL_TARGET_AD),
            "pas": PATHS.relative(FULL_TARGET_PAS),
            "pdp": PATHS.relative(FULL_TARGET_PDP),
            "calibrated_pas": PATHS.relative(FULL_CALIBRATED_PAS),
            "calibrated_pdp": PATHS.relative(FULL_CALIBRATED_PDP),
        },
        "folds": folds,
        "seconds": float(time.time() - started),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    payload = {
        "experiment": VERSION,
        "status": "identity_gate_running",
        "contract": {
            "final_local_goal": 0.677521,
            "final_online_goal": 0.70,
            "current_goal": "build strict full-4000 V7-domain supervision",
            "does_not_do": [
                "train correction student",
                "read Sionna or V12 point outputs",
                "generate test channel or submission archive",
            ],
        },
        "gpu": gpu_report(device),
    }
    if args.phase == "identity":
        payload["identity_gate"] = identity_gate(device, args.batch_size)
        payload["status"] = (
            "identity_gate_passed"
            if payload["identity_gate"]["passed"]
            else "identity_gate_failed"
        )
    elif args.phase in ("carriers", "clean_carriers"):
        if REPORT.is_file():
            previous = json.loads(REPORT.read_text(encoding="utf-8"))
            if "identity_gate" in previous:
                payload["identity_gate"] = previous["identity_gate"]
        payload["full4000_raw_stack"] = build_full_raw_stack(
            device,
            args.batch_size,
            args.force,
            require_identity=args.phase == "carriers",
        )
        payload["strict_family_isolation"] = args.phase == "clean_carriers"
        payload["status"] = "full4000_raw_stack_complete"
    elif args.phase in ("core", "clean_core"):
        if REPORT.is_file():
            previous = json.loads(REPORT.read_text(encoding="utf-8"))
            for key in ("identity_gate", "full4000_raw_stack"):
                if key in previous:
                    payload[key] = previous[key]
        payload["full4000_v7_core"] = build_full_v7_core(
            device,
            args.batch_size,
            args.v3_batch_size,
            args.rbf_batch_size,
            args.force,
        )
        payload["strict_family_isolation"] = args.phase == "clean_core"
        payload["status"] = "full4000_v7_core_complete"
    payload["seconds"] = float(time.time() - started)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if payload["status"] == "identity_gate_failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
