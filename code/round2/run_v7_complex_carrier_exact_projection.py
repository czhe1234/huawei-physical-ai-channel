#!/usr/bin/env python3
"""Evaluate a learned complex carrier under the frozen V4 spectral projection."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    continue_trajectory,
    initialize_trajectory,
)
from analyze_v4_predictability_calibrated_groups import (
    GROUPED_PAS,
    GROUPED_PDP,
    blend_group_pas,
    blend_group_pdp,
    globally_shrunk_targets,
    predict_grouped,
)
from analyze_v7_complex_carrier_stacking import (
    StackConfig,
    _scope_labels,
    compute_gram,
    fit_weights,
)
from build_v3_anchored_neural_rbf import (
    TEST_WEIGHTS,
    TRAIN_AD,
    TRAIN_PAS,
    TRAIN_PDP,
    spectral_target,
)
from build_v7_complex_carrier_bank import OOF_BANK, TEST_BANK
from data import load_positions
from evaluate_v3_anchored_projection import target_batch
from official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics,
)
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, OOF_META, optimal_scale, scaled_error
from run_v4_calibrated_exact_projection import (
    CALIBRATION_MODEL,
    EXACT_PATTERN,
    OOF_CALIBRATED_PAS,
    OOF_CALIBRATED_PDP,
    PAS_CALIBRATION_STRENGTH,
    PAS_GROUP_BLEND,
    PDP_CALIBRATION_STRENGTH,
    PDP_GROUP_BLEND,
    SELECTION as V4_SELECTION,
    action_by_name,
    apply_calibration,
    mode_targets,
)
from evaluate_v3_anchored_projection import (
    TARGET_AD,
    TARGET_PAS,
    TARGET_PDP,
)


VERSION = "Round2-E46-complex-carrier-exact-projection"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "complex_carrier_stacking"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_complex_carrier_stacking"
STACK_SELECTION = ARTIFACT_DIR / "complex_stack_selection.json"
OOF_STACK = ARTIFACT_DIR / "oof_stacked_complex_channel.npy"
OOF_PROJECTED = ARTIFACT_DIR / "oof_stacked_v4_projection.npy"
FULL_WEIGHTS = ARTIFACT_DIR / "full_complex_stack_weights.npz"
TEST_STACK = ARTIFACT_DIR / "test_stacked_complex_channel.npy"
TEST_PROJECTED = (
    ARTIFACT_DIR / "Round2_V7_Complex_Carrier_Stacked_Projection_Test_Channel.npy"
)
DEPLOYMENT = ARTIFACT_DIR / "complex_carrier_deployment.json"
REPORT = ANALYSIS_DIR / "complex_carrier_exact_projection.json"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"


def stack_config_from_record(record: dict[str, object]) -> StackConfig:
    return StackConfig(
        scope=str(record["scope"]),
        ports=str(record["ports"]),
        ridge=float(record["ridge"]),
        sum_penalty=float(record["sum_penalty"]),
    )


def row_weights(
    labels: np.ndarray,
    rows: np.ndarray,
    weights: dict[int, np.ndarray],
    config: StackConfig,
) -> np.ndarray:
    output = np.empty((len(rows), 8, next(iter(weights.values())).shape[-1]), dtype=np.complex64)
    for destination, row in enumerate(rows):
        value = weights[int(labels[row])]
        if config.ports == "shared":
            value = np.repeat(value, 8, axis=0)
        output[destination] = value
    return output


def combine_block(bank: np.ndarray, rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    carrier_count = bank.shape[0]
    block = np.asarray(bank[:, rows], dtype=np.complex64)
    candidate = (
        block.reshape(carrier_count, len(rows), 2, 16, 8, 4, 192)
        .transpose(1, 2, 5, 0, 3, 4, 6)
        .reshape(len(rows), 8, carrier_count, 16, 8, 192)
    )
    combined = np.einsum(
        "bgcvhd,bgc->bgvhd", candidate, weights, optimize=True
    )
    return (
        combined.reshape(len(rows), 2, 4, 16, 8, 192)
        .transpose(0, 1, 3, 4, 2, 5)
        .reshape(len(rows), 256, 4, 192)
        .astype(np.complex64)
    )


def save_full_weights(
    weights: dict[int, np.ndarray], config: StackConfig
) -> dict[str, object]:
    arrays = {
        f"label_{label}": np.asarray(value, dtype=np.complex64)
        for label, value in weights.items()
    }
    arrays["labels"] = np.asarray(sorted(weights), dtype=np.int16)
    arrays["scope"] = np.asarray(config.scope)
    arrays["ports"] = np.asarray(config.ports)
    arrays["ridge"] = np.asarray(config.ridge, dtype=np.float64)
    arrays["sum_penalty"] = np.asarray(config.sum_penalty, dtype=np.float64)
    np.savez_compressed(FULL_WEIGHTS, **arrays)
    return {
        "path": PATHS.relative(FULL_WEIGHTS),
        "labels": sorted(int(label) for label in weights),
        "weight_norm": {
            str(label): float(np.linalg.norm(value))
            for label, value in weights.items()
        },
    }


def build_test_stack(
    cache: dict[str, np.ndarray],
    config: StackConfig,
    oof_site: np.ndarray,
    oof_regime: np.ndarray,
    test_site: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    bank = np.load(TEST_BANK, mmap_mode="r")
    fit_rows = np.arange(len(oof_site), dtype=np.int64)
    fit_labels = _scope_labels(config, oof_site, oof_regime)
    weights = fit_weights(cache, fit_rows, fit_labels, config)
    weight_report = save_full_weights(weights, config)
    test_labels = _scope_labels(
        config, test_site, np.zeros(len(test_site), dtype=np.int8)
    )
    output = np.lib.format.open_memmap(
        TEST_STACK,
        mode="w+",
        dtype=np.complex64,
        shape=tuple(bank.shape[1:]),
    )
    for start in range(0, len(test_site), batch_size):
        rows = np.arange(start, min(start + batch_size, len(test_site)), dtype=np.int64)
        output[rows] = combine_block(
            bank,
            rows,
            row_weights(test_labels, rows, weights, config),
        )
        output.flush()
        print(f"stacked test carrier {int(rows[-1]) + 1}/{len(test_site)}", flush=True)
    return {
        "path": PATHS.relative(TEST_STACK),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "weights": weight_report,
    }


def load_calibration_models() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cached = np.load(CALIBRATION_MODEL, allow_pickle=False)
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for site in (0, 1):
        prefix = f"site{site}"
        models[site] = (
            cached[f"{prefix}_pas_mean"],
            cached[f"{prefix}_pas_vectors"],
            cached[f"{prefix}_pas_slope"],
        )
    return models


def load_pdp_calibration_models() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cached = np.load(CALIBRATION_MODEL, allow_pickle=False)
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for site in (0, 1):
        prefix = f"site{site}"
        models[site] = (
            cached[f"{prefix}_pdp_mean"],
            cached[f"{prefix}_pdp_vectors"],
            cached[f"{prefix}_pdp_slope"],
        )
    return models


def test_group_calibration(test_site: np.ndarray) -> dict[str, np.ndarray]:
    weights = np.load(TEST_WEIGHTS, allow_pickle=False)
    donor_ids = weights["donor_ids"].astype(np.int64)
    pas_weights = weights["pas_weights"].astype(np.float32)
    pdp_weights = weights["pdp_weights"].astype(np.float32)
    pas_amplitude = np.load(GROUPED_PAS, mmap_mode="r")
    pdp_amplitude = np.load(GROUPED_PDP, mmap_mode="r")
    test_pas = predict_grouped(pas_amplitude, donor_ids, pas_weights)
    test_pdp = predict_grouped(pdp_amplitude, donor_ids, pdp_weights)
    return {
        "pas": apply_calibration(
            test_pas,
            test_site,
            load_calibration_models(),
            PAS_CALIBRATION_STRENGTH,
        ),
        "pdp": apply_calibration(
            test_pdp,
            test_site,
            load_pdp_calibration_models(),
            PDP_CALIBRATION_STRENGTH,
        ),
    }


def deployment_scales(report: dict[str, object]) -> dict[int, float]:
    decisions = report["projected_score"]["full_oof"]["scale_decisions"]
    scales = {int(row["site"]): float(row["scale"]) for row in decisions}
    if sorted(scales) != [0, 1]:
        raise RuntimeError(f"missing deployment scales: {scales}")
    return scales


def project_test_v4_policy(
    device: torch.device,
    batch_size: int,
    test_site: np.ndarray,
    scales: dict[int, float],
) -> dict[str, object]:
    selection = json.loads(V4_SELECTION.read_text(encoding="utf-8"))
    policy = {int(row["site"]): row for row in selection["final_policy"]}
    weights = np.load(TEST_WEIGHTS, allow_pickle=False)
    donor_ids = weights["donor_ids"].astype(np.int64)
    pas_weights = weights["pas_weights"].astype(np.float32)
    pdp_weights = weights["pdp_weights"].astype(np.float32)
    train_ad = np.load(TRAIN_AD, mmap_mode="r")
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    calibration = test_group_calibration(test_site)
    base = np.load(TEST_STACK, mmap_mode="r")
    output = np.lib.format.open_memmap(
        TEST_PROJECTED,
        mode="w+",
        dtype=np.complex64,
        shape=base.shape,
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        ad_rows: list[np.ndarray] = []
        pas_rows: list[np.ndarray] = []
        pdp_rows: list[np.ndarray] = []
        for row in range(start, end):
            ad_rows.append(
                spectral_target(train_ad, donor_ids[row], pas_weights[row], (0, 1, 2))
            )
            pas_rows.append(
                spectral_target(train_pas, donor_ids[row], pas_weights[row], 0)
            )
            pdp_rows.append(
                spectral_target(train_pdp, donor_ids[row], pdp_weights[row], 2)
            )
        raw_pas, raw_pdp = globally_shrunk_targets(
            np.stack(pas_rows), np.stack(pdp_rows)
        )
        calibrated_pas = blend_group_pas(
            raw_pas, calibration["pas"][start:end], PAS_GROUP_BLEND
        )
        calibrated_pdp = blend_group_pdp(
            raw_pdp, calibration["pdp"][start:end], PDP_GROUP_BLEND
        )
        ad_tensor = torch.as_tensor(np.stack(ad_rows), device=device)
        raw_pas_tensor = torch.as_tensor(raw_pas, device=device)
        raw_pdp_tensor = torch.as_tensor(raw_pdp, device=device)
        calibrated_pas_tensor = torch.as_tensor(calibrated_pas, device=device)
        calibrated_pdp_tensor = torch.as_tensor(calibrated_pdp, device=device)
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        block = np.empty_like(np.asarray(base[start:end]))
        block_site = test_site[start:end]
        with torch.inference_mode():
            for site_value in np.unique(block_site):
                local = np.flatnonzero(block_site == site_value)
                name = str(policy[int(site_value)]["candidate"])
                if name.startswith("legacy_"):
                    from pre_sionna_spectral import project_channel

                    action = action_by_name(name.removeprefix("legacy_"))
                    prediction = project_channel(
                        base_tensor[local],
                        ad_tensor[local],
                        raw_pas_tensor[local],
                        raw_pdp_tensor[local],
                        action.alpha_ad,
                        action.alpha_pas,
                        action.alpha_pdp,
                    )
                else:
                    match = EXACT_PATTERN.fullmatch(name)
                    if match is None:
                        raise ValueError(f"cannot parse V4 candidate {name}")
                    mode, order, action_name, cycles_text = match.groups()
                    action_index = next(
                        index
                        for index, action in enumerate(ACTIONS)
                        if action.name == action_name
                    )
                    pas, pdp = mode_targets(
                        mode,
                        raw_pas_tensor[local],
                        raw_pdp_tensor[local],
                        calibrated_pas_tensor[local],
                        calibrated_pdp_tensor[local],
                    )
                    prediction, desired_pas, desired_pdp, reference_power = (
                        initialize_trajectory(
                            base_tensor[local],
                            ad_tensor[local],
                            pas,
                            pdp,
                            action_index,
                            order,
                        )
                    )
                    for _ in range(2, int(cycles_text) + 1):
                        prediction = continue_trajectory(
                            prediction,
                            desired_pas,
                            desired_pdp,
                            reference_power,
                            order,
                        )
                block[local] = (
                    prediction.cpu().numpy() * np.float32(scales[int(site_value)])
                ).astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"stacked V4 test projection {end}/{len(base)}", flush=True)
    finite = True
    for start in range(0, len(output), 8):
        value = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(value.real).all() and np.isfinite(value.imag).all()
        )
    if not finite:
        raise ValueError("V7 output contains non-finite values")
    return {
        "path": PATHS.relative(TEST_PROJECTED),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(TEST_PROJECTED.stat().st_size),
        "site_scales": {str(site): scale for site, scale in scales.items()},
    }
def build_oof_stack(
    cache: dict[str, np.ndarray],
    config: StackConfig,
    site: np.ndarray,
    regime: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    bank = np.load(OOF_BANK, mmap_mode="r")
    fold_id = cache["fold_id"].astype(np.int8)
    labels = _scope_labels(config, site, regime)
    output = np.lib.format.open_memmap(
        OOF_STACK,
        mode="w+",
        dtype=np.complex64,
        shape=tuple(bank.shape[1:]),
    )
    decisions: list[dict[str, object]] = []
    for fold in np.unique(fold_id):
        fit = np.flatnonzero(fold_id != fold)
        held = np.flatnonzero(fold_id == fold)
        weights = fit_weights(cache, fit, labels, config)
        decisions.append(
            {
                "fold": int(fold),
                "fit_count": int(len(fit)),
                "held_count": int(len(held)),
                "weight_norm": {
                    str(label): float(np.linalg.norm(value))
                    for label, value in weights.items()
                },
            }
        )
        for start in range(0, len(held), batch_size):
            rows = held[start : start + batch_size]
            output[rows] = combine_block(
                bank,
                rows,
                row_weights(labels, rows, weights, config),
            )
            output.flush()
        print(f"stacked OOF fold={int(fold)} count={len(held)}", flush=True)
    return {
        "path": PATHS.relative(OOF_STACK),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "decisions": decisions,
    }


def aggregate_points(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    return aggregate_statistics(PointMetricStatistics(pas, pdp, error, energy), mask)


def fit_scales(
    predicted_energy: np.ndarray,
    truth_energy: np.ndarray,
    cross: np.ndarray,
    site: np.ndarray,
    fold_id: np.ndarray,
    nested: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    scale = np.empty(len(site), dtype=np.float32)
    decisions: list[dict[str, object]] = []
    folds = np.unique(fold_id) if nested else np.asarray([-1], dtype=np.int8)
    for fold in folds:
        for site_value in np.unique(site):
            if nested:
                fit = (fold_id != fold) & (site == site_value)
                query = (fold_id == fold) & (site == site_value)
            else:
                fit = site == site_value
                query = site == site_value
            value = optimal_scale(predicted_energy, cross, fit)
            scale[query] = value
            decisions.append(
                {
                    "fold": None if not nested else int(fold),
                    "site": int(site_value),
                    "scale": float(value),
                    "fit_count": int(np.sum(fit)),
                    "query_count": int(np.sum(query)),
                }
            )
    return scale, decisions


def score_channel(
    prediction: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    site: np.ndarray,
    fold_id: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    count = len(query_ids)
    pas = np.empty(count, dtype=np.float64)
    pdp = np.empty(count, dtype=np.float64)
    truth_energy = np.empty(count, dtype=np.float64)
    predicted_energy = np.empty(count, dtype=np.float64)
    cross = np.empty(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        value = np.asarray(prediction[start:end], dtype=np.complex64)
        truth = np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64)
        point = point_metric_statistics(value, truth)
        pas[start:end] = point.pas
        pdp[start:end] = point.pdp
        truth_energy[start:end] = point.truth_energy
        predicted_energy[start:end] = np.sum(
            np.abs(value) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        cross[start:end] = np.real(
            np.sum(value * np.conj(truth), axis=(1, 2, 3), dtype=np.complex128)
        )
    reports: dict[str, object] = {}
    for name, nested in (("nested", True), ("full_oof", False)):
        scale, decisions = fit_scales(
            predicted_energy, truth_energy, cross, site, fold_id, nested
        )
        error = scaled_error(predicted_energy, truth_energy, cross, scale)
        reports[name] = {
            "overall": aggregate_points(pas, pdp, error, truth_energy),
            "folds": [
                {
                    "fold": int(fold),
                    **aggregate_points(pas, pdp, error, truth_energy, fold_id == fold),
                }
                for fold in np.unique(fold_id)
            ],
            "scale_decisions": decisions,
        }
    return reports


def project_v4_policy(
    device: torch.device,
    batch_size: int,
    query_ids: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    selection = json.loads(V4_SELECTION.read_text(encoding="utf-8"))
    policy = {int(row["site"]): row for row in selection["final_policy"]}
    base = np.load(OOF_STACK, mmap_mode="r")
    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    calibrated_pas_cache = np.load(OOF_CALIBRATED_PAS, mmap_mode="r")
    calibrated_pdp_cache = np.load(OOF_CALIBRATED_PDP, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OOF_PROJECTED,
        mode="w+",
        dtype=np.complex64,
        shape=base.shape,
    )
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        ad, raw_pas, raw_pdp = target_batch(
            ad_cache, pas_cache, pdp_cache, start, end, device
        )
        calibrated_pas = torch.as_tensor(
            np.asarray(calibrated_pas_cache[start:end], dtype=np.float32), device=device
        )
        calibrated_pdp = torch.as_tensor(
            np.asarray(calibrated_pdp_cache[start:end], dtype=np.float32), device=device
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        block = np.empty_like(np.asarray(base[start:end]))
        block_site = site[start:end]
        with torch.inference_mode():
            for site_value in np.unique(block_site):
                local = np.flatnonzero(block_site == site_value)
                name = str(policy[int(site_value)]["candidate"])
                if name.startswith("legacy_"):
                    action = action_by_name(name.removeprefix("legacy_"))
                    from pre_sionna_spectral import project_channel

                    prediction = project_channel(
                        base_tensor[local],
                        ad[local],
                        raw_pas[local],
                        raw_pdp[local],
                        action.alpha_ad,
                        action.alpha_pas,
                        action.alpha_pdp,
                    )
                else:
                    match = EXACT_PATTERN.fullmatch(name)
                    if match is None:
                        raise ValueError(f"cannot parse V4 candidate {name}")
                    mode, order, action_name, cycles_text = match.groups()
                    action_index = next(
                        index
                        for index, action in enumerate(ACTIONS)
                        if action.name == action_name
                    )
                    pas, pdp = mode_targets(
                        mode,
                        raw_pas[local],
                        raw_pdp[local],
                        calibrated_pas[local],
                        calibrated_pdp[local],
                    )
                    prediction, desired_pas, desired_pdp, reference_power = (
                        initialize_trajectory(
                            base_tensor[local],
                            ad[local],
                            pas,
                            pdp,
                            action_index,
                            order,
                        )
                    )
                    for _ in range(2, int(cycles_text) + 1):
                        prediction = continue_trajectory(
                            prediction,
                            desired_pas,
                            desired_pdp,
                            reference_power,
                            order,
                        )
                block[local] = prediction.cpu().numpy().astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"stacked V4 projection {end}/{len(query_ids)}", flush=True)
    return {
        "path": PATHS.relative(OOF_PROJECTED),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cache = compute_gram(torch.device(args.device))
    stack_selection = json.loads(STACK_SELECTION.read_text(encoding="utf-8"))
    config = stack_config_from_record(
        stack_selection["posterior_best"]["config"]
    )
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, test_site = load_positions()
    site = train_site[query_ids].astype(np.int8)
    from run_v6_regime_component_lifecycle import REGIME_CACHE

    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    config_payload = {
        "name": config.name,
        "scope": config.scope,
        "ports": config.ports,
        "ridge": config.ridge,
        "sum_penalty": config.sum_penalty,
    }
    hypothesis = (
        "A complex ridge stack that improves carrier coherence can preserve that "
        "gain after the frozen, already validated V4 PAS/PDP exact projection."
    )
    payload: dict[str, object]
    if args.phase in ("oof", "all"):
        stack = build_oof_stack(cache, config, site, regime, args.batch_size)
        truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
        raw_score = score_channel(
            np.load(OOF_STACK, mmap_mode="r"),
            truth_all,
            query_ids,
            site,
            fold_id,
            args.batch_size,
        )
        projection = project_v4_policy(
            torch.device(args.device), args.batch_size, query_ids, site
        )
        projected_score = score_channel(
            np.load(OOF_PROJECTED, mmap_mode="r"),
            truth_all,
            query_ids,
            site,
            fold_id,
            args.batch_size,
        )
        v4_selection = json.loads(V4_SELECTION.read_text(encoding="utf-8"))
        payload = {
            "experiment": VERSION,
            "status": "analysis_only",
            "official_inputs_only": True,
            "sionna_used": False,
            "primary_hypothesis": hypothesis,
            "stack_config": config_payload,
            "stack": stack,
            "raw_score": raw_score,
            "projection": projection,
            "projected_score": projected_score,
            "v4_reference": {
                "nested": v4_selection["nested_selected"],
                "full_oof": v4_selection["final_policy_oof"]["overall"],
            },
            "seconds": float(time.time() - started),
        }
    else:
        payload = json.loads(REPORT.read_text(encoding="utf-8"))

    if args.phase in ("test", "all"):
        if not TEST_BANK.is_file():
            raise FileNotFoundError(TEST_BANK)
        scales = deployment_scales(payload)
        test_stack = build_test_stack(
            cache,
            config,
            site,
            regime,
            test_site.astype(np.int8),
            args.batch_size,
        )
        test_projection = project_test_v4_policy(
            torch.device(args.device),
            args.batch_size,
            test_site.astype(np.int8),
            scales,
        )
        deployment = {
            "experiment": VERSION,
            "status": "candidate_ready",
            "official_inputs_only": True,
            "sionna_used": False,
            "primary_hypothesis": hypothesis,
            "stack_config": config_payload,
            "local_validation": payload["projected_score"],
            "v4_reference": payload["v4_reference"],
            "test_stack": test_stack,
            "test_projection": test_projection,
            "seconds": float(time.time() - started),
        }
        DEPLOYMENT.write_text(
            json.dumps(deployment, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        payload["test_deployment"] = deployment
        payload["status"] = "candidate_ready"

    payload["seconds"] = float(time.time() - started)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
