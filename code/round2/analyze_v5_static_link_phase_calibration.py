#!/usr/bin/env python3
"""Test fit-only static link phase calibration on strict Round 2 V4 OOF channels."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from analyze_v4_exact_joint_projection import continue_trajectory, initialize_trajectory
from official_metrics import point_metric_statistics
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, OOF_BASE, OOF_META, statistics
from run_v4_calibrated_exact_projection import (
    EXACT_PATTERN,
    OOF_CALIBRATED_PAS,
    OOF_CALIBRATED_PDP,
    TARGET_AD,
    TARGET_PAS,
    TARGET_PDP,
    action_by_name,
    mode_targets,
)
from evaluate_v3_anchored_projection import target_batch


VERSION = "Round2-E26-static-link-phase-calibration"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_static_link_phase_calibration"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_static_link_phase_calibration"
V4_OOF_CHANNEL = ARTIFACT_DIR / "v4_nested_oof_channel.npy"
FACTOR_CACHE = ARTIFACT_DIR / "oof_static_phase_factors.npz"
METRIC_CACHE = ARTIFACT_DIR / "oof_static_phase_metrics.npz"
REPORT = ANALYSIS_DIR / "static_link_phase_calibration.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
V4_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_v4_calibrated_exact_projection"
    / "calibrated_exact_projection.json"
)

TINY = 1e-30


@dataclass(frozen=True)
class Config:
    batch_size: int = 8
    separable_iterations: int = 20
    seed: int = 20_260_805


def parse_candidate(name: str) -> tuple[str, str, int, int]:
    matched = EXACT_PATTERN.match(name)
    if matched is None:
        raise ValueError(name)
    mode, order, action_name, cycles = matched.groups()
    action = action_by_name(action_name)
    action_index = next(index for index, value in enumerate(ACTIONS) if value == action)
    return mode, order, action_index, int(cycles)


@torch.inference_mode()
def build_v4_oof(device: torch.device, batch_size: int) -> dict[str, float]:
    if V4_OOF_CHANNEL.is_file():
        return {"reused": True, "bytes": int(V4_OOF_CHANNEL.stat().st_size)}
    payload = json.loads(V4_REPORT.read_text(encoding="utf-8"))
    decisions = {
        (int(row["held_fold"]), int(row["site"])): row
        for row in payload["nested_selected"]["decisions"]
    }
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    site = np.load(
        PATHS.artifacts
        / "round2"
        / "v4_calibrated_exact_projection"
        / "calibrated_exact_projection_metrics.npz"
    )["site"].astype(np.int8)
    base = np.load(OOF_BASE, mmap_mode="r")
    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    calibrated_pas_cache = np.load(OOF_CALIBRATED_PAS, mmap_mode="r")
    calibrated_pdp_cache = np.load(OOF_CALIBRATED_PDP, mmap_mode="r")
    output = np.lib.format.open_memmap(
        V4_OOF_CHANNEL,
        mode="w+",
        dtype=np.complex64,
        shape=(len(query_ids), 256, 4, 192),
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
        batch_output = torch.empty_like(base_tensor)
        local_fold = fold_id[start:end]
        local_site = site[start:end]
        for fold_value in np.unique(local_fold):
            for site_value in np.unique(local_site[local_fold == fold_value]):
                rows = np.flatnonzero(
                    (local_fold == fold_value) & (local_site == site_value)
                )
                row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
                decision = decisions[(int(fold_value), int(site_value))]
                mode, order, action_index, cycles = parse_candidate(
                    str(decision["candidate"])
                )
                pas, pdp = mode_targets(
                    mode,
                    raw_pas[row_tensor],
                    raw_pdp[row_tensor],
                    calibrated_pas[row_tensor],
                    calibrated_pdp[row_tensor],
                )
                candidate, desired_pas, desired_pdp, reference_power = initialize_trajectory(
                    base_tensor[row_tensor],
                    ad[row_tensor],
                    pas,
                    pdp,
                    action_index,
                    order,
                )
                for _ in range(2, cycles + 1):
                    candidate = continue_trajectory(
                        candidate,
                        desired_pas,
                        desired_pdp,
                        reference_power,
                        order,
                    )
                candidate *= float(decision["scale"])
                batch_output[row_tensor] = candidate
        output[start:end] = batch_output.cpu().numpy()
        print(f"strict V4 OOF channel {end}/{len(query_ids)}", flush=True)
    output.flush()
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    metric_parts = []
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        metric_parts.append(
            point_metric_statistics(
                np.asarray(output[start:end]),
                np.asarray(truth[query_ids[start:end]], dtype=np.complex64),
            )
        )
    pas = np.concatenate([value.pas for value in metric_parts])
    pdp = np.concatenate([value.pdp for value in metric_parts])
    error = np.concatenate([value.squared_error for value in metric_parts])
    energy = np.concatenate([value.truth_energy for value in metric_parts])
    result = statistics(pas, pdp, error, energy)
    result["reused"] = False
    result["bytes"] = int(V4_OOF_CHANNEL.stat().st_size)
    return result


def unit_phase(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    return np.where(magnitude > TINY, value / np.maximum(magnitude, TINY), 1.0 + 0.0j)


def fit_cross_tensor(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    cross = np.zeros((2, 16, 8, 4), dtype=np.complex128)
    for start in range(0, len(rows), batch_size):
        ids = rows[start : start + batch_size]
        source = np.asarray(prediction[ids], dtype=np.complex64).reshape(-1, 2, 16, 8, 4, 192)
        target_array = np.asarray(
            truth[query_ids[ids]], dtype=np.complex64
        ).reshape(-1, 2, 16, 8, 4, 192)
        cross += np.sum(np.conj(source) * target_array, axis=(0, 5), dtype=np.complex128)
    return cross


def polue_factor(cross: np.ndarray) -> np.ndarray:
    grouped = np.sum(cross, axis=(1, 2), dtype=np.complex128)
    return np.broadcast_to(unit_phase(grouped)[:, None, None, :], cross.shape).copy()


def separable_factor(cross: np.ndarray, iterations: int) -> np.ndarray:
    factors = [
        np.ones(size, dtype=np.complex128)
        for size in (cross.shape[0], cross.shape[1], cross.shape[2], cross.shape[3])
    ]
    for _ in range(iterations):
        for axis in range(4):
            other = np.ones(cross.shape, dtype=np.complex128)
            for other_axis, factor in enumerate(factors):
                if other_axis == axis:
                    continue
                shape = [1, 1, 1, 1]
                shape[other_axis] = len(factor)
                other *= factor.reshape(shape)
            axes = tuple(value for value in range(4) if value != axis)
            summary = np.sum(np.conj(other) * cross, axis=axes, dtype=np.complex128)
            factors[axis] = unit_phase(summary)
    output = np.ones(cross.shape, dtype=np.complex128)
    for axis, factor in enumerate(factors):
        shape = [1, 1, 1, 1]
        shape[axis] = len(factor)
        output *= factor.reshape(shape)
    return unit_phase(output)


def blend_phase(factor: np.ndarray, alpha: float) -> np.ndarray:
    return unit_phase((1.0 - alpha) + alpha * factor)


def fit_factors(config: Config) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if FACTOR_CACHE.is_file():
        cached = np.load(FACTOR_CACHE)
        names = cached["factor_names"].tolist()
        return (
            {name: cached[name].astype(np.complex64) for name in names},
            json.loads(str(cached["diagnostics_json"])),
        )
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    site = np.load(
        PATHS.artifacts
        / "round2"
        / "v4_calibrated_exact_projection"
        / "calibrated_exact_projection_metrics.npz"
    )["site"].astype(np.int8)
    prediction = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    names = ("polue", "separable", "full_link", "full_link_shuffle")
    factors = {
        name: np.ones((len(query_ids), 2, 16, 8, 4), dtype=np.complex64)
        for name in names
    }
    diagnostics: dict[str, object] = {"rows": []}
    rng = np.random.default_rng(config.seed)
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            fit_rows = np.flatnonzero((fold_id != held_fold) & (site == site_value))
            held_rows = np.flatnonzero((fold_id == held_fold) & (site == site_value))
            cross = fit_cross_tensor(
                prediction, truth_all, query_ids, fit_rows, config.batch_size
            )
            fitted = {
                "polue": polue_factor(cross),
                "separable": separable_factor(cross, config.separable_iterations),
                "full_link": unit_phase(cross),
            }
            permutation = rng.permutation(16 * 8)
            shuffled = fitted["full_link"].reshape(2, 16 * 8, 4)[:, permutation]
            fitted["full_link_shuffle"] = shuffled.reshape(2, 16, 8, 4)
            for name, factor in fitted.items():
                factors[name][held_rows] = factor.astype(np.complex64)
            diagnostics["rows"].append(
                {
                    "fold": int(held_fold),
                    "site": int(site_value),
                    "fit_count": int(len(fit_rows)),
                    "held_count": int(len(held_rows)),
                    "cross_phase_concentration": float(
                        np.abs(np.sum(cross)) / max(float(np.sum(np.abs(cross))), TINY)
                    ),
                }
            )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FACTOR_CACHE,
        **factors,
        factor_names=np.asarray(names),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    return factors, diagnostics


@torch.inference_mode()
def evaluate_factors(
    factors: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    if METRIC_CACHE.is_file():
        cached = np.load(METRIC_CACHE)
        names = cached["variant_names"].tolist()
        arrays = {key: cached[key] for key in ("pas", "pdp", "error", "energy")}
        return summarize_metrics(names, arrays, cached["fold_id"].astype(np.int8))
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    prediction = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    names = ["identity"] + [
        f"{name}_a{alpha:.2f}"
        for name in factors
        for alpha in (0.25, 0.50, 1.00)
    ]
    arrays = {
        key: np.zeros((len(names), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        source = torch.as_tensor(
            np.array(prediction[start:end], dtype=np.complex64, copy=True), device=device
        )
        target = np.asarray(truth[query_ids[start:end]], dtype=np.complex64)
        variants = {"identity": source}
        for factor_name, factor_values in factors.items():
            factor = torch.as_tensor(
                factor_values[start:end], dtype=torch.complex64, device=device
            ).reshape(end - start, 256, 4, 1)
            for alpha in (0.25, 0.50, 1.00):
                blended = torch.as_tensor(
                    blend_phase(factor.cpu().numpy(), alpha), device=device
                )
                variants[f"{factor_name}_a{alpha:.2f}"] = source * blended
        for index, name in enumerate(names):
            metric = point_metric_statistics(variants[name].cpu().numpy(), target)
            arrays["pas"][index, start:end] = metric.pas
            arrays["pdp"][index, start:end] = metric.pdp
            arrays["error"][index, start:end] = metric.squared_error
            arrays["energy"][index, start:end] = metric.truth_energy
        print(f"static link phase evaluation {end}/{len(query_ids)}", flush=True)
    np.savez_compressed(
        METRIC_CACHE,
        **arrays,
        variant_names=np.asarray(names),
        fold_id=fold_id,
    )
    return summarize_metrics(names, arrays, fold_id)


def summarize_metrics(
    names: list[str], arrays: dict[str, np.ndarray], fold_id: np.ndarray
) -> dict[str, object]:
    variants: dict[str, object] = {}
    for index, name in enumerate(names):
        overall = statistics(
            arrays["pas"][index],
            arrays["pdp"][index],
            arrays["error"][index],
            arrays["energy"][index],
        )
        folds = [
            {
                "fold": int(fold),
                **statistics(
                    arrays["pas"][index],
                    arrays["pdp"][index],
                    arrays["error"][index],
                    arrays["energy"][index],
                    fold_id == fold,
                ),
            }
            for fold in np.unique(fold_id)
        ]
        variants[name] = {"all": overall, "folds": folds}
    baseline = float(variants["identity"]["all"]["score"])
    baseline_folds = variants["identity"]["folds"]
    for name, values in variants.items():
        values["delta_vs_v4"] = float(values["all"]["score"] - baseline)
        values["fold_delta_vs_v4"] = [
            float(row["score"] - reference["score"])
            for row, reference in zip(values["folds"], baseline_folds)
        ]
        values["positive_fold_count"] = int(
            np.sum(np.asarray(values["fold_delta_vs_v4"]) > 0.0)
        )
    return {
        "variants": variants,
        "ranking_high_score_first": sorted(
            names,
            key=lambda name: float(variants[name]["all"]["score"]),
            reverse=True,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(batch_size=args.batch_size)
    if args.rebuild:
        for path in (V4_OOF_CHANNEL, FACTOR_CACHE, METRIC_CACHE):
            if path.is_file():
                path.unlink()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    v4_integrity = build_v4_oof(device, args.batch_size)
    factors, diagnostics = fit_factors(config)
    evaluation = evaluate_factors(factors, device, args.batch_size)
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "A fixed antenna/polarization/UE response phase is shared across positions; estimating "
            "that response from the other OOF spatial holes should improve V4 NMSE and PAS without "
            "predicting rapidly varying absolute path phase."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "v4_oof_integrity": v4_integrity,
        "factor_diagnostics": diagnostics,
        "evaluation": evaluation,
        "outputs": {
            "v4_oof_channel": str(V4_OOF_CHANNEL.relative_to(PATHS.root)),
            "factors": str(FACTOR_CACHE.relative_to(PATHS.root)),
            "metrics": str(METRIC_CACHE.relative_to(PATHS.root)),
        },
        "seconds": float(time.time() - started),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
