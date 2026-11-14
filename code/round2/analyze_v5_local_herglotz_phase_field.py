#!/usr/bin/env python3
"""Fit query-local fixed-wavenumber Herglotz fields for Round 2 phase."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from analyze_v5_angular_spectrum_phase_kernel import (
    MODEL_CACHE as ANGULAR_MODEL_CACHE,
    point_errors,
    summarize_candidates,
)
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E29-local-herglotz-phase-field"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_local_herglotz_phase_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_local_herglotz_phase_field"
INNER_CACHE = ARTIFACT_DIR / "oof_local_herglotz_inner_products.npz"
REPORT = ANALYSIS_DIR / "local_herglotz_phase_field.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
POWER_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_train_point_power.npy"
SPLITS = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
V4_OOF_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "v5_static_link_phase_calibration"
    / "v4_nested_oof_channel.npy"
)
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    directions: int = 72
    sectors: int = 8
    maximum_neighbors: int = 48
    coefficient_l1_cap: float = 4.0
    seed: int = 20_260_805


@dataclass(frozen=True)
class Variant:
    name: str
    source: str
    neighbors: int
    ridge: float
    conjugate: bool = False


def variants() -> list[Variant]:
    output: list[Variant] = []
    for source in ("uniform", "sector"):
        for neighbors in (16, 32, 48):
            for ridge in (0.03, 0.30, 3.00):
                output.append(
                    Variant(
                        name=f"{source}_k{neighbors}_r{ridge:g}",
                        source=source,
                        neighbors=neighbors,
                        ridge=ridge,
                    )
                )
    output.extend(
        [
            Variant("global_k32_r0.3", "global", 32, 0.30),
            Variant("radial_control_k32_r0.3", "radial_control", 32, 0.30),
            Variant("rotated_control_k32_r0.3", "rotated_control", 32, 0.30),
            Variant(
                "sector_conjugate_control_k32_r0.3",
                "sector",
                32,
                0.30,
                conjugate=True,
            ),
        ]
    )
    return output


def load_models(config: Config) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    cached = np.load(ANGULAR_MODEL_CACHE)
    models: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for fold in range(5):
        for site in range(2):
            prefix = f"fold{fold}_site{site}"
            global_spectrum = cached[f"{prefix}_global"].astype(np.float64)
            sector_spectra = cached[f"{prefix}_sectors"].astype(np.float64)
            if len(global_spectrum) != config.directions:
                raise ValueError("angular direction count does not match E27 cache")
            models[(fold, site)] = {
                "global": global_spectrum,
                "sectors": sector_spectra,
                "beta": np.asarray(cached[f"{prefix}_beta"], dtype=np.float64),
            }
    return models


def query_spectra(
    query_position: np.ndarray,
    bs_position: np.ndarray,
    model: dict[str, np.ndarray],
    config: Config,
) -> dict[str, np.ndarray]:
    radial = query_position[:2] - bs_position[:2]
    azimuth = math.atan2(float(radial[1]), float(radial[0]))
    sector = int(
        math.floor(
            ((azimuth + math.pi) % (2.0 * math.pi))
            * config.sectors
            / (2.0 * math.pi)
        )
    )
    uniform = np.full(config.directions, 1.0 / config.directions, dtype=np.float64)
    radial_only = np.zeros(config.directions, dtype=np.float64)
    radial_only[config.directions // 2] = 1.0
    # Relative angles run from -pi to pi, so index directions/2 is zero radians.
    return {
        "uniform": uniform,
        "global": model["global"],
        "sector": model["sectors"][sector],
        "rotated_control": np.roll(model["sectors"][sector], 7),
        "radial_control": radial_only,
    }


def herglotz_coefficients(
    donor_position: np.ndarray,
    query_position: np.ndarray,
    bs_position: np.ndarray,
    beta: float,
    spectrum: np.ndarray,
    ridge: float,
    l1_cap: float,
) -> tuple[np.ndarray, float, float]:
    radial = query_position[:2] - bs_position[:2]
    radial_angle = math.atan2(float(radial[1]), float(radial[0]))
    relative_angle = np.linspace(-np.pi, np.pi, len(spectrum), endpoint=False)
    angle = radial_angle + relative_angle
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    delta = donor_position[:, :2] - query_position[None, :2]
    feature = np.exp(-1j * beta * (delta @ direction.T))
    feature *= np.sqrt(np.maximum(spectrum, 0.0))[None]
    query_feature = np.sqrt(np.maximum(spectrum, 0.0)).astype(np.complex128)
    distance = np.linalg.norm(delta, axis=1)
    scale = max(float(np.median(distance)), 1.0)
    spatial_weight = np.exp(-0.5 * (distance / (1.5 * scale)) ** 2)
    kernel = feature @ np.conj(feature.T)
    query_kernel = query_feature @ np.conj(feature.T)
    system = kernel + ridge * np.diag(1.0 / np.maximum(spatial_weight, 1e-4))
    try:
        coefficient = np.linalg.solve(system.T, query_kernel.T).T
    except np.linalg.LinAlgError:
        coefficient = query_kernel @ np.linalg.pinv(system, rcond=1e-8)
    l1_before = float(np.sum(np.abs(coefficient)))
    if l1_before > l1_cap:
        coefficient *= l1_cap / l1_before
    condition = float(np.linalg.cond(system))
    return coefficient.astype(np.complex64), l1_before, condition


def coefficient_bank(
    donor_position: np.ndarray,
    query_position: np.ndarray,
    bs_position: np.ndarray,
    model: dict[str, np.ndarray],
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra = query_spectra(query_position, bs_position, model, config)
    rows: list[np.ndarray] = []
    l1: list[float] = []
    condition: list[float] = []
    for variant in variants():
        coefficient, raw_l1, matrix_condition = herglotz_coefficients(
            donor_position[: variant.neighbors],
            query_position,
            bs_position,
            float(model["beta"]),
            spectra[variant.source],
            variant.ridge,
            config.coefficient_l1_cap,
        )
        if variant.conjugate:
            coefficient = np.conj(coefficient)
        padded = np.zeros(config.maximum_neighbors, dtype=np.complex64)
        padded[: variant.neighbors] = coefficient
        rows.append(padded)
        l1.append(raw_l1)
        condition.append(matrix_condition)
    return np.stack(rows), np.asarray(l1), np.asarray(condition)


@torch.inference_mode()
def generate_inner_products(
    config: Config, device: torch.device, rebuild: bool
) -> dict[str, np.ndarray]:
    if INNER_CACHE.is_file() and not rebuild:
        return {key: value for key, value in np.load(INNER_CACHE).items()}
    setup, positions, _, site, _ = load_positions()
    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    reference = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    power = np.load(POWER_CACHE).astype(np.float64)
    split = np.load(SPLITS)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    model_bank = load_models(config)
    configs = variants()
    arrays: dict[str, np.ndarray] = {
        "predicted_energy": np.zeros((len(configs), len(query_ids)), dtype=np.float64),
        "predicted_truth_cross_real": np.zeros(
            (len(configs), len(query_ids)), dtype=np.float64
        ),
        "predicted_truth_cross_imag": np.zeros(
            (len(configs), len(query_ids)), dtype=np.float64
        ),
        "reference_predicted_cross_real": np.zeros(
            (len(configs), len(query_ids)), dtype=np.float64
        ),
        "reference_predicted_cross_imag": np.zeros(
            (len(configs), len(query_ids)), dtype=np.float64
        ),
        "coefficient_l1": np.zeros((len(configs), len(query_ids)), dtype=np.float32),
        "system_condition": np.zeros((len(configs), len(query_ids)), dtype=np.float32),
        "truth_energy": np.zeros(len(query_ids), dtype=np.float64),
        "reference_energy": np.zeros(len(query_ids), dtype=np.float64),
        "reference_truth_cross_real": np.zeros(len(query_ids), dtype=np.float64),
        "reference_truth_cross_imag": np.zeros(len(query_ids), dtype=np.float64),
    }
    row_by_id = np.full(len(positions), -1, dtype=np.int64)
    row_by_id[query_ids] = np.arange(len(query_ids))

    for fold in range(5):
        fit = split[f"fold{fold}_fit"].astype(np.int64)
        held = query_ids[fold_id == fold]
        for site_value in np.unique(site[held]):
            fit_site = fit[(site[fit] == site_value) & (power[fit] > TINY)]
            held_site = held[site[held] == site_value]
            tree = cKDTree(positions[fit_site, :2])
            _, local = tree.query(
                positions[held_site, :2], k=config.maximum_neighbors, workers=-1
            )
            donors = fit_site[local]
            model = model_bank[(fold, int(site_value))]
            for local_row, query_id in enumerate(held_site):
                output_row = int(row_by_id[query_id])
                coefficients, l1, condition = coefficient_bank(
                    positions[donors[local_row]],
                    positions[query_id],
                    setup.bs_positions[int(site_value)],
                    model,
                    config,
                )
                donor_tensor = torch.as_tensor(
                    np.array(
                        channel[donors[local_row]], dtype=np.complex64, copy=True
                    ).reshape(config.maximum_neighbors, -1),
                    device=device,
                )
                prediction = torch.as_tensor(coefficients, device=device) @ donor_tensor
                truth = torch.as_tensor(
                    np.array(channel[query_id], dtype=np.complex64, copy=True).reshape(-1),
                    device=device,
                )
                ref = torch.as_tensor(
                    np.array(reference[output_row], dtype=np.complex64, copy=True).reshape(-1),
                    device=device,
                )
                pred_truth = torch.sum(torch.conj(prediction) * truth[None], dim=1)
                ref_pred = torch.sum(torch.conj(ref[None]) * prediction, dim=1)
                arrays["predicted_energy"][:, output_row] = (
                    prediction.abs().square().sum(dim=1).double().cpu().numpy()
                )
                arrays["predicted_truth_cross_real"][:, output_row] = (
                    pred_truth.real.double().cpu().numpy()
                )
                arrays["predicted_truth_cross_imag"][:, output_row] = (
                    pred_truth.imag.double().cpu().numpy()
                )
                arrays["reference_predicted_cross_real"][:, output_row] = (
                    ref_pred.real.double().cpu().numpy()
                )
                arrays["reference_predicted_cross_imag"][:, output_row] = (
                    ref_pred.imag.double().cpu().numpy()
                )
                arrays["coefficient_l1"][:, output_row] = l1.astype(np.float32)
                arrays["system_condition"][:, output_row] = np.minimum(
                    condition, np.finfo(np.float32).max
                ).astype(np.float32)
                arrays["truth_energy"][output_row] = float(
                    truth.abs().square().sum().double().cpu()
                )
                arrays["reference_energy"][output_row] = float(
                    ref.abs().square().sum().double().cpu()
                )
                ref_truth = torch.sum(torch.conj(ref) * truth)
                arrays["reference_truth_cross_real"][output_row] = float(
                    ref_truth.real.double().cpu()
                )
                arrays["reference_truth_cross_imag"][output_row] = float(
                    ref_truth.imag.double().cpu()
                )
                if (output_row + 1) % 25 == 0:
                    print(f"local Herglotz OOF row={output_row + 1}/500", flush=True)
    arrays["variant_names"] = np.asarray([variant.name for variant in configs])
    arrays["query_ids"] = query_ids
    arrays["fold_id"] = fold_id
    arrays["site"] = site[query_ids].astype(np.int8)
    np.savez_compressed(INNER_CACHE, **arrays)
    return arrays


def summarize_matrix(values: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        variant.name: {
            "mean": float(np.mean(values[index])),
            "p50": float(np.median(values[index])),
            "p90": float(np.quantile(values[index], 0.90)),
            "max": float(np.max(values[index])),
        }
        for index, variant in enumerate(variants())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = generate_inner_products(config, torch.device(args.device), args.rebuild)
    names, errors = point_errors(arrays)
    candidate_rows, nested = summarize_candidates(names, errors, arrays)
    report = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "Irregular official spatial samples identify a query-local superposition of "
            "fixed-wavenumber plane waves when donor coefficients are solved jointly, "
            "even though E27's direct covariance-weighted average failed."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": (
            "fixed five complete spatial holes; local regression uses only each fold's "
            "fit donors and fit-only E27 angular spectra"
        ),
        "config": asdict(config),
        "variants": [asdict(variant) for variant in variants()],
        "nested_nmse_selection": nested,
        "top_candidates": candidate_rows[:30],
        "coefficient_l1": summarize_matrix(arrays["coefficient_l1"]),
        "system_condition": summarize_matrix(arrays["system_condition"]),
        "outputs": {"inner_products": PATHS.relative(INNER_CACHE)},
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nested_nmse_selection": nested,
                "top_candidates": candidate_rows[:8],
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
