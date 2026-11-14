#!/usr/bin/env python3
"""Resolve path-carrier phase cycles across Round 2 spatial holes."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from data import load_positions
from project_paths import PATHS
from run_path_component_transport import (
    PORTS,
    S_SC,
    TINY,
    ComponentCache,
    Config as TransportConfig,
    circular_delta,
    component_energy,
    match_components,
    synthesize_batch,
    weighted_intercept,
)
from run_rbf_spectral_field import OOF_META


VERSION = "Round2-E24-cycle-resolved-path-phase"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_cycle_resolved_path_phase"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_cycle_resolved_path_phase"
PREDICTIONS = ARTIFACT_DIR / "oof_cycle_resolved_path_coefficients.npz"
REPORT = ANALYSIS_DIR / "cycle_resolved_path_phase.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
COMPONENT_CACHE = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
TRANSPORT_PREDICTION = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "oof_path_parameters_k16_os2_ref2_n40_dir3600_r1.npz"
)
MAP_PHASE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_v4_map_virtual_source_phase"
    / "map_virtual_source_phase.json"
)

LIGHT_SPEED = 299_792_458.0
SUBCARRIER_SPACING = 30_000.0
METERS_PER_DELAY_BIN = LIGHT_SPEED / (S_SC * SUBCARRIER_SPACING)


@dataclass(frozen=True)
class Config:
    paths: int = 16
    donors: int = 40
    minimum_observations: int = 6
    cycle_iterations: int = 4
    irls_iterations: int = 4
    linear_ridge: float = 0.04
    quadratic_ridge: float = 0.35
    coarse_delay_weight: float = 0.02
    maximum_gradient: float = 1.05
    seed: int = 20_260_805


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def load_components() -> ComponentCache:
    cached = np.load(COMPONENT_CACHE)
    return ComponentCache(
        coordinates=cached["coordinates"].astype(np.float32),
        coefficients=cached["coefficients"].astype(np.complex64),
        residual_fraction=cached["residual_fraction"].astype(np.float32),
        point_power=cached["point_power"].astype(np.float64),
    )


def load_beta_table() -> dict[tuple[int, int], float]:
    payload = json.loads(MAP_PHASE_REPORT.read_text(encoding="utf-8"))
    rows = payload["diagnostics"]["beta_calibration"]
    return {
        (int(row["fold"]), int(row["site"])): float(row["beta_rad_per_m"])
        for row in rows
    }


def design_matrix(offsets: np.ndarray, mode: str, scale: float) -> np.ndarray:
    normalized = np.asarray(offsets, dtype=np.float64) / scale
    x = normalized[:, 0]
    y = normalized[:, 1]
    if mode == "linear":
        return np.column_stack([x, y])
    if mode == "quadratic":
        return np.column_stack([x, y, 0.5 * x * x, x * y, 0.5 * y * y])
    raise ValueError(mode)


def weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    coarse: np.ndarray,
    weight: np.ndarray,
    ridge: float,
    coarse_weight: float,
    irls_iterations: int,
) -> np.ndarray:
    robust = np.ones(len(target), dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.full(design.shape[1], ridge, dtype=np.float64)
    penalty[:2] *= 0.25
    for _ in range(irls_iterations):
        effective = np.maximum(weight * robust, 1e-8)
        combined_weight = effective * (1.0 + coarse_weight)
        combined_target = (target + coarse_weight * coarse) / (1.0 + coarse_weight)
        gram = design.T @ (combined_weight[:, None] * design)
        gram += np.diag(penalty)
        right = design.T @ (combined_weight * combined_target)
        try:
            coefficients = np.linalg.solve(gram, right)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(gram, right, rcond=1e-8)[0]
        residual = target - design @ coefficients
        center = float(np.median(residual))
        deviation = np.abs(residual - center)
        scale = max(1.4826 * float(np.median(deviation)), 1e-4)
        robust = np.minimum(1.0, 2.5 * scale / np.maximum(deviation, 1e-12))
    return coefficients


def constrain_gradient(
    coefficients: np.ndarray, scale: float, maximum_gradient: float
) -> np.ndarray:
    output = np.array(coefficients, dtype=np.float64, copy=True)
    gradient = output[:2] / scale
    norm = float(np.linalg.norm(gradient))
    if norm > maximum_gradient:
        output[:2] *= maximum_gradient / norm
    return output


def fit_cycle_surface(
    offsets: np.ndarray,
    query_offset: np.ndarray,
    phase: np.ndarray,
    coarse_length: np.ndarray,
    weight: np.ndarray,
    beta: float,
    mode: str,
    config: Config,
    shuffle_delay: bool,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, dict[str, float]]:
    spatial_scale = max(float(np.median(np.linalg.norm(offsets, axis=1))), 1.0)
    design = design_matrix(offsets, mode, spatial_scale)
    query_design = design_matrix(query_offset[None], mode, spatial_scale)[0]
    coarse = np.array(coarse_length, dtype=np.float64, copy=True)
    if shuffle_delay and len(coarse) > 2:
        order = np.arange(len(coarse))
        order[1:] = rng.permutation(order[1:])
        coarse = coarse[order]
        coarse[0] = 0.0

    cycle = np.rint((beta * coarse - phase) / (2.0 * math.pi))
    initial_cycle = cycle.copy()
    unwrapped = (phase + 2.0 * math.pi * cycle) / beta
    ridge = config.linear_ridge if mode == "linear" else config.quadratic_ridge
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(config.cycle_iterations):
        coefficients = weighted_ridge(
            design,
            unwrapped,
            coarse,
            weight,
            ridge,
            config.coarse_delay_weight,
            config.irls_iterations,
        )
        coefficients = constrain_gradient(
            coefficients, spatial_scale, config.maximum_gradient
        )
        predicted = design @ coefficients
        cycle = np.rint((beta * predicted - phase) / (2.0 * math.pi))
        unwrapped = (phase + 2.0 * math.pi * cycle) / beta

    predicted = design @ coefficients
    residual_phase = phase - beta * predicted
    normalized_weight = weight / max(float(np.sum(weight)), TINY)
    coherence = float(
        np.abs(np.sum(normalized_weight * np.exp(1j * residual_phase)))
    )
    delay_residual = unwrapped - coarse_length
    gradient_norm = float(np.linalg.norm(coefficients[:2] / spatial_scale))
    return float(query_design @ coefficients), unwrapped, {
        "fit_coherence": coherence,
        "gradient_norm": gradient_norm,
        "cycle_change_fraction": float(np.mean(cycle != initial_cycle)),
        "delay_residual_abs_median_m": float(np.median(np.abs(delay_residual))),
    }


def path_observations(
    donor_ids: np.ndarray,
    cache: ComponentCache,
    transport_config: TransportConfig,
    paths: int,
) -> list[list[tuple[int, int, float]]]:
    anchor_id = int(donor_ids[0])
    anchor_coordinates = cache.coordinates[anchor_id, :paths]
    anchor_coefficients = cache.coefficients[anchor_id, :paths]
    observations: list[list[tuple[int, int, float]]] = [
        [(anchor_id, path, 1.0)] for path in range(paths)
    ]
    for donor_id in donor_ids[1:]:
        donor_id = int(donor_id)
        matched, _, coherence = match_components(
            anchor_coordinates,
            anchor_coefficients,
            cache.coordinates[donor_id],
            cache.coefficients[donor_id],
            transport_config,
        )
        for path, component in enumerate(matched):
            if component >= 0:
                observations[path].append(
                    (donor_id, int(component), float(coherence[path]))
                )
    return observations


def predict_query(
    row: int,
    query_id: int,
    donor_ids: np.ndarray,
    query_position: np.ndarray,
    positions: np.ndarray,
    cache: ComponentCache,
    predicted_coordinates: np.ndarray,
    wave_coefficients: np.ndarray,
    beta: float,
    transport_config: TransportConfig,
    config: Config,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    observations = path_observations(
        donor_ids, cache, transport_config, config.paths
    )
    variants = {
        "cycle_linear": np.array(wave_coefficients[row], copy=True),
        "cycle_quadratic": np.array(wave_coefficients[row], copy=True),
        "delay_shuffle_control": np.array(wave_coefficients[row], copy=True),
    }
    diagnostics: list[dict[str, float]] = []
    rng = np.random.default_rng(config.seed + int(query_id) * 1009)

    for path, rows in enumerate(observations):
        if len(rows) < config.minimum_observations:
            continue
        ids = np.asarray([value[0] for value in rows], dtype=np.int64)
        components = np.asarray([value[1] for value in rows], dtype=np.int64)
        match_coherence = np.asarray([value[2] for value in rows], dtype=np.float64)
        coordinates = cache.coordinates[ids, components]
        coefficients = cache.coefficients[ids, components]
        norms = np.linalg.norm(coefficients, axis=1)
        valid = norms > 1e-20
        if int(np.sum(valid)) < config.minimum_observations:
            continue
        ids = ids[valid]
        coordinates = coordinates[valid]
        coefficients = coefficients[valid]
        norms = norms[valid]
        match_coherence = match_coherence[valid]

        anchor_position = positions[ids[0], :2]
        offsets = positions[ids, :2] - anchor_position[None]
        query_offset = query_position[:2] - anchor_position
        query_distance = np.linalg.norm(positions[ids, :2] - query_position[None, :2], axis=1)
        donor_scale = max(float(np.median(query_distance)), 1.0)
        spatial_weight = np.exp(-0.5 * (query_distance / max(1.5 * donor_scale, 1.0)) ** 2)
        weight = spatial_weight * np.maximum(match_coherence, 0.10) * np.sqrt(
            np.maximum(norms / max(float(norms[0]), 1e-20), 1e-4)
        )
        weight /= max(float(np.sum(weight)), TINY)

        unit = coefficients / norms[:, None]
        inner = np.sum(unit[0][None] * np.conj(unit), axis=1)
        phase = np.angle(inner).astype(np.float64)
        phase[0] = 0.0
        coarse_length = circular_delta(
            coordinates[:, 2], coordinates[0, 2], S_SC
        ).astype(np.float64) * METERS_PER_DELAY_BIN
        coarse_length[0] = 0.0

        model_rows: dict[str, dict[str, float]] = {}
        for name, mode, shuffled in (
            ("cycle_linear", "linear", False),
            ("cycle_quadratic", "quadratic", False),
            ("delay_shuffle_control", "linear", True),
        ):
            query_length, observed_length, diagnostic = fit_cycle_surface(
                offsets,
                query_offset,
                phase,
                coarse_length,
                weight,
                beta,
                mode,
                config,
                shuffled,
                rng,
            )
            transported = coefficients * np.exp(
                -1j * beta * (query_length - observed_length)
            )[:, None]
            vector = np.sum(weight[:, None] * transported, axis=0)
            predicted_log_norm = weighted_intercept(
                np.log(np.maximum(norms, 1e-20)),
                positions[ids, :2] - query_position[None, :2],
                weight,
                2.0,
                0.75,
            )
            vector_norm = max(float(np.linalg.norm(vector)), 1e-20)
            vector *= math.exp(predicted_log_norm) / vector_norm
            variants[name][path] = vector.astype(np.complex64)
            model_rows[name] = diagnostic
        diagnostics.append(
            {
                "path_energy": float(np.sum(np.abs(variants["cycle_linear"][path]) ** 2)),
                "observations": float(len(ids)),
                **{
                    f"{name}_{key}": value
                    for name, values in model_rows.items()
                    for key, value in values.items()
                },
            }
        )
    return variants, diagnostics


def predict_oof(config: Config) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    if PREDICTIONS.is_file():
        cached = np.load(PREDICTIONS)
        names = cached["variant_names"].tolist()
        variants = {name: cached[name].astype(np.complex64) for name in names}
        return (
            cached["coordinates"].astype(np.float32),
            variants,
            json.loads(str(cached["diagnostics_json"])),
        )

    cache = load_components()
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    folds = np.load(FOLD_CACHE)
    transport = np.load(TRANSPORT_PREDICTION)
    predicted_coordinates = transport["coordinates"].astype(np.float32)
    wave_coefficients = transport["coefficients"].astype(np.complex64)
    setup, positions, _, site, _ = load_positions()
    del setup
    beta_table = load_beta_table()
    transport_config = TransportConfig(
        components=24,
        transported_components=config.paths,
        oversample=2,
        refinement_sweeps=2,
        donor_count=config.donors,
        direction_bins=3600,
        beta=140.40,
        radius_fractions=(1.0,),
    )
    variant_names = ("cycle_linear", "cycle_quadratic", "delay_shuffle_control")
    outputs = {
        name: np.array(wave_coefficients, dtype=np.complex64, copy=True)
        for name in variant_names
    }
    valid = cache.point_power > TINY
    diagnostic_rows: list[dict[str, float]] = []

    for held_fold in np.unique(fold_id):
        rows = np.flatnonzero(fold_id == held_fold)
        fit_ids = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
        for site_value in np.unique(site[query_ids[rows]]):
            selected_rows = rows[site[query_ids[rows]] == site_value]
            bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
            _, local = cKDTree(positions[bank, :2]).query(
                positions[query_ids[selected_rows], :2],
                k=min(config.donors, len(bank)),
                workers=-1,
            )
            donor_table = bank[local]
            beta = beta_table[(int(held_fold), int(site_value))]
            for local_row, output_row in enumerate(selected_rows):
                variants, diagnostics = predict_query(
                    int(output_row),
                    int(query_ids[output_row]),
                    donor_table[local_row],
                    positions[query_ids[output_row]],
                    positions,
                    cache,
                    predicted_coordinates,
                    wave_coefficients,
                    beta,
                    transport_config,
                    config,
                )
                for name in variant_names:
                    outputs[name][output_row] = variants[name]
                diagnostic_rows.extend(diagnostics)
        print(f"cycle phase prediction fold={int(held_fold)} rows={len(rows)}", flush=True)

    diagnostics: dict[str, object] = {
        "path_rows": int(len(diagnostic_rows)),
        "observations": summarize(
            np.asarray([row["observations"] for row in diagnostic_rows])
        ),
    }
    for name in variant_names:
        for key in (
            "fit_coherence",
            "gradient_norm",
            "cycle_change_fraction",
            "delay_residual_abs_median_m",
        ):
            diagnostics[f"{name}_{key}"] = summarize(
                np.asarray([row[f"{name}_{key}"] for row in diagnostic_rows])
            )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PREDICTIONS,
        coordinates=predicted_coordinates,
        wavevector=wave_coefficients,
        **outputs,
        variant_names=np.asarray(variant_names),
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    return predicted_coordinates, outputs, diagnostics


@torch.inference_mode()
def evaluate_raw_paths(
    coordinates: np.ndarray,
    variants: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    transport = np.load(TRANSPORT_PREDICTION)
    all_variants = {"wavevector": transport["coefficients"].astype(np.complex64), **variants}
    accumulators = {
        name: {
            "error": np.zeros(len(query_ids), dtype=np.float64),
            "truth": np.zeros(len(query_ids), dtype=np.float64),
            "prediction": np.zeros(len(query_ids), dtype=np.float64),
            "cross_real": np.zeros(len(query_ids), dtype=np.float64),
            "cross_abs": np.zeros(len(query_ids), dtype=np.float64),
        }
        for name in all_variants
    }
    cache = load_components()
    extraction_error = np.zeros(len(query_ids), dtype=np.float64)

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        truth = np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64)
        truth_tensor = torch.as_tensor(truth, device=device)
        truth_energy = np.sum(np.abs(truth) ** 2, axis=(1, 2, 3), dtype=np.float64)
        true_path = synthesize_batch(
            cache.coordinates[query_ids[start:end]],
            cache.coefficients[query_ids[start:end]],
            device,
        )
        extraction_error[start:end] = np.sum(
            np.abs(true_path.cpu().numpy() - truth) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        for name, coefficients in all_variants.items():
            prediction = synthesize_batch(
                coordinates[start:end], coefficients[start:end], device
            )
            error = (prediction - truth_tensor).abs().square().sum(dim=(1, 2, 3))
            predicted_energy = prediction.abs().square().sum(dim=(1, 2, 3))
            cross = (torch.conj(prediction) * truth_tensor).sum(dim=(1, 2, 3))
            values = accumulators[name]
            values["error"][start:end] = error.cpu().numpy()
            values["truth"][start:end] = truth_energy
            values["prediction"][start:end] = predicted_energy.cpu().numpy()
            values["cross_real"][start:end] = cross.real.cpu().numpy()
            values["cross_abs"][start:end] = cross.abs().cpu().numpy()
        print(f"cycle phase raw evaluation {end}/{len(query_ids)}", flush=True)

    results: dict[str, object] = {
        "truth_component_extraction_nmse": float(
            np.sum(extraction_error)
            / max(float(np.sum(next(iter(accumulators.values()))["truth"])), TINY)
        ),
        "variants": {},
    }
    for name, values in accumulators.items():
        total_truth = float(np.sum(values["truth"]))
        total_prediction = float(np.sum(values["prediction"]))
        nmse = float(np.sum(values["error"]) / max(total_truth, TINY))
        signed_coherence = float(
            np.sum(values["cross_real"])
            / max(math.sqrt(total_truth * total_prediction), TINY)
        )
        absolute_coherence = float(
            np.sum(values["cross_abs"])
            / max(
                float(np.sum(np.sqrt(values["truth"] * values["prediction"]))),
                TINY,
            )
        )
        folds = []
        for fold in np.unique(fold_id):
            mask = fold_id == fold
            fold_truth = float(np.sum(values["truth"][mask]))
            fold_prediction = float(np.sum(values["prediction"][mask]))
            folds.append(
                {
                    "fold": int(fold),
                    "nmse": float(
                        np.sum(values["error"][mask]) / max(fold_truth, TINY)
                    ),
                    "signed_coherence": float(
                        np.sum(values["cross_real"][mask])
                        / max(math.sqrt(fold_truth * fold_prediction), TINY)
                    ),
                }
            )
        results["variants"][name] = {
            "nmse": nmse,
            "signed_coherence": signed_coherence,
            "absolute_point_coherence": absolute_coherence,
            "folds": folds,
        }
    results["ranking_low_nmse_first"] = sorted(
        accumulators,
        key=lambda name: float(results["variants"][name]["nmse"]),
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.recompute and PREDICTIONS.is_file():
        PREDICTIONS.unlink()
    device = torch.device(args.device)
    coordinates, variants, diagnostics = predict_oof(Config())
    evaluation = evaluate_raw_paths(coordinates, variants, device, args.batch_size)
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "Continuous delay resolves the integer-cycle branch of each tracked path phase; "
            "a robust eikonal-constrained local phase surface can then extrapolate path complex "
            "coefficients across a complete spatial hole better than free wavevector fitting."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(Config()),
        "diagnostics": diagnostics,
        "raw_path_evaluation": evaluation,
        "outputs": {
            "predictions": str(PREDICTIONS.relative_to(PATHS.root)),
        },
        "seconds": float(time.time() - started),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
