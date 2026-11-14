#!/usr/bin/env python3
"""Deploy fit-only wall-identity carrier-phase transport into the V4 carrier."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from analyze_v4_map_virtual_source_phase import (
    COMPONENT_CACHE,
    TRANSPORT_PREDICTION,
    Config as MapConfig,
    build_wall_bank,
    image_sources,
)
from analyze_v5_explicit_virtual_path_identity import (
    ASSIGNMENT_CACHE,
    FOLDS,
    LOS_ID,
    UNASSIGNED,
    Config as IdentityConfig,
    assign_point_components,
    load_fold_parameters,
    source_lengths,
    unit_vector,
)
from analyze_v5_path_relative_port_phase import (
    TRAIN_CHANNEL,
    V4_OOF_CHANNEL,
    fixed_coordinate_decomposition,
    ports_to_channel,
    preserve_power,
)
from data import load_positions
from official_metrics import point_metric_statistics
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, statistics


VERSION = "Round2-E31-deployable-wall-phase-transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_deployable_wall_phase_transport"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_v5_deployable_wall_phase_transport"
)
PHASE_CACHE = ARTIFACT_DIR / "oof_wall_phase_predictions.npz"
METRIC_CACHE = ARTIFACT_DIR / "oof_wall_phase_metrics.npz"
REPORT = ANALYSIS_DIR / "deployable_wall_phase_transport.json"
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    donor_count: int = 24
    minimum_observations: int = 4
    maximum_donor_distance_m: float = 55.0
    distance_scale_floor_m: float = 5.0
    wrong_beta_offset: float = 0.20
    decomposition_ridge: float = 0.003
    alphas: tuple[float, ...] = (0.10, 0.25, 0.50)


def load_component_coefficients() -> np.ndarray:
    return np.load(COMPONENT_CACHE)["coefficients"].astype(np.complex64)


def weighted_unit_average(value: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float]:
    raw = np.sum(weight[:, None] * value, axis=0)
    return unit_vector(raw), float(np.linalg.norm(raw))


def build_phase_predictions(
    config: Config,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if PHASE_CACHE.is_file() and not rebuild:
        return {key: value for key, value in np.load(PHASE_CACHE).items()}
    setup, positions, _, site, _ = load_positions()
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    folds = np.load(FOLDS)
    assignments = np.load(ASSIGNMENT_CACHE)
    components = load_component_coefficients()
    transport = np.load(TRANSPORT_PREDICTION)
    coordinates = transport["coordinates"].astype(np.float32)
    fallback = transport["coefficients"].astype(np.complex64)
    identity_config = IdentityConfig()
    map_config = MapConfig()
    wall_bank = build_wall_bank(map_config)
    fold_parameters = load_fold_parameters()

    output = np.array(fallback, copy=True)
    rotated_output = np.array(fallback, copy=True)
    wrong_beta_output = np.array(fallback, copy=True)
    active = np.zeros(coordinates.shape[:2], dtype=bool)
    path_id = np.full(coordinates.shape[:2], UNASSIGNED, dtype=np.int32)
    fit_coherence = np.zeros(coordinates.shape[:2], dtype=np.float32)
    rotated_coherence = np.zeros(coordinates.shape[:2], dtype=np.float32)
    observation_count = np.zeros(coordinates.shape[:2], dtype=np.int16)
    nearest_distance = np.full(coordinates.shape[:2], np.inf, dtype=np.float32)

    for fold in range(5):
        fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
        rows = np.flatnonzero(fold_id == fold)
        for site_value in range(2):
            local_rows = rows[site[query_ids[rows]] == site_value]
            fit_site = fit_ids[site[fit_ids] == site_value]
            sources = image_sources(wall_bank, setup.bs_positions[site_value])
            parameters = fold_parameters[(fold, site_value)]
            beta = parameters["beta"]
            identity_bank: dict[int, list[tuple[int, int]]] = {}
            for point_id in fit_site:
                for component, identity in enumerate(
                    assignments["path_id"][fold, point_id]
                ):
                    if identity >= 0:
                        identity_bank.setdefault(int(identity), []).append(
                            (int(point_id), int(component))
                        )

            for output_row in local_rows:
                query_id = int(query_ids[output_row])
                assigned = assign_point_components(
                    positions[query_id],
                    setup.bs_positions[site_value],
                    coordinates[output_row],
                    sources,
                    wall_bank,
                    map_config,
                    parameters["orientation"],
                    identity_config,
                )
                for path, identity in enumerate(assigned[0]):
                    path_id[output_row, path] = identity
                    if identity < 0:
                        continue
                    observations = identity_bank.get(int(identity), [])
                    if len(observations) < config.minimum_observations:
                        continue
                    ids = np.asarray([value[0] for value in observations], dtype=np.int64)
                    parts = np.asarray([value[1] for value in observations], dtype=np.int64)
                    distance = np.linalg.norm(
                        positions[ids, :2] - positions[query_id, None, :2], axis=1
                    )
                    keep = distance <= config.maximum_donor_distance_m
                    ids = ids[keep]
                    parts = parts[keep]
                    distance = distance[keep]
                    if len(ids) < config.minimum_observations:
                        continue
                    order = np.argsort(distance)[: config.donor_count]
                    ids = ids[order]
                    parts = parts[order]
                    distance = distance[order]
                    scale = max(
                        float(np.median(distance)), config.distance_scale_floor_m
                    )
                    weight = np.exp(-0.5 * (distance / (1.5 * scale)) ** 2)
                    weight /= max(float(np.sum(weight)), TINY)
                    observed = components[ids, parts]
                    observed /= np.maximum(
                        np.linalg.norm(observed, axis=1, keepdims=True), 1e-20
                    )
                    donor_length = assignments["path_length"][fold, ids, parts]
                    query_length = float(assigned[1][path])
                    latent, coherence = weighted_unit_average(
                        observed * np.exp(1j * beta * donor_length)[:, None], weight
                    )
                    prediction = latent * np.exp(-1j * beta * query_length)

                    rotated_donor_length = source_lengths(
                        ids,
                        int(identity),
                        positions,
                        setup.bs_positions[site_value],
                        sources,
                        True,
                    )
                    rotated_query_length = float(
                        source_lengths(
                            np.asarray([query_id]),
                            int(identity),
                            positions,
                            setup.bs_positions[site_value],
                            sources,
                            True,
                        )[0]
                    )
                    rotated_latent, rotated_fit = weighted_unit_average(
                        observed
                        * np.exp(1j * beta * rotated_donor_length)[:, None],
                        weight,
                    )
                    rotated_prediction = rotated_latent * np.exp(
                        -1j * beta * rotated_query_length
                    )
                    wrong_beta = beta + config.wrong_beta_offset
                    wrong_latent, _ = weighted_unit_average(
                        observed * np.exp(1j * wrong_beta * donor_length)[:, None],
                        weight,
                    )
                    wrong_prediction = wrong_latent * np.exp(
                        -1j * wrong_beta * query_length
                    )

                    output[output_row, path] = prediction
                    rotated_output[output_row, path] = rotated_prediction
                    wrong_beta_output[output_row, path] = wrong_prediction
                    active[output_row, path] = True
                    fit_coherence[output_row, path] = coherence
                    rotated_coherence[output_row, path] = rotated_fit
                    observation_count[output_row, path] = len(ids)
                    nearest_distance[output_row, path] = distance[0]
        print(f"deployable wall phase fold={fold} rows={len(rows)}", flush=True)

    arrays = {
        "coordinates": coordinates,
        "source": output,
        "rotated_source": rotated_output,
        "wrong_beta_source": wrong_beta_output,
        "active": active,
        "path_id": path_id,
        "fit_coherence": fit_coherence,
        "rotated_coherence": rotated_coherence,
        "observation_count": observation_count,
        "nearest_distance": nearest_distance,
        "query_ids": query_ids,
        "fold_id": fold_id,
        "site": site[query_ids].astype(np.int8),
    }
    np.savez_compressed(PHASE_CACHE, **arrays)
    return arrays


def gate_masks(arrays: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    active = arrays["active"].astype(bool)
    quality = arrays["fit_coherence"] - arrays["rotated_coherence"]
    fold_id = arrays["fold_id"]
    site = arrays["site"]
    names = [
        "all_wall",
        "quality_positive",
        "quality_top75",
        "quality_top50",
        "quality_top25",
        "joint_top50",
    ]
    masks = np.zeros((len(names), *active.shape), dtype=bool)
    masks[0] = active
    masks[1] = active & (quality > 0.0)
    for fold in range(5):
        for site_value in range(2):
            point = (fold_id == fold) & (site == site_value)
            local_active = active[point]
            if not np.any(local_active):
                continue
            local_quality = quality[point][local_active]
            local_coherence = arrays["fit_coherence"][point][local_active]
            for index, quantile in ((2, 0.25), (3, 0.50), (4, 0.75)):
                threshold = float(np.quantile(local_quality, quantile))
                masks[index, point] = local_active & (quality[point] >= threshold)
            quality_threshold = float(np.quantile(local_quality, 0.50))
            coherence_threshold = float(np.quantile(local_coherence, 0.50))
            masks[5, point] = (
                local_active
                & (quality[point] >= quality_threshold)
                & (arrays["fit_coherence"][point] >= coherence_threshold)
            )
    return names, masks


def mix_full_phase(
    base: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    base_unit = base / base.abs().clamp_min(1e-20)
    source_unit = source / source.abs().clamp_min(1e-20)
    direction = (1.0 - alpha) * base_unit + alpha * source_unit
    direction /= direction.abs().clamp_min(1e-20)
    candidate = base.abs() * direction
    return torch.where(mask[:, :, None], candidate, base)


def variant_names(gates: list[str], config: Config) -> list[str]:
    return ["identity"] + [
        f"{gate}_a{alpha:.2f}" for gate in gates for alpha in config.alphas
    ]


@torch.inference_mode()
def evaluate_channels(
    arrays: dict[str, np.ndarray],
    config: Config,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> dict[str, object]:
    gate_names, masks = gate_masks(arrays)
    names = variant_names(gate_names, config)
    if METRIC_CACHE.is_file() and not rebuild:
        cached = np.load(METRIC_CACHE)
        metrics = {key: cached[key] for key in ("pas", "pdp", "error", "energy")}
        return summarize_metrics(names, metrics, arrays, masks)

    query_ids = arrays["query_ids"].astype(np.int64)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    reference_all = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    metric = {
        key: np.zeros((len(names), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    phase_diagnostics = {
        key: np.zeros((len(query_ids), arrays["coordinates"].shape[1]), dtype=np.float32)
        for key in ("correct", "rotated", "wrong_beta", "truth_path_energy")
    }

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        reference = torch.as_tensor(
            np.array(reference_all[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        coordinates = arrays["coordinates"][start:end]
        base_coefficient, residual, atoms = fixed_coordinate_decomposition(
            reference, coordinates, config.decomposition_ridge, device
        )
        truth_coefficient, _, _ = fixed_coordinate_decomposition(
            truth, coordinates, config.decomposition_ridge, device
        )
        truth_norm = truth_coefficient.abs().square().sum(dim=2).sqrt().clamp_min(1e-20)
        truth_unit = truth_coefficient / truth_norm[:, :, None]
        for diagnostic_name, source_name in (
            ("correct", "source"),
            ("rotated", "rotated_source"),
            ("wrong_beta", "wrong_beta_source"),
        ):
            source = torch.as_tensor(arrays[source_name][start:end], device=device)
            source /= source.abs().square().sum(dim=2, keepdim=True).sqrt().clamp_min(1e-20)
            phase_diagnostics[diagnostic_name][start:end] = torch.real(
                torch.sum(torch.conj(source) * truth_unit, dim=2)
            ).cpu().numpy()
        phase_diagnostics["truth_path_energy"][start:end] = (
            truth_norm.square().cpu().numpy()
        )

        variants: dict[str, torch.Tensor] = {"identity": reference}
        source = torch.as_tensor(arrays["source"][start:end], device=device)
        for gate_index, gate in enumerate(gate_names):
            mask = torch.as_tensor(masks[gate_index, start:end], device=device)
            for alpha in config.alphas:
                coefficient = mix_full_phase(base_coefficient, source, mask, alpha)
                ports = residual + torch.einsum(
                    "bkc,bkhvs->bchvs", coefficient, atoms
                )
                candidate = preserve_power(ports_to_channel(ports), reference)
                variants[f"{gate}_a{alpha:.2f}"] = candidate
        truth_numpy = truth.cpu().numpy()
        for index, name in enumerate(names):
            point = point_metric_statistics(variants[name].cpu().numpy(), truth_numpy)
            metric["pas"][index, start:end] = point.pas
            metric["pdp"][index, start:end] = point.pdp
            metric["error"][index, start:end] = point.squared_error
            metric["energy"][index, start:end] = point.truth_energy
        print(f"wall phase full metric {end}/{len(query_ids)}", flush=True)

    np.savez_compressed(
        METRIC_CACHE,
        **metric,
        **phase_diagnostics,
        variant_names=np.asarray(names),
        gate_names=np.asarray(gate_names),
    )
    return summarize_metrics(names, metric, arrays, masks, phase_diagnostics)


def summarize_metrics(
    names: list[str],
    metric: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    masks: np.ndarray,
    phase_diagnostics: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    fold_id = arrays["fold_id"]
    site = arrays["site"]
    rows = []
    for index, name in enumerate(names):
        overall = statistics(
            metric["pas"][index],
            metric["pdp"][index],
            metric["error"][index],
            metric["energy"][index],
        )
        rows.append(
            {
                "name": name,
                **overall,
                "fold_score": [
                    statistics(
                        metric["pas"][index],
                        metric["pdp"][index],
                        metric["error"][index],
                        metric["energy"][index],
                        fold_id == fold,
                    )["score"]
                    for fold in range(5)
                ],
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)

    selected = np.zeros(len(fold_id), dtype=np.int64)
    decisions = []
    for held_fold in range(5):
        for site_value in range(2):
            train = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            scores = [
                statistics(
                    metric["pas"][index],
                    metric["pdp"][index],
                    metric["error"][index],
                    metric["energy"][index],
                    train,
                )["score"]
                for index in range(len(names))
            ]
            chosen = int(np.argmax(scores))
            selected[held] = chosen
            decisions.append(
                {
                    "held_fold": held_fold,
                    "site": site_value,
                    "candidate": names[chosen],
                    "training_score": float(scores[chosen]),
                }
            )
    point = np.arange(len(fold_id))
    nested = statistics(
        metric["pas"][selected, point],
        metric["pdp"][selected, point],
        metric["error"][selected, point],
        metric["energy"][selected, point],
    )
    baseline = statistics(
        metric["pas"][0], metric["pdp"][0], metric["error"][0], metric["energy"][0]
    )
    nested["delta_vs_identity"] = float(nested["score"] - baseline["score"])
    nested["fold_delta_vs_identity"] = []
    for fold in range(5):
        mask = fold_id == fold
        current = statistics(
            metric["pas"][selected, point],
            metric["pdp"][selected, point],
            metric["error"][selected, point],
            metric["energy"][selected, point],
            mask,
        )
        old = statistics(
            metric["pas"][0],
            metric["pdp"][0],
            metric["error"][0],
            metric["energy"][0],
            mask,
        )
        nested["fold_delta_vs_identity"].append(float(current["score"] - old["score"]))
    nested["positive_fold_count"] = int(
        np.sum(np.asarray(nested["fold_delta_vs_identity"]) > 0.0)
    )
    nested["decisions"] = decisions

    if phase_diagnostics is None:
        cached = np.load(METRIC_CACHE)
        phase_diagnostics = {
            key: cached[key]
            for key in ("correct", "rotated", "wrong_beta", "truth_path_energy")
        }
    active = arrays["active"]
    weight = phase_diagnostics["truth_path_energy"] * active
    phase = {
        name: float(
            np.sum(weight * phase_diagnostics[name])
            / max(float(np.sum(weight)), TINY)
        )
        for name in ("correct", "rotated", "wrong_beta")
    }
    return {
        "baseline": baseline,
        "top_candidates": rows[:20],
        "nested_selection": nested,
        "phase_sensitive_cosine_on_deployable_assignments": phase,
        "gate_active_path_fraction": [float(np.mean(mask)) for mask in masks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = build_phase_predictions(config, args.rebuild)
    evaluation = evaluate_channels(
        arrays,
        config,
        torch.device(args.device),
        args.batch_size,
        args.rebuild,
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "Fit-only dephased consistency can select predicted wall identities whose "
            "official-map path length improves V4 carrier phase without held path identity."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": (
            "fixed five complete spatial holes; predicted E7 support is assigned to walls "
            "without held support or held path identity"
        ),
        "config": asdict(config),
        "evaluation": evaluation,
        "outputs": {
            "phase_predictions": PATHS.relative(PHASE_CACHE),
            "metrics": PATHS.relative(METRIC_CACHE),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
