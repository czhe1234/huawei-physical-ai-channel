#!/usr/bin/env python3
"""Apply wall-identity phase transport on components extracted from V4 itself."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
)
from analyze_v4_map_virtual_source_phase import (
    COMPONENT_CACHE,
    Config as MapConfig,
    build_wall_bank,
    image_sources,
)
from analyze_v5_deployable_wall_phase_transport import gate_masks, mix_full_phase
from analyze_v5_explicit_virtual_path_identity import (
    ASSIGNMENT_CACHE,
    FOLDS,
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
from run_path_component_transport import (
    MH,
    MP,
    MV,
    N_UE,
    PORTS,
    S_SC,
    _component_atom,
    _refine_one_atom,
)
from run_rbf_spectral_field import OOF_META, statistics


VERSION = "Round2-E33-v4-native-wall-phase"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_v4_native_wall_phase"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_v4_native_wall_phase"
COMPONENT_OUTPUT = ARTIFACT_DIR / "v4_native_components_k24_os2_ref2.npz"
PHASE_CACHE = ARTIFACT_DIR / "oof_v4_native_wall_phase_predictions.npz"
METRIC_CACHE = ARTIFACT_DIR / "oof_v4_native_wall_phase_metrics.npz"
REPORT = ANALYSIS_DIR / "v4_native_wall_phase.json"
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    components: int = 24
    paths: int = 16
    oversample: int = 2
    refinement_sweeps: int = 2
    donor_count: int = 24
    minimum_observations: int = 4
    maximum_donor_distance_m: float = 55.0
    distance_scale_floor_m: float = 5.0
    wrong_beta_offset: float = 0.20
    phase_alpha: float = 0.10
    decomposition_ridge: float = 0.003
    projection_cycles: int = 4


def channel_to_ports(channel: torch.Tensor) -> torch.Tensor:
    return (
        channel.reshape(-1, MP, MH, MV, N_UE, S_SC)
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(-1, PORTS, MH, MV, S_SC)
    )


@torch.inference_mode()
def extract_v4_components(
    config: Config,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if COMPONENT_OUTPUT.is_file() and not rebuild:
        return {key: value for key, value in np.load(COMPONENT_OUTPUT).items()}
    source_all = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    count = len(source_all)
    coordinates = np.zeros((count, config.components, 3), dtype=np.float32)
    coefficients = np.zeros(
        (count, config.components, PORTS), dtype=np.complex64
    )
    explained = np.zeros(count, dtype=np.float32)
    os = config.oversample

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        source = torch.as_tensor(
            np.array(source_all[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        source = channel_to_ports(source)
        residual = source.clone()
        source_energy = source.abs().square().sum(dim=(1, 2, 3, 4))
        coordinate = torch.zeros(
            (len(source), config.components, 3), dtype=torch.float32, device=device
        )
        coefficient = torch.zeros(
            (len(source), config.components, PORTS),
            dtype=torch.complex64,
            device=device,
        )
        for component in range(config.components):
            transformed = torch.fft.ifft(
                torch.fft.fft2(
                    residual, s=(MH * os, MV * os), dim=(-3, -2)
                ),
                n=S_SC * os,
                dim=-1,
            )
            power = transformed.abs().square().sum(dim=1).reshape(len(source), -1)
            flat = torch.argmax(power, dim=1)
            delay = flat % (S_SC * os)
            partial = flat // (S_SC * os)
            vertical = partial % (MV * os)
            horizontal = partial // (MV * os)
            selected = torch.stack([horizontal, vertical, delay], dim=1).float() / os
            atom = _component_atom(selected[:, None], residual.dtype, device)[:, 0]
            value = torch.sum(
                torch.conj(atom[:, None]) * residual, dim=(-3, -2, -1)
            )
            residual -= value[:, :, None, None, None] * atom[:, None]
            coordinate[:, component] = selected
            coefficient[:, component] = value

        initial = coordinate.clone()
        atoms = _component_atom(coordinate, torch.complex64, device)
        model = torch.einsum("bkc,bkhvs->bchvs", coefficient, atoms)
        for _ in range(config.refinement_sweeps):
            for component in range(config.components):
                residual_plus = (
                    source
                    - model
                    + coefficient[:, component, :, None, None, None]
                    * atoms[:, component, None]
                )
                estimate, updated, updated_atom = _refine_one_atom(
                    residual_plus, initial[:, component]
                )
                model = (
                    model
                    - coefficient[:, component, :, None, None, None]
                    * atoms[:, component, None]
                    + updated[:, :, None, None, None] * updated_atom[:, None]
                )
                coordinate[:, component] = estimate
                coefficient[:, component] = updated
                atoms[:, component] = updated_atom
        energy = coefficient.abs().square().sum(dim=2)
        order = torch.argsort(energy, dim=1, descending=True)
        coordinate = torch.gather(
            coordinate, 1, order[:, :, None].expand(-1, -1, 3)
        )
        coefficient = torch.gather(
            coefficient, 1, order[:, :, None].expand(-1, -1, PORTS)
        )
        residual = source - model
        residual_energy = residual.abs().square().sum(dim=(1, 2, 3, 4))
        coordinates[start:end] = coordinate.cpu().numpy()
        coefficients[start:end] = coefficient.cpu().numpy()
        explained[start:end] = (
            1.0 - residual_energy / source_energy.clamp_min(TINY)
        ).cpu().numpy()
        print(
            f"V4 native component extraction {end}/{count} "
            f"median_explained={np.median(explained[start:end]):.6f}",
            flush=True,
        )
    arrays = {
        "coordinates": coordinates,
        "coefficients": coefficients,
        "explained_fraction": explained,
    }
    np.savez_compressed(COMPONENT_OUTPUT, **arrays)
    return arrays


def weighted_unit_average(value: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float]:
    raw = np.sum(weight[:, None] * value, axis=0)
    return unit_vector(raw), float(np.linalg.norm(raw))


def build_phase_predictions(
    components: dict[str, np.ndarray],
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
    train_coefficients = np.load(COMPONENT_CACHE)["coefficients"].astype(np.complex64)
    coordinates = components["coordinates"][:, : config.paths]
    fallback = components["coefficients"][:, : config.paths]
    identity_config = IdentityConfig()
    map_config = MapConfig()
    wall_bank = build_wall_bank(map_config)
    fold_parameters = load_fold_parameters()

    source_output = np.array(fallback, copy=True)
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
                    observed = np.array(train_coefficients[ids, parts], copy=True)
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
                    source_output[output_row, path] = prediction
                    rotated_output[output_row, path] = rotated_prediction
                    wrong_beta_output[output_row, path] = wrong_prediction
                    active[output_row, path] = True
                    fit_coherence[output_row, path] = coherence
                    rotated_coherence[output_row, path] = rotated_fit
                    observation_count[output_row, path] = len(ids)
                    nearest_distance[output_row, path] = distance[0]
        print(f"V4-native wall phase fold={fold} rows={len(rows)}", flush=True)
    arrays = {
        "coordinates": coordinates,
        "source": source_output,
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


def candidate_names(gates: list[str]) -> list[str]:
    return ["identity"] + [f"{gate}_a0.10_pas_pdp_c4" for gate in gates]


@torch.inference_mode()
def evaluate(
    arrays: dict[str, np.ndarray],
    config: Config,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> dict[str, object]:
    gates, masks = gate_masks(arrays)
    names = candidate_names(gates)
    if METRIC_CACHE.is_file() and not rebuild:
        cached = np.load(METRIC_CACHE)
        metric = {key: cached[key] for key in ("pas", "pdp", "error", "energy")}
        controls = {key: cached[key] for key in ("correct", "rotated", "wrong_beta", "path_energy")}
        return summarize(names, metric, arrays, controls)

    query_ids = arrays["query_ids"].astype(np.int64)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    reference_all = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    metric = {
        key: np.zeros((len(names), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    controls = {
        key: np.zeros(arrays["active"].shape, dtype=np.float32)
        for key in ("correct", "rotated", "wrong_beta", "path_energy")
    }
    index_by_name = {name: index for index, name in enumerate(names)}

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        reference = torch.as_tensor(
            np.array(reference_all[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True), device=device
        )
        target_pas = pas_distribution(reference)
        target_pdp = pdp_distribution(reference)
        base, residual, atoms = fixed_coordinate_decomposition(
            reference,
            arrays["coordinates"][start:end],
            config.decomposition_ridge,
            device,
        )
        truth_coefficient, _, _ = fixed_coordinate_decomposition(
            truth,
            arrays["coordinates"][start:end],
            config.decomposition_ridge,
            device,
        )
        truth_norm = truth_coefficient.abs().square().sum(dim=2).sqrt().clamp_min(1e-20)
        truth_unit = truth_coefficient / truth_norm[:, :, None]
        for control, source_name in (
            ("correct", "source"),
            ("rotated", "rotated_source"),
            ("wrong_beta", "wrong_beta_source"),
        ):
            source = torch.as_tensor(arrays[source_name][start:end], device=device)
            source /= source.abs().square().sum(dim=2, keepdim=True).sqrt().clamp_min(1e-20)
            controls[control][start:end] = torch.real(
                torch.sum(torch.conj(source) * truth_unit, dim=2)
            ).cpu().numpy()
        controls["path_energy"][start:end] = truth_norm.square().cpu().numpy()

        variants: dict[str, torch.Tensor] = {"identity": reference}
        source = torch.as_tensor(arrays["source"][start:end], device=device)
        for gate_index, gate in enumerate(gates):
            mask = torch.as_tensor(masks[gate_index, start:end], device=device)
            coefficient = mix_full_phase(base, source, mask, config.phase_alpha)
            ports = residual + torch.einsum("bkc,bkhvs->bchvs", coefficient, atoms)
            candidate = preserve_power(ports_to_channel(ports), reference)
            for _ in range(config.projection_cycles):
                candidate = exact_pas_projection(candidate, target_pas)
                candidate = exact_pdp_projection(candidate, target_pdp)
                candidate = preserve_power(candidate, reference)
            variants[f"{gate}_a0.10_pas_pdp_c4"] = candidate
        truth_numpy = truth.cpu().numpy()
        for name, candidate in variants.items():
            point = point_metric_statistics(candidate.cpu().numpy(), truth_numpy)
            index = index_by_name[name]
            metric["pas"][index, start:end] = point.pas
            metric["pdp"][index, start:end] = point.pdp
            metric["error"][index, start:end] = point.squared_error
            metric["energy"][index, start:end] = point.truth_energy
        if end % 20 == 0 or end == len(query_ids):
            print(f"V4-native wall phase metric {end}/{len(query_ids)}", flush=True)
    np.savez_compressed(METRIC_CACHE, **metric, **controls, variant_names=np.asarray(names))
    return summarize(names, metric, arrays, controls)


def summarize(
    names: list[str],
    metric: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    controls: dict[str, np.ndarray],
) -> dict[str, object]:
    fold_id = arrays["fold_id"]
    site = arrays["site"]
    candidate_rows = []
    for index, name in enumerate(names):
        candidate_rows.append(
            {
                "name": name,
                **statistics(
                    metric["pas"][index], metric["pdp"][index],
                    metric["error"][index], metric["energy"][index]
                ),
                "fold_score": [
                    statistics(
                        metric["pas"][index], metric["pdp"][index],
                        metric["error"][index], metric["energy"][index], fold_id == fold
                    )["score"]
                    for fold in range(5)
                ],
            }
        )
    candidate_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = np.zeros(len(fold_id), dtype=np.int64)
    decisions = []
    for held_fold in range(5):
        for site_value in range(2):
            train = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            scores = np.asarray([
                statistics(
                    metric["pas"][index], metric["pdp"][index],
                    metric["error"][index], metric["energy"][index], train
                )["score"]
                for index in range(len(names))
            ])
            chosen = int(np.argmax(scores))
            selected[held] = chosen
            decisions.append({
                "held_fold": held_fold, "site": site_value,
                "candidate": names[chosen], "training_score": float(scores[chosen])
            })
    point = np.arange(len(fold_id))
    nested = statistics(
        metric["pas"][selected, point], metric["pdp"][selected, point],
        metric["error"][selected, point], metric["energy"][selected, point]
    )
    baseline = statistics(
        metric["pas"][0], metric["pdp"][0], metric["error"][0], metric["energy"][0]
    )
    nested["delta_vs_identity"] = float(nested["score"] - baseline["score"])
    fold_delta = []
    for fold in range(5):
        mask = fold_id == fold
        current = statistics(
            metric["pas"][selected, point], metric["pdp"][selected, point],
            metric["error"][selected, point], metric["energy"][selected, point], mask
        )
        old = statistics(
            metric["pas"][0], metric["pdp"][0], metric["error"][0], metric["energy"][0], mask
        )
        fold_delta.append(float(current["score"] - old["score"]))
    nested["fold_delta_vs_identity"] = fold_delta
    nested["positive_fold_count"] = int(np.sum(np.asarray(fold_delta) > 0.0))
    nested["decisions"] = decisions
    active = arrays["active"]
    weight = controls["path_energy"] * active
    phase_controls = {
        name: float(np.sum(weight * controls[name]) / max(float(np.sum(weight)), TINY))
        for name in ("correct", "rotated", "wrong_beta")
    }
    return {
        "baseline": baseline,
        "top_candidates": candidate_rows,
        "nested_selection": nested,
        "phase_sensitive_cosine_on_deployable_assignments": phase_controls,
        "active_path_fraction": float(np.mean(active)),
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
    device = torch.device(args.device)
    components = extract_v4_components(config, device, args.batch_size, args.rebuild)
    phase = build_phase_predictions(components, config, args.rebuild)
    evaluation = evaluate(phase, config, device, args.batch_size, args.rebuild)
    report = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "Extracting phase-action coordinates from V4 itself removes the 65.8% carrier "
            "alignment ceiling of E31/E32 and makes wall-phase gains stable."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "component_explained_fraction": {
            "mean": float(np.mean(components["explained_fraction"])),
            "p50": float(np.median(components["explained_fraction"])),
            "p10": float(np.quantile(components["explained_fraction"], 0.10)),
        },
        "evaluation": evaluation,
        "outputs": {
            "components": PATHS.relative(COMPONENT_OUTPUT),
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
