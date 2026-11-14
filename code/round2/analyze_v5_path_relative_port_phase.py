#!/usr/bin/env python3
"""Predict path-relative polarization/UE phase while retaining V4 carrier phase."""

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
from official_metrics import point_metric_statistics
from project_paths import PATHS
from run_path_component_transport import (
    MH,
    MP,
    MV,
    N_UE,
    PORTS,
    S_SC,
    ComponentCache,
    Config as TransportConfig,
    _component_atom,
    match_components,
)
from run_rbf_spectral_field import statistics


VERSION = "Round2-E28-path-relative-port-phase"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_path_relative_port_phase"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_path_relative_port_phase"
FINGERPRINT_CACHE = ARTIFACT_DIR / "oof_projective_port_fingerprints.npz"
METRIC_CACHE = ARTIFACT_DIR / "oof_path_relative_port_phase_metrics.npz"
REPORT = ANALYSIS_DIR / "path_relative_port_phase.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
TRAIN_COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
OOF_PATH_PARAMETERS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "oof_path_parameters_k16_os2_ref2_n40_dir3600_r1.npz"
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
    paths: int = 16
    donors: int = 40
    decomposition_ridge: float = 0.003
    seed: int = 20_260_805


def load_components() -> ComponentCache:
    cached = np.load(TRAIN_COMPONENTS)
    return ComponentCache(
        coordinates=cached["coordinates"].astype(np.float32),
        coefficients=cached["coefficients"].astype(np.complex64),
        residual_fraction=cached["residual_fraction"].astype(np.float32),
        point_power=cached["point_power"].astype(np.float64),
    )


def unit_vector(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-20)


def align_to_reference(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cross = np.sum(np.conj(value) * reference, axis=-1, keepdims=True)
    phase = cross / np.maximum(np.abs(cross), 1e-20)
    return value * phase


def projective_fingerprint(
    coefficients: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    normalized = unit_vector(coefficients)
    aligned = align_to_reference(normalized, reference[None])
    vector = np.sum(weights[:, None] * aligned, axis=0)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-20:
        return unit_vector(reference[None])[0].astype(np.complex64)
    return (vector / norm).astype(np.complex64)


def predict_query_fingerprints(
    query_position: np.ndarray,
    donor_ids: np.ndarray,
    positions: np.ndarray,
    cache: ComponentCache,
    transport: TransportConfig,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    anchor_id = int(donor_ids[0])
    anchor_coordinates = cache.coordinates[anchor_id, : config.paths]
    anchor_coefficients = cache.coefficients[anchor_id, : config.paths]
    observations: list[list[tuple[int, int, float]]] = [
        [(anchor_id, path, 1.0)] for path in range(config.paths)
    ]
    for donor_id in donor_ids[1:]:
        donor_id = int(donor_id)
        matched, _, coherence = match_components(
            anchor_coordinates,
            anchor_coefficients,
            cache.coordinates[donor_id],
            cache.coefficients[donor_id],
            transport,
        )
        for path, component in enumerate(matched):
            if component >= 0:
                observations[path].append(
                    (donor_id, int(component), float(coherence[path]))
                )

    projective = np.zeros((config.paths, PORTS), dtype=np.complex64)
    anchor = unit_vector(anchor_coefficients).astype(np.complex64)
    observation_count = np.zeros(config.paths, dtype=np.float64)
    agreement = np.zeros(config.paths, dtype=np.float64)
    donor_scale = max(
        float(
            np.median(
                np.linalg.norm(
                    positions[donor_ids, :2] - query_position[None, :2], axis=1
                )
            )
        ),
        1.0,
    )
    anchor_unit = unit_vector(anchor_coefficients)
    for path, rows in enumerate(observations):
        ids = np.asarray([row[0] for row in rows], dtype=np.int64)
        components = np.asarray([row[1] for row in rows], dtype=np.int64)
        coherence = np.asarray([row[2] for row in rows], dtype=np.float64)
        distance = np.linalg.norm(
            positions[ids, :2] - query_position[None, :2], axis=1
        )
        weight = np.exp(-0.5 * (distance / max(1.5 * donor_scale, 1.0)) ** 2)
        weight *= np.maximum(coherence, 0.10)
        weight /= max(float(np.sum(weight)), TINY)
        values = cache.coefficients[ids, components]
        projective[path] = projective_fingerprint(
            values, weight, anchor_unit[path]
        )
        aligned = align_to_reference(unit_vector(values), anchor_unit[path][None])
        agreement[path] = float(np.linalg.norm(np.sum(weight[:, None] * aligned, axis=0)))
        observation_count[path] = len(rows)
    energy = np.sum(np.abs(anchor_coefficients) ** 2, axis=1, dtype=np.float64)
    energy /= max(float(np.sum(energy)), TINY)
    diagnostics = {
        "mean_observations": float(np.sum(energy * observation_count)),
        "mean_projective_agreement": float(np.sum(energy * agreement)),
        "well_supported_energy": float(np.sum(energy[observation_count >= 6])),
    }
    return projective, anchor, diagnostics


def build_fingerprints(
    config: Config, rebuild: bool
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if FINGERPRINT_CACHE.is_file() and not rebuild:
        cached = np.load(FINGERPRINT_CACHE)
        return (
            {
                "projective": cached["projective"],
                "anchor": cached["anchor"],
                "e7_transport": cached["e7_transport"],
                "coordinates": cached["coordinates"],
            },
            json.loads(str(cached["diagnostics_json"].item())),
        )
    setup, positions, _, site, _ = load_positions()
    del setup
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    folds = np.load(FOLD_CACHE)
    path = np.load(OOF_PATH_PARAMETERS)
    coordinates = path["coordinates"].astype(np.float32)
    e7_transport = unit_vector(path["coefficients"].astype(np.complex64)).astype(
        np.complex64
    )
    cache = load_components()
    transport_payload = json.loads(str(path["config_json"].item()))
    transport = TransportConfig(**transport_payload)
    projective = np.zeros((len(query_ids), config.paths, PORTS), dtype=np.complex64)
    anchor = np.zeros_like(projective)
    rows: list[dict[str, float]] = []
    valid = cache.point_power > TINY
    for held_fold in np.unique(fold_id):
        output_rows = np.flatnonzero(fold_id == held_fold)
        fit = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
        for site_value in np.unique(site[query_ids[output_rows]]):
            selected = output_rows[site[query_ids[output_rows]] == site_value]
            bank = fit[(site[fit] == site_value) & valid[fit]]
            _, local = cKDTree(positions[bank, :2]).query(
                positions[query_ids[selected], :2], k=config.donors, workers=-1
            )
            donor_table = bank[local]
            for local_row, output_row in enumerate(selected):
                predicted, nearest, diagnostic = predict_query_fingerprints(
                    positions[query_ids[output_row]],
                    donor_table[local_row],
                    positions,
                    cache,
                    transport,
                    config,
                )
                projective[output_row] = predicted
                anchor[output_row] = nearest
                rows.append(diagnostic)
        print(f"projective fingerprints fold={int(held_fold)}", flush=True)
    diagnostics = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "p10": float(np.quantile([row[key] for row in rows], 0.10)),
            "p50": float(np.quantile([row[key] for row in rows], 0.50)),
            "p90": float(np.quantile([row[key] for row in rows], 0.90)),
        }
        for key in rows[0]
    }
    np.savez_compressed(
        FINGERPRINT_CACHE,
        projective=projective,
        anchor=anchor,
        e7_transport=e7_transport,
        coordinates=coordinates,
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
    )
    return {
        "projective": projective,
        "anchor": anchor,
        "e7_transport": e7_transport,
        "coordinates": coordinates,
    }, diagnostics


def channel_to_ports(channel: torch.Tensor) -> torch.Tensor:
    return (
        channel.reshape(-1, MP, MH, MV, N_UE, S_SC)
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(-1, PORTS, MH, MV, S_SC)
    )


def ports_to_channel(channel: torch.Tensor) -> torch.Tensor:
    return (
        channel.reshape(-1, MP, N_UE, MH, MV, S_SC)
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(-1, MP * MH * MV, N_UE, S_SC)
    )


def fixed_coordinate_decomposition(
    channel: torch.Tensor,
    coordinates: np.ndarray,
    ridge: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinate = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
    atoms = _component_atom(coordinate, torch.complex64, device)
    ports = channel_to_ports(channel)
    gram = torch.einsum("bkhvs,blhvs->bkl", torch.conj(atoms), atoms)
    identity = torch.eye(
        gram.shape[-1], dtype=gram.dtype, device=device
    )[None]
    right = torch.einsum("bkhvs,bchvs->bkc", torch.conj(atoms), ports)
    coefficient = torch.linalg.solve(gram + ridge * identity, right)
    reconstruction = torch.einsum("bkc,bkhvs->bchvs", coefficient, atoms)
    residual = ports - reconstruction
    return coefficient, residual, atoms


def relative_phase_coefficients(
    base: torch.Tensor,
    source: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    source = source / source.abs().square().sum(dim=2, keepdim=True).sqrt().clamp_min(1e-20)
    cross = torch.sum(torch.conj(source) * base, dim=2, keepdim=True)
    aligned = source * cross / cross.abs().clamp_min(1e-20)
    base_unit = base / base.abs().clamp_min(1e-20)
    source_unit = aligned / aligned.abs().clamp_min(1e-20)
    direction = (1.0 - alpha) * base_unit + alpha * source_unit
    direction = direction / direction.abs().clamp_min(1e-20)
    output = base.abs() * direction
    return torch.where(base.abs() > 1e-12, output, base)


def preserve_power(channel: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    current = channel.abs().square().sum(dim=(1, 2, 3))
    target = reference.abs().square().sum(dim=(1, 2, 3))
    scale = torch.sqrt(target / current.clamp_min(TINY))
    return channel * scale[:, None, None, None]


def metric_names() -> list[str]:
    names = ["identity"]
    sources = (
        "projective",
        "anchor",
        "e7_transport",
        "projective_port_shuffle_control",
        "projective_path_shuffle_control",
        "projective_conjugate_control",
        "oracle",
    )
    for source in sources:
        for alpha in (0.25, 0.50, 1.00):
            names.append(f"{source}_a{alpha:.2f}")
    return names


@torch.inference_mode()
def evaluate(
    fingerprints: dict[str, np.ndarray],
    config: Config,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> dict[str, object]:
    if METRIC_CACHE.is_file() and not rebuild:
        cached = np.load(METRIC_CACHE)
        arrays = {key: cached[key] for key in ("pas", "pdp", "error", "energy")}
        return summarize_metrics(
            cached["variant_names"].tolist(),
            arrays,
            cached["fold_id"].astype(np.int8),
            cached["site"].astype(np.int8),
            cached["fingerprint_coherence"],
            cached["reconstruction_fraction"],
        )
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids]
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    reference_all = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    names = metric_names()
    arrays = {
        key: np.zeros((len(names), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    fingerprint_coherence = np.zeros(
        (3, len(query_ids)), dtype=np.float32
    )
    reconstruction_fraction = np.zeros((2, len(query_ids)), dtype=np.float32)
    permutation = torch.as_tensor([3, 6, 1, 4, 7, 0, 5, 2], device=device)

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
        coordinates = fingerprints["coordinates"][start:end]
        base_coefficient, residual, atoms = fixed_coordinate_decomposition(
            reference, coordinates, config.decomposition_ridge, device
        )
        truth_coefficient, truth_residual, _ = fixed_coordinate_decomposition(
            truth, coordinates, config.decomposition_ridge, device
        )
        reference_ports = channel_to_ports(reference)
        truth_ports = channel_to_ports(truth)
        reconstruction_fraction[0, start:end] = (
            1.0
            - residual.abs().square().sum(dim=(1, 2, 3, 4))
            / reference_ports.abs().square().sum(dim=(1, 2, 3, 4)).clamp_min(TINY)
        ).cpu().numpy()
        reconstruction_fraction[1, start:end] = (
            1.0
            - truth_residual.abs().square().sum(dim=(1, 2, 3, 4))
            / truth_ports.abs().square().sum(dim=(1, 2, 3, 4)).clamp_min(TINY)
        ).cpu().numpy()

        source = {
            key: torch.as_tensor(fingerprints[key][start:end], device=device)
            for key in ("projective", "anchor", "e7_transport")
        }
        source["projective_port_shuffle_control"] = source["projective"][:, :, permutation]
        source["projective_path_shuffle_control"] = torch.roll(
            source["projective"], shifts=1, dims=1
        )
        source["projective_conjugate_control"] = torch.conj(source["projective"])
        source["oracle"] = truth_coefficient

        truth_unit = truth_coefficient / truth_coefficient.abs().square().sum(
            dim=2, keepdim=True
        ).sqrt().clamp_min(1e-20)
        path_energy = base_coefficient.abs().square().sum(dim=2)
        path_weight = path_energy / path_energy.sum(dim=1, keepdim=True).clamp_min(TINY)
        for source_index, source_name in enumerate(
            ("projective", "anchor", "e7_transport")
        ):
            value = source[source_name]
            value = value / value.abs().square().sum(dim=2, keepdim=True).sqrt().clamp_min(1e-20)
            coherence = torch.abs(torch.sum(torch.conj(value) * truth_unit, dim=2))
            fingerprint_coherence[source_index, start:end] = torch.sum(
                path_weight * coherence, dim=1
            ).cpu().numpy()

        variants: dict[str, torch.Tensor] = {"identity": reference}
        for source_name, source_value in source.items():
            for alpha in (0.25, 0.50, 1.00):
                coefficient = relative_phase_coefficients(
                    base_coefficient, source_value, alpha
                )
                ports = residual + torch.einsum(
                    "bkc,bkhvs->bchvs", coefficient, atoms
                )
                candidate = preserve_power(ports_to_channel(ports), reference)
                variants[f"{source_name}_a{alpha:.2f}"] = candidate
        truth_numpy = truth.cpu().numpy()
        for index, name in enumerate(names):
            metric = point_metric_statistics(variants[name].cpu().numpy(), truth_numpy)
            arrays["pas"][index, start:end] = metric.pas
            arrays["pdp"][index, start:end] = metric.pdp
            arrays["error"][index, start:end] = metric.squared_error
            arrays["energy"][index, start:end] = metric.truth_energy
        print(f"path-relative phase {end}/{len(query_ids)}", flush=True)

    np.savez_compressed(
        METRIC_CACHE,
        **arrays,
        variant_names=np.asarray(names),
        fold_id=fold_id,
        site=site,
        fingerprint_coherence=fingerprint_coherence,
        reconstruction_fraction=reconstruction_fraction,
    )
    return summarize_metrics(
        names,
        arrays,
        fold_id,
        site,
        fingerprint_coherence,
        reconstruction_fraction,
    )


def summarize_metrics(
    names: list[str],
    arrays: dict[str, np.ndarray],
    fold_id: np.ndarray,
    site: np.ndarray,
    fingerprint_coherence: np.ndarray,
    reconstruction_fraction: np.ndarray,
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
    baseline = variants["identity"]
    for value in variants.values():
        value["delta_vs_v4"] = float(value["all"]["score"] - baseline["all"]["score"])
        value["fold_delta_vs_v4"] = [
            float(current["score"] - old["score"])
            for current, old in zip(value["folds"], baseline["folds"])
        ]
        value["positive_fold_count"] = int(
            sum(delta > 0.0 for delta in value["fold_delta_vs_v4"])
        )

    deployable = [
        index
        for index, name in enumerate(names)
        if name == "identity"
        or (
            "control" not in name
            and "oracle" not in name
            and "conjugate" not in name
        )
    ]
    selected = np.zeros(len(fold_id), dtype=np.int64)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            fit = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            scores = []
            for index in deployable:
                scores.append(
                    statistics(
                        arrays["pas"][index],
                        arrays["pdp"][index],
                        arrays["error"][index],
                        arrays["energy"][index],
                        fit,
                    )["score"]
                )
            chosen = deployable[int(np.argmax(scores))]
            selected[held] = chosen
            decisions.append(
                {
                    "held_fold": int(held_fold),
                    "site": int(site_value),
                    "candidate": names[chosen],
                    "training_score": float(np.max(scores)),
                    "held_count": int(np.sum(held)),
                }
            )
    point = np.arange(len(fold_id))
    nested = statistics(
        arrays["pas"][selected, point],
        arrays["pdp"][selected, point],
        arrays["error"][selected, point],
        arrays["energy"][selected, point],
    )
    nested_folds = [
        {
            "fold": int(fold),
            **statistics(
                arrays["pas"][selected, point],
                arrays["pdp"][selected, point],
                arrays["error"][selected, point],
                arrays["energy"][selected, point],
                fold_id == fold,
            ),
        }
        for fold in np.unique(fold_id)
    ]
    return {
        "variants": variants,
        "ranking_high_score_first": sorted(
            names,
            key=lambda name: float(variants[name]["all"]["score"]),
            reverse=True,
        ),
        "nested_selection": {
            "overall": nested,
            "folds": nested_folds,
            "decisions": decisions,
            "delta_vs_v4": float(nested["score"] - baseline["all"]["score"]),
            "fold_delta_vs_v4": [
                float(new["score"] - old["score"])
                for new, old in zip(nested_folds, baseline["folds"])
            ],
        },
        "fingerprint_coherence": {
            name: {
                "mean": float(np.mean(fingerprint_coherence[index])),
                "p10": float(np.quantile(fingerprint_coherence[index], 0.10)),
                "p50": float(np.quantile(fingerprint_coherence[index], 0.50)),
                "p90": float(np.quantile(fingerprint_coherence[index], 0.90)),
            }
            for index, name in enumerate(("projective", "anchor", "e7_transport"))
        },
        "reconstruction_fraction": {
            "v4_mean": float(np.mean(reconstruction_fraction[0])),
            "truth_mean": float(np.mean(reconstruction_fraction[1])),
            "v4_p50": float(np.median(reconstruction_fraction[0])),
            "truth_p50": float(np.median(reconstruction_fraction[1])),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rebuild-fingerprints", action="store_true")
    parser.add_argument("--rebuild-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    fingerprints, diagnostics = build_fingerprints(
        config, args.rebuild_fingerprints
    )
    evaluation = evaluate(
        fingerprints,
        config,
        torch.device(args.device),
        args.batch_size,
        args.rebuild_metrics,
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "A path's projective polarization-by-UE fingerprint varies more slowly than "
            "its unobservable common carrier phase; transporting only that fingerprint "
            "can improve V4 relative link phase without resolving integer cycles."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": (
            "fixed five complete spatial holes; every donor fingerprint and path match "
            "comes from the corresponding outer-fit set"
        ),
        "config": asdict(config),
        "fingerprint_diagnostics": diagnostics,
        "evaluation": evaluation,
        "outputs": {
            "fingerprints": PATHS.relative(FINGERPRINT_CACHE),
            "metrics": PATHS.relative(METRIC_CACHE),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fingerprint_coherence": evaluation["fingerprint_coherence"],
                "reconstruction_fraction": evaluation["reconstruction_fraction"],
                "nested_selection": evaluation["nested_selection"],
                "top": [
                    {
                        "name": name,
                        **evaluation["variants"][name],
                    }
                    for name in evaluation["ranking_high_score_first"][:5]
                ],
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
