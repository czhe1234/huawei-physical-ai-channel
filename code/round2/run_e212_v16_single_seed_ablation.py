#!/usr/bin/env python3
"""Run the V16 seed=42-only diagnostic in an isolated artifact tree.

The historical V16 code uses module-level paths.  This driver redirects only
seed-dependent inputs and outputs while keeping the original model code,
hyperparameters, folds, renderer, and official scorer unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch

from project_paths import PATHS


VERSION = "Round2-E212-V16-single-seed-ablation"
SEED = 42
BASE_NAME = "primary_d16_base_v4lambda_1m"
SCATTER_NAME = "primary_d10_scatter015_v4lambda_1m"
BASELINE_SCORE = 0.686375004
BASELINE_ZIP = PATHS.root / "final_code" / "V16_source_online_0.686754.zip"
BASELINE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_official_sionna_path_calibrator"
    / "fixed500_report.json"
)


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    reports: Path
    logs: Path
    spectra: Path
    tokens: Path
    token_cache: Path
    features: Path
    e40: Path
    v9: Path
    link: Path
    canonical: Path
    e85: Path
    e99: Path
    e190: Path
    final_report: Path


def experiment_paths() -> ExperimentPaths:
    root = PATHS.artifacts / "round2" / "e212_v16_single_seed_ablation"
    reports = (
        PATHS.analysis / "results" / "round2_e212_v16_single_seed_ablation"
    )
    return ExperimentPaths(
        root=root,
        reports=reports,
        logs=root / "logs",
        spectra=root / "sionna_multiseed",
        tokens=root / "joint_path_token_oracle",
        token_cache=(
            root
            / "joint_path_token_oracle"
            / "n4000_s01_k128_b096_d064_scatter1_sel106"
        ),
        features=root / "explicit_path_pdp_predictor",
        e40=root / "sionna_official_calibration",
        v9=root / "v9_joint_sionna_endpoints",
        link=root / "full_path_pdp_link_gate",
        canonical=root / "canonical_500_version_replay",
        e85=root / "surface_interaction_energy_model",
        e99=root / "corrected_array_surface_model",
        e190=root / "official_sionna_path_calibrator",
        final_report=reports / "fixed500_report.json",
    )


def ensure_directories(paths: ExperimentPaths) -> None:
    for path in (paths.root, paths.reports, paths.logs):
        path.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def invoke_main(module: Any, arguments: list[str]) -> None:
    previous = sys.argv
    sys.argv = [str(getattr(module, "__file__", module.__name__)), *arguments]
    try:
        module.main()
    finally:
        sys.argv = previous


def historical_path_root(name: str) -> Path:
    return (
        PATHS.artifacts / "round2" / "sionna_multiseed" / name / "full" / "paths"
    )


def source_h5(name: str) -> Path:
    return historical_path_root(name) / f"train_seed{SEED}.h5"


def configure_spectra(paths: ExperimentPaths):
    import prepare_sionna_calibration_dataset as prepare
    import train_sionna_spectral_calibrator as e40
    import train_v9_joint_sionna_endpoints as v9

    prepare.SIONNA_ROOT = paths.spectra
    prepare.SEEDS = 1

    e40.ARTIFACT_DIR = paths.e40
    e40.ANALYSIS_DIR = paths.reports / "e40"
    e40.SCREEN_REPORT = e40.ANALYSIS_DIR / "calibrator_screen.json"
    e40.OOF_REPORT = e40.ANALYSIS_DIR / "calibrator_oof.json"
    e40.SEEDS = 1

    v9.ARTIFACT_DIR = paths.v9
    v9.ANALYSIS_DIR = paths.reports / "v9"
    v9.SCREEN_REPORT = v9.ANALYSIS_DIR / "joint_endpoint_screen.json"
    v9.OOF_REPORT = v9.ANALYSIS_DIR / "joint_endpoint_oof.json"
    v9.TEST_REPORT = v9.ANALYSIS_DIR / "joint_endpoint_test.json"
    v9.SEEDS = 1
    return prepare, e40, v9


def configure_paths(paths: ExperimentPaths):
    import analyze_joint_path_token_oracle as joint

    _, e40, v9 = configure_spectra(paths)
    joint.ARTIFACT_DIR = paths.tokens
    joint.ANALYSIS_DIR = paths.reports / "tokens"
    joint.REPORT = joint.ANALYSIS_DIR / "joint_path_token_oracle.json"
    joint.BASE_PATH_ROOT = historical_path_root(BASE_NAME)
    joint.SCATTER_PATH_ROOT = historical_path_root(SCATTER_NAME)
    joint.OLD_PAS = paths.e40 / "oof_pas_mean_f16.npy"
    joint.OLD_PDP = paths.e40 / "oof_pdp_mean_f16.npy"
    joint.JOINT_PAS = v9.output_path("pas", "oof")
    joint.JOINT_PDP = v9.output_path("pdp", "oof")
    return joint


def configure_features(paths: ExperimentPaths):
    import build_explicit_path_model_features as builder

    configure_paths(paths)
    builder.TRAIN_CACHE = paths.token_cache
    builder.ARTIFACT_DIR = paths.features
    return builder


def configure_link(paths: ExperimentPaths):
    import train_full_path_pdp_link_gate as link

    configure_features(paths)
    _, _, v9 = configure_spectra(paths)
    historical = PATHS.artifacts / "round2" / "full_path_pdp_link_gate"
    link.PATH_CACHE = paths.token_cache
    link.TOKEN_FEATURES = paths.features / "path_features_train_n4000.npy"
    link.TRAIN_REGIME = paths.features / "path_regime_train_n4000.npy"
    link.RELIABILITY = v9.reliability_path("pdp", "train")
    link.ARTIFACT_DIR = paths.link
    link.ANALYSIS_DIR = paths.reports / "link_gate"
    link.FOLD_IDS = historical / "train_fold_id.npy"
    link.BASE_PDP = historical / "train_rbf_base_pdp_f16.npy"
    link.RAW_PDP = paths.link / "train_raw_path_pdp_f16.npy"
    link.POINT_CONTEXT = paths.link / "train_point_context.npy"
    link.OOF_ALPHA = paths.link / "train_oof_alpha_f16.npy"
    link.TRAIN_REPORT = link.ANALYSIS_DIR / "full_path_pdp_link_gate_train.json"
    link.V9_REPORT = link.ANALYSIS_DIR / "full_path_pdp_link_gate_v9_oof.json"
    return link


def configure_canonical(paths: ExperimentPaths):
    import analyze_round2_canonical_500_versions as canonical

    prepare, _, v9 = configure_spectra(paths)
    joint = configure_paths(paths)
    link = configure_link(paths)
    canonical.ARTIFACT_DIR = paths.canonical
    canonical.ANALYSIS_DIR = paths.reports / "canonical"
    canonical.JSON_REPORT = canonical.ANALYSIS_DIR / "canonical_500_version_scores.json"
    canonical.CSV_REPORT = canonical.ANALYSIS_DIR / "canonical_500_version_scores.csv"
    canonical.POINT_METRICS = paths.canonical / "canonical_v9_v11_point_metrics.npz"
    canonical.V9_CHANNEL = paths.canonical / "V9_Canonical_500_OOF_Channel.npy"
    canonical.V10_CHANNEL = paths.canonical / "V10_Canonical_500_OOF_Channel.npy"
    canonical.V11_CHANNEL = paths.canonical / "V11_Canonical_500_OOF_Channel.npy"
    canonical.V9_PDP = paths.canonical / "v9_base_pdp_f16.npy"
    canonical.V10_ALPHA = paths.canonical / "v10_cross_fitted_alpha_f16.npy"
    canonical.V11_GATE_FEATURES = paths.canonical / "v11_pair_rnorm_raw_features.npy"
    canonical.V11_GATE = paths.canonical / "v11_pair_rnorm_gate_f16.npy"
    canonical.FULL_PATH_DIR = paths.link
    canonical.RAW_PATH_PDP = link.RAW_PDP
    canonical.POINT_CONTEXT = link.POINT_CONTEXT
    canonical.FOLD_IDS = link.FOLD_IDS
    canonical.E40_PAS = paths.e40 / "oof_pas_mean_f16.npy"
    canonical.V9_ENDPOINT_PAS = v9.output_path("pas", "oof")
    canonical.E39_BASE_PAS = prepare.running_path(
        BASE_NAME, "train", "pas", "mean"
    )
    canonical.E39_SCATTER_PAS = prepare.running_path(
        SCATTER_NAME, "train", "pas", "mean"
    )
    joint.OLD_PAS = paths.e40 / "oof_pas_mean_f16.npy"
    joint.OLD_PDP = paths.e40 / "oof_pdp_mean_f16.npy"
    return canonical


def configure_surface(paths: ExperimentPaths):
    import analyze_path_candidate_information_audit as e84
    import train_surface_interaction_energy_model as e85
    import analyze_e86_surface_endpoint_v11_blend as e86
    import evaluate_e86_frozen_policy as frozen
    import optimize_e98_global_array_angle_mapping as e98
    import train_e99_corrected_array_surface_model as e99
    import analyze_e100_locked_energy_angle_correction as e100

    canonical = configure_canonical(paths)
    historical_link = PATHS.artifacts / "round2" / "full_path_pdp_link_gate"
    historical_state = (
        PATHS.artifacts
        / "round2"
        / "sionna_multiseed"
        / BASE_NAME
        / "full"
        / "state.json"
    )
    feature_path = paths.features / "path_features_train_n4000.npy"
    regime_path = paths.features / "path_regime_train_n4000.npy"

    e84.PATH_CACHE = paths.token_cache
    e84.FULL_FOLD = historical_link / "train_fold_id.npy"
    e84.FULL_REGIME = regime_path
    e84.SIONNA_STATE = historical_state

    e85.PATH_CACHE = paths.token_cache
    e85.PATH_FEATURES = feature_path
    e85.FULL_FOLD = historical_link / "train_fold_id.npy"
    e85.FULL_REGIME = regime_path
    e85.SIONNA_STATE = historical_state
    e85.ARTIFACT_DIR = paths.e85
    e85.ANALYSIS_DIR = paths.reports / "e85"
    e85.TARGET_PAS = paths.e85 / "official_pas_cosine_sufficient_target_f32.npy"
    e85.TARGET_PDP = paths.e85 / "official_pdp_cosine_sufficient_target_f32.npy"
    e85.POINT_POWER = paths.e85 / "official_point_power_f64.npy"
    e85.BASE_COORDINATE = paths.e85 / "sionna_aligned_coordinates_f32.npy"
    e85.PORT_PRIOR = paths.e85 / "sionna_path_port_prior_f16.npy"
    e85.SEQUENCE_GEOMETRY = paths.e85 / "sequence_geometry_f16.npy"
    e85.SEQUENCE_HASH = paths.e85 / "sequence_hash_buckets_i32.npy"
    e85.PREPARE_REPORT = e85.ANALYSIS_DIR / "surface_interaction_training_cache.json"
    e85.SCREEN_REPORT = e85.ANALYSIS_DIR / "surface_interaction_screen.json"
    e85.FULL_REPORT = e85.ANALYSIS_DIR / "surface_interaction_five_fold.json"
    e85.STRICT_REPORT = e85.ANALYSIS_DIR / "surface_interaction_strict_holes.json"
    e85.PRESENCE_SEED_COUNT = 1

    e86.ARTIFACT_DIR = paths.e85
    e86.ANALYSIS_DIR = paths.reports / "e86"
    e86.REPORT = e86.ANALYSIS_DIR / "e86_surface_endpoint_v11_blend.json"
    e86.METRICS_PATH = paths.e85 / "e86_candidate_metrics.npz"
    e86.NESTED_CHANNEL = paths.e85 / "e86_nested_oof_channel.npy"
    e86.RENDERED = {
        "geometry_sequence": (
            paths.e85 / "e86_geometry_sequence_rendered_pas_f16.npy",
            paths.e85 / "e86_geometry_sequence_rendered_pdp_f16.npy",
            paths.e85 / "strict_oof_geometry_sequence_path_endpoint.npz",
        ),
        "no_sequence": (
            paths.e85 / "e86_no_sequence_rendered_pas_f16.npy",
            paths.e85 / "e86_no_sequence_rendered_pdp_f16.npy",
            paths.e85 / "strict_oof_no_sequence_path_endpoint.npz",
        ),
    }
    e86.V11_CHANNEL = canonical.V11_CHANNEL

    frozen.ARTIFACT_DIR = paths.e85
    frozen.ANALYSIS_DIR = paths.reports / "e86"
    frozen.REPORT = frozen.ANALYSIS_DIR / "e86_frozen_policy.json"
    frozen.OUTPUT = paths.e85 / "e86_frozen_geometry_sequence_oof_channel.npy"
    frozen.V11_CHANNEL = canonical.V11_CHANNEL
    frozen.GEOMETRY_PAS = e86.RENDERED["geometry_sequence"][0]
    frozen.GEOMETRY_PDP = e86.RENDERED["geometry_sequence"][1]
    frozen.NO_SEQUENCE_PAS = e86.RENDERED["no_sequence"][0]
    frozen.NO_SEQUENCE_PDP = e86.RENDERED["no_sequence"][1]

    e99.ARTIFACT_DIR = paths.e99
    e99.ANALYSIS_DIR = paths.reports / "e99"
    e99.REPORT = e99.ANALYSIS_DIR / "corrected_array_surface_model.json"
    e99.SMOKE_REPORT = e99.ANALYSIS_DIR / "smoke.json"
    e99.ENDPOINT = paths.e99 / "strict_oof_corrected_geometry_sequence_endpoint.npz"
    e99.RENDERED_PAS = paths.e99 / "strict_oof_corrected_rendered_pas_f16.npy"
    e99.RENDERED_PDP = paths.e99 / "strict_oof_corrected_rendered_pdp_f16.npy"
    e99.OOF_CHANNEL = paths.e99 / "strict_oof_corrected_v11_blend_channel.npy"
    e99.MAPPING_PARAMETERS = paths.e99 / "strict_mapping_parameters.npz"
    e99.V12_CHANNEL = frozen.OUTPUT
    e99.OLD_ENDPOINT = e86.RENDERED["geometry_sequence"][2]

    e100.V12_CHANNEL = frozen.OUTPUT
    e100.OLD_ENDPOINT = e86.RENDERED["geometry_sequence"][2]
    e100.OLD_RENDERED_PDP = e86.RENDERED["geometry_sequence"][1]
    e100.MAPPING_PARAMETERS = e99.MAPPING_PARAMETERS
    return e84, e85, e86, frozen, e98, e99, e100


def configure_e190(paths: ExperimentPaths):
    import train_e190_official_sionna_path_calibrator as e190

    _, e85, e86, frozen, e98, e99, e100 = configure_surface(paths)
    e190.ARTIFACT_DIR = paths.e190
    e190.ANALYSIS_DIR = paths.reports / "e190"
    e190.FULL_REPORT = e190.ANALYSIS_DIR / "fixed500_report.json"
    e190.FULL_CHANNEL = paths.e190 / "fixed500_channel.npy"
    e190.MAPPING_PARAMETERS = e99.MAPPING_PARAMETERS
    e190.OLD_ENDPOINT = e85.ARTIFACT_DIR / "strict_oof_geometry_sequence_path_endpoint.npz"
    e190.OLD_RENDERED_PAS = e86.RENDERED["geometry_sequence"][0]
    e190.OLD_RENDERED_PDP = e86.RENDERED["geometry_sequence"][1]
    e190.PRESENCE_SEED_COUNT = 1
    e100.V12_CHANNEL = frozen.OUTPUT
    return e190


def spectra_output_root(paths: ExperimentPaths, name: str) -> Path:
    return paths.spectra / name / "full"


def open_array(path: Path, dtype: Any, shape: tuple[int, ...], reset: bool):
    if reset:
        path.unlink(missing_ok=True)
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"invalid cached array {path}: {value.shape}/{value.dtype}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value[...] = 0
    value.flush()
    return value


def reconstruct_single_seed_spectra(
    paths: ExperimentPaths, name: str, device: torch.device, force: bool
) -> dict[str, Any]:
    import generate_multiseed_sionna_dataset as generator
    import analyze_joint_path_token_oracle as joint

    started = time.time()
    source = source_h5(name)
    output_root = spectra_output_root(paths, name)
    output_root.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as handle:
        if int(handle.attrs["seed"]) != SEED or str(handle.attrs["split"]) != "train":
            raise ValueError(f"unexpected source identity: {source}")
        if not bool(handle.attrs["complete"]):
            raise ValueError(f"incomplete source HDF5: {source}")
        configuration = json.loads(str(handle.attrs["configuration"]))
        coefficient_mode = str(handle.attrs["coefficient_mode"])
        tasks = [
            (site_name, batch_name)
            for site_name in sorted(handle.keys())
            for batch_name in sorted(handle[site_name].keys())
        ]
        row_count = sum(len(handle[site][batch]["row_indices"]) for site, batch in tasks)
    if row_count != 4000:
        raise ValueError(f"expected 4000 rows in {source}, found {row_count}")

    arrays: dict[str, dict[str, np.memmap]] = {}
    for domain in generator.DOMAINS:
        def statistic_path(kind: str) -> Path:
            return output_root / "running" / f"train_{domain}_{kind}_n1.npy"

        arrays[domain] = {
            "mean": open_array(
                statistic_path("mean"),
                np.float32,
                generator.domain_shape(domain, row_count),
                force,
            ),
            "m2": open_array(
                statistic_path("m2"),
                np.float32,
                generator.domain_shape(domain, row_count),
                force,
            ),
            "count": open_array(
                statistic_path("count"),
                np.uint16,
                generator.count_shape(domain, row_count),
                force,
            ),
        }
    done_path = output_root / "running" / "train_seed42_reconstruction_done.npy"
    done = open_array(done_path, np.bool_, (len(tasks),), force)
    task_path_count = open_array(
        output_root / "running" / "train_seed42_reconstruction_path_count.npy",
        np.int64,
        (len(tasks),),
        force,
    )
    task_identity = open_array(
        output_root / "running" / "train_seed42_reference_identity.npy",
        np.float64,
        (len(tasks), 7),
        force,
    )
    frequencies = torch.as_tensor(
        np.arange(-96, 96, dtype=np.float32) * 30_000.0,
        device=device,
    )
    with h5py.File(source, "r") as handle:
        for task_index, (site_name, batch_name) in enumerate(tasks):
            if bool(done[task_index]):
                continue
            site = int(site_name.removeprefix("site"))
            batch = handle[site_name][batch_name]
            rows = np.asarray(batch["row_indices"], dtype=np.int64)
            offsets = np.asarray(batch["offsets"], dtype=np.int64)
            tau = np.asarray(batch["tau"], dtype=np.float32)
            if coefficient_mode == "full":
                coefficient = np.asarray(batch["full_coefficient"], dtype=np.complex64)
                expanded = joint.expand_reference_coefficient(
                    np.asarray(batch["reference_coefficient"], dtype=np.complex64),
                    np.asarray(batch["theta_t"], dtype=np.float32),
                    np.asarray(batch["phi_t"], dtype=np.float32),
                    configuration,
                    site,
                )
                full_flat = coefficient.reshape(-1).astype(np.complex128)
                expanded_flat = expanded.reshape(-1).astype(np.complex128)
                difference = expanded_flat - full_flat
                dot = np.vdot(full_flat, expanded_flat)
                task_identity[task_index] = np.asarray(
                    (
                        float(dot.real),
                        float(dot.imag),
                        float(np.vdot(expanded_flat, expanded_flat).real),
                        float(np.vdot(full_flat, full_flat).real),
                        float(np.vdot(difference, difference).real),
                        float(np.max(np.abs(difference), initial=0.0)),
                        float(full_flat.size),
                    ),
                    dtype=np.float64,
                )
                task_identity.flush()
            elif coefficient_mode == "reference":
                coefficient = joint.expand_reference_coefficient(
                    np.asarray(batch["reference_coefficient"], dtype=np.complex64),
                    np.asarray(batch["theta_t"], dtype=np.float32),
                    np.asarray(batch["phi_t"], dtype=np.float32),
                    configuration,
                    site,
                )
            else:
                raise ValueError(f"unsupported coefficient mode {coefficient_mode}")

            coefficient_tensor = torch.as_tensor(coefficient, device=device)
            tau_tensor = torch.as_tensor(tau, device=device)
            channels = []
            for local_index in range(len(rows)):
                start = int(offsets[local_index])
                end = int(offsets[local_index + 1])
                if end <= start:
                    channels.append(np.zeros((256, 4, 192), dtype=np.complex64))
                    continue
                phase = torch.exp(
                    -2j
                    * torch.pi
                    * tau_tensor[start:end, None]
                    * frequencies[None, :]
                )
                channel = torch.einsum(
                    "pmu,ps->mus", coefficient_tensor[start:end], phase
                )
                channels.append(channel.cpu().numpy().astype(np.complex64, copy=False))
            local = generator.score_aligned_statistics(np.stack(channels))
            path_count = np.diff(offsets).astype(np.float32)
            values = {
                "pas": (local["pas"], local["pas_valid"]),
                "pdp": (local["pdp"], local["pdp_valid"]),
                "ad": (local["ad"], local["ad_valid"]),
                "log_power": (local["log_power"], local["log_power_valid"]),
                "path_count": (path_count, np.ones(len(rows), dtype=bool)),
            }
            for domain, (value, valid) in values.items():
                arrays[domain]["mean"][rows] = value
                arrays[domain]["count"][rows] = np.asarray(valid, dtype=np.uint16)
                arrays[domain]["mean"].flush()
                arrays[domain]["count"].flush()
            task_path_count[task_index] = int(offsets[-1])
            task_path_count.flush()
            done[task_index] = True
            done.flush()
            print(
                f"E212 spectra {name} {task_index + 1}/{len(tasks)} "
                f"rows={len(rows)} paths={int(offsets[-1])}",
                flush=True,
            )
    if not bool(np.all(done)):
        raise RuntimeError(f"incomplete spectra reconstruction for {name}")
    for domain in generator.DOMAINS:
        arrays[domain]["m2"].flush()
    identity_values = np.asarray(task_identity, dtype=np.float64)
    identity = {
        "dot_real": float(identity_values[:, 0].sum()),
        "dot_imag": float(identity_values[:, 1].sum()),
        "expanded_norm2": float(identity_values[:, 2].sum()),
        "full_norm2": float(identity_values[:, 3].sum()),
        "diff_norm2": float(identity_values[:, 4].sum()),
        "maximum_absolute_error": float(identity_values[:, 5].max(initial=0.0)),
        "coefficient_count": int(identity_values[:, 6].sum()),
    }
    if identity["coefficient_count"]:
        denominator = np.sqrt(identity["full_norm2"] * identity["expanded_norm2"])
        identity["complex_cosine"] = float(
            np.hypot(identity["dot_real"], identity["dot_imag"]) / denominator
        )
        identity["relative_l2"] = float(
            np.sqrt(identity["diff_norm2"] / identity["full_norm2"])
        )
    report = {
        "experiment": VERSION,
        "stage": "spectra",
        "status": "complete",
        "configuration": name,
        "seed": SEED,
        "source": str(source),
        "source_size": source.stat().st_size,
        "coefficient_mode": coefficient_mode,
        "row_count": row_count,
        "task_count": len(tasks),
        "path_count_processed": int(np.asarray(task_path_count).sum()),
        "reference_expansion_identity": identity if identity["coefficient_count"] else None,
        "output": str(output_root),
        "seconds": float(time.time() - started),
    }
    atomic_json(paths.reports / f"spectra_{name}.json", report)
    return report


def stage_identity(paths: ExperimentPaths) -> None:
    import analyze_joint_path_token_oracle as joint
    from round2.data import load_positions

    _, train_positions, _, _, _ = load_positions()
    sources = {}
    for name in (BASE_NAME, SCATTER_NAME):
        source = source_h5(name)
        with h5py.File(source, "r") as handle:
            sources[name] = {
                "path": str(source),
                "size": source.stat().st_size,
                "seed": int(handle.attrs["seed"]),
                "split": str(handle.attrs["split"]),
                "complete": bool(handle.attrs["complete"]),
                "coefficient_mode": str(handle.attrs["coefficient_mode"]),
            }
        selected = joint.path_files(historical_path_root(name), 1, "train")
        if selected != [source]:
            raise RuntimeError(f"seed-count=1 does not select seed42 for {name}: {selected}")
    payload = {
        "experiment": VERSION,
        "stage": "identity",
        "status": "complete",
        "baseline_zip": {
            "path": str(BASELINE_ZIP),
            "sha256": sha256(BASELINE_ZIP),
        },
        "baseline_report": str(BASELINE_REPORT),
        "train_count": int(len(train_positions)),
        "seed": SEED,
        "sources": sources,
        "accepted_scatter_reconstruction_relative_l2": 1.5759e-5,
    }
    expected = "18a7e257ac42a9a88c2c4941b9ed547537084b25b425ad8584b32d79689939a6"
    if payload["baseline_zip"]["sha256"] != expected:
        raise RuntimeError("V16 baseline ZIP hash changed")
    atomic_json(paths.reports / "identity.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def stage_tokens(paths: ExperimentPaths, force: bool) -> None:
    joint = configure_paths(paths)
    from round2.data import load_positions

    _, train_positions, _, train_site, _ = load_positions()
    args = SimpleNamespace(
        seed_count=1,
        max_paths=128,
        base_reserve=48,
        base_per_seed=96,
        scatter_per_seed=64,
        scatter_pool=256,
        include_scatter=True,
        selection_seed=106,
        reuse_path_cache=not force,
    )
    started = time.time()
    cache = joint.cache_paths(
        args,
        np.arange(len(train_positions), dtype=np.int64),
        np.arange(len(train_positions), dtype=np.int64),
        train_site,
        split="train",
    )
    mask = np.asarray(cache["mask"], dtype=bool)
    seed = np.asarray(cache["seed"], dtype=np.int32)
    if np.any(seed[mask] != SEED):
        raise RuntimeError("single-seed path cache contains a non-42 seed")
    payload = {
        "experiment": VERSION,
        "stage": "tokens",
        "status": "complete",
        "configuration": vars(args),
        "cache": str(paths.token_cache),
        "valid_paths": int(mask.sum()),
        "mean_paths": float(mask.sum(axis=1).mean()),
        "empty_points": int(np.sum(mask.sum(axis=1) == 0)),
        "seconds": float(time.time() - started),
    }
    atomic_json(paths.reports / "tokens.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def stage_features(paths: ExperimentPaths) -> None:
    builder = configure_features(paths)
    invoke_main(builder, ["--split", "train", "--presence-seed-count", "1"])


def initialize_oof_outputs(paths: ExperimentPaths, force: bool) -> None:
    _, _, v9 = configure_spectra(paths)
    outputs = {
        paths.e40 / "oof_pas_mean_f16.npy": (500, 2, 16, 8, 4, 192),
        paths.e40 / "oof_pdp_mean_f16.npy": (500, 256, 4, 192),
        v9.output_path("pas", "oof"): (500, 2, 16, 8, 4, 192),
        v9.output_path("pdp", "oof"): (500, 256, 4, 192),
    }
    for path, shape in outputs.items():
        open_array(path, np.float16, shape, force)
    print(json.dumps({str(path): list(shape) for path, shape in outputs.items()}, indent=2))


def stage_e40(paths: ExperimentPaths, domain: str, fold: int, device: str) -> None:
    _, e40, _ = configure_spectra(paths)
    e40.OOF_REPORT = paths.reports / "e40" / f"oof_{domain}_fold{fold}.json"
    invoke_main(
        e40,
        [
            "--phase", "oof", "--modes", "mean", "--domains", domain,
            "--folds", str(fold), "--device", device,
        ],
    )


def stage_v9_reliability(paths: ExperimentPaths, domain: str) -> None:
    _, _, v9 = configure_spectra(paths)
    paths.v9.mkdir(parents=True, exist_ok=True)
    v9.spectral_reliability(domain, "train", force=True)


def stage_v9(paths: ExperimentPaths, domain: str, fold: int, device: str) -> None:
    _, _, v9 = configure_spectra(paths)
    v9.OOF_REPORT = paths.reports / "v9" / f"oof_{domain}_fold{fold}.json"
    invoke_main(
        v9,
        [
            "--phase", "oof", "--domains", domain, "--folds", str(fold),
            "--device", device,
        ],
    )


def stage_link(paths: ExperimentPaths, phase: str, device: str, force: bool) -> None:
    link = configure_link(paths)
    arguments = ["--phase", phase, "--device", device]
    if force and phase == "build":
        for path in (link.RAW_PDP, link.POINT_CONTEXT):
            path.unlink(missing_ok=True)
    elif force:
        arguments.append("--force")
    invoke_main(link, arguments)


def stage_canonical(paths: ExperimentPaths, device: str, force: bool) -> None:
    canonical = configure_canonical(paths)
    arguments = [
        "--device",
        device,
        "--batch-size",
        "4",
        "--skip-historical-control-assertions",
    ]
    if force:
        arguments.append("--force")
    invoke_main(canonical, arguments)


def stage_e85(paths: ExperimentPaths, device: str, force: bool) -> None:
    _, e85, _, _, _, _, _ = configure_surface(paths)
    arguments = ["--stage", "strict", "--device", device, "--batch-size", "48"]
    if force:
        arguments.append("--force-prepare")
    invoke_main(e85, arguments)


def stage_e86(paths: ExperimentPaths, device: str) -> None:
    _, _, e86, frozen, _, _, _ = configure_surface(paths)
    torch_device = torch.device(device)
    for _, (pas_path, pdp_path, endpoint_path) in e86.RENDERED.items():
        e86.render_cache(
            str(endpoint_path), str(pas_path), str(pdp_path), torch_device, 8, True
        )
    invoke_main(frozen, [])


def stage_e99(paths: ExperimentPaths, device: str) -> None:
    _, _, _, _, _, e99, _ = configure_surface(paths)
    invoke_main(e99, ["--device", device, "--folds", "0,1,2,3,4"])


def stage_e190(paths: ExperimentPaths, outer: int, device: str) -> None:
    e190 = configure_e190(paths)
    invoke_main(
        e190,
        ["--stage", "screen", "--outer", str(outer), "--device", device],
    )


def stage_aggregate(paths: ExperimentPaths, device: str) -> None:
    e190 = configure_e190(paths)
    invoke_main(e190, ["--stage", "aggregate", "--device", device])


def stage_finalize(paths: ExperimentPaths) -> None:
    e190 = configure_e190(paths)
    candidate_report = json.loads(e190.FULL_REPORT.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    selected = candidate_report["selected"]
    score = float(selected["score"])
    payload = {
        "experiment": VERSION,
        "status": "single_seed_passed" if score >= BASELINE_SCORE else "single_seed_failed",
        "scope": "diagnostic_only_fixed500_no_test_no_zip_no_online_submission",
        "seed": SEED,
        "accepted_reconstruction_difference": {
            "scatter_reference_to_full_relative_l2": 1.5759e-5,
            "reason": "user selected compressed historical Scatter HDF5 reconstruction",
        },
        "baseline": {
            "source": str(BASELINE_REPORT),
            "pas": 0.676941576,
            "pdp": 0.780347398,
            "nmse_score": 0.517297070,
            "score": BASELINE_SCORE,
            "online_reference": 0.686754,
            "machine_report": baseline_report,
        },
        "single_seed": selected,
        "delta": {
            "pas": float(selected["pas"] - 0.676941576),
            "pdp": float(selected["pdp"] - 0.780347398),
            "nmse_score": float(selected["nmse_score"] - 0.517297070),
            "score": float(score - BASELINE_SCORE),
        },
        "folds": candidate_report["folds"],
        "decision": {
            "threshold": BASELINE_SCORE,
            "passed": bool(score >= BASELINE_SCORE),
            "interpretation": (
                "the other 23 seeds are eligible for removal"
                if score >= BASELINE_SCORE
                else "the other 23 seeds provide measurable accuracy in the V16 method"
            ),
        },
        "output": {
            "channel": str(e190.FULL_CHANNEL),
            "sha256": sha256(e190.FULL_CHANNEL),
            "shape": list(np.load(e190.FULL_CHANNEL, mmap_mode="r").shape),
        },
        "known_fold_caveat": (
            "V16/E190 model training uses FULL_FOLD while canonical fixed500 scoring "
            "uses the canonical query fold; only 124/500 assignments agree."
        ),
    }
    atomic_json(paths.final_report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def stage_status(paths: ExperimentPaths) -> None:
    required = {
        "identity": paths.reports / "identity.json",
        "base_spectra": paths.reports / f"spectra_{BASE_NAME}.json",
        "scatter_spectra": paths.reports / f"spectra_{SCATTER_NAME}.json",
        "tokens": paths.reports / "tokens.json",
        "features": paths.features / "path_features_train_n4000.npy",
        "e40_pas": paths.e40 / "oof_pas_mean_f16.npy",
        "e40_pdp": paths.e40 / "oof_pdp_mean_f16.npy",
        "v9_pas": paths.v9 / "oof_pas_joint_f16.npy",
        "v9_pdp": paths.v9 / "oof_pdp_joint_f16.npy",
        "link_raw": paths.link / "train_raw_path_pdp_f16.npy",
        "v11": paths.canonical / "V11_Canonical_500_OOF_Channel.npy",
        "e85_endpoint": paths.e85 / "strict_oof_geometry_sequence_path_endpoint.npz",
        "e99_mapping": paths.e99 / "strict_mapping_parameters.npz",
        "e190": paths.e190 / "fixed500_channel.npy",
        "final": paths.final_report,
    }
    print(
        json.dumps(
            {
                name: {
                    "exists": path.exists(),
                    "path": str(path),
                    "size": path.stat().st_size if path.exists() else None,
                }
                for name, path in required.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "identity", "spectra", "tokens", "features", "init-oof", "e40",
            "v9-reliability", "v9", "link-build", "link-train", "canonical",
            "e85", "e86", "e99", "e190", "aggregate", "finalize", "status",
        ),
    )
    parser.add_argument("--config", choices=(BASE_NAME, SCATTER_NAME))
    parser.add_argument("--domain", choices=("pas", "pdp"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--outer", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = experiment_paths()
    ensure_directories(paths)
    if args.stage == "identity":
        stage_identity(paths)
    elif args.stage == "spectra":
        if args.config is None:
            raise ValueError("--config is required for spectra")
        reconstruct_single_seed_spectra(paths, args.config, torch.device(args.device), args.force)
    elif args.stage == "tokens":
        stage_tokens(paths, args.force)
    elif args.stage == "features":
        stage_features(paths)
    elif args.stage == "init-oof":
        initialize_oof_outputs(paths, args.force)
    elif args.stage == "e40":
        if args.domain is None or args.fold not in range(5):
            raise ValueError("e40 requires --domain and --fold 0..4")
        stage_e40(paths, args.domain, args.fold, args.device)
    elif args.stage == "v9-reliability":
        if args.domain is None:
            raise ValueError("v9-reliability requires --domain")
        stage_v9_reliability(paths, args.domain)
    elif args.stage == "v9":
        if args.domain is None or args.fold not in range(5):
            raise ValueError("v9 requires --domain and --fold 0..4")
        stage_v9(paths, args.domain, args.fold, args.device)
    elif args.stage in ("link-build", "link-train"):
        stage_link(paths, args.stage.removeprefix("link-"), args.device, args.force)
    elif args.stage == "canonical":
        stage_canonical(paths, args.device, args.force)
    elif args.stage == "e85":
        stage_e85(paths, args.device, args.force)
    elif args.stage == "e86":
        stage_e86(paths, args.device)
    elif args.stage == "e99":
        stage_e99(paths, args.device)
    elif args.stage == "e190":
        if args.outer not in range(5):
            raise ValueError("e190 requires --outer 0..4")
        stage_e190(paths, args.outer, args.device)
    elif args.stage == "aggregate":
        stage_aggregate(paths, args.device)
    elif args.stage == "finalize":
        stage_finalize(paths)
    elif args.stage == "status":
        stage_status(paths)


if __name__ == "__main__":
    main()
