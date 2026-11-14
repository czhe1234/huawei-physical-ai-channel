#!/usr/bin/env python3
"""Run V16 RT-information ablations on the historical fixed-500 pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from project_paths import PATHS
import run_e212_v16_single_seed_ablation as shared


VERSION = "Round2-E216-V16-single-seed-information-ablation"
BASE_NAME = shared.BASE_NAME
SCATTER_NAME = shared.SCATTER_NAME
BASELINE_SCORE = 0.6863750036074729
MINIMUM_SCORE = 0.6800000000000000
BASELINE_METRICS = {
    "pas": 0.6769415758539015,
    "pdp": 0.7803473980573273,
    "nmse": 0.9331251955178451,
    "nmse_score": 0.5172970702149068,
    "score": BASELINE_SCORE,
}
EXPECTED_ZIP_SHA256 = (
    "18a7e257ac42a9a88c2c4941b9ed547537084b25b425ad8584b32d79689939a6"
)

ROOT = PATHS.artifacts / "round2" / "e216_v16_rt_information_ablation"
REPORT_ROOT = (
    PATHS.analysis / "results" / "round2_e216_v16_rt_information_ablation"
)
HIST_SPECTRA = PATHS.artifacts / "round2" / "sionna_multiseed"
HIST_TOKENS = PATHS.artifacts / "round2" / "joint_path_token_oracle"
HIST_TOKEN_CACHE = (
    HIST_TOKENS / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
HIST_FEATURES = PATHS.artifacts / "round2" / "explicit_path_pdp_predictor"
HIST_E40 = PATHS.artifacts / "round2" / "sionna_official_calibration"
HIST_V9 = PATHS.artifacts / "round2" / "v9_joint_sionna_endpoints"
HIST_E85 = PATHS.artifacts / "round2" / "surface_interaction_energy_model"
HIST_E99 = PATHS.artifacts / "round2" / "corrected_array_surface_model"
BASELINE_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "official_sionna_path_calibrator"
    / "fixed500_channel.npy"
)


@dataclass(frozen=True)
class Variant:
    name: str
    spectral_seeds: int = 1
    path_seeds: int = 1
    physical_view: str = "both"
    token_view: str = "both"
    use_presence: bool = True
    rebuild_e40: bool = True
    rebuild_v9: bool = True
    path_models_changed: bool = True


VARIANTS = {
    item.name: item
    for item in (
        Variant(name="seed42_both"),
        Variant(
            name="seed42_base_only",
            physical_view="base",
            token_view="base",
        ),
        Variant(
            name="seed42_scatter_only",
            physical_view="scatter",
            token_view="scatter",
        ),
    )
}

PRIMARY_VARIANT = "seed42_both"


def token_tag(variant: Variant) -> str:
    include_scatter = variant.token_view != "base"
    tag = (
        f"n4000_s{variant.path_seeds:02d}_k128_b096_d064_"
        f"scatter{int(include_scatter)}_sel106"
    )
    if variant.token_view == "scatter":
        tag += "_base0"
    if not variant.use_presence:
        tag += "_presence0"
    return tag


def experiment_paths(variant: Variant) -> shared.ExperimentPaths:
    root = ROOT / "variants" / variant.name
    reports = REPORT_ROOT / variant.name
    token_changed = variant.path_models_changed
    spectra = ROOT / "shared_seed42_spectra"
    tokens = root / "joint_path_token_oracle"
    token_cache = tokens / token_tag(variant)
    features = root / "explicit_path_pdp_predictor"
    e40 = root / "sionna_official_calibration"
    v9 = root / "v9_joint_sionna_endpoints"
    e99 = root / "corrected_array_surface_model"
    return shared.ExperimentPaths(
        root=root,
        reports=reports,
        logs=root / "logs",
        spectra=spectra,
        tokens=tokens,
        token_cache=token_cache,
        features=features,
        e40=e40,
        v9=v9,
        link=root / "full_path_pdp_link_gate",
        canonical=root / "canonical_500_version_replay",
        e85=root / "surface_interaction_energy_model",
        e99=e99,
        e190=root / "official_sionna_path_calibrator",
        final_report=reports / "fixed500_report.json",
    )


def physical_configs(view: str) -> tuple[str, str]:
    if view == "both":
        return BASE_NAME, SCATTER_NAME
    if view == "base":
        return BASE_NAME, BASE_NAME
    if view == "scatter":
        return SCATTER_NAME, SCATTER_NAME
    raise ValueError(view)


def install_variant(variant: Variant) -> None:
    original_spectra = shared.configure_spectra
    original_canonical = shared.configure_canonical
    original_surface = shared.configure_surface
    original_e190 = shared.configure_e190

    def configure_spectra(paths: shared.ExperimentPaths):
        prepare, e40, v9 = original_spectra(paths)
        base_config, scatter_config = physical_configs(variant.physical_view)
        prepare.SIONNA_ROOT = paths.spectra
        prepare.SEEDS = variant.spectral_seeds
        e40.SEEDS = variant.spectral_seeds
        e40.BASE_CONFIG = base_config
        e40.SCATTER_CONFIG = scatter_config
        v9.SEEDS = variant.spectral_seeds
        v9.BASE_CONFIG = base_config
        v9.SCATTER_CONFIG = scatter_config
        return prepare, e40, v9

    shared.configure_spectra = configure_spectra

    def configure_canonical(paths: shared.ExperimentPaths):
        canonical = original_canonical(paths)
        prepare, _, _ = shared.configure_spectra(paths)
        base_config, scatter_config = physical_configs(variant.physical_view)
        canonical.E39_BASE_PAS = prepare.running_path(
            base_config, "train", "pas", "mean"
        )
        canonical.E39_SCATTER_PAS = prepare.running_path(
            scatter_config, "train", "pas", "mean"
        )
        return canonical

    shared.configure_canonical = configure_canonical

    def configure_surface(paths: shared.ExperimentPaths):
        modules = original_surface(paths)
        e84, e85, e86, frozen, e98, e99, e100 = modules
        e85.PRESENCE_SEED_COUNT = variant.path_seeds
        if variant.path_models_changed:
            return modules

        canonical = shared.configure_canonical(paths)
        e85.ARTIFACT_DIR = HIST_E85
        e85.ANALYSIS_DIR = PATHS.analysis / "results" / "round2_surface_interaction_energy_model"
        e85.TARGET_PAS = HIST_E85 / "official_pas_cosine_sufficient_target_f32.npy"
        e85.TARGET_PDP = HIST_E85 / "official_pdp_cosine_sufficient_target_f32.npy"
        e85.POINT_POWER = HIST_E85 / "official_point_power_f64.npy"
        e85.BASE_COORDINATE = HIST_E85 / "sionna_aligned_coordinates_f32.npy"
        e85.PORT_PRIOR = HIST_E85 / "sionna_path_port_prior_f16.npy"
        e85.SEQUENCE_GEOMETRY = HIST_E85 / "sequence_geometry_f16.npy"
        e85.SEQUENCE_HASH = HIST_E85 / "sequence_hash_buckets_i32.npy"
        e85.PREPARE_REPORT = e85.ANALYSIS_DIR / "surface_interaction_training_cache.json"
        e85.SCREEN_REPORT = e85.ANALYSIS_DIR / "surface_interaction_screen.json"
        e85.FULL_REPORT = e85.ANALYSIS_DIR / "surface_interaction_five_fold.json"
        e85.STRICT_REPORT = e85.ANALYSIS_DIR / "surface_interaction_strict_holes.json"

        e86.ARTIFACT_DIR = paths.e85
        e86.ANALYSIS_DIR = paths.reports / "e86"
        e86.REPORT = e86.ANALYSIS_DIR / "e86_surface_endpoint_v11_blend.json"
        e86.METRICS_PATH = paths.e85 / "e86_candidate_metrics.npz"
        e86.NESTED_CHANNEL = paths.e85 / "e86_nested_oof_channel.npy"
        e86.RENDERED = {
            "geometry_sequence": (
                HIST_E85 / "e86_geometry_sequence_rendered_pas_f16.npy",
                HIST_E85 / "e86_geometry_sequence_rendered_pdp_f16.npy",
                HIST_E85 / "strict_oof_geometry_sequence_path_endpoint.npz",
            ),
            "no_sequence": (
                HIST_E85 / "e86_no_sequence_rendered_pas_f16.npy",
                HIST_E85 / "e86_no_sequence_rendered_pdp_f16.npy",
                HIST_E85 / "strict_oof_no_sequence_path_endpoint.npz",
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

        e99.ARTIFACT_DIR = HIST_E99
        e99.MAPPING_PARAMETERS = HIST_E99 / "strict_mapping_parameters.npz"
        e100.V12_CHANNEL = frozen.OUTPUT
        e100.OLD_ENDPOINT = e86.RENDERED["geometry_sequence"][2]
        return modules

    shared.configure_surface = configure_surface

    def configure_e190(paths: shared.ExperimentPaths):
        e190 = original_e190(paths)
        e190.PRESENCE_SEED_COUNT = variant.path_seeds
        return e190

    shared.configure_e190 = configure_e190


def install_mean_only_reliability(v9: Any) -> None:
    if not hasattr(v9, "_e216_source_scalar_matrix"):
        v9._e216_source_scalar_matrix = v9.source_scalar_matrix
        v9._e216_spectral_reliability = v9.spectral_reliability

    original_scalar = v9._e216_source_scalar_matrix
    original_reliability = v9._e216_spectral_reliability

    def source_scalar_matrix(config: str, split: str) -> np.ndarray:
        value = np.array(original_scalar(config, split), dtype=np.float32, copy=True)
        value[:, [1, 2, 4, 5]] = 0.0
        return value

    def spectral_reliability(domain: str, split: str, force: bool) -> np.ndarray:
        value = original_reliability(domain, split, force)
        path = v9.reliability_path(domain, split)
        writable = np.load(path, mmap_mode="r+")
        writable[:, 2:6] = 0.0
        writable.flush()
        return np.load(path, mmap_mode="r")

    v9.source_scalar_matrix = source_scalar_matrix
    v9.spectral_reliability = spectral_reliability


def stage_identity(device: torch.device) -> None:
    import analyze_e203_v15_information_contribution as e203

    zip_hash = shared.sha256(shared.BASELINE_ZIP)
    if zip_hash != EXPECTED_ZIP_SHA256:
        raise RuntimeError("V16 baseline ZIP identity changed")
    e203.configure_experiment("v11")
    evaluator = e203.CanonicalBaseEvaluator(device)
    metrics = evaluator.channel_metrics(np.load(BASELINE_CHANNEL, mmap_mode="r"))
    error = max(
        abs(float(metrics[name]) - BASELINE_METRICS[name])
        for name in ("pas", "pdp", "nmse", "nmse_score", "score")
    )
    if error > 1e-7:
        raise RuntimeError(f"V16 fixed500 identity error {error} exceeds 1e-7")
    payload = {
        "experiment": VERSION,
        "stage": "identity",
        "status": "complete",
        "baseline_zip": PATHS.relative(shared.BASELINE_ZIP),
        "baseline_zip_sha256": zip_hash,
        "baseline_channel": PATHS.relative(BASELINE_CHANNEL),
        "metrics": metrics,
        "maximum_metric_error": float(error),
        "minimum_acceptable_score": MINIMUM_SCORE,
        "fold_assignment_match": 124,
        "fold_assignment_label_access_risk": 376,
        "variants": {name: vars(item) for name, item in VARIANTS.items()},
    }
    shared.atomic_json(REPORT_ROOT / "identity.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def stage_audit(variant: Variant, paths: shared.ExperimentPaths) -> None:
    prepare, e40, v9 = shared.configure_spectra(paths)
    joint = shared.configure_paths(paths)
    canonical = shared.configure_canonical(paths)
    e190 = shared.configure_e190(paths)
    base_config, scatter_config = physical_configs(variant.physical_view)
    required = {
        "token_cache": paths.token_cache / "mask.npy",
        "base_pas": prepare.running_path(base_config, "train", "pas", "mean"),
        "scatter_pas": prepare.running_path(
            scatter_config, "train", "pas", "mean"
        ),
        "truth_pas": e40.TRUTH_PAS,
        "truth_pdp": e40.TRUTH_PDP,
        "fold_ids": e190.e85.FULL_FOLD if hasattr(e190, "e85") else joint.FULL_FOLD,
    }
    payload = {
        "experiment": VERSION,
        "variant": variant.name,
        "stage": "audit",
        "status": "complete",
        "configuration": vars(variant),
        "resolved": {
            "spectral_root": PATHS.relative(prepare.SIONNA_ROOT),
            "spectral_seeds": int(prepare.SEEDS),
            "base_config": e40.BASE_CONFIG,
            "scatter_config": e40.SCATTER_CONFIG,
            "v9_base_config": v9.BASE_CONFIG,
            "v9_scatter_config": v9.SCATTER_CONFIG,
            "token_cache": PATHS.relative(paths.token_cache),
            "token_artifact_root": PATHS.relative(joint.ARTIFACT_DIR),
            "canonical_base_pas": PATHS.relative(canonical.E39_BASE_PAS),
            "canonical_scatter_pas": PATHS.relative(canonical.E39_SCATTER_PAS),
            "path_presence_denominator": int(e190.PRESENCE_SEED_COUNT),
        },
        "required": {
            name: {
                "path": PATHS.relative(path),
                "exists": path.is_file(),
                "size": path.stat().st_size if path.is_file() else None,
            }
            for name, path in required.items()
        },
    }
    shared.atomic_json(paths.reports / "audit.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def stage_tokens(variant: Variant, paths: shared.ExperimentPaths, force: bool) -> None:
    if not variant.path_models_changed:
        raise ValueError(f"{variant.name} reuses the historical token cache")
    joint = shared.configure_paths(paths)
    from round2.data import load_positions

    _, train_positions, _, train_site, _ = load_positions()
    args = SimpleNamespace(
        seed_count=variant.path_seeds,
        max_paths=128,
        base_reserve=48,
        base_per_seed=96,
        scatter_per_seed=64,
        scatter_pool=256,
        include_base=variant.token_view != "scatter",
        include_scatter=variant.token_view != "base",
        use_presence=variant.use_presence,
        selection_seed=106,
        reuse_path_cache=not force,
    )
    cache = joint.cache_paths(
        args,
        np.arange(len(train_positions), dtype=np.int64),
        np.arange(len(train_positions), dtype=np.int64),
        train_site,
        split="train",
    )
    if Path(cache["rows"].filename).parent != paths.token_cache:
        raise RuntimeError("token cache path does not match the registered variant")
    mask = np.asarray(cache["mask"], dtype=bool)
    presence = np.asarray(cache["presence"], dtype=np.uint8)
    seed = np.asarray(cache["seed"], dtype=np.int32)
    if np.any(seed[mask] != shared.SEED):
        raise RuntimeError("single-seed path cache contains a non-42 seed")
    if np.any(presence[mask] < 1):
        raise RuntimeError("single-seed path cache contains invalid zero presence")
    payload = {
        "experiment": VERSION,
        "variant": variant.name,
        "stage": "tokens",
        "status": "complete",
        "configuration": vars(args),
        "cache": PATHS.relative(paths.token_cache),
        "valid_paths": int(mask.sum()),
        "mean_paths": float(mask.sum(axis=1).mean()),
        "base_paths": int(np.sum(mask & (np.asarray(cache["source"]) == 0))),
        "scatter_paths": int(np.sum(mask & (np.asarray(cache["source"]) == 1))),
        "presence_min": int(presence[mask].min()) if np.any(mask) else None,
        "presence_max": int(presence[mask].max()) if np.any(mask) else None,
        "within_seed_duplicate_identity_paths": int(np.sum(mask & (presence > 1))),
        "seed_values": sorted(int(value) for value in np.unique(seed[mask])),
    }
    shared.atomic_json(paths.reports / "tokens.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def stage_features(variant: Variant, paths: shared.ExperimentPaths) -> None:
    if not variant.path_models_changed:
        raise ValueError(f"{variant.name} reuses historical path features")
    builder = shared.configure_features(paths)
    shared.invoke_main(
        builder,
        ["--split", "train", "--presence-seed-count", str(variant.path_seeds)],
    )


def stage_surface(variant: Variant, paths: shared.ExperimentPaths, device: str) -> None:
    if variant.path_models_changed:
        shared.stage_e86(paths, device)
        return
    paths.e85.mkdir(parents=True, exist_ok=True)
    _, _, _, frozen, _, _, _ = shared.configure_surface(paths)
    shared.invoke_main(frozen, [])


def stage_finalize(variant: Variant, paths: shared.ExperimentPaths) -> None:
    e190 = shared.configure_e190(paths)
    report = json.loads(e190.FULL_REPORT.read_text(encoding="utf-8"))
    selected = report["selected"]
    delta = {
        name: float(selected[name]) - BASELINE_METRICS[name]
        for name in ("pas", "pdp", "nmse", "nmse_score", "score")
    }
    primary_selected = selected
    if variant.name != PRIMARY_VARIANT:
        primary_report = (
            experiment_paths(VARIANTS[PRIMARY_VARIANT]).reports
            / "e190"
            / "fixed500_report.json"
        )
        primary_selected = json.loads(primary_report.read_text(encoding="utf-8"))["selected"]
    delta_from_primary = {
        name: float(selected[name]) - float(primary_selected[name])
        for name in ("pas", "pdp", "nmse", "nmse_score", "score")
    }
    payload = {
        "experiment": VERSION,
        "variant": variant.name,
        "status": (
            "minimum_score_passed"
            if float(selected["score"]) >= MINIMUM_SCORE
            else "minimum_score_failed"
        ),
        "configuration": vars(variant),
        "baseline": BASELINE_METRICS,
        "selected": selected,
        "delta_from_v16": delta,
        "single_seed_both": primary_selected,
        "delta_from_single_seed_both": delta_from_primary,
        "folds": report["folds"],
        "decision": {
            "threshold": MINIMUM_SCORE,
            "passed": bool(float(selected["score"]) >= MINIMUM_SCORE),
            "view_removal_preserves_single_seed_score": bool(
                delta_from_primary["score"] >= 0.0
            ),
        },
        "output": {
            "channel": PATHS.relative(e190.FULL_CHANNEL),
            "sha256": shared.sha256(e190.FULL_CHANNEL),
            "shape": list(np.load(e190.FULL_CHANNEL, mmap_mode="r").shape),
        },
        "known_fold_caveat": "124/500 canonical assignments match FULL_FOLD",
    }
    shared.atomic_json(paths.final_report, payload)
    print(json.dumps(payload, indent=2), flush=True)


def stage_summary() -> None:
    reports = {
        name: json.loads(experiment_paths(variant).final_report.read_text(encoding="utf-8"))
        for name, variant in VARIANTS.items()
    }
    scores = {
        name: float(report["selected"]["score"])
        for name, report in reports.items()
    }
    payload = {
        "experiment": VERSION,
        "status": "complete",
        "scope": "diagnostic_only_fixed500_no_test_no_zip_no_online_submission",
        "seed": shared.SEED,
        "baseline": BASELINE_METRICS,
        "minimum_acceptable_score": MINIMUM_SCORE,
        "variants": reports,
        "best_variant": max(scores, key=scores.get),
        "best_score": max(scores.values()),
    }
    shared.atomic_json(REPORT_ROOT / "summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def stage_status(variant: Variant, paths: shared.ExperimentPaths) -> None:
    required = {
        "identity": REPORT_ROOT / "identity.json",
        "tokens": paths.token_cache / "mask.npy",
        "features": paths.features / "path_features_train_n4000.npy",
        "e40_pas": paths.e40 / "oof_pas_mean_f16.npy",
        "e40_pdp": paths.e40 / "oof_pdp_mean_f16.npy",
        "v9_pas": paths.v9 / "oof_pas_joint_f16.npy",
        "v9_pdp": paths.v9 / "oof_pdp_joint_f16.npy",
        "link_raw": paths.link / "train_raw_path_pdp_f16.npy",
        "v11": paths.canonical / "V11_Canonical_500_OOF_Channel.npy",
        "v12": paths.e85 / "e86_frozen_geometry_sequence_oof_channel.npy",
        "mapping": (
            paths.e99 / "strict_mapping_parameters.npz"
            if variant.path_models_changed
            else HIST_E99 / "strict_mapping_parameters.npz"
        ),
        "channel": paths.e190 / "fixed500_channel.npy",
        "final": paths.final_report,
    }
    print(
        json.dumps(
            {
                name: {
                    "exists": path.is_file(),
                    "path": str(path),
                    "size": path.stat().st_size if path.is_file() else None,
                }
                for name, path in required.items()
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "identity",
            "audit",
            "spectra",
            "tokens",
            "features",
            "init-oof",
            "e40",
            "v9-reliability",
            "v9",
            "link-build",
            "link-train",
            "canonical",
            "e85",
            "e86",
            "e99",
            "e190",
            "aggregate",
            "finalize",
            "summary",
            "status",
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
    if args.stage == "identity":
        stage_identity(torch.device(args.device))
        return
    if args.stage == "summary":
        stage_summary()
        return
    if args.variant is None:
        raise ValueError("--variant is required except for identity")
    variant = VARIANTS[args.variant]
    paths = experiment_paths(variant)
    shared.ensure_directories(paths)
    install_variant(variant)

    if args.stage == "audit":
        stage_audit(variant, paths)
    elif args.stage == "spectra":
        if variant.spectral_seeds != 1 or args.config is None:
            raise ValueError("spectra requires a single-seed variant and --config")
        shared.reconstruct_single_seed_spectra(
            paths, args.config, torch.device(args.device), args.force
        )
    elif args.stage == "tokens":
        stage_tokens(variant, paths, args.force)
    elif args.stage == "features":
        stage_features(variant, paths)
    elif args.stage == "init-oof":
        shared.initialize_oof_outputs(paths, args.force)
    elif args.stage == "e40":
        if not variant.rebuild_e40 or args.domain is None or args.fold not in range(5):
            raise ValueError("e40 requires a rebuilding variant, --domain, and --fold 0..4")
        shared.stage_e40(paths, args.domain, args.fold, args.device)
    elif args.stage == "v9-reliability":
        if not variant.rebuild_v9 or args.domain is None:
            raise ValueError("v9-reliability requires a rebuilding variant and --domain")
        shared.stage_v9_reliability(paths, args.domain)
    elif args.stage == "v9":
        if not variant.rebuild_v9 or args.domain is None or args.fold not in range(5):
            raise ValueError("v9 requires a rebuilding variant, --domain, and --fold 0..4")
        shared.stage_v9(paths, args.domain, args.fold, args.device)
    elif args.stage in ("link-build", "link-train"):
        shared.stage_link(
            paths, args.stage.removeprefix("link-"), args.device, args.force
        )
    elif args.stage == "canonical":
        shared.stage_canonical(paths, args.device, args.force)
    elif args.stage == "e85":
        if not variant.path_models_changed:
            raise ValueError(f"{variant.name} reuses historical E85 endpoints")
        shared.stage_e85(paths, args.device, args.force)
    elif args.stage == "e86":
        stage_surface(variant, paths, args.device)
    elif args.stage == "e99":
        if not variant.path_models_changed:
            raise ValueError(f"{variant.name} reuses historical E99 mapping")
        shared.stage_e99(paths, args.device)
    elif args.stage == "e190":
        if args.outer not in range(5):
            raise ValueError("e190 requires --outer 0..4")
        shared.stage_e190(paths, args.outer, args.device)
    elif args.stage == "aggregate":
        shared.stage_aggregate(paths, args.device)
    elif args.stage == "finalize":
        stage_finalize(variant, paths)
    elif args.stage == "status":
        stage_status(variant, paths)


if __name__ == "__main__":
    main()
