#!/usr/bin/env python3
"""Build strict family-isolated workspaces and E183 Clean action labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E183-strict-clean-action-labels"
FROZEN_V7_STACK_CONFIG = {
    "name": "site_port8_ridge1_sum0",
    "scope": "site",
    "ports": "port8",
    "ridge": 1.0,
    "sum_penalty": 0.0,
}
SOURCE_ROOT = PATHS.root
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e183_strict_action_labels"
WORKSPACE_DIR = Path(
    os.environ.get("E183_WORKSPACE_DIR", str(ARTIFACT_DIR / "workspaces"))
).resolve()
LIBRARY = (
    PATHS.artifacts
    / "round2"
    / "e183_test_hole_specialization"
    / "matched_pseudo_hole_library.npz"
)
IDENTITY_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e183_hybrid_specialization"
    / "canonical_13_action_identity_gate.json"
)
GLOBAL_FOLD_IDS = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
ALLOWED_FAMILIES = (
    "S0H0",
    "S0H1",
    "S0H2",
    "S0H3",
    "S0H4",
    "S1H1",
    "S1H2",
    "S1H3",
    "S1H4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "pipeline", "status"), required=True)
    parser.add_argument("--family", choices=ALLOWED_FAMILIES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", default="4")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stop-step", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def family_root(family: str) -> Path:
    return WORKSPACE_DIR / family


def load_family(family: str) -> tuple[np.ndarray, list[np.ndarray], dict[str, object]]:
    library = np.load(LIBRARY, allow_pickle=False)
    names = library["target_hole_ids"].astype(str)
    target = int(np.flatnonzero(names == family)[0])
    rows = np.flatnonzero(library["target_index"].astype(np.int64) == target)
    rows = rows[np.argsort(library["rank"][rows], kind="stable")]
    if len(rows) != 30:
        raise RuntimeError(f"{family} has {len(rows)} analogs, expected 30")
    offsets = library["point_offsets"].astype(np.int64)
    flat = library["point_ids_flat"].astype(np.int64)
    holes = [flat[offsets[row] : offsets[row + 1]].copy() for row in rows]
    union = np.unique(np.concatenate(holes)).astype(np.int64)
    metadata = {
        "family": family,
        "library_rows": rows.astype(int).tolist(),
        "hole_count": len(holes),
        "hole_sizes": [int(len(value)) for value in holes],
        "union_count": int(len(union)),
        "match_cost": library["match_cost"][rows].astype(float).tolist(),
        "mean_match_cost": float(np.mean(library["match_cost"][rows], dtype=np.float64)),
    }
    return union, holes, metadata


def ensure_identity_gate() -> dict[str, object]:
    report = json.loads(IDENTITY_REPORT.read_text(encoding="utf-8"))
    if not bool(report.get("gate", {}).get("overall_pass")):
        raise RuntimeError("E183 canonical identity gate has not passed")
    return report


def write_npy_subset(
    source: Path,
    destination: Path,
    rows: np.ndarray,
    chunk_size: int = 16,
) -> None:
    value = np.load(source, mmap_mode="r")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=value.dtype,
        shape=(len(rows), *value.shape[1:]),
    )
    for start in range(0, len(rows), chunk_size):
        end = min(start + chunk_size, len(rows))
        output[start:end] = np.asarray(value[rows[start:end]])
        output.flush()


def ensure_official_target_caches(root: Path, external: np.ndarray) -> None:
    source_dir = (
        PATHS.artifacts / "round2" / "surface_interaction_energy_model"
    )
    target_dir = (
        root / "artifacts" / "round2" / "surface_interaction_energy_model"
    )
    targets = (
        ("official_pas_cosine_sufficient_target_f32.npy", 64),
        ("official_pdp_cosine_sufficient_target_f32.npy", 64),
        ("official_point_power_f64.npy", 256),
    )
    for name, chunk_size in targets:
        destination = target_dir / name
        if destination.is_file():
            continue
        write_npy_subset(
            source_dir / name,
            destination,
            external,
            chunk_size=chunk_size,
        )


def subset_npz(
    source: Path,
    destination: Path,
    train_rows: np.ndarray,
    test_rows: np.ndarray | None = None,
) -> None:
    cached = np.load(source, allow_pickle=False)
    payload: dict[str, np.ndarray] = {}
    for key in cached.files:
        value = cached[key]
        if key.startswith("train_") and len(value.shape) > 0 and value.shape[0] == 4000:
            payload[key] = np.asarray(value[train_rows])
        elif (
            key.startswith("test_")
            and test_rows is not None
            and len(value.shape) > 0
            and value.shape[0] == 500
        ):
            raise RuntimeError(
                f"cannot map original test-only cache into E183 train-point queries: {key}"
            )
        else:
            payload[key] = np.asarray(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def subset_train_test_npz(
    source: Path,
    destination: Path,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
) -> None:
    cached = np.load(source, allow_pickle=False)
    payload: dict[str, np.ndarray] = {}
    for key in cached.files:
        value = cached[key]
        if key.startswith("train_") and len(value.shape) > 0 and value.shape[0] == 4000:
            payload[key] = np.asarray(value[train_rows])
        elif key.startswith("test_") and len(value.shape) > 0 and value.shape[0] == 500:
            train_key = "train_" + key.removeprefix("test_")
            if train_key not in cached:
                raise RuntimeError(f"missing {train_key} needed to construct {key}")
            payload[key] = np.asarray(cached[train_key][test_rows])
        else:
            payload[key] = np.asarray(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def build_internal_folds(
    root: Path,
    external_ids: np.ndarray,
    original_site: np.ndarray,
) -> tuple[Path, np.ndarray]:
    original_fold = np.load(GLOBAL_FOLD_IDS, allow_pickle=False).astype(np.int8)
    local_fold = original_fold[external_ids]
    local_site = original_site[external_ids]
    payload: dict[str, np.ndarray] = {}
    all_local = np.arange(len(external_ids), dtype=np.int64)
    rng = np.random.default_rng(260_811_183)
    selected_all: list[np.ndarray] = []
    for fold in range(5):
        selected_parts = []
        for site_value in (0, 1):
            candidates = np.flatnonzero((local_fold == fold) & (local_site == site_value))
            if len(candidates) < 50:
                raise RuntimeError(
                    f"family fold/site has only {len(candidates)} rows: fold={fold} site={site_value}"
                )
            order = rng.permutation(candidates)
            selected_parts.append(np.sort(order[:50]))
        query = np.sort(np.concatenate(selected_parts)).astype(np.int64)
        fit = all_local[~np.isin(all_local, query)]
        payload[f"fold{fold}_query"] = query.astype(np.int32)
        payload[f"fold{fold}_fit"] = fit.astype(np.int32)
        payload[f"fold{fold}_ignored"] = np.empty(0, dtype=np.int32)
        selected_all.append(query)
    selected = np.concatenate(selected_all)
    if len(selected) != 500 or len(np.unique(selected)) != 500:
        raise RuntimeError("E183 internal OOF queries must be 500 distinct rows")
    path = root / "artifacts" / "round2" / "e183_internal_folds.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    fold_path = (
        root
        / "artifacts"
        / "round2"
        / "full_path_pdp_link_gate"
        / "train_fold_id.npy"
    )
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(fold_path, local_fold.astype(np.int8))
    return path, local_fold


def prepare_workspace(family: str, force: bool) -> dict[str, object]:
    ensure_identity_gate()
    root = family_root(family)
    state_path = root / "e183_workspace_state.json"
    started = time.perf_counter()
    union, holes, metadata = load_family(family)
    all_ids = np.arange(4000, dtype=np.int64)
    external = all_ids[~np.isin(all_ids, union)]
    if state_path.is_file() and not force:
        ensure_official_target_caches(root, external)
        return json.loads(state_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    (root / ".hw-project").write_text("e183 strict isolated workspace\n", encoding="utf-8")
    dataset = root / "datasets" / "round2"
    dataset.mkdir(parents=True, exist_ok=True)

    _, source_positions, _, original_site, _ = load_positions()
    source_setup = json.loads(
        (PATHS.round2 / "Round2_Setup.json").read_text(encoding="utf-8")
    )
    hidden_sites = np.unique(original_site[union])
    expected_site = int(family[1])
    if hidden_sites.tolist() != [expected_site]:
        raise RuntimeError(
            f"{family} hidden union has sites {hidden_sites.tolist()}, "
            f"expected only site {expected_site}"
        )
    np.save(dataset / "Round2_Train_Pos.npy", source_positions[external])
    np.save(dataset / "Round2_Test_Pos.npy", source_positions[union])
    write_npy_subset(
        PATHS.round2 / "Round2_Train_Channel.npy",
        dataset / "Round2_Train_Channel.npy",
        external,
        chunk_size=8,
    )
    source_setup["P_Train"] = int(len(external))
    source_setup["P_Test"] = int(len(union))
    (dataset / "Round2_Setup.json").write_text(
        json.dumps(source_setup, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(PATHS.round2 / "Round2_Map.ply", dataset / "Round2_Map.ply")

    mapping_dir = root / "artifacts" / "round2" / "e183_mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(holes) + 1, dtype=np.int64)
    for index, hole in enumerate(holes):
        offsets[index + 1] = offsets[index] + len(hole)
    union_to_local = np.full(4000, -1, dtype=np.int32)
    union_to_local[union] = np.arange(len(union), dtype=np.int32)
    np.savez_compressed(
        mapping_dir / "family_mapping.npz",
        external_global_ids=external.astype(np.int32),
        test_global_ids=union.astype(np.int32),
        hole_global_ids_flat=np.concatenate(holes).astype(np.int32),
        hole_offsets=offsets,
        hole_test_rows_flat=union_to_local[np.concatenate(holes)].astype(np.int32),
    )

    fold_cache, local_fold = build_internal_folds(root, external, original_site)

    component_source = np.load(
        PATHS.artifacts
        / "round2"
        / "v4_path_component_transport"
        / "train_components_k24_os2_ref2.npz",
        allow_pickle=False,
    )
    component_target = (
        root
        / "artifacts"
        / "round2"
        / "v4_path_component_transport"
        / "train_components_k24_os2_ref2.npz"
    )
    component_target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        component_target,
        coordinates=component_source["coordinates"][external],
        coefficients=component_source["coefficients"][external],
        residual_fraction=component_source["residual_fraction"][external],
        point_power=component_source["point_power"][external],
        components=component_source["components"],
        oversample=component_source["oversample"],
        refinement_sweeps=component_source["refinement_sweeps"],
    )

    subset_train_test_npz(
        PATHS.artifacts
        / "round2"
        / "regime_lifecycle_support"
        / "geodesic_corridor_features.npz",
        root
        / "artifacts"
        / "round2"
        / "regime_lifecycle_support"
        / "geodesic_corridor_features.npz",
        external,
        union,
    )
    subset_train_test_npz(
        PATHS.artifacts
        / "round2"
        / "route_topology"
        / "round2_route_topology_features.npz",
        root
        / "artifacts"
        / "round2"
        / "route_topology"
        / "round2_route_topology_features.npz",
        external,
        union,
    )
    subset_train_test_npz(
        PATHS.artifacts
        / "round2"
        / "offline_sionna_geometry_student"
        / "official_map_geometry_rasters.npz",
        root
        / "artifacts"
        / "round2"
        / "offline_sionna_geometry_student"
        / "official_map_geometry_rasters.npz",
        external,
        union,
    )

    grouped_dir = root / "artifacts" / "round2" / "v3_anchored_neural_rbf"
    write_npy_subset(
        PATHS.artifacts
        / "round2"
        / "v3_anchored_neural_rbf"
        / "grouped_pas_amplitude_f16.npy",
        grouped_dir / "grouped_pas_amplitude_f16.npy",
        external,
        chunk_size=64,
    )
    write_npy_subset(
        PATHS.artifacts
        / "round2"
        / "v3_anchored_neural_rbf"
        / "grouped_pdp_amplitude_f16.npy",
        grouped_dir / "grouped_pdp_amplitude_f16.npy",
        external,
        chunk_size=64,
    )
    denoised_dir = root / "artifacts" / "round2" / "sparse_denoised_v7_carrier"
    write_npy_subset(
        PATHS.artifacts
        / "round2"
        / "sparse_denoised_v7_carrier"
        / "official_train_channel_k24_denoised.npy",
        denoised_dir / "official_train_channel_k24_denoised.npy",
        external,
        chunk_size=8,
    )
    write_npy_subset(
        PATHS.artifacts
        / "round2"
        / "sparse_denoised_v7_carrier"
        / "official_train_channel_k24_power.npy",
        denoised_dir / "official_train_channel_k24_power.npy",
        external,
        chunk_size=256,
    )
    ensure_official_target_caches(root, external)

    state = {
        "experiment": VERSION,
        "phase": "workspace_prepared",
        **metadata,
        "workspace": (
            str(root.relative_to(SOURCE_ROOT))
            if root.is_relative_to(SOURCE_ROOT)
            else f"runtime_scratch/{family}"
        ),
        "external_train_count": int(len(external)),
        "hidden_union_count": int(len(union)),
        "hidden_site": expected_site,
        "internal_oof_count": 500,
        "internal_fold_counts": np.bincount(local_fold, minlength=5).astype(int).tolist(),
        "fold_cache": str(fold_cache.relative_to(root)),
        "sionna_used": False,
        "v8_v12_used": False,
        "seconds": float(time.perf_counter() - started),
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return state


def pipeline_steps(device: str) -> list[tuple[str, list[str]]]:
    python = sys.executable
    code = SOURCE_ROOT / "code" / "round2"
    return [
        ("foundation", [python, "-u", str(code / "run_foundation_benchmark.py"), "--workers", "4", "--device", device, "--gpu-batch-size", "8"]),
        ("pre_sionna", [python, "-u", str(code / "run_pre_sionna_pipeline.py"), "--batch-size", "128"]),
        (
            "rbf",
            [python, "-u", str(code / "run_rbf_spectral_field.py"), "--device", device, "--batch-size", "16"],
        ),
        (
            "v3_folds",
            [python, "-u", str(code / "run_v3_anchored_neural_rbf.py"), "--device", device, "--folds", "0,1,2,3,4", "--epochs", "6"],
        ),
        (
            "v3_targets",
            [python, "-u", str(code / "evaluate_v3_anchored_full_targets.py"), "--device", device],
        ),
        (
            "v3_projection",
            [python, "-u", str(code / "evaluate_v3_anchored_projection.py"), "--device", device, "--batch-size", "16"],
        ),
        (
            "v4_calibration",
            [python, "-u", str(code / "analyze_v4_predictability_calibrated_groups.py"), "--device", device, "--batch-size", "16"],
        ),
        (
            "v3_test",
            [python, "-u", str(code / "build_v3_anchored_neural_rbf.py"), "--device", device, "--batch-size", "16"],
        ),
        (
            "v4",
            [python, "-u", str(code / "run_v4_calibrated_exact_projection.py"), "--device", device, "--batch-size", "16"],
        ),
        (
            "v6_prepare",
            [python, "-u", str(code / "run_v6_regime_component_lifecycle.py"), "--phase", "prepare", "--device", device, "--batch-size", "16"],
        ),
        (
            "v6_evaluate",
            [python, "-u", str(code / "run_v6_regime_component_lifecycle.py"), "--phase", "evaluate", "--device", device, "--batch-size", "16"],
        ),
        (
            "v6_test",
            [python, "-u", str(code / "run_v6_regime_component_lifecycle.py"), "--phase", "test", "--device", device, "--batch-size", "16"],
        ),
        (
            "carrier_bank",
            [
                python,
                "-u",
                str(code / "build_v7_complex_carrier_bank.py"),
                "--phase",
                "all",
                "--carrier-pairs",
                "500",
                "--workers",
                "4",
                "--device",
                device,
                "--gpu-batch-size",
                "32",
            ],
        ),
        (
            "carrier_stack",
            [python, "-u", str(code / "analyze_v7_complex_carrier_stacking.py"), "--phase", "all", "--device", device],
        ),
        (
            "v7_projection",
            [python, "-u", str(code / "run_v7_complex_carrier_exact_projection.py"), "--phase", "all", "--device", device, "--batch-size", "16"],
        ),
        (
            "v7_lifecycle",
            [python, "-u", str(code / "analyze_v7_stacked_lifecycle_overlay.py"), "--device", device, "--batch-size", "16"],
        ),
        (
            "e142",
            [
                python,
                "-u",
                str(code / "run_e142_denoised_v7_carrier.py"),
                "--phase",
                "all",
                "--device",
                device,
                "--batch-size",
                "16",
                "--workers",
                "4",
                "--gpu-batch-size",
                "32",
            ],
        ),
        (
            "e120_prepare",
            [python, "-u", str(code / "run_e120_fivefold_unique_path_v12.py"), "--stage", "prepare", "--device", device, "--batch-size", "20"],
        ),
        (
            "e120_train",
            [python, "-u", str(code / "run_e120_fivefold_unique_path_v12.py"), "--stage", "train", "--device", device, "--batch-size", "20", "--core-epochs", "24", "--scene-epochs", "16"],
        ),
        (
            "e120_endpoint",
            [python, "-u", str(code / "generate_e120_unique_path_v12_test_channel.py"), "--device", device, "--batch-size", "20", "--endpoint-only"],
        ),
        (
            "family_actions_11",
            [python, "-u", str(code / "build_e183_clean_family_actions.py"), "--device", device, "--batch-size", "16", "--gram-batch-size", "8"],
        ),
        (
            "e160_clean_carriers",
            [python, "-u", str(code / "build_e160_full4000_v7_supervision.py"), "--phase", "clean_carriers", "--device", device, "--batch-size", "8"],
        ),
        (
            "e160_clean_core",
            [python, "-u", str(code / "build_e160_full4000_v7_supervision.py"), "--phase", "clean_core", "--device", device, "--batch-size", "8", "--v3-batch-size", "128", "--rbf-batch-size", "256"],
        ),
        (
            "e161_e162_actions",
            [python, "-u", str(code / "build_e183_e161_e162_actions.py"), "--phase", "all", "--device", device, "--batch-size", "16"],
        ),
    ]


def portable_command(command: list[str]) -> list[str]:
    output: list[str] = []
    for index, value in enumerate(command):
        path = Path(value)
        if index == 0 and path == Path(sys.executable):
            output.append("python")
        elif path.is_absolute() and path.is_relative_to(SOURCE_ROOT):
            output.append(str(path.relative_to(SOURCE_ROOT)))
        else:
            output.append(value)
    return output


def freeze_v7_stack_selection(root: Path) -> dict[str, object]:
    selection_path = (
        root
        / "artifacts"
        / "round2"
        / "complex_carrier_stacking"
        / "complex_stack_selection.json"
    )
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing E183 carrier selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    visible_best = selection.get("e183_visible_posterior_best")
    if visible_best is None:
        visible_best = selection.get("posterior_best")
    selection["e183_visible_posterior_best"] = visible_best
    selection["posterior_best"] = {
        "config": dict(FROZEN_V7_STACK_CONFIG),
        "selection_mode": "e183_frozen_canonical_v7",
    }
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = {
        "experiment": VERSION,
        "status": "frozen_v7_stack_config_applied",
        "visible_posterior_best": visible_best,
        "frozen_config": dict(FROZEN_V7_STACK_CONFIG),
        "selection": str(selection_path.relative_to(root)),
    }
    audit_path = root / "e183_frozen_v7_stack_selection.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def run_pipeline(
    family: str,
    device: str,
    physical_gpu: str,
    start_step: int,
    stop_step: int,
) -> dict[str, object]:
    root = family_root(family)
    state_path = root / "e183_workspace_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"prepare family first: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    fold_cache = root / str(state["fold_cache"])
    env = os.environ.copy()
    env["HW_PROJECT_ROOT"] = str(root)
    env["HW_SPATIAL_FOLD_CACHE"] = str(fold_cache)
    env["CUDA_VISIBLE_DEVICES"] = physical_gpu
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(SOURCE_ROOT / "code"),
            str(SOURCE_ROOT / "code" / "common"),
            str(SOURCE_ROOT / "code" / "round1"),
            str(SOURCE_ROOT / "code" / "round2"),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    steps = pipeline_steps(device)
    final_step = len(steps) - 1 if stop_step < 0 else min(stop_step, len(steps) - 1)
    log_dir = root / "analysis" / "results" / "round2_e183_strict_action_labels" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    reusable_outputs = {
        "e160_clean_carriers": root
        / "artifacts"
        / "round2"
        / "e160_full4000_v7_supervision"
        / "full4000_v7_raw_stack.npy",
        "e160_clean_core": root
        / "artifacts"
        / "round2"
        / "e160_full4000_v7_supervision"
        / "full4000_v7_core.npy",
    }

    def reusable_output_ready(name: str, path: Path) -> bool:
        if not path.is_file():
            return False
        value = np.load(path, mmap_mode="r")
        expected = int(state["external_train_count"])
        if value.shape != (expected, 256, 4, 192) or value.dtype != np.complex64:
            return False
        e160_dir = root / "artifacts" / "round2" / "e160_full4000_v7_supervision"
        weights = e160_dir / "full4000_v7_stack_weights.npz"
        report_path = (
            root
            / "analysis"
            / "results"
            / "round2_e160_full4000_v7_supervision"
            / "full4000_v7_supervision.json"
        )
        if not weights.is_file() or not report_path.is_file():
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = str(report.get("status", ""))
        if name == "e160_clean_carriers":
            return status in {
                "full4000_raw_stack_complete",
                "full4000_v7_core_complete",
            }
        return status == "full4000_v7_core_complete"

    for index, (name, command) in enumerate(steps):
        if index < start_step or index > final_step:
            continue
        if index == 14:
            freeze_v7_stack_selection(root)
        reusable = reusable_outputs.get(name)
        if reusable is not None and reusable_output_ready(name, reusable):
            records.append(
                {
                    "index": index,
                    "name": name,
                    "command": portable_command(command),
                    "return_code": 0,
                    "seconds": 0.0,
                    "log": None,
                    "reused": True,
                }
            )
            print(f"E183 {family} step {index}/{len(steps) - 1}: {name} reused", flush=True)
            continue
        step_started = time.perf_counter()
        log_path = log_dir / f"{index:02d}_{name}.log"
        print(f"E183 {family} step {index}/{len(steps) - 1}: {name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=SOURCE_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
        elapsed = time.perf_counter() - step_started
        records.append(
            {
                "index": index,
                "name": name,
                "command": portable_command(command),
                "return_code": return_code,
                "seconds": elapsed,
                "log": str(log_path.relative_to(root)),
            }
        )
        if return_code != 0:
            raise RuntimeError(f"E183 {family} pipeline failed at {index}:{name}")
    report_path = root / "e183_pipeline_report.json"
    previous_steps: list[dict[str, object]] = []
    previous_seconds = 0.0
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_steps = list(previous.get("steps", []))
        previous_seconds = float(previous.get("seconds", 0.0))
    merged = {int(row["index"]): row for row in previous_steps}
    merged.update({int(row["index"]): row for row in records})
    invocation_seconds = float(time.perf_counter() - started)
    report = {
        "experiment": VERSION,
        "phase": "family_pipeline",
        "family": family,
        "last_start_step": start_step,
        "last_stop_step": final_step,
        "steps": [merged[index] for index in sorted(merged)],
        "seconds": previous_seconds + invocation_seconds,
        "last_invocation_seconds": invocation_seconds,
        "physical_gpu": physical_gpu,
        "device": device,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def status(family: str) -> dict[str, object]:
    root = family_root(family)
    paths = {
        "workspace_state": root / "e183_workspace_state.json",
        "v4": root
        / "artifacts"
        / "round2"
        / "v4_calibrated_exact_projection"
        / "Round2_V4_Calibrated_Exact_Projection_Test_Channel.npy",
        "v7": root
        / "artifacts"
        / "round2"
        / "complex_carrier_lifecycle_overlay"
        / "Round2_V7_Stacked_Lifecycle_Test_Channel.npy",
        "carrier_bank": root
        / "artifacts"
        / "round2"
        / "complex_carrier_bank"
        / "test_complex_carrier_bank.npy",
        "e142_report": root
        / "analysis"
        / "results"
        / "round2_sparse_denoised_v7_carrier"
        / "sparse_denoised_v7_carrier.json",
        "e120_report": root
        / "analysis"
        / "results"
        / "round2_fivefold_unique_path_v12"
        / "fivefold_unique_path_v12.json",
        "e161_action": root
        / "artifacts"
        / "round2"
        / "e183_family_actions"
        / "E161_knn.npy",
        "e162_action": root
        / "artifacts"
        / "round2"
        / "e183_family_actions"
        / "E162_knn.npy",
    }
    return {
        "family": family,
        "workspace": str(root),
        "files": {
            name: {
                "exists": path.is_file(),
                "bytes": int(path.stat().st_size) if path.is_file() else 0,
            }
            for name, path in paths.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        payload = prepare_workspace(args.family, args.force)
    elif args.phase == "pipeline":
        payload = run_pipeline(
            args.family,
            args.device,
            args.physical_gpu,
            args.start_step,
            args.stop_step,
        )
    else:
        payload = status(args.family)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
