#!/usr/bin/env python3
"""Build strict family-isolated E161/E162 Clean actions for E183."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_e160_v7_correction_capacity as e160
import analyze_e162_v7_relative_peak_relation as e162
import analyze_v7_stacked_lifecycle_overlay as lifecycle
import train_e161_full4000_v7_correction_student as e161
from data import load_positions
from project_paths import PATHS
from run_v6_regime_component_lifecycle import (
    REGIME_CACHE,
    TARGET_PATHS,
    _candidate_from_name,
    apply_candidate,
)
from run_v7_complex_carrier_exact_projection import TEST_PROJECTED
from train_full_path_pdp_link_gate import FOLD_IDS


VERSION = "Round2-E183-strict-E161-E162-actions"
E160_DIR = PATHS.artifacts / "round2" / "e160_full4000_v7_supervision"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e183_family_actions"
RELATIVE_DIR = PATHS.artifacts / "round2" / "e183_e162_relative_cache"
REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e183_strict_action_labels"
    / "e161_e162_action_generation.json"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
FULL_CORE = E160_DIR / "full4000_v7_core.npy"
E161_PRE = ARTIFACT_DIR / "E161_knn_pre_lifecycle.npy"
E161_OUTPUT = ARTIFACT_DIR / "E161_knn.npy"
E162_PRE = ARTIFACT_DIR / "E162_knn_pre_lifecycle.npy"
E162_OUTPUT = ARTIFACT_DIR / "E162_knn.npy"
MAPPING = PATHS.artifacts / "round2" / "e183_mapping" / "family_mapping.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("caches", "e161", "e162", "all"), default="all"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def valid_channel(path: Path, count: int) -> bool:
    if not path.is_file():
        return False
    value = np.load(path, mmap_mode="r")
    if value.shape != (count, 256, 4, 192) or value.dtype != np.complex64:
        return False
    for start in range(0, count, 8):
        block = np.asarray(value[start : start + 8])
        if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
            return False
    return True


def verify_isolation(train_count: int, query_count: int) -> dict[str, object]:
    mapping = np.load(MAPPING, allow_pickle=False)
    external = mapping["external_global_ids"].astype(np.int64)
    hidden = mapping["test_global_ids"].astype(np.int64)
    overlap = np.intersect1d(external, hidden)
    if len(external) != train_count or len(hidden) != query_count or len(overlap):
        raise RuntimeError("E183 family mapping does not match the isolated dataset")
    return {
        "visible_count": int(len(external)),
        "hidden_count": int(len(hidden)),
        "visible_hidden_overlap": int(len(overlap)),
    }


@torch.inference_mode()
def grouped_query_endpoints(
    channel: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pas = np.empty((len(channel), *e161.PAS_SHAPE), dtype=np.float32)
    pdp = np.empty((len(channel), *e161.PDP_SHAPE), dtype=np.float32)
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        block = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        local_pas, local_pdp = e160.grouped_endpoints(block)
        pas[start:end] = local_pas.cpu().numpy()
        pdp[start:end] = local_pdp.cpu().numpy()
    return pas, pdp


def build_supervision_caches(
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, object]:
    started = time.time()
    e160.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    core = np.load(FULL_CORE, mmap_mode="r")
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    if core.shape != truth.shape:
        raise RuntimeError(f"visible core/truth mismatch: {core.shape} != {truth.shape}")
    core_pas, core_pdp = e160.build_endpoint_cache(
        core, "core", device, batch_size, force
    )
    truth_pas, truth_pdp = e160.build_endpoint_cache(
        truth, "truth", device, batch_size, force
    )
    residual_pas, residual_pdp = e160.build_residual_cache(
        core_pas, core_pdp, truth_pas, truth_pdp, force
    )
    return {
        "count": int(len(core)),
        "core_pas_shape": list(core_pas.shape),
        "core_pdp_shape": list(core_pdp.shape),
        "residual_pas_shape": list(residual_pas.shape),
        "residual_pdp_shape": list(residual_pdp.shape),
        "seconds": float(time.time() - started),
    }


@torch.inference_mode()
def apply_test_lifecycle(
    source: np.ndarray,
    output_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    selection = json.loads(lifecycle.SELECTION.read_text(encoding="utf-8"))
    winner = selection.get("winner")
    if not isinstance(winner, dict):
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.complex64, shape=source.shape
        )
        for start in range(0, len(source), max(batch_size, 64)):
            end = min(start + max(batch_size, 64), len(source))
            output[start:end] = np.asarray(source[start:end], dtype=np.complex64)
        output.flush()
        return np.load(output_path, mmap_mode="r")
    policy = {
        (int(row["site"]), int(row["regime_index"])): str(row["candidate"])
        for row in winner["final_policy"]
    }
    target = np.load(TARGET_PATHS[str(winner["stage"])][1], mmap_mode="r")
    regimes = np.load(REGIME_CACHE, allow_pickle=False)["test_regime"].astype(np.int8)
    _, _, _, _, site = load_positions()
    if len(source) != len(target) or len(source) != len(regimes):
        raise RuntimeError("lifecycle query arrays have inconsistent counts")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        block = torch.as_tensor(
            np.array(source[start:end], dtype=np.complex64, copy=True), device=device
        )
        current = e160.current_ad(block)
        block_site = site[start:end]
        block_regime = regimes[start:end]
        target_tensor = torch.as_tensor(
            np.asarray(target[start:end], dtype=np.float32), device=device
        )
        for site_value in np.unique(block_site):
            for regime_value in np.unique(block_regime):
                rows = np.flatnonzero(
                    (block_site == site_value) & (block_regime == regime_value)
                )
                if len(rows) == 0:
                    continue
                name = policy[(int(site_value), int(regime_value))]
                if name == "identity":
                    continue
                block[rows] = apply_candidate(
                    block[rows],
                    current[rows],
                    target_tensor[rows],
                    block_regime[rows],
                    block_site[rows],
                    _candidate_from_name(name),
                )
        output[start:end] = block.cpu().numpy().astype(np.complex64)
    output.flush()
    return np.load(output_path, mmap_mode="r")


def query_neighbors(
    query_positions: np.ndarray,
    donor_rows: np.ndarray,
    donor_positions: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = min(count, len(donor_rows))
    distance, local = cKDTree(donor_positions[donor_rows, :2]).query(
        query_positions[:, :2], k=keep, workers=-1
    )
    if keep == 1:
        distance = distance[:, None]
        local = local[:, None]
    selected = donor_rows[np.asarray(local, dtype=np.int64)]
    weight = 1.0 / np.square(np.asarray(distance, dtype=np.float32) + 1.0)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return selected, weight.astype(np.float32)


def build_e161(
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, object]:
    started = time.time()
    _, train_positions, test_positions, train_site, test_site = load_positions()
    if valid_channel(E161_OUTPUT, len(test_positions)) and not force:
        return {
            "path": PATHS.relative(E161_OUTPUT),
            "shape": list(np.load(E161_OUTPUT, mmap_mode="r").shape),
            "reused": True,
            "seconds": float(time.time() - started),
        }
    core_pas = np.load(e160.FULL_CORE_PAS, mmap_mode="r")
    core_pdp = np.load(e160.FULL_CORE_PDP, mmap_mode="r")
    residual_pas = np.load(e160.FULL_RESIDUAL_PAS, mmap_mode="r")
    residual_pdp = np.load(e160.FULL_RESIDUAL_PDP, mmap_mode="r")
    query_core = np.load(TEST_PROJECTED, mmap_mode="r")
    query_pas, query_pdp = grouped_query_endpoints(query_core, device, batch_size)
    target_pas = np.array(query_pas, dtype=np.float32, copy=True)
    target_pdp = np.array(query_pdp, dtype=np.float32, copy=True)
    records = []
    config = e161.Config(metric_batch_size=batch_size)
    for site_value in np.unique(test_site):
        fit_rows = np.flatnonzero(train_site == site_value)
        query_rows = np.flatnonzero(test_site == site_value)
        state = e161.fit_state(
            fit_rows,
            residual_pas,
            residual_pdp,
            core_pas,
            core_pdp,
            device,
            config.seed + 900_000 + int(site_value) * 31,
        )
        fit_coefficient = e161.coefficients(
            fit_rows, residual_pas, residual_pdp, state
        )
        selected, weight = query_neighbors(
            test_positions[query_rows], fit_rows, train_positions, config.neighbors
        )
        local = np.searchsorted(fit_rows, selected)
        if not np.array_equal(fit_rows[local], selected):
            raise RuntimeError("E161 donor coefficient index mismatch")
        prediction = np.einsum(
            "qk,qkd->qd", weight, fit_coefficient[local], optimize=True
        ).astype(np.float32)
        base_pas_root = torch.as_tensor(
            np.sqrt(np.maximum(query_pas[query_rows], 0.0)), device=device
        )
        base_pdp_root = torch.as_tensor(
            np.sqrt(np.maximum(query_pdp[query_rows], 0.0)), device=device
        )
        local_pas, local_pdp = e161.reconstruct_grouped_torch(
            base_pas_root,
            base_pdp_root,
            torch.as_tensor(prediction, device=device),
            state,
            device,
        )
        target_pas[query_rows] = local_pas.cpu().numpy()
        target_pdp[query_rows] = local_pdp.cpu().numpy()
        records.append(
            {
                "site": int(site_value),
                "fit_count": int(len(fit_rows)),
                "query_count": int(len(query_rows)),
                "pas_basis_explained": float(state.pas_explained),
                "pdp_basis_explained": float(state.pdp_explained),
            }
        )
    pre = e161.project_canonical_targets(
        query_core,
        target_pas,
        target_pdp,
        test_site,
        E161_PRE,
        config,
        device,
        np.ones(len(test_positions), dtype=bool),
    )
    output = apply_test_lifecycle(pre, E161_OUTPUT, device, batch_size)
    if not valid_channel(E161_OUTPUT, len(test_positions)):
        raise RuntimeError("invalid strict E161 action")
    return {
        "path": PATHS.relative(E161_OUTPUT),
        "pre_lifecycle": PATHS.relative(E161_PRE),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": True,
        "reused": False,
        "sites": records,
        "seconds": float(time.time() - started),
    }


def relative_cache_paths() -> dict[str, Path]:
    return {
        "pas": RELATIVE_DIR / "visible_aligned_pas_residual_f16.npy",
        "pdp": RELATIVE_DIR / "visible_aligned_pdp_residual_f16.npy",
    }


@torch.inference_mode()
def build_relative_caches(
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, np.ndarray]:
    core_pas = np.load(e160.FULL_CORE_PAS, mmap_mode="r")
    core_pdp = np.load(e160.FULL_CORE_PDP, mmap_mode="r")
    truth_pas = np.load(e160.FULL_TRUTH_PAS, mmap_mode="r")
    truth_pdp = np.load(e160.FULL_TRUTH_PDP, mmap_mode="r")
    count = len(core_pas)
    paths = relative_cache_paths()
    shapes = {
        "pas": (count, e162.PAS_GROUPS, e162.PAS_CELLS),
        "pdp": (count, e162.PDP_GROUPS, e162.PDP_CELLS),
    }
    if not force and all(
        path.is_file()
        and np.load(path, mmap_mode="r").shape == shapes[key]
        and np.load(path, mmap_mode="r").dtype == np.float16
        for key, path in paths.items()
    ):
        return {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
    RELATIVE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        key: np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shapes[key])
        for key, path in paths.items()
    }
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        base_pas = torch.as_tensor(
            np.asarray(core_pas[start:end], dtype=np.float32), device=device
        )
        target_pas = torch.as_tensor(
            np.asarray(truth_pas[start:end], dtype=np.float32), device=device
        )
        aligned_base_pas, aligned_truth_pas, _ = e162.align_pas(
            base_pas, target_pas
        )
        outputs["pas"][start:end] = (
            aligned_truth_pas.clamp_min(0.0).sqrt()
            - aligned_base_pas.clamp_min(0.0).sqrt()
        ).reshape(end - start, e162.PAS_GROUPS, e162.PAS_CELLS).cpu().numpy().astype(np.float16)
        base_pdp = torch.as_tensor(
            np.asarray(core_pdp[start:end], dtype=np.float32), device=device
        )
        target_pdp = torch.as_tensor(
            np.asarray(truth_pdp[start:end], dtype=np.float32), device=device
        )
        aligned_base_pdp, aligned_truth_pdp, _ = e162.align_pdp(
            base_pdp, target_pdp
        )
        outputs["pdp"][start:end] = (
            aligned_truth_pdp.clamp_min(0.0).sqrt()
            - aligned_base_pdp.clamp_min(0.0).sqrt()
        ).cpu().numpy().astype(np.float16)
        for output in outputs.values():
            output.flush()
    return {key: np.load(path, mmap_mode="r") for key, path in paths.items()}


def relation_mask(
    aligned: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    folds: np.ndarray,
    site_value: int,
    config: e162.Config,
    domain_offset: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    selected = np.ones(aligned.shape[1:], dtype=bool)
    records = []
    for fold in range(5):
        query_rows = np.flatnonzero((folds == fold) & (site == site_value))
        donor_rows = np.flatnonzero((folds != fold) & (site == site_value))
        neighbor_ids, weights = e162.neighbor_table(
            query_rows, donor_rows, positions, config.neighbors
        )
        stats = e162.relation_statistics(
            aligned,
            aligned,
            query_rows,
            donor_rows,
            neighbor_ids,
            weights,
            config.group_chunk,
            config.seed + int(site_value) * 100 + fold * 7 + domain_offset,
        )
        fold_pass = (
            np.asarray(stats["sse_real"]) < np.asarray(stats["sse_zero"])
        ) & (
            np.asarray(stats["sse_real"]) < np.asarray(stats["sse_shuffle"])
        )
        selected &= fold_pass
        records.append(
            {
                "fold": fold,
                "query_count": int(len(query_rows)),
                "donor_count": int(len(donor_rows)),
                "passing_cells": int(fold_pass.sum()),
            }
        )
    return selected, records


def predicted_relative_action(
    aligned: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    masks: dict[int, np.ndarray],
    neighbors: int,
) -> np.ndarray:
    output = np.zeros((len(test_positions), *aligned.shape[1:]), dtype=np.float32)
    for site_value in np.unique(test_site):
        fit_rows = np.flatnonzero(train_site == site_value)
        query_rows = np.flatnonzero(test_site == site_value)
        selected, weight = query_neighbors(
            test_positions[query_rows], fit_rows, train_positions, neighbors
        )
        for start in range(0, aligned.shape[1], 4):
            end = min(start + 4, aligned.shape[1])
            donor = np.asarray(aligned[selected, start:end], dtype=np.float32)
            output[query_rows, start:end] = np.einsum(
                "qk,qkgc->qgc", weight, donor, optimize=True
            ) * masks[int(site_value)][start:end][None]
    return output


@torch.inference_mode()
def query_relative_state(
    base_pas: np.ndarray,
    base_pdp: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    base_pas_tensor = torch.as_tensor(base_pas, device=device)
    aligned_pas, _, pas_centers = e162.align_pas(base_pas_tensor, base_pas_tensor)
    base_pdp_tensor = torch.as_tensor(base_pdp, device=device)
    aligned_pdp, _, pdp_centers = e162.align_pdp(base_pdp_tensor, base_pdp_tensor)
    return {
        "pas_base_root": aligned_pas.clamp_min(0.0).sqrt().reshape(
            len(base_pas), e162.PAS_GROUPS, e162.PAS_CELLS
        ).cpu().numpy(),
        "pas_centers": pas_centers.cpu().numpy(),
        "pdp_base_root": aligned_pdp.clamp_min(0.0).sqrt().cpu().numpy(),
        "pdp_centers": pdp_centers.cpu().numpy(),
    }


def build_e162(
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, object]:
    started = time.time()
    _, train_positions, test_positions, train_site, test_site = load_positions()
    if valid_channel(E162_OUTPUT, len(test_positions)) and not force:
        return {
            "path": PATHS.relative(E162_OUTPUT),
            "shape": list(np.load(E162_OUTPUT, mmap_mode="r").shape),
            "reused": True,
            "seconds": float(time.time() - started),
        }
    aligned = build_relative_caches(device, max(batch_size, 32), force)
    folds = np.load(FOLD_IDS, allow_pickle=False).astype(np.int8)
    config = e162.Config(metric_batch_size=batch_size)
    pas_masks: dict[int, np.ndarray] = {}
    pdp_masks: dict[int, np.ndarray] = {}
    records = []
    for site_value in np.unique(test_site):
        pas_mask, pas_records = relation_mask(
            aligned["pas"],
            train_positions,
            train_site,
            folds,
            int(site_value),
            config,
            0,
        )
        pdp_mask, pdp_records = relation_mask(
            aligned["pdp"],
            train_positions,
            train_site,
            folds,
            int(site_value),
            config,
            1,
        )
        pas_masks[int(site_value)] = pas_mask
        pdp_masks[int(site_value)] = pdp_mask
        records.append(
            {
                "site": int(site_value),
                "pas_selected": int(pas_mask.sum()),
                "pas_total": int(pas_mask.size),
                "pdp_selected": int(pdp_mask.sum()),
                "pdp_total": int(pdp_mask.size),
                "pas_folds": pas_records,
                "pdp_folds": pdp_records,
            }
        )
    pas_action = predicted_relative_action(
        aligned["pas"],
        train_positions,
        test_positions,
        train_site,
        test_site,
        pas_masks,
        config.neighbors,
    )
    pdp_action = predicted_relative_action(
        aligned["pdp"],
        train_positions,
        test_positions,
        train_site,
        test_site,
        pdp_masks,
        config.neighbors,
    )
    query_core = np.load(TEST_PROJECTED, mmap_mode="r")
    base_pas, base_pdp = grouped_query_endpoints(query_core, device, batch_size)
    state = query_relative_state(base_pas, base_pdp, device)
    grouped_pas, grouped_pdp = e162.render_grouped_targets(
        state, pas_action, pdp_action, device
    )
    pre = e162.project_channel(
        query_core,
        grouped_pas,
        grouped_pdp,
        test_site,
        E162_PRE,
        device,
        batch_size,
        config.cycles,
    )
    output = apply_test_lifecycle(pre, E162_OUTPUT, device, batch_size)
    if not valid_channel(E162_OUTPUT, len(test_positions)):
        raise RuntimeError("invalid strict E162 action")
    return {
        "path": PATHS.relative(E162_OUTPUT),
        "pre_lifecycle": PATHS.relative(E162_PRE),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": True,
        "reused": False,
        "sites": records,
        "seconds": float(time.time() - started),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    _, train_positions, test_positions, _, _ = load_positions()
    isolation = verify_isolation(len(train_positions), len(test_positions))
    if not FULL_CORE.is_file():
        raise FileNotFoundError(f"missing strict visible V7 core: {FULL_CORE}")
    if not TEST_PROJECTED.is_file():
        raise FileNotFoundError(f"missing strict hidden-query V7 core: {TEST_PROJECTED}")
    payload: dict[str, object] = {
        "experiment": VERSION,
        "phase": args.phase,
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "hidden_truth_read": False,
        "isolation": isolation,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    if args.phase in ("caches", "all"):
        payload["supervision_caches"] = build_supervision_caches(
            device, args.batch_size, args.force
        )
    else:
        required = (
            e160.FULL_CORE_PAS,
            e160.FULL_CORE_PDP,
            e160.FULL_TRUTH_PAS,
            e160.FULL_TRUTH_PDP,
            e160.FULL_RESIDUAL_PAS,
            e160.FULL_RESIDUAL_PDP,
        )
        if not all(path.is_file() for path in required):
            raise FileNotFoundError("build E160 grouped supervision caches first")
    if args.phase in ("e161", "all"):
        payload["E161_knn"] = build_e161(device, args.batch_size, args.force)
    if args.phase in ("e162", "all"):
        payload["E162_knn"] = build_e162(device, args.batch_size, args.force)
    payload["status"] = "complete"
    payload["seconds"] = float(time.time() - started)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
