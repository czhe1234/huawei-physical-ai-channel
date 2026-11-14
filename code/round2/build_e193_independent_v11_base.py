#!/usr/bin/env python3
"""Rebuild the V11-equivalent OOF carrier from original Round 2 inputs."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

import analyze_joint_path_token_oracle as joint_paths
import analyze_round2_canonical_500_versions as canonical
import analyze_v7_complex_carrier_stacking as stacking
import analyze_v7_stacked_lifecycle_overlay as lifecycle_overlay
import build_v7_complex_carrier_bank as carrier_bank
import run_v6_regime_component_lifecycle as lifecycle
import run_v7_complex_carrier_exact_projection as v7_projection
from data import load_positions
from project_paths import PATHS
from run_foundation_benchmark import load_or_compute_power
from run_rbf_spectral_field import OOF_META
from run_v9_joint_sionna_migration import Candidate as V9Candidate
from run_v9_joint_sionna_migration import apply_candidate as apply_v9_candidate


VERSION = "Round2-E193-independent-v11-base"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e193_propagation_memory"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e193_propagation_memory"
REPORT = ANALYSIS_DIR / "independent_v11_base.json"

CARRIER_BANK = ARTIFACT_DIR / "oof_complex_carrier_bank.npy"
GRAM_CACHE = ARTIFACT_DIR / "oof_port_gram.npz"
STACK_CHANNEL = ARTIFACT_DIR / "oof_stacked_complex_channel.npy"
PROJECTED_CHANNEL = ARTIFACT_DIR / "oof_stacked_v4_projection.npy"
SCALED_CHANNEL = ARTIFACT_DIR / "oof_stacked_v4_projection_nested_scaled.npy"
CURRENT_AD = ARTIFACT_DIR / "oof_stacked_current_ad_f16.npy"
V7_CHANNEL = ARTIFACT_DIR / "oof_v7_channel.npy"
V9_CHANNEL = ARTIFACT_DIR / "oof_v9_channel.npy"
V9_PDP = ARTIFACT_DIR / "oof_v9_pdp_f16.npy"
V10_ALPHA = ARTIFACT_DIR / "oof_v10_alpha_f16.npy"
V10_CHANNEL = ARTIFACT_DIR / "oof_v10_channel.npy"
V11_GATE_FEATURES = ARTIFACT_DIR / "oof_v11_gate_features.npy"
V11_GATE = ARTIFACT_DIR / "oof_v11_gate_f16.npy"
V11_CHANNEL = ARTIFACT_DIR / "oof_v11_channel.npy"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
EXPECTED_SHAPE = (500, 256, 4, 192)
TINY = 1e-30

FORBIDDEN_CHANNELS = {
    PATHS.artifacts
    / "round2"
    / "canonical_500_version_replay"
    / "V11_Canonical_500_OOF_Channel.npy",
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_nested_oof_channel.npy",
    PATHS.artifacts
    / "round2"
    / "official_sionna_path_calibrator"
    / "fixed500_channel.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_channel(path: Path) -> np.memmap:
    value = np.load(path, mmap_mode="r")
    if value.shape != EXPECTED_SHAPE or value.dtype != np.complex64:
        raise ValueError(
            f"invalid E193 channel {PATHS.relative(path)}: {value.shape}/{value.dtype}"
        )
    for start in range(0, len(value), 8):
        block = np.asarray(value[start : start + 8])
        if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
            raise ValueError(f"non-finite E193 channel: {PATHS.relative(path)}")
    return value


def replace_partial(partial: Path, final: Path) -> None:
    if not partial.is_file():
        raise FileNotFoundError(partial)
    os.replace(partial, final)


def build_carriers(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    device: torch.device,
    gpu_batch_size: int,
    force: bool,
) -> np.memmap:
    configs = carrier_bank.carrier_configs()
    expected = (len(configs), *EXPECTED_SHAPE)
    if CARRIER_BANK.is_file() and not force:
        cached = np.load(CARRIER_BANK, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.complex64:
            return cached
    partial = CARRIER_BANK.with_name(CARRIER_BANK.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    original_output = carrier_bank.OOF_BANK
    carrier_bank.OOF_BANK = partial
    device_channel = None
    try:
        if device.type == "cuda":
            device_channel = torch.as_tensor(
                np.asarray(train_channel), dtype=torch.complex64, device=device
            )
        value, _ = carrier_bank.build_oof_bank(
            train_channel,
            train_positions,
            train_site,
            bs_positions,
            point_power,
            configs,
            carrier_pairs=500,
            workers=1,
            device_channel=device_channel,
            gpu_batch_size=gpu_batch_size,
        )
        if value.shape != expected or value.dtype != np.complex64:
            raise ValueError(f"invalid rebuilt carrier bank: {value.shape}/{value.dtype}")
        del value
    finally:
        carrier_bank.OOF_BANK = original_output
        del device_channel
        if device.type == "cuda":
            torch.cuda.empty_cache()
    replace_partial(partial, CARRIER_BANK)
    return np.load(CARRIER_BANK, mmap_mode="r")


def build_gram(device: torch.device, force: bool) -> dict[str, np.ndarray]:
    if force:
        GRAM_CACHE.unlink(missing_ok=True)
    original_bank = stacking.OOF_BANK
    original_cache = stacking.GRAM_CACHE
    stacking.OOF_BANK = CARRIER_BANK
    stacking.GRAM_CACHE = GRAM_CACHE
    try:
        return stacking.compute_gram(device)
    finally:
        stacking.OOF_BANK = original_bank
        stacking.GRAM_CACHE = original_cache


def build_stack(
    gram: dict[str, np.ndarray],
    site: np.ndarray,
    regime: np.ndarray,
    batch_size: int,
    force: bool,
) -> tuple[np.memmap, object]:
    selection = json.loads(v7_projection.STACK_SELECTION.read_text(encoding="utf-8"))
    config = v7_projection.stack_config_from_record(
        selection["posterior_best"]["config"]
    )
    if STACK_CHANNEL.is_file() and not force:
        return validate_channel(STACK_CHANNEL), config
    partial = STACK_CHANNEL.with_name(STACK_CHANNEL.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    original_bank = v7_projection.OOF_BANK
    original_output = v7_projection.OOF_STACK
    v7_projection.OOF_BANK = CARRIER_BANK
    v7_projection.OOF_STACK = partial
    try:
        v7_projection.build_oof_stack(gram, config, site, regime, batch_size)
    finally:
        v7_projection.OOF_BANK = original_bank
        v7_projection.OOF_STACK = original_output
    replace_partial(partial, STACK_CHANNEL)
    return validate_channel(STACK_CHANNEL), config


def build_v4_projection(
    device: torch.device,
    batch_size: int,
    query_ids: np.ndarray,
    site: np.ndarray,
    force: bool,
) -> np.memmap:
    if PROJECTED_CHANNEL.is_file() and not force:
        return validate_channel(PROJECTED_CHANNEL)
    partial = PROJECTED_CHANNEL.with_name(PROJECTED_CHANNEL.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    original_stack = v7_projection.OOF_STACK
    original_output = v7_projection.OOF_PROJECTED
    v7_projection.OOF_STACK = STACK_CHANNEL
    v7_projection.OOF_PROJECTED = partial
    try:
        v7_projection.project_v4_policy(device, batch_size, query_ids, site)
    finally:
        v7_projection.OOF_STACK = original_stack
        v7_projection.OOF_PROJECTED = original_output
    replace_partial(partial, PROJECTED_CHANNEL)
    return validate_channel(PROJECTED_CHANNEL)


def build_scaled_projection(
    projected_score: dict[str, object],
    fold_id: np.ndarray,
    site: np.ndarray,
    force: bool,
) -> np.memmap:
    if SCALED_CHANNEL.is_file() and not force:
        return validate_channel(SCALED_CHANNEL)
    scale = lifecycle_overlay.nested_scale_lookup(
        {"projected_score": projected_score}, fold_id, site
    )
    partial = SCALED_CHANNEL.with_name(SCALED_CHANNEL.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    base = np.load(PROJECTED_CHANNEL, mmap_mode="r")
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=EXPECTED_SHAPE
    )
    for start in range(0, len(base), 8):
        end = min(start + 8, len(base))
        output[start:end] = (
            np.asarray(base[start:end], dtype=np.complex64)
            * scale[start:end, None, None, None]
        ).astype(np.complex64)
    output.flush()
    del output
    replace_partial(partial, SCALED_CHANNEL)
    return validate_channel(SCALED_CHANNEL)


def build_v7_lifecycle(
    device: torch.device,
    batch_size: int,
    fold_id: np.ndarray,
    site: np.ndarray,
    regime: np.ndarray,
    force: bool,
) -> np.memmap:
    if V7_CHANNEL.is_file() and not force:
        return validate_channel(V7_CHANNEL)
    current = lifecycle.channel_ad_support(
        np.load(SCALED_CHANNEL, mmap_mode="r"), CURRENT_AD, device, batch_size
    )
    selection = json.loads(lifecycle_overlay.SELECTION.read_text(encoding="utf-8"))
    winner = selection["winner"]
    if winner is None:
        raise RuntimeError("V7 lifecycle selection has no winner")
    chosen = np.full(len(fold_id), "identity", dtype="U72")
    regime_lookup = {"clear": 0, "light": 1, "medium": 2, "heavy": 3}
    for row in winner["nested_decisions"]:
        regime_index = int(
            regime_lookup.get(
                str(row.get("regime", "")), int(row.get("regime_index", -1))
            )
        )
        mask = (
            (fold_id == int(row["held_fold"]))
            & (site == int(row["site"]))
            & (regime == regime_index)
        )
        chosen[mask] = str(row["candidate"])
    target = np.load(
        lifecycle.TARGET_PATHS[str(winner["stage"])][0], mmap_mode="r"
    )
    base = np.load(SCALED_CHANNEL, mmap_mode="r")
    partial = V7_CHANNEL.with_name(V7_CHANNEL.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=EXPECTED_SHAPE
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        block = np.array(base[start:end], dtype=np.complex64, copy=True)
        for name in np.unique(chosen[start:end]):
            rows = np.flatnonzero(chosen[start:end] == name)
            if name == "identity" or len(rows) == 0:
                continue
            spec = lifecycle._candidate_from_name(str(name))
            with torch.inference_mode():
                prediction = lifecycle.apply_candidate(
                    torch.as_tensor(block[rows], device=device),
                    torch.as_tensor(
                        np.asarray(current[start:end][rows], dtype=np.float32),
                        device=device,
                    ),
                    torch.as_tensor(
                        np.asarray(target[start:end][rows], dtype=np.float32),
                        device=device,
                    ),
                    regime[start:end][rows],
                    site[start:end][rows],
                    spec,
                )
            block[rows] = prediction.cpu().numpy().astype(np.complex64)
        output[start:end] = block
    output.flush()
    del output
    replace_partial(partial, V7_CHANNEL)
    return validate_channel(V7_CHANNEL)


def build_v9(device: torch.device, batch_size: int, force: bool) -> np.memmap:
    if V9_CHANNEL.is_file() and not force:
        return validate_channel(V9_CHANNEL)
    old_pas = np.load(joint_paths.OLD_PAS, mmap_mode="r")
    old_pdp = np.load(joint_paths.OLD_PDP, mmap_mode="r")
    joint_pas = np.load(joint_paths.JOINT_PAS, mmap_mode="r")
    joint_pdp = np.load(joint_paths.JOINT_PDP, mmap_mode="r")
    base = np.load(V7_CHANNEL, mmap_mode="r")
    spec = V9Candidate(
        name="v9",
        pas_beta=0.75,
        pdp_beta=0.50,
        alpha_pas=0.45,
        alpha_pdp=0.35,
        cycles=4,
    )
    partial = V9_CHANNEL.with_name(V9_CHANNEL.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=EXPECTED_SHAPE
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        with torch.inference_mode():
            prediction = apply_v9_candidate(
                torch.as_tensor(
                    np.array(base[start:end], dtype=np.complex64, copy=True),
                    device=device,
                ),
                torch.as_tensor(
                    np.asarray(old_pas[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(joint_pas[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(old_pdp[start:end], dtype=np.float32), device=device
                ),
                torch.as_tensor(
                    np.asarray(joint_pdp[start:end], dtype=np.float32), device=device
                ),
                spec,
            )
        output[start:end] = prediction.cpu().numpy().astype(np.complex64)
    output.flush()
    del output
    replace_partial(partial, V9_CHANNEL)
    return validate_channel(V9_CHANNEL)


def build_v10_v11(
    device: torch.device,
    batch_size: int,
    query_ids: np.ndarray,
    force: bool,
) -> np.memmap:
    original = {
        "V9_PDP": canonical.V9_PDP,
        "V10_ALPHA": canonical.V10_ALPHA,
        "V10_CHANNEL": canonical.V10_CHANNEL,
        "V11_GATE_FEATURES": canonical.V11_GATE_FEATURES,
        "V11_GATE": canonical.V11_GATE,
        "V11_CHANNEL": canonical.V11_CHANNEL,
    }
    canonical.V9_PDP = V9_PDP
    canonical.V10_ALPHA = V10_ALPHA
    canonical.V10_CHANNEL = V10_CHANNEL
    canonical.V11_GATE_FEATURES = V11_GATE_FEATURES
    canonical.V11_GATE = V11_GATE
    canonical.V11_CHANNEL = V11_CHANNEL
    try:
        v9 = np.load(V9_CHANNEL, mmap_mode="r")
        base_pdp = canonical.build_v9_pdp(v9, device, batch_size, force)
        alpha = canonical.predict_v10_alpha(
            query_ids, base_pdp, device, batch_size, force
        )
        v10 = canonical.build_v10(
            v9, query_ids, base_pdp, alpha, device, batch_size, force
        )
        sources = canonical.pas_sources()
        gate = canonical.build_v11_gate(
            v10, sources, query_ids, device, batch_size, force
        )
        value = canonical.build_v11(
            v10, sources, query_ids, gate, device, batch_size, force
        )
    finally:
        for name, value_path in original.items():
            setattr(canonical, name, value_path)
    return validate_channel(V11_CHANNEL)


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0 or args.gpu_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if any(path.is_file() and path in FORBIDDEN_CHANNELS for path in FORBIDDEN_CHANNELS):
        # The files may exist in the workspace, but this script never lists or opens them.
        pass

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    setup, train_positions, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(lifecycle.REGIME_CACHE, allow_pickle=False)[
        "oof_regime"
    ].astype(np.int8)
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = load_or_compute_power(train_channel, 16)

    timings: dict[str, float] = {}
    phase = time.time()
    build_carriers(
        train_channel,
        train_positions,
        train_site,
        np.asarray(setup.bs_positions),
        point_power,
        device,
        args.gpu_batch_size,
        args.force,
    )
    timings["carrier_bank"] = time.time() - phase

    phase = time.time()
    gram = build_gram(device, args.force)
    timings["gram"] = time.time() - phase

    phase = time.time()
    build_stack(gram, site, regime, args.batch_size, args.force)
    timings["ridge_stack"] = time.time() - phase

    phase = time.time()
    build_v4_projection(device, args.batch_size, query_ids, site, args.force)
    projected_score = v7_projection.score_channel(
        np.load(PROJECTED_CHANNEL, mmap_mode="r"),
        train_channel,
        query_ids,
        site,
        fold_id,
        args.batch_size,
    )
    build_scaled_projection(projected_score, fold_id, site, args.force)
    timings["v4_projection_and_scaling"] = time.time() - phase

    phase = time.time()
    build_v7_lifecycle(
        device, args.batch_size, fold_id, site, regime, args.force
    )
    timings["v7_lifecycle"] = time.time() - phase

    phase = time.time()
    build_v9(device, args.batch_size, args.force)
    timings["v9"] = time.time() - phase

    phase = time.time()
    final = build_v10_v11(device, args.batch_size, query_ids, args.force)
    timings["v10_v11"] = time.time() - phase

    phase = time.time()
    overall, folds, _ = canonical.score_channel(
        final, train_channel, query_ids, fold_id, args.batch_size
    )
    timings["official_scoring"] = time.time() - phase

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        peak_memory = float(torch.cuda.max_memory_allocated(device) / 2**30)
    else:
        gpu_name = "cpu"
        peak_memory = 0.0
    report = {
        "experiment": VERSION,
        "status": "complete",
        "data_lineage": {
            "official_train_channel": PATHS.relative(TRAIN_CHANNEL),
            "historical_complete_channel_read": False,
            "forbidden_complete_channels": [
                PATHS.relative(path) for path in sorted(FORBIDDEN_CHANNELS)
            ],
            "reused_inputs": (
                "trained model parameters, fold definitions, official-data spectral "
                "statistics, raw Sionna path caches, and raw Sionna spectral caches"
            ),
            "rebuilt_complete_channels": [
                PATHS.relative(path)
                for path in (
                    STACK_CHANNEL,
                    PROJECTED_CHANNEL,
                    SCALED_CHANNEL,
                    V7_CHANNEL,
                    V9_CHANNEL,
                    V10_CHANNEL,
                    V11_CHANNEL,
                )
            ],
        },
        "metrics": {"overall": overall, "folds": folds},
        "output": {
            "path": PATHS.relative(V11_CHANNEL),
            "shape": list(final.shape),
            "dtype": str(final.dtype),
            "finite": True,
        },
        "performance": {
            "device": gpu_name,
            "peak_torch_memory_gib": peak_memory,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "phase_seconds": timings,
            "measured_bottleneck": max(timings, key=timings.get),
            "wall_seconds": float(time.time() - started),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
