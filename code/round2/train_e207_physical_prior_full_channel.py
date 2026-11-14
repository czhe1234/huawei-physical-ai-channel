#!/usr/bin/env python3
"""Run E207 self-built PathTokens through the strict E205 full-H interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import analyze_e114_path_state_variable_ceiling as e114
import train_e205_ordered_map_path_calibrator as e205
from project_paths import PATHS
from round2.data import load_positions
from round2.physical_prior.config import PhysicalPriorConfig
from round2.physical_prior.geometry import geometry_summary, load_geometry
from round2.physical_prior.model import (
    FeatureBuilder,
    PhysicalPriorLayer,
    select_candidates,
)
from round2.physical_prior.pipeline import (
    GEOMETRY_CACHE,
    RAW_TOKEN_CACHE,
    action_utility_labels,
    atomic_npz,
    fold_coordinate,
    load_tokens,
    training_candidates,
)
from run_path_component_transport import component_energy


VERSION = "Round2-E207-physical-prior-full-channel"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e207_physical_prior_full_channel"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e207_physical_prior_full_channel"
)
COMMON_DIR = ARTIFACT_DIR / "common"
COMMON_STATE = COMMON_DIR / "state.json"
TARGET_PAS = COMMON_DIR / "official_pas_target_f32.npy"
TARGET_PDP = COMMON_DIR / "official_pdp_target_f32.npy"
FULL_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
FULL_REPORT = ANALYSIS_DIR / "fixed500_report.json"

E205_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e205_ordered_map_path_calibrator"
    / "fixed500_report.json"
)

CANDIDATES = 128
SEQUENCE_STEPS = 16
RAW_CANDIDATES = 512
RANK_SAMPLE_COUNT = 160
RANK_SCORE_BATCH = 8
FEATURE_BATCH = 32

RUNTIME_FORCE = False
_ORIGINAL_BUILD_FOLD_CACHE = e205.build_fold_cache


def fold_dir(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}"


def fold_input_dir(outer: int) -> Path:
    return fold_dir(outer) / "inputs"


def fold_ranker_path(outer: int) -> Path:
    return fold_dir(outer) / "strict_ranker.pt"


def fold_report_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}.json"


def fold_channel_path(outer: int) -> Path:
    return fold_dir(outer) / "held_channel.npy"


def fold_endpoint_path(outer: int) -> Path:
    return fold_dir(outer) / "held_endpoint.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "fold", "aggregate"), default="prepare"
    )
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--carrier-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_e205() -> None:
    """Point the proven E205 training/rendering code at E207-owned artifacts."""

    e205.VERSION = VERSION
    e205.ARTIFACT_DIR = ARTIFACT_DIR
    e205.ANALYSIS_DIR = ANALYSIS_DIR
    e205.COMMON_DIR = COMMON_DIR
    e205.COMMON_STATE = COMMON_STATE
    e205.TARGET_PAS = TARGET_PAS
    e205.TARGET_PDP = TARGET_PDP
    e205.FULL_CHANNEL = FULL_CHANNEL
    e205.FULL_REPORT = FULL_REPORT
    e205.CANDIDATES = CANDIDATES
    e205.SEQUENCE_STEPS = SEQUENCE_STEPS
    e205.build_fold_cache = build_fold_cache
    e205.candidate_coordinate = candidate_coordinate


def required_inputs() -> tuple[Path, ...]:
    return (
        e205.TRAIN_CHANNEL,
        e205.BASE_V7,
        e205.OOF_META,
        e205.FULL_FOLD,
        e114.COMPONENTS,
        RAW_TOKEN_CACHE,
        GEOMETRY_CACHE,
        E205_REPORT,
    )


def split_ids(outer: int, validation_offset: int) -> tuple[np.ndarray, ...]:
    fold, query_ids, canonical = e205.canonical_full_fold()
    validation = (outer + validation_offset) % 5
    fit_ids = np.flatnonzero((fold != outer) & (fold != validation)).astype(np.int64)
    validation_ids = query_ids[canonical == validation]
    held_ids = query_ids[canonical == outer]
    excluded = np.concatenate((validation_ids, held_ids))
    if np.intersect1d(fit_ids, excluded).size:
        raise RuntimeError("E207 ranker fit rows overlap validation/held labels")
    return fit_ids, validation_ids, held_ids


def ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids).view(np.uint8)).hexdigest()


def input_paths(outer: int) -> dict[str, Path]:
    directory = fold_input_dir(outer)
    return {
        "state": directory / "state.json",
        "adapter": directory / "selected_adapter.npz",
        "coordinate": directory / "coordinate_f32.npy",
        "candidate": directory / "candidate_feature_f16.npy",
        "point": directory / "point_feature_f32.npy",
        "sequence": directory / "sequence_feature_f16.npy",
        "interaction": directory / "interaction_type_i8.npy",
        "sequence_mask": directory / "sequence_mask.npy",
        "selected_index": directory / "selected_index_i32.npy",
    }


def configure_fold_paths(outer: int) -> None:
    paths = input_paths(outer)
    e205.COMMON_CANDIDATE_FEATURE = paths["candidate"]
    e205.COMMON_POINT_FEATURE = paths["point"]
    e205.COMMON_SEQUENCE_FEATURE = paths["sequence"]
    e205.ORDERED_TYPE = paths["interaction"]
    e205.ORDERED_MASK = paths["sequence_mask"]
    e205.e146.CANDIDATE_CACHE = paths["adapter"]


def candidate_coordinate(
    outer: int,
    fit_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    setup: Any,
    component: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    del positions, site, setup, component
    paths = input_paths(outer)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    expected_digest = ids_digest(fit_ids)
    if state.get("fit_ids_sha256") != expected_digest:
        raise RuntimeError("E207 selected candidate split does not match state fit split")
    coordinate = np.load(paths["coordinate"], mmap_mode="r")
    return np.asarray(coordinate, dtype=np.float32), state["array_axis_fit"]


def ranker_checkpoint_matches(
    checkpoint: dict[str, Any],
    outer: int,
    validation_fold: int,
    fit_digest: str,
    rank_config: PhysicalPriorConfig,
) -> bool:
    return bool(
        checkpoint.get("outer") == outer
        and checkpoint.get("validation_fold") == validation_fold
        and checkpoint.get("fit_ids_sha256") == fit_digest
        and checkpoint.get("config") == rank_config.payload()
        and checkpoint.get("candidate_dim") == 27
        and checkpoint.get("sequence_dim") == 18
        and checkpoint.get("point_dim") == 10
    )


def train_or_load_ranker(
    outer: int,
    validation_fold: int,
    fit_ids: np.ndarray,
    fit_digest: str,
    labels: np.ndarray,
    tokens: dict[str, np.ndarray],
    builder: FeatureBuilder,
    rank_config: PhysicalPriorConfig,
    device: torch.device,
    force: bool,
) -> tuple[PhysicalPriorLayer, list[float], bool, float]:
    checkpoint_path = fold_ranker_path(outer)
    model = PhysicalPriorLayer(
        builder.candidate_dim,
        builder.sequence_dim,
        builder.point_dim,
        rank_config,
    ).to(device)
    if checkpoint_path.is_file() and not force:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ranker_checkpoint_matches(
            checkpoint, outer, validation_fold, fit_digest, rank_config
        ):
            model.load_state_dict(checkpoint["state_dict"])
            return model, list(checkpoint["epoch_loss"]), True, 0.0

    started = time.time()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=rank_config.rank_learning_rate,
        weight_decay=rank_config.rank_weight_decay,
    )
    rng = np.random.default_rng(rank_config.ranking_seed + outer * 1009)
    epoch_loss: list[float] = []
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        for epoch in range(rank_config.rank_epochs):
            shuffled = rng.permutation(fit_ids)
            total_loss = 0.0
            batches = 0
            model.train()
            for start in range(0, len(shuffled), rank_config.rank_batch_points):
                rows = shuffled[start : start + rank_config.rank_batch_points]
                candidate_ids = training_candidates(
                    rows, labels, tokens, RANK_SAMPLE_COUNT, rng
                )
                batch = builder.batch(rows, candidate_ids)
                target = torch.as_tensor(
                    labels[rows[:, None], candidate_ids], device=device
                )
                valid = batch["candidate_mask"].to(torch.float32)
                logits = model(batch)
                positive = (target > 0.0).to(torch.float32)
                weight = 1.0 + rank_config.rank_positive_weight * positive
                bce = F.binary_cross_entropy_with_logits(
                    logits, positive, reduction="none"
                )
                regression = F.smooth_l1_loss(
                    torch.sigmoid(logits), target, reduction="none", beta=0.05
                )
                loss = ((weight * bce + 2.0 * weight * regression) * valid).sum()
                loss = loss / (weight * valid).sum().clamp_min(1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"E207 outer={outer} rank epoch={epoch + 1} non-finite loss"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1
            epoch_loss.append(total_loss / max(batches, 1))
            print(
                f"E207 outer={outer} rank epoch={epoch + 1}/"
                f"{rank_config.rank_epochs} loss={epoch_loss[-1]:.6f}",
                flush=True,
            )
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    e205.atomic_torch_save(
        checkpoint_path,
        {
            "experiment": VERSION,
            "outer": outer,
            "validation_fold": validation_fold,
            "fit_ids_sha256": fit_digest,
            "config": rank_config.payload(),
            "candidate_dim": builder.candidate_dim,
            "sequence_dim": builder.sequence_dim,
            "point_dim": builder.point_dim,
            "epoch_loss": epoch_loss,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
    )
    return model, epoch_loss, False, float(time.time() - started)


def select_all_candidates(
    model: PhysicalPriorLayer,
    builder: FeatureBuilder,
    tokens: dict[str, np.ndarray],
    coordinate: np.ndarray,
    rank_config: PhysicalPriorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    started = time.time()
    point_count = len(tokens["mask"])
    scores = np.full(tokens["mask"].shape, -np.inf, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, point_count, RANK_SCORE_BATCH):
            end = min(start + RANK_SCORE_BATCH, point_count)
            rows = np.arange(start, end, dtype=np.int64)
            scores[rows] = model(builder.batch(rows)).cpu().numpy().astype(np.float32)
            if end % 400 == 0 or end == point_count:
                print(f"E207 rank score {end}/{point_count}", flush=True)

    selected_index = np.full((point_count, CANDIDATES), -1, dtype=np.int32)
    selected_coordinate = np.zeros((point_count, CANDIDATES, 3), dtype=np.float32)
    selected_mask = np.zeros((point_count, CANDIDATES), dtype=bool)
    for row in range(point_count):
        chosen = select_candidates(
            scores[row],
            tokens["mask"][row],
            tokens["source"][row],
            coordinate[row],
            rank_config,
        )
        count = min(len(chosen), CANDIDATES)
        selected_index[row, :count] = chosen[:count].astype(np.int32)
        selected_coordinate[row, :count] = coordinate[row, chosen[:count]]
        selected_mask[row, :count] = True
    return (
        selected_index,
        selected_coordinate,
        selected_mask,
        float(time.time() - started),
    )


def write_selected_features(
    temporary: Path,
    builder: FeatureBuilder,
    selected_index: np.ndarray,
    selected_coordinate: np.ndarray,
    selected_mask: np.ndarray,
) -> float:
    started = time.time()
    point_count = len(selected_index)
    candidate_output = np.lib.format.open_memmap(
        temporary / "candidate_feature_f16.npy",
        mode="w+",
        dtype=np.float16,
        shape=(point_count, CANDIDATES, builder.candidate_dim),
    )
    point_output = np.lib.format.open_memmap(
        temporary / "point_feature_f32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(point_count, builder.point_dim),
    )
    sequence_output = np.lib.format.open_memmap(
        temporary / "sequence_feature_f16.npy",
        mode="w+",
        dtype=np.float16,
        shape=(
            point_count,
            CANDIDATES,
            SEQUENCE_STEPS,
            builder.sequence_dim,
        ),
    )
    interaction_output = np.lib.format.open_memmap(
        temporary / "interaction_type_i8.npy",
        mode="w+",
        dtype=np.int8,
        shape=(point_count, CANDIDATES, SEQUENCE_STEPS),
    )
    sequence_mask_output = np.lib.format.open_memmap(
        temporary / "sequence_mask.npy",
        mode="w+",
        dtype=bool,
        shape=(point_count, CANDIDATES, SEQUENCE_STEPS),
    )
    for start in range(0, point_count, FEATURE_BATCH):
        end = min(start + FEATURE_BATCH, point_count)
        rows = np.arange(start, end, dtype=np.int64)
        safe = np.maximum(selected_index[start:end], 0)
        batch = np.arange(end - start, dtype=np.int64)[:, None]
        candidate = builder._candidate(rows)[batch, safe]
        sequence, interaction, sequence_mask = builder._sequence(rows)
        sequence = sequence[batch, safe]
        interaction = interaction[batch, safe]
        sequence_mask = sequence_mask[batch, safe]
        valid = selected_mask[start:end]
        candidate[~valid] = 0.0
        sequence[~valid] = 0.0
        interaction[~valid] = 0
        sequence_mask[~valid] = False
        interaction = np.where(interaction == 4, 3, interaction)
        if np.any((interaction < 0) | (interaction > 3)):
            raise ValueError("E207 interaction mapping produced an invalid class")
        candidate_output[start:end] = candidate.astype(np.float16)
        point_output[start:end] = builder._point(rows)
        sequence_output[start:end] = sequence.astype(np.float16)
        interaction_output[start:end] = interaction.astype(np.int8)
        sequence_mask_output[start:end] = sequence_mask
        if end % 400 == 0 or end == point_count:
            print(f"E207 selected features {end}/{point_count}", flush=True)
    for output in (
        candidate_output,
        point_output,
        sequence_output,
        interaction_output,
        sequence_mask_output,
    ):
        output.flush()
    np.save(temporary / "coordinate_f32.npy", selected_coordinate.astype(np.float32))
    np.save(temporary / "selected_index_i32.npy", selected_index.astype(np.int32))
    with (temporary / "selected_adapter.npz").open("wb") as handle:
        np.savez(handle, mask=selected_mask)
    return float(time.time() - started)


def input_state_complete(outer: int, fit_digest: str) -> bool:
    paths = input_paths(outer)
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        state.get("status") == "complete"
        and state.get("experiment") == VERSION
        and state.get("fit_ids_sha256") == fit_digest
        and state.get("rank_config") == PhysicalPriorConfig().payload()
    )


def build_fold_inputs(
    outer: int,
    config: e205.Config,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    started = time.time()
    validation_fold = (outer + config.validation_fold_offset) % 5
    fit_ids, validation_ids, held_ids = split_ids(
        outer, config.validation_fold_offset
    )
    if e205.FORCE_BASE_GLOBAL_IDS:
        excluded = np.fromiter(e205.FORCE_BASE_GLOBAL_IDS, dtype=np.int64)
        fit_ids = fit_ids[~np.isin(fit_ids, excluded)]
    fit_digest = ids_digest(fit_ids)
    if input_state_complete(outer, fit_digest) and not force:
        state = json.loads(input_paths(outer)["state"].read_text(encoding="utf-8"))
        return {"reused": True, **state}

    rank_config = PhysicalPriorConfig()
    setup, positions, _, site, _ = load_positions()
    geometry = load_geometry(GEOMETRY_CACHE)
    tokens = load_tokens(RAW_TOKEN_CACHE)
    if tokens["mask"].shape != (len(positions), RAW_CANDIDATES):
        raise ValueError(f"unexpected E206 token shape {tokens['mask'].shape}")
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_energy = component_energy(
        component["coefficients"].astype(np.complex64)
    ).astype(np.float64)
    truth_power = component["point_power"].astype(np.float64)

    coordinate_started = time.time()
    coordinate, axis_record, _, _ = fold_coordinate(
        outer,
        fit_ids,
        tokens,
        positions,
        site,
        setup,
        component,
        rank_config,
    )
    if coordinate.shape != (len(positions), RAW_CANDIDATES, 3):
        raise ValueError(f"unexpected E207 coordinate shape {coordinate.shape}")
    if not np.isfinite(coordinate[tokens["mask"]]).all():
        raise FloatingPointError("E207 contains non-finite valid coordinates")
    coordinate_seconds = float(time.time() - coordinate_started)

    label_started = time.time()
    labels = action_utility_labels(
        coordinate,
        tokens["mask"],
        truth_coordinate,
        truth_energy,
        truth_power,
        fit_ids,
        rank_config,
    )
    label_seconds = float(time.time() - label_started)
    if np.any(labels[validation_ids] != 0.0) or np.any(labels[held_ids] != 0.0):
        raise RuntimeError("E207 rank labels leaked into validation/held rows")

    builder = FeatureBuilder(
        tokens,
        positions,
        site,
        setup.bs_positions,
        geometry,
        coordinate,
        device,
    )
    ranker, epoch_loss, ranker_reused, rank_seconds = train_or_load_ranker(
        outer,
        validation_fold,
        fit_ids,
        fit_digest,
        labels,
        tokens,
        builder,
        rank_config,
        device,
        force,
    )
    selected_index, selected_coordinate, selected_mask, selection_seconds = (
        select_all_candidates(ranker, builder, tokens, coordinate, rank_config)
    )
    if np.any(selected_mask.sum(axis=1) == 0):
        raise RuntimeError("E207 selected an empty candidate set")

    directory = fold_input_dir(outer)
    temporary = directory.with_name(directory.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    feature_seconds = write_selected_features(
        temporary,
        builder,
        selected_index,
        selected_coordinate,
        selected_mask,
    )
    state = {
        "experiment": VERSION,
        "status": "complete",
        "outer": outer,
        "validation_fold": validation_fold,
        "fit_count": int(len(fit_ids)),
        "validation_count": int(len(validation_ids)),
        "held_count": int(len(held_ids)),
        "fit_ids_sha256": fit_digest,
        "zero_rank_label_overlap": True,
        "rank_config": rank_config.payload(),
        "array_axis_fit": axis_record,
        "ranker_reused": ranker_reused,
        "ranker_epoch_loss": epoch_loss,
        "selected_candidate_count": {
            "minimum": int(selected_mask.sum(axis=1).min()),
            "median": float(np.median(selected_mask.sum(axis=1))),
            "maximum": int(selected_mask.sum(axis=1).max()),
        },
        "feature_dimensions": {
            "candidate": builder.candidate_dim,
            "sequence": builder.sequence_dim,
            "point": builder.point_dim,
        },
        "timing": {
            "coordinate_seconds": coordinate_seconds,
            "label_seconds": label_seconds,
            "ranker_train_seconds": rank_seconds,
            "ranker_score_and_select_seconds": selection_seconds,
            "selected_feature_seconds": feature_seconds,
            "total_seconds": float(time.time() - started),
        },
        "gpu": e205.gpu_record(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    e205.atomic_json(temporary / "state.json", state)
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(temporary, directory)
    return {"reused": False, **state}


def build_fold_cache(
    outer: int,
    config: e205.Config,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    input_record = build_fold_inputs(
        outer, config, device, force=bool(force or RUNTIME_FORCE)
    )
    configure_fold_paths(outer)
    cache = _ORIGINAL_BUILD_FOLD_CACHE(
        outer, config, device, force=bool(force or RUNTIME_FORCE)
    )
    cache["physical_prior_inputs"] = input_record
    return cache


def prepare(device: torch.device, force: bool) -> dict[str, Any]:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    COMMON_DIR.mkdir(parents=True, exist_ok=True)
    for path in required_inputs():
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(RAW_TOKEN_CACHE, allow_pickle=False) as cache:
        mask = cache["mask"].astype(bool)
    if mask.shape != (4000, RAW_CANDIDATES):
        raise ValueError(f"unexpected raw token mask {mask.shape}")
    target_record = e205.build_official_targets(device, force)
    _, query_ids, _ = e205.canonical_full_fold()
    base = torch.as_tensor(
        np.array(np.load(e205.BASE_V7, mmap_mode="r"), copy=True), device=device
    )
    truth_all = np.load(e205.TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    baseline, _ = e205.score_channels(base, truth, e205.Config().score_batch_size)
    if abs(float(baseline["score"]) - e205.EXPECTED_V7_SCORE) > 1e-8:
        raise RuntimeError("E207 V7 identity check failed")
    smoke = e205.model_smoke(device)
    geometry = load_geometry(GEOMETRY_CACHE)
    state = {
        "experiment": VERSION,
        "status": "prepare_complete",
        "baseline_v7": baseline,
        "targets": target_record,
        "raw_tokens": {
            "shape": list(mask.shape),
            "valid_count": int(mask.sum()),
            "minimum_per_point": int(mask.sum(axis=1).min()),
            "median_per_point": float(np.median(mask.sum(axis=1))),
            "maximum_per_point": int(mask.sum(axis=1).max()),
        },
        "geometry": geometry_summary(geometry),
        "smoke": smoke,
        "gpu": e205.gpu_record(device),
        "seconds": float(time.time() - started),
    }
    e205.atomic_json(COMMON_STATE, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return state


def metric_delta(
    selected: dict[str, float | int], baseline: dict[str, float | int]
) -> dict[str, float]:
    return {
        "pas": float(selected["pas"] - baseline["pas"]),
        "pdp": float(selected["pdp"] - baseline["pdp"]),
        "nmse": float(selected["nmse"] - baseline["nmse"]),
        "nmse_score": float(selected["nmse_score"] - baseline["nmse_score"]),
        "score": float(selected["score"] - baseline["score"]),
    }


def refresh_fold_performance_records() -> None:
    for outer in range(5):
        report_path = fold_report_path(outer)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        physical = report["cache"]["physical_prior_inputs"]["timing"]
        carrier = report["cache"]["timing"]
        stages = {
            "strict physical ranker training": float(
                physical["ranker_train_seconds"]
            ),
            "ranker scoring and top128 selection": float(
                physical["ranker_score_and_select_seconds"]
            ),
            "selected token feature materialization": float(
                physical["selected_feature_seconds"]
            ),
            "strict 17-carrier field cache": float(carrier["carrier_field_total"]),
        }
        report["performance"]["stage_seconds"] = stages
        report["performance"]["measured_bottleneck"] = max(
            stages, key=stages.get
        )
        e205.atomic_json(report_path, report)


def aggregate(device: torch.device, config: e205.Config) -> dict[str, Any]:
    refresh_fold_performance_records()
    preliminary = e205.aggregate(device, config)
    e205_report = json.loads(E205_REPORT.read_text(encoding="utf-8"))
    e205_folds = {int(row["outer"]): row for row in e205_report["folds"]}
    fold_deltas: list[float] = []
    for row in preliminary["folds"]:
        baseline_v7 = row["baseline"]
        delta_v7 = row["delta"]
        reference = e205_folds[int(row["outer"])]["selected"]
        delta_reference = metric_delta(row["selected"], reference)
        row["baseline_v7"] = baseline_v7
        row["delta_vs_v7"] = delta_v7
        row["baseline"] = reference
        row["delta"] = delta_reference
        fold_deltas.append(delta_reference["score"])

    selected = preliminary["selected"]
    baseline_v7 = preliminary["baseline"]
    delta_v7 = preliminary["delta"]
    baseline_e205 = e205_report["selected"]
    delta_e205 = metric_delta(selected, baseline_e205)
    positive_folds = int(sum(value > 0.0 for value in fold_deltas))
    worst_fold_delta = float(min(fold_deltas))
    component_protection = bool(
        delta_e205["pas"] >= -config.maximum_component_drop
        and delta_e205["pdp"] >= -config.maximum_component_drop
        and delta_e205["nmse_score"] >= -config.maximum_component_drop
    )
    passed = bool(
        float(selected["score"]) >= config.required_score
        and positive_folds >= config.required_positive_folds
        and worst_fold_delta >= config.minimum_fold_delta
        and component_protection
    )
    preliminary.update(
        {
            "experiment": VERSION,
            "status": "fixed500_passed" if passed else "fixed500_failed",
            "baseline": baseline_e205,
            "selected": selected,
            "delta": delta_e205,
            "baseline_v7": baseline_v7,
            "delta_vs_v7": delta_v7,
            "comparison_targets": {
                "v12_fixed500": 0.677521,
                "v15_fixed500": 0.686375,
            },
            "gate": {
                "required_score": config.required_score,
                "required_positive_folds_vs_e205": config.required_positive_folds,
                "minimum_fold_delta_vs_e205": config.minimum_fold_delta,
                "maximum_component_drop_vs_e205": config.maximum_component_drop,
                "positive_folds_vs_e205": positive_folds,
                "worst_fold_delta_vs_e205": worst_fold_delta,
                "component_protection_vs_e205": component_protection,
                "passed": passed,
            },
            "data_lineage": {
                "official_train_h": True,
                "official_train_positions": True,
                "official_map_setup": True,
                "canonical_v7_base": PATHS.relative(e205.BASE_V7),
                "self_built_physical_tokens": PATHS.relative(RAW_TOKEN_CACHE),
                "strict_ranker_excludes_outer_and_validation": True,
                "sionna_used": False,
                "mitsuba_drjit_pathsolver_used": False,
                "v8_v15_derived_inputs_used": False,
                "predicted_heads": ["path_mass", "8_port_power"],
                "frozen": ["path_angle", "path_delay", "v7_phase", "point_power"],
            },
            "failure_conclusion_if_failed": (
                "High-coverage E206 PathTokens do not convert to V12-level complete-H "
                "score through the frozen E205 mass/port state interface."
            ),
        }
    )
    run_report_path = ANALYSIS_DIR / "parallel_run.json"
    if run_report_path.is_file():
        run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
        if run_report.get("status") == "complete":
            preliminary["performance"][
                "parallel_wall_time_recorded_by_runner"
            ] = float(run_report["parallel_wall_seconds"])
    e205.atomic_json(FULL_REPORT, preliminary)
    print(json.dumps(preliminary, ensure_ascii=False, indent=2), flush=True)
    return preliminary


def main() -> None:
    global RUNTIME_FORCE
    args = parse_args()
    if args.outer not in range(5):
        raise ValueError(f"invalid outer fold {args.outer}")
    RUNTIME_FORCE = bool(args.force)
    configure_e205()
    config = e205.Config(
        carrier_batch_size=args.carrier_batch_size,
        train_batch_size=args.train_batch_size,
        epochs=args.epochs,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    torch.manual_seed(config.seed + args.outer)
    np.random.seed((config.seed + args.outer) % (2**32 - 1))
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(device, args.force)
    elif args.stage == "fold":
        for path in required_inputs():
            if not path.is_file():
                raise FileNotFoundError(path)
        if not COMMON_STATE.is_file():
            raise FileNotFoundError(COMMON_STATE)
        e205.train_fold(args.outer, config, device, args.force)
    else:
        aggregate(device, config)


configure_e205()


if __name__ == "__main__":
    main()
