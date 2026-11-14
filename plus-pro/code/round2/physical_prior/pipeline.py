"""Strict five-fold training and coverage evaluation for PhysicalPriorLayer."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e146_map_propagation_graph_coverage as e146
import train_e150_v7_pathwise_carrier_operator as e150
from project_paths import PATHS
from round2.data import load_positions
from run_path_component_transport import component_energy

from .config import METERS_PER_DELAY_BIN, PhysicalPriorConfig
from .geometry import GeometryData, geometry_summary
from .model import FeatureBuilder, PhysicalPriorLayer, select_candidates
from .tokens import FINAL_DIR, TOKEN_DTYPES, empty_token_arrays


ARTIFACT_DIR = PATHS.artifacts / "round2" / "physical_prior"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_physical_prior"
GEOMETRY_CACHE = ARTIFACT_DIR / "geometry.npz"
RAW_TOKEN_CACHE = ARTIFACT_DIR / "raw_tokens_n4000_k512_v2.npz"
FINAL_TOKEN_CACHE = ARTIFACT_DIR / "fixed500_path_tokens_k128.npz"
FINAL_REPORT = ANALYSIS_DIR / "fixed500_support_report.json"
RUN_REPORT = ANALYSIS_DIR / "parallel_run.json"

OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FULL_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def fold_dir(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}"


def fold_report_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}.json"


def fold_output_path(outer: int) -> Path:
    return fold_dir(outer) / "held_tokens.npz"


def fold_checkpoint_path(outer: int) -> Path:
    return fold_dir(outer) / "ranker.pt"


def canonical_fold() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical = metadata["fold_id"].astype(np.int8)
    fold = np.load(FULL_FOLD, allow_pickle=False).astype(np.int8)
    fold = np.array(fold, copy=True)
    fold[query_ids] = canonical
    return fold, query_ids, canonical


def load_tokens(path: Path = RAW_TOKEN_CACHE) -> dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=False)
    return {name: cache[name] for name in TOKEN_DTYPES}


def direct_los(
    tokens: dict[str, np.ndarray],
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
) -> np.ndarray:
    direct = np.linalg.norm(positions - bs_positions[site], axis=1)
    delta = np.abs(tokens["path_length"] - direct[:, None])
    candidate = (
        tokens["mask"]
        & (tokens["source"] == 0)
        & (delta <= 0.02)
    )
    index = np.argmin(np.where(candidate, delta, np.inf), axis=1)
    exists = np.any(candidate, axis=1)
    count = tokens["interaction_count"][np.arange(len(tokens["mask"])), index]
    return exists & (count == 0)


def fold_coordinate(
    outer: int,
    fit_ids: np.ndarray,
    tokens: dict[str, np.ndarray],
    positions: np.ndarray,
    site: np.ndarray,
    setup: Any,
    component: Any,
    config: PhysicalPriorConfig,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    geometric_los = direct_los(tokens, positions, site, setup.bs_positions)
    observations = e146.select_los_observations(
        positions,
        site,
        setup.bs_positions,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
        geometric_los,
    )
    mapping_config = e146.Config(
        horizontal_scale=config.horizontal_scale,
        vertical_scale=config.vertical_scale,
        delay_scale=config.delay_scale,
        match_cost_limit=config.match_cost_limit,
        match_bandwidth=config.match_bandwidth,
        minimum_truth_fraction=config.minimum_truth_fraction,
    )
    horizontal, vertical, record = e150.fit_strict_array_axes(
        outer, fit_ids, observations, site, mapping_config
    )
    arrays = {
        "theta": tokens["theta_t"],
        "phi": tokens["phi_t"],
        "delay": np.mod(
            tokens["path_length"].astype(np.float64) / METERS_PER_DELAY_BIN,
            192.0,
        ).astype(np.float32),
    }
    coordinate = e146.map_coordinates_linear_array(
        arrays, horizontal[site], vertical[site]
    )
    return coordinate, record, horizontal, vertical


def support_cost(
    candidate: np.ndarray, truth: np.ndarray, config: PhysicalPriorConfig
) -> np.ndarray:
    periods = np.asarray((16.0, 8.0, 192.0), dtype=np.float32)
    scales = np.asarray(
        (config.horizontal_scale, config.vertical_scale, config.delay_scale),
        dtype=np.float32,
    )
    delta = np.remainder(
        candidate[:, None] - truth[None] + periods / 2.0, periods
    ) - periods / 2.0
    return np.sum(np.square(delta / scales), axis=2)


def action_utility_labels(
    coordinate: np.ndarray,
    mask: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_energy: np.ndarray,
    truth_power: np.ndarray,
    rows: np.ndarray,
    config: PhysicalPriorConfig,
) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.float32)
    total = truth_energy.sum(axis=1, dtype=np.float64)
    fraction = np.divide(
        truth_energy,
        np.maximum(total[:, None], 1e-30),
        out=np.zeros_like(truth_energy, dtype=np.float64),
    )
    power_fraction = np.divide(
        truth_energy,
        np.maximum(truth_power[:, None], 1e-30),
        out=np.zeros_like(truth_energy, dtype=np.float64),
    )
    valid_truth = power_fraction >= config.minimum_truth_fraction
    for progress, row in enumerate(rows):
        candidate_ids = np.flatnonzero(mask[row])
        truth_ids = np.flatnonzero(valid_truth[row])
        if len(candidate_ids) == 0 or len(truth_ids) == 0:
            continue
        cost = support_cost(
            coordinate[row, candidate_ids], truth_coordinate[row, truth_ids], config
        )
        utility = fraction[row, truth_ids][None] * np.exp(
            -0.5 * cost / config.match_bandwidth
        )
        utility[cost > config.match_cost_limit] = 0.0
        candidate_local, truth_local = linear_sum_assignment(-utility)
        accepted = utility[candidate_local, truth_local] > 0.0
        labels[row, candidate_ids[candidate_local[accepted]]] = utility[
            candidate_local[accepted], truth_local[accepted]
        ].astype(np.float32)
        if (progress + 1) % 500 == 0:
            print(f"physical labels {progress + 1}/{len(rows)}", flush=True)
    return labels


def training_candidates(
    rows: np.ndarray,
    labels: np.ndarray,
    tokens: dict[str, np.ndarray],
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.zeros((len(rows), maximum), dtype=np.int64)
    for local, row in enumerate(rows):
        valid = np.flatnonzero(tokens["mask"][row])
        positive = valid[labels[row, valid] > 0.0]
        priority = np.argsort(
            -(
                tokens["relative_energy"][row, valid]
                * (0.5 + 0.5 * tokens["geometry_confidence"][row, valid])
            )
        )
        hard = valid[priority[: maximum // 2]]
        chosen = np.unique(np.concatenate((positive, hard)))
        if len(chosen) < maximum:
            remaining = np.setdiff1d(valid, chosen, assume_unique=False)
            if len(remaining):
                add = rng.choice(
                    remaining,
                    size=min(maximum - len(chosen), len(remaining)),
                    replace=False,
                )
                chosen = np.concatenate((chosen, add))
        if len(chosen) > maximum:
            positive_set = set(int(value) for value in positive)
            ordered = sorted(
                (int(value) for value in chosen),
                key=lambda value: (
                    value not in positive_set,
                    -float(labels[row, value]),
                    -float(tokens["relative_energy"][row, value]),
                ),
            )
            chosen = np.asarray(ordered[:maximum], dtype=np.int64)
        if len(chosen) < maximum:
            fill = np.resize(chosen if len(chosen) else valid[:1], maximum - len(chosen))
            chosen = np.concatenate((chosen, fill))
        rng.shuffle(chosen)
        output[local] = chosen[:maximum]
    return output


def gpu_record(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "cuda": False}
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "cuda": True,
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
    }


def evaluate_selected(
    coordinate: np.ndarray,
    mask: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_energy: np.ndarray,
    truth_power: np.ndarray,
    config: PhysicalPriorConfig,
) -> dict[str, Any]:
    mapping_config = e146.Config(
        horizontal_scale=config.horizontal_scale,
        vertical_scale=config.vertical_scale,
        delay_scale=config.delay_scale,
        match_cost_limit=config.match_cost_limit,
        match_bandwidth=config.match_bandwidth,
        minimum_truth_fraction=config.minimum_truth_fraction,
    )
    candidate_type = np.zeros(mask.shape, dtype=np.int8)
    result = e146.evaluate_coverage(
        coordinate,
        mask,
        candidate_type,
        truth_coordinate,
        truth_energy,
        truth_power,
        mapping_config,
    )
    return result


def train_fold(
    outer: int,
    geometry: GeometryData,
    config: PhysicalPriorConfig,
    device_name: str,
    force: bool = False,
) -> dict[str, Any]:
    report_path = fold_report_path(outer)
    output_path = fold_output_path(outer)
    if report_path.is_file() and output_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    started = time.time()
    random.seed(config.ranking_seed + outer)
    np.random.seed((config.ranking_seed + outer) % (2**32 - 1))
    torch.manual_seed(config.ranking_seed + outer)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config.ranking_seed + outer)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    fold, query_ids, canonical = canonical_fold()
    fit_ids = np.flatnonzero(fold != outer).astype(np.int64)
    held_ids = query_ids[canonical == outer]
    if np.intersect1d(fit_ids, held_ids).size:
        raise RuntimeError("fit/held overlap")
    setup, positions, _, site, _ = load_positions()
    tokens = load_tokens()
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_coefficient = component["coefficients"].astype(np.complex64)
    truth_energy = component_energy(truth_coefficient).astype(np.float64)
    truth_power = component["point_power"].astype(np.float64)
    coordinate, axis_record, horizontal, vertical = fold_coordinate(
        outer,
        fit_ids,
        tokens,
        positions,
        site,
        setup,
        component,
        config,
    )
    label_started = time.time()
    labels = action_utility_labels(
        coordinate,
        tokens["mask"],
        truth_coordinate,
        truth_energy,
        truth_power,
        fit_ids,
        config,
    )
    label_seconds = time.time() - label_started
    builder = FeatureBuilder(
        tokens,
        positions,
        site,
        setup.bs_positions,
        geometry,
        coordinate,
        device,
    )
    model = PhysicalPriorLayer(
        builder.candidate_dim,
        builder.sequence_dim,
        builder.point_dim,
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.rank_learning_rate,
        weight_decay=config.rank_weight_decay,
    )
    rng = np.random.default_rng(config.ranking_seed + outer * 1009)
    train_started = time.time()
    epoch_loss: list[float] = []
    sample_count = min(160, config.raw_candidate_count)
    for epoch in range(config.rank_epochs):
        shuffled = rng.permutation(fit_ids)
        total_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, len(shuffled), config.rank_batch_points):
            rows = shuffled[start : start + config.rank_batch_points]
            candidate_ids = training_candidates(
                rows, labels, tokens, sample_count, rng
            )
            batch = builder.batch(rows, candidate_ids)
            batch_index = np.arange(len(rows), dtype=np.int64)[:, None]
            target = torch.as_tensor(
                labels[rows[:, None], candidate_ids], device=device
            )
            valid = batch["candidate_mask"].to(torch.float32)
            logits = model(batch)
            positive = (target > 0.0).to(torch.float32)
            weight = 1.0 + config.rank_positive_weight * positive
            bce = F.binary_cross_entropy_with_logits(
                logits, positive, reduction="none"
            )
            regression = F.smooth_l1_loss(
                torch.sigmoid(logits), target, reduction="none", beta=0.05
            )
            loss = ((weight * bce + 2.0 * weight * regression) * valid).sum()
            loss = loss / (weight * valid).sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        epoch_loss.append(total_loss / max(batches, 1))
        print(
            f"physical rank outer={outer} epoch={epoch + 1}/{config.rank_epochs} "
            f"loss={epoch_loss[-1]:.6f}",
            flush=True,
        )
    train_seconds = time.time() - train_started
    atomic_torch(
        fold_checkpoint_path(outer),
        {
            "state_dict": model.state_dict(),
            "config": config.payload(),
            "candidate_dim": builder.candidate_dim,
            "sequence_dim": builder.sequence_dim,
            "point_dim": builder.point_dim,
            "outer": outer,
        },
    )

    predict_started = time.time()
    score = np.full(tokens["mask"].shape, -np.inf, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(held_ids), 8):
            rows = held_ids[start : start + 8]
            score[rows] = model(builder.batch(rows)).cpu().numpy().astype(np.float32)
    selected_index = np.zeros(
        (len(held_ids), config.output_candidate_count), dtype=np.int64
    )
    selected_coordinate = np.zeros(
        (len(held_ids), config.output_candidate_count, 3), dtype=np.float32
    )
    selected_mask = np.zeros(
        (len(held_ids), config.output_candidate_count), dtype=bool
    )
    selected_tokens = empty_token_arrays(
        len(held_ids), config.output_candidate_count, config.maximum_interactions
    )
    for local, row in enumerate(held_ids):
        chosen = select_candidates(
            score[row],
            tokens["mask"][row],
            tokens["source"][row],
            coordinate[row],
            config,
        )
        selected_index[local, : len(chosen)] = chosen
        selected_coordinate[local, : len(chosen)] = coordinate[row, chosen]
        selected_mask[local, : len(chosen)] = True
        for name in selected_tokens:
            selected_tokens[name][local, : len(chosen)] = tokens[name][row, chosen]
    selected_metrics = evaluate_selected(
        selected_coordinate,
        selected_mask,
        truth_coordinate[held_ids],
        truth_energy[held_ids],
        truth_power[held_ids],
        config,
    )
    baseline_cache = np.load(e146.CANDIDATE_CACHE, allow_pickle=False)
    baseline_arrays = {
        key: baseline_cache[key]
        for key in baseline_cache.files
        if key != "config_json"
    }
    baseline_coordinate = e146.map_coordinates_linear_array(
        baseline_arrays, horizontal[site], vertical[site]
    ).astype(np.float32)
    baseline_metrics = evaluate_selected(
        baseline_coordinate[held_ids],
        baseline_arrays["mask"][held_ids].astype(bool),
        truth_coordinate[held_ids],
        truth_energy[held_ids],
        truth_power[held_ids],
        config,
    )
    predict_seconds = time.time() - predict_started
    atomic_npz(
        output_path,
        point_ids=held_ids.astype(np.int64),
        selected_index=selected_index,
        selected_coordinate=selected_coordinate,
        selected_mask=selected_mask,
        selected_point_coverage=np.asarray(
            selected_metrics["point_coverage"], dtype=np.float32
        ),
        baseline_point_coverage=np.asarray(
            baseline_metrics["point_coverage"], dtype=np.float32
        ),
        truth_energy_total=truth_energy[held_ids].sum(axis=1).astype(np.float64),
        **selected_tokens,
    )
    report = {
        "status": "complete",
        "outer": int(outer),
        "fit_count": int(len(fit_ids)),
        "held_count": int(len(held_ids)),
        "zero_label_overlap": True,
        "selected_energy_coverage": float(
            selected_metrics["global_energy_coverage"]
        ),
        "selected_quality_coverage": float(
            selected_metrics["global_quality_coverage"]
        ),
        "baseline_energy_coverage": float(
            baseline_metrics["global_energy_coverage"]
        ),
        "baseline_quality_coverage": float(
            baseline_metrics["global_quality_coverage"]
        ),
        "coverage_gain": float(
            selected_metrics["global_energy_coverage"]
            - baseline_metrics["global_energy_coverage"]
        ),
        "axis_mapping": axis_record,
        "epoch_loss": epoch_loss,
        "performance": {
            "device": gpu_record(device),
            "label_seconds": float(label_seconds),
            "train_seconds": float(train_seconds),
            "predict_and_score_seconds": float(predict_seconds),
            "total_seconds": float(time.time() - started),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "measured_bottleneck": "candidate sequence ranker training",
        },
        "checkpoint": PATHS.relative(fold_checkpoint_path(outer)),
        "output": PATHS.relative(output_path),
    }
    atomic_json(report_path, report)
    return report


def aggregate_folds(
    geometry: GeometryData,
    config: PhysicalPriorConfig,
) -> dict[str, Any]:
    started = time.time()
    _, query_ids, canonical = canonical_fold()
    query_row = {int(point): row for row, point in enumerate(query_ids)}
    final = empty_token_arrays(
        len(query_ids), config.output_candidate_count, config.maximum_interactions
    )
    selected_point = np.zeros(len(query_ids), dtype=np.float32)
    baseline_point = np.zeros(len(query_ids), dtype=np.float32)
    truth_total = np.zeros(len(query_ids), dtype=np.float64)
    coverage = np.zeros(len(query_ids), dtype=np.uint8)
    folds: list[dict[str, Any]] = []
    for outer in range(5):
        report_path = fold_report_path(outer)
        output_path = fold_output_path(outer)
        if not report_path.is_file() or not output_path.is_file():
            raise FileNotFoundError(f"missing outer{outer}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        folds.append(report)
        with np.load(output_path, allow_pickle=False) as cache:
            point_ids = cache["point_ids"].astype(np.int64)
            rows = np.asarray([query_row[int(point)] for point in point_ids], dtype=np.int64)
            for name in final:
                final[name][rows] = cache[name]
            selected_point[rows] = cache["selected_point_coverage"]
            baseline_point[rows] = cache["baseline_point_coverage"]
            truth_total[rows] = cache["truth_energy_total"]
            coverage[rows] += 1
    if not np.all(coverage == 1):
        raise RuntimeError(
            f"fixed500 fold coverage failure {np.unique(coverage, return_counts=True)}"
        )
    denominator = max(float(np.sum(truth_total)), 1e-30)
    selected = float(np.sum(selected_point * truth_total) / denominator)
    baseline = float(np.sum(baseline_point * truth_total) / denominator)
    gain = selected - baseline
    fold_gains = [float(row["coverage_gain"]) for row in folds]
    positive = int(sum(value > 0.0 for value in fold_gains))
    success = bool(
        selected >= 0.87
        and gain >= 0.02
        and positive >= 4
    )
    atomic_npz(
        FINAL_TOKEN_CACHE,
        point_ids=query_ids.astype(np.int64),
        fold_id=canonical.astype(np.int8),
        **final,
    )
    report = {
        "experiment": "Round2-E206-self-built-physical-prior",
        "status": "passed" if success else "failed",
        "configuration": config.payload(),
        "geometry": geometry_summary(geometry),
        "fixed500": {
            "selected_energy_coverage": selected,
            "e205_energy_coverage": baseline,
            "gain": gain,
            "positive_folds": positive,
            "required": {
                "selected_energy_coverage": 0.87,
                "gain_vs_e205": 0.02,
                "positive_folds": 4,
            },
        },
        "folds": [
            {
                "outer": int(row["outer"]),
                "selected_energy_coverage": float(
                    row["selected_energy_coverage"]
                ),
                "e205_energy_coverage": float(row["baseline_energy_coverage"]),
                "gain": float(row["coverage_gain"]),
                "seconds": float(row["performance"]["total_seconds"]),
                "gpu": row["performance"]["device"],
                "peak_cuda_bytes": int(row["performance"]["peak_cuda_bytes"]),
            }
            for row in folds
        ],
        "output": PATHS.relative(FINAL_TOKEN_CACHE),
        "seconds": float(time.time() - started),
        "decision": (
            "physical_prior_support_reproduced"
            if success
            else "current_self_built_three_seed_support_did_not_reproduce_e185"
        ),
    }
    atomic_json(FINAL_REPORT, report)
    return report
