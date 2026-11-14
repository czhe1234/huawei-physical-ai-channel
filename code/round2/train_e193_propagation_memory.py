#!/usr/bin/env python3
"""Calibrate target Sionna paths from propagation-similar official-H donors."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

import analyze_e185_sionna_test_propagation_similarity as e185
import analyze_e86_surface_endpoint_v11_blend as e86
import analyze_e100_locked_energy_angle_correction as e100
import optimize_e98_global_array_angle_mapping as e98
import train_e99_corrected_array_surface_model as e99
import train_e190_official_sionna_path_calibrator as e190
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from build_e193_independent_v11_base import V11_CHANNEL as INDEPENDENT_V11
from project_paths import PATHS
from round2.data import load_positions
from run_path_component_transport import component_energy
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E193-propagation-similar-official-path-memory"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e193_propagation_memory"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e193_propagation_memory"
STATIC_DIR = ARTIFACT_DIR / "static"
FULL_REPORT = ANALYSIS_DIR / "fixed500_report.json"
FULL_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
V15_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_official_sionna_path_calibrator"
    / "fixed500_report.json"
)

TRAIN_BASE_EMBEDDING = STATIC_DIR / "train_base_embedding_f16.npy"
TRAIN_SCATTER_EMBEDDING = STATIC_DIR / "train_scatter_embedding_f16.npy"
PATH_DESCRIPTOR = STATIC_DIR / "path_descriptor_f16.npy"
PATH_VALID = STATIC_DIR / "path_valid.npy"
STATIC_REPORT = ANALYSIS_DIR / "static_memory_inputs.json"

PERIODS = np.asarray((e85.MH, e85.MV, e85.S_SC), dtype=np.float32)
PORTS = e85.MP * e85.N_UE
TINY = 1e-30
LIGHT_SPEED_MPS = 299_792_458.0
TOP_OVERALL = 4
TOP_NONLOCAL = 4
DONOR_COUNT = TOP_OVERALL + TOP_NONLOCAL
NONLOCAL_DISTANCE_M = 30.0
MEMORY_BATCH_SIZE = 24
PATH_DESCRIPTOR_DIM = 22
MEMORY_FEATURE_DIM = 21

FORBIDDEN_RUNTIME_INPUTS = (
    e86.V11_CHANNEL,
    e100.V12_CHANNEL,
    e190.FULL_CHANNEL,
    e190.OLD_ENDPOINT,
)


@dataclass
class MemoryArrays:
    features: np.ndarray
    prior_mass: np.ndarray
    prior_coordinate: np.ndarray
    prior_port: np.ndarray
    donor_ids: np.ndarray
    donor_similarity: np.ndarray
    report: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "fold", "aggregate"), default="prepare"
    )
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fold_report_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}_report.json"


def fold_channel_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_channel.npy"


def fold_endpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_endpoint.npz"


def fold_checkpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_model.pt"


def fold_memory_path(outer: int, split: str) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_{split}_memory.npz"


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    partial = path.with_name(path.stem + ".partial.npy")
    partial.unlink(missing_ok=True)
    np.save(partial, value)
    os.replace(partial, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    partial = path.with_name(path.stem + ".partial.npz")
    partial.unlink(missing_ok=True)
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)


def atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    torch.save(payload, partial)
    os.replace(partial, path)


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def prepare_static_inputs(force: bool) -> dict[str, object]:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ready = all(
        path.is_file()
        for path in (
            TRAIN_BASE_EMBEDDING,
            TRAIN_SCATTER_EMBEDDING,
            PATH_DESCRIPTOR,
            PATH_VALID,
            STATIC_REPORT,
        )
    )
    if ready and not force:
        return json.loads(STATIC_REPORT.read_text(encoding="utf-8"))

    started = time.time()
    setup, train_positions, _, train_sites, _ = load_positions()
    positions = np.asarray(train_positions, dtype=np.float64)
    sites = np.asarray(train_sites, dtype=np.int8)
    bs_positions = np.asarray(setup.bs_positions, dtype=np.float64)
    cache = e185.load_cache(e185.TRAIN_CACHE, len(positions))

    base, base_report = e185.structural_embedding(
        cache, positions, sites, bs_positions, 0
    )
    scatter, scatter_report = e185.structural_embedding(
        cache, positions, sites, bs_positions, 1
    )
    atomic_save_npy(TRAIN_BASE_EMBEDDING, base.astype(np.float16))
    atomic_save_npy(TRAIN_SCATTER_EMBEDDING, scatter.astype(np.float16))

    mask = np.asarray(cache["mask"], dtype=bool)
    tau = np.asarray(cache["tau"], dtype=np.float64)
    angles = np.asarray(cache["angles"], dtype=np.float64)
    source = np.asarray(cache["source"], dtype=np.int64)
    depth = np.asarray(cache["interaction_count"], dtype=np.int64)
    interactions = np.asarray(cache["interactions"], dtype=np.int64)

    bs = bs_positions[sites]
    offset = positions - bs
    direct_distance = np.linalg.norm(offset, axis=1)
    bearing_tx = np.arctan2(offset[:, 1], offset[:, 0])
    bearing_rx = np.arctan2(-offset[:, 1], -offset[:, 0])
    direct_theta_tx = np.arccos(
        np.clip(offset[:, 2] / np.maximum(direct_distance, 1e-9), -1.0, 1.0)
    )
    direct_theta_rx = np.arccos(
        np.clip(-offset[:, 2] / np.maximum(direct_distance, 1e-9), -1.0, 1.0)
    )

    excess = np.maximum(tau * LIGHT_SPEED_MPS - direct_distance[:, None], 0.0)
    log_excess = np.log1p(np.minimum(excess, 800.0)) / np.log1p(400.0)
    relative_angles = np.stack(
        (
            wrap_angle(angles[..., 1] - bearing_tx[:, None]),
            wrap_angle(angles[..., 3] - bearing_rx[:, None]),
            wrap_angle(angles[..., 0] - direct_theta_tx[:, None]),
            wrap_angle(angles[..., 2] - direct_theta_rx[:, None]),
        ),
        axis=-1,
    )
    first = np.zeros_like(depth)
    last = np.zeros_like(depth)
    active_depth = depth > 0
    first[active_depth] = interactions[..., 0][active_depth]
    row, slot = np.nonzero(active_depth)
    last[row, slot] = interactions[row, slot, depth[row, slot] - 1]
    first = np.clip(first, 0, 4)
    last = np.clip(last, 0, 4)
    source = np.clip(source, 0, 1)

    parts = [log_excess[..., None].astype(np.float32)]
    for axis in range(4):
        parts.append((1.25 * np.cos(relative_angles[..., axis]))[..., None].astype(np.float32))
        parts.append((1.25 * np.sin(relative_angles[..., axis]))[..., None].astype(np.float32))
    parts.append((depth[..., None] / 8.0).astype(np.float32))
    parts.append((0.65 * np.eye(2, dtype=np.float32)[source]))
    parts.append((0.70 * np.eye(5, dtype=np.float32)[first]))
    parts.append((0.70 * np.eye(5, dtype=np.float32)[last]))
    descriptor = np.concatenate(parts, axis=-1).astype(np.float32)
    descriptor[~mask] = 0.0
    if descriptor.shape != (4000, e85.PATHS_PER_POINT, PATH_DESCRIPTOR_DIM):
        raise RuntimeError(f"unexpected path descriptor shape: {descriptor.shape}")
    atomic_save_npy(PATH_DESCRIPTOR, descriptor.astype(np.float16))
    atomic_save_npy(PATH_VALID, mask)

    report = {
        "experiment": VERSION,
        "status": "static_inputs_ready",
        "base_embedding": {
            "path": PATHS.relative(TRAIN_BASE_EMBEDDING),
            "shape": list(base.shape),
            "report": base_report,
        },
        "scatter_embedding": {
            "path": PATHS.relative(TRAIN_SCATTER_EMBEDDING),
            "shape": list(scatter.shape),
            "report": scatter_report,
        },
        "path_descriptor": {
            "path": PATHS.relative(PATH_DESCRIPTOR),
            "shape": list(descriptor.shape),
            "valid_fraction": float(np.mean(mask)),
            "definition": (
                "relative excess delay, four relative angle sin/cos pairs, depth, "
                "base/scatter source, first interaction, and last interaction"
            ),
        },
        "seconds": float(time.time() - started),
    }
    STATIC_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def select_donors(
    allowed: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = torch.as_tensor(
        np.asarray(np.load(TRAIN_BASE_EMBEDDING, mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    scatter = torch.as_tensor(
        np.asarray(np.load(TRAIN_SCATTER_EMBEDDING, mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    donor_ids = np.full((len(positions), DONOR_COUNT), -1, dtype=np.int32)
    donor_similarity = np.zeros((len(positions), DONOR_COUNT), dtype=np.float32)
    donor_nonlocal = np.zeros((len(positions), DONOR_COUNT), dtype=bool)
    query_all = np.arange(len(positions), dtype=np.int64)

    with torch.inference_mode():
        for site_value in np.unique(sites):
            query = query_all[sites == site_value]
            pool = allowed[sites[allowed] == site_value]
            if len(pool) < DONOR_COUNT + 1:
                raise RuntimeError(f"insufficient donor pool for site {site_value}")
            score = torch.minimum(
                base[query] @ base[pool].T,
                scatter[query] @ scatter[pool].T,
            ).clamp(0.0, 1.0)
            query_tensor = torch.as_tensor(query, device=device)
            pool_tensor = torch.as_tensor(pool, device=device)
            self_mask = query_tensor[:, None] == pool_tensor[None, :]
            score = score.masked_fill(self_mask, -1.0)
            overall_values, overall_local = torch.topk(
                score, k=min(12, len(pool)), dim=1
            )
            xy_query = torch.as_tensor(
                positions[query, :2], dtype=torch.float32, device=device
            )
            xy_pool = torch.as_tensor(
                positions[pool, :2], dtype=torch.float32, device=device
            )
            distance = torch.cdist(xy_query, xy_pool)
            nonlocal_score = score.masked_fill(distance < NONLOCAL_DISTANCE_M, -1.0)
            nonlocal_values, nonlocal_local = torch.topk(
                nonlocal_score, k=min(TOP_NONLOCAL, len(pool)), dim=1
            )

            overall_ids = pool[overall_local.cpu().numpy()]
            overall_scores = overall_values.cpu().numpy()
            nonlocal_ids = pool[nonlocal_local.cpu().numpy()]
            nonlocal_scores = nonlocal_values.cpu().numpy()
            for local_row, global_row in enumerate(query):
                selected: list[tuple[int, float, bool]] = []
                seen: set[int] = set()
                for donor, value in zip(
                    overall_ids[local_row, :TOP_OVERALL],
                    overall_scores[local_row, :TOP_OVERALL],
                ):
                    if int(donor) not in seen and float(value) >= 0.0:
                        selected.append((int(donor), float(value), False))
                        seen.add(int(donor))
                for donor, value in zip(
                    nonlocal_ids[local_row], nonlocal_scores[local_row]
                ):
                    if int(donor) not in seen and float(value) >= 0.0:
                        selected.append((int(donor), float(value), True))
                        seen.add(int(donor))
                for donor, value in zip(
                    overall_ids[local_row], overall_scores[local_row]
                ):
                    if len(selected) >= DONOR_COUNT:
                        break
                    if int(donor) not in seen and float(value) >= 0.0:
                        is_nonlocal = bool(
                            np.linalg.norm(
                                positions[int(donor), :2] - positions[global_row, :2]
                            )
                            >= NONLOCAL_DISTANCE_M
                        )
                        selected.append((int(donor), float(value), is_nonlocal))
                        seen.add(int(donor))
                if len(selected) != DONOR_COUNT:
                    raise RuntimeError(
                        f"query {global_row} received {len(selected)} donors"
                    )
                for slot, (donor, value, is_nonlocal) in enumerate(selected):
                    donor_ids[global_row, slot] = donor
                    donor_similarity[global_row, slot] = value
                    donor_nonlocal[global_row, slot] = is_nonlocal
    return donor_ids, donor_similarity, donor_nonlocal


def build_memory(
    labels: e190.PathLabels,
    fallback_mass: np.ndarray,
    fallback_port: np.ndarray,
    query_coordinate: np.ndarray,
    allowed: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    candidate_mask: np.ndarray,
    device: torch.device,
    split: str,
    query_limit: int | None = None,
) -> MemoryArrays:
    started = time.time()
    donor_ids, donor_similarity, donor_nonlocal = select_donors(
        allowed, positions, sites, device
    )
    if not bool(np.isin(donor_ids, allowed).all()):
        raise RuntimeError(f"{split} donor pool leakage")
    if np.any(donor_ids == np.arange(len(donor_ids))[:, None]):
        raise RuntimeError(f"{split} donor pool contains query self")

    descriptor = torch.as_tensor(
        np.asarray(np.load(PATH_DESCRIPTOR, mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    valid = torch.as_tensor(
        np.asarray(np.load(PATH_VALID, mmap_mode="r"), dtype=bool), device=device
    )
    label_mass = torch.as_tensor(labels.mass, device=device)
    label_delta = torch.as_tensor(labels.delta, device=device)
    label_port = torch.as_tensor(labels.port, device=device)
    raw_mass = torch.as_tensor(fallback_mass, device=device)
    raw_port = torch.as_tensor(fallback_port, device=device)
    donor_ids_tensor = torch.as_tensor(donor_ids, dtype=torch.long, device=device)
    donor_similarity_tensor = torch.as_tensor(donor_similarity, device=device)
    donor_nonlocal_tensor = torch.as_tensor(
        donor_nonlocal, dtype=torch.float32, device=device
    )
    donor_label_valid = torch.as_tensor(labels.valid, device=device)
    query_valid = valid

    count = len(positions)
    process_count = count if query_limit is None else min(count, query_limit)
    features = np.zeros(
        (count, e85.PATHS_PER_POINT, MEMORY_FEATURE_DIM), dtype=np.float32
    )
    prior_mass = np.zeros((count, e85.PATHS_PER_POINT), dtype=np.float32)
    prior_coordinate = np.zeros(
        (count, e85.PATHS_PER_POINT, 3), dtype=np.float32
    )
    prior_port = np.zeros(
        (count, e85.PATHS_PER_POINT, PORTS), dtype=np.float32
    )
    active_support_values: list[float] = []
    reliability_values: list[float] = []
    cost_values: list[float] = []

    with torch.inference_mode():
        for start in range(0, process_count, MEMORY_BATCH_SIZE):
            end = min(start + MEMORY_BATCH_SIZE, process_count)
            rows = torch.arange(start, end, device=device)
            local_donor = donor_ids_tensor[start:end]
            batch = end - start
            query_desc = descriptor[rows]
            donor_desc = descriptor[local_donor]
            query_flat = (
                query_desc[:, None]
                .expand(batch, DONOR_COUNT, e85.PATHS_PER_POINT, PATH_DESCRIPTOR_DIM)
                .reshape(batch * DONOR_COUNT, e85.PATHS_PER_POINT, PATH_DESCRIPTOR_DIM)
            )
            donor_flat = donor_desc.reshape(
                batch * DONOR_COUNT, e85.PATHS_PER_POINT, PATH_DESCRIPTOR_DIM
            )
            cost = torch.cdist(query_flat, donor_flat).square().reshape(
                batch,
                DONOR_COUNT,
                e85.PATHS_PER_POINT,
                e85.PATHS_PER_POINT,
            )
            donor_valid = valid[local_donor]
            cost = cost.masked_fill(~donor_valid[:, :, None, :], float("inf"))
            match_cost, match_index = cost.min(dim=3)
            match_quality = torch.exp(-0.5 * match_cost.clamp_max(30.0))
            match_quality = torch.where(
                query_valid[rows, None], match_quality, torch.zeros_like(match_quality)
            )

            def gather_path(values: torch.Tensor) -> torch.Tensor:
                local = values[local_donor]
                suffix = local.shape[3:]
                index = match_index
                for _ in suffix:
                    index = index.unsqueeze(-1)
                if suffix:
                    index = index.expand(*match_index.shape, *suffix)
                return torch.gather(local, 2, index)

            mapped_mass = gather_path(label_mass)
            mapped_delta = gather_path(label_delta)
            mapped_port = gather_path(label_port)
            mapped_raw_mass = gather_path(raw_mass)
            mapped_raw_port = gather_path(raw_port)

            point_weight = donor_similarity_tensor[start:end, :, None]
            weight = point_weight * match_quality
            total_weight = weight.sum(dim=1).clamp_min(1e-8)
            active = (mapped_mass > 0.0) & donor_label_valid[local_donor, None]
            active_weight = weight * active
            active_total = active_weight.sum(dim=1)
            active_safe = active_total.clamp_min(1e-8)

            existence = active_total / total_weight
            official_mass = (weight * mapped_mass * active).sum(dim=1) / total_weight
            donor_raw_mass = (weight * mapped_raw_mass).sum(dim=1) / total_weight
            log_mass_ratio = torch.log(
                (official_mass + 1e-6) / (donor_raw_mass + 1e-6)
            ).clamp(-4.0, 4.0)
            delta_mean = (active_weight[..., None] * mapped_delta).sum(dim=1) / active_safe[..., None]
            delta_variance = (
                active_weight[..., None]
                * (mapped_delta - delta_mean[:, None]).square()
            ).sum(dim=1) / active_safe[..., None]
            delta_std = torch.sqrt(delta_variance.clamp_min(0.0))
            official_port = (
                active_weight[..., None] * mapped_port
            ).sum(dim=1) / active_safe[..., None]
            donor_raw_port = (
                active_weight[..., None] * mapped_raw_port
            ).sum(dim=1) / active_safe[..., None]
            port_log_ratio = torch.log(
                (official_port + 1e-5) / (donor_raw_port + 1e-5)
            ).clamp(-2.0, 2.0)
            mean_similarity = (
                weight * point_weight
            ).sum(dim=1) / total_weight
            mean_quality = (weight * match_quality).sum(dim=1) / total_weight
            support_fraction = (match_quality >= 0.20).float().mean(dim=1)
            nonlocal_fraction = (
                weight * donor_nonlocal_tensor[start:end, :, None]
            ).sum(dim=1) / total_weight
            reliability = (
                mean_similarity
                * mean_quality
                * torch.clamp(support_fraction / 0.50, 0.0, 1.0)
            ).clamp(0.0, 1.0)

            query_mass = raw_mass[rows]
            candidate_log_mass = torch.log(query_mass.clamp_min(1e-12)) + reliability * log_mass_ratio
            candidate_log_mass = candidate_log_mass.masked_fill(
                ~query_valid[rows], -30.0
            )
            local_prior_mass = torch.softmax(candidate_log_mass, dim=1)
            local_prior_coordinate = torch.remainder(
                torch.as_tensor(query_coordinate[start:end], device=device)
                + reliability[..., None] * delta_mean,
                torch.as_tensor(PERIODS, device=device),
            )
            query_port = raw_port[rows]
            local_prior_port = torch.softmax(
                torch.log(query_port.clamp_min(1e-8))
                + reliability[..., None] * port_log_ratio,
                dim=2,
            )

            scale = torch.as_tensor((2.0, 4.0, 2.0), device=device)
            local_features = torch.cat(
                (
                    existence[..., None],
                    (log_mass_ratio / 4.0)[..., None],
                    delta_mean / scale,
                    delta_std / scale,
                    reliability[..., None],
                    mean_similarity[..., None],
                    mean_quality[..., None],
                    support_fraction[..., None],
                    nonlocal_fraction[..., None],
                    port_log_ratio / 2.0,
                ),
                dim=2,
            )
            if local_features.shape[2] != MEMORY_FEATURE_DIM:
                raise RuntimeError(
                    f"memory feature width mismatch: {local_features.shape}"
                )
            local_features = torch.where(
                query_valid[rows, :, None],
                local_features,
                torch.zeros_like(local_features),
            )
            features[start:end] = local_features.cpu().numpy().astype(np.float32)
            prior_mass[start:end] = local_prior_mass.cpu().numpy().astype(np.float32)
            prior_coordinate[start:end] = (
                local_prior_coordinate.cpu().numpy().astype(np.float32)
            )
            prior_port[start:end] = local_prior_port.cpu().numpy().astype(np.float32)
            active_support_values.append(float(existence.mean().item()))
            reliability_values.append(float(reliability.mean().item()))
            finite_cost = match_cost[torch.isfinite(match_cost)]
            if finite_cost.numel():
                cost_values.append(float(finite_cost.mean().item()))
            del cost, match_cost, match_index

    if not np.isfinite(features).all():
        raise FloatingPointError(f"non-finite {split} memory features")
    if not np.isfinite(prior_mass).all() or not np.isfinite(prior_port).all():
        raise FloatingPointError(f"non-finite {split} memory priors")
    report = {
        "split": split,
        "allowed_donor_count": int(len(allowed)),
        "donor_count_per_query": DONOR_COUNT,
        "top_overall": TOP_OVERALL,
        "top_nonlocal": TOP_NONLOCAL,
        "nonlocal_distance_m": NONLOCAL_DISTANCE_M,
        "donor_similarity": {
            "mean": float(np.mean(donor_similarity)),
            "median": float(np.median(donor_similarity)),
            "p10": float(np.quantile(donor_similarity, 0.10)),
        },
        "nonlocal_fraction": float(np.mean(donor_nonlocal)),
        "mapped_existence_mean": float(np.mean(active_support_values)),
        "reliability_mean": float(np.mean(reliability_values)),
        "path_match_cost_mean": float(np.mean(cost_values)),
        "seconds": float(time.time() - started),
        "processed_query_count": int(process_count),
        "zero_donor_label_overlap_with_forbidden_rows": True,
    }
    return MemoryArrays(
        features,
        prior_mass,
        prior_coordinate,
        prior_port,
        donor_ids,
        donor_similarity,
        report,
    )


def save_memory(path: Path, memory: MemoryArrays) -> None:
    atomic_save_npz(
        path,
        features=memory.features.astype(np.float16),
        prior_mass=memory.prior_mass,
        prior_coordinate=memory.prior_coordinate,
        prior_port=memory.prior_port.astype(np.float16),
        donor_ids=memory.donor_ids,
        donor_similarity=memory.donor_similarity.astype(np.float16),
        report_json=np.asarray(json.dumps(memory.report, ensure_ascii=False)),
    )


def load_memory(path: Path) -> MemoryArrays:
    payload = np.load(path, allow_pickle=False)
    return MemoryArrays(
        payload["features"].astype(np.float32),
        payload["prior_mass"].astype(np.float32),
        payload["prior_coordinate"].astype(np.float32),
        payload["prior_port"].astype(np.float32),
        payload["donor_ids"].astype(np.int32),
        payload["donor_similarity"].astype(np.float32),
        json.loads(str(payload["report_json"].item())),
    )


def memory_store(
    coordinate: np.ndarray, memory: MemoryArrays
) -> e85.DataStore:
    store = e85.DataStore()
    original_base = np.asarray(store.base, dtype=np.float32)
    store.base = np.concatenate((original_base, memory.features), axis=2).astype(
        np.float32
    )
    store.fraction = memory.prior_mass
    store.presence = np.full(memory.prior_mass.shape, 24.0, dtype=np.float32)
    store.coordinate = memory.prior_coordinate
    store.port = memory.prior_port.reshape(
        len(memory.prior_port), e85.PATHS_PER_POINT, e85.MP, e85.N_UE
    )
    if store.base.shape[2] != original_base.shape[2] + MEMORY_FEATURE_DIM:
        raise RuntimeError("memory feature concatenation failed")
    return store


def labels_for_memory_prior(
    labels: e190.PathLabels,
    original_coordinate: np.ndarray,
    prior_coordinate: np.ndarray,
) -> e190.PathLabels:
    target_coordinate = np.mod(original_coordinate + labels.delta, PERIODS)
    delta = np.empty_like(labels.delta, dtype=np.float32)
    for axis, period in enumerate(PERIODS):
        delta[..., axis] = e190.periodic_delta(
            target_coordinate[..., axis],
            prior_coordinate[..., axis],
            float(period),
        ).astype(np.float32)
    delta[~labels.valid] = 0.0
    return e190.PathLabels(
        mass=labels.mass,
        delta=delta,
        port=labels.port,
        matched_count=labels.matched_count,
        matched_fraction=labels.matched_fraction,
        valid=labels.valid,
    )


def prior_endpoint(
    store: e85.DataStore, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = np.asarray(store.fraction[rows], dtype=np.float32)
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), 1e-12)
    coordinate = np.mod(
        np.asarray(store.coordinate[rows], dtype=np.float32), PERIODS
    ).astype(np.float32)
    port = np.asarray(store.port[rows], dtype=np.float32)
    port /= np.maximum(port.sum(axis=(2, 3), keepdims=True), 1e-12)
    return mass, coordinate, port


class IndependentEvaluator:
    def __init__(self, device: torch.device) -> None:
        metadata = np.load(e85.OOF_META, allow_pickle=False)
        self.query_ids = metadata["query_ids"].astype(np.int64)
        self.fold_id = metadata["fold_id"].astype(np.int8)
        self.lookup = np.full(4000, -1, dtype=np.int64)
        self.lookup[self.query_ids] = np.arange(len(self.query_ids), dtype=np.int64)
        self.base = np.load(INDEPENDENT_V11, mmap_mode="r")
        self.truth_all = np.load(e86.TRAIN_CHANNEL, mmap_mode="r")
        self.device = device

    def evaluate(
        self,
        global_rows: np.ndarray,
        endpoint_pas: np.ndarray,
        endpoint_pdp: np.ndarray,
        collect_channel: bool = False,
    ) -> tuple[dict[str, float | int], np.ndarray | None]:
        local = self.lookup[global_rows]
        if np.any(local < 0):
            raise ValueError("E193 complete evaluation requires canonical rows")
        base = torch.as_tensor(
            np.array(self.base[local], dtype=np.complex64, copy=True),
            device=self.device,
        )
        truth = torch.as_tensor(
            np.asarray(self.truth_all[global_rows], dtype=np.complex64),
            device=self.device,
        )
        base_pas = pas_distribution(base)
        base_pdp = pdp_distribution(base)
        truth_pas = pas_distribution(truth)
        truth_pdp = pdp_distribution(truth)
        spec = e86.Candidate(
            endpoint="e193_memory",
            mode="sqrt",
            pas_beta=0.20,
            pdp_beta=0.20,
            order="pas_pdp",
            cycles=4,
        )
        with torch.inference_mode():
            candidate = e86.make_candidate(
                base,
                base_pas,
                base_pdp,
                torch.as_tensor(endpoint_pas, device=self.device),
                torch.as_tensor(endpoint_pdp, device=self.device),
                spec,
            )
            values = point_metrics(candidate, truth, truth_pas, truth_pdp)
        metric = statistics(*values)
        channel = (
            candidate.cpu().numpy().astype(np.complex64)
            if collect_channel
            else None
        )
        return metric, channel


def train_screen(
    store: e85.DataStore,
    labels: e190.PathLabels,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    evaluator: IndependentEvaluator,
    config: e190.Config,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | None, int, np.ndarray, np.ndarray, list[dict[str, object]]]:
    mean, std = store.normalization(fit_rows)
    mass, coordinate, port = prior_endpoint(store, validation_rows)
    pas, pdp = e190.render_endpoint(mass, coordinate, port, device)
    baseline, _ = evaluator.evaluate(validation_rows, pas, pdp)
    best_score = float(baseline["score"])
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, object]] = [
        {"epoch": 0, "validation": baseline, "source": "memory_prior"}
    ]

    torch.manual_seed(config.random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_state)
    model = e190.make_model(store, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=config.learning_rate * 0.08,
    )
    rng = np.random.default_rng(config.random_state)
    for epoch in range(1, config.epochs + 1):
        train = e190.train_epoch(
            model,
            optimizer,
            store,
            labels,
            fit_rows,
            mean,
            std,
            config,
            device,
            rng,
        )
        scheduler.step()
        record: dict[str, object] = {"epoch": epoch, "train": train}
        if epoch >= config.minimum_epochs and epoch % config.validation_interval == 0:
            mass, coordinate, port = e190.endpoint_arrays(
                model,
                store,
                validation_rows,
                mean,
                std,
                device,
                config.batch_size,
            )
            pas, pdp = e190.render_endpoint(mass, coordinate, port, device)
            selected, _ = evaluator.evaluate(validation_rows, pas, pdp)
            record["validation"] = selected
            record["delta_vs_prior"] = float(selected["score"] - baseline["score"])
            if float(selected["score"]) > best_score:
                best_score = float(selected["score"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch % config.validation_interval == 0:
            print(
                f"E193 outer={config.outer} epoch={epoch}/{config.epochs} "
                f"mass={train['mass_affinity']:.6f} "
                f"pas={train['pas']:.6f} pdp={train['pdp']:.6f} "
                f"best={best_score:.6f}",
                flush=True,
            )
    del model
    return best_state, best_epoch, mean, std, history


def train_fold(args: argparse.Namespace) -> dict[str, object]:
    if args.outer not in range(5):
        raise ValueError("outer must be in [0,4]")
    started = time.time()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    prepare_static_inputs(False)
    if not INDEPENDENT_V11.is_file():
        raise FileNotFoundError(
            f"independent E193 base missing: {PATHS.relative(INDEPENDENT_V11)}"
        )
    for forbidden in FORBIDDEN_RUNTIME_INPUTS:
        if forbidden == INDEPENDENT_V11:
            raise RuntimeError("independent base aliases a forbidden historical input")

    config = e190.Config(
        outer=args.outer,
        validation_fold=(args.outer + 1) % 5,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    evaluator = IndependentEvaluator(device)
    all_rows = np.arange(4000, dtype=np.int64)
    strict_outer = evaluator.query_ids[evaluator.fold_id == args.outer]
    strict_validation = evaluator.query_ids[
        evaluator.fold_id == config.validation_fold
    ]
    held_mask = np.zeros(4000, dtype=bool)
    validation_mask = np.zeros(4000, dtype=bool)
    held_mask[strict_outer] = True
    validation_mask[strict_validation] = True
    screen_fit = all_rows[~held_mask & ~validation_mask]
    refit_rows = all_rows[~held_mask]
    setup, positions, _, sites, _ = load_positions()
    del setup
    positions = np.asarray(positions, dtype=np.float64)
    sites = np.asarray(sites, dtype=np.int8)

    phase_time: dict[str, float] = {}
    load_start = time.time()
    raw_store = e85.DataStore()
    fallback_mass, fallback_port = e190.raw_prior(raw_store)
    candidate_mask = np.asarray(
        np.load(e85.PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool
    )
    component = np.load(e190.COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    base_coordinate = np.asarray(raw_store.coordinate, dtype=np.float32)
    mapping = np.load(e190.MAPPING_PARAMETERS, allow_pickle=False)
    mapping_data = e98.MappingData(device)
    phase_time["load"] = time.time() - load_start

    screen_start = time.time()
    screen_coordinate = e190.corrected_coordinates(
        base_coordinate,
        mapping_data,
        mapping["screen_latent"][args.outer].astype(np.float32),
        device,
    )
    screen_labels, screen_label_report = e190.build_labels(
        screen_coordinate,
        candidate_mask,
        truth_coordinate,
        truth_coefficient,
        point_power,
        fallback_mass,
        fallback_port,
        config,
    )
    screen_memory_file = fold_memory_path(args.outer, "screen")
    if screen_memory_file.is_file() and not args.force:
        screen_memory = load_memory(screen_memory_file)
    else:
        screen_memory = build_memory(
            screen_labels,
            fallback_mass,
            fallback_port,
            screen_coordinate,
            screen_fit,
            positions,
            sites,
            candidate_mask,
            device,
            "screen",
        )
        save_memory(screen_memory_file, screen_memory)
    if np.intersect1d(
        np.unique(screen_memory.donor_ids[strict_validation]), strict_validation
    ).size:
        raise RuntimeError("screen validation label leakage through donor memory")
    if np.intersect1d(
        np.unique(screen_memory.donor_ids[strict_validation]), strict_outer
    ).size:
        raise RuntimeError("screen outer leakage through donor memory")
    screen_store = memory_store(screen_coordinate, screen_memory)
    screen_training_labels = labels_for_memory_prior(
        screen_labels, screen_coordinate, screen_memory.prior_coordinate
    )
    phase_time["screen_labels_and_memory"] = time.time() - screen_start

    train_start = time.time()
    best_state, best_epoch, _, _, screen_history = train_screen(
        screen_store,
        screen_training_labels,
        screen_fit,
        strict_validation,
        evaluator,
        config,
        device,
    )
    phase_time["screen_training"] = time.time() - train_start
    del screen_store, screen_memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    refit_start = time.time()
    refit_coordinate = e190.corrected_coordinates(
        base_coordinate,
        mapping_data,
        mapping["refit_latent"][args.outer].astype(np.float32),
        device,
    )
    refit_labels, refit_label_report = e190.build_labels(
        refit_coordinate,
        candidate_mask,
        truth_coordinate,
        truth_coefficient,
        point_power,
        fallback_mass,
        fallback_port,
        config,
    )
    refit_memory_file = fold_memory_path(args.outer, "refit")
    if refit_memory_file.is_file() and not args.force:
        refit_memory = load_memory(refit_memory_file)
    else:
        refit_memory = build_memory(
            refit_labels,
            fallback_mass,
            fallback_port,
            refit_coordinate,
            refit_rows,
            positions,
            sites,
            candidate_mask,
            device,
            "refit",
        )
        save_memory(refit_memory_file, refit_memory)
    if np.intersect1d(
        np.unique(refit_memory.donor_ids[strict_outer]), strict_outer
    ).size:
        raise RuntimeError("outer label leakage through donor memory")
    refit_store = memory_store(refit_coordinate, refit_memory)
    refit_training_labels = labels_for_memory_prior(
        refit_labels, refit_coordinate, refit_memory.prior_coordinate
    )
    phase_time["refit_labels_and_memory"] = time.time() - refit_start

    refit_train_start = time.time()
    if best_state is None or best_epoch == 0:
        mass, coordinate, port = prior_endpoint(refit_store, strict_outer)
        refit_history: list[dict[str, float]] = []
        checkpoint_payload = None
    else:
        mean_refit, std_refit = refit_store.normalization(refit_rows)
        model, refit_history = e190.refit_model(
            refit_store,
            refit_training_labels,
            refit_rows,
            mean_refit,
            std_refit,
            best_epoch,
            config,
            device,
        )
        mass, coordinate, port = e190.endpoint_arrays(
            model,
            refit_store,
            strict_outer,
            mean_refit,
            std_refit,
            device,
            config.batch_size,
        )
        checkpoint_payload = {
            "state_dict": model.state_dict(),
            "configuration": asdict(config),
            "feature_mean": mean_refit,
            "feature_std": std_refit,
            "best_epoch": best_epoch,
            "outer": args.outer,
            "mapping_latent": mapping["refit_latent"][args.outer].astype(np.float32),
            "memory_feature_dim": MEMORY_FEATURE_DIM,
        }
        del model
    phase_time["refit_training"] = time.time() - refit_train_start

    score_start = time.time()
    endpoint_pas, endpoint_pdp = e190.render_endpoint(
        mass, coordinate, port, device
    )
    selected, final_channel = evaluator.evaluate(
        strict_outer, endpoint_pas, endpoint_pdp, collect_channel=True
    )
    if final_channel is None:
        raise RuntimeError("E193 evaluator did not return a channel")
    v15_payload = json.loads(V15_REPORT.read_text(encoding="utf-8"))
    v15_fold = next(
        row for row in v15_payload["folds"] if int(row["outer"]) == args.outer
    )["selected"]
    delta = {
        key: float(selected[key] - v15_fold[key])
        for key in ("pas", "pdp", "nmse_score", "score")
    }
    phase_time["render_and_score"] = time.time() - score_start

    atomic_save_npz(
        fold_endpoint_path(args.outer),
        query_ids=strict_outer.astype(np.int32),
        mass=mass,
        coordinate=coordinate,
        port=port,
    )
    atomic_save_npy(fold_channel_path(args.outer), final_channel.astype(np.complex64))
    if checkpoint_payload is not None:
        atomic_torch_save(fold_checkpoint_path(args.outer), checkpoint_payload)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        peak_memory = float(torch.cuda.max_memory_allocated(device) / 2**30)
    else:
        gpu_name = "cpu"
        peak_memory = 0.0
    report = {
        "experiment": VERSION,
        "status": "fold_complete",
        "configuration": asdict(config),
        "split": {
            "outer": args.outer,
            "validation_fold": config.validation_fold,
            "screen_fit_count": int(len(screen_fit)),
            "refit_count": int(len(refit_rows)),
            "outer_count": int(len(strict_outer)),
            "screen_donor_forbidden_folds": [args.outer, config.validation_fold],
            "refit_donor_forbidden_folds": [args.outer],
            "zero_validation_or_outer_label_overlap": True,
        },
        "labels": {
            "screen": screen_label_report,
            "refit": refit_label_report,
        },
        "memory": {
            "screen": load_memory(screen_memory_file).report,
            "refit": refit_memory.report,
        },
        "training": {
            "best_epoch": best_epoch,
            "screen_history": screen_history,
            "refit_history": refit_history,
        },
        "selected": selected,
        "v15_fold_baseline": v15_fold,
        "delta_vs_v15": delta,
        "data_lineage": {
            "independent_base": PATHS.relative(INDEPENDENT_V11),
            "historical_complete_channel_read": False,
            "historical_endpoint_read": False,
            "forbidden_runtime_inputs": [
                PATHS.relative(path) for path in FORBIDDEN_RUNTIME_INPUTS
            ],
            "official_train_h_used_for": (
                "strict donor labels, candidate model supervision, and final scoring"
            ),
            "sionna_used_for": (
                "target/donor candidate paths and propagation-state retrieval only; "
                "absolute Sionna complex phase is not used"
            ),
        },
        "outputs": {
            "channel": PATHS.relative(fold_channel_path(args.outer)),
            "endpoint": PATHS.relative(fold_endpoint_path(args.outer)),
            "checkpoint": (
                PATHS.relative(fold_checkpoint_path(args.outer))
                if checkpoint_payload is not None
                else None
            ),
        },
        "performance": {
            "device": gpu_name,
            "peak_torch_memory_gib": peak_memory,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "phase_seconds": phase_time,
            "measured_bottleneck": max(phase_time, key=phase_time.get),
            "wall_seconds": float(time.time() - started),
        },
    }
    fold_report_path(args.outer).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def aggregate(device: torch.device, batch_size: int) -> dict[str, object]:
    started = time.time()
    metadata = np.load(e85.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    lookup = np.full(4000, -1, dtype=np.int64)
    lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    channel = np.empty((500, 256, 4, 192), dtype=np.complex64)
    fold_reports = []
    for outer in range(5):
        report_path = fold_report_path(outer)
        endpoint_path = fold_endpoint_path(outer)
        local_channel_path = fold_channel_path(outer)
        missing = [
            PATHS.relative(path)
            for path in (report_path, endpoint_path, local_channel_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing E193 fold outputs: {missing}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        endpoint = np.load(endpoint_path, allow_pickle=False)
        local_ids = endpoint["query_ids"].astype(np.int64)
        expected = query_ids[fold_id == outer]
        if not np.array_equal(local_ids, expected):
            raise RuntimeError(f"outer{outer} query identity mismatch")
        local_channel = np.load(local_channel_path, mmap_mode="r")
        if local_channel.shape != (len(local_ids), 256, 4, 192):
            raise RuntimeError(f"outer{outer} channel shape mismatch")
        channel[lookup[local_ids]] = np.asarray(local_channel, dtype=np.complex64)
        fold_reports.append(report)
    if not np.isfinite(channel.real).all() or not np.isfinite(channel.imag).all():
        raise FloatingPointError("non-finite E193 fixed500 channel")
    atomic_save_npy(FULL_CHANNEL, channel)

    truth_all = np.load(e86.TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    prediction = torch.as_tensor(channel, device=device)
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    with torch.inference_mode():
        values = point_metrics(prediction, truth, truth_pas, truth_pdp)
    selected = statistics(*values)
    folds = [statistics(*values, fold_id == outer) for outer in range(5)]

    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))
    v15_selected = v15["selected"]
    v15_folds = {
        int(row["outer"]): row["selected"] for row in v15["folds"]
    }
    deltas = [
        float(folds[outer]["score"] - v15_folds[outer]["score"])
        for outer in range(5)
    ]
    overall_delta = {
        key: float(selected[key] - v15_selected[key])
        for key in ("pas", "pdp", "nmse_score", "score")
    }
    positive_folds = int(sum(delta > 0.0 for delta in deltas))
    worst_fold_delta = float(min(deltas))
    meaningful_passed = bool(
        float(selected["score"]) >= 0.695
        and positive_folds >= 4
        and worst_fold_delta >= -0.001
    )
    formal_passed = bool(
        float(selected["score"]) >= 0.700
        and positive_folds >= 4
        and worst_fold_delta >= -0.001
    )
    performance = {
        "fold_wall_seconds": [
            float(report["performance"]["wall_seconds"]) for report in fold_reports
        ],
        "fold_devices": [
            str(report["performance"]["device"]) for report in fold_reports
        ],
        "fold_peak_torch_memory_gib": [
            float(report["performance"]["peak_torch_memory_gib"])
            for report in fold_reports
        ],
        "measured_bottlenecks": [
            str(report["performance"]["measured_bottleneck"])
            for report in fold_reports
        ],
        "aggregate_seconds": float(time.time() - started),
    }
    report = {
        "experiment": VERSION,
        "status": (
            "formal_gate_passed"
            if formal_passed
            else "meaningful_gate_passed"
            if meaningful_passed
            else "failed_and_retired"
        ),
        "selected": selected,
        "v15_baseline": v15_selected,
        "delta_vs_v15": overall_delta,
        "folds": [
            {
                "outer": outer,
                "selected": folds[outer],
                "v15_baseline": v15_folds[outer],
                "score_delta": deltas[outer],
            }
            for outer in range(5)
        ],
        "gate": {
            "meaningful_score": 0.695,
            "formal_score": 0.700,
            "minimum_positive_folds": 4,
            "minimum_worst_fold_delta": -0.001,
            "positive_folds": positive_folds,
            "worst_fold_delta": worst_fold_delta,
            "meaningful_passed": meaningful_passed,
            "formal_passed": formal_passed,
        },
        "data_lineage": {
            "historical_complete_channel_read": False,
            "historical_endpoint_read": False,
            "independent_base": PATHS.relative(INDEPENDENT_V11),
            "official_train_channel": PATHS.relative(e86.TRAIN_CHANNEL),
        },
        "output": {
            "channel": PATHS.relative(FULL_CHANNEL),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": True,
        },
        "performance": performance,
    }
    FULL_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        report = prepare_static_inputs(args.force)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    if args.stage == "fold":
        train_fold(args)
        return
    aggregate(torch.device(args.device), args.batch_size)


if __name__ == "__main__":
    main()
