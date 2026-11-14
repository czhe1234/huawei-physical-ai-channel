#!/usr/bin/env python3
"""Audit whether V7 spectral corrections are predictable in peak-relative frames."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from analyze_e160_v7_correction_capacity import (
    FULL_CORE_PAS,
    FULL_CORE_PDP,
    FULL_TRUTH_PAS,
    FULL_TRUTH_PDP,
    apply_lifecycle,
    grouped_endpoints,
    score_channel,
)
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)
from analyze_v7_stacked_lifecycle_overlay import OOF_SCALED
from data import load_positions
from official_metrics import OfficialMetricAccumulator
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META
from run_v6_regime_component_lifecycle import REGIME_CACHE
from train_full_path_pdp_link_gate import FOLD_IDS


VERSION = "Round2-E162-v7-relative-peak-relation"
E160_DIR = PATHS.artifacts / "round2" / "e160_full4000_v7_supervision"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e162_v7_relative_peak_relation"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e162_v7_relative_peak_relation"
REPORT = ANALYSIS_DIR / "e162_v7_relative_peak_relation.json"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
FULL_CORE = E160_DIR / "full4000_v7_core.npy"
CANONICAL_V7_COMPLETE = E160_DIR / "canonical_v7_complete.npy"

ALIGNED_PAS_RESIDUAL = ARTIFACT_DIR / "full4000_aligned_pas_residual_f16.npy"
ALIGNED_PDP_RESIDUAL = ARTIFACT_DIR / "full4000_aligned_pdp_residual_f16.npy"
ABSOLUTE_PAS_RESIDUAL = ARTIFACT_DIR / "full4000_absolute_pas_residual_f16.npy"
ABSOLUTE_PDP_RESIDUAL = ARTIFACT_DIR / "full4000_absolute_pdp_residual_f16.npy"

PAS_GROUPS = 4 * 8
PAS_CELLS = 2 * 16 * 8
PDP_GROUPS = 2 * 4 * 2 * 4
PDP_CELLS = 192
TINY = 1e-30
V7_SCORE = 0.644456345060792
V12_SCORE = 0.677521
IDENTITY_TOLERANCE = 5e-6


@dataclass(frozen=True)
class Config:
    neighbors: int = 32
    batch_size: int = 64
    metric_batch_size: int = 8
    group_chunk: int = 4
    cycles: int = 4
    seed: int = 260_810_162


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument(
        "--metric-batch-size", type=int, default=Config.metric_batch_size
    )
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def normalize_numpy(value: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = output.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        output,
        np.maximum(total, TINY),
        out=np.zeros_like(output),
        where=total > 0.0,
    )


def align_pas(
    base: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_spectrum = base.permute(0, 4, 5, 1, 2, 3).contiguous()
    truth_spectrum = truth.permute(0, 4, 5, 1, 2, 3).contiguous()
    batch, ue, frequency, polarization, height, vertical = base_spectrum.shape
    strongest = base_spectrum.reshape(batch, ue, frequency, -1).argmax(dim=-1)
    center_h = torch.div(strongest, vertical, rounding_mode="floor") % height
    center_v = strongest % vertical

    output_h = torch.arange(height, device=base.device)
    source_h = (
        output_h[None, None, None, :] - height // 2 + center_h[..., None]
    ) % height
    gather_h = source_h[:, :, :, None, :, None].expand(
        batch, ue, frequency, polarization, height, vertical
    )
    base_aligned = torch.gather(base_spectrum, 4, gather_h)
    truth_aligned = torch.gather(truth_spectrum, 4, gather_h)

    output_v = torch.arange(vertical, device=base.device)
    source_v = (
        output_v[None, None, None, :] - vertical // 2 + center_v[..., None]
    ) % vertical
    gather_v = source_v[:, :, :, None, None, :].expand(
        batch, ue, frequency, polarization, height, vertical
    )
    base_aligned = torch.gather(base_aligned, 5, gather_v)
    truth_aligned = torch.gather(truth_aligned, 5, gather_v)
    centers = torch.stack((center_h, center_v), dim=-1)
    return base_aligned, truth_aligned, centers


def inverse_align_pas(value: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    batch, ue, frequency, polarization, height, vertical = value.shape
    center_h = centers[..., 0]
    center_v = centers[..., 1]
    original_h = torch.arange(height, device=value.device)
    source_h = (
        original_h[None, None, None, :] + height // 2 - center_h[..., None]
    ) % height
    gather_h = source_h[:, :, :, None, :, None].expand(
        batch, ue, frequency, polarization, height, vertical
    )
    output = torch.gather(value, 4, gather_h)
    original_v = torch.arange(vertical, device=value.device)
    source_v = (
        original_v[None, None, None, :] + vertical // 2 - center_v[..., None]
    ) % vertical
    gather_v = source_v[:, :, :, None, None, :].expand(
        batch, ue, frequency, polarization, height, vertical
    )
    output = torch.gather(output, 5, gather_v)
    return output.permute(0, 3, 4, 5, 1, 2).contiguous()


def align_pdp(
    base: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(base)
    base_spectrum = base.reshape(batch, PDP_GROUPS, PDP_CELLS)
    truth_spectrum = truth.reshape(batch, PDP_GROUPS, PDP_CELLS)
    center = base_spectrum.argmax(dim=-1)
    output_d = torch.arange(PDP_CELLS, device=base.device)
    source_d = (
        output_d[None, None, :] - PDP_CELLS // 2 + center[..., None]
    ) % PDP_CELLS
    return (
        torch.gather(base_spectrum, 2, source_d),
        torch.gather(truth_spectrum, 2, source_d),
        center,
    )


def inverse_align_pdp(value: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    original_d = torch.arange(PDP_CELLS, device=value.device)
    source_d = (
        original_d[None, None, :] + PDP_CELLS // 2 - centers[..., None]
    ) % PDP_CELLS
    output = torch.gather(value, 2, source_d)
    return output.reshape(len(value), 2, 4, 2, 4, PDP_CELLS)


def cache_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "aligned_pas": (4000, PAS_GROUPS, PAS_CELLS),
        "aligned_pdp": (4000, PDP_GROUPS, PDP_CELLS),
        "absolute_pas": (4000, PAS_GROUPS, PAS_CELLS),
        "absolute_pdp": (4000, PDP_GROUPS, PDP_CELLS),
    }


def cache_valid(path, shape: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    value = np.load(path, mmap_mode="r")
    return value.shape == shape and value.dtype == np.float16


@torch.inference_mode()
def build_residual_caches(
    device: torch.device, batch_size: int, force: bool
) -> dict[str, np.ndarray]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "aligned_pas": ALIGNED_PAS_RESIDUAL,
        "aligned_pdp": ALIGNED_PDP_RESIDUAL,
        "absolute_pas": ABSOLUTE_PAS_RESIDUAL,
        "absolute_pdp": ABSOLUTE_PDP_RESIDUAL,
    }
    shapes = cache_shapes()
    if not force and all(cache_valid(paths[key], shapes[key]) for key in paths):
        return {key: np.load(path, mmap_mode="r") for key, path in paths.items()}

    core_pas = np.load(FULL_CORE_PAS, mmap_mode="r")
    truth_pas = np.load(FULL_TRUTH_PAS, mmap_mode="r")
    core_pdp = np.load(FULL_CORE_PDP, mmap_mode="r")
    truth_pdp = np.load(FULL_TRUTH_PDP, mmap_mode="r")
    outputs = {
        key: np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shapes[key])
        for key, path in paths.items()
    }

    for start in range(0, 4000, batch_size):
        end = min(start + batch_size, 4000)
        base_pas = torch.as_tensor(
            np.asarray(core_pas[start:end], dtype=np.float32), device=device
        )
        target_pas = torch.as_tensor(
            np.asarray(truth_pas[start:end], dtype=np.float32), device=device
        )
        aligned_base_pas, aligned_truth_pas, _ = align_pas(base_pas, target_pas)
        absolute_base_pas = base_pas.permute(0, 4, 5, 1, 2, 3).reshape(
            end - start, PAS_GROUPS, PAS_CELLS
        )
        absolute_truth_pas = target_pas.permute(0, 4, 5, 1, 2, 3).reshape(
            end - start, PAS_GROUPS, PAS_CELLS
        )
        outputs["aligned_pas"][start:end] = (
            aligned_truth_pas.clamp_min(0.0).sqrt()
            - aligned_base_pas.clamp_min(0.0).sqrt()
        ).reshape(end - start, PAS_GROUPS, PAS_CELLS).cpu().numpy().astype(np.float16)
        outputs["absolute_pas"][start:end] = (
            absolute_truth_pas.clamp_min(0.0).sqrt()
            - absolute_base_pas.clamp_min(0.0).sqrt()
        ).cpu().numpy().astype(np.float16)

        base_pdp = torch.as_tensor(
            np.asarray(core_pdp[start:end], dtype=np.float32), device=device
        )
        target_pdp = torch.as_tensor(
            np.asarray(truth_pdp[start:end], dtype=np.float32), device=device
        )
        aligned_base_pdp, aligned_truth_pdp, _ = align_pdp(base_pdp, target_pdp)
        outputs["aligned_pdp"][start:end] = (
            aligned_truth_pdp.clamp_min(0.0).sqrt()
            - aligned_base_pdp.clamp_min(0.0).sqrt()
        ).cpu().numpy().astype(np.float16)
        outputs["absolute_pdp"][start:end] = (
            target_pdp.reshape(end - start, PDP_GROUPS, PDP_CELLS).clamp_min(0.0).sqrt()
            - base_pdp.reshape(end - start, PDP_GROUPS, PDP_CELLS).clamp_min(0.0).sqrt()
        ).cpu().numpy().astype(np.float16)

        for output in outputs.values():
            output.flush()
        print(f"E162 residual cache {end}/4000", flush=True)
    return {key: np.load(path, mmap_mode="r") for key, path in paths.items()}


def neighbor_table(
    query_rows: np.ndarray,
    donor_rows: np.ndarray,
    positions: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = min(neighbors, len(donor_rows))
    distance, local = cKDTree(positions[donor_rows, :2]).query(
        positions[query_rows, :2], k=keep
    )
    if keep == 1:
        distance = distance[:, None]
        local = local[:, None]
    selected = donor_rows[np.asarray(local, dtype=np.int64)]
    weight = 1.0 / np.square(np.asarray(distance, dtype=np.float32) + 1.0)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return selected, weight.astype(np.float32)


def relation_statistics(
    aligned: np.ndarray,
    absolute: np.ndarray,
    query_rows: np.ndarray,
    donor_rows: np.ndarray,
    neighbor_ids: np.ndarray,
    weights: np.ndarray,
    group_chunk: int,
    seed: int,
    selection: np.ndarray | None = None,
    canonical_index: np.ndarray | None = None,
    canonical_prediction: np.ndarray | None = None,
    canonical_shuffle: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    groups = aligned.shape[1]
    cells = aligned.shape[2]
    sse_real = np.zeros((groups, cells), dtype=np.float64)
    sse_zero = np.zeros((groups, cells), dtype=np.float64)
    sse_shuffle = np.zeros((groups, cells), dtype=np.float64)
    sse_absolute = np.zeros((groups, cells), dtype=np.float64)
    sse_absolute_zero = np.zeros((groups, cells), dtype=np.float64)
    dot_real = norm_real = norm_truth = 0.0
    dot_shuffle = norm_shuffle = 0.0

    rng = np.random.default_rng(seed)
    permuted_rows = np.arange(len(aligned), dtype=np.int64)
    permuted_rows[donor_rows] = rng.permutation(donor_rows)
    shuffled_ids = permuted_rows[neighbor_ids]
    canonical_local = None
    if canonical_index is not None:
        canonical_local = np.flatnonzero(canonical_index[query_rows] >= 0)

    for start in range(0, groups, group_chunk):
        end = min(start + group_chunk, groups)
        true = np.asarray(aligned[query_rows, start:end], dtype=np.float32)
        donor = np.asarray(aligned[neighbor_ids, start:end], dtype=np.float32)
        predicted = np.einsum("qk,qkgc->qgc", weights, donor, optimize=True)
        shuffled = np.einsum(
            "qk,qkgc->qgc",
            weights,
            np.asarray(aligned[shuffled_ids, start:end], dtype=np.float32),
            optimize=True,
        )
        absolute_true = np.asarray(absolute[query_rows, start:end], dtype=np.float32)
        absolute_predicted = np.einsum(
            "qk,qkgc->qgc",
            weights,
            np.asarray(absolute[neighbor_ids, start:end], dtype=np.float32),
            optimize=True,
        )
        sse_real[start:end] = np.square(predicted - true).sum(axis=0, dtype=np.float64)
        sse_zero[start:end] = np.square(true).sum(axis=0, dtype=np.float64)
        sse_shuffle[start:end] = np.square(shuffled - true).sum(
            axis=0, dtype=np.float64
        )
        sse_absolute[start:end] = np.square(absolute_predicted - absolute_true).sum(
            axis=0, dtype=np.float64
        )
        sse_absolute_zero[start:end] = np.square(absolute_true).sum(
            axis=0, dtype=np.float64
        )

        if selection is not None:
            local_mask = selection[start:end][None, :, :]
            masked_prediction = predicted * local_mask
            masked_shuffle = shuffled * local_mask
            masked_truth = true * local_mask
            dot_real += float(np.sum(masked_prediction * masked_truth, dtype=np.float64))
            norm_real += float(np.sum(masked_prediction * masked_prediction, dtype=np.float64))
            norm_truth += float(np.sum(masked_truth * masked_truth, dtype=np.float64))
            dot_shuffle += float(np.sum(masked_shuffle * masked_truth, dtype=np.float64))
            norm_shuffle += float(np.sum(masked_shuffle * masked_shuffle, dtype=np.float64))

        if canonical_local is not None and len(canonical_local) > 0:
            destination = canonical_index[query_rows[canonical_local]]
            if canonical_prediction is not None:
                canonical_prediction[destination, start:end] = predicted[canonical_local]
            if canonical_shuffle is not None:
                canonical_shuffle[destination, start:end] = shuffled[canonical_local]

    selected_count = int(selection.sum()) if selection is not None else 0
    if selection is None or selected_count == 0:
        cosine_real = 0.0
        cosine_shuffle = 0.0
    else:
        cosine_real = dot_real / max(np.sqrt(norm_real * norm_truth), TINY)
        cosine_shuffle = dot_shuffle / max(np.sqrt(norm_shuffle * norm_truth), TINY)
    return {
        "sse_real": sse_real,
        "sse_zero": sse_zero,
        "sse_shuffle": sse_shuffle,
        "sse_absolute": sse_absolute,
        "sse_absolute_zero": sse_absolute_zero,
        "selected_count": selected_count,
        "selected_cosine_real": float(cosine_real),
        "selected_cosine_shuffle": float(cosine_shuffle),
    }


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, TINY))


def selected_summary(stats: dict[str, np.ndarray | float | int], mask: np.ndarray) -> dict[str, float | int]:
    real = float(np.asarray(stats["sse_real"])[mask].sum(dtype=np.float64))
    zero = float(np.asarray(stats["sse_zero"])[mask].sum(dtype=np.float64))
    shuffled = float(np.asarray(stats["sse_shuffle"])[mask].sum(dtype=np.float64))
    return {
        "selected_count": int(mask.sum()),
        "real_over_zero": ratio(real, zero),
        "shuffle_over_zero": ratio(shuffled, zero),
        "real_over_shuffle": ratio(real, shuffled),
        "cosine_real": float(stats["selected_cosine_real"]),
        "cosine_shuffle": float(stats["selected_cosine_shuffle"]),
    }


def all_mode_summary(stats: dict[str, np.ndarray | float | int]) -> dict[str, float]:
    real = float(np.asarray(stats["sse_real"]).sum(dtype=np.float64))
    zero = float(np.asarray(stats["sse_zero"]).sum(dtype=np.float64))
    shuffled = float(np.asarray(stats["sse_shuffle"]).sum(dtype=np.float64))
    absolute = float(np.asarray(stats["sse_absolute"]).sum(dtype=np.float64))
    absolute_zero = float(
        np.asarray(stats["sse_absolute_zero"]).sum(dtype=np.float64)
    )
    return {
        "aligned_real_over_zero": ratio(real, zero),
        "aligned_shuffle_over_zero": ratio(shuffled, zero),
        "aligned_real_over_shuffle": ratio(real, shuffled),
        "absolute_real_over_zero": ratio(absolute, absolute_zero),
    }


@torch.inference_mode()
def canonical_endpoints(
    channel: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pas = np.empty((len(query_ids), 2, 16, 8, 4, 8), dtype=np.float32)
    pdp = np.empty((len(query_ids), 2, 4, 2, 4, 192), dtype=np.float32)
    truth_pas = np.empty_like(pas)
    truth_pdp = np.empty_like(pdp)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        base_pas, base_pdp = grouped_endpoints(base)
        target_pas, target_pdp = grouped_endpoints(truth)
        pas[start:end] = base_pas.cpu().numpy()
        pdp[start:end] = base_pdp.cpu().numpy()
        truth_pas[start:end] = target_pas.cpu().numpy()
        truth_pdp[start:end] = target_pdp.cpu().numpy()
    return pas, pdp, truth_pas, truth_pdp


@torch.inference_mode()
def aligned_canonical_state(
    base_pas: np.ndarray,
    base_pdp: np.ndarray,
    truth_pas: np.ndarray,
    truth_pdp: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    base_pas_tensor = torch.as_tensor(base_pas, device=device)
    truth_pas_tensor = torch.as_tensor(truth_pas, device=device)
    aligned_base_pas, aligned_truth_pas, pas_centers = align_pas(
        base_pas_tensor, truth_pas_tensor
    )
    base_pdp_tensor = torch.as_tensor(base_pdp, device=device)
    truth_pdp_tensor = torch.as_tensor(truth_pdp, device=device)
    aligned_base_pdp, aligned_truth_pdp, pdp_centers = align_pdp(
        base_pdp_tensor, truth_pdp_tensor
    )
    return {
        "pas_base_root": aligned_base_pas.clamp_min(0.0).sqrt().reshape(
            len(base_pas), PAS_GROUPS, PAS_CELLS
        ).cpu().numpy(),
        "pas_truth_residual": (
            aligned_truth_pas.clamp_min(0.0).sqrt()
            - aligned_base_pas.clamp_min(0.0).sqrt()
        ).reshape(len(base_pas), PAS_GROUPS, PAS_CELLS).cpu().numpy(),
        "pas_centers": pas_centers.cpu().numpy(),
        "pdp_base_root": aligned_base_pdp.clamp_min(0.0).sqrt().cpu().numpy(),
        "pdp_truth_residual": (
            aligned_truth_pdp.clamp_min(0.0).sqrt()
            - aligned_base_pdp.clamp_min(0.0).sqrt()
        ).cpu().numpy(),
        "pdp_centers": pdp_centers.cpu().numpy(),
    }


@torch.inference_mode()
def render_grouped_targets(
    state: dict[str, np.ndarray],
    pas_action: np.ndarray,
    pdp_action: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pas_root = torch.as_tensor(state["pas_base_root"] + pas_action, device=device)
    pas_power = pas_root.clamp_min(0.0).square().reshape(-1, 4, 8, 2, 16, 8)
    pas_total = pas_power.sum(dim=(3, 4, 5), keepdim=True)
    pas_power = torch.where(
        pas_total > TINY, pas_power / pas_total.clamp_min(TINY), torch.zeros_like(pas_power)
    )
    pas = inverse_align_pas(
        pas_power, torch.as_tensor(state["pas_centers"], device=device)
    )

    pdp_root = torch.as_tensor(state["pdp_base_root"] + pdp_action, device=device)
    pdp_power = pdp_root.clamp_min(0.0).square()
    pdp_total = pdp_power.sum(dim=-1, keepdim=True)
    pdp_power = torch.where(
        pdp_total > TINY, pdp_power / pdp_total.clamp_min(TINY), torch.zeros_like(pdp_power)
    )
    pdp = inverse_align_pdp(
        pdp_power, torch.as_tensor(state["pdp_centers"], device=device)
    )
    return pas.cpu().numpy().astype(np.float32), pdp.cpu().numpy().astype(np.float32)


@torch.inference_mode()
def project_channel(
    source: np.ndarray,
    grouped_pas: np.ndarray,
    grouped_pdp: np.ndarray,
    site: np.ndarray,
    output_path,
    device: torch.device,
    batch_size: int,
    cycles: int,
) -> np.ndarray:
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        block = torch.as_tensor(
            np.array(source[start:end], dtype=np.complex64, copy=True), device=device
        )
        base_pas = pas_distribution(block).reshape(
            end - start, 2, 16, 8, 4, 192
        )
        current_pas_group = base_pas.reshape(
            end - start, 2, 16, 8, 4, 8, 24
        ).mean(dim=-1)
        current_pas_group = current_pas_group / current_pas_group.sum(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(TINY)
        target_pas_group = torch.as_tensor(grouped_pas[start:end], device=device)
        expanded_current_pas = current_pas_group[..., None].expand(
            end - start, 2, 16, 8, 4, 8, 24
        ).reshape(end - start, 2, 16, 8, 4, 192)
        expanded_target_pas = target_pas_group[..., None].expand(
            end - start, 2, 16, 8, 4, 8, 24
        ).reshape(end - start, 2, 16, 8, 4, 192)
        pas = torch.where(
            expanded_current_pas > TINY,
            base_pas * expanded_target_pas / expanded_current_pas.clamp_min(TINY),
            expanded_target_pas,
        )
        pas = pas / pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)

        base_pdp = pdp_distribution(block).reshape(
            end - start, 2, 16, 8, 4, 192
        )
        grouped_base_pdp = base_pdp.reshape(
            end - start, 2, 4, 4, 2, 4, 4, 192
        ).mean(dim=(3, 5))
        grouped_base_pdp = grouped_base_pdp / grouped_base_pdp.sum(
            dim=-1, keepdim=True
        ).clamp_min(TINY)
        target_pdp_group = torch.as_tensor(grouped_pdp[start:end], device=device)
        expanded_current_pdp = grouped_base_pdp[:, :, :, None, :, None, :, :].expand(
            end - start, 2, 4, 4, 2, 4, 4, 192
        ).reshape(end - start, 2, 16, 8, 4, 192)
        expanded_target_pdp = target_pdp_group[:, :, :, None, :, None, :, :].expand(
            end - start, 2, 4, 4, 2, 4, 4, 192
        ).reshape(end - start, 2, 16, 8, 4, 192)
        pdp = torch.where(
            expanded_current_pdp > TINY,
            base_pdp * expanded_target_pdp / expanded_current_pdp.clamp_min(TINY),
            expanded_target_pdp,
        ).reshape(end - start, 256, 4, 192)
        pdp = pdp / pdp.sum(dim=-1, keepdim=True).clamp_min(TINY)
        reference_power = block.abs().square().sum(dim=(1, 2, 3))
        for _ in range(cycles):
            for site_value in np.unique(site[start:end]):
                rows = np.flatnonzero(site[start:end] == site_value)
                local = block[rows]
                if int(site_value) == 0:
                    local = exact_pdp_projection(local, pdp[rows])
                    local = exact_pas_projection(local, pas[rows])
                else:
                    local = exact_pas_projection(local, pas[rows])
                    local = exact_pdp_projection(local, pdp[rows])
                block[rows] = preserve_point_power(local, reference_power[rows])
        output[start:end] = block.cpu().numpy().astype(np.complex64)
        output.flush()
    return np.load(output_path, mmap_mode="r")


def score_by_fold(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float | int]]:
    rows = []
    for fold in range(5):
        selected = np.flatnonzero(labels == fold)
        accumulator = OfficialMetricAccumulator()
        accumulator.add(
            np.asarray(prediction[selected], dtype=np.complex64),
            np.asarray(truth[query_ids[selected]], dtype=np.complex64),
        )
        values = accumulator.result()
        rows.append({"fold": fold, **values})
    return rows


def official_with_lifecycle(
    name: str,
    source: np.ndarray,
    grouped_pas: np.ndarray,
    grouped_pdp: np.ndarray,
    canonical_fold: np.ndarray,
    site: np.ndarray,
    regime: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    broad_fold: np.ndarray,
    device: torch.device,
    config: Config,
) -> dict[str, object]:
    pre_path = ARTIFACT_DIR / f"canonical_{name}_pre_lifecycle.npy"
    complete_path = ARTIFACT_DIR / f"canonical_{name}_complete.npy"
    pre = project_channel(
        source,
        grouped_pas,
        grouped_pdp,
        site,
        pre_path,
        device,
        config.metric_batch_size,
        config.cycles,
    )
    pre_score = score_channel(pre, truth, query_ids, config.metric_batch_size)
    complete = apply_lifecycle(
        pre,
        complete_path,
        canonical_fold,
        site,
        regime,
        device,
        config.metric_batch_size,
    )
    complete_score = score_channel(
        complete, truth, query_ids, config.metric_batch_size
    )
    return {
        "pre_lifecycle": {
            "path": PATHS.relative(pre_path),
            "official": pre_score,
        },
        "complete": {
            "path": PATHS.relative(complete_path),
            "official": complete_score,
            "broad_fold": score_by_fold(complete, truth, query_ids, broad_fold),
            "canonical_fold": score_by_fold(
                complete, truth, query_ids, canonical_fold
            ),
        },
    }


def run(device: torch.device, args: argparse.Namespace) -> dict[str, object]:
    config = Config(
        batch_size=args.batch_size,
        metric_batch_size=args.metric_batch_size,
    )
    caches = build_residual_caches(device, config.batch_size, args.force_cache)
    _, train_positions, _, train_site, _ = load_positions()
    positions = np.asarray(train_positions, dtype=np.float32)
    site_all = np.asarray(train_site, dtype=np.int8)
    if positions.shape != (4000, 3) or site_all.shape != (4000,):
        raise ValueError(
            f"unexpected position/site shapes: {positions.shape}/{site_all.shape}"
        )
    broad_fold_all = np.load(FOLD_IDS, allow_pickle=False).astype(np.int8)

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    canonical_broad_fold = broad_fold_all[query_ids]
    canonical_site = site_all[query_ids]
    canonical_index = np.full(4000, -1, dtype=np.int64)
    canonical_index[query_ids] = np.arange(len(query_ids), dtype=np.int64)

    canonical_prediction_pas = np.zeros(
        (len(query_ids), PAS_GROUPS, PAS_CELLS), dtype=np.float32
    )
    canonical_shuffle_pas = np.zeros_like(canonical_prediction_pas)
    canonical_prediction_pdp = np.zeros(
        (len(query_ids), PDP_GROUPS, PDP_CELLS), dtype=np.float32
    )
    canonical_shuffle_pdp = np.zeros_like(canonical_prediction_pdp)
    canonical_mask_pas = np.zeros_like(canonical_prediction_pas, dtype=bool)
    canonical_mask_pdp = np.zeros_like(canonical_prediction_pdp, dtype=bool)

    relation_records: list[dict[str, object]] = []
    masks: dict[tuple[int, int, str], np.ndarray] = {}
    relation_pass_flags: list[bool] = []

    for outer in range(5):
        for site_value in (0, 1):
            fit_rows = np.flatnonzero(
                (broad_fold_all != outer) & (site_all == site_value)
            )
            held_rows = np.flatnonzero(
                (broad_fold_all == outer) & (site_all == site_value)
            )
            if len(held_rows) == 0:
                continue
            for domain in ("pas", "pdp"):
                aligned = caches[f"aligned_{domain}"]
                absolute = caches[f"absolute_{domain}"]
                inner_statistics = []
                for inner in range(5):
                    if inner == outer:
                        continue
                    inner_rows = np.flatnonzero(
                        (broad_fold_all == inner) & (site_all == site_value)
                    )
                    donor_rows = fit_rows[broad_fold_all[fit_rows] != inner]
                    neighbor_ids, weights = neighbor_table(
                        inner_rows, donor_rows, positions, config.neighbors
                    )
                    inner_statistics.append(
                        relation_statistics(
                            aligned,
                            absolute,
                            inner_rows,
                            donor_rows,
                            neighbor_ids,
                            weights,
                            config.group_chunk,
                            config.seed + outer * 1000 + site_value * 100 + inner * 7 + (0 if domain == "pas" else 1),
                        )
                    )
                selected = np.ones(aligned.shape[1:], dtype=bool)
                for stats in inner_statistics:
                    selected &= np.asarray(stats["sse_real"]) < np.asarray(stats["sse_zero"])
                    selected &= np.asarray(stats["sse_real"]) < np.asarray(stats["sse_shuffle"])
                masks[(outer, site_value, domain)] = selected

                neighbor_ids, weights = neighbor_table(
                    held_rows, fit_rows, positions, config.neighbors
                )
                prediction_target = (
                    canonical_prediction_pas if domain == "pas" else canonical_prediction_pdp
                )
                shuffle_target = (
                    canonical_shuffle_pas if domain == "pas" else canonical_shuffle_pdp
                )
                outer_stats = relation_statistics(
                    aligned,
                    absolute,
                    held_rows,
                    fit_rows,
                    neighbor_ids,
                    weights,
                    config.group_chunk,
                    config.seed + outer * 1000 + site_value * 100 + 90 + (0 if domain == "pas" else 1),
                    selection=selected,
                    canonical_index=canonical_index,
                    canonical_prediction=prediction_target,
                    canonical_shuffle=shuffle_target,
                )
                selected_result = selected_summary(outer_stats, selected)
                passed = (
                    selected_result["selected_count"] > 0
                    and selected_result["real_over_zero"] < 1.0
                    and selected_result["real_over_shuffle"] < 1.0
                    and selected_result["cosine_real"] > selected_result["cosine_shuffle"]
                )
                relation_pass_flags.append(bool(passed))
                relation_records.append(
                    {
                        "outer": outer,
                        "site": site_value,
                        "domain": domain,
                        "fit_count": int(len(fit_rows)),
                        "held_count": int(len(held_rows)),
                        "selected": selected_result,
                        "all_outer_modes": all_mode_summary(outer_stats),
                        "inner": [all_mode_summary(item) for item in inner_statistics],
                        "passed": bool(passed),
                    }
                )
                canonical_rows = np.flatnonzero(
                    (canonical_broad_fold == outer) & (canonical_site == site_value)
                )
                target_mask = canonical_mask_pas if domain == "pas" else canonical_mask_pdp
                target_mask[canonical_rows] = selected[None, :, :]
                print(
                    f"E162 outer={outer} site={site_value} domain={domain} "
                    f"selected={selected.sum()}/{selected.size} "
                    f"real/zero={selected_result['real_over_zero']:.4f} "
                    f"real/shuffle={selected_result['real_over_shuffle']:.4f}",
                    flush=True,
                )

    canonical_core = np.load(OOF_SCALED, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    base_pas, base_pdp, truth_pas, truth_pdp = canonical_endpoints(
        canonical_core,
        truth_all,
        query_ids,
        device,
        config.metric_batch_size,
    )
    state = aligned_canonical_state(base_pas, base_pdp, truth_pas, truth_pdp, device)

    predicted_pas_action = canonical_prediction_pas * canonical_mask_pas
    predicted_pdp_action = canonical_prediction_pdp * canonical_mask_pdp
    shuffled_pas_action = canonical_shuffle_pas * canonical_mask_pas
    shuffled_pdp_action = canonical_shuffle_pdp * canonical_mask_pdp
    oracle_pas_action = state["pas_truth_residual"] * canonical_mask_pas
    oracle_pdp_action = state["pdp_truth_residual"] * canonical_mask_pdp

    predicted_targets = render_grouped_targets(
        state, predicted_pas_action, predicted_pdp_action, device
    )
    shuffled_targets = render_grouped_targets(
        state, shuffled_pas_action, shuffled_pdp_action, device
    )
    oracle_targets = render_grouped_targets(
        state, oracle_pas_action, oracle_pdp_action, device
    )
    zero_targets = render_grouped_targets(
        state,
        np.zeros_like(predicted_pas_action),
        np.zeros_like(predicted_pdp_action),
        device,
    )

    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    methods = {
        "zero_action_identity": official_with_lifecycle(
            "zero_action_identity",
            canonical_core,
            *zero_targets,
            canonical_fold,
            canonical_site,
            regime,
            truth_all,
            query_ids,
            canonical_broad_fold,
            device,
            config,
        ),
        "knn_relative_action": official_with_lifecycle(
            "knn_relative_action",
            canonical_core,
            *predicted_targets,
            canonical_fold,
            canonical_site,
            regime,
            truth_all,
            query_ids,
            canonical_broad_fold,
            device,
            config,
        ),
        "shuffled_relative_action": official_with_lifecycle(
            "shuffled_relative_action",
            canonical_core,
            *shuffled_targets,
            canonical_fold,
            canonical_site,
            regime,
            truth_all,
            query_ids,
            canonical_broad_fold,
            device,
            config,
        ),
        "predictable_subspace_oracle": official_with_lifecycle(
            "predictable_subspace_oracle",
            canonical_core,
            *oracle_targets,
            canonical_fold,
            canonical_site,
            regime,
            truth_all,
            query_ids,
            canonical_broad_fold,
            device,
            config,
        ),
    }

    baseline = np.load(CANONICAL_V7_COMPLETE, mmap_mode="r")
    baseline_score = score_channel(
        baseline, truth_all, query_ids, config.metric_batch_size
    )
    identity_error = abs(
        float(methods["zero_action_identity"]["complete"]["official"]["score"])
        - float(baseline_score["score"])
    )
    if identity_error > IDENTITY_TOLERANCE:
        raise RuntimeError(
            f"E162 zero-action identity failed: {identity_error:.9g}"
        )
    baseline_broad = score_by_fold(
        baseline, truth_all, query_ids, canonical_broad_fold
    )
    for method in methods.values():
        method_score = float(method["complete"]["official"]["score"])
        method["complete"]["delta_vs_v7"] = method_score - float(
            baseline_score["score"]
        )
        method["complete"]["broad_fold_delta_vs_v7"] = [
            float(current["score"] - reference["score"])
            for current, reference in zip(
                method["complete"]["broad_fold"], baseline_broad, strict=True
            )
        ]

    oracle_score = float(
        methods["predictable_subspace_oracle"]["complete"]["official"]["score"]
    )
    relation_passed = bool(all(relation_pass_flags))
    return {
        "config": config.__dict__,
        "baseline": {
            "path": PATHS.relative(CANONICAL_V7_COMPLETE),
            "official": baseline_score,
            "broad_fold": baseline_broad,
            "zero_action_identity_error": identity_error,
            "identity_tolerance": IDENTITY_TOLERANCE,
        },
        "relation": {
            "records": relation_records,
            "all_outer_site_domain_passed": relation_passed,
            "pass_count": int(sum(relation_pass_flags)),
            "total_count": int(len(relation_pass_flags)),
        },
        "methods": methods,
        "decision": {
            "required_score": V12_SCORE,
            "oracle_score": oracle_score,
            "oracle_margin": oracle_score - V12_SCORE,
            "relation_passed": relation_passed,
            "passed": relation_passed and oracle_score > V12_SCORE,
        },
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    result = run(torch.device(args.device), args)
    payload = {
        "experiment": VERSION,
        "status": "relation_gate_passed" if result["decision"]["passed"] else "relation_gate_failed",
        "diagnostic_only": True,
        "official_inputs_only": True,
        "sionna_used": False,
        "result": result,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
