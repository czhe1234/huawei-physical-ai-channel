#!/usr/bin/env python3
"""Test whether V12 exposes the same sparse transport law as the true V7 error."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch

import analyze_e138_v12_guided_official_correction_student as e138
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from data import load_positions
from official_metrics import OfficialMetricAccumulator
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META
from run_v6_regime_component_lifecycle import REGIME_CACHE
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E163-v12-sparse-unbalanced-transport-mechanism"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e163_v12_transport_mechanism"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e163_v12_transport_mechanism"
REPORT = ANALYSIS_DIR / "e163_v12_transport_mechanism.json"

ZERO_COMPLETE = ARTIFACT_DIR / "canonical_zero_transport_complete.npy"
SHARED_COMPLETE = ARTIFACT_DIR / "canonical_shared_transport_oracle_complete.npy"
ALL_COMPLETE = ARTIFACT_DIR / "canonical_all_transport_oracle_complete.npy"

PAS_GROUPS = 4
PAS_CELLS = 2 * 16 * 8
PAS_SIGNATURE = 3 * 16 * 8 + 2
PAS_DESTROY = PAS_SIGNATURE - 2
PAS_BIRTH = PAS_SIGNATURE - 1
PDP_GROUPS = 2 * 4
PDP_CELLS = 192
PDP_SIGNATURE = 192 + 2
PDP_DESTROY = PDP_SIGNATURE - 2
PDP_BIRTH = PDP_SIGNATURE - 1
TINY = 1e-30
V7_SCORE = 0.644456345060792
V12_SCORE = 0.677521
IDENTITY_TOLERANCE = 5e-6


@dataclass(frozen=True)
class Config:
    top_k: int = 16
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 8192
    sinkhorn_check_interval: int = 64
    sinkhorn_tolerance: float = 1e-5
    dust_cost: float = 1.0
    pas_h_scale: float = 2.0
    pas_v_scale: float = 1.0
    pas_polarization_cost: float = 1.0
    pdp_delay_scale: float = 3.0
    shuffles: int = 16
    plan_batch_size: int = 512
    channel_batch_size: int = 8
    cycles: int = 4
    seed: int = 260_811_163


@dataclass
class SparsePlan:
    source_index: np.ndarray
    target_index: np.ndarray
    flow: np.ndarray
    signature: np.ndarray
    residual_mass: np.ndarray
    marginal_error: float
    maximum_iterations_used: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--plan-batch-size", type=int, default=Config.plan_batch_size
    )
    parser.add_argument(
        "--channel-batch-size", type=int, default=Config.channel_batch_size
    )
    return parser.parse_args()


def signed_periodic_delta(
    source: torch.Tensor, target: torch.Tensor, period: int
) -> torch.Tensor:
    return (target - source + period // 2) % period - period // 2


def pas_cost_and_code(
    source: torch.Tensor, target: torch.Tensor, config: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    source_p = torch.div(source, 16 * 8, rounding_mode="floor")
    source_h = torch.div(source, 8, rounding_mode="floor") % 16
    source_v = source % 8
    target_p = torch.div(target, 16 * 8, rounding_mode="floor")
    target_h = torch.div(target, 8, rounding_mode="floor") % 16
    target_v = target % 8
    dp = target_p[:, None, :] - source_p[:, :, None]
    dh = signed_periodic_delta(
        source_h[:, :, None], target_h[:, None, :], 16
    )
    dv = signed_periodic_delta(
        source_v[:, :, None], target_v[:, None, :], 8
    )
    cost = (
        dp.float().square() * config.pas_polarization_cost
        + (dh.float() / config.pas_h_scale).square()
        + (dv.float() / config.pas_v_scale).square()
    )
    code = (dp + 1) * (16 * 8) + (dh + 8) * 8 + (dv + 4)
    return cost, code.long()


def pdp_cost_and_code(
    source: torch.Tensor, target: torch.Tensor, config: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    delay = signed_periodic_delta(source[:, :, None], target[:, None, :], 192)
    return (delay.float() / config.pdp_delay_scale).square(), (delay + 96).long()


def log_sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    iterations: int,
    check_interval: int,
    tolerance: float,
) -> tuple[torch.Tensor, int]:
    batch = len(a)
    output = torch.zeros_like(cost)
    active = (a.sum(dim=1) > 1e-12) & (b.sum(dim=1) > 1e-12)
    if not bool(active.any()):
        return output, 0
    local_a = a[active].double()
    local_b = b[active].double()
    log_a = torch.where(
        local_a > 0.0, torch.log(local_a.clamp_min(1e-30)), torch.full_like(local_a, -torch.inf)
    )
    log_b = torch.where(
        local_b > 0.0, torch.log(local_b.clamp_min(1e-30)), torch.full_like(local_b, -torch.inf)
    )
    log_kernel = -cost[active].double() / epsilon
    log_u = torch.zeros_like(local_a)
    log_v = torch.zeros_like(local_b)
    iterations_used = 0
    local_flow = None
    for iteration in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, :, None], dim=1)
        iterations_used = iteration + 1
        if iterations_used % check_interval == 0 or iterations_used == iterations:
            local_flow = torch.exp(
                log_u[:, :, None] + log_kernel + log_v[:, None, :]
            )
            row_error = (local_flow.sum(dim=2) - local_a).abs().amax()
            column_error = (local_flow.sum(dim=1) - local_b).abs().amax()
            if float(torch.maximum(row_error, column_error)) <= tolerance:
                break
    if local_flow is None:
        raise RuntimeError("Sinkhorn completed without producing a transport plan")
    output[active] = local_flow.to(output.dtype)
    return output, iterations_used


@torch.inference_mode()
def sparse_transport_plan(
    source: np.ndarray,
    target: np.ndarray,
    domain: str,
    device: torch.device,
    config: Config,
) -> SparsePlan:
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError(f"invalid transport shapes: {source.shape}/{target.shape}")
    count, groups, cells = source.shape
    flattened_source = np.asarray(source, dtype=np.float32).reshape(-1, cells)
    flattened_target = np.asarray(target, dtype=np.float32).reshape(-1, cells)
    total_spectra = len(flattened_source)
    source_index = np.empty((total_spectra, config.top_k), dtype=np.int16)
    target_index = np.empty_like(source_index)
    flow = np.zeros(
        (total_spectra, config.top_k + 1, config.top_k + 1), dtype=np.float32
    )
    signature_size = PAS_SIGNATURE if domain == "pas" else PDP_SIGNATURE
    signature = np.zeros((count, signature_size), dtype=np.float64)
    residual_mass = np.zeros(count, dtype=np.float64)
    maximum_marginal_error = 0.0
    maximum_iterations_used = 0

    for start in range(0, total_spectra, config.plan_batch_size):
        end = min(start + config.plan_batch_size, total_spectra)
        source_tensor = torch.as_tensor(flattened_source[start:end], device=device)
        target_tensor = torch.as_tensor(flattened_target[start:end], device=device)
        common = torch.minimum(source_tensor, target_tensor)
        supply = (source_tensor - common).clamp_min(0.0)
        demand = (target_tensor - common).clamp_min(0.0)
        supply_mass, supply_index = torch.topk(
            supply, k=config.top_k, dim=1, largest=True, sorted=False
        )
        demand_mass, demand_index = torch.topk(
            demand, k=config.top_k, dim=1, largest=True, sorted=False
        )
        supply_total = supply_mass.sum(dim=1)
        demand_total = demand_mass.sum(dim=1)
        a = torch.cat((supply_mass, demand_total[:, None]), dim=1)
        b = torch.cat((demand_mass, supply_total[:, None]), dim=1)

        if domain == "pas":
            real_cost, offset_code = pas_cost_and_code(
                supply_index, demand_index, config
            )
            destroy_code = PAS_DESTROY
            birth_code = PAS_BIRTH
        elif domain == "pdp":
            real_cost, offset_code = pdp_cost_and_code(
                supply_index, demand_index, config
            )
            destroy_code = PDP_DESTROY
            birth_code = PDP_BIRTH
        else:
            raise ValueError(domain)
        cost = torch.full(
            (end - start, config.top_k + 1, config.top_k + 1),
            config.dust_cost,
            device=device,
            dtype=torch.float32,
        )
        cost[:, : config.top_k, : config.top_k] = real_cost
        cost[:, config.top_k, config.top_k] = 0.0
        local_flow, iterations_used = log_sinkhorn(
            a,
            b,
            cost,
            config.sinkhorn_epsilon,
            config.sinkhorn_iterations,
            config.sinkhorn_check_interval,
            config.sinkhorn_tolerance,
        )
        maximum_iterations_used = max(maximum_iterations_used, iterations_used)
        row_error = (local_flow.sum(dim=2) - a).abs().amax()
        column_error = (local_flow.sum(dim=1) - b).abs().amax()
        maximum_marginal_error = max(
            maximum_marginal_error,
            float(torch.maximum(row_error, column_error).cpu()),
        )
        local_source_index = supply_index.cpu().numpy().astype(np.int16)
        local_target_index = demand_index.cpu().numpy().astype(np.int16)
        local_flow_np = local_flow.cpu().numpy().astype(np.float32)
        source_index[start:end] = local_source_index
        target_index[start:end] = local_target_index
        flow[start:end] = local_flow_np

        point_index = np.arange(start, end, dtype=np.int64) // groups
        local_code = offset_code.cpu().numpy().astype(np.int64)
        real_flow = local_flow_np[:, : config.top_k, : config.top_k]
        np.add.at(
            signature,
            (
                np.repeat(point_index, config.top_k * config.top_k),
                local_code.reshape(-1),
            ),
            real_flow.reshape(-1),
        )
        np.add.at(
            signature,
            (np.repeat(point_index, config.top_k), destroy_code),
            local_flow_np[:, : config.top_k, config.top_k].reshape(-1),
        )
        np.add.at(
            signature,
            (np.repeat(point_index, config.top_k), birth_code),
            local_flow_np[:, config.top_k, : config.top_k].reshape(-1),
        )
        np.add.at(
            residual_mass,
            point_index,
            supply_total.cpu().numpy().astype(np.float64),
        )
        print(
            f"E163 {domain} transport {end}/{total_spectra}", flush=True
        )
    return SparsePlan(
        source_index=source_index,
        target_index=target_index,
        flow=flow,
        signature=signature.astype(np.float32),
        residual_mass=residual_mass.astype(np.float32),
        marginal_error=maximum_marginal_error,
        maximum_iterations_used=maximum_iterations_used,
    )


def compact_spectra(
    pas: np.ndarray, pdp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pas_spectra = np.transpose(pas, (0, 4, 1, 2, 3)).reshape(
        len(pas), PAS_GROUPS, PAS_CELLS
    )
    pdp_spectra = pdp.reshape(len(pdp), PDP_GROUPS, PDP_CELLS)
    return pas_spectra.astype(np.float32), pdp_spectra.astype(np.float32)


def compact_from_spectra(
    pas: np.ndarray, pdp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pas_compact = pas.reshape(len(pas), 4, 2, 16, 8).transpose(0, 2, 3, 4, 1)
    pdp_compact = pdp.reshape(len(pdp), 2, 4, 192)
    return pas_compact.astype(np.float32), pdp_compact.astype(np.float32)


def shuffled_indices(
    site: np.ndarray,
    regime: np.ndarray,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    strata = [
        np.flatnonzero((site == site_value) & (regime == regime_value))
        for site_value in np.unique(site)
        for regime_value in np.unique(regime)
    ]
    output = []
    for iteration in range(count):
        rng = np.random.default_rng(seed + iteration * 1009)
        permutation = np.arange(len(site), dtype=np.int64)
        for rows in strata:
            if len(rows) <= 1:
                continue
            shuffled = rng.permutation(rows)
            if np.array_equal(shuffled, rows):
                shuffled = np.roll(shuffled, 1)
            permutation[rows] = shuffled
        output.append(permutation)
    return output


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=1, dtype=np.float64)
        * np.sum(right * right, axis=1, dtype=np.float64)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > TINY,
    )


def relation_audit(
    truth: np.ndarray,
    teacher: np.ndarray,
    fold_id: np.ndarray,
    permutations: list[np.ndarray],
) -> dict[str, object]:
    same = row_cosine(truth, teacher)
    controls = np.stack(
        [row_cosine(truth, teacher[permutation]) for permutation in permutations],
        axis=0,
    )
    folds = []
    for fold in range(5):
        rows = fold_id == fold
        same_value = float(np.mean(same[rows]))
        control_values = controls[:, rows].mean(axis=1)
        control_mean = float(np.mean(control_values))
        folds.append(
            {
                "fold": fold,
                "count": int(np.sum(rows)),
                "same_point_cosine": same_value,
                "shuffle_mean_cosine": control_mean,
                "margin": same_value - control_mean,
                "shuffle_p95": float(np.quantile(control_values, 0.95)),
            }
        )
    return {
        "same_point_cosine": float(np.mean(same)),
        "shuffle_mean_cosine": float(np.mean(controls)),
        "margin": float(np.mean(same) - np.mean(controls)),
        "positive_folds": int(sum(row["margin"] > 0.0 for row in folds)),
        "folds": folds,
    }


def action_masks(
    truth: np.ndarray,
    teacher: np.ndarray,
    fold_id: np.ndarray,
    permutations: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    masks = np.zeros((5, truth.shape[1]), dtype=bool)
    records = []
    square_truth = np.sqrt(np.maximum(truth, 0.0))
    square_teacher = np.sqrt(np.maximum(teacher, 0.0))
    for outer in range(5):
        selected = np.ones(truth.shape[1], dtype=bool)
        inner_rows = []
        for inner in range(5):
            if inner == outer:
                continue
            rows = fold_id == inner
            same = np.sum(
                square_truth[rows] * square_teacher[rows], axis=0, dtype=np.float64
            )
            controls = np.stack(
                [
                    np.sum(
                        square_truth[rows]
                        * square_teacher[permutation[rows]],
                        axis=0,
                        dtype=np.float64,
                    )
                    for permutation in permutations
                ],
                axis=0,
            )
            control_mean = controls.mean(axis=0)
            local = same > control_mean
            selected &= local
            inner_rows.append(
                {
                    "fold": inner,
                    "positive_actions": int(local.sum()),
                    "same_overlap": float(same.sum()),
                    "shuffle_overlap": float(control_mean.sum()),
                }
            )
        fit = fold_id != outer
        selected &= truth[fit].sum(axis=0) > 1e-10
        selected &= teacher[fit].sum(axis=0) > 1e-10
        masks[outer] = selected
        records.append(
            {
                "outer": outer,
                "selected_actions": int(selected.sum()),
                "total_actions": int(len(selected)),
                "inner": inner_rows,
            }
        )
    return masks, records


def pas_offset_code(source: int, target: int) -> int:
    source_p, remainder = divmod(source, 16 * 8)
    source_h, source_v = divmod(remainder, 8)
    target_p, remainder = divmod(target, 16 * 8)
    target_h, target_v = divmod(remainder, 8)
    dp = target_p - source_p
    dh = (target_h - source_h + 8) % 16 - 8
    dv = (target_v - source_v + 4) % 8 - 4
    return int((dp + 1) * 16 * 8 + (dh + 8) * 8 + (dv + 4))


def pdp_offset_code(source: int, target: int) -> int:
    return int((target - source + 96) % 192)


def apply_selected_plan(
    source: np.ndarray,
    plan: SparsePlan,
    fold_id: np.ndarray,
    masks: np.ndarray,
    domain: str,
    all_actions: bool,
    config: Config,
) -> np.ndarray:
    output = np.array(source, dtype=np.float64, copy=True)
    groups = source.shape[1]
    for spectrum in range(len(plan.flow)):
        point = spectrum // groups
        selected = None if all_actions else masks[int(fold_id[point])]
        source_row = plan.source_index[spectrum]
        target_row = plan.target_index[spectrum]
        local_flow = plan.flow[spectrum]
        for left in range(config.top_k):
            source_cell = int(source_row[left])
            for right in range(config.top_k):
                value = float(local_flow[left, right])
                if value <= 1e-12:
                    continue
                code = (
                    pas_offset_code(source_cell, int(target_row[right]))
                    if domain == "pas"
                    else pdp_offset_code(source_cell, int(target_row[right]))
                )
                if all_actions or bool(selected[code]):
                    output[point, spectrum % groups, source_cell] -= value
                    output[point, spectrum % groups, int(target_row[right])] += value
            destroy = float(local_flow[left, config.top_k])
            destroy_code = PAS_DESTROY if domain == "pas" else PDP_DESTROY
            if destroy > 1e-12 and (all_actions or bool(selected[destroy_code])):
                output[point, spectrum % groups, source_cell] -= destroy
        birth_code = PAS_BIRTH if domain == "pas" else PDP_BIRTH
        if all_actions or bool(selected[birth_code]):
            for right in range(config.top_k):
                value = float(local_flow[config.top_k, right])
                if value > 1e-12:
                    output[point, spectrum % groups, int(target_row[right])] += value
    output = np.maximum(output, 0.0)
    total = output.sum(axis=-1, keepdims=True, dtype=np.float64)
    return np.divide(
        output,
        np.maximum(total, TINY),
        out=np.zeros_like(output),
        where=total > 0.0,
    ).astype(np.float32)


@torch.inference_mode()
def project_compact_targets(
    source: np.ndarray,
    desired_pas: np.ndarray,
    desired_pdp: np.ndarray,
    output_path,
    device: torch.device,
    config: Config,
) -> np.ndarray:
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), config.channel_batch_size):
        end = min(start + config.channel_batch_size, len(source))
        base = torch.as_tensor(
            np.array(source[start:end], dtype=np.complex64, copy=True), device=device
        )
        full_pas = pas_distribution(base).reshape(
            end - start, 2, 16, 8, 4, 192
        )
        compact_pas = full_pas.mean(dim=-1)
        compact_pas = compact_pas / compact_pas.sum(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(TINY)
        target_pas = torch.as_tensor(desired_pas[start:end], device=device)
        expanded_current_pas = compact_pas[..., None].expand_as(full_pas)
        expanded_target_pas = target_pas[..., None].expand_as(full_pas)
        full_target_pas = torch.where(
            expanded_current_pas > TINY,
            full_pas * expanded_target_pas / expanded_current_pas.clamp_min(TINY),
            expanded_target_pas,
        )
        full_target_pas = full_target_pas / full_target_pas.sum(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(TINY)

        full_pdp = pdp_distribution(base).reshape(
            end - start, 2, 16, 8, 4, 192
        )
        compact_pdp = full_pdp.mean(dim=(2, 3))
        compact_pdp = compact_pdp / compact_pdp.sum(
            dim=-1, keepdim=True
        ).clamp_min(TINY)
        target_pdp = torch.as_tensor(desired_pdp[start:end], device=device)
        expanded_current_pdp = compact_pdp[:, :, None, None, :, :].expand_as(full_pdp)
        expanded_target_pdp = target_pdp[:, :, None, None, :, :].expand_as(full_pdp)
        full_target_pdp = torch.where(
            expanded_current_pdp > TINY,
            full_pdp * expanded_target_pdp / expanded_current_pdp.clamp_min(TINY),
            expanded_target_pdp,
        )
        full_target_pdp = full_target_pdp / full_target_pdp.sum(
            dim=-1, keepdim=True
        ).clamp_min(TINY)
        prediction = project_joint(
            base,
            full_target_pas,
            full_target_pdp.reshape(end - start, 256, 4, 192),
            config.cycles,
            "pas_pdp",
        )
        output[start:end] = prediction.cpu().numpy().astype(np.complex64)
        output.flush()
    return np.load(output_path, mmap_mode="r")


def score_channel(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> dict[str, float | int]:
    accumulator = OfficialMetricAccumulator()
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        accumulator.add(
            np.asarray(prediction[start:end], dtype=np.complex64),
            np.asarray(truth[query_ids[start:end]], dtype=np.complex64),
        )
    return accumulator.result()


def score_by_fold(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
) -> list[dict[str, float | int]]:
    output = []
    for fold in range(5):
        rows = np.flatnonzero(fold_id == fold)
        accumulator = OfficialMetricAccumulator()
        accumulator.add(
            np.asarray(prediction[rows], dtype=np.complex64),
            np.asarray(truth[query_ids[rows]], dtype=np.complex64),
        )
        output.append({"fold": fold, **accumulator.result()})
    return output


def endpoint_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(len(left), -1).astype(np.float64)
    b = right.reshape(len(right), -1).astype(np.float64)
    numerator = np.sum(a * b, axis=1)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return float(
        np.mean(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > TINY,
            )
        )
    )


def run(device: torch.device, args: argparse.Namespace) -> dict[str, object]:
    config = Config(
        plan_batch_size=args.plan_batch_size,
        channel_batch_size=args.channel_batch_size,
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = np.asarray(train_site[query_ids], dtype=np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)

    v7 = np.load(e138.V7_CHANNEL, mmap_mode="r")
    v12 = np.load(e138.V12_CHANNEL, mmap_mode="r")
    truth_all = np.load(e138.TRAIN_CHANNEL, mmap_mode="r")
    truth = np.asarray(truth_all[query_ids], dtype=np.complex64)
    if v7.shape != truth.shape or v12.shape != truth.shape:
        raise ValueError(f"channel shape mismatch: {v7.shape}/{v12.shape}/{truth.shape}")

    v7_pas, v7_pdp, _ = e138.compact_channel_endpoints(
        v7, device, config.channel_batch_size
    )
    v12_pas, v12_pdp, _ = e138.compact_channel_endpoints(
        v12, device, config.channel_batch_size
    )
    truth_pas, truth_pdp, _ = e138.compact_channel_endpoints(
        truth, device, config.channel_batch_size
    )
    v7_pas_spectra, v7_pdp_spectra = compact_spectra(v7_pas, v7_pdp)
    v12_pas_spectra, v12_pdp_spectra = compact_spectra(v12_pas, v12_pdp)
    truth_pas_spectra, truth_pdp_spectra = compact_spectra(truth_pas, truth_pdp)

    truth_pas_plan = sparse_transport_plan(
        v7_pas_spectra, truth_pas_spectra, "pas", device, config
    )
    teacher_pas_plan = sparse_transport_plan(
        v7_pas_spectra, v12_pas_spectra, "pas", device, config
    )
    truth_pdp_plan = sparse_transport_plan(
        v7_pdp_spectra, truth_pdp_spectra, "pdp", device, config
    )
    teacher_pdp_plan = sparse_transport_plan(
        v7_pdp_spectra, v12_pdp_spectra, "pdp", device, config
    )

    permutations = shuffled_indices(site, regime, config.shuffles, config.seed)
    pas_relation = relation_audit(
        truth_pas_plan.signature,
        teacher_pas_plan.signature,
        fold_id,
        permutations,
    )
    pdp_relation = relation_audit(
        truth_pdp_plan.signature,
        teacher_pdp_plan.signature,
        fold_id,
        permutations,
    )
    pas_masks, pas_mask_records = action_masks(
        truth_pas_plan.signature,
        teacher_pas_plan.signature,
        fold_id,
        permutations,
    )
    pdp_masks, pdp_mask_records = action_masks(
        truth_pdp_plan.signature,
        teacher_pdp_plan.signature,
        fold_id,
        permutations,
    )

    shared_pas_spectra = apply_selected_plan(
        v7_pas_spectra,
        truth_pas_plan,
        fold_id,
        pas_masks,
        "pas",
        False,
        config,
    )
    shared_pdp_spectra = apply_selected_plan(
        v7_pdp_spectra,
        truth_pdp_plan,
        fold_id,
        pdp_masks,
        "pdp",
        False,
        config,
    )
    all_pas_spectra = apply_selected_plan(
        v7_pas_spectra,
        truth_pas_plan,
        fold_id,
        pas_masks,
        "pas",
        True,
        config,
    )
    all_pdp_spectra = apply_selected_plan(
        v7_pdp_spectra,
        truth_pdp_plan,
        fold_id,
        pdp_masks,
        "pdp",
        True,
        config,
    )
    shared_pas, shared_pdp = compact_from_spectra(
        shared_pas_spectra, shared_pdp_spectra
    )
    all_pas, all_pdp = compact_from_spectra(all_pas_spectra, all_pdp_spectra)

    zero = project_compact_targets(
        v7, v7_pas, v7_pdp, ZERO_COMPLETE, device, config
    )
    shared = project_compact_targets(
        v7, shared_pas, shared_pdp, SHARED_COMPLETE, device, config
    )
    all_transport = project_compact_targets(
        v7, all_pas, all_pdp, ALL_COMPLETE, device, config
    )

    baseline_score = score_channel(v7, truth_all, query_ids, config.channel_batch_size)
    v12_score = score_channel(v12, truth_all, query_ids, config.channel_batch_size)
    zero_score = score_channel(zero, truth_all, query_ids, config.channel_batch_size)
    shared_score = score_channel(
        shared, truth_all, query_ids, config.channel_batch_size
    )
    all_score = score_channel(
        all_transport, truth_all, query_ids, config.channel_batch_size
    )
    identity_error = abs(float(zero_score["score"]) - float(baseline_score["score"]))
    if identity_error > IDENTITY_TOLERANCE:
        raise RuntimeError(f"E163 zero transport identity failed: {identity_error:.9g}")

    baseline_folds = score_by_fold(v7, truth_all, query_ids, fold_id)
    shared_folds = score_by_fold(shared, truth_all, query_ids, fold_id)
    all_folds = score_by_fold(all_transport, truth_all, query_ids, fold_id)
    relation_passed = (
        pas_relation["positive_folds"] == 5 and pdp_relation["positive_folds"] == 5
    )
    capacity_passed = float(shared_score["score"]) > V12_SCORE
    return {
        "config": config.__dict__,
        "baseline": {
            "v7": baseline_score,
            "v12": v12_score,
            "zero_transport": zero_score,
            "identity_error": identity_error,
            "identity_tolerance": IDENTITY_TOLERANCE,
        },
        "transport": {
            "pas_truth_residual_mass_mean": float(np.mean(truth_pas_plan.residual_mass)),
            "pas_teacher_residual_mass_mean": float(np.mean(teacher_pas_plan.residual_mass)),
            "pdp_truth_residual_mass_mean": float(np.mean(truth_pdp_plan.residual_mass)),
            "pdp_teacher_residual_mass_mean": float(np.mean(teacher_pdp_plan.residual_mass)),
            "maximum_marginal_error": float(
                max(
                    truth_pas_plan.marginal_error,
                    teacher_pas_plan.marginal_error,
                    truth_pdp_plan.marginal_error,
                    teacher_pdp_plan.marginal_error,
                )
            ),
            "maximum_sinkhorn_iterations_used": int(
                max(
                    truth_pas_plan.maximum_iterations_used,
                    teacher_pas_plan.maximum_iterations_used,
                    truth_pdp_plan.maximum_iterations_used,
                    teacher_pdp_plan.maximum_iterations_used,
                )
            ),
            "pas_relation": pas_relation,
            "pdp_relation": pdp_relation,
            "pas_action_masks": pas_mask_records,
            "pdp_action_masks": pdp_mask_records,
        },
        "capacity": {
            "shared_transport_oracle": {
                "path": PATHS.relative(SHARED_COMPLETE),
                "official": shared_score,
                "delta_vs_v7": float(shared_score["score"] - baseline_score["score"]),
                "margin_vs_v12": float(shared_score["score"] - V12_SCORE),
                "pas_endpoint_cosine": endpoint_cosine(shared_pas, truth_pas),
                "pdp_endpoint_cosine": endpoint_cosine(shared_pdp, truth_pdp),
                "fold_delta_vs_v7": [
                    float(current["score"] - reference["score"])
                    for current, reference in zip(shared_folds, baseline_folds, strict=True)
                ],
            },
            "all_transport_oracle": {
                "path": PATHS.relative(ALL_COMPLETE),
                "official": all_score,
                "delta_vs_v7": float(all_score["score"] - baseline_score["score"]),
                "pas_endpoint_cosine": endpoint_cosine(all_pas, truth_pas),
                "pdp_endpoint_cosine": endpoint_cosine(all_pdp, truth_pdp),
                "fold_delta_vs_v7": [
                    float(current["score"] - reference["score"])
                    for current, reference in zip(all_folds, baseline_folds, strict=True)
                ],
            },
        },
        "decision": {
            "required_score": V12_SCORE,
            "relation_passed": relation_passed,
            "capacity_passed": capacity_passed,
            "passed": relation_passed and capacity_passed,
        },
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    result = run(torch.device(args.device), args)
    payload = {
        "experiment": VERSION,
        "status": "mechanism_gate_passed" if result["decision"]["passed"] else "mechanism_gate_failed",
        "diagnostic_only": True,
        "official_inputs_only_final_target": True,
        "v12_used_as_privileged_reference": True,
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
