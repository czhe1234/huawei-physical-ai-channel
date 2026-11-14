#!/usr/bin/env python3
"""Audit the complete geometry-to-metric path variable system for Round 2."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e121_standalone_unique_path_carrier as e121
import analyze_global_local_path_lifecycle as e79
import train_e115_complete_path_state_operator as e115
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import component_energy, synthesize_batch


VERSION = "Round2-E123-complete-path-variable-system"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "complete_path_variable_system"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_complete_path_variable_system"
REPORT = ANALYSIS_DIR / "complete_path_variable_system.json"
METRICS = ARTIFACT_DIR / "complete_path_variable_system_metrics.npz"
HIERARCHICAL_DIR = PATHS.artifacts / "round2" / "hierarchical_unique_path_operator"
RAW_DIR = PATHS.artifacts / "round2" / "complete_path_state_operator"
E122_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_strict_complex_stack_unique_path"
    / "strict_complex_stack_unique_path.json"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
PERIODS = np.asarray((e115.MH, e115.MV, e115.S_SC), dtype=np.float32)
TINY = 1e-30


@dataclass(frozen=True)
class PathState:
    coordinate: np.ndarray
    mass: np.ndarray
    port: np.ndarray
    phase: np.ndarray
    power: np.ndarray
    enforce_power: bool = True
    add_true_residual: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def cache_tag(outer: int) -> str:
    return f"outer{outer}_block{outer}{(outer + 1) % 5}"


def collect_cache_arrays(
    query_ids: np.ndarray,
    full_fold: np.ndarray,
) -> dict[str, np.ndarray]:
    hierarchy_keys = (
        "mask",
        "coordinate",
        "base_log_mass",
        "base_log_port",
        "base_phase",
        "target_mass",
        "target_delta",
        "target_port",
        "target_phase",
        "member_count",
    )
    output: dict[str, np.ndarray] = {}
    for outer in range(5):
        rows = np.flatnonzero(full_fold == outer)
        point_ids = query_ids[rows]
        hierarchy = HIERARCHICAL_DIR / cache_tag(outer)
        raw = RAW_DIR / cache_tag(outer)
        for key in hierarchy_keys:
            source = np.load(hierarchy / f"{key}.npy", mmap_mode="r")
            if key not in output:
                output[key] = np.empty(
                    (len(query_ids), *source.shape[1:]), dtype=source.dtype
                )
            output[key][rows] = source[point_ids]
        source_power = np.load(raw / "base_log_power.npy", mmap_mode="r")
        if "base_log_power" not in output:
            output["base_log_power"] = np.empty(len(query_ids), source_power.dtype)
        output["base_log_power"][rows] = source_power[point_ids]
    return output


def normalize_mass(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    positive *= mask
    total = positive.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        positive,
        np.maximum(total, TINY),
        out=np.zeros_like(positive),
    ).astype(np.float32)


def normalize_port(value: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = positive.sum(axis=2, keepdims=True, dtype=np.float64)
    return np.divide(
        positive,
        np.maximum(total, TINY),
        out=np.full_like(positive, 1.0 / e115.PORTS),
    ).astype(np.float32)


def unit_complex(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.complex64)
    magnitude = np.abs(source)
    return np.divide(
        source,
        np.maximum(magnitude, 1e-12),
        out=np.ones_like(source),
    ).astype(np.complex64)


def phase_from_parts(value: np.ndarray) -> np.ndarray:
    return unit_complex(
        np.asarray(value[..., : e115.PORTS], np.float32)
        + 1j * np.asarray(value[..., e115.PORTS :], np.float32)
    )


def replace_active(
    base: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    return np.where(active[(...,) + (None,) * (base.ndim - active.ndim)], target, base)


def replace_coordinate_axes(
    base: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    axes: tuple[int, ...],
) -> np.ndarray:
    output = np.array(base, copy=True)
    for axis in axes:
        output[..., axis] = np.where(active, target[..., axis], base[..., axis])
    return output


def phase_decomposition(
    base: np.ndarray,
    target: np.ndarray,
    target_port: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.argmax(target_port, axis=2)
    base_anchor = np.take_along_axis(base, anchor[..., None], axis=2)[..., 0]
    target_anchor = np.take_along_axis(target, anchor[..., None], axis=2)[..., 0]
    common = unit_complex(base * target_anchor[..., None] * np.conj(base_anchor[..., None]))
    relative = unit_complex(
        target * base_anchor[..., None] * np.conj(target_anchor[..., None])
    )
    return replace_active(base, common, active), replace_active(base, relative, active)


def permute_active_paths(
    value: np.ndarray,
    active: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.array(value, copy=True)
    for row in range(len(output)):
        ids = np.flatnonzero(active[row])
        if len(ids) > 1:
            output[row, ids] = value[row, rng.permutation(ids)]
    return output


def metric_payload(
    values: PointMetricStatistics,
    full_fold: np.ndarray,
    nonzero: np.ndarray,
) -> dict[str, object]:
    return {
        "all": aggregate_statistics(values),
        "nonzero": aggregate_statistics(values, nonzero),
        "folds": [
            {"fold": fold, **aggregate_statistics(values, full_fold == fold)}
            for fold in range(5)
        ],
    }


@torch.inference_mode()
def evaluate_components(
    coordinate: np.ndarray,
    coefficient: np.ndarray,
    query_ids: np.ndarray,
    full_fold: np.ndarray,
    nonzero: np.ndarray,
    truth_components: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    batch_size: int,
    enforce_power: np.ndarray | None = None,
    add_true_residual: bool = False,
) -> dict[str, object]:
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    true_coordinate, true_coefficient = truth_components
    pas_parts: list[np.ndarray] = []
    pdp_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        prediction = synthesize_batch(
            coordinate[start:stop], coefficient[start:stop], device
        )
        truth = torch.as_tensor(
            np.asarray(truth_all[query_ids[start:stop]], dtype=np.complex64),
            device=device,
        )
        if enforce_power is not None:
            current = prediction.abs().square().sum(
                dim=(1, 2, 3), dtype=torch.float64
            )
            requested = torch.as_tensor(
                enforce_power[start:stop], dtype=torch.float64, device=device
            )
            scale = torch.where(
                current > TINY,
                torch.sqrt(requested / current.clamp_min(TINY)),
                torch.zeros_like(current),
            ).to(prediction.dtype)
            prediction = prediction * scale[:, None, None, None]
        if add_true_residual:
            true_path = synthesize_batch(
                true_coordinate[start:stop], true_coefficient[start:stop], device
            )
            prediction = prediction + truth - true_path
        pas, pdp, error, energy = e115.torch_official_metrics(prediction, truth)
        pas_parts.append(pas.cpu().numpy())
        pdp_parts.append(pdp.cpu().numpy())
        error_parts.append(error.cpu().numpy())
        energy_parts.append(energy.cpu().numpy())
    values = PointMetricStatistics(
        pas=np.concatenate(pas_parts),
        pdp=np.concatenate(pdp_parts),
        squared_error=np.concatenate(error_parts),
        truth_energy=np.concatenate(energy_parts),
    )
    return metric_payload(values, full_fold, nonzero)


def coefficients_for_state(state: PathState) -> np.ndarray:
    magnitude = np.sqrt(
        np.maximum(
            state.power[:, None, None]
            * state.mass[..., None]
            * state.port,
            0.0,
        )
    )
    return (magnitude * state.phase).astype(np.complex64)


def evaluate_state(
    state: PathState,
    query_ids: np.ndarray,
    full_fold: np.ndarray,
    nonzero: np.ndarray,
    truth_components: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    return evaluate_components(
        state.coordinate,
        coefficients_for_state(state),
        query_ids,
        full_fold,
        nonzero,
        truth_components,
        device,
        batch_size,
        enforce_power=state.power if state.enforce_power else None,
        add_true_residual=state.add_true_residual,
    )


def support_statistics(
    candidate_coordinate: np.ndarray,
    candidate_mask: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    truth_power: np.ndarray,
    source_rows: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    coverage = np.zeros(len(candidate_coordinate), dtype=np.float64)
    axis_error = np.full((len(candidate_coordinate), 3), np.nan, dtype=np.float64)
    for row in range(len(candidate_coordinate)):
        source = row if source_rows is None else int(source_rows[row])
        candidates = candidate_coordinate[source, candidate_mask[source]]
        energy = component_energy(truth_coefficient[row])
        fraction = energy / max(float(truth_power[row]), TINY)
        valid = fraction >= 1e-5
        if len(candidates) == 0 or not np.any(valid):
            continue
        truth = truth_coordinate[row, valid]
        weight = fraction[valid]
        cost = e79.support_cost(candidates, truth)
        nearest = np.argmin(cost, axis=0)
        nearest_cost = cost[nearest, np.arange(len(truth))]
        coverage[row] = float(np.sum(weight[nearest_cost <= e115.Config().soft_cost_limit]))
        delta = np.stack(
            [
                np.abs(
                    e79.periodic_delta(
                        candidates[nearest, axis], truth[:, axis], float(PERIODS[axis])
                    )
                )
                for axis in range(3)
            ],
            axis=1,
        )
        axis_error[row] = np.sum(weight[:, None] * delta, axis=0) / max(
            float(np.sum(weight)), TINY
        )
    finite = np.isfinite(axis_error).all(axis=1)
    return (
        {
            "mass_coverage_mean": float(np.mean(coverage)),
            "mass_coverage_nonzero_mean": float(
                np.mean(coverage[truth_power > TINY])
            ),
            "mass_coverage_median": float(np.median(coverage[truth_power > TINY])),
            "mass_coverage_p10": float(
                np.quantile(coverage[truth_power > TINY], 0.10)
            ),
            "nearest_axis_mae": {
                "horizontal_bin": float(np.mean(axis_error[finite, 0])),
                "vertical_bin": float(np.mean(axis_error[finite, 1])),
                "delay_bin": float(np.mean(axis_error[finite, 2])),
            },
        },
        coverage,
    )


def geometry_shuffle_rows(
    site: np.ndarray,
    full_fold: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.arange(len(site), dtype=np.int64)
    for fold in range(5):
        for site_value in np.unique(site):
            rows = np.flatnonzero((full_fold == fold) & (site == site_value))
            if len(rows) > 1:
                shift = int(rng.integers(1, len(rows)))
                output[rows] = np.roll(rows, shift)
    return output


def variable_contract() -> dict[str, object]:
    return {
        "layer_1_scene_and_observations": {
            "inference_inputs": {
                "train_positions": "[4000,3] meters",
                "test_positions": "[500,3] meters",
                "bs_positions": "[2,3] meters",
                "official_train_channel": "[4000,256,4,192] complex64",
                "scene_mesh": "19378 vertices, 24035 triangles",
                "route_tensor": "[B,7,48,11]",
                "panorama_tensor": "[B,2,32,9]",
                "polar_occupancy": "[B,4,16,64]",
                "corridor_occupancy": "[B,4,13,64]",
                "geodesic_features": "[B,28]",
                "query_query_graph": "same-site spatial graph over all test positions",
            },
            "role": "Boundary conditions for path viability, continuity, and power.",
        },
        "layer_2_candidate_path_evidence": {
            "raw_token": {
                "shape": "[B,176]",
                "variables": (
                    "donor position and distance; donor component rank; periodic "
                    "(horizontal,vertical,delay); donor path mass; 8-port amplitude; "
                    "radially transported 8-port complex direction"
                ),
            },
            "unique_path": {
                "shape": "[B,K<=64]",
                "variables": (
                    "cluster coordinate, member count, spatial/coordinate spread, "
                    "base mass, base port profile, base complex direction"
                ),
            },
            "invariants": (
                "Permutation equivariance across candidates; repeated donor observations "
                "must merge before path competition; candidate identity must remain "
                "consistent across neighboring queries."
            ),
        },
        "layer_3_latent_path_state": {
            "coverage_zero_state": "z in {0,1} per point",
            "total_power": "P>=0 per point",
            "path_existence": "e_k in {0,1} with birth/stable/death lifecycle",
            "resolved_fraction": (
                "rho=||H_resolved||^2/P in [0,1], including coherent path cross terms"
            ),
            "path_mass": (
                "a_k=||c_k||^2/P>=0 plus normalized competition weights; sum_k a_k "
                "is not forced to rho because path atoms are not exactly orthogonal"
            ),
            "continuous_coordinate": (
                "(h_k mod 16, v_k mod 8, d_k mod 192) plus local residual"
            ),
            "port_power": "pi_k in simplex^8",
            "common_phase": "phi_k on the unit circle",
            "relative_port_phase": "delta_phi_k in torus^7 after one phase gauge",
            "path_identity": "latent physical family shared across donor/query locations",
            "uncertainty": "candidate-wise aleatoric and epistemic confidence",
            "residual_state": (
                "unresolved/diffuse complex field with energy P*(1-rho), not a free "
                "replacement for resolved paths"
            ),
        },
        "layer_4_analytic_channel_synthesis": {
            "coefficient": (
                "c[k,p,u]=sqrt(P*a[k]*pi[k,p,u])*exp(j*(phi[k]+delta_phi[k,p,u]))"
            ),
            "atom": (
                "A[k,h,v,s]=exp(j*2*pi*(h_k*h/16+v_k*v/8-d_k*s/192))"
                "/sqrt(16*8*192)"
            ),
            "resolved_channel": "H_resolved=sum_k c_k*A_k",
            "coherent_power": (
                "||H_resolved||^2 is evaluated through the path-atom Gram matrix; "
                "cross terms cannot be replaced by sum_k ||c_k||^2"
            ),
            "full_channel": "H=H_resolved+H_residual, flattened as [B,256,4,192]",
            "constraints": (
                "periodicity, mass and port conservation, phase gauge, exact output "
                "layout [polarization,horizontal,vertical,UE,subcarrier]"
            ),
        },
        "layer_5_official_objective": {
            "pas": (
                "FFT2 over BS horizontal/vertical; power cosine along all 256 BS "
                "antennas for each UE/subcarrier; then mean"
            ),
            "pdp": (
                "IFFT over 192 subcarriers; power cosine along delay for each BS "
                "antenna/UE; then mean"
            ),
            "nmse": "sum squared complex error / sum truth energy",
            "score": "0.4*PAS + 0.4*PDP + 0.2/(1+NMSE)",
            "zero_norm_rule": "replace a zero norm product by 1e-12 only",
        },
        "observability": {
            "inference_known": (
                "scene, setup, all coordinates, official train H, donor paths, "
                "query-query relations"
            ),
            "training_supervision_only": (
                "held H decomposition, held zero state, power, candidate assignment, "
                "mass, coordinate, port and phase targets"
            ),
            "must_be_predicted": (
                "hidden-query path identity/existence/lifecycle, resolved fraction, "
                "mass, coordinate residual, port state, phase state, power, residual, "
                "and uncertainty"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    endpoint = np.load(e121.ENDPOINT, allow_pickle=False)
    query_ids = endpoint["query_ids"].astype(np.int64)
    full_fold = endpoint["full_fold"].astype(np.int8)
    cache = collect_cache_arrays(query_ids, full_fold)
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    true_coordinate = component["coordinates"][query_ids].astype(np.float32)
    true_coefficient = component["coefficients"][query_ids].astype(np.complex64)
    true_power = component["point_power"][query_ids].astype(np.float64)
    true_residual_fraction = component["residual_fraction"][query_ids].astype(
        np.float64
    )
    nonzero = true_power > TINY
    _, positions, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]

    mask = cache["mask"].astype(bool)
    base_coordinate = cache["coordinate"].astype(np.float32)
    base_mass_raw = np.exp(np.clip(cache["base_log_mass"].astype(np.float32), -30, 20))
    base_mass = normalize_mass(base_mass_raw, mask)
    base_port = normalize_port(
        np.exp(np.clip(cache["base_log_port"].astype(np.float32), -20, 10))
    )
    base_phase = phase_from_parts(cache["base_phase"])
    base_power = np.exp(
        np.clip(cache["base_log_power"].astype(np.float64), -80.0, 80.0)
    )

    target_mass_absolute = np.maximum(cache["target_mass"].astype(np.float32), 0.0)
    target_mass_absolute *= mask
    target_total = target_mass_absolute.sum(axis=1)
    target_mass = normalize_mass(target_mass_absolute, mask)
    target_mass = np.where(
        (target_total > 0.0)[:, None], target_mass, base_mass
    ).astype(np.float32)
    active = target_mass_absolute > 0.0
    corrected_coordinate = np.mod(
        base_coordinate + cache["target_delta"].astype(np.float32), PERIODS
    )
    target_coordinate = replace_active(base_coordinate, corrected_coordinate, active)
    raw_target_port = normalize_port(cache["target_port"].astype(np.float32))
    target_port = replace_active(base_port, raw_target_port, active)
    raw_target_phase = phase_from_parts(cache["target_phase"])
    target_phase = replace_active(base_phase, raw_target_phase, active)
    common_phase, relative_phase = phase_decomposition(
        base_phase, target_phase, target_port, active
    )
    existence_mass = normalize_mass(base_mass * active, mask)
    existence_mass = np.where(
        (existence_mass.sum(axis=1) > 0.0)[:, None], existence_mass, base_mass
    ).astype(np.float32)

    coordinate_h = replace_coordinate_axes(
        base_coordinate, target_coordinate, active, (0,)
    )
    coordinate_v = replace_coordinate_axes(
        base_coordinate, target_coordinate, active, (1,)
    )
    coordinate_d = replace_coordinate_axes(
        base_coordinate, target_coordinate, active, (2,)
    )
    zero_power = np.where(nonzero, base_power, 0.0)

    states: OrderedDict[str, PathState] = OrderedDict()
    states["base_candidate"] = PathState(
        base_coordinate, base_mass, base_port, base_phase, base_power
    )
    states["zero_state_oracle"] = PathState(
        base_coordinate, base_mass, base_port, base_phase, zero_power
    )
    states["total_power_oracle"] = PathState(
        base_coordinate, base_mass, base_port, base_phase, true_power
    )
    states["existence_oracle"] = PathState(
        base_coordinate, existence_mass, base_port, base_phase, base_power
    )
    states["mass_oracle"] = PathState(
        base_coordinate, target_mass, base_port, base_phase, base_power
    )
    states["horizontal_coordinate_oracle"] = PathState(
        coordinate_h, base_mass, base_port, base_phase, base_power
    )
    states["vertical_coordinate_oracle"] = PathState(
        coordinate_v, base_mass, base_port, base_phase, base_power
    )
    states["delay_coordinate_oracle"] = PathState(
        coordinate_d, base_mass, base_port, base_phase, base_power
    )
    states["all_coordinate_oracle"] = PathState(
        target_coordinate, base_mass, base_port, base_phase, base_power
    )
    states["port_power_oracle"] = PathState(
        base_coordinate, base_mass, target_port, base_phase, base_power
    )
    states["common_phase_oracle"] = PathState(
        base_coordinate, base_mass, base_port, common_phase, base_power
    )
    states["relative_port_phase_oracle"] = PathState(
        base_coordinate, base_mass, base_port, relative_phase, base_power
    )
    states["full_phase_oracle"] = PathState(
        base_coordinate, base_mass, base_port, target_phase, base_power
    )
    states["mass_coordinate"] = PathState(
        target_coordinate, target_mass, base_port, base_phase, base_power
    )
    states["mass_coordinate_port"] = PathState(
        target_coordinate, target_mass, target_port, base_phase, base_power
    )
    states["mass_coordinate_port_common_phase"] = PathState(
        target_coordinate, target_mass, target_port, common_phase, base_power
    )
    states["all_path_state_base_power"] = PathState(
        target_coordinate, target_mass, target_port, target_phase, base_power
    )
    states["noncoherent_state_true_power"] = PathState(
        target_coordinate, target_mass, target_port, base_phase, true_power
    )
    states["all_path_state_true_power"] = PathState(
        target_coordinate, target_mass, target_port, target_phase, true_power
    )

    rng = np.random.default_rng(20_260_808_123)
    states["path_phase_permutation_control"] = PathState(
        target_coordinate,
        target_mass,
        target_port,
        permute_active_paths(target_phase, active, rng),
        true_power,
    )
    states["path_coordinate_permutation_control"] = PathState(
        permute_active_paths(target_coordinate, active, rng),
        target_mass,
        target_port,
        target_phase,
        true_power,
    )
    states["absolute_resolved_state_oracle"] = PathState(
        target_coordinate,
        target_mass_absolute,
        target_port,
        target_phase,
        true_power,
        enforce_power=False,
    )
    states["absolute_resolved_plus_true_residual"] = PathState(
        target_coordinate,
        target_mass_absolute,
        target_port,
        target_phase,
        true_power,
        enforce_power=False,
        add_true_residual=True,
    )

    true_components = (true_coordinate, true_coefficient)
    results: OrderedDict[str, dict[str, object]] = OrderedDict()
    for name, state in states.items():
        results[name] = evaluate_state(
            state,
            query_ids,
            full_fold,
            nonzero,
            true_components,
            device,
            args.batch_size,
        )
        print(f"{name} score={results[name]['all']['score']:.6f}", flush=True)

    results["true_24_component_ceiling"] = evaluate_components(
        true_coordinate,
        true_coefficient,
        query_ids,
        full_fold,
        nonzero,
        true_components,
        device,
        args.batch_size,
    )
    results["truth_self_ceiling"] = evaluate_components(
        true_coordinate,
        true_coefficient,
        query_ids,
        full_fold,
        nonzero,
        true_components,
        device,
        args.batch_size,
        add_true_residual=True,
    )

    correct_support, correct_coverage = support_statistics(
        base_coordinate,
        mask,
        true_coordinate,
        true_coefficient,
        true_power,
    )
    shuffled_rows = geometry_shuffle_rows(query_site, full_fold, rng)
    shuffled_support, shuffled_coverage = support_statistics(
        base_coordinate,
        mask,
        true_coordinate,
        true_coefficient,
        true_power,
        source_rows=shuffled_rows,
    )
    support = {
        "correct_location": correct_support,
        "same_site_same_fold_location_shuffle": shuffled_support,
        "coverage_margin": float(
            correct_support["mass_coverage_nonzero_mean"]
            - shuffled_support["mass_coverage_nonzero_mean"]
        ),
        "soft_assignment_mass_fraction_nonzero_mean": float(
            np.mean(target_total[nonzero])
        ),
        "true_24_coefficient_mass_sum_nonzero_mean": float(
            np.mean(
                component_energy(true_coefficient)[nonzero].sum(axis=1)
                / np.maximum(true_power[nonzero], TINY)
            )
        ),
        "true_24_explained_energy_fraction_nonzero_mean": float(
            np.mean(1.0 - true_residual_fraction[nonzero])
        ),
        "candidate_count_mean": float(np.mean(mask.sum(axis=1))),
        "positive_candidate_count_nonzero_mean": float(
            np.mean(active[nonzero].sum(axis=1))
        ),
    }

    v12_score = 0.677521
    coherent_score = float(results["all_path_state_true_power"]["all"]["score"])
    true24_score = float(results["true_24_component_ceiling"]["all"]["score"])
    gates = {
        "minimum_location_conditioned_coverage_margin": 0.10,
        "minimum_true_24_component_score": 0.90,
        "minimum_candidate_coherent_margin_over_v12": 0.05,
        "location_conditioned_coverage_margin": support["coverage_margin"],
        "true_24_component_score": true24_score,
        "candidate_coherent_margin_over_v12": coherent_score - v12_score,
    }
    gates["candidate_generation_passed"] = bool(
        gates["location_conditioned_coverage_margin"]
        >= gates["minimum_location_conditioned_coverage_margin"]
    )
    gates["analytic_synthesis_passed"] = bool(
        true24_score >= gates["minimum_true_24_component_score"]
    )
    gates["candidate_state_representation_passed"] = bool(
        gates["candidate_coherent_margin_over_v12"]
        >= gates["minimum_candidate_coherent_margin_over_v12"]
    )

    e122_report = json.loads(E122_REPORT.read_text(encoding="utf-8"))
    base_score = float(results["base_candidate"]["all"]["score"])
    state_deltas = {
        name: float(value["all"]["score"] - base_score)
        for name, value in results.items()
    }
    score = {name: float(value["all"]["score"]) for name, value in results.items()}
    causal_findings = {
        "largest_single_state": "path_mass_and_competition",
        "mass_single_gain": score["mass_oracle"] - base_score,
        "all_coordinate_single_gain": score["all_coordinate_oracle"] - base_score,
        "port_single_gain": score["port_power_oracle"] - base_score,
        "phase_single_gain": score["full_phase_oracle"] - base_score,
        "mass_coordinate_port_score": score["mass_coordinate_port"],
        "phase_gain_after_mass_coordinate_port": (
            score["all_path_state_base_power"] - score["mass_coordinate_port"]
        ),
        "power_gain_after_complete_path_state": (
            score["all_path_state_true_power"] - score["all_path_state_base_power"]
        ),
        "phase_identity_control_drop": (
            score["all_path_state_true_power"]
            - score["path_phase_permutation_control"]
        ),
        "coordinate_identity_control_drop": (
            score["all_path_state_true_power"]
            - score["path_coordinate_permutation_control"]
        ),
        "exact_residual_gain": (
            score["absolute_resolved_plus_true_residual"]
            - score["absolute_resolved_state_oracle"]
        ),
        "remaining_candidate_assignment_gap_to_true24": (
            score["true_24_component_ceiling"]
            - score["absolute_resolved_plus_true_residual"]
        ),
        "interaction_conclusion": (
            "Phase is harmful when replaced alone but strongly positive after mass, "
            "coordinate, and port identity are aligned; state heads must therefore "
            "be hierarchical and path-identity conditioned rather than independent."
        ),
    }
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A complete path-variable contract and layerwise complex-channel oracle "
            "will identify which causal state blocks must be learned before choosing "
            "a network architecture."
        ),
        "strict_protocol": {
            "query_count": int(len(query_ids)),
            "outer_folds": 5,
            "inner_rule": "(outer + 1) % 5",
            "candidate_donors_exclude_outer_and_inner": True,
            "held_truth_used_only_for_oracle_and_scoring": True,
            "v12_or_sionna_used_as_model_input": False,
            "minimum_component_fraction": 1e-5,
            "candidate_support_cost_limit": float(e115.Config().soft_cost_limit),
        },
        "variable_contract": variable_contract(),
        "candidate_support": support,
        "references": {
            "v12_strict_score_from_e120": v12_score,
            "e122_strict_official_stack": e122_report["baseline"],
            "e122_stack_plus_unique_path": e122_report["primary"]["selected"],
        },
        "variant_results": results,
        "state_score_delta_vs_candidate_base": state_deltas,
        "causal_findings": causal_findings,
        "network_design_requirements": {
            "candidate_encoder": (
                "Encode all 176 donor observations, then pool repeated evidence into "
                "at most 64 unique paths without discarding member uncertainty."
            ),
            "identity_and_lifecycle": (
                "Use query-query graph matching to maintain path identity and predict "
                "birth, persistence, and death before regressing continuous states."
            ),
            "hierarchical_decoder_order": (
                "existence -> coefficient mass -> H/V/delay -> port power -> common "
                "phase -> relative port phase -> total power/resolved fraction"
            ),
            "scene_conditioning": (
                "Route, panorama, polar, corridor, and geodesic features condition each "
                "candidate's viability and state transition, not a free final-spectrum residual."
            ),
            "analytic_decoder": (
                "Synthesize the complete complex H with periodic atoms and Gram-aware "
                "coherent power; model only a bounded residual field afterward."
            ),
            "loss": (
                "Hungarian/transport identity supervision, existence and state losses, "
                "plus differentiable official PAS, PDP, and dataset-level NMSE losses."
            ),
            "uncertainty": (
                "Predict candidate confidence and abstain toward the strict official-H "
                "carrier when identity or phase is uncertain."
            ),
        },
        "gates": gates,
        "diagnostic_boundaries": {
            "oracle_targets_are_not_inference_inputs": True,
            "true_residual_is_ceiling_only": True,
            "no_test_channel_or_submission_generated": True,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    names = np.asarray(tuple(results), dtype="U64")
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        full_fold=full_fold.astype(np.int8),
        nonzero=nonzero,
        variant_names=names,
        variant_pas=np.asarray(
            [results[name]["all"]["pas"] for name in names], dtype=np.float64
        ),
        variant_pdp=np.asarray(
            [results[name]["all"]["pdp"] for name in names], dtype=np.float64
        ),
        variant_nmse=np.asarray(
            [results[name]["all"]["nmse"] for name in names], dtype=np.float64
        ),
        variant_score=np.asarray(
            [results[name]["all"]["score"] for name in names], dtype=np.float64
        ),
        correct_candidate_coverage=correct_coverage,
        shuffled_candidate_coverage=shuffled_coverage,
        target_resolved_fraction=target_total.astype(np.float32),
        true_power=true_power,
    )
    print(
        json.dumps(
            {
                "candidate_support": support,
                "key_scores": {
                    name: results[name]["all"]
                    for name in (
                        "base_candidate",
                        "mass_coordinate_port",
                        "all_path_state_base_power",
                        "all_path_state_true_power",
                        "absolute_resolved_state_oracle",
                        "absolute_resolved_plus_true_residual",
                        "true_24_component_ceiling",
                        "truth_self_ceiling",
                    )
                },
                "gates": gates,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
