#!/usr/bin/env python3
"""Calibrate same-position external paths with one-to-one official-H path labels."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

import analyze_e86_surface_endpoint_v11_blend as e86
import analyze_e100_locked_energy_angle_correction as e100
import analyze_global_local_path_lifecycle as e79
import analyze_v4_path_track_spectral_targets as track_targets
import optimize_e98_global_array_angle_mapping as e98
import train_e99_corrected_array_surface_model as e99
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_path_component_transport import component_energy
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E190-official-sionna-path-calibrator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "official_sionna_path_calibrator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_official_sionna_path_calibrator"
FULL_REPORT = ANALYSIS_DIR / "fixed500_report.json"
FULL_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"

COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
MAPPING_PARAMETERS = (
    PATHS.artifacts
    / "round2"
    / "corrected_array_surface_model"
    / "strict_mapping_parameters.npz"
)
OLD_ENDPOINT = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "strict_oof_geometry_sequence_path_endpoint.npz"
)
OLD_RENDERED_PAS = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_geometry_sequence_rendered_pas_f16.npy"
)
OLD_RENDERED_PDP = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_geometry_sequence_rendered_pdp_f16.npy"
)

PERIODS = np.asarray((e85.MH, e85.MV, e85.S_SC), dtype=np.float32)
PORTS = e85.MP * e85.N_UE
TINY = 1e-30
PRESENCE_SEED_COUNT = 24


@dataclass(frozen=True)
class Config:
    outer: int = 0
    validation_fold: int = 1
    epochs: int = 48
    minimum_epochs: int = 16
    validation_interval: int = 4
    batch_size: int = 64
    hidden: int = 96
    heads: int = 4
    sequence_layers: int = 2
    set_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.08
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    score_limit: float = 4.0
    port_limit: float = 2.0
    match_cost_limit: float = 1.0
    match_bandwidth: float = 0.25
    minimum_path_fraction: float = 1e-5
    spectral_weight: float = 1.0
    mass_weight: float = 0.75
    coordinate_weight: float = 0.20
    port_weight: float = 0.20
    existence_weight: float = 0.05
    residual_weight: float = 0.002
    random_state: int = 20260812190
    capability_minimum_delta: float = 0.010
    screen_minimum_delta: float = 0.005
    maximum_component_drop: float = 0.001


@dataclass
class PathLabels:
    mass: np.ndarray
    delta: np.ndarray
    port: np.ndarray
    matched_count: np.ndarray
    matched_fraction: np.ndarray
    valid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("capability", "screen", "aggregate"),
        default="capability",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--outer", type=int, default=0)
    return parser.parse_args()


def capability_report_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}_capability.json"


def screen_report_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}_screen.json"


def screen_endpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_screen_endpoint.npz"


def screen_channel_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_screen_channel.npy"


def screen_checkpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_screen_model.pt"


def refit_checkpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_refit_model.pt"


def required_inputs() -> tuple[Path, ...]:
    return (
        COMPONENTS,
        MAPPING_PARAMETERS,
        OLD_ENDPOINT,
        OLD_RENDERED_PAS,
        OLD_RENDERED_PDP,
        e85.PATH_CACHE,
        e85.PATH_FEATURES,
        e85.SEQUENCE_GEOMETRY,
        e85.SEQUENCE_HASH,
        e85.BASE_COORDINATE,
        e85.PORT_PRIOR,
        e86.V11_CHANNEL,
        e86.TRAIN_CHANNEL,
        e100.V12_CHANNEL,
    )


def periodic_delta(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    return (left - right + period / 2.0) % period - period / 2.0


def normalize_rows(value: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = positive.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        positive,
        np.maximum(total, TINY),
        out=np.zeros_like(positive),
    ).astype(np.float32)


def raw_prior(store: e85.DataStore) -> tuple[np.ndarray, np.ndarray]:
    fraction = np.asarray(store.fraction, dtype=np.float32)
    presence = np.asarray(store.presence, dtype=np.float32)
    anchored = np.maximum(fraction, 1e-12) * np.sqrt(
        np.maximum(presence, 1.0) / float(PRESENCE_SEED_COUNT)
    )
    mass = normalize_rows(anchored)
    port = np.asarray(store.port, dtype=np.float32).reshape(
        len(mass), e85.PATHS_PER_POINT, PORTS
    )
    port /= np.maximum(port.sum(axis=2, keepdims=True), TINY)
    return mass, port.astype(np.float32)


def corrected_coordinates(
    base_coordinate: np.ndarray,
    mapping_data: e98.MappingData,
    latent: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    return e99.coordinates_from_mapping(
        base_coordinate,
        mapping_data,
        torch.as_tensor(latent, dtype=torch.float32, device=device),
        device,
    )


def build_labels(
    candidate_coordinate: np.ndarray,
    candidate_mask: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    point_power: np.ndarray,
    fallback_mass: np.ndarray,
    fallback_port: np.ndarray,
    config: Config,
) -> tuple[PathLabels, dict[str, object]]:
    started = time.time()
    count, candidates, _ = candidate_coordinate.shape
    mass_absolute = np.zeros((count, candidates), dtype=np.float32)
    delta = np.zeros((count, candidates, 3), dtype=np.float32)
    port = np.array(fallback_port, dtype=np.float32, copy=True)
    matched_count = np.zeros(count, dtype=np.int16)
    matched_fraction = np.zeros(count, dtype=np.float64)
    available_fraction = np.zeros(count, dtype=np.float64)
    accepted_cost: list[np.ndarray] = []

    for row in range(count):
        candidate_ids = np.flatnonzero(candidate_mask[row])
        if len(candidate_ids) == 0 or point_power[row] <= TINY:
            continue
        truth_energy = component_energy(truth_coefficient[row])
        truth_fraction = truth_energy / max(float(point_power[row]), TINY)
        truth_ids = np.flatnonzero(truth_fraction >= config.minimum_path_fraction)
        available_fraction[row] = float(np.sum(truth_fraction[truth_ids]))
        if len(truth_ids) == 0:
            continue

        local_candidate = candidate_coordinate[row, candidate_ids]
        local_truth = truth_coordinate[row, truth_ids]
        cost = e79.support_cost(local_candidate, local_truth)
        utility = truth_fraction[truth_ids][None] * np.exp(
            -0.5 * cost / config.match_bandwidth
        )
        utility[cost > config.match_cost_limit] = 0.0
        candidate_local, truth_local = linear_sum_assignment(-utility)
        accepted = utility[candidate_local, truth_local] > 0.0
        candidate_local = candidate_local[accepted]
        truth_local = truth_local[accepted]
        if len(candidate_local) == 0:
            continue

        assigned_candidate = candidate_ids[candidate_local]
        assigned_truth = truth_ids[truth_local]
        coefficient = truth_coefficient[row, assigned_truth]
        assigned_fraction = truth_fraction[assigned_truth]
        mass_absolute[row, assigned_candidate] = assigned_fraction.astype(np.float32)
        for axis, period in enumerate(PERIODS):
            delta[row, assigned_candidate, axis] = periodic_delta(
                truth_coordinate[row, assigned_truth, axis],
                candidate_coordinate[row, assigned_candidate, axis],
                float(period),
            ).astype(np.float32)
        port[row, assigned_candidate] = normalize_rows(np.abs(coefficient) ** 2)
        matched_count[row] = len(assigned_candidate)
        matched_fraction[row] = float(np.sum(assigned_fraction))
        accepted_cost.append(cost[candidate_local, truth_local])

    mass = normalize_rows(mass_absolute)
    valid = matched_fraction > 0.0
    mass[~valid] = fallback_mass[~valid]
    delta[~valid] = 0.0
    port[~valid] = fallback_port[~valid]
    cost_values = (
        np.concatenate(accepted_cost) if accepted_cost else np.empty(0, dtype=np.float64)
    )
    nonzero = point_power > TINY
    report = {
        "point_count": int(count),
        "nonzero_count": int(np.sum(nonzero)),
        "valid_label_count": int(np.sum(valid)),
        "matched_count_nonzero_mean": float(np.mean(matched_count[nonzero])),
        "matched_count_nonzero_median": float(np.median(matched_count[nonzero])),
        "matched_fraction_nonzero_mean": float(np.mean(matched_fraction[nonzero])),
        "matched_fraction_available_ratio": float(
            np.sum(matched_fraction) / max(float(np.sum(available_fraction)), TINY)
        ),
        "accepted_cost_mean": float(np.mean(cost_values)) if len(cost_values) else 0.0,
        "accepted_cost_p90": float(np.quantile(cost_values, 0.90))
        if len(cost_values)
        else 0.0,
        "seconds": float(time.time() - started),
    }
    return PathLabels(mass, delta, port, matched_count, matched_fraction, valid), report


def save_labels(path: Path, labels: PathLabels, report: dict[str, object]) -> None:
    np.savez_compressed(
        path,
        mass=labels.mass,
        delta=labels.delta,
        port=labels.port,
        matched_count=labels.matched_count,
        matched_fraction=labels.matched_fraction,
        valid=labels.valid,
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )


def endpoint_arrays(
    model: nn.Module,
    store: e85.DataStore,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mass = np.zeros((len(rows), e85.PATHS_PER_POINT), dtype=np.float32)
    coordinate = np.zeros((len(rows), e85.PATHS_PER_POINT, 3), dtype=np.float32)
    port = np.zeros((len(rows), e85.PATHS_PER_POINT, e85.MP, e85.N_UE), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            local = rows[start:end]
            batch, _, _ = store.batch(local, mean, std, device)
            output = model(batch)
            mass[start:end] = output["mass"].cpu().numpy().astype(np.float32)
            coordinate[start:end] = (
                output["coordinate"].cpu().numpy() % PERIODS
            ).astype(np.float32)
            port[start:end] = output["port"].cpu().numpy().astype(np.float32)
    return mass, coordinate, port


def render_endpoint(
    mass: np.ndarray,
    coordinate: np.ndarray,
    port: np.ndarray,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(mass)
    pas = np.empty((count, 2, 16, 8, 4, 192), dtype=np.float32)
    pdp = np.empty((count, 256, 4, 192), dtype=np.float32)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        coefficient = np.sqrt(
            np.maximum(
                mass[start:end, :, None]
                * port[start:end].reshape(end - start, e85.PATHS_PER_POINT, PORTS),
                0.0,
            )
        ).astype(np.complex64)
        _, local_pas, local_pdp = track_targets.render_noncoherent(
            coordinate[start:end], coefficient, device
        )
        pas[start:end] = local_pas.reshape(end - start, 2, 16, 8, 4, 192)
        pdp[start:end] = local_pdp
    return pas, pdp


class CanonicalEvaluator:
    def __init__(self, device: torch.device) -> None:
        metadata = np.load(e85.OOF_META, allow_pickle=False)
        self.query_ids = metadata["query_ids"].astype(np.int64)
        self.fold_id = metadata["fold_id"].astype(np.int8)
        self.lookup = np.full(4000, -1, dtype=np.int64)
        self.lookup[self.query_ids] = np.arange(len(self.query_ids), dtype=np.int64)
        self.base_np = np.load(e86.V11_CHANNEL, mmap_mode="r")
        self.v12_np = np.load(e100.V12_CHANNEL, mmap_mode="r")
        self.truth_all = np.load(e86.TRAIN_CHANNEL, mmap_mode="r")
        self.device = device

    def evaluate(
        self,
        global_rows: np.ndarray,
        endpoint_pas: np.ndarray,
        endpoint_pdp: np.ndarray,
        collect_channel: bool = False,
    ) -> tuple[dict[str, float | int], dict[str, float | int], np.ndarray | None]:
        local = self.lookup[global_rows]
        if np.any(local < 0):
            raise ValueError("complete-channel evaluation requested non-canonical rows")
        base = torch.as_tensor(np.array(self.base_np[local], copy=True), device=self.device)
        truth = torch.as_tensor(
            np.asarray(self.truth_all[global_rows], dtype=np.complex64), device=self.device
        )
        v12 = torch.as_tensor(np.array(self.v12_np[local], copy=True), device=self.device)
        truth_pas = pas_distribution(truth)
        truth_pdp = pdp_distribution(truth)
        base_pas = pas_distribution(base)
        base_pdp = pdp_distribution(base)
        spec = e86.Candidate(
            endpoint="official_path_calibrator",
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
            selected_metrics = point_metrics(candidate, truth, truth_pas, truth_pdp)
            baseline_metrics = point_metrics(v12, truth, truth_pas, truth_pdp)
        selected = statistics(*selected_metrics)
        baseline = statistics(*baseline_metrics)
        channel = candidate.cpu().numpy().astype(np.complex64) if collect_channel else None
        return selected, baseline, channel

    def identity_error(self) -> float:
        old_pas = np.asarray(np.load(OLD_RENDERED_PAS, mmap_mode="r"), dtype=np.float32)
        old_pdp = np.asarray(np.load(OLD_RENDERED_PDP, mmap_mode="r"), dtype=np.float32)
        base = torch.as_tensor(np.array(self.base_np, copy=True), device=self.device)
        base_pas = pas_distribution(base)
        base_pdp = pdp_distribution(base)
        spec = e86.Candidate(
            endpoint="geometry_sequence",
            mode="sqrt",
            pas_beta=0.20,
            pdp_beta=0.20,
            order="pas_pdp",
            cycles=4,
        )
        with torch.inference_mode():
            replay = e86.make_candidate(
                base,
                base_pas,
                base_pdp,
                torch.as_tensor(old_pas, device=self.device),
                torch.as_tensor(old_pdp, device=self.device),
                spec,
            )
            reference = torch.as_tensor(
                np.array(self.v12_np, copy=True), device=self.device
            )
            return float(torch.max(torch.abs(replay - reference)).item())


def aggregate_five_folds(device: torch.device) -> dict[str, object]:
    started = time.time()
    evaluator = CanonicalEvaluator(device)
    old_endpoint = np.load(OLD_ENDPOINT, allow_pickle=False)
    old_query_ids = old_endpoint["query_ids"].astype(np.int64)
    old_lookup = np.full(4000, -1, dtype=np.int64)
    old_lookup[old_query_ids] = np.arange(len(old_query_ids), dtype=np.int64)
    fold_reports = []
    endpoint_mass = np.zeros(
        (len(evaluator.query_ids), e85.PATHS_PER_POINT), dtype=np.float32
    )
    endpoint_coordinate = np.zeros(
        (len(evaluator.query_ids), e85.PATHS_PER_POINT, 3), dtype=np.float32
    )
    endpoint_port = np.zeros(
        (len(evaluator.query_ids), e85.PATHS_PER_POINT, e85.MP, e85.N_UE),
        dtype=np.float32,
    )
    channel = np.zeros(
        (len(evaluator.query_ids), 256, 4, 192), dtype=np.complex64
    )
    for outer in range(5):
        report_path = screen_report_path(outer)
        endpoint_path = screen_endpoint_path(outer)
        channel_path = screen_channel_path(outer)
        missing = [
            PATHS.relative(path)
            for path in (report_path, channel_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"missing E190 fold outputs: {missing}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = evaluator.query_ids[evaluator.fold_id == outer]
        status = report.get("status")
        if status == "screen_complete":
            if not endpoint_path.exists():
                raise FileNotFoundError(
                    f"missing E190 fold endpoint: {PATHS.relative(endpoint_path)}"
                )
            endpoint = np.load(endpoint_path, allow_pickle=False)
            query_ids = endpoint["query_ids"].astype(np.int64)
            mass = endpoint["mass"].astype(np.float32)
            coordinate = endpoint["coordinate"].astype(np.float32)
            port = endpoint["port"].astype(np.float32)
            fallback = False
        elif status == "screen_failed_validation_fallback":
            if int(report["screen"]["best_epoch"]) != 0:
                raise RuntimeError(
                    f"E190 outer{outer} fallback has a nonzero best epoch"
                )
            query_ids = expected
            old_rows = old_lookup[query_ids]
            if np.any(old_rows < 0):
                raise RuntimeError(
                    f"E190 outer{outer} fallback endpoint identity mismatch"
                )
            mass = old_endpoint["mass"][old_rows].astype(np.float32)
            coordinate = old_endpoint["coordinate"][old_rows].astype(np.float32)
            port = old_endpoint["port"][old_rows].astype(np.float32)
            fallback = True
        else:
            raise RuntimeError(
                f"E190 outer{outer} screen has unsupported status {status!r}"
            )
        if not np.array_equal(query_ids, expected):
            raise RuntimeError(f"E190 outer{outer} query identity mismatch")
        rows = evaluator.lookup[query_ids]
        endpoint_mass[rows] = mass
        endpoint_coordinate[rows] = coordinate
        endpoint_port[rows] = port
        local_channel = np.load(channel_path, mmap_mode="r")
        if local_channel.shape != (len(query_ids), 256, 4, 192):
            raise RuntimeError(f"E190 outer{outer} channel shape mismatch")
        channel[rows] = np.asarray(local_channel, dtype=np.complex64)
        fold_reports.append(
            {
                "outer": outer,
                "count": int(len(query_ids)),
                "selected": report["screen"]["selected"],
                "baseline": report["screen"]["baseline"],
                "delta": report["screen"]["delta"],
                "best_epoch": int(report["screen"]["best_epoch"]),
                "fallback": fallback,
            }
        )

    truth = torch.as_tensor(
        np.asarray(evaluator.truth_all[evaluator.query_ids], dtype=np.complex64),
        device=device,
    )
    prediction = torch.as_tensor(channel, device=device)
    baseline_channel = torch.as_tensor(
        np.array(evaluator.v12_np, dtype=np.complex64, copy=True), device=device
    )
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    with torch.inference_mode():
        selected_metrics = point_metrics(prediction, truth, truth_pas, truth_pdp)
        baseline_metrics = point_metrics(
            baseline_channel, truth, truth_pas, truth_pdp
        )
    selected = statistics(*selected_metrics)
    baseline = statistics(*baseline_metrics)
    delta = metric_delta(selected, baseline)
    positive_folds = int(sum(row["delta"]["score"] > 0.0 for row in fold_reports))
    worst_fold_delta = float(min(row["delta"]["score"] for row in fold_reports))
    passed = bool(
        float(selected["score"]) >= 0.690000
        and positive_folds >= 4
        and worst_fold_delta >= -0.001000
    )
    np.save(FULL_CHANNEL, channel)
    np.savez_compressed(
        ARTIFACT_DIR / "fixed500_endpoint.npz",
        query_ids=evaluator.query_ids.astype(np.int32),
        fold_id=evaluator.fold_id,
        mass=endpoint_mass,
        coordinate=endpoint_coordinate,
        port=endpoint_port,
    )
    report = {
        "experiment": VERSION,
        "stage": "aggregate",
        "status": "fixed500_passed" if passed else "fixed500_failed",
        "selected": selected,
        "baseline": baseline,
        "delta": delta,
        "folds": fold_reports,
        "gate": {
            "minimum_score": 0.690000,
            "minimum_positive_folds": 4,
            "minimum_worst_fold_delta": -0.001000,
            "positive_folds": positive_folds,
            "worst_fold_delta": worst_fold_delta,
            "passed": passed,
        },
        "output": {
            "channel": PATHS.relative(FULL_CHANNEL),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": bool(np.isfinite(channel).all()),
        },
        "seconds": float(time.time() - started),
    }
    FULL_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def metric_delta(
    selected: dict[str, float | int], baseline: dict[str, float | int]
) -> dict[str, float]:
    return {
        "pas": float(selected["pas"] - baseline["pas"]),
        "pdp": float(selected["pdp"] - baseline["pdp"]),
        "nmse_score": float(selected["nmse_score"] - baseline["nmse_score"]),
        "score": float(selected["score"] - baseline["score"]),
    }


def label_objective(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    target_mass: torch.Tensor,
    target_delta: torch.Tensor,
    target_port: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pas, pdp = e85.render_spectra(output)
    pas_score = e85.cosine_sufficient_score(pas, target_pas, (1, 2, 3), (1,))
    pdp_score = e85.cosine_sufficient_score(pdp, target_pdp, (3,), (1, 2))
    spectral_score = 0.5 * (pas_score + pdp_score)
    mass_affinity = (
        torch.sqrt(output["mass"].clamp_min(1e-12))
        * torch.sqrt(target_mass.clamp_min(0.0))
    ).sum(dim=1)
    active = target_mass > 0.0
    weight = target_mass / target_mass.sum(dim=1, keepdim=True).clamp_min(TINY)
    coordinate_error = F.smooth_l1_loss(
        output["coordinate_residual"], target_delta, reduction="none"
    ).mean(dim=2)
    coordinate_loss = (weight * coordinate_error).sum(dim=1)
    port_loss = -(
        target_port.clamp_min(0.0)
        * torch.log(output["port"].reshape_as(target_port).clamp_min(1e-8))
    ).sum(dim=2)
    port_loss = (weight * port_loss).sum(dim=1)
    positive = active.sum().clamp_min(1)
    negative = (~active).sum().clamp_min(1)
    pos_weight = (negative / positive).clamp(1.0, 8.0).to(
        output["score_residual"].dtype
    )
    existence_loss = F.binary_cross_entropy_with_logits(
        output["score_residual"], active.to(output["score_residual"].dtype),
        pos_weight=pos_weight,
    )
    residual = (
        output["score_residual"].square().mean()
        + output["coordinate_residual"].square().mean()
        + output["port_residual"].square().mean()
    )
    loss = (
        config.spectral_weight * (1.0 - spectral_score.mean())
        + config.mass_weight * (1.0 - mass_affinity.mean())
        + config.coordinate_weight * coordinate_loss.mean()
        + config.port_weight * port_loss.mean()
        + config.existence_weight * existence_loss
        + config.residual_weight * residual
    )
    return loss, {
        "pas": pas_score.mean(),
        "pdp": pdp_score.mean(),
        "mass_affinity": mass_affinity.mean(),
        "coordinate": coordinate_loss.mean(),
        "port": port_loss.mean(),
        "existence": existence_loss,
    }


def train_epoch(
    model: e85.SurfaceInteractionModel,
    optimizer: torch.optim.Optimizer,
    store: e85.DataStore,
    labels: PathLabels,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    eligible = rows[labels.valid[rows] & (np.asarray(store.power[rows]) > 0.0)]
    order = rng.permutation(eligible)
    totals: dict[str, list[float]] = {
        name: [] for name in ("loss", "pas", "pdp", "mass_affinity", "coordinate", "port", "existence")
    }
    for start in range(0, len(order), config.batch_size):
        local = order[start : start + config.batch_size]
        batch, target_pas, target_pdp = store.batch(local, mean, std, device)
        output = model(batch)
        loss, values = label_objective(
            output,
            batch,
            target_pas,
            target_pdp,
            torch.as_tensor(labels.mass[local], device=device),
            torch.as_tensor(labels.delta[local], device=device),
            torch.as_tensor(labels.port[local], device=device),
            config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite E190 gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        totals["loss"].append(float(loss.detach().item()))
        for name, value in values.items():
            totals[name].append(float(value.detach().item()))
    return {name: float(np.mean(value)) for name, value in totals.items()}


def make_model(store: e85.DataStore, config: Config, device: torch.device) -> e85.SurfaceInteractionModel:
    model_config = e85.Config(
        hidden=config.hidden,
        heads=config.heads,
        sequence_layers=config.sequence_layers,
        set_layers=config.set_layers,
        feedforward=config.feedforward,
        dropout=config.dropout,
        epochs=config.epochs,
        minimum_epochs=config.minimum_epochs,
        validation_interval=config.validation_interval,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        score_limit=config.score_limit,
        port_limit=config.port_limit,
        random_state=config.random_state,
    )
    return e85.SurfaceInteractionModel(
        store.base.shape[-1],
        store.sequence_geometry.shape[-1],
        model_config,
        "geometry",
    ).to(device)


def train_screen_model(
    store: e85.DataStore,
    labels: PathLabels,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    evaluator: CanonicalEvaluator,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | None, int, list[dict[str, object]]]:
    torch.manual_seed(config.random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_state)
    model = make_model(store, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1), eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(config.random_state)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, object]] = []

    old = np.load(OLD_ENDPOINT, allow_pickle=False)
    lookup = np.full(4000, -1, dtype=np.int64)
    lookup[old["query_ids"].astype(np.int64)] = np.arange(len(old["query_ids"]), dtype=np.int64)
    validation_local = lookup[validation_rows]
    old_pas, old_pdp = render_endpoint(
        old["mass"][validation_local].astype(np.float32),
        old["coordinate"][validation_local].astype(np.float32),
        old["port"][validation_local].astype(np.float32),
        device,
    )
    _, baseline, _ = evaluator.evaluate(validation_rows, old_pas, old_pdp)
    best_score = float(baseline["score"])
    history.append({"epoch": 0, "validation": baseline, "fallback": "V12"})

    for epoch in range(1, config.epochs + 1):
        train = train_epoch(
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
            mass, coordinate, port = endpoint_arrays(
                model, store, validation_rows, mean, std, device, config.batch_size
            )
            endpoint_pas, endpoint_pdp = render_endpoint(
                mass, coordinate, port, device
            )
            selected, baseline, _ = evaluator.evaluate(
                validation_rows, endpoint_pas, endpoint_pdp
            )
            record["validation"] = selected
            record["validation_delta"] = metric_delta(selected, baseline)
            if float(selected["score"]) > best_score:
                best_score = float(selected["score"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            validation_score = (
                float(record["validation"]["score"])
                if "validation" in record
                else float("nan")
            )
            print(
                f"E190 epoch={epoch}/{config.epochs} "
                f"train_mass={train['mass_affinity']:.6f} "
                f"train_pas={train['pas']:.6f} train_pdp={train['pdp']:.6f} "
                f"validation={validation_score:.6f}",
                flush=True,
            )
    return best_state, best_epoch, history


def refit_model(
    store: e85.DataStore,
    labels: PathLabels,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    config: Config,
    device: torch.device,
) -> tuple[e85.SurfaceInteractionModel, list[dict[str, float]]]:
    torch.manual_seed(config.random_state + 100000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_state + 100000)
    model = make_model(store, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(config.random_state + 100000)
    history = []
    for epoch in range(1, epochs + 1):
        values = train_epoch(
            model,
            optimizer,
            store,
            labels,
            rows,
            mean,
            std,
            config,
            device,
            rng,
        )
        scheduler.step()
        history.append({"epoch": epoch, **values})
        if epoch % config.validation_interval == 0 or epoch == epochs:
            print(
                f"E190 refit epoch={epoch}/{epochs} "
                f"mass={values['mass_affinity']:.6f} "
                f"pas={values['pas']:.6f} pdp={values['pdp']:.6f}",
                flush=True,
            )
    return model, history


def main() -> None:
    args = parse_args()
    started = time.time()
    phase_time: dict[str, float] = {}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if args.stage == "aggregate":
        report = aggregate_five_folds(device)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    if args.outer not in range(5):
        raise ValueError("E190 outer must be in [0, 4]")
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E190 inputs: {missing}")
    config = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        outer=args.outer,
        validation_fold=(args.outer + 1) % 5,
    )

    loaded = time.time()
    store = e85.DataStore()
    evaluator = CanonicalEvaluator(device)
    identity_error = evaluator.identity_error()
    if identity_error > 1e-6:
        raise RuntimeError(f"V12 identity replay failed: {identity_error}")
    component = np.load(COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    base_coordinate = np.array(store.coordinate, dtype=np.float32, copy=True)
    candidate_mask = np.asarray(
        np.load(e85.PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool
    )
    fallback_mass, fallback_port = raw_prior(store)
    mapping = np.load(MAPPING_PARAMETERS, allow_pickle=False)
    mapping_data = e98.MappingData(device)
    phase_time["load_and_identity"] = float(time.time() - loaded)

    screen_start = time.time()
    screen_coordinate = corrected_coordinates(
        base_coordinate,
        mapping_data,
        mapping["screen_latent"][config.outer].astype(np.float32),
        device,
    )
    screen_labels, screen_label_report = build_labels(
        screen_coordinate,
        candidate_mask,
        truth_coordinate,
        truth_coefficient,
        point_power,
        fallback_mass,
        fallback_port,
        config,
    )
    save_labels(
        ARTIFACT_DIR / f"outer{config.outer}_screen_labels.npz",
        screen_labels,
        screen_label_report,
    )
    phase_time["screen_labels"] = float(time.time() - screen_start)

    refit_start = time.time()
    refit_coordinate = corrected_coordinates(
        base_coordinate,
        mapping_data,
        mapping["refit_latent"][config.outer].astype(np.float32),
        device,
    )
    refit_labels, refit_label_report = build_labels(
        refit_coordinate,
        candidate_mask,
        truth_coordinate,
        truth_coefficient,
        point_power,
        fallback_mass,
        fallback_port,
        config,
    )
    save_labels(
        ARTIFACT_DIR / f"outer{config.outer}_refit_labels.npz",
        refit_labels,
        refit_label_report,
    )
    phase_time["refit_labels"] = float(time.time() - refit_start)

    strict_outer = evaluator.query_ids[evaluator.fold_id == config.outer]
    strict_validation = evaluator.query_ids[
        evaluator.fold_id == config.validation_fold
    ]
    all_rows = np.arange(len(store.base), dtype=np.int64)
    screen_fit = all_rows[
        (store.fold != config.outer) & (store.fold != config.validation_fold)
    ]
    refit_rows = all_rows[store.fold != config.outer]

    capability_start = time.time()
    outer_mass = refit_labels.mass[strict_outer]
    outer_coordinate = np.mod(
        refit_coordinate[strict_outer] + refit_labels.delta[strict_outer], PERIODS
    ).astype(np.float32)
    outer_port = refit_labels.port[strict_outer].reshape(
        len(strict_outer), e85.PATHS_PER_POINT, e85.MP, e85.N_UE
    )
    capability_pas, capability_pdp = render_endpoint(
        outer_mass, outer_coordinate, outer_port, device
    )
    capability_selected, capability_baseline, _ = evaluator.evaluate(
        strict_outer, capability_pas, capability_pdp
    )
    capability_delta = metric_delta(capability_selected, capability_baseline)
    capability_passed = bool(
        capability_delta["score"] >= config.capability_minimum_delta
    )
    phase_time["capability"] = float(time.time() - capability_start)

    base_report: dict[str, object] = {
        "experiment": VERSION,
        "stage": args.stage,
        "status": "capability_failed" if not capability_passed else "capability_passed",
        "configuration": asdict(config),
        "identity": {"v12_replay_max_abs_error": identity_error},
        "labels": {
            "screen": screen_label_report,
            "refit": refit_label_report,
        },
        "split": {
            "outer": config.outer,
            "validation_fold": config.validation_fold,
            "screen_fit_count": int(len(screen_fit)),
            "refit_count": int(len(refit_rows)),
            "strict_outer_count": int(len(strict_outer)),
            "strict_validation_count": int(len(strict_validation)),
        },
        "capability_gate": {
            "selected": capability_selected,
            "baseline": capability_baseline,
            "delta": capability_delta,
            "minimum_score_delta": config.capability_minimum_delta,
            "passed": capability_passed,
            "truth_usage": "held official paths used only for an undeployable representation ceiling",
        },
        "timing_seconds": phase_time,
    }
    base_report["seconds"] = float(time.time() - started)
    capability_report_path(config.outer).write_text(
        json.dumps(base_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.stage == "capability" or (
        config.outer == 0 and not capability_passed
    ):
        print(json.dumps(base_report, ensure_ascii=False, indent=2), flush=True)
        return

    train_start = time.time()
    store.coordinate = screen_coordinate
    mean_screen, std_screen = store.normalization(screen_fit)
    best_state, best_epoch, screen_history = train_screen_model(
        store,
        screen_labels,
        screen_fit,
        strict_validation,
        mean_screen,
        std_screen,
        evaluator,
        config,
        device,
    )
    phase_time["screen_training"] = float(time.time() - train_start)

    if best_state is None or best_epoch == 0:
        selected = capability_baseline
        baseline = capability_baseline
        screen_delta = metric_delta(selected, baseline)
        final_channel = np.asarray(
            evaluator.v12_np[evaluator.lookup[strict_outer]], dtype=np.complex64
        )
        endpoint_payload = None
        refit_history: list[dict[str, float]] = []
        status = "screen_failed_validation_fallback"
    else:
        torch.save(
            {
                "state_dict": best_state,
                "config": asdict(config),
                "feature_mean": mean_screen,
                "feature_std": std_screen,
                "best_epoch": best_epoch,
                "outer": config.outer,
                "validation_fold": config.validation_fold,
            },
            screen_checkpoint_path(config.outer),
        )
        refit_start = time.time()
        store.coordinate = refit_coordinate
        mean_refit, std_refit = store.normalization(refit_rows)
        model, refit_history = refit_model(
            store,
            refit_labels,
            refit_rows,
            mean_refit,
            std_refit,
            best_epoch,
            config,
            device,
        )
        phase_time["refit_training"] = float(time.time() - refit_start)
        mass, coordinate, port = endpoint_arrays(
            model,
            store,
            strict_outer,
            mean_refit,
            std_refit,
            device,
            config.batch_size,
        )
        endpoint_pas, endpoint_pdp = render_endpoint(mass, coordinate, port, device)
        selected, baseline, final_channel = evaluator.evaluate(
            strict_outer, endpoint_pas, endpoint_pdp, collect_channel=True
        )
        screen_delta = metric_delta(selected, baseline)
        endpoint_payload = {
            "query_ids": strict_outer.astype(np.int32),
            "mass": mass,
            "coordinate": coordinate,
            "port": port,
        }
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "feature_mean": mean_refit,
                "feature_std": std_refit,
                "best_epoch": best_epoch,
                "outer": config.outer,
                "mapping_latent": mapping["refit_latent"][config.outer].astype(np.float32),
            },
            refit_checkpoint_path(config.outer),
        )
        del model
        status = "screen_complete"

    component_protection = min(
        screen_delta["pas"], screen_delta["pdp"], screen_delta["nmse_score"]
    ) >= -config.maximum_component_drop
    screen_passed = bool(
        screen_delta["score"] >= config.screen_minimum_delta and component_protection
    )
    if endpoint_payload is not None:
        np.savez_compressed(screen_endpoint_path(config.outer), **endpoint_payload)
    np.save(screen_channel_path(config.outer), final_channel.astype(np.complex64))
    phase_time["total_before_report"] = float(time.time() - started)

    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        peak_memory = float(torch.cuda.max_memory_allocated(device) / 2**30)
    else:
        gpu = "cpu"
        peak_memory = 0.0
    report = {
        **base_report,
        "status": status,
        "screen": {
            "best_epoch": best_epoch,
            "selected": selected,
            "baseline": baseline,
            "delta": screen_delta,
            "minimum_score_delta": config.screen_minimum_delta,
            "maximum_component_drop": config.maximum_component_drop,
            "component_protection_passed": component_protection,
            "passed": screen_passed,
        },
        "training": {
            "screen_history": screen_history,
            "refit_history": refit_history,
        },
        "outputs": {
            "endpoint": PATHS.relative(screen_endpoint_path(config.outer))
            if endpoint_payload is not None
            else None,
            "channel": PATHS.relative(screen_channel_path(config.outer)),
            "screen_checkpoint": PATHS.relative(screen_checkpoint_path(config.outer))
            if best_state is not None
            else None,
            "refit_checkpoint": PATHS.relative(refit_checkpoint_path(config.outer))
            if endpoint_payload is not None
            else None,
        },
        "performance": {
            "device": gpu,
            "peak_torch_memory_gib": peak_memory,
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
            "timing_seconds": phase_time,
            "measured_bottleneck": max(phase_time, key=phase_time.get),
        },
        "seconds": float(time.time() - started),
    }
    screen_report_path(config.outer).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
