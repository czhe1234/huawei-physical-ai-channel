#!/usr/bin/env python3
"""Train the E198 cross-round complex-field conditional generator."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

import analyze_e86_surface_endpoint_v11_blend as e86
import analyze_global_local_path_lifecycle as e79
import prepare_e198_cross_round_complex_field as prepare
import train_e190_official_sionna_path_calibrator as e190
import train_e195_directional_probe_calibrator as e195
import train_e197_cross_round_official as e197
import train_e192_multiseed_relation_probe as e192
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics_torch,
)
from project_paths import PATHS
from run_path_component_transport import component_energy


VERSION = "Round2-E198-cross-round-complex-field"
ARTIFACT_DIR = prepare.ARTIFACT_DIR
FOLD_DIR = ARTIFACT_DIR / "folds"
ANALYSIS_DIR = prepare.ANALYSIS_DIR
LOG_DIR = ANALYSIS_DIR / "logs"
REPORT = ANALYSIS_DIR / "fixed500_report.json"
OUTPUT_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
V15_REPORT = e195.V15_REPORT

POINTS = 4000
CANDIDATES = e85.PATHS_PER_POINT
PORTS = e85.MP * e85.N_UE
PERIODS = np.asarray((e85.MH, e85.MV, e85.S_SC), dtype=np.float32)
FIELD_TOKENS = prepare.DONORS * prepare.COMPONENTS
FIELD_VALID_INDEX = 59
V15_SCORE = 0.6863750036074728
TARGET_SCORE = 0.700000
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    outer: int = 0
    validation_fold: int = 1
    epochs: int = 32
    validation_interval: int = 4
    state_batch_size: int = 64
    official_batch_size: int = 8
    metric_batch_size: int = 8
    field_layers: int = 1
    field_learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    phase_weight: float = 0.30
    state_weight_in_official: float = 0.10
    maximum_complex_mix: float = 0.40
    maximum_log_power_delta: float = 4.0
    random_state: int = 20260813198


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "fold", "aggregate"), default="fold")
    parser.add_argument("--outer", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force-fold", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def row_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def fold_paths(outer: int) -> dict[str, Path]:
    return {
        "screen_phase": FOLD_DIR / f"outer{outer}_screen_phase_labels.npz",
        "refit_phase": FOLD_DIR / f"outer{outer}_refit_phase_labels.npz",
        "screen": FOLD_DIR / f"outer{outer}_screen.pt",
        "refit": FOLD_DIR / f"outer{outer}_refit.pt",
        "channel": FOLD_DIR / f"outer{outer}_channel.npy",
        "endpoint": FOLD_DIR / f"outer{outer}_endpoint.npz",
        "report": ANALYSIS_DIR / f"outer{outer}_report.json",
        "failure": ANALYSIS_DIR / f"outer{outer}_failure.json",
    }


def required_inputs() -> tuple[Path, ...]:
    return (
        *prepare.expected_files(),
        e195.INDEPENDENT_BASE,
        e195.INDEPENDENT_BASE_REPORT,
        e195.TRAIN_CHANNEL,
        e195.FOLD_CACHE,
        e190.COMPONENTS,
        e85.PATH_CACHE / "mask.npy",
        V15_REPORT,
        *tuple(
            e195.fold_paths(outer)[name]
            for outer in range(5)
            for name in (
                "mapping",
                "screen_labels",
                "refit_labels",
                "screen_base",
                "refit_base",
            )
        ),
    )


def unit_complex(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    return np.divide(
        value,
        np.maximum(magnitude, 1e-12),
        out=np.ones_like(value, dtype=np.complex64),
    ).astype(np.complex64)


def load_candidate_phase() -> np.ndarray:
    raw = np.asarray(np.load(prepare.CANDIDATE_PHASE, mmap_mode="r"), dtype=np.float32)
    phase = raw[..., :PORTS] + 1j * raw[..., PORTS:]
    return unit_complex(phase.astype(np.complex64))


def build_phase_labels(
    path: Path,
    coordinate: np.ndarray,
    coordinate_digest: str,
    label_path: Path,
    config: e190.Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if path.is_file():
        cached = np.load(path, allow_pickle=False)
        if str(cached["coordinate_digest"].item()) == coordinate_digest:
            return (
                cached["phase"].astype(np.complex64),
                cached["active"].astype(bool),
                {
                    "status": "reused",
                    "path": PATHS.relative(path),
                    "coordinate_digest": coordinate_digest,
                },
            )

    started = time.time()
    candidate_mask = np.asarray(
        np.load(e85.PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool
    )
    component = np.load(e190.COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    existing = np.load(label_path, allow_pickle=False)
    expected_active = existing["mass"].astype(np.float32) > 0.0
    valid_point = existing["valid"].astype(bool)

    phase = load_candidate_phase()
    active = np.zeros((POINTS, CANDIDATES), dtype=bool)
    accepted_count = 0
    for row in range(POINTS):
        candidate_ids = np.flatnonzero(candidate_mask[row])
        if len(candidate_ids) == 0 or point_power[row] <= TINY:
            continue
        truth_energy = component_energy(truth_coefficient[row])
        truth_fraction = truth_energy / max(float(point_power[row]), TINY)
        truth_ids = np.flatnonzero(truth_fraction >= config.minimum_path_fraction)
        if len(truth_ids) == 0:
            continue
        cost = e79.support_cost(
            coordinate[row, candidate_ids], truth_coordinate[row, truth_ids]
        )
        utility = truth_fraction[truth_ids][None] * np.exp(
            -0.5 * cost / config.match_bandwidth
        )
        utility[cost > config.match_cost_limit] = 0.0
        candidate_local, truth_local = linear_sum_assignment(-utility)
        accepted = utility[candidate_local, truth_local] > 0.0
        assigned_candidate = candidate_ids[candidate_local[accepted]]
        assigned_truth = truth_ids[truth_local[accepted]]
        if len(assigned_candidate) == 0:
            continue
        phase[row, assigned_candidate] = unit_complex(
            truth_coefficient[row, assigned_truth]
        )
        active[row, assigned_candidate] = True
        accepted_count += len(assigned_candidate)

    mismatch = int(np.sum((active != expected_active) & valid_point[:, None]))
    if mismatch:
        raise RuntimeError(f"E198 phase-label identity mismatch: {mismatch}")
    atomic_save_npz(
        path,
        phase=phase,
        active=active,
        coordinate_digest=np.asarray(coordinate_digest),
    )
    return phase, active, {
        "status": "created",
        "path": PATHS.relative(path),
        "coordinate_digest": coordinate_digest,
        "active_paths": accepted_count,
        "seconds": float(time.time() - started),
    }


class CrossRoundStore(e195.ResidentStore):
    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        self.round1_field = torch.as_tensor(
            np.asarray(np.load(prepare.DONOR_FEATURE, mmap_mode="r"), dtype=np.float16),
            device=device,
        )
        self.coverage = torch.as_tensor(
            np.asarray(np.load(prepare.COVERAGE, mmap_mode="r"), dtype=np.float32),
            device=device,
        )
        raw_phase = np.asarray(
            np.load(prepare.CANDIDATE_PHASE, mmap_mode="r"), dtype=np.float32
        )
        phase = unit_complex(
            (raw_phase[..., :PORTS] + 1j * raw_phase[..., PORTS:]).astype(
                np.complex64
            )
        )
        self.base_phase = torch.as_tensor(phase, device=device)
        self.target_phase = torch.ones(
            (POINTS, CANDIDATES, PORTS), dtype=torch.complex64, device=device
        )
        self.target_phase_active = torch.zeros(
            (POINTS, CANDIDATES), dtype=torch.bool, device=device
        )

    def set_phase_labels(self, phase: np.ndarray, active: np.ndarray) -> None:
        self.target_phase = torch.as_tensor(phase, device=self.device)
        self.target_phase_active = torch.as_tensor(active, device=self.device)

    def batch(
        self,
        rows: np.ndarray,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        batch, target_pas, target_pdp = super().batch(
            rows, mean, std, include_probe=False
        )
        ids = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        batch.update(
            {
                "round1_field": self.round1_field[ids].float(),
                "coverage": self.coverage[ids],
                "base_phase": self.base_phase[ids],
                "target_phase": self.target_phase[ids],
                "target_phase_active": self.target_phase_active[ids],
            }
        )
        return batch, target_pas, target_pdp


class CrossRoundComplexModel(e85.SurfaceInteractionModel):
    def __init__(
        self,
        checkpoint: dict[str, Any],
        store: CrossRoundStore,
        config: Config,
    ) -> None:
        model_config = e195.model_config(checkpoint)
        super().__init__(
            store.base.shape[-1],
            store.sequence_geometry.shape[-1],
            model_config,
            "geometry",
        )
        self.e198_config = config
        missing, unexpected = self.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"E198 base checkpoint mismatch missing={missing} unexpected={unexpected}"
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        hidden = model_config.hidden
        self.field_input = nn.Sequential(
            nn.Linear(prepare.FIELD_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        field_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=model_config.heads,
            dim_feedforward=model_config.feedforward,
            dropout=model_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.field_encoder = nn.TransformerEncoder(
            field_layer, num_layers=config.field_layers
        )
        self.field_query_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(
            hidden, model_config.heads, dropout=model_config.dropout, batch_first=True
        )
        self.field_output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.phase_head = nn.Linear(hidden, PORTS)
        self.point_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )
        nn.init.zeros_(self.field_output[-1].weight)
        nn.init.zeros_(self.field_output[-1].bias)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = self.base_input(batch["base"])
        sequence = self.sequence_state(
            batch["sequence_geometry"],
            batch["interactions"],
            batch["objects"],
            batch["primitives"],
            batch["depth"],
            batch["sequence_hash"],
        )
        core = self.fusion(torch.cat((base, sequence), dim=-1))

        field_raw = batch["round1_field"].reshape(
            len(core), FIELD_TOKENS, prepare.FIELD_FEATURES
        )
        field_mask = field_raw[..., FIELD_VALID_INDEX] > 0.5
        safe_field_mask = field_mask.clone()
        empty = ~safe_field_mask.any(dim=1)
        safe_field_mask[empty, 0] = True
        field = self.field_input(field_raw)
        field = self.field_encoder(field, src_key_padding_mask=~safe_field_mask)
        attended, _ = self.cross_attention(
            self.field_query_norm(core),
            field,
            field,
            key_padding_mask=~safe_field_mask,
            need_weights=False,
        )
        coverage = batch["coverage"].clamp(0.0, 1.0)
        core = core + coverage[:, None, None] * self.field_output(attended)
        hidden = self.set_encoder(core)

        score_residual = self.config.score_limit * torch.tanh(
            self.score_head(hidden)[..., 0]
        )
        coordinate_residual = torch.tanh(self.coordinate_head(hidden)) * torch.tensor(
            (2.0, 4.0, 2.0), dtype=hidden.dtype, device=hidden.device
        )
        port_residual = self.config.port_limit * torch.tanh(self.port_head(hidden))
        mass = torch.softmax(batch["base_log_mass"] + score_residual, dim=1)
        port = torch.softmax(batch["base_log_port"] + port_residual, dim=2).reshape(
            len(hidden), CANDIDATES, e85.MP, e85.N_UE
        )
        coordinate = batch["coordinate"] + coordinate_residual
        phase_delta = (
            torch.pi
            * coverage[:, None, None]
            * torch.tanh(self.phase_head(hidden))
        )
        phase = batch["base_phase"] * torch.exp(1j * phase_delta)
        candidate_weight = batch["candidate_mask"].to(hidden.dtype)
        pooled = (hidden * candidate_weight[..., None]).sum(dim=1) / candidate_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        point = self.point_head(pooled)
        complex_mix = (
            self.e198_config.maximum_complex_mix
            * coverage
            * torch.tanh(point[:, 0])
        )
        log_power_delta = (
            self.e198_config.maximum_log_power_delta
            * coverage
            * torch.tanh(point[:, 1])
        )
        return {
            "mass": mass,
            "port": port,
            "coordinate": coordinate,
            "phase": phase,
            "complex_mix": complex_mix,
            "log_power_delta": log_power_delta,
            "score_residual": score_residual,
            "coordinate_residual": coordinate_residual,
            "port_residual": port_residual,
        }


def make_model(
    checkpoint: dict[str, Any],
    store: CrossRoundStore,
    config: Config,
    device: torch.device,
    seed: int,
) -> CrossRoundComplexModel:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return CrossRoundComplexModel(checkpoint, store, config).to(device)


def endpoint_spectra(
    output: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_small, pdp_small = e85.render_spectra(output)
    pas = pas_small[..., None].expand(-1, -1, -1, -1, -1, e85.S_SC)
    pdp = (
        pdp_small[:, :, None, None, :, :]
        .expand(-1, -1, e85.MH, e85.MV, -1, -1)
        .reshape(-1, 256, e85.N_UE, e85.S_SC)
    )
    return pas, pdp


def synthesize_paths(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    desired_power: torch.Tensor,
) -> torch.Tensor:
    coordinate = torch.remainder(
        output["coordinate"],
        torch.tensor(PERIODS, dtype=torch.float32, device=desired_power.device),
    )
    port = output["port"].reshape(len(coordinate), CANDIDATES, PORTS)
    coefficient_power = output["mass"][..., None] * port
    amplitude = torch.sqrt(coefficient_power.clamp_min(1e-30))
    amplitude = amplitude * batch["candidate_mask"][..., None]
    coefficient = amplitude.to(torch.complex64) * output["phase"]
    atom = e192.component_atom_torch(coordinate)
    channel = torch.einsum("bkc,bkhvs->bchvs", coefficient, atom)
    channel = (
        channel.reshape(-1, 2, 4, 16, 8, 192)
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(-1, 256, 4, 192)
    )
    current_power = channel.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    scale = torch.sqrt(desired_power / current_power.clamp_min(TINY)).to(torch.float32)
    return channel * scale[:, None, None, None]


class CompleteEvaluator:
    def __init__(self, device: torch.device) -> None:
        base = e195.IndependentEvaluator(device)
        self.query_ids = base.query_ids
        self.fold_id = base.fold_id
        self.lookup = base.lookup
        self.base = base.base
        self.truth = base.truth
        self.truth_pas = base.truth_pas.reshape_as(base.truth)
        self.truth_pdp = base.truth_pdp
        self.device = device

    def canonical(
        self, global_rows: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.lookup[global_rows]
        if np.any(local < 0):
            raise ValueError("E198 official batch contains non-canonical rows")
        ids = torch.as_tensor(local, dtype=torch.long, device=self.device)
        return (
            self.base[ids],
            self.truth[ids],
            self.truth_pas[ids],
            self.truth_pdp[ids],
        )


def render_complete(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    base: torch.Tensor,
) -> torch.Tensor:
    endpoint_pas, endpoint_pdp = endpoint_spectra(output)
    spec = e86.Candidate(
        endpoint="e198_cross_round_complex_field",
        mode="sqrt",
        pas_beta=0.20,
        pdp_beta=0.20,
        order="pas_pdp",
        cycles=4,
    )
    projected = e86.make_candidate(
        base,
        pas_distribution(base),
        pdp_distribution(base),
        endpoint_pas,
        endpoint_pdp,
        spec,
    )
    projected_power = projected.abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    coherent = synthesize_paths(output, batch, projected_power)
    mix = output["complex_mix"].to(projected.dtype)
    prediction = projected + mix[:, None, None, None] * (coherent - projected)
    amplitude_scale = torch.exp(0.5 * output["log_power_delta"])
    return prediction * amplitude_scale[:, None, None, None]


def phase_objective(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_mass: torch.Tensor,
    target_port: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = batch["target_phase_active"] & (target_mass > 0.0)
    if not active.any():
        zero = output["mass"].sum() * 0.0
        return zero, zero
    weight = target_mass[active]
    weight = weight / weight.mean().clamp_min(TINY)
    phase_score = (
        torch.real(output["phase"][active] * torch.conj(batch["target_phase"][active]))
        * target_port[active]
    ).sum(dim=1)
    score = (phase_score * weight).sum() / weight.sum().clamp_min(TINY)
    return 1.0 - score, score


def state_objective(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    labels: e195.ResidentLabels,
    rows: np.ndarray,
    config: Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ids = torch.as_tensor(rows, dtype=torch.long, device=target_pas.device)
    base_loss, values = e190.label_objective(
        output,
        batch,
        target_pas,
        target_pdp,
        labels.mass[ids],
        labels.delta[ids],
        labels.port[ids],
        e190.Config(outer=config.outer, validation_fold=config.validation_fold),
    )
    phase_loss, phase_score = phase_objective(
        output, batch, labels.mass[ids], labels.port[ids]
    )
    regularization = (
        output["complex_mix"].square().mean()
        + 0.05 * output["log_power_delta"].square().mean()
    )
    loss = base_loss + config.phase_weight * phase_loss + 0.001 * regularization
    return loss, {**values, "phase": phase_score, "regularization": regularization}


def stable_step(
    loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm_squared = torch.zeros((), dtype=torch.float64, device=loss.device)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("non-finite E198 gradient")
        norm_squared += parameter.grad.double().square().sum()
    norm = torch.sqrt(norm_squared)
    scale = min(1.0, 2.0 / max(float(norm.detach().cpu()), 1e-30))
    if scale < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    optimizer.step()
    return float(norm.detach().cpu())


def optimizer_for(
    model: CrossRoundComplexModel, config: Config, epochs: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.field_learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=config.field_learning_rate * 0.08,
    )
    return optimizer, scheduler


def run_state_epoch(
    model: CrossRoundComplexModel,
    store: CrossRoundStore,
    labels: e195.ResidentLabels,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: Config,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    eligible = rows[labels.valid[rows] & (store.power[rows] > 0.0)]
    order = rng.permutation(eligible)
    totals: dict[str, list[float]] = {
        name: []
        for name in (
            "loss",
            "pas",
            "pdp",
            "mass_affinity",
            "coordinate",
            "port",
            "existence",
            "phase",
            "gradient",
        )
    }
    for start in range(0, len(order), config.state_batch_size):
        local = order[start : start + config.state_batch_size]
        batch, target_pas, target_pdp = store.batch(local, mean, std)
        output = model(batch)
        loss, values = state_objective(
            output, batch, target_pas, target_pdp, labels, local, config
        )
        gradient = stable_step(loss, model, optimizer)
        totals["loss"].append(float(loss.detach().cpu()))
        totals["gradient"].append(gradient)
        for name in totals:
            if name in ("loss", "gradient"):
                continue
            totals[name].append(float(values[name].detach().cpu()))
    return {name: float(np.mean(value)) for name, value in totals.items()}


def run_official_epoch(
    model: CrossRoundComplexModel,
    store: CrossRoundStore,
    labels: e195.ResidentLabels,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: CompleteEvaluator,
    optimizer: torch.optim.Optimizer,
    config: Config,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    order = rng.permutation(rows)
    scores: list[float] = []
    gradients: list[float] = []
    mixes: list[float] = []
    for start in range(0, len(order), config.official_batch_size):
        local = order[start : start + config.official_batch_size]
        batch, target_pas, target_pdp = store.batch(local, mean, std)
        output = model(batch)
        base, truth, truth_pas, truth_pdp = evaluator.canonical(local)
        prediction = render_complete(output, batch, base)
        official_loss, official = e192.official_objective(
            prediction, truth, truth_pas, truth_pdp
        )
        state_loss, _ = state_objective(
            output, batch, target_pas, target_pdp, labels, local, config
        )
        loss = official_loss + config.state_weight_in_official * state_loss
        gradients.append(stable_step(loss, model, optimizer))
        scores.append(float(official["score"].detach().cpu()))
        mixes.append(float(output["complex_mix"].abs().mean().detach().cpu()))
    return {
        "score": float(np.mean(scores)),
        "gradient": float(np.mean(gradients)),
        "absolute_mix": float(np.mean(mixes)),
    }


@torch.inference_mode()
def evaluate_rows(
    model: CrossRoundComplexModel,
    store: CrossRoundStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: CompleteEvaluator,
    config: Config,
    collect_channel: bool = False,
) -> tuple[dict[str, float | int], dict[str, np.ndarray], np.ndarray | None, dict[str, float]]:
    model.eval()
    point = PointMetricStatistics(
        pas=np.empty(len(rows), dtype=np.float64),
        pdp=np.empty(len(rows), dtype=np.float64),
        squared_error=np.empty(len(rows), dtype=np.float64),
        truth_energy=np.empty(len(rows), dtype=np.float64),
    )
    endpoint = {
        "mass": np.empty((len(rows), CANDIDATES), dtype=np.float32),
        "coordinate": np.empty((len(rows), CANDIDATES, 3), dtype=np.float32),
        "port": np.empty((len(rows), CANDIDATES, e85.MP, e85.N_UE), dtype=np.float32),
        "phase": np.empty((len(rows), CANDIDATES, PORTS), dtype=np.complex64),
    }
    channel = (
        np.empty((len(rows), 256, 4, 192), dtype=np.complex64)
        if collect_channel
        else None
    )
    mixes: list[np.ndarray] = []
    power_delta: list[np.ndarray] = []
    for start in range(0, len(rows), config.metric_batch_size):
        end = min(start + config.metric_batch_size, len(rows))
        local = rows[start:end]
        batch, _, _ = store.batch(local, mean, std)
        output = model(batch)
        base, truth, _, _ = evaluator.canonical(local)
        prediction = render_complete(output, batch, base)
        values = point_metric_statistics_torch(prediction, truth)
        point.pas[start:end] = values.pas
        point.pdp[start:end] = values.pdp
        point.squared_error[start:end] = values.squared_error
        point.truth_energy[start:end] = values.truth_energy
        endpoint["mass"][start:end] = output["mass"].cpu().numpy().astype(np.float32)
        endpoint["coordinate"][start:end] = (
            output["coordinate"].remainder(
                torch.tensor(PERIODS, device=output["coordinate"].device)
            ).cpu().numpy().astype(np.float32)
        )
        endpoint["port"][start:end] = output["port"].cpu().numpy().astype(np.float32)
        endpoint["phase"][start:end] = output["phase"].cpu().numpy().astype(np.complex64)
        mixes.append(output["complex_mix"].cpu().numpy())
        power_delta.append(output["log_power_delta"].cpu().numpy())
        if channel is not None:
            channel[start:end] = prediction.cpu().numpy().astype(np.complex64)
    diagnostics = {
        "absolute_complex_mix_mean": float(np.mean(np.abs(np.concatenate(mixes)))),
        "log_power_delta_mean": float(np.mean(np.concatenate(power_delta))),
    }
    return aggregate_statistics(point), endpoint, channel, diagnostics


@torch.inference_mode()
def identity_check(
    model: CrossRoundComplexModel,
    checkpoint: dict[str, Any],
    store: CrossRoundStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    baseline = e85.SurfaceInteractionModel(
        store.base.shape[-1],
        store.sequence_geometry.shape[-1],
        e195.model_config(checkpoint),
        "geometry",
    ).to(device)
    baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    baseline.eval()
    model.eval()
    batch, _, _ = store.batch(rows, mean, std)
    left = baseline(batch)
    right = model(batch)
    endpoint_error = max(
        float(torch.max(torch.abs(left[name] - right[name])).item())
        for name in ("mass", "port", "coordinate")
    )
    return {
        "endpoint_max_abs_error": endpoint_error,
        "complex_mix_max_abs": float(right["complex_mix"].abs().max().item()),
        "log_power_delta_max_abs": float(right["log_power_delta"].abs().max().item()),
        "phase_residual_max_abs": float(
            torch.max(torch.abs(right["phase"] - batch["base_phase"])).item()
        ),
    }


def train_screen(
    model: CrossRoundComplexModel,
    store: CrossRoundStore,
    labels: e195.ResidentLabels,
    fit_rows: np.ndarray,
    official_fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: CompleteEvaluator,
    config: Config,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, Any]], dict[str, Any]]:
    baseline, _, _, baseline_diagnostics = evaluate_rows(
        model, store, validation_rows, mean, std, evaluator, config
    )
    best_score = float(baseline["score"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = [
        {"epoch": 0, "validation": baseline, "diagnostics": baseline_diagnostics}
    ]
    optimizer, scheduler = optimizer_for(model, config, config.epochs)
    rng = np.random.default_rng(config.random_state + config.outer)
    for epoch in range(1, config.epochs + 1):
        state = run_state_epoch(
            model, store, labels, fit_rows, mean, std, optimizer, config, rng
        )
        official = run_official_epoch(
            model,
            store,
            labels,
            official_fit_rows,
            mean,
            std,
            evaluator,
            optimizer,
            config,
            rng,
        )
        scheduler.step()
        record: dict[str, Any] = {"epoch": epoch, "state": state, "official": official}
        if epoch % config.validation_interval == 0:
            validation, _, _, diagnostics = evaluate_rows(
                model, store, validation_rows, mean, std, evaluator, config
            )
            record["validation"] = validation
            record["diagnostics"] = diagnostics
            record["delta_vs_epoch0"] = e190.metric_delta(validation, baseline)
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E198 outer={config.outer} epoch={epoch}/{config.epochs} "
                f"validation={float(validation['score']):.6f} best={best_score:.6f}",
                flush=True,
            )
        history.append(record)
    return best_state, best_epoch, history, {
        "baseline": baseline,
        "best_validation_score": best_score,
    }


def refit(
    model: CrossRoundComplexModel,
    store: CrossRoundStore,
    labels: e195.ResidentLabels,
    rows: np.ndarray,
    official_rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    evaluator: CompleteEvaluator,
    epochs: int,
    config: Config,
) -> list[dict[str, Any]]:
    if epochs == 0:
        return []
    optimizer, scheduler = optimizer_for(model, config, epochs)
    rng = np.random.default_rng(config.random_state + 100_000 + config.outer)
    history = []
    for epoch in range(1, epochs + 1):
        state = run_state_epoch(
            model, store, labels, rows, mean, std, optimizer, config, rng
        )
        official = run_official_epoch(
            model,
            store,
            labels,
            official_rows,
            mean,
            std,
            evaluator,
            optimizer,
            config,
            rng,
        )
        scheduler.step()
        history.append({"epoch": epoch, "state": state, "official": official})
        if epoch % config.validation_interval == 0 or epoch == epochs:
            print(
                f"E198 refit outer={config.outer} epoch={epoch}/{epochs} "
                f"official_train={official['score']:.6f}",
                flush=True,
            )
    return history


def load_mode(
    store: CrossRoundStore,
    outer: int,
    mode: str,
    phase_path: Path,
    device: torch.device,
) -> tuple[e195.ResidentLabels, dict[str, Any], dict[str, Any]]:
    labels, coordinate, lineage = e197.round2_mode(store, outer, mode, device)
    mapping = np.load(e195.fold_paths(outer)["mapping"], allow_pickle=False)
    latent = mapping[f"{mode}_latent"].astype(np.float32)
    digest = row_digest(latent, coordinate)
    base_config = e190.Config(outer=outer, validation_fold=(outer + 1) % 5)
    phase, active, phase_report = build_phase_labels(
        phase_path,
        coordinate,
        digest,
        e195.fold_paths(outer)[f"{mode}_labels"],
        base_config,
    )
    store.set_phase_labels(phase, active)
    return labels, lineage, phase_report


def load_checkpoint(outer: int, mode: str, device: torch.device) -> dict[str, Any]:
    path = e195.fold_paths(outer)[f"{mode}_base"]
    return torch.load(path, map_location=device, weights_only=False)


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    config = Config(outer=args.outer, validation_fold=(args.outer + 1) % 5)
    store = CrossRoundStore(device)
    evaluator = CompleteEvaluator(device)
    split = e195.canonical_split(config.outer, config.validation_fold, evaluator)
    labels, lineage, phase = load_mode(
        store,
        config.outer,
        "screen",
        fold_paths(config.outer)["screen_phase"],
        device,
    )
    checkpoint = load_checkpoint(config.outer, "screen", device)
    mean = torch.as_tensor(checkpoint["feature_mean"], device=device)
    std = torch.as_tensor(checkpoint["feature_std"], device=device)
    model = make_model(checkpoint, store, config, device, config.random_state)
    identity = identity_check(
        model, checkpoint, store, split["screen_fit"][:4], mean, std, device
    )
    if max(identity.values()) > 1e-6:
        raise RuntimeError(f"E198 epoch0 identity failed: {identity}")
    optimizer, _ = optimizer_for(model, config, 1)
    state_rows = split["screen_fit"][
        labels.valid[split["screen_fit"]] & (store.power[split["screen_fit"]] > 0.0)
    ][:8]
    state = run_state_epoch(
        model,
        store,
        labels,
        state_rows,
        mean,
        std,
        optimizer,
        Config(**{**asdict(config), "state_batch_size": 8}),
        np.random.default_rng(config.random_state),
    )
    official_rows = evaluator.query_ids[
        (evaluator.fold_id != config.outer)
        & (evaluator.fold_id != config.validation_fold)
    ][:4]
    official = run_official_epoch(
        model,
        store,
        labels,
        official_rows,
        mean,
        std,
        evaluator,
        optimizer,
        Config(**{**asdict(config), "official_batch_size": 4}),
        np.random.default_rng(config.random_state + 1),
    )
    return {
        "experiment": VERSION,
        "status": "profile_complete",
        "identity": identity,
        "state_gradient": state,
        "official_gradient": official,
        "lineage": lineage,
        "phase_labels": phase,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "peak_torch_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else 0.0,
        "seconds": float(time.time() - started),
    }


def run_fold(args: argparse.Namespace) -> dict[str, Any]:
    paths = fold_paths(args.outer)
    if paths["report"].is_file() and not args.force_fold:
        return json.loads(paths["report"].read_text(encoding="utf-8"))
    started = time.time()
    phase_time: dict[str, float] = {}
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    config = Config(outer=args.outer, validation_fold=(args.outer + 1) % 5)
    lineage = e195.validate_lineage()

    load_started = time.time()
    store = CrossRoundStore(device)
    evaluator = CompleteEvaluator(device)
    split = e195.canonical_split(config.outer, config.validation_fold, evaluator)
    official_screen = evaluator.query_ids[
        (evaluator.fold_id != config.outer)
        & (evaluator.fold_id != config.validation_fold)
    ]
    official_refit = evaluator.query_ids[evaluator.fold_id != config.outer]
    phase_time["resident_load"] = float(time.time() - load_started)

    screen_started = time.time()
    screen_labels, screen_lineage, screen_phase_report = load_mode(
        store, config.outer, "screen", paths["screen_phase"], device
    )
    screen_checkpoint = load_checkpoint(config.outer, "screen", device)
    mean_screen = torch.as_tensor(screen_checkpoint["feature_mean"], device=device)
    std_screen = torch.as_tensor(screen_checkpoint["feature_std"], device=device)
    model = make_model(
        screen_checkpoint,
        store,
        config,
        device,
        config.random_state + config.outer,
    )
    identity = identity_check(
        model,
        screen_checkpoint,
        store,
        split["screen_fit"][:4],
        mean_screen,
        std_screen,
        device,
    )
    if max(identity.values()) > 1e-6:
        raise RuntimeError(f"E198 screen identity failed: {identity}")
    best_state, best_epoch, screen_history, screen_selection = train_screen(
        model,
        store,
        screen_labels,
        split["screen_fit"],
        official_screen,
        split["validation"],
        mean_screen,
        std_screen,
        evaluator,
        config,
    )
    atomic_torch_save(
        paths["screen"],
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "configuration": asdict(config),
            "history": screen_history,
            "split_digest": row_digest(
                split["screen_fit"], split["validation"], split["outer"]
            ),
        },
    )
    phase_time["screen_training"] = float(time.time() - screen_started)
    del model, screen_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()

    refit_started = time.time()
    refit_labels, refit_lineage, refit_phase_report = load_mode(
        store, config.outer, "refit", paths["refit_phase"], device
    )
    refit_checkpoint = load_checkpoint(config.outer, "refit", device)
    mean_refit = torch.as_tensor(refit_checkpoint["feature_mean"], device=device)
    std_refit = torch.as_tensor(refit_checkpoint["feature_std"], device=device)
    model = make_model(
        refit_checkpoint,
        store,
        config,
        device,
        config.random_state + 100_000 + config.outer,
    )
    refit_identity = identity_check(
        model,
        refit_checkpoint,
        store,
        split["refit"][:4],
        mean_refit,
        std_refit,
        device,
    )
    if max(refit_identity.values()) > 1e-6:
        raise RuntimeError(f"E198 refit identity failed: {refit_identity}")
    refit_history = refit(
        model,
        store,
        refit_labels,
        split["refit"],
        official_refit,
        mean_refit,
        std_refit,
        evaluator,
        best_epoch,
        config,
    )
    selected, endpoint, channel, diagnostics = evaluate_rows(
        model,
        store,
        split["outer"],
        mean_refit,
        std_refit,
        evaluator,
        config,
        collect_channel=True,
    )
    if channel is None:
        raise RuntimeError("E198 outer channel missing")
    phase_time["refit_and_outer"] = float(time.time() - refit_started)

    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))
    v15_fold = next(row for row in v15["folds"] if int(row["outer"]) == config.outer)
    delta = e190.metric_delta(selected, v15_fold["selected"])
    atomic_save_npy(paths["channel"], channel)
    atomic_save_npz(
        paths["endpoint"],
        query_ids=split["outer"].astype(np.int32),
        mass=endpoint["mass"],
        coordinate=endpoint["coordinate"],
        port=endpoint["port"],
        phase=endpoint["phase"],
    )
    atomic_torch_save(
        paths["refit"],
        {
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "configuration": asdict(config),
            "feature_mean": mean_refit.cpu().numpy(),
            "feature_std": std_refit.cpu().numpy(),
            "history": refit_history,
            "split_digest": row_digest(split["refit"], split["outer"]),
        },
    )
    phase_time["save"] = float(time.time() - started - sum(phase_time.values()))
    report = {
        "experiment": VERSION,
        "status": "fold_complete",
        "configuration": asdict(config),
        "outer": config.outer,
        "validation_fold": config.validation_fold,
        "split": {
            "screen_fit": int(len(split["screen_fit"])),
            "official_screen": int(len(official_screen)),
            "refit": int(len(split["refit"])),
            "official_refit": int(len(official_refit)),
            "validation": int(len(split["validation"])),
            "outer": int(len(split["outer"])),
            "zero_label_overlap": split["leakage"],
        },
        "identity": {"screen": identity, "refit": refit_identity},
        "selection": {
            "best_epoch": best_epoch,
            "screen": screen_selection,
            "screen_history": screen_history,
            "refit_history": refit_history,
        },
        "selected": selected,
        "v15": v15_fold["selected"],
        "delta_vs_v15": delta,
        "diagnostics": diagnostics,
        "data_lineage": {
            "round1_official_complex_field_used": True,
            "round1_absolute_coordinates": True,
            "round1_scene_translation_applied": False,
            "round2_same_position_external_candidates_used": True,
            "round2_outer_h_used_for_training": False,
            "historical_v15_complete_h_read": False,
            "screen": screen_lineage,
            "refit": refit_lineage,
            "screen_phase": screen_phase_report,
            "refit_phase": refit_phase_report,
            "independent_base_lineage": lineage.get("data_lineage", {}),
        },
        "output": {
            "channel": PATHS.relative(paths["channel"]),
            "endpoint": PATHS.relative(paths["endpoint"]),
            "checkpoint": PATHS.relative(paths["refit"]),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": bool(np.isfinite(channel).all()),
        },
        "performance": {
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated(device) / 2**30
            )
            if device.type == "cuda"
            else 0.0,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "timing_seconds": phase_time,
            "measured_bottleneck": max(phase_time, key=phase_time.get),
        },
        "seconds": float(time.time() - started),
    }
    atomic_json(paths["report"], report)
    paths["failure"].unlink(missing_ok=True)
    return report


def aggregate(device: torch.device) -> dict[str, Any]:
    started = time.time()
    evaluator = CompleteEvaluator(device)
    output = np.empty((500, 256, 4, 192), dtype=np.complex64)
    fold_reports = []
    deltas = []
    for outer in range(5):
        paths = fold_paths(outer)
        if not paths["report"].is_file() or not paths["channel"].is_file():
            raise FileNotFoundError(f"missing E198 outer {outer}")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        rows = evaluator.fold_id == outer
        local = np.load(paths["channel"], mmap_mode="r")
        expected = (int(rows.sum()), 256, 4, 192)
        if local.shape != expected or local.dtype != np.complex64:
            raise ValueError(f"invalid E198 outer{outer}: {local.shape}/{local.dtype}")
        output[rows] = np.asarray(local, dtype=np.complex64)
        fold_reports.append(
            {
                "outer": outer,
                "count": int(rows.sum()),
                "selected": report["selected"],
                "v15": report["v15"],
                "delta_vs_v15": report["delta_vs_v15"],
                "best_epoch": report["selection"]["best_epoch"],
                "performance": report["performance"],
            }
        )
        deltas.append(float(report["delta_vs_v15"]["score"]))
    prediction = torch.as_tensor(output, device=device)
    statistics = point_metric_statistics_torch(prediction, evaluator.truth)
    selected = aggregate_statistics(statistics)
    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))["selected"]
    delta = e190.metric_delta(selected, v15)
    positive_folds = int(sum(value > 0.0 for value in deltas))
    worst_fold = float(min(deltas))
    passed = bool(
        float(selected["score"]) >= TARGET_SCORE
        and positive_folds >= 4
        and worst_fold >= -0.001
    )
    atomic_save_npy(OUTPUT_CHANNEL, output)
    report = {
        "experiment": VERSION,
        "status": "fixed500_passed" if passed else "fixed500_failed",
        "selected": selected,
        "v15": v15,
        "delta_vs_v15": delta,
        "delta_vs_target": float(selected["score"] - TARGET_SCORE),
        "folds": fold_reports,
        "gate": {
            "target_score": TARGET_SCORE,
            "minimum_positive_folds": 4,
            "minimum_worst_fold_delta": -0.001,
            "positive_folds": positive_folds,
            "worst_fold_delta": worst_fold,
            "passed": passed,
        },
        "output": {
            "channel": PATHS.relative(OUTPUT_CHANNEL),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output).all()),
        },
        "seconds": float(time.time() - started),
    }
    atomic_json(REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E198 inputs: {missing}")
    try:
        if args.stage == "profile":
            report = run_profile(args)
        elif args.stage == "fold":
            report = run_fold(args)
        else:
            report = aggregate(torch.device(args.device))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        if args.stage == "fold":
            atomic_json(
                fold_paths(args.outer)["failure"],
                {
                    "experiment": VERSION,
                    "status": "failed",
                    "outer": args.outer,
                    "traceback": traceback.format_exc(),
                },
            )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
