#!/usr/bin/env python3
"""Train the strict full-4000 E161 V7 correction coefficient student."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from analyze_e160_v7_correction_capacity import (
    FULL_CORE_PAS,
    FULL_CORE_PDP,
    FULL_RESIDUAL_PAS,
    FULL_RESIDUAL_PDP,
    FULL_TRUTH_PAS,
    FULL_TRUTH_PDP,
    apply_lifecycle,
    fit_basis,
    render_full_targets,
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
from official_metrics import aggregate_statistics, PointMetricStatistics
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META
from run_v6_regime_component_lifecycle import REGIME_CACHE
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pdp_link_gate import FOLD_IDS


VERSION = "Round2-E161-full4000-v7-correction-student"
E160_DIR = PATHS.artifacts / "round2" / "e160_full4000_v7_supervision"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e161_v7_correction_student"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e161_v7_correction_student"
REPORT = ANALYSIS_DIR / "e161_v7_correction_student.json"
METRICS = ARTIFACT_DIR / "e161_v7_correction_student_metrics.npz"
FULL_CORE = E160_DIR / "full4000_v7_core.npy"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
GEOMETRY = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_geometry_visibility_features.npz"
)

PAS_RANK = 128
PDP_RANK = 128
PAS_SHAPE = (2, 16, 8, 4, 8)
PDP_SHAPE = (2, 4, 2, 4, 192)
TOTAL_RANK = PAS_RANK + PDP_RANK
TINY = 1e-30
V7_SCORE = 0.644456345060792
V12_SCORE = 0.677521


@dataclass(frozen=True)
class Config:
    neighbors: int = 32
    hidden: int = 256
    heads: int = 8
    layers: int = 2
    dropout: float = 0.05
    batch_size: int = 96
    maximum_steps: int = 5000
    validation_interval: int = 100
    patience_steps: int = 800
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    coefficient_weight: float = 0.20
    action_regularization: float = 0.001
    project_cycles: int = 4
    metric_batch_size: int = 8
    seed: int = 260_810_161


@dataclass
class BasisState:
    pas_basis: np.ndarray
    pdp_basis: np.ndarray
    pas_scale: np.ndarray
    pdp_scale: np.ndarray
    pas_input_mean: np.ndarray
    pas_input_scale: np.ndarray
    pdp_input_mean: np.ndarray
    pdp_input_scale: np.ndarray
    pas_explained: float
    pdp_explained: float


@dataclass
class ModelTable:
    query_rows: np.ndarray
    query_state: torch.Tensor
    donor_tokens: torch.Tensor
    anchor: torch.Tensor
    dispersion: torch.Tensor
    target: torch.Tensor
    base_pas_root: torch.Tensor
    base_pdp_root: torch.Tensor
    truth_pas: torch.Tensor
    truth_pdp: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--maximum-steps", type=int, default=Config.maximum_steps)
    parser.add_argument("--validation-interval", type=int, default=Config.validation_interval)
    parser.add_argument("--patience-steps", type=int, default=Config.patience_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--metric-batch-size", type=int, default=Config.metric_batch_size)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_numpy(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = output.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        output,
        np.maximum(total, TINY),
        out=np.zeros_like(output),
        where=total > 0.0,
    )


def normalize_torch(value: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    output = value.clamp_min(0.0)
    total = output.sum(dim=axes, keepdim=True)
    return torch.where(
        total > TINY,
        output / total.clamp_min(TINY),
        torch.zeros_like(output),
    )


def endpoint_cosine_torch(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    axes: tuple[int, ...],
) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=axes, dtype=torch.float64)
    product = prediction.square().sum(dim=axes, dtype=torch.float64) * truth.square().sum(
        dim=axes, dtype=torch.float64
    )
    denominator = torch.sqrt(product.clamp_min(1e-24))
    cosine = torch.where(
        product > 0.0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return cosine.reshape(len(cosine), -1).mean(dim=1).float()


def spatial_features(
    positions: np.ndarray,
    site: np.ndarray,
    geometry: np.ndarray,
    bs_positions: np.ndarray,
) -> np.ndarray:
    source = bs_positions[site]
    delta = positions - source
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    azimuth = np.arctan2(delta[:, 1], delta[:, 0])[:, None]
    values: list[np.ndarray] = [
        geometry.astype(np.float32),
        (positions[:, :2] / 200.0).astype(np.float32),
        (delta[:, :2] / 200.0).astype(np.float32),
        (distance / 200.0).astype(np.float32),
        np.sin(azimuth).astype(np.float32),
        np.cos(azimuth).astype(np.float32),
    ]
    for scale in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        phase = 2.0 * np.pi * positions[:, :2] / scale
        values.extend((np.sin(phase).astype(np.float32), np.cos(phase).astype(np.float32)))
    output = np.concatenate(values, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("non-finite spatial features")
    return output


def fit_state(
    fit_rows: np.ndarray,
    residual_pas: np.ndarray,
    residual_pdp: np.ndarray,
    core_pas: np.ndarray,
    core_pdp: np.ndarray,
    device: torch.device,
    seed: int,
) -> BasisState:
    pas_basis, pas_explained = fit_basis(
        residual_pas, fit_rows, PAS_RANK, device, seed
    )
    pdp_basis, pdp_explained = fit_basis(
        residual_pdp, fit_rows, PDP_RANK, device, seed + 1
    )
    pas_residual = np.asarray(residual_pas[fit_rows], dtype=np.float32).reshape(
        len(fit_rows), -1
    )
    pdp_residual = np.asarray(residual_pdp[fit_rows], dtype=np.float32).reshape(
        len(fit_rows), -1
    )
    pas_coefficient = pas_residual @ pas_basis.T
    pdp_coefficient = pdp_residual @ pdp_basis.T
    pas_scale = np.maximum(
        np.std(pas_coefficient, axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    pdp_scale = np.maximum(
        np.std(pdp_coefficient, axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    pas_base = np.sqrt(
        np.maximum(np.asarray(core_pas[fit_rows], dtype=np.float32), 0.0)
    ).reshape(len(fit_rows), -1)
    pdp_base = np.sqrt(
        np.maximum(np.asarray(core_pdp[fit_rows], dtype=np.float32), 0.0)
    ).reshape(len(fit_rows), -1)
    pas_input = pas_base @ pas_basis.T
    pdp_input = pdp_base @ pdp_basis.T
    pas_input_mean = np.mean(pas_input, axis=0, dtype=np.float64).astype(np.float32)
    pdp_input_mean = np.mean(pdp_input, axis=0, dtype=np.float64).astype(np.float32)
    pas_input_scale = np.maximum(
        np.std(pas_input, axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    pdp_input_scale = np.maximum(
        np.std(pdp_input, axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    return BasisState(
        pas_basis=pas_basis,
        pdp_basis=pdp_basis,
        pas_scale=pas_scale,
        pdp_scale=pdp_scale,
        pas_input_mean=pas_input_mean,
        pas_input_scale=pas_input_scale,
        pdp_input_mean=pdp_input_mean,
        pdp_input_scale=pdp_input_scale,
        pas_explained=pas_explained,
        pdp_explained=pdp_explained,
    )


def coefficients(
    rows: np.ndarray,
    residual_pas: np.ndarray,
    residual_pdp: np.ndarray,
    state: BasisState,
) -> np.ndarray:
    pas = np.asarray(residual_pas[rows], dtype=np.float32).reshape(len(rows), -1)
    pdp = np.asarray(residual_pdp[rows], dtype=np.float32).reshape(len(rows), -1)
    pas_value = (pas @ state.pas_basis.T) / state.pas_scale
    pdp_value = (pdp @ state.pdp_basis.T) / state.pdp_scale
    return np.concatenate((pas_value, pdp_value), axis=1).astype(np.float32)


def base_codes(
    pas: np.ndarray,
    pdp: np.ndarray,
    state: BasisState,
) -> np.ndarray:
    pas_root = np.sqrt(np.maximum(np.asarray(pas, dtype=np.float32), 0.0)).reshape(
        len(pas), -1
    )
    pdp_root = np.sqrt(np.maximum(np.asarray(pdp, dtype=np.float32), 0.0)).reshape(
        len(pdp), -1
    )
    pas_code = (pas_root @ state.pas_basis.T - state.pas_input_mean) / state.pas_input_scale
    pdp_code = (pdp_root @ state.pdp_basis.T - state.pdp_input_mean) / state.pdp_input_scale
    return np.concatenate((pas_code, pdp_code), axis=1).astype(np.float32)


def query_neighbors(
    query_positions: np.ndarray,
    donor_positions: np.ndarray,
    donor_rows: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = min(count, len(donor_rows))
    if keep < 1:
        raise ValueError("empty donor pool")
    distance, local = cKDTree(donor_positions[:, :2]).query(
        query_positions[:, :2], k=keep
    )
    if keep == 1:
        distance = distance[:, None]
        local = local[:, None]
    selected = donor_rows[np.asarray(local, dtype=np.int64)]
    distance = np.asarray(distance, dtype=np.float32)
    weight = 1.0 / np.square(distance + 1.0)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return selected, distance, weight.astype(np.float32)


def build_neighbor_index(
    query_rows: np.ndarray,
    fit_rows: np.ndarray,
    broad_fold: np.ndarray,
    positions: np.ndarray,
    neighbors: int,
    exclude_query_fold: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.empty((len(query_rows), neighbors), dtype=np.int64)
    distance = np.empty((len(query_rows), neighbors), dtype=np.float32)
    weight = np.empty((len(query_rows), neighbors), dtype=np.float32)
    if exclude_query_fold:
        for fold_value in np.unique(broad_fold[query_rows]):
            local_rows = np.flatnonzero(broad_fold[query_rows] == fold_value)
            donor_rows = fit_rows[broad_fold[fit_rows] != fold_value]
            ids, dist, local_weight = query_neighbors(
                positions[query_rows[local_rows]],
                positions[donor_rows],
                donor_rows,
                neighbors,
            )
            selected[local_rows] = ids
            distance[local_rows] = dist
            weight[local_rows] = local_weight
    else:
        ids, dist, local_weight = query_neighbors(
            positions[query_rows], positions[fit_rows], fit_rows, neighbors
        )
        selected[:] = ids
        distance[:] = dist
        weight[:] = local_weight
    return selected, distance, weight


def prepare_table(
    query_rows: np.ndarray,
    fit_rows: np.ndarray,
    broad_fold: np.ndarray,
    positions: np.ndarray,
    spatial: np.ndarray,
    core_pas: np.ndarray,
    core_pdp: np.ndarray,
    truth_pas: np.ndarray,
    truth_pdp: np.ndarray,
    residual_pas: np.ndarray,
    residual_pdp: np.ndarray,
    state: BasisState,
    config: Config,
    device: torch.device,
    exclude_query_fold: bool,
    query_core_pas: np.ndarray | None = None,
    query_core_pdp: np.ndarray | None = None,
    query_truth_pas: np.ndarray | None = None,
    query_truth_pdp: np.ndarray | None = None,
) -> ModelTable:
    fit_coefficient = coefficients(fit_rows, residual_pas, residual_pdp, state)
    fit_code = base_codes(core_pas[fit_rows], core_pdp[fit_rows], state)
    coefficient_by_row = {int(row): index for index, row in enumerate(fit_rows)}
    selected, distance, weight = build_neighbor_index(
        query_rows,
        fit_rows,
        broad_fold,
        positions,
        config.neighbors,
        exclude_query_fold,
    )
    local_index = np.vectorize(coefficient_by_row.__getitem__, otypes=[np.int64])(selected)
    donor_coefficient = fit_coefficient[local_index]
    donor_code = fit_code[local_index]
    local_core_pas = core_pas[query_rows] if query_core_pas is None else query_core_pas
    local_core_pdp = core_pdp[query_rows] if query_core_pdp is None else query_core_pdp
    local_truth_pas = truth_pas[query_rows] if query_truth_pas is None else query_truth_pas
    local_truth_pdp = truth_pdp[query_rows] if query_truth_pdp is None else query_truth_pdp
    query_code = base_codes(local_core_pas, local_core_pdp, state)
    anchor = np.einsum("bk,bkd->bd", weight, donor_coefficient, optimize=True)
    centered = donor_coefficient - anchor[:, None]
    dispersion = np.sqrt(
        np.einsum("bk,bkd->bd", weight, centered * centered, optimize=True)
        + 1e-6
    ).astype(np.float32)
    relative = positions[selected, :2] - positions[query_rows, None, :2]
    angle = np.arctan2(relative[..., 1], relative[..., 0])
    relation = np.stack(
        (
            relative[..., 0] / 32.0,
            relative[..., 1] / 32.0,
            distance / 32.0,
            np.sin(angle),
            np.cos(angle),
            weight,
        ),
        axis=-1,
    ).astype(np.float32)
    donor_tokens = np.concatenate(
        (
            donor_coefficient,
            donor_code - query_code[:, None],
            relation,
        ),
        axis=-1,
    ).astype(np.float32)
    query_state = np.concatenate(
        (query_code, anchor, dispersion, spatial[query_rows]), axis=1
    ).astype(np.float32)
    if query_core_pas is None:
        target = coefficients(query_rows, residual_pas, residual_pdp, state)
    else:
        local_residual_pas = (
            np.sqrt(np.maximum(np.asarray(local_truth_pas, dtype=np.float32), 0.0))
            - np.sqrt(np.maximum(np.asarray(local_core_pas, dtype=np.float32), 0.0))
        ).reshape(len(query_rows), -1)
        local_residual_pdp = (
            np.sqrt(np.maximum(np.asarray(local_truth_pdp, dtype=np.float32), 0.0))
            - np.sqrt(np.maximum(np.asarray(local_core_pdp, dtype=np.float32), 0.0))
        ).reshape(len(query_rows), -1)
        target = np.concatenate(
            (
                (local_residual_pas @ state.pas_basis.T) / state.pas_scale,
                (local_residual_pdp @ state.pdp_basis.T) / state.pdp_scale,
            ),
            axis=1,
        ).astype(np.float32)
    return ModelTable(
        query_rows=query_rows,
        query_state=torch.as_tensor(query_state, device=device),
        donor_tokens=torch.as_tensor(donor_tokens, device=device),
        anchor=torch.as_tensor(anchor, device=device),
        dispersion=torch.as_tensor(dispersion, device=device),
        target=torch.as_tensor(target, device=device),
        base_pas_root=torch.as_tensor(
            np.sqrt(np.maximum(np.asarray(local_core_pas, dtype=np.float32), 0.0)),
            device=device,
        ),
        base_pdp_root=torch.as_tensor(
            np.sqrt(np.maximum(np.asarray(local_core_pdp, dtype=np.float32), 0.0)),
            device=device,
        ),
        truth_pas=torch.as_tensor(
            np.asarray(local_truth_pas, dtype=np.float32), device=device
        ),
        truth_pdp=torch.as_tensor(
            np.asarray(local_truth_pdp, dtype=np.float32), device=device
        ),
    )


class CrossBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.donor_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.feed_norm = nn.LayerNorm(hidden)
        self.feed = nn.Sequential(
            nn.Linear(hidden, hidden * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
        )

    def forward(self, query: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
        update, _ = self.attention(
            self.query_norm(query),
            self.donor_norm(donor),
            self.donor_norm(donor),
            need_weights=False,
        )
        query = query + update
        return query + self.feed(self.feed_norm(query))


class CorrectionStudent(nn.Module):
    def __init__(self, query_dim: int, donor_dim: int, config: Config) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(query_dim, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        self.donor = nn.Sequential(
            nn.Linear(donor_dim, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        self.blocks = nn.ModuleList(
            CrossBlock(config.hidden, config.heads, config.dropout)
            for _ in range(config.layers)
        )
        self.output_norm = nn.LayerNorm(config.hidden)
        self.delta = nn.Linear(config.hidden, TOTAL_RANK)
        self.gate = nn.Linear(config.hidden, 2)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self,
        query_state: torch.Tensor,
        donor_tokens: torch.Tensor,
        anchor: torch.Tensor,
        dispersion: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(query_state)[:, None]
        donor = self.donor(donor_tokens)
        for block in self.blocks:
            query = block(query, donor)
        hidden = self.output_norm(query[:, 0])
        delta = torch.tanh(self.delta(hidden)) * (0.5 + dispersion.clamp_max(3.0))
        candidate = anchor + delta
        gate = torch.sigmoid(self.gate(hidden))
        return torch.cat(
            (
                candidate[:, :PAS_RANK] * gate[:, :1],
                candidate[:, PAS_RANK:] * gate[:, 1:],
            ),
            dim=1,
        )


def reconstruct_grouped_torch(
    base_pas_root: torch.Tensor,
    base_pdp_root: torch.Tensor,
    prediction: torch.Tensor,
    state: BasisState,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_basis = torch.as_tensor(state.pas_basis, device=device)
    pdp_basis = torch.as_tensor(state.pdp_basis, device=device)
    pas_scale = torch.as_tensor(state.pas_scale, device=device)
    pdp_scale = torch.as_tensor(state.pdp_scale, device=device)
    pas_delta = (prediction[:, :PAS_RANK] * pas_scale) @ pas_basis
    pdp_delta = (prediction[:, PAS_RANK:] * pdp_scale) @ pdp_basis
    pas_amplitude = torch.relu(base_pas_root.reshape(len(prediction), -1) + pas_delta)
    pdp_amplitude = torch.relu(base_pdp_root.reshape(len(prediction), -1) + pdp_delta)
    pas = normalize_torch(pas_amplitude.reshape(len(prediction), *PAS_SHAPE).square(), (1, 2, 3))
    pdp = normalize_torch(pdp_amplitude.reshape(len(prediction), *PDP_SHAPE).square(), (5,))
    return pas, pdp


def model_loss(
    model: CorrectionStudent,
    table: ModelTable,
    rows: torch.Tensor,
    state: BasisState,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = model(
        table.query_state[rows],
        table.donor_tokens[rows],
        table.anchor[rows],
        table.dispersion[rows],
    )
    pas, pdp = reconstruct_grouped_torch(
        table.base_pas_root[rows],
        table.base_pdp_root[rows],
        prediction,
        state,
        device,
    )
    pas_cosine = endpoint_cosine_torch(pas, table.truth_pas[rows], (1, 2, 3)).mean()
    pdp_cosine = endpoint_cosine_torch(pdp, table.truth_pdp[rows], (5,)).mean()
    coefficient_loss = torch.nn.functional.smooth_l1_loss(prediction, table.target[rows])
    action = prediction.square().mean()
    loss = (
        0.40 * (1.0 - pas_cosine)
        + 0.40 * (1.0 - pdp_cosine)
        + config.coefficient_weight * coefficient_loss
        + config.action_regularization * action
    )
    return loss, {
        "pas": float(pas_cosine.detach().cpu()),
        "pdp": float(pdp_cosine.detach().cpu()),
        "coefficient": float(coefficient_loss.detach().cpu()),
        "action_rms": float(torch.sqrt(action).detach().cpu()),
    }


@torch.inference_mode()
def predict_table(
    model: CorrectionStudent,
    table: ModelTable,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    output = torch.empty(
        (len(table.query_rows), TOTAL_RANK),
        device=table.query_state.device,
        dtype=torch.float32,
    )
    for start in range(0, len(output), batch_size):
        end = min(start + batch_size, len(output))
        output[start:end] = model(
            table.query_state[start:end],
            table.donor_tokens[start:end],
            table.anchor[start:end],
            table.dispersion[start:end],
        )
    return output


def summarize_arrays(values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, float | int]:
    return aggregate_statistics(
        PointMetricStatistics(
            pas=values[0],
            pdp=values[1],
            squared_error=values[2],
            truth_energy=values[3],
        )
    )


@torch.inference_mode()
def evaluate_inner_official(
    prediction: torch.Tensor,
    table: ModelTable,
    state: BasisState,
    full_core: np.ndarray,
    truth_all: np.ndarray,
    config: Config,
    device: torch.device,
    site_value: int,
) -> dict[str, float | int]:
    pas, pdp = reconstruct_grouped_torch(
        table.base_pas_root,
        table.base_pdp_root,
        prediction,
        state,
        device,
    )
    full_pas = pas.repeat_interleave(24, dim=-1).reshape(-1, 2, 16, 8, 4, 192)
    full_pdp = (
        pdp.repeat_interleave(4, dim=2)
        .repeat_interleave(4, dim=3)
        .reshape(-1, 256, 4, 192)
    )
    values = tuple(
        np.empty(len(table.query_rows), dtype=np.float64) for _ in range(4)
    )
    for start in range(0, len(table.query_rows), config.metric_batch_size):
        end = min(start + config.metric_batch_size, len(table.query_rows))
        rows = table.query_rows[start:end]
        base = torch.as_tensor(
            np.array(full_core[rows], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[rows], dtype=np.complex64, copy=True), device=device
        )
        candidate = base.clone()
        active = torch.nonzero(
            prediction[start:end].abs().amax(dim=1) > 1e-8,
            as_tuple=False,
        ).flatten()
        if len(active):
            local = base[active]
            reference_power = local.abs().square().sum(dim=(1, 2, 3))
            for _ in range(config.project_cycles):
                if site_value == 0:
                    local = exact_pdp_projection(local, full_pdp[start:end][active])
                    local = exact_pas_projection(local, full_pas[start:end][active])
                else:
                    local = exact_pas_projection(local, full_pas[start:end][active])
                    local = exact_pdp_projection(local, full_pdp[start:end][active])
                local = preserve_point_power(local, reference_power)
            candidate[active] = local
        block = point_metrics(
            candidate,
            truth,
            pas_distribution(truth),
            pdp_distribution(truth),
        )
        for destination, source in zip(values, block, strict=True):
            destination[start:end] = source
    return summarize_arrays(values)


def train_model(
    table: ModelTable,
    state: BasisState,
    config: Config,
    device: torch.device,
    steps: int,
    seed: int,
    validation_table: ModelTable | None = None,
    full_core: np.ndarray | None = None,
    truth_all: np.ndarray | None = None,
    site_value: int = 0,
) -> tuple[CorrectionStudent, int, list[dict[str, object]]]:
    set_seed(seed)
    model = CorrectionStudent(
        table.query_state.shape[1], table.donor_tokens.shape[2], config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 17)
    history: list[dict[str, object]] = []
    best_step = 0
    best_score = -math.inf
    if validation_table is not None:
        zero = torch.zeros(
            (len(validation_table.query_rows), TOTAL_RANK), device=device
        )
        baseline = evaluate_inner_official(
            zero,
            validation_table,
            state,
            full_core,
            truth_all,
            config,
            device,
            site_value,
        )
        best_score = float(baseline["score"])
        history.append({"step": 0, "validation": baseline})
    rolling: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        model.train()
        rows = torch.randint(
            0,
            len(table.query_rows),
            (min(config.batch_size, len(table.query_rows)),),
            generator=generator,
            device=device,
        )
        loss, summary = model_loss(model, table, rows, state, config, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()
        summary["loss"] = float(loss.detach().cpu())
        rolling.append(summary)
        if step % config.validation_interval == 0 or step == steps:
            train_summary = {
                key: float(np.mean([item[key] for item in rolling]))
                for key in rolling[0]
            }
            record: dict[str, object] = {"step": step, "train": train_summary}
            if validation_table is not None:
                prediction = predict_table(model, validation_table, config.batch_size)
                validation = evaluate_inner_official(
                    prediction,
                    validation_table,
                    state,
                    full_core,
                    truth_all,
                    config,
                    device,
                    site_value,
                )
                record["validation"] = validation
                score = float(validation["score"])
                if score > best_score + 1e-7:
                    best_score = score
                    best_step = step
                print(
                    f"E161 select step={step}/{steps} train={train_summary['loss']:.6f} "
                    f"inner={score:.6f} best={best_score:.6f}@{best_step}",
                    flush=True,
                )
            else:
                print(
                    f"E161 final step={step}/{steps} train={train_summary['loss']:.6f}",
                    flush=True,
                )
            history.append(record)
            rolling.clear()
            if (
                validation_table is not None
                and step - best_step >= config.patience_steps
                and step >= config.patience_steps
            ):
                break
    return model, best_step if validation_table is not None else steps, history


def coefficient_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    difference = prediction - target
    mse = float(np.mean(difference * difference, dtype=np.float64))
    target_energy = float(np.mean(target * target, dtype=np.float64))
    flat_prediction = prediction.reshape(-1).astype(np.float64)
    flat_target = target.reshape(-1).astype(np.float64)
    correlation_defined = bool(
        np.std(flat_prediction) > 1e-12 and np.std(flat_target) > 1e-12
    )
    correlation = (
        float(np.corrcoef(flat_prediction, flat_target)[0, 1])
        if correlation_defined
        else 0.0
    )
    numerator = np.sum(prediction * target, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(prediction * prediction, axis=1, dtype=np.float64)
        * np.sum(target * target, axis=1, dtype=np.float64)
    )
    direction = np.divide(
        numerator,
        np.where(denominator == 0.0, 1e-12, denominator),
    )
    return {
        "mse": mse,
        "zero_baseline_mse": target_energy,
        "flat_pearson": correlation,
        "flat_pearson_defined": correlation_defined,
        "mean_direction_cosine": float(np.mean(direction)),
    }


def project_canonical_targets(
    canonical_core: np.ndarray,
    target_pas: np.ndarray,
    target_pdp: np.ndarray,
    site: np.ndarray,
    output_path,
    config: Config,
    device: torch.device,
    action_mask: np.ndarray | None = None,
) -> np.ndarray:
    full_pas_np, full_pdp_np = render_full_targets(target_pas, target_pdp)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=canonical_core.shape
    )
    for start in range(0, len(canonical_core), config.metric_batch_size):
        end = min(start + config.metric_batch_size, len(canonical_core))
        base = torch.as_tensor(
            np.array(canonical_core[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        pas = torch.as_tensor(full_pas_np[start:end], device=device).reshape(
            end - start, 2, 16, 8, 4, 192
        )
        pdp = torch.as_tensor(full_pdp_np[start:end], device=device)
        candidate = base.clone()
        enabled = (
            np.ones(end - start, dtype=bool)
            if action_mask is None
            else np.asarray(action_mask[start:end], dtype=bool)
        )
        for _ in range(config.project_cycles):
            for site_value in np.unique(site[start:end]):
                rows = np.flatnonzero(
                    (site[start:end] == site_value) & enabled
                )
                if len(rows) == 0:
                    continue
                local = candidate[rows]
                reference_power = base[rows].abs().square().sum(dim=(1, 2, 3))
                if int(site_value) == 0:
                    local = exact_pdp_projection(local, pdp[rows])
                    local = exact_pas_projection(local, pas[rows])
                else:
                    local = exact_pas_projection(local, pas[rows])
                    local = exact_pdp_projection(local, pdp[rows])
                candidate[rows] = preserve_point_power(local, reference_power)
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        output.flush()
    return np.load(output_path, mmap_mode="r")


def fold_scores(
    prediction: np.ndarray,
    baseline: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    batch_size: int,
) -> list[dict[str, object]]:
    records = []
    for fold_value in range(5):
        rows = np.flatnonzero(fold_id == fold_value)
        predicted = score_channel(prediction[rows], truth_all, query_ids[rows], batch_size)
        base = score_channel(baseline[rows], truth_all, query_ids[rows], batch_size)
        records.append(
            {
                "fold": fold_value,
                "count": int(len(rows)),
                "baseline": base,
                "student": predicted,
                "delta": float(predicted["score"] - base["score"]),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    config = Config(
        maximum_steps=args.maximum_steps,
        validation_interval=args.validation_interval,
        patience_steps=args.patience_steps,
        batch_size=args.batch_size,
        metric_batch_size=args.metric_batch_size,
    )
    started = time.time()
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    full_core = np.load(FULL_CORE, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    core_pas = np.load(FULL_CORE_PAS, mmap_mode="r")
    core_pdp = np.load(FULL_CORE_PDP, mmap_mode="r")
    truth_pas = np.load(FULL_TRUTH_PAS, mmap_mode="r")
    truth_pdp = np.load(FULL_TRUTH_PDP, mmap_mode="r")
    residual_pas = np.load(FULL_RESIDUAL_PAS, mmap_mode="r")
    residual_pdp = np.load(FULL_RESIDUAL_PDP, mmap_mode="r")
    broad_fold = np.load(FOLD_IDS, allow_pickle=False).astype(np.int8)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    canonical_broad = broad_fold[query_ids]
    setup, positions, _, full_site, _ = load_positions()
    full_site = full_site.astype(np.int8)
    canonical_site = full_site[query_ids]
    geometry = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    spatial = spatial_features(positions, full_site, geometry, setup.bs_positions)
    if args.smoke:
        nested_rows = np.flatnonzero(
            (broad_fold != 0) & (broad_fold != 1) & (full_site == 0)
        )
        state = fit_state(
            nested_rows,
            residual_pas,
            residual_pdp,
            core_pas,
            core_pdp,
            device,
            config.seed,
        )
        table = prepare_table(
            nested_rows,
            nested_rows,
            broad_fold,
            positions,
            spatial,
            core_pas,
            core_pdp,
            truth_pas,
            truth_pdp,
            residual_pas,
            residual_pdp,
            state,
            config,
            device,
            True,
        )
        model = CorrectionStudent(
            table.query_state.shape[1], table.donor_tokens.shape[2], config
        ).to(device)
        rows = torch.arange(min(config.batch_size, len(table.query_rows)), device=device)
        loss, summary = model_loss(model, table, rows, state, config, device)
        loss.backward()
        prediction = predict_table(model, table, config.batch_size)
        pas, pdp = reconstruct_grouped_torch(
            table.base_pas_root, table.base_pdp_root, prediction, state, device
        )
        print(
            json.dumps(
                {
                    "status": "smoke_passed",
                    "train_count": int(len(nested_rows)),
                    "query_state_shape": list(table.query_state.shape),
                    "donor_token_shape": list(table.donor_tokens.shape),
                    "prediction_shape": list(prediction.shape),
                    "pas_shape": list(pas.shape),
                    "pdp_shape": list(pdp.shape),
                    "loss": float(loss.detach().cpu()),
                    "summary": summary,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    canonical_core = np.load(OOF_SCALED, mmap_mode="r")
    canonical_core_tensor = torch.as_tensor(
        np.array(canonical_core, dtype=np.complex64, copy=True), device=device
    )
    with torch.inference_mode():
        canonical_core_pas = pas_distribution(canonical_core_tensor).reshape(
            len(query_ids), 2, 16, 8, 4, 8, 24
        ).mean(dim=6)
        canonical_core_pas = normalize_torch(canonical_core_pas, (1, 2, 3)).cpu().numpy()
        canonical_core_pdp = pdp_distribution(canonical_core_tensor).reshape(
            len(query_ids), 2, 4, 4, 2, 4, 4, 192
        ).mean(dim=(3, 5))
        canonical_core_pdp = normalize_torch(canonical_core_pdp, (5,)).cpu().numpy()
        canonical_truth_tensor = torch.as_tensor(
            np.array(truth_all[query_ids], dtype=np.complex64, copy=True), device=device
        )
        canonical_truth_pas = pas_distribution(canonical_truth_tensor).reshape(
            len(query_ids), 2, 16, 8, 4, 8, 24
        ).mean(dim=6)
        canonical_truth_pas = normalize_torch(canonical_truth_pas, (1, 2, 3)).cpu().numpy()
        canonical_truth_pdp = pdp_distribution(canonical_truth_tensor).reshape(
            len(query_ids), 2, 4, 4, 2, 4, 4, 192
        ).mean(dim=(3, 5))
        canonical_truth_pdp = normalize_torch(canonical_truth_pdp, (5,)).cpu().numpy()
    del canonical_core_tensor, canonical_truth_tensor
    torch.cuda.empty_cache()

    student_pas = np.array(canonical_core_pas, dtype=np.float32, copy=True)
    student_pdp = np.array(canonical_core_pdp, dtype=np.float32, copy=True)
    knn_pas = np.array(canonical_core_pas, dtype=np.float32, copy=True)
    knn_pdp = np.array(canonical_core_pdp, dtype=np.float32, copy=True)
    student_coefficients = np.zeros((len(query_ids), TOTAL_RANK), dtype=np.float32)
    knn_coefficients = np.zeros_like(student_coefficients)
    target_coefficients = np.zeros_like(student_coefficients)
    student_action_mask = np.zeros(len(query_ids), dtype=bool)
    training_records: list[dict[str, object]] = []

    for outer in range(5):
        inner = (outer + 1) % 5
        for site_value in (0, 1):
            nested_rows = np.flatnonzero(
                (broad_fold != outer)
                & (broad_fold != inner)
                & (full_site == site_value)
            )
            inner_rows = np.flatnonzero(
                (broad_fold == inner) & (full_site == site_value)
            )
            nested_state = fit_state(
                nested_rows,
                residual_pas,
                residual_pdp,
                core_pas,
                core_pdp,
                device,
                config.seed + outer * 1009 + site_value * 31,
            )
            nested_table = prepare_table(
                nested_rows,
                nested_rows,
                broad_fold,
                positions,
                spatial,
                core_pas,
                core_pdp,
                truth_pas,
                truth_pdp,
                residual_pas,
                residual_pdp,
                nested_state,
                config,
                device,
                True,
            )
            inner_table = prepare_table(
                inner_rows,
                nested_rows,
                broad_fold,
                positions,
                spatial,
                core_pas,
                core_pdp,
                truth_pas,
                truth_pdp,
                residual_pas,
                residual_pdp,
                nested_state,
                config,
                device,
                False,
            )
            print(
                f"E161 outer={outer} site={site_value} select "
                f"train={len(nested_rows)} inner={len(inner_rows)}",
                flush=True,
            )
            _, selected_step, selection_history = train_model(
                nested_table,
                nested_state,
                config,
                device,
                config.maximum_steps,
                config.seed + outer * 100_003 + site_value * 997,
                inner_table,
                full_core,
                truth_all,
                site_value,
            )
            del nested_table, inner_table
            torch.cuda.empty_cache()

            fit_rows = np.flatnonzero(
                (broad_fold != outer) & (full_site == site_value)
            )
            held_canonical = np.flatnonzero(
                (canonical_broad == outer) & (canonical_site == site_value)
            )
            held_rows = query_ids[held_canonical]
            state = fit_state(
                fit_rows,
                residual_pas,
                residual_pdp,
                core_pas,
                core_pdp,
                device,
                config.seed + 500_000 + outer * 1009 + site_value * 31,
            )
            train_table = prepare_table(
                fit_rows,
                fit_rows,
                broad_fold,
                positions,
                spatial,
                core_pas,
                core_pdp,
                truth_pas,
                truth_pdp,
                residual_pas,
                residual_pdp,
                state,
                config,
                device,
                True,
            )
            held_table = prepare_table(
                held_rows,
                fit_rows,
                broad_fold,
                positions,
                spatial,
                core_pas,
                core_pdp,
                truth_pas,
                truth_pdp,
                residual_pas,
                residual_pdp,
                state,
                config,
                device,
                False,
                canonical_core_pas[held_canonical],
                canonical_core_pdp[held_canonical],
                canonical_truth_pas[held_canonical],
                canonical_truth_pdp[held_canonical],
            )
            print(
                f"E161 outer={outer} site={site_value} final "
                f"train={len(fit_rows)} held={len(held_rows)} steps={selected_step}",
                flush=True,
            )
            if selected_step > 0:
                model, _, final_history = train_model(
                    train_table,
                    state,
                    config,
                    device,
                    selected_step,
                    config.seed + 900_000 + outer * 100_003 + site_value * 997,
                    site_value=site_value,
                )
                predicted = predict_table(model, held_table, config.batch_size)
                checkpoint = ARTIFACT_DIR / f"outer{outer}_site{site_value}_student.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(config),
                        "outer": outer,
                        "site": site_value,
                        "selected_step": selected_step,
                        "pas_basis": state.pas_basis,
                        "pdp_basis": state.pdp_basis,
                        "pas_scale": state.pas_scale,
                        "pdp_scale": state.pdp_scale,
                        "pas_input_mean": state.pas_input_mean,
                        "pas_input_scale": state.pas_input_scale,
                        "pdp_input_mean": state.pdp_input_mean,
                        "pdp_input_scale": state.pdp_input_scale,
                    },
                    checkpoint,
                )
            else:
                final_history = []
                predicted = torch.zeros(
                    (len(held_rows), TOTAL_RANK), device=device, dtype=torch.float32
                )
                checkpoint = None
            knn_prediction = held_table.anchor
            with torch.inference_mode():
                predicted_pas, predicted_pdp = reconstruct_grouped_torch(
                    held_table.base_pas_root,
                    held_table.base_pdp_root,
                    predicted,
                    state,
                    device,
                )
                local_knn_pas, local_knn_pdp = reconstruct_grouped_torch(
                    held_table.base_pas_root,
                    held_table.base_pdp_root,
                    knn_prediction,
                    state,
                    device,
                )
            student_pas[held_canonical] = predicted_pas.cpu().numpy()
            student_pdp[held_canonical] = predicted_pdp.cpu().numpy()
            knn_pas[held_canonical] = local_knn_pas.cpu().numpy()
            knn_pdp[held_canonical] = local_knn_pdp.cpu().numpy()
            student_coefficients[held_canonical] = predicted.cpu().numpy()
            knn_coefficients[held_canonical] = knn_prediction.cpu().numpy()
            target_coefficients[held_canonical] = held_table.target.cpu().numpy()
            student_action_mask[held_canonical] = selected_step > 0
            training_records.append(
                {
                    "outer": outer,
                    "inner": inner,
                    "site": site_value,
                    "nested_train_count": int(len(nested_rows)),
                    "inner_count": int(len(inner_rows)),
                    "final_train_count": int(len(fit_rows)),
                    "held_count": int(len(held_rows)),
                    "selected_step": int(selected_step),
                    "nested_basis_explained": {
                        "pas": nested_state.pas_explained,
                        "pdp": nested_state.pdp_explained,
                    },
                    "final_basis_explained": {
                        "pas": state.pas_explained,
                        "pdp": state.pdp_explained,
                    },
                    "selection_history": selection_history,
                    "final_history": final_history,
                    "checkpoint": PATHS.relative(checkpoint) if checkpoint else None,
                }
            )
            del train_table, held_table
            if selected_step > 0:
                del model
            torch.cuda.empty_cache()

    student_pre_path = ARTIFACT_DIR / "canonical_student_pre_lifecycle.npy"
    student_complete_path = ARTIFACT_DIR / "canonical_student_complete.npy"
    knn_pre_path = ARTIFACT_DIR / "canonical_knn_pre_lifecycle.npy"
    knn_complete_path = ARTIFACT_DIR / "canonical_knn_complete.npy"
    student_pre = project_canonical_targets(
        canonical_core,
        student_pas,
        student_pdp,
        canonical_site,
        student_pre_path,
        config,
        device,
        student_action_mask,
    )
    knn_pre = project_canonical_targets(
        canonical_core,
        knn_pas,
        knn_pdp,
        canonical_site,
        knn_pre_path,
        config,
        device,
        np.ones(len(query_ids), dtype=bool),
    )
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    student_complete = apply_lifecycle(
        student_pre,
        student_complete_path,
        canonical_fold,
        canonical_site,
        regime,
        device,
        config.metric_batch_size,
    )
    knn_complete = apply_lifecycle(
        knn_pre,
        knn_complete_path,
        canonical_fold,
        canonical_site,
        regime,
        device,
        config.metric_batch_size,
    )
    baseline_complete = np.load(E160_DIR / "canonical_v7_complete.npy", mmap_mode="r")
    baseline_score = score_channel(
        baseline_complete, truth_all, query_ids, config.metric_batch_size
    )
    student_pre_score = score_channel(
        student_pre, truth_all, query_ids, config.metric_batch_size
    )
    student_complete_score = score_channel(
        student_complete, truth_all, query_ids, config.metric_batch_size
    )
    knn_pre_score = score_channel(knn_pre, truth_all, query_ids, config.metric_batch_size)
    knn_complete_score = score_channel(
        knn_complete, truth_all, query_ids, config.metric_batch_size
    )
    folds = fold_scores(
        student_complete,
        baseline_complete,
        truth_all,
        query_ids,
        canonical_fold,
        config.metric_batch_size,
    )
    student_diagnostics = coefficient_diagnostics(
        student_coefficients, target_coefficients
    )
    knn_diagnostics = coefficient_diagnostics(knn_coefficients, target_coefficients)
    passed = float(student_complete_score["score"]) > V12_SCORE
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        canonical_fold=canonical_fold,
        broad_fold=canonical_broad,
        site=canonical_site,
        student_pas=student_pas.astype(np.float16),
        student_pdp=student_pdp.astype(np.float16),
        knn_pas=knn_pas.astype(np.float16),
        knn_pdp=knn_pdp.astype(np.float16),
        student_coefficients=student_coefficients.astype(np.float16),
        knn_coefficients=knn_coefficients.astype(np.float16),
        target_coefficients=target_coefficients.astype(np.float16),
        student_action_mask=student_action_mask,
    )
    payload = {
        "experiment": VERSION,
        "status": "student_gate_passed" if passed else "student_gate_failed",
        "official_inputs_only_at_inference": True,
        "sionna_used": False,
        "held_truth_used_as_model_input": False,
        "config": asdict(config),
        "success_gate": {
            "required_score": V12_SCORE,
            "actual_score": float(student_complete_score["score"]),
            "passed": passed,
        },
        "official": {
            "v7_complete": baseline_score,
            "student_pre_lifecycle": student_pre_score,
            "student_complete": student_complete_score,
            "knn_pre_lifecycle": knn_pre_score,
            "knn_complete": knn_complete_score,
        },
        "delta_to_v7": float(student_complete_score["score"] - baseline_score["score"]),
        "gap_to_v12": float(student_complete_score["score"] - V12_SCORE),
        "positive_folds": int(sum(float(row["delta"]) > 0.0 for row in folds)),
        "folds": folds,
        "coefficient_diagnostics": {
            "student": student_diagnostics,
            "knn": knn_diagnostics,
        },
        "training": training_records,
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "student_pre_lifecycle": PATHS.relative(student_pre_path),
            "student_complete": PATHS.relative(student_complete_path),
            "knn_pre_lifecycle": PATHS.relative(knn_pre_path),
            "knn_complete": PATHS.relative(knn_complete_path),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
