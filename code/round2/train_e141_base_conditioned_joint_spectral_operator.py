#!/usr/bin/env python3
"""Train an official-only, base-conditioned joint PAS/PDP correction operator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from torch import nn

from analyze_e138_v12_guided_official_correction_student import (
    V12_CHANNEL,
    V7_CHANNEL,
    expanded_endpoint,
)
from analyze_e86_surface_endpoint_v11_blend import endpoint_target
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import FOLD_CACHE, OOF_META, statistics
from run_v3_anchored_neural_rbf import (
    Config as ContextConfig,
    ContextTable,
    build_train_contexts,
    set_seed,
    validation_context,
)
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint
from train_surface_interaction_energy_model import POINT_POWER, TARGET_PAS, TARGET_PDP


VERSION = "Round2-E141-base-conditioned-joint-spectral-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "base_conditioned_joint_spectral_operator"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_base_conditioned_joint_spectral_operator"
)
REPORT = ANALYSIS_DIR / "base_conditioned_joint_spectral_operator.json"
METRICS = ARTIFACT_DIR / "base_conditioned_joint_spectral_operator_metrics.npz"
GEOMETRY = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_geometry_visibility_features.npz"
)
LOS_LABELS = (
    PATHS.artifacts / "round2" / "geometric_los_inventory" / "geometric_los_labels.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PAS_SIZE = MP * MH * MV * N_UE
PDP_SIZE = MP * N_UE * S_SC
TINY = 1e-30
SPEED_OF_LIGHT = 299_792_458.0
SUBCARRIER_SPACING_HZ = 30_000.0
V12_LOCAL_SCORE = 0.677521


@dataclass(frozen=True)
class Config:
    pas_rank: int = 128
    pdp_rank: int = 128
    hidden: int = 320
    heads: int = 8
    cross_layers: int = 2
    dropout: float = 0.08
    epochs: int = 40
    batch_size: int = 48
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    coefficient_weight: float = 0.08
    action_regularization: float = 0.002
    output_bound: float = 3.5
    basis_guard_index: int = 2
    beta: float = 0.20
    cycles: int = 4
    order: str = "pas_pdp"
    seed: int = 20_260_809_141


@dataclass(frozen=True)
class ResidualBasis:
    components: np.ndarray
    residual_scale: np.ndarray
    input_mean: np.ndarray
    input_scale: np.ndarray
    explained_energy: float

    @property
    def rank(self) -> int:
        return int(len(self.components))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="3")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--pas-rank", type=int, default=Config.pas_rank)
    parser.add_argument("--pdp-rank", type=int, default=Config.pdp_rank)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def normalize_probability(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def normalize_amplitude_torch(
    value: torch.Tensor, shape: tuple[int, ...], axes: tuple[int, ...]
) -> torch.Tensor:
    shaped = torch.relu(value).reshape(len(value), *shape)
    norm = torch.sqrt(shaped.square().sum(dim=axes, keepdim=True).clamp_min(1e-20))
    return (shaped / norm).reshape(len(value), -1)


def probability_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pas = normalize_probability(np.load(TARGET_PAS, mmap_mode="r"), (1, 2, 3))
    pdp = normalize_probability(np.load(TARGET_PDP, mmap_mode="r"), (3,))
    pas_amplitude = np.sqrt(np.maximum(pas, 0.0)).reshape(len(pas), -1)
    pdp_amplitude = np.sqrt(np.maximum(pdp, 0.0)).reshape(len(pdp), -1)
    valid = np.asarray(np.load(POINT_POWER, mmap_mode="r") > 0.0)
    return (
        pas_amplitude.astype(np.float32),
        pdp_amplitude.astype(np.float32),
        valid,
    )


def build_state_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    setup, positions, _, site, _ = load_positions()
    geometry = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    los = np.load(LOS_LABELS, allow_pickle=False)
    source = setup.bs_positions[site]
    delta = positions - source
    distance = np.linalg.norm(delta, axis=1)
    azimuth = np.arctan2(delta[:, 1], delta[:, 0])
    delay_bin = distance * (S_SC * SUBCARRIER_SPACING_HZ / SPEED_OF_LIGHT)

    blocks: list[np.ndarray] = [
        geometry,
        (positions[:, :2] / 200.0).astype(np.float32),
        (delta[:, :2] / 200.0).astype(np.float32),
        (distance[:, None] / 200.0).astype(np.float32),
        np.sin(azimuth)[:, None].astype(np.float32),
        np.cos(azimuth)[:, None].astype(np.float32),
        np.eye(2, dtype=np.float32)[site],
    ]
    for scale in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        phase = 2.0 * np.pi * positions[:, :2] / scale
        blocks.extend((np.sin(phase).astype(np.float32), np.cos(phase).astype(np.float32)))
    delay_features = []
    for harmonic in (1.0, 2.0, 4.0):
        phase = 2.0 * np.pi * harmonic * delay_bin / S_SC
        delay_features.extend((np.sin(phase), np.cos(phase)))
    blocks.append(np.stack(delay_features, axis=1).astype(np.float32))

    fan_blocks = []
    for key in (
        "train_clear_fan_r010cm",
        "train_clear_fan_r025cm",
        "train_clear_fan_r050cm",
    ):
        fan = los[key].astype(np.float32)
        fan_blocks.extend(
            (
                fan.mean(axis=1, keepdims=True),
                ((fan.sum(axis=1) > 0) & (fan.sum(axis=1) < fan.shape[1]))[:, None].astype(
                    np.float32
                ),
            )
        )
    blocks.append(los["train_center_clear"][:, None].astype(np.float32))
    blocks.extend(fan_blocks)
    state = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("non-finite state features")
    return state, positions.astype(np.float32), site.astype(np.int64)


def anchor_amplitude_numpy(
    amplitude: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, ...],
    axes: tuple[int, ...],
    batch_size: int = 64,
) -> np.ndarray:
    output = np.empty((len(donor_ids), amplitude.shape[1]), dtype=np.float32)
    for start in range(0, len(donor_ids), batch_size):
        end = min(start + batch_size, len(donor_ids))
        donor = np.asarray(amplitude[donor_ids[start:end]], dtype=np.float32)
        value = np.einsum(
            "bk,bkd->bd", weights[start:end].astype(np.float32), donor, optimize=True
        )
        shaped = np.maximum(value, 0.0).reshape(end - start, *shape)
        norm = np.sqrt(
            np.sum(shaped * shaped, axis=axes, keepdims=True, dtype=np.float64)
        )
        shaped = np.divide(
            shaped,
            np.maximum(norm, 1e-20),
            out=np.zeros_like(shaped),
            where=norm > 0.0,
        )
        output[start:end] = shaped.reshape(end - start, -1)
    return output


def fit_residual_basis(
    truth: np.ndarray,
    anchor: np.ndarray,
    rank: int,
    seed: int,
) -> ResidualBasis:
    residual = np.asarray(truth - anchor, dtype=np.float32)
    keep = min(rank, len(residual) - 1, residual.shape[1])
    _, singular, components = randomized_svd(
        residual,
        n_components=keep,
        n_iter=5,
        random_state=seed % (2**32),
    )
    coefficient = residual @ components.T
    residual_scale = np.maximum(
        np.std(coefficient, axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    absolute = truth @ components.T
    input_mean = np.mean(absolute, axis=0, dtype=np.float64).astype(np.float32)
    input_scale = np.maximum(
        np.std(absolute, axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    total = float(np.sum(residual * residual, dtype=np.float64))
    explained = float(np.sum(singular * singular, dtype=np.float64) / max(total, TINY))
    return ResidualBasis(
        components=components.astype(np.float32),
        residual_scale=residual_scale,
        input_mean=input_mean,
        input_scale=input_scale,
        explained_energy=explained,
    )


def encode_input(value: np.ndarray, basis: ResidualBasis) -> np.ndarray:
    projection = np.asarray(value, dtype=np.float32) @ basis.components.T
    return ((projection - basis.input_mean) / basis.input_scale).astype(np.float32)


def residual_coefficients(
    truth: torch.Tensor,
    anchor: torch.Tensor,
    components: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return ((truth - anchor) @ components.T) / scale


def amplitude_cosine(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> torch.Tensor:
    predicted_power = predicted.reshape(len(predicted), *shape).square()
    truth_power = truth.reshape(len(truth), *shape).square()
    numerator = (predicted_power * truth_power).sum(dim=axes)
    product = predicted_power.square().sum(dim=axes) * truth_power.square().sum(dim=axes)
    denominator = torch.sqrt(product.clamp_min(1e-30))
    value = torch.where(product > 0.0, numerator / denominator, torch.zeros_like(numerator))
    return value.reshape(len(value), -1).mean(dim=1)


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


class JointSpectralOperator(nn.Module):
    def __init__(self, state_dim: int, config: Config) -> None:
        super().__init__()
        code_dim = config.pas_rank + config.pdp_rank
        hidden = config.hidden
        donor_input = code_dim + state_dim + 9
        query_input = state_dim + 3 * code_dim + 4
        self.donor_encoder = nn.Sequential(
            nn.Linear(donor_input, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(query_input, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.cross = nn.ModuleList(
            CrossBlock(hidden, config.heads, config.dropout)
            for _ in range(config.cross_layers)
        )
        self.shared_head = self._head(hidden, code_dim)
        self.site_heads = nn.ModuleList(self._head(hidden, code_dim) for _ in range(2))
        self.los_head = self._head(hidden, code_dim)
        self.output_bound = float(config.output_bound)

    @staticmethod
    def _head(hidden: int, output: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def forward(
        self,
        query_state: torch.Tensor,
        query_position: torch.Tensor,
        query_site: torch.Tensor,
        query_los: torch.Tensor,
        donor_state: torch.Tensor,
        donor_position: torch.Tensor,
        donor_code: torch.Tensor,
        anchor_code: torch.Tensor,
        distances: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        relative = donor_position - query_position[:, None, :]
        direction = relative / distances[:, :, None].clamp_min(1e-4)
        rank = torch.linspace(
            0.0, 1.0, donor_code.shape[1], device=donor_code.device
        )[None, :, None].expand(len(donor_code), -1, -1)
        numeric = torch.cat(
            (
                relative / 20.0,
                distances[:, :, None] / 20.0,
                torch.log1p(distances[:, :, None]),
                direction,
                weights[:, :, None].clamp(-2.0, 2.0),
                weights.abs()[:, :, None].clamp(0.0, 2.0),
                rank,
            ),
            dim=2,
        )
        donor = self.donor_encoder(
            torch.cat(
                (
                    donor_code,
                    donor_state - query_state[:, None, :],
                    numeric,
                ),
                dim=2,
            )
        )
        mean_code = donor_code.mean(dim=1)
        weighted_code = torch.sum(weights[:, :, None] * donor_code, dim=1)
        spread = torch.sqrt(
            (donor_code - mean_code[:, None, :]).square().mean(dim=1).clamp_min(1e-8)
        )
        distance_summary = torch.stack(
            (
                distances[:, 0] / 20.0,
                distances.mean(dim=1) / 20.0,
                distances.std(dim=1) / 20.0,
                weights.abs().sum(dim=1),
            ),
            dim=1,
        )
        query = self.query_encoder(
            torch.cat(
                (
                    query_state,
                    anchor_code,
                    weighted_code,
                    spread,
                    distance_summary,
                ),
                dim=1,
            )
        )
        query = query + donor.mean(dim=1) + torch.sum(weights[:, :, None] * donor, dim=1)
        query = query[:, None, :]
        for block in self.cross:
            query = block(query, donor)
        hidden = query[:, 0]
        raw = self.shared_head(hidden)
        site_values = torch.stack([head(hidden) for head in self.site_heads], dim=1)
        raw = raw + site_values[torch.arange(len(hidden), device=hidden.device), query_site]
        raw = raw + query_los[:, None] * self.los_head(hidden)
        return self.output_bound * torch.tanh(raw / self.output_bound)


@dataclass
class FoldTensors:
    state: torch.Tensor
    position: torch.Tensor
    site: torch.Tensor
    los: torch.Tensor
    pas_amplitude: torch.Tensor
    pdp_amplitude: torch.Tensor
    donor_code: torch.Tensor
    pas_components: torch.Tensor
    pdp_components: torch.Tensor
    pas_scale: torch.Tensor
    pdp_scale: torch.Tensor


def tensor_context(table: ContextTable, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "query_ids": torch.as_tensor(table.query_ids, dtype=torch.long, device=device),
        "donor_ids": torch.as_tensor(table.donor_ids, dtype=torch.long, device=device),
        "weights": torch.as_tensor(table.weights, dtype=torch.float32, device=device),
        "distances": torch.as_tensor(table.distances, dtype=torch.float32, device=device),
    }


def anchor_torch(
    amplitude: torch.Tensor,
    donor_ids: torch.Tensor,
    weights: torch.Tensor,
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> torch.Tensor:
    value = torch.einsum("bk,bkd->bd", weights, amplitude[donor_ids])
    return normalize_amplitude_torch(value, shape, axes)


def encode_anchor_torch(
    value: torch.Tensor,
    components: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (value @ components.T - mean) / scale


def forward_batch(
    model: JointSpectralOperator,
    context: dict[str, torch.Tensor],
    rows: torch.Tensor,
    guard: torch.Tensor | None,
    tensors: FoldTensors,
    pas_input_mean: torch.Tensor,
    pas_input_scale: torch.Tensor,
    pdp_input_mean: torch.Tensor,
    pdp_input_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_ids = context["query_ids"][rows]
    if context["donor_ids"].ndim == 3:
        if guard is None:
            raise ValueError("training context requires guard indices")
        donor_ids = context["donor_ids"][rows, guard]
        weights = context["weights"][rows, guard]
        distances = context["distances"][rows, guard]
    else:
        donor_ids = context["donor_ids"][rows]
        weights = context["weights"][rows]
        distances = context["distances"][rows]
    pas_anchor = anchor_torch(
        tensors.pas_amplitude, donor_ids, weights, (MP, MH, MV, N_UE), (1, 2, 3)
    )
    pdp_anchor = anchor_torch(
        tensors.pdp_amplitude, donor_ids, weights, (MP, N_UE, S_SC), (3,)
    )
    anchor_code = torch.cat(
        (
            encode_anchor_torch(
                pas_anchor,
                tensors.pas_components,
                pas_input_mean,
                pas_input_scale,
            ),
            encode_anchor_torch(
                pdp_anchor,
                tensors.pdp_components,
                pdp_input_mean,
                pdp_input_scale,
            ),
        ),
        dim=1,
    )
    coefficient = model(
        tensors.state[query_ids],
        tensors.position[query_ids, :2],
        tensors.site[query_ids],
        tensors.los[query_ids],
        tensors.state[donor_ids],
        tensors.position[donor_ids, :2],
        tensors.donor_code[donor_ids],
        anchor_code,
        distances,
        weights,
    )
    pas_coefficient = coefficient[:, : tensors.pas_components.shape[0]]
    pdp_coefficient = coefficient[:, tensors.pas_components.shape[0] :]
    pas_delta = (pas_coefficient * tensors.pas_scale) @ tensors.pas_components
    pdp_delta = (pdp_coefficient * tensors.pdp_scale) @ tensors.pdp_components
    pas_prediction = normalize_amplitude_torch(
        pas_anchor + pas_delta, (MP, MH, MV, N_UE), (1, 2, 3)
    )
    pdp_prediction = normalize_amplitude_torch(
        pdp_anchor + pdp_delta, (MP, N_UE, S_SC), (3,)
    )
    return pas_prediction, pdp_prediction, coefficient, query_ids, torch.cat(
        (pas_anchor, pdp_anchor), dim=1
    )


@torch.no_grad()
def predict_table(
    model: JointSpectralOperator,
    table: ContextTable,
    tensors: FoldTensors,
    pas_basis: ResidualBasis,
    pdp_basis: ResidualBasis,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    model.eval()
    context = tensor_context(table, device)
    pas_mean = torch.as_tensor(pas_basis.input_mean, device=device)
    pas_input_scale = torch.as_tensor(pas_basis.input_scale, device=device)
    pdp_mean = torch.as_tensor(pdp_basis.input_mean, device=device)
    pdp_input_scale = torch.as_tensor(pdp_basis.input_scale, device=device)
    pas_rows = []
    pdp_rows = []
    anchor_pas_rows = []
    anchor_pdp_rows = []
    coefficient_rows = []
    for start in range(0, len(table.query_ids), batch_size):
        end = min(start + batch_size, len(table.query_ids))
        rows = torch.arange(start, end, device=device)
        pas, pdp, coefficient, _, anchor = forward_batch(
            model,
            context,
            rows,
            None,
            tensors,
            pas_mean,
            pas_input_scale,
            pdp_mean,
            pdp_input_scale,
        )
        pas_rows.append(pas.cpu().numpy())
        pdp_rows.append(pdp.cpu().numpy())
        anchor_pas_rows.append(anchor[:, :PAS_SIZE].cpu().numpy())
        anchor_pdp_rows.append(anchor[:, PAS_SIZE:].cpu().numpy())
        coefficient_rows.append(coefficient.cpu().numpy())
    return {
        "pas_amplitude": np.concatenate(pas_rows).astype(np.float32),
        "pdp_amplitude": np.concatenate(pdp_rows).astype(np.float32),
        "anchor_pas_amplitude": np.concatenate(anchor_pas_rows).astype(np.float32),
        "anchor_pdp_amplitude": np.concatenate(anchor_pdp_rows).astype(np.float32),
        "coefficient": np.concatenate(coefficient_rows).astype(np.float32),
    }


def endpoint_summary(
    pas_amplitude: np.ndarray,
    pdp_amplitude: np.ndarray,
    truth_pas: np.ndarray,
    truth_pdp: np.ndarray,
) -> dict[str, float]:
    pas = pas_amplitude.reshape(-1, MP, MH, MV, N_UE) ** 2
    pdp = pdp_amplitude.reshape(-1, MP, N_UE, S_SC) ** 2
    target_pas = truth_pas.reshape(-1, MP, MH, MV, N_UE) ** 2
    target_pdp = truth_pdp.reshape(-1, MP, N_UE, S_SC) ** 2

    def cosine(a: np.ndarray, b: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        numerator = np.sum(a * b, axis=axes, dtype=np.float64)
        product = np.sum(a * a, axis=axes, dtype=np.float64) * np.sum(
            b * b, axis=axes, dtype=np.float64
        )
        return np.divide(
            numerator,
            np.sqrt(np.maximum(product, 1e-30)),
            out=np.zeros_like(numerator),
            where=product > 0.0,
        )

    pas_value = cosine(pas, target_pas, (1, 2, 3)).mean(axis=1)
    pdp_value = cosine(pdp, target_pdp, (3,)).mean(axis=(1, 2))
    return {
        "pas": float(np.mean(pas_value)),
        "pdp": float(np.mean(pdp_value)),
        "mean": float(np.mean(0.5 * (pas_value + pdp_value))),
    }


def complete_point_metrics(
    channel_path,
    canonical_rows: np.ndarray,
    query_ids: np.ndarray,
    truth_all: np.ndarray,
    device: torch.device,
    batch_size: int,
    endpoint_pas: np.ndarray | None = None,
    endpoint_pdp: np.ndarray | None = None,
    config: Config | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel = np.load(channel_path, mmap_mode="r")
    values = [np.empty(len(query_ids), dtype=np.float64) for _ in range(4)]
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        prediction = torch.as_tensor(
            np.array(channel[canonical_rows[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        with torch.inference_mode():
            truth_pas = pas_distribution(truth)
            truth_pdp = pdp_distribution(truth)
            if endpoint_pas is not None and endpoint_pdp is not None:
                if config is None:
                    raise ValueError("projection config is required")
                full_pas, full_pdp = expanded_endpoint(
                    endpoint_pas[start:end], endpoint_pdp[start:end], device
                )
                desired_pas = endpoint_target(
                    pas_distribution(prediction),
                    full_pas,
                    config.beta,
                    "sqrt",
                    (1, 2, 3),
                )
                desired_pdp = endpoint_target(
                    pdp_distribution(prediction),
                    full_pdp,
                    config.beta,
                    "sqrt",
                    (3,),
                )
                prediction = project_joint(
                    prediction, desired_pas, desired_pdp, config.cycles, config.order
                )
            point = point_metrics(prediction, truth, truth_pas, truth_pdp)
        for destination, source in zip(values, point):
            destination[start:end] = source
    return tuple(values)


def train_fold(
    outer: int,
    config: Config,
    state_raw: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    pas_amplitude: np.ndarray,
    pdp_amplitude: np.ndarray,
    valid: np.ndarray,
    canonical_query_ids: np.ndarray,
    canonical_fold: np.ndarray,
    truth_all: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    set_seed((config.seed + outer * 1009) % (2**32 - 1))
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{outer}_fit"].astype(np.int64)
    train_table = build_train_contexts(
        outer,
        positions,
        site,
        valid,
        fit_ids,
        ContextConfig(),
    )
    held_table = validation_context(outer, canonical_query_ids, canonical_fold)
    if not np.array_equal(
        np.sort(held_table.query_ids), np.sort(canonical_query_ids[canonical_fold == outer])
    ):
        raise RuntimeError(f"fold {outer} held query mismatch")

    state_mean = np.mean(state_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.maximum(
        np.std(state_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    state = np.clip((state_raw - state_mean) / state_std, -8.0, 8.0).astype(np.float32)
    guard = config.basis_guard_index
    basis_donor = train_table.donor_ids[:, guard].astype(np.int64)
    basis_weight = train_table.weights[:, guard].astype(np.float32)
    anchor_pas = anchor_amplitude_numpy(
        pas_amplitude,
        basis_donor,
        basis_weight,
        (MP, MH, MV, N_UE),
        (1, 2, 3),
    )
    anchor_pdp = anchor_amplitude_numpy(
        pdp_amplitude,
        basis_donor,
        basis_weight,
        (MP, N_UE, S_SC),
        (3,),
    )
    train_ids = train_table.query_ids.astype(np.int64)
    pas_basis = fit_residual_basis(
        pas_amplitude[train_ids], anchor_pas, config.pas_rank, config.seed + outer * 17
    )
    pdp_basis = fit_residual_basis(
        pdp_amplitude[train_ids], anchor_pdp, config.pdp_rank, config.seed + outer * 19
    )
    donor_code = np.concatenate(
        (
            encode_input(pas_amplitude, pas_basis),
            encode_input(pdp_amplitude, pdp_basis),
        ),
        axis=1,
    ).astype(np.float32)

    tensors = FoldTensors(
        state=torch.as_tensor(state, device=device),
        position=torch.as_tensor(positions, device=device),
        site=torch.as_tensor(site, dtype=torch.long, device=device),
        los=torch.as_tensor(
            np.load(LOS_LABELS, allow_pickle=False)["train_center_clear"].astype(np.float32),
            device=device,
        ),
        pas_amplitude=torch.as_tensor(pas_amplitude, device=device),
        pdp_amplitude=torch.as_tensor(pdp_amplitude, device=device),
        donor_code=torch.as_tensor(donor_code, device=device),
        pas_components=torch.as_tensor(pas_basis.components, device=device),
        pdp_components=torch.as_tensor(pdp_basis.components, device=device),
        pas_scale=torch.as_tensor(pas_basis.residual_scale, device=device),
        pdp_scale=torch.as_tensor(pdp_basis.residual_scale, device=device),
    )
    pas_input_mean = torch.as_tensor(pas_basis.input_mean, device=device)
    pas_input_scale = torch.as_tensor(pas_basis.input_scale, device=device)
    pdp_input_mean = torch.as_tensor(pdp_basis.input_mean, device=device)
    pdp_input_scale = torch.as_tensor(pdp_basis.input_scale, device=device)
    context = tensor_context(train_table, device)
    model = JointSpectralOperator(state.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed + outer * 2003)
    history = []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_ids))
        rows_epoch = []
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guards = torch.as_tensor(
                rng.integers(0, len(ContextConfig().guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            predicted_pas, predicted_pdp, coefficient, query_ids, anchor = forward_batch(
                model,
                context,
                rows,
                guards,
                tensors,
                pas_input_mean,
                pas_input_scale,
                pdp_input_mean,
                pdp_input_scale,
            )
            truth_pas = tensors.pas_amplitude[query_ids]
            truth_pdp = tensors.pdp_amplitude[query_ids]
            pas_cosine = amplitude_cosine(
                predicted_pas, truth_pas, (MP, MH, MV, N_UE), (1, 2, 3)
            )
            pdp_cosine = amplitude_cosine(
                predicted_pdp, truth_pdp, (MP, N_UE, S_SC), (3,)
            )
            pas_target = residual_coefficients(
                truth_pas,
                anchor[:, :PAS_SIZE],
                tensors.pas_components,
                tensors.pas_scale,
            )
            pdp_target = residual_coefficients(
                truth_pdp,
                anchor[:, PAS_SIZE:],
                tensors.pdp_components,
                tensors.pdp_scale,
            )
            target = torch.cat((pas_target, pdp_target), dim=1)
            coefficient_loss = (coefficient - target).square().mean()
            action = coefficient.square().mean()
            loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean())
            loss = loss + config.coefficient_weight * coefficient_loss
            loss = loss + config.action_regularization * action
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            rows_epoch.append(
                (
                    float(loss.detach().cpu()),
                    float(pas_cosine.mean().detach().cpu()),
                    float(pdp_cosine.mean().detach().cpu()),
                    float(coefficient_loss.detach().cpu()),
                    float(action.detach().sqrt().cpu()),
                )
            )
        row = {
            "epoch": epoch,
            "loss": float(np.mean([item[0] for item in rows_epoch])),
            "pas": float(np.mean([item[1] for item in rows_epoch])),
            "pdp": float(np.mean([item[2] for item in rows_epoch])),
            "coefficient_mse": float(np.mean([item[3] for item in rows_epoch])),
            "action_rms": float(np.mean([item[4] for item in rows_epoch])),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
            print(
                f"E141 fold={outer} epoch={epoch}/{config.epochs} "
                f"train={row['pas']:.6f}/{row['pdp']:.6f} "
                f"coef={row['coefficient_mse']:.5f} action={row['action_rms']:.4f}",
                flush=True,
            )

    prediction = predict_table(
        model,
        held_table,
        tensors,
        pas_basis,
        pdp_basis,
        config.batch_size,
        device,
    )
    held_ids = held_table.query_ids.astype(np.int64)
    strict_rows = np.flatnonzero(canonical_fold == outer)
    if not np.array_equal(held_ids, canonical_query_ids[strict_rows]):
        raise RuntimeError(f"fold {outer} held order mismatch")
    truth_pas = pas_amplitude[held_ids]
    truth_pdp = pdp_amplitude[held_ids]
    endpoint = endpoint_summary(
        prediction["pas_amplitude"],
        prediction["pdp_amplitude"],
        truth_pas,
        truth_pdp,
    )
    anchor_endpoint = endpoint_summary(
        prediction["anchor_pas_amplitude"],
        prediction["anchor_pdp_amplitude"],
        truth_pas,
        truth_pdp,
    )

    pas_true_coefficient = (
        (truth_pas - prediction["anchor_pas_amplitude"]) @ pas_basis.components.T
    ) / pas_basis.residual_scale
    pdp_true_coefficient = (
        (truth_pdp - prediction["anchor_pdp_amplitude"]) @ pdp_basis.components.T
    ) / pdp_basis.residual_scale
    oracle_pas = prediction["anchor_pas_amplitude"] + (
        pas_true_coefficient * pas_basis.residual_scale
    ) @ pas_basis.components
    oracle_pdp = prediction["anchor_pdp_amplitude"] + (
        pdp_true_coefficient * pdp_basis.residual_scale
    ) @ pdp_basis.components
    oracle_pas = np.maximum(oracle_pas, 0.0).reshape(-1, MP, MH, MV, N_UE)
    oracle_pdp = np.maximum(oracle_pdp, 0.0).reshape(-1, MP, N_UE, S_SC)
    oracle_pas /= np.sqrt(
        np.maximum(np.sum(oracle_pas**2, axis=(1, 2, 3), keepdims=True), 1e-20)
    )
    oracle_pdp /= np.sqrt(
        np.maximum(np.sum(oracle_pdp**2, axis=(3,), keepdims=True), 1e-20)
    )
    oracle_pas = oracle_pas.reshape(len(held_ids), -1)
    oracle_pdp = oracle_pdp.reshape(len(held_ids), -1)

    predicted_pas = prediction["pas_amplitude"].reshape(-1, MP, MH, MV, N_UE) ** 2
    predicted_pdp = prediction["pdp_amplitude"].reshape(-1, MP, N_UE, S_SC) ** 2
    anchor_pas_probability = (
        prediction["anchor_pas_amplitude"].reshape(-1, MP, MH, MV, N_UE) ** 2
    )
    anchor_pdp_probability = (
        prediction["anchor_pdp_amplitude"].reshape(-1, MP, N_UE, S_SC) ** 2
    )
    oracle_pas_probability = oracle_pas.reshape(-1, MP, MH, MV, N_UE) ** 2
    oracle_pdp_probability = oracle_pdp.reshape(-1, MP, N_UE, S_SC) ** 2

    v7_point = complete_point_metrics(
        V7_CHANNEL,
        strict_rows,
        held_ids,
        truth_all,
        device,
        config.batch_size,
    )
    v12_point = complete_point_metrics(
        V12_CHANNEL,
        strict_rows,
        held_ids,
        truth_all,
        device,
        config.batch_size,
    )
    anchor_point = complete_point_metrics(
        V7_CHANNEL,
        strict_rows,
        held_ids,
        truth_all,
        device,
        config.batch_size,
        anchor_pas_probability,
        anchor_pdp_probability,
        config,
    )
    model_point = complete_point_metrics(
        V7_CHANNEL,
        strict_rows,
        held_ids,
        truth_all,
        device,
        config.batch_size,
        predicted_pas,
        predicted_pdp,
        config,
    )
    oracle_point = complete_point_metrics(
        V7_CHANNEL,
        strict_rows,
        held_ids,
        truth_all,
        device,
        config.batch_size,
        oracle_pas_probability,
        oracle_pdp_probability,
        config,
    )
    methods = {
        "v7": statistics(*v7_point),
        "v12": statistics(*v12_point),
        "anchor": statistics(*anchor_point),
        "model": statistics(*model_point),
        "residual_basis_oracle": statistics(*oracle_point),
    }
    v7_score = float(methods["v7"]["score"])
    oracle_gain = float(methods["residual_basis_oracle"]["score"] - v7_score)
    model_gain = float(methods["model"]["score"] - v7_score)
    recovered = model_gain / max(oracle_gain, 1e-12)
    checkpoint = ARTIFACT_DIR / f"fold{outer}_joint_spectral_operator.pt"
    torch.save(
        {
            "experiment": VERSION,
            "fold": outer,
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "pas_basis": asdict(pas_basis),
            "pdp_basis": asdict(pdp_basis),
            "sionna_required_for_inference": False,
        },
        checkpoint,
    )
    record = {
        "fold": outer,
        "fit_count": int(len(fit_ids)),
        "train_count": int(len(train_ids)),
        "held_count": int(len(held_ids)),
        "state_dim": int(state.shape[1]),
        "basis": {
            "pas_rank": pas_basis.rank,
            "pdp_rank": pdp_basis.rank,
            "pas_explained_energy": pas_basis.explained_energy,
            "pdp_explained_energy": pdp_basis.explained_energy,
        },
        "endpoint": {"anchor": anchor_endpoint, "model": endpoint},
        "complete": methods,
        "delta_to_v7": model_gain,
        "delta_to_v12": float(methods["model"]["score"] - methods["v12"]["score"]),
        "recovered_oracle_fraction": float(recovered),
        "screen_passed": bool(
            model_gain >= 0.002
            and endpoint["pas"] >= anchor_endpoint["pas"]
            and endpoint["pdp"] >= anchor_endpoint["pdp"]
        ),
        "training": history,
        "checkpoint": PATHS.relative(checkpoint),
        "seconds": float(time.time() - started),
    }
    arrays = {
        "query_ids": held_ids,
        "canonical_rows": strict_rows.astype(np.int32),
        "predicted_pas": predicted_pas.astype(np.float16),
        "predicted_pdp": predicted_pdp.astype(np.float16),
        "anchor_pas": anchor_pas_probability.astype(np.float16),
        "anchor_pdp": anchor_pdp_probability.astype(np.float16),
        "oracle_pas": oracle_pas_probability.astype(np.float16),
        "oracle_pdp": oracle_pdp_probability.astype(np.float16),
        "predicted_coefficient": np.asarray(prediction["coefficient"], dtype=np.float16),
    }
    for name, point in (
        ("v7", v7_point),
        ("v12", v12_point),
        ("anchor", anchor_point),
        ("model", model_point),
        ("oracle", oracle_point),
    ):
        for metric, value in zip(("pas", "pdp", "error", "energy"), point):
            arrays[f"{name}_{metric}"] = value
    del model, tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, arrays


def aggregate_selected(
    records: list[dict[str, object]], arrays: list[dict[str, np.ndarray]]
) -> dict[str, object]:
    methods = {}
    for name in ("v7", "v12", "anchor", "model", "oracle"):
        point = tuple(
            np.concatenate([value[f"{name}_{metric}"] for value in arrays])
            for metric in ("pas", "pdp", "error", "energy")
        )
        methods[name] = statistics(*point)
    complete_five = len(records) == 5 and {int(row["fold"]) for row in records} == set(
        range(5)
    )
    fold_delta = [float(row["delta_to_v7"]) for row in records]
    model = methods["model"]
    v7 = methods["v7"]
    gate = {
        "complete_five_folds": complete_five,
        "required_score_above_v12": V12_LOCAL_SCORE,
        "actual_score": float(model["score"]),
        "positive_folds": int(np.sum(np.asarray(fold_delta) > 0.0)),
        "pas_non_worse": bool(model["pas"] >= v7["pas"]),
        "pdp_non_worse": bool(model["pdp"] >= v7["pdp"]),
        "nmse_non_worse": bool(model["nmse"] <= v7["nmse"]),
    }
    gate["passed"] = bool(
        complete_five
        and model["score"] > V12_LOCAL_SCORE
        and gate["positive_folds"] >= 4
        and gate["pas_non_worse"]
        and gate["pdp_non_worse"]
        and gate["nmse_non_worse"]
    )
    return {"methods": methods, "fold_delta": fold_delta, "deployment_gate": gate}


def main() -> None:
    args = parse_args()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    if not folds or any(value not in range(5) for value in folds) or len(set(folds)) != len(folds):
        raise ValueError("--folds must contain unique values in [0,4]")
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        pas_rank=args.pas_rank,
        pdp_rank=args.pdp_rank,
    )
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    state, positions, site = build_state_features()
    pas_amplitude, pdp_amplitude, valid = probability_targets()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    records = []
    arrays = []
    started = time.time()
    for outer in folds:
        record, values = train_fold(
            outer,
            config,
            state,
            positions,
            site,
            pas_amplitude,
            pdp_amplitude,
            valid,
            query_ids,
            fold_id,
            truth_all,
            device,
        )
        records.append(record)
        arrays.append(values)
        print(
            f"E141 fold={outer} complete model={record['complete']['model']['score']:.6f} "
            f"v7={record['complete']['v7']['score']:.6f} "
            f"v12={record['complete']['v12']['score']:.6f} "
            f"delta={record['delta_to_v7']:+.6f}",
            flush=True,
        )

    np.savez_compressed(
        METRICS,
        folds=np.asarray(folds, dtype=np.int8),
        **{
            f"fold{record['fold']}_{key}": value
            for record, fold_arrays in zip(records, arrays)
            for key, value in fold_arrays.items()
        },
    )
    aggregate = aggregate_selected(records, arrays)
    payload = {
        "experiment": VERSION,
        "status": "screen_complete" if len(folds) < 5 else "fivefold_complete",
        "primary_hypothesis": (
            "The improvement direction exposed by V12 is learnable from official data when "
            "the student is conditioned on an exact-group official-neighbor spectral base, "
            "multi-scale donor disagreement, official geometry, and the verified LOS delay law."
        ),
        "total_goal": (
            "An end-to-end official-input AI method whose canonical complete score exceeds "
            f"V12={V12_LOCAL_SCORE:.6f}, without test-position Sionna inputs."
        ),
        "official_labels_only": True,
        "sionna_test_inputs_used": False,
        "v12_role": (
            "Method design only: joint grouped PAS/PDP correction, path-stability evidence, "
            "and propagation-state branching. No V12 endpoint is a model input or label."
        ),
        "config": asdict(config),
        "folds": records,
        "aggregate": aggregate,
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
