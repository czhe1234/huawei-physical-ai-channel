#!/usr/bin/env python3
"""Select a propagation-operator architecture in Sionna, then retrain on official H.

The simulator stage may select only a model structure. It does not save a model
checkpoint. The official stage reads the locked structure name, initializes every
weight again, and trains only on official Round 2 channels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    preserve_point_power,
    truth_views,
)
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import solve_linear_rbf_weights, statistics
from run_v8_full_sionna_migration import point_metrics


VERSION = "Round2-E113-Sionna-selected-group-propagation-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_selected_group_operator"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_sionna_selected_group_operator"
)
CONTEXT_DIR = ARTIFACT_DIR / "all_point_contexts"
SIONNA_SELECTION = ARTIFACT_DIR / "sionna_structure_selection.json"
SIONNA_SCORES = ARTIFACT_DIR / "sionna_structure_scores.npz"
OFFICIAL_ENDPOINT = ARTIFACT_DIR / "official_oof_group_endpoint.npz"
OFFICIAL_CHANNEL = ARTIFACT_DIR / "official_oof_projected_channel.npy"
REPORT = ANALYSIS_DIR / "sionna_selected_group_operator.json"

SIONNA_DIR = PATHS.artifacts / "round2" / "v5_sionna_upper_bound"
SIONNA_SOURCES = {
    "depth16_base_v4lambda": "d16_base_v4lambda",
    "depth10_scatter015_v4lambda": "d10_scatter015_v4lambda",
    "depth16_base_v05lambda": "d16_base_v05lambda",
}
GEOMETRY = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
OFFICIAL_PAS = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "official_pas_cosine_sufficient_target_f32.npy"
)
OFFICIAL_PDP = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "official_pdp_cosine_sufficient_target_f32.npy"
)
OOF_V7 = (
    PATHS.artifacts
    / "round2"
    / "full_sionna_migration"
    / "oof_v7_nested_channel.npy"
)
OOF_V12 = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)

PAS_GROUPS = 4
PDP_GROUPS = 8
TOTAL_GROUPS = PAS_GROUPS + PDP_GROUPS
PAS_LENGTH = 256
PDP_LENGTH = 192
COMMON_LENGTH = PAS_LENGTH + PDP_LENGTH
TINY = 1e-20

# These thresholds and the channel action are frozen before the official stage.
ROBUSTNESS_PENALTY = 0.5
MIN_SIONNA_GAIN = 0.005
MIN_POSITIVE_SOURCES = 3
MIN_POSITIVE_SOURCE_FOLDS = 12
COARSE_ACTION_ALPHA = 0.25
COARSE_ACTION_CYCLES = 4
MIN_COMPLETE_GAIN_VS_V12 = 0.005
MIN_COMPLETE_POSITIVE_FOLDS = 4
MAX_COMPONENT_DROP = 0.001
MAX_NMSE_INCREASE = 0.002


@dataclass(frozen=True)
class TrainConfig:
    neighbors: int = 64
    candidate_neighbors: int = 384
    guard_radii_m: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0)
    smoothing: float = 10.0
    hidden: int = 96
    spectral_hidden: int = 48
    low_rank_heads: int = 4
    correction_l1_bound: float = 0.65
    correction_regularization: float = 0.012
    epochs: int = 8
    batch_size: int = 192
    learning_rate: float = 8e-4
    weight_decay: float = 2e-5
    seed: int = 2_026_080_113


@dataclass(frozen=True)
class Variant:
    name: str
    head_mode: str
    use_map: bool


VARIANTS = (
    Variant("shared_head_no_map", "shared", False),
    Variant("shared_head_map", "shared", True),
    Variant("pas_pdp_dual_head_map", "dual", True),
    Variant("low_rank4_group_head_map", "low_rank", True),
    Variant("group12_head_map", "grouped", True),
    Variant("group12_head_no_map", "grouped", False),
)


@dataclass(frozen=True)
class ContextTable:
    query_ids: np.ndarray
    donor_ids: np.ndarray
    weights: np.ndarray
    distances: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("smoke", "sionna", "official"), required=True
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--force-contexts", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_power(value: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def power_to_amplitude(value: np.ndarray) -> np.ndarray:
    return np.sqrt(normalize_power(value), dtype=np.float32)


def common_amplitude(pas: np.ndarray, pdp: np.ndarray) -> np.ndarray:
    pas_power = normalize_power(np.square(pas, dtype=np.float32).mean(axis=1))
    pdp_power = normalize_power(np.square(pdp, dtype=np.float32).mean(axis=1))
    return np.concatenate(
        [np.sqrt(pas_power, dtype=np.float32), np.sqrt(pdp_power, dtype=np.float32)],
        axis=1,
    ).astype(np.float32)


def sionna_path(prefix: str, kind: str) -> Path:
    suffix = {
        "ids": "ids.npy",
        "pas": "grouped_pas_f16.npy",
        "pdp": "grouped_pdp_f16.npy",
    }[kind]
    return SIONNA_DIR / f"{prefix}_train_full4000_{suffix}"


def load_sionna_targets(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.load(sionna_path(prefix, "ids"), allow_pickle=False)
    if not np.array_equal(ids, np.arange(len(ids), dtype=ids.dtype)):
        raise ValueError(f"Sionna source {prefix} is not in official training-ID order")
    pas_raw = np.asarray(np.load(sionna_path(prefix, "pas"), mmap_mode="r"), dtype=np.float32)
    pdp_raw = np.asarray(np.load(sionna_path(prefix, "pdp"), mmap_mode="r"), dtype=np.float32)
    if pas_raw.shape != (4000, 32, PAS_LENGTH):
        raise ValueError(f"unexpected Sionna PAS shape: {pas_raw.shape}")
    if pdp_raw.shape != (4000, PDP_GROUPS, PDP_LENGTH):
        raise ValueError(f"unexpected Sionna PDP shape: {pdp_raw.shape}")

    # Each PAS frequency block is first converted to a direction. The eight
    # directions for one UE port are then averaged, matching the official
    # cosine-sufficient target as closely as this compact cache permits.
    pas_power = np.square(pas_raw, dtype=np.float32).reshape(
        len(pas_raw), PAS_GROUPS, 8, PAS_LENGTH
    )
    pas_power = normalize_power(pas_power).mean(axis=2, dtype=np.float64).astype(np.float32)
    pas = power_to_amplitude(pas_power)
    pdp = power_to_amplitude(np.square(pdp_raw, dtype=np.float32))
    return pas, pdp, common_amplitude(pas, pdp)


def load_official_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pas_raw = np.asarray(np.load(OFFICIAL_PAS, mmap_mode="r"), dtype=np.float32)
    pdp_raw = np.asarray(np.load(OFFICIAL_PDP, mmap_mode="r"), dtype=np.float32)
    pas_power = pas_raw.reshape(len(pas_raw), PAS_LENGTH, PAS_GROUPS).transpose(0, 2, 1)
    pdp_power = pdp_raw.reshape(len(pdp_raw), PDP_GROUPS, PDP_LENGTH)
    pas = power_to_amplitude(pas_power)
    pdp = power_to_amplitude(pdp_power)
    return pas, pdp, common_amplitude(pas, pdp)


def context_path(fold: int) -> Path:
    return CONTEXT_DIR / f"fold{fold}_all_points_k64_s10_guards_4_8_12_16.npz"


def _build_context(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    config: TrainConfig,
    guards: tuple[float, ...] | None,
) -> ContextTable:
    guard_values = guards or (0.0,)
    shape = (len(query_ids), len(guard_values), config.neighbors)
    donor_ids = np.empty(shape, dtype=np.int32)
    weights = np.empty(shape, dtype=np.float32)
    distances = np.empty(shape, dtype=np.float32)
    for site_value in np.unique(site[query_ids]):
        rows = np.flatnonzero(site[query_ids] == site_value)
        bank = bank_ids[site[bank_ids] == site_value]
        count = min(config.candidate_neighbors, len(bank))
        candidate_distance, candidate_local = cKDTree(positions[bank, :2]).query(
            positions[query_ids[rows], :2], k=count, workers=-1
        )
        candidate_distance = np.atleast_2d(candidate_distance)
        candidate_ids = bank[np.atleast_2d(candidate_local)]
        for local_row, output_row in enumerate(rows):
            query_id = query_ids[output_row]
            for guard_index, guard in enumerate(guard_values):
                keep = (candidate_ids[local_row] != query_id) & (
                    candidate_distance[local_row] >= guard
                )
                selected = candidate_ids[local_row, keep][: config.neighbors]
                selected_distance = candidate_distance[local_row, keep][
                    : config.neighbors
                ]
                if len(selected) != config.neighbors:
                    raise RuntimeError(
                        f"query {query_id} guard {guard} has only {len(selected)} donors"
                    )
                donor_ids[output_row, guard_index] = selected
                distances[output_row, guard_index] = selected_distance
                weights[output_row, guard_index] = solve_linear_rbf_weights(
                    positions[selected, :2], positions[query_id, :2], config.smoothing
                )
    if guards is None:
        return ContextTable(
            query_ids,
            donor_ids[:, 0],
            weights[:, 0],
            distances[:, 0],
        )
    return ContextTable(query_ids, donor_ids, weights, distances)


def load_or_build_contexts(
    fold: int, config: TrainConfig, force: bool = False
) -> tuple[ContextTable, ContextTable, np.ndarray]:
    path = context_path(fold)
    if path.is_file() and not force:
        cached = np.load(path, allow_pickle=False)
        compatible = (
            int(cached["neighbors"]) == config.neighbors
            and np.isclose(float(cached["smoothing"]), config.smoothing)
            and np.allclose(cached["guard_radii_m"], config.guard_radii_m)
        )
        if compatible:
            train = ContextTable(
                cached["train_query_ids"].astype(np.int64),
                cached["train_donor_ids"].astype(np.int64),
                cached["train_weights"].astype(np.float32),
                cached["train_distances"].astype(np.float32),
            )
            held = ContextTable(
                cached["held_query_ids"].astype(np.int64),
                cached["held_donor_ids"].astype(np.int64),
                cached["held_weights"].astype(np.float32),
                cached["held_distances"].astype(np.float32),
            )
            return train, held, cached["held_rows"].astype(np.int64)

    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    _, positions, _, site, _ = load_positions()
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    metadata = np.load(OOF_META, allow_pickle=False)
    oof_query_ids = metadata["query_ids"].astype(np.int64)
    oof_fold_id = metadata["fold_id"].astype(np.int8)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    held_rows = np.flatnonzero(oof_fold_id == fold)
    held_ids = oof_query_ids[held_rows]
    train = _build_context(
        fit_ids, fit_ids, positions, site, config, config.guard_radii_m
    )
    held = _build_context(held_ids, fit_ids, positions, site, config, None)
    np.savez(
        path,
        train_query_ids=train.query_ids.astype(np.int32),
        train_donor_ids=train.donor_ids.astype(np.int32),
        train_weights=train.weights,
        train_distances=train.distances,
        held_rows=held_rows.astype(np.int32),
        held_query_ids=held.query_ids.astype(np.int32),
        held_donor_ids=held.donor_ids.astype(np.int32),
        held_weights=held.weights,
        held_distances=held.distances,
        neighbors=np.int32(config.neighbors),
        smoothing=np.float32(config.smoothing),
        guard_radii_m=np.asarray(config.guard_radii_m, dtype=np.float32),
        official_h_mask_used=np.bool_(False),
    )
    print(f"E113 built all-point context fold={fold}", flush=True)
    return train, held, held_rows


def standardize_geometry(
    geometry: np.ndarray, fit_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(geometry[:, 3:], dtype=np.float32)
    mean = selected[fit_ids].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = selected[fit_ids].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return ((selected - mean) / std).astype(np.float32), mean, std


def unit_amplitude(value: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(value.square().sum(dim=-1, keepdim=True).clamp_min(TINY))
    return value / norm


def power_cosine(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    predicted_power = predicted.square()
    truth_power = truth.square()
    numerator = (predicted_power * truth_power).sum(dim=-1)
    product = (
        predicted_power.square().sum(dim=-1)
        * truth_power.square().sum(dim=-1)
    )
    denominator = torch.sqrt(product.clamp_min(TINY))
    return numerator / denominator


class GroupPropagationOperator(nn.Module):
    def __init__(
        self, geometry_size: int, variant: Variant, config: TrainConfig
    ) -> None:
        super().__init__()
        self.variant = variant
        self.config = config
        if variant.use_map:
            self.query_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(geometry_size, config.hidden),
                nn.SiLU(),
                nn.LayerNorm(config.hidden),
                nn.Linear(config.hidden, config.hidden),
            )
            geometry_pair = geometry_size + config.hidden
        else:
            self.query_encoder = None
            geometry_pair = 0
        self.spectral_encoder = nn.Sequential(
            nn.Linear(COMMON_LENGTH, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.spectral_hidden),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(geometry_pair + config.spectral_hidden + 9, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
        )
        output_size = {
            "shared": 1,
            "dual": 2,
            "low_rank": config.low_rank_heads,
            "grouped": TOTAL_GROUPS,
        }[variant.head_mode]
        self.correction_head = nn.Sequential(
            nn.Linear(config.hidden * 3, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, output_size),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        if variant.head_mode == "low_rank":
            group = np.linspace(-1.0, 1.0, TOTAL_GROUPS, dtype=np.float32)
            initial = np.stack(
                [
                    np.ones(TOTAL_GROUPS, dtype=np.float32),
                    np.r_[np.ones(PAS_GROUPS), -np.ones(PDP_GROUPS)].astype(np.float32),
                    group,
                    np.cos(np.pi * group).astype(np.float32),
                ],
                axis=0,
            )[: config.low_rank_heads]
            initial /= np.maximum(
                np.linalg.norm(initial, axis=1, keepdims=True), 1e-6
            )
            self.low_rank_mixing = nn.Parameter(torch.as_tensor(initial))
        else:
            self.register_parameter("low_rank_mixing", None)

    def _expand_heads(self, raw: torch.Tensor) -> torch.Tensor:
        if self.variant.head_mode == "shared":
            return raw.expand(-1, -1, TOTAL_GROUPS)
        if self.variant.head_mode == "dual":
            return torch.cat(
                [
                    raw[:, :, :1].expand(-1, -1, PAS_GROUPS),
                    raw[:, :, 1:].expand(-1, -1, PDP_GROUPS),
                ],
                dim=2,
            )
        if self.variant.head_mode == "low_rank":
            if self.low_rank_mixing is None:
                raise RuntimeError("missing low-rank mixing matrix")
            return torch.einsum("bkr,rg->bkg", raw, self.low_rank_mixing)
        return raw

    def forward(
        self,
        query_geometry: torch.Tensor,
        query_position: torch.Tensor,
        donor_geometry: torch.Tensor,
        donor_position: torch.Tensor,
        donor_common: torch.Tensor,
        donor_pas: torch.Tensor,
        donor_pdp: torch.Tensor,
        distances: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_parts: list[torch.Tensor] = []
        if self.query_encoder is not None:
            query_latent = self.query_encoder(query_geometry)
            pair_parts.extend(
                [
                    query_latent[:, None, :].expand(-1, donor_common.shape[1], -1),
                    donor_geometry - query_geometry[:, None, :],
                ]
            )
        pair_parts.append(self.spectral_encoder(donor_common))
        position_delta = donor_position - query_position[:, None, :]
        direction = position_delta / distances[:, :, None].clamp_min(1e-4)
        rank = torch.linspace(
            0.0, 1.0, donor_common.shape[1], device=donor_common.device
        )[None, :, None].expand(len(query_position), -1, -1)
        numeric = torch.cat(
            [
                position_delta / 20.0,
                distances[:, :, None] / 20.0,
                torch.log1p(distances[:, :, None]),
                direction,
                base_weights[:, :, None].clamp(-2.0, 2.0),
                base_weights.abs()[:, :, None].clamp(0.0, 2.0),
                rank,
            ],
            dim=-1,
        )
        pair_parts.append(numeric)
        pair = self.pair_encoder(torch.cat(pair_parts, dim=-1))
        pooled_mean = pair.mean(dim=1, keepdim=True).expand_as(pair)
        pooled_anchor = (
            (base_weights[:, :, None] * pair)
            .sum(dim=1, keepdim=True)
            .expand_as(pair)
        )
        raw = self._expand_heads(
            self.correction_head(torch.cat([pair, pooled_mean, pooled_anchor], dim=-1))
        )
        raw = torch.tanh(raw)
        raw = raw - raw.mean(dim=1, keepdim=True)
        correction = self.config.correction_l1_bound / self.config.neighbors * raw
        weights = base_weights[:, :, None] + correction
        pas = unit_amplitude(
            torch.relu(
                torch.einsum(
                    "bkg,bkgl->bgl", weights[:, :, :PAS_GROUPS], donor_pas
                )
            )
        )
        pdp = unit_amplitude(
            torch.relu(
                torch.einsum(
                    "bkg,bkgl->bgl", weights[:, :, PAS_GROUPS:], donor_pdp
                )
            )
        )
        return pas, pdp, correction


def tensor_context(table: ContextTable, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "query_ids": torch.as_tensor(table.query_ids, dtype=torch.long, device=device),
        "donor_ids": torch.as_tensor(table.donor_ids, dtype=torch.long, device=device),
        "weights": torch.as_tensor(table.weights, dtype=torch.float32, device=device),
        "distances": torch.as_tensor(table.distances, dtype=torch.float32, device=device),
    }


def forward_context(
    model: GroupPropagationOperator,
    context: dict[str, torch.Tensor],
    rows: torch.Tensor,
    guard: torch.Tensor | None,
    positions: torch.Tensor,
    geometry: torch.Tensor,
    pas: torch.Tensor,
    pdp: torch.Tensor,
    common: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    prediction_pas, prediction_pdp, correction = model(
        geometry[query_ids],
        positions[query_ids, :2],
        geometry[donor_ids],
        positions[donor_ids, :2],
        common[donor_ids],
        pas[donor_ids],
        pdp[donor_ids],
        distances,
        weights,
    )
    return prediction_pas, prediction_pdp, correction, query_ids


@torch.no_grad()
def evaluate_model(
    model: GroupPropagationOperator,
    table: ContextTable,
    positions: torch.Tensor,
    geometry: torch.Tensor,
    pas: torch.Tensor,
    pdp: torch.Tensor,
    common: torch.Tensor,
    batch_size: int,
    return_predictions: bool = False,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray] | None]:
    model.eval()
    context = tensor_context(table, positions.device)
    pas_values: list[np.ndarray] = []
    pdp_values: list[np.ndarray] = []
    correction_values: list[np.ndarray] = []
    pas_predictions: list[np.ndarray] = []
    pdp_predictions: list[np.ndarray] = []
    for start in range(0, len(table.query_ids), batch_size):
        end = min(start + batch_size, len(table.query_ids))
        rows = torch.arange(start, end, device=positions.device)
        pred_pas, pred_pdp, correction, query_ids = forward_context(
            model,
            context,
            rows,
            None,
            positions,
            geometry,
            pas,
            pdp,
            common,
        )
        pas_values.append(power_cosine(pred_pas, pas[query_ids]).mean(dim=1).cpu().numpy())
        pdp_values.append(power_cosine(pred_pdp, pdp[query_ids]).mean(dim=1).cpu().numpy())
        correction_values.append(correction.abs().sum(dim=1).mean(dim=1).cpu().numpy())
        if return_predictions:
            pas_predictions.append(pred_pas.cpu().numpy().astype(np.float32))
            pdp_predictions.append(pred_pdp.cpu().numpy().astype(np.float32))
    pas_point = np.concatenate(pas_values)
    pdp_point = np.concatenate(pdp_values)
    result = {
        "pas": float(np.mean(pas_point)),
        "pdp": float(np.mean(pdp_point)),
        "combined": float(np.mean(0.5 * (pas_point + pdp_point))),
        "correction_l1": float(np.mean(np.concatenate(correction_values))),
    }
    predictions = None
    if return_predictions:
        predictions = (np.concatenate(pas_predictions), np.concatenate(pdp_predictions))
    return result, predictions


@torch.no_grad()
def evaluate_anchor(
    table: ContextTable,
    pas: torch.Tensor,
    pdp: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    device = pas.device
    pas_values: list[np.ndarray] = []
    pdp_values: list[np.ndarray] = []
    for start in range(0, len(table.query_ids), batch_size):
        end = min(start + batch_size, len(table.query_ids))
        donor_ids = torch.as_tensor(
            table.donor_ids[start:end], dtype=torch.long, device=device
        )
        weights = torch.as_tensor(
            table.weights[start:end], dtype=torch.float32, device=device
        )
        query_ids = torch.as_tensor(
            table.query_ids[start:end], dtype=torch.long, device=device
        )
        pred_pas = unit_amplitude(
            torch.relu(torch.einsum("bk,bkgl->bgl", weights, pas[donor_ids]))
        )
        pred_pdp = unit_amplitude(
            torch.relu(torch.einsum("bk,bkgl->bgl", weights, pdp[donor_ids]))
        )
        pas_values.append(power_cosine(pred_pas, pas[query_ids]).mean(dim=1).cpu().numpy())
        pdp_values.append(power_cosine(pred_pdp, pdp[query_ids]).mean(dim=1).cpu().numpy())
    pas_point = np.concatenate(pas_values)
    pdp_point = np.concatenate(pdp_values)
    return {
        "pas": float(np.mean(pas_point)),
        "pdp": float(np.mean(pdp_point)),
        "combined": float(np.mean(0.5 * (pas_point + pdp_point))),
    }


def train_one(
    variant: Variant,
    config: TrainConfig,
    train_table: ContextTable,
    held_table: ContextTable,
    positions: torch.Tensor,
    geometry: torch.Tensor,
    pas: torch.Tensor,
    pdp: torch.Tensor,
    common: torch.Tensor,
    device: torch.device,
    seed: int,
    return_predictions: bool = False,
    save_checkpoint: Path | None = None,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray] | None]:
    set_seed(seed)
    model = GroupPropagationOperator(geometry.shape[1], variant, config).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1), eta_min=config.learning_rate * 0.10
    )
    context = tensor_context(train_table, device)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_table.query_ids))
        loss_values: list[float] = []
        pas_values: list[float] = []
        pdp_values: list[float] = []
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guard = torch.as_tensor(
                rng.integers(0, len(config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            pred_pas, pred_pdp, correction, query_ids = forward_context(
                model,
                context,
                rows,
                guard,
                positions,
                geometry,
                pas,
                pdp,
                common,
            )
            pas_score = power_cosine(pred_pas, pas[query_ids]).mean(dim=1)
            pdp_score = power_cosine(pred_pdp, pdp[query_ids]).mean(dim=1)
            regularization = config.correction_regularization * (
                correction.square().sum(dim=1).mean()
                * config.neighbors
                / max(config.correction_l1_bound**2, 1e-8)
            )
            loss = 1.0 - 0.5 * (pas_score.mean() + pdp_score.mean())
            loss = loss + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))
            pas_values.append(float(pas_score.mean().detach().cpu()))
            pdp_values.append(float(pdp_score.mean().detach().cpu()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(loss_values)),
                "pas": float(np.mean(pas_values)),
                "pdp": float(np.mean(pdp_values)),
            }
        )
    held, predictions = evaluate_model(
        model,
        held_table,
        positions,
        geometry,
        pas,
        pdp,
        common,
        config.batch_size,
        return_predictions,
    )
    if save_checkpoint is not None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "variant": asdict(variant),
                "config": asdict(config),
                "seed": seed,
                "official_h_only": True,
            },
            save_checkpoint,
        )
    record: dict[str, object] = {
        "parameter_count": parameter_count,
        "history": history,
        "held": held,
    }
    del optimizer, scheduler, model, context
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, predictions


def prepare_fold_tensors(
    fit_ids: np.ndarray,
    positions_numpy: np.ndarray,
    geometry_numpy: np.ndarray,
    pas_numpy: np.ndarray,
    pdp_numpy: np.ndarray,
    common_numpy: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    geometry_standard, mean, std = standardize_geometry(geometry_numpy, fit_ids)
    return (
        torch.as_tensor(positions_numpy, dtype=torch.float32, device=device),
        torch.as_tensor(geometry_standard, dtype=torch.float32, device=device),
        torch.as_tensor(pas_numpy, dtype=torch.float32, device=device),
        torch.as_tensor(pdp_numpy, dtype=torch.float32, device=device),
        torch.as_tensor(common_numpy, dtype=torch.float32, device=device),
        mean,
        std,
    )


def run_smoke(config: TrainConfig, device: torch.device, force_contexts: bool) -> None:
    smoke_config = TrainConfig(**{**asdict(config), "epochs": 1, "batch_size": 96})
    pas, pdp, common = load_sionna_targets(next(iter(SIONNA_SOURCES.values())))
    _, positions, _, _, _ = load_positions()
    geometry = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    train, held, _ = load_or_build_contexts(0, smoke_config, force_contexts)
    train = ContextTable(
        train.query_ids[:512],
        train.donor_ids[:512],
        train.weights[:512],
        train.distances[:512],
    )
    tensors = prepare_fold_tensors(
        train.query_ids, positions, geometry, pas, pdp, common, device
    )
    positions_t, geometry_t, pas_t, pdp_t, common_t, _, _ = tensors
    for index in (1, 4):
        variant = VARIANTS[index]
        record, _ = train_one(
            variant,
            smoke_config,
            train,
            held,
            positions_t,
            geometry_t,
            pas_t,
            pdp_t,
            common_t,
            device,
            smoke_config.seed + index,
        )
        print(f"E113 smoke {variant.name}: {record['held']}", flush=True)


def run_sionna(config: TrainConfig, device: torch.device, force_contexts: bool) -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _, positions, _, _, _ = load_positions()
    geometry_numpy = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    source_scores = np.empty(
        (len(SIONNA_SOURCES), len(VARIANTS), 5), dtype=np.float64
    )
    baseline_scores = np.empty((len(SIONNA_SOURCES), 5), dtype=np.float64)
    fold_counts = np.empty(5, dtype=np.int64)
    records: dict[str, dict[str, object]] = {}

    for source_index, (source_name, prefix) in enumerate(SIONNA_SOURCES.items()):
        pas, pdp, common = load_sionna_targets(prefix)
        source_records: dict[str, object] = {}
        for fold in range(5):
            train, held, _ = load_or_build_contexts(fold, config, force_contexts)
            fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
            tensors = prepare_fold_tensors(
                fit_ids, positions, geometry_numpy, pas, pdp, common, device
            )
            positions_t, geometry_t, pas_t, pdp_t, common_t, _, _ = tensors
            baseline = evaluate_anchor(held, pas_t, pdp_t, config.batch_size)
            baseline_scores[source_index, fold] = baseline["combined"]
            fold_counts[fold] = len(held.query_ids)
            fold_records: dict[str, object] = {"anchor": baseline}
            for variant_index, variant in enumerate(VARIANTS):
                seed = config.seed + source_index * 100_000 + fold * 1009 + variant_index * 17
                record, _ = train_one(
                    variant,
                    config,
                    train,
                    held,
                    positions_t,
                    geometry_t,
                    pas_t,
                    pdp_t,
                    common_t,
                    device,
                    seed,
                )
                source_scores[source_index, variant_index, fold] = float(
                    record["held"]["combined"]
                )
                fold_records[variant.name] = record
                print(
                    f"E113 Sionna source={source_name} fold={fold} "
                    f"variant={variant.name} score={source_scores[source_index, variant_index, fold]:.6f} "
                    f"gain={source_scores[source_index, variant_index, fold] - baseline_scores[source_index, fold]:+.6f}",
                    flush=True,
                )
            source_records[str(fold)] = fold_records
            del positions_t, geometry_t, pas_t, pdp_t, common_t
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        records[source_name] = source_records
        del pas, pdp, common

    weights = fold_counts / fold_counts.sum()
    source_combined = np.sum(source_scores * weights[None, None, :], axis=2)
    baseline_combined = np.sum(baseline_scores * weights[None, :], axis=1)
    source_gain = source_combined - baseline_combined[:, None]
    robust_gain = source_gain.mean(axis=0) - ROBUSTNESS_PENALTY * source_gain.std(axis=0)
    selected_index = int(np.argmax(robust_gain))
    selected = VARIANTS[selected_index]
    selected_fold_gain = source_scores[:, selected_index] - baseline_scores
    positive_sources = int(np.sum(source_gain[:, selected_index] > 0.0))
    positive_source_folds = int(np.sum(selected_fold_gain > 0.0))
    gate_passed = bool(
        float(robust_gain[selected_index]) >= MIN_SIONNA_GAIN
        and positive_sources >= MIN_POSITIVE_SOURCES
        and positive_source_folds >= MIN_POSITIVE_SOURCE_FOLDS
    )
    ranking = []
    order = np.argsort(-robust_gain, kind="stable")
    for rank, variant_index in enumerate(order, start=1):
        ranking.append(
            {
                "rank": rank,
                "variant": asdict(VARIANTS[variant_index]),
                "source_scores": {
                    name: float(source_combined[source_index, variant_index])
                    for source_index, name in enumerate(SIONNA_SOURCES)
                },
                "source_gains": {
                    name: float(source_gain[source_index, variant_index])
                    for source_index, name in enumerate(SIONNA_SOURCES)
                },
                "mean_gain": float(source_gain[:, variant_index].mean()),
                "std_gain": float(source_gain[:, variant_index].std()),
                "robust_gain": float(robust_gain[variant_index]),
                "positive_source_folds": int(
                    np.sum(source_scores[:, variant_index] - baseline_scores > 0.0)
                ),
            }
        )
    payload = {
        "experiment": VERSION,
        "stage": "sionna_structure_selection",
        "status": "selection_locked",
        "selection_locked": True,
        "official_h_loaded": False,
        "official_h_mask_used_for_contexts": False,
        "sionna_checkpoint_saved": False,
        "primary_hypothesis": (
            "Sionna can prospectively choose whether score-sufficient PAS/PDP groups "
            "need shared, metric-specific, low-rank, or independent donor competition."
        ),
        "config": asdict(config),
        "selection_rule": {
            "score": "mean_source_gain - 0.5 * std_source_gain",
            "minimum_robust_gain": MIN_SIONNA_GAIN,
            "minimum_positive_sources": MIN_POSITIVE_SOURCES,
            "minimum_positive_source_folds": MIN_POSITIVE_SOURCE_FOLDS,
        },
        "baseline_sources": {
            name: float(baseline_combined[source_index])
            for source_index, name in enumerate(SIONNA_SOURCES)
        },
        "ranking": ranking,
        "selected_variant": asdict(selected),
        "selected_robust_gain": float(robust_gain[selected_index]),
        "selected_positive_sources": positive_sources,
        "selected_positive_source_folds": positive_source_folds,
        "sionna_gate_passed": gate_passed,
        "records": records,
        "seconds": float(time.time() - started),
    }
    np.savez_compressed(
        SIONNA_SCORES,
        variant_names=np.asarray([variant.name for variant in VARIANTS]),
        source_names=np.asarray(list(SIONNA_SOURCES)),
        source_scores=source_scores,
        baseline_scores=baseline_scores,
        source_gain=source_gain,
        robust_gain=robust_gain,
        fold_counts=fold_counts,
    )
    SIONNA_SELECTION.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "selected_variant",
        "selected_robust_gain",
        "selected_positive_sources",
        "selected_positive_source_folds",
        "sionna_gate_passed",
        "seconds",
    )}, indent=2), flush=True)


def coarse_pas_direction(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    power = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    norm = torch.sqrt(power.square().sum(dim=(1, 2, 3), keepdim=True))
    direction = torch.where(norm > 0.0, power / norm.clamp_min(TINY), torch.zeros_like(power))
    coarse = direction.mean(dim=5).permute(0, 4, 1, 2, 3).reshape(len(channel), 4, 256)
    return coarse / coarse.sum(dim=2, keepdim=True).clamp_min(TINY)


def coarse_pdp_direction(channel: torch.Tensor) -> torch.Tensor:
    shaped = torch.fft.ifft(channel, dim=-1).reshape(len(channel), 2, 16, 8, 4, 192)
    power = shaped.abs().square().float()
    norm = torch.sqrt(power.square().sum(dim=5, keepdim=True))
    direction = torch.where(norm > 0.0, power / norm.clamp_min(TINY), torch.zeros_like(power))
    coarse = direction.mean(dim=(2, 3)).reshape(len(channel), 8, 192)
    return coarse / coarse.sum(dim=2, keepdim=True).clamp_min(TINY)


def coarse_preserving_action(
    base: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
) -> torch.Tensor:
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    candidate = base
    current_pas = coarse_pas_direction(candidate)
    pas_ratio = (
        target_pas.clamp_min(1e-12) / current_pas.clamp_min(1e-12)
    ).pow(COARSE_ACTION_ALPHA)
    shaped = candidate.reshape(len(candidate), 2, 16, 8, 4, 192)
    spatial = torch.fft.fft2(shaped, dim=(2, 3))
    pas_power = spatial.abs().square().float()
    ratio_pas = pas_ratio.reshape(len(candidate), 4, 2, 16, 8).permute(0, 2, 3, 4, 1)
    desired_pas = pas_power * ratio_pas[:, :, :, :, :, None]
    desired_pas = desired_pas / desired_pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)

    current_pdp = coarse_pdp_direction(candidate)
    pdp_ratio = (
        target_pdp.clamp_min(1e-12) / current_pdp.clamp_min(1e-12)
    ).pow(COARSE_ACTION_ALPHA)
    delay = torch.fft.ifft(candidate, dim=-1).reshape(len(candidate), 2, 16, 8, 4, 192)
    pdp_power = delay.abs().square().float()
    ratio_pdp = pdp_ratio.reshape(len(candidate), 2, 4, 192)
    desired_pdp = pdp_power * ratio_pdp[:, :, None, None, :, :]
    desired_pdp = desired_pdp / desired_pdp.sum(dim=5, keepdim=True).clamp_min(TINY)
    desired_pdp = desired_pdp.reshape(len(candidate), 256, 4, 192)

    for _ in range(COARSE_ACTION_CYCLES):
        candidate = exact_pas_projection(candidate, desired_pas)
        candidate = exact_pdp_projection(candidate, desired_pdp)
        candidate = preserve_point_power(candidate, reference_power)
    return candidate


def aggregate_channel_metrics(
    channels: dict[str, np.ndarray],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    target_pas: np.ndarray,
    target_pdp: np.ndarray,
    device: torch.device,
    batch_size: int = 4,
) -> tuple[dict[str, object], np.ndarray]:
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    values = {
        name: {
            key: np.empty(len(query_ids), dtype=np.float64)
            for key in ("pas", "pdp", "error", "energy")
        }
        for name in channels
    }
    output = np.lib.format.open_memmap(
        OFFICIAL_CHANNEL,
        mode="w+",
        dtype=np.complex64,
        shape=channels["official_operator"].shape,
    )
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        with torch.inference_mode():
            truth_pas, truth_pdp = truth_views(truth)
            for name, channel in channels.items():
                prediction = torch.as_tensor(
                    np.array(channel[start:end], dtype=np.complex64, copy=True),
                    device=device,
                )
                if name == "official_operator":
                    prediction = coarse_preserving_action(
                        prediction,
                        torch.as_tensor(target_pas[start:end], dtype=torch.float32, device=device),
                        torch.as_tensor(target_pdp[start:end], dtype=torch.float32, device=device),
                    )
                    output[start:end] = prediction.cpu().numpy().astype(np.complex64)
                metrics = point_metrics(prediction, truth, truth_pas, truth_pdp)
                for key, metric in zip(("pas", "pdp", "error", "energy"), metrics):
                    values[name][key][start:end] = metric
        print(f"E113 complete-channel replay {end}/{len(query_ids)}", flush=True)
    output.flush()

    summaries: dict[str, object] = {}
    for name, row in values.items():
        overall = statistics(row["pas"], row["pdp"], row["error"], row["energy"])
        folds = [
            statistics(
                row["pas"], row["pdp"], row["error"], row["energy"], fold_id == fold
            )
            for fold in range(5)
        ]
        summaries[name] = {"overall": overall, "folds": folds}
    candidate_delta = np.asarray(
        [
            summaries["official_operator"]["folds"][fold]["score"]
            - summaries["V12"]["folds"][fold]["score"]
            for fold in range(5)
        ],
        dtype=np.float64,
    )
    return summaries, candidate_delta


def run_official(config: TrainConfig, device: torch.device, force_contexts: bool) -> None:
    started = time.time()
    if not SIONNA_SELECTION.is_file():
        raise FileNotFoundError(
            f"Run --stage sionna first; locked selection is missing: {SIONNA_SELECTION}"
        )
    selection = json.loads(SIONNA_SELECTION.read_text(encoding="utf-8"))
    if not selection.get("selection_locked"):
        raise RuntimeError("Sionna architecture selection is not locked")
    if not selection.get("sionna_gate_passed"):
        raise RuntimeError("Sionna winner did not pass the preregistered transfer gate")
    variant_payload = selection["selected_variant"]
    selected = next(
        variant for variant in VARIANTS if variant.name == variant_payload["name"]
    )

    # Official H-derived targets are intentionally loaded only after the locked
    # simulator selection above has been validated.
    pas, pdp, common = load_official_targets()
    _, positions, _, _, _ = load_positions()
    geometry_numpy = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    endpoint_pas = np.empty((len(query_ids), PAS_GROUPS, PAS_LENGTH), dtype=np.float32)
    endpoint_pdp = np.empty((len(query_ids), PDP_GROUPS, PDP_LENGTH), dtype=np.float32)
    baseline_fold: list[dict[str, float]] = []
    selected_fold: list[dict[str, float]] = []
    records: list[dict[str, object]] = []
    checkpoint_paths: list[str] = []

    for fold in range(5):
        train, held, held_rows = load_or_build_contexts(fold, config, force_contexts)
        fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
        tensors = prepare_fold_tensors(
            fit_ids, positions, geometry_numpy, pas, pdp, common, device
        )
        positions_t, geometry_t, pas_t, pdp_t, common_t, mean, std = tensors
        baseline = evaluate_anchor(held, pas_t, pdp_t, config.batch_size)
        checkpoint = ARTIFACT_DIR / f"official_fold{fold}_{selected.name}.pt"
        record, predictions = train_one(
            selected,
            config,
            train,
            held,
            positions_t,
            geometry_t,
            pas_t,
            pdp_t,
            common_t,
            device,
            config.seed + 9_000_000 + fold * 1009,
            return_predictions=True,
            save_checkpoint=checkpoint,
        )
        if predictions is None:
            raise RuntimeError("official endpoint predictions were not returned")
        endpoint_pas[held_rows] = predictions[0]
        endpoint_pdp[held_rows] = predictions[1]
        record["fold"] = fold
        record["baseline"] = baseline
        record["delta_endpoint"] = float(record["held"]["combined"] - baseline["combined"])
        record["feature_mean"] = mean.tolist()
        record["feature_std"] = std.tolist()
        record["checkpoint"] = PATHS.relative(checkpoint)
        records.append(record)
        baseline_fold.append(baseline)
        selected_fold.append(record["held"])
        checkpoint_paths.append(PATHS.relative(checkpoint))
        print(
            f"E113 official fold={fold} selected={record['held']['combined']:.6f} "
            f"baseline={baseline['combined']:.6f} delta={record['delta_endpoint']:+.6f}",
            flush=True,
        )
        del positions_t, geometry_t, pas_t, pdp_t, common_t
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_counts = np.bincount(fold_id, minlength=5).astype(np.float64)
    fold_weights = fold_counts / fold_counts.sum()
    baseline_endpoint = {
        key: float(sum(fold_weights[fold] * baseline_fold[fold][key] for fold in range(5)))
        for key in ("pas", "pdp", "combined")
    }
    selected_endpoint = {
        key: float(sum(fold_weights[fold] * selected_fold[fold][key] for fold in range(5)))
        for key in ("pas", "pdp", "combined", "correction_l1")
    }
    endpoint_gain = float(selected_endpoint["combined"] - baseline_endpoint["combined"])
    np.savez_compressed(
        OFFICIAL_ENDPOINT,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        pas=endpoint_pas.astype(np.float16),
        pdp=endpoint_pdp.astype(np.float16),
        selected_variant=np.asarray(selected.name),
        official_h_only=np.bool_(True),
    )

    channels = {
        "official_operator": np.load(OOF_V7, mmap_mode="r"),
        "V7": np.load(OOF_V7, mmap_mode="r"),
        "V12": np.load(OOF_V12, mmap_mode="r"),
    }
    complete, fold_delta = aggregate_channel_metrics(
        channels, query_ids, fold_id, endpoint_pas, endpoint_pdp, device
    )
    candidate = complete["official_operator"]["overall"]
    v12 = complete["V12"]["overall"]
    complete_gain = float(candidate["score"] - v12["score"])
    positive_folds = int(np.sum(fold_delta > 0.0))
    gate_passed = bool(
        endpoint_gain > 0.0
        and complete_gain >= MIN_COMPLETE_GAIN_VS_V12
        and positive_folds >= MIN_COMPLETE_POSITIVE_FOLDS
        and candidate["pas"] >= v12["pas"] - MAX_COMPONENT_DROP
        and candidate["pdp"] >= v12["pdp"] - MAX_COMPONENT_DROP
        and candidate["nmse"] <= v12["nmse"] + MAX_NMSE_INCREASE
    )
    finite_endpoint = bool(
        np.isfinite(endpoint_pas).all() and np.isfinite(endpoint_pdp).all()
    )
    output_map = np.load(OFFICIAL_CHANNEL, mmap_mode="r")
    finite_channel = True
    for start in range(0, len(output_map), 8):
        block = np.asarray(output_map[start : start + 8])
        finite_channel &= bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    payload = {
        "experiment": VERSION,
        "status": "official_five_fold_complete",
        "primary_hypothesis": selection["primary_hypothesis"],
        "protocol": {
            "sionna_selected_structure_only": True,
            "sionna_checkpoint_loaded": False,
            "official_weights_reinitialized": True,
            "official_h_only_final_training": True,
            "selected_variant": asdict(selected),
            "fixed_epochs": config.epochs,
            "coarse_action_alpha": COARSE_ACTION_ALPHA,
            "coarse_action_cycles": COARSE_ACTION_CYCLES,
        },
        "sionna_selection": {
            "path": PATHS.relative(SIONNA_SELECTION),
            "robust_gain": selection["selected_robust_gain"],
            "positive_source_folds": selection["selected_positive_source_folds"],
        },
        "config": asdict(config),
        "official_endpoint": {
            "anchor": baseline_endpoint,
            "selected": selected_endpoint,
            "gain": endpoint_gain,
            "folds": records,
            "path": PATHS.relative(OFFICIAL_ENDPOINT),
        },
        "complete_channel": {
            "results": complete,
            "delta_vs_V12": complete_gain,
            "fold_delta_vs_V12": fold_delta.tolist(),
            "positive_folds_vs_V12": positive_folds,
            "output": {
                "path": PATHS.relative(OFFICIAL_CHANNEL),
                "shape": list(output_map.shape),
                "dtype": str(output_map.dtype),
                "finite": finite_channel,
            },
        },
        "deployment_gate": {
            "minimum_gain_vs_V12": MIN_COMPLETE_GAIN_VS_V12,
            "minimum_positive_folds": MIN_COMPLETE_POSITIVE_FOLDS,
            "maximum_pas_or_pdp_drop": MAX_COMPONENT_DROP,
            "maximum_nmse_increase": MAX_NMSE_INCREASE,
            "passed": gate_passed,
        },
        "integrity": {
            "official_endpoint_finite": finite_endpoint,
            "official_channel_finite": finite_channel,
            "official_checkpoints": checkpoint_paths,
            "sionna_checkpoints_retained": [],
        },
        "candidate_generated": False,
        "candidate_reason": (
            "V13 requires a separate full-data fit and test replay after this gate passes."
            if gate_passed
            else "The preregistered complete-channel deployment gate did not pass."
        ),
        "seconds": float(time.time() - started),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected_variant": selected.name,
        "official_endpoint_gain": endpoint_gain,
        "candidate_complete": candidate,
        "V12": v12,
        "delta_vs_V12": complete_gain,
        "fold_delta_vs_V12": fold_delta.tolist(),
        "deployment_gate_passed": gate_passed,
        "seconds": payload["seconds"],
    }, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.stage == "smoke":
        run_smoke(config, device, args.force_contexts)
    elif args.stage == "sionna":
        run_sionna(config, device, args.force_contexts)
    else:
        run_official(config, device, args.force_contexts)


if __name__ == "__main__":
    main()
