#!/usr/bin/env python3
"""Train group-conditioned donor fields for official linkwise PAS and PDP."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from data import load_positions
from project_paths import PATHS
from run_v3_anchored_neural_rbf import (
    Config as ContextConfig,
    FOLD_CACHE,
    OOF_META,
    build_train_contexts,
    set_seed,
    tensor_context,
    validation_context,
)
from run_v3_grouped_link_refinement import build_grouped_descriptors


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_group_conditioned_link_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_group_conditioned_link_field"
REPORT = ANALYSIS_DIR / "group_conditioned_training.json"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
GEOMETRY = PRE_DIR / "round2_geometry_visibility_features.npz"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
ROUTE_FEATURES = (
    PATHS.artifacts / "round2" / "route_topology" / "round2_route_topology_features.npz"
)

PAS_GROUPS = 32
PDP_GROUPS = 8
PAS_BINS = 32
PDP_BINS = 24


@dataclass(frozen=True)
class ModelConfig:
    neighbors: int = 64
    state_hidden: int = 128
    pair_hidden: int = 128
    spectrum_hidden: int = 48
    interaction_rank: int = 48
    correction_l1_bound: float = 1.20
    correction_regularization: float = 0.004
    ranking_weight: float = 0.015
    ranking_temperature: float = 0.20
    pas_smoothness: float = 0.001
    epochs: int = 12
    batch_size: int = 24
    learning_rate: float = 5e-4
    weight_decay: float = 2e-5
    seed: int = 20_260_805


def route_state_features(split: str) -> np.ndarray:
    cache = np.load(ROUTE_FEATURES)
    hit = cache[f"{split}_route_hit"].astype(np.float32)
    occupancy = cache[f"{split}_route_occupancy"].astype(np.float32)
    count = cache[f"{split}_route_count"].astype(np.float32)
    panorama = cache[f"{split}_panorama"].astype(np.float32)
    if hit.shape[-1] != 48:
        raise ValueError(f"unexpected route bins: {hit.shape}")
    hit_pooled = hit.reshape(len(hit), 7, 12, 4).mean(axis=3)
    occupancy_pooled = occupancy.reshape(len(hit), 7, 12, 4).mean(axis=3)
    count = np.clip(count / 16.0, 0.0, 1.0)
    return np.concatenate(
        [
            hit_pooled.reshape(len(hit), -1),
            occupancy_pooled.reshape(len(hit), -1),
            count,
            panorama.reshape(len(hit), -1),
        ],
        axis=1,
    ).astype(np.float32)


def load_state_features() -> tuple[np.ndarray, np.ndarray]:
    geometry = np.load(GEOMETRY)
    train_geometry = geometry["train"].astype(np.float32)[:, 3:]
    test_geometry = geometry["test"].astype(np.float32)[:, 3:]
    train = np.concatenate([train_geometry, route_state_features("train")], axis=1)
    test = np.concatenate([test_geometry, route_state_features("test")], axis=1)
    return train.astype(np.float32), test.astype(np.float32)


def standardize_state(
    train: np.ndarray,
    test: np.ndarray,
    fit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return (
        ((train - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
        mean,
        std,
    )


def compact_group_features(amplitude: np.ndarray, bins: int) -> np.ndarray:
    count, groups, length = amplitude.shape
    if length % bins != 0:
        raise ValueError((amplitude.shape, bins))
    output = np.empty((count, groups, bins + 6), dtype=np.float32)
    axis = np.linspace(0.0, 1.0, length, dtype=np.float32)
    for start in range(0, count, 128):
        end = min(start + 128, count)
        values = np.asarray(amplitude[start:end], dtype=np.float32)
        power = np.square(np.maximum(values, 0.0))
        energy = power.sum(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
        normalized = np.divide(
            power,
            np.maximum(energy, 1e-20),
            out=np.zeros_like(power),
            where=energy > 0.0,
        )
        pooled = normalized.reshape(end - start, groups, bins, length // bins).sum(axis=3)
        centroid = np.sum(normalized * axis[None, None, :], axis=2)
        spread = np.sqrt(
            np.sum(
                normalized * np.square(axis[None, None, :] - centroid[:, :, None]),
                axis=2,
            )
        )
        peak = np.argmax(normalized, axis=2).astype(np.float32) / max(length - 1, 1)
        peak_value = np.max(normalized, axis=2)
        entropy = -np.sum(
            normalized * np.log(np.maximum(normalized, 1e-20)), axis=2
        ) / math.log(length)
        log_energy = np.clip(np.log10(np.maximum(energy[:, :, 0], 1e-20)), -12.0, 12.0) / 12.0
        output[start:end] = np.concatenate(
            [
                pooled,
                centroid[:, :, None],
                spread[:, :, None],
                peak[:, :, None],
                peak_value[:, :, None],
                entropy[:, :, None],
                log_energy[:, :, None],
            ],
            axis=2,
        )
    return output


def unit_amplitude(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(value.square().sum(dim=-1, keepdim=True).clamp_min(1e-20))


def stable_power_cosine(
    predicted_amplitude: torch.Tensor,
    truth_amplitude: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted_amplitude.square()
    truth = truth_amplitude.square()
    numerator = torch.sum(predicted * truth, dim=-1)
    product = (
        torch.sum(predicted.square(), dim=-1)
        * torch.sum(truth.square(), dim=-1)
    )
    return numerator / torch.sqrt(product.clamp_min(1e-20))


def compact_group_features_torch(amplitude: torch.Tensor, bins: int) -> torch.Tensor:
    length = amplitude.shape[-1]
    power = amplitude.square().clamp_min(0.0)
    energy = power.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        energy > 0.0,
        power / energy.clamp_min(1e-20),
        torch.zeros_like(power),
    )
    pooled = normalized.reshape(*normalized.shape[:-1], bins, length // bins).sum(dim=-1)
    axis = torch.linspace(0.0, 1.0, length, device=amplitude.device)
    centroid = torch.sum(normalized * axis, dim=-1)
    spread = torch.sqrt(
        torch.sum(normalized * (axis - centroid[..., None]).square(), dim=-1).clamp_min(1e-12)
    )
    peak_value, peak_index = torch.max(normalized, dim=-1)
    peak = peak_index.to(amplitude.dtype) / max(length - 1, 1)
    entropy = -torch.sum(
        normalized * torch.log(normalized.clamp_min(1e-20)), dim=-1
    ) / math.log(length)
    log_energy = torch.log10(energy[..., 0].clamp_min(1e-20)).clamp(-12.0, 12.0) / 12.0
    return torch.cat(
        [
            pooled,
            centroid[..., None],
            spread[..., None],
            peak[..., None],
            peak_value[..., None],
            entropy[..., None],
            log_energy[..., None],
        ],
        dim=-1,
    )


class MetricGroupScorer(nn.Module):
    def __init__(
        self,
        groups: int,
        feature_size: int,
        pair_hidden: int,
        query_hidden: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        hidden = config.spectrum_hidden
        rank = config.interaction_rank
        self.group_embedding = nn.Embedding(groups, hidden)
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(feature_size, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.query_context = nn.Sequential(
            nn.Linear(query_hidden + hidden * 2, rank),
            nn.SiLU(),
            nn.LayerNorm(rank),
        )
        self.donor_context = nn.Sequential(
            nn.Linear(pair_hidden + hidden * 3 + 5, rank),
            nn.SiLU(),
            nn.LayerNorm(rank),
        )
        self.score_head = nn.Linear(rank, 1)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(
        self,
        query_latent: torch.Tensor,
        pair_latent: torch.Tensor,
        donor_features: torch.Tensor,
        anchor_features: torch.Tensor,
        anchor_similarity: torch.Tensor,
        base_weights: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        batch, donors, groups, _ = donor_features.shape
        group = self.group_embedding.weight[None].expand(batch, -1, -1)
        anchor = self.spectrum_encoder(anchor_features)
        query = self.query_context(torch.cat([query_latent[:, None, :].expand(-1, groups, -1), anchor, group], dim=-1))
        donor = self.spectrum_encoder(donor_features)
        group_expanded = group[:, None].expand(-1, donors, -1, -1)
        anchor_expanded = anchor[:, None].expand(-1, donors, -1, -1)
        pair_expanded = pair_latent[:, :, None, :].expand(-1, -1, groups, -1)
        relation = torch.stack(
            [
                anchor_similarity,
                (donor_features[..., -6] - anchor_features[:, None, :, -6]).abs(),
                (donor_features[..., -4] - anchor_features[:, None, :, -4]).abs(),
                base_weights[:, :, None].expand(-1, -1, groups),
                (distances[:, :, None] / 20.0).expand(-1, -1, groups),
            ],
            dim=-1,
        )
        donor_context = self.donor_context(
            torch.cat(
                [pair_expanded, donor, anchor_expanded, group_expanded, relation],
                dim=-1,
            )
        )
        interaction = donor_context * query[:, None]
        return self.score_head(interaction).squeeze(-1).permute(0, 2, 1)


class GroupConditionedDonorField(nn.Module):
    def __init__(self, state_size: int, pas_feature_size: int, pdp_feature_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query_encoder = nn.Sequential(
            nn.Linear(state_size, config.state_hidden),
            nn.SiLU(),
            nn.LayerNorm(config.state_hidden),
            nn.Linear(config.state_hidden, config.state_hidden),
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(state_size, config.state_hidden // 2),
            nn.SiLU(),
            nn.LayerNorm(config.state_hidden // 2),
        )
        self.numeric_encoder = nn.Sequential(
            nn.Linear(9, config.state_hidden // 2),
            nn.SiLU(),
            nn.LayerNorm(config.state_hidden // 2),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(config.state_hidden * 2, config.pair_hidden),
            nn.SiLU(),
            nn.LayerNorm(config.pair_hidden),
            nn.Linear(config.pair_hidden, config.pair_hidden),
        )
        self.pas_scorer = MetricGroupScorer(
            PAS_GROUPS,
            pas_feature_size,
            config.pair_hidden,
            config.state_hidden,
            config,
        )
        self.pdp_scorer = MetricGroupScorer(
            PDP_GROUPS,
            pdp_feature_size,
            config.pair_hidden,
            config.state_hidden,
            config,
        )

    def pair_latents(
        self,
        query_state: torch.Tensor,
        query_position: torch.Tensor,
        donor_state: torch.Tensor,
        donor_position: torch.Tensor,
        distances: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_encoder(query_state)
        delta = self.delta_encoder(donor_state - query_state[:, None, :])
        position_delta = donor_position - query_position[:, None, :]
        direction = position_delta / distances[:, :, None].clamp_min(1e-4)
        rank = torch.linspace(0.0, 1.0, donor_state.shape[1], device=donor_state.device)
        rank = rank[None, :, None].expand(len(query_state), -1, -1)
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
        numeric_latent = self.numeric_encoder(numeric)
        query_expanded = query[:, None, :].expand(-1, donor_state.shape[1], -1)
        pair = self.pair_encoder(torch.cat([query_expanded, delta, numeric_latent], dim=-1))
        return query, pair

    def metric_weights(
        self,
        scorer: MetricGroupScorer,
        bins: int,
        query_latent: torch.Tensor,
        pair_latent: torch.Tensor,
        donor_amplitude: torch.Tensor,
        donor_features: torch.Tensor,
        base_weights: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = unit_amplitude(
            torch.relu(torch.einsum("bk,bkgd->bgd", base_weights, donor_amplitude))
        )
        anchor_features = compact_group_features_torch(anchor, bins)
        donor_power = donor_amplitude.square()
        anchor_power = anchor.square()
        numerator = torch.sum(donor_power * anchor_power[:, None], dim=-1)
        product = (
            torch.sum(donor_power.square(), dim=-1)
            * torch.sum(anchor_power.square(), dim=-1)[:, None]
        )
        similarity = numerator / torch.sqrt(product.clamp_min(1e-20))
        logits = scorer(
            query_latent,
            pair_latent,
            donor_features,
            anchor_features,
            similarity,
            base_weights,
            distances,
        )
        raw = torch.tanh(logits)
        raw = raw - raw.mean(dim=2, keepdim=True)
        raw = raw - raw.mean(dim=1, keepdim=True)
        raw = raw - raw.mean(dim=2, keepdim=True)
        correction = self.config.correction_l1_bound / self.config.neighbors * raw
        weights = base_weights[:, None, :] + correction
        prediction = unit_amplitude(
            torch.relu(torch.einsum("bgk,bkgd->bgd", weights, donor_amplitude))
        )
        return prediction, weights, logits

    def forward(
        self,
        query_state: torch.Tensor,
        query_position: torch.Tensor,
        donor_state: torch.Tensor,
        donor_position: torch.Tensor,
        distances: torch.Tensor,
        base_weights: torch.Tensor,
        donor_pas_amplitude: torch.Tensor,
        donor_pdp_amplitude: torch.Tensor,
        donor_pas_features: torch.Tensor,
        donor_pdp_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query, pair = self.pair_latents(
            query_state,
            query_position,
            donor_state,
            donor_position,
            distances,
            base_weights,
        )
        pas, pas_weights, pas_logits = self.metric_weights(
            self.pas_scorer,
            PAS_BINS,
            query,
            pair,
            donor_pas_amplitude,
            donor_pas_features,
            base_weights,
            distances,
        )
        pdp, pdp_weights, pdp_logits = self.metric_weights(
            self.pdp_scorer,
            PDP_BINS,
            query,
            pair,
            donor_pdp_amplitude,
            donor_pdp_features,
            base_weights,
            distances,
        )
        return {
            "pas": pas,
            "pdp": pdp,
            "pas_weights": pas_weights,
            "pdp_weights": pdp_weights,
            "pas_logits": pas_logits,
            "pdp_logits": pdp_logits,
        }


def selected_context(
    context: dict[str, torch.Tensor],
    rows: torch.Tensor,
    guard: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_ids = context["query_ids"][rows]
    if context["donor_ids"].ndim == 3:
        if guard is None:
            raise ValueError("guard is required for training contexts")
        donor_ids = context["donor_ids"][rows, guard]
        weights = context["weights"][rows, guard]
        distances = context["distances"][rows, guard]
    else:
        donor_ids = context["donor_ids"][rows]
        weights = context["weights"][rows]
        distances = context["distances"][rows]
    return query_ids, donor_ids, weights, distances


def forward_context(
    model: GroupConditionedDonorField,
    context: dict[str, torch.Tensor],
    rows: torch.Tensor,
    guard: torch.Tensor | None,
    state: torch.Tensor,
    positions: torch.Tensor,
    pas_amplitude: torch.Tensor,
    pdp_amplitude: torch.Tensor,
    pas_features: torch.Tensor,
    pdp_features: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    query_ids, donor_ids, weights, distances = selected_context(context, rows, guard)
    output = model(
        state[query_ids],
        positions[query_ids, :2],
        state[donor_ids],
        positions[donor_ids, :2],
        distances,
        weights,
        pas_amplitude[donor_ids],
        pdp_amplitude[donor_ids],
        pas_features[donor_ids],
        pdp_features[donor_ids],
    )
    return output, query_ids, donor_ids


def donor_similarity(
    donor_amplitude: torch.Tensor,
    truth_amplitude: torch.Tensor,
) -> torch.Tensor:
    donor_power = donor_amplitude.square()
    truth_power = truth_amplitude.square()[:, None]
    numerator = torch.sum(donor_power * truth_power, dim=-1)
    product = (
        torch.sum(donor_power.square(), dim=-1)
        * torch.sum(truth_power.square(), dim=-1)
    )
    return (numerator / torch.sqrt(product.clamp_min(1e-20))).permute(0, 2, 1)


def training_loss(
    output: dict[str, torch.Tensor],
    donor_ids: torch.Tensor,
    query_ids: torch.Tensor,
    pas_amplitude: torch.Tensor,
    pdp_amplitude: torch.Tensor,
    config: ModelConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    truth_pas = pas_amplitude[query_ids]
    truth_pdp = pdp_amplitude[query_ids]
    pas_cosine = stable_power_cosine(output["pas"], truth_pas)
    pdp_cosine = stable_power_cosine(output["pdp"], truth_pdp)
    pas_similarity = donor_similarity(pas_amplitude[donor_ids], truth_pas)
    pdp_similarity = donor_similarity(pdp_amplitude[donor_ids], truth_pdp)
    pas_label = torch.argmax(pas_similarity, dim=2)
    pdp_label = torch.argmax(pdp_similarity, dim=2)
    pas_rank = functional.cross_entropy(
        (output["pas_logits"] / config.ranking_temperature).reshape(-1, config.neighbors),
        pas_label.reshape(-1),
    )
    pdp_rank = functional.cross_entropy(
        (output["pdp_logits"] / config.ranking_temperature).reshape(-1, config.neighbors),
        pdp_label.reshape(-1),
    )
    pas_correction = output["pas_weights"] - output["pas_weights"].mean(dim=1, keepdim=True)
    pdp_correction = output["pdp_weights"] - output["pdp_weights"].mean(dim=1, keepdim=True)
    regularization = config.correction_regularization * 0.5 * (
        pas_correction.square().mean() + pdp_correction.square().mean()
    ) * config.neighbors**2 / max(config.correction_l1_bound**2, 1e-8)
    pas_grid = output["pas_weights"].reshape(-1, 4, 8, config.neighbors)
    smoothness = config.pas_smoothness * (
        pas_grid[:, :, 1:] - pas_grid[:, :, :-1]
    ).square().mean() * config.neighbors
    metric_loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean())
    ranking = config.ranking_weight * 0.5 * (pas_rank + pdp_rank)
    loss = metric_loss + ranking + regularization + smoothness
    diagnostics = {
        "loss": float(loss.detach().cpu()),
        "pas": float(pas_cosine.mean().detach().cpu()),
        "pdp": float(pdp_cosine.mean().detach().cpu()),
        "pas_rank": float(pas_rank.detach().cpu()),
        "pdp_rank": float(pdp_rank.detach().cpu()),
        "pas_l1": float(
            (output["pas_weights"] - output["pas_weights"].mean(dim=1, keepdim=True))
            .abs()
            .sum(dim=2)
            .mean()
            .detach()
            .cpu()
        ),
        "pdp_l1": float(
            (output["pdp_weights"] - output["pdp_weights"].mean(dim=1, keepdim=True))
            .abs()
            .sum(dim=2)
            .mean()
            .detach()
            .cpu()
        ),
    }
    return loss, diagnostics


@torch.no_grad()
def evaluate(
    model: GroupConditionedDonorField,
    table,
    state: torch.Tensor,
    positions: torch.Tensor,
    pas_amplitude: torch.Tensor,
    pdp_amplitude: torch.Tensor,
    pas_features: torch.Tensor,
    pdp_features: torch.Tensor,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    context = tensor_context(table, state.device)
    pas_values: list[np.ndarray] = []
    pdp_values: list[np.ndarray] = []
    pas_l1: list[np.ndarray] = []
    pdp_l1: list[np.ndarray] = []
    for start in range(0, len(table.query_ids), batch_size):
        end = min(start + batch_size, len(table.query_ids))
        rows = torch.arange(start, end, device=state.device)
        output, query_ids, _ = forward_context(
            model,
            context,
            rows,
            None,
            state,
            positions,
            pas_amplitude,
            pdp_amplitude,
            pas_features,
            pdp_features,
        )
        pas_values.append(stable_power_cosine(output["pas"], pas_amplitude[query_ids]).cpu().numpy())
        pdp_values.append(stable_power_cosine(output["pdp"], pdp_amplitude[query_ids]).cpu().numpy())
        pas_l1.append(
            (output["pas_weights"] - output["pas_weights"].mean(dim=1, keepdim=True))
            .abs()
            .sum(dim=2)
            .cpu()
            .numpy()
        )
        pdp_l1.append(
            (output["pdp_weights"] - output["pdp_weights"].mean(dim=1, keepdim=True))
            .abs()
            .sum(dim=2)
            .cpu()
            .numpy()
        )
    pas = np.concatenate(pas_values)
    pdp = np.concatenate(pdp_values)
    return {
        "pas": float(np.mean(pas)),
        "pdp": float(np.mean(pdp)),
        "mean": float(0.5 * (np.mean(pas) + np.mean(pdp))),
        "pas_fold_points": int(len(pas)),
        "pas_group_l1": float(np.mean(np.concatenate(pas_l1))),
        "pdp_group_l1": float(np.mean(np.concatenate(pdp_l1))),
    }


def checkpoint_path(fold: int, epoch: int) -> Path:
    return ARTIFACT_DIR / f"fold{fold}_epoch{epoch}.pt"


def train_fold(
    fold: int,
    config: ModelConfig,
    context_config: ContextConfig,
    device: torch.device,
    train_state_raw: np.ndarray,
    test_state_raw: np.ndarray,
    pas_amplitude: torch.Tensor,
    pdp_amplitude: torch.Tensor,
    pas_features: torch.Tensor,
    pdp_features: torch.Tensor,
) -> dict[str, object]:
    set_seed(config.seed + fold * 1009)
    _, positions_numpy, _, site, _ = load_positions()
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    folds = np.load(FOLD_CACHE)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    metadata = np.load(OOF_META)
    oof_query_ids = metadata["query_ids"].astype(np.int64)
    oof_fold_id = metadata["fold_id"].astype(np.int8)
    train_table = build_train_contexts(
        fold,
        positions_numpy,
        site,
        valid,
        fit_ids,
        context_config,
    )
    validation_table = validation_context(fold, oof_query_ids, oof_fold_id)
    train_state, _, feature_mean, feature_std = standardize_state(
        train_state_raw, test_state_raw, fit_ids
    )
    state = torch.as_tensor(train_state, dtype=torch.float32, device=device)
    positions = torch.as_tensor(positions_numpy, dtype=torch.float32, device=device)
    train_context = tensor_context(train_table, device)
    model = GroupConditionedDonorField(
        state.shape[1], pas_features.shape[2], pdp_features.shape[2], config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed + fold * 1009)
    baseline = evaluate(
        model,
        validation_table,
        state,
        positions,
        pas_amplitude,
        pdp_amplitude,
        pas_features,
        pdp_features,
        config.batch_size,
    )
    history: list[dict[str, object]] = [{"epoch": 0, "validation": baseline}]
    best_score = baseline["mean"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "epoch": 0,
            "fold": fold,
        },
        checkpoint_path(fold, 0),
    )
    print(f"group-field fold={fold} epoch=0 validation={baseline}", flush=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_table.query_ids))
        train_rows: list[dict[str, float]] = []
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guard = torch.as_tensor(
                rng.integers(0, len(context_config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            output, query_ids, donor_ids = forward_context(
                model,
                train_context,
                rows,
                guard,
                state,
                positions,
                pas_amplitude,
                pdp_amplitude,
                pas_features,
                pdp_features,
            )
            loss, diagnostics = training_loss(
                output,
                donor_ids,
                query_ids,
                pas_amplitude,
                pdp_amplitude,
                config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            train_rows.append(diagnostics)
        validation = evaluate(
            model,
            validation_table,
            state,
            positions,
            pas_amplitude,
            pdp_amplitude,
            pas_features,
            pdp_features,
            config.batch_size,
        )
        summary = {
            key: float(np.mean([row[key] for row in train_rows]))
            for key in train_rows[0]
        }
        history.append({"epoch": epoch, "train": summary, "validation": validation})
        if validation["mean"] > best_score:
            best_score = validation["mean"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "epoch": epoch,
                "fold": fold,
            },
            checkpoint_path(fold, epoch),
        )
        print(
            f"group-field fold={fold} epoch={epoch}/{config.epochs} "
            f"train={summary['pas']:.6f}/{summary['pdp']:.6f} "
            f"valid={validation['pas']:.6f}/{validation['pdp']:.6f}/"
            f"{validation['mean']:.6f} best={best_score:.6f}@{best_epoch}",
            flush=True,
        )

    best_path = ARTIFACT_DIR / f"fold{fold}_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": asdict(config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "epoch": best_epoch,
            "fold": fold,
        },
        best_path,
    )
    return {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "train_valid_count": int(len(train_table.query_ids)),
        "validation_count": int(len(validation_table.query_ids)),
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "delta_vs_base": float(best_score - baseline["mean"]),
        "history": history,
        "best_checkpoint": PATHS.relative(best_path),
    }


def summarize_epochs(records: list[dict[str, object]], epochs: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for epoch in range(epochs + 1):
        pas = [record["history"][epoch]["validation"]["pas"] for record in records]
        pdp = [record["history"][epoch]["validation"]["pdp"] for record in records]
        mean = [record["history"][epoch]["validation"]["mean"] for record in records]
        base = [record["baseline"]["mean"] for record in records]
        output.append(
            {
                "epoch": epoch,
                "pas": float(np.mean(pas)),
                "pdp": float(np.mean(pdp)),
                "mean": float(np.mean(mean)),
                "fold_delta": [float(value - anchor) for value, anchor in zip(mean, base)],
                "positive_folds": int(sum(value > anchor for value, anchor in zip(mean, base))),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", default="0")
    parser.add_argument("--epochs", type=int, default=ModelConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=ModelConfig.batch_size)
    parser.add_argument("--correction-l1-bound", type=float, default=ModelConfig.correction_l1_bound)
    parser.add_argument("--ranking-weight", type=float, default=ModelConfig.ranking_weight)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (GEOMETRY, DESCRIPTORS, ROUTE_FEATURES, FOLD_CACHE, OOF_META):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = ModelConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        correction_l1_bound=args.correction_l1_bound,
        ranking_weight=args.ranking_weight,
    )
    context_config = ContextConfig()
    device = torch.device(args.device)
    train_state, test_state = load_state_features()
    pas_numpy, pdp_numpy = build_grouped_descriptors()
    pas_float = np.asarray(pas_numpy, dtype=np.float32)
    pdp_float = np.asarray(pdp_numpy, dtype=np.float32)
    pas_compact = compact_group_features(pas_float, PAS_BINS)
    pdp_compact = compact_group_features(pdp_float, PDP_BINS)
    pas_amplitude = torch.as_tensor(pas_float, dtype=torch.float32, device=device)
    pdp_amplitude = torch.as_tensor(pdp_float, dtype=torch.float32, device=device)
    pas_features = torch.as_tensor(pas_compact, dtype=torch.float32, device=device)
    pdp_features = torch.as_tensor(pdp_compact, dtype=torch.float32, device=device)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    records = [
        train_fold(
            fold,
            config,
            context_config,
            device,
            train_state,
            test_state,
            pas_amplitude,
            pdp_amplitude,
            pas_features,
            pdp_features,
        )
        for fold in folds
    ]
    epoch_summary = summarize_epochs(records, config.epochs)
    payload = {
        "experiment": "Round2-E13-group-conditioned-link-field",
        "status": "analysis_only",
        "primary_hypothesis": (
            "UE/frequency and TX-polarization/UE groups require distinct donor mixtures; "
            "bounded group-specific corrections can realize the E12 ceiling while preserving "
            "the reliable V2 point-global anchor."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": "fixed Round 2 spatial-hole folds with fit-only guarded contexts",
        "test_test_usage": (
            "No label or spectral smoothing; route topology is an observable query/donor "
            "condition. Test-test consistency is deferred until the grouped field passes OOF."
        ),
        "device": str(device),
        "config": asdict(config),
        "context_config": asdict(context_config),
        "state_feature_size": int(train_state.shape[1]),
        "pas_feature_size": int(pas_compact.shape[2]),
        "pdp_feature_size": int(pdp_compact.shape[2]),
        "folds": records,
        "epoch_summary": epoch_summary,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
