"""Neural ranking layer over self-built physical PathTokens."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import (
    DELAY_BINS,
    HORIZONTAL_BINS,
    PhysicalPriorConfig,
    VERTICAL_BINS,
)
from .geometry import GeometryData


class PhysicalPriorLayer(nn.Module):
    """Score candidate propagation supports without generating a channel."""

    def __init__(
        self,
        candidate_dim: int,
        sequence_dim: int,
        point_dim: int,
        config: PhysicalPriorConfig,
    ) -> None:
        super().__init__()
        hidden = config.rank_hidden
        sequence_hidden = config.rank_sequence_hidden
        self.interaction_embedding = nn.Embedding(5, 16)
        self.sequence_input = nn.Sequential(
            nn.Linear(sequence_dim + 16, sequence_hidden),
            nn.GELU(),
            nn.LayerNorm(sequence_hidden),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=sequence_hidden,
            nhead=4,
            dim_feedforward=sequence_hidden * 3,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.candidate_input = nn.Sequential(
            nn.Linear(candidate_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.point_input = nn.Sequential(
            nn.Linear(point_dim, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden + sequence_hidden + hidden // 2, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate = self.candidate_input(batch["candidate"])
        point = self.point_input(batch["point"])
        interaction = self.interaction_embedding(
            batch["interaction"].clamp(0, 4)
        )
        sequence = self.sequence_input(
            torch.cat((batch["sequence"], interaction), dim=-1)
        )
        batch_size, candidate_count, steps, hidden = sequence.shape
        flat_sequence = sequence.reshape(batch_size * candidate_count, steps, hidden)
        flat_mask = batch["sequence_mask"].reshape(
            batch_size * candidate_count, steps
        )
        encoded = self.sequence_encoder(
            flat_sequence, src_key_padding_mask=~flat_mask
        )
        weight = flat_mask.to(encoded.dtype)
        pooled = (encoded * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pooled = pooled.reshape(batch_size, candidate_count, hidden)
        point = point[:, None].expand(-1, candidate_count, -1)
        logits = self.output(torch.cat((candidate, pooled, point), dim=-1))[..., 0]
        return logits.masked_fill(~batch["candidate_mask"], -20.0)


class FeatureBuilder:
    """Build bounded model inputs lazily so the large token bank stays compact."""

    def __init__(
        self,
        tokens: dict[str, np.ndarray],
        positions: np.ndarray,
        site: np.ndarray,
        bs_positions: np.ndarray,
        geometry: GeometryData,
        coordinate: np.ndarray,
        device: torch.device,
    ) -> None:
        self.tokens = tokens
        self.positions = np.asarray(positions, dtype=np.float32)
        self.site = np.asarray(site, dtype=np.int64)
        self.bs_positions = np.asarray(bs_positions, dtype=np.float32)
        self.geometry = geometry
        self.coordinate = np.asarray(coordinate, dtype=np.float32)
        self.device = device
        self.candidate_dim = 27
        self.sequence_dim = 18
        self.point_dim = 10

    def _candidate(self, rows: np.ndarray) -> np.ndarray:
        token = self.tokens
        theta_t = token["theta_t"][rows]
        phi_t = token["phi_t"][rows]
        theta_r = token["theta_r"][rows]
        phi_r = token["phi_r"][rows]
        coordinate = self.coordinate[rows]
        path_length = token["path_length"][rows]
        position = self.positions[rows]
        bs = self.bs_positions[self.site[rows]]
        direct = np.linalg.norm(position - bs, axis=1).astype(np.float32)
        source = token["source"][rows]
        interaction = token["interactions"][rows]
        count = np.maximum(token["interaction_count"][rows], 1)
        reflection = np.sum(interaction == 1, axis=2, dtype=np.int16)
        scatter = np.sum(interaction == 2, axis=2, dtype=np.int16)
        transmission = np.sum(interaction == 4, axis=2, dtype=np.int16)
        features = np.stack(
            (
                np.sin(theta_t),
                np.cos(theta_t),
                np.sin(phi_t),
                np.cos(phi_t),
                np.sin(theta_r),
                np.cos(theta_r),
                np.sin(phi_r),
                np.cos(phi_r),
                np.sin(2.0 * np.pi * coordinate[..., 0] / HORIZONTAL_BINS),
                np.cos(2.0 * np.pi * coordinate[..., 0] / HORIZONTAL_BINS),
                np.sin(2.0 * np.pi * coordinate[..., 1] / VERTICAL_BINS),
                np.cos(2.0 * np.pi * coordinate[..., 1] / VERTICAL_BINS),
                np.sin(2.0 * np.pi * coordinate[..., 2] / DELAY_BINS),
                np.cos(2.0 * np.pi * coordinate[..., 2] / DELAY_BINS),
                np.log1p(path_length) / 7.0,
                (path_length - direct[:, None]) / 200.0,
                path_length / np.maximum(direct[:, None], 1.0),
                np.log1p(np.maximum(token["relative_energy"][rows], 0.0) * 1e6)
                / 14.0,
                token["geometry_confidence"][rows],
                token["presence"][rows].astype(np.float32) / 3.0,
                token["interaction_count"][rows].astype(np.float32) / 16.0,
                reflection.astype(np.float32) / count,
                scatter.astype(np.float32) / count,
                transmission.astype(np.float32) / count,
                (source == 0).astype(np.float32),
                (source == 1).astype(np.float32),
                np.broadcast_to(
                    self.site[rows, None].astype(np.float32), path_length.shape
                ),
            ),
            axis=2,
        ).astype(np.float32)
        features[~token["mask"][rows]] = 0.0
        return features

    def _sequence(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        token = self.tokens
        family = token["surface_family"][rows]
        safe = np.maximum(family, 0)
        normal = self.geometry.family_normal[safe]
        centroid = self.geometry.family_centroid[safe]
        area = self.geometry.family_area[safe]
        vertices = token["vertices"][rows]
        position = self.positions[rows, None, None]
        bs = self.bs_positions[self.site[rows], None, None]
        relative_bs = (vertices - bs) / 200.0
        relative_ue = (vertices - position) / 200.0
        family_relative_bs = (centroid - bs) / 200.0
        sequence = np.concatenate(
            (
                relative_bs,
                relative_ue,
                normal,
                family_relative_bs,
                (np.log1p(area) / 14.0)[..., None],
                token["incidence"][rows][..., None],
                (vertices[..., 2:3] / 50.0),
                (centroid[..., 2:3] / 50.0),
                (
                    np.linalg.norm(vertices - bs, axis=3, keepdims=True) / 400.0
                ),
                (
                    np.linalg.norm(vertices - position, axis=3, keepdims=True)
                    / 400.0
                ),
            ),
            axis=3,
        ).astype(np.float32)
        count = token["interaction_count"][rows].astype(np.int64)
        steps = np.arange(sequence.shape[2])[None, None]
        mask = steps < count[..., None]
        direct = count == 0
        direct_row, direct_candidate = np.nonzero(direct)
        mask[direct_row, direct_candidate, 0] = True
        sequence[~mask] = 0.0
        interaction = token["interactions"][rows].astype(np.int64)
        interaction[~mask] = 0
        return sequence, interaction, mask

    def _point(self, rows: np.ndarray) -> np.ndarray:
        position = self.positions[rows]
        bs = self.bs_positions[self.site[rows]]
        delta = position - bs
        distance = np.linalg.norm(delta, axis=1)
        azimuth = np.arctan2(delta[:, 1], delta[:, 0])
        elevation = np.arctan2(delta[:, 2], np.linalg.norm(delta[:, :2], axis=1))
        return np.stack(
            (
                position[:, 0] / 250.0,
                position[:, 1] / 300.0,
                delta[:, 0] / 250.0,
                delta[:, 1] / 300.0,
                delta[:, 2] / 30.0,
                distance / 400.0,
                np.sin(azimuth),
                np.cos(azimuth),
                np.sin(elevation),
                self.site[rows].astype(np.float32),
            ),
            axis=1,
        ).astype(np.float32)

    def batch(
        self, rows: np.ndarray, candidate_ids: np.ndarray | None = None
    ) -> dict[str, torch.Tensor]:
        rows = np.asarray(rows, dtype=np.int64)
        sequence, interaction, sequence_mask = self._sequence(rows)
        candidate = self._candidate(rows)
        candidate_mask = self.tokens["mask"][rows]
        if candidate_ids is not None:
            candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
            if candidate_ids.shape[0] != len(rows):
                raise ValueError("candidate_ids batch dimension mismatch")
            batch = np.arange(len(rows), dtype=np.int64)[:, None]
            candidate = candidate[batch, candidate_ids]
            sequence = sequence[batch, candidate_ids]
            interaction = interaction[batch, candidate_ids]
            sequence_mask = sequence_mask[batch, candidate_ids]
            candidate_mask = candidate_mask[batch, candidate_ids]
        return {
            "candidate": torch.as_tensor(
                candidate, device=self.device
            ),
            "sequence": torch.as_tensor(sequence, device=self.device),
            "interaction": torch.as_tensor(interaction, device=self.device),
            "sequence_mask": torch.as_tensor(sequence_mask, device=self.device),
            "point": torch.as_tensor(self._point(rows), device=self.device),
            "candidate_mask": torch.as_tensor(
                candidate_mask, device=self.device
            ),
        }


def select_candidates(
    scores: np.ndarray,
    mask: np.ndarray,
    source: np.ndarray,
    coordinate: np.ndarray,
    config: PhysicalPriorConfig,
) -> np.ndarray:
    """Greedy top-128 selection with the historical 48-base reservation."""

    valid = np.flatnonzero(mask)
    selected: list[int] = []
    base = valid[source[valid] == 0]
    base = base[np.argsort(-scores[base])]
    selected.extend(int(value) for value in base[: config.base_reserve])
    order = valid[np.argsort(-scores[valid])]
    deferred: list[int] = []
    periods = np.asarray(
        (HORIZONTAL_BINS, VERTICAL_BINS, DELAY_BINS), dtype=np.float32
    )
    scales = np.asarray(
        (config.horizontal_scale, config.vertical_scale, config.delay_scale),
        dtype=np.float32,
    )
    for candidate in order:
        candidate = int(candidate)
        if candidate in selected:
            continue
        if selected:
            delta = np.remainder(
                coordinate[selected] - coordinate[candidate] + periods / 2.0,
                periods,
            ) - periods / 2.0
            cost = np.sum(np.square(delta / scales), axis=1)
            if float(np.min(cost)) < config.rank_diversity_cost:
                deferred.append(candidate)
                continue
        selected.append(candidate)
        if len(selected) >= config.output_candidate_count:
            break
    if len(selected) < config.output_candidate_count:
        for candidate in deferred:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= config.output_candidate_count:
                break
    if len(selected) < config.output_candidate_count:
        for candidate in order:
            candidate = int(candidate)
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= config.output_candidate_count:
                break
    return np.asarray(selected, dtype=np.int64)
