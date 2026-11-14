"""Frozen runtime primitives for path link core."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .exact_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
    truth_views,
)


class LinkGate(nn.Module):

    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -3.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features)[..., 0])


def link_coordinates() -> np.ndarray:
    antenna = np.arange(256, dtype=np.int64)
    polarization = antenna // 128
    local = antenna % 128
    horizontal = local // 8
    vertical = local % 8
    ue = np.arange(4, dtype=np.int64)
    p = np.broadcast_to(polarization[:, None], (256, 4)).reshape(-1)
    h = np.broadcast_to(horizontal[:, None], (256, 4)).reshape(-1)
    v = np.broadcast_to(vertical[:, None], (256, 4)).reshape(-1)
    u = np.broadcast_to(ue[None, :], (256, 4)).reshape(-1)
    return np.concatenate(
        (
            np.eye(2, dtype=np.float32)[p],
            np.eye(4, dtype=np.float32)[u],
            (h / 15.0)[:, None].astype(np.float32),
            (v / 7.0)[:, None].astype(np.float32),
            np.sin(2.0 * np.pi * h / 16.0)[:, None].astype(np.float32),
            np.cos(2.0 * np.pi * h / 16.0)[:, None].astype(np.float32),
            np.sin(2.0 * np.pi * v / 8.0)[:, None].astype(np.float32),
            np.cos(2.0 * np.pi * v / 8.0)[:, None].astype(np.float32),
        ),
        axis=1,
    )


def spectral_features(
    base: torch.Tensor, raw: torch.Tensor, context: torch.Tensor, link: torch.Tensor
) -> torch.Tensor:
    count = len(base)
    base_group = base.reshape(count, 1024, 24, 8).sum(dim=3)
    raw_group = raw.reshape(count, 1024, 24, 8).sum(dim=3)
    base_root = torch.sqrt(base_group.clamp_min(0.0))
    raw_root = torch.sqrt(raw_group.clamp_min(0.0))
    difference = raw_root - base_root
    product = raw_root * base_root
    base_entropy = -(base * torch.log(base.clamp_min(1e-12))).sum(dim=2, keepdim=True)
    raw_entropy = -(raw * torch.log(raw.clamp_min(1e-12))).sum(dim=2, keepdim=True)
    agreement = (base_root * raw_root).sum(dim=2, keepdim=True)
    summary = torch.cat((base_entropy, raw_entropy, agreement), dim=2)
    context_block = context[:, None].expand(-1, 1024, -1)
    link_block = link[None].expand(count, -1, -1)
    return torch.cat(
        (base_root, raw_root, difference, product, summary, context_block, link_block),
        dim=2,
    )


def cosine_by_link(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=2, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=2, dtype=torch.float64)
        * truth.square().sum(dim=2, dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def project(
    base: torch.Tensor, desired_pdp: torch.Tensor, cycles: int, order: str
) -> torch.Tensor:
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    desired_pas = pas_distribution(base)
    output = base
    for _ in range(cycles):
        for domain in order.split("_"):
            if domain == "pas":
                output = exact_pas_projection(output, desired_pas)
            else:
                output = exact_pdp_projection(output, desired_pdp)
        output = preserve_point_power(output, reference_power)
    return output
