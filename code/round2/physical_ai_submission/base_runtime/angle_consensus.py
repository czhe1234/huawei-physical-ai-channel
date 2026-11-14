"""Frozen runtime primitives for angle consensus."""

from __future__ import annotations
import torch


def build_consensus(
    base: torch.Tensor, sources: list[torch.Tensor], indices: tuple[int, ...]
) -> torch.Tensor:
    residual = torch.stack([sources[index] - base for index in indices])
    norm = residual.square().sum(dim=4, keepdim=True).sqrt().clamp_min(1e-08)
    unit = residual / norm
    mean_direction = unit.mean(dim=0)
    agreement = mean_direction.square().sum(dim=3, keepdim=True).sqrt()
    direction = mean_direction / agreement.clamp_min(1e-08)
    median_scale = residual.square().sum(dim=4).sqrt().median(dim=0).values
    return base + direction * median_scale[..., None] * agreement
