"""Frozen runtime primitives for angle residual."""

from __future__ import annotations
import torch

TINY = 1e-30


def normalize_pas(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=(1, 2, 3), keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def pas_slices(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 4, 5, 1, 2, 3).reshape(len(value), 4, 192, 256)


def pas_from_slices(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(len(value), 4, 192, 2, 16, 8).permute(0, 3, 4, 5, 1, 2)
