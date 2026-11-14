"""Frozen runtime primitives for angle reliability."""

from __future__ import annotations
import torch


def percentile_rank(value: torch.Tensor) -> torch.Tensor:
    flat = value.reshape(-1)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(
        1, len(flat) + 1, device=value.device, dtype=value.dtype
    ) / float(len(flat))
    return ranks.reshape_as(value)
