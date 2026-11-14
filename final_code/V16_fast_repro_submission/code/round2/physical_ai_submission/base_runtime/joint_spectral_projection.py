"""Frozen runtime primitives for joint spectral projection."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from .exact_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    fixed_blend,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)

TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    name: str
    pas_beta: float = 0.0
    pdp_beta: float = 0.0
    alpha_pas: float = 0.0
    alpha_pdp: float = 0.0
    order: str = "pas_pdp"
    cycles: int = 4


def normalize(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def endpoint_blend(
    old: torch.Tensor, joint: torch.Tensor, beta: float, dims: tuple[int, ...]
) -> torch.Tensor:
    if beta <= 0.0:
        return old
    if beta >= 1.0:
        return joint
    return normalize((1.0 - beta) * old + beta * joint, dims)


def apply_candidate(
    base: torch.Tensor,
    old_pas: torch.Tensor,
    joint_pas: torch.Tensor,
    old_pdp: torch.Tensor,
    joint_pdp: torch.Tensor,
    spec: Candidate,
) -> torch.Tensor:
    if spec.name == "identity":
        return base
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    pas_target = endpoint_blend(old_pas, joint_pas, spec.pas_beta, (1, 2, 3))
    pdp_target = endpoint_blend(old_pdp, joint_pdp, spec.pdp_beta, (3,))
    desired_pas = fixed_blend(
        pas_distribution(base), pas_target, spec.alpha_pas, (1, 2, 3)
    )
    desired_pdp = fixed_blend(pdp_distribution(base), pdp_target, spec.alpha_pdp, (3,))
    candidate = base
    for _ in range(spec.cycles):
        for domain in spec.order.split("_"):
            if domain == "pas":
                candidate = exact_pas_projection(candidate, desired_pas)
            elif domain == "pdp":
                candidate = exact_pdp_projection(candidate, desired_pdp)
            else:
                raise ValueError(domain)
        candidate = preserve_point_power(candidate, reference_power)
    return candidate
