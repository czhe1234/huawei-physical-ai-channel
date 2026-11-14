"""Frozen runtime primitives for angle slice model."""

from __future__ import annotations
import torch
from .exact_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
    truth_views,
)


def project_joint(
    base: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
    cycles: int,
    order: str,
) -> torch.Tensor:
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    output = base
    for _ in range(cycles):
        for domain in order.split("_"):
            if domain == "pas":
                output = exact_pas_projection(output, desired_pas)
            elif domain == "pdp":
                output = exact_pdp_projection(output, desired_pdp)
            else:
                raise ValueError(domain)
        output = preserve_point_power(output, reference_power)
    return output
