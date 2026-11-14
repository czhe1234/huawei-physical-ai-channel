"""Frozen runtime primitives for exact projection."""

from __future__ import annotations
import torch
from .spectral_transforms import project_angle_delay
from .radial_spectral_field import (
    ACTIONS,
    OOF_BASE,
    OOF_META,
    optimal_scale,
    scaled_error,
    statistics,
)

TINY = 1e-30


def normalized(
    value: torch.Tensor, dims: tuple[int, ...], fallback: torch.Tensor | None = None
) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    result = value / total.clamp_min(TINY)
    if fallback is not None:
        result = torch.where(total > TINY, result, fallback)
    return result


def preserve_point_power(
    channel: torch.Tensor, reference_power: torch.Tensor
) -> torch.Tensor:
    current = channel.abs().square().sum(dim=(1, 2, 3))
    scale = torch.sqrt(reference_power / current.clamp_min(TINY))
    return channel * scale[:, None, None, None]


def pas_distribution(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    spectrum = torch.fft.fft2(shaped, dim=(2, 3))
    power = spectrum.abs().square()
    return normalized(power, (1, 2, 3))


def pdp_distribution(channel: torch.Tensor) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1)
    return normalized(delay.abs().square(), (3,))


def fixed_blend(
    current: torch.Tensor, target: torch.Tensor, alpha: float, dims: tuple[int, ...]
) -> torch.Tensor:
    target = normalized(target, dims, fallback=current)
    return normalized((1.0 - alpha) * current + alpha * target, dims)


def exact_pas_projection(channel: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    spectrum = torch.fft.fft2(shaped, dim=(2, 3))
    magnitude = spectrum.abs()
    slice_energy = magnitude.square().sum(dim=(1, 2, 3), keepdim=True)
    desired_magnitude = torch.sqrt((desired * slice_energy).clamp_min(0.0))
    unit = spectrum / magnitude.clamp_min(TINY)
    unit = torch.where(
        magnitude > TINY,
        unit,
        torch.ones((), dtype=spectrum.dtype, device=spectrum.device),
    )
    restored = torch.fft.ifft2(desired_magnitude * unit, dim=(2, 3))
    return restored.reshape_as(channel)


def exact_pdp_projection(channel: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1)
    magnitude = delay.abs()
    link_energy = magnitude.square().sum(dim=3, keepdim=True)
    desired_magnitude = torch.sqrt((desired * link_energy).clamp_min(0.0))
    unit = delay / magnitude.clamp_min(TINY)
    unit = torch.where(
        magnitude > TINY, unit, torch.ones((), dtype=delay.dtype, device=delay.device)
    )
    return torch.fft.fft(desired_magnitude * unit, dim=-1)


def exact_ad_projection(channel: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    transformed = torch.fft.ifft(torch.fft.fft2(shaped, dim=(2, 3)), dim=-1)
    power = transformed.abs().square()
    collapsed = power.sum(dim=(1, 4))
    total = collapsed.sum(dim=(1, 2, 3), keepdim=True)
    gain = torch.sqrt((desired * total).clamp_min(0.0) / collapsed.clamp_min(TINY))
    transformed = transformed * gain[:, None, :, :, None, :]
    restored = torch.fft.ifft2(torch.fft.fft(transformed, dim=-1), dim=(2, 3))
    return restored.reshape_as(channel)


def initialize_trajectory(
    base: torch.Tensor,
    ad_target: torch.Tensor,
    pas_target: torch.Tensor,
    pdp_target: torch.Tensor,
    action_index: int,
    order: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action = ACTIONS[action_index]
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    candidate = project_angle_delay(base, ad_target, action.alpha_ad)
    target_pas = pas_target.reshape(len(base), 2, 16, 8, 4, 192)
    if order == "pas_pdp":
        desired_pas = fixed_blend(
            pas_distribution(candidate), target_pas, action.alpha_pas, (1, 2, 3)
        )
        candidate = exact_pas_projection(candidate, desired_pas)
        desired_pdp = fixed_blend(
            pdp_distribution(candidate), pdp_target, action.alpha_pdp, (3,)
        )
        candidate = exact_pdp_projection(candidate, desired_pdp)
    elif order == "pdp_pas":
        desired_pdp = fixed_blend(
            pdp_distribution(candidate), pdp_target, action.alpha_pdp, (3,)
        )
        candidate = exact_pdp_projection(candidate, desired_pdp)
        desired_pas = fixed_blend(
            pas_distribution(candidate), target_pas, action.alpha_pas, (1, 2, 3)
        )
        candidate = exact_pas_projection(candidate, desired_pas)
    else:
        raise ValueError(order)
    return (
        preserve_point_power(candidate, reference_power),
        desired_pas,
        desired_pdp,
        reference_power,
    )


def continue_trajectory(
    candidate: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
    reference_power: torch.Tensor,
    order: str,
) -> torch.Tensor:
    if order == "pas_pdp":
        candidate = exact_pas_projection(candidate, desired_pas)
        candidate = exact_pdp_projection(candidate, desired_pdp)
    elif order == "pdp_pas":
        candidate = exact_pdp_projection(candidate, desired_pdp)
        candidate = exact_pas_projection(candidate, desired_pas)
    else:
        raise ValueError(order)
    return preserve_point_power(candidate, reference_power)


def truth_views(truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shaped = truth.reshape(len(truth), 2, 16, 8, 4, 192)
    pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
    return (pas, pdp)
