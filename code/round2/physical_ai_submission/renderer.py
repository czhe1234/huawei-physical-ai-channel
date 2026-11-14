"""Noncoherent path rendering and constrained complete-channel synthesis."""

from __future__ import annotations

import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
import torch

from .config import PipelineConfig, RuntimePaths
from .io_utils import atomic_json, portable_path, validate_channel


TINY = 1e-30


def _normalized(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(
        total > TINY, value / total.clamp_min(TINY), torch.zeros_like(value)
    )


def pas_distribution(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), 2, 16, 8, 4, 192)
    spectrum = torch.fft.fft2(shaped, dim=(2, 3))
    return _normalized(spectrum.abs().square(), (1, 2, 3))


def pdp_distribution(channel: torch.Tensor) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1)
    return _normalized(delay.abs().square(), (3,))


def _spectral_kernels(
    coordinate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizontal_index = torch.arange(16, dtype=torch.float32, device=coordinate.device)
    vertical_index = torch.arange(8, dtype=torch.float32, device=coordinate.device)
    delay_index = torch.arange(192, dtype=torch.float32, device=coordinate.device)
    horizontal_wave = torch.exp(
        2j * torch.pi * coordinate[..., 0, None] * horizontal_index / 16.0
    )
    vertical_wave = torch.exp(
        2j * torch.pi * coordinate[..., 1, None] * vertical_index / 8.0
    )
    delay_wave = torch.exp(
        -2j * torch.pi * coordinate[..., 2, None] * delay_index / 192.0
    )
    horizontal = torch.fft.fft(horizontal_wave, dim=-1).abs().square().float()
    vertical = torch.fft.fft(vertical_wave, dim=-1).abs().square().float()
    delay = torch.fft.ifft(delay_wave, dim=-1).abs().square().float()
    horizontal /= horizontal.sum(dim=-1, keepdim=True).clamp_min(TINY)
    vertical /= vertical.sum(dim=-1, keepdim=True).clamp_min(TINY)
    delay /= delay.sum(dim=-1, keepdim=True).clamp_min(TINY)
    return horizontal, vertical, delay


@torch.inference_mode()
def render_endpoint(
    mass: np.ndarray,
    coordinate: np.ndarray,
    port: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinate_tensor = torch.as_tensor(coordinate, dtype=torch.float32, device=device)
    energy = torch.as_tensor(
        mass[..., None, None] * port, dtype=torch.float32, device=device
    )
    horizontal, vertical, delay = _spectral_kernels(coordinate_tensor)
    small_pas = torch.einsum("bkpu,bkh,bkv->bphvu", energy, horizontal, vertical)
    small_pas = _normalized(small_pas, (1, 2, 3))
    pas = small_pas[..., None].expand(-1, -1, -1, -1, -1, 192)
    small_pdp = torch.einsum("bkpu,bkd->bpud", energy, delay)
    small_pdp = _normalized(small_pdp, (3,))
    pdp = (
        small_pdp[:, :, None, None]
        .expand(-1, -1, 16, 8, -1, -1)
        .reshape(-1, 256, 4, 192)
    )
    return pas, pdp


def _exact_pas_projection(channel: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
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
    return torch.fft.ifft2(desired_magnitude * unit, dim=(2, 3)).reshape_as(channel)


def _exact_pdp_projection(channel: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1)
    magnitude = delay.abs()
    link_energy = magnitude.square().sum(dim=3, keepdim=True)
    desired_magnitude = torch.sqrt((desired * link_energy).clamp_min(0.0))
    unit = delay / magnitude.clamp_min(TINY)
    unit = torch.where(
        magnitude > TINY,
        unit,
        torch.ones((), dtype=delay.dtype, device=delay.device),
    )
    return torch.fft.fft(desired_magnitude * unit, dim=-1)


def _preserve_power(channel: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    current = channel.abs().square().sum(dim=(1, 2, 3))
    scale = torch.sqrt(reference / current.clamp_min(TINY))
    return channel * scale[:, None, None, None]


def _sqrt_target(
    base: torch.Tensor,
    endpoint: torch.Tensor,
    beta: float,
    dims: tuple[int, ...],
) -> torch.Tensor:
    mixed = (
        (1.0 - beta) * torch.sqrt(base.clamp_min(0.0))
        + beta * torch.sqrt(endpoint.clamp_min(0.0))
    ).square()
    return _normalized(mixed, dims)


@torch.inference_mode()
def project_complete_channel(
    base: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    config: PipelineConfig,
) -> torch.Tensor:
    desired_pas = _sqrt_target(
        pas_distribution(base), endpoint_pas, config.pas_blend, (1, 2, 3)
    )
    desired_pdp = _sqrt_target(
        pdp_distribution(base), endpoint_pdp, config.pdp_blend, (3,)
    )
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    output = base
    for _ in range(config.projection_cycles):
        output = _exact_pas_projection(output, desired_pas)
        output = _exact_pdp_projection(output, desired_pdp)
        output = _preserve_power(output, reference_power)
    return output


def synthesize_channel(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    base_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    started = time.time()
    state_path = paths.work_dir / "path_state" / f"{split}.npz"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    with np.load(state_path, allow_pickle=False) as loaded:
        mass = np.array(loaded["mass"], dtype=np.float32, copy=True)
        coordinate = np.array(loaded["coordinate"], dtype=np.float32, copy=True)
        port = np.array(loaded["port"], dtype=np.float32, copy=True)
    base = np.load(base_path, mmap_mode="r")
    expected = (len(mass), 256, 4, 192)
    if base.shape != expected or base.dtype != np.complex64:
        raise ValueError(f"invalid base field: {base.shape}/{base.dtype}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial.npy")
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=expected
    )
    power_error: list[np.ndarray] = []
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        endpoint_pas, endpoint_pdp = render_endpoint(
            mass[start:end], coordinate[start:end], port[start:end], device
        )
        prediction = project_complete_channel(
            base_tensor, endpoint_pas, endpoint_pdp, config
        )
        output[start:end] = prediction.cpu().numpy().astype(np.complex64)
        ratio = prediction.abs().square().sum(
            dim=(1, 2, 3), dtype=torch.float64
        ) / base_tensor.abs().square().sum(
            dim=(1, 2, 3), dtype=torch.float64
        ).clamp_min(
            TINY
        )
        power_error.append((ratio - 1.0).abs().cpu().numpy())
    output.flush()
    del output
    os.replace(partial, output_path)
    validation = validate_channel(output_path, len(base))
    errors = np.concatenate(power_error)
    report = {
        "stage": "complete_channel_synthesis",
        "status": "complete",
        "split": split,
        "base_field": portable_path(base_path),
        "output": validation,
        "power_ratio_max_abs_error": float(np.max(errors)),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_rss_gib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        ),
        "wall_seconds": float(time.time() - started),
        "measured_bottleneck": "batched FFT projection and output write",
    }
    atomic_json(paths.report_dir / f"synthesis_{split}.json", report)
    return report
