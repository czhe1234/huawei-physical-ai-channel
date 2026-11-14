"""Spectral caches and projections for the Round 2 pre-Sionna model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from project_paths import PATHS


MP = 2
MH = 16
MV = 8
M = MP * MH * MV
N = 4
S = 192
TINY = 1e-20

ARTIFACT_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_AD = ARTIFACT_DIR / "round2_train_ad_support_f16.npy"
TRAIN_PAS = ARTIFACT_DIR / "round2_train_pas_link_support_f16.npy"
TRAIN_PDP = ARTIFACT_DIR / "round2_train_pdp_link_support_f16.npy"
TRAIN_DESCRIPTORS = ARTIFACT_DIR / "round2_train_spectral_descriptors.npz"


@dataclass(frozen=True)
class SpectralCache:
    ad: np.ndarray
    pas: np.ndarray
    pdp: np.ndarray
    global_pas: np.ndarray
    global_pdp: np.ndarray
    total_power: np.ndarray
    valid: np.ndarray
    pas_entropy: np.ndarray
    pdp_entropy: np.ndarray


def _normalize_torch(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), torch.zeros_like(value))


def channel_spectra_torch(channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = channel.shape[0]
    shaped = channel.reshape(batch, MP, MH, MV, N, S)
    spatial = torch.fft.fft2(shaped, dim=(2, 3))
    pas_power = spatial.abs().square()
    pas = _normalize_torch(pas_power, (1, 2, 3)).reshape(batch, M, N, S)

    delay = torch.fft.ifft(shaped, dim=-1)
    pdp_power = delay.abs().square()
    pdp = _normalize_torch(pdp_power, (5,)).reshape(batch, M, N, S)

    angle_delay = torch.fft.ifft(spatial, dim=-1)
    ad_power = angle_delay.abs().square().sum(dim=(1, 4))
    ad = _normalize_torch(ad_power, (1, 2, 3))
    return ad.float(), pas.float(), pdp.float()


def channel_spectra_numpy(channel: np.ndarray, device: torch.device | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.as_tensor(np.asarray(channel, dtype=np.complex64), device=selected)
    with torch.inference_mode():
        ad, pas, pdp = channel_spectra_torch(tensor)
    return ad.cpu().numpy(), pas.cpu().numpy(), pdp.cpu().numpy()


def _entropy(distribution: np.ndarray, axis: int) -> np.ndarray:
    value = np.asarray(distribution, dtype=np.float64)
    return -np.sum(value * np.log(np.maximum(value, 1e-30)), axis=axis).astype(np.float32)


def build_train_spectral_cache(
    train_channel: np.ndarray,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> SpectralCache:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    expected = (len(train_channel), M, N, S)
    cache_ready = (
        TRAIN_AD.is_file()
        and TRAIN_PAS.is_file()
        and TRAIN_PDP.is_file()
        and TRAIN_DESCRIPTORS.is_file()
    )
    if cache_ready:
        ad = np.load(TRAIN_AD, mmap_mode="r")
        pas = np.load(TRAIN_PAS, mmap_mode="r")
        pdp = np.load(TRAIN_PDP, mmap_mode="r")
        if ad.shape == (len(train_channel), MH, MV, S) and pas.shape == expected and pdp.shape == expected:
            data = np.load(TRAIN_DESCRIPTORS)
            return SpectralCache(
                ad=ad,
                pas=pas,
                pdp=pdp,
                global_pas=data["global_pas"].astype(np.float32),
                global_pdp=data["global_pdp"].astype(np.float32),
                total_power=data["total_power"].astype(np.float64),
                valid=data["valid"].astype(bool),
                pas_entropy=data["pas_entropy"].astype(np.float32),
                pdp_entropy=data["pdp_entropy"].astype(np.float32),
            )

    selected = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ad_out = np.lib.format.open_memmap(
        TRAIN_AD, mode="w+", dtype=np.float16, shape=(len(train_channel), MH, MV, S)
    )
    pas_out = np.lib.format.open_memmap(TRAIN_PAS, mode="w+", dtype=np.float16, shape=expected)
    pdp_out = np.lib.format.open_memmap(TRAIN_PDP, mode="w+", dtype=np.float16, shape=expected)
    global_pas = np.zeros((len(train_channel), M), dtype=np.float32)
    global_pdp = np.zeros((len(train_channel), S), dtype=np.float32)
    total_power = np.zeros(len(train_channel), dtype=np.float64)

    for start in range(0, len(train_channel), batch_size):
        end = min(start + batch_size, len(train_channel))
        block = np.asarray(train_channel[start:end], dtype=np.complex64)
        total_power[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        ad, pas, pdp = channel_spectra_numpy(block, selected)
        ad_out[start:end] = ad.astype(np.float16)
        pas_out[start:end] = pas.astype(np.float16)
        pdp_out[start:end] = pdp.astype(np.float16)
        global_pas[start:end] = pas.mean(axis=(2, 3), dtype=np.float64).astype(np.float32)
        global_pdp[start:end] = pdp.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
        print(f"train spectral cache {end}/{len(train_channel)}", flush=True)

    ad_out.flush()
    pas_out.flush()
    pdp_out.flush()

    valid = total_power > 1e-30
    pas_entropy = _entropy(global_pas, axis=1)
    pdp_entropy = _entropy(global_pdp, axis=1)
    np.savez_compressed(
        TRAIN_DESCRIPTORS,
        global_pas=global_pas,
        global_pdp=global_pdp,
        total_power=total_power,
        valid=valid,
        pas_entropy=pas_entropy,
        pdp_entropy=pdp_entropy,
    )
    return SpectralCache(
        ad=np.load(TRAIN_AD, mmap_mode="r"),
        pas=np.load(TRAIN_PAS, mmap_mode="r"),
        pdp=np.load(TRAIN_PDP, mmap_mode="r"),
        global_pas=global_pas,
        global_pdp=global_pdp,
        total_power=total_power,
        valid=valid,
        pas_entropy=pas_entropy,
        pdp_entropy=pdp_entropy,
    )


def _preserve_point_power(channel: torch.Tensor, reference_power: torch.Tensor) -> torch.Tensor:
    current = channel.abs().square().sum(dim=(1, 2, 3))
    scale = torch.sqrt(reference_power / current.clamp_min(TINY))
    return channel * scale[:, None, None, None]


def project_angle_delay(
    channel: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha <= 0.0:
        return channel
    reference_power = channel.abs().square().sum(dim=(1, 2, 3))
    shaped = channel.reshape(len(channel), MP, MH, MV, N, S)
    transformed = torch.fft.ifft(torch.fft.fft2(shaped, dim=(2, 3)), dim=-1)
    power = transformed.abs().square()
    collapsed = power.sum(dim=(1, 4))
    normalized = _normalize_torch(collapsed, (1, 2, 3))
    desired = _normalize_torch((1.0 - alpha) * normalized + alpha * target, (1, 2, 3))
    total = collapsed.sum(dim=(1, 2, 3), keepdim=True)
    gain = torch.sqrt((desired * total).clamp_min(1e-30) / collapsed.clamp_min(1e-30))
    gain = gain.clamp(0.35, 2.85)
    transformed = transformed * gain[:, None, :, :, None, :]
    restored = torch.fft.ifft2(torch.fft.fft(transformed, dim=-1), dim=(2, 3))
    return _preserve_point_power(restored.reshape_as(channel), reference_power)


def project_pas(
    channel: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha <= 0.0:
        return channel
    reference_power = channel.abs().square().sum(dim=(1, 2, 3))
    shaped = channel.reshape(len(channel), MP, MH, MV, N, S)
    spectrum = torch.fft.fft2(shaped, dim=(2, 3))
    power = spectrum.abs().square()
    current = _normalize_torch(power, (1, 2, 3))
    target_shaped = target.reshape(len(channel), MP, MH, MV, N, S)
    desired = _normalize_torch((1.0 - alpha) * current + alpha * target_shaped, (1, 2, 3))
    total = power.sum(dim=(1, 2, 3), keepdim=True)
    gain = torch.sqrt((desired * total).clamp_min(1e-30) / power.clamp_min(1e-30))
    gain = gain.clamp(0.40, 2.50)
    restored = torch.fft.ifft2(spectrum * gain, dim=(2, 3)).reshape_as(channel)
    return _preserve_point_power(restored, reference_power)


def project_pdp(
    channel: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha <= 0.0:
        return channel
    reference_power = channel.abs().square().sum(dim=(1, 2, 3))
    delay = torch.fft.ifft(channel, dim=-1)
    power = delay.abs().square()
    current = _normalize_torch(power, (3,))
    desired = _normalize_torch((1.0 - alpha) * current + alpha * target, (3,))
    total = power.sum(dim=3, keepdim=True)
    gain = torch.sqrt((desired * total).clamp_min(1e-30) / power.clamp_min(1e-30))
    gain = gain.clamp(0.40, 2.75)
    restored = torch.fft.fft(delay * gain, dim=-1)
    return _preserve_point_power(restored, reference_power)


def project_channel(
    channel: torch.Tensor,
    ad_target: torch.Tensor,
    pas_target: torch.Tensor,
    pdp_target: torch.Tensor,
    alpha_ad: float,
    alpha_pas: float,
    alpha_pdp: float,
) -> torch.Tensor:
    output = project_angle_delay(channel, ad_target, alpha_ad)
    output = project_pas(output, pas_target, alpha_pas)
    output = project_pdp(output, pdp_target, alpha_pdp)
    return output
