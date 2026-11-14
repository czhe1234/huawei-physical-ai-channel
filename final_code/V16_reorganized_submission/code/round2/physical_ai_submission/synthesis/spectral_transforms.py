from __future__ import annotations
import torch

MP = 2
MH = 16
MV = 8
N = 4
S = 192
TINY = 1e-20


def _normalize_torch(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(
        total > TINY, value / total.clamp_min(TINY), torch.zeros_like(value)
    )


def _preserve_point_power(
    channel: torch.Tensor, reference_power: torch.Tensor
) -> torch.Tensor:
    current = channel.abs().square().sum(dim=(1, 2, 3))
    scale = torch.sqrt(reference_power / current.clamp_min(TINY))
    return channel * scale[:, None, None, None]


def project_angle_delay(
    channel: torch.Tensor, target: torch.Tensor, alpha: float
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
