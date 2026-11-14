"""Frozen runtime primitives for grouped prediction."""

from __future__ import annotations
import numpy as np
import torch
from .anchored_projection import (
    PAS_GLOBAL_SHRINK,
    PDP_GLOBAL_SHRINK,
    TARGET_PAS,
    TARGET_PDP,
)
from .grouped_link_field import GROUPED_PAS, GROUPED_PDP

PAS_COMPONENTS = 128
PDP_COMPONENTS = 64
TINY = 1e-30


def normalize_power(value: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = value.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        value, np.maximum(total, TINY), out=np.zeros_like(value), where=total > 0.0
    )


def normalize_group_amplitude(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    norm = np.sqrt(np.sum(value * value, axis=-1, keepdims=True, dtype=np.float64))
    return np.divide(
        value, np.maximum(norm, TINY), out=np.zeros_like(value), where=norm > 0.0
    )


def predict_grouped(
    amplitude: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    output = np.empty(
        (len(donor_ids), amplitude.shape[1], amplitude.shape[2]), dtype=np.float32
    )
    for row, (ids, raw_weights) in enumerate(zip(donor_ids, weights)):
        selected_weights = np.asarray(raw_weights, dtype=np.float64).copy()
        if allowed is not None:
            selected_weights *= allowed[ids]
            total = float(np.sum(selected_weights))
            if abs(total) < 0.1:
                selected_weights = np.maximum(selected_weights, 0.0)
                total = float(np.sum(selected_weights))
            if abs(total) < 1e-08:
                nearest = int(np.flatnonzero(allowed[ids])[0])
                selected_weights[:] = 0.0
                selected_weights[nearest] = 1.0
            else:
                selected_weights /= total
        output[row] = np.maximum(
            np.tensordot(
                selected_weights.astype(np.float32),
                np.asarray(amplitude[ids], dtype=np.float32),
                axes=(0, 0),
            ),
            0.0,
        )
    return normalize_group_amplitude(output)


def pca_basis(
    fit_amplitude: np.ndarray, components: int, device: torch.device, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    value = torch.as_tensor(fit_amplitude, dtype=torch.float32, device=device)
    mean = value.mean(dim=0)
    centered = value - mean
    count = min(components, len(value) - 1, value.shape[1])
    _, _, vectors = torch.pca_lowrank(centered, q=count, center=False, niter=4)
    return (mean.cpu().numpy(), vectors.cpu().numpy())


def encode(value: np.ndarray, mean: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float32) - mean[None, :]) @ vectors


def globally_shrunk_targets(
    pas: np.ndarray, pdp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pas = normalize_power(pas, 1)
    pdp = normalize_power(pdp, 3)
    shared_pas = np.broadcast_to(
        pas.mean(axis=(2, 3), keepdims=True, dtype=np.float64).astype(np.float32),
        pas.shape,
    )
    shared_pdp = np.broadcast_to(
        pdp.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32),
        pdp.shape,
    )
    pas_amplitude = (1.0 - PAS_GLOBAL_SHRINK) * np.sqrt(
        np.maximum(pas, 0.0)
    ) + PAS_GLOBAL_SHRINK * np.sqrt(np.maximum(shared_pas, 0.0))
    pdp_amplitude = (1.0 - PDP_GLOBAL_SHRINK) * np.sqrt(
        np.maximum(pdp, 0.0)
    ) + PDP_GLOBAL_SHRINK * np.sqrt(np.maximum(shared_pdp, 0.0))
    return (
        normalize_power(pas_amplitude * pas_amplitude, 1),
        normalize_power(pdp_amplitude * pdp_amplitude, 3),
    )


def blend_group_pas(
    target: np.ndarray, grouped: np.ndarray, alpha: float
) -> np.ndarray:
    if alpha <= 0.0:
        return target
    batch = len(target)
    amplitude = np.sqrt(np.maximum(target, 0.0)).reshape(batch, 256, 4, 8, 24)
    group_amplitude = np.transpose(grouped, (0, 2, 1)).reshape(batch, 256, 4, 8, 1)
    blended = (1.0 - alpha) * amplitude + alpha * group_amplitude
    power = blended * blended
    return normalize_power(power, 1).reshape(batch, 256, 4, 192)


def blend_group_pdp(
    target: np.ndarray, grouped: np.ndarray, alpha: float
) -> np.ndarray:
    if alpha <= 0.0:
        return target
    batch = len(target)
    amplitude = np.sqrt(np.maximum(target, 0.0)).reshape(batch, 2, 16, 8, 4, 192)
    group_amplitude = grouped.reshape(batch, 2, 4, 192)[:, :, None, None]
    blended = (1.0 - alpha) * amplitude + alpha * group_amplitude
    power = blended * blended
    return normalize_power(power, 5).reshape(batch, 256, 4, 192)
