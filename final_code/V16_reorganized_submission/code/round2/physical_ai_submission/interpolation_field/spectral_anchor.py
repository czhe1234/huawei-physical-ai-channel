from __future__ import annotations
import numpy as np


def normalize(value: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        array, np.maximum(total, 1e-20), out=np.zeros_like(array), where=total > 0.0
    )


def spectral_target(
    cache: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    axes: int | tuple[int, ...],
) -> np.ndarray:
    donor = np.sqrt(np.maximum(np.asarray(cache[donor_ids], dtype=np.float32), 0.0))
    amplitude = np.maximum(
        np.tensordot(weights.astype(np.float32), donor, axes=(0, 0)), 0.0
    )
    return normalize(np.square(amplitude), axes)
