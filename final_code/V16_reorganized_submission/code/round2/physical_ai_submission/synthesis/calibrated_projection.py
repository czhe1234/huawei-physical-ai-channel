from __future__ import annotations
import numpy as np
import torch
from ..interpolation_field.grouped_spectrum import (
    encode,
    normalize_group_amplitude,
)


def mode_targets(
    mode: str,
    raw_pas: torch.Tensor,
    raw_pdp: torch.Tensor,
    calibrated_pas: torch.Tensor,
    calibrated_pdp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas = calibrated_pas if mode in ("pascal", "bothcal") else raw_pas
    pdp = calibrated_pdp if mode in ("pdpcal", "bothcal") else raw_pdp
    return (pas, pdp)


def apply_calibration(
    prediction: np.ndarray,
    query_site: np.ndarray,
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    strength: float,
) -> np.ndarray:
    output = np.empty_like(prediction, dtype=np.float32)
    shape = prediction.shape[1:]
    for site_value, (mean, vectors, slope) in models.items():
        rows = np.flatnonzero(query_site == site_value)
        if len(rows) == 0:
            continue
        flat = prediction[rows].reshape(len(rows), -1)
        coefficients = encode(flat, mean, vectors)
        gain = 1.0 + strength * (slope - 1.0)
        corrected = flat + (coefficients * gain - coefficients) @ vectors.T
        output[rows] = normalize_group_amplitude(corrected.reshape(len(rows), *shape))
    return output
