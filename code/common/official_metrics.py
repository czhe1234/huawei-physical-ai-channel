"""Official PAS/PDP/NMSE metric for MIMO-OFDM channel tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as scipy_fft


@dataclass(frozen=True)
class ArrayLayout:
    tx_polarizations: int = 2
    tx_horizontal: int = 16
    tx_vertical: int = 8

    @property
    def tx_antennas(self) -> int:
        return self.tx_polarizations * self.tx_horizontal * self.tx_vertical


@dataclass
class PointMetricStatistics:
    pas: np.ndarray
    pdp: np.ndarray
    squared_error: np.ndarray
    truth_energy: np.ndarray


def _validate_channel(channel: np.ndarray, layout: ArrayLayout) -> None:
    if channel.ndim != 4:
        raise ValueError(f"channel must have shape [point,M,N,S], got {channel.shape}")
    if channel.shape[1] != layout.tx_antennas:
        raise ValueError(
            f"M={channel.shape[1]} does not match array layout M={layout.tx_antennas}"
        )


def official_cosine(
    prediction: np.ndarray,
    truth: np.ndarray,
    vector_axis: int,
) -> np.ndarray:
    """Cosine with the corrected official zero-norm denominator rule."""

    pred = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(truth, dtype=np.float32)
    numerator = np.sum(pred * target, axis=vector_axis, dtype=np.float64)
    pred_norm_sq = np.sum(pred * pred, axis=vector_axis, dtype=np.float64)
    truth_norm_sq = np.sum(target * target, axis=vector_axis, dtype=np.float64)
    denominator = np.sqrt(pred_norm_sq * truth_norm_sq)
    denominator = np.where(denominator == 0.0, 1e-12, denominator)
    return (numerator / denominator).astype(np.float32)


def pas_power(channel: np.ndarray, layout: ArrayLayout = ArrayLayout()) -> np.ndarray:
    """Return BS PAS power as [point,M,N,S], with M ordered [P,H,V]."""

    value = np.asarray(channel)
    _validate_channel(value, layout)
    shaped = value.reshape(
        len(value),
        layout.tx_polarizations,
        layout.tx_horizontal,
        layout.tx_vertical,
        value.shape[2],
        value.shape[3],
    )
    spectrum = scipy_fft.fft2(shaped, axes=(2, 3), workers=-1)
    power = np.square(np.abs(spectrum), dtype=np.float32)
    return power.reshape(value.shape)


def pdp_power(channel: np.ndarray) -> np.ndarray:
    value = np.asarray(channel)
    delay = scipy_fft.ifft(value, axis=-1, workers=-1)
    return np.square(np.abs(delay), dtype=np.float32)


def point_metric_statistics(
    prediction: np.ndarray,
    truth: np.ndarray,
    layout: ArrayLayout = ArrayLayout(),
) -> PointMetricStatistics:
    pred = np.asarray(prediction, dtype=np.complex64)
    target = np.asarray(truth, dtype=np.complex64)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/truth shape mismatch: {pred.shape} != {target.shape}")
    _validate_channel(pred, layout)

    pred_pas = pas_power(pred, layout)
    truth_pas = pas_power(target, layout)
    pas = official_cosine(pred_pas, truth_pas, vector_axis=1).mean(
        axis=(1, 2), dtype=np.float64
    )

    pred_pdp = pdp_power(pred)
    truth_pdp = pdp_power(target)
    pdp = official_cosine(pred_pdp, truth_pdp, vector_axis=3).mean(
        axis=(1, 2), dtype=np.float64
    )

    squared_error = np.sum(
        np.abs(pred - target) ** 2, axis=(1, 2, 3), dtype=np.float64
    )
    truth_energy = np.sum(
        np.abs(target) ** 2, axis=(1, 2, 3), dtype=np.float64
    )
    return PointMetricStatistics(
        pas=np.asarray(pas, dtype=np.float64),
        pdp=np.asarray(pdp, dtype=np.float64),
        squared_error=squared_error,
        truth_energy=truth_energy,
    )


def point_metric_statistics_torch(
    prediction,
    truth,
    layout: ArrayLayout = ArrayLayout(),
) -> PointMetricStatistics:
    """GPU-capable equivalent of ``point_metric_statistics`` for batched tensors."""
    import torch

    pred = torch.as_tensor(prediction, dtype=torch.complex64)
    target = torch.as_tensor(truth, dtype=torch.complex64, device=pred.device)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/truth shape mismatch: {pred.shape} != {target.shape}")
    if pred.ndim != 4 or pred.shape[1] != layout.tx_antennas:
        raise ValueError(f"invalid channel shape for official metric: {tuple(pred.shape)}")

    def cosine(value, reference, vector_axis: int):
        numerator = (value * reference).sum(dim=vector_axis, dtype=torch.float64)
        value_norm = (value * value).sum(dim=vector_axis, dtype=torch.float64)
        reference_norm = (reference * reference).sum(
            dim=vector_axis, dtype=torch.float64
        )
        denominator = torch.sqrt(value_norm * reference_norm)
        denominator = torch.where(
            denominator == 0.0,
            torch.full_like(denominator, 1e-12),
            denominator,
        )
        return (numerator / denominator).to(torch.float32)

    shaped_pred = pred.reshape(
        len(pred),
        layout.tx_polarizations,
        layout.tx_horizontal,
        layout.tx_vertical,
        pred.shape[2],
        pred.shape[3],
    )
    shaped_truth = target.reshape_as(shaped_pred)
    pred_pas = torch.fft.fft2(shaped_pred, dim=(2, 3)).abs().square().reshape_as(pred)
    truth_pas = (
        torch.fft.fft2(shaped_truth, dim=(2, 3)).abs().square().reshape_as(target)
    )
    pas = cosine(pred_pas, truth_pas, 1).mean(dim=(1, 2), dtype=torch.float64)

    pred_pdp = torch.fft.ifft(pred, dim=-1).abs().square()
    truth_pdp = torch.fft.ifft(target, dim=-1).abs().square()
    pdp = cosine(pred_pdp, truth_pdp, 3).mean(dim=(1, 2), dtype=torch.float64)

    squared_error = torch.abs(pred - target).square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    truth_energy = torch.abs(target).square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    return PointMetricStatistics(
        pas=pas.cpu().numpy().astype(np.float64, copy=False),
        pdp=pdp.cpu().numpy().astype(np.float64, copy=False),
        squared_error=squared_error.cpu().numpy().astype(np.float64, copy=False),
        truth_energy=truth_energy.cpu().numpy().astype(np.float64, copy=False),
    )


def aggregate_statistics(
    statistics: PointMetricStatistics,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    if mask is None:
        selected = np.ones(len(statistics.pas), dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
    count = int(np.sum(selected))
    if count == 0:
        return {
            "count": 0,
            "pas": float("nan"),
            "pdp": float("nan"),
            "nmse": float("nan"),
            "nmse_score": float("nan"),
            "score": float("nan"),
        }

    pas = float(np.mean(statistics.pas[selected], dtype=np.float64))
    pdp = float(np.mean(statistics.pdp[selected], dtype=np.float64))
    truth_energy = float(np.sum(statistics.truth_energy[selected], dtype=np.float64))
    error = float(np.sum(statistics.squared_error[selected], dtype=np.float64))
    nmse = error / max(truth_energy, 1e-30)
    nmse_score = 1.0 / (1.0 + nmse)
    return {
        "count": count,
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "nmse_score": nmse_score,
        "score": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
    }


class OfficialMetricAccumulator:
    def __init__(self, layout: ArrayLayout = ArrayLayout()) -> None:
        self.layout = layout
        self._pas: list[np.ndarray] = []
        self._pdp: list[np.ndarray] = []
        self._error: list[np.ndarray] = []
        self._energy: list[np.ndarray] = []

    def add(self, prediction: np.ndarray, truth: np.ndarray) -> None:
        values = point_metric_statistics(prediction, truth, self.layout)
        self._pas.append(values.pas)
        self._pdp.append(values.pdp)
        self._error.append(values.squared_error)
        self._energy.append(values.truth_energy)

    def statistics(self) -> PointMetricStatistics:
        if not self._pas:
            raise RuntimeError("no metric batches were added")
        return PointMetricStatistics(
            pas=np.concatenate(self._pas),
            pdp=np.concatenate(self._pdp),
            squared_error=np.concatenate(self._error),
            truth_energy=np.concatenate(self._energy),
        )

    def result(self) -> dict[str, float | int]:
        return aggregate_statistics(self.statistics())
