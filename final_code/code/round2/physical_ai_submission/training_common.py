"""Shared deterministic utilities for the official-data retraining pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from scipy.spatial import distance_matrix

from .io_utils import atomic_json


TINY = 1e-30


@dataclass(frozen=True)
class FoldMetadata:
    query_ids: np.ndarray
    fold_id: np.ndarray
    fit_indices: tuple[np.ndarray, ...]
    query_indices: tuple[np.ndarray, ...]
    ignored_indices: tuple[np.ndarray, ...]

    @property
    def query_lookup(self) -> np.ndarray:
        size = int(
            max(
                np.max(self.query_ids, initial=-1),
                *(np.max(value, initial=-1) for value in self.fit_indices),
            )
            + 1
        )
        lookup = np.full(size, -1, dtype=np.int64)
        lookup[self.query_ids] = np.arange(len(self.query_ids), dtype=np.int64)
        return lookup


@dataclass
class MetricArrays:
    pas: np.ndarray
    pdp: np.ndarray
    squared_error: np.ndarray
    truth_energy: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_fold_metadata(fold_cache: Path, fold_metadata: Path) -> FoldMetadata:
    with np.load(fold_cache, allow_pickle=False) as cached:
        fit = tuple(
            cached[f"fold{fold}_fit"].astype(np.int64) for fold in range(5)
        )
        query = tuple(
            cached[f"fold{fold}_query"].astype(np.int64) for fold in range(5)
        )
        ignored = tuple(
            cached[f"fold{fold}_ignored"].astype(np.int64) for fold in range(5)
        )
    with np.load(fold_metadata, allow_pickle=False) as metadata:
        query_ids = metadata["query_ids"].astype(np.int64)
        fold_id = metadata["fold_id"].astype(np.int8)
    if len(query_ids) != 500 or fold_id.shape != (500,):
        raise RuntimeError("spatial-hole metadata must describe exactly 500 queries")
    lookup = np.full(int(np.max(query_ids)) + 1, -1, dtype=np.int64)
    lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    for fold in range(5):
        rows = lookup[query[fold]]
        if np.any(rows < 0) or not np.all(fold_id[rows] == fold):
            raise RuntimeError(f"inconsistent spatial-hole fold {fold}")
    return FoldMetadata(query_ids, fold_id, fit, query, ignored)


def solve_linear_rbf_weights(
    donor_xy: np.ndarray, query_xy: np.ndarray, smoothing: float
) -> np.ndarray:
    donor_xy = np.asarray(donor_xy, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    count = len(donor_xy)
    system = np.empty((count + 1, count + 1), dtype=np.float64)
    system[:count, :count] = -distance_matrix(donor_xy, donor_xy)
    system[np.arange(count), np.arange(count)] += smoothing
    system[:count, count] = 1.0
    system[count, :count] = 1.0
    system[count, count] = 0.0
    right = np.concatenate(
        (-np.linalg.norm(donor_xy - query_xy[None, :], axis=1), np.ones(1))
    )
    try:
        weights = np.linalg.solve(system, right)[:count]
    except np.linalg.LinAlgError:
        weights = (np.linalg.pinv(system) @ right)[:count]
    return weights.astype(np.float32)


def normalize_probability(
    value: np.ndarray, axis: int | tuple[int, ...]
) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def official_metric_arrays_torch(
    prediction: torch.Tensor, truth: torch.Tensor
) -> MetricArrays:
    prediction = prediction.to(dtype=torch.complex64)
    truth = truth.to(device=prediction.device, dtype=torch.complex64)
    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError(
            f"invalid metric tensors: {tuple(prediction.shape)} / {tuple(truth.shape)}"
        )

    def cosine(left: torch.Tensor, right: torch.Tensor, axis: int) -> torch.Tensor:
        numerator = (left * right).sum(dim=axis, dtype=torch.float64)
        denominator = torch.sqrt(
            left.square().sum(dim=axis, dtype=torch.float64)
            * right.square().sum(dim=axis, dtype=torch.float64)
        )
        denominator = torch.where(
            denominator == 0.0,
            torch.full_like(denominator, 1e-12),
            denominator,
        )
        return numerator / denominator

    shaped_prediction = prediction.reshape(len(prediction), 2, 16, 8, 4, 192)
    shaped_truth = truth.reshape_as(shaped_prediction)
    prediction_pas = (
        torch.fft.fft2(shaped_prediction, dim=(2, 3))
        .abs()
        .square()
        .reshape_as(prediction)
    )
    truth_pas = (
        torch.fft.fft2(shaped_truth, dim=(2, 3))
        .abs()
        .square()
        .reshape_as(truth)
    )
    pas = cosine(prediction_pas, truth_pas, 1).mean(dim=(1, 2))
    prediction_pdp = torch.fft.ifft(prediction, dim=-1).abs().square()
    truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square()
    pdp = cosine(prediction_pdp, truth_pdp, 3).mean(dim=(1, 2))
    squared_error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    truth_energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return MetricArrays(
        pas.detach().cpu().numpy().astype(np.float64),
        pdp.detach().cpu().numpy().astype(np.float64),
        squared_error.detach().cpu().numpy().astype(np.float64),
        truth_energy.detach().cpu().numpy().astype(np.float64),
    )


def aggregate_metrics(
    arrays: MetricArrays, mask: np.ndarray | None = None
) -> dict[str, float | int]:
    selected = (
        np.ones(len(arrays.pas), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
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
    pas = float(np.mean(arrays.pas[selected], dtype=np.float64))
    pdp = float(np.mean(arrays.pdp[selected], dtype=np.float64))
    error = float(np.sum(arrays.squared_error[selected], dtype=np.float64))
    truth_energy = float(np.sum(arrays.truth_energy[selected], dtype=np.float64))
    nmse = error / max(truth_energy, TINY)
    nmse_score = 1.0 / (1.0 + nmse)
    return {
        "count": count,
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "nmse_score": nmse_score,
        "score": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
    }


def evaluate_channel_file(
    prediction_path: Path,
    truth: np.ndarray,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    prediction = np.load(prediction_path, mmap_mode="r")
    values: list[MetricArrays] = []
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        values.append(
            official_metric_arrays_torch(
                torch.as_tensor(
                    np.array(prediction[start:end], dtype=np.complex64, copy=True),
                    device=device,
                ),
                torch.as_tensor(
                    np.array(
                        truth[query_ids[start:end]], dtype=np.complex64, copy=True
                    ),
                    device=device,
                ),
            )
        )
    return aggregate_metrics(
        MetricArrays(
            pas=np.concatenate([value.pas for value in values]),
            pdp=np.concatenate([value.pdp for value in values]),
            squared_error=np.concatenate([value.squared_error for value in values]),
            truth_energy=np.concatenate([value.truth_energy for value in values]),
        )
    )


def write_failure_report(path: Path, stage: str, error: BaseException) -> None:
    import traceback

    atomic_json(
        path,
        {
            "stage": stage,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )
