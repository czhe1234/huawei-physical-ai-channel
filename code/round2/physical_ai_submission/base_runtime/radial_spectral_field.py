"""Frozen runtime primitives for radial spectral field."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial import cKDTree, distance_matrix
from official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics,
)
from project_paths import PATHS

PRE_DIR = PATHS.artifacts / "round2" / "pre_physical"
OOF_BASE = PRE_DIR / "round2_pre_physical_oof_foundation.npy"
OOF_META = PRE_DIR / "round2_pre_physical_oof_foundation_meta.npz"


@dataclass(frozen=True)
class ProjectionAction:
    name: str
    alpha_ad: float
    alpha_pas: float
    alpha_pdp: float


ACTIONS = (
    ProjectionAction("ad000_pas090_pdp090", 0.0, 0.9, 0.9),
    ProjectionAction("ad050_pas075_pdp080", 0.5, 0.75, 0.8),
    ProjectionAction("ad050_pas090_pdp090", 0.5, 0.9, 0.9),
    ProjectionAction("ad075_pas090_pdp090", 0.75, 0.9, 0.9),
    ProjectionAction("ad100_pas090_pdp090", 1.0, 0.9, 0.9),
    ProjectionAction("ad075_pas100_pdp100", 0.75, 1.0, 1.0),
    ProjectionAction("ad100_pas100_pdp100", 1.0, 1.0, 1.0),
)


def solve_linear_rbf_weights(
    donor_xy: np.ndarray, query_xy: np.ndarray, smoothing: float
) -> np.ndarray:
    count = len(donor_xy)
    system = np.empty((count + 1, count + 1), dtype=np.float64)
    system[:count, :count] = -distance_matrix(donor_xy, donor_xy)
    system[np.arange(count), np.arange(count)] += smoothing
    system[:count, count] = 1.0
    system[count, :count] = 1.0
    system[count, count] = 0.0
    right = np.concatenate(
        [-np.linalg.norm(donor_xy - query_xy[None, :], axis=1), np.ones(1)]
    )
    weights = np.linalg.solve(system, right)[:count]
    return weights.astype(np.float32)


def optimal_scale(
    predicted_energy: np.ndarray, cross: np.ndarray, mask: np.ndarray
) -> float:
    denominator = float(np.sum(predicted_energy[mask], dtype=np.float64))
    numerator = float(np.sum(cross[mask], dtype=np.float64))
    if denominator <= 1e-30:
        return 1.0
    return float(np.clip(numerator / denominator, 0.05, 1.5))


def scaled_error(
    predicted_energy: np.ndarray,
    truth_energy: np.ndarray,
    cross: np.ndarray,
    scale: np.ndarray | float,
) -> np.ndarray:
    value = (
        np.asarray(scale) ** 2 * predicted_energy
        + truth_energy
        - 2.0 * np.asarray(scale) * cross
    )
    return np.maximum(value, 0.0)


def statistics(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    truth_energy: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    return aggregate_statistics(
        PointMetricStatistics(pas, pdp, error, truth_energy), mask
    )
