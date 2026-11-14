from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CarrierEstimate:
    beta: float
    gamma: float
    objective: float
    pair_count: int
    median_pair_distance_m: float
    median_radial_delta_m: float
    median_link_coherence: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "beta": self.beta,
            "gamma": self.gamma,
            "objective": self.objective,
            "pair_count": self.pair_count,
            "median_pair_distance_m": self.median_pair_distance_m,
            "median_radial_delta_m": self.median_radial_delta_m,
            "median_link_coherence": self.median_link_coherence,
        }


def _candidate_pairs(
    positions: np.ndarray,
    site_ids: np.ndarray,
    fit_indices: np.ndarray,
    point_power: np.ndarray,
    max_pairs_per_site: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for site in np.unique(site_ids[fit_indices]):
        site_fit = np.asarray(
            fit_indices[site_ids[fit_indices] == site], dtype=np.int64
        )
        valid = site_fit[point_power[site_fit] > 1e-30]
        tree = cKDTree(positions[valid, :2])
        distances, local = tree.query(positions[valid, :2], k=2, workers=-1)
        left = valid
        right = valid[local[:, 1]]
        separation = distances[:, 1]
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        pairs = np.column_stack([low, high])
        _, unique_rows = np.unique(pairs, axis=0, return_index=True)
        left = left[unique_rows]
        right = right[unique_rows]
        separation = separation[unique_rows]
        keep = (separation <= 8.0) & (separation >= 0.15)
        left = left[keep]
        right = right[keep]
        separation = separation[keep]
        if len(left) > max_pairs_per_site:
            chosen = np.sort(
                rng.choice(len(left), size=max_pairs_per_site, replace=False)
            )
            left = left[chosen]
            right = right[chosen]
            separation = separation[chosen]
        left_parts.append(left)
        right_parts.append(right)
        distance_parts.append(separation)
    return (
        np.concatenate(left_parts).astype(np.int64),
        np.concatenate(right_parts).astype(np.int64),
        np.concatenate(distance_parts).astype(np.float64),
    )


def _cross_observations(
    train_channel: np.ndarray, left: np.ndarray, right: np.ndarray, batch_size: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    cross = np.empty((len(left), train_channel.shape[-1]), dtype=np.complex64)
    coherence = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), batch_size):
        end = min(start + batch_size, len(left))
        first = np.asarray(train_channel[left[start:end]], dtype=np.complex64)
        second = np.asarray(train_channel[right[start:end]], dtype=np.complex64)
        value = np.sum(np.conj(first) * second, axis=(1, 2), dtype=np.complex64)
        first_energy = np.sum(np.abs(first) ** 2, axis=(1, 2), dtype=np.float64)
        second_energy = np.sum(np.abs(second) ** 2, axis=(1, 2), dtype=np.float64)
        normalization = np.sqrt(first_energy * second_energy)
        normalized = value / np.maximum(normalization, 1e-30)
        cross[start:end] = normalized.astype(np.complex64)
        coherence[start:end] = np.mean(np.abs(normalized), axis=1).astype(np.float32)
    return (cross, coherence)


def _score_beta(
    cross: np.ndarray, radial_delta: np.ndarray, gamma: float, beta_values: np.ndarray
) -> np.ndarray:
    subcarrier = np.linspace(-0.5, 0.5, cross.shape[1], dtype=np.float64)
    gamma_phase = np.exp(1j * gamma * radial_delta[:, None] * subcarrier[None, :])
    summary = np.sum(cross * gamma_phase, axis=1, dtype=np.complex128)
    phase = np.exp(1j * radial_delta[:, None] * beta_values[None, :])
    numerator = np.real(np.sum(summary[:, None] * phase, axis=0))
    denominator = max(float(np.sum(np.abs(cross), dtype=np.float64)), 1e-30)
    return numerator / denominator


def _score_gamma(
    cross: np.ndarray, radial_delta: np.ndarray, beta: float, gamma_values: np.ndarray
) -> np.ndarray:
    subcarrier = np.linspace(-0.5, 0.5, cross.shape[1], dtype=np.float64)
    beta_phase = np.exp(1j * beta * radial_delta)[:, None]
    weighted = cross * beta_phase
    scores = np.empty(len(gamma_values), dtype=np.float64)
    denominator = max(float(np.sum(np.abs(cross), dtype=np.float64)), 1e-30)
    for index, gamma in enumerate(gamma_values):
        phase = np.exp(1j * gamma * radial_delta[:, None] * subcarrier[None, :])
        scores[index] = float(np.real(np.sum(weighted * phase))) / denominator
    return scores


def estimate_carrier(
    train_channel: np.ndarray,
    positions: np.ndarray,
    site_ids: np.ndarray,
    bs_positions: np.ndarray,
    fit_indices: np.ndarray,
    point_power: np.ndarray,
    max_pairs_per_site: int = 500,
    seed: int = 20260805,
) -> dict[int, CarrierEstimate]:
    pass
    estimates: dict[int, CarrierEstimate] = {}
    for site in np.unique(site_ids[fit_indices]):
        site_fit = fit_indices[site_ids[fit_indices] == site]
        left, right, pair_distance = _candidate_pairs(
            positions,
            site_ids,
            site_fit,
            point_power,
            max_pairs_per_site,
            seed + int(site),
        )
        cross, coherence = _cross_observations(train_channel, left, right)
        threshold = float(np.quantile(coherence, 0.5))
        keep = coherence >= max(threshold, 0.02)
        left = left[keep]
        right = right[keep]
        pair_distance = pair_distance[keep]
        cross = cross[keep]
        coherence = coherence[keep]
        radial = np.linalg.norm(positions - bs_positions[int(site)][None, :], axis=1)
        radial_delta = radial[right] - radial[left]
        informative = np.abs(radial_delta) >= 0.05
        radial_delta = radial_delta[informative]
        pair_distance = pair_distance[informative]
        cross = cross[informative]
        coherence = coherence[informative]
        if len(radial_delta) < 32:
            raise RuntimeError(
                f"site {site } has only {len (radial_delta )} carrier pairs"
            )
        coarse_beta = np.arange(40.0, 180.0001, 0.1, dtype=np.float64)
        coarse_scores = _score_beta(cross, radial_delta, 0.0, coarse_beta)
        beta = float(coarse_beta[int(np.argmax(coarse_scores))])
        gamma = 0.0
        for _ in range(2):
            gamma_values = np.arange(-0.3, 0.3001, 0.005, dtype=np.float64)
            gamma_scores = _score_gamma(cross, radial_delta, beta, gamma_values)
            gamma = float(gamma_values[int(np.argmax(gamma_scores))])
            beta_values = np.arange(beta - 0.2, beta + 0.2001, 0.0025, dtype=np.float64)
            beta_scores = _score_beta(cross, radial_delta, gamma, beta_values)
            beta = float(beta_values[int(np.argmax(beta_scores))])
        objective = float(
            _score_beta(cross, radial_delta, gamma, np.asarray([beta]))[0]
        )
        estimates[int(site)] = CarrierEstimate(
            beta=beta,
            gamma=gamma,
            objective=objective,
            pair_count=int(len(radial_delta)),
            median_pair_distance_m=float(np.median(pair_distance)),
            median_radial_delta_m=float(np.median(np.abs(radial_delta))),
            median_link_coherence=float(np.median(coherence)),
        )
    return estimates
