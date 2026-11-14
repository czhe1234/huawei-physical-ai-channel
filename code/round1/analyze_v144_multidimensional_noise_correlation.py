#!/usr/bin/env python3
"""Analyze whether post-v140 train residuals contain exploitable structure."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

from analyze_post_v143_hidden_information import (
    normalize_without_floor,
    regression_oof,
    support_score,
)
from analyze_v143_matched_residual_probability import basic_features
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import (
    build_risk,
    row_cos,
    support_features,
)
from generate_v143_transductive_geodesic_risk import apply_risk_shift
from powerfield_common import TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(144)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v144_multidimensional_noise_correlation_analysis.json"
POINT_CSV = REPORT_DIR / "v144_noise_point_metrics.csv"
DISTANCE_CSV = REPORT_DIR / "v144_noise_distance_curve.csv"
NEIGHBOR_CSV = REPORT_DIR / "v144_noise_neighbor_spaces.csv"
MODES_OUT = REPORT_DIR / "v144_noise_component_modes.npz"

RANDOM_STATE = 14471
PERMUTATIONS = 200
DISTANCE_BINS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 200.0, np.inf)
K_VALUES = (4, 8, 16, 32)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or x[valid].std() == 0 or y[valid].std() == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": float(pearsonr(x[valid], y[valid]).statistic),
        "spearman": float(spearmanr(x[valid], y[valid]).statistic),
    }


def normalize_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


def normalize_complex_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.complex64)
    norm = np.sqrt(np.sum(np.abs(x) ** 2, axis=1, keepdims=True))
    return (x / np.maximum(norm, 1e-12)).astype(np.complex64)


def marginal(support: np.ndarray, axis: str) -> np.ndarray:
    if axis == "H":
        return support.sum(axis=(2, 3))
    if axis == "V":
        return support.sum(axis=(1, 3))
    if axis == "D":
        return support.sum(axis=(1, 2))
    raise ValueError(axis)


def entropy(support: np.ndarray) -> np.ndarray:
    flat = support.reshape(len(support), -1)
    return (
        -(flat * np.log(np.maximum(flat, 1e-30))).sum(axis=1)
        / math.log(flat.shape[1])
    ).astype(np.float32)


def probability_entropy(probability: np.ndarray) -> np.ndarray:
    return (
        -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1)
        / math.log(probability.shape[1])
    ).astype(np.float32)


def js_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    flat_a = a.reshape(len(a), -1)
    flat_b = b.reshape(len(b), -1)
    middle = 0.5 * (flat_a + flat_b)
    return (
        0.5
        * np.sum(
            flat_a
            * (np.log(np.maximum(flat_a, 1e-30)) - np.log(np.maximum(middle, 1e-30))),
            axis=1,
        )
        + 0.5
        * np.sum(
            flat_b
            * (np.log(np.maximum(flat_b, 1e-30)) - np.log(np.maximum(middle, 1e-30))),
            axis=1,
        )
    ).astype(np.float32)


def robust_scale(train: np.ndarray, test: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    median = np.median(train, axis=0, keepdims=True)
    scale = np.quantile(train, 0.75, axis=0, keepdims=True) - np.quantile(
        train, 0.25, axis=0, keepdims=True
    )
    scale = np.maximum(scale, 1e-3)
    train_scaled = np.clip((train - median) / scale, -8.0, 8.0).astype(np.float32)
    if test is None:
        return train_scaled, None
    test_scaled = np.clip((test - median) / scale, -8.0, 8.0).astype(np.float32)
    return train_scaled, test_scaled


def permutation_summary(observed: float, null: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(null, dtype=np.float64)
    std = float(values.std())
    return {
        "observed": float(observed),
        "null_mean": float(values.mean()),
        "null_std": std,
        "z": float((observed - values.mean()) / max(std, 1e-12)),
        "p_positive": float((1 + np.sum(values >= observed)) / (1 + len(values))),
        "p_negative": float((1 + np.sum(values <= observed)) / (1 + len(values))),
        "p_two_sided": float(
            min(
                1.0,
                2.0
                * min(
                    (1 + np.sum(values >= observed)) / (1 + len(values)),
                    (1 + np.sum(values <= observed)) / (1 + len(values)),
                ),
            )
        ),
        "permutations": int(len(values)),
    }


def complex_residual_metrics(
    valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    base = np.load(
        PATHS.shared_features / "train_v125eq_crossfit_base_ad.npy", mmap_mode="r"
    )
    residual = np.load(
        PATHS.shared_features / "train_v125eq_crossfit_residual_ad.npy", mmap_mode="r"
    )
    rng = np.random.default_rng(RANDOM_STATE)
    feature_count = int(np.prod(base.shape[1:]))
    sketch_indices = np.sort(rng.choice(feature_count, size=512, replace=False))
    n = len(base)
    base_power = np.empty(n, dtype=np.float64)
    residual_power = np.empty(n, dtype=np.float64)
    truth_power = np.empty(n, dtype=np.float64)
    inner_base_residual = np.empty(n, dtype=np.complex128)
    sketch = np.empty((n, len(sketch_indices)), dtype=np.complex64)
    for start in range(0, n, 20):
        end = min(start + 20, n)
        b = np.asarray(base[start:end]).reshape(end - start, -1)
        r = np.asarray(residual[start:end]).reshape(end - start, -1)
        t = b + r
        base_power[start:end] = np.sum(np.abs(b) ** 2, axis=1, dtype=np.float64)
        residual_power[start:end] = np.sum(
            np.abs(r) ** 2, axis=1, dtype=np.float64
        )
        truth_power[start:end] = np.sum(np.abs(t) ** 2, axis=1, dtype=np.float64)
        inner_base_residual[start:end] = np.sum(
            np.conj(b) * r, axis=1, dtype=np.complex128
        )
        sketch[start:end] = r[:, sketch_indices]

    inner_base_truth = base_power + inner_base_residual
    coherence = np.abs(inner_base_truth) / np.sqrt(
        np.maximum(base_power * truth_power, 1e-30)
    )
    residual_nmse = residual_power / np.maximum(truth_power, 1e-30)
    residual_to_base = residual_power / np.maximum(base_power, 1e-30)
    residual_alignment = np.real(inner_base_residual) / np.sqrt(
        np.maximum(base_power * residual_power, 1e-30)
    )
    sketch_valid = normalize_complex_rows(sketch[valid])
    report = {
        "source": "v125-equivalent cross-fit complex AD base/residual",
        "sketch_components": int(len(sketch_indices)),
        "residual_nmse": summarize(residual_nmse[valid]),
        "residual_to_base_power": summarize(residual_to_base[valid]),
        "base_truth_coherence": summarize(coherence[valid]),
        "signed_base_residual_alignment": summarize(residual_alignment[valid]),
    }
    return (
        {
            "residual_nmse": residual_nmse.astype(np.float32),
            "residual_to_base": residual_to_base.astype(np.float32),
            "base_truth_coherence": coherence.astype(np.float32),
            "base_residual_alignment": residual_alignment.astype(np.float32),
            "truth_power_ad": truth_power.astype(np.float64),
        },
        sketch_valid,
        report,
    )


def zero_cluster_test(
    positions: np.ndarray, zero: np.ndarray, rng: np.random.Generator
) -> dict[str, object]:
    selected = np.flatnonzero(zero)
    if len(selected) < 2:
        return {"count": int(len(selected))}

    def mean_nearest(indices: np.ndarray) -> float:
        tree = cKDTree(positions[indices, :2])
        distance, _ = tree.query(positions[indices, :2], k=2)
        return float(distance[:, 1].mean())

    observed = mean_nearest(selected)
    null = np.empty(2000, dtype=np.float64)
    for index in range(len(null)):
        null[index] = mean_nearest(
            rng.choice(len(positions), size=len(selected), replace=False)
        )
    return {
        "count": int(len(selected)),
        "mean_nearest_distance": permutation_summary(observed, null),
        "indices": selected.tolist(),
    }


def distance_dependence(
    positions: np.ndarray,
    regimes: np.ndarray,
    axial_corr: np.ndarray,
    complex_corr_real: np.ndarray,
    complex_corr_abs: np.ndarray,
    truth_corr: np.ndarray,
    badness: np.ndarray,
    log_power: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    n = len(positions)
    pair_i, pair_j = np.triu_indices(n, 1)
    delta = positions[pair_i, :2] - positions[pair_j, :2]
    distance = np.linalg.norm(delta, axis=1)
    rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    permutations = [rng.permutation(n) for _ in range(PERMUTATIONS)]
    for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        selected = np.flatnonzero((distance >= low) & (distance < high))
        if not len(selected):
            continue
        sample = selected
        if len(sample) > 20000:
            sample = rng.choice(sample, size=20000, replace=False)
        i = pair_i[sample]
        j = pair_j[sample]
        observed = float(axial_corr[i, j].mean())
        null = np.asarray(
            [axial_corr[p[i], p[j]].mean() for p in permutations], dtype=np.float64
        )
        row = {
            "distance_low": float(low),
            "distance_high": None if not np.isfinite(high) else float(high),
            "count": int(len(selected)),
            "sample_count": int(len(sample)),
            "distance": summarize(distance[sample]),
            "axial_residual_cosine": permutation_summary(observed, null),
            "complex_residual_real_correlation": float(complex_corr_real[i, j].mean()),
            "complex_residual_absolute_coherence": float(complex_corr_abs[i, j].mean()),
            "truth_axial_cosine": float(truth_corr[i, j].mean()),
            "badness_semivariance": float(
                0.5 * np.mean((badness[i] - badness[j]) ** 2)
            ),
            "log_power_semivariance": float(
                0.5 * np.mean((log_power[i] - log_power[j]) ** 2)
            ),
            "same_regime_fraction": float(np.mean(regimes[i] == regimes[j])),
        }
        rows.append(row)
        csv_rows.append(
            {
                "distance_low": low,
                "distance_high": high,
                "count": len(selected),
                "axial_residual_cosine": observed,
                "axial_null_mean": float(null.mean()),
                "axial_z": row["axial_residual_cosine"]["z"],
                "axial_p_positive": row["axial_residual_cosine"]["p_positive"],
                "complex_residual_real": row["complex_residual_real_correlation"],
                "complex_residual_abs": row["complex_residual_absolute_coherence"],
                "truth_axial_cosine": row["truth_axial_cosine"],
                "badness_semivariance": row["badness_semivariance"],
                "log_power_semivariance": row["log_power_semivariance"],
                "same_regime_fraction": row["same_regime_fraction"],
            }
        )

    bearing = np.degrees(np.arctan2(np.abs(delta[:, 1]), np.abs(delta[:, 0])))
    anisotropy = []
    local = (distance >= 0.5) & (distance < 12.0)
    for low, high in zip((0, 15, 30, 45, 60, 75), (15, 30, 45, 60, 75, 90.0001)):
        selected = np.flatnonzero(local & (bearing >= low) & (bearing < high))
        if not len(selected):
            continue
        anisotropy.append(
            {
                "bearing_low": low,
                "bearing_high": high,
                "count": int(len(selected)),
                "distance": summarize(distance[selected]),
                "axial_residual_cosine": float(
                    axial_corr[pair_i[selected], pair_j[selected]].mean()
                ),
                "badness_semivariance": float(
                    0.5
                    * np.mean(
                        (badness[pair_i[selected]] - badness[pair_j[selected]]) ** 2
                    )
                ),
            }
        )
    return rows, anisotropy, csv_rows


def moran_test(
    values: np.ndarray,
    neighbor_indices: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x - x.mean()
    rows = np.repeat(np.arange(len(x)), neighbor_indices.shape[1])
    cols = neighbor_indices.reshape(-1)
    denominator = np.sum(x * x)

    def statistic(current: np.ndarray) -> float:
        return float(
            len(current)
            * np.sum(current[rows] * current[cols])
            / max(len(rows) * np.sum(current * current), 1e-30)
        )

    observed = statistic(x)
    null = np.asarray(
        [statistic(x[rng.permutation(len(x))]) for _ in range(PERMUTATIONS)]
    )
    return permutation_summary(observed, null)


def neighbor_space_analysis(
    embeddings: dict[str, np.ndarray],
    axial_corr: np.ndarray,
    complex_corr_real: np.ndarray,
    complex_corr_abs: np.ndarray,
    truth_corr: np.ndarray,
    badness: np.ndarray,
    regimes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    neighbor_cache: dict[str, np.ndarray] = {}
    permutations = [rng.permutation(len(badness)) for _ in range(PERMUTATIONS)]
    for name, raw in embeddings.items():
        scaled, _ = robust_scale(raw)
        model = NearestNeighbors(n_neighbors=max(K_VALUES) + 1, n_jobs=-1)
        model.fit(scaled)
        distance, indices = model.kneighbors(scaled)
        indices = indices[:, 1:]
        distance = distance[:, 1:]
        neighbor_cache[name] = indices
        by_k = {}
        for k in K_VALUES:
            rows = np.repeat(np.arange(len(raw)), k)
            cols = indices[:, :k].reshape(-1)
            observed = float(axial_corr[rows, cols].mean())
            null = np.asarray(
                [axial_corr[p[rows], p[cols]].mean() for p in permutations]
            )
            bad_corr = safe_corr(badness[rows], badness[cols])
            payload = {
                "neighbor_distance": summarize(distance[:, :k]),
                "axial_residual_cosine": permutation_summary(observed, null),
                "complex_residual_real_correlation": float(
                    complex_corr_real[rows, cols].mean()
                ),
                "complex_residual_absolute_coherence": float(
                    complex_corr_abs[rows, cols].mean()
                ),
                "truth_axial_cosine": float(truth_corr[rows, cols].mean()),
                "badness_neighbor_correlation": bad_corr,
                "same_regime_fraction": float(np.mean(regimes[rows] == regimes[cols])),
            }
            by_k[str(k)] = payload
            csv_rows.append(
                {
                    "space": name,
                    "k": k,
                    "axial_residual_cosine": observed,
                    "axial_null_mean": float(null.mean()),
                    "axial_z": payload["axial_residual_cosine"]["z"],
                    "axial_p_positive": payload["axial_residual_cosine"]["p_positive"],
                    "complex_residual_real": payload[
                        "complex_residual_real_correlation"
                    ],
                    "complex_residual_abs": payload[
                        "complex_residual_absolute_coherence"
                    ],
                    "truth_axial_cosine": payload["truth_axial_cosine"],
                    "badness_pearson": bad_corr["pearson"],
                    "same_regime_fraction": payload["same_regime_fraction"],
                }
            )
        report[name] = {"feature_dim": int(raw.shape[1]), "by_k": by_k}
    return report, csv_rows, neighbor_cache


def axis_autocorrelation(
    values: np.ndarray,
    lags: tuple[int, ...],
    cyclic: bool,
    center_rows: bool = False,
) -> dict[str, object]:
    if center_rows:
        values = values - values.mean(axis=1, keepdims=True)
    result = {}
    width = values.shape[1]
    for lag in lags:
        correlations = []
        pairs = range(width) if cyclic else range(width - lag)
        for left in pairs:
            right = (left + lag) % width
            correlations.append(safe_corr(values[:, left], values[:, right])["pearson"])
        result[str(lag)] = summarize(np.asarray(correlations))
    return result


def pca_structure(
    values: np.ndarray,
    positions: np.ndarray,
    spatial_neighbors: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, vt = randomized_svd(
        centered, n_components=24, random_state=RANDOM_STATE, n_iter=5
    )
    total = float(np.sum(centered * centered, dtype=np.float64))
    explained = singular.astype(np.float64) ** 2 / max(total, 1e-30)
    scores = centered @ vt[:8].T

    null = np.array(centered, copy=True)
    for column in range(null.shape[1]):
        rng.shuffle(null[:, column])
    _, null_singular, _ = randomized_svd(
        null, n_components=24, random_state=RANDOM_STATE + 1, n_iter=5
    )
    null_total = float(np.sum(null * null, dtype=np.float64))
    null_explained = null_singular.astype(np.float64) ** 2 / max(null_total, 1e-30)
    score_moran = {
        f"pc{index + 1}": moran_test(scores[:, index], spatial_neighbors, rng)
        for index in range(min(5, scores.shape[1]))
    }
    return (
        {
            "shape": list(values.shape),
            "explained_variance_ratio_top24": explained.tolist(),
            "cumulative_top1": float(explained[:1].sum()),
            "cumulative_top5": float(explained[:5].sum()),
            "cumulative_top10": float(explained[:10].sum()),
            "cumulative_top24": float(explained.sum()),
            "permuted_null_top24": null_explained.tolist(),
            "top1_over_null": float(explained[0] / max(null_explained[0], 1e-12)),
            "top5_over_null": float(
                explained[:5].sum() / max(null_explained[:5].sum(), 1e-12)
            ),
            "score_spatial_moran": score_moran,
        },
        {"components": vt, "scores": scores},
    )


def oof_neighbor_transfer(
    embeddings: dict[str, np.ndarray],
    current: np.ndarray,
    truth: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    selected = np.flatnonzero(valid)
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions[:, :2])
    axes = {"H": 0.25, "V": 0.50, "D": 0.50}
    targets = {}
    current_marginals = {}
    truth_marginals = {}
    for axis in axes:
        current_marginals[axis] = marginal(current, axis)
        truth_marginals[axis] = marginal(truth, axis)
        targets[axis] = np.clip(
            np.log(np.maximum(truth_marginals[axis], 1e-10))
            - np.log(np.maximum(current_marginals[axis], 1e-10)),
            -3.0,
            3.0,
        ).astype(np.float32)

    predictions = {
        name: {
            k: {axis: np.zeros_like(targets[axis]) for axis in axes}
            for k in K_VALUES
        }
        for name in embeddings
    }
    splitter = GroupKFold(n_splits=5)
    for train_local, validation_local in splitter.split(
        selected, groups=groups[selected]
    ):
        train_index = selected[train_local]
        validation_index = selected[validation_local]
        for name, raw in embeddings.items():
            train_scaled, validation_scaled = robust_scale(
                raw[train_index], raw[validation_index]
            )
            model = NearestNeighbors(n_neighbors=max(K_VALUES), n_jobs=-1)
            model.fit(train_scaled)
            distance, local = model.kneighbors(validation_scaled)
            neighbors = train_index[local]
            inverse = 1.0 / np.maximum(distance, 0.25)
            for k in K_VALUES:
                weight = inverse[:, :k]
                weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
                for axis in axes:
                    predictions[name][k][axis][validation_index] = np.sum(
                        weight[..., None] * targets[axis][neighbors[:, :k]], axis=1
                    )

    report: dict[str, object] = {}
    best: dict[str, tuple[float, str, int, np.ndarray]] = {}
    for name in embeddings:
        by_k = {}
        for k in K_VALUES:
            by_axis = {}
            for axis, strength in axes.items():
                pred = predictions[name][k][axis]
                adjusted = current_marginals[axis] * np.exp(
                    np.clip(pred, -3.0, 3.0) * strength
                )
                adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), TINY)
                base_score = row_cos(
                    current_marginals[axis], truth_marginals[axis]
                )
                adjusted_score = row_cos(adjusted, truth_marginals[axis])
                gain = adjusted_score - base_score
                corr = safe_corr(pred[valid], targets[axis][valid])
                payload = {
                    "strength": strength,
                    "target_prediction_correlation": corr,
                    "marginal_cosine_gain": summarize(gain[valid]),
                    "positive_fraction": float(np.mean(gain[valid] > 0)),
                }
                by_axis[axis] = payload
                mean_gain = float(np.mean(gain[valid]))
                if axis not in best or mean_gain > best[axis][0]:
                    best[axis] = (mean_gain, name, k, pred)
            by_k[str(k)] = by_axis
        report[name] = by_k

    best_report = {}
    for axis, (marginal_gain, name, k, pred) in best.items():
        strength = axes[axis]
        if axis == "H":
            ratio = np.exp(np.clip(pred, -3.0, 3.0) * strength)[:, :, None, None]
        elif axis == "V":
            ratio = np.exp(np.clip(pred, -3.0, 3.0) * strength)[:, None, :, None]
        else:
            ratio = np.exp(np.clip(pred, -3.0, 3.0) * strength)[:, None, None, :]
        adjusted = (current * ratio).astype(np.float32)
        adjusted /= np.maximum(adjusted.sum(axis=(1, 2, 3), keepdims=True), TINY)
        full_gain = support_score(adjusted, truth) - support_score(current, truth)
        best_report[axis] = {
            "space": name,
            "k": k,
            "marginal_gain_mean": marginal_gain,
            "full_support_gain": summarize(full_gain[valid]),
            "full_support_positive_fraction": float(np.mean(full_gain[valid] > 0)),
        }
    return {"all": report, "best_by_axis": best_report}


def write_rows(path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = height["train_blocked"].astype(np.float32)
    regimes_all = regime_of(blocked)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(
        PATHS.shared_features / "official_pointcloud_scene_cache.npz"
    )
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    probability = np.load(
        PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
    ).astype(np.float32)

    source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    raw_truth = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    truth_power = raw_truth.sum(axis=(1, 2, 3)).astype(np.float64)
    valid = truth_power > 0
    truth = normalize_without_floor(raw_truth)
    risk = build_risk(
        regimes_all,
        probability,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    current, _ = apply_risk_shift(source, risk)

    residual = (truth - current).astype(np.float32)
    h_residual = marginal(truth, "H") - marginal(current, "H")
    v_residual = marginal(truth, "V") - marginal(current, "V")
    d_residual = marginal(truth, "D") - marginal(current, "D")
    h_log_ratio = np.clip(
        np.log(np.maximum(marginal(truth, "H"), 1e-10))
        - np.log(np.maximum(marginal(current, "H"), 1e-10)),
        -3.0,
        3.0,
    )
    v_log_ratio = np.clip(
        np.log(np.maximum(marginal(truth, "V"), 1e-10))
        - np.log(np.maximum(marginal(current, "V"), 1e-10)),
        -3.0,
        3.0,
    )
    d_log_ratio = np.clip(
        np.log(np.maximum(marginal(truth, "D"), 1e-10))
        - np.log(np.maximum(marginal(current, "D"), 1e-10)),
        -3.0,
        3.0,
    )

    score = support_score(current, truth)
    badness = 1.0 - score
    residual_l1 = np.sum(np.abs(residual), axis=(1, 2, 3))
    residual_l2 = np.sqrt(np.sum(residual * residual, axis=(1, 2, 3)))
    h_l1 = np.sum(np.abs(h_residual), axis=1)
    v_l1 = np.sum(np.abs(v_residual), axis=1)
    d_l1 = np.sum(np.abs(d_residual), axis=1)
    js = js_divergence(current, truth)
    truth_entropy = entropy(truth)
    current_entropy = entropy(current)
    prob_entropy = probability_entropy(probability)
    log_power = np.log10(np.maximum(truth_power, 1e-20))

    complex_metrics, complex_sketch, complex_report = complex_residual_metrics(valid)
    valid_index = np.flatnonzero(valid)
    positions_valid = positions[valid]
    regimes = regimes_all[valid]
    badness_valid = badness[valid]
    log_power_valid = log_power[valid]

    axial_residual = np.concatenate(
        [h_residual[valid], v_residual[valid], d_residual[valid]], axis=1
    )
    axial_norm = normalize_rows(axial_residual)
    axial_corr = axial_norm @ axial_norm.T
    truth_axial = normalize_rows(
        np.concatenate(
            [marginal(truth, axis)[valid] for axis in ("H", "V", "D")], axis=1
        )
    )
    truth_corr = truth_axial @ truth_axial.T
    complex_inner = complex_sketch @ complex_sketch.conj().T
    complex_corr_real = np.real(complex_inner).astype(np.float32)
    complex_corr_abs = np.abs(complex_inner).astype(np.float32)

    basic, _ = basic_features(positions, blocked)
    geometry = np.concatenate(
        [
            basic,
            geodesic["train_features"].astype(np.float32),
            pointcloud["train_features"].astype(np.float32),
        ],
        axis=1,
    )
    geometry = np.nan_to_num(
        geometry, nan=0.0, posinf=8.0, neginf=-8.0
    ).astype(np.float32)
    signature = np.nan_to_num(
        support_features(current), nan=0.0, posinf=8.0, neginf=-8.0
    ).astype(np.float32)
    full_features = np.concatenate(
        [geometry, probability, signature], axis=1
    ).astype(np.float32)
    embeddings = {
        "xy": positions_valid[:, :2],
        "official_geometry": geometry[valid],
        "matched_residual_probability": probability[valid],
        "current_support_signature": signature[valid],
        "full_519": full_features[valid],
    }

    distance_rows, anisotropy, distance_csv = distance_dependence(
        positions_valid,
        regimes,
        axial_corr,
        complex_corr_real,
        complex_corr_abs,
        truth_corr,
        badness_valid,
        log_power_valid,
        rng,
    )
    neighbor_report, neighbor_csv, neighbor_cache = neighbor_space_analysis(
        embeddings,
        axial_corr,
        complex_corr_real,
        complex_corr_abs,
        truth_corr,
        badness_valid,
        regimes,
        rng,
    )
    spatial_neighbors = neighbor_cache["xy"][:, :8]

    scalar_metrics = {
        "badness": badness_valid,
        "support_residual_l1": residual_l1[valid],
        "support_residual_l2": residual_l2[valid],
        "h_marginal_l1": h_l1[valid],
        "v_marginal_l1": v_l1[valid],
        "delay_marginal_l1": d_l1[valid],
        "js_divergence": js[valid],
        "complex_residual_nmse": complex_metrics["residual_nmse"][valid],
        "log10_truth_power": log_power_valid,
    }
    moran = {
        name: moran_test(values, spatial_neighbors, rng)
        for name, values in scalar_metrics.items()
    }

    coarse = residual[valid].reshape(len(valid_index), 16, 8, 24, 8).sum(axis=4)
    coarse_flat = coarse.reshape(len(coarse), -1)
    coarse_shape = normalize_rows(coarse_flat)
    pca_report, pca_arrays = pca_structure(
        coarse_shape, positions_valid, spatial_neighbors, rng
    )
    np.savez_compressed(
        MODES_OUT,
        components=pca_arrays["components"].astype(np.float32),
        scores=pca_arrays["scores"].astype(np.float32),
        valid_indices=valid_index.astype(np.int32),
        coarse_shape=np.asarray([16, 8, 24], dtype=np.int32),
    )

    covariates = {
        "x": positions[:, 0],
        "y": positions[:, 1],
        "distance_bs": np.linalg.norm(
            positions[:, :2] - np.asarray([50.0, 0.0], dtype=np.float32), axis=1
        ),
        "blocked": blocked,
        "geodesic_risk": risk,
        "log10_truth_power": log_power,
        "truth_entropy": truth_entropy,
        "current_entropy": current_entropy,
        "residual_probability_entropy": prob_entropy,
        "residual_probability_top1": probability.max(axis=1),
    }
    heteroscedasticity = []
    for metric_name, metric in {
        "badness": badness,
        "support_residual_l1": residual_l1,
        "v_marginal_l1": v_l1,
        "complex_residual_nmse": complex_metrics["residual_nmse"],
    }.items():
        for covariate_name, covariate in covariates.items():
            correlations = safe_corr(metric[valid], covariate[valid])
            heteroscedasticity.append(
                {
                    "metric": metric_name,
                    "covariate": covariate_name,
                    **correlations,
                }
            )
    heteroscedasticity.sort(key=lambda row: -abs(float(row["spearman"])))

    regime_report = {}
    for regime in ("clear", "light", "medium", "heavy"):
        selected = valid & (regimes_all == regime)
        regime_report[regime] = {
            "count": int(selected.sum()),
            **{
                name: summarize(values[selected])
                for name, values in {
                    "badness": badness,
                    "support_residual_l1": residual_l1,
                    "v_marginal_l1": v_l1,
                    "complex_residual_nmse": complex_metrics["residual_nmse"],
                    "truth_power": truth_power,
                }.items()
            },
        }

    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions[:, :2])
    predictability = {
        "badness": regression_oof(full_features, badness, groups, valid),
        "support_residual_l1": regression_oof(
            full_features, residual_l1, groups, valid
        ),
        "complex_residual_nmse_log": regression_oof(
            full_features,
            np.log1p(np.maximum(complex_metrics["residual_nmse"], 0.0)),
            groups,
            valid,
        ),
    }
    transfer_report = oof_neighbor_transfer(
        {
            "xy": positions[:, :2],
            "official_geometry": geometry,
            "matched_residual_probability": probability,
            "current_support_signature": signature,
            "full_519": full_features,
        },
        current,
        truth,
        positions,
        valid,
    )

    zero = truth_power == 0
    tree = cKDTree(positions[:, :2])
    nearest_distance, _ = tree.query(positions[:, :2], k=2)
    data_quality = {
        "points": int(len(positions)),
        "unique_xyz": int(len(np.unique(positions, axis=0))),
        "unique_xy": int(len(np.unique(positions[:, :2], axis=0))),
        "nearest_neighbor_distance": summarize(nearest_distance[:, 1]),
        "truth_power": summarize(truth_power),
        "exact_zero": zero_cluster_test(positions, zero, rng),
        "identifiability_limit": (
            "there is one observation per position and no duplicate coordinates; "
            "the analyzed residual mixes model bias, unmodeled propagation, and "
            "measurement noise, so pure aleatoric noise is not separately identifiable"
        ),
    }

    point_rows = []
    for index in range(len(positions)):
        point_rows.append(
            {
                "index": index,
                "x": float(positions[index, 0]),
                "y": float(positions[index, 1]),
                "regime": str(regimes_all[index]),
                "valid_truth": int(valid[index]),
                "truth_power": float(truth_power[index]),
                "log10_truth_power": float(log_power[index]),
                "geodesic_risk": float(risk[index]),
                "support_score": float(score[index]),
                "badness": float(badness[index]),
                "support_residual_l1": float(residual_l1[index]),
                "support_residual_l2": float(residual_l2[index]),
                "h_marginal_l1": float(h_l1[index]),
                "v_marginal_l1": float(v_l1[index]),
                "delay_marginal_l1": float(d_l1[index]),
                "js_divergence": float(js[index]),
                "complex_residual_nmse": float(
                    complex_metrics["residual_nmse"][index]
                ),
                "base_truth_coherence": float(
                    complex_metrics["base_truth_coherence"][index]
                ),
                "truth_entropy": float(truth_entropy[index]),
                "current_entropy": float(current_entropy[index]),
                "residual_probability_entropy": float(prob_entropy[index]),
            }
        )
    write_rows(POINT_CSV, point_rows)
    write_rows(DISTANCE_CSV, distance_csv)
    write_rows(NEIGHBOR_CSV, neighbor_csv)

    point_distributions = {}
    for name, values in {
        "support_score": score[valid],
        "badness": badness[valid],
        "support_residual_l1": residual_l1[valid],
        "support_residual_l2": residual_l2[valid],
        "h_marginal_l1": h_l1[valid],
        "v_marginal_l1": v_l1[valid],
        "delay_marginal_l1": d_l1[valid],
        "js_divergence": js[valid],
    }.items():
        point_distributions[name] = {
            **summarize(values),
            "skew": float(skew(values)),
            "excess_kurtosis": float(kurtosis(values)),
        }

    axis_metric_correlation = {}
    axis_metrics = {"H": h_l1[valid], "V": v_l1[valid], "D": d_l1[valid]}
    for left in axis_metrics:
        for right in axis_metrics:
            if left < right:
                axis_metric_correlation[f"{left}_{right}"] = safe_corr(
                    axis_metrics[left], axis_metrics[right]
                )

    report = {
        "method": "v144_multidimensional_noise_correlation_analysis",
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "post-v140 residuals are structured across space, propagation-feature "
            "neighborhoods, and H/V/delay components rather than iid noise"
        ),
        "residual_definition": (
            "truth normalized collapsed AD support minus matched-risk v140 train "
            "analogue; complex diagnostics use v125-equivalent cross-fit AD residual"
        ),
        "data_quality": data_quality,
        "point_error_distributions": point_distributions,
        "complex_residual": complex_report,
        "regime_heteroscedasticity": regime_report,
        "top_covariate_correlations": heteroscedasticity[:30],
        "spatial_dependence": {
            "distance_curve": distance_rows,
            "local_anisotropy_0p5_to_12m": anisotropy,
            "moran_k8_xy": moran,
        },
        "neighbor_spaces": neighbor_report,
        "component_structure": {
            "axis_error_correlation": axis_metric_correlation,
            "H_log_ratio_autocorrelation_raw": axis_autocorrelation(
                h_log_ratio[valid], (1, 2, 4, 8), True
            ),
            "H_log_ratio_autocorrelation_row_centered": axis_autocorrelation(
                h_log_ratio[valid], (1, 2, 4, 8), True, center_rows=True
            ),
            "V_log_ratio_autocorrelation_raw": axis_autocorrelation(
                v_log_ratio[valid], (1, 2, 4), True
            ),
            "V_log_ratio_autocorrelation_row_centered": axis_autocorrelation(
                v_log_ratio[valid], (1, 2, 4), True, center_rows=True
            ),
            "D_log_ratio_autocorrelation_raw": axis_autocorrelation(
                d_log_ratio[valid], (1, 2, 4, 8, 16, 32, 64), False
            ),
            "D_log_ratio_autocorrelation_row_centered": axis_autocorrelation(
                d_log_ratio[valid],
                (1, 2, 4, 8, 16, 32, 64),
                False,
                center_rows=True,
            ),
            "coarse_joint_residual_shape_pca": pca_report,
        },
        "spatial_oof_predictability": predictability,
        "spatial_oof_neighbor_transfer": transfer_report,
        "actionable_conclusions": [
            (
                "Pure measurement noise is not identifiable because every position "
                "has one observation and there are no duplicate coordinates; all "
                "claims concern the mixed cross-fit residual process."
            ),
            (
                "Normalized axial support residuals are strongly correlated below "
                "12 m, remain weakly above the permutation baseline at 12-20 m, "
                "and are effectively at the null by 20-35 m."
            ),
            (
                "Residual magnitude is a spatially structured heteroscedastic field: "
                "support-residual L1 spatial OOF Spearman is 0.618 and badness "
                "Spearman is 0.531; gating by predicted error magnitude is more "
                "reliable than unconditional residual replacement."
            ),
            (
                "H/V/delay errors share a common point-difficulty factor, but row-"
                "centered axis autocorrelation also remains, especially for delay; "
                "the residual is functional/low-dimensional rather than independent "
                "per-bin noise."
            ),
            (
                "Simple spatial or feature-space neighbor transfer is useful only "
                "for a weak H correction in this audit (best full support gain "
                "+0.001207); direct V and delay neighbor averaging is neutral or "
                "negative, so their structure requires conditional nonlinear models."
            ),
            (
                "Complex residuals have high absolute local coherence but signed "
                "correlation changes sign at the shortest distances; direct complex "
                "averaging is unsafe without relative-phase alignment or a learned "
                "phase-gradient model."
            ),
            (
                "The 16 exact-zero points are strongly spatially clustered and mostly "
                "clear-regime, so they are not random isolated corruption; prior "
                "online evidence still forbids deleting them."
            ),
        ],
        "next_stage_priority": [
            "After v144 feedback, test a separately versioned H residual field using spatial/official-geometry neighbors and an error-magnitude gate.",
            "Represent delay residual shape in a smooth low-rank basis, then predict basis coefficients conditionally; do not copy raw neighbor delay ratios.",
            "Use predicted badness/residual magnitude as a selector for any correction, with strict spatial OOF calibration and conservative coverage.",
            "Treat exact-zero clustering as a latent propagation-region feature, not as a data-cleaning deletion rule.",
            "Keep complex phase correction separate from support correction and require relative-phase alignment before any neighbor aggregation.",
        ],
        "outputs": {
            "report": PATHS.relative(REPORT),
            "point_metrics": PATHS.relative(POINT_CSV),
            "distance_curve": PATHS.relative(DISTANCE_CSV),
            "neighbor_spaces": PATHS.relative(NEIGHBOR_CSV),
            "component_modes": PATHS.relative(MODES_OUT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {REPORT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
