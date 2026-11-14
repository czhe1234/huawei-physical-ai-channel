#!/usr/bin/env python3
"""Analyze official train-channel noise proxies and residual distribution.

This is an analysis-only experiment.  The official clean channel and the actual
noise realization are not observed, so the script separates two questions:

1. What does the low delay-domain spectral floor imply about noise variance?
2. Do leave-one-out spatial residuals look like white complex Gaussian noise?

The second residual necessarily contains propagation/model error.  It is used
as a negative control: if it is strongly structured, it must not be interpreted
as the organizer's random noise directly.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


VERSION = 565
REPORT_DIR = PATHS.version_report_dir(VERSION)
FIG_DIR = REPORT_DIR / "figures"
CACHE_DIR = PATHS.cache_dir(VERSION)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v565_train_noise_distribution_analysis.json"
POINT_CSV = REPORT_DIR / "v565_train_noise_point_metrics.csv"
SUMMARY_CSV = REPORT_DIR / "v565_train_noise_regime_summary.csv"
CACHE = CACHE_DIR / "v565_train_noise_metrics.npz"

MP, MH, MV, N_UE, S_SC = 2, 16, 8, 4, 192
N_LINK = MP * MH * MV * N_UE
BATCH = 8
KNN_K = 8
BETA_RAD_PER_M = 140.164
RNG_SEED = 20260729 + VERSION
REGIMES = ("clear", "light", "medium", "heavy")


def setup_plot_style() -> None:
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ):
        path = Path(font_path)
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def safe_log_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)


def safe_pearson_log(x: np.ndarray, y: np.ndarray) -> float:
    mask = safe_log_mask(x, y)
    if int(mask.sum()) < 4:
        return 0.0
    return float(np.corrcoef(np.log10(x[mask]), np.log10(y[mask]))[0, 1])


def safe_spearman_log(x: np.ndarray, y: np.ndarray) -> float:
    mask = safe_log_mask(x, y)
    if int(mask.sum()) < 4:
        return 0.0
    return float(stats.spearmanr(np.log10(x[mask]), np.log10(y[mask])).statistic)


def fit_noise_models(observed_power: np.ndarray, noise_proxy: np.ndarray) -> dict[str, object]:
    mask = safe_log_mask(observed_power, noise_proxy)
    x = np.asarray(observed_power, dtype=np.float64)[mask]
    y = np.asarray(noise_proxy, dtype=np.float64)[mask]
    lx = np.log10(x)
    ly = np.log10(y)
    gamma, intercept = np.polyfit(lx, ly, deg=1)
    proportional_ratio = float(np.median(y / np.maximum(x, TINY)))
    fixed_variance = float(np.median(y))
    power_a = float(10.0**intercept)

    predictions = {
        "fixed": np.full_like(y, fixed_variance),
        "proportional": proportional_ratio * x,
        "power_law": power_a * np.power(x, gamma),
    }
    diagnostics: dict[str, dict[str, float]] = {}
    for name, pred in predictions.items():
        error = np.log10(np.maximum(pred, TINY)) - ly
        diagnostics[name] = {
            "log_rmse": float(np.sqrt(np.mean(error * error))),
            "log_mae": float(np.mean(np.abs(error))),
        }
    return {
        "fit_count": int(mask.sum()),
        "proxy_power_log_pearson": safe_pearson_log(observed_power, noise_proxy),
        "proxy_power_log_spearman": safe_spearman_log(observed_power, noise_proxy),
        "models": {
            "fixed": {"variance": fixed_variance},
            "proportional": {"ratio": proportional_ratio},
            "power_law": {"a": power_a, "gamma": float(gamma)},
        },
        "fit_diagnostics": diagnostics,
    }


def top_fraction(flat_power: np.ndarray, top_k: int) -> np.ndarray:
    if top_k >= flat_power.shape[1]:
        return np.ones(flat_power.shape[0], dtype=np.float32)
    part = np.partition(flat_power, flat_power.shape[1] - top_k, axis=1)[:, -top_k:]
    total = np.sum(flat_power, axis=1, dtype=np.float64)
    return (np.sum(part, axis=1, dtype=np.float64) / np.maximum(total, TINY)).astype(
        np.float32
    )


def compute_tail_noise_metrics(train_h: np.ndarray) -> dict[str, np.ndarray]:
    n = len(train_h)
    observed_mean_power = np.zeros(n, dtype=np.float64)
    total_power = np.zeros(n, dtype=np.float64)
    noise_proxy_cfr = np.zeros(n, dtype=np.float64)
    cir_floor_median = np.zeros(n, dtype=np.float64)
    cir_q25_ratio_median = np.zeros(n, dtype=np.float64)
    cir_q50_over_q25_median = np.zeros(n, dtype=np.float64)
    cir_q90_over_q50_median = np.zeros(n, dtype=np.float64)
    cir_top1_frac = np.zeros(n, dtype=np.float32)
    cir_top16_frac = np.zeros(n, dtype=np.float32)
    cir_top1024_frac = np.zeros(n, dtype=np.float32)

    low_tail_samples: list[np.ndarray] = []
    all_cell_samples: list[np.ndarray] = []
    rng = np.random.default_rng(RNG_SEED)

    quantile_indices = {
        "q10": int(round((S_SC - 1) * 0.10)),
        "q25": int(round((S_SC - 1) * 0.25)),
        "q50": int(round((S_SC - 1) * 0.50)),
        "q90": int(round((S_SC - 1) * 0.90)),
    }
    kth = sorted(quantile_indices.values())
    q25_exp = -math.log(0.75)

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        block = np.asarray(train_h[start:end], dtype=np.complex64)
        power_cfr = np.abs(block).astype(np.float32) ** 2
        total_power[start:end] = np.sum(power_cfr, axis=(1, 2, 3), dtype=np.float64)
        observed_mean_power[start:end] = np.mean(power_cfr, axis=(1, 2, 3), dtype=np.float64)

        cir = np.fft.ifft(block, axis=-1)
        link_power = (
            np.abs(cir.reshape(end - start, N_LINK, S_SC)).astype(np.float32) ** 2
        )
        partitioned = np.partition(link_power, kth, axis=2)
        q10 = partitioned[:, :, quantile_indices["q10"]].astype(np.float64)
        q25 = partitioned[:, :, quantile_indices["q25"]].astype(np.float64)
        q50 = partitioned[:, :, quantile_indices["q50"]].astype(np.float64)
        q90 = partitioned[:, :, quantile_indices["q90"]].astype(np.float64)

        link_floor_cir = q25 / q25_exp
        point_floor_cir = np.median(link_floor_cir, axis=1)
        cir_floor_median[start:end] = point_floor_cir
        noise_proxy_cfr[start:end] = point_floor_cir * S_SC
        cir_q25_ratio_median[start:end] = np.median(
            q25 / np.maximum(np.mean(link_power, axis=2, dtype=np.float64), TINY), axis=1
        )
        cir_q50_over_q25_median[start:end] = np.median(
            q50 / np.maximum(q25, TINY), axis=1
        )
        cir_q90_over_q50_median[start:end] = np.median(
            q90 / np.maximum(q50, TINY), axis=1
        )
        flat_power = link_power.reshape(end - start, -1)
        cir_top1_frac[start:end] = top_fraction(flat_power, 1)
        cir_top16_frac[start:end] = top_fraction(flat_power, 16)
        cir_top1024_frac[start:end] = top_fraction(flat_power, 1024)

        for row in range(end - start):
            if total_power[start + row] <= 0:
                continue
            links = rng.integers(0, N_LINK, size=256)
            delays = rng.integers(0, S_SC, size=256)
            values = link_power[row, links, delays].astype(np.float64)
            sigma = np.maximum(link_floor_cir[row, links], TINY)
            normalized = values / sigma
            all_cell_samples.append(normalized.astype(np.float32))
            low_tail_samples.append(normalized[normalized <= 1.0].astype(np.float32))

        print(f"v565 tail-floor noise metrics {end}/{n}", flush=True)

    return {
        "observed_mean_power": observed_mean_power,
        "total_power": total_power,
        "noise_proxy_cfr": noise_proxy_cfr,
        "cir_floor_median": cir_floor_median,
        "cir_q25_ratio_median": cir_q25_ratio_median,
        "cir_q50_over_q25_median": cir_q50_over_q25_median,
        "cir_q90_over_q50_median": cir_q90_over_q50_median,
        "cir_top1_frac": cir_top1_frac,
        "cir_top16_frac": cir_top16_frac,
        "cir_top1024_frac": cir_top1024_frac,
        "tail_sample_power_over_floor": np.concatenate(low_tail_samples)
        if low_tail_samples
        else np.empty(0, dtype=np.float32),
        "all_sample_power_over_floor": np.concatenate(all_cell_samples)
        if all_cell_samples
        else np.empty(0, dtype=np.float32),
    }


def effective_neighbor_graph(
    pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    xy = pos[:, :2].astype(np.float64)
    nn = NearestNeighbors(n_neighbors=min(96, len(pos)), algorithm="auto")
    nn.fit(xy)
    dist, idx = nn.kneighbors(xy)
    out_idx = np.empty((len(pos), k), dtype=np.int32)
    out_w = np.empty((len(pos), k), dtype=np.float32)
    for i in range(len(pos)):
        candidates = idx[i]
        dxy = dist[i]
        keep = candidates != i
        candidates = candidates[keep]
        dxy = dxy[keep]
        effective = (
            dxy
            + 0.18 * np.abs(blocked[candidates] - blocked[i]).astype(np.float64)
            + 4.0 * (regimes[candidates] != regimes[i]).astype(np.float64)
        )
        order = np.argsort(effective)[:k]
        selected = candidates[order]
        selected_eff = effective[order]
        w = 1.0 / np.maximum(selected_eff, 0.35) ** 2
        w /= np.maximum(np.sum(w), TINY)
        out_idx[i] = selected
        out_w[i] = w.astype(np.float32)
    return out_idx, out_w


def complex_coherence(a: np.ndarray, b: np.ndarray) -> float:
    denom = math.sqrt(
        float(np.sum(np.abs(a) ** 2, dtype=np.float64))
        * float(np.sum(np.abs(b) ** 2, dtype=np.float64))
    )
    if denom <= TINY:
        return 0.0
    return float(abs(np.vdot(a.reshape(-1), b.reshape(-1))) / denom)


def lag1_complex_corr(x: np.ndarray, axis: int = -1) -> float:
    a = np.take(x, indices=range(0, x.shape[axis] - 1), axis=axis)
    b = np.take(x, indices=range(1, x.shape[axis]), axis=axis)
    denom = math.sqrt(
        float(np.sum(np.abs(a) ** 2, dtype=np.float64))
        * float(np.sum(np.abs(b) ** 2, dtype=np.float64))
    )
    if denom <= TINY:
        return 0.0
    return float(np.real(np.vdot(a.reshape(-1), b.reshape(-1))) / denom)


def compute_knn_residual_metrics(
    train_h: np.ndarray,
    pos: np.ndarray,
    idx: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(train_h)
    los_distance = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    residual_power_direct = np.zeros(n, dtype=np.float64)
    residual_power_carrier = np.zeros(n, dtype=np.float64)
    coherence_direct = np.zeros(n, dtype=np.float32)
    coherence_carrier = np.zeros(n, dtype=np.float32)
    carrier_sign = np.zeros(n, dtype=np.int8)
    residual_freq_lag1 = np.zeros(n, dtype=np.float32)
    residual_cir_top1_frac = np.zeros(n, dtype=np.float32)
    residual_cir_top16_frac = np.zeros(n, dtype=np.float32)
    residual_cir_top1024_frac = np.zeros(n, dtype=np.float32)

    real_samples: list[np.ndarray] = []
    imag_samples: list[np.ndarray] = []
    residual_power_samples: list[np.ndarray] = []
    rng = np.random.default_rng(RNG_SEED + 17)

    for i in range(n):
        truth = np.asarray(train_h[i], dtype=np.complex64)
        nbr = np.asarray(train_h[idx[i]], dtype=np.complex64)
        w = weights[i].astype(np.complex64)

        pred_direct = np.einsum("k,kabc->abc", w, nbr, optimize=True).astype(np.complex64)
        delta_distance = los_distance[i] - los_distance[idx[i]]
        phase_plus = np.exp(1j * BETA_RAD_PER_M * delta_distance).astype(np.complex64)
        phase_minus = np.conj(phase_plus)
        pred_plus = np.einsum(
            "k,kabc->abc", w * phase_plus, nbr, optimize=True
        ).astype(np.complex64)
        pred_minus = np.einsum(
            "k,kabc->abc", w * phase_minus, nbr, optimize=True
        ).astype(np.complex64)

        direct_residual = truth - pred_direct
        plus_residual = truth - pred_plus
        minus_residual = truth - pred_minus
        direct_power = float(np.mean(np.abs(direct_residual) ** 2, dtype=np.float64))
        plus_power = float(np.mean(np.abs(plus_residual) ** 2, dtype=np.float64))
        minus_power = float(np.mean(np.abs(minus_residual) ** 2, dtype=np.float64))

        if plus_power <= minus_power:
            best_residual = plus_residual
            best_pred = pred_plus
            best_power = plus_power
            carrier_sign[i] = 1
        else:
            best_residual = minus_residual
            best_pred = pred_minus
            best_power = minus_power
            carrier_sign[i] = -1

        residual_power_direct[i] = direct_power
        residual_power_carrier[i] = best_power
        coherence_direct[i] = complex_coherence(truth, pred_direct)
        coherence_carrier[i] = complex_coherence(truth, best_pred)
        residual_freq_lag1[i] = lag1_complex_corr(best_residual, axis=-1)

        residual_cir = np.fft.ifft(best_residual[None, ...], axis=-1)
        flat_power = (
            np.abs(residual_cir.reshape(1, -1)).astype(np.float32) ** 2
        )
        residual_cir_top1_frac[i] = top_fraction(flat_power, 1)[0]
        residual_cir_top16_frac[i] = top_fraction(flat_power, 16)[0]
        residual_cir_top1024_frac[i] = top_fraction(flat_power, 1024)[0]

        if best_power > 0.0:
            flat = best_residual.reshape(-1)
            sample_idx = rng.integers(0, flat.size, size=384)
            sampled = flat[sample_idx].astype(np.complex64)
            sigma_real = math.sqrt(best_power / 2.0)
            real_samples.append((sampled.real / sigma_real).astype(np.float32))
            imag_samples.append((sampled.imag / sigma_real).astype(np.float32))
            residual_power_samples.append(
                (np.abs(sampled).astype(np.float32) ** 2 / best_power).astype(np.float32)
            )

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"v565 KNN complex residual {i + 1}/{n}", flush=True)

    return {
        "residual_power_direct": residual_power_direct,
        "residual_power_carrier": residual_power_carrier,
        "coherence_direct": coherence_direct,
        "coherence_carrier": coherence_carrier,
        "carrier_sign": carrier_sign,
        "residual_freq_lag1": residual_freq_lag1,
        "residual_cir_top1_frac": residual_cir_top1_frac,
        "residual_cir_top16_frac": residual_cir_top16_frac,
        "residual_cir_top1024_frac": residual_cir_top1024_frac,
        "standard_real": np.concatenate(real_samples)
        if real_samples
        else np.empty(0, dtype=np.float32),
        "standard_imag": np.concatenate(imag_samples)
        if imag_samples
        else np.empty(0, dtype=np.float32),
        "standard_power": np.concatenate(residual_power_samples)
        if residual_power_samples
        else np.empty(0, dtype=np.float32),
    }


def distribution_diagnostics(samples: np.ndarray, reference: str) -> dict[str, object]:
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    if x.size > 200000:
        rng = np.random.default_rng(RNG_SEED + 31)
        x = rng.choice(x, size=200000, replace=False)
    out: dict[str, object] = summarize(x)
    if reference == "normal":
        out.update(
            {
                "skew": float(stats.skew(x)),
                "kurtosis_fisher": float(stats.kurtosis(x, fisher=True)),
                "ks_stat_to_normal": float(stats.kstest(x, "norm").statistic),
            }
        )
    elif reference == "exponential_mean1":
        out.update(
            {
                "mean_expected": 1.0,
                "median_expected": float(math.log(2.0)),
                "p90_expected": float(-math.log(0.1)),
                "ks_stat_to_exponential": float(stats.kstest(x, "expon").statistic),
            }
        )
    return out


def by_group_summary(
    names: list[str],
    masks: list[np.ndarray],
    metrics: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, mask in zip(names, masks):
        row: dict[str, object] = {"group": name, "count": int(np.sum(mask))}
        if not np.any(mask):
            rows.append(row)
            continue
        for key in (
            "observed_mean_power",
            "noise_proxy_cfr",
            "noise_to_power_ratio",
            "residual_power_carrier",
            "residual_to_power_ratio",
            "coherence_carrier",
            "residual_freq_lag1",
            "cir_top1024_frac",
            "residual_cir_top1024_frac",
        ):
            value = np.asarray(metrics[key])
            row[f"{key}_mean"] = float(np.nanmean(value[mask]))
            row[f"{key}_median"] = float(np.nanmedian(value[mask]))
            row[f"{key}_p90"] = float(np.nanquantile(value[mask], 0.90))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_point_csv(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    metrics: dict[str, np.ndarray],
    exact_zero: np.ndarray,
) -> None:
    keys = [
        "observed_mean_power",
        "noise_proxy_cfr",
        "noise_to_power_ratio",
        "cir_top1024_frac",
        "residual_power_carrier",
        "residual_to_power_ratio",
        "coherence_carrier",
        "residual_freq_lag1",
        "residual_cir_top1024_frac",
    ]
    with POINT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["index", "x", "y", "z", "blocked", "regime", "exact_zero"] + keys
        )
        for i in range(len(pos)):
            writer.writerow(
                [
                    i,
                    float(pos[i, 0]),
                    float(pos[i, 1]),
                    float(pos[i, 2]),
                    float(blocked[i]),
                    str(regimes[i]),
                    int(exact_zero[i]),
                ]
                + [float(metrics[key][i]) for key in keys]
            )


def scatter_log_power(metrics: dict[str, np.ndarray], exact_zero: np.ndarray) -> None:
    power = metrics["observed_mean_power"]
    noise = metrics["noise_proxy_cfr"]
    ratio = metrics["noise_to_power_ratio"]
    mask = (~exact_zero) & (power > 0) & (noise > 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(power[mask], noise[mask], s=10, alpha=0.55, c=np.log10(ratio[mask]), cmap="viridis")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("观测平均功率 E|H|²")
    ax.set_ylabel("低时延谱底噪声方差代理")
    ax.set_title("噪声代理随功率变化")
    fit = fit_noise_models(power, noise)["models"]["power_law"]
    xs = np.logspace(np.log10(power[mask].min()), np.log10(power[mask].max()), 200)
    ys = fit["a"] * xs ** fit["gamma"]
    ax.plot(xs, ys, color="crimson", lw=2, label=f"斜率 γ={fit['gamma']:.3f}")
    ax.legend()

    ax = axes[1]
    ax.hist(np.log10(ratio[mask]), bins=70, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("log10(噪声代理 / 观测功率)")
    ax.set_ylabel("训练点数量")
    ax.set_title("相对噪声强度分布")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_功率与噪声方差代理.png")
    plt.close(fig)


def plot_power_distribution(metrics: dict[str, np.ndarray], exact_zero: np.ndarray) -> None:
    power = metrics["observed_mean_power"]
    nonzero = (~exact_zero) & (power > 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log10(power[nonzero]), bins=80, color="#5f8dd3", alpha=0.9)
    ax.axvline(np.log10(np.median(power[nonzero])), color="crimson", lw=2, label="非零中位数")
    ax.set_xlabel("log10(观测平均功率 E|H|²)")
    ax.set_ylabel("训练点数量")
    ax.set_title(f"训练点功率分布；exact-zero={int(exact_zero.sum())}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "00_训练点功率分布.png")
    plt.close(fig)


def plot_residual_structure(metrics: dict[str, np.ndarray], exact_zero: np.ndarray) -> None:
    power = metrics["observed_mean_power"]
    residual = metrics["residual_power_carrier"]
    mask = (~exact_zero) & (power > 0) & (residual > 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(power[mask], residual[mask], s=10, alpha=0.55, c=metrics["coherence_carrier"][mask], cmap="magma")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("观测平均功率 E|H|²")
    ax.set_ylabel("KNN留一复残差功率")
    ax.set_title("空间留一残差随功率变化")

    ax = axes[1]
    ax.scatter(
        metrics["residual_freq_lag1"][mask],
        metrics["residual_cir_top1024_frac"][mask],
        s=10,
        alpha=0.55,
        c=np.log10(power[mask]),
        cmap="viridis",
    )
    ax.set_xlabel("残差相邻频点相关")
    ax.set_ylabel("残差CIR top1024能量占比")
    ax.set_title("残差是否像白噪声")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_空间留一残差结构.png")
    plt.close(fig)


def plot_distribution_checks(metrics: dict[str, np.ndarray]) -> None:
    real = metrics["standard_real"]
    imag = metrics["standard_imag"]
    power = metrics["standard_power"]
    floor_power = metrics["tail_sample_power_over_floor"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    xs = np.linspace(-5, 5, 500)
    ax = axes[0, 0]
    ax.hist(real, bins=120, density=True, alpha=0.65, label="残差实部")
    ax.plot(xs, stats.norm.pdf(xs), color="crimson", lw=2, label="标准正态")
    ax.set_xlim(-5, 5)
    ax.set_title("标准化复残差实部")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(imag, bins=120, density=True, alpha=0.65, label="残差虚部")
    ax.plot(xs, stats.norm.pdf(xs), color="crimson", lw=2, label="标准正态")
    ax.set_xlim(-5, 5)
    ax.set_title("标准化复残差虚部")
    ax.legend()

    ax = axes[1, 0]
    clipped_power = power[np.isfinite(power) & (power <= 8)]
    ax.hist(clipped_power, bins=120, density=True, alpha=0.65, label="|残差|² / 均值")
    xx = np.linspace(0, 8, 500)
    ax.plot(xx, np.exp(-xx), color="crimson", lw=2, label="均值1指数分布")
    ax.set_title("残差功率分布")
    ax.legend()

    ax = axes[1, 1]
    clipped_floor = floor_power[np.isfinite(floor_power) & (floor_power <= 3)]
    ax.hist(clipped_floor, bins=100, density=True, alpha=0.65, label="低谱底样本")
    xx = np.linspace(0, 3, 400)
    ax.plot(xx, np.exp(-xx), color="crimson", lw=2, label="均值1指数分布")
    ax.set_title("低时延谱底样本功率")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_高斯性与指数分布检查.png")
    plt.close(fig)


def main() -> None:
    setup_plot_style()
    train_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geom["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked)

    if CACHE.is_file():
        data = np.load(CACHE)
        metrics = {key: data[key] for key in data.files}
        print(f"reuse {PATHS.relative(CACHE)}", flush=True)
    else:
        tail = compute_tail_noise_metrics(train_h)
        idx, weights = effective_neighbor_graph(pos, blocked, regimes, KNN_K)
        residual = compute_knn_residual_metrics(train_h, pos, idx, weights)
        metrics = {**tail, **residual}
        metrics["neighbor_idx"] = idx
        metrics["neighbor_weights"] = weights
        np.savez_compressed(CACHE, **metrics)
        print(f"saved {PATHS.relative(CACHE)}", flush=True)

    exact_zero = metrics["total_power"] <= 0.0
    nonzero = ~exact_zero
    metrics["noise_to_power_ratio"] = metrics["noise_proxy_cfr"] / np.maximum(
        metrics["observed_mean_power"], TINY
    )
    metrics["residual_to_power_ratio"] = metrics["residual_power_carrier"] / np.maximum(
        metrics["observed_mean_power"], TINY
    )

    group_names = ["all_2000", "nonzero_1984", "exact_zero_16"] + [
        f"regime_{name}" for name in REGIMES
    ]
    group_masks = [
        np.ones(len(pos), dtype=bool),
        nonzero,
        exact_zero,
    ] + [regimes == name for name in REGIMES]
    group_rows = by_group_summary(group_names, group_masks, metrics)
    write_csv(SUMMARY_CSV, group_rows)
    save_point_csv(pos, blocked, regimes, metrics, exact_zero)

    plot_power_distribution(metrics, exact_zero)
    scatter_log_power(metrics, exact_zero)
    plot_residual_structure(metrics, exact_zero)
    plot_distribution_checks(metrics)

    nonzero_fit = fit_noise_models(
        metrics["observed_mean_power"][nonzero],
        metrics["noise_proxy_cfr"][nonzero],
    )
    residual_fit = fit_noise_models(
        metrics["observed_mean_power"][nonzero],
        metrics["residual_power_carrier"][nonzero],
    )

    report = {
        "method": "v565_train_noise_distribution_analysis",
        "status": "analysis_complete_no_submission",
        "inputs": {
            "train_channel": PATHS.relative(PATHS.round1 / "Round1_Train_Channel.npy"),
            "train_pos": PATHS.relative(PATHS.round1 / "Round1_Train_Pos.npy"),
            "geometry_cache": PATHS.relative(PATHS.shared_features / "height_geometry_cache.npz"),
            "shape": list(train_h.shape),
            "dtype": str(train_h.dtype),
        },
        "core_caveat": (
            "Official clean H and random noise realization are not observed. "
            "The low-delay-floor proxy is the closest train-only noise proxy; "
            "KNN complex residuals are propagation/model-error residuals and are "
            "reported mainly to show what must not be called pure white noise."
        ),
        "counts": {
            "all": int(len(pos)),
            "nonzero": int(nonzero.sum()),
            "exact_zero": int(exact_zero.sum()),
            "by_regime": {
                name: int(np.sum(regimes == name)) for name in REGIMES
            },
            "exact_zero_by_regime": {
                name: int(np.sum((regimes == name) & exact_zero)) for name in REGIMES
            },
        },
        "tail_floor_noise_proxy": {
            "estimator": (
                "For each official train point, reshape to 1024 links x 192 frequency bins, "
                "IFFT on the historical scorer axis, estimate each link's exponential "
                "noise floor from q25/(-log(0.75)), take the median across links, "
                "and multiply by 192 to express it as a CFR variance proxy."
            ),
            "fit_vs_observed_power_nonzero": nonzero_fit,
            "noise_to_power_ratio": summarize(metrics["noise_to_power_ratio"][nonzero]),
            "cir_quantile_shape": {
                "q50_over_q25_expected_exponential": float(math.log(2.0) / -math.log(0.75)),
                "q90_over_q50_expected_exponential": float(-math.log(0.10) / math.log(2.0)),
                "q50_over_q25_median": summarize(
                    metrics["cir_q50_over_q25_median"][nonzero]
                ),
                "q90_over_q50_median": summarize(
                    metrics["cir_q90_over_q50_median"][nonzero]
                ),
                "low_tail_power_over_floor": distribution_diagnostics(
                    metrics["tail_sample_power_over_floor"], "exponential_mean1"
                ),
                "all_sample_power_over_floor": distribution_diagnostics(
                    metrics["all_sample_power_over_floor"], "exponential_mean1"
                ),
            },
        },
        "knn_complex_residual": {
            "estimator": (
                f"Leave-one-out {KNN_K}-NN complex average with the better of ±LOS-carrier "
                f"phase rotations at beta={BETA_RAD_PER_M} rad/m. This is not a clean "
                "noise estimator; it measures local prediction residual."
            ),
            "fit_vs_observed_power_nonzero": residual_fit,
            "residual_to_power_ratio": summarize(metrics["residual_to_power_ratio"][nonzero]),
            "coherence_carrier": summarize(metrics["coherence_carrier"][nonzero]),
            "residual_freq_lag1": summarize(metrics["residual_freq_lag1"][nonzero]),
            "residual_cir_top1024_frac": summarize(
                metrics["residual_cir_top1024_frac"][nonzero]
            ),
            "standard_real": distribution_diagnostics(metrics["standard_real"], "normal"),
            "standard_imag": distribution_diagnostics(metrics["standard_imag"], "normal"),
            "standard_power": distribution_diagnostics(
                metrics["standard_power"], "exponential_mean1"
            ),
        },
        "by_group_summary_csv": PATHS.relative(SUMMARY_CSV),
        "point_metrics_csv": PATHS.relative(POINT_CSV),
        "figures": [
            PATHS.relative(path) for path in sorted(FIG_DIR.glob("*.png"))
        ],
        "interpretation": {
            "power_relation": (
                "The train-only low-floor proxy is expected to show whether absolute "
                "noise variance rises with channel power. A gamma close to 1 means "
                "roughly constant relative noise/SNR, not fixed additive variance."
            ),
            "exact_zero": (
                "Exact-zero channels are incompatible with ordinary continuous "
                "complex Gaussian noise and should be treated as explicit erasures or "
                "invalid labels. Their online usefulness still requires submission "
                "ablation, as earlier exact-zero pruning did not consistently help."
            ),
            "residual_structure": (
                "If KNN residuals have high frequency lag correlation or concentrated "
                "CIR support, they are dominated by missing propagation/path structure, "
                "so denoising them as IID Gaussian noise would remove real multipath."
            ),
        },
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["tail_floor_noise_proxy"]["fit_vs_observed_power_nonzero"], ensure_ascii=False, indent=2))
    print(f"wrote {PATHS.relative(REPORT)}", flush=True)


if __name__ == "__main__":
    main()
