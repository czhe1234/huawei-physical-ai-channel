#!/usr/bin/env python3
"""Strictly identify the observable IID complex-white-noise component in train H."""

from __future__ import annotations

import argparse
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
from scipy.integrate import cumulative_trapezoid
import torch

from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v568_strict_noise_identification"
OUT_DIR = PATHS.cache_dir(568)
REPORT_DIR = PATHS.version_report_dir(568)
FIG_DIR = REPORT_DIR / "figures"
REPORT = REPORT_DIR / "v568_strict_noise_identification.json"
POINT_CSV = REPORT_DIR / "v568_official_point_noise_bounds.csv"
METRICS_CACHE = OUT_DIR / "v568_strict_noise_metrics.npz"

OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
V208_CLEAN = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
V208_NOISY = PATHS.cache_dir(208) / "strict_dataset" / "Round1_Train_Channel.npy"
V208_NOISE_META = PATHS.cache_dir(208) / "strict_dataset" / "active_noise_metadata.json"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"

N_POINT = 2000
N_ROW = 256 * 4
N_FREQ = 192
BATCH = 8
MP_FRACTION = 0.50
CROSSFIT_RANKS = (16, 24, 32, 48, 64)
CROSSFIT_SELECTED_RANK = 32
RANK_THRESHOLDS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
SNR_DB_GRID = (15, 20, 25, 30, 40, 60, 80, 100, 110, 120, 130)
TINY = 1e-30


def configure_chinese_font() -> None:
    families = {font.name for font in font_manager.fontManager.ttflist}
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        if family in families:
            selected = family
            break
    plt.rcParams.update(
        {
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute cached spectra.")
    parser.add_argument("--device", default="cuda", help="Torch device; default cuda.")
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return {"count": 0}
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
        "min": float(np.min(value)),
        "p01": float(np.quantile(value, 0.01)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "p99": float(np.quantile(value, 0.99)),
        "max": float(np.max(value)),
    }


def mp_quantiles(rows: int = N_ROW, cols: int = N_FREQ) -> np.ndarray:
    beta = cols / rows
    lower = (1.0 - math.sqrt(beta)) ** 2
    upper = (1.0 + math.sqrt(beta)) ** 2
    grid = np.linspace(lower + 1e-9, upper - 1e-9, 300_000)
    density = np.sqrt((upper - grid) * (grid - lower)) / (
        2.0 * math.pi * beta * grid
    )
    cdf = np.concatenate([[0.0], cumulative_trapezoid(density, grid)])
    cdf /= cdf[-1]
    probability = (np.arange(cols, dtype=np.float64) + 0.5) / cols
    return np.interp(probability, cdf, grid)


def mp_estimate(eigen_relative: np.ndarray, mp_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = max(8, int(round(len(mp_q) * MP_FRACTION)))
    ratio = eigen_relative[:, :count] / np.maximum(mp_q[:count], TINY)
    estimate = np.nanmedian(ratio, axis=1)
    normalized = ratio / np.maximum(estimate[:, None], TINY)
    fit_rmse = np.sqrt(np.nanmean(np.log(np.maximum(normalized, TINY)) ** 2, axis=1))
    return estimate, fit_rmse


def q25_floor_ratio(block: np.ndarray, point_power: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(block, axis=-1).reshape(len(block), N_ROW, N_FREQ)
    cir_power = np.abs(cir).astype(np.float32) ** 2
    q_index = int(round((N_FREQ - 1) * 0.25))
    q25 = np.partition(cir_power, q_index, axis=2)[:, :, q_index]
    link_floor = q25 / (-math.log(0.75))
    floor_cfr = np.median(link_floor, axis=1) * N_FREQ
    return floor_cfr / np.maximum(point_power, TINY)


def dataset_spectral_metrics(
    name: str,
    path: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    channel = np.load(path, mmap_mode="r")
    if channel.shape != (N_POINT, 256, 4, N_FREQ) or channel.dtype != np.complex64:
        raise ValueError(f"Invalid {name}: {channel.shape} {channel.dtype}")

    power = np.zeros(N_POINT, dtype=np.float64)
    eigen_relative = np.full((N_POINT, N_FREQ), np.nan, dtype=np.float64)
    q25_relative = np.full(N_POINT, np.nan, dtype=np.float64)

    for start in range(0, N_POINT, BATCH):
        end = min(start + BATCH, N_POINT)
        raw = np.array(channel[start:end], dtype=np.complex64, copy=True)
        local_power = np.mean(np.abs(raw) ** 2, axis=(1, 2, 3), dtype=np.float64)
        power[start:end] = local_power
        valid = local_power > 0.0
        if np.any(valid):
            target = np.flatnonzero(valid) + start
            q25_relative[target] = q25_floor_ratio(raw[valid], local_power[valid])
            matrix = raw[valid].reshape(-1, N_ROW, N_FREQ)
            matrix /= np.sqrt(local_power[valid])[:, None, None].astype(np.float32)
            singular = torch.linalg.svdvals(torch.from_numpy(matrix).to(device))
            singular = singular.detach().cpu().numpy()[:, ::-1].astype(np.float64)
            eigen_relative[target] = singular**2 / N_ROW
        print(f"v568 spectra {name} {end}/{N_POINT}", flush=True)

    return {
        "power": power,
        "eigen_relative": eigen_relative,
        "q25_relative": q25_relative,
    }


def crossfit_metrics(
    name: str,
    path: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    channel = np.load(path, mmap_mode="r")
    row_grid = np.arange(N_ROW).reshape(2, 16, 8, 4)
    split_a = torch.as_tensor(row_grid[:, :, 0::2, :].reshape(-1), device=device)
    split_b = torch.as_tensor(row_grid[:, :, 1::2, :].reshape(-1), device=device)
    energy = np.full((N_POINT, len(CROSSFIT_RANKS)), np.nan, dtype=np.float64)
    circularity = np.full(N_POINT, np.nan, dtype=np.float64)
    real_imag_ratio = np.full(N_POINT, np.nan, dtype=np.float64)

    selected_index = CROSSFIT_RANKS.index(CROSSFIT_SELECTED_RANK)
    for start in range(0, N_POINT, BATCH):
        end = min(start + BATCH, N_POINT)
        raw = np.array(channel[start:end], dtype=np.complex64, copy=True)
        local_power = np.mean(np.abs(raw) ** 2, axis=(1, 2, 3), dtype=np.float64)
        valid = local_power > 0.0
        if np.any(valid):
            matrix = raw[valid].reshape(-1, N_ROW, N_FREQ)
            matrix /= np.sqrt(local_power[valid])[:, None, None].astype(np.float32)
            tensor = torch.from_numpy(matrix).to(device)
            directional_energy: list[list[torch.Tensor]] = [
                [] for _ in CROSSFIT_RANKS
            ]
            directional_circularity: list[torch.Tensor] = []
            directional_ri: list[torch.Tensor] = []
            for fit, held in ((split_a, split_b), (split_b, split_a)):
                basis = torch.linalg.svd(tensor[:, fit], full_matrices=False).Vh
                coefficient = torch.matmul(tensor[:, held], basis.mH)
                for rank_index, rank in enumerate(CROSSFIT_RANKS):
                    value = torch.mean(torch.abs(coefficient[:, :, rank:]) ** 2, dim=(1, 2))
                    directional_energy[rank_index].append(value)
                selected = coefficient[:, :, CROSSFIT_SELECTED_RANK:]
                selected_power = torch.mean(torch.abs(selected) ** 2, dim=(1, 2))
                pseudo = torch.abs(torch.mean(selected * selected, dim=(1, 2)))
                directional_circularity.append(pseudo / selected_power.clamp_min(1e-30))
                real_var = torch.mean(selected.real**2, dim=(1, 2))
                imag_var = torch.mean(selected.imag**2, dim=(1, 2))
                directional_ri.append(real_var / imag_var.clamp_min(1e-30))

            local_energy = torch.stack(
                [torch.mean(torch.stack(values), dim=0) for values in directional_energy],
                dim=1,
            )
            target = np.flatnonzero(valid) + start
            energy[target] = local_energy.detach().cpu().numpy().astype(np.float64)
            circularity[target] = (
                torch.mean(torch.stack(directional_circularity), dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            real_imag_ratio[target] = (
                torch.mean(torch.stack(directional_ri), dim=0).detach().cpu().numpy()
            )
        print(f"v568 crossfit {name} {end}/{N_POINT}", flush=True)

    return {
        "energy": energy,
        "circularity": circularity,
        "real_imag_ratio": real_imag_ratio,
        "selected_energy": energy[:, selected_index],
    }


def exact_noise_relative(clean_path: Path, noisy_path: Path) -> np.ndarray:
    clean = np.load(clean_path, mmap_mode="r")
    noisy = np.load(noisy_path, mmap_mode="r")
    result = np.full(N_POINT, np.nan, dtype=np.float64)
    for start in range(0, N_POINT, BATCH):
        end = min(start + BATCH, N_POINT)
        clean_block = np.asarray(clean[start:end], dtype=np.complex64)
        noisy_block = np.asarray(noisy[start:end], dtype=np.complex64)
        observed_power = np.mean(np.abs(noisy_block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        noise_power = np.mean(
            np.abs(noisy_block - clean_block) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        valid = observed_power > 0.0
        local = np.full(end - start, np.nan, dtype=np.float64)
        local[valid] = noise_power[valid] / observed_power[valid]
        result[start:end] = local
    return result


def representative_indices(power: np.ndarray, count: int = 48) -> np.ndarray:
    valid = np.flatnonzero(power > 0.0)
    ordered = valid[np.argsort(power[valid])]
    return np.unique(ordered[np.linspace(0, len(ordered) - 1, count, dtype=int)])


def array_mp_metrics(
    value: np.ndarray,
    mp_q: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_power = np.mean(np.abs(value) ** 2, axis=(1, 2), dtype=np.float64)
    normalized = value / np.sqrt(np.maximum(point_power, TINY))[:, None, None].astype(np.float32)
    singular = torch.linalg.svdvals(torch.from_numpy(normalized).to(device))
    singular = singular.detach().cpu().numpy()[:, ::-1].astype(np.float64)
    eigen_relative = singular**2 / N_ROW
    estimate, fit = mp_estimate(eigen_relative, mp_q)
    rank32_residual = np.sum(eigen_relative[:, : N_FREQ - 32], axis=1) / np.sum(
        eigen_relative, axis=1
    )
    return estimate, fit, rank32_residual


def injection_controls(
    clean_path: Path,
    clean_power: np.ndarray,
    mp_q: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clean = np.load(clean_path, mmap_mode="r")
    indices = representative_indices(clean_power)
    base = np.array(clean[indices], dtype=np.complex64, copy=True).reshape(-1, N_ROW, N_FREQ)
    base_power = np.mean(np.abs(base) ** 2, axis=(1, 2), dtype=np.float64)
    rng = np.random.default_rng(568_001)
    snr_rows: list[dict[str, object]] = []

    for snr_db in SNR_DB_GRID:
        target_ratio = 10.0 ** (-snr_db / 10.0)
        real = rng.standard_normal(base.shape, dtype=np.float32)
        imag = rng.standard_normal(base.shape, dtype=np.float32)
        noise = (real + 1j * imag) * np.sqrt(base_power * target_ratio / 2.0)[
            :, None, None
        ].astype(np.float32)
        observed = (base + noise.astype(np.complex64)).astype(np.complex64)
        actual = np.mean(np.abs(noise) ** 2, axis=(1, 2), dtype=np.float64) / np.mean(
            np.abs(observed) ** 2, axis=(1, 2), dtype=np.float64
        )
        estimate, fit, residual = array_mp_metrics(observed, mp_q, device)
        snr_rows.append(
            {
                "snr_db": int(snr_db),
                "target_noise_to_signal": float(target_ratio),
                "actual_noise_to_observed": summarize(actual),
                "mp_estimate": summarize(estimate),
                "estimate_over_actual": summarize(estimate / np.maximum(actual, TINY)),
                "mp_fit_rmse": summarize(fit),
                "rank32_residual_fraction": summarize(residual),
            }
        )
        print(f"v568 injected SNR {snr_db} dB", flush=True)

    ratio = 10.0 ** (-25.0 / 10.0)
    fingerprint_rows: list[dict[str, object]] = []

    def add_fingerprint(name: str, observed: np.ndarray, delta: np.ndarray) -> None:
        actual = np.mean(np.abs(delta) ** 2, axis=(1, 2), dtype=np.float64) / np.mean(
            np.abs(observed) ** 2, axis=(1, 2), dtype=np.float64
        )
        estimate, fit, residual = array_mp_metrics(observed, mp_q, device)
        fingerprint_rows.append(
            {
                "name": name,
                "actual_noise_to_observed": summarize(actual),
                "mp_estimate": summarize(estimate),
                "estimate_over_actual": summarize(estimate / np.maximum(actual, TINY)),
                "mp_fit_rmse": summarize(fit),
                "rank32_residual_fraction": summarize(residual),
            }
        )

    real = rng.standard_normal(base.shape, dtype=np.float32)
    imag = rng.standard_normal(base.shape, dtype=np.float32)
    iid_delta = (real + 1j * imag) * np.sqrt(base_power * ratio / 2.0)[
        :, None, None
    ].astype(np.float32)
    add_fingerprint("iid_additive_per_coefficient", base + iid_delta, iid_delta)

    scalar = (
        rng.standard_normal((len(base), 1, 1), dtype=np.float32)
        + 1j * rng.standard_normal((len(base), 1, 1), dtype=np.float32)
    ) * math.sqrt(ratio / 2.0)
    delta = base * scalar
    add_fingerprint("point_scalar_multiplicative", base + delta, delta)

    link = (
        rng.standard_normal((len(base), N_ROW, 1), dtype=np.float32)
        + 1j * rng.standard_normal((len(base), N_ROW, 1), dtype=np.float32)
    ) * math.sqrt(ratio / 2.0)
    delta = base * link
    add_fingerprint("per_link_multiplicative", base + delta, delta)

    frequency = (
        rng.standard_normal((len(base), 1, N_FREQ), dtype=np.float32)
        + 1j * rng.standard_normal((len(base), 1, N_FREQ), dtype=np.float32)
    ) * math.sqrt(ratio / 2.0)
    delta = base * frequency
    add_fingerprint("per_frequency_multiplicative", base + delta, delta)

    return snr_rows, fingerprint_rows


def distribution_shape(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return {"count": 0}
    log_value = np.log10(np.maximum(value, TINY))
    return {
        **summarize(value),
        "log10_mean": float(np.mean(log_value)),
        "log10_std": float(np.std(log_value)),
        "log10_skew": float(stats.skew(log_value)),
        "log10_kurtosis_fisher": float(stats.kurtosis(log_value)),
    }


def save_point_csv(
    official: dict[str, np.ndarray],
    mp_value: np.ndarray,
    crossfit_value: np.ndarray,
    regimes: np.ndarray,
    blocked: np.ndarray,
) -> None:
    rank = {
        threshold: np.sum(official["eigen_relative"] > threshold, axis=1)
        for threshold in RANK_THRESHOLDS
    }
    residual32 = np.nansum(official["eigen_relative"][:, : N_FREQ - 32], axis=1) / np.maximum(
        np.nansum(official["eigen_relative"], axis=1), TINY
    )
    with POINT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "blocked",
                "regime",
                "mean_power",
                "q25_floor_relative",
                "mp_iid_noise_relative",
                "crossfit_iid_noise_relative",
                "mp_fit_rmse",
                "rank_gt_1e-12",
                "rank_gt_1e-8",
                "rank32_residual_fraction",
            ]
        )
        for index in range(N_POINT):
            writer.writerow(
                [
                    index,
                    int(blocked[index]),
                    regimes[index],
                    float(official["power"][index]),
                    float(official["q25_relative"][index]),
                    float(mp_value[index]),
                    float(crossfit_value[index]),
                    float(official["mp_fit_rmse"][index]),
                    int(rank[1e-12][index]),
                    int(rank[1e-8][index]),
                    float(residual32[index]),
                ]
            )


def make_figures(
    datasets: dict[str, dict[str, np.ndarray]],
    exact_relative: np.ndarray,
    snr_rows: list[dict[str, object]],
    fingerprints: list[dict[str, object]],
    regimes: np.ndarray,
    mp_calibration: float,
    crossfit_calibration: float,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["v208干净", "v208已知25dB", "官方训练H"]
    q25_data = [
        datasets["v208_clean"]["q25_relative"],
        datasets["v208_noisy25"]["q25_relative"],
        datasets["official"]["q25_relative"],
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [np.log10(x[np.isfinite(x)]) for x in q25_data],
        tick_labels=labels,
        showfliers=False,
    )
    ax.axhline(math.log10(10 ** (-25 / 10)), color="black", linestyle="--", label="真实25dB噪声比例")
    ax.set_ylabel("log10(低能量CIR谱底 / 点功率)")
    ax.set_title("v565分位数代理在干净信道中同样存在")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "00_v565代理严格控制.png", dpi=180)
    plt.close(fig)

    target = np.array([row["actual_noise_to_observed"]["median"] for row in snr_rows])
    estimate = np.array([row["mp_estimate"]["median"] / mp_calibration for row in snr_rows])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.loglog(target, estimate, "o-", label="MP估计")
    limit = [min(target.min(), estimate.min()), max(target.max(), estimate.max())]
    ax.loglog(limit, limit, "--", color="black", label="理想 y=x")
    ax.set_xlabel("真实IID噪声功率 / 观测功率")
    ax.set_ylabel("估计IID噪声功率 / 观测功率")
    ax.set_title("已知噪声从15dB到130dB的检出校准")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_随机矩阵白噪声校准.png", dpi=180)
    plt.close(fig)

    mp_values = [
        datasets["v208_clean"]["mp_estimate"] / mp_calibration,
        datasets["v208_noisy25"]["mp_estimate"] / mp_calibration,
        datasets["official"]["mp_estimate"] / mp_calibration,
    ]
    cross_values = [
        datasets["v208_clean"]["crossfit_selected"] / crossfit_calibration,
        datasets["v208_noisy25"]["crossfit_selected"] / crossfit_calibration,
        datasets["official"]["crossfit_selected"] / crossfit_calibration,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, values, title in zip(
        axes,
        (mp_values, cross_values),
        ("完整矩阵MP谱", "跨阵列交叉子空间"),
    ):
        ax.boxplot(
            [np.log10(x[np.isfinite(x)]) for x in values],
            tick_labels=labels,
            showfliers=False,
        )
        ax.axhline(math.log10(np.nanmedian(exact_relative)), color="black", linestyle="--", label="v208真实25dB")
        ax.set_title(title)
        ax.set_ylabel("log10(IID白噪声相对功率)")
        ax.legend()
    fig.suptitle("官方H没有出现逐系数IID白噪声的满秩谱底")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_官方白噪声上界.png", dpi=180)
    plt.close(fig)

    thresholds = np.array(RANK_THRESHOLDS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, label in zip(("v208_clean", "v208_noisy25", "official"), labels):
        rank_median = [
            np.nanmedian(np.sum(datasets[name]["eigen_relative"] > threshold, axis=1))
            for threshold in thresholds
        ]
        axes[0].semilogx(thresholds, rank_median, "o-", label=label)
    axes[0].invert_xaxis()
    axes[0].set_xlabel("相对特征值阈值")
    axes[0].set_ylabel("有效秩中位数")
    axes[0].set_title("完整1024×192信道矩阵有效秩")
    axes[0].legend()
    residual_labels = []
    residual_data = []
    for name, label in zip(("v208_clean", "v208_noisy25", "official"), labels):
        eigen = datasets[name]["eigen_relative"]
        residual = np.nansum(eigen[:, : N_FREQ - 32], axis=1) / np.maximum(
            np.nansum(eigen, axis=1), TINY
        )
        residual_labels.append(label)
        residual_data.append(np.log10(np.maximum(residual[np.isfinite(residual)], TINY)))
    axes[1].boxplot(residual_data, tick_labels=residual_labels, showfliers=False)
    axes[1].set_ylabel("log10(rank-32残余能量比例)")
    axes[1].set_title("IID噪声会留下满秩残余，官方H没有")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_有效秩与低秩残差.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    regime_names = ("clear", "light", "medium", "heavy")
    regime_labels = ("无遮挡", "轻遮挡", "中遮挡", "重遮挡")
    data = []
    for regime in regime_names:
        mask = (regimes == regime) & np.isfinite(datasets["official"]["mp_estimate"])
        data.append(np.log10(datasets["official"]["mp_estimate"][mask] / mp_calibration))
    ax.boxplot(data, tick_labels=regime_labels, showfliers=False)
    ax.set_ylabel("log10(MP可检测IID白噪声相对功率)")
    ax.set_title("各传播区均处于数值精度谱底")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_传播区白噪声上界.png", dpi=180)
    plt.close(fig)

    fingerprint_labels = {
        "iid_additive_per_coefficient": "逐系数IID加性噪声",
        "point_scalar_multiplicative": "逐点标量乘性扰动",
        "per_link_multiplicative": "逐链路乘性扰动",
        "per_frequency_multiplicative": "逐频点乘性扰动",
    }
    names = [fingerprint_labels[row["name"]] for row in fingerprints] + ["官方训练H"]
    values = [row["mp_estimate"]["median"] / mp_calibration for row in fingerprints]
    values.append(float(np.nanmedian(datasets["official"]["mp_estimate"] / mp_calibration)))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(np.arange(len(names)), np.log10(np.maximum(values, TINY)))
    ax.set_xticks(np.arange(len(names)), names, rotation=20, ha="right")
    ax.set_ylabel("log10(MP检测到的满秩白噪声比例)")
    ax.set_title("只有逐系数IID加性噪声产生明显满秩指纹")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_噪声模型指纹对比.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    mp_q = mp_quantiles()

    if METRICS_CACHE.exists() and not args.force:
        cached = np.load(METRICS_CACHE, allow_pickle=False)
        datasets: dict[str, dict[str, np.ndarray]] = {}
        for name in ("v208_clean", "v208_noisy25", "official"):
            datasets[name] = {
                key: cached[f"{name}_{key}"]
                for key in (
                    "power",
                    "eigen_relative",
                    "q25_relative",
                    "mp_estimate",
                    "mp_fit_rmse",
                    "crossfit_energy",
                    "crossfit_selected",
                    "crossfit_circularity",
                    "crossfit_real_imag_ratio",
                )
            }
        exact_relative = cached["exact_relative"]
        print(f"v568 loaded {PATHS.relative(METRICS_CACHE)}", flush=True)
    else:
        paths = {
            "v208_clean": V208_CLEAN,
            "v208_noisy25": V208_NOISY,
            "official": OFFICIAL,
        }
        datasets = {}
        for name, path in paths.items():
            spectral = dataset_spectral_metrics(name, path, device)
            estimate, fit = mp_estimate(spectral["eigen_relative"], mp_q)
            crossfit = crossfit_metrics(name, path, device)
            datasets[name] = {
                **spectral,
                "mp_estimate": estimate,
                "mp_fit_rmse": fit,
                "crossfit_energy": crossfit["energy"],
                "crossfit_selected": crossfit["selected_energy"],
                "crossfit_circularity": crossfit["circularity"],
                "crossfit_real_imag_ratio": crossfit["real_imag_ratio"],
            }
        exact_relative = exact_noise_relative(V208_CLEAN, V208_NOISY)
        payload: dict[str, np.ndarray] = {"exact_relative": exact_relative}
        for name, values in datasets.items():
            for key, value in values.items():
                payload[f"{name}_{key}"] = value
        np.savez_compressed(METRICS_CACHE, **payload)
        print(f"v568 wrote {PATHS.relative(METRICS_CACHE)}", flush=True)

    valid_control = (
        np.isfinite(exact_relative)
        & np.isfinite(datasets["v208_noisy25"]["mp_estimate"])
        & (exact_relative > 0.0)
        & (datasets["v208_clean"]["power"] > 0.0)
        & (datasets["v208_noisy25"]["power"] > 0.0)
    )
    mp_calibration = float(
        np.median(datasets["v208_noisy25"]["mp_estimate"][valid_control] / exact_relative[valid_control])
    )
    crossfit_calibration = float(
        np.median(
            datasets["v208_noisy25"]["crossfit_selected"][valid_control]
            / exact_relative[valid_control]
        )
    )
    snr_rows, fingerprint_rows = injection_controls(
        V208_CLEAN,
        datasets["v208_clean"]["power"],
        mp_q,
        device,
    )

    geometry = np.load(GEOMETRY)
    blocked = np.asarray(geometry["train_blocked"], dtype=np.int32)
    regimes = regime_of(blocked)
    official_mp = datasets["official"]["mp_estimate"] / mp_calibration
    official_crossfit = datasets["official"]["crossfit_selected"] / crossfit_calibration
    save_point_csv(datasets["official"], official_mp, official_crossfit, regimes, blocked)

    group_report: dict[str, object] = {}
    for regime in ("all", "clear", "light", "medium", "heavy"):
        mask = datasets["official"]["power"] > 0.0
        if regime != "all":
            mask &= regimes == regime
        eigen = datasets["official"]["eigen_relative"][mask]
        residual32 = np.nansum(eigen[:, : N_FREQ - 32], axis=1) / np.maximum(
            np.nansum(eigen, axis=1), TINY
        )
        group_report[regime] = {
            "count": int(np.sum(mask)),
            "q25_floor_relative": summarize(datasets["official"]["q25_relative"][mask]),
            "mp_iid_relative": summarize(official_mp[mask]),
            "crossfit_iid_relative": summarize(official_crossfit[mask]),
            "mp_fit_rmse": summarize(datasets["official"]["mp_fit_rmse"][mask]),
            "rank_gt_1e-12": summarize(np.sum(eigen > 1e-12, axis=1)),
            "rank_gt_1e-8": summarize(np.sum(eigen > 1e-8, axis=1)),
            "rank32_residual_fraction": summarize(residual32),
        }

    clean_mp_floor = float(
        np.nanquantile(datasets["v208_clean"]["mp_estimate"] / mp_calibration, 0.90)
    )
    clean_crossfit_floor = float(
        np.nanquantile(
            datasets["v208_clean"]["crossfit_selected"] / crossfit_calibration,
            0.90,
        )
    )
    joint_floor = max(clean_mp_floor, clean_crossfit_floor)
    tested_actual = np.array(
        [row["actual_noise_to_observed"]["median"] for row in snr_rows], dtype=np.float64
    )
    tested_ratio = np.array(
        [row["estimate_over_actual"]["median"] / mp_calibration for row in snr_rows],
        dtype=np.float64,
    )
    reliable = (tested_ratio >= 0.8) & (tested_ratio <= 1.2) & (tested_actual >= 10.0 * joint_floor)
    conservative_detection_limit = float(np.min(tested_actual[reliable])) if np.any(reliable) else 1e-10
    conservative_detection_limit = max(conservative_detection_limit, 1e-10)
    joint_detected = (
        np.isfinite(official_mp)
        & np.isfinite(official_crossfit)
        & (official_mp >= conservative_detection_limit)
        & (official_crossfit >= conservative_detection_limit)
    )

    make_figures(
        datasets,
        exact_relative,
        snr_rows,
        fingerprint_rows,
        regimes,
        mp_calibration,
        crossfit_calibration,
    )

    meta = json.loads(V208_NOISE_META.read_text(encoding="utf-8"))
    report = {
        "method": VERSION,
        "status": "analysis_complete_no_submission",
        "primary_hypothesis": (
            "An IID circular complex Gaussian perturbation added independently to every "
            "H coefficient creates a full-rank Marchenko-Pastur singular-value bulk and "
            "an independently measurable held-link residual outside the shared frequency "
            "subspace. Physical multipath and in-path perturbations remain low-rank."
        ),
        "identifiability_boundary": {
            "identifiable": [
                "observable per-coefficient IID complex-white-noise variance",
                "a conservative upper bound when the estimate reaches numerical precision",
                "whether the observed H contains a full-rank white-noise fingerprint",
            ],
            "not_identifiable_from_one_H": [
                "the exact random noise realization at every coefficient",
                "Gaussian perturbations applied inside path coefficients or path parameters",
                "a separation between two components occupying the same physical path subspace",
            ],
        },
        "inputs": {
            "official": PATHS.relative(OFFICIAL),
            "v208_clean": PATHS.relative(V208_CLEAN),
            "v208_known_noisy": PATHS.relative(V208_NOISY),
            "v208_noise_metadata": meta,
            "matrix_unfolding": [N_ROW, N_FREQ],
            "device": str(device),
        },
        "estimators": {
            "mp_full_matrix": {
                "definition": (
                    "Reshape each point to 1024 link rows x 192 frequencies, normalize by "
                    "observed mean power, and robustly scale the lower half of the empirical "
                    "eigenvalues to the Marchenko-Pastur quantiles for beta=192/1024."
                ),
                "calibration_factor_on_v208_25db": mp_calibration,
            },
            "crossfit_frequency_subspace": {
                "definition": (
                    "Split the physical array by vertical parity. Learn the frequency right-"
                    "singular subspace on one disjoint half, measure held-half energy outside "
                    f"rank {CROSSFIT_SELECTED_RANK}, then reverse the split and average."
                ),
                "ranks": list(CROSSFIT_RANKS),
                "selected_rank": CROSSFIT_SELECTED_RANK,
                "calibration_factor_on_v208_25db": crossfit_calibration,
            },
            "v565_q25_control": (
                "The historical q25 CIR floor is retained only as a negative control because "
                "it is scale-homogeneous and also measures unresolved clean multipath."
            ),
        },
        "known_noise_control": {
            "paired_nonzero_calibration": {
                "count": int(np.sum(valid_control)),
                "clean_nonzero_count": int(np.sum(datasets["v208_clean"]["power"] > 0.0)),
                "noisy_nonzero_count": int(
                    np.sum(datasets["v208_noisy25"]["power"] > 0.0)
                ),
                "selection": (
                    "Use only indices where clean and noisy H are both nonzero and the exact "
                    "noise ratio and spectral estimate are finite. Calibration uses the median "
                    "because the v208 control also contains a fixed absolute noise floor that "
                    "dominates a few extremely weak points."
                ),
            },
            "true_noise_relative_to_observed": summarize(exact_relative[valid_control]),
            "mp_uncalibrated_over_true": summarize(
                datasets["v208_noisy25"]["mp_estimate"][valid_control]
                / exact_relative[valid_control]
            ),
            "mp_calibrated_over_true": summarize(
                datasets["v208_noisy25"]["mp_estimate"][valid_control]
                / mp_calibration
                / exact_relative[valid_control]
            ),
            "crossfit_uncalibrated_over_true": summarize(
                datasets["v208_noisy25"]["crossfit_selected"][valid_control]
                / exact_relative[valid_control]
            ),
            "crossfit_calibrated_over_true": summarize(
                datasets["v208_noisy25"]["crossfit_selected"][valid_control]
                / crossfit_calibration
                / exact_relative[valid_control]
            ),
            "clean_false_positive_mp_relative": summarize(
                datasets["v208_clean"]["mp_estimate"] / mp_calibration
            ),
            "clean_false_positive_crossfit_relative": summarize(
                datasets["v208_clean"]["crossfit_selected"] / crossfit_calibration
            ),
            "multi_snr_injection": snr_rows,
            "noise_model_fingerprints": fingerprint_rows,
        },
        "v565_proxy_falsification": {
            "v208_clean_q25_relative": summarize(datasets["v208_clean"]["q25_relative"]),
            "v208_noisy25_q25_relative": summarize(
                datasets["v208_noisy25"]["q25_relative"][valid_control]
            ),
            "official_q25_relative": summarize(datasets["official"]["q25_relative"]),
            "interpretation": (
                "A median q25 ratio near the official value is already present in completely "
                "clean v208 H. Therefore the v565 0.008742 ratio is primarily a clean multipath/"
                "finite-bandwidth floor and cannot be used as an IID noise variance."
            ),
        },
        "official_results": {
            "by_regime": group_report,
            "mp_relative_all": distribution_shape(official_mp),
            "crossfit_relative_all": distribution_shape(official_crossfit),
            "conservative_joint_detection_limit": conservative_detection_limit,
            "equivalent_snr_lower_bound_db": float(
                -10.0 * math.log10(conservative_detection_limit)
            ),
            "joint_point_detection_count": int(np.sum(joint_detected)),
            "iid_per_coefficient_awgn_detected": bool(np.any(joint_detected)),
            "interpretation": (
                "Official H is numerically low-rank across 1024 links x 192 frequencies. Its "
                "lower singular spectrum is at the complex64/numerical floor and is many orders "
                "below the validated IID-noise signature. Any non-ideal perturbation must either "
                "be below the conservative detection limit or live inside the physical path "
                "subspace, where it is not separately identifiable from a single H realization."
            ),
        },
        "exact_zero": {
            "official_count": int(np.sum(datasets["official"]["power"] == 0.0)),
            "interpretation": (
                "Exact-zero rows are not evidence for or against output-level IID noise once the "
                "organizer may explicitly mask rows or scale noise by clean power. They remain a "
                "separate missing/no-path mechanism."
            ),
        },
        "conclusion": {
            "strict_result": (
                "The provided official H does not contain a detectable per-coefficient IID "
                "complex Gaussian white-noise component at a practically relevant level."
            ),
            "modeling_action": (
                "Do not subtract 0.008742 times point/link power and do not train a post-hoc "
                "white-noise denoiser on v559/v567 outputs. Treat the observed low-energy floor "
                "as weak physical multipath. Future robustness work should model uncertainty in "
                "path amplitudes/support, not coefficient-wise AWGN."
            ),
        },
        "outputs": {
            "metrics_cache": PATHS.relative(METRICS_CACHE),
            "point_csv": PATHS.relative(POINT_CSV),
            "figures": [PATHS.relative(path) for path in sorted(FIG_DIR.glob("*.png"))],
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["official_results"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {PATHS.relative(REPORT)}", flush=True)


if __name__ == "__main__":
    main()
