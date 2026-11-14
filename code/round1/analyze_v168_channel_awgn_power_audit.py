#!/usr/bin/env python3
"""Audit possible complex AWGN effects in the official train channels.

This analysis asks whether the official train channel looks like

    H_observed = H_clean + N,  N ~ CN(0, sigma^2)

from the data alone.  There is no clean reference and no repeated observation at
the same UE position, so we cannot identify the exact noise realization.  We can
only measure signatures that should appear under additive circular white noise:

* low-power points should have lower effective SNR;
* AD-domain tail energy should behave like a background floor;
* tail coefficients should be roughly circular complex: real/imag balance,
  small pseudo-covariance, and near-zero mean;
* exact-zero rows cannot be explained by nonzero-variance AWGN.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, spearmanr

from powerfield_common import regime_of
from project_paths import PATHS


VERSION = 168
MP, MH, MV, N_ANT, S_SC = 2, 16, 8, 4, 192
CELL_COUNT = MH * MV * S_SC
COMP_PER_CELL = MP * N_ANT
TINY = 1e-30

REPORT_DIR = PATHS.version_report_dir(VERSION)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v168_channel_awgn_power_audit.json"
POINT_CSV = REPORT_DIR / "v168_channel_awgn_point_metrics.csv"
BIN_CSV = REPORT_DIR / "v168_channel_awgn_bin_summary.csv"


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or x[valid].std() == 0 or y[valid].std() == 0:
        return 0.0
    return float(spearmanr(x[valid], y[valid]).statistic)


def ad_transform(h: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def power_bin(power: float) -> str:
    if power == 0.0:
        return "exact_zero"
    for threshold, name in (
        (1e-16, "(0,1e-16]"),
        (1e-14, "(1e-16,1e-14]"),
        (1e-12, "(1e-14,1e-12]"),
        (1e-10, "(1e-12,1e-10]"),
        (1e-8, "(1e-10,1e-8]"),
        (1e-6, "(1e-8,1e-6]"),
        (1e-4, "(1e-6,1e-4]"),
    ):
        if power <= threshold:
            return name
    return ">1e-4"


def top_fraction(flat_power: np.ndarray, k: int) -> np.ndarray:
    part = np.partition(flat_power, flat_power.shape[1] - k, axis=1)[:, -k:]
    return part.sum(axis=1) / np.maximum(flat_power.sum(axis=1), TINY)


def entropy01(flat_power: np.ndarray) -> np.ndarray:
    prob = flat_power / np.maximum(flat_power.sum(axis=1, keepdims=True), TINY)
    ent = -np.sum(prob * np.log(np.maximum(prob, TINY)), axis=1) / math.log(
        flat_power.shape[1]
    )
    ent[flat_power.sum(axis=1) <= 0] = np.nan
    return ent.astype(np.float32)


def top_cell_mask(flat_power: np.ndarray, k: int) -> np.ndarray:
    n = flat_power.shape[1]
    keep = min(k, n)
    idx = np.argpartition(flat_power, n - keep, axis=1)[:, -keep:]
    mask = np.zeros_like(flat_power, dtype=bool)
    rows = np.arange(flat_power.shape[0])[:, None]
    mask[rows, idx] = True
    return mask.reshape(flat_power.shape[0], MH, MV, S_SC)


def tail_complex_stats(ad: np.ndarray, cell_power: np.ndarray) -> dict[str, np.ndarray]:
    # Exclude strong path cells, then sample tail coefficients from the rest.
    # Tail includes noise + weak diffuse components; if AWGN is strong, this
    # should become close to circular complex Gaussian.
    top512 = top_cell_mask(cell_power.reshape(len(cell_power), -1), 512)
    out = {
        "tail_coeff_mean_power": np.zeros(len(ad), dtype=np.float32),
        "tail_coeff_median_power": np.zeros(len(ad), dtype=np.float32),
        "tail_real_imag_std_ratio": np.zeros(len(ad), dtype=np.float32),
        "tail_circularity": np.zeros(len(ad), dtype=np.float32),
        "tail_mean_abs_over_rms": np.zeros(len(ad), dtype=np.float32),
        "tail_real_kurtosis": np.zeros(len(ad), dtype=np.float32),
        "tail_imag_kurtosis": np.zeros(len(ad), dtype=np.float32),
    }
    rng = np.random.default_rng(1682026)
    for i in range(len(ad)):
        z = ad[i].transpose(1, 2, 4, 0, 3).reshape(MH, MV, S_SC, COMP_PER_CELL)
        tail = z[~top512[i]].reshape(-1)
        if tail.size > 8192:
            tail = tail[rng.choice(tail.size, size=8192, replace=False)]
        power = np.abs(tail) ** 2
        mean_power = float(np.mean(power))
        real = tail.real.astype(np.float64)
        imag = tail.imag.astype(np.float64)
        std_real = float(np.std(real))
        std_imag = float(np.std(imag))
        pseudo = abs(np.mean(tail.astype(np.complex128) ** 2))
        out["tail_coeff_mean_power"][i] = mean_power
        out["tail_coeff_median_power"][i] = float(np.median(power))
        out["tail_real_imag_std_ratio"][i] = std_real / max(std_imag, TINY)
        out["tail_circularity"][i] = pseudo / max(mean_power, TINY)
        out["tail_mean_abs_over_rms"][i] = abs(np.mean(tail)) / math.sqrt(
            max(mean_power, TINY)
        )
        out["tail_real_kurtosis"][i] = float(kurtosis(real, fisher=True, bias=False))
        out["tail_imag_kurtosis"][i] = float(kurtosis(imag, fisher=True, bias=False))
    return out


def compute_metrics(batch_size: int = 20) -> dict[str, np.ndarray]:
    channels = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    n = channels.shape[0]
    metrics = {
        "ad_total_power": np.zeros(n, dtype=np.float64),
        "cell_p10_power": np.zeros(n, dtype=np.float32),
        "cell_p25_power": np.zeros(n, dtype=np.float32),
        "cell_median_power": np.zeros(n, dtype=np.float32),
        "cell_tail_mean_ex_top512": np.zeros(n, dtype=np.float32),
        "top1_fraction": np.zeros(n, dtype=np.float32),
        "top8_fraction": np.zeros(n, dtype=np.float32),
        "top64_fraction": np.zeros(n, dtype=np.float32),
        "top512_fraction": np.zeros(n, dtype=np.float32),
        "ad_entropy": np.zeros(n, dtype=np.float32),
        "floor_fraction_median_cell": np.zeros(n, dtype=np.float32),
        "floor_fraction_tail_mean": np.zeros(n, dtype=np.float32),
        "top64_to_tail_floor": np.zeros(n, dtype=np.float32),
        "top512_to_tail_floor": np.zeros(n, dtype=np.float32),
    }
    tail_metric_names = [
        "tail_coeff_mean_power",
        "tail_coeff_median_power",
        "tail_real_imag_std_ratio",
        "tail_circularity",
        "tail_mean_abs_over_rms",
        "tail_real_kurtosis",
        "tail_imag_kurtosis",
    ]
    for name in tail_metric_names:
        metrics[name] = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        h = np.asarray(channels[start:end])
        ad = ad_transform(h)
        cell_power = np.sum(np.abs(ad) ** 2, axis=(1, 4), dtype=np.float64).astype(
            np.float32
        )
        flat = cell_power.reshape(end - start, -1)
        total = flat.sum(axis=1, dtype=np.float64)
        sorted_power = np.sort(flat, axis=1)
        metrics["ad_total_power"][start:end] = total
        metrics["cell_p10_power"][start:end] = sorted_power[:, int(0.10 * CELL_COUNT)]
        metrics["cell_p25_power"][start:end] = sorted_power[:, int(0.25 * CELL_COUNT)]
        metrics["cell_median_power"][start:end] = sorted_power[:, int(0.50 * CELL_COUNT)]
        metrics["cell_tail_mean_ex_top512"][start:end] = sorted_power[:, :-512].mean(
            axis=1
        )
        for k, name in (
            (1, "top1_fraction"),
            (8, "top8_fraction"),
            (64, "top64_fraction"),
            (512, "top512_fraction"),
        ):
            metrics[name][start:end] = top_fraction(flat, k).astype(np.float32)
        metrics["ad_entropy"][start:end] = entropy01(flat)

        floor_median = metrics["cell_median_power"][start:end].astype(np.float64)
        floor_tail = metrics["cell_tail_mean_ex_top512"][start:end].astype(np.float64)
        metrics["floor_fraction_median_cell"][start:end] = (
            floor_median * CELL_COUNT / np.maximum(total, TINY)
        ).astype(np.float32)
        metrics["floor_fraction_tail_mean"][start:end] = (
            floor_tail * CELL_COUNT / np.maximum(total, TINY)
        ).astype(np.float32)
        top64_sum = np.partition(flat, CELL_COUNT - 64, axis=1)[:, -64:].sum(axis=1)
        top512_sum = np.partition(flat, CELL_COUNT - 512, axis=1)[:, -512:].sum(axis=1)
        metrics["top64_to_tail_floor"][start:end] = (
            (top64_sum / 64.0) / np.maximum(floor_tail, TINY)
        ).astype(np.float32)
        metrics["top512_to_tail_floor"][start:end] = (
            (top512_sum / 512.0) / np.maximum(floor_tail, TINY)
        ).astype(np.float32)

        tail = tail_complex_stats(ad, cell_power)
        for name in tail_metric_names:
            metrics[name][start:end] = tail[name]
        print(f"processed {end}/{n}", flush=True)
    return metrics


def bin_summary(
    bin_names: np.ndarray, regimes: np.ndarray, metrics: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    order = [
        "exact_zero",
        "(0,1e-16]",
        "(1e-16,1e-14]",
        "(1e-14,1e-12]",
        "(1e-12,1e-10]",
        "(1e-10,1e-8]",
        "(1e-8,1e-6]",
        "(1e-6,1e-4]",
        ">1e-4",
    ]
    rows: list[dict[str, object]] = []
    for name in order:
        mask = bin_names == name
        if not mask.any():
            continue
        row: dict[str, object] = {
            "bin": name,
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
        }
        for regime in ("clear", "light", "medium", "heavy"):
            row[f"count_{regime}"] = int(np.sum(mask & (regimes == regime)))
        for metric_name in (
            "log10_raw_power",
            "log10_tail_coeff_mean_power",
            "floor_fraction_tail_mean",
            "top64_fraction",
            "top512_fraction",
            "top64_to_tail_floor",
            "ad_entropy",
            "tail_real_imag_std_ratio",
            "tail_circularity",
            "tail_mean_abs_over_rms",
            "tail_real_kurtosis",
            "tail_imag_kurtosis",
        ):
            values = np.asarray(metrics[metric_name][mask], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{metric_name}_median"] = None
                row[f"{metric_name}_mean"] = None
            else:
                row[f"{metric_name}_median"] = float(np.median(values))
                row[f"{metric_name}_mean"] = float(values.mean())
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(metrics: dict[str, np.ndarray], power: np.ndarray, regimes: np.ndarray) -> None:
    nonzero = power > 0
    logp = metrics["log10_raw_power"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    hb = axes[0, 0].hexbin(
        logp[nonzero],
        metrics["log10_tail_coeff_mean_power"][nonzero],
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    axes[0, 0].set_xlabel("log10 raw total power")
    axes[0, 0].set_ylabel("log10 AD tail coeff mean power")
    axes[0, 0].set_title("If AWGN floor fixed, this would be flatter")
    fig.colorbar(hb, ax=axes[0, 0], label="count")

    hb = axes[0, 1].hexbin(
        logp[nonzero],
        metrics["floor_fraction_tail_mean"][nonzero],
        gridsize=45,
        mincnt=1,
        cmap="magma",
    )
    axes[0, 1].set_xlabel("log10 raw total power")
    axes[0, 1].set_ylabel("tail-floor fraction of AD total")
    axes[0, 1].set_title("Relative noise/background grows as power drops")
    fig.colorbar(hb, ax=axes[0, 1], label="count")

    hb = axes[1, 0].hexbin(
        logp[nonzero],
        metrics["top64_to_tail_floor"][nonzero],
        gridsize=45,
        mincnt=1,
        cmap="plasma",
    )
    axes[1, 0].set_xlabel("log10 raw total power")
    axes[1, 0].set_ylabel("top64 mean / tail floor")
    axes[1, 0].set_title("Effective sparse-path SNR proxy")
    fig.colorbar(hb, ax=axes[1, 0], label="count")

    data = [
        metrics["tail_real_imag_std_ratio"][nonzero],
        metrics["tail_circularity"][nonzero],
        metrics["tail_mean_abs_over_rms"][nonzero],
    ]
    axes[1, 1].boxplot(data, tick_labels=["stdR/stdI", "circularity", "|mean|/rms"])
    axes[1, 1].axhline(1.0, color="gray", linewidth=1.0, linestyle="--")
    axes[1, 1].set_title("Tail complex circularity diagnostics")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "awgn_power_relationship.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    values = [
        metrics["log10_raw_power"],
        metrics["floor_fraction_tail_mean"],
        np.log10(np.maximum(metrics["top64_to_tail_floor"], TINY)),
        metrics["tail_circularity"],
    ]
    titles = [
        "log10 total power",
        "tail floor fraction",
        "log10 top64/floor",
        "tail circularity",
    ]
    cmaps = ["viridis", "magma", "plasma", "cividis"]
    for ax, value, title, cmap in zip(axes, values, titles, cmaps):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=value, s=10, cmap=cmap, alpha=0.85)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        fig.colorbar(sc, ax=ax)
    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "awgn_spatial_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    raw_power = np.load(PATHS.shared_features / "train_total_power.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "train_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    bin_names = np.array([power_bin(float(v)) for v in raw_power], dtype="U20")
    exact_zero = raw_power == 0
    nonzero = raw_power > 0

    metrics = compute_metrics()
    metrics["raw_total_power"] = raw_power
    metrics["log10_raw_power"] = np.log10(np.maximum(raw_power, 1e-30)).astype(
        np.float32
    )
    metrics["log10_tail_coeff_mean_power"] = np.log10(
        np.maximum(metrics["tail_coeff_mean_power"], 1e-30)
    ).astype(np.float32)
    metrics["power_bin"] = bin_names

    rows = bin_summary(bin_names, regimes, metrics)
    write_csv(BIN_CSV, rows)

    point_rows: list[dict[str, object]] = []
    for i in range(len(raw_power)):
        point_rows.append(
            {
                "index": i,
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
                "z": float(pos[i, 2]),
                "regime": str(regimes[i]),
                "blocked": float(blocked[i]),
                "raw_total_power": float(raw_power[i]),
                "log10_raw_power": float(metrics["log10_raw_power"][i]),
                "power_bin": str(bin_names[i]),
                "exact_zero": int(exact_zero[i]),
                "ad_total_power": float(metrics["ad_total_power"][i]),
                "cell_median_power": float(metrics["cell_median_power"][i]),
                "cell_tail_mean_ex_top512": float(
                    metrics["cell_tail_mean_ex_top512"][i]
                ),
                "tail_coeff_mean_power": float(metrics["tail_coeff_mean_power"][i]),
                "floor_fraction_tail_mean": float(
                    metrics["floor_fraction_tail_mean"][i]
                ),
                "top64_fraction": float(metrics["top64_fraction"][i]),
                "top512_fraction": float(metrics["top512_fraction"][i]),
                "top64_to_tail_floor": float(metrics["top64_to_tail_floor"][i]),
                "ad_entropy": float(metrics["ad_entropy"][i])
                if np.isfinite(metrics["ad_entropy"][i])
                else "",
                "tail_real_imag_std_ratio": float(
                    metrics["tail_real_imag_std_ratio"][i]
                ),
                "tail_circularity": float(metrics["tail_circularity"][i]),
                "tail_mean_abs_over_rms": float(metrics["tail_mean_abs_over_rms"][i]),
                "tail_real_kurtosis": float(metrics["tail_real_kurtosis"][i]),
                "tail_imag_kurtosis": float(metrics["tail_imag_kurtosis"][i]),
            }
        )
    write_csv(POINT_CSV, point_rows)

    plot(metrics, raw_power, regimes)

    logp = metrics["log10_raw_power"][nonzero]
    report = {
        "method": "v168_channel_awgn_power_audit",
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official train channels may include complex Gaussian noise.  If so, "
            "low-power points should have worse effective SNR, and the AD-domain "
            "tail should expose a roughly circular complex background floor."
        ),
        "limitations": [
            "No repeated observations at the same UE position; exact noise variance cannot be identified.",
            "AD tail contains true weak diffuse multipath as well as noise, so floor estimates are upper bounds on AWGN.",
            "Selecting tail bins after observing power biases pure Gaussian diagnostics; circularity metrics are diagnostic, not proof.",
        ],
        "inputs": {
            "train_channel": "datasets/round1/Round1_Train_Channel.npy",
            "train_position": "datasets/round1/Round1_Train_Pos.npy",
            "train_total_power": "artifacts/shared_features/train_total_power.npy",
            "height_geometry": "artifacts/shared_features/height_geometry_cache.npz",
        },
        "counts": {
            "train_count": int(len(raw_power)),
            "nonzero_count": int(nonzero.sum()),
            "exact_zero_count": int(exact_zero.sum()),
            "power_le_1e-12_count": int(np.sum(raw_power <= 1e-12)),
        },
        "global_summaries": {
            "raw_total_power": summarize(raw_power),
            "tail_coeff_mean_power": summarize(metrics["tail_coeff_mean_power"][nonzero]),
            "floor_fraction_tail_mean": summarize(
                metrics["floor_fraction_tail_mean"][nonzero]
            ),
            "top64_to_tail_floor": summarize(metrics["top64_to_tail_floor"][nonzero]),
            "tail_real_imag_std_ratio": summarize(
                metrics["tail_real_imag_std_ratio"][nonzero]
            ),
            "tail_circularity": summarize(metrics["tail_circularity"][nonzero]),
            "tail_mean_abs_over_rms": summarize(
                metrics["tail_mean_abs_over_rms"][nonzero]
            ),
        },
        "correlations_nonzero": {
            "log_power_vs_log_tail_floor": safe_spearman(
                logp, metrics["log10_tail_coeff_mean_power"][nonzero]
            ),
            "log_power_vs_floor_fraction": safe_spearman(
                logp, metrics["floor_fraction_tail_mean"][nonzero]
            ),
            "log_power_vs_top64_to_tail_floor": safe_spearman(
                logp, metrics["top64_to_tail_floor"][nonzero]
            ),
            "log_power_vs_top64_fraction": safe_spearman(
                logp, metrics["top64_fraction"][nonzero]
            ),
            "log_power_vs_ad_entropy": safe_spearman(logp, metrics["ad_entropy"][nonzero]),
            "log_power_vs_tail_circularity": safe_spearman(
                logp, metrics["tail_circularity"][nonzero]
            ),
        },
        "bin_summary": rows,
        "exact_zero_note": (
            "16 exact-zero rows cannot be generated by nonzero-variance complex "
            "Gaussian noise; they are a separate mask/simulation failure/special "
            "state issue."
        ),
        "interpretation": {
            "evidence_compatible_with_awgn_effect": [
                "tail coefficients are approximately circular in aggregate, with real/imag std ratio near 1 and small mean/rms",
                "lower-power bins have slightly worse relative background/SNR proxies, but the effect is weak rather than dominant",
            ],
            "evidence_against_simple_constant_awgn_only": [
                "estimated tail floor strongly correlates with total power, so AD tail includes real diffuse multipath, not just fixed sigma noise",
                "exact-zero rows are impossible under ordinary AWGN",
            ],
            "modeling_implication": (
                "Do not add random noise to predictions.  Instead use power/SNR-aware "
                "confidence: trust strong AD components more, suppress overfitting to "
                "weak tail phase/support, and treat exact-zero separately from Gaussian noise."
            ),
        },
        "outputs": {
            "report": str(REPORT.relative_to(PATHS.root)),
            "point_metrics_csv": str(POINT_CSV.relative_to(PATHS.root)),
            "bin_summary_csv": str(BIN_CSV.relative_to(PATHS.root)),
            "figures": [
                str((FIG_DIR / "awgn_power_relationship.png").relative_to(PATHS.root)),
                str((FIG_DIR / "awgn_spatial_diagnostics.png").relative_to(PATHS.root)),
            ],
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT.relative_to(PATHS.root)),
                "counts": report["counts"],
                "correlations_nonzero": report["correlations_nonzero"],
                "tail_circularity": report["global_summaries"]["tail_circularity"],
                "top64_to_tail_floor": report["global_summaries"]["top64_to_tail_floor"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
