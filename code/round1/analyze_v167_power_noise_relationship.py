#!/usr/bin/env python3
"""Analyze the relationship between train power and noise/anomaly proxies.

This is an analysis-only experiment.  It does not generate a submission zip.

Noise is not directly observed, so the script measures several proxies:

* exact-zero and low-power membership;
* local log-power residual against nearby nonzero train points;
* local normalized joint AD support inconsistency against nearby nonzero train
  points;
* existing leave-one-out support-model badness from the v98/current-best
  visualization pipeline.
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from powerfield_common import regime_of
from project_paths import PATHS


VERSION = 167
REPORT_DIR = PATHS.version_report_dir(VERSION)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v167_power_noise_relationship_analysis.json"
POINT_CSV = REPORT_DIR / "v167_power_noise_point_metrics.csv"
BIN_CSV = REPORT_DIR / "v167_power_noise_bin_summary.csv"
TOP_CSV = REPORT_DIR / "v167_power_noise_top_anomalies.csv"

BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)
TINY = 1e-30
K = 8
K_QUERY = 64


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


def nan_summary3(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None, None
    return (
        float(x.mean()),
        float(np.quantile(x, 0.50)),
        float(np.quantile(x, 0.90)),
    )


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        return 0.0
    if np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return 0.0
    return float(spearmanr(x[valid], y[valid]).statistic)


def rank01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.nan_to_num(x, nan=-np.inf))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(order), dtype=np.float64)
    return ranks.astype(np.float32)


def normalized_support(coll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = coll.sum(axis=(1, 2, 3), dtype=np.float64)
    support = np.asarray(coll, dtype=np.float32)
    support = support / np.maximum(total[:, None, None, None], TINY).astype(np.float32)
    support[total <= 0] = 0.0
    return support.reshape(len(support), -1).astype(np.float32), total


def support_concentration(flat_support: np.ndarray) -> dict[str, np.ndarray]:
    sorted_desc = np.sort(flat_support, axis=1)[:, ::-1]
    top1 = sorted_desc[:, 0]
    top8 = sorted_desc[:, :8].sum(axis=1)
    top64 = sorted_desc[:, :64].sum(axis=1)
    entropy = -np.sum(
        flat_support * np.log(np.maximum(flat_support, TINY)), axis=1
    ) / math.log(flat_support.shape[1])
    zero = flat_support.sum(axis=1) <= 0
    entropy[zero] = np.nan
    return {
        "top1": top1.astype(np.float32),
        "top8": top8.astype(np.float32),
        "top64": top64.astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }


def nonzero_knn(pos_xy: np.ndarray, nonzero: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nonzero_idx = np.flatnonzero(nonzero)
    nn = NearestNeighbors(n_neighbors=min(K_QUERY, len(nonzero_idx)), algorithm="auto")
    nn.fit(pos_xy[nonzero_idx])
    dist, local = nn.kneighbors(pos_xy)
    idx = nonzero_idx[local]
    return dist.astype(np.float32), idx.astype(np.int32)


def select_neighbors(
    point_index: int,
    candidate_dist: np.ndarray,
    candidate_idx: np.ndarray,
    regimes: np.ndarray,
    same_regime: bool,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray]:
    idx = candidate_idx[point_index]
    dist = candidate_dist[point_index]
    keep = idx != point_index
    if same_regime:
        keep &= regimes[idx] == regimes[point_index]
    idx = idx[keep]
    dist = dist[keep]
    if idx.size < k and same_regime:
        return select_neighbors(
            point_index, candidate_dist, candidate_idx, regimes, same_regime=False, k=k
        )
    idx = idx[:k]
    dist = dist[:k]
    weight = 1.0 / np.maximum(dist.astype(np.float64), 0.5) ** 2
    weight = weight / np.maximum(weight.sum(), TINY)
    return idx, weight.astype(np.float32)


def local_metrics(
    pos_xy: np.ndarray,
    regimes: np.ndarray,
    power: np.ndarray,
    flat_support: np.ndarray,
    nonzero: np.ndarray,
) -> dict[str, np.ndarray]:
    dist, idx = nonzero_knn(pos_xy, nonzero)
    logp = np.log10(np.maximum(power, 1e-30)).astype(np.float32)
    l2 = np.sqrt(np.sum(flat_support * flat_support, axis=1)).astype(np.float32)

    out = {
        "nearest_nonzero_dist": np.zeros(len(power), dtype=np.float32),
        "local_log_power_mean": np.zeros(len(power), dtype=np.float32),
        "local_log_power_residual": np.zeros(len(power), dtype=np.float32),
        "local_log_power_abs_z": np.zeros(len(power), dtype=np.float32),
        "local_support_cosine": np.zeros(len(power), dtype=np.float32),
        "local_support_inconsistency": np.ones(len(power), dtype=np.float32),
        "same_regime_support_cosine": np.zeros(len(power), dtype=np.float32),
        "same_regime_support_inconsistency": np.ones(len(power), dtype=np.float32),
    }

    for i in range(len(power)):
        for same_regime, prefix in ((False, "local"), (True, "same_regime")):
            nbr, w = select_neighbors(i, dist, idx, regimes, same_regime=same_regime)
            if prefix == "local":
                out["nearest_nonzero_dist"][i] = float(dist[i, idx[i] != i][0])
                p_n = logp[nbr]
                mean = float(np.sum(w * p_n))
                std = float(np.sqrt(np.sum(w * (p_n - mean) ** 2)))
                out["local_log_power_mean"][i] = mean
                out["local_log_power_residual"][i] = float(logp[i] - mean)
                out["local_log_power_abs_z"][i] = float(
                    abs(logp[i] - mean) / max(std, 0.25)
                )
            avg = np.sum(flat_support[nbr] * w[:, None], axis=0).astype(np.float32)
            avg_norm = float(np.sqrt(np.sum(avg * avg)))
            denom = max(float(l2[i]) * avg_norm, TINY)
            cosine = float(np.sum(flat_support[i] * avg) / denom) if l2[i] > 0 else 0.0
            out[f"{prefix}_support_cosine"][i] = np.clip(cosine, 0.0, 1.0)
            out[f"{prefix}_support_inconsistency"][i] = 1.0 - out[
                f"{prefix}_support_cosine"
            ][i]
    return out


def load_current_bad_top() -> np.ndarray:
    mask = np.zeros(2000, dtype=bool)
    path = PATHS.analysis / "visualizations/current_best_bad_points/data/train_true_bad_top.csv"
    if not path.is_file():
        return mask
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mask[int(row["index"])] = True
    return mask


def bin_name(power: float) -> str:
    if power == 0.0:
        return "exact_zero"
    thresholds = [
        (1e-16, "(0,1e-16]"),
        (1e-14, "(1e-16,1e-14]"),
        (1e-12, "(1e-14,1e-12]"),
        (1e-10, "(1e-12,1e-10]"),
        (1e-8, "(1e-10,1e-8]"),
        (1e-6, "(1e-8,1e-6]"),
        (1e-4, "(1e-6,1e-4]"),
    ]
    for threshold, name in thresholds:
        if power <= threshold:
            return name
    return ">1e-4"


def bin_summary(
    names: np.ndarray, regimes: np.ndarray, metrics: dict[str, np.ndarray]
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
        mask = names == name
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
            "log10_power",
            "local_log_power_abs_z",
            "local_support_inconsistency",
            "same_regime_support_inconsistency",
            "v98_badness",
            "noise_proxy",
            "top1",
            "top8",
            "entropy",
        ):
            values = metrics[metric_name][mask]
            mean, median, p90 = nan_summary3(values)
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_median"] = median
            row[f"{metric_name}_p90"] = p90
        row["current_bad_top80_fraction"] = float(np.mean(metrics["current_bad_top80"][mask]))
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


def plot_results(
    pos: np.ndarray,
    power: np.ndarray,
    zero: np.ndarray,
    metrics: dict[str, np.ndarray],
    rows: list[dict[str, object]],
) -> None:
    nonzero = power > 0
    logp = metrics["log10_power"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].hist(logp[nonzero], bins=60, color="#4361ee", alpha=0.85)
    axes[0].axvline(-12, color="#d00000", linestyle="--", linewidth=1.2)
    axes[0].axvline(-8, color="#f77f00", linestyle="--", linewidth=1.2)
    axes[0].set_title("Train power distribution")
    axes[0].set_xlabel("log10(total power)")
    axes[0].set_ylabel("count")

    hb = axes[1].hexbin(
        logp[nonzero],
        metrics["local_support_inconsistency"][nonzero],
        gridsize=45,
        cmap="viridis",
        mincnt=1,
    )
    axes[1].set_title("Power vs local AD-support inconsistency")
    axes[1].set_xlabel("log10(total power)")
    axes[1].set_ylabel("1 - cosine to local nonzero support")
    fig.colorbar(hb, ax=axes[1], label="count")

    hb = axes[2].hexbin(
        logp[nonzero],
        metrics["v98_badness"][nonzero],
        gridsize=45,
        cmap="magma",
        mincnt=1,
    )
    axes[2].scatter(
        np.full(int(zero.sum()), logp[nonzero].min() - 0.5),
        metrics["v98_badness"][zero],
        s=18,
        c="#e63946",
        label="exact-zero",
    )
    axes[2].set_title("Power vs LOO support-model badness")
    axes[2].set_xlabel("log10(total power)")
    axes[2].set_ylabel("1 - LOO support score")
    axes[2].legend(frameon=False, loc="upper right")
    fig.colorbar(hb, ax=axes[2], label="count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "power_vs_noise_scatter.png", dpi=180)
    plt.close(fig)

    labels = [row["bin"] for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].bar(x - 0.2, [row["count"] for row in rows], width=0.4, color="#4cc9f0")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("Power bins")
    axes[0].bar(
        x + 0.2,
        [max(row["count_heavy"], 0.5) for row in rows],
        width=0.4,
        color="#7209b7",
        alpha=0.75,
        label="heavy count",
    )
    axes[0].legend(frameon=False)
    axes[1].plot(
        x,
        [row["local_support_inconsistency_mean"] for row in rows],
        marker="o",
        label="local support inconsistency",
    )
    axes[1].plot(
        x,
        [row["v98_badness_mean"] for row in rows],
        marker="o",
        label="LOO badness",
    )
    axes[1].plot(
        x,
        [row["local_log_power_abs_z_mean"] / 10.0 for row in rows],
        marker="o",
        label="local power |z| / 10",
    )
    axes[1].set_ylabel("mean proxy value")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "power_bin_noise_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
    sc = axes[0].scatter(
        pos[:, 0], pos[:, 1], c=logp, s=12, cmap="viridis", alpha=0.85
    )
    axes[0].scatter(pos[zero, 0], pos[zero, 1], c="#e63946", s=38, marker="x")
    axes[0].set_title("Power; red x = exact-zero")
    fig.colorbar(sc, ax=axes[0], label="log10 power")

    sc = axes[1].scatter(
        pos[:, 0],
        pos[:, 1],
        c=metrics["local_support_inconsistency"],
        s=12,
        cmap="magma",
        alpha=0.85,
        vmin=0,
        vmax=np.nanquantile(metrics["local_support_inconsistency"], 0.98),
    )
    axes[1].set_title("Local support inconsistency")
    fig.colorbar(sc, ax=axes[1], label="1 - cosine")

    sc = axes[2].scatter(
        pos[:, 0],
        pos[:, 1],
        c=metrics["noise_proxy"],
        s=12,
        cmap="plasma",
        alpha=0.85,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Aggregate anomaly proxy")
    fig.colorbar(sc, ax=axes[2], label="rank-combined proxy")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spatial_power_noise_overlay.png", dpi=180)
    plt.close(fig)


def main() -> None:
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    power = np.load(PATHS.shared_features / "train_total_power.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "train_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    coll = np.load(PATHS.shared_features / "truth_coll_power.npy", mmap_mode="r")
    flat_support, coll_total = normalized_support(np.asarray(coll, dtype=np.float32))

    zero = power == 0.0
    nonzero = power > 0.0
    logp = np.log10(np.maximum(power, 1e-30)).astype(np.float32)
    power_bins = np.array([bin_name(float(v)) for v in power], dtype="U20")

    loo_path = PATHS.analysis / "visualizations/v98_bad_points/data/train_loo_metrics.npz"
    loo = np.load(loo_path)
    v98_badness = (1.0 - loo["score"].astype(np.float32)).astype(np.float32)

    current_bad_top80 = load_current_bad_top()
    concentration = support_concentration(flat_support)
    local = local_metrics(pos[:, :2], regimes, power, flat_support, nonzero)

    noise_proxy = (
        0.35 * rank01(local["local_support_inconsistency"])
        + 0.25 * rank01(local["local_log_power_abs_z"])
        + 0.25 * rank01(v98_badness)
        + 0.15 * rank01(1.0 - concentration["top8"])
    ).astype(np.float32)

    metrics: dict[str, np.ndarray] = {
        "power": power,
        "log10_power": logp,
        "blocked": blocked.astype(np.float32),
        "current_bad_top80": current_bad_top80.astype(np.float32),
        "v98_badness": v98_badness,
        "noise_proxy": noise_proxy,
        **concentration,
        **local,
    }

    rows = bin_summary(power_bins, regimes, metrics)
    write_csv(BIN_CSV, rows)

    point_rows: list[dict[str, object]] = []
    for i in range(len(pos)):
        point_rows.append(
            {
                "index": i,
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
                "z": float(pos[i, 2]),
                "regime": str(regimes[i]),
                "blocked": float(blocked[i]),
                "power": float(power[i]),
                "log10_power": float(logp[i]),
                "power_bin": str(power_bins[i]),
                "exact_zero": int(zero[i]),
                "near_zero_le_1e_12": int(power[i] <= 1e-12),
                "local_log_power_residual": float(local["local_log_power_residual"][i]),
                "local_log_power_abs_z": float(local["local_log_power_abs_z"][i]),
                "local_support_inconsistency": float(
                    local["local_support_inconsistency"][i]
                ),
                "same_regime_support_inconsistency": float(
                    local["same_regime_support_inconsistency"][i]
                ),
                "v98_badness": float(v98_badness[i]),
                "noise_proxy": float(noise_proxy[i]),
                "top1": float(concentration["top1"][i]),
                "top8": float(concentration["top8"][i]),
                "top64": float(concentration["top64"][i]),
                "entropy": float(concentration["entropy"][i])
                if np.isfinite(concentration["entropy"][i])
                else "",
                "current_bad_top80": int(current_bad_top80[i]),
            }
        )
    write_csv(POINT_CSV, point_rows)

    top_order = np.argsort(-noise_proxy)[:120]
    write_csv(TOP_CSV, [point_rows[int(i)] | {"rank": rank + 1} for rank, i in enumerate(top_order)])

    plot_results(pos, power, zero, metrics, rows)

    valid_nonzero = nonzero & np.isfinite(v98_badness)
    top10_noise = noise_proxy >= np.quantile(noise_proxy, 0.90)
    top10_badness = v98_badness >= np.quantile(v98_badness, 0.90)
    low_rank_score = -logp

    aucs: dict[str, float | None] = {}
    for name, target in {
        "top10_noise_proxy_from_low_power": top10_noise,
        "top10_v98_badness_from_low_power": top10_badness,
        "current_bad_top80_from_low_power": current_bad_top80,
        "exact_zero_from_noise_proxy": zero,
    }.items():
        try:
            if name == "exact_zero_from_noise_proxy":
                aucs[name] = float(roc_auc_score(target.astype(int), noise_proxy))
            else:
                aucs[name] = float(
                    roc_auc_score(target[valid_nonzero].astype(int), low_rank_score[valid_nonzero])
                )
        except ValueError:
            aucs[name] = None

    low_masks = {
        "exact_zero": zero,
        "power_le_1e-16": power <= 1e-16,
        "power_le_1e-14": power <= 1e-14,
        "power_le_1e-12": power <= 1e-12,
        "power_le_1e-10": power <= 1e-10,
        "power_le_1e-8": power <= 1e-8,
        "power_le_1e-6": power <= 1e-6,
    }
    threshold_summary: dict[str, dict[str, object]] = {}
    for name, mask in low_masks.items():
        threshold_summary[name] = {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "regime_counts": {
                regime: int(np.sum(mask & (regimes == regime)))
                for regime in ("clear", "light", "medium", "heavy")
            },
            "mean_noise_proxy": float(np.nanmean(noise_proxy[mask])) if mask.any() else None,
            "mean_v98_badness": float(np.nanmean(v98_badness[mask])) if mask.any() else None,
            "mean_local_support_inconsistency": float(
                np.nanmean(local["local_support_inconsistency"][mask])
            )
            if mask.any()
            else None,
            "current_bad_top80_count": int(np.sum(mask & current_bad_top80)),
        }

    correlations = {
        "nonzero_log_power_vs_local_power_abs_z": safe_spearman(
            logp[valid_nonzero], local["local_log_power_abs_z"][valid_nonzero]
        ),
        "nonzero_log_power_vs_local_support_inconsistency": safe_spearman(
            logp[valid_nonzero], local["local_support_inconsistency"][valid_nonzero]
        ),
        "nonzero_log_power_vs_same_regime_support_inconsistency": safe_spearman(
            logp[valid_nonzero],
            local["same_regime_support_inconsistency"][valid_nonzero],
        ),
        "nonzero_log_power_vs_v98_badness": safe_spearman(
            logp[valid_nonzero], v98_badness[valid_nonzero]
        ),
        "nonzero_log_power_vs_noise_proxy": safe_spearman(
            logp[valid_nonzero], noise_proxy[valid_nonzero]
        ),
        "nonzero_log_power_vs_top8_fraction": safe_spearman(
            logp[valid_nonzero], concentration["top8"][valid_nonzero]
        ),
    }

    zero_indices = np.flatnonzero(zero).astype(int).tolist()
    exact_zero_detail = [
        {
            "index": int(i),
            "x": float(pos[i, 0]),
            "y": float(pos[i, 1]),
            "regime": str(regimes[i]),
            "nearest_nonzero_dist": float(local["nearest_nonzero_dist"][i]),
            "local_log_power_mean": float(local["local_log_power_mean"][i]),
            "local_support_inconsistency": float(local["local_support_inconsistency"][i]),
            "v98_badness": float(v98_badness[i]),
            "noise_proxy": float(noise_proxy[i]),
            "current_bad_top80": bool(current_bad_top80[i]),
        }
        for i in zero_indices
    ]

    report = {
        "method": "v167_power_noise_relationship_analysis",
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "If train noise mainly comes from invalid simulation rows, anomaly "
            "proxies should concentrate at exact/near-zero power. If it is mostly "
            "propagation discontinuity, low power and badness should correlate only "
            "partially."
        ),
        "inputs": {
            "positions": "datasets/round1/Round1_Train_Pos.npy",
            "power": "artifacts/shared_features/train_total_power.npy",
            "collapsed_truth_support": "artifacts/shared_features/truth_coll_power.npy",
            "blocked": "artifacts/shared_features/height_geometry_cache.npz/train_blocked",
            "loo_metrics": "analysis/visualizations/v98_bad_points/data/train_loo_metrics.npz",
            "current_bad_top": "analysis/visualizations/current_best_bad_points/data/train_true_bad_top.csv",
        },
        "definitions": {
            "local_power_noise": "absolute z-like residual of log10 power vs 8 nearby nonzero train points",
            "local_support_noise": "1 - cosine between normalized joint AD support and 8-neighbor nonzero support average",
            "v98_badness": "1 - v98 leave-one-out support score",
            "noise_proxy": "rank combination: 35% local support inconsistency, 25% local power z, 25% v98 badness, 15% low top8 concentration",
        },
        "counts": {
            "train_count": int(len(power)),
            "nonzero_count": int(nonzero.sum()),
            "exact_zero_count": int(zero.sum()),
            "near_zero_le_1e_12_count": int(np.sum(power <= 1e-12)),
            "current_bad_top80_count": int(current_bad_top80.sum()),
        },
        "power": summarize(power),
        "log10_power_nonzero": summarize(logp[nonzero]),
        "threshold_summary": threshold_summary,
        "correlations": correlations,
        "aucs": aucs,
        "bin_summary": rows,
        "exact_zero_indices": zero_indices,
        "exact_zero_detail": exact_zero_detail,
        "top_noise_indices": [int(i) for i in top_order[:40]],
        "top_noise_exact_zero_count": int(np.sum(zero[top_order[:40]])),
        "top_noise_power_le_1e_12_count": int(np.sum(power[top_order[:40]] <= 1e-12)),
        "top_noise_current_bad_top80_count": int(np.sum(current_bad_top80[top_order[:40]])),
        "outputs": {
            "report": str(REPORT.relative_to(PATHS.root)),
            "point_metrics_csv": str(POINT_CSV.relative_to(PATHS.root)),
            "bin_summary_csv": str(BIN_CSV.relative_to(PATHS.root)),
            "top_anomalies_csv": str(TOP_CSV.relative_to(PATHS.root)),
            "figures": [
                str((FIG_DIR / "power_vs_noise_scatter.png").relative_to(PATHS.root)),
                str((FIG_DIR / "power_bin_noise_summary.png").relative_to(PATHS.root)),
                str((FIG_DIR / "spatial_power_noise_overlay.png").relative_to(PATHS.root)),
            ],
        },
        "conclusion": {
            "short": (
                "Exact-zero points are extreme anomalies, but power alone is not "
                "a reliable global noise label. Near-zero pruning is risky because "
                "many low-power nonzero points belong to heavy/blocked propagation "
                "and are not equivalent to exact-zero rows."
            ),
            "next_use": (
                "Use power-derived noise only as a narrow reliability weight or "
                "latent region feature, not as a deletion rule and not as a broad "
                "support target."
            ),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str(REPORT.relative_to(PATHS.root)),
        "exact_zero_count": int(zero.sum()),
        "near_zero_le_1e_12_count": int(np.sum(power <= 1e-12)),
        "correlations": correlations,
        "aucs": aucs,
        "top_noise_exact_zero_count": int(np.sum(zero[top_order[:40]])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
