#!/usr/bin/env python3
"""Correlate historical online scores with v206 and v205 Sionna references."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

from analyze_v195_scorer_mouth_reverse_inference import (
    cached_output,
    parse_online_scores,
    parse_score_csv,
)
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v207_online_score_sionna_correlation"
REPORT_DIR = PATHS.version_report_dir(207)
REPORT = REPORT_DIR / "v207_online_score_sionna_correlation.json"
FEATURE_CSV = REPORT_DIR / "v207_historical_sionna_reference_features.csv"
CORRELATION_CSV = REPORT_DIR / "v207_historical_sionna_correlations.csv"
FIGURE_DIR = REPORT_DIR / "figures"
REFERENCE_FEATURE_DIR = PATHS.cache_dir(207) / "reference_features"
SCORE_CSV = PATHS.analysis / "visualizations" / "version_scores" / "online_scores_by_version.csv"
V206_REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
V205_REFERENCE = PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p0" / "test_clean_H.npy"
FLOOR = 1e-30
REGIMES = ("clear", "light", "medium", "heavy")


def normalize_last(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    norm = np.sqrt(np.sum(value * value, axis=axes, keepdims=True, dtype=np.float64))
    return (value / np.maximum(norm, FLOOR)).astype(np.float32)


def transform_channel(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(h)
    value6 = h.reshape(n, 2, 16, 8, 4, 192)
    leaked = value6.transpose(0, 1, 5, 2, 4, 3)
    leaked = np.abs(np.fft.fft2(leaked, axes=(2, 3))).astype(np.float32) ** 2
    leaked = leaked.transpose(0, 1, 4, 5, 2, 3)
    leaked_norm = normalize_last(leaked, (4, 5))
    collapsed = np.sum(leaked, axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    collapsed_norm = normalize_last(collapsed, (1, 2))
    delay = np.abs(np.fft.ifft(h, axis=-1)).astype(np.float32) ** 2
    delay_norm = normalize_last(delay, (3,))
    power = np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return leaked_norm, collapsed_norm, delay_norm, power


def ensure_reference_features(name: str, path: Path, batch: int = 8) -> dict[str, Path]:
    out_dir = REFERENCE_FEATURE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "leaked": out_dir / "leaked_norm.npy",
        "collapsed": out_dir / "collapsed_norm.npy",
        "delay": out_dir / "delay_norm.npy",
        "power": out_dir / "point_power.npy",
        "metadata": out_dir / "metadata.json",
    }
    data_keys = ("leaked", "collapsed", "delay", "power")
    if all(outputs[key].is_file() for key in data_keys) and outputs["metadata"].is_file():
        return outputs
    if all(outputs[key].is_file() for key in data_keys):
        arrays = {key: np.load(outputs[key], mmap_mode="r") for key in data_keys}
        finite = True
        for start in range(0, 500, batch):
            end = min(start + batch, 500)
            finite = finite and all(bool(np.isfinite(arrays[key][start:end]).all()) for key in data_keys)
        metadata = {
            "reference": PATHS.relative(path),
            "zero_power_count": int(np.sum(np.asarray(arrays["power"]) == 0.0)),
            "finite": finite,
        }
        outputs["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return outputs
    h = np.load(path, mmap_mode="r")
    if h.shape != (500, 256, 4, 192) or h.dtype != np.complex64:
        raise ValueError(f"Invalid reference {path}: {h.shape} {h.dtype}")
    leaked_out = np.lib.format.open_memmap(
        outputs["leaked"], mode="w+", dtype=np.float32, shape=(500, 2, 4, 8, 192, 16)
    )
    collapsed_out = np.lib.format.open_memmap(
        outputs["collapsed"], mode="w+", dtype=np.float32, shape=(500, 192, 16)
    )
    delay_out = np.lib.format.open_memmap(
        outputs["delay"], mode="w+", dtype=np.float32, shape=(500, 256, 4, 192)
    )
    power_out = np.lib.format.open_memmap(outputs["power"], mode="w+", dtype=np.float64, shape=(500,))
    for start in range(0, len(h), batch):
        end = min(start + batch, len(h))
        leaked, collapsed, delay, power = transform_channel(np.asarray(h[start:end], dtype=np.complex64))
        leaked_out[start:end] = leaked
        collapsed_out[start:end] = collapsed
        delay_out[start:end] = delay
        power_out[start:end] = power
    leaked_out.flush()
    collapsed_out.flush()
    delay_out.flush()
    power_out.flush()
    finite = True
    for start in range(0, 500, batch):
        end = min(start + batch, 500)
        finite = finite and all(
            bool(np.isfinite(array[start:end]).all())
            for array in (leaked_out, collapsed_out, delay_out, power_out)
        )
    metadata = {
        "reference": PATHS.relative(path),
        "zero_power_count": int(np.sum(np.asarray(power_out) == 0.0)),
        "finite": finite,
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return outputs


def online_versions() -> list[tuple[int, float, Path]]:
    scores = parse_online_scores()
    scores.update(parse_score_csv(SCORE_CSV))
    versions = []
    for version, metadata in sorted(scores.items()):
        output = cached_output(version)
        if output is not None:
            versions.append((version, float(metadata["online"]), output))
    return versions


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 4 or np.std(x[valid]) == 0.0 or np.std(y[valid]) == 0.0:
        return None
    if method == "pearson":
        return float(pearsonr(x[valid], y[valid]).statistic)
    if method == "spearman":
        return float(spearmanr(x[valid], y[valid]).statistic)
    return float(kendalltau(x[valid], y[valid]).statistic)


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(len(controls)), controls])
    coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficient


def polynomial_version_controls(version: np.ndarray) -> np.ndarray:
    normalized = (version - np.mean(version)) / max(float(np.std(version)), 1.0)
    return np.column_stack([normalized, normalized**2, normalized**3])


def aggregate(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def score_version(
    output_path: Path,
    reference_h: dict[str, np.memmap],
    reference_features: dict[str, dict[str, np.memmap]],
    regime_masks: dict[str, np.ndarray],
    batch: int = 4,
) -> dict[str, float]:
    current = np.load(output_path, mmap_mode="r")
    if current.shape != (500, 256, 4, 192) or current.dtype != np.complex64:
        raise ValueError(f"Invalid historical output {output_path}: {current.shape} {current.dtype}")
    point_metrics = {
        name: {
            "leaked": np.empty(500, dtype=np.float32),
            "collapsed": np.empty(500, dtype=np.float32),
            "pdp": np.empty(500, dtype=np.float32),
            "coherence": np.empty(500, dtype=np.float32),
        }
        for name in reference_features
    }
    current_power = np.empty(500, dtype=np.float64)
    global_accum = {
        name: {"dot": 0j, "current": 0.0, "reference": 0.0} for name in reference_features
    }
    for start in range(0, len(current), batch):
        end = min(start + batch, len(current))
        h = np.asarray(current[start:end], dtype=np.complex64)
        leaked, collapsed, delay, power = transform_channel(h)
        current_power[start:end] = power
        flat = h.reshape(len(h), -1)
        for name, features in reference_features.items():
            point_metrics[name]["leaked"][start:end] = np.mean(
                np.sum(leaked * features["leaked"][start:end], axis=(4, 5), dtype=np.float64),
                axis=(1, 2, 3),
            )
            point_metrics[name]["collapsed"][start:end] = np.sum(
                collapsed * features["collapsed"][start:end], axis=(1, 2), dtype=np.float64
            )
            point_metrics[name]["pdp"][start:end] = np.mean(
                np.sum(delay * features["delay"][start:end], axis=3, dtype=np.float64), axis=(1, 2)
            )
            ref_block = np.asarray(reference_h[name][start:end], dtype=np.complex64)
            ref_flat = ref_block.reshape(len(ref_block), -1)
            dot = np.sum(np.conj(flat) * ref_flat, axis=1, dtype=np.complex128)
            ref_power = features["power"][start:end]
            point_metrics[name]["coherence"][start:end] = (
                np.abs(dot) / np.maximum(np.sqrt(power * ref_power), FLOOR)
            ).astype(np.float32)
            global_accum[name]["dot"] += complex(np.sum(np.conj(flat) * ref_flat, dtype=np.complex128))
            global_accum[name]["current"] += float(np.sum(power))
            global_accum[name]["reference"] += float(np.sum(ref_power))

    row: dict[str, float] = {}
    for name, features in reference_features.items():
        ref_power = np.asarray(features["power"], dtype=np.float64)
        nonzero = ref_power > 0.0
        masks = {"all": np.ones(500, dtype=bool), "nonzero": nonzero, **regime_masks}
        for subset, mask in masks.items():
            leaked_mean = aggregate(point_metrics[name]["leaked"], mask)
            pdp_mean = aggregate(point_metrics[name]["pdp"], mask)
            power_spearman = safe_corr(
                np.log(np.maximum(current_power[mask], FLOOR)),
                np.log(np.maximum(ref_power[mask], FLOOR)),
                "spearman",
            )
            row[f"{name}_leaked_{subset}"] = leaked_mean
            row[f"{name}_collapsed_{subset}"] = aggregate(point_metrics[name]["collapsed"], mask)
            row[f"{name}_pdp_{subset}"] = pdp_mean
            row[f"{name}_coherence_{subset}"] = aggregate(point_metrics[name]["coherence"], mask)
            row[f"{name}_shape_{subset}"] = 0.5 * (leaked_mean + pdp_mean)
            row[f"{name}_power_spearman_{subset}"] = float(power_spearman or 0.0)
            row[f"{name}_selection_{subset}"] = (
                0.45 * leaked_mean + 0.45 * pdp_mean + 0.10 * (float(power_spearman or 0.0) + 1.0) / 2.0
            )
        accum = global_accum[name]
        coefficient = accum["dot"] / max(accum["current"], FLOOR)
        residual_energy = (
            accum["reference"]
            - 2.0 * float(np.real(np.conj(coefficient) * accum["dot"]))
            + abs(coefficient) ** 2 * accum["current"]
        )
        scaled_nmse = max(residual_energy, 0.0) / max(accum["reference"], FLOOR)
        row[f"{name}_complex_scaled_nmse"] = float(scaled_nmse)
        row[f"{name}_complex_scaled_score"] = float(1.0 / (1.0 + scaled_nmse))
    return row


def windows(version: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(version), dtype=bool),
        "v98_plus": version >= 98,
        "v138_plus": version >= 138,
        "denoise_track_v169_v186": (version >= 169) & (version <= 186),
        "metric_power_v187_v202": (version >= 187) & (version <= 202),
        "power_v196_v202": (version >= 196) & (version <= 202),
    }


def correlation_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    version = np.asarray([row["version"] for row in rows], dtype=np.float64)
    online = np.asarray([row["online"] for row in rows], dtype=np.float64)
    feature_names = [
        key for key in rows[0] if key not in {"version", "online", "output"} and isinstance(rows[0][key], float)
    ]
    output = []
    for window_name, mask in windows(version).items():
        if np.sum(mask) < 4:
            continue
        controls = polynomial_version_controls(version[mask])
        online_residual = residualize(online[mask], controls)
        for feature in feature_names:
            values = np.asarray([row[feature] for row in rows], dtype=np.float64)[mask]
            output.append(
                {
                    "window": window_name,
                    "count": int(np.sum(mask)),
                    "feature": feature,
                    "pearson": safe_corr(values, online[mask], "pearson"),
                    "spearman": safe_corr(values, online[mask], "spearman"),
                    "kendall": safe_corr(values, online[mask], "kendall"),
                    "version_detrended_pearson": safe_corr(
                        residualize(values, controls), online_residual, "pearson"
                    ),
                    "version_detrended_spearman": safe_corr(
                        residualize(values, controls), online_residual, "spearman"
                    ),
                }
            )
    return output


def partial_correlation(y: np.ndarray, x: np.ndarray, control: np.ndarray) -> float | None:
    return safe_corr(residualize(x, control), residualize(y, control), "pearson")


def bootstrap_difference(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    method: str,
    seed: int,
    repetitions: int = 2000,
) -> dict[str, float | None]:
    observed_a = safe_corr(a, y, method)
    observed_b = safe_corr(b, y, method)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        index = rng.integers(0, len(y), size=len(y))
        ca = safe_corr(a[index], y[index], method)
        cb = safe_corr(b[index], y[index], method)
        if ca is not None and cb is not None:
            differences.append(ca - cb)
    interval = np.quantile(differences, [0.025, 0.5, 0.975]) if differences else [np.nan] * 3
    return {
        "v206": observed_a,
        "v205": observed_b,
        "difference": None if observed_a is None or observed_b is None else observed_a - observed_b,
        "bootstrap_p025": float(interval[0]),
        "bootstrap_p50": float(interval[1]),
        "bootstrap_p975": float(interval[2]),
    }


def primary_comparison(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    version = np.asarray([row["version"] for row in rows], dtype=np.float64)
    online = np.asarray([row["online"] for row in rows], dtype=np.float64)
    output = []
    suffixes = ("selection_all", "shape_all", "leaked_all", "pdp_all", "selection_heavy")
    for window_name, mask in windows(version).items():
        if np.sum(mask) < 5:
            continue
        for suffix_index, suffix in enumerate(suffixes):
            v206 = np.asarray([row[f"v206_{suffix}"] for row in rows], dtype=np.float64)[mask]
            v205 = np.asarray([row[f"v205_{suffix}"] for row in rows], dtype=np.float64)[mask]
            y = online[mask]
            controls = polynomial_version_controls(version[mask])
            output.append(
                {
                    "window": window_name,
                    "count": int(np.sum(mask)),
                    "metric": suffix,
                    "pearson": bootstrap_difference(y, v206, v205, "pearson", 2070 + suffix_index),
                    "spearman": bootstrap_difference(y, v206, v205, "spearman", 2170 + suffix_index),
                    "v206_partial_controlling_v205": partial_correlation(y, v206, v205),
                    "v205_partial_controlling_v206": partial_correlation(y, v205, v206),
                    "v206_version_detrended_pearson": safe_corr(
                        residualize(v206, controls), residualize(y, controls), "pearson"
                    ),
                    "v205_version_detrended_pearson": safe_corr(
                        residualize(v205, controls), residualize(y, controls), "pearson"
                    ),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows: list[dict[str, object]], comparisons: list[dict[str, object]]) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    online = np.asarray([row["online"] for row in rows])
    version = np.asarray([row["version"] for row in rows])
    v206 = np.asarray([row["v206_selection_all"] for row in rows])
    v205 = np.asarray([row["v205_selection_all"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(v205, online, c=version, cmap="viridis", s=30)
    axes[0].set_title("Online vs v205 reference")
    axes[0].set_xlabel("v205 selection score")
    axes[0].set_ylabel("online score")
    scatter = axes[1].scatter(v206, online, c=version, cmap="viridis", s=30)
    axes[1].set_title("Online vs v206 reference")
    axes[1].set_xlabel("v206 selection score")
    fig.colorbar(scatter, ax=axes, label="version")
    fig.tight_layout()
    scatter_path = FIGURE_DIR / "v207_online_vs_sionna_references.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)

    primary = [item for item in comparisons if item["metric"] == "selection_all"]
    labels = [item["window"] for item in primary]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, [item["spearman"]["v205"] for item in primary], width, label="v205")
    ax.bar(x + width / 2, [item["spearman"]["v206"] for item in primary], width, label="v206")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Spearman with online score")
    ax.set_title("Historical online correlation by window")
    ax.legend()
    fig.tight_layout()
    bar_path = FIGURE_DIR / "v207_window_spearman_comparison.png"
    fig.savefig(bar_path, dpi=180)
    plt.close(fig)
    return [PATHS.relative(scatter_path), PATHS.relative(bar_path)]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    references = {"v206": V206_REFERENCE, "v205": V205_REFERENCE}
    reference_feature_paths = {
        name: ensure_reference_features(name, path) for name, path in references.items()
    }
    reference_features = {
        name: {
            key: np.load(paths[key], mmap_mode="r") for key in ("leaked", "collapsed", "delay", "power")
        }
        for name, paths in reference_feature_paths.items()
    }
    reference_h = {name: np.load(path, mmap_mode="r") for name, path in references.items()}
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"]
    regime = regime_of(blocked.astype(np.float32)).astype(str)
    regime_masks = {name: regime == name for name in REGIMES}

    rows: list[dict[str, object]] = []
    versions = online_versions()
    for index, (version, online, output) in enumerate(versions, start=1):
        row: dict[str, object] = {
            "version": version,
            "online": online,
            "output": PATHS.relative(output),
        }
        row.update(score_version(output, reference_h, reference_features, regime_masks))
        rows.append(row)
        print(f"v{version} scored ({index}/{len(versions)})", flush=True)

    correlations = correlation_table(rows)
    comparisons = primary_comparison(rows)
    figures = make_figures(rows, comparisons)
    write_csv(FEATURE_CSV, rows)
    write_csv(CORRELATION_CSV, correlations)

    v195_path = PATHS.analysis / "results" / "scorer_mouth_reverse_inference_v195_all_available" / "scorer_mouth_reverse_inference.json"
    v195 = json.loads(v195_path.read_text(encoding="utf-8")) if v195_path.is_file() else None
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A useful metric-equivalent Sionna configuration should order historical submissions like the hidden "
            "online target, and should outperform the v205 reference after controlling version progression."
        ),
        "references": {
            name: {
                "channel": PATHS.relative(path),
                "metadata": json.loads(reference_feature_paths[name]["metadata"].read_text(encoding="utf-8")),
            }
            for name, path in references.items()
        },
        "version_count": len(rows),
        "version_range": [int(rows[0]["version"]), int(rows[-1]["version"])],
        "versions": [int(row["version"]) for row in rows],
        "windows": {name: int(np.sum(mask)) for name, mask in windows(np.asarray([row["version"] for row in rows])).items()},
        "primary_comparison": comparisons,
        "all_correlations": correlations,
        "v195_empirical_reference_control": None
        if v195 is None
        else {
            "reference_version": v195["reference_version"],
            "version_count": v195["version_count"],
            "best_all_pearson": max(
                item["pearson_all"] for item in v195["correlations"] if item["pearson_all"] is not None
            ),
            "best_all_spearman": max(
                item["spearman_all"] for item in v195["correlations"] if item["spearman_all"] is not None
            ),
            "interpretation": "This is an ancestry/current-output control, not independent physical truth.",
        },
        "limitations": [
            "Historical versions are dependent and mostly improve sequentially; full-history correlation alone is not causal.",
            "The v206 channel is metric-equivalent on official train holdouts, not the hidden test truth.",
            "Only versions with both an online score and a retained full test-output cache are included.",
            "Within-window and version-detrended statistics have smaller sample size and wider uncertainty.",
        ],
        "outputs": {
            "report": PATHS.relative(REPORT),
            "feature_csv": PATHS.relative(FEATURE_CSV),
            "correlation_csv": PATHS.relative(CORRELATION_CSV),
            "figures": figures,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("version_count", "windows", "primary_comparison", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
