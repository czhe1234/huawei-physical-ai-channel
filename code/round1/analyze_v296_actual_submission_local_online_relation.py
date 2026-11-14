#!/usr/bin/env python3
"""Score retained historical submissions against one local truth and compare online.

Only actual retained prediction arrays are paired with online scores. Missing old
submissions are reported rather than reconstructed approximately. The local score
is mathematically identical to v208 ``score_traceback_leaked`` while omitting
unneeded PAS/AD diagnostics for speed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

import analyze_v208_strict_historical_replay as v208
import analyze_v205_sionna_config_calibration as calibration
from project_paths import PATHS


VERSION_NUMBER = 296
VERSION = "v296_actual_submission_local_online_relation"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ONLINE_CSV = PATHS.analysis / "visualizations/version_scores/online_scores_by_version.csv"
SCORES_CSV = REPORT_DIR / "v296_actual_submission_local_scores.csv"
REPORT = REPORT_DIR / f"{VERSION}.json"
SCATTER = REPORT_DIR / "v296_local_vs_online_scatter.png"
CURVE = REPORT_DIR / "v296_local_online_version_curve.png"
DELTA_PLOT = REPORT_DIR / "v296_local_online_first_differences.png"
PDP_SCATTER = REPORT_DIR / "v296_local_pdp_vs_online_scatter.png"
TRUTH_DELAY = OUT_DIR / "v296_truth_delay_power.npy"
TRUTH_LEAKED = OUT_DIR / "v296_truth_leaked_power.npy"
TRUTH_ENERGY = OUT_DIR / "v296_truth_energy.npy"

BATCH = 8
EXPECTED_SHAPE = (500, 256, 4, 192)
REFERENCE_V212 = 0.6456036433098431


@dataclass(frozen=True)
class OnlineRow:
    version: int
    score: float
    method: str
    archive: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute existing rows")
    parser.add_argument(
        "--versions",
        default="",
        help="Optional comma-separated subset for smoke tests",
    )
    return parser.parse_args()


def load_online_rows() -> dict[int, OnlineRow]:
    rows: dict[int, OnlineRow] = {}
    with ONLINE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            version = int(row["version"])
            rows[version] = OnlineRow(
                version,
                float(row["score"]),
                row.get("method", ""),
                row.get("file") or None,
            )
    pattern = re.compile(r"^v(\d+)_(.+)_online_([0-9.]+)\.zip$")
    for archive in PATHS.round1_versions.glob("v*_online_*.zip"):
        match = pattern.match(archive.name)
        if match is None:
            continue
        version = int(match.group(1))
        score = float(match.group(3))
        previous = rows.get(version)
        if previous is not None and abs(previous.score - score) > 5e-7:
            raise ValueError(
                f"conflicting online score for v{version}: {previous.score} vs {score}"
            )
        rows[version] = OnlineRow(
            version,
            score,
            match.group(2),
            PATHS.relative(archive),
        )
    return rows


def version_from_path(path: Path) -> int | None:
    for value in (path.name, path.parent.name, str(path)):
        match = re.search(r"(?:^|[^a-zA-Z])v0*(\d+)", value)
        if match is not None:
            return int(match.group(1))
    return None


def prediction_priority(path: Path, version: int) -> tuple[int, int, str]:
    score = 0
    if path.name == f"v{version}_Round1_Test_Channel.npy":
        score += 100
    if path.parent.name in (f"v{version}", f"v{version:03d}"):
        score += 60
    if "version_cache" in path.parts:
        score += 30
    if "temporary" in path.parts:
        score += 10
    return score, -len(str(path)), str(path)


def discover_predictions() -> dict[int, Path]:
    candidates: dict[int, list[Path]] = {}
    for path in PATHS.artifacts.rglob("*Round1_Test_Channel.npy"):
        version = version_from_path(path)
        if version is not None:
            candidates.setdefault(version, []).append(path)
    selected = {
        version: max(paths, key=lambda path: prediction_priority(path, version))
        for version, paths in candidates.items()
    }
    # v246 is a byte-identical online package of the v239 strict output.
    v239 = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
    if v239.is_file():
        selected[246] = v239
    return selected


def leaked_power(h: np.ndarray) -> np.ndarray:
    value = h.reshape(len(h), 2, 16, 8, 4, 192)
    leaked = value.transpose(0, 1, 5, 2, 4, 3)
    leaked = np.abs(np.fft.fft2(leaked, axes=(2, 3))).astype(np.float32) ** 2
    return leaked.transpose(0, 1, 4, 5, 2, 3).reshape(
        len(h), 2, 4, 8, -1
    )


def build_truth_cache() -> None:
    expected_leaked = (EXPECTED_SHAPE[0], 2, 4, 8, 3072)
    cache_valid = False
    if TRUTH_DELAY.is_file() and TRUTH_LEAKED.is_file() and TRUTH_ENERGY.is_file():
        try:
            delay = np.load(TRUTH_DELAY, mmap_mode="r")
            leaked = np.load(TRUTH_LEAKED, mmap_mode="r")
            energy = np.load(TRUTH_ENERGY, mmap_mode="r")
            cache_valid = (
                delay.shape == EXPECTED_SHAPE
                and delay.dtype == np.float32
                and leaked.shape == expected_leaked
                and leaked.dtype == np.float32
                and energy.shape == (EXPECTED_SHAPE[0],)
            )
        except (OSError, ValueError):
            cache_valid = False
    if cache_valid:
        return

    truth = np.load(v208.TEST_TRUTH, mmap_mode="r")
    if truth.shape != EXPECTED_SHAPE or truth.dtype != np.complex64:
        raise ValueError(f"invalid local truth: {truth.shape} {truth.dtype}")
    delay_out = np.lib.format.open_memmap(
        TRUTH_DELAY, mode="w+", dtype=np.float32, shape=EXPECTED_SHAPE
    )
    leaked_out = np.lib.format.open_memmap(
        TRUTH_LEAKED, mode="w+", dtype=np.float32, shape=expected_leaked
    )
    energy_out = np.lib.format.open_memmap(
        TRUTH_ENERGY, mode="w+", dtype=np.float64, shape=(EXPECTED_SHAPE[0],)
    )
    for start in range(0, len(truth), BATCH):
        end = min(start + BATCH, len(truth))
        block = np.asarray(truth[start:end], dtype=np.complex64)
        delay_out[start:end] = (
            np.abs(np.fft.ifft(block, axis=-1)).astype(np.float32) ** 2
        )
        leaked_out[start:end] = leaked_power(block)
        energy_out[start:end] = np.sum(
            np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        if end % 100 == 0 or end == len(truth):
            print(f"v296 truth cache {end}/{len(truth)}", flush=True)
    delay_out.flush()
    leaked_out.flush()
    energy_out.flush()


def score_prediction(path: Path) -> dict[str, object]:
    prediction = np.load(path, mmap_mode="r")
    truth = np.load(v208.TEST_TRUTH, mmap_mode="r")
    truth_delay = np.load(TRUTH_DELAY, mmap_mode="r")
    truth_leaked = np.load(TRUTH_LEAKED, mmap_mode="r")
    truth_energy = np.load(TRUTH_ENERGY, mmap_mode="r")
    if prediction.shape != EXPECTED_SHAPE or prediction.dtype != np.complex64:
        raise ValueError(f"invalid prediction {path}: {prediction.shape} {prediction.dtype}")

    pdp_parts: list[np.ndarray] = []
    leaked_parts: list[np.ndarray] = []
    error = 0.0
    energy = 0.0
    finite = True
    for start in range(0, len(prediction), BATCH):
        end = min(start + BATCH, len(prediction))
        pred = np.asarray(prediction[start:end], dtype=np.complex64)
        target = np.asarray(truth[start:end], dtype=np.complex64)
        finite = finite and bool(np.isfinite(pred).all())
        pred_delay = np.abs(np.fft.ifft(pred, axis=-1)).astype(np.float32) ** 2
        pred_leaked = leaked_power(pred)
        pdp = calibration.cosine(
            pred_delay, np.asarray(truth_delay[start:end]), (3,)
        ).mean(axis=(1, 2))
        leaked = calibration.cosine(
            pred_leaked, np.asarray(truth_leaked[start:end]), (4,)
        ).mean(axis=(1, 2, 3))
        pred_energy = np.sum(
            np.abs(pred) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        target_energy = np.asarray(truth_energy[start:end])
        true_zero = target_energy <= 1e-25
        pred_zero = pred_energy <= 1e-25
        pdp[true_zero & pred_zero] = 1.0
        pdp[true_zero & ~pred_zero] = 0.0
        leaked[true_zero & pred_zero] = 1.0
        leaked[true_zero & ~pred_zero] = 0.0
        pdp_parts.append(pdp.astype(np.float32))
        leaked_parts.append(leaked.astype(np.float32))
        error += float(np.sum(np.abs(pred - target) ** 2, dtype=np.float64))
        energy += float(np.sum(np.abs(target) ** 2, dtype=np.float64))

    nmse = error / max(energy, 1e-30)
    nmse_score = 1.0 / (1.0 + nmse)
    pdp_mean = float(np.mean(np.concatenate(pdp_parts)))
    leaked_mean = float(np.mean(np.concatenate(leaked_parts)))
    return {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "finite": finite,
        "pdp": pdp_mean,
        "leaked_sh": leaked_mean,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "score_traceback_leaked": (
            0.4 * leaked_mean + 0.4 * pdp_mean + 0.2 * nmse_score
        ),
    }


CSV_FIELDS = [
    "version",
    "method",
    "online_score",
    "local_score",
    "pdp",
    "leaked_sh",
    "nmse_score",
    "global_nmse",
    "prediction",
]


def load_existing_scores() -> dict[int, dict[str, str]]:
    if not SCORES_CSV.is_file():
        return {}
    with SCORES_CSV.open(newline="", encoding="utf-8") as handle:
        return {int(row["version"]): row for row in csv.DictReader(handle)}


def write_scores(rows: dict[int, dict[str, object]]) -> None:
    with SCORES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for version in sorted(rows):
            writer.writerow(rows[version])


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if len(x) < 3 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return {"count": int(len(x))}
    return {
        "count": int(len(x)),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
    }


def residualize(value: np.ndarray, version: np.ndarray) -> np.ndarray:
    design = np.stack(
        [np.ones(len(version)), version, version**2], axis=1
    )
    coefficient = np.linalg.lstsq(design, value, rcond=None)[0]
    return value - design @ coefficient


def first_difference_summary(
    local: np.ndarray, online: np.ndarray
) -> dict[str, float | int]:
    delta_local = np.diff(local)
    delta_online = np.diff(online)
    nonzero = (np.abs(delta_local) > 1e-12) & (np.abs(delta_online) > 1e-12)
    result = correlation(delta_local, delta_online)
    result["direction_accuracy"] = (
        float(np.mean(np.sign(delta_local[nonzero]) == np.sign(delta_online[nonzero])))
        if np.any(nonzero)
        else float("nan")
    )
    result["nonzero_pair_count"] = int(np.sum(nonzero))
    return result


def stage_of(version: int) -> str:
    if version < 100:
        return "legacy_geometry_residual"
    if version <= 140:
        return "support_template"
    if version <= 187:
        return "track_denoising"
    if version <= 209:
        return "metric_aligned_official"
    if version <= 212:
        return "sionna_target"
    return "strict_physical_ai"


def analyze(rows: list[dict[str, object]], online_total: int, missing: list[int]) -> dict[str, object]:
    version = np.asarray([row["version"] for row in rows], dtype=np.float64)
    local = np.asarray([row["local_score"] for row in rows], dtype=np.float64)
    online = np.asarray([row["online_score"] for row in rows], dtype=np.float64)
    pdp = np.asarray([row["pdp"] for row in rows], dtype=np.float64)
    leaked_sh = np.asarray([row["leaked_sh"] for row in rows], dtype=np.float64)
    nmse_score = np.asarray([row["nmse_score"] for row in rows], dtype=np.float64)
    fit = np.polyfit(local, online, 1)
    fitted = np.polyval(fit, local)

    by_stage: dict[str, object] = {}
    stages = sorted({stage_of(int(value)) for value in version})
    for stage in stages:
        mask = np.asarray([stage_of(int(value)) == stage for value in version])
        by_stage[stage] = correlation(local[mask], online[mask])

    exclude_sionna = ~np.isin(version.astype(np.int64), [210, 211, 212])
    exclude_sionna_window = ~(
        (version.astype(np.int64) >= 210) & (version.astype(np.int64) <= 218)
    )
    through_v209 = version <= 209
    submetrics = {
        "pdp": pdp,
        "leaked_sh": leaked_sh,
        "nmse_score": nmse_score,
    }

    def submetric_summary(mask: np.ndarray) -> dict[str, object]:
        return {
            name: {
                "absolute": correlation(values[mask], online[mask]),
                "first_difference": first_difference_summary(
                    values[mask], online[mask]
                ),
            }
            for name, values in submetrics.items()
        }

    report = {
        "paired_count": int(len(rows)),
        "known_online_count": int(online_total),
        "coverage": float(len(rows) / max(online_total, 1)),
        "missing_online_versions": missing,
        "global": correlation(local, online),
        "quadratic_version_detrended": correlation(
            residualize(local, version), residualize(online, version)
        ),
        "first_difference": first_difference_summary(local, online),
        "first_difference_through_v209": first_difference_summary(
            local[through_v209], online[through_v209]
        ),
        "first_difference_excluding_v210_v218": first_difference_summary(
            local[exclude_sionna_window], online[exclude_sionna_window]
        ),
        "excluding_v210_v212": correlation(
            local[exclude_sionna], online[exclude_sionna]
        ),
        "excluding_v210_v218": correlation(
            local[exclude_sionna_window], online[exclude_sionna_window]
        ),
        "through_v209": correlation(local[through_v209], online[through_v209]),
        "submetrics": {
            "global": submetric_summary(np.ones(len(version), dtype=bool)),
            "through_v209": submetric_summary(through_v209),
            "excluding_v210_v218": submetric_summary(exclude_sionna_window),
        },
        "linear_mapping": {
            "online_intercept": float(fit[1]),
            "online_per_local": float(fit[0]),
            "rmse": float(np.sqrt(np.mean((online - fitted) ** 2))),
            "mae": float(np.mean(np.abs(online - fitted))),
            "r2": float(1.0 - np.sum((online - fitted) ** 2) / np.sum((online - np.mean(online)) ** 2)),
        },
        "by_stage": by_stage,
        "local_best": rows[int(np.argmax(local))],
        "online_best": rows[int(np.argmax(online))],
    }
    return report


def make_plots(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    version = np.asarray([row["version"] for row in rows], dtype=np.int64)
    local = np.asarray([row["local_score"] for row in rows], dtype=np.float64)
    online = np.asarray([row["online_score"] for row in rows], dtype=np.float64)
    milestones = {26, 55, 102, 121, 140, 163, 187, 189, 200, 209, 210, 211, 212, 215, 218, 246}

    figure, axis = plt.subplots(figsize=(10, 7))
    points = axis.scatter(local, online, c=version, cmap="viridis", s=34, alpha=0.85)
    slope = float(summary["linear_mapping"]["online_per_local"])
    intercept = float(summary["linear_mapping"]["online_intercept"])
    grid = np.linspace(float(np.min(local)), float(np.max(local)), 200)
    axis.plot(grid, intercept + slope * grid, color="#c23b22", linewidth=1.8)
    for row in rows:
        if int(row["version"]) in milestones:
            axis.annotate(
                f"v{row['version']}",
                (float(row["local_score"]), float(row["online_score"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Local score_traceback_leaked")
    axis.set_ylabel("Online score")
    axis.set_title("Actual retained submissions: local vs online")
    axis.grid(alpha=0.22)
    figure.colorbar(points, ax=axis, label="Version")
    figure.tight_layout()
    figure.savefig(SCATTER, dpi=180)
    plt.close(figure)

    pdp = np.asarray([row["pdp"] for row in rows], dtype=np.float64)
    pdp_fit = np.polyfit(pdp, online, 1)
    figure, axis = plt.subplots(figsize=(10, 7))
    points = axis.scatter(pdp, online, c=version, cmap="viridis", s=34, alpha=0.85)
    grid = np.linspace(float(np.min(pdp)), float(np.max(pdp)), 200)
    axis.plot(grid, np.polyval(pdp_fit, grid), color="#c23b22", linewidth=1.8)
    for row in rows:
        if int(row["version"]) in milestones:
            axis.annotate(
                f"v{row['version']}",
                (float(row["pdp"]), float(row["online_score"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Local PDP cosine")
    axis.set_ylabel("Online score")
    axis.set_title("Actual retained submissions: local PDP vs online")
    axis.grid(alpha=0.22)
    figure.colorbar(points, ax=axis, label="Version")
    figure.tight_layout()
    figure.savefig(PDP_SCATTER, dpi=180)
    plt.close(figure)

    figure, left = plt.subplots(figsize=(13, 6.5))
    right = left.twinx()
    left.plot(version, local, "o-", color="#1f77b4", markersize=3.2, linewidth=1.2, label="local")
    right.plot(version, online, "o-", color="#c23b22", markersize=3.2, linewidth=1.2, label="online")
    left.set_xlabel("Version")
    left.set_ylabel("Local score", color="#1f77b4")
    right.set_ylabel("Online score", color="#c23b22")
    left.set_title("Local and online score curves for retained actual submissions")
    left.grid(alpha=0.20)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="lower right")
    figure.tight_layout()
    figure.savefig(CURVE, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 6.5))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.plot(version[1:], np.diff(local), "o-", label="local delta", markersize=3)
    axis.plot(version[1:], np.diff(online), "o-", label="online delta", markersize=3)
    axis.set_xlabel("Destination version")
    axis.set_ylabel("Score change from previous retained online version")
    axis.set_title("First differences: does local predict online direction?")
    axis.grid(alpha=0.20)
    axis.legend()
    figure.tight_layout()
    figure.savefig(DELTA_PLOT, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    build_truth_cache()
    online = load_online_rows()
    predictions = discover_predictions()
    requested = {
        int(value) for value in args.versions.split(",") if value.strip()
    }
    versions = sorted(set(online) & set(predictions))
    if requested:
        versions = [version for version in versions if version in requested]
    missing = sorted(set(online) - set(predictions))

    stored = load_existing_scores()
    rows: dict[int, dict[str, object]] = {
        version: {
            key: (
                int(value)
                if key == "version"
                else float(value)
                if key in {"online_score", "local_score", "pdp", "leaked_sh", "nmse_score", "global_nmse"}
                else value
            )
            for key, value in row.items()
        }
        for version, row in stored.items()
        if not requested or version in requested
    }
    for index, version in enumerate(versions, start=1):
        if version in rows and not args.force:
            continue
        path = predictions[version]
        score = score_prediction(path)
        rows[version] = {
            "version": version,
            "method": online[version].method,
            "online_score": online[version].score,
            "local_score": score["score_traceback_leaked"],
            "pdp": score["pdp"],
            "leaked_sh": score["leaked_sh"],
            "nmse_score": score["nmse_score"],
            "global_nmse": score["global_nmse"],
            "prediction": PATHS.relative(path),
        }
        write_scores(rows)
        print(
            f"v296 score {index}/{len(versions)} v{version} "
            f"local={float(score['score_traceback_leaked']):.9f} "
            f"online={online[version].score:.6f}",
            flush=True,
        )
        if version == 212 and abs(float(score["score_traceback_leaked"]) - REFERENCE_V212) > 1e-6:
            raise AssertionError(
                f"v212 fast scorer mismatch: {score['score_traceback_leaked']} vs {REFERENCE_V212}"
            )

    paired = [rows[version] for version in sorted(rows) if version in online]
    if requested:
        print(f"v296 subset complete: {len(paired)} rows", flush=True)
        return
    summary = analyze(paired, len(online), missing)
    make_plots(paired, summary)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Scoring retained actual submissions against one fixed local Sionna truth "
            "reveals whether the v208 local gate predicts absolute online rank or only "
            "the shared chronological trend."
        ),
        "score_definition": (
            "0.4 * leaked_SH + 0.4 * PDP + 0.2 * 1/(1+global_NMSE), "
            "identical to v208 score_traceback_leaked"
        ),
        "data_policy": {
            "paired_outputs": "actual retained submission arrays only",
            "v246": "mapped to byte-identical v239 source output",
            "missing_versions": "reported, never approximated by retraining",
            "online_sources": [
                PATHS.relative(ONLINE_CSV),
                "versions/v*_online_SCORE.zip filenames",
            ],
        },
        "summary": summary,
        "rows": paired,
        "outputs": {
            "scores_csv": PATHS.relative(SCORES_CSV),
            "scatter": PATHS.relative(SCATTER),
            "pdp_scatter": PATHS.relative(PDP_SCATTER),
            "curve": PATHS.relative(CURVE),
            "first_difference_plot": PATHS.relative(DELTA_PLOT),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
