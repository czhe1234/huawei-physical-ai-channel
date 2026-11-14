#!/usr/bin/env python3
"""Score a Sionna CFR candidate against the portable official-H subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from project_paths import PATHS


INPUT_DIR = PATHS.artifacts / "task_inputs"
TRUTH_PATH = INPUT_DIR / "official_channel_subset.npy"
METADATA_PATH = INPUT_DIR / "official_subset_metadata.npz"
SPLIT_NAMES = {0: "calibration", 1: "validation", 2: "audit"}
REGIME_NAMES = {0: "clear", 1: "light", 2: "medium", 3: "heavy"}


def cosine(a: np.ndarray, b: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    numerator = np.sum(a * b, axis=axes, dtype=np.float64)
    aa = np.sum(a * a, axis=axes, dtype=np.float64)
    bb = np.sum(b * b, axis=axes, dtype=np.float64)
    value = numerator / np.maximum(np.sqrt(aa * bb), 1e-30)
    value = np.where((aa <= 1e-30) & (bb <= 1e-30), 1.0, value)
    return value.astype(np.float32)


def metric_values(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    if prediction.shape != truth.shape:
        raise ValueError(f"shape mismatch: prediction={prediction.shape}, truth={truth.shape}")
    if prediction.ndim != 4 or prediction.shape[1:] != (256, 4, 192):
        raise ValueError(f"unexpected channel shape: {prediction.shape}")
    count = len(prediction)
    pred6 = prediction.reshape(count, 2, 16, 8, 4, 192)
    truth6 = truth.reshape(count, 2, 16, 8, 4, 192)

    pred_leaked = pred6.transpose(0, 1, 5, 2, 4, 3)
    truth_leaked = truth6.transpose(0, 1, 5, 2, 4, 3)
    pred_leaked = np.abs(np.fft.fft2(pred_leaked, axes=(2, 3))).astype(np.float32) ** 2
    truth_leaked = np.abs(np.fft.fft2(truth_leaked, axes=(2, 3))).astype(np.float32) ** 2
    leaked_full = cosine(pred_leaked, truth_leaked, (2, 3)).mean(axis=(1, 2, 3))
    pred_leaked_sum = np.sum(pred_leaked, axis=(1, 4, 5), dtype=np.float64)
    truth_leaked_sum = np.sum(truth_leaked, axis=(1, 4, 5), dtype=np.float64)
    leaked_collapsed = cosine(pred_leaked_sum, truth_leaked_sum, (1, 2))

    pred_delay = np.abs(np.fft.ifft(prediction, axis=-1)).astype(np.float32) ** 2
    truth_delay = np.abs(np.fft.ifft(truth, axis=-1)).astype(np.float32) ** 2
    pdp = cosine(pred_delay, truth_delay, (3,)).mean(axis=(1, 2))

    pred_flat = prediction.reshape(count, -1)
    truth_flat = truth.reshape(count, -1)
    pred_power = np.sum(np.abs(pred_flat) ** 2, axis=1, dtype=np.float64)
    truth_power = np.sum(np.abs(truth_flat) ** 2, axis=1, dtype=np.float64)
    complex_dot = np.sum(np.conj(pred_flat) * truth_flat, axis=1, dtype=np.complex128)
    coherence = np.abs(complex_dot) / np.maximum(np.sqrt(pred_power * truth_power), 1e-30)
    return {
        "leaked_full": leaked_full.astype(np.float32),
        "leaked_collapsed": leaked_collapsed.astype(np.float32),
        "pdp": pdp.astype(np.float32),
        "complex_coherence": coherence.astype(np.float32),
        "pred_log_power": np.log(np.maximum(pred_power, 1e-30)).astype(np.float32),
        "truth_log_power": np.log(np.maximum(truth_power, 1e-30)).astype(np.float32),
    }


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ranked_a = rankdata(np.asarray(a, dtype=np.float64))
    ranked_b = rankdata(np.asarray(b, dtype=np.float64))
    if np.std(ranked_a) == 0.0 or np.std(ranked_b) == 0.0:
        return 0.0
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def summarize(point_metrics: dict[str, np.ndarray]) -> dict[str, float | int]:
    power_spearman = rank_correlation(
        point_metrics["pred_log_power"], point_metrics["truth_log_power"]
    )
    power_rank = (power_spearman + 1.0) / 2.0
    offset = float(
        np.median(point_metrics["truth_log_power"] - point_metrics["pred_log_power"])
    )
    calibrated_power_error = np.abs(
        point_metrics["pred_log_power"] + offset - point_metrics["truth_log_power"]
    )
    coverage = float(np.mean(point_metrics["pred_log_power"] > -60.0))
    leaked_full = float(np.mean(point_metrics["leaked_full"]))
    pdp = float(np.mean(point_metrics["pdp"]))
    return {
        "count": int(len(point_metrics["pdp"])),
        "leaked_full_mean": leaked_full,
        "leaked_collapsed_mean": float(np.mean(point_metrics["leaked_collapsed"])),
        "pdp_mean": pdp,
        "complex_coherence_mean": float(np.mean(point_metrics["complex_coherence"])),
        "power_spearman": power_spearman,
        "power_offset_median": offset,
        "calibrated_log_power_mae": float(np.mean(calibrated_power_error)),
        "coverage": coverage,
        "simulated_zero_count": int(np.sum(point_metrics["pred_log_power"] <= -60.0)),
        "legacy_selection": 0.45 * leaked_full + 0.45 * pdp + 0.10 * power_rank,
        "online_informed_selection": (
            0.60 * pdp + 0.25 * leaked_full + 0.10 * power_rank + 0.05 * coverage
        ),
    }


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def score_prediction(
    prediction: np.ndarray,
    truth: np.ndarray,
    split_code: np.ndarray,
    regime_code: np.ndarray,
    *,
    batch_size: int = 8,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    parts: list[dict[str, np.ndarray]] = []
    for start in range(0, len(prediction), batch_size):
        end = min(start + batch_size, len(prediction))
        parts.append(
            metric_values(
                np.asarray(prediction[start:end], dtype=np.complex64),
                np.asarray(truth[start:end], dtype=np.complex64),
            )
        )
    point_metrics = concatenate(parts)
    report: dict[str, object] = {"overall": summarize(point_metrics), "by_split": {}}
    for code, name in SPLIT_NAMES.items():
        split_mask = split_code == code
        if not np.any(split_mask):
            continue
        split_metrics = {key: value[split_mask] for key, value in point_metrics.items()}
        split_report: dict[str, object] = summarize(split_metrics)
        split_report["by_regime"] = {}
        for regime, regime_name in REGIME_NAMES.items():
            mask = split_mask & (regime_code == regime)
            if np.any(mask):
                split_report["by_regime"][regime_name] = summarize(
                    {key: value[mask] for key, value in point_metrics.items()}
                )
        report["by_split"][name] = split_report
    return report, point_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--point-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = np.load(METADATA_PATH)
    truth = np.load(TRUTH_PATH, mmap_mode="r")
    prediction = np.load(args.candidate, mmap_mode="r")
    if len(prediction) > len(truth):
        raise ValueError(f"candidate has too many rows: {len(prediction)} > {len(truth)}")
    count = len(prediction)
    report, point_metrics = score_prediction(
        prediction,
        truth[:count],
        metadata["split_code"][:count],
        metadata["regime_code"][:count],
        batch_size=args.batch_size,
    )
    report["candidate"] = str(args.candidate)
    report["row_count"] = count
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    if args.point_output:
        args.point_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.point_output, **point_metrics)
    print(text)


if __name__ == "__main__":
    main()
