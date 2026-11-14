#!/usr/bin/env python3
"""Screen Wasserstein interpolation of Round 2 global PDP distributions."""

from __future__ import annotations

import json
import time

import numpy as np

from analyze_v3_spectral_interpolators import cosine, predict_spectrum
from data import load_positions
from project_paths import PATHS


ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_spectral_transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_spectral_transport"
REPORT = ANALYSIS_DIR / "wasserstein_pdp_screen.json"
PREDICTIONS = ARTIFACT_DIR / "wasserstein_pdp_oof.npz"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
OOF_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)

QUANTILE_COUNT = 512
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)


def normalized(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, 1e-20),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def quantile_table(distributions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    levels = (np.arange(QUANTILE_COUNT, dtype=np.float64) + 0.5) / QUANTILE_COUNT
    output = np.zeros((len(distributions), QUANTILE_COUNT), dtype=np.float32)
    for row in np.flatnonzero(valid):
        cdf = np.cumsum(distributions[row], dtype=np.float64)
        cdf /= max(float(cdf[-1]), 1e-30)
        output[row] = np.searchsorted(cdf, levels, side="left").astype(np.float32)
    return output


def distribution_from_quantiles(quantiles: np.ndarray, bins: int) -> np.ndarray:
    value = np.clip(np.maximum.accumulate(quantiles), 0.0, bins - 1.0)
    low = np.floor(value).astype(np.int64)
    high = np.minimum(low + 1, bins - 1)
    fraction = value - low
    output = np.zeros(bins, dtype=np.float32)
    mass = np.float32(1.0 / len(value))
    np.add.at(output, low, (1.0 - fraction).astype(np.float32) * mass)
    np.add.at(output, high, fraction.astype(np.float32) * mass)
    return output


def idw_weights(distance: np.ndarray, count: int, power: float) -> np.ndarray:
    selected = np.asarray(distance[:count], dtype=np.float64)
    if power == 0.0:
        value = np.ones(count, dtype=np.float64)
    else:
        value = 1.0 / np.maximum(selected, 0.25) ** power
    return (value / value.sum()).astype(np.float32)


def blend(base: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    amplitude = (1.0 - alpha) * np.sqrt(np.maximum(base, 0.0)) + alpha * np.sqrt(
        np.maximum(target, 0.0)
    )
    return normalized(amplitude * amplitude)


def summarize(values: np.ndarray, fold: np.ndarray, site: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [float(np.mean(values[fold == item])) for item in np.unique(fold)],
        "site": {
            str(int(item)): float(np.mean(values[site == item]))
            for item in np.unique(site)
        },
    }


def main() -> None:
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    descriptors = np.load(DESCRIPTORS)
    global_pdp = descriptors["global_pdp"].astype(np.float32)
    valid = descriptors["valid"].astype(bool)
    weights_cache = np.load(OOF_WEIGHTS)
    query_ids = weights_cache["query_ids"].astype(np.int64)
    fold_id = weights_cache["fold_id"].astype(np.int8)
    donor_ids = weights_cache["neighbor_ids"].astype(np.int64)
    rbf_weights = weights_cache["weights"].astype(np.float32)
    _, positions, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]
    distance = np.linalg.norm(
        positions[donor_ids, :2] - positions[query_ids, None, :2], axis=2
    )

    quantiles = quantile_table(global_pdp, valid)
    base = predict_spectrum(global_pdp, donor_ids, rbf_weights)
    truth = global_pdp[query_ids]
    baseline_score = cosine(base, truth)
    weight_configs: list[tuple[str, np.ndarray]] = [("signed_rbf", rbf_weights)]
    for count in (4, 8, 16, 32, 64):
        for power in (0.0, 1.0, 2.0):
            values = np.zeros_like(rbf_weights)
            for row in range(len(query_ids)):
                values[row, :count] = idw_weights(distance[row], count, power)
            weight_configs.append((f"idw_k{count}_p{power:g}", values))

    rows: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id,
        "base": base.astype(np.float16),
    }
    for name, interpolation_weights in weight_configs:
        barycenter = np.empty_like(base)
        for row in range(len(query_ids)):
            selected_quantiles = quantiles[donor_ids[row]]
            predicted_quantiles = interpolation_weights[row] @ selected_quantiles
            barycenter[row] = distribution_from_quantiles(
                predicted_quantiles, global_pdp.shape[1]
            )
        saved[name] = barycenter.astype(np.float16)
        action_rows: list[dict[str, object]] = []
        for alpha in ALPHAS:
            score = cosine(blend(base, barycenter, alpha), truth)
            action_rows.append(
                {
                    "alpha": alpha,
                    "score": summarize(score, fold_id, query_site),
                    "delta_vs_v2": float(np.mean(score - baseline_score)),
                    "delta_fold": [
                        float(np.mean((score - baseline_score)[fold_id == item]))
                        for item in np.unique(fold_id)
                    ],
                }
            )
        best = max(action_rows, key=lambda value: float(value["score"]["mean"]))
        rows.append({"weights": name, "best": best, "actions": action_rows})
        print(
            f"{name}: alpha={best['alpha']:.2f} PDP={best['score']['mean']:.6f} "
            f"delta={best['delta_vs_v2']:+.6f}",
            flush=True,
        )

    ranking = sorted(rows, key=lambda value: float(value["best"]["score"]["mean"]), reverse=True)
    payload = {
        "experiment": "Round2-E4-V3-Wasserstein-PDP-screen",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes",
        "quantile_count": QUANTILE_COUNT,
        "baseline": summarize(baseline_score, fold_id, query_site),
        "results": rows,
        "ranking": ranking,
        "seconds": float(time.time() - started),
    }
    np.savez_compressed(PREDICTIONS, **saved)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(ranking[:8], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
