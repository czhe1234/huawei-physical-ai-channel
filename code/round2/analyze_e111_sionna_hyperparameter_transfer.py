#!/usr/bin/env python3
"""Audit whether Sionna can select transferable spectral-field hyperparameters.

This experiment never uses Sionna spectra as model inputs, labels, checkpoints, or
test-point corrections.  It replays a frozen family of interpolation algorithms on
Sionna channels at official *training* coordinates, selects one configuration from
three simulator variants, and only then compares that selection with the existing
official-H spatial-hole ranking.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analyze_v3_spectral_interpolators import (
    CONFIGS,
    REPORT as OFFICIAL_INTERPOLATOR_REPORT,
    WEIGHTS as INTERPOLATOR_WEIGHTS,
    cosine,
    predict_spectrum,
)
from project_paths import PATHS


VERSION = "Round2-E111-Sionna-hyperparameter-transfer"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_hyperparameter_transfer"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sionna_hyperparameter_transfer"
SCORE_TABLE = ARTIFACT_DIR / "e111_interpolator_scores.npz"
REPORT = ANALYSIS_DIR / "sionna_hyperparameter_transfer.json"

SIONNA_DIR = PATHS.artifacts / "round2" / "v5_sionna_upper_bound"
SOURCES = {
    "depth16_base_v4lambda": "d16_base_v4lambda",
    "depth10_scatter015_v4lambda": "d10_scatter015_v4lambda",
    "depth16_base_v05lambda": "d16_base_v05lambda",
}

# Frozen before any E111 score is computed.  The standard-deviation penalty avoids
# choosing a parameter that only works for one simulator configuration.
ROBUSTNESS_PENALTY = 0.5
MIN_SPEARMAN = 0.50
MAX_OFFICIAL_RANK = 5
MAX_OFFICIAL_REGRET = 0.005
PERMUTATIONS = 20_000
TINY = 1e-30


def source_path(prefix: str, kind: str) -> str:
    suffix = {
        "ids": "ids.npy",
        "pas": "grouped_pas_f16.npy",
        "pdp": "grouped_pdp_f16.npy",
    }[kind]
    return str(SIONNA_DIR / f"{prefix}_train_full4000_{suffix}")


def normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def collapse_grouped_amplitude(path: str, batch_size: int = 128) -> np.ndarray:
    """Convert stored sqrt-power groups to one normalized global spectrum."""

    grouped = np.load(path, mmap_mode="r")
    if grouped.ndim != 3:
        raise ValueError(f"unexpected grouped spectrum shape: {grouped.shape}")
    output = np.empty((len(grouped), grouped.shape[-1]), dtype=np.float32)
    for start in range(0, len(grouped), batch_size):
        end = min(start + batch_size, len(grouped))
        amplitude = np.asarray(grouped[start:end], dtype=np.float32)
        power = np.square(amplitude, dtype=np.float32).sum(
            axis=1, dtype=np.float64
        )
        output[start:end] = normalize_rows(power.astype(np.float32))
    return output


def summarize(values: np.ndarray, fold_id: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "folds": [
            float(np.mean(values[fold_id == fold]))
            for fold in np.unique(fold_id)
        ],
    }


def evaluate_source(
    prefix: str,
    cached_weights: np.lib.npyio.NpzFile,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray]:
    ids = np.load(source_path(prefix, "ids"))
    expected_ids = np.arange(len(ids), dtype=ids.dtype)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{prefix} does not cover all training IDs in order")

    global_pas = collapse_grouped_amplitude(source_path(prefix, "pas"))
    global_pdp = collapse_grouped_amplitude(source_path(prefix, "pdp"))
    rows: list[dict[str, object]] = []
    combined_scores = np.empty(len(CONFIGS), dtype=np.float64)
    for config_index, config in enumerate(CONFIGS):
        donor_ids = cached_weights[f"{config.name}_donor_ids"].astype(np.int64)
        weights = cached_weights[f"{config.name}_weights"].astype(np.float32)
        predicted_pas = predict_spectrum(global_pas, donor_ids, weights)
        predicted_pdp = predict_spectrum(global_pdp, donor_ids, weights)
        pas = cosine(predicted_pas, global_pas[query_ids])
        pdp = cosine(predicted_pdp, global_pdp[query_ids])
        combined = 0.5 * (pas + pdp)
        combined_scores[config_index] = float(np.mean(combined))
        rows.append(
            {
                "config": asdict(config),
                "pas": summarize(pas, fold_id),
                "pdp": summarize(pdp, fold_id),
                "combined": summarize(combined, fold_id),
            }
        )
    return rows, combined_scores


def rank_map(values: np.ndarray, names: list[str]) -> dict[str, int]:
    order = np.argsort(-np.asarray(values), kind="stable")
    return {names[index]: rank + 1 for rank, index in enumerate(order)}


def permutation_pvalue(
    left: np.ndarray, right: np.ndarray, observed: float, seed: int = 20260808111
) -> float:
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(right)
        value = float(spearmanr(left, permuted).statistic)
        exceed += abs(value) >= abs(observed)
    return float((exceed + 1) / (PERMUTATIONS + 1))


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if not INTERPOLATOR_WEIGHTS.is_file():
        raise FileNotFoundError(INTERPOLATOR_WEIGHTS)
    if not OFFICIAL_INTERPOLATOR_REPORT.is_file():
        raise FileNotFoundError(OFFICIAL_INTERPOLATOR_REPORT)

    cached_weights = np.load(INTERPOLATOR_WEIGHTS, allow_pickle=False)
    query_ids = cached_weights["query_ids"].astype(np.int64)
    fold_id = cached_weights["fold_id"].astype(np.int8)
    names = [config.name for config in CONFIGS]
    source_rows: dict[str, list[dict[str, object]]] = {}
    source_scores = np.empty((len(SOURCES), len(CONFIGS)), dtype=np.float64)

    for source_index, (source_name, prefix) in enumerate(SOURCES.items()):
        rows, scores = evaluate_source(
            prefix, cached_weights, query_ids, fold_id
        )
        source_rows[source_name] = rows
        source_scores[source_index] = scores
        best = int(np.argmax(scores))
        print(
            f"E111 source={source_name} best={names[best]} "
            f"score={scores[best]:.6f}",
            flush=True,
        )

    synthetic_mean = source_scores.mean(axis=0)
    synthetic_std = source_scores.std(axis=0)
    robust_score = synthetic_mean - ROBUSTNESS_PENALTY * synthetic_std
    selected_index = int(np.argmax(robust_score))
    selected_name = names[selected_index]

    # Official results are read only after the simulator-only winner is frozen.
    official_payload = json.loads(
        OFFICIAL_INTERPOLATOR_REPORT.read_text(encoding="utf-8")
    )
    official_by_name = {
        str(row["config"]["name"]): row for row in official_payload["results"]
    }
    if set(official_by_name) != set(names):
        raise ValueError("official and simulator configuration families differ")
    official_scores = np.asarray(
        [official_by_name[name]["combined"]["mean"] for name in names],
        dtype=np.float64,
    )
    official_ranks = rank_map(official_scores, names)
    synthetic_ranks = rank_map(robust_score, names)
    pearson = float(pearsonr(robust_score, official_scores).statistic)
    spearman = float(spearmanr(robust_score, official_scores).statistic)
    pvalue = permutation_pvalue(robust_score, official_scores, spearman)
    official_best_index = int(np.argmax(official_scores))
    official_regret = float(
        official_scores[official_best_index] - official_scores[selected_index]
    )
    selected_official_rank = int(official_ranks[selected_name])

    source_correlations = {}
    for source_index, source_name in enumerate(SOURCES):
        source_correlations[source_name] = {
            "pearson": float(
                pearsonr(source_scores[source_index], official_scores).statistic
            ),
            "spearman": float(
                spearmanr(source_scores[source_index], official_scores).statistic
            ),
        }

    top5_synthetic = {
        name for name, rank in synthetic_ranks.items() if rank <= 5
    }
    top5_official = {name for name, rank in official_ranks.items() if rank <= 5}
    top5_overlap = sorted(top5_synthetic & top5_official)
    passed = bool(
        spearman >= MIN_SPEARMAN
        and selected_official_rank <= MAX_OFFICIAL_RANK
        and official_regret <= MAX_OFFICIAL_REGRET
    )

    ranking = []
    for index in np.argsort(-robust_score, kind="stable"):
        ranking.append(
            {
                "name": names[index],
                "synthetic_rank": int(synthetic_ranks[names[index]]),
                "synthetic_mean": float(synthetic_mean[index]),
                "synthetic_std": float(synthetic_std[index]),
                "robust_score": float(robust_score[index]),
                "source_scores": {
                    source_name: float(source_scores[source_index, index])
                    for source_index, source_name in enumerate(SOURCES)
                },
                "official_rank": int(official_ranks[names[index]]),
                "official_score": float(official_scores[index]),
            }
        )

    np.savez_compressed(
        SCORE_TABLE,
        config_names=np.asarray(names),
        source_names=np.asarray(list(SOURCES)),
        source_scores=source_scores,
        synthetic_mean=synthetic_mean,
        synthetic_std=synthetic_std,
        robust_score=robust_score,
        official_scores=official_scores,
        query_ids=query_ids,
        fold_id=fold_id,
    )
    payload = {
        "experiment": VERSION,
        "status": "strict_diagnostic_complete",
        "primary_hypothesis": (
            "A hyperparameter family ranked only on Sionna channels at official "
            "training coordinates preserves its ordering on official-H spatial holes."
        ),
        "data_boundary": {
            "sionna_train_coordinates_only": True,
            "sionna_test_coordinates_used": False,
            "sionna_weights_or_predictions_retained_by_final_model": False,
            "official_test_channel_used": False,
            "query_count": int(len(query_ids)),
            "sources": SOURCES,
        },
        "selection_rule": {
            "formula": "mean_across_sources - 0.5 * std_across_sources",
            "robustness_penalty": ROBUSTNESS_PENALTY,
            "frozen_before_official_comparison": True,
            "selected": selected_name,
        },
        "transfer": {
            "pearson": pearson,
            "spearman": spearman,
            "spearman_permutation_pvalue": pvalue,
            "source_correlations": source_correlations,
            "selected_official_rank": selected_official_rank,
            "selected_official_score": float(official_scores[selected_index]),
            "official_best": names[official_best_index],
            "official_best_score": float(official_scores[official_best_index]),
            "official_regret": official_regret,
            "top5_overlap": top5_overlap,
            "top5_overlap_count": len(top5_overlap),
        },
        "gate": {
            "minimum_spearman": MIN_SPEARMAN,
            "maximum_official_rank": MAX_OFFICIAL_RANK,
            "maximum_official_regret": MAX_OFFICIAL_REGRET,
            "passed": passed,
            "next_step": (
                "prospective_simulator_only_search_then_official_retrain"
                if passed
                else "stop_using_sionna_as_numeric_hyperparameter_selector"
            ),
        },
        "ranking": ranking,
        "source_results": source_rows,
        "outputs": {
            "score_table": PATHS.relative(SCORE_TABLE),
            "candidate_generated": False,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": selected_name,
                "transfer": payload["transfer"],
                "gate": payload["gate"],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
