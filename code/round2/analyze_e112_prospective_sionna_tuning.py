#!/usr/bin/env python3
"""Prospectively select new spectral interpolation knobs using Sionna only."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from analyze_e111_sionna_hyperparameter_transfer import (
    ROBUSTNESS_PENALTY,
    SOURCES,
    collapse_grouped_amplitude,
    source_path,
)
from analyze_v3_spectral_interpolators import (
    CONFIGS,
    WEIGHTS as INTERPOLATOR_WEIGHTS,
    cosine,
    predict_spectrum,
)
from project_paths import PATHS


VERSION = "Round2-E112-prospective-Sionna-hyperparameter-tuning"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "prospective_sionna_tuning"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_prospective_sionna_tuning"
REPORT = ANALYSIS_DIR / "prospective_sionna_tuning.json"
SCORE_TABLE = ARTIFACT_DIR / "prospective_sionna_scores.npz"
SELECTED_ENDPOINT = ARTIFACT_DIR / "selected_official_global_endpoint.npz"
E111_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_sionna_hyperparameter_transfer"
    / "sionna_hyperparameter_transfer.json"
)
OFFICIAL_DESCRIPTORS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_spectral_descriptors.npz"
)

TOP_BASE_CONFIGS = 5
ENCODING_EXPONENTS = (0.25, 1.0 / 3.0, 0.40, 0.50, 0.60, 0.75, 1.00)
SIGNED_WEIGHT_SHRINKAGES = (0.00, 0.15, 0.30, 0.50, 0.75, 1.00)
MIN_GAIN_OVER_PRIOR_BEST = 0.0002
MIN_POSITIVE_FOLDS = 3
TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    base: str
    exponent: float
    shrinkage: float

    @property
    def name(self) -> str:
        exponent = f"{self.exponent:.6f}".rstrip("0").rstrip(".")
        shrinkage = f"{self.shrinkage:.2f}".rstrip("0").rstrip(".")
        return f"{self.base}_q{exponent}_shrink{shrinkage}"


def normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def shrink_weights(weights: np.ndarray, shrinkage: float) -> np.ndarray:
    original = np.asarray(weights, dtype=np.float32)
    if shrinkage == 0.0:
        return original
    positive = np.maximum(original, 0.0)
    total = positive.sum(axis=1, keepdims=True, dtype=np.float64)
    fallback = np.full_like(positive, 1.0 / positive.shape[1])
    positive = np.divide(
        positive,
        np.maximum(total, TINY),
        out=fallback,
        where=total > 0.0,
    )
    return (
        (1.0 - shrinkage) * original + shrinkage * positive
    ).astype(np.float32)


def predict_encoded(
    cache: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    exponent: float,
) -> np.ndarray:
    donor = np.maximum(np.asarray(cache[donor_ids], dtype=np.float32), 0.0)
    encoded = np.power(donor, exponent, dtype=np.float32)
    interpolated = np.maximum(
        np.einsum("qk,qkd->qd", weights, encoded, optimize=True), 0.0
    )
    prediction = np.power(interpolated, 1.0 / exponent, dtype=np.float32)
    return normalize_rows(prediction)


def score_candidate(
    pas: np.ndarray,
    pdp: np.ndarray,
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_weights = shrink_weights(weights, candidate.shrinkage)
    predicted_pas = predict_encoded(
        pas, donor_ids, local_weights, candidate.exponent
    )
    predicted_pdp = predict_encoded(
        pdp, donor_ids, local_weights, candidate.exponent
    )
    pas_score = cosine(predicted_pas, pas[query_ids])
    pdp_score = cosine(predicted_pdp, pdp[query_ids])
    return (
        pas_score,
        pdp_score,
        0.5 * (pas_score + pdp_score),
        predicted_pas,
        predicted_pdp,
    )


def fold_means(values: np.ndarray, fold_id: np.ndarray) -> list[float]:
    return [
        float(np.mean(values[fold_id == fold])) for fold in np.unique(fold_id)
    ]


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    e111 = json.loads(E111_REPORT.read_text(encoding="utf-8"))
    base_names = [
        str(row["name"]) for row in e111["ranking"][:TOP_BASE_CONFIGS]
    ]
    config_by_name = {config.name: config for config in CONFIGS}
    if not set(base_names).issubset(config_by_name):
        raise ValueError("E111 top configurations do not match the frozen family")
    candidates = [
        Candidate(base, exponent, shrinkage)
        for base in base_names
        for exponent in ENCODING_EXPONENTS
        for shrinkage in SIGNED_WEIGHT_SHRINKAGES
    ]

    cached = np.load(INTERPOLATOR_WEIGHTS, allow_pickle=False)
    query_ids = cached["query_ids"].astype(np.int64)
    fold_id = cached["fold_id"].astype(np.int8)
    source_scores = np.empty((len(SOURCES), len(candidates)), dtype=np.float64)

    for source_index, (source_name, prefix) in enumerate(SOURCES.items()):
        pas = collapse_grouped_amplitude(source_path(prefix, "pas"))
        pdp = collapse_grouped_amplitude(source_path(prefix, "pdp"))
        for candidate_index, candidate in enumerate(candidates):
            donor_ids = cached[f"{candidate.base}_donor_ids"].astype(np.int64)
            weights = cached[f"{candidate.base}_weights"].astype(np.float32)
            _, _, combined, _, _ = score_candidate(
                pas, pdp, query_ids, donor_ids, weights, candidate
            )
            source_scores[source_index, candidate_index] = float(
                np.mean(combined)
            )
        best = int(np.argmax(source_scores[source_index]))
        print(
            f"E112 source={source_name} best={candidates[best].name} "
            f"score={source_scores[source_index, best]:.6f}",
            flush=True,
        )

    synthetic_mean = source_scores.mean(axis=0)
    synthetic_std = source_scores.std(axis=0)
    robust_score = synthetic_mean - ROBUSTNESS_PENALTY * synthetic_std
    selected_index = int(np.argmax(robust_score))
    selected = candidates[selected_index]

    # From this line onward the Sionna-only selection is immutable.
    official = np.load(OFFICIAL_DESCRIPTORS, allow_pickle=False)
    official_pas = official["global_pas"].astype(np.float32)
    official_pdp = official["global_pdp"].astype(np.float32)
    controls = (
        Candidate("linear_const_v2", 0.50, 0.00),
        Candidate("linear_affine_s10", 0.50, 0.00),
        Candidate("linear_affine_s3", 0.50, 0.00),
    )
    evaluated = (selected, *controls)
    official_results: dict[str, dict[str, object]] = {}
    selected_pas = None
    selected_pdp = None
    for candidate in evaluated:
        donor_ids = cached[f"{candidate.base}_donor_ids"].astype(np.int64)
        weights = cached[f"{candidate.base}_weights"].astype(np.float32)
        pas_score, pdp_score, combined, predicted_pas, predicted_pdp = score_candidate(
            official_pas,
            official_pdp,
            query_ids,
            donor_ids,
            weights,
            candidate,
        )
        official_results[candidate.name] = {
            "candidate": asdict(candidate),
            "pas": float(np.mean(pas_score)),
            "pdp": float(np.mean(pdp_score)),
            "combined": float(np.mean(combined)),
            "combined_folds": fold_means(combined, fold_id),
        }
        if candidate == selected:
            selected_pas = predicted_pas
            selected_pdp = predicted_pdp

    if selected_pas is None or selected_pdp is None:
        raise RuntimeError("selected endpoint was not evaluated")
    np.savez_compressed(
        SELECTED_ENDPOINT,
        query_ids=query_ids,
        fold_id=fold_id,
        global_pas=selected_pas.astype(np.float16),
        global_pdp=selected_pdp.astype(np.float16),
        base=np.asarray(selected.base),
        exponent=np.float32(selected.exponent),
        shrinkage=np.float32(selected.shrinkage),
    )
    np.savez_compressed(
        SCORE_TABLE,
        candidate_names=np.asarray([candidate.name for candidate in candidates]),
        source_names=np.asarray(list(SOURCES)),
        source_scores=source_scores,
        synthetic_mean=synthetic_mean,
        synthetic_std=synthetic_std,
        robust_score=robust_score,
    )

    selected_result = official_results[selected.name]
    prior_best = official_results[Candidate("linear_affine_s10", 0.50, 0.00).name]
    fold_delta = np.asarray(selected_result["combined_folds"]) - np.asarray(
        prior_best["combined_folds"]
    )
    gain = float(selected_result["combined"] - prior_best["combined"])
    positive_folds = int(np.sum(fold_delta > 0.0))
    passed = bool(
        gain >= MIN_GAIN_OVER_PRIOR_BEST
        and positive_folds >= MIN_POSITIVE_FOLDS
    )

    top_rows = []
    for index in np.argsort(-robust_score, kind="stable")[:20]:
        top_rows.append(
            {
                "candidate": asdict(candidates[index]),
                "name": candidates[index].name,
                "synthetic_mean": float(synthetic_mean[index]),
                "synthetic_std": float(synthetic_std[index]),
                "robust_score": float(robust_score[index]),
                "source_scores": {
                    source_name: float(source_scores[source_index, index])
                    for source_index, source_name in enumerate(SOURCES)
                },
            }
        )

    payload = {
        "experiment": VERSION,
        "status": "strict_diagnostic_complete",
        "primary_hypothesis": (
            "A spectrum encoding exponent and signed-weight regularizer selected "
            "without official target values improves the official-H spatial field."
        ),
        "data_boundary": {
            "selection_uses_sionna_train_coordinates_only": True,
            "selection_uses_official_target_values": False,
            "sionna_test_coordinates_or_channels_used": False,
            "final_endpoint_uses_official_H_only": True,
            "sionna_weights_retained": False,
        },
        "search": {
            "base_configs": base_names,
            "encoding_exponents": list(ENCODING_EXPONENTS),
            "signed_weight_shrinkages": list(SIGNED_WEIGHT_SHRINKAGES),
            "candidate_count": len(candidates),
            "selection_formula": "mean_across_sources - 0.5 * std_across_sources",
            "selected": asdict(selected),
            "selected_name": selected.name,
            "top20": top_rows,
        },
        "official_frozen_evaluation": {
            "results": official_results,
            "selected_gain_over_prior_best": gain,
            "selected_fold_delta_over_prior_best": fold_delta.tolist(),
            "positive_folds": positive_folds,
        },
        "gate": {
            "minimum_gain_over_prior_best": MIN_GAIN_OVER_PRIOR_BEST,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "passed": passed,
            "next_step": (
                "expand_selected_rule_to_full_PAS_PDP_and_complete_score"
                if passed
                else "retain_E111_as_evidence_but_do_not_deploy_this_tuning"
            ),
        },
        "outputs": {
            "score_table": PATHS.relative(SCORE_TABLE),
            "selected_endpoint": PATHS.relative(SELECTED_ENDPOINT),
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
                "selected": asdict(selected),
                "official": official_results,
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
