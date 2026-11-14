#!/usr/bin/env python3
"""Likelihood-free Bayesian model averaging of existing Sionna spectra."""

from __future__ import annotations

import json

import numpy as np

import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 529
VERSION = "v529_sbi_configuration_posterior"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = CACHE_DIR / f"{VERSION}_oof.npz"

PRIOR_DRAWS = 4096
PRIOR_CONCENTRATION = 0.5
TARGET_ESS_FRACTION = 0.08
SPATIAL_STD_PENALTY = 0.25
RANDOM_SEED = 20260724


def point_terms(
    truth: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress spectra into the terms needed to score arbitrary mixtures."""
    truth64 = np.asarray(truth, dtype=np.float64)
    candidates64 = np.asarray(candidates, dtype=np.float64)
    truth_norm = np.sqrt(np.sum(truth64 * truth64, axis=1))
    target_dot = np.einsum(
        "knd,nd->nk", candidates64, truth64, optimize=True
    )
    gram = np.einsum(
        "knd,lnd->nkl", candidates64, candidates64, optimize=True
    )
    return target_dot, gram, truth_norm


def mixture_cosine_from_terms(
    target_dot: np.ndarray,
    gram: np.ndarray,
    truth_norm: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weights64 = np.asarray(weights, dtype=np.float64)
    numerator = target_dot @ weights64
    mixture_norm_sq = np.einsum(
        "k,nkl,l->n", weights64, gram, weights64, optimize=True
    )
    denominator = truth_norm * np.sqrt(np.maximum(mixture_norm_sq, 1e-30))
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


def sampled_mixture_cosines(
    target_dot: np.ndarray,
    gram: np.ndarray,
    truth_norm: np.ndarray,
    draws: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    output = np.empty((len(target_dot), len(draws)), dtype=np.float32)
    target_dot64 = np.asarray(target_dot, dtype=np.float64)
    gram64 = np.asarray(gram, dtype=np.float64)
    truth_norm64 = np.asarray(truth_norm, dtype=np.float64)
    for start in range(0, len(draws), batch_size):
        stop = min(start + batch_size, len(draws))
        batch = np.asarray(draws[start:stop], dtype=np.float64)
        numerator = target_dot64 @ batch.T
        norm_sq = np.einsum(
            "bk,nkl,bl->nb", batch, gram64, batch, optimize=True
        )
        denominator = truth_norm64[:, None] * np.sqrt(
            np.maximum(norm_sq, 1e-30)
        )
        output[:, start:stop] = (
            numerator / np.maximum(denominator, 1e-30)
        ).astype(np.float32)
    return output


def normalized_importance(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64)
    shifted -= np.max(shifted)
    probability = np.exp(np.clip(shifted, -745.0, 0.0))
    probability /= np.maximum(np.sum(probability), 1e-300)
    return probability


def effective_sample_size(probability: np.ndarray) -> float:
    return float(1.0 / np.maximum(np.sum(probability * probability), 1e-300))


def select_temperature(scores: np.ndarray, target_ess: float) -> float:
    spread = float(np.max(scores) - np.min(scores))
    if spread <= 1e-12:
        return 1.0
    low = max(spread * 1e-8, 1e-12)
    high = max(spread * 100.0, 1e-6)
    for _ in range(64):
        middle = np.sqrt(low * high)
        probability = normalized_importance((scores - np.max(scores)) / middle)
        if effective_sample_size(probability) < target_ess:
            low = middle
        else:
            high = middle
    return float(high)


def infer_posterior(
    sampled_cosine: np.ndarray,
    row_ids: np.ndarray,
    fold_id: np.ndarray,
    draws: np.ndarray,
) -> dict[str, np.ndarray | float]:
    present_folds = np.unique(fold_id[row_ids])
    selected = np.zeros(len(fold_id), dtype=bool)
    selected[row_ids] = True
    fold_scores = np.stack(
        [
            np.mean(
                sampled_cosine[selected & (fold_id == fold)],
                axis=0,
                dtype=np.float64,
            )
            for fold in present_folds
        ],
        axis=0,
    )
    robust_score = np.mean(fold_scores, axis=0) - SPATIAL_STD_PENALTY * np.std(
        fold_scores, axis=0
    )
    target_ess = max(TARGET_ESS_FRACTION * len(draws), 2.0)
    temperature = select_temperature(robust_score, target_ess)
    probability = normalized_importance(
        (robust_score - np.max(robust_score)) / temperature
    )
    mean_weight = probability @ draws
    centered = draws - mean_weight[None, :]
    covariance = (centered * probability[:, None]).T @ centered
    return {
        "weight": mean_weight.astype(np.float32),
        "covariance": covariance.astype(np.float32),
        "temperature": float(temperature),
        "ess": effective_sample_size(probability),
        "map_weight": draws[int(np.argmax(probability))].astype(np.float32),
        "robust_score_mean": float(probability @ robust_score),
        "robust_score_map": float(np.max(robust_score)),
    }


def weighted_rows(
    candidates: np.ndarray,
    weights: dict[str, np.ndarray],
    regimes: np.ndarray,
) -> np.ndarray:
    output = np.empty((candidates.shape[1], candidates.shape[2]), dtype=np.float32)
    for regime in v494.REGIMES:
        local = regimes == regime
        if not np.any(local):
            continue
        output[local] = np.tensordot(
            weights[regime], candidates[:, local], axes=(0, 0)
        ).astype(np.float32)
    return v494.normalize_rows(output)


def best_single_weights(
    truth: np.ndarray,
    candidates: np.ndarray,
    ids: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for regime in v494.REGIMES:
        local_ids = ids[regimes[ids] == regime]
        score = np.asarray(
            [
                np.mean(v432.row_cosine(truth[local_ids], source[local_ids]))
                for source in candidates
            ],
            dtype=np.float64,
        )
        weight = np.zeros(len(candidates), dtype=np.float32)
        weight[int(np.argmax(score))] = 1.0
        result[regime] = weight
    return result


def summarize(value: np.ndarray) -> dict[str, float]:
    value64 = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value64)),
        "p10": float(np.quantile(value64, 0.10)),
        "median": float(np.median(value64)),
        "p90": float(np.quantile(value64, 0.90)),
    }


def main() -> None:
    for path in (v494.OFFICIAL, v494.POSITIONS, v494.ALL_INTERACTION, v434.OOF_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    truth, sources, quality = v494.load_sources()
    positions = np.load(v494.POSITIONS).astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    fold_id = np.load(v434.OOF_CACHE)["fold_id"].astype(np.int16)
    draws = np.random.default_rng(RANDOM_SEED).dirichlet(
        np.full(len(sources), PRIOR_CONCENTRATION, dtype=np.float64),
        size=PRIOR_DRAWS,
    ).astype(np.float32)

    baseline_prediction = np.zeros_like(truth)
    old_prediction = np.zeros_like(truth)
    single_prediction = np.zeros_like(truth)
    stacked_prediction = np.zeros_like(truth)
    sbi_prediction = np.zeros_like(truth)
    sbi_map_prediction = np.zeros_like(truth)
    fold_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []

    for outer in range(5):
        query_ids = np.flatnonzero(fold_id == outer)
        fit_ids = np.flatnonzero(fold_id != outer)
        inner_baseline = np.zeros_like(truth)
        inner_candidates = np.zeros(
            (len(sources), len(truth), truth.shape[1]), dtype=np.float32
        )
        for inner in range(5):
            if inner == outer:
                continue
            inner_query = np.flatnonzero(fold_id == inner)
            inner_fit = np.flatnonzero((fold_id != outer) & (fold_id != inner))
            inner_baseline[inner_query] = v432.neighbor_prediction(
                inner_fit,
                inner_query,
                positions,
                blocked,
                regimes,
                truth,
                k=48,
            )
            inner_candidates[:, inner_query] = v494.calibrated_sources(
                truth, sources, inner_fit, inner_query, regimes
            )

        query_baseline = v432.neighbor_prediction(
            fit_ids,
            query_ids,
            positions,
            blocked,
            regimes,
            truth,
            k=48,
        )
        query_candidates = v494.calibrated_sources(
            truth, sources, fit_ids, query_ids, regimes
        )

        fit_candidates = inner_candidates[:, fit_ids]
        target_dot, gram, truth_norm = point_terms(truth[fit_ids], fit_candidates)
        sampled_cosine = sampled_mixture_cosines(
            target_dot, gram, truth_norm, draws
        )
        fit_fold_id = fold_id[fit_ids]
        fit_regimes = regimes[fit_ids]

        inner_sbi_target = np.zeros_like(truth)
        inner_sbi_map_target = np.zeros_like(truth)
        final_sbi_weights: dict[str, np.ndarray] = {}
        final_sbi_map_weights: dict[str, np.ndarray] = {}
        posterior_summary: dict[str, dict[str, object]] = {}
        for regime in v494.REGIMES:
            regime_rows = np.flatnonzero(fit_regimes == regime)
            final_posterior = infer_posterior(
                sampled_cosine,
                regime_rows,
                fit_fold_id,
                draws,
            )
            final_sbi_weights[regime] = np.asarray(
                final_posterior["weight"], dtype=np.float32
            )
            final_sbi_map_weights[regime] = np.asarray(
                final_posterior["map_weight"], dtype=np.float32
            )
            posterior_summary[regime] = {
                "weight": final_sbi_weights[regime].tolist(),
                "map_weight": np.asarray(final_posterior["map_weight"]).tolist(),
                "temperature": float(final_posterior["temperature"]),
                "ess": float(final_posterior["ess"]),
                "weight_std": np.sqrt(
                    np.maximum(np.diag(np.asarray(final_posterior["covariance"])), 0.0)
                ).tolist(),
            }

        for inner in range(5):
            if inner == outer:
                continue
            validation_global_ids = np.flatnonzero(fold_id == inner)
            validation_rows = np.flatnonzero(fit_fold_id == inner)
            train_rows = np.flatnonzero(fit_fold_id != inner)
            local_weights: dict[str, np.ndarray] = {}
            local_map_weights: dict[str, np.ndarray] = {}
            for regime in v494.REGIMES:
                regime_train_rows = train_rows[fit_regimes[train_rows] == regime]
                posterior = infer_posterior(
                    sampled_cosine,
                    regime_train_rows,
                    fit_fold_id,
                    draws,
                )
                local_weights[regime] = np.asarray(posterior["weight"], dtype=np.float32)
                local_map_weights[regime] = np.asarray(
                    posterior["map_weight"], dtype=np.float32
                )
            inner_sbi_target[validation_global_ids] = weighted_rows(
                inner_candidates[:, validation_global_ids],
                local_weights,
                regimes[validation_global_ids],
            )
            inner_sbi_map_target[validation_global_ids] = weighted_rows(
                inner_candidates[:, validation_global_ids],
                local_map_weights,
                regimes[validation_global_ids],
            )
            if len(validation_rows) != len(validation_global_ids):
                raise RuntimeError("inner fold row mapping mismatch")

        sbi_query_target = weighted_rows(
            query_candidates, final_sbi_weights, regimes[query_ids]
        )
        sbi_map_query_target = weighted_rows(
            query_candidates, final_sbi_map_weights, regimes[query_ids]
        )

        stacked_weights: dict[str, np.ndarray] = {}
        for regime in v494.REGIMES:
            regime_fit = fit_ids[regimes[fit_ids] == regime]
            stacked_weights[regime] = v494.fit_convex_weights(
                truth, inner_candidates, regime_fit
            )
        inner_stacked_target = np.zeros_like(truth)
        inner_stacked_target[fit_ids] = v494.weighted_target(
            inner_candidates, fit_ids, regimes, stacked_weights
        )
        stacked_query_target = v494.weighted_target(
            query_candidates,
            np.arange(len(query_ids)),
            regimes[query_ids],
            stacked_weights,
        )

        single_weights = best_single_weights(
            truth, inner_candidates, fit_ids, regimes
        )
        inner_single_target = np.zeros_like(truth)
        inner_single_target[fit_ids] = weighted_rows(
            inner_candidates[:, fit_ids], single_weights, regimes[fit_ids]
        )
        single_query_target = weighted_rows(
            query_candidates, single_weights, regimes[query_ids]
        )

        old_alpha = v494.select_alpha(
            truth, inner_baseline, inner_candidates[0], quality, fit_ids, regimes
        )
        single_alpha = v494.select_alpha(
            truth,
            inner_baseline,
            inner_single_target,
            quality,
            fit_ids,
            regimes,
        )
        stacked_alpha = v494.select_alpha(
            truth,
            inner_baseline,
            inner_stacked_target,
            quality,
            fit_ids,
            regimes,
        )
        sbi_alpha = v494.select_alpha(
            truth,
            inner_baseline,
            inner_sbi_target,
            quality,
            fit_ids,
            regimes,
        )
        sbi_map_alpha = v494.select_alpha(
            truth,
            inner_baseline,
            inner_sbi_map_target,
            quality,
            fit_ids,
            regimes,
        )

        baseline_prediction[query_ids] = v494.normalize_rows(query_baseline)
        old_prediction[query_ids] = v494.blend_target(
            query_baseline,
            query_candidates[0],
            quality,
            np.arange(len(query_ids)),
            regimes[query_ids],
            old_alpha,
        )
        single_prediction[query_ids] = v494.blend_target(
            query_baseline,
            single_query_target,
            quality,
            np.arange(len(query_ids)),
            regimes[query_ids],
            single_alpha,
        )
        stacked_prediction[query_ids] = v494.blend_target(
            query_baseline,
            stacked_query_target,
            quality,
            np.arange(len(query_ids)),
            regimes[query_ids],
            stacked_alpha,
        )
        sbi_prediction[query_ids] = v494.blend_target(
            query_baseline,
            sbi_query_target,
            quality,
            np.arange(len(query_ids)),
            regimes[query_ids],
            sbi_alpha,
        )
        sbi_map_prediction[query_ids] = v494.blend_target(
            query_baseline,
            sbi_map_query_target,
            quality,
            np.arange(len(query_ids)),
            regimes[query_ids],
            sbi_map_alpha,
        )

        old_score = v432.row_cosine(truth[query_ids], old_prediction[query_ids])
        single_score = v432.row_cosine(
            truth[query_ids], single_prediction[query_ids]
        )
        stacked_score = v432.row_cosine(
            truth[query_ids], stacked_prediction[query_ids]
        )
        sbi_score = v432.row_cosine(truth[query_ids], sbi_prediction[query_ids])
        sbi_map_score = v432.row_cosine(
            truth[query_ids], sbi_map_prediction[query_ids]
        )
        fold_rows.append(
            {
                "fold": outer,
                "count": int(len(query_ids)),
                "old_robust": float(np.mean(old_score)),
                "best_single": float(np.mean(single_score)),
                "deterministic_stack": float(np.mean(stacked_score)),
                "sbi": float(np.mean(sbi_score)),
                "sbi_map": float(np.mean(sbi_map_score)),
                "sbi_gain_vs_old": float(np.mean(sbi_score - old_score)),
                "sbi_gain_vs_stack": float(np.mean(sbi_score - stacked_score)),
                "sbi_map_gain_vs_old": float(np.mean(sbi_map_score - old_score)),
                "sbi_map_gain_vs_stack": float(
                    np.mean(sbi_map_score - stacked_score)
                ),
                "single_source": {
                    regime: v494.SOURCE_NAMES[int(np.argmax(single_weights[regime]))]
                    for regime in v494.REGIMES
                },
                "old_alpha": old_alpha,
                "single_alpha": single_alpha,
                "stacked_alpha": stacked_alpha,
                "sbi_alpha": sbi_alpha,
                "sbi_map_alpha": sbi_map_alpha,
            }
        )
        posterior_rows.append(
            {
                "fold": outer,
                "posterior": posterior_summary,
                "deterministic_weights": {
                    regime: stacked_weights[regime].tolist()
                    for regime in v494.REGIMES
                },
            }
        )
        print(
            f"v529 outer {outer + 1}/5 "
            f"gain_old={fold_rows[-1]['sbi_gain_vs_old']:+.6f} "
            f"gain_stack={fold_rows[-1]['sbi_gain_vs_stack']:+.6f} "
            f"map_stack={fold_rows[-1]['sbi_map_gain_vs_stack']:+.6f}",
            flush=True,
        )

    baseline_score = v432.row_cosine(truth, baseline_prediction)
    old_score = v432.row_cosine(truth, old_prediction)
    single_score = v432.row_cosine(truth, single_prediction)
    stacked_score = v432.row_cosine(truth, stacked_prediction)
    sbi_score = v432.row_cosine(truth, sbi_prediction)
    sbi_map_score = v432.row_cosine(truth, sbi_map_prediction)
    gain_old = sbi_score - old_score
    gain_stack = sbi_score - stacked_score
    map_gain_old = sbi_map_score - old_score
    map_gain_stack = sbi_map_score - stacked_score
    pass_gate = bool(
        max(float(np.mean(gain_stack)), float(np.mean(map_gain_stack))) >= 0.001
        and max(
            sum(float(row["sbi_gain_vs_stack"]) > 0.0 for row in fold_rows),
            sum(float(row["sbi_map_gain_vs_stack"]) > 0.0 for row in fold_rows),
        )
        >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A spatially robust likelihood-free posterior over Sionna configuration "
            "mixtures generalizes better than a single configuration or deterministic "
            "least-squares stacking."
        ),
        "sbi": {
            "parameter": "six-source simplex weight per propagation regime",
            "prior": f"Dirichlet({PRIOR_CONCENTRATION})",
            "prior_draws": PRIOR_DRAWS,
            "generalized_likelihood": (
                "exp((spatial-fold mean cosine - 0.25*fold std)/temperature)"
            ),
            "target_ess_fraction": TARGET_ESS_FRACTION,
            "random_seed": RANDOM_SEED,
        },
        "evidence_boundary": {
            "outer_split": "five mutually exclusive spatial folds",
            "posterior_fit": "cross-fitted source spectra from outer-fit folds only",
            "query_fold_used_for_posterior": False,
            "test_sionna_used": False,
            "online_feedback_used_for_parameters": False,
        },
        "sources": list(v494.SOURCE_NAMES),
        "scores": {
            "baseline": summarize(baseline_score),
            "old_robust": summarize(old_score),
            "best_single": summarize(single_score),
            "deterministic_stack": summarize(stacked_score),
            "sbi": summarize(sbi_score),
            "sbi_map": summarize(sbi_map_score),
            "sbi_gain_vs_old": summarize(gain_old),
            "sbi_gain_vs_stack": summarize(gain_stack),
            "sbi_map_gain_vs_old": summarize(map_gain_old),
            "sbi_map_gain_vs_stack": summarize(map_gain_stack),
            "fold_wins_vs_old": int(
                sum(float(row["sbi_gain_vs_old"]) > 0.0 for row in fold_rows)
            ),
            "fold_wins_vs_stack": int(
                sum(float(row["sbi_gain_vs_stack"]) > 0.0 for row in fold_rows)
            ),
            "map_fold_wins_vs_old": int(
                sum(float(row["sbi_map_gain_vs_old"]) > 0.0 for row in fold_rows)
            ),
            "map_fold_wins_vs_stack": int(
                sum(float(row["sbi_map_gain_vs_stack"]) > 0.0 for row in fold_rows)
            ),
        },
        "folds": fold_rows,
        "posteriors": posterior_rows,
        "decision_gate": {
            "required_gain_vs_deterministic_stack": 0.001,
            "required_fold_wins_vs_stack": 4,
            "pass": pass_gate,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        truth=truth,
        baseline=baseline_prediction,
        old_robust=old_prediction,
        best_single=single_prediction,
        deterministic_stack=stacked_prediction,
        sbi=sbi_prediction,
        sbi_map=sbi_map_prediction,
        baseline_score=baseline_score,
        old_score=old_score,
        single_score=single_score,
        stacked_score=stacked_score,
        sbi_score=sbi_score,
        sbi_map_score=sbi_map_score,
        sbi_gain_vs_old=gain_old,
        sbi_gain_vs_stack=gain_stack,
        sbi_map_gain_vs_old=map_gain_old,
        sbi_map_gain_vs_stack=map_gain_stack,
        source_names=np.asarray(v494.SOURCE_NAMES),
        prior_draws=draws,
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
