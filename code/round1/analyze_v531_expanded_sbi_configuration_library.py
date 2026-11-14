#!/usr/bin/env python3
"""Audit deterministic stacking and SBI after adding heavy-specialist spectra."""

from __future__ import annotations

import argparse
import json

import numpy as np

import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v529_sbi_configuration_posterior as v529
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 531
VERSION = "v531_expanded_sbi_configuration_library"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = CACHE_DIR / f"{VERSION}_oof.npz"
V530_DIR = PATHS.cache_dir(530)
SHALLOW_TRAIN = V530_DIR / "depth10_scatter015_synthetic_train_joint_sh.npy"
SHALLOW_DONE = V530_DIR / "depth10_scatter015_synthetic_train_done.npy"
SHALLOW_SEED_PATHS = (
    SHALLOW_TRAIN,
    V530_DIR / "depth10_scatter015_synthetic_seed206_train_joint_sh.npy",
    V530_DIR / "depth10_scatter015_synthetic_seed420_train_joint_sh.npy",
)
SHALLOW_SEED_DONE = (
    SHALLOW_DONE,
    V530_DIR / "depth10_scatter015_synthetic_seed206_train_done.npy",
    V530_DIR / "depth10_scatter015_synthetic_seed420_train_done.npy",
)
DEEP_TRAIN = V530_DIR / "depth20_scatter015_synthetic_seed206_train_joint_sh.npy"
DEEP_DONE = V530_DIR / "depth20_scatter015_synthetic_seed206_train_done.npy"
DEEP_SEED_PATHS = (
    DEEP_TRAIN,
    V530_DIR / "depth20_scatter015_synthetic_seed42_train_joint_sh.npy",
)
DEEP_SEED_DONE = (
    DEEP_DONE,
    V530_DIR / "depth20_scatter015_synthetic_seed42_train_done.npy",
)
NONMETAL_TRAIN = (
    V530_DIR / "depth18_scatter015_nonmetal_synthetic_seed206_train_joint_sh.npy"
)
NONMETAL_DONE = (
    V530_DIR / "depth18_scatter015_nonmetal_synthetic_seed206_train_done.npy"
)
EDGE_TRAIN = V530_DIR / "depth16_nonsynthetic_edge_train_heavy_joint_sh.npy"
EDGE_DONE = V530_DIR / "depth16_nonsynthetic_edge_train_heavy_done.npy"
EDGE_IDS = V530_DIR / "depth16_nonsynthetic_edge_train_heavy_indices.npy"

PRIOR_DRAWS = 8192
PRIOR_CONCENTRATION = 0.5
RANDOM_SEED = 20260725
EDGE_SOURCE_NAME = "depth16_nonsynthetic_edge_heavy"
HYBRID_METHOD = "expanded_sbi_map_chlm_stack_heavy"


def clean_source(source: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = v494.normalize_rows(np.asarray(source, dtype=np.float32))
    valid = value.sum(axis=1) > 0.5
    value = value.copy()
    value[~valid] = fallback[~valid]
    return v494.normalize_rows(value)


def load_seed_marginalized_source(
    paths: tuple,
    done_paths: tuple,
    fallback: np.ndarray,
) -> tuple[np.ndarray, int]:
    complete = []
    for path, done_path in zip(paths, done_paths, strict=True):
        if not path.is_file() or not done_path.is_file():
            continue
        done = np.load(done_path)
        value = np.load(path, mmap_mode="r")
        if value.shape[0] != len(fallback) or not np.all(done):
            raise ValueError(f"incomplete seed cache: {path}")
        complete.append(clean_source(value.reshape(len(fallback), -1), fallback))
    if not complete:
        raise FileNotFoundError(paths[0])
    return v494.normalize_rows(np.mean(complete, axis=0)), len(complete)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-nonmetal", action="store_true")
    return parser.parse_args()


def load_expanded_sources(
    include_nonmetal: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[str]]:
    truth, old_sources, quality = v494.load_sources()
    sources = list(old_sources)
    names = list(v494.SOURCE_NAMES)
    if not SHALLOW_TRAIN.is_file() or not SHALLOW_DONE.is_file():
        raise FileNotFoundError("complete v530 shallow-scattering spectra are required")
    shallow, shallow_seed_count = load_seed_marginalized_source(
        SHALLOW_SEED_PATHS, SHALLOW_SEED_DONE, sources[0]
    )
    sources.append(shallow)
    names.append(
        "depth10_scatter015_synthetic"
        if shallow_seed_count == 1
        else f"depth10_scatter015_synthetic_seed_marginal_{shallow_seed_count}"
    )

    if DEEP_TRAIN.is_file() and DEEP_DONE.is_file():
        deep, deep_seed_count = load_seed_marginalized_source(
            DEEP_SEED_PATHS, DEEP_SEED_DONE, sources[0]
        )
        sources.append(deep)
        names.append(
            "depth20_scatter015_synthetic_seed206"
            if deep_seed_count == 1
            else f"depth20_scatter015_synthetic_seed_marginal_{deep_seed_count}"
        )

    if include_nonmetal and NONMETAL_TRAIN.is_file() and NONMETAL_DONE.is_file():
        nonmetal_done = np.load(NONMETAL_DONE)
        nonmetal = np.load(NONMETAL_TRAIN, mmap_mode="r")
        if nonmetal.shape[0] != len(truth) or not np.all(nonmetal_done):
            raise ValueError("v530 depth18 nonmetal-scattering cache is incomplete")
        sources.append(clean_source(nonmetal.reshape(len(truth), -1), sources[0]))
        names.append("depth18_scatter015_nonmetal_synthetic_seed206")

    edge_paths = (EDGE_TRAIN, EDGE_DONE, EDGE_IDS)
    if all(path.is_file() for path in edge_paths):
        edge_done = np.load(EDGE_DONE)
        edge_ids = np.load(EDGE_IDS).astype(np.int64)
        edge_rows = np.load(EDGE_TRAIN, mmap_mode="r")
        if len(edge_rows) != len(edge_ids):
            raise ValueError("v530 non-synthetic edge cache index mismatch")
        if np.all(edge_done):
            edge_source = sources[0].copy()
            edge_source[edge_ids] = clean_source(
                edge_rows.reshape(len(edge_ids), -1), edge_source[edge_ids]
            )
            sources.append(v494.normalize_rows(edge_source))
            names.append("depth16_nonsynthetic_edge_heavy")
    return truth, sources, quality, names


def fit_targets(
    truth: np.ndarray,
    candidates: np.ndarray,
    fit_ids: np.ndarray,
    query_candidates: np.ndarray,
    regimes: np.ndarray,
    query_regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    weights: dict[str, np.ndarray] = {}
    for regime in v494.REGIMES:
        local_ids = fit_ids[regimes[fit_ids] == regime]
        weights[regime] = v494.fit_convex_weights(truth, candidates, local_ids)
    inner_target = np.zeros_like(truth)
    inner_target[fit_ids] = v494.weighted_target(
        candidates, fit_ids, regimes, weights
    )
    query_target = v494.weighted_target(
        query_candidates,
        np.arange(query_candidates.shape[1]),
        query_regimes,
        weights,
    )
    return inner_target, query_target, weights


def by_regime(values: np.ndarray, regimes: np.ndarray) -> dict[str, float]:
    return {
        regime: float(np.mean(values[regimes == regime]))
        for regime in v494.REGIMES
    }


def prior_draws_by_regime(source_names: list[str]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    edge_index = (
        source_names.index(EDGE_SOURCE_NAME)
        if EDGE_SOURCE_NAME in source_names
        else None
    )
    output: dict[str, np.ndarray] = {}
    for regime in v494.REGIMES:
        active = np.arange(len(source_names), dtype=np.int64)
        if edge_index is not None and regime != "heavy":
            active = active[active != edge_index]
        local = rng.dirichlet(
            np.full(len(active), PRIOR_CONCENTRATION, dtype=np.float64),
            size=PRIOR_DRAWS,
        ).astype(np.float32)
        draws = np.zeros((PRIOR_DRAWS, len(source_names)), dtype=np.float32)
        draws[:, active] = local
        output[regime] = draws
    return output


def main() -> None:
    args = parse_args()
    for path in (v494.OFFICIAL, v494.POSITIONS, v434.OOF_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    truth, sources, quality, source_names = load_expanded_sources(
        include_nonmetal=args.include_nonmetal
    )
    old_count = len(v494.SOURCE_NAMES)
    positions = np.load(v494.POSITIONS).astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    fold_id = np.load(v434.OOF_CACHE)["fold_id"].astype(np.int16)
    regime_draws = prior_draws_by_regime(source_names)

    predictions = {
        name: np.zeros_like(truth)
        for name in (
            "baseline",
            "old_robust",
            "old_stack",
            "expanded_stack",
            "expanded_sbi",
            "expanded_sbi_map",
            HYBRID_METHOD,
        )
    }
    source_score = np.zeros((len(sources), len(truth)), dtype=np.float32)
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
            fit_ids, query_ids, positions, blocked, regimes, truth, k=48
        )
        query_candidates = v494.calibrated_sources(
            truth, sources, fit_ids, query_ids, regimes
        )
        old_inner_target, old_query_target, old_weights = fit_targets(
            truth,
            inner_candidates[:old_count],
            fit_ids,
            query_candidates[:old_count],
            regimes,
            regimes[query_ids],
        )
        expanded_inner_target, expanded_query_target, expanded_weights = fit_targets(
            truth,
            inner_candidates,
            fit_ids,
            query_candidates,
            regimes,
            regimes[query_ids],
        )

        fit_candidates = inner_candidates[:, fit_ids]
        target_dot, gram, truth_norm = v529.point_terms(
            truth[fit_ids], fit_candidates
        )
        sampled_cosine = {
            regime: v529.sampled_mixture_cosines(
                target_dot, gram, truth_norm, regime_draws[regime]
            )
            for regime in v494.REGIMES
        }
        fit_fold_id = fold_id[fit_ids]
        fit_regimes = regimes[fit_ids]
        inner_sbi_target = np.zeros_like(truth)
        inner_sbi_map_target = np.zeros_like(truth)
        final_weights: dict[str, np.ndarray] = {}
        final_map_weights: dict[str, np.ndarray] = {}
        posterior_summary: dict[str, dict[str, object]] = {}
        for regime in v494.REGIMES:
            rows = np.flatnonzero(fit_regimes == regime)
            posterior = v529.infer_posterior(
                sampled_cosine[regime],
                rows,
                fit_fold_id,
                regime_draws[regime],
            )
            final_weights[regime] = np.asarray(posterior["weight"], dtype=np.float32)
            final_map_weights[regime] = np.asarray(
                posterior["map_weight"], dtype=np.float32
            )
            posterior_summary[regime] = {
                "weight": final_weights[regime].tolist(),
                "map_weight": final_map_weights[regime].tolist(),
                "temperature": float(posterior["temperature"]),
                "ess": float(posterior["ess"]),
            }

        for inner in range(5):
            if inner == outer:
                continue
            validation_ids = np.flatnonzero(fold_id == inner)
            train_rows = np.flatnonzero(fit_fold_id != inner)
            local_weights: dict[str, np.ndarray] = {}
            local_map_weights: dict[str, np.ndarray] = {}
            for regime in v494.REGIMES:
                regime_rows = train_rows[fit_regimes[train_rows] == regime]
                posterior = v529.infer_posterior(
                    sampled_cosine[regime],
                    regime_rows,
                    fit_fold_id,
                    regime_draws[regime],
                )
                local_weights[regime] = np.asarray(
                    posterior["weight"], dtype=np.float32
                )
                local_map_weights[regime] = np.asarray(
                    posterior["map_weight"], dtype=np.float32
                )
            inner_sbi_target[validation_ids] = v529.weighted_rows(
                inner_candidates[:, validation_ids],
                local_weights,
                regimes[validation_ids],
            )
            inner_sbi_map_target[validation_ids] = v529.weighted_rows(
                inner_candidates[:, validation_ids],
                local_map_weights,
                regimes[validation_ids],
            )

        sbi_query_target = v529.weighted_rows(
            query_candidates, final_weights, regimes[query_ids]
        )
        sbi_map_query_target = v529.weighted_rows(
            query_candidates, final_map_weights, regimes[query_ids]
        )
        old_alpha = v494.select_alpha(
            truth,
            inner_baseline,
            inner_candidates[0],
            quality,
            fit_ids,
            regimes,
        )
        old_stack_alpha = v494.select_alpha(
            truth, inner_baseline, old_inner_target, quality, fit_ids, regimes
        )
        expanded_alpha = v494.select_alpha(
            truth,
            inner_baseline,
            expanded_inner_target,
            quality,
            fit_ids,
            regimes,
        )
        sbi_alpha = v494.select_alpha(
            truth, inner_baseline, inner_sbi_target, quality, fit_ids, regimes
        )
        sbi_map_alpha = v494.select_alpha(
            truth,
            inner_baseline,
            inner_sbi_map_target,
            quality,
            fit_ids,
            regimes,
        )

        local_regimes = regimes[query_ids]
        local_ids = np.arange(len(query_ids))
        predictions["baseline"][query_ids] = v494.normalize_rows(query_baseline)
        predictions["old_robust"][query_ids] = v494.blend_target(
            query_baseline,
            query_candidates[0],
            quality,
            local_ids,
            local_regimes,
            old_alpha,
        )
        predictions["old_stack"][query_ids] = v494.blend_target(
            query_baseline,
            old_query_target,
            quality,
            local_ids,
            local_regimes,
            old_stack_alpha,
        )
        predictions["expanded_stack"][query_ids] = v494.blend_target(
            query_baseline,
            expanded_query_target,
            quality,
            local_ids,
            local_regimes,
            expanded_alpha,
        )
        predictions["expanded_sbi"][query_ids] = v494.blend_target(
            query_baseline,
            sbi_query_target,
            quality,
            local_ids,
            local_regimes,
            sbi_alpha,
        )
        predictions["expanded_sbi_map"][query_ids] = v494.blend_target(
            query_baseline,
            sbi_map_query_target,
            quality,
            local_ids,
            local_regimes,
            sbi_map_alpha,
        )
        predictions[HYBRID_METHOD][query_ids] = predictions[
            "expanded_sbi_map"
        ][query_ids]
        heavy_query = local_regimes == "heavy"
        predictions[HYBRID_METHOD][query_ids[heavy_query]] = predictions[
            "expanded_stack"
        ][query_ids[heavy_query]]
        for source_index in range(len(sources)):
            source_alpha = v494.select_alpha(
                truth,
                inner_baseline,
                inner_candidates[source_index],
                quality,
                fit_ids,
                regimes,
            )
            source_prediction = v494.blend_target(
                query_baseline,
                query_candidates[source_index],
                quality,
                local_ids,
                local_regimes,
                source_alpha,
            )
            source_score[source_index, query_ids] = v432.row_cosine(
                truth[query_ids], source_prediction
            )

        scores = {
            name: v432.row_cosine(truth[query_ids], value[query_ids])
            for name, value in predictions.items()
        }
        fold_rows.append(
            {
                "fold": outer,
                "count": int(len(query_ids)),
                **{name: float(np.mean(value)) for name, value in scores.items()},
                "stack_gain_vs_old_stack": float(
                    np.mean(scores["expanded_stack"] - scores["old_stack"])
                ),
                "sbi_gain_vs_old_stack": float(
                    np.mean(scores["expanded_sbi"] - scores["old_stack"])
                ),
                "sbi_map_gain_vs_old_stack": float(
                    np.mean(scores["expanded_sbi_map"] - scores["old_stack"])
                ),
                "hybrid_gain_vs_old_stack": float(
                    np.mean(scores[HYBRID_METHOD] - scores["old_stack"])
                ),
                "old_weights": {
                    regime: old_weights[regime].tolist()
                    for regime in v494.REGIMES
                },
                "expanded_weights": {
                    regime: expanded_weights[regime].tolist()
                    for regime in v494.REGIMES
                },
                "alphas": {
                    "old_stack": old_stack_alpha,
                    "expanded_stack": expanded_alpha,
                    "expanded_sbi": sbi_alpha,
                    "expanded_sbi_map": sbi_map_alpha,
                },
            }
        )
        posterior_rows.append({"fold": outer, "posterior": posterior_summary})
        print(
            f"v531 outer {outer + 1}/5 "
            f"stack={fold_rows[-1]['stack_gain_vs_old_stack']:+.6f} "
            f"sbi={fold_rows[-1]['sbi_gain_vs_old_stack']:+.6f} "
            f"map={fold_rows[-1]['sbi_map_gain_vs_old_stack']:+.6f}",
            flush=True,
        )

    scores = {
        name: v432.row_cosine(truth, value) for name, value in predictions.items()
    }
    gains = {
        name: value - scores["old_stack"]
        for name, value in scores.items()
        if name
        in ("expanded_stack", "expanded_sbi", "expanded_sbi_map", HYBRID_METHOD)
    }
    source_oracle = np.max(source_score, axis=0)
    source_oracle_index = np.argmax(source_score, axis=0).astype(np.int16)
    best_method = max(gains, key=lambda name: float(np.mean(gains[name])))
    best_gain = float(np.mean(gains[best_method]))
    fold_gain_key = {
        "expanded_stack": "stack_gain_vs_old_stack",
        "expanded_sbi": "sbi_gain_vs_old_stack",
        "expanded_sbi_map": "sbi_map_gain_vs_old_stack",
        HYBRID_METHOD: "hybrid_gain_vs_old_stack",
    }[best_method]
    fold_wins = sum(float(row[fold_gain_key]) > 0.0 for row in fold_rows)
    pass_gate = bool(best_gain >= 0.003 and fold_wins >= 4)
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Adding physically distinct heavy-specialist Sionna configurations supplies "
            "missing path support that cannot be recovered by retuning weights over the "
            "old six-source library."
        ),
        "evidence_boundary": {
            "outer_split": "five mutually exclusive spatial folds",
            "inner_selection": "outer-fit folds only",
            "test_sionna_used": False,
            "online_feedback_used_for_parameters": False,
            "fixed_regime_decision": (
                "SBI MAP for clear/light/medium; deterministic expanded stack "
                "for heavy, where posterior mode uncertainty makes averaging safer"
            ),
        },
        "sources": source_names,
        "include_nonmetal": bool(args.include_nonmetal),
        "scores": {
            **{name: v529.summarize(value) for name, value in scores.items()},
            **{f"{name}_gain_vs_old_stack": v529.summarize(value) for name, value in gains.items()},
            "source_oracle": v529.summarize(source_oracle),
            "source_oracle_gain_vs_old_stack": float(
                np.mean(source_oracle - scores["old_stack"])
            ),
        },
        "by_regime": {
            name: by_regime(value, regimes) for name, value in scores.items()
        },
        "gain_by_regime": {
            name: by_regime(value, regimes) for name, value in gains.items()
        },
        "source_oracle_counts": {
            source_names[index]: int(np.sum(source_oracle_index == index))
            for index in range(len(source_names))
        },
        "folds": fold_rows,
        "posteriors": posterior_rows,
        "decision_gate": {
            "best_method": best_method,
            "best_mean_gain_vs_old_stack": best_gain,
            "fold_wins": int(fold_wins),
            "required_gain": 0.003,
            "required_fold_wins": 4,
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
        source_names=np.asarray(source_names),
        source_score=source_score,
        source_oracle=source_oracle,
        source_oracle_index=source_oracle_index,
        **predictions,
        **{f"{name}_score": value for name, value in scores.items()},
        **{f"{name}_gain_vs_old_stack": value for name, value in gains.items()},
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
