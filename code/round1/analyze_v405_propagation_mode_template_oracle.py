#!/usr/bin/env python3
"""Measure the OOF ceiling of whole-spectrum propagation-mode templates."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 405
VERSION = "v405_propagation_mode_template_oracle"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v405_propagation_mode_template_oracle_oof.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

ANCHOR_K = 16
TEMPLATE_K = 32
DISTANCE_FLOOR = 0.25
ALPHAS = np.asarray([0.25, 0.50, 0.75, 1.00, 1.25], dtype=np.float32)
MIN_ORACLE_DELTA = 0.02
MIN_FOLD_WINS = 5


def neighbors_excluding_self(
    query_ids: np.ndarray, bank_ids: np.ndarray, xy: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(xy[bank_ids])
    raw_count = min(count + 1, len(bank_ids))
    distance, local = tree.query(xy[query_ids], k=raw_count)
    if raw_count == 1:
        distance = distance[:, None]
        local = local[:, None]
    mapped = bank_ids[local]
    output_ids = np.empty((len(query_ids), min(count, len(bank_ids) - 1)), dtype=np.int64)
    output_distance = np.empty_like(output_ids, dtype=np.float64)
    for row, point_id in enumerate(query_ids):
        keep = mapped[row] != point_id
        selected_id = mapped[row, keep][: output_ids.shape[1]]
        selected_distance = distance[row, keep][: output_ids.shape[1]]
        if len(selected_id) != output_ids.shape[1]:
            raise AssertionError("self-excluding neighbor bank is too small")
        output_ids[row] = selected_id
        output_distance[row] = selected_distance
    weight = 1.0 / np.maximum(output_distance, DISTANCE_FLOOR) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return output_ids, weight.astype(np.float32)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def score_candidates(candidate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    sh = cosine_rows(candidate, np.broadcast_to(truth, candidate.shape))
    pdp = cosine_rows(
        candidate.sum(axis=2),
        np.broadcast_to(truth.sum(axis=2), candidate.sum(axis=2).shape),
    )
    return 0.5 * (sh + pdp)


def oracle_select(
    anchor: np.ndarray,
    truth: np.ndarray,
    donor_target: np.ndarray,
    donor_anchor: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    methods = ("full_template", "linear_residual", "sqrt_residual")
    best_score = {name: np.full(len(truth), -np.inf, dtype=np.float32) for name in methods}
    best_prediction = {name: np.empty_like(truth) for name in methods}
    best_rank = {name: np.full(len(truth), -1, dtype=np.int16) for name in methods}
    best_alpha = {name: np.zeros(len(truth), dtype=np.float32) for name in methods}
    anchor_sqrt = np.sqrt(np.maximum(anchor, 0.0))
    donor_sqrt_residual = np.sqrt(np.maximum(donor_target, 0.0)) - np.sqrt(
        np.maximum(donor_anchor, 0.0)
    )
    donor_linear_residual = donor_target - donor_anchor

    for rank in range(donor_target.shape[1]):
        donor = donor_target[:, rank]
        linear = donor_linear_residual[:, rank]
        sqrt_residual = donor_sqrt_residual[:, rank]
        for alpha in ALPHAS:
            variants = {
                "full_template": v402.normalize(
                    (1.0 - min(float(alpha), 1.0)) * anchor
                    + min(float(alpha), 1.0) * donor
                ),
                "linear_residual": v402.normalize(
                    np.maximum(anchor + float(alpha) * linear, 0.0)
                ),
                "sqrt_residual": v402.normalize(
                    np.square(np.maximum(anchor_sqrt + float(alpha) * sqrt_residual, 0.0))
                ),
            }
            for name, prediction in variants.items():
                score = score_candidates(prediction, truth)
                improved = score > best_score[name]
                best_score[name][improved] = score[improved]
                best_prediction[name][improved] = prediction[improved]
                best_rank[name][improved] = rank
                best_alpha[name][improved] = alpha
    diagnostics = {
        name: {
            "selected_rank": summarize(best_rank[name]),
            "selected_alpha": summarize(best_alpha[name]),
            "top8_rank_fraction": float(np.mean(best_rank[name] < 8)),
        }
        for name in methods
    }
    return best_prediction, diagnostics


def fold_experiment(
    fold: int,
    query_ids: np.ndarray,
    valid_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    xy = positions[:, :2]
    anchor_ids, anchor_weight = v402.neighbors(query_ids, fit_ids, xy)
    anchor = v402.knn_average(target, anchor_ids, anchor_weight)

    fit_anchor_ids, fit_anchor_weight = neighbors_excluding_self(
        fit_ids, fit_ids, xy, ANCHOR_K
    )
    fit_anchor = v402.knn_average(target, fit_anchor_ids, fit_anchor_weight)
    template_distance, template_local = cKDTree(xy[fit_ids]).query(
        xy[query_ids], k=min(TEMPLATE_K, len(fit_ids))
    )
    if np.ndim(template_local) == 1:
        template_local = template_local[:, None]
        template_distance = template_distance[:, None]
    donor_ids = fit_ids[template_local]
    lookup = np.full(len(target), -1, dtype=np.int64)
    lookup[fit_ids] = np.arange(len(fit_ids))
    predictions, diagnostic = oracle_select(
        anchor,
        target[query_ids],
        target[donor_ids],
        fit_anchor[lookup[donor_ids]],
    )
    baseline_values = v402.point_metrics(anchor, target[query_ids])
    regimes = regime_of(blocked[query_ids]).astype(str)
    methods: dict[str, object] = {}
    for name, prediction in predictions.items():
        values = v402.point_metrics(prediction, target[query_ids])
        delta = values["weighted"] - baseline_values["weighted"]
        methods[name] = {
            "metrics": v402.metric_summary(prediction, target[query_ids]),
            "delta": {
                key: float(np.mean(values[key] - baseline_values[key]))
                for key in values
            },
            "point_gain": summarize(delta),
            "point_win_fraction": float(np.mean(delta > 0.0)),
            "by_regime": {
                regime: float(np.mean(delta[regimes == regime]))
                for regime in ("clear", "light", "medium", "heavy")
                if np.any(regimes == regime)
            },
            "selection": diagnostic[name],
        }
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "baseline": v402.metric_summary(anchor, target[query_ids]),
        "methods": methods,
        "template_distance": summarize(np.asarray(template_distance)),
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "query_truth_used_for_oracle_selection_only": True,
        },
    }
    arrays = {
        "ids": query_ids,
        "baseline": anchor,
        "truth": target[query_ids],
        **predictions,
    }
    return row, arrays


def main() -> None:
    for path in (TARGET_CACHE, GEOMETRY_CACHE, v297.COMPONENT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGET_CACHE)
    target = v402.normalize(targets["sh"].astype(np.float32))
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    blocked = np.load(GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, positions[:, :2])
    rows: list[dict[str, object]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for fold, query_ids in enumerate(folds):
        row, part = fold_experiment(
            fold, query_ids, valid_ids, positions, blocked, target
        )
        rows.append(row)
        arrays.append(part)
        print(
            f"v405 fold={fold} full={row['methods']['full_template']['delta']['weighted']:+.5f} "
            f"linear={row['methods']['linear_residual']['delta']['weighted']:+.5f} "
            f"sqrt={row['methods']['sqrt_residual']['delta']['weighted']:+.5f}",
            flush=True,
        )

    aggregate: dict[str, object] = {}
    for name in ("full_template", "linear_residual", "sqrt_residual"):
        delta = np.asarray([row["methods"][name]["delta"]["weighted"] for row in rows])
        aggregate[name] = {
            "weighted_delta": summarize(delta),
            "fold_wins": int(np.sum(delta >= MIN_ORACLE_DELTA)),
            "passed": bool(
                np.mean(delta) >= MIN_ORACLE_DELTA
                and np.sum(delta >= MIN_ORACLE_DELTA) >= MIN_FOLD_WINS
            ),
        }
    best_name = max(
        aggregate,
        key=lambda name: aggregate[name]["weighted_delta"]["mean"],
    )
    passed = bool(aggregate[best_name]["passed"])
    ids = np.concatenate([part["ids"] for part in arrays])
    if len(ids) != len(np.unique(ids)) or len(ids) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover")
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [
                np.full(len(part["ids"]), fold, dtype=np.int16)
                for fold, part in enumerate(arrays)
            ]
        ),
        **{
            key: np.concatenate([part[key] for part in arrays])
            for key in (
                "baseline",
                "truth",
                "full_template",
                "linear_residual",
                "sqrt_residual",
            )
        },
    )
    report = {
        "method": VERSION,
        "status": "oracle_passed_needs_router" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Whole-spectrum propagation-mode templates preserve mutually exclusive path "
            "states that are destroyed by per-path candidate averaging."
        ),
        "compliance": {
            "uses": ["official train H spectra", "official train coordinates"],
            "does_not_use": [
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic target or online labels",
            ],
            "oracle_truth": "outer H is used only to measure candidate-template ceiling",
        },
        "parameters": {
            "anchor_k": ANCHOR_K,
            "template_k": TEMPLATE_K,
            "alphas": ALPHAS.tolist(),
            "folds": "five mutually exclusive isolation-stratified folds",
        },
        "fold_rows": rows,
        "aggregate": aggregate,
        "selection": {
            "best_method": best_name,
            "minimum_mean_delta": MIN_ORACLE_DELTA,
            "minimum_fold_wins": f"{MIN_FOLD_WINS}/5 at delta >= {MIN_ORACLE_DELTA}",
            "passed": passed,
            "next_action": (
                "train a strict official-only neural mode router"
                if passed
                else "terminate whole-spectrum template routing"
            ),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
