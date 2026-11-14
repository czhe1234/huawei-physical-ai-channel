#!/usr/bin/env python3
"""Test whether the v405 donor identity survives independent array views."""

from __future__ import annotations

import json

import numpy as np

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v406_neural_propagation_mode_router as v406
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 410
VERSION = "v410_cross_view_template_identity_audit"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v410_cross_view_template_identity_audit_oof.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
MIN_CROSS_VIEW_DELTA = 0.02
MIN_FOLD_WINS = 5


def make_router_data(
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    anchor: np.ndarray,
    scores: np.ndarray,
) -> v406.RouterData:
    empty = np.empty((0,), dtype=np.float32)
    return v406.RouterData(
        query_ids=query_ids,
        donor_ids=donor_ids,
        anchor=anchor,
        scores=scores,
        q_geometry=empty,
        donor_geometry=empty,
        relation_geometry=empty,
        q_full=empty,
        donor_full=empty,
        relation_full=empty,
    )


def query_bank(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    xy: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids, distance = v406.find_neighbors(
        query_ids, fit_ids, xy, v406.DONOR_K
    )
    anchor, _ = v406.weighted_anchor(target, donor_ids, distance)
    return anchor, donor_ids


def select_choices(
    anchor: np.ndarray,
    donor_ids: np.ndarray,
    target: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = v406.option_scores(anchor, target[donor_ids], truth)
    return scores.argmax(axis=1), scores


def render_choices(
    query_ids: np.ndarray,
    anchor: np.ndarray,
    donor_ids: np.ndarray,
    choices: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    data = make_router_data(
        query_ids,
        donor_ids,
        anchor,
        np.empty((len(query_ids), 1 + v406.DONOR_K * len(v406.ALPHAS))),
    )
    return v406.render_choices(data, choices, target)


def weighted_delta(
    prediction: np.ndarray, anchor: np.ndarray, truth: np.ndarray
) -> tuple[float, np.ndarray]:
    base = v402.point_metrics(anchor, truth)["weighted"]
    value = v402.point_metrics(prediction, truth)["weighted"]
    delta = value - base
    return float(np.mean(delta)), delta


def evaluate_direction(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    donor_ids: np.ndarray,
    xy: np.ndarray,
    selection_target: np.ndarray,
    evaluation_target: np.ndarray,
    full_target: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    selection_anchor, selection_donor = query_bank(
        query_ids, fit_ids, xy, selection_target
    )
    if not np.array_equal(selection_donor, donor_ids):
        raise AssertionError("coordinate donor bank changed across views")
    evaluation_anchor, evaluation_donor = query_bank(
        query_ids, fit_ids, xy, evaluation_target
    )
    full_anchor, full_donor = query_bank(query_ids, fit_ids, xy, full_target)
    if not (
        np.array_equal(evaluation_donor, donor_ids)
        and np.array_equal(full_donor, donor_ids)
    ):
        raise AssertionError("coordinate donor bank changed across views")
    choices, selection_scores = select_choices(
        selection_anchor,
        donor_ids,
        selection_target,
        selection_target[query_ids],
    )
    selection_prediction = render_choices(
        query_ids, selection_anchor, donor_ids, choices, selection_target
    )
    evaluation_prediction = render_choices(
        query_ids, evaluation_anchor, donor_ids, choices, evaluation_target
    )
    full_prediction = render_choices(
        query_ids, full_anchor, donor_ids, choices, full_target
    )
    same_delta, same_point = weighted_delta(
        selection_prediction, selection_anchor, selection_target[query_ids]
    )
    cross_delta, cross_point = weighted_delta(
        evaluation_prediction, evaluation_anchor, evaluation_target[query_ids]
    )
    full_delta, full_point = weighted_delta(
        full_prediction, full_anchor, full_target[query_ids]
    )
    option = np.maximum(choices - 1, 0)
    active = choices > 0
    row = {
        "same_view_delta": same_delta,
        "cross_view_delta": cross_delta,
        "full_target_delta": full_delta,
        "same_view_win_fraction": float(np.mean(same_point > 0.0)),
        "cross_view_win_fraction": float(np.mean(cross_point > 0.0)),
        "full_target_win_fraction": float(np.mean(full_point > 0.0)),
        "no_action_fraction": float(np.mean(~active)),
        "selected_rank": summarize(
            (option[active] // len(v406.ALPHAS)).astype(np.float32)
        ),
        "selected_alpha": summarize(
            v406.ALPHAS[(option[active] % len(v406.ALPHAS)).astype(np.int64)]
        ),
        "selection_score": summarize(
            selection_scores[np.arange(len(choices)), choices]
        ),
    }
    arrays = {
        "same": same_point,
        "cross": cross_point,
        "full": full_point,
    }
    return row, arrays


def fold_experiment(
    fold: int,
    query_ids: np.ndarray,
    valid_ids: np.ndarray,
    xy: np.ndarray,
    views: dict[str, tuple[np.ndarray, np.ndarray]],
    full_target: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    donor_ids, _ = v406.find_neighbors(
        query_ids, fit_ids, xy, v406.DONOR_K
    )
    pair_rows: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {"ids": query_ids}
    for name, (left, right) in views.items():
        left_row, left_arrays = evaluate_direction(
            query_ids,
            fit_ids,
            donor_ids,
            xy,
            left,
            right,
            full_target,
        )
        right_row, right_arrays = evaluate_direction(
            query_ids,
            fit_ids,
            donor_ids,
            xy,
            right,
            left,
            full_target,
        )
        pair_rows[name] = {
            "left_to_right": left_row,
            "right_to_left": right_row,
            "mean_same_view_delta": 0.5
            * (left_row["same_view_delta"] + right_row["same_view_delta"]),
            "mean_cross_view_delta": 0.5
            * (left_row["cross_view_delta"] + right_row["cross_view_delta"]),
            "mean_full_target_delta": 0.5
            * (left_row["full_target_delta"] + right_row["full_target_delta"]),
        }
        for direction, part in (("lr", left_arrays), ("rl", right_arrays)):
            for metric, value in part.items():
                arrays[f"{name}_{direction}_{metric}"] = value
    full_anchor, full_donor = query_bank(query_ids, fit_ids, xy, full_target)
    full_choice, _ = select_choices(
        full_anchor,
        full_donor,
        full_target,
        full_target[query_ids],
    )
    full_prediction = render_choices(
        query_ids, full_anchor, full_donor, full_choice, full_target
    )
    full_delta, full_point = weighted_delta(
        full_prediction, full_anchor, full_target[query_ids]
    )
    arrays["full_oracle"] = full_point
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "pairs": pair_rows,
        "full_oracle_delta": full_delta,
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "all_query_views_used_for_oracle_selection_or_scoring_only": True,
            "deployable_prediction": False,
        },
    }
    return row, arrays


def main() -> None:
    for path in (TARGET_CACHE, v297.COMPONENT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGET_CACHE)
    full_target = v406.normalize(targets["sh"].astype(np.float32))
    views = {
        "polar": (
            v406.normalize(targets["psh"][:, 0].astype(np.float32)),
            v406.normalize(targets["psh"][:, 1].astype(np.float32)),
        ),
        "ue_even_odd": (
            v406.normalize(targets["shue"][:, :, :, [0, 2]].sum(axis=3)),
            v406.normalize(targets["shue"][:, :, :, [1, 3]].sum(axis=3)),
        ),
        "vertical_even_odd": (
            v406.normalize(targets["shv"][:, :, :, 0::2].sum(axis=3)),
            v406.normalize(targets["shv"][:, :, :, 1::2].sum(axis=3)),
        ),
    }
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    xy = positions[:, :2]
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, xy)
    rows: list[dict[str, object]] = []
    parts: list[dict[str, np.ndarray]] = []
    for fold, query_ids in enumerate(folds):
        row, arrays = fold_experiment(
            fold, query_ids, valid_ids, xy, views, full_target
        )
        rows.append(row)
        parts.append(arrays)
        print(
            f"v410 fold={fold} full={row['full_oracle_delta']:+.5f} "
            + " ".join(
                f"{name}={row['pairs'][name]['mean_cross_view_delta']:+.5f}"
                for name in views
            ),
            flush=True,
        )
    aggregate: dict[str, object] = {}
    for name in views:
        same = np.asarray(
            [row["pairs"][name]["mean_same_view_delta"] for row in rows]
        )
        cross = np.asarray(
            [row["pairs"][name]["mean_cross_view_delta"] for row in rows]
        )
        full = np.asarray(
            [row["pairs"][name]["mean_full_target_delta"] for row in rows]
        )
        aggregate[name] = {
            "same_view_delta": summarize(same),
            "cross_view_delta": summarize(cross),
            "full_target_delta": summarize(full),
            "cross_view_fold_wins": int(np.sum(cross >= MIN_CROSS_VIEW_DELTA)),
            "retained_fraction": float(np.mean(cross) / max(np.mean(same), 1e-20)),
        }
    best_name = max(
        aggregate,
        key=lambda name: aggregate[name]["cross_view_delta"]["mean"],
    )
    passed = bool(
        aggregate[best_name]["cross_view_delta"]["mean"] >= MIN_CROSS_VIEW_DELTA
        and aggregate[best_name]["cross_view_fold_wins"] >= MIN_FOLD_WINS
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != len(np.unique(ids)) or len(ids) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover")
    array_names = sorted(set(parts[0]) - {"ids"})
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [
                np.full(len(part["ids"]), fold, dtype=np.int16)
                for fold, part in enumerate(parts)
            ]
        ),
        **{
            name: np.concatenate([part[name] for part in parts])
            for name in array_names
        },
    )
    report = {
        "method": VERSION,
        "status": "cross_view_identity_passed" if passed else "oracle_not_stable",
        "primary_hypothesis": (
            "A physical propagation-mode donor should remain useful when selected in one "
            "array view and evaluated in a disjoint view; noise-matched donors should not."
        ),
        "compliance": {
            "uses": [
                "official train H-derived polarization/UE/vertical spectral views",
                "official train coordinates",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic targets or online labels",
            ],
            "oracle_only": "all query-view selection is diagnostic and never deployable",
        },
        "parameters": {
            "anchor_k": v406.ANCHOR_K,
            "donor_k": v406.DONOR_K,
            "alphas": v406.ALPHAS.tolist(),
            "minimum_cross_view_delta": MIN_CROSS_VIEW_DELTA,
        },
        "fold_rows": rows,
        "aggregate": aggregate,
        "selection": {
            "best_view_pair": best_name,
            "minimum_cross_view_delta": MIN_CROSS_VIEW_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "passed": passed,
            "next_action": (
                "learn view-consistent donor identity from official train H"
                if passed
                else "terminate the v405 donor-oracle branch as view-specific matching"
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
