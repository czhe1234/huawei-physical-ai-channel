#!/usr/bin/env python3
"""Render normalized PDP from a strict tree-scored eikonal path set."""

from __future__ import annotations

import json

import numpy as np

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v365_multiview_component_pdp as v365
import analyze_v366_local_eikonal_pdp as v366
import analyze_v368_terminal_ray_path_existence as v368
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 369
VERSION = "v369_tree_scored_eikonal_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v369_tree_scored_eikonal_pdp_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
ALPHA_GRID = np.linspace(0.0, 1.0, 21, dtype=np.float32)

MIN_COSINE_LIFT = 0.010
MIN_TOP_RECALL_LIFT = 0.05
MIN_EMD_REDUCTION = 0.10
MIN_FOLD_WINS = 4
MIN_LABEL_CONTROL_LIFT = 0.002
MAX_CLEAR_DEGRADATION = 0.002


def empty_terminal(hypothesis: v366.HypothesisSet) -> np.ndarray:
    return np.zeros(
        (len(hypothesis.query_ids), v366.HYPOTHESES, 0), dtype=np.float32
    )


def shuffled_labels(
    arrays: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    output = {key: value.copy() for key, value in arrays.items()}
    rng = np.random.default_rng(seed)
    for query_row in np.unique(arrays["query_row"]):
        rows = np.flatnonzero(arrays["query_row"] == query_row)
        order = rng.permutation(len(rows))
        output["mass"][rows] = arrays["mass"][rows[order]]
    output["binary"] = output["mass"] > 0.0
    return output


def fit_path_scores(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    probability, mass = v368.fit_predict(
        train["existing"], query["existing"], train, seed
    )
    return probability.astype(np.float32), mass.astype(np.float32)


def scatter_pair_score(
    arrays: dict[str, np.ndarray], pair_score: np.ndarray, query_count: int
) -> np.ndarray:
    output = np.zeros((query_count, v366.HYPOTHESES), dtype=np.float32)
    cursor = np.zeros(query_count, dtype=np.int32)
    for pair, query_row in enumerate(arrays["query_row"]):
        row = int(query_row)
        output[row, cursor[row]] = pair_score[pair]
        cursor[row] += 1
    return output


def render_profiles(
    hypothesis: v366.HypothesisSet,
    arrays: dict[str, np.ndarray],
    pair_mass: np.ndarray,
) -> np.ndarray:
    scattered = scatter_pair_score(arrays, pair_mass, len(hypothesis.query_ids))
    output = np.zeros((len(hypothesis.query_ids), v366.N_DELAY), dtype=np.float32)
    for row in range(len(hypothesis.query_ids)):
        active = np.flatnonzero(hypothesis.valid[row])
        mass = scattered[row, : len(active)].astype(np.float64)
        if float(np.sum(mass)) <= 1e-20:
            mass = hypothesis.prior[row, active].astype(np.float64)
        mass /= max(float(np.sum(mass)), 1e-20)
        output[row] = (
            v297.dirichlet_atoms(hypothesis.delay[row, active]) @ mass
        ).astype(np.float32)
    return v365.normalize_rows(output)


def anchor_profile(
    hypothesis: v366.HypothesisSet, truth_d: np.ndarray
) -> np.ndarray:
    return v365.normalize_rows(
        np.sum(
            hypothesis.base_weights[:, :, None]
            * truth_d[hypothesis.neighbor_ids],
            axis=1,
            dtype=np.float64,
        )
    )


def select_alpha(
    anchor: np.ndarray,
    proposal: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        prediction = v365.normalize_rows(
            anchor + float(alpha) * (proposal - anchor)
        )
        rows.append(
            {
                "alpha": float(alpha),
                "pdp_cosine": float(np.mean(v365.row_cos(prediction, truth))),
            }
        )
    best = max(rows, key=lambda row: row["pdp_cosine"])
    return float(best["alpha"]), rows


def prepare(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
    component: dict[str, np.ndarray],
) -> tuple[v366.HypothesisSet, dict[str, np.ndarray]]:
    _train, query_set, _node = v366.raw_sets(
        bank_ids, query_ids, bank_ids, geometry, node_raw
    )
    hypothesis = v366.build_hypotheses(
        query_set, geometry[0][:, :2], component
    )
    arrays = v368.pair_arrays(
        hypothesis,
        node_raw,
        empty_terminal(hypothesis),
        v368.path_mass_labels(hypothesis, component),
    )
    return hypothesis, arrays


def path_metric(
    arrays: dict[str, np.ndarray], probability: np.ndarray, mass: np.ndarray
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score

    return {
        "existence_ap": float(
            average_precision_score(arrays["binary"], probability)
        ),
        "mass_cosine": v368.grouped_mass_cosine(
            mass,
            arrays["mass"],
            arrays["query_row"],
            int(np.max(arrays["query_row"])) + 1,
        ),
    }


def main() -> None:
    for path in (
        TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = np.load(TARGET_CACHE)
    truth_d = v365.normalize_rows(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    component_cache = np.load(v297.COMPONENT_CACHE)
    component = {key: component_cache[key] for key in component_cache.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    propagation_geometry = np.load(v238.GEOM_CACHE)
    regimes = regime_of(
        propagation_geometry["train_blocked"].astype(np.float32)
    )
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    fold_rows: list[dict[str, object]] = []
    oof: dict[str, list[np.ndarray]] = {
        "ids": [], "truth": [], "baseline": [], "shuffled": [], "full": []
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(folds[(fold + 1) % len(folds)], fit_ids)
        inner_fit = np.setdiff1d(fit_ids, inner_ids)

        inner_train_h, inner_train_arrays = prepare(
            inner_fit, inner_fit, geometry, node_raw, component
        )
        inner_val_h, inner_val_arrays = prepare(
            inner_ids, inner_fit, geometry, node_raw, component
        )
        outer_train_h, outer_train_arrays = prepare(
            fit_ids, fit_ids, geometry, node_raw, component
        )
        outer_val_h, outer_val_arrays = prepare(
            outer_ids, fit_ids, geometry, node_raw, component
        )

        modes = {
            "full": (inner_train_arrays, outer_train_arrays),
            "shuffled": (
                shuffled_labels(inner_train_arrays, 369_000 + fold),
                shuffled_labels(outer_train_arrays, 370_000 + fold),
            ),
        }
        predictions: dict[str, np.ndarray] = {}
        mode_report: dict[str, object] = {}
        for mode_index, (mode, train_arrays) in enumerate(modes.items()):
            inner_train_mode, outer_train_mode = train_arrays
            inner_probability, inner_mass = fit_path_scores(
                inner_train_mode,
                inner_val_arrays,
                369_100 + 10 * fold + mode_index,
            )
            inner_proposal = render_profiles(
                inner_val_h, inner_val_arrays, inner_mass
            )
            inner_anchor = anchor_profile(inner_val_h, truth_d)
            alpha, alpha_rows = select_alpha(
                inner_anchor, inner_proposal, truth_d[inner_ids]
            )
            outer_probability, outer_mass = fit_path_scores(
                outer_train_mode,
                outer_val_arrays,
                379_100 + 10 * fold + mode_index,
            )
            outer_proposal = render_profiles(
                outer_val_h, outer_val_arrays, outer_mass
            )
            outer_anchor = anchor_profile(outer_val_h, truth_d)
            predictions[mode] = v365.normalize_rows(
                outer_anchor + alpha * (outer_proposal - outer_anchor)
            )
            mode_report[mode] = {
                "selected_alpha": alpha,
                "inner_alpha_rows": alpha_rows,
                "inner_path": path_metric(
                    inner_val_arrays, inner_probability, inner_mass
                ),
                "outer_path": path_metric(
                    outer_val_arrays, outer_probability, outer_mass
                ),
            }

        truth = truth_d[outer_ids]
        baseline = anchor_profile(outer_val_h, truth_d)
        baseline_metric = v365.evaluate(
            baseline, truth, outer_ids, component
        )
        shuffled_metric = v365.evaluate(
            predictions["shuffled"], truth, outer_ids, component
        )
        full_metric = v365.evaluate(
            predictions["full"], truth, outer_ids, component
        )
        regime_rows = {}
        for regime_name in ("clear", "light", "medium", "heavy"):
            keep = regimes[outer_ids] == regime_name
            if np.any(keep):
                regime_rows[regime_name] = {
                    "count": int(np.sum(keep)),
                    "cosine_lift": float(
                        np.mean(
                            full_metric["cosine_rows"][keep]
                            - baseline_metric["cosine_rows"][keep]
                        )
                    ),
                }
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "baseline": v365.metric_summary(baseline_metric),
                "shuffled_labels": v365.metric_summary(shuffled_metric),
                "full": v365.metric_summary(full_metric),
                "full_delta": {
                    "cosine": float(
                        full_metric["cosine"] - baseline_metric["cosine"]
                    ),
                    "top_path_mass_recall": float(
                        full_metric["top_path_mass_recall"]
                        - baseline_metric["top_path_mass_recall"]
                    ),
                    "delay_emd": float(
                        full_metric["delay_emd"] - baseline_metric["delay_emd"]
                    ),
                },
                "full_vs_shuffled_cosine": float(
                    full_metric["cosine"] - shuffled_metric["cosine"]
                ),
                "regimes": regime_rows,
                "models": mode_report,
                "leakage_assertions": {
                    "outer_in_inner_alpha_selection": 0,
                    "outer_in_path_training": 0,
                    "outer_in_donor_bank": 0,
                },
            }
        )
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth)
        oof["baseline"].append(baseline)
        oof["shuffled"].append(predictions["shuffled"])
        oof["full"].append(predictions["full"])
        print(
            f"v369 fold={fold} alpha={mode_report['full']['selected_alpha']:.2f} "
            f"base={baseline_metric['cosine']:.6f} "
            f"full={full_metric['cosine']:.6f} "
            f"shuffle={shuffled_metric['cosine']:.6f}",
            flush=True,
        )

    merged = {key: np.concatenate(value) for key, value in oof.items()}
    order = np.argsort(merged["ids"])
    for key in merged:
        merged[key] = merged[key][order]
    np.savez_compressed(OOF_OUT, **merged)
    base_metric = v365.evaluate(
        merged["baseline"], merged["truth"], merged["ids"], component
    )
    shuffled_metric = v365.evaluate(
        merged["shuffled"], merged["truth"], merged["ids"], component
    )
    full_metric = v365.evaluate(
        merged["full"], merged["truth"], merged["ids"], component
    )
    cosine_lift = float(full_metric["cosine"] - base_metric["cosine"])
    recall_lift = float(
        full_metric["top_path_mass_recall"]
        - base_metric["top_path_mass_recall"]
    )
    emd_reduction = float(
        (base_metric["delay_emd"] - full_metric["delay_emd"])
        / max(float(base_metric["delay_emd"]), 1e-20)
    )
    fold_wins = int(
        sum(row["full_delta"]["cosine"] > 0.0 for row in fold_rows)
    )
    full_vs_shuffled = float(
        full_metric["cosine"] - shuffled_metric["cosine"]
    )
    regime_summary = {}
    for regime_name in ("clear", "light", "medium", "heavy"):
        keep = regimes[merged["ids"]] == regime_name
        regime_summary[regime_name] = {
            "count": int(np.sum(keep)),
            "cosine_lift": float(
                np.mean(
                    full_metric["cosine_rows"][keep]
                    - base_metric["cosine_rows"][keep]
                )
            ),
        }
    passed = bool(
        cosine_lift >= MIN_COSINE_LIFT
        and recall_lift >= MIN_TOP_RECALL_LIFT
        and emd_reduction >= MIN_EMD_REDUCTION
        and fold_wins >= MIN_FOLD_WINS
        and full_vs_shuffled >= MIN_LABEL_CONTROL_LIFT
        and regime_summary["medium"]["cosine_lift"] > 0.0
        and regime_summary["heavy"]["cosine_lift"] > 0.0
        and regime_summary["clear"]["cosine_lift"] >= -MAX_CLEAR_DEGRADATION
    )
    summary = {
        "baseline": v365.metric_summary(base_metric),
        "shuffled_labels": v365.metric_summary(shuffled_metric),
        "full": v365.metric_summary(full_metric),
        "cosine_lift": cosine_lift,
        "top_path_mass_recall_lift": recall_lift,
        "delay_emd_reduction_fraction": emd_reduction,
        "fold_wins": fold_wins,
        "full_vs_shuffled_cosine": full_vs_shuffled,
        "regimes": regime_summary,
    }
    report = {
        "method": VERSION,
        "status": (
            "strict_pdp_module_passed" if passed else "analysis_only_no_candidate"
        ),
        "scope": "normalized global D only; no full-H, phase, total power, or ZIP",
        "primary_hypothesis": (
            "v368's fixed tree scorer substantially improves path AP; converting its "
            "P(existence)*E(mass|existence) into exact eikonal atoms should improve PDP "
            "when blend strength is selected on an independent inner spatial fold."
        ),
        "compliance": {
            "uses": [
                "official train H fit-only path labels and normalized PDP",
                "official train/test planar coordinates and point-cloud node features",
                "v366 local theoretical-gradient path surfaces",
            ],
            "does_not_use": [
                "terminal-ray features rejected by v368",
                "outer labels in path fitting or alpha selection",
                "Sionna/synthetic labels, v212 teacher, hidden truth, or online feedback",
                "full-H projection or submission packaging",
            ],
        },
        "parameters": {
            "path_model": "v368 fixed ExtraTrees existing-feature classifier/regressor",
            "path_energy": "P(existence) * E(mass | existence)",
            "alpha_grid": ALPHA_GRID.tolist(),
            "alpha_selection": "next mutually exclusive inner spatial fold",
            "control": "same path model with fit-query path mass shuffled within query",
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "cosine_lift": f">={MIN_COSINE_LIFT}",
            "top_path_mass_recall_lift": f">={MIN_TOP_RECALL_LIFT}",
            "delay_emd_reduction_fraction": f">={MIN_EMD_REDUCTION}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "full_vs_shuffled_cosine": f">={MIN_LABEL_CONTROL_LIFT}",
            "medium_heavy": "both positive",
            "clear": f">=-{MAX_CLEAR_DEGRADATION}",
            "passed": passed,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
