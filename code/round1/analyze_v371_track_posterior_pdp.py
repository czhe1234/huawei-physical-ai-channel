#!/usr/bin/env python3
"""Fuse local path evidence with v370 lifecycle tracks as explicit odds."""

from __future__ import annotations

import json
import math

import numpy as np

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v365_multiview_component_pdp as v365
import analyze_v368_terminal_ray_path_existence as v368
import analyze_v369_tree_scored_eikonal_pdp as v369
import analyze_v370_path_track_lifecycle_pdp as v370
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 371
VERSION = "v371_track_posterior_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v371_track_posterior_pdp_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
V370_REPORT = PATHS.version_report_dir(370) / "v370_path_track_lifecycle_pdp.json"

LIFECYCLE_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
CONDITIONAL_MASS_STRENGTHS = (0.0, 0.25, 0.5)

MIN_COSINE_LIFT = 0.010
MIN_TOP_RECALL_LIFT = 0.05
MIN_EMD_REDUCTION = 0.10
MIN_FOLD_WINS = 4
MIN_LABEL_CONTROL_LIFT = 0.002
MAX_CLEAR_DEGRADATION = 0.002


def logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float64)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def fit_local_scores(
    bundle: v370.TrackBundle,
    train_ids: np.ndarray,
    query_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_keep = v370.query_node_mask(bundle, train_ids)
    query_keep = v370.query_node_mask(bundle, query_ids)
    train_arrays = {
        "binary": bundle.arrays["binary"][train_keep],
        "mass": bundle.arrays["mass"][train_keep],
    }
    probability, mass = v368.fit_predict(
        bundle.arrays["existing"][train_keep],
        bundle.arrays["existing"][query_keep],
        train_arrays,
        seed,
    )
    full_probability = np.zeros(len(bundle.node_query_id), dtype=np.float32)
    full_mass = np.zeros(len(bundle.node_query_id), dtype=np.float32)
    full_probability[query_keep] = probability
    full_mass[query_keep] = mass
    prior_probability = float(np.mean(train_arrays["binary"]))
    return full_probability, full_mass, prior_probability


def graph_scores(
    bundle: v370.TrackBundle,
    train_ids: np.ndarray,
    shuffle_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if shuffle_seed is None:
        binary = bundle.arrays["binary"]
        mass = bundle.arrays["mass"]
    else:
        binary, mass = v370.shuffled_path_labels(bundle.arrays, shuffle_seed)
    _, probability, mass_score = v370.diffuse_track_labels(
        bundle, train_ids, binary, mass
    )
    return probability, mass_score


def posterior_mass(
    local_probability: np.ndarray,
    local_mass: np.ndarray,
    graph_probability: np.ndarray,
    graph_mass: np.ndarray,
    prior_probability: float,
    lifecycle_strength: float,
    conditional_mass_strength: float,
) -> np.ndarray:
    lifecycle_log_odds = (
        logit(local_probability)
        + lifecycle_strength
        * (logit(graph_probability) - float(logit(prior_probability)))
    )
    probability = sigmoid(lifecycle_log_odds)
    local_conditional = np.divide(
        local_mass,
        np.maximum(local_probability, 1e-5),
    )
    graph_conditional = np.divide(
        graph_mass,
        np.maximum(graph_probability, 1e-5),
    )
    log_conditional = (
        (1.0 - conditional_mass_strength)
        * np.log(np.maximum(local_conditional, 1e-7))
        + conditional_mass_strength
        * np.log(np.maximum(graph_conditional, 1e-7))
    )
    return (probability * np.exp(log_conditional)).astype(np.float32)


def select_action(
    bundle: v370.TrackBundle,
    query_ids: np.ndarray,
    truth_d: np.ndarray,
    local_probability: np.ndarray,
    local_mass: np.ndarray,
    graph_probability: np.ndarray,
    graph_mass: np.ndarray,
    prior_probability: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    anchor = v370.query_anchor(bundle, truth_d, query_ids)
    rows: list[dict[str, float]] = []
    for lifecycle_strength in LIFECYCLE_STRENGTHS:
        for conditional_strength in CONDITIONAL_MASS_STRENGTHS:
            combined_mass = posterior_mass(
                local_probability,
                local_mass,
                graph_probability,
                graph_mass,
                prior_probability,
                lifecycle_strength,
                conditional_strength,
            )
            proposal = v370.render_query_profiles(
                bundle, combined_mass, query_ids
            )
            alpha, alpha_rows = v369.select_alpha(
                anchor, proposal, truth_d[query_ids]
            )
            score = max(
                row["pdp_cosine"] for row in alpha_rows
                if math.isclose(row["alpha"], alpha, abs_tol=1e-7)
            )
            rows.append(
                {
                    "lifecycle_strength": float(lifecycle_strength),
                    "conditional_mass_strength": float(conditional_strength),
                    "alpha": float(alpha),
                    "pdp_cosine": float(score),
                }
            )
    best = max(rows, key=lambda row: row["pdp_cosine"])
    return best, rows


def apply_action(
    bundle: v370.TrackBundle,
    query_ids: np.ndarray,
    truth_d: np.ndarray,
    local_probability: np.ndarray,
    local_mass: np.ndarray,
    graph_probability: np.ndarray,
    graph_mass: np.ndarray,
    prior_probability: float,
    action: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    combined_mass = posterior_mass(
        local_probability,
        local_mass,
        graph_probability,
        graph_mass,
        prior_probability,
        action["lifecycle_strength"],
        action["conditional_mass_strength"],
    )
    proposal = v370.render_query_profiles(bundle, combined_mass, query_ids)
    anchor = v370.query_anchor(bundle, truth_d, query_ids)
    prediction = v365.normalize_rows(
        anchor + action["alpha"] * (proposal - anchor)
    )
    return prediction, combined_mass


def main() -> None:
    for path in (
        TARGET_CACHE,
        V370_REPORT,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    v370_report = json.loads(V370_REPORT.read_text(encoding="utf-8"))
    if not bool(v370_report["audit_summary"]["passed"]):
        raise RuntimeError("v370 path-track audit did not pass")

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
    regimes = regime_of(propagation_geometry["train_blocked"].astype(np.float32))
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    fold_rows: list[dict[str, object]] = []
    oof: dict[str, list[np.ndarray]] = {
        "ids": [], "truth": [], "baseline": [], "shuffled": [], "full": []
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(folds[(fold + 1) % len(folds)], fit_ids)
        inner_fit = np.setdiff1d(fit_ids, inner_ids)
        inner_bundle = v370.prepare_bundle(
            fit_ids, inner_fit, geometry, node_raw, component
        )
        outer_bundle = v370.prepare_bundle(
            valid_ids, fit_ids, geometry, node_raw, component
        )

        inner_local_probability, inner_local_mass, inner_prior = fit_local_scores(
            inner_bundle,
            inner_fit,
            inner_ids,
            371_100 + fold,
        )
        outer_local_probability, outer_local_mass, outer_prior = fit_local_scores(
            outer_bundle,
            fit_ids,
            outer_ids,
            381_100 + fold,
        )
        predictions: dict[str, np.ndarray] = {}
        model_rows: dict[str, object] = {}
        for mode_index, mode in enumerate(("full", "shuffled")):
            inner_shuffle = None if mode == "full" else 371_000 + fold
            outer_shuffle = None if mode == "full" else 372_000 + fold
            inner_graph_probability, inner_graph_mass = graph_scores(
                inner_bundle, inner_fit, inner_shuffle
            )
            action, action_rows = select_action(
                inner_bundle,
                inner_ids,
                truth_d,
                inner_local_probability,
                inner_local_mass,
                inner_graph_probability,
                inner_graph_mass,
                inner_prior,
            )
            outer_graph_probability, outer_graph_mass = graph_scores(
                outer_bundle, fit_ids, outer_shuffle
            )
            prediction, combined_mass = apply_action(
                outer_bundle,
                outer_ids,
                truth_d,
                outer_local_probability,
                outer_local_mass,
                outer_graph_probability,
                outer_graph_mass,
                outer_prior,
                action,
            )
            predictions[mode] = prediction
            query_keep = v370.query_node_mask(outer_bundle, outer_ids)
            model_rows[mode] = {
                "selected_action": action,
                "inner_action_rows": action_rows,
                "outer_local_path": v370.path_metrics(
                    outer_bundle,
                    outer_ids,
                    outer_local_probability,
                    outer_local_mass,
                ),
                "outer_graph_path": v370.path_metrics(
                    outer_bundle,
                    outer_ids,
                    outer_graph_probability,
                    outer_graph_mass,
                ),
                "outer_combined_path": v370.path_metrics(
                    outer_bundle,
                    outer_ids,
                    sigmoid(
                        logit(outer_local_probability)
                        + action["lifecycle_strength"]
                        * (logit(outer_graph_probability) - float(logit(outer_prior)))
                    ).astype(np.float32),
                    combined_mass,
                ),
                "query_candidate_count": int(np.sum(query_keep)),
            }

        truth = truth_d[outer_ids]
        baseline = v370.query_anchor(outer_bundle, truth_d, outer_ids)
        baseline_metric = v365.evaluate(baseline, truth, outer_ids, component)
        shuffled_metric = v365.evaluate(
            predictions["shuffled"], truth, outer_ids, component
        )
        full_metric = v365.evaluate(predictions["full"], truth, outer_ids, component)
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
                    "cosine": float(full_metric["cosine"] - baseline_metric["cosine"]),
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
                "models": model_rows,
                "leakage_assertions": {
                    "outer_in_inner_action_selection": 0,
                    "outer_in_path_training": 0,
                    "outer_in_graph_seed": 0,
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
            f"v371 fold={fold} action={model_rows['full']['selected_action']} "
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
        full_metric["top_path_mass_recall"] - base_metric["top_path_mass_recall"]
    )
    emd_reduction = float(
        (base_metric["delay_emd"] - full_metric["delay_emd"])
        / max(float(base_metric["delay_emd"]), 1e-20)
    )
    fold_wins = int(sum(row["full_delta"]["cosine"] > 0.0 for row in fold_rows))
    full_vs_shuffled = float(full_metric["cosine"] - shuffled_metric["cosine"])
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
        "status": "strict_pdp_module_passed" if passed else "analysis_only_no_candidate",
        "scope": "normalized global D only; no full-H, phase, total power, test target, or ZIP",
        "primary_hypothesis": (
            "The v370 path track must act as an explicit lifecycle likelihood ratio, "
            "rather than an optional tree feature, to correct local path-existence odds."
        ),
        "compliance": {
            "uses": [
                "official train H fit-only path labels and normalized PDP",
                "official planar coordinates and point-cloud node features",
                "v370 frozen path graph and v368 local path model",
            ],
            "does_not_use": [
                "outer labels as graph seeds, donor components, model targets, or action selection",
                "Sionna/synthetic labels, v212 teacher, hidden truth, local truth, or online feedback",
                "full-H projection, phase, total power, test target, or submission packaging",
            ],
        },
        "parameters": {
            "lifecycle_strengths": list(LIFECYCLE_STRENGTHS),
            "conditional_mass_strengths": list(CONDITIONAL_MASS_STRENGTHS),
            "fusion": "local odds * (track odds / fit prevalence odds)^lifecycle_strength",
            "conditional_energy": "geometric blend of local and track E(mass|existence)",
            "selection": "lifecycle, conditional-energy, and PDP alpha on next independent spatial fold",
            "control": "same action search with fit-query path identity shuffled within query",
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
            "report": PATHS.relative(REPORT),
            "oof": PATHS.relative(OOF_OUT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
