#!/usr/bin/env python3
"""Transductive point-cloud graph propagation for virtual-source visibility."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v274_planar_barrier_attention as v274
import analyze_v275_planar_graph_conductance as v275
import analyze_v238_raw_path_conditional_angle as v238
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v354_nested_virtual_source_transport as v354
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 357
VERSION = "v357_transductive_source_visibility_graph"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v357_transductive_source_graph_oof.npz"

K_ANCHOR = 16
GRAPH_MIX = 0.30
GRAPH_ITERATIONS = 8
EXPERT_REGIMES = ("medium", "heavy")
MIN_MEAN_DELTA = 0.01
MIN_FOLD_WINS = 4


def knn_mass_all(
    known_ids: np.ndarray,
    known_target: np.ndarray,
    all_xy: np.ndarray,
) -> np.ndarray:
    take = min(K_ANCHOR, len(known_ids))
    distance, local = cKDTree(all_xy[known_ids]).query(all_xy, k=take)
    if take == 1:
        distance = distance[:, None]
        local = local[:, None]
    weight = 1.0 / np.maximum(distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(
        weight[:, :, None] * known_target[local], axis=1, dtype=np.float64
    ).astype(np.float32)


def propagate_mass(
    known_ids: np.ndarray,
    known_target: np.ndarray,
    all_xy: np.ndarray,
    matrix,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = knn_mass_all(known_ids, known_target, all_xy)
    field = anchor.copy()
    field[known_ids] = known_target
    unknown = np.setdiff1d(np.arange(len(all_xy), dtype=np.int64), known_ids)
    for _ in range(GRAPH_ITERATIONS):
        update = matrix @ field
        field[unknown] = (
            GRAPH_MIX * anchor[unknown]
            + (1.0 - GRAPH_MIX) * update[unknown]
        )
        field[known_ids] = known_target
    return np.maximum(field, 0.0).astype(np.float32), anchor


def main() -> None:
    target = np.load(v354.TARGET_CACHE)
    truth_sh = v354.normalize(target["sh"].astype(np.float32))
    truth_d = v354.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    observation, observation_report = v354.build_physical_observations()
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    (
        all_pos,
        signatures,
        blocker_count,
        _scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = v274.load_geometry()
    all_ids = np.arange(len(all_pos), dtype=np.int64)
    raw_graph = v274.build_raw_set(
        all_ids,
        all_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    matrix = v275.graph_matrix(raw_graph, raw_graph.base_weights, len(all_pos))
    train_regime = regime_of(
        np.load(v238.GEOM_CACHE)["train_blocked"].astype(np.float32)
    ).astype(str)

    fold_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_anchor: list[np.ndarray] = []
    oof_prediction: list[np.ndarray] = []
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = v311.discover_sources(observation, fit_ids)
        fit_target, _ = v311.assign_targets(
            observation, fit_ids, centers, train_pos
        )
        outer_target, _ = v311.assign_targets(
            observation, outer_ids, centers, train_pos
        )
        graph_field, graph_anchor = propagate_mass(
            fit_ids, fit_target, all_pos[:, :2], matrix
        )
        graph_mass = graph_field[outer_ids]
        raw_mass = graph_anchor[outer_ids]
        neighbor = v354.neighbors(
            outer_ids, fit_ids, train_pos[:, :2], False
        )
        if np.intersect1d(np.unique(neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered v357 anchor")
        anchor = v354.anchor_from_neighbors(truth_sh, neighbor)
        graph_prediction, diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit_target,
            centers,
            train_pos[:, :2],
            query_mass_override=graph_mass,
        )
        raw_prediction, _ = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit_target,
            centers,
            train_pos[:, :2],
            query_mass_override=raw_mass,
        )
        expert = np.isin(train_regime[outer_ids], EXPERT_REGIMES)
        prediction = anchor.copy()
        prediction[expert] = graph_prediction[expert]
        raw_expert_prediction = anchor.copy()
        raw_expert_prediction[expert] = raw_prediction[expert]
        base_metric = v354.metrics(
            anchor, truth_sh[outer_ids], truth_d[outer_ids]
        )
        raw_metric = v354.metrics(
            raw_expert_prediction, truth_sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = v354.metrics(
            prediction, truth_sh[outer_ids], truth_d[outer_ids]
        )
        delta = {
            key: float(model_metric[key] - base_metric[key])
            for key in ("sh", "d", "weighted")
        }
        delta_vs_raw = {
            key: float(model_metric[key] - raw_metric[key])
            for key in ("sh", "d", "weighted")
        }
        fold_rows.append(
            {
                "fold": fold,
                "source_discovery": source_report,
                "expert_count": int(np.sum(expert)),
                "expert_fraction": float(np.mean(expert)),
                "raw_mass_metrics": v311.evaluate_prediction(raw_mass, outer_target),
                "graph_mass_metrics": v311.evaluate_prediction(
                    graph_mass, outer_target
                ),
                "raw_total_mass": summarize(raw_mass.sum(axis=1)),
                "graph_total_mass": summarize(graph_mass.sum(axis=1)),
                "true_total_mass": summarize(outer_target.sum(axis=1)),
                "baseline": base_metric,
                "raw_expert_transport": raw_metric,
                "prediction": model_metric,
                "delta": delta,
                "delta_vs_raw_expert_transport": delta_vs_raw,
                "transport": {
                    key: summarize(value) for key, value in diagnostics.items()
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_graph_labels": 0,
                    "outer_in_knn_bank": 0,
                    "test_h_labels": 0,
                },
            }
        )
        oof_ids.append(outer_ids)
        oof_anchor.append(anchor)
        oof_prediction.append(prediction)
        print(
            f"v357 fold={fold} expert={np.sum(expert)} "
            f"raw_ap={fold_rows[-1]['raw_mass_metrics']['pair_ap']:.4f} "
            f"graph_ap={fold_rows[-1]['graph_mass_metrics']['pair_ap']:.4f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )

    mean_delta = {
        key: float(np.mean([row["delta"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    mean_vs_raw = {
        key: float(
            np.mean(
                [row["delta_vs_raw_expert_transport"][key] for row in fold_rows]
            )
        )
        for key in ("sh", "d", "weighted")
    }
    wins = int(sum(row["delta"]["weighted"] > 0.0 for row in fold_rows))
    passed = bool(mean_delta["weighted"] >= MIN_MEAN_DELTA and wins >= MIN_FOLD_WINS)
    np.savez_compressed(
        OOF_OUT,
        ids=np.concatenate(oof_ids),
        anchor=np.concatenate(oof_anchor),
        prediction=np.concatenate(oof_prediction),
    )
    report = {
        "method": VERSION,
        "status": "strict_oof_breakthrough" if passed else "analysis_only_no_candidate",
        "round": 4,
        "primary_hypothesis": (
            "Virtual-source visibility follows multi-hop planar connectivity; source mass "
            "should diffuse through the official train/test geometry graph and be applied "
            "only in medium/heavy regions where KNN path identity is weakest."
        ),
        "compliance": {
            "uses": [
                "official outer-fit H source labels",
                "official train/test coordinates as unlabeled graph nodes",
                "official Round1_Map point-cloud geometry edge weights",
            ],
            "does_not_use": [
                "outer/test H as graph labels",
                "v207/v212/synthetic teacher, restored scene RT, local or online labels",
            ],
        },
        "parameters": {
            "graph": "v274 official-pointcloud geometry base conductance",
            "mix": GRAPH_MIX,
            "iterations": GRAPH_ITERATIONS,
            "selection_source": "frozen v275 strict official evidence",
            "expert_regimes": list(EXPERT_REGIMES),
            "transport": "frozen v354",
            "parameter_search": "none",
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_delta": mean_delta,
            "mean_delta_vs_raw_expert_transport": mean_vs_raw,
            "fold_wins": wins,
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
