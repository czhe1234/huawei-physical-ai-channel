#!/usr/bin/env python3
"""Source-specific mass transport with frozen planar Matern kriging weights."""

from __future__ import annotations

import json

import numpy as np

import analyze_v294_heteroscedastic_planar_gp as v294
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v354_nested_virtual_source_transport as v354
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 358
VERSION = "v358_virtual_source_matern_field"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v358_virtual_source_matern_oof.npz"

NEIGHBORS = 32
KERNEL = v294.Kernel(6.5, 0.08, 0.0, 0.0, 0.0, True)
MIN_MEAN_DELTA = 0.01
MIN_FOLD_WINS = 4


def matern_mass(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    fit_target: np.ndarray,
    train_pos: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    indices, query_distance = v294.nearest(
        query_ids,
        fit_ids,
        train_pos,
        train_pos,
        NEIGHBORS,
        False,
    )
    selected_pos = train_pos[indices, :2]
    pair_distance = np.linalg.norm(
        selected_pos[:, :, None, :] - selected_pos[:, None, :, :], axis=3
    ).astype(np.float32)
    zeros_query = np.zeros_like(query_distance, dtype=np.float32)
    zeros_pair = np.zeros_like(pair_distance, dtype=np.float32)
    neighborhood = v294.Neighborhood(
        indices=indices,
        query_distance=query_distance,
        pair_distance=pair_distance,
        query_signature_distance=zeros_query,
        pair_signature_distance=zeros_pair,
        query_path_distance=zeros_query,
        pair_path_distance=zeros_pair,
    )
    weight = v294.kriging_weights(
        neighborhood,
        KERNEL,
        np.zeros(len(train_pos), dtype=np.float32),
    )
    lookup = np.full(len(train_pos), -1, dtype=np.int64)
    lookup[fit_ids] = np.arange(len(fit_ids), dtype=np.int64)
    rows = lookup[indices]
    if np.any(rows < 0):
        raise AssertionError("Matern neighbor outside fit bank")
    prediction = np.sum(
        weight[:, :, None] * fit_target[rows], axis=1, dtype=np.float64
    ).astype(np.float32)
    return prediction, {
        "effective_neighbors": summarize(
            1.0 / np.maximum(np.sum(weight**2, axis=1), 1e-20)
        ),
        "weight_max": summarize(np.max(weight, axis=1)),
        "negative_weight_fraction": float(np.mean(weight < 0.0)),
    }


def main() -> None:
    target = np.load(v354.TARGET_CACHE)
    truth_sh = v354.normalize(target["sh"].astype(np.float32))
    truth_d = v354.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    observation, observation_report = v354.build_physical_observations()
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

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
        predicted_mass, weight_diagnostics = matern_mass(
            outer_ids, fit_ids, fit_target, train_pos
        )
        knn_mass = v311.knn_prediction(
            outer_ids, fit_ids, fit_target, train_pos
        )
        neighbor = v354.neighbors(
            outer_ids, fit_ids, train_pos[:, :2], False
        )
        if np.intersect1d(np.unique(neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered v358 anchor")
        anchor = v354.anchor_from_neighbors(truth_sh, neighbor)
        prediction, diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit_target,
            centers,
            train_pos[:, :2],
            query_mass_override=predicted_mass,
        )
        base_metric = v354.metrics(
            anchor, truth_sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = v354.metrics(
            prediction, truth_sh[outer_ids], truth_d[outer_ids]
        )
        delta = {
            key: float(model_metric[key] - base_metric[key])
            for key in ("sh", "d", "weighted")
        }
        fold_rows.append(
            {
                "fold": fold,
                "source_discovery": source_report,
                "matern_mass_metrics": v311.evaluate_prediction(
                    predicted_mass, outer_target
                ),
                "knn_mass_metrics": v311.evaluate_prediction(
                    knn_mass, outer_target
                ),
                "predicted_total_mass": summarize(predicted_mass.sum(axis=1)),
                "true_total_mass": summarize(outer_target.sum(axis=1)),
                "weight_diagnostics": weight_diagnostics,
                "baseline": base_metric,
                "prediction": model_metric,
                "delta": delta,
                "transport": {
                    key: summarize(value) for key, value in diagnostics.items()
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_kriging_bank": 0,
                    "outer_in_anchor_bank": 0,
                },
            }
        )
        oof_ids.append(outer_ids)
        oof_anchor.append(anchor)
        oof_prediction.append(prediction)
        print(
            f"v358 fold={fold} sources={len(centers)} "
            f"knn_ap={fold_rows[-1]['knn_mass_metrics']['pair_ap']:.4f} "
            f"matern_ap={fold_rows[-1]['matern_mass_metrics']['pair_ap']:.4f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )

    mean_delta = {
        key: float(np.mean([row["delta"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    wins = int(sum(row["delta"]["weighted"] > 0.0 for row in fold_rows))
    ap_gain = float(
        np.mean(
            [
                row["matern_mass_metrics"]["pair_ap"]
                - row["knn_mass_metrics"]["pair_ap"]
                for row in fold_rows
            ]
        )
    )
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
        "round": 5,
        "primary_hypothesis": (
            "Each virtual source has its own smooth planar mass field; frozen positive-weight "
            "Matern ordinary kriging should estimate that field more faithfully than inverse-"
            "distance KNN while retaining source identity."
        ),
        "compliance": {
            "uses": [
                "official outer-fit H source labels and train coordinates",
                "v294 Matern kernel frozen from prior strict official evidence",
            ],
            "does_not_use": [
                "outer H in source discovery or kriging bank",
                "synthetic/v212 teacher, restored RT channel, local or online labels",
            ],
        },
        "parameters": {
            "neighbors": NEIGHBORS,
            "kernel": KERNEL.name,
            "kernel_selection": "frozen v294 modal SH kernel",
            "transport": "frozen v354",
            "parameter_search": "none",
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_delta": mean_delta,
            "fold_wins": wins,
            "matern_pair_ap_gain_vs_knn": ap_gain,
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
