#!/usr/bin/env python3
"""Replace total-power KNN D with scorer-aligned link-normalized KNN D."""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v238_raw_path_conditional_angle as v238
import analyze_v239_nnls_path_energy_decoder as v239
import analyze_v254_route_topology_spectral_field as v254
import analyze_v344_linkwise_delay_attention as v344
import analyze_v345_scorer_aligned_path_selector as v345
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 346
VERSION = "v346_scorer_aligned_knn_correction"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v346_Round1_Test_Channel.npy"
TARGETS_OUT = OUT_DIR / "v346_scorer_aligned_knn_targets.npz"
SOURCE = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"

OUTER_SEEDS = (346, 1346, 2346, 3346, 4346)
OUTER_COUNT = 320
ALPHA_GRID = np.linspace(0.0, 1.5, 31, dtype=np.float32)
MIN_DELTA = 0.005
K = v238.ANCHOR_K


def predict(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    bank_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices, weights = v218.neighbor_index_weights(
        query_pos, query_blocked, bank_pos, bank_blocked, k=K
    )
    value = v239.normalize_d(
        np.sum(weights[:, :, None] * bank_target[indices], axis=1, dtype=np.float64)
    )
    return value, indices, weights


def main() -> None:
    clean, _raw, _power, valid_old = v215.load_targets()
    old_d = v239.normalize_d(clean["d"])
    link_d = v344.link_global_d(PATHS.round1 / "Round1_Train_Channel.npy")
    valid_ids = np.flatnonzero(valid_old & (link_d.sum(axis=1) > 1e-20)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v238.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    rows = []
    ids_parts = []; old_parts = []; link_parts = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        old_pred, _, _ = predict(
            train_pos[outer_ids], train_blocked[outer_ids],
            train_pos[fit_ids], train_blocked[fit_ids], old_d[fit_ids]
        )
        link_pred, _, _ = predict(
            train_pos[outer_ids], train_blocked[outer_ids],
            train_pos[fit_ids], train_blocked[fit_ids], link_d[fit_ids]
        )
        old_score = float(np.mean(v239.row_cos(old_pred, link_d[outer_ids])))
        link_score = float(np.mean(v239.row_cos(link_pred, link_d[outer_ids])))
        rows.append({"fold": fold, "seed": seed, "old_total_d": old_score, "link_d": link_score, "raw_delta": link_score - old_score})
        ids_parts.append(outer_ids); old_parts.append(old_pred); link_parts.append(link_pred)
        print(f"v346 fold={fold+1}/5 raw_delta={link_score-old_score:+.6f}", flush=True)
    ids_all = np.concatenate(ids_parts); old_all = np.concatenate(old_parts); link_all = np.concatenate(link_parts)
    alpha_rows = []
    for alpha in ALPHA_GRID:
        value = v239.normalize_d(old_all + float(alpha) * (link_all - old_all))
        alpha_rows.append({"alpha": float(alpha), "score": float(np.mean(v239.row_cos(value, link_d[ids_all])))})
    best = max(alpha_rows, key=lambda item: item["score"]); alpha = float(best["alpha"])
    fold_delta = []
    for row, ids, old_pred, link_pred in zip(rows, ids_parts, old_parts, link_parts):
        value = v239.normalize_d(old_pred + alpha * (link_pred - old_pred))
        delta = float(np.mean(v239.row_cos(value, link_d[ids]))) - float(row["old_total_d"])
        row["blended_delta"] = delta; fold_delta.append(delta)
    mean_delta = float(np.mean(fold_delta)); wins = int(np.sum(np.asarray(fold_delta) > 0))
    selected = bool(alpha > 0 and mean_delta >= MIN_DELTA and wins >= 4)
    local_score = None; projection = None
    if selected:
        old_test, indices, weights = predict(
            test_pos, test_blocked, train_pos[valid_ids], train_blocked[valid_ids], old_d[valid_ids]
        )
        link_test, _, _ = predict(
            test_pos, test_blocked, train_pos[valid_ids], train_blocked[valid_ids], link_d[valid_ids]
        )
        current = v344.link_global_d(SOURCE)
        target_test = v239.normalize_d(current + alpha * (link_test - old_test))
        np.savez_compressed(
            TARGETS_OUT,
            old_test=old_test,
            link_test=link_test,
            current=current,
            target=target_test,
            indices=valid_ids[indices],
            weights=weights,
            alpha=np.asarray(alpha, dtype=np.float32),
        )
        projection = v345.project_linkwise(
            SOURCE, OUTPUT, {"d": target_test}, np.zeros(len(test_pos)), 1.0
        )
        local_score = v208.score_prediction(OUTPUT)
    report = {
        "method": VERSION,
        "status": "strict_complete_local_diagnostic" if selected else "analysis_only_no_candidate",
        "primary_hypothesis": "The evaluator normalizes each link PDP independently, so KNN donors must be formed from link-normalized D rather than total-power D.",
        "compliance_boundary": {"uses": ["official train H", "official train/test coordinates", "official blocked geometry", "strict v284 carrier after official selection"], "does_not_use": ["v211/v212 output or labels", "synthetic training labels", "restored scene/RT coefficients", "online-score optimization"]},
        "parameters": {"neighbors": K, "outer_seeds": list(OUTER_SEEDS), "minimum_delta": MIN_DELTA, "source": PATHS.relative(SOURCE)},
        "folds": rows,
        "official_selection": {"alpha": alpha, "alpha_rows": alpha_rows, "fold_delta": fold_delta, "mean_delta": mean_delta, "wins": wins, "selected": selected},
        "projection": {key: summarize(value) for key, value in projection.items()} if projection else None,
        "local_score": local_score,
        "outputs": {"prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None, "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None, "report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["official_selection"], ensure_ascii=False, indent=2), flush=True)
    if local_score is not None: print(json.dumps(local_score, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
