#!/usr/bin/env python3
"""Audit outer-label exposure in the reused v285 cross-fit current."""

from __future__ import annotations

import json

import numpy as np

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v281_v239_crossfit_pdp_current as v281
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 353
VERSION = "v353_crossfold_current_leakage_audit"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
SAMPLES = PATHS.cache_dir(285) / "v285_v239_crossfit_joint_current.npz"
V281_CACHE = PATHS.cache_dir(281) / "v281_v239_crossfit_pdp_current.npz"


def main() -> None:
    for path in (SAMPLES, V281_CACHE, v238.GEOM_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    sample = np.load(SAMPLES)
    source = np.load(V281_CACHE)
    query_ids = sample["query_ids"].astype(np.int64)
    assigned_fold = sample["assigned_fold"].astype(np.int16)
    all_source_ids = source["ids"].astype(np.int64)
    all_source_folds = source["fold"].astype(np.int16)

    _clean, _raw, _power, valid_delay = v215.load_targets()
    _support, valid_support, _log_power = v238.build_raw_support()
    valid_ids = np.flatnonzero(valid_delay & valid_support).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    blocked = np.load(v238.GEOM_CACHE)["train_blocked"].astype(np.float32)

    rows: list[dict[str, object]] = []
    all_neighbor_weight: list[np.ndarray] = []
    all_path_exposure: list[np.ndarray] = []
    for outer_fold in range(5):
        outer_ids = query_ids[assigned_fold == outer_fold]
        outer_set = set(int(value) for value in outer_ids)
        train_rows = np.flatnonzero(assigned_fold != outer_fold)
        fold_neighbor_weights: list[np.ndarray] = []
        fold_path_exposure: list[np.ndarray] = []
        source_fold_rows: list[dict[str, object]] = []
        for source_fold in range(5):
            rows_in_source = train_rows[assigned_fold[train_rows] == source_fold]
            if len(rows_in_source) == 0:
                continue
            source_val_ids = np.unique(
                all_source_ids[all_source_folds == source_fold]
            )
            bank_ids = np.setdiff1d(valid_ids, source_val_ids)
            exposed_outer = np.asarray(
                [value for value in outer_ids if int(value) in set(bank_ids.tolist())],
                dtype=np.int64,
            )
            query = query_ids[rows_in_source]
            indices, weights = v218.neighbor_index_weights(
                train_pos[query],
                blocked[query],
                train_pos[bank_ids],
                blocked[bank_ids],
                query_ids=query,
                bank_ids=bank_ids,
                k=v238.ANCHOR_K,
            )
            neighbor_ids = bank_ids[indices]
            leaked = np.isin(neighbor_ids, outer_ids)
            leaked_weight = np.sum(weights * leaked, axis=1).astype(np.float32)
            path_exposure = np.full(
                len(query), len(exposed_outer) / max(len(outer_ids), 1), dtype=np.float32
            )
            fold_neighbor_weights.append(leaked_weight)
            fold_path_exposure.append(path_exposure)
            source_fold_rows.append(
                {
                    "source_fold": source_fold,
                    "training_rows": int(len(query)),
                    "outer_ids": int(len(outer_ids)),
                    "outer_ids_in_path_model_bank": int(len(exposed_outer)),
                    "path_model_outer_label_fraction": float(
                        len(exposed_outer) / max(len(outer_ids), 1)
                    ),
                    "neighbor_leak_weight": summarize(leaked_weight),
                    "training_rows_with_outer_h_neighbor": float(
                        np.mean(leaked_weight > 0.0)
                    ),
                }
            )
        neighbor_weight = np.concatenate(fold_neighbor_weights)
        path_exposure = np.concatenate(fold_path_exposure)
        all_neighbor_weight.append(neighbor_weight)
        all_path_exposure.append(path_exposure)
        rows.append(
            {
                "outer_fold": outer_fold,
                "outer_count": int(len(outer_ids)),
                "training_count": int(len(train_rows)),
                "path_model_outer_label_fraction": summarize(path_exposure),
                "neighbor_leak_weight": summarize(neighbor_weight),
                "training_rows_with_outer_h_neighbor": float(
                    np.mean(neighbor_weight > 0.0)
                ),
                "source_folds": source_fold_rows,
            }
        )
        print(
            f"v353 outer={outer_fold} path_exposure={path_exposure.mean():.3f} "
            f"neighbor_rows={np.mean(neighbor_weight > 0.0):.3f} "
            f"neighbor_weight={neighbor_weight.mean():.4f}",
            flush=True,
        )

    neighbor_weight = np.concatenate(all_neighbor_weight)
    path_exposure = np.concatenate(all_path_exposure)
    invalid = bool(np.any(neighbor_weight > 0.0) or np.any(path_exposure > 0.0))
    report = {
        "method": VERSION,
        "status": "crossfold_leakage_confirmed" if invalid else "no_leakage_found",
        "purpose": (
            "Determine whether v350-v352 reused currents are valid outer-fold features."
        ),
        "audit_logic": (
            "For each new outer fold, reconstruct every training row's original v281 source "
            "bank. Count outer labels present in the source path-model fit bank and the "
            "actual 72-NN support weight assigned to outer H."
        ),
        "overall": {
            "path_model_outer_label_fraction": summarize(path_exposure),
            "neighbor_leak_weight": summarize(neighbor_weight),
            "training_rows_with_outer_h_neighbor": float(
                np.mean(neighbor_weight > 0.0)
            ),
            "invalidates_v350_v351_v352_strict_oof": invalid,
        },
        "fold_rows": rows,
        "conclusion": (
            "The reused per-point current is not nested with respect to the new outer fold; "
            "v350-v352 OOF values cannot be used as strict generalization evidence."
            if invalid
            else "No direct outer-label exposure was detected."
        ),
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
