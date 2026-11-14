#!/usr/bin/env python3
"""Frozen carrier-robust composition of v284 and the v351 expert residual."""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 352
VERSION = "v352_carrier_robust_occluded_expert"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v352_carrier_robust_oof.npz"
TARGETS_OUT = OUT_DIR / "v352_carrier_robust_targets.npz"
OUTPUT = OUT_DIR / "v352_Round1_Test_Channel.npy"
V351_OOF = PATHS.cache_dir(351) / "v351_occluded_expert_oof.npz"
V351_TARGETS = PATHS.cache_dir(351) / "v351_occluded_expert_targets.npz"
V285_SAMPLES = PATHS.cache_dir(285) / "v285_v239_crossfit_joint_current.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
V284_BASE = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"

FOLD_COUNT = 5
MIN_MEAN_DELTA = 0.01
MIN_FOLD_WINS = 4
LOCAL_GATE = 0.645604


def main() -> None:
    for path in (V351_OOF, V351_TARGETS, V285_SAMPLES, TARGET_CACHE, V284_BASE):
        if not path.is_file():
            raise FileNotFoundError(path)

    oof = np.load(V351_OOF)
    sample = np.load(V285_SAMPLES)
    target_data = np.load(TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))
    query_ids = sample["query_ids"].astype(np.int64)
    assigned_fold = sample["assigned_fold"].astype(np.int16)
    independent_carrier = sample["reconstructed_anchor_sh"].astype(np.float32)

    sample_lookup = np.full(len(truth_sh), -1, dtype=np.int64)
    sample_lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    expert_rows = sample_lookup[oof["ids"].astype(np.int64)]
    if np.any(expert_rows < 0):
        raise RuntimeError("v351 OOF ids are not present in v285 samples")
    residual = oof["prediction"].astype(np.float32) - oof["base"].astype(np.float32)
    carrier_prediction = independent_carrier.copy()
    carrier_prediction[expert_rows] = v279.normalize(
        independent_carrier[expert_rows] + residual
    )

    fold_rows: list[dict[str, object]] = []
    for fold in range(FOLD_COUNT):
        rows = np.flatnonzero(assigned_fold == fold)
        ids = query_ids[rows]
        baseline = v279.metrics(
            independent_carrier[rows], truth_sh[ids], truth_d[ids]
        )
        prediction = v279.metrics(
            carrier_prediction[rows], truth_sh[ids], truth_d[ids]
        )
        delta = {
            key: float(prediction[key] - baseline[key])
            for key in ("sh", "d", "weighted")
        }
        fold_rows.append(
            {
                "fold": fold,
                "count": int(len(rows)),
                "expert_count": int(np.sum(np.isin(rows, expert_rows))),
                "baseline": baseline,
                "prediction": prediction,
                "delta": delta,
            }
        )
        print(f"v352 fold={fold} delta={delta['weighted']:+.6f}", flush=True)

    mean_delta = {
        key: float(np.mean([row["delta"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    wins = int(sum(row["delta"]["weighted"] > 0.0 for row in fold_rows))
    official_selected = bool(
        mean_delta["weighted"] >= MIN_MEAN_DELTA and wins >= MIN_FOLD_WINS
    )
    np.savez_compressed(
        OOF_OUT,
        ids=query_ids,
        carrier=independent_carrier,
        prediction=carrier_prediction,
        fold=assigned_fold,
        expert_rows=expert_rows,
    )

    local_score = None
    projection = None
    test_diagnostics = None
    if official_selected:
        v351_target = np.load(V351_TARGETS)
        expert_residual = (
            v351_target["target_sh"].astype(np.float32)
            - v351_target["current_sh"].astype(np.float32)
        )
        v284_h = np.load(V284_BASE, mmap_mode="r")
        v284_sh, v284_d = v274.extract_current_marginals(V284_BASE)
        target_sh = v279.normalize(v284_sh + expert_residual)
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            carrier_sh=v284_sh,
            carrier_d=v284_d,
            expert_residual=expert_residual,
            target_sh=target_sh,
            target_d=target_d,
        )
        old_base = v247.BASE
        try:
            v247.BASE = V284_BASE
            projection = v247.project(target_sh, target_d, OUTPUT)
        finally:
            v247.BASE = old_base
        local_score = v208.score_prediction(OUTPUT)
        test_diagnostics = {
            "carrier_residual_l1": summarize(
                np.sum(np.abs(expert_residual), axis=(1, 2))
            ),
            "target_vs_carrier_sh_l1": summarize(
                np.sum(np.abs(target_sh - v284_sh), axis=(1, 2))
            ),
            "target_vs_carrier_d_l1": summarize(
                np.sum(np.abs(target_d - v284_d), axis=1)
            ),
            "source_shape": list(v284_h.shape),
            "source_dtype": str(v284_h.dtype),
        }

    report = {
        "method": VERSION,
        "status": "invalidated_crossfold_current_leakage",
        "round": 5,
        "validity_audit": {
            "strict_oof_usable": False,
            "invalidated_by": "v353_crossfold_current_leakage_audit",
            "reason": "the frozen v351 residual was selected from leaked OOF currents",
            "numerical_selection_before_audit": official_selected,
        },
        "primary_hypothesis": (
            "The v351 medium/heavy expert learned a carrier-independent propagation residual; "
            "the same frozen signed support correction should improve an independent OOF KNN "
            "carrier and can then be composed without retuning on the strict v284 carrier."
        ),
        "compliance": {
            "uses": [
                "official-only v351 OOF expert residual",
                "independent official-H KNN joint OOF carrier",
                "strict official-only v284 test carrier",
            ],
            "does_not_use": [
                "v207/v212/synthetic teacher or labels",
                "restored scene/material/RT channel",
                "local truth for residual strength or composition selection",
                "online labels",
            ],
            "local_truth_usage": (
                "one frozen diagnostic after carrier-robust official OOF selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "composition": "new_carrier + frozen_v351_signed_residual",
            "additional_alpha": 1.0,
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "parameter_search": "none",
        },
        "fold_rows": fold_rows,
        "official_selection": {
            "mean_delta": mean_delta,
            "fold_wins": wins,
            "selected": False,
            "selected_numerically_before_audit": official_selected,
        },
        "test_diagnostics": test_diagnostics,
        "projection": (
            {key: summarize(value) for key, value in projection.items()}
            if projection is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(local_score["score_traceback_leaked"] >= LOCAL_GATE),
                "gap": float(local_score["score_traceback_leaked"] - LOCAL_GATE),
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["official_selection"], ensure_ascii=False, indent=2), flush=True)
    if local_score is not None:
        print(json.dumps(local_score, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
