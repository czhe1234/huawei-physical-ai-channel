#!/usr/bin/env python3
"""v268: frozen v267 patch-delay retrieval deployed as analysis-only target."""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v266_learned_scatterer_joint_projection as v266
import analyze_v267_patch_conditioned_neighbor_retrieval as v267
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 268
VERSION = "v268_patch_delay_retrieval_deployment"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v268_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v268_patch_delay_targets.npz"
ACTION = OUT_DIR / "v268_joint_projection_actions.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
TEST_CANDIDATES = PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

POWER = 2.0
PATCH_LAMBDA = 0.0
DELAY_LAMBDA = 9.0
K = 12
ALPHA = 0.70
LOCAL_GATE = 0.645604


def main() -> None:
    for path in (TARGET_CACHE, TRAIN_CANDIDATES, TEST_CANDIDATES, BASE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    target_data = np.load(TARGET_CACHE)
    train_sh = v267.normalize_l1(target_data["sh"].astype(np.float32))
    train_d = v267.normalize_l1(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_patch, train_patch_delay = v267.build_patch_signatures(
        dict(np.load(TRAIN_CANDIDATES))
    )
    test_patch, test_patch_delay = v267.build_patch_signatures(
        dict(np.load(TEST_CANDIDATES))
    )
    patch_sim = (test_patch @ train_patch[valid_ids].T).toarray().astype(np.float32)
    delay_sim = (test_patch_delay @ train_patch_delay[valid_ids].T).toarray().astype(
        np.float32
    )
    zero = np.zeros_like(delay_sim)
    base_indices, base_weights = v267.kernel_neighbors(
        test_pos[:, :2],
        train_pos[valid_ids, :2],
        zero,
        zero,
        2.0,
        0.0,
        0.0,
        16,
    )
    patch_indices, patch_weights = v267.kernel_neighbors(
        test_pos[:, :2],
        train_pos[valid_ids, :2],
        patch_sim,
        delay_sim,
        POWER,
        PATCH_LAMBDA,
        DELAY_LAMBDA,
        K,
    )
    base_sh, base_d = v267.exact_prediction(
        train_sh[valid_ids], train_d[valid_ids], base_indices, base_weights
    )
    patch_sh, patch_d = v267.exact_prediction(
        train_sh[valid_ids], train_d[valid_ids], patch_indices, patch_weights
    )
    retrieved_sh = v267.normalize_l1((1.0 - ALPHA) * base_sh + ALPHA * patch_sh)
    retrieved_d = v267.normalize_l1((1.0 - ALPHA) * base_d + ALPHA * patch_d)

    source = np.load(BASE, mmap_mode="r")
    source_array = np.asarray(source, dtype=np.complex64)
    current_sh = v266.current_sh(source_array)
    _unused_sh, current_d = v247.extract_current(source_array)
    np.savez_compressed(
        TARGETS,
        sh=retrieved_sh,
        d=retrieved_d,
        current_sh=current_sh,
        current_d=current_d,
        base_sh=base_sh,
        base_d=base_d,
        patch_sh=patch_sh,
        patch_d=patch_d,
        neighbor_indices=valid_ids[patch_indices],
        neighbor_weights=patch_weights,
    )
    actions = v266.alternating_project(BASE, OUTPUT, retrieved_sh, retrieved_d)
    np.savez_compressed(ACTION, **actions)
    local_score = v208.score_prediction(OUTPUT)
    selected_patch = np.take_along_axis(patch_sim, patch_indices, axis=1)
    selected_delay = np.take_along_axis(delay_sim, patch_indices, axis=1)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The nested-holdout-positive v267 patch-delay kernel transfers to public test "
            "points and improves the strict local diagnostic when projected jointly."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP bank",
                "official train/test coordinates",
                "official point-cloud patch-delay signatures",
                "frozen v267 nested-holdout parameters",
                "strict v239 carrier",
            ],
            "does_not_use": [
                "Sionna or synthetic labels for generation",
                "teacher output",
                "local replay truth for parameter selection",
                "hidden labels",
                "raw RT H",
            ],
            "note": "Non-neural diagnostic only; never a final submission candidate.",
        },
        "parameters": {
            "power": POWER,
            "patch_lambda": PATCH_LAMBDA,
            "delay_lambda": DELAY_LAMBDA,
            "k": K,
            "alpha": ALPHA,
        },
        "diagnostics": {
            "target_sh_l1_to_current": summarize(
                np.sum(np.abs(retrieved_sh - current_sh), axis=(1, 2))
            ),
            "target_d_l1_to_current": summarize(
                np.sum(np.abs(retrieved_d - current_d), axis=1)
            ),
            "selected_patch_cos": summarize(selected_patch),
            "selected_patch_delay_cos": summarize(selected_delay),
            "projection": {key: summarize(value) for key, value in actions.items()},
        },
        "local_gate": {
            "required": LOCAL_GATE,
            "score": local_score,
            "passed": bool(float(local_score["score_traceback_leaked"]) > LOCAL_GATE),
            "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS),
            "actions": PATHS.relative(ACTION),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
