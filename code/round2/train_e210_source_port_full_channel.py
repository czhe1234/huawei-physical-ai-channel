#!/usr/bin/env python3
"""Run E210 source-port PathTokens through the frozen E208 full-H code."""

from __future__ import annotations

import json
import sys

import numpy as np

from project_paths import PATHS
from round2.physical_prior.source_port_tokens import RAW_TOKEN_CACHE, VERSION
import train_e207_physical_prior_full_channel as implementation


ARTIFACT_DIR = PATHS.artifacts / "round2" / "e210_source_port_full_channel"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e210_source_port_full_channel"
COMMON_DIR = ARTIFACT_DIR / "common"
COMMON_STATE = COMMON_DIR / "state.json"
TARGET_PAS = COMMON_DIR / "official_pas_target_f32.npy"
TARGET_PDP = COMMON_DIR / "official_pdp_target_f32.npy"
FULL_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
FULL_REPORT = ANALYSIS_DIR / "fixed500_report.json"


def configure() -> None:
    implementation.VERSION = VERSION
    implementation.ARTIFACT_DIR = ARTIFACT_DIR
    implementation.ANALYSIS_DIR = ANALYSIS_DIR
    implementation.COMMON_DIR = COMMON_DIR
    implementation.COMMON_STATE = COMMON_STATE
    implementation.TARGET_PAS = TARGET_PAS
    implementation.TARGET_PDP = TARGET_PDP
    implementation.FULL_CHANNEL = FULL_CHANNEL
    implementation.FULL_REPORT = FULL_REPORT
    implementation.RAW_TOKEN_CACHE = RAW_TOKEN_CACHE
    with np.load(RAW_TOKEN_CACHE, allow_pickle=False) as cache:
        fallback = np.flatnonzero(
            np.any(cache["mask"] & (cache["source"] < 0), axis=1)
        )
    implementation.e205.FORCE_BASE_GLOBAL_IDS = frozenset(
        int(value) for value in fallback
    )


configure()

fold_report_path = implementation.fold_report_path
fold_channel_path = implementation.fold_channel_path
e205 = implementation.e205


def main() -> None:
    implementation.main()
    if "--stage" not in sys.argv:
        return
    stage_index = sys.argv.index("--stage") + 1
    if stage_index >= len(sys.argv) or sys.argv[stage_index] != "aggregate":
        return
    report = json.loads(FULL_REPORT.read_text(encoding="utf-8"))
    lineage = report.setdefault("data_lineage", {})
    lineage["sionna_package_used"] = False
    lineage["historical_sionna_cache_used"] = False
    lineage["vendored_sionna_rt_source_used"] = True
    lineage["mitsuba_drjit_pathsolver_used"] = True
    report["failure_conclusion_if_failed"] = (
        "Exact E185 source-port PathTokens still do not convert to V12-level "
        "complete-H score through the frozen mass/port state interface."
    )
    e205.atomic_json(FULL_REPORT, report)


if __name__ == "__main__":
    main()
