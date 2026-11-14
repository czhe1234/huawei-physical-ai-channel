#!/usr/bin/env python3
"""Run source-aligned E208 PathTokens through the frozen E207 full-H code."""

from __future__ import annotations

import numpy as np

from project_paths import PATHS
from round2.physical_prior.source_aligned import RAW_TOKEN_CACHE
import train_e207_physical_prior_full_channel as implementation


VERSION = "Round2-E208-source-aligned-full-channel"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e208_source_aligned_full_channel"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e208_source_aligned_full_channel"
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
main = implementation.main


if __name__ == "__main__":
    main()
