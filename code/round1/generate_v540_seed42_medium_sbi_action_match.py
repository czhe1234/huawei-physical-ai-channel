#!/usr/bin/env python3
"""Package the validated v540 seed42 corrected-array medium-only action."""

from __future__ import annotations

import analyze_v540_seed42_medium_only as v540
import generate_v538_corrected_array_medium_sbi_action_match as deployment
from project_paths import PATHS


def main() -> None:
    deployment.VERSION_NUMBER = 540
    deployment.VERSION = "v540_seed42_corrected_array_medium_sbi_action_match"
    deployment.ANALYSIS = v540
    deployment.SOURCE_COMBINATION = "mean"
    deployment.ENTRY_SCRIPT = "generate_v540_seed42_medium_sbi_action_match.py"
    deployment.ANALYSIS_SCRIPT = "analyze_v540_seed42_medium_only.py"
    deployment.PRIMARY_HYPOTHESIS = (
        "The preselected seed42 corrected-array physical source improves medium "
        "joint-S-H support, while exact retention of v533 outside medium prevents "
        "the observed light-regime regression."
    )
    deployment.CACHE_DIR = PATHS.cache_dir(540)
    deployment.REPORT_DIR = PATHS.version_report_dir(540)
    deployment.OUTPUT = deployment.CACHE_DIR / "v540_Round1_Test_Channel.npy"
    deployment.TARGET = deployment.CACHE_DIR / "v540_seed42_medium_joint_target.npy"
    deployment.ACTIONS = deployment.CACHE_DIR / "v540_seed42_medium_actions.npz"
    deployment.REPORT = deployment.REPORT_DIR / f"{deployment.VERSION}.json"
    deployment.ZIP_OUT = PATHS.round1_versions / f"{deployment.VERSION}_candidate_pending.zip"
    deployment.ANALYSIS_REPORT = (
        PATHS.version_report_dir(540) / "v540_seed42_corrected_array_medium_only.json"
    )
    deployment.main()


if __name__ == "__main__":
    main()
