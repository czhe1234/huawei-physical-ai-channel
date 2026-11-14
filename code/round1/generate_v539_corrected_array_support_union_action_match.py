#!/usr/bin/env python3
"""Package the validated v539 corrected-array support-union action."""

from __future__ import annotations

import analyze_v539_corrected_array_support_union as v539
import generate_v538_corrected_array_medium_sbi_action_match as deployment
from project_paths import PATHS


def main() -> None:
    deployment.VERSION_NUMBER = 539
    deployment.VERSION = "v539_corrected_array_support_union_action_match"
    deployment.ANALYSIS = v539
    deployment.SOURCE_COMBINATION = "max"
    deployment.ENTRY_SCRIPT = "generate_v539_corrected_array_support_union_action_match.py"
    deployment.ANALYSIS_SCRIPT = "analyze_v539_corrected_array_support_union.py"
    deployment.PRIMARY_HYPOTHESIS = (
        "A three-seed corrected-array support union preserves stochastic path births "
        "and is deployed only for medium points while retaining v533 elsewhere."
    )
    deployment.CACHE_DIR = PATHS.cache_dir(539)
    deployment.REPORT_DIR = PATHS.version_report_dir(539)
    deployment.OUTPUT = deployment.CACHE_DIR / "v539_Round1_Test_Channel.npy"
    deployment.TARGET = deployment.CACHE_DIR / "v539_corrected_array_union_joint_target.npy"
    deployment.ACTIONS = deployment.CACHE_DIR / "v539_corrected_array_union_actions.npz"
    deployment.REPORT = deployment.REPORT_DIR / f"{deployment.VERSION}.json"
    deployment.ZIP_OUT = PATHS.round1_versions / f"{deployment.VERSION}_candidate_pending.zip"
    deployment.ANALYSIS_REPORT = (
        PATHS.version_report_dir(539) / "v539_corrected_array_support_union.json"
    )
    deployment.main()


if __name__ == "__main__":
    main()
