#!/usr/bin/env python3
"""Build E211 full-state physical tokens on all selected GPUs."""

from __future__ import annotations

from project_paths import PATHS
import run_e210_source_port_parallel as runner
from round2.physical_prior import source_port_full_state_tokens as tokens


runner.SCRIPT = PATHS.root / "code" / "round2" / "run_e211_source_port_prior.py"
runner.LOG_DIR = tokens.ANALYSIS_DIR / "logs" / "physical"
runner.RUN_REPORT = tokens.ANALYSIS_DIR / "physical_parallel_run.json"
runner.VERSION = tokens.VERSION
runner.RAW_TOKEN_CACHE = tokens.RAW_TOKEN_CACHE
runner.TOKEN_REPORT = tokens.TOKEN_REPORT
runner.batch_count = tokens.batch_count
runner.complete = tokens.complete
runner.support_path = tokens.support_path


if __name__ == "__main__":
    runner.main()
