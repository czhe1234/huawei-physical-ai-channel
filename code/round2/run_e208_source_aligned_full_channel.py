#!/usr/bin/env python3
"""Run E208 prepare, five resumable GPU folds, and fixed500 aggregation."""

from __future__ import annotations

from project_paths import PATHS
import run_e207_physical_prior_full_channel as runner
import train_e208_source_aligned_full_channel as e208


runner.e207 = e208
runner.SCRIPT = (
    PATHS.root / "code" / "round2" / "train_e208_source_aligned_full_channel.py"
)


if __name__ == "__main__":
    runner.main()

