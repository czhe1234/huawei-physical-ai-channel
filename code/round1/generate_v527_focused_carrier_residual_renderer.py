#!/usr/bin/env python3
"""Run v526 with training and deployment both focused on light/medium points."""

from __future__ import annotations

import sys

import generate_v526_full_carrier_residual_renderer as v526
from project_paths import PATHS


VERSION_NUMBER = 527
VERSION = "v527_focused_carrier_residual_renderer"


def configure() -> None:
    v526.VERSION_NUMBER = VERSION_NUMBER
    v526.VERSION = VERSION
    v526.CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
    v526.FOLD_DIR = v526.CACHE_DIR / "fold_results"
    v526.CHECKPOINT_DIR = PATHS.checkpoints / VERSION
    v526.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
    v526.REPORT = v526.REPORT_DIR / f"{VERSION}.json"
    v526.OUTPUT = v526.CACHE_DIR / "v527_Round1_Test_Channel.npy"
    v526.ACTION_CACHE = v526.CACHE_DIR / "v527_path_pdp_actions.npz"
    v526.INFERENCE_BUNDLE = v526.CACHE_DIR / "v527_inference_bundle.pt"
    v526.ARCHIVE = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
    v526.TRAINING_REGIMES = (1, 2)
    v526.ENTRY_SCRIPT = "generate_v527_focused_carrier_residual_renderer.py"


def main() -> None:
    configure()
    if "--seed" not in sys.argv:
        sys.argv.extend(["--seed", "527001"])
    v526.main()


if __name__ == "__main__":
    main()
