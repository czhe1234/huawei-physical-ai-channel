"""Frozen runtime primitives for physical spectral dataset."""

from __future__ import annotations
from pathlib import Path
from project_paths import PATHS

ARTIFACT_DIR = PATHS.artifacts / "round2" / "physical_official_calibration"
TRUTH_PAS = ARTIFACT_DIR / "official_train_pas_f16.npy"
TRUTH_PDP = ARTIFACT_DIR / "official_train_pdp_f16.npy"
TRUTH_POWER = ARTIFACT_DIR / "official_train_point_power_f64.npy"
PHYSICAL_ROOT = PATHS.artifacts / "round2" / "physical_multiseed"
BASE_CONFIG = "primary_d16_base_v4lambda_1m"
SCATTER_CONFIG = "primary_d10_scatter015_v4lambda_1m"
SEEDS = 24


def running_path(config: str, split: str, kind: str, statistic: str) -> Path:
    return (
        PHYSICAL_ROOT
        / config
        / "full"
        / "running"
        / f"{split}_{kind}_{statistic}_n{SEEDS}.npy"
    )
