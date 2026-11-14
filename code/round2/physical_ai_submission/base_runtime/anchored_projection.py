"""Frozen runtime primitives for anchored projection."""

from __future__ import annotations
from project_paths import PATHS

ARTIFACT_DIR = PATHS.artifacts / "round2" / "donor_anchored_neural_rbf"
TARGET_PAS = ARTIFACT_DIR / "oof_pas_neural_f16.npy"
TARGET_PDP = ARTIFACT_DIR / "oof_pdp_neural_f16.npy"
PAS_GLOBAL_SHRINK = 0.1
PDP_GLOBAL_SHRINK = 0.55
