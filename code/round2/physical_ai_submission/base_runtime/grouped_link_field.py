"""Frozen runtime primitives for grouped link field."""

from __future__ import annotations
from .anchored_donor_field import (
    ARTIFACT_DIR,
    DESCRIPTORS,
    FOLD_CACHE,
    GEOMETRY,
    OOF_META,
    PAS_SIZE,
    AnchoredWeightCorrector,
    Config,
    build_train_contexts,
    forward_rows,
    power_cosine,
    set_seed,
    standardize_geometry,
    tensor_context,
    unit_amplitude,
    validation_context,
)

GROUPED_PAS = ARTIFACT_DIR / "grouped_pas_amplitude_f16.npy"
GROUPED_PDP = ARTIFACT_DIR / "grouped_pdp_amplitude_f16.npy"
