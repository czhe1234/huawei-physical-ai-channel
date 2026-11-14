"""Frozen runtime primitives for path token oracle."""

from __future__ import annotations
import numpy as np

N_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30000.0
FREQUENCIES = (
    np.arange(-N_SUBCARRIERS // 2, N_SUBCARRIERS // 2, dtype=np.float32)
    * SUBCARRIER_SPACING
)
