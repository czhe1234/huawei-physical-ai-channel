"""Round 2 input loading and deterministic two-site assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project_paths import PATHS


@dataclass(frozen=True)
class Round2Setup:
    train_count: int
    test_count: int
    tx_antennas: int
    rx_antennas: int
    subcarriers: int
    tx_horizontal: int
    tx_vertical: int
    tx_polarizations: int
    bs_positions: np.ndarray
    metric_weights: np.ndarray


def load_setup(path: Path | None = None) -> Round2Setup:
    source = path or (PATHS.round2 / "Round2_Setup.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return Round2Setup(
        train_count=int(payload["P_Train"]),
        test_count=int(payload["P_Test"]),
        tx_antennas=int(payload["M"]),
        rx_antennas=int(payload["N"]),
        subcarriers=int(payload["S"]),
        tx_horizontal=int(payload["M_H"]),
        tx_vertical=int(payload["M_V"]),
        tx_polarizations=int(payload["M_P"]),
        bs_positions=np.asarray(payload["X"], dtype=np.float64),
        metric_weights=np.asarray(payload["w"], dtype=np.float64),
    )


def infer_site_ids(
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    bs_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover the two official sites from the largest empty train-Y interval."""

    if len(bs_positions) != 2:
        raise ValueError(f"Round 2 expects two sites, got {len(bs_positions)}")
    sorted_y = np.sort(np.asarray(train_positions, dtype=np.float64)[:, 1])
    gaps = np.diff(sorted_y)
    split = int(np.argmax(gaps))
    boundary = float(0.5 * (sorted_y[split] + sorted_y[split + 1]))

    low_site = int(np.argmin(bs_positions[:, 1]))
    high_site = int(np.argmax(bs_positions[:, 1]))

    def assign(positions: np.ndarray) -> np.ndarray:
        low = np.asarray(positions)[:, 1] < boundary
        return np.where(low, low_site, high_site).astype(np.int8)

    return assign(train_positions), assign(query_positions), boundary


def load_positions() -> tuple[Round2Setup, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    setup = load_setup()
    train_positions = np.load(PATHS.round2 / "Round2_Train_Pos.npy").astype(np.float64)
    test_positions = np.load(PATHS.round2 / "Round2_Test_Pos.npy").astype(np.float64)
    train_site, test_site, _ = infer_site_ids(
        train_positions, test_positions, setup.bs_positions
    )
    return setup, train_positions, test_positions, train_site, test_site
