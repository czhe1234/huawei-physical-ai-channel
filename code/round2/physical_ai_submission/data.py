"""Strict loading and validation of the five official Round 2 inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OfficialSetup:
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


@dataclass(frozen=True)
class OfficialData:
    setup: OfficialSetup
    train_positions: np.ndarray
    test_positions: np.ndarray
    train_sites: np.ndarray
    test_sites: np.ndarray
    site_boundary_y: float
    train_channel_path: Path
    map_path: Path

    def train_channel(self, mmap_mode: str | None = "r") -> np.ndarray:
        value = np.load(self.train_channel_path, mmap_mode=mmap_mode)
        expected = (
            self.setup.train_count,
            self.setup.tx_antennas,
            self.setup.rx_antennas,
            self.setup.subcarriers,
        )
        if value.shape != expected or value.dtype != np.complex64:
            raise ValueError(
                f"invalid training channel: {value.shape}/{value.dtype}, expected "
                f"{expected}/complex64"
            )
        return value


REQUIRED_FILES = (
    "Round2_Train_Channel.npy",
    "Round2_Train_Pos.npy",
    "Round2_Test_Pos.npy",
    "Round2_Map.ply",
    "Round2_Setup.json",
)


def infer_sites(
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    bs_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(bs_positions) != 2:
        raise ValueError(f"expected two base stations, got {len(bs_positions)}")
    sorted_y = np.sort(np.asarray(train_positions, dtype=np.float64)[:, 1])
    split = int(np.argmax(np.diff(sorted_y)))
    boundary = float(0.5 * (sorted_y[split] + sorted_y[split + 1]))
    low_site = int(np.argmin(bs_positions[:, 1]))
    high_site = int(np.argmax(bs_positions[:, 1]))

    def assign(positions: np.ndarray) -> np.ndarray:
        return np.where(
            np.asarray(positions)[:, 1] < boundary, low_site, high_site
        ).astype(np.int8)

    return assign(train_positions), assign(query_positions), boundary


def load_official_data(data_dir: Path) -> OfficialData:
    data_dir = Path(data_dir).resolve()
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing official inputs in {data_dir}: {missing}")
    payload = json.loads((data_dir / "Round2_Setup.json").read_text(encoding="utf-8"))
    setup = OfficialSetup(
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
    train_positions = np.load(data_dir / "Round2_Train_Pos.npy").astype(
        np.float64, copy=False
    )
    test_positions = np.load(data_dir / "Round2_Test_Pos.npy").astype(
        np.float64, copy=False
    )
    if train_positions.shape != (setup.train_count, 3):
        raise ValueError(f"invalid train positions: {train_positions.shape}")
    if test_positions.shape != (setup.test_count, 3):
        raise ValueError(f"invalid test positions: {test_positions.shape}")
    train_sites, test_sites, boundary = infer_sites(
        train_positions, test_positions, setup.bs_positions
    )
    data = OfficialData(
        setup=setup,
        train_positions=train_positions,
        test_positions=test_positions,
        train_sites=train_sites,
        test_sites=test_sites,
        site_boundary_y=boundary,
        train_channel_path=data_dir / "Round2_Train_Channel.npy",
        map_path=data_dir / "Round2_Map.ply",
    )
    data.train_channel(mmap_mode="r")
    return data
