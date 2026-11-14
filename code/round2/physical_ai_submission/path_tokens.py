"""Aggregate raw propagation shards into the fixed 128-token model interface."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .io_utils import atomic_json, atomic_npy


@dataclass
class PathRecord:
    tau: float
    angles: np.ndarray
    coefficient: np.ndarray
    interactions: np.ndarray
    objects: np.ndarray
    primitives: np.ndarray
    vertices: np.ndarray
    interaction_count: int
    energy: float
    energy_fraction: float
    source: int
    seed: int
    presence: int = 1


TOKEN_FIELDS = (
    "coefficient",
    "tau",
    "mask",
    "source",
    "seed",
    "angles",
    "energy",
    "energy_fraction",
    "presence",
    "interaction_count",
    "interactions",
    "objects",
    "primitives",
    "vertices",
    "rows",
)

RECORD_FIELDS = (
    "point_ids",
    "offsets",
    "relative_energy",
    "interaction_count",
    "tau",
    "angles",
    "full_complex_coefficient",
    "interactions",
    "objects",
    "primitives",
    "vertices",
    "absolute_energy",
    "seed",
)


def cache_dir(paths: RuntimePaths, split: str) -> Path:
    return paths.token_dir / split


def _periodic_coordinate(
    tau: float,
    theta: float,
    phi: float,
    site: int,
    source: int,
    config: PipelineConfig,
) -> tuple[float, float, float]:
    yaw_deg = (-44.0, 73.5)[site]
    tilt_deg = (-11.5, -10.5)[site]
    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)
    horizontal_cycles = 0.5 * math.sin(theta) * math.sin(phi - yaw)
    vertical_cycles = 4.0 * (
        math.sin(tilt) * math.sin(theta) * math.cos(phi - yaw)
        - math.cos(tilt) * math.cos(theta)
    )
    delay = (tau * config.subcarrier_spacing_hz * config.subcarriers) % 192.0
    return delay, (16.0 * horizontal_cycles) % 16.0, (8.0 * vertical_cycles) % 8.0


def _base_identity(
    record: PathRecord, site: int, config: PipelineConfig
) -> tuple[Any, ...]:
    delay, horizontal, vertical = _periodic_coordinate(
        record.tau,
        float(record.angles[0]),
        float(record.angles[1]),
        site,
        record.source,
        config,
    )
    depth = record.interaction_count
    sequence = tuple(
        (
            int(record.interactions[index]),
            int(record.objects[index]),
            int(record.primitives[index]),
        )
        for index in range(depth)
    )
    return (
        int(round(delay * 50.0)),
        int(round(horizontal * 50.0)),
        int(round(vertical * 50.0)),
        sequence,
    )


def _scatter_identity(
    record: PathRecord, site: int, config: PipelineConfig
) -> tuple[int, ...]:
    delay, horizontal, vertical = _periodic_coordinate(
        record.tau,
        float(record.angles[0]),
        float(record.angles[1]),
        site,
        record.source,
        config,
    )
    shaped = record.coefficient.reshape(2, 16, 8, 4)
    port_power = np.sum(np.abs(shaped) ** 2, axis=(1, 2))
    return (
        int(round(delay * 4.0)),
        int(round(horizontal * 4.0)),
        int(round(vertical * 4.0)),
        min(record.interaction_count, 15),
        int(np.argmax(port_power.reshape(-1))),
    )


def _score_align_coefficient(value: np.ndarray) -> np.ndarray:
    ramp = np.exp(2j * np.pi * 2.0 * np.arange(8, dtype=np.float32) / 8.0).astype(
        np.complex64
    )
    shaped = np.asarray(value, dtype=np.complex64).reshape(2, 16, 8, 4)
    return (shaped * ramp[None, None, :, None]).reshape(256, 4)


def _records_for_point(
    loaded: Mapping[str, np.ndarray],
    local_row: int,
    per_seed: int,
    source: int,
) -> list[PathRecord]:
    start = int(loaded["offsets"][local_row])
    end = int(loaded["offsets"][local_row + 1])
    if end <= start:
        return []
    fraction = np.asarray(loaded["relative_energy"][start:end], dtype=np.float64)
    count = min(per_seed, len(fraction))
    chosen = np.argpartition(fraction, len(fraction) - count)[-count:]
    indices = np.sort(start + chosen.astype(np.int64))
    output: list[PathRecord] = []
    for index in indices:
        depth = int(loaded["interaction_count"][index])
        output.append(
            PathRecord(
                tau=float(loaded["tau"][index]),
                angles=np.array(loaded["angles"][index], dtype=np.float32, copy=True),
                coefficient=_score_align_coefficient(
                    loaded["full_complex_coefficient"][index]
                ),
                interactions=np.array(
                    loaded["interactions"][index], dtype=np.uint8, copy=True
                ),
                objects=np.array(loaded["objects"][index], dtype=np.uint32, copy=True),
                primitives=np.array(
                    loaded["primitives"][index], dtype=np.uint32, copy=True
                ),
                vertices=np.array(
                    loaded["vertices"][index], dtype=np.float32, copy=True
                ),
                interaction_count=depth,
                energy=float(loaded["absolute_energy"][index]),
                energy_fraction=float(loaded["relative_energy"][index]),
                source=source,
                seed=int(loaded["seed"][index]),
            )
        )
    return output


def _iter_seed_shards(
    paths: RuntimePaths,
    split: str,
    configuration: str,
    seed: int,
    site: int,
) -> list[Path]:
    root = (
        paths.propagation_dir
        / split
        / configuration
        / f"seed_{seed:04d}"
        / f"site_{site}"
    )
    shards = sorted(root.glob("batch_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no propagation shards in {root}")
    return shards


def _empty_arrays(count: int, config: PipelineConfig) -> dict[str, np.ndarray]:
    shape = (count, config.paths_per_point)
    depth = config.interaction_steps
    return {
        "coefficient": np.zeros(shape + (256, 4), dtype=np.complex64),
        "tau": np.zeros(shape, dtype=np.float32),
        "mask": np.zeros(shape, dtype=bool),
        "source": np.full(shape, -1, dtype=np.int8),
        "seed": np.full(shape, -1, dtype=np.int32),
        "angles": np.zeros(shape + (4,), dtype=np.float32),
        "energy": np.zeros(shape, dtype=np.float32),
        "energy_fraction": np.zeros(shape, dtype=np.float32),
        "presence": np.zeros(shape, dtype=np.uint8),
        "interaction_count": np.zeros(shape, dtype=np.uint8),
        "interactions": np.zeros(shape + (depth,), dtype=np.uint8),
        "objects": np.zeros(shape + (depth,), dtype=np.uint32),
        "primitives": np.zeros(shape + (depth,), dtype=np.uint32),
        "vertices": np.zeros(shape + (depth, 3), dtype=np.float32),
        "rows": np.arange(count, dtype=np.int64),
    }


def _write_record(
    arrays: dict[str, np.ndarray], row: int, slot: int, record: PathRecord
) -> None:
    arrays["coefficient"][row, slot] = record.coefficient
    arrays["tau"][row, slot] = record.tau
    arrays["mask"][row, slot] = True
    arrays["source"][row, slot] = record.source
    arrays["seed"][row, slot] = record.seed
    arrays["angles"][row, slot] = record.angles
    arrays["energy"][row, slot] = record.energy
    arrays["energy_fraction"][row, slot] = record.energy_fraction
    arrays["presence"][row, slot] = min(record.presence, 255)
    depth = min(record.interaction_count, arrays["interactions"].shape[-1])
    arrays["interaction_count"][row, slot] = depth
    arrays["interactions"][row, slot, :depth] = record.interactions[:depth]
    arrays["objects"][row, slot, :depth] = record.objects[:depth]
    arrays["primitives"][row, slot, :depth] = record.primitives[:depth]
    arrays["vertices"][row, slot, :depth] = record.vertices[:depth]


def build_path_tokens(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    output_dir = cache_dir(paths, split)
    output_files = {name: output_dir / f"{name}.npy" for name in TOKEN_FIELDS}
    if not force and all(path.is_file() for path in output_files.values()):
        mask = np.load(output_files["mask"], mmap_mode="r")
        return {
            "stage": "path_tokens",
            "status": "reused",
            "split": split,
            "shape": list(mask.shape),
            "valid_paths_mean": float(np.mean(mask.sum(axis=1))),
        }

    data = load_official_data(paths.data_dir)
    positions = data.train_positions if split == "train" else data.test_positions
    sites = data.train_sites if split == "train" else data.test_sites
    count = len(positions)
    base_bank: list[dict[tuple[Any, ...], PathRecord]] = [dict() for _ in range(count)]
    scatter_bank: list[list[PathRecord]] = [[] for _ in range(count)]

    def consume_site(configuration: str, site: int) -> None:
        for seed in config.seeds:
            source = 0 if configuration == "base" else 1
            per_seed = (
                config.base_paths_per_seed
                if source == 0
                else config.scatter_paths_per_seed
            )
            for shard in _iter_seed_shards(paths, split, configuration, seed, site):
                with np.load(shard, allow_pickle=False) as loaded:
                    fields = {name: np.asarray(loaded[name]) for name in RECORD_FIELDS}
                for local_row, point_id_value in enumerate(fields["point_ids"]):
                    point_id = int(point_id_value)
                    records = _records_for_point(
                        fields, local_row, per_seed, source=source
                    )
                    if source == 0:
                        bank = base_bank[point_id]
                        for record in records:
                            key = _base_identity(record, site, config)
                            previous = bank.get(key)
                            if previous is None:
                                bank[key] = record
                            else:
                                presence = previous.presence + 1
                                if record.energy_fraction > previous.energy_fraction:
                                    record.presence = presence
                                    bank[key] = record
                                else:
                                    previous.presence = presence
                    else:
                        local_best: dict[tuple[int, ...], PathRecord] = {}
                        for record in records:
                            key = _scatter_identity(record, site, config)
                            previous = local_best.get(key)
                            if (
                                previous is None
                                or record.energy_fraction > previous.energy_fraction
                            ):
                                local_best[key] = record
                        bank = scatter_bank[point_id]
                        bank.extend(local_best.values())
                        if len(bank) > config.scatter_pool:
                            bank.sort(
                                key=lambda value: value.energy_fraction, reverse=True
                            )
                            del bank[config.scatter_pool :]

    jobs = [
        (configuration, site)
        for configuration in ("base", "scatter")
        for site in range(len(data.setup.bs_positions))
    ]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(consume_site, *job) for job in jobs]
        for future in futures:
            future.result()

    arrays = _empty_arrays(count, config)
    for row in range(count):
        base_records = list(base_bank[row].values())
        base_records.sort(
            key=lambda value: value.energy_fraction
            * (1.0 + 0.05 * (value.presence - 1)),
            reverse=True,
        )
        take = min(config.base_reserve, config.paths_per_point, len(base_records))
        selected = base_records[:take]
        remaining = config.paths_per_point - len(selected)
        if remaining:
            scatter_records = sorted(
                scatter_bank[row], key=lambda value: value.energy_fraction, reverse=True
            )
            selected.extend(scatter_records[:remaining])
        if len(selected) < config.paths_per_point:
            selected.extend(
                base_records[take : take + config.paths_per_point - len(selected)]
            )
        for slot, record in enumerate(selected[: config.paths_per_point]):
            _write_record(arrays, row, slot, record)

    for name, output in output_files.items():
        atomic_npy(output, arrays[name])
    valid = arrays["mask"].sum(axis=1)
    report = {
        "stage": "path_tokens",
        "status": "complete",
        "split": split,
        "seed_count": config.seed_count,
        "shape": list(arrays["mask"].shape),
        "valid_paths": {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "min": int(np.min(valid)),
            "max": int(np.max(valid)),
        },
        "base_candidates": int(np.sum(arrays["source"] == 0)),
        "scatter_candidates": int(np.sum(arrays["source"] == 1)),
        "wall_seconds": float(time.time() - started),
        "measured_bottleneck": "CPU path identity aggregation",
    }
    atomic_json(paths.report_dir / f"path_tokens_{split}.json", report)
    return report
