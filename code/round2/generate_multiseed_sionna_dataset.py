#!/usr/bin/env python3
"""Generate resumable high-budget multi-seed Sionna data for Round 2."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np

from project_paths import PATHS
from round2.data import load_positions
from generate_v5_sionna_spectral_library import (
    SCENE_XML,
    configure_materials,
    prepare_scene,
)


VERSION = "Round2-E39-high-budget-multiseed-Sionna-dataset"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_multiseed"
REPORT_DIR = PATHS.analysis / "results" / "round2_sionna_multiseed"
REPORT = REPORT_DIR / "sionna_multiseed_generation.json"
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
TINY = 1e-30
SEEDS = (
    42,
    101,
    211,
    307,
    401,
    503,
    601,
    701,
    809,
    907,
    1009,
    1103,
    1201,
    1301,
    1409,
    1511,
    1601,
    1709,
    1801,
    1901,
    2003,
    2111,
    2203,
    2309,
)
CHECKPOINTS = (6, 12, 24)
DOMAINS = ("pas", "pdp", "ad", "log_power", "path_count")


@dataclass(frozen=True)
class GenerationConfig:
    name: str
    frequency_hz: float
    site_yaw_deg: tuple[float, float]
    site_tilt_deg: tuple[float, float]
    tx_vertical_spacing: float
    tx_horizontal_spacing: float
    max_depth: int
    synthetic_array: bool
    diffuse_reflection: bool
    scattering_coefficient: float
    diffraction: bool
    edge_diffraction: bool
    samples_per_src: int
    max_num_paths_per_src: int
    path_coefficient_mode: str


CONFIGS = (
    GenerationConfig(
        name="primary_d16_base_v4lambda_1m",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=16,
        synthetic_array=True,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
        diffraction=False,
        edge_diffraction=False,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        path_coefficient_mode="full",
    ),
    GenerationConfig(
        name="primary_d10_scatter015_v4lambda_1m",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=10,
        synthetic_array=True,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        diffraction=False,
        edge_diffraction=False,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        path_coefficient_mode="reference",
    ),
    GenerationConfig(
        name="aux_d8_edge_v4lambda_1m",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=8,
        synthetic_array=True,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
        diffraction=True,
        edge_diffraction=True,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        path_coefficient_mode="reference",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=("all", "primary", *(config.name for config in CONFIGS)),
        default="primary",
    )
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-seeds", type=int, default=len(SEEDS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-count-per-site", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260805)
    parser.add_argument("--selection-tag", default="full")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--skip-path-data", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def sample_ids(
    positions: np.ndarray,
    site: np.ndarray,
    count_per_site: int,
    seed: int,
) -> np.ndarray:
    if count_per_site <= 0:
        return np.arange(len(positions), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for site_value in np.unique(site):
        candidates = np.flatnonzero(site == site_value)
        count = min(count_per_site, len(candidates))
        selected.append(np.sort(rng.choice(candidates, size=count, replace=False)))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def select_drjit_cuda_device(dr, device: int) -> None:
    library_path = Path(dr.__file__).with_name("libdrjit-core.so")
    library = ctypes.CDLL(str(library_path))
    setter = getattr(library, "_Z19jit_cuda_set_devicei")
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    setter(int(device))


def tx_target(position: np.ndarray, yaw_deg: float, tilt_deg: float, mi):
    radius = 140.0
    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)
    return mi.Point3f(
        float(position[0] + radius * math.cos(yaw)),
        float(position[1] + radius * math.sin(yaw)),
        float(position[2] + radius * math.tan(tilt)),
    )


def chunked_cfr(paths, frequencies: np.ndarray, mi, chunk_size: int = 32):
    try:
        parts = []
        for start in range(0, len(frequencies), chunk_size):
            end = min(start + chunk_size, len(frequencies))
            parts.append(
                paths.cfr(
                    mi.TensorXf(frequencies[start:end]),
                    normalize_delays=False,
                    normalize=False,
                    out_type="numpy",
                )
            )
        return np.concatenate(parts, axis=-1)
    except RuntimeError as error:
        if chunk_size <= 1 or "exceeds the limit of 2^32" not in str(error):
            raise
        return chunked_cfr(paths, frequencies, mi, chunk_size // 2)


def convert_cfr(raw: np.ndarray, count: int) -> np.ndarray:
    expected = (count, 4, 1, 256, 1, NUM_SUBCARRIERS)
    if raw.shape != expected:
        raise ValueError(f"unexpected CFR shape {raw.shape}, expected {expected}")
    return np.transpose(raw[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def score_aligned_statistics(channel: np.ndarray) -> dict[str, np.ndarray]:
    count = len(channel)
    shaped = channel.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    spatial = np.fft.fft2(shaped, axes=(2, 3))
    pas_power = np.square(np.abs(spatial), dtype=np.float32)
    pas_total = pas_power.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    pas_valid = pas_total > TINY
    pas = np.divide(
        pas_power,
        np.maximum(pas_total[:, None, None, None, :, :], TINY),
        out=np.zeros_like(pas_power),
        where=pas_valid[:, None, None, None, :, :],
    )

    cir = np.fft.ifft(channel, axis=-1)
    pdp_power = np.square(np.abs(cir), dtype=np.float32)
    pdp_total = pdp_power.sum(axis=-1, dtype=np.float64).astype(np.float32)
    pdp_valid = pdp_total > TINY
    pdp = np.divide(
        pdp_power,
        np.maximum(pdp_total[..., None], TINY),
        out=np.zeros_like(pdp_power),
        where=pdp_valid[..., None],
    )

    joint = np.fft.ifft(spatial, axis=-1)
    ad_power = np.square(np.abs(joint), dtype=np.float32).sum(
        axis=(1, 4), dtype=np.float64
    ).astype(np.float32)
    ad_total = ad_power.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    ad_valid = ad_total > TINY
    ad = np.divide(
        ad_power,
        np.maximum(ad_total[:, None, None, None], TINY),
        out=np.zeros_like(ad_power),
        where=ad_valid[:, None, None, None],
    )

    point_power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
    point_valid = point_power > TINY
    log_power = np.zeros(count, dtype=np.float32)
    log_power[point_valid] = np.log(point_power[point_valid]).astype(np.float32)
    return {
        "pas": pas.astype(np.float32),
        "pas_valid": pas_valid,
        "pdp": pdp.astype(np.float32),
        "pdp_valid": pdp_valid,
        "ad": ad.astype(np.float32),
        "ad_valid": ad_valid,
        "log_power": log_power,
        "log_power_valid": point_valid,
    }


def domain_shape(name: str, rows: int) -> tuple[int, ...]:
    return {
        "pas": (rows, 2, 16, 8, 4, NUM_SUBCARRIERS),
        "pdp": (rows, 256, 4, NUM_SUBCARRIERS),
        "ad": (rows, 16, 8, NUM_SUBCARRIERS),
        "log_power": (rows,),
        "path_count": (rows,),
    }[name]


def count_shape(name: str, rows: int) -> tuple[int, ...]:
    return {
        "pas": (rows, 4, NUM_SUBCARRIERS),
        "pdp": (rows, 256, 4),
        "ad": (rows,),
        "log_power": (rows,),
        "path_count": (rows,),
    }[name]


def expand_count(name: str, value: np.ndarray) -> np.ndarray:
    if name == "pas":
        return value[:, None, None, None, :, :]
    if name == "pdp":
        return value[..., None]
    if name == "ad":
        return value[:, None, None, None]
    return value


def open_memmap(
    path: Path,
    dtype,
    shape: tuple[int, ...],
    reset: bool = False,
    initialize: bool = False,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        path.unlink(missing_ok=True)
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"cache mismatch for {path}: {value.shape}/{value.dtype}")
        return value
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if initialize:
        value[...] = 0
        value.flush()
    return value


def running_path(root: Path, split: str, name: str, kind: str, count: int) -> Path:
    return root / "running" / f"{split}_{name}_{kind}_n{count:02d}.npy"


def active_done_path(root: Path, split: str, count: int) -> Path:
    return root / "running" / f"{split}_active_n{count:02d}_done.npy"


def checkpoint_path(
    root: Path, checkpoint: int, split: str, name: str, kind: str
) -> Path:
    return root / "checkpoints" / f"n{checkpoint:02d}_{split}_{name}_{kind}.npy"


def initial_state(
    config: GenerationConfig,
    tag: str,
    ids: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "configuration": json.loads(json.dumps(asdict(config))),
        "selection_tag": tag,
        "seed_order": list(SEEDS),
        "ids": {
            split: {"count": int(len(value)), "sha256": ids_digest(value)}
            for split, value in ids.items()
        },
        "splits": {
            split: {"count": 0, "completed_seeds": [], "active_seed": None}
            for split in ids
        },
    }


def load_state(
    path: Path,
    config: GenerationConfig,
    tag: str,
    ids: dict[str, np.ndarray],
) -> dict[str, Any]:
    expected = initial_state(config, tag, ids)
    if not path.is_file():
        atomic_json(path, expected)
        return expected
    state = json.loads(path.read_text(encoding="utf-8"))
    for key in ("configuration", "selection_tag", "seed_order", "ids"):
        if state.get(key) != expected[key]:
            raise ValueError(f"generation state changed for {path}: {key}")
    return state


def begin_seed(
    root: Path,
    state_path: Path,
    state: dict[str, Any],
    split: str,
    seed: int,
    rows: int,
) -> tuple[
    int,
    dict[str, dict[str, np.memmap]],
    dict[str, dict[str, np.memmap]] | None,
    np.memmap,
]:
    split_state = state["splits"][split]
    old_count = int(split_state["count"])
    new_count = old_count + 1
    active = split_state.get("active_seed")
    resume = active == seed
    if active not in (None, seed):
        raise RuntimeError(f"unfinished seed {active} blocks seed {seed} for {split}")

    stores: dict[str, dict[str, np.memmap]] = {}
    old_stores: dict[str, dict[str, np.memmap]] | None = {} if old_count else None
    for name in DOMAINS:
        stores[name] = {
            "mean": open_memmap(
                running_path(root, split, name, "mean", new_count),
                np.float32,
                domain_shape(name, rows),
                reset=not resume,
            ),
            "m2": open_memmap(
                running_path(root, split, name, "m2", new_count),
                np.float32,
                domain_shape(name, rows),
                reset=not resume,
            ),
            "count": open_memmap(
                running_path(root, split, name, "count", new_count),
                np.uint16,
                count_shape(name, rows),
                reset=not resume,
            ),
        }
        if old_stores is not None:
            old_stores[name] = {
                kind: np.load(
                    running_path(root, split, name, kind, old_count), mmap_mode="r"
                )
                for kind in ("mean", "m2", "count")
            }
    done = open_memmap(
        active_done_path(root, split, new_count),
        np.bool_,
        (rows,),
        reset=not resume,
        initialize=not resume,
    )
    if not resume:
        split_state["active_seed"] = int(seed)
        atomic_json(state_path, state)
    return old_count, stores, old_stores, done


def update_welford(
    *,
    root: Path,
    split: str,
    name: str,
    rows: np.ndarray,
    value: np.ndarray,
    valid: np.ndarray,
    old_count: int,
    stores: dict[str, dict[str, np.memmap]],
    old_stores: dict[str, dict[str, np.memmap]] | None,
) -> None:
    current = stores[name]
    valid_uint = np.asarray(valid, dtype=np.uint16)
    if old_count:
        assert old_stores is not None
        previous = np.asarray(old_stores[name]["mean"][rows], dtype=np.float32)
        previous_m2 = np.asarray(old_stores[name]["m2"][rows], dtype=np.float32)
        previous_count = np.asarray(old_stores[name]["count"][rows], dtype=np.uint16)
    else:
        previous = np.zeros_like(value, dtype=np.float32)
        previous_m2 = np.zeros_like(value, dtype=np.float32)
        previous_count = np.zeros_like(valid_uint, dtype=np.uint16)

    updated_count = previous_count + valid_uint
    denominator = np.maximum(expand_count(name, updated_count), 1).astype(np.float32)
    mask = expand_count(name, np.asarray(valid, dtype=bool))
    delta = value - previous
    updated = np.where(mask, previous + delta / denominator, previous)
    updated_m2 = np.where(
        mask,
        previous_m2 + delta * (value - updated),
        previous_m2,
    )
    current["mean"][rows] = updated
    current["m2"][rows] = updated_m2
    current["count"][rows] = updated_count


def compressed_dataset(group, name: str, value: np.ndarray) -> None:
    value = np.asarray(value)
    kwargs: dict[str, Any] = {}
    if value.size:
        kwargs.update(compression="lzf", shuffle=True)
    group.create_dataset(name, data=value, **kwargs)


def path_batch(
    traced,
    row_indices: np.ndarray,
    point_ids: np.ndarray,
    coefficient_mode: str,
) -> dict[str, np.ndarray]:
    valid = np.asarray(traced.valid.numpy(), dtype=bool)
    tau = np.asarray(traced.tau.numpy(), dtype=np.float32)
    theta_t = np.asarray(traced.theta_t.numpy(), dtype=np.float32)
    phi_t = np.asarray(traced.phi_t.numpy(), dtype=np.float32)
    theta_r = np.asarray(traced.theta_r.numpy(), dtype=np.float32)
    phi_r = np.asarray(traced.phi_r.numpy(), dtype=np.float32)
    interactions = np.asarray(traced.interactions.numpy(), dtype=np.uint32)
    objects = np.asarray(traced.objects.numpy(), dtype=np.uint32)
    primitives = np.asarray(traced.primitives.numpy(), dtype=np.uint32)
    vertices = np.asarray(traced.vertices.numpy(), dtype=np.float32)
    raw_a_value = traced.a
    if isinstance(raw_a_value, tuple):
        real = np.asarray(raw_a_value[0].numpy(), dtype=np.float32)
        imag = np.asarray(raw_a_value[1].numpy(), dtype=np.float32)
    else:
        raw_a = np.asarray(raw_a_value.numpy())
        if np.iscomplexobj(raw_a):
            coefficients = raw_a.astype(np.complex64)
            real = imag = None
        elif raw_a.ndim == 6 and raw_a.shape[0] == 2:
            real = np.asarray(raw_a[0], dtype=np.float32)
            imag = np.asarray(raw_a[1], dtype=np.float32)
        else:
            raise ValueError(f"unexpected path coefficient representation {raw_a.shape}")
    if real is not None and imag is not None:
        coefficients = (real + 1j * imag).astype(np.complex64)

    offsets = [0]
    banks: dict[str, list[np.ndarray]] = {
        "tau": [],
        "theta_t": [],
        "phi_t": [],
        "theta_r": [],
        "phi_r": [],
        "interactions": [],
        "objects": [],
        "primitives": [],
        "vertices": [],
        "path_energy": [],
        "path_energy_fraction": [],
        "port_energy": [],
        "reference_coefficient": [],
        "interaction_count": [],
    }
    if coefficient_mode == "full":
        banks["full_coefficient"] = []

    for local_index in range(len(row_indices)):
        indices = np.flatnonzero(valid[local_index, 0])
        count = len(indices)
        offsets.append(offsets[-1] + count)
        if count == 0:
            continue
        banks["tau"].append(tau[local_index, 0, indices])
        banks["theta_t"].append(theta_t[local_index, 0, indices])
        banks["phi_t"].append(phi_t[local_index, 0, indices])
        banks["theta_r"].append(theta_r[local_index, 0, indices])
        banks["phi_r"].append(phi_r[local_index, 0, indices])
        local_interactions = np.take(
            interactions[:, local_index, 0, :], indices, axis=-1
        ).T
        banks["interactions"].append(local_interactions.astype(np.uint8))
        banks["objects"].append(
            np.take(objects[:, local_index, 0, :], indices, axis=-1).T
        )
        banks["primitives"].append(
            np.take(primitives[:, local_index, 0, :], indices, axis=-1).T
        )
        banks["vertices"].append(
            np.take(vertices[:, local_index, 0, :, :], indices, axis=1).transpose(
                1, 0, 2
            )
        )
        banks["interaction_count"].append(
            np.count_nonzero(local_interactions, axis=1).astype(np.uint8)
        )

        local_coefficients = np.take(
            coefficients[local_index, :, 0, :, :], indices, axis=-1
        ).transpose(2, 1, 0)
        shaped = local_coefficients.reshape(count, 2, 16, 8, 4)
        port_energy = np.square(np.abs(shaped), dtype=np.float32).sum(
            axis=(2, 3), dtype=np.float64
        ).astype(np.float32)
        path_energy = port_energy.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
        banks["port_energy"].append(port_energy)
        banks["path_energy"].append(path_energy)
        banks["path_energy_fraction"].append(
            path_energy / np.maximum(path_energy.sum(dtype=np.float64), TINY)
        )
        banks["reference_coefficient"].append(shaped[:, :, 0, 0, :])
        if coefficient_mode == "full":
            banks["full_coefficient"].append(local_coefficients)

    depth = interactions.shape[0]
    empty_shapes = {
        "tau": (0,),
        "theta_t": (0,),
        "phi_t": (0,),
        "theta_r": (0,),
        "phi_r": (0,),
        "interactions": (0, depth),
        "objects": (0, depth),
        "primitives": (0, depth),
        "vertices": (0, depth, 3),
        "path_energy": (0,),
        "path_energy_fraction": (0,),
        "port_energy": (0, 2, 4),
        "reference_coefficient": (0, 2, 4),
        "interaction_count": (0,),
        "full_coefficient": (0, 256, 4),
    }
    output: dict[str, np.ndarray] = {
        "row_indices": np.asarray(row_indices, dtype=np.int64),
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "offsets": np.asarray(offsets, dtype=np.int64),
    }
    for name, parts in banks.items():
        if parts:
            output[name] = np.concatenate(parts, axis=0)
        else:
            dtype = np.complex64 if "coefficient" in name else np.float32
            if name in ("interactions", "interaction_count"):
                dtype = np.uint8
            elif name in ("objects", "primitives"):
                dtype = np.uint32
            output[name] = np.empty(empty_shapes[name], dtype=dtype)
    return output


def write_path_group(
    handle: h5py.File,
    group_name: str,
    values: dict[str, np.ndarray],
) -> None:
    if group_name in handle:
        del handle[group_name]
    group = handle.create_group(group_name)
    for name, value in values.items():
        compressed_dataset(group, name, value)
    handle.flush()


def prepare_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    select_drjit_cuda_device(dr, args.cuda_device)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"expected Sionna 2.0.1, got {sionna.__version__}")
    frequencies = np.asarray(
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
    ).astype(np.float32)
    return {
        "dr": dr,
        "mi": mi,
        "sionna": sionna,
        "frequencies": frequencies,
        "PathSolver": PathSolver,
        "PlanarArray": PlanarArray,
        "Receiver": Receiver,
        "Transmitter": Transmitter,
        "load_scene": load_scene,
    }


def finish_seed(
    *,
    root: Path,
    state_path: Path,
    state: dict[str, Any],
    split: str,
    seed: int,
    old_count: int,
    rows: int,
) -> None:
    new_count = old_count + 1
    if new_count in CHECKPOINTS:
        for name in DOMAINS:
            mean = np.load(
                running_path(root, split, name, "mean", new_count), mmap_mode="r"
            )
            m2 = np.load(
                running_path(root, split, name, "m2", new_count), mmap_mode="r"
            )
            count = np.load(
                running_path(root, split, name, "count", new_count), mmap_mode="r"
            )
            mean_copy = open_memmap(
                checkpoint_path(root, new_count, split, name, "mean"),
                np.float32,
                domain_shape(name, rows),
                reset=True,
            )
            variance = open_memmap(
                checkpoint_path(root, new_count, split, name, "variance"),
                np.float32,
                domain_shape(name, rows),
                reset=True,
            )
            count_copy = open_memmap(
                checkpoint_path(root, new_count, split, name, "valid_count"),
                np.uint16,
                count_shape(name, rows),
                reset=True,
            )
            for start in range(0, rows, 8):
                end = min(start + 8, rows)
                local_count = np.asarray(count[start:end], dtype=np.uint16)
                mean_copy[start:end] = mean[start:end]
                denominator = np.maximum(
                    expand_count(name, local_count).astype(np.float32) - 1.0, 1.0
                )
                variance[start:end] = np.asarray(m2[start:end], dtype=np.float32) / denominator
                count_copy[start:end] = local_count
            for value in (mean_copy, variance, count_copy):
                value.flush()

    split_state = state["splits"][split]
    split_state["count"] = new_count
    split_state["completed_seeds"] = [
        *split_state["completed_seeds"],
        int(seed),
    ]
    split_state["active_seed"] = None
    atomic_json(state_path, state)
    active_done_path(root, split, new_count).unlink(missing_ok=True)
    if old_count:
        for name in DOMAINS:
            for kind in ("mean", "m2", "count"):
                running_path(root, split, name, kind, old_count).unlink(missing_ok=True)


def generate_seed(
    *,
    config: GenerationConfig,
    root: Path,
    state_path: Path,
    state: dict[str, Any],
    split: str,
    seed: int,
    positions: np.ndarray,
    site: np.ndarray,
    ids: np.ndarray,
    args: argparse.Namespace,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    selected_positions = positions[ids].astype(np.float32)
    selected_site = site[ids].astype(np.int8)
    old_count, stores, old_stores, done = begin_seed(
        root, state_path, state, split, seed, len(ids)
    )
    path_file = root / "paths" / f"{split}_seed{seed}.h5"
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_handle = h5py.File(path_file, "a") if not args.skip_path_data else None
    if path_handle is not None:
        path_handle.attrs["version"] = VERSION
        path_handle.attrs["configuration"] = json.dumps(asdict(config), sort_keys=True)
        path_handle.attrs["split"] = split
        path_handle.attrs["seed"] = int(seed)
        path_handle.attrs["coefficient_mode"] = config.path_coefficient_mode
        path_handle.attrs["complete"] = False

    setup, _, _, _, _ = load_positions()
    total_path_count = np.zeros(len(ids), dtype=np.int32)
    started = time.time()
    materials: dict[str, float] = {}
    for site_value in np.unique(selected_site):
        scene = runtime["load_scene"](
            str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
        )
        scene.frequency = config.frequency_hz
        scene.tx_array = runtime["PlanarArray"](
            num_rows=8,
            num_cols=16,
            vertical_spacing=config.tx_vertical_spacing,
            horizontal_spacing=config.tx_horizontal_spacing,
            pattern="tr38901",
            polarization="VH",
        )
        scene.rx_array = runtime["PlanarArray"](
            num_rows=2,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="cross",
        )
        materials = configure_materials(scene, config.scattering_coefficient)
        bs_position = setup.bs_positions[int(site_value)]
        transmitter = runtime["Transmitter"](
            name=f"tx_site{int(site_value)}_seed{seed}",
            position=runtime["mi"].Point3f(*map(float, bs_position)),
            look_at=tx_target(
                bs_position,
                config.site_yaw_deg[int(site_value)],
                config.site_tilt_deg[int(site_value)],
                runtime["mi"],
            ),
            power_dbm=44.0,
        )
        scene.add(transmitter)
        solver = runtime["PathSolver"]()
        receiver_names: list[str] = []
        local_rows = np.flatnonzero(selected_site == site_value)
        for offset in range(0, len(local_rows), args.batch_size):
            batch_rows = local_rows[offset : offset + args.batch_size]
            if np.all(done[batch_rows]):
                continue
            for name in receiver_names:
                scene.remove(name)
            receiver_names = []
            for row in batch_rows:
                name = f"rx_{split}_{int(ids[row]):05d}"
                scene.add(
                    runtime["Receiver"](
                        name=name,
                        position=runtime["mi"].Point3f(
                            *map(float, selected_positions[row])
                        ),
                        look_at=transmitter,
                    )
                )
                receiver_names.append(name)
            select_drjit_cuda_device(runtime["dr"], args.cuda_device)
            solver_seed = (
                int(seed)
                + 100_000 * int(site_value)
                + 1_000_000 * int(split == "test")
                + int(ids[batch_rows[0]])
            )
            traced = solver(
                scene,
                max_depth=config.max_depth,
                max_num_paths_per_src=config.max_num_paths_per_src,
                samples_per_src=config.samples_per_src,
                synthetic_array=config.synthetic_array,
                los=True,
                specular_reflection=True,
                diffuse_reflection=config.diffuse_reflection,
                refraction=True,
                diffraction=config.diffraction,
                edge_diffraction=config.edge_diffraction,
                diffraction_lit_region=True,
                seed=solver_seed,
            )
            valid_paths = np.asarray(traced.valid.numpy(), dtype=bool)
            local_path_count = valid_paths.reshape(len(batch_rows), -1).sum(axis=1)
            total_path_count[batch_rows] = local_path_count
            raw = chunked_cfr(
                traced, runtime["frequencies"], runtime["mi"]
            )
            channel = convert_cfr(np.asarray(raw), len(batch_rows))
            local = score_aligned_statistics(channel)
            update_welford(
                root=root,
                split=split,
                name="pas",
                rows=batch_rows,
                value=local["pas"],
                valid=local["pas_valid"],
                old_count=old_count,
                stores=stores,
                old_stores=old_stores,
            )
            update_welford(
                root=root,
                split=split,
                name="pdp",
                rows=batch_rows,
                value=local["pdp"],
                valid=local["pdp_valid"],
                old_count=old_count,
                stores=stores,
                old_stores=old_stores,
            )
            update_welford(
                root=root,
                split=split,
                name="ad",
                rows=batch_rows,
                value=local["ad"],
                valid=local["ad_valid"],
                old_count=old_count,
                stores=stores,
                old_stores=old_stores,
            )
            update_welford(
                root=root,
                split=split,
                name="log_power",
                rows=batch_rows,
                value=local["log_power"],
                valid=local["log_power_valid"],
                old_count=old_count,
                stores=stores,
                old_stores=old_stores,
            )
            update_welford(
                root=root,
                split=split,
                name="path_count",
                rows=batch_rows,
                value=local_path_count.astype(np.float32),
                valid=np.ones(len(batch_rows), dtype=bool),
                old_count=old_count,
                stores=stores,
                old_stores=old_stores,
            )
            if path_handle is not None:
                group_name = (
                    f"site{int(site_value)}/batch_{int(batch_rows[0]):05d}_"
                    f"{int(batch_rows[-1]):05d}"
                )
                write_path_group(
                    path_handle,
                    group_name,
                    path_batch(
                        traced,
                        batch_rows,
                        ids[batch_rows],
                        config.path_coefficient_mode,
                    ),
                )
            done[batch_rows] = True
            for domain in stores.values():
                for value in domain.values():
                    value.flush()
            done.flush()
            del traced, raw, channel, local
            runtime["dr"].sync_thread()
            completed = int(np.sum(done))
            if completed == len(ids) or completed % 160 < len(batch_rows):
                print(
                    f"{config.name} seed={seed} {split} site={site_value} "
                    f"{completed}/{len(ids)} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        for name in receiver_names:
            scene.remove(name)
        del scene, solver
        runtime["dr"].sync_thread()

    if not np.all(done):
        raise RuntimeError(f"incomplete seed {seed} for {config.name}/{split}")
    if path_handle is not None:
        path_handle.attrs["complete"] = True
        path_handle.flush()
        path_handle.close()
    finish_seed(
        root=root,
        state_path=state_path,
        state=state,
        split=split,
        seed=seed,
        old_count=old_count,
        rows=len(ids),
    )
    return {
        "config": config.name,
        "split": split,
        "seed": int(seed),
        "rows": int(len(ids)),
        "seconds": float(time.time() - started),
        "mean_path_count": float(np.mean(total_path_count)),
        "zero_path_fraction": float(np.mean(total_path_count == 0)),
        "materials": materials,
        "path_file": PATHS.relative(path_file) if path_file.is_file() else None,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 1 <= args.max_seeds <= len(SEEDS):
        raise ValueError(f"--max-seeds must be in [1, {len(SEEDS)}]")
    prepare_scene()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    setup, train_pos, test_pos, train_site, test_site = load_positions()
    del setup
    split_positions = {"train": train_pos, "test": test_pos}
    split_sites = {"train": train_site, "test": test_site}
    selected_splits = ("train", "test") if args.split == "all" else (args.split,)
    ids = {
        split: sample_ids(
            split_positions[split],
            split_sites[split],
            args.sample_count_per_site,
            args.sample_seed + 10_000 * int(split == "test"),
        )
        for split in selected_splits
    }
    if args.config == "all":
        selected_configs = list(CONFIGS)
    elif args.config == "primary":
        selected_configs = [config for config in CONFIGS if config.name.startswith("primary_")]
    else:
        selected_configs = [config for config in CONFIGS if config.name == args.config]

    runtime = prepare_runtime(args)
    new_runs: list[dict[str, Any]] = []
    for config in selected_configs:
        root = ARTIFACT_DIR / config.name / args.selection_tag
        root.mkdir(parents=True, exist_ok=True)
        selected_ids = {split: ids[split] for split in selected_splits}
        for split, values in selected_ids.items():
            id_path = root / f"{split}_ids.npy"
            if id_path.is_file():
                cached = np.load(id_path)
                if not np.array_equal(cached, values):
                    raise ValueError(f"ids changed for {id_path}")
            else:
                np.save(id_path, values)
        state_path = root / "state.json"
        state = load_state(state_path, config, args.selection_tag, selected_ids)
        requested = SEEDS[: args.max_seeds]
        for seed in requested:
            for split in selected_splits:
                if seed in state["splits"][split]["completed_seeds"]:
                    continue
                new_runs.append(
                    generate_seed(
                        config=config,
                        root=root,
                        state_path=state_path,
                        state=state,
                        split=split,
                        seed=seed,
                        positions=split_positions[split],
                        site=split_sites[split],
                        ids=selected_ids[split],
                        args=args,
                        runtime=runtime,
                    )
                )

    report = {
        "experiment": VERSION,
        "status": "physical_source_generation",
        "primary_hypothesis": (
            "High-budget independent Sionna realizations provide scorer-aligned PAS/PDP "
            "expectations, uncertainty and path-level candidates that can be calibrated "
            "against all 4000 official Round 2 training channels."
        ),
        "configs": [asdict(config) for config in selected_configs],
        "seed_order": list(SEEDS),
        "requested_seed_count": int(args.max_seeds),
        "checkpoints": list(CHECKPOINTS),
        "arguments": vars(args),
        "environment": {
            "sionna": runtime["sionna"].__version__,
            "mitsuba_variant": runtime["mi"].variant(),
            "single_visible_gpu_required": True,
        },
        "new_runs": new_runs,
        "output_root": PATHS.relative(ARTIFACT_DIR),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
