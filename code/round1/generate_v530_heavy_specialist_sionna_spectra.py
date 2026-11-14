#!/usr/bin/env python3
"""Generate complete compact spectra for validated heavy-specialist configurations."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v504_scene_topology_sionna_inverse as v504
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 530
VERSION = "v530_heavy_specialist_sionna_spectra"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
SCENE_XML = (
    PATHS.artifacts
    / "external"
    / "restored_sionna_official_etoile"
    / "etoile.xml"
)
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
TINY = 1e-30


@dataclass(frozen=True)
class SpectrumConfig:
    name: str
    frequency_hz: float
    tx_azimuth_deg: float
    tx_tilt_deg: float
    max_depth: int
    synthetic_array: bool
    diffraction: bool
    edge_diffraction: bool
    diffuse_reflection: bool
    scattering_coefficient: float
    samples_per_src: int
    max_num_paths_per_src: int
    batch_size: int
    seed: int
    scattering_scope: str = "all"
    tx_vertical_spacing: float = 0.5


CONFIGS = (
    SpectrumConfig(
        name="depth10_scatter015_synthetic",
        frequency_hz=6.5e9,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-12.0,
        max_depth=10,
        synthetic_array=True,
        diffraction=False,
        edge_diffraction=False,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=42,
    ),
    SpectrumConfig(
        name="depth10_scatter015_synthetic_seed206",
        frequency_hz=6.5e9,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-12.0,
        max_depth=10,
        synthetic_array=True,
        diffraction=False,
        edge_diffraction=False,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=206,
    ),
    SpectrumConfig(
        name="depth10_scatter015_synthetic_seed420",
        frequency_hz=6.5e9,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-12.0,
        max_depth=10,
        synthetic_array=True,
        diffraction=False,
        edge_diffraction=False,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=420,
    ),
    SpectrumConfig(
        name="depth20_scatter015_synthetic_seed206",
        frequency_hz=6.5e9,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-12.0,
        max_depth=20,
        synthetic_array=True,
        diffraction=False,
        edge_diffraction=False,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=206,
    ),
    SpectrumConfig(
        name="depth20_scatter015_synthetic_seed42",
        frequency_hz=6.5e9,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-12.0,
        max_depth=20,
        synthetic_array=True,
        diffraction=False,
        edge_diffraction=False,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=42,
    ),
    SpectrumConfig(
        name="depth18_scatter015_nonmetal_synthetic_seed206",
        frequency_hz=6.5e9,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-12.0,
        max_depth=18,
        synthetic_array=True,
        diffraction=False,
        edge_diffraction=False,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=206,
        scattering_scope="nonmetal",
    ),
    SpectrumConfig(
        name="depth16_nonsynthetic_edge",
        frequency_hz=6.693e9,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-9.53,
        max_depth=16,
        synthetic_array=False,
        diffraction=True,
        edge_diffraction=True,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
        samples_per_src=3_000_000,
        max_num_paths_per_src=100_000,
        batch_size=2,
        seed=42,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=("all", *(config.name for config in CONFIGS)),
        default="all",
    )
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--selection", choices=("all", "heavy"), default="all")
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def output_paths(
    config: SpectrumConfig, split: str, selection: str
) -> dict[str, object]:
    suffix = "" if selection == "all" else f"_{selection}"
    prefix = f"{config.name}_{split}{suffix}"
    return {
        "joint_sh": OUT_DIR / f"{prefix}_joint_sh.npy",
        "log_power": OUT_DIR / f"{prefix}_log_power.npy",
        "path_count": OUT_DIR / f"{prefix}_path_count.npy",
        "done": OUT_DIR / f"{prefix}_done.npy",
        "indices": OUT_DIR / f"{prefix}_indices.npy",
    }


def open_or_create(path, dtype, shape, force: bool):
    if force:
        path.unlink(missing_ok=True)
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(
                f"cache mismatch for {path}: {value.shape} {value.dtype}, "
                f"expected {shape} {np.dtype(dtype)}"
            )
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value[...] = 0
    value.flush()
    return value


def joint_sh(channel: np.ndarray) -> np.ndarray:
    count = len(channel)
    value = channel.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    value = value.transpose(0, 1, 5, 2, 4, 3)
    spectrum = np.fft.fft2(value, axes=(2, 3))
    power = np.abs(spectrum).astype(np.float32) ** 2
    output = power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    output /= np.maximum(output.sum(axis=(1, 2), keepdims=True), TINY)
    return output.astype(np.float32)


def target_point(config: SpectrumConfig, mi):
    radius = 140.0
    azimuth = math.radians(config.tx_azimuth_deg)
    tilt = math.radians(config.tx_tilt_deg)
    return mi.Point3f(
        float(BS_POSITION[0] + radius * math.cos(azimuth)),
        float(BS_POSITION[1] + radius * math.sin(azimuth)),
        float(BS_POSITION[2] + radius * math.tan(tilt)),
    )


def configure_materials(scene, config: SpectrumConfig) -> dict[str, object]:
    audit = v504.material_audit(scene)
    if set(audit) != {"concrete", "marble", "metal", "wood"}:
        raise RuntimeError(f"unexpected canonical materials: {audit}")
    for name, material in scene.radio_materials.items():
        active = config.scattering_scope == "all" or name != "metal"
        material.scattering_coefficient = (
            config.scattering_coefficient if active else 0.0
        )
    return v504.material_audit(scene)


def chunked_cfr(paths, frequencies: np.ndarray, mi, chunk_size: int = 32):
    try:
        parts = []
        for start in range(0, len(frequencies), chunk_size):
            stop = min(start + chunk_size, len(frequencies))
            parts.append(
                paths.cfr(
                    mi.TensorXf(frequencies[start:stop]),
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


def select_drjit_cuda_device(dr, device: int | None) -> None:
    if device is None:
        return
    library_path = Path(dr.__file__).with_name("libdrjit-core.so")
    library = ctypes.CDLL(str(library_path))
    setter = getattr(library, "_Z19jit_cuda_set_devicei")
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    setter(int(device))


def generate_split(
    *,
    config: SpectrumConfig,
    split: str,
    positions: np.ndarray,
    point_ids: np.ndarray,
    args: argparse.Namespace,
    mi,
    dr,
    rt_api,
) -> dict[str, object]:
    paths_out = output_paths(config, split, args.selection)
    count = len(positions)
    shape = (count, NUM_SUBCARRIERS, 16)
    sh_out = open_or_create(paths_out["joint_sh"], np.float32, shape, args.force)
    log_power_out = open_or_create(
        paths_out["log_power"], np.float32, (count,), args.force
    )
    path_count_out = open_or_create(
        paths_out["path_count"], np.int32, (count,), args.force
    )
    done = open_or_create(paths_out["done"], np.bool_, (count,), args.force)
    indices_path = paths_out["indices"]
    if args.force:
        indices_path.unlink(missing_ok=True)
    if indices_path.is_file():
        cached_ids = np.load(indices_path)
        if not np.array_equal(cached_ids, point_ids):
            raise ValueError(f"point index mismatch for {indices_path}")
    else:
        np.save(indices_path, point_ids.astype(np.int64))
    if np.all(done):
        return {
            "reused": True,
            "count": count,
            "seconds": 0.0,
            "outputs": {
                name: PATHS.relative(path) for name, path in paths_out.items()
            },
        }

    load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies = rt_api
    scene = load_scene(
        str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
    )
    scene.frequency = config.frequency_hz
    materials = configure_materials(scene, config)
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=config.tx_vertical_spacing,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="VH",
    )
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )
    tx = Transmitter(
        name=f"tx_{config.name}",
        position=mi.Point3f(*map(float, BS_POSITION)),
        look_at=target_point(config, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()
    receiver_names: list[str] = []
    batch_size = args.batch_size or config.batch_size
    samples = max(1, int(round(config.samples_per_src * args.budget_scale)))
    max_paths = max(1, int(round(config.max_num_paths_per_src * args.budget_scale)))
    started = time.time()
    for batch_index, start in enumerate(range(0, count, batch_size)):
        end = min(start + batch_size, count)
        if np.all(done[start:end]):
            continue
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for local_id, position in enumerate(positions[start:end], start=start):
            point_id = int(point_ids[local_id])
            name = f"rx_{split}_{point_id:05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, position)),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        select_drjit_cuda_device(dr, args.cuda_device)
        paths = solver(
            scene,
            max_depth=config.max_depth,
            max_num_paths_per_src=max_paths,
            samples_per_src=samples,
            synthetic_array=config.synthetic_array,
            los=True,
            specular_reflection=True,
            diffuse_reflection=config.diffuse_reflection,
            refraction=True,
            diffraction=config.diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=True,
            seed=config.seed
            + (1_000_000 if split == "test" else 0)
            + batch_index,
        )
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        select_drjit_cuda_device(dr, args.cuda_device)
        raw = chunked_cfr(
            paths,
            frequencies(config.frequency_hz),
            mi,
        )
        channel = v501.convert_cfr(np.asarray(raw), end - start)
        sh_out[start:end] = joint_sh(channel)
        energy = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
        log_power_out[start:end] = np.log(np.maximum(energy, TINY)).astype(np.float32)
        path_count_out[start:end] = valid.reshape(end - start, -1).sum(axis=1)
        done[start:end] = True
        sh_out.flush()
        log_power_out.flush()
        path_count_out.flush()
        done.flush()
        del paths, valid, raw, channel
        dr.sync_thread()
        if end == count or (batch_index + 1) % 10 == 0:
            print(
                f"v530 {config.name} {split}: {end}/{count} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    for name in receiver_names:
        scene.remove(name)
    dr.sync_thread()
    flush = getattr(dr, "flush_malloc_cache", None)
    if callable(flush):
        flush()
    if not np.all(done):
        raise RuntimeError(f"incomplete output for {config.name} {split}")
    if not np.all(np.isfinite(sh_out)) or not np.all(np.isfinite(log_power_out)):
        raise FloatingPointError(f"non-finite output for {config.name} {split}")
    return {
        "reused": False,
        "count": count,
        "seconds": float(time.time() - started),
        "samples_per_src": samples,
        "max_num_paths_per_src": max_paths,
        "zero_spectrum_count": int(np.sum(np.asarray(sh_out).sum(axis=(1, 2)) == 0.0)),
        "mean_path_count": float(np.mean(path_count_out)),
        "materials": materials,
        "outputs": {name: PATHS.relative(path) for name, path in paths_out.items()},
    }


def main() -> None:
    args = parse_args()
    if args.budget_scale <= 0.0:
        raise ValueError("--budget-scale must be positive")
    for path in (
        SCENE_XML,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    select_drjit_cuda_device(dr, args.cuda_device)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
    select_drjit_cuda_device(dr, args.cuda_device)

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v530 requires Sionna 2.0.1, got {sionna.__version__}")

    def frequencies(frequency_hz: float):
        del frequency_hz
        return np.asarray(
            subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
        ).astype(np.float32)

    rt_api = (load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies)
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_ids = np.arange(len(train_positions), dtype=np.int64)
    test_ids = np.arange(len(test_positions), dtype=np.int64)
    if args.selection == "heavy":
        geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
        train_ids = np.flatnonzero(
            regime_of(geometry["train_blocked"]) == "heavy"
        ).astype(np.int64)
        test_ids = np.flatnonzero(
            regime_of(geometry["test_blocked"]) == "heavy"
        ).astype(np.int64)
    train_ids = train_ids[: min(max(args.train_count, 0), len(train_ids))]
    test_ids = test_ids[: min(max(args.test_count, 0), len(test_ids))]
    train_positions = train_positions[train_ids]
    test_positions = test_positions[test_ids]
    selected_configs = [
        config for config in CONFIGS if args.config in ("all", config.name)
    ]
    selected_splits = ("train", "test") if args.split == "all" else (args.split,)
    results: dict[str, object] = {}
    for config in selected_configs:
        split_results = {}
        for split in selected_splits:
            positions = train_positions if split == "train" else test_positions
            point_ids = train_ids if split == "train" else test_ids
            split_results[split] = generate_split(
                config=config,
                split=split,
                positions=positions,
                point_ids=point_ids,
                args=args,
                mi=mi,
                dr=dr,
                rt_api=rt_api,
            )
        results[config.name] = {
            "configuration": asdict(config),
            "splits": split_results,
        }
    report = {
        "method": VERSION,
        "status": "complete",
        "primary_hypothesis": (
            "The existing six-source Sionna library lacks configurations that are "
            "independently stronger on medium/heavy propagation. Complete spectra from "
            "the v502 shallow-scattering and v513 non-synthetic edge configurations may "
            "add missing path birth, disappearance, delay, and angle support."
        ),
        "environment": {
            "mitsuba_variant": mi.variant(),
            "sionna_version": sionna.__version__,
            "single_visible_gpu_required": True,
            "requested_cuda_device": args.cuda_device,
        },
        "results": results,
        "report": PATHS.relative(REPORT),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
