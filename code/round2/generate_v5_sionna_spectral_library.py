#!/usr/bin/env python3
"""Generate compact Round 2 Sionna spectra on the official triangle mesh.

The generator intentionally stores phase-insensitive targets rather than full
complex CFRs: grouped PAS, grouped linkwise PDP, collapsed angle-delay power,
point power, and path counts.  A small optional CFR sample is retained for
checking whether absolute Sionna phase is useful before any deployment.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from project_paths import PATHS
from round2.data import load_positions


VERSION = "Round2-E37-Sionna-spectral-library"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_sionna_upper_bound"
SCENE_DIR = ARTIFACT_DIR / "scene"
SCENE_XML = SCENE_DIR / "round2_scene.xml"
GROUND_OBJ = SCENE_DIR / "ground.obj"
REPORT_DIR = PATHS.analysis / "results" / "round2_v5_sionna_upper_bound"
REPORT = REPORT_DIR / "sionna_spectral_library.json"
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
TINY = 1e-30


@dataclass(frozen=True)
class SpectrumConfig:
    name: str
    frequency_hz: float
    site_yaw_deg: tuple[float, float]
    site_tilt_deg: tuple[float, float]
    tx_vertical_spacing: float
    max_depth: int
    synthetic_array: bool
    diffuse_reflection: bool
    scattering_coefficient: float
    diffraction: bool
    edge_diffraction: bool
    samples_per_src: int
    max_num_paths_per_src: int
    seed: int


CONFIGS = (
    SpectrumConfig(
        name="d8_base_v4lambda",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        max_depth=8,
        synthetic_array=True,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
        diffraction=False,
        edge_diffraction=False,
        samples_per_src=200_000,
        max_num_paths_per_src=200_000,
        seed=42,
    ),
    SpectrumConfig(
        name="d16_base_v4lambda",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        max_depth=16,
        synthetic_array=True,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
        diffraction=False,
        edge_diffraction=False,
        samples_per_src=400_000,
        max_num_paths_per_src=400_000,
        seed=42,
    ),
    SpectrumConfig(
        name="d10_scatter015_v4lambda",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        max_depth=10,
        synthetic_array=True,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        diffraction=False,
        edge_diffraction=False,
        samples_per_src=400_000,
        max_num_paths_per_src=400_000,
        seed=42,
    ),
    SpectrumConfig(
        name="d8_edge_v4lambda",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        max_depth=8,
        synthetic_array=True,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
        diffraction=True,
        edge_diffraction=True,
        samples_per_src=400_000,
        max_num_paths_per_src=200_000,
        seed=42,
    ),
    SpectrumConfig(
        name="d16_base_v05lambda",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=0.5,
        max_depth=16,
        synthetic_array=True,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
        diffraction=False,
        edge_diffraction=False,
        samples_per_src=400_000,
        max_num_paths_per_src=400_000,
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
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--sample-count-per-site", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260805)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--ids-key", default="query_ids")
    parser.add_argument("--selection-tag")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--save-cfr-sample", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def prepare_scene() -> None:
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    map_path = PATHS.round2 / "Round2_Map.ply"
    if not map_path.is_file():
        raise FileNotFoundError(map_path)

    positions = np.load(PATHS.round2 / "Round2_Train_Pos.npy")
    pad = 120.0
    x0 = float(positions[:, 0].min() - pad)
    x1 = float(positions[:, 0].max() + pad)
    y0 = float(positions[:, 1].min() - pad)
    y1 = float(positions[:, 1].max() + pad)
    GROUND_OBJ.write_text(
        "\n".join(
            [
                "# Round 2 generated ground",
                f"v {x0:.6f} {y0:.6f} 0.0",
                f"v {x1:.6f} {y0:.6f} 0.0",
                f"v {x1:.6f} {y1:.6f} 0.0",
                f"v {x0:.6f} {y1:.6f} 0.0",
                "f 1 2 3 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    relative_map = os.path.relpath(map_path, SCENE_DIR)
    SCENE_XML.write_text(
        "\n".join(
            [
                '<scene version="2.1.0">',
                '  <bsdf type="itu-radio-material" id="concrete">',
                '    <string name="type" value="concrete"/>',
                '    <float name="thickness" value="0.1"/>',
                "  </bsdf>",
                '  <shape type="ply" id="round2-map">',
                f'    <string name="filename" value="{relative_map}"/>',
                '    <boolean name="face_normals" value="true"/>',
                '    <ref id="concrete" name="bsdf"/>',
                "  </shape>",
                '  <shape type="obj" id="ground">',
                '    <string name="filename" value="ground.obj"/>',
                '    <boolean name="face_normals" value="true"/>',
                '    <ref id="concrete" name="bsdf"/>',
                "  </shape>",
                "</scene>",
                "",
            ]
        ),
        encoding="utf-8",
    )


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


def configure_materials(scene, coefficient: float) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, material in scene.radio_materials.items():
        material.scattering_coefficient = coefficient
        value = np.asarray(material.scattering_coefficient.numpy()).reshape(-1)
        output[str(name)] = float(value[0])
    return output


def convert_cfr(raw: np.ndarray, count: int) -> np.ndarray:
    expected = (4, 1, 256, 1, NUM_SUBCARRIERS)
    if raw.ndim != 6 or raw.shape[0] != count or tuple(raw.shape[1:]) != expected:
        raise ValueError(f"unexpected CFR shape {raw.shape}")
    return np.transpose(raw[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def compact_spectra(channel: np.ndarray) -> dict[str, np.ndarray]:
    count = len(channel)
    shaped = channel.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    spatial = np.fft.fft2(shaped, axes=(2, 3))
    pas = np.square(np.abs(spatial), dtype=np.float32)
    pas = pas.reshape(count, 256, 4, 8, 24).mean(axis=4, dtype=np.float64)
    grouped_pas = np.transpose(pas, (0, 2, 3, 1)).reshape(count, 32, 256)

    cir = np.fft.ifft(channel, axis=-1)
    pdp = np.square(np.abs(cir), dtype=np.float32)
    grouped_pdp = pdp.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    grouped_pdp = grouped_pdp.mean(axis=(2, 3), dtype=np.float64).reshape(
        count, 8, NUM_SUBCARRIERS
    )

    joint = np.fft.ifft(spatial, axis=-1)
    joint_power = np.square(np.abs(joint), dtype=np.float32)
    joint_ad = joint_power.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    point_power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return {
        "grouped_pas": np.sqrt(np.maximum(grouped_pas, 0.0)).astype(np.float16),
        "grouped_pdp": np.sqrt(np.maximum(grouped_pdp, 0.0)).astype(np.float16),
        "joint_ad": np.sqrt(np.maximum(joint_ad, 0.0)).astype(np.float16),
        "log_power": np.log(np.maximum(point_power, TINY)).astype(np.float32),
    }


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
        ids = np.flatnonzero(site == site_value)
        xy = positions[ids, :2]
        center = np.mean(xy, axis=0)
        radius = np.linalg.norm(xy - center, axis=1)
        angle = np.mod(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]), 2 * np.pi)
        radial_bin = np.minimum((radius / max(float(radius.max()), 1e-6) * 4).astype(int), 3)
        angle_bin = np.minimum((angle / (2 * np.pi) * 8).astype(int), 7)
        strata = radial_bin * 8 + angle_bin
        site_selected: list[int] = []
        for stratum in np.unique(strata):
            candidates = ids[strata == stratum]
            quota = max(1, int(round(count_per_site * len(candidates) / len(ids))))
            quota = min(quota, len(candidates))
            site_selected.extend(rng.choice(candidates, size=quota, replace=False).tolist())
        if len(site_selected) > count_per_site:
            site_selected = rng.choice(site_selected, size=count_per_site, replace=False).tolist()
        elif len(site_selected) < count_per_site:
            remaining = np.setdiff1d(ids, np.asarray(site_selected, dtype=np.int64))
            extra = rng.choice(
                remaining, size=min(count_per_site - len(site_selected), len(remaining)), replace=False
            )
            site_selected.extend(extra.tolist())
        selected.append(np.asarray(sorted(site_selected), dtype=np.int64))
    return np.sort(np.concatenate(selected))


def output_paths(
    config: SpectrumConfig, split: str, point_count: int, selection_tag: str | None
) -> dict[str, Path]:
    suffix = selection_tag or f"n{point_count}"
    prefix = f"{config.name}_{split}_{suffix}"
    return {
        "ids": ARTIFACT_DIR / f"{prefix}_ids.npy",
        "grouped_pas": ARTIFACT_DIR / f"{prefix}_grouped_pas_f16.npy",
        "grouped_pdp": ARTIFACT_DIR / f"{prefix}_grouped_pdp_f16.npy",
        "joint_ad": ARTIFACT_DIR / f"{prefix}_joint_ad_f16.npy",
        "log_power": ARTIFACT_DIR / f"{prefix}_log_power.npy",
        "path_count": ARTIFACT_DIR / f"{prefix}_path_count.npy",
        "done": ARTIFACT_DIR / f"{prefix}_done.npy",
        "cfr_sample": ARTIFACT_DIR / f"{prefix}_cfr_sample.npz",
    }


def open_memmap(path: Path, dtype, shape, force: bool):
    if force:
        path.unlink(missing_ok=True)
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"cache mismatch {path}: {value.shape}/{value.dtype} != {shape}/{dtype}")
        return value
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value[...] = 0
    value.flush()
    return value


def generate_split(
    config: SpectrumConfig,
    split: str,
    positions: np.ndarray,
    site: np.ndarray,
    ids: np.ndarray,
    selection_tag: str | None,
    args: argparse.Namespace,
    mi,
    dr,
    rt_api,
) -> dict[str, object]:
    selected_positions = positions[ids].astype(np.float32)
    selected_site = site[ids].astype(np.int8)
    paths = output_paths(config, split, len(ids), selection_tag)
    if args.force:
        paths["ids"].unlink(missing_ok=True)
        paths["cfr_sample"].unlink(missing_ok=True)
    if paths["ids"].is_file():
        cached = np.load(paths["ids"])
        if not np.array_equal(cached, ids):
            raise ValueError(f"point ids changed for {paths['ids']}")
    else:
        np.save(paths["ids"], ids)

    grouped_pas = open_memmap(paths["grouped_pas"], np.float16, (len(ids), 32, 256), args.force)
    grouped_pdp = open_memmap(paths["grouped_pdp"], np.float16, (len(ids), 8, 192), args.force)
    joint_ad = open_memmap(paths["joint_ad"], np.float16, (len(ids), 16, 8, 192), args.force)
    log_power = open_memmap(paths["log_power"], np.float32, (len(ids),), args.force)
    path_count = open_memmap(paths["path_count"], np.int32, (len(ids),), args.force)
    done = open_memmap(paths["done"], np.bool_, (len(ids),), args.force)
    if np.all(done):
        return {"reused": True, "count": int(len(ids)), "outputs": {k: PATHS.relative(v) for k, v in paths.items() if v.is_file()}}

    load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies = rt_api
    setup, _, _, _, _ = load_positions()
    materials: dict[str, float] = {}
    cfr_ids: list[int] = []
    cfr_values: list[np.ndarray] = []
    samples = max(1, int(round(config.samples_per_src * args.budget_scale)))
    max_paths = max(1, int(round(config.max_num_paths_per_src * args.budget_scale)))
    started = time.time()

    for site_value in np.unique(selected_site):
        scene = load_scene(
            str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
        )
        scene.frequency = config.frequency_hz
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
        materials = configure_materials(scene, config.scattering_coefficient)
        position = setup.bs_positions[int(site_value)]
        tx = Transmitter(
            name=f"tx_site{int(site_value)}",
            position=mi.Point3f(*map(float, position)),
            look_at=tx_target(
                position,
                config.site_yaw_deg[int(site_value)],
                config.site_tilt_deg[int(site_value)],
                mi,
            ),
            power_dbm=44.0,
        )
        scene.add(tx)
        solver = PathSolver()
        receiver_names: list[str] = []
        rows = np.flatnonzero(selected_site == site_value)
        for batch_index, offset in enumerate(range(0, len(rows), args.batch_size)):
            batch_rows = rows[offset : offset + args.batch_size]
            if np.all(done[batch_rows]):
                continue
            for name in receiver_names:
                scene.remove(name)
            receiver_names = []
            for row in batch_rows:
                name = f"rx_{split}_{int(ids[row]):05d}"
                scene.add(
                    Receiver(
                        name=name,
                        position=mi.Point3f(*map(float, selected_positions[row])),
                        look_at=tx,
                    )
                )
                receiver_names.append(name)
            select_drjit_cuda_device(dr, args.cuda_device)
            traced = solver(
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
                seed=config.seed + 100_000 * int(site_value) + 1_000_000 * (split == "test") + batch_index,
            )
            raw = traced.cfr(
                mi.TensorXf(frequencies),
                normalize_delays=False,
                normalize=False,
                out_type="numpy",
            )
            channel = convert_cfr(np.asarray(raw), len(batch_rows))
            compact = compact_spectra(channel)
            grouped_pas[batch_rows] = compact["grouped_pas"]
            grouped_pdp[batch_rows] = compact["grouped_pdp"]
            joint_ad[batch_rows] = compact["joint_ad"]
            log_power[batch_rows] = compact["log_power"]
            path_count[batch_rows] = np.asarray(traced.valid.numpy(), dtype=bool).reshape(len(batch_rows), -1).sum(axis=1)
            done[batch_rows] = True
            if args.save_cfr_sample > 0 and len(cfr_ids) < args.save_cfr_sample:
                take = min(args.save_cfr_sample - len(cfr_ids), len(batch_rows))
                cfr_ids.extend(ids[batch_rows[:take]].tolist())
                cfr_values.append(channel[:take].copy())
            for value in (grouped_pas, grouped_pdp, joint_ad, log_power, path_count, done):
                value.flush()
            del traced, raw, channel, compact
            dr.sync_thread()
            completed = int(np.sum(done))
            print(
                f"{config.name} {split} site={site_value} {completed}/{len(ids)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        for name in receiver_names:
            scene.remove(name)
        del scene, solver
        dr.sync_thread()
    if cfr_values:
        np.savez_compressed(
            paths["cfr_sample"],
            ids=np.asarray(cfr_ids, dtype=np.int64),
            channel=np.concatenate(cfr_values).astype(np.complex64),
        )
    if not np.all(done):
        raise RuntimeError(f"incomplete Sionna cache for {config.name}/{split}")
    if not np.all(np.isfinite(grouped_pas)) or not np.all(np.isfinite(grouped_pdp)):
        raise FloatingPointError(f"non-finite spectra for {config.name}/{split}")
    return {
        "reused": False,
        "count": int(len(ids)),
        "seconds": float(time.time() - started),
        "samples_per_src": samples,
        "max_num_paths_per_src": max_paths,
        "mean_path_count": float(np.mean(path_count)),
        "zero_spectrum_count": int(np.sum(np.asarray(grouped_pdp).sum(axis=(1, 2)) == 0.0)),
        "materials": materials,
        "outputs": {k: PATHS.relative(v) for k, v in paths.items() if v.is_file()},
    }


def main() -> None:
    args = parse_args()
    if args.budget_scale <= 0.0:
        raise ValueError("--budget-scale must be positive")
    prepare_scene()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    rt_api = (load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies)

    _, train_pos, test_pos, train_site, test_site = load_positions()
    selected_configs = [config for config in CONFIGS if args.config in ("all", config.name)]
    selected_splits = ("train", "test") if args.split == "all" else (args.split,)
    results: dict[str, object] = {}
    for config in selected_configs:
        split_results: dict[str, object] = {}
        for split in selected_splits:
            positions = train_pos if split == "train" else test_pos
            site = train_site if split == "train" else test_site
            if args.ids_file is not None:
                loaded_ids = np.load(args.ids_file)
                if isinstance(loaded_ids, np.lib.npyio.NpzFile):
                    if args.ids_key not in loaded_ids.files:
                        raise KeyError(f"{args.ids_key} not found in {args.ids_file}")
                    ids = loaded_ids[args.ids_key].astype(np.int64)
                else:
                    ids = loaded_ids.astype(np.int64)
                if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(positions)):
                    raise ValueError(f"invalid ids in {args.ids_file} for {split}")
            else:
                ids = sample_ids(
                    positions,
                    site,
                    args.sample_count_per_site,
                    args.sample_seed + 10_000 * (split == "test"),
                )
            split_results[split] = generate_split(
                config,
                split,
                positions,
                site,
                ids,
                args.selection_tag,
                args,
                mi,
                dr,
                rt_api,
            )
        results[config.name] = {"configuration": asdict(config), "splits": split_results}

    payload = {
        "experiment": VERSION,
        "status": "complete",
        "purpose": "Strong Sionna upper-bound diagnostic for Round 2 PAS/PDP support",
        "environment": {
            "sionna_version": sionna.__version__,
            "mitsuba_variant": mi.variant(),
            "cuda_device": args.cuda_device,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene": PATHS.relative(SCENE_XML),
        "results": results,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
