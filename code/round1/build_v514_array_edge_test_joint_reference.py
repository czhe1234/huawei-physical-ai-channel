#!/usr/bin/env python3
"""Build compact test-coordinate joint S-H references for v514."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v504_scene_topology_sionna_inverse as v504
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 514
VERSION = "v514_targeted_array_edge_pas_diagnostic"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
CONFIG_DIR = CACHE_DIR / "reference_configs"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
REFERENCE = CACHE_DIR / "v514_array_edge_test_joint_reference.npz"
REFERENCE_REPORT = REPORT_DIR / "v514_array_edge_test_joint_reference.json"

BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
FREQUENCY_HZ = 6.693e9
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
MAX_DEPTH = 16
SAMPLES_PER_SRC = 3_000_000
MAX_PATHS_PER_SRC = 100_000
SEED = 42
TINY = 1e-30


@dataclass(frozen=True)
class ReferenceConfig:
    name: str
    synthetic_array: bool
    edge_diffraction: bool


CONFIGS = (
    ReferenceConfig("synthetic_base", True, False),
    ReferenceConfig("nonsynthetic_edge", False, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--point-limit", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def cache_prefix(config: ReferenceConfig, point_count: int) -> str:
    return f"{config.name}_points{point_count}"


def config_paths(config: ReferenceConfig, point_count: int) -> dict[str, object]:
    prefix = cache_prefix(config, point_count)
    return {
        "sh": CONFIG_DIR / f"{prefix}_sh.npy",
        "log_power": CONFIG_DIR / f"{prefix}_log_power.npy",
        "path_count": CONFIG_DIR / f"{prefix}_path_count.npy",
        "done": CONFIG_DIR / f"{prefix}_done.npy",
        "report": REPORT_DIR / "configs" / f"{prefix}.json",
    }


def open_or_create(path, dtype, shape):
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"cache mismatch for {path}: {value.shape} {value.dtype}")
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
    sh = power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    sh /= np.maximum(sh.sum(axis=(1, 2), keepdims=True), TINY)
    return sh.astype(np.float32)


def run_config(
    config: ReferenceConfig,
    indices: np.ndarray,
    positions: np.ndarray,
    batch_size: int,
    mi,
    dr,
    rt_api,
) -> dict[str, object]:
    paths_out = config_paths(config, len(indices))
    shape_sh = (len(indices), NUM_SUBCARRIERS, 16)
    sh_out = open_or_create(paths_out["sh"], np.float32, shape_sh)
    log_power_out = open_or_create(paths_out["log_power"], np.float32, (len(indices),))
    path_count_out = open_or_create(paths_out["path_count"], np.int32, (len(indices),))
    done = open_or_create(paths_out["done"], np.bool_, (len(indices),))
    if np.all(done) and paths_out["report"].is_file():
        return json.loads(paths_out["report"].read_text(encoding="utf-8"))

    load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies = rt_api
    scene = load_scene(
        str(SCENE_XML),
        merge_shapes=True,
        remove_duplicate_vertices=True,
    )
    scene.frequency = FREQUENCY_HZ
    materials = v504.material_audit(scene)
    if set(materials) != {"concrete", "marble", "metal", "wood"}:
        raise RuntimeError(f"unexpected materials: {materials}")
    if any(
        item["scattering_coefficient"] != 0.0 or item["xpd_coefficient"] != 0.0
        for item in materials.values()
    ):
        raise RuntimeError(f"fixed material audit failed: {materials}")

    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=0.5,
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
    search = v501.SearchConfig(
        name=config.name,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-9.53,
        rx_orientation="track_bs",
        synthetic_array=config.synthetic_array,
        frequency_hz=FREQUENCY_HZ,
        max_depth=MAX_DEPTH,
        diffraction=config.edge_diffraction,
        edge_diffraction=config.edge_diffraction,
        samples_per_src=SAMPLES_PER_SRC,
        max_num_paths_per_src=MAX_PATHS_PER_SRC,
        seed=SEED,
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*[float(value) for value in BS_POSITION]),
        look_at=v501.tx_target(search, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()
    receiver_names: list[str] = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(indices), batch_size)):
        end = min(start + batch_size, len(indices))
        if np.all(done[start:end]):
            continue
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        batch_ids = indices[start:end]
        for point_id in batch_ids:
            name = f"rx_{int(point_id):05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(
                        *[float(value) for value in positions[point_id]]
                    ),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        paths = solver(
            scene,
            max_depth=MAX_DEPTH,
            max_num_paths_per_src=MAX_PATHS_PER_SRC,
            samples_per_src=SAMPLES_PER_SRC,
            synthetic_array=config.synthetic_array,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=config.edge_diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=True,
            seed=SEED + batch_index,
        )
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        channel = v501.convert_cfr(np.asarray(raw), len(batch_ids))
        sh_out[start:end] = joint_sh(channel)
        energy = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
        log_power_out[start:end] = np.log(np.maximum(energy, TINY)).astype(np.float32)
        path_count_out[start:end] = valid.reshape(len(batch_ids), -1).sum(axis=1)
        done[start:end] = True
        sh_out.flush()
        log_power_out.flush()
        path_count_out.flush()
        done.flush()
        del paths, valid, raw, channel
        dr.sync_thread()
        if end == len(indices) or (batch_index + 1) % 10 == 0:
            print(
                f"v514 {config.name}: {end}/{len(indices)} "
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
        raise RuntimeError(f"incomplete reference cache for {config.name}")
    if not np.all(np.isfinite(sh_out)) or not np.all(np.isfinite(log_power_out)):
        raise FloatingPointError(f"non-finite reference for {config.name}")
    report = {
        "method": VERSION,
        "config": asdict(config),
        "point_count": int(len(indices)),
        "samples_per_src": SAMPLES_PER_SRC,
        "max_num_paths_per_src": MAX_PATHS_PER_SRC,
        "max_depth": MAX_DEPTH,
        "frequency_hz": FREQUENCY_HZ,
        "materials": materials,
        "seconds_this_process": float(time.time() - started),
        "mean_path_count": float(np.mean(path_count_out)),
        "outputs": {key: PATHS.relative(value) for key, value in paths_out.items()},
    }
    paths_out["report"].parent.mkdir(parents=True, exist_ok=True)
    paths_out["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v514 requires Sionna 2.0.1, got {sionna.__version__}")
    frequencies = np.asarray(
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
    ).astype(np.float32)
    rt_api = (load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies)

    positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["test_blocked"]).astype("U8")
    indices = np.flatnonzero(regimes == "heavy").astype(np.int64)
    default_limit = 4 if args.stage == "smoke" else len(indices)
    point_limit = min(args.point_limit or default_limit, len(indices))
    indices = indices[:point_limit]

    if args.force:
        for config in CONFIGS:
            for path in config_paths(config, len(indices)).values():
                path.unlink(missing_ok=True)
        if args.stage == "full":
            REFERENCE.unlink(missing_ok=True)
            REFERENCE_REPORT.unlink(missing_ok=True)

    reports = [
        run_config(
            config,
            indices,
            positions,
            args.batch_size,
            mi,
            dr,
            rt_api,
        )
        for config in CONFIGS
    ]
    base_paths = config_paths(CONFIGS[0], len(indices))
    combo_paths = config_paths(CONFIGS[1], len(indices))
    output = (
        REFERENCE
        if args.stage == "full"
        else CACHE_DIR / f"v514_array_edge_test_joint_reference_smoke{len(indices)}.npz"
    )
    np.savez_compressed(
        output,
        indices=indices,
        regimes=regimes[indices],
        baseline_sh=np.asarray(np.load(base_paths["sh"], mmap_mode="r")),
        combo_sh=np.asarray(np.load(combo_paths["sh"], mmap_mode="r")),
        baseline_log_power=np.asarray(
            np.load(base_paths["log_power"], mmap_mode="r")
        ),
        combo_log_power=np.asarray(
            np.load(combo_paths["log_power"], mmap_mode="r")
        ),
        baseline_path_count=np.asarray(
            np.load(base_paths["path_count"], mmap_mode="r")
        ),
        combo_path_count=np.asarray(
            np.load(combo_paths["path_count"], mmap_mode="r")
        ),
    )
    with np.load(output) as data:
        for name in data.files:
            value = np.asarray(data[name])
            if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                raise FloatingPointError(f"non-finite output {name}")
    report = {
        "method": VERSION,
        "status": "reference_complete" if args.stage == "full" else "smoke_complete",
        "stage": args.stage,
        "compliance_boundary": {
            "diagnostic_only": True,
            "reason": "runs Sionna at official test coordinates",
        },
        "point_count": int(len(indices)),
        "indices": indices.tolist(),
        "environment": {
            "sionna": sionna.__version__,
            "mitsuba_variant": mi.variant(),
            "visible_gpu_count": 1,
        },
        "configs": reports,
        "output": PATHS.relative(output),
    }
    report_path = (
        REFERENCE_REPORT
        if args.stage == "full"
        else REPORT_DIR / f"v514_array_edge_test_joint_reference_smoke{len(indices)}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": report["output"]}, indent=2))


if __name__ == "__main__":
    main()
