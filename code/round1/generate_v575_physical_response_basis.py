#!/usr/bin/env python3
"""Generate paired Sionna PAS/PDP response fields for v575."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np

import generate_v549_multiseed_v207_pas_statistics as v549
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 575
VERSION = "v575_physical_response_basis"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR = CACHE_DIR / "physical_responses"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / "v575_physical_response_generation.json"
SELECTION = CACHE_DIR / "v575_selection.npz"
SCENE_XML = (
    PATHS.artifacts
    / "external"
    / "restored_sionna_official_etoile"
    / "etoile.xml"
)

BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
PAIR_SEEDS = (57_501, 57_503)
MATERIALS = ("concrete", "marble", "metal", "wood")
TINY = 1e-30


@dataclass(frozen=True)
class PhysicalConfig:
    frequency_hz: float = 6.693e9
    tx_azimuth_deg: float = -29.70
    tx_tilt_deg: float = -7.65
    tx_vertical_spacing: float = 4.0
    tx_horizontal_spacing: float = 0.5
    rx_vertical_spacing: float = 0.5
    max_depth: int = 16
    samples_per_src: int = 200_000
    max_num_paths_per_src: int = 200_000
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    scattering_coefficient: float = 0.0
    scattering_scope: str = "all"
    xpd_coefficient: float = 0.0
    material_name: str = ""
    material_scale: float = 1.0
    batch_size: int = 8


@dataclass(frozen=True)
class ResponsePair:
    name: str
    category: str
    interpretation: str
    minus: PhysicalConfig
    plus: PhysicalConfig


BASE = PhysicalConfig()


def pair(
    name: str,
    category: str,
    interpretation: str,
    minus: PhysicalConfig,
    plus: PhysicalConfig,
) -> ResponsePair:
    return ResponsePair(name, category, interpretation, minus, plus)


PAIRS = (
    pair(
        "frequency",
        "frequency",
        "carrier-frequency response from 6.0 to 7.2 GHz",
        replace(BASE, frequency_hz=6.0e9),
        replace(BASE, frequency_hz=7.2e9),
    ),
    pair(
        "tx_azimuth",
        "array",
        "BS horizontal pointing response across +/-4 degrees",
        replace(BASE, tx_azimuth_deg=-33.70),
        replace(BASE, tx_azimuth_deg=-25.70),
    ),
    pair(
        "tx_tilt",
        "array",
        "BS vertical pointing response across +/-4 degrees",
        replace(BASE, tx_tilt_deg=-11.65),
        replace(BASE, tx_tilt_deg=-3.65),
    ),
    pair(
        "tx_vertical_spacing",
        "array",
        "TX vertical aperture response from 3 to 5 wavelengths",
        replace(BASE, tx_vertical_spacing=3.0),
        replace(BASE, tx_vertical_spacing=5.0),
    ),
    pair(
        "tx_horizontal_spacing",
        "array",
        "TX horizontal aperture response from 0.4 to 0.6 wavelengths",
        replace(BASE, tx_horizontal_spacing=0.4),
        replace(BASE, tx_horizontal_spacing=0.6),
    ),
    pair(
        "rx_vertical_spacing",
        "array",
        "UE vertical aperture response from 0.35 to 0.65 wavelengths",
        replace(BASE, rx_vertical_spacing=0.35),
        replace(BASE, rx_vertical_spacing=0.65),
    ),
    pair(
        "path_depth",
        "path_lifecycle",
        "path birth and disappearance between depth 6 and depth 26",
        replace(BASE, max_depth=6),
        replace(BASE, max_depth=26),
    ),
    pair(
        "path_truncation",
        "path_lifecycle",
        "weak-path truncation response at fixed 500k sampling budget",
        replace(
            BASE,
            samples_per_src=500_000,
            max_num_paths_per_src=50_000,
            batch_size=4,
        ),
        replace(
            BASE,
            samples_per_src=500_000,
            max_num_paths_per_src=500_000,
            batch_size=4,
        ),
    ),
    pair(
        "scattering_strength",
        "scattering",
        "diffuse-scattering response from coefficient 0 to 0.30",
        BASE,
        replace(
            BASE,
            diffuse_reflection=True,
            scattering_coefficient=0.30,
        ),
    ),
    pair(
        "scattering_scope",
        "scattering",
        "response to suppressing diffuse scattering on metal surfaces",
        replace(
            BASE,
            diffuse_reflection=True,
            scattering_coefficient=0.15,
            scattering_scope="all",
        ),
        replace(
            BASE,
            diffuse_reflection=True,
            scattering_coefficient=0.15,
            scattering_scope="nonmetal",
        ),
    ),
    pair(
        "diffraction",
        "diffraction",
        "ordinary diffraction path birth response",
        BASE,
        replace(BASE, diffraction=True),
    ),
    pair(
        "edge_diffraction",
        "diffraction",
        "edge-diffraction response relative to ordinary diffraction",
        replace(BASE, diffraction=True),
        replace(BASE, diffraction=True, edge_diffraction=True, batch_size=4),
    ),
    pair(
        "refraction",
        "path_mechanism",
        "transmission/refraction path presence response",
        replace(BASE, refraction=False),
        BASE,
    ),
    pair(
        "los",
        "path_mechanism",
        "direct-path presence response",
        replace(BASE, los=False),
        BASE,
    ),
    pair(
        "specular_reflection",
        "path_mechanism",
        "specular-reflection path presence response",
        replace(BASE, specular_reflection=False),
        BASE,
    ),
    pair(
        "physical_array",
        "array",
        "full physical-array propagation response relative to synthetic array",
        replace(
            BASE,
            synthetic_array=True,
            max_num_paths_per_src=100_000,
            batch_size=2,
        ),
        replace(
            BASE,
            synthetic_array=False,
            max_num_paths_per_src=100_000,
            batch_size=2,
        ),
    ),
    pair(
        "concrete_material",
        "material",
        "concrete Fresnel-response perturbation",
        replace(BASE, material_name="concrete", material_scale=0.6),
        replace(BASE, material_name="concrete", material_scale=1.4),
    ),
    pair(
        "marble_material",
        "material",
        "marble Fresnel-response perturbation",
        replace(BASE, material_name="marble", material_scale=0.6),
        replace(BASE, material_name="marble", material_scale=1.4),
    ),
    pair(
        "wood_material",
        "material",
        "wood Fresnel-response perturbation",
        replace(BASE, material_name="wood", material_scale=0.6),
        replace(BASE, material_name="wood", material_scale=1.4),
    ),
    pair(
        "cross_polar_scattering",
        "polarization",
        "diffuse cross-polarization response from XPD 0 to 0.40",
        replace(
            BASE,
            diffuse_reflection=True,
            scattering_coefficient=0.15,
            xpd_coefficient=0.0,
        ),
        replace(
            BASE,
            diffuse_reflection=True,
            scattering_coefficient=0.15,
            xpd_coefficient=0.40,
        ),
    ),
)


PAS_SHAPES = {
    "sh": (NUM_SUBCARRIERS, 16),
    "psh": (2, NUM_SUBCARRIERS, 16),
    "shue": (NUM_SUBCARRIERS, 16, 4),
    "shv": (NUM_SUBCARRIERS, 16, 8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        choices=("all", *(item.name for item in PAIRS)),
        default="all",
    )
    parser.add_argument(
        "--seed",
        choices=("all", *(str(value) for value in PAIR_SEEDS)),
        default="all",
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_drjit_cuda_device(dr, device: int | None) -> None:
    if device is None:
        return
    library_path = Path(dr.__file__).with_name("libdrjit-core.so")
    library = ctypes.CDLL(str(library_path))
    setter = getattr(library, "_Z19jit_cuda_set_devicei")
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    setter(int(device))


def create_selection() -> dict[str, object]:
    if SELECTION.is_file():
        cached = np.load(SELECTION)
        return {
            "selected_ids": cached["selected_ids"].astype(np.int64),
            "fold_id": cached["fold_id"].astype(np.int16),
            "regimes": cached["regimes"].astype("U8"),
            "replacements": cached["replacements"].astype(np.int64),
        }

    pilot = np.load(PATHS.cache_dir(551) / "v551_pilot_selection.npz")
    selected_ids = pilot["selected_ids"].astype(np.int64).copy()
    fold_id = pilot["fold_id"].astype(np.int16).copy()
    regimes = pilot["regimes"].astype("U8").copy()
    all_fold = np.load(
        PATHS.cache_dir(434) / "v434_sionna_joint_sh_calibration_oof.npz"
    )["fold_id"].astype(np.int16)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    all_regimes = regime_of(geometry["train_blocked"]).astype("U8")
    selected = set(map(int, selected_ids))
    replacements: list[tuple[int, int, int]] = []
    for row in np.flatnonzero(power[selected_ids] == 0.0):
        old_id = int(selected_ids[row])
        candidates = np.flatnonzero(
            (all_fold == fold_id[row])
            & (all_regimes == regimes[row])
            & (power > 0.0)
        )
        candidates = np.asarray(
            [value for value in candidates if int(value) not in selected],
            dtype=np.int64,
        )
        distance = np.linalg.norm(
            positions[candidates, :2] - positions[old_id, :2], axis=1
        )
        new_id = int(candidates[np.argmin(distance)])
        selected.remove(old_id)
        selected.add(new_id)
        selected_ids[row] = new_id
        replacements.append((int(row), old_id, new_id))

    if len(np.unique(selected_ids)) != 400 or np.any(power[selected_ids] == 0.0):
        raise RuntimeError("v575 selection must contain 400 unique nonzero points")
    replacement_array = np.asarray(replacements, dtype=np.int64).reshape(-1, 3)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SELECTION,
        selected_ids=selected_ids,
        fold_id=fold_id,
        regimes=regimes,
        replacements=replacement_array,
    )
    return {
        "selected_ids": selected_ids,
        "fold_id": fold_id,
        "regimes": regimes,
        "replacements": replacement_array,
    }


def response_paths(item: ResponsePair, seed: int) -> dict[str, Path]:
    root = OUT_DIR / item.name / f"seed_{seed}"
    outputs = {
        name: root / f"{name}_sqrt_response.npy" for name in PAS_SHAPES
    }
    outputs["pdp"] = root / "pdp_sqrt_response.npy"
    outputs["summary"] = root / "summary.npz"
    outputs["done"] = root / "done.json"
    return outputs


def completed(item: ResponsePair, seed: int) -> bool:
    paths = response_paths(item, seed)
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        done = json.loads(paths["done"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return done.get("pair") == item.name and done.get("seed") == seed


def target_point(config: PhysicalConfig, mi):
    radius = 140.0
    azimuth = math.radians(config.tx_azimuth_deg)
    tilt = math.radians(config.tx_tilt_deg)
    return mi.Point3f(
        float(BS_POSITION[0] + radius * math.cos(azimuth)),
        float(BS_POSITION[1] + radius * math.sin(azimuth)),
        float(BS_POSITION[2] + radius * math.tan(tilt)),
    )


def configure_materials(scene, config: PhysicalConfig) -> dict[str, dict[str, float]]:
    audit: dict[str, dict[str, float]] = {}
    scene_materials = {
        name: material
        for name, material in scene.radio_materials.items()
        if name in MATERIALS
    }
    if set(scene_materials) != set(MATERIALS):
        raise RuntimeError(f"unexpected material set: {sorted(scene_materials)}")
    for name, material in scene_materials.items():
        relative_permittivity = float(
            np.asarray(material.relative_permittivity).reshape(-1)[0]
        )
        conductivity = float(np.asarray(material.conductivity).reshape(-1)[0])
        if name == config.material_name:
            relative_permittivity *= config.material_scale
            conductivity *= config.material_scale
            material.relative_permittivity = relative_permittivity
            material.conductivity = conductivity
        scatter_active = config.scattering_scope == "all" or name != "metal"
        material.scattering_coefficient = (
            config.scattering_coefficient if scatter_active else 0.0
        )
        material.xpd_coefficient = config.xpd_coefficient
        audit[name] = {
            "relative_permittivity": relative_permittivity,
            "conductivity": conductivity,
            "scattering_coefficient": float(
                config.scattering_coefficient if scatter_active else 0.0
            ),
            "xpd_coefficient": float(config.xpd_coefficient),
        }
    return audit


def build_runtime(config: PhysicalConfig, label: str, runtime: dict[str, object]):
    mi = runtime["mi"]
    scene = runtime["load_scene"](
        str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
    )
    scene.frequency = config.frequency_hz
    materials = configure_materials(scene, config)
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
        vertical_spacing=config.rx_vertical_spacing,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )
    tx = runtime["Transmitter"](
        name=f"tx_v575_{label}",
        position=mi.Point3f(*map(float, BS_POSITION)),
        look_at=target_point(config, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    return {
        "scene": scene,
        "tx": tx,
        "solver": runtime["PathSolver"](),
        "receiver_names": [],
        "materials": materials,
    }


def normalized_linkwise_pdp(channel: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(channel, axis=-1)
    power = np.abs(cir).astype(np.float32) ** 2
    total = np.sum(power, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
    output = power / np.maximum(total, TINY)
    output[total[..., 0] <= TINY] = 0.0
    return output.reshape(len(channel), 1024, NUM_SUBCARRIERS).astype(np.float32)


def spectra(channel: np.ndarray) -> dict[str, np.ndarray]:
    output = {
        name: value.reshape(len(channel), -1).astype(np.float32)
        for name, value in v549.pas_marginals(channel).items()
    }
    output["pdp"] = normalized_linkwise_pdp(channel)
    return output


def solve_batch(
    holder: dict[str, object],
    config: PhysicalConfig,
    positions: np.ndarray,
    point_ids: np.ndarray,
    seed: int,
    batch_start: int,
    label: str,
    runtime: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    scene = holder["scene"]
    tx = holder["tx"]
    for name in holder["receiver_names"]:
        scene.remove(name)
    receiver_names: list[str] = []
    mi = runtime["mi"]
    for local, position in enumerate(positions):
        point_id = int(point_ids[local])
        name = f"rx_v575_{label}_{seed}_{point_id:05d}"
        scene.add(
            runtime["Receiver"](
                name=name,
                position=mi.Point3f(*map(float, position)),
                look_at=tx,
            )
        )
        receiver_names.append(name)
    holder["receiver_names"] = receiver_names
    select_drjit_cuda_device(runtime["dr"], runtime["cuda_device"])
    paths = holder["solver"](
        scene,
        max_depth=config.max_depth,
        max_num_paths_per_src=config.max_num_paths_per_src,
        samples_per_src=config.samples_per_src,
        synthetic_array=config.synthetic_array,
        los=config.los,
        specular_reflection=config.specular_reflection,
        diffuse_reflection=config.diffuse_reflection,
        refraction=config.refraction,
        diffraction=config.diffraction,
        edge_diffraction=config.edge_diffraction,
        diffraction_lit_region=config.diffraction_lit_region,
        seed=seed + batch_start,
    )
    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    path_count = valid.reshape(len(positions), -1).sum(axis=1).astype(np.int32)
    select_drjit_cuda_device(runtime["dr"], runtime["cuda_device"])
    raw = v549.chunked_cfr(paths, runtime["frequencies"], mi)
    channel = v549.convert_cfr(np.asarray(raw), len(positions))
    point_power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
    result = spectra(channel)
    del paths, valid, raw, channel
    runtime["dr"].sync_thread()
    return result, path_count, point_power


def open_outputs(item: ResponsePair, seed: int, rows: int, force: bool):
    paths = response_paths(item, seed)
    paths["done"].parent.mkdir(parents=True, exist_ok=True)
    if force:
        for path in paths.values():
            path.unlink(missing_ok=True)
    outputs = {}
    for name, local_shape in PAS_SHAPES.items():
        outputs[name] = np.lib.format.open_memmap(
            paths[name],
            mode="w+",
            dtype=np.float16,
            shape=(rows, int(np.prod(local_shape))),
        )
    outputs["pdp"] = np.lib.format.open_memmap(
        paths["pdp"],
        mode="w+",
        dtype=np.float16,
        shape=(rows, 1024, NUM_SUBCARRIERS),
    )
    return paths, outputs


def generate_pair_seed(
    item: ResponsePair,
    seed: int,
    selection: dict[str, object],
    runtime: dict[str, object],
    force: bool,
) -> dict[str, object]:
    if not force and completed(item, seed):
        return {
            "pair": item.name,
            "seed": seed,
            "status": "reused",
            "outputs": {
                name: PATHS.relative(path)
                for name, path in response_paths(item, seed).items()
            },
        }
    selected_ids = selection["selected_ids"]
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    positions = train_positions[selected_ids]
    rows = len(positions)
    paths, outputs = open_outputs(item, seed, rows, force=True)
    summaries = {
        "minus_path_count": np.zeros(rows, dtype=np.int32),
        "plus_path_count": np.zeros(rows, dtype=np.int32),
        "minus_log_power": np.zeros(rows, dtype=np.float32),
        "plus_log_power": np.zeros(rows, dtype=np.float32),
    }
    for name in (*PAS_SHAPES, "pdp"):
        summaries[f"{name}_response_rms"] = np.zeros(rows, dtype=np.float32)

    minus_runtime = build_runtime(item.minus, f"{item.name}_minus", runtime)
    plus_runtime = build_runtime(item.plus, f"{item.name}_plus", runtime)
    batch_size = min(item.minus.batch_size, item.plus.batch_size)
    started = time.time()
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        local_positions = positions[start:stop]
        local_ids = selected_ids[start:stop]
        minus, minus_count, minus_power = solve_batch(
            minus_runtime,
            item.minus,
            local_positions,
            local_ids,
            seed,
            start,
            f"{item.name}_minus",
            runtime,
        )
        plus, plus_count, plus_power = solve_batch(
            plus_runtime,
            item.plus,
            local_positions,
            local_ids,
            seed,
            start,
            f"{item.name}_plus",
            runtime,
        )
        summaries["minus_path_count"][start:stop] = minus_count
        summaries["plus_path_count"][start:stop] = plus_count
        summaries["minus_log_power"][start:stop] = np.log(
            np.maximum(minus_power, TINY)
        ).astype(np.float32)
        summaries["plus_log_power"][start:stop] = np.log(
            np.maximum(plus_power, TINY)
        ).astype(np.float32)
        for name in outputs:
            response = np.float32(0.5) * (
                np.sqrt(np.maximum(plus[name], 0.0))
                - np.sqrt(np.maximum(minus[name], 0.0))
            )
            outputs[name][start:stop] = response.astype(np.float16)
            summaries[f"{name}_response_rms"][start:stop] = np.sqrt(
                np.mean(response.astype(np.float64) ** 2, axis=tuple(range(1, response.ndim)))
            ).astype(np.float32)
        if stop == rows or stop % 40 == 0:
            for value in outputs.values():
                value.flush()
            print(
                f"v575 pair={item.name} seed={seed} {stop}/{rows} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    for holder in (minus_runtime, plus_runtime):
        for name in holder["receiver_names"]:
            holder["scene"].remove(name)
    runtime["dr"].sync_thread()
    np.savez_compressed(paths["summary"], **summaries)
    output_shapes = {}
    for name, output in outputs.items():
        output.flush()
        output_shapes[name] = list(output.shape)
        if not np.all(np.isfinite(output)):
            raise FloatingPointError(f"non-finite v575 response: {item.name} {seed} {name}")
    result = {
        "pair": item.name,
        "seed": seed,
        "status": "complete",
        "seconds": float(time.time() - started),
        "minus": asdict(item.minus),
        "plus": asdict(item.plus),
        "output_shapes": output_shapes,
        "outputs": {
            name: PATHS.relative(path) for name, path in paths.items() if name != "done"
        },
    }
    atomic_json(paths["done"], result)
    return result


def load_report() -> dict[str, object]:
    if not REPORT.is_file():
        return {}
    return json.loads(REPORT.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    for path in (
        SCENE_XML,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Train_Channel.npy",
        PATHS.cache_dir(551) / "v551_pilot_selection.npz",
        PATHS.cache_dir(434) / "v434_sionna_joint_sh_calibration_oof.npz",
        PATHS.shared_features / "height_geometry_cache.npz",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selection = create_selection()

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    select_drjit_cuda_device(dr, args.cuda_device)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v575 requires Sionna 2.0.1, got {sionna.__version__}")
    select_drjit_cuda_device(dr, args.cuda_device)
    runtime = {
        "mi": mi,
        "dr": dr,
        "load_scene": load_scene,
        "PlanarArray": PlanarArray,
        "Receiver": Receiver,
        "Transmitter": Transmitter,
        "PathSolver": PathSolver,
        "frequencies": np.asarray(
            subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
        ).astype(np.float32),
        "cuda_device": args.cuda_device,
    }
    selected_pairs = [item for item in PAIRS if args.pair in ("all", item.name)]
    selected_seeds = [
        seed for seed in PAIR_SEEDS if args.seed in ("all", str(seed))
    ]
    report = load_report()
    results = dict(report.get("results", {}))
    for item in selected_pairs:
        pair_results = dict(results.get(item.name, {}))
        for seed in selected_seeds:
            pair_results[str(seed)] = generate_pair_seed(
                item, seed, selection, runtime, args.force
            )
            results[item.name] = pair_results
            report = {
                "method": VERSION,
                "status": "physical_response_generation",
                "primary_hypothesis": (
                    "Paired changes of Sionna physics provide directional PAS/PDP response "
                    "fields that are more transferable than selecting one absolute simulator output."
                ),
                "selection": {
                    "count": 400,
                    "source": "v551 stratified 80-per-spatial-fold pilot",
                    "zero_power_replacements": selection["replacements"].tolist(),
                    "path": PATHS.relative(SELECTION),
                },
                "response_definition": (
                    "0.5 * (sqrt(normalized plus spectrum) - "
                    "sqrt(normalized minus spectrum))"
                ),
                "pair_seeds": list(PAIR_SEEDS),
                "pairs": [asdict(value) for value in PAIRS],
                "environment": {
                    "sionna": sionna.__version__,
                    "mitsuba_variant": mi.variant(),
                    "requested_cuda_device": args.cuda_device,
                    "single_gpu_required": True,
                },
                "results": results,
                "report": PATHS.relative(REPORT),
            }
            atomic_json(REPORT, report)
    print(
        json.dumps(
            {
                "report": PATHS.relative(REPORT),
                "completed_pairs": sorted(results),
                "selection": PATHS.relative(SELECTION),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
