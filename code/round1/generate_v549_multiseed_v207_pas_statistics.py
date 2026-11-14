#!/usr/bin/env python3
"""Generate streaming multi-seed PAS statistics for the v207 Sionna setup."""

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


VERSION_NUMBER = 549
VERSION = "v549_multiseed_v207_pas_statistics"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER) / "multiseed_pas"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
STATE = OUT_DIR / "state.json"
SCENE_XML = (
    PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
)
TRAIN_SEED42 = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
TEST_SEED42 = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
NUM_SUBCARRIERS = 192
NUM_HORIZONTAL = 16
TINY = 1e-30

# These seeds were fixed before looking at any result from this experiment.
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
MARGINALS = ("sh", "psh", "shue", "shv")


@dataclass(frozen=True)
class Configuration:
    frequency_hz: float = 6.0e9
    tx_azimuth_deg: float = -30.0
    tx_tilt_deg: float = -9.53
    max_depth: int = 16
    samples_per_src: int = 1_000_000
    max_num_paths_per_src: int = 1_000_000
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    tx_vertical_spacing: float = 0.5
    tx_horizontal_spacing: float = 0.5
    seed_policy: str = "realization_seed + batch_start"


CONFIG = Configuration()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-seeds", type=int, default=len(SEEDS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def initial_state() -> dict[str, object]:
    return {
        "version": VERSION,
        "configuration": asdict(CONFIG),
        "seed_order": list(SEEDS),
        "splits": {
            "train": {"count": 0, "completed_seeds": []},
            "test": {"count": 0, "completed_seeds": []},
        },
    }


def load_state() -> dict[str, object]:
    expected = initial_state()
    if not STATE.is_file():
        atomic_json(STATE, expected)
        return expected
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("configuration") != expected["configuration"]:
        raise ValueError("v549 configuration changed after generation started")
    if state.get("seed_order") != expected["seed_order"]:
        raise ValueError("v549 seed order changed after generation started")
    return state


def marginal_shape(split: str, name: str, rows: int) -> tuple[int, ...]:
    del split
    shapes = {
        "sh": (rows, NUM_SUBCARRIERS, NUM_HORIZONTAL),
        "psh": (rows, 2, NUM_SUBCARRIERS, NUM_HORIZONTAL),
        "shue": (rows, NUM_SUBCARRIERS, NUM_HORIZONTAL, 4),
        "shv": (rows, NUM_SUBCARRIERS, NUM_HORIZONTAL, 8),
    }
    return shapes[name]


def open_output(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def normalize(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 1e-12)
    output /= np.maximum(
        np.sum(output, axis=axes, keepdims=True, dtype=np.float64), TINY
    )
    return output.astype(np.float32)


def pas_marginals(channel: np.ndarray) -> dict[str, np.ndarray]:
    count = len(channel)
    array = channel.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    spectrum = np.fft.fft2(array.transpose(0, 1, 5, 2, 4, 3), axes=(2, 3))
    power = np.abs(spectrum).astype(np.float32) ** 2
    total = np.sum(power, axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
    support = power / np.maximum(total[:, None, None, None, None, None], TINY)
    return {
        "sh": normalize(support.sum(axis=(1, 4, 5), dtype=np.float64), (1, 2)),
        "psh": normalize(support.sum(axis=(4, 5), dtype=np.float64), (1, 2, 3)),
        "shue": normalize(support.sum(axis=(1, 5), dtype=np.float64), (1, 2, 3)),
        "shv": normalize(support.sum(axis=(1, 4), dtype=np.float64), (1, 2, 3)),
    }


def complex_coherence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    flat_left = left.reshape(len(left), -1).astype(np.complex128)
    flat_right = right.reshape(len(right), -1).astype(np.complex128)
    inner = np.sum(np.conj(flat_left) * flat_right, axis=1)
    denominator = np.sqrt(
        np.sum(np.abs(flat_left) ** 2, axis=1) * np.sum(np.abs(flat_right) ** 2, axis=1)
    )
    return (np.abs(inner) / np.maximum(denominator, TINY)).astype(np.float32)


def convert_cfr(raw: np.ndarray, count: int) -> np.ndarray:
    expected = (count, 4, 1, 256, 1, NUM_SUBCARRIERS)
    if raw.shape != expected:
        raise ValueError(f"unexpected CFR shape {raw.shape}, expected {expected}")
    return np.transpose(raw[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


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


def target_point(mi):
    radius = 140.0
    azimuth = math.radians(CONFIG.tx_azimuth_deg)
    tilt = math.radians(CONFIG.tx_tilt_deg)
    return mi.Point3f(
        float(BS_POSITION[0] + radius * math.cos(azimuth)),
        float(BS_POSITION[1] + radius * math.sin(azimuth)),
        float(BS_POSITION[2] + radius * math.tan(tilt)),
    )


def prepare_runtime(args: argparse.Namespace):
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
        raise RuntimeError(f"v549 requires Sionna 2.0.1, got {sionna.__version__}")
    frequencies = np.asarray(subcarrier_frequencies(NUM_SUBCARRIERS, 30_000.0)).astype(
        np.float32
    )
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


def generate_seed_sample(
    *,
    split: str,
    seed: int,
    positions: np.ndarray,
    args: argparse.Namespace,
    runtime: dict[str, object] | None,
) -> tuple[dict[str, Path], dict[str, object]]:
    rows = len(positions)
    prefix = OUT_DIR / "temporary" / f"{split}_seed{seed}"
    done_path = prefix.with_name(prefix.name + "_done.json")
    paths = {name: prefix.with_name(prefix.name + f"_{name}.npy") for name in MARGINALS}
    coherence_path = OUT_DIR / "seed_diagnostics" / f"{split}_seed{seed}_coherence.npy"
    log_power_path = OUT_DIR / "seed_diagnostics" / f"{split}_seed{seed}_log_power.npy"
    if done_path.is_file() and all(path.is_file() for path in paths.values()):
        return paths, json.loads(done_path.read_text(encoding="utf-8"))

    for path in (*paths.values(), coherence_path, log_power_path, done_path):
        path.unlink(missing_ok=True)
    outputs = {
        name: open_output(path, marginal_shape(split, name, rows))
        for name, path in paths.items()
    }
    coherence_path.parent.mkdir(parents=True, exist_ok=True)
    coherence = np.lib.format.open_memmap(
        coherence_path, mode="w+", dtype=np.float32, shape=(rows,)
    )
    log_power = np.lib.format.open_memmap(
        log_power_path, mode="w+", dtype=np.float32, shape=(rows,)
    )
    seed42_path = TRAIN_SEED42 if split == "train" else TEST_SEED42
    seed42 = np.load(seed42_path, mmap_mode="r")
    if seed42.shape != (rows, 256, 4, NUM_SUBCARRIERS):
        raise ValueError(f"seed42 reference mismatch: {seed42_path} {seed42.shape}")

    scene = None
    receiver_names: list[str] = []
    path_count = np.zeros(rows, dtype=np.int32)
    if seed != 42:
        if runtime is None:
            raise RuntimeError("Sionna runtime is required for non-reused seeds")
        mi = runtime["mi"]
        scene = runtime["load_scene"](str(SCENE_XML), merge_shapes=True)
        scene.frequency = CONFIG.frequency_hz
        scene.tx_array = runtime["PlanarArray"](
            num_rows=8,
            num_cols=16,
            vertical_spacing=CONFIG.tx_vertical_spacing,
            horizontal_spacing=CONFIG.tx_horizontal_spacing,
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
        transmitter = runtime["Transmitter"](
            name=f"tx_v549_seed{seed}",
            position=mi.Point3f(*map(float, BS_POSITION)),
            look_at=target_point(mi),
            power_dbm=44.0,
        )
        scene.add(transmitter)
        solver = runtime["PathSolver"]()

    started = time.time()
    for start in range(0, rows, args.batch_size):
        end = min(start + args.batch_size, rows)
        if seed == 42:
            channel = np.asarray(seed42[start:end], dtype=np.complex64)
        else:
            assert runtime is not None and scene is not None
            mi = runtime["mi"]
            for name in receiver_names:
                scene.remove(name)
            receiver_names = []
            for index, position in enumerate(positions[start:end], start=start):
                name = f"rx_{split}_{seed}_{index:05d}"
                scene.add(
                    runtime["Receiver"](
                        name=name,
                        position=mi.Point3f(*map(float, position)),
                        look_at=transmitter,
                    )
                )
                receiver_names.append(name)
            select_drjit_cuda_device(runtime["dr"], args.cuda_device)
            solved = solver(
                scene,
                max_depth=CONFIG.max_depth,
                max_num_paths_per_src=CONFIG.max_num_paths_per_src,
                samples_per_src=CONFIG.samples_per_src,
                synthetic_array=CONFIG.synthetic_array,
                los=CONFIG.los,
                specular_reflection=CONFIG.specular_reflection,
                diffuse_reflection=CONFIG.diffuse_reflection,
                refraction=CONFIG.refraction,
                diffraction=CONFIG.diffraction,
                edge_diffraction=CONFIG.edge_diffraction,
                diffraction_lit_region=True,
                seed=seed + start,
            )
            path_count[start:end] = (
                np.asarray(solved.valid.numpy(), dtype=bool)
                .reshape(end - start, -1)
                .sum(axis=1)
            )
            select_drjit_cuda_device(runtime["dr"], args.cuda_device)
            raw = chunked_cfr(solved, runtime["frequencies"], mi)
            channel = convert_cfr(np.asarray(raw), end - start)

        local = pas_marginals(channel)
        for name in MARGINALS:
            outputs[name][start:end] = local[name]
        reference = np.asarray(seed42[start:end], dtype=np.complex64)
        coherence[start:end] = complex_coherence(channel, reference)
        log_power[start:end] = np.log(
            np.maximum(
                np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64),
                TINY,
            )
        ).astype(np.float32)
        if seed != 42:
            del solved, raw
            runtime["dr"].sync_thread()
        if end == rows or end % 160 == 0:
            print(
                f"v549 seed={seed} split={split} generated={end}/{rows} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if scene is not None:
        for name in receiver_names:
            scene.remove(name)
        runtime["dr"].sync_thread()
    for value in (*outputs.values(), coherence, log_power):
        value.flush()
    diagnostics = {
        "seed": seed,
        "split": split,
        "rows": rows,
        "seconds": float(time.time() - started),
        "coherence_to_seed42_mean": float(np.mean(coherence)),
        "coherence_to_seed42_p10": float(np.quantile(coherence, 0.10)),
        "mean_path_count": float(np.mean(path_count)) if seed != 42 else None,
        "zero_spectrum_rows": int(
            np.sum(np.asarray(outputs["sh"]).sum(axis=(1, 2)) <= 0.0)
        ),
    }
    atomic_json(done_path, diagnostics)
    return paths, diagnostics


def running_path(split: str, name: str, kind: str, count: int) -> Path:
    return OUT_DIR / "running" / f"{split}_{name}_{kind}_n{count:02d}.npy"


def update_running_statistics(
    *, split: str, seed_paths: dict[str, Path], rows: int, state: dict[str, object]
) -> None:
    split_state = state["splits"][split]
    old_count = int(split_state["count"])
    new_count = old_count + 1
    generated: list[Path] = []
    for name in MARGINALS:
        shape = marginal_shape(split, name, rows)
        sample = np.load(seed_paths[name], mmap_mode="r")
        mean_out_path = running_path(split, name, "mean", new_count)
        m2_out_path = running_path(split, name, "m2", new_count)
        mean_out = open_output(mean_out_path, shape)
        m2_out = open_output(m2_out_path, shape)
        old_mean = (
            np.load(running_path(split, name, "mean", old_count), mmap_mode="r")
            if old_count
            else None
        )
        old_m2 = (
            np.load(running_path(split, name, "m2", old_count), mmap_mode="r")
            if old_count
            else None
        )
        for start in range(0, rows, 32):
            end = min(start + 32, rows)
            value = np.asarray(sample[start:end], dtype=np.float32)
            if old_count == 0:
                mean_out[start:end] = value
                m2_out[start:end] = 0.0
            else:
                previous = np.asarray(old_mean[start:end], dtype=np.float32)
                delta = value - previous
                updated = previous + delta / np.float32(new_count)
                mean_out[start:end] = updated
                m2_out[start:end] = np.asarray(
                    old_m2[start:end], dtype=np.float32
                ) + delta * (value - updated)
        mean_out.flush()
        m2_out.flush()
        generated.extend((mean_out_path, m2_out_path))

    if new_count in CHECKPOINTS:
        for name in MARGINALS:
            mean = np.load(running_path(split, name, "mean", new_count), mmap_mode="r")
            m2 = np.load(running_path(split, name, "m2", new_count), mmap_mode="r")
            mean_path = (
                OUT_DIR / "checkpoints" / f"n{new_count:02d}_{split}_{name}_mean.npy"
            )
            variance_path = (
                OUT_DIR
                / "checkpoints"
                / f"n{new_count:02d}_{split}_{name}_variance.npy"
            )
            mean_copy = open_output(mean_path, mean.shape)
            variance = open_output(variance_path, m2.shape)
            for start in range(0, rows, 32):
                end = min(start + 32, rows)
                mean_copy[start:end] = mean[start:end]
                variance[start:end] = np.asarray(
                    m2[start:end], dtype=np.float32
                ) / np.float32(max(new_count - 1, 1))
            mean_copy.flush()
            variance.flush()

    completed = [*split_state["completed_seeds"], int(seed_paths["seed"])]
    split_state["count"] = new_count
    split_state["completed_seeds"] = completed
    atomic_json(STATE, state)

    if old_count:
        for name in MARGINALS:
            running_path(split, name, "mean", old_count).unlink(missing_ok=True)
            running_path(split, name, "m2", old_count).unlink(missing_ok=True)
    del generated


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 1 <= args.max_seeds <= len(SEEDS):
        raise ValueError(f"--max-seeds must be in [1, {len(SEEDS)}]")
    for path in (
        SCENE_XML,
        TRAIN_SEED42,
        TEST_SEED42,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    positions = {
        "train": np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32),
        "test": np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32),
    }
    splits = ("train", "test") if args.split == "all" else (args.split,)
    requested = SEEDS[: args.max_seeds]
    needs_runtime = any(
        seed != 42 and seed not in state["splits"][split]["completed_seeds"]
        for split in splits
        for seed in requested
    )
    runtime = prepare_runtime(args) if needs_runtime else None
    diagnostics: list[dict[str, object]] = []
    for seed in requested:
        for split in splits:
            if seed in state["splits"][split]["completed_seeds"]:
                continue
            sample_paths, result = generate_seed_sample(
                split=split,
                seed=seed,
                positions=positions[split],
                args=args,
                runtime=runtime,
            )
            update_running_statistics(
                split=split,
                seed_paths={**sample_paths, "seed": seed},
                rows=len(positions[split]),
                state=state,
            )
            diagnostics.append(result)
            for path in sample_paths.values():
                path.unlink(missing_ok=True)
            done = OUT_DIR / "temporary" / f"{split}_seed{seed}_done.json"
            done.unlink(missing_ok=True)

    report = {
        "method": VERSION,
        "status": "physical_source_generation",
        "primary_hypothesis": (
            "Independent high-budget Sionna realizations estimate a lower-variance PAS "
            "expectation, while their cellwise variance identifies unreliable paths."
        ),
        "configuration": asdict(CONFIG),
        "seed_order": list(SEEDS),
        "requested_seed_count": args.max_seeds,
        "checkpoints": list(CHECKPOINTS),
        "state": state,
        "new_runs": diagnostics,
        "environment": {
            "sionna": (
                runtime["sionna"].__version__ if runtime is not None else "reused"
            ),
            "mitsuba_variant": (
                runtime["mi"].variant() if runtime is not None else "reused"
            ),
            "single_visible_gpu_required": True,
        },
        "outputs": {
            "directory": PATHS.relative(OUT_DIR),
            "state": PATHS.relative(STATE),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
