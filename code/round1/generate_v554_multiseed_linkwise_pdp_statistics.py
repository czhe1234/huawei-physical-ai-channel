#!/usr/bin/env python3
"""Generate streaming multi-seed linkwise-PDP statistics for v554."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import generate_v549_multiseed_v207_pas_statistics as source
from project_paths import PATHS


VERSION_NUMBER = 554
VERSION = "v554_multiseed_linkwise_pdp_statistics"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER) / "multiseed_linkwise_pdp"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
STATE = OUT_DIR / "state.json"
NUM_SUBCARRIERS = source.NUM_SUBCARRIERS
SEEDS = source.SEEDS
CHECKPOINTS = source.CHECKPOINTS
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-seeds", type=int, default=len(SEEDS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--update-batch-size", type=int, default=8)
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
        "configuration": asdict(source.CONFIG),
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
    for key in ("configuration", "seed_order"):
        if state.get(key) != expected[key]:
            raise ValueError(f"v554 {key} changed after generation started")
    return state


def pdp_shape(rows: int) -> tuple[int, int, int, int]:
    return (rows, 256, 4, NUM_SUBCARRIERS)


def count_shape(rows: int) -> tuple[int, int, int]:
    return (rows, 256, 4)


def open_output(path: Path, shape: tuple[int, ...], dtype=np.float32) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def normalized_linkwise_pdp(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cir = np.fft.ifft(channel, axis=-1)
    power = np.abs(cir).astype(np.float32) ** 2
    total = np.sum(power, axis=-1, dtype=np.float64).astype(np.float32)
    valid = total > TINY
    shape = power / np.maximum(total[..., None], TINY)
    shape[~valid] = 0.0
    return shape.astype(np.float32), valid


def generate_seed_sample(
    *,
    split: str,
    seed: int,
    positions: np.ndarray,
    args: argparse.Namespace,
    runtime: dict[str, object] | None,
) -> tuple[Path, Path, dict[str, object]]:
    rows = len(positions)
    prefix = OUT_DIR / "temporary" / f"{split}_seed{seed}"
    pdp_path = prefix.with_name(prefix.name + "_linkwise_pdp.npy")
    valid_path = prefix.with_name(prefix.name + "_valid.npy")
    done_path = prefix.with_name(prefix.name + "_done.json")
    if done_path.is_file() and pdp_path.is_file() and valid_path.is_file():
        return pdp_path, valid_path, json.loads(done_path.read_text(encoding="utf-8"))

    for path in (pdp_path, valid_path, done_path):
        path.unlink(missing_ok=True)
    output = open_output(pdp_path, pdp_shape(rows))
    validity = open_output(valid_path, count_shape(rows), dtype=np.uint8)
    seed42_path = source.TRAIN_SEED42 if split == "train" else source.TEST_SEED42
    seed42 = np.load(seed42_path, mmap_mode="r")
    if seed42.shape != pdp_shape(rows):
        raise ValueError(f"seed42 reference mismatch: {seed42_path} {seed42.shape}")

    scene = None
    receiver_names: list[str] = []
    path_count = np.zeros(rows, dtype=np.int32)
    if seed != 42:
        if runtime is None:
            raise RuntimeError("Sionna runtime is required for non-reused seeds")
        mi = runtime["mi"]
        scene = runtime["load_scene"](str(source.SCENE_XML), merge_shapes=True)
        scene.frequency = source.CONFIG.frequency_hz
        scene.tx_array = runtime["PlanarArray"](
            num_rows=8,
            num_cols=16,
            vertical_spacing=source.CONFIG.tx_vertical_spacing,
            horizontal_spacing=source.CONFIG.tx_horizontal_spacing,
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
        radius = 140.0
        azimuth = math.radians(source.CONFIG.tx_azimuth_deg)
        tilt = math.radians(source.CONFIG.tx_tilt_deg)
        target = mi.Point3f(
            float(source.BS_POSITION[0] + radius * math.cos(azimuth)),
            float(source.BS_POSITION[1] + radius * math.sin(azimuth)),
            float(source.BS_POSITION[2] + radius * math.tan(tilt)),
        )
        transmitter = runtime["Transmitter"](
            name=f"tx_v554_seed{seed}",
            position=mi.Point3f(*map(float, source.BS_POSITION)),
            look_at=target,
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
            source.select_drjit_cuda_device(runtime["dr"], args.cuda_device)
            solved = solver(
                scene,
                max_depth=source.CONFIG.max_depth,
                max_num_paths_per_src=source.CONFIG.max_num_paths_per_src,
                samples_per_src=source.CONFIG.samples_per_src,
                synthetic_array=source.CONFIG.synthetic_array,
                los=source.CONFIG.los,
                specular_reflection=source.CONFIG.specular_reflection,
                diffuse_reflection=source.CONFIG.diffuse_reflection,
                refraction=source.CONFIG.refraction,
                diffraction=source.CONFIG.diffraction,
                edge_diffraction=source.CONFIG.edge_diffraction,
                diffraction_lit_region=True,
                seed=seed + start,
            )
            path_count[start:end] = (
                np.asarray(solved.valid.numpy(), dtype=bool)
                .reshape(end - start, -1)
                .sum(axis=1)
            )
            source.select_drjit_cuda_device(runtime["dr"], args.cuda_device)
            raw = source.chunked_cfr(solved, runtime["frequencies"], mi)
            channel = source.convert_cfr(np.asarray(raw), end - start)

        local_pdp, local_valid = normalized_linkwise_pdp(channel)
        output[start:end] = local_pdp
        validity[start:end] = local_valid.astype(np.uint8)
        if seed != 42:
            del solved, raw
            runtime["dr"].sync_thread()
        if end == rows or end % 160 == 0:
            print(
                f"v554 seed={seed} split={split} generated={end}/{rows} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if scene is not None:
        for name in receiver_names:
            scene.remove(name)
        runtime["dr"].sync_thread()
    output.flush()
    validity.flush()
    link_valid = np.asarray(validity, dtype=bool)
    point_valid = np.any(link_valid.reshape(rows, -1), axis=1)
    diagnostics = {
        "seed": seed,
        "split": split,
        "rows": rows,
        "seconds": float(time.time() - started),
        "valid_point_fraction": float(np.mean(point_valid)),
        "valid_link_fraction": float(np.mean(link_valid)),
        "mean_path_count": float(np.mean(path_count)) if seed != 42 else None,
    }
    atomic_json(done_path, diagnostics)
    return pdp_path, valid_path, diagnostics


def running_path(split: str, kind: str, count: int) -> Path:
    return OUT_DIR / "running" / f"{split}_{kind}_n{count:02d}.npy"


def checkpoint_path(checkpoint: int, split: str, kind: str) -> Path:
    return OUT_DIR / "checkpoints" / f"n{checkpoint:02d}_{split}_{kind}.npy"


def update_running_statistics(
    *,
    split: str,
    pdp_path: Path,
    valid_path: Path,
    seed: int,
    rows: int,
    update_batch_size: int,
    state: dict[str, object],
) -> None:
    split_state = state["splits"][split]
    old_seed_count = int(split_state["count"])
    new_seed_count = old_seed_count + 1
    sample = np.load(pdp_path, mmap_mode="r")
    sample_valid = np.load(valid_path, mmap_mode="r")
    mean_out = open_output(running_path(split, "mean", new_seed_count), pdp_shape(rows))
    m2_out = open_output(running_path(split, "m2", new_seed_count), pdp_shape(rows))
    count_out = open_output(
        running_path(split, "valid_count", new_seed_count),
        count_shape(rows),
        dtype=np.uint16,
    )
    old_mean = (
        np.load(running_path(split, "mean", old_seed_count), mmap_mode="r")
        if old_seed_count
        else None
    )
    old_m2 = (
        np.load(running_path(split, "m2", old_seed_count), mmap_mode="r")
        if old_seed_count
        else None
    )
    old_count = (
        np.load(running_path(split, "valid_count", old_seed_count), mmap_mode="r")
        if old_seed_count
        else None
    )
    for start in range(0, rows, update_batch_size):
        end = min(start + update_batch_size, rows)
        value = np.asarray(sample[start:end], dtype=np.float32)
        valid = np.asarray(sample_valid[start:end], dtype=bool)
        if old_seed_count:
            previous = np.asarray(old_mean[start:end], dtype=np.float32)
            previous_m2 = np.asarray(old_m2[start:end], dtype=np.float32)
            previous_count = np.asarray(old_count[start:end], dtype=np.uint16)
        else:
            previous = np.zeros_like(value)
            previous_m2 = np.zeros_like(value)
            previous_count = np.zeros(valid.shape, dtype=np.uint16)
        updated_count = previous_count + valid.astype(np.uint16)
        denominator = np.maximum(updated_count, 1).astype(np.float32)[..., None]
        delta = value - previous
        updated = np.where(valid[..., None], previous + delta / denominator, previous)
        updated_m2 = np.where(
            valid[..., None],
            previous_m2 + delta * (value - updated),
            previous_m2,
        )
        mean_out[start:end] = updated
        m2_out[start:end] = updated_m2
        count_out[start:end] = updated_count
    mean_out.flush()
    m2_out.flush()
    count_out.flush()

    if new_seed_count in CHECKPOINTS:
        mean_copy = open_output(
            checkpoint_path(new_seed_count, split, "mean"), pdp_shape(rows)
        )
        variance = open_output(
            checkpoint_path(new_seed_count, split, "variance"), pdp_shape(rows)
        )
        count_copy = open_output(
            checkpoint_path(new_seed_count, split, "valid_count"),
            count_shape(rows),
            dtype=np.uint16,
        )
        for start in range(0, rows, update_batch_size):
            end = min(start + update_batch_size, rows)
            local_count = np.asarray(count_out[start:end], dtype=np.uint16)
            mean_copy[start:end] = mean_out[start:end]
            variance[start:end] = (
                np.asarray(m2_out[start:end], dtype=np.float32)
                / np.maximum(local_count.astype(np.float32) - 1.0, 1.0)[..., None]
            )
            count_copy[start:end] = local_count
        mean_copy.flush()
        variance.flush()
        count_copy.flush()

    split_state["count"] = new_seed_count
    split_state["completed_seeds"] = [
        *split_state["completed_seeds"],
        int(seed),
    ]
    atomic_json(STATE, state)
    if old_seed_count:
        for kind in ("mean", "m2", "valid_count"):
            running_path(split, kind, old_seed_count).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.update_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not 1 <= args.max_seeds <= len(SEEDS):
        raise ValueError(f"--max-seeds must be in [1, {len(SEEDS)}]")
    for path in (
        source.SCENE_XML,
        source.TRAIN_SEED42,
        source.TEST_SEED42,
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
    runtime = source.prepare_runtime(args) if needs_runtime else None
    diagnostics: list[dict[str, object]] = []
    for seed in requested:
        for split in splits:
            if seed in state["splits"][split]["completed_seeds"]:
                continue
            pdp_path, valid_path, result = generate_seed_sample(
                split=split,
                seed=seed,
                positions=positions[split],
                args=args,
                runtime=runtime,
            )
            update_running_statistics(
                split=split,
                pdp_path=pdp_path,
                valid_path=valid_path,
                seed=seed,
                rows=len(positions[split]),
                update_batch_size=args.update_batch_size,
                state=state,
            )
            diagnostics.append(result)
            pdp_path.unlink(missing_ok=True)
            valid_path.unlink(missing_ok=True)
            (OUT_DIR / "temporary" / f"{split}_seed{seed}_done.json").unlink(
                missing_ok=True
            )

    report = {
        "method": VERSION,
        "status": "physical_source_generation",
        "primary_hypothesis": (
            "The v549 Monte Carlo expectation and variance mechanism extends to the "
            "scorer-aligned normalized 192-bin PDP of every 256x4 link."
        ),
        "configuration": asdict(source.CONFIG),
        "seed_order": list(SEEDS),
        "requested_seed_count": args.max_seeds,
        "checkpoints": list(CHECKPOINTS),
        "invalid_realization_policy": (
            "skip each invalid link in Welford statistics; retain its per-link valid count"
        ),
        "state": state,
        "new_runs": diagnostics,
        "environment": {
            "sionna": runtime["sionna"].__version__ if runtime else "reused",
            "mitsuba_variant": runtime["mi"].variant() if runtime else "reused",
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
