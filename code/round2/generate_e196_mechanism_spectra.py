#!/usr/bin/env python3
"""Generate resumable mechanism-separated Sionna spectra for Round 2 E196."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import traceback
from typing import Any

import numpy as np

from generate_multiseed_sionna_dataset import (
    NUM_SUBCARRIERS,
    SCENE_XML,
    SEEDS,
    chunked_cfr,
    convert_cfr,
    prepare_runtime,
    score_aligned_statistics,
    select_drjit_cuda_device,
    tx_target,
)
from generate_v5_sionna_spectral_library import configure_materials, prepare_scene
from project_paths import PATHS
from round2.data import load_positions


VERSION = "Round2-E196-mechanism-separated-spectra"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e196_mechanism_spectra"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e196_mechanism_spectra"
NUM_SEEDS = len(SEEDS)
TINY = 1e-30
SPECTRAL_DOMAINS = ("pas", "pdp", "ad")
SCALAR_DOMAINS = ("log_power", "path_count")
DOMAINS = (*SPECTRAL_DOMAINS, *SCALAR_DOMAINS)


@dataclass(frozen=True)
class MechanismConfig:
    name: str
    tx_vertical_spacing: float
    tx_horizontal_spacing: float
    max_depth: int
    los: bool
    specular_reflection: bool
    diffuse_reflection: bool
    refraction: bool
    diffraction: bool
    edge_diffraction: bool


CONFIGS = (
    MechanismConfig(
        name="d16_v05lambda",
        tx_vertical_spacing=0.5,
        tx_horizontal_spacing=0.5,
        max_depth=16,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False,
        edge_diffraction=False,
    ),
    MechanismConfig(
        name="d8_edge_v4lambda",
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=8,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=True,
        edge_diffraction=True,
    ),
    MechanismConfig(
        name="d16_no_refraction",
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=16,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        edge_diffraction=False,
    ),
    MechanismConfig(
        name="d16_no_specular",
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=16,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False,
        edge_diffraction=False,
    ),
    MechanismConfig(
        name="d4_shallow_v4lambda",
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=4,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False,
        edge_diffraction=False,
    ),
)
CONFIG_BY_NAME = {config.name: config for config in CONFIGS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=tuple(CONFIG_BY_NAME), required=True)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-count-per-site", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260813196)
    parser.add_argument("--selection-tag", default="full")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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


def buffer_path(root: Path, split: str, name: str, kind: str, count: int) -> Path:
    return root / "running" / f"{split}_{name}_{kind}_n{count:02d}.npy"


def done_path(root: Path, split: str, count: int) -> Path:
    return root / "running" / f"{split}_active_n{count:02d}_done.npy"


def final_path(root: Path, split: str, name: str) -> Path:
    suffix = "f16" if name in SPECTRAL_DOMAINS else "f32"
    return root / "final" / f"{split}_{name}_mean_{suffix}.npy"


def open_memmap(
    path: Path,
    dtype: np.dtype,
    shape: tuple[int, ...],
    reset: bool,
    initialize: bool = False,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        path.unlink(missing_ok=True)
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"cache mismatch {path}: {value.shape}/{value.dtype}")
        return value
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if initialize:
        value[...] = 0
        value.flush()
    return value


def sample_ids(site: np.ndarray, count_per_site: int, seed: int) -> np.ndarray:
    if count_per_site <= 0:
        return np.arange(len(site), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for site_value in np.unique(site):
        local = np.flatnonzero(site == site_value)
        count = min(int(count_per_site), len(local))
        selected.append(np.sort(rng.choice(local, size=count, replace=False)))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def initial_state(
    config: MechanismConfig, selection_tag: str, split_ids: dict[str, np.ndarray]
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "configuration": asdict(config),
        "selection_tag": selection_tag,
        "seed_order": list(SEEDS),
        "ids": {
            split: {
                "count": int(len(ids)),
                "sha256": __import__("hashlib").sha256(
                    np.asarray(ids, dtype=np.int64).tobytes()
                ).hexdigest(),
            }
            for split, ids in split_ids.items()
        },
        "splits": {
            split: {"count": 0, "completed_seeds": [], "active_seed": None}
            for split in split_ids
        },
        "runs": [],
    }


def load_state(
    path: Path,
    config: MechanismConfig,
    selection_tag: str,
    split_ids: dict[str, np.ndarray],
    force: bool,
) -> dict[str, Any]:
    expected = initial_state(config, selection_tag, split_ids)
    if force and path.is_file():
        shutil.rmtree(path.parent)
    if not path.is_file():
        atomic_json(path, expected)
        return expected
    state = json.loads(path.read_text(encoding="utf-8"))
    for key in ("version", "configuration", "selection_tag", "seed_order", "ids"):
        if state.get(key) != expected[key]:
            raise ValueError(f"state mismatch for {path}: {key}")
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
        raise RuntimeError(f"unfinished seed {active} blocks {seed} for {split}")
    stores: dict[str, dict[str, np.memmap]] = {}
    old_stores: dict[str, dict[str, np.memmap]] | None = {} if old_count else None
    for name in DOMAINS:
        mean_target = buffer_path(root, split, name, "mean", new_count)
        count_target = buffer_path(root, split, name, "count", new_count)
        stores[name] = {
            "mean": open_memmap(
                mean_target, np.float32, domain_shape(name, rows), reset=not resume
            ),
            "count": open_memmap(
                count_target, np.uint16, count_shape(name, rows), reset=not resume
            ),
        }
        if old_stores is not None:
            old_stores[name] = {
                "mean": np.load(
                    buffer_path(root, split, name, "mean", old_count), mmap_mode="r"
                ),
                "count": np.load(
                    buffer_path(root, split, name, "count", old_count), mmap_mode="r"
                ),
            }
    done = open_memmap(
        done_path(root, split, new_count),
        np.bool_,
        (rows,),
        not resume,
        initialize=not resume,
    )
    if not resume:
        split_state["active_seed"] = int(seed)
        atomic_json(state_path, state)
    return old_count, stores, old_stores, done


def update_mean(
    name: str,
    rows: np.ndarray,
    value: np.ndarray,
    valid: np.ndarray,
    old_count: int,
    stores: dict[str, dict[str, np.memmap]],
    old_stores: dict[str, dict[str, np.memmap]] | None,
) -> None:
    mean_store = stores[name]["mean"]
    count_store = stores[name]["count"]
    if old_count:
        assert old_stores is not None
        previous = np.asarray(old_stores[name]["mean"][rows], dtype=np.float32)
        previous_count = np.asarray(
            old_stores[name]["count"][rows], dtype=np.uint16
        )
    else:
        previous = np.zeros_like(value, dtype=np.float32)
        previous_count = np.zeros_like(valid, dtype=np.uint16)
    valid_uint = np.asarray(valid, dtype=np.uint16)
    updated_count = previous_count + valid_uint
    denominator = np.maximum(expand_count(name, updated_count), 1).astype(np.float32)
    expanded_valid = expand_count(name, np.asarray(valid, dtype=bool))
    updated = np.where(
        expanded_valid,
        previous + (np.asarray(value, dtype=np.float32) - previous) / denominator,
        previous,
    )
    mean_store[rows] = updated
    count_store[rows] = updated_count


def physical_gpu_index(logical_index: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return int(logical_index)
    values = [value.strip() for value in visible.split(",") if value.strip()]
    if logical_index >= len(values):
        raise ValueError(
            f"cuda device {logical_index} is outside CUDA_VISIBLE_DEVICES={visible}"
        )
    return int(values[logical_index])


def gpu_snapshot(logical_index: int) -> dict[str, Any]:
    physical = physical_gpu_index(logical_index)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(physical),
            ],
            text=True,
        ).strip()
        name, used, total = [value.strip() for value in output.split(",")]
        return {
            "logical_index": int(logical_index),
            "physical_index": int(physical),
            "name": name,
            "memory_used_mib_at_report": int(used),
            "memory_total_mib": int(total),
        }
    except Exception as error:
        return {
            "logical_index": int(logical_index),
            "physical_index": int(physical),
            "query_error": repr(error),
        }


def finish_seed(
    root: Path,
    state_path: Path,
    state: dict[str, Any],
    split: str,
    seed: int,
    old_count: int,
    run: dict[str, Any],
) -> None:
    new_count = old_count + 1
    split_state = state["splits"][split]
    split_state["count"] = new_count
    split_state["completed_seeds"].append(int(seed))
    split_state["active_seed"] = None
    state["runs"].append(run)
    atomic_json(state_path, state)
    done_path(root, split, new_count).unlink(missing_ok=True)
    if old_count:
        for name in DOMAINS:
            buffer_path(root, split, name, "mean", old_count).unlink(missing_ok=True)
            buffer_path(root, split, name, "count", old_count).unlink(missing_ok=True)


def finalize_split(
    root: Path,
    split: str,
    rows: int,
    count: int,
    required_count: int,
) -> dict[str, Any]:
    if count != required_count:
        return {
            "status": "not_final",
            "seed_count": int(count),
            "required_seed_count": int(required_count),
        }
    outputs = {}
    for name in DOMAINS:
        target = final_path(root, split, name)
        dtype = np.float16 if name in SPECTRAL_DOMAINS else np.float32
        if target.is_file():
            cached = np.load(target, mmap_mode="r")
            if cached.shape != domain_shape(name, rows) or cached.dtype != np.dtype(dtype):
                raise ValueError(f"invalid final cache {target}: {cached.shape}/{cached.dtype}")
            outputs[name] = PATHS.relative(target)
            continue
        source = np.load(
            buffer_path(root, split, name, "mean", count), mmap_mode="r"
        )
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npy")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=dtype, shape=source.shape
        )
        chunk = 1 if name == "pas" else 4 if name in ("pdp", "ad") else 512
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            output[start:end] = np.asarray(source[start:end], dtype=dtype)
        output.flush()
        del output
        os.replace(temporary, target)
        outputs[name] = PATHS.relative(target)
    return {
        "status": "complete",
        "seed_count": int(count),
        "required_seed_count": int(required_count),
        "outputs": outputs,
    }


def generate_seed(
    config: MechanismConfig,
    root: Path,
    state_path: Path,
    state: dict[str, Any],
    split: str,
    seed: int,
    positions: np.ndarray,
    sites: np.ndarray,
    ids: np.ndarray,
    args: argparse.Namespace,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    selected_positions = positions[ids].astype(np.float32)
    selected_sites = sites[ids].astype(np.int8)
    old_count, stores, old_stores, done = begin_seed(
        root, state_path, state, split, seed, len(ids)
    )
    setup, _, _, _, _ = load_positions()
    started = time.time()
    path_counts = np.zeros(len(ids), dtype=np.int32)
    materials: dict[str, float] = {}

    for site_value in np.unique(selected_sites):
        scene = runtime["load_scene"](
            str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
        )
        scene.frequency = 6.69e9
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
        materials = configure_materials(scene, 0.0)
        bs_position = setup.bs_positions[int(site_value)]
        transmitter = runtime["Transmitter"](
            name=f"tx_site{int(site_value)}_seed{seed}",
            position=runtime["mi"].Point3f(*map(float, bs_position)),
            look_at=tx_target(
                bs_position,
                (-44.0, 73.5)[int(site_value)],
                (-11.5, -10.5)[int(site_value)],
                runtime["mi"],
            ),
            power_dbm=44.0,
        )
        scene.add(transmitter)
        solver = runtime["PathSolver"]()
        receiver_names: list[str] = []
        local_rows = np.flatnonzero(selected_sites == site_value)
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
                max_num_paths_per_src=1_000_000,
                samples_per_src=1_000_000,
                synthetic_array=True,
                los=config.los,
                specular_reflection=config.specular_reflection,
                diffuse_reflection=config.diffuse_reflection,
                refraction=config.refraction,
                diffraction=config.diffraction,
                edge_diffraction=config.edge_diffraction,
                diffraction_lit_region=True,
                seed=solver_seed,
            )
            local_count = np.asarray(traced.valid.numpy(), dtype=bool).reshape(
                len(batch_rows), -1
            ).sum(axis=1)
            path_counts[batch_rows] = local_count
            raw = chunked_cfr(traced, runtime["frequencies"], runtime["mi"])
            channel = convert_cfr(np.asarray(raw), len(batch_rows))
            local = score_aligned_statistics(channel)
            update_mean(
                "pas",
                batch_rows,
                local["pas"],
                local["pas_valid"],
                old_count,
                stores,
                old_stores,
            )
            update_mean(
                "pdp",
                batch_rows,
                local["pdp"],
                local["pdp_valid"],
                old_count,
                stores,
                old_stores,
            )
            update_mean(
                "ad",
                batch_rows,
                local["ad"],
                local["ad_valid"],
                old_count,
                stores,
                old_stores,
            )
            update_mean(
                "log_power",
                batch_rows,
                local["log_power"],
                local["log_power_valid"],
                old_count,
                stores,
                old_stores,
            )
            update_mean(
                "path_count",
                batch_rows,
                local_count.astype(np.float32),
                np.ones(len(batch_rows), dtype=bool),
                old_count,
                stores,
                old_stores,
            )
            for domain in stores.values():
                domain["mean"].flush()
                domain["count"].flush()
            done[batch_rows] = True
            done.flush()
            del traced, raw, channel, local
            runtime["dr"].sync_thread()
            completed = int(done.sum())
            if completed == len(ids) or completed % 320 < len(batch_rows):
                print(
                    f"E196 {config.name} seed={seed} {split} "
                    f"{completed}/{len(ids)} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        for name in receiver_names:
            scene.remove(name)
        del scene, solver
        runtime["dr"].sync_thread()

    if not np.all(done):
        raise RuntimeError(f"incomplete seed {seed} for {config.name}/{split}")
    run = {
        "config": config.name,
        "split": split,
        "seed": int(seed),
        "rows": int(len(ids)),
        "seconds": float(time.time() - started),
        "mean_path_count": float(path_counts.mean()),
        "zero_path_fraction": float(np.mean(path_counts == 0)),
        "materials": materials,
    }
    finish_seed(root, state_path, state, split, seed, old_count, run)
    return run


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 1 <= args.max_seeds <= NUM_SEEDS:
        raise ValueError(f"--max-seeds must be in [1,{NUM_SEEDS}]")
    started = time.time()
    config = CONFIG_BY_NAME[args.config]
    prepare_scene()
    setup, train_pos, test_pos, train_site, test_site = load_positions()
    del setup
    positions = {"train": train_pos, "test": test_pos}
    sites = {"train": train_site, "test": test_site}
    selected_splits = ("train", "test") if args.split == "all" else (args.split,)
    split_ids = {
        split: sample_ids(
            sites[split],
            args.sample_count_per_site,
            args.sample_seed + 10_000 * int(split == "test"),
        )
        for split in selected_splits
    }
    root = ARTIFACT_DIR / config.name / args.selection_tag
    state_path = root / "state.json"
    state = load_state(
        state_path, config, args.selection_tag, split_ids, args.force
    )
    root.mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        ids_path = root / f"{split}_ids.npy"
        if ids_path.is_file():
            if not np.array_equal(np.load(ids_path), ids):
                raise ValueError(f"ids changed for {ids_path}")
        else:
            np.save(ids_path, ids)

    runtime = prepare_runtime(args)
    status = "complete"
    error = None
    try:
        for seed in SEEDS[: args.max_seeds]:
            for split in selected_splits:
                if seed in state["splits"][split]["completed_seeds"]:
                    continue
                generate_seed(
                    config,
                    root,
                    state_path,
                    state,
                    split,
                    seed,
                    positions[split],
                    sites[split],
                    split_ids[split],
                    args,
                    runtime,
                )
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        raise
    finally:
        final = {
            split: finalize_split(
                root,
                split,
                len(split_ids[split]),
                int(state["splits"][split]["count"]),
                int(args.max_seeds),
            )
            for split in selected_splits
        }
        report = {
            "experiment": VERSION,
            "status": status,
            "configuration": asdict(config),
            "arguments": vars(args),
            "splits": state["splits"],
            "final": final,
            "performance": {
                "gpu": gpu_snapshot(args.cuda_device),
                "peak_rss_gib": float(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
                ),
                "wall_seconds": float(time.time() - started),
                "measured_bottleneck": (
                    "Sionna PathSolver and full-spectrum writes; scorer FFT remains on CPU "
                    "because the dedicated Sionna environment's PyTorch CUDA build requires "
                    "a newer driver, while the established NumPy implementation is exact"
                ),
            },
            "error": error,
        }
        report_path = ANALYSIS_DIR / f"{config.name}_{args.selection_tag}.json"
        atomic_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
