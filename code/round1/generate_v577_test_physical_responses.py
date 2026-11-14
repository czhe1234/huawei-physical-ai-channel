#!/usr/bin/env python3
"""Generate the v575 paired physical responses at all 500 test positions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

import generate_v575_physical_response_basis as v575
from project_paths import PATHS


VERSION_NUMBER = 577
VERSION = "v577_dual_metric_physical_response_probe"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR = CACHE_DIR / "test_physical_responses"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / "v577_test_physical_response_generation.json"
ROWS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        choices=("all", *(item.name for item in v575.PAIRS)),
        default="all",
    )
    parser.add_argument(
        "--seed",
        choices=("all", *(str(value) for value in v575.PAIR_SEEDS)),
        default="all",
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def response_paths(item: v575.ResponsePair, seed: int) -> dict[str, Path]:
    local = OUT_DIR / item.name / f"seed_{seed}"
    return {
        "sh": local / "sh_sqrt_response.npy",
        "psh": local / "psh_sqrt_response.npy",
        "shue": local / "shue_sqrt_response.npy",
        "shv": local / "shv_sqrt_response.npy",
        "pdp": local / "pdp_sqrt_response.npy",
        "summary": local / "summary.npz",
        "done": local / "done.json",
    }


def completed(item: v575.ResponsePair, seed: int) -> bool:
    paths = response_paths(item, seed)
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        done = json.loads(paths["done"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        done.get("pair") == item.name
        and done.get("seed") == seed
        and done.get("rows") == ROWS
    )


def open_outputs(item: v575.ResponsePair, seed: int, force: bool):
    paths = response_paths(item, seed)
    paths["done"].parent.mkdir(parents=True, exist_ok=True)
    if force:
        for path in paths.values():
            path.unlink(missing_ok=True)
    outputs = {}
    for name, local_shape in v575.PAS_SHAPES.items():
        outputs[name] = np.lib.format.open_memmap(
            paths[name],
            mode="w+",
            dtype=np.float16,
            shape=(ROWS, int(np.prod(local_shape))),
        )
    outputs["pdp"] = np.lib.format.open_memmap(
        paths["pdp"],
        mode="w+",
        dtype=np.float16,
        shape=(ROWS, 1024, v575.NUM_SUBCARRIERS),
    )
    return paths, outputs


def generate_pair_seed(
    item: v575.ResponsePair,
    seed: int,
    positions: np.ndarray,
    runtime: dict[str, object],
    force: bool,
) -> dict[str, object]:
    if not force and completed(item, seed):
        return {
            "pair": item.name,
            "seed": seed,
            "rows": ROWS,
            "status": "reused",
            "outputs": {
                name: PATHS.relative(path)
                for name, path in response_paths(item, seed).items()
            },
        }

    paths, outputs = open_outputs(item, seed, force=True)
    point_ids = np.arange(ROWS, dtype=np.int64)
    summaries = {
        "minus_path_count": np.zeros(ROWS, dtype=np.int32),
        "plus_path_count": np.zeros(ROWS, dtype=np.int32),
        "minus_log_power": np.zeros(ROWS, dtype=np.float32),
        "plus_log_power": np.zeros(ROWS, dtype=np.float32),
    }
    for name in (*v575.PAS_SHAPES, "pdp"):
        summaries[f"{name}_response_rms"] = np.zeros(ROWS, dtype=np.float32)

    minus_runtime = v575.build_runtime(
        item.minus, f"v577_{item.name}_minus", runtime
    )
    plus_runtime = v575.build_runtime(
        item.plus, f"v577_{item.name}_plus", runtime
    )
    batch_size = min(item.minus.batch_size, item.plus.batch_size)
    started = time.time()
    for start in range(0, ROWS, batch_size):
        stop = min(start + batch_size, ROWS)
        minus, minus_count, minus_power = v575.solve_batch(
            minus_runtime,
            item.minus,
            positions[start:stop],
            point_ids[start:stop],
            seed,
            start,
            f"v577_{item.name}_minus",
            runtime,
        )
        plus, plus_count, plus_power = v575.solve_batch(
            plus_runtime,
            item.plus,
            positions[start:stop],
            point_ids[start:stop],
            seed,
            start,
            f"v577_{item.name}_plus",
            runtime,
        )
        summaries["minus_path_count"][start:stop] = minus_count
        summaries["plus_path_count"][start:stop] = plus_count
        summaries["minus_log_power"][start:stop] = np.log(
            np.maximum(minus_power, v575.TINY)
        ).astype(np.float32)
        summaries["plus_log_power"][start:stop] = np.log(
            np.maximum(plus_power, v575.TINY)
        ).astype(np.float32)
        for name in outputs:
            response = np.float32(0.5) * (
                np.sqrt(np.maximum(plus[name], 0.0))
                - np.sqrt(np.maximum(minus[name], 0.0))
            )
            outputs[name][start:stop] = response.astype(np.float16)
            summaries[f"{name}_response_rms"][start:stop] = np.sqrt(
                np.mean(
                    response.astype(np.float64) ** 2,
                    axis=tuple(range(1, response.ndim)),
                )
            ).astype(np.float32)
        if stop == ROWS or stop % 50 == 0:
            for output in outputs.values():
                output.flush()
            print(
                f"v577 pair={item.name} seed={seed} {stop}/{ROWS} "
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
            raise FloatingPointError(
                f"non-finite v577 response: {item.name} {seed} {name}"
            )
    result = {
        "pair": item.name,
        "seed": seed,
        "rows": ROWS,
        "status": "complete",
        "seconds": float(time.time() - started),
        "minus": asdict(item.minus),
        "plus": asdict(item.plus),
        "output_shapes": output_shapes,
        "outputs": {
            name: PATHS.relative(path)
            for name, path in paths.items()
            if name != "done"
        },
    }
    v575.atomic_json(paths["done"], result)
    return result


def load_report() -> dict[str, object]:
    if not REPORT.is_file():
        return {}
    return json.loads(REPORT.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    for path in (v575.SCENE_XML, PATHS.round1 / "Round1_Test_Pos.npy"):
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    if positions.shape != (ROWS, 3):
        raise ValueError(f"unexpected test positions shape: {positions.shape}")

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    v575.select_drjit_cuda_device(dr, args.cuda_device)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v577 requires Sionna 2.0.1, got {sionna.__version__}")
    v575.select_drjit_cuda_device(dr, args.cuda_device)
    runtime = {
        "mi": mi,
        "dr": dr,
        "load_scene": load_scene,
        "PlanarArray": PlanarArray,
        "Receiver": Receiver,
        "Transmitter": Transmitter,
        "PathSolver": PathSolver,
        "frequencies": np.asarray(
            subcarrier_frequencies(v575.NUM_SUBCARRIERS, v575.SUBCARRIER_SPACING)
        ).astype(np.float32),
        "cuda_device": args.cuda_device,
    }
    selected_pairs = [
        item for item in v575.PAIRS if args.pair in ("all", item.name)
    ]
    selected_seeds = [
        seed
        for seed in v575.PAIR_SEEDS
        if args.seed in ("all", str(seed))
    ]
    report = load_report()
    results = dict(report.get("results", {}))
    for item in selected_pairs:
        pair_results = dict(results.get(item.name, {}))
        for seed in selected_seeds:
            pair_results[str(seed)] = generate_pair_seed(
                item, seed, positions, runtime, args.force
            )
            results[item.name] = pair_results
            report = {
                "method": VERSION,
                "status": "test_physical_response_generation",
                "rows": ROWS,
                "response_definition": (
                    "0.5 * (sqrt(normalized plus spectrum) - "
                    "sqrt(normalized minus spectrum))"
                ),
                "pair_seeds": list(v575.PAIR_SEEDS),
                "pairs": [asdict(value) for value in v575.PAIRS],
                "environment": {
                    "sionna": sionna.__version__,
                    "mitsuba_variant": mi.variant(),
                    "requested_cuda_device": args.cuda_device,
                    "single_gpu_required": True,
                },
                "results": results,
                "report": PATHS.relative(REPORT),
            }
            v575.atomic_json(REPORT, report)
    print(
        json.dumps(
            {
                "report": PATHS.relative(REPORT),
                "completed_pairs": sorted(results),
                "rows": ROWS,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
