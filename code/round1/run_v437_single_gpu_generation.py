#!/usr/bin/env python3
"""Run disjoint v437 generator workers on one visible GPU."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from project_paths import PATHS


GENERATOR = PATHS.code / "generate_v437_single_gpu_sionna_pretraining_dataset.py"
LOG_DIR = PATHS.cache_dir(437) / "worker_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--frequency", type=float, default=6.693e9)
    parser.add_argument("--seed", type=int, default=43_701)
    parser.add_argument("--no-save-full-h", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_one_visible_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(
            "The v437 launcher requires exactly one CUDA_VISIBLE_DEVICES entry; "
            f"received {visible!r}"
        )
    return devices[0]


def main() -> None:
    args = parse_args()
    physical_gpu = require_one_visible_gpu()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    total_shards = math.ceil(args.count / args.shard_size)
    end_shard = total_shards if args.end_shard is None else args.end_shard
    if not (0 <= args.start_shard < end_shard <= total_shards):
        raise ValueError(
            f"invalid shard interval [{args.start_shard}, {end_shard}) "
            f"for {total_shards} shards"
        )
    shard_count = end_shard - args.start_shard
    worker_count = min(args.workers, shard_count)
    base, remainder = divmod(shard_count, worker_count)
    assignments: list[list[int]] = []
    cursor = args.start_shard
    for worker_id in range(worker_count):
        size = base + int(worker_id < remainder)
        assignments.append(list(range(cursor, cursor + size)))
        cursor += size

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[str], object, Path]] = []
    started = time.time()
    for worker_id in range(len(assignments)):
        worker_shards = assignments[worker_id]
        log_path = LOG_DIR / f"worker_{worker_id:02d}.log"
        handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(GENERATOR),
            "--count",
            str(args.count),
            "--shard-size",
            str(args.shard_size),
            "--batch-size",
            str(args.batch_size),
            "--samples-per-src",
            str(args.samples_per_src),
            "--max-num-paths-per-src",
            str(args.max_num_paths_per_src),
            "--max-depth",
            str(args.max_depth),
            "--max-paths",
            str(args.max_paths),
            "--frequency",
            str(args.frequency),
            "--seed",
            str(args.seed),
            "--start-shard",
            str(worker_shards[0]),
            "--limit-shards",
            str(worker_shards[-1] - worker_shards[0] + 1),
            "--mitsuba-variant",
            "cuda_ad_mono_polarized",
        ]
        if args.no_save_full_h:
            command.append("--no-save-full-h")
        if args.overwrite:
            command.append("--overwrite-shards")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        processes.append((worker_id, process, handle, log_path))
        print(
            f"worker={worker_id} pid={process.pid} gpu={physical_gpu} "
            f"range={worker_shards[0]}:{worker_shards[-1] + 1} log={PATHS.relative(log_path)}",
            flush=True,
        )

    failed = []
    try:
        for worker_id, process, handle, log_path in processes:
            return_code = process.wait()
            handle.close()
            print(
                f"worker={worker_id} exit={return_code} log={PATHS.relative(log_path)}",
                flush=True,
            )
            if return_code != 0:
                failed.append((worker_id, return_code, log_path))
    except BaseException:
        for _, process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
        raise
    if failed:
        details = ", ".join(
            f"worker {worker_id} exit {code} ({PATHS.relative(path)})"
            for worker_id, code, path in failed
        )
        raise RuntimeError(f"v437 generation failed: {details}")
    print(
        f"completed workers={len(processes)} gpu={physical_gpu} "
        f"seconds={time.time() - started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
