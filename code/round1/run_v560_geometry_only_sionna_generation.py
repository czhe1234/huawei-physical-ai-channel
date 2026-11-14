#!/usr/bin/env python3
"""Launch disjoint v560 geometry-only workers on one visible GPU."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

from project_paths import PATHS


GENERATOR = PATHS.code / "generate_v560_geometry_only_sionna_dataset.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pilot", "official"), default="pilot")
    parser.add_argument("--count", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-paths", type=int, default=256)
    parser.add_argument("--frequency", type=float, default=6.693e9)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_one_visible_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one CUDA_VISIBLE_DEVICES entry, got {visible!r}")
    return devices[0]


def main() -> None:
    args = parse_args()
    physical_gpu = require_one_visible_gpu()
    count = args.count or (20_000 if args.dataset == "pilot" else 2_500)
    shard_size = args.shard_size or (1_000 if args.dataset == "pilot" else 100)
    seed = args.seed or (44_201 if args.dataset == "pilot" else 475_001)
    if args.dataset == "official" and count != 2_500:
        raise ValueError("Official generation requires count=2500")
    total_shards = math.ceil(count / shard_size)
    end_shard = total_shards if args.end_shard is None else args.end_shard
    if not (0 <= args.start_shard < end_shard <= total_shards):
        raise ValueError(f"Invalid shard interval {args.start_shard}:{end_shard}")
    shard_count = end_shard - args.start_shard
    worker_count = min(args.workers, shard_count)
    base, remainder = divmod(shard_count, worker_count)
    assignments = []
    cursor = args.start_shard
    for worker_id in range(worker_count):
        size = base + int(worker_id < remainder)
        assignments.append((cursor, cursor + size))
        cursor += size

    # Build the shared manifest once before workers start writing disjoint shards.
    manifest_command = [
        sys.executable,
        str(GENERATOR),
        "--dataset",
        args.dataset,
        "--count",
        str(count),
        "--shard-size",
        str(shard_size),
        "--batch-size",
        str(args.batch_size),
        "--max-paths",
        str(args.max_paths),
        "--seed",
        str(seed),
        "--start-shard",
        "0",
        "--limit-shards",
        "0",
    ]
    subprocess.run(manifest_command, check=True, env=os.environ.copy())

    log_dir = PATHS.cache_dir(560) / args.dataset / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    started = time.time()
    for worker_id, (start, stop) in enumerate(assignments):
        log_path = log_dir / f"worker_{worker_id:02d}.log"
        handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(GENERATOR),
            "--dataset",
            args.dataset,
            "--count",
            str(count),
            "--shard-size",
            str(shard_size),
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
            str(seed),
            "--start-shard",
            str(start),
            "--limit-shards",
            str(stop - start),
            "--no-save-full-h",
        ]
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
            f"range={start}:{stop} log={PATHS.relative(log_path)}",
            flush=True,
        )

    failed = []
    try:
        for worker_id, process, handle, log_path in processes:
            return_code = process.wait()
            handle.close()
            print(f"worker={worker_id} exit={return_code}", flush=True)
            if return_code:
                failed.append((worker_id, return_code, log_path))
    except BaseException:
        for _, process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
        raise
    if failed:
        details = ", ".join(
            f"worker {worker} exit {code} ({PATHS.relative(path)})"
            for worker, code, path in failed
        )
        raise RuntimeError(details)
    print(
        f"completed dataset={args.dataset} workers={worker_count} "
        f"seconds={time.time() - started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
