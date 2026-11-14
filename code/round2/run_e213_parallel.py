#!/usr/bin/env python3
"""Run the five E213 folds in parallel and aggregate the fixed-500 score."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from project_paths import PATHS


SCRIPT = Path(__file__).resolve().with_name("train_e213_no_rt_spectral_residual.py")
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e213_no_rt_spectral_residual"
RUNNER_REPORT = ANALYSIS_DIR / "parallel_run.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channel-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def command(stage: str, epochs: int, batch_size: int, channel_batch_size: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(SCRIPT),
        "--stage",
        stage,
        "--device",
        "cuda:0",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--channel-batch-size",
        str(channel_batch_size),
    ]


def main() -> None:
    args = parse_args()
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) < 5:
        raise ValueError("five GPU ids are required")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    processes: list[tuple[int, str, subprocess.Popen[bytes], object, Path]] = []
    for outer in range(5):
        gpu = gpu_ids[outer]
        log_path = ANALYSIS_DIR / f"outer{outer}.log"
        log_handle = log_path.open("wb")
        local_command = command(
            "train", args.epochs, args.batch_size, args.channel_batch_size
        ) + ["--outer", str(outer)]
        if args.force:
            local_command.append("--force")
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(
            local_command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((outer, gpu, process, log_handle, log_path))
        print(f"E213 launched outer={outer} gpu={gpu} pid={process.pid}", flush=True)

    results = []
    failed = False
    for outer, gpu, process, log_handle, log_path in processes:
        exit_code = process.wait()
        log_handle.close()
        exit_path = ANALYSIS_DIR / f"outer{outer}.exitcode"
        exit_path.write_text(f"{exit_code}\n", encoding="ascii")
        results.append(
            {
                "outer": outer,
                "gpu": gpu,
                "pid": process.pid,
                "exit_code": exit_code,
                "log": PATHS.relative(log_path),
            }
        )
        failed = failed or exit_code != 0
        print(f"E213 finished outer={outer} exit={exit_code}", flush=True)

    aggregate_result: dict[str, object] | None = None
    if not failed:
        log_path = ANALYSIS_DIR / "aggregate.log"
        aggregate_command = command(
            "aggregate", args.epochs, args.batch_size, args.channel_batch_size
        )
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        with log_path.open("wb") as handle:
            exit_code = subprocess.call(
                aggregate_command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        (ANALYSIS_DIR / "aggregate.exitcode").write_text(
            f"{exit_code}\n", encoding="ascii"
        )
        aggregate_result = {
            "gpu": gpu_ids[0],
            "exit_code": exit_code,
            "log": PATHS.relative(log_path),
        }
        failed = exit_code != 0
        print(f"E213 aggregate exit={exit_code}", flush=True)

    report = {
        "experiment": "Round2-E213-no-rt-spectral-residual",
        "status": "failed" if failed else "complete",
        "folds": results,
        "aggregate": aggregate_result,
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(RUNNER_REPORT, report)
    print(json.dumps(report, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
