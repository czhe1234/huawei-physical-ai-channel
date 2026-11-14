#!/usr/bin/env python3
"""Run E205 prepare, five resumable folds, and fixed500 aggregation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from project_paths import PATHS
import train_e205_ordered_map_path_calibrator as e205


SCRIPT = PATHS.root / "code" / "round2" / "train_e205_ordered_map_path_calibrator.py"
RUN_IN_ENV = PATHS.root / "code" / "tools" / "run_in_env.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,3,2,4")
    parser.add_argument("--carrier-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def complete(outer: int) -> bool:
    path = e205.fold_report_path(outer)
    if not path.is_file() or not e205.fold_channel_path(outer).is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def command(
    stage: str,
    gpu: int,
    args: argparse.Namespace,
    outer: int | None = None,
) -> list[str]:
    output = [
        str(RUN_IN_ENV),
        "python",
        str(SCRIPT),
        "--stage",
        stage,
        "--device",
        f"cuda:{gpu}",
        "--carrier-batch-size",
        str(args.carrier_batch_size),
        "--train-batch-size",
        str(args.train_batch_size),
        "--epochs",
        str(args.epochs),
    ]
    if outer is not None:
        output.extend(("--outer", str(outer)))
    if args.force:
        output.append("--force")
    return output


def run_blocking(
    stage: str,
    gpu: int,
    args: argparse.Namespace,
    log_path: Path,
) -> None:
    with log_path.open("ab") as handle:
        result = subprocess.run(
            command(stage, gpu, args),
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
            },
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"E205 {stage} failed with {result.returncode}; see {log_path}")


def main() -> None:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) < 5:
        raise ValueError("E205 requires at least five GPU ids for five parallel folds")
    if not torch.cuda.is_available():
        raise RuntimeError("E205 runner requires CUDA")
    if max(gpus) >= torch.cuda.device_count():
        raise ValueError(f"invalid GPU list {gpus}")
    e205.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = e205.ANALYSIS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_report_path = e205.ANALYSIS_DIR / "parallel_run.json"
    started = time.time()

    prepare_log = log_dir / "prepare.log"
    run_blocking("prepare", gpus[0], args, prepare_log)
    prepare_seconds = float(time.time() - started)

    jobs = [outer for outer in range(5) if args.force or not complete(outer)]
    active: dict[int, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    for outer in jobs:
        gpu = gpus[outer]
        log_path = log_dir / f"outer{outer}.log"
        handle = log_path.open("ab")
        process = subprocess.Popen(
            command("fold", gpu, args, outer),
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
            },
        )
        active[outer] = {
            "process": process,
            "handle": handle,
            "gpu": gpu,
            "log": PATHS.relative(log_path),
            "started": time.time(),
        }
        print(f"E205 launch outer={outer} gpu={gpu}", flush=True)

    while active:
        time.sleep(2.0)
        for outer, job in list(active.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            record = {
                "outer": outer,
                "gpu": job["gpu"],
                "log": job["log"],
                "exit_code": int(returncode),
                "seconds": float(time.time() - job["started"]),
            }
            e205.atomic_json(log_dir / f"outer{outer}_exit.json", record)
            finished.append(record)
            del active[outer]
            print(
                f"E205 finish outer={outer} gpu={record['gpu']} "
                f"exit={returncode} seconds={record['seconds']:.1f}",
                flush=True,
            )

    failed = [row for row in finished if row["exit_code"] != 0]
    if failed:
        report = {
            "experiment": e205.VERSION,
            "status": "failed",
            "prepare_seconds": prepare_seconds,
            "folds": finished,
            "failed": failed,
            "seconds": float(time.time() - started),
        }
        e205.atomic_json(run_report_path, report)
        raise RuntimeError(f"E205 fold failures: {failed}")

    aggregate_started = time.time()
    aggregate_log = log_dir / "aggregate.log"
    run_blocking("aggregate", gpus[0], args, aggregate_log)
    report = {
        "experiment": e205.VERSION,
        "status": "complete",
        "prepare_seconds": prepare_seconds,
        "folds": finished,
        "aggregate_seconds": float(time.time() - aggregate_started),
        "gpus": gpus[:5],
        "carrier_batch_size": args.carrier_batch_size,
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "parallel_wall_seconds": float(time.time() - started),
        "final_report": PATHS.relative(e205.FULL_REPORT),
    }
    e205.atomic_json(run_report_path, report)
    if e205.FULL_REPORT.is_file():
        final = json.loads(e205.FULL_REPORT.read_text(encoding="utf-8"))
        final["performance"]["parallel_wall_time_recorded_by_runner"] = report[
            "parallel_wall_seconds"
        ]
        e205.atomic_json(e205.FULL_REPORT, final)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
