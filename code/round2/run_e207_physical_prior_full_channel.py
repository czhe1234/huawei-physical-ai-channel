#!/usr/bin/env python3
"""Run E207 prepare, five resumable GPU folds, and fixed500 aggregation."""

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
import train_e207_physical_prior_full_channel as e207


SCRIPT = (
    PATHS.root
    / "code"
    / "round2"
    / "train_e207_physical_prior_full_channel.py"
)
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
    report_path = e207.fold_report_path(outer)
    channel_path = e207.fold_channel_path(outer)
    if not report_path.is_file() or not channel_path.is_file():
        return False
    try:
        return json.loads(report_path.read_text(encoding="utf-8")).get(
            "status"
        ) == "complete"
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


def process_environment() -> dict[str, str]:
    return {
        **os.environ,
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    }


def run_blocking(
    stage: str,
    gpu: int,
    args: argparse.Namespace,
    log_path: Path,
    exit_path: Path,
) -> dict[str, Any]:
    started = time.time()
    with log_path.open("ab") as handle:
        result = subprocess.run(
            command(stage, gpu, args),
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=process_environment(),
            check=False,
        )
    record = {
        "stage": stage,
        "gpu": gpu,
        "log": PATHS.relative(log_path),
        "exit_code": int(result.returncode),
        "seconds": float(time.time() - started),
    }
    e207.e205.atomic_json(exit_path, record)
    if result.returncode != 0:
        raise RuntimeError(
            f"E207 {stage} failed with {result.returncode}; full traceback in {log_path}"
        )
    return record


def main() -> None:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) < 5:
        raise ValueError("E207 requires five GPU ids for five parallel folds")
    if not torch.cuda.is_available():
        raise RuntimeError("E207 runner requires CUDA")
    if min(gpus) < 0 or max(gpus) >= torch.cuda.device_count():
        raise ValueError(f"invalid GPU list {gpus}")

    e207.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = e207.ANALYSIS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_report_path = e207.ANALYSIS_DIR / "parallel_run.json"
    started = time.time()

    prepare = run_blocking(
        "prepare",
        gpus[0],
        args,
        log_dir / "prepare.log",
        log_dir / "prepare_exit.json",
    )

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
            env=process_environment(),
        )
        active[outer] = {
            "process": process,
            "handle": handle,
            "gpu": gpu,
            "log": PATHS.relative(log_path),
            "started": time.time(),
        }
        print(f"E207 launch outer={outer} gpu={gpu}", flush=True)

    while active:
        time.sleep(2.0)
        for outer, job in list(active.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            record = {
                "stage": "fold",
                "outer": outer,
                "gpu": job["gpu"],
                "log": job["log"],
                "exit_code": int(returncode),
                "seconds": float(time.time() - job["started"]),
            }
            e207.e205.atomic_json(log_dir / f"outer{outer}_exit.json", record)
            finished.append(record)
            del active[outer]
            print(
                f"E207 finish outer={outer} gpu={record['gpu']} "
                f"exit={returncode} seconds={record['seconds']:.1f}",
                flush=True,
            )

    failed = [row for row in finished if row["exit_code"] != 0]
    if failed:
        report = {
            "experiment": e207.VERSION,
            "status": "failed",
            "prepare": prepare,
            "folds": finished,
            "failed": failed,
            "seconds": float(time.time() - started),
        }
        e207.e205.atomic_json(run_report_path, report)
        raise RuntimeError(
            f"E207 fold failures; full tracebacks are in the recorded logs: {failed}"
        )

    aggregate = run_blocking(
        "aggregate",
        gpus[0],
        args,
        log_dir / "aggregate.log",
        log_dir / "aggregate_exit.json",
    )
    report = {
        "experiment": e207.VERSION,
        "status": "complete",
        "prepare": prepare,
        "folds": finished,
        "aggregate": aggregate,
        "gpus": gpus[:5],
        "carrier_batch_size": args.carrier_batch_size,
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "parallel_wall_seconds": float(time.time() - started),
        "final_report": PATHS.relative(e207.FULL_REPORT),
    }
    e207.e205.atomic_json(run_report_path, report)
    final = json.loads(e207.FULL_REPORT.read_text(encoding="utf-8"))
    final["performance"]["parallel_wall_time_recorded_by_runner"] = report[
        "parallel_wall_seconds"
    ]
    e207.e205.atomic_json(e207.FULL_REPORT, final)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
