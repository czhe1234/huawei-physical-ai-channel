#!/usr/bin/env python3
"""Run resumable E203 variant/fold jobs across idle CUDA devices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

import analyze_e203_v15_information_contribution as e203
from project_paths import PATHS


SCRIPT = PATHS.root / "code" / "round2" / "analyze_e203_v15_information_contribution.py"
RUN_IN_ENV = PATHS.root / "code" / "tools" / "run_in_env.sh"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=tuple(e203.EXPERIMENTS), default="v7")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def complete(variant: str, outer: int) -> bool:
    path = e203.fold_report_path(variant, outer)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    e203.configure_experiment(args.base)
    run_report = e203.ANALYSIS_DIR / "parallel_run.json"
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("at least one GPU is required")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("E203 parallel runner requires CUDA")
    if max(gpus) >= torch.cuda.device_count():
        raise ValueError(f"invalid GPU list {gpus}")
    e203.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = e203.ANALYSIS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (variant, outer)
        for variant in e203.VARIANTS
        for outer in range(5)
        if args.force or not complete(variant, outer)
    ]
    started = time.time()
    active: dict[tuple[int, int], dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    next_job = 0
    while next_job < len(jobs) or active:
        for gpu in gpus:
            for slot in range(args.workers_per_gpu):
                key = (gpu, slot)
                if key in active or next_job >= len(jobs):
                    continue
                variant, outer = jobs[next_job]
                next_job += 1
                log_path = log_dir / f"{variant}_outer{outer}.log"
                handle = log_path.open("ab")
                command = [
                    str(RUN_IN_ENV),
                    "python",
                    str(SCRIPT),
                    "--stage",
                    "train",
                    "--base",
                    args.base,
                    "--variant",
                    variant,
                    "--outer",
                    str(outer),
                    "--device",
                    f"cuda:{gpu}",
                    "--batch-size",
                    str(args.batch_size),
                    "--epochs",
                    str(args.epochs),
                ]
                if args.force:
                    command.append("--force")
                environment = dict(os.environ)
                environment.update(
                    {
                        "OMP_NUM_THREADS": "4",
                        "MKL_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "4",
                    }
                )
                process = subprocess.Popen(
                    command,
                    cwd=PATHS.root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                active[key] = {
                    "process": process,
                    "handle": handle,
                    "variant": variant,
                    "outer": outer,
                    "gpu": gpu,
                    "slot": slot,
                    "log": PATHS.relative(log_path),
                    "started": time.time(),
                }
                print(
                    f"{e203.VERSION} launch {variant}/outer{outer} gpu={gpu} slot={slot}",
                    flush=True,
                )
        time.sleep(1.0)
        for key, job in list(active.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            record = {
                key: value
                for key, value in job.items()
                if key not in ("process", "handle", "started")
            }
            record.update(
                {
                    "exit_code": int(returncode),
                    "seconds": float(time.time() - job["started"]),
                }
            )
            status_path = log_dir / f"{job['variant']}_outer{job['outer']}_exit.json"
            e203.atomic_json(status_path, record)
            finished.append(record)
            del active[key]
            print(
                f"{e203.VERSION} finish {job['variant']}/outer{job['outer']} "
                f"gpu={job['gpu']} slot={job['slot']} exit={returncode} "
                f"seconds={record['seconds']:.1f}",
                flush=True,
            )
            if returncode != 0:
                report = {
                    "status": "failed",
                    "failed_job": record,
                    "completed_jobs": finished,
                    "seconds": float(time.time() - started),
                }
                e203.atomic_json(run_report, report)
                for other in active.values():
                    other["process"].terminate()
                for other in active.values():
                    other["process"].wait()
                    other["handle"].close()
                raise RuntimeError(
                    f"E203 failed at {job['variant']}/outer{job['outer']}; see {job['log']}"
                )
    report = {
        "status": "folds_complete",
        "requested_jobs": len(jobs),
        "completed_jobs": finished,
        "gpus": list(gpus),
        "workers_per_gpu": args.workers_per_gpu,
        "seconds": float(time.time() - started),
    }
    e203.atomic_json(run_report, report)
    aggregate_log = log_dir / "aggregate.log"
    command = [
        str(RUN_IN_ENV),
        "python",
        str(SCRIPT),
        "--stage",
        "aggregate",
        "--base",
        args.base,
        "--device",
        f"cuda:{gpus[0]}",
    ]
    with aggregate_log.open("ab") as handle:
        result = subprocess.run(
            command,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or not e203.FINAL_REPORT.is_file():
        raise RuntimeError(
            f"E203 aggregate failed with {result.returncode}; see {aggregate_log}"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
