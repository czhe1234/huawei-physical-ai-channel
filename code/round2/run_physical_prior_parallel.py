#!/usr/bin/env python3
"""Run all resumable physical-prior stages across the idle GPU fleet."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from project_paths import PATHS
from round2.data import load_positions
from round2.physical_prior.config import PhysicalPriorConfig
from round2.physical_prior.pipeline import (
    ANALYSIS_DIR,
    FINAL_REPORT,
    RUN_REPORT,
    atomic_json,
    fold_output_path,
    fold_report_path,
)
from round2.physical_prior.tokens import raw_shard_path, raw_shard_report_path
from round2.physical_prior.tracer import (
    route_bank_path,
    scatter_bank_path,
    shard_count,
    trace_path,
    trace_report_path,
)


SCRIPT = PATHS.root / "code" / "round2" / "run_physical_prior.py"
RUN_IN_ENV = PATHS.root / "code" / "tools" / "run_in_env.sh"
LOG_DIR = ANALYSIS_DIR / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--token-shard-size", type=int, default=250)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def command(stage: str, gpu: int | None = None, **kwargs: Any) -> list[str]:
    output = [str(RUN_IN_ENV), "python", "-u", str(SCRIPT), "--stage", stage]
    if gpu is not None:
        output.extend(("--device", f"cuda:{gpu}"))
    for key, value in kwargs.items():
        if value is None:
            continue
        output.extend((f"--{key.replace('_', '-')}", str(value)))
    return output


def run_one(
    stage: str,
    log_path: Path,
    gpu: int | None = None,
    force: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(stage, gpu, **kwargs)
    if force:
        cmd.append("--force")
    started = time.time()
    with log_path.open("ab") as handle:
        result = subprocess.run(
            cmd,
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
    record = {
        "stage": stage,
        "gpu": gpu,
        "exit_code": int(result.returncode),
        "seconds": float(time.time() - started),
        "log": PATHS.relative(log_path),
        "arguments": kwargs,
    }
    atomic_json(log_path.with_suffix(".exit.json"), record)
    if result.returncode != 0:
        raise RuntimeError(f"stage {stage} failed; see {log_path}")
    return record


def run_parallel(
    jobs: list[dict[str, Any]],
    gpus: list[int],
    force: bool,
) -> list[dict[str, Any]]:
    pending = list(jobs)
    active: dict[int, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            log_path = Path(job.pop("log_path"))
            stage = str(job.pop("stage"))
            cmd = command(stage, gpu, **job)
            if force:
                cmd.append("--force")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
            process = subprocess.Popen(
                cmd,
                cwd=PATHS.root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                },
            )
            active[gpu] = {
                "process": process,
                "handle": handle,
                "stage": stage,
                "arguments": job,
                "log": log_path,
                "started": time.time(),
            }
            print(f"launch {stage} gpu={gpu} args={job}", flush=True)
        time.sleep(1.0)
        for gpu, job in list(active.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            record = {
                "stage": job["stage"],
                "gpu": gpu,
                "exit_code": int(returncode),
                "seconds": float(time.time() - job["started"]),
                "log": PATHS.relative(job["log"]),
                "arguments": job["arguments"],
            }
            atomic_json(job["log"].with_suffix(".exit.json"), record)
            finished.append(record)
            del active[gpu]
            print(
                f"finish {record['stage']} gpu={gpu} exit={returncode} "
                f"seconds={record['seconds']:.1f}",
                flush=True,
            )
            if returncode != 0:
                for remaining in active.values():
                    remaining["process"].terminate()
                    remaining["handle"].close()
                raise RuntimeError(f"parallel job failed: {record}")
    return finished


def completed(report_path: Path, output_path: Path) -> bool:
    if not report_path.is_file() or not output_path.is_file():
        return False
    try:
        return json.loads(report_path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
        raise RuntimeError(f"invalid CUDA fleet {gpus}")
    if len(gpus) < 5:
        raise RuntimeError("at least five GPUs are required for parallel folds")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config = PhysicalPriorConfig()
    started = time.time()
    stages: list[dict[str, Any]] = []
    stages.append(
        run_one("geometry", LOG_DIR / "geometry.log", force=args.force)
    )
    stages.append(
        run_one("profile", LOG_DIR / "profile.log", gpu=gpus[0], force=args.force)
    )

    trace_jobs: list[dict[str, Any]] = []
    for site in range(2):
        for branch in config.branches:
            for seed in config.seeds:
                for shard in range(shard_count(branch)):
                    if not args.force and completed(
                        trace_report_path(site, branch.name, seed, shard),
                        trace_path(site, branch.name, seed, shard),
                    ):
                        continue
                    trace_jobs.append(
                        {
                            "stage": "trace",
                            "site": site,
                            "branch": branch.name,
                            "seed": seed,
                            "shard": shard,
                            "log_path": LOG_DIR
                            / "trace"
                            / f"site{site}_{branch.name}_seed{seed}_shard{shard}.log",
                        }
                    )
    trace_results = run_parallel(trace_jobs, gpus, args.force)
    stages.append({"stage": "trace", "jobs": trace_results})

    route_jobs: list[dict[str, Any]] = []
    for site in range(2):
        for branch in config.branches:
            for seed in config.seeds:
                output_path = (
                    route_bank_path(site, seed)
                    if branch.name == config.base.name
                    else scatter_bank_path(site, seed)
                )
                if not args.force and completed(
                    output_path.with_suffix(".json"), output_path
                ):
                    continue
                route_jobs.append(
                    {
                        "stage": "route",
                        "site": site,
                        "branch": branch.name,
                        "seed": seed,
                        "log_path": LOG_DIR
                        / "routes"
                        / f"site{site}_{branch.name}_seed{seed}.log",
                    }
                )
    route_results = run_parallel(route_jobs, gpus, args.force)
    stages.append({"stage": "route", "jobs": route_results})

    _, positions, _, site_id, _ = load_positions()
    token_jobs: list[dict[str, Any]] = []
    for site in range(2):
        count = int(np.sum(site_id == site))
        for shard in range(int(math.ceil(count / args.token_shard_size))):
            if not args.force and completed(
                raw_shard_report_path(site, shard), raw_shard_path(site, shard)
            ):
                continue
            token_jobs.append(
                {
                    "stage": "tokens",
                    "site": site,
                    "shard": shard,
                    "token_shard_size": args.token_shard_size,
                    "log_path": LOG_DIR / "tokens" / f"site{site}_shard{shard}.log",
                }
            )
    token_results = run_parallel(token_jobs, gpus, args.force)
    stages.append({"stage": "tokens", "jobs": token_results})
    stages.append(
        run_one(
            "assemble",
            LOG_DIR / "assemble.log",
            force=args.force,
            token_shard_size=args.token_shard_size,
        )
    )

    fold_jobs = []
    for outer in range(5):
        if not args.force and completed(
            fold_report_path(outer), fold_output_path(outer)
        ):
            continue
        fold_jobs.append(
            {
                "stage": "fold",
                "outer": outer,
                "log_path": LOG_DIR / "folds" / f"outer{outer}.log",
            }
        )
    fold_results = run_parallel(fold_jobs, gpus[:5], args.force)
    stages.append({"stage": "fold", "jobs": fold_results})
    stages.append(run_one("aggregate", LOG_DIR / "aggregate.log", gpu=gpus[0]))
    final = json.loads(FINAL_REPORT.read_text(encoding="utf-8"))
    report = {
        "status": "complete",
        "gpus": gpus,
        "token_shard_size": args.token_shard_size,
        "parallel_wall_seconds": float(time.time() - started),
        "stages": stages,
        "final_report": PATHS.relative(FINAL_REPORT),
        "final_status": final["status"],
        "fixed500": final["fixed500"],
    }
    atomic_json(RUN_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
