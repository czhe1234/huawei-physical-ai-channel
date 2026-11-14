#!/usr/bin/env python3
"""Build all E208 source-aligned caches across the available GPU fleet."""

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
from round2.physical_prior.source_aligned import (
    ANALYSIS_DIR,
    RAW_TOKEN_CACHE,
    atomic_json,
    diffuse_path,
    route_path,
    shard_count,
    token_shard_path,
    trace_path,
)
from round2.physical_prior.source_aligned_config import SourceAlignedConfig


SCRIPT = PATHS.root / "code" / "round2" / "run_e208_source_aligned_prior.py"
RUN_IN_ENV = PATHS.root / "code" / "tools" / "run_in_env.sh"
LOG_DIR = ANALYSIS_DIR / "logs"
RUN_REPORT = ANALYSIS_DIR / "parallel_run.json"


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
        output.extend((f"--{key.replace('_', '-')}", str(value)))
    return output


def completed(output: Path) -> bool:
    report = output.with_suffix(".json")
    if not output.is_file() or not report.is_file():
        return False
    try:
        return json.loads(report.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def environment(threads: int) -> dict[str, str]:
    return {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }


def run_one(
    stage: str,
    log_path: Path,
    *,
    gpu: int | None = None,
    force: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    cmd = command(stage, gpu, **kwargs)
    if force:
        cmd.append("--force")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("ab") as handle:
        result = subprocess.run(
            cmd,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment(8),
            check=False,
        )
    record = {
        "stage": stage,
        "gpu": gpu,
        "arguments": kwargs,
        "exit_code": int(result.returncode),
        "seconds": float(time.time() - started),
        "log": PATHS.relative(log_path),
    }
    atomic_json(log_path.with_suffix(".exit.json"), record)
    if result.returncode != 0:
        raise RuntimeError(f"E208 stage {stage} failed; see {log_path}")
    return record


def run_parallel(
    jobs: list[dict[str, Any]], gpus: list[int], force: bool
) -> list[dict[str, Any]]:
    pending = list(jobs)
    active: dict[int, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            stage = str(job.pop("stage"))
            log_path = Path(job.pop("log_path"))
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
                env=environment(4),
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
                "arguments": job["arguments"],
                "exit_code": int(returncode),
                "seconds": float(time.time() - job["started"]),
                "log": PATHS.relative(job["log"]),
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
                raise RuntimeError(f"E208 parallel job failed: {record}")
    return finished


def main() -> None:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
        raise RuntimeError(f"invalid CUDA fleet {gpus}")
    config = SourceAlignedConfig()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stages: list[dict[str, Any]] = []
    stages.append(run_one("geometry", LOG_DIR / "geometry.log", force=args.force))
    stages.append(
        run_one(
            "profile",
            LOG_DIR / "profile.log",
            gpu=gpus[0],
            force=args.force,
        )
    )

    trace_jobs: list[dict[str, Any]] = []
    for site in range(2):
        for branch in config.branches:
            for seed in config.seeds:
                for shard in range(shard_count(branch)):
                    output = trace_path(site, branch.name, seed, shard)
                    if not args.force and completed(output):
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
    stages.append(
        {"stage": "trace", "jobs": run_parallel(trace_jobs, gpus, args.force)}
    )

    route_jobs: list[dict[str, Any]] = []
    for site in range(2):
        for branch in config.branches:
            for seed in config.seeds:
                output = route_path(site, branch.name, seed)
                if not args.force and completed(output):
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
    stages.append(
        {"stage": "route", "jobs": run_parallel(route_jobs, gpus, args.force)}
    )

    diffuse_jobs: list[dict[str, Any]] = []
    for site in range(2):
        for seed in config.seeds:
            output = diffuse_path(site, seed)
            if not args.force and completed(output):
                continue
            diffuse_jobs.append(
                {
                    "stage": "diffuse",
                    "site": site,
                    "seed": seed,
                    "log_path": LOG_DIR / "diffuse" / f"site{site}_seed{seed}.log",
                }
            )
    stages.append(
        {"stage": "diffuse", "jobs": run_parallel(diffuse_jobs, gpus, args.force)}
    )

    _, positions, _, site_id, _ = load_positions()
    token_jobs: list[dict[str, Any]] = []
    for site in range(2):
        count = int(np.sum(site_id == site))
        for shard in range(int(math.ceil(count / args.token_shard_size))):
            output = token_shard_path(site, shard)
            if not args.force and completed(output):
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
    stages.append(
        {"stage": "tokens", "jobs": run_parallel(token_jobs, gpus, args.force)}
    )
    stages.append(
        run_one(
            "assemble",
            LOG_DIR / "assemble.log",
            force=args.force,
            token_shard_size=args.token_shard_size,
        )
    )
    if not RAW_TOKEN_CACHE.is_file():
        raise FileNotFoundError(RAW_TOKEN_CACHE)
    report = {
        "experiment": "Round2-E208-source-aligned-prior",
        "status": "complete",
        "gpus": gpus,
        "token_shard_size": args.token_shard_size,
        "stages": stages,
        "parallel_wall_seconds": float(time.time() - started),
        "raw_tokens": PATHS.relative(RAW_TOKEN_CACHE),
    }
    atomic_json(RUN_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

