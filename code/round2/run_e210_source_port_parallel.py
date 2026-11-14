#!/usr/bin/env python3
"""Build E210 supports on all idle GPUs and assemble raw512 PathTokens."""

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
from round2.physical_prior.source_port_common import CONFIGS, E185_SEEDS
from round2.physical_prior.source_port_tokens import (
    ANALYSIS_DIR,
    RAW_TOKEN_CACHE,
    TOKEN_REPORT,
    VERSION,
    atomic_json,
    batch_count,
    complete,
    support_path,
)


SCRIPT = PATHS.root / "code" / "round2" / "run_e210_source_port_prior.py"
RUN_IN_ENV = PATHS.root / "code" / "tools" / "run_in_env.sh"
LOG_DIR = ANALYSIS_DIR / "logs" / "physical"
RUN_REPORT = ANALYSIS_DIR / "physical_parallel_run.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,3,2,4,5,6,7")
    parser.add_argument("--token-workers", type=int, default=24)
    parser.add_argument("--physical-env")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def environment(threads: int, physical_env: str | None = None) -> dict[str, str]:
    output = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }
    if physical_env:
        output["HW_CONDA_ENV"] = physical_env
        output["HW_SKIP_ENV_CHECK"] = "1"
    return output


def command_support(job: dict[str, Any], gpu: int, force: bool) -> list[str]:
    output = [
        str(RUN_IN_ENV),
        "python",
        "-u",
        str(SCRIPT),
        "--stage",
        "support",
        "--config",
        str(job["config"]),
        "--seed",
        str(job["seed"]),
        "--site",
        str(job["site"]),
        "--cuda-device",
        str(gpu),
    ]
    if force:
        output.append("--force")
    return output


def support_complete(config: str, seed: int, site: int) -> bool:
    return all(
        complete(support_path(config, seed, site, batch))
        for batch in range(batch_count(site))
    )


def gpu_state() -> list[tuple[int, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output: list[tuple[int, int]] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            memory, utilization = line.split(",")
            output.append((int(memory.strip()), int(utilization.strip())))
    return output


def run_blocking(stage: str, workers: int, force: bool) -> dict[str, Any]:
    command = [
        str(RUN_IN_ENV),
        "python",
        "-u",
        str(SCRIPT),
        "--stage",
        stage,
    ]
    if stage == "tokens":
        command.extend(("--workers", str(workers)))
    if force:
        command.append("--force")
    log_path = LOG_DIR / f"{stage}.log"
    started = time.time()
    with log_path.open("ab") as handle:
        result = subprocess.run(
            command,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment(1 if stage == "tokens" else 8),
            check=False,
        )
    record = {
        "stage": stage,
        "exit_code": int(result.returncode),
        "seconds": float(time.time() - started),
        "log": PATHS.relative(log_path),
    }
    atomic_json(log_path.with_suffix(".exit.json"), record)
    if result.returncode != 0:
        raise RuntimeError(f"E210 {stage} failed; see {log_path}")
    return record


def main() -> None:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
        raise RuntimeError(f"invalid CUDA fleet {gpus}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs = [
        {"config": config, "seed": seed, "site": site}
        for config in ("scatter", "base")
        for seed in E185_SEEDS
        for site in (0, 1)
        if args.force or not support_complete(config, seed, site)
    ]
    pending = list(jobs)
    active: dict[int, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    peak_memory = {gpu: 0 for gpu in gpus}
    peak_utilization = {gpu: 0 for gpu in gpus}
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            log_path = (
                LOG_DIR
                / "support"
                / f"{job['config']}_seed{job['seed']}_site{job['site']}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
            process = subprocess.Popen(
                command_support(job, gpu, args.force),
                cwd=PATHS.root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment(4, args.physical_env),
            )
            active[gpu] = {
                **job,
                "process": process,
                "handle": handle,
                "log": log_path,
                "started": time.time(),
            }
            print(f"E210 launch gpu={gpu} job={job}", flush=True)
        time.sleep(2.0)
        state = gpu_state()
        for gpu in gpus:
            if gpu < len(state):
                peak_memory[gpu] = max(peak_memory[gpu], state[gpu][0])
                peak_utilization[gpu] = max(peak_utilization[gpu], state[gpu][1])
        for gpu, job in list(active.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            record = {
                "stage": "support",
                "gpu": gpu,
                "config": job["config"],
                "seed": job["seed"],
                "site": job["site"],
                "exit_code": int(returncode),
                "seconds": float(time.time() - job["started"]),
                "log": PATHS.relative(job["log"]),
            }
            atomic_json(job["log"].with_suffix(".exit.json"), record)
            finished.append(record)
            del active[gpu]
            print(
                f"E210 finish gpu={gpu} config={record['config']} "
                f"seed={record['seed']} site={record['site']} "
                f"exit={returncode} seconds={record['seconds']:.1f}",
                flush=True,
            )
            if returncode != 0:
                for remaining in active.values():
                    remaining["process"].terminate()
                    remaining["handle"].close()
                raise RuntimeError(f"E210 support failed: {record}")

    token_stage = run_blocking("tokens", args.token_workers, args.force)
    assemble_stage = run_blocking("assemble", args.token_workers, args.force)
    report = {
        "status": "complete",
        "version": VERSION,
        "support_jobs": finished,
        "tokens": token_stage,
        "assemble": assemble_stage,
        "gpus": gpus,
        "physical_env": args.physical_env,
        "peak_memory_mib": peak_memory,
        "peak_utilization_percent": peak_utilization,
        "parallel_wall_seconds": float(time.time() - started),
        "raw_token_cache": PATHS.relative(RAW_TOKEN_CACHE),
        "raw_token_report": PATHS.relative(TOKEN_REPORT),
    }
    atomic_json(RUN_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
