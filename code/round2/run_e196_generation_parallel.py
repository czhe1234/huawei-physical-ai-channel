#!/usr/bin/env python3
"""Run E196 mechanism-spectrum jobs in parallel with resumable logs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from generate_e196_mechanism_spectra import ANALYSIS_DIR, CONFIGS
from project_paths import PATHS


VERSION = "Round2-E196-mechanism-generation-runner"
DEFAULT_GPUS = (4, 5, 0, 1, 2)


@dataclass
class Job:
    config: str
    gpu: int
    command: list[str]
    log: str
    exit_code_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-tag", default="full")
    parser.add_argument("--max-seeds", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-count-per-site", type=int, default=0)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--gpus", default=",".join(map(str, DEFAULT_GPUS)))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    gpus = tuple(int(value.strip()) for value in args.gpus.split(",") if value.strip())
    if len(gpus) < len(CONFIGS):
        raise ValueError(f"need at least {len(CONFIGS)} GPUs, got {gpus}")
    started = time.time()
    run_dir = ANALYSIS_DIR / "logs" / args.selection_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    processes: list[tuple[Job, subprocess.Popen[str], Any]] = []
    try:
        for config, gpu in zip(CONFIGS, gpus):
            log_path = run_dir / f"{config.name}.log"
            exit_path = run_dir / f"{config.name}.exit.json"
            command = [
                str(PATHS.root / "code" / "tools" / "run_in_env.sh"),
                "python",
                "-u",
                str(PATHS.root / "code" / "round2" / "generate_e196_mechanism_spectra.py"),
                "--config",
                config.name,
                "--split",
                args.split,
                "--max-seeds",
                str(args.max_seeds),
                "--batch-size",
                str(args.batch_size),
                "--sample-count-per-site",
                str(args.sample_count_per_site),
                "--selection-tag",
                args.selection_tag,
                "--cuda-device",
                "0",
            ]
            if args.force:
                command.append("--force")
            job = Job(
                config=config.name,
                gpu=gpu,
                command=command,
                log=PATHS.relative(log_path),
                exit_code_path=PATHS.relative(exit_path),
            )
            jobs.append(job)
            handle = log_path.open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "HW_CONDA_ENV": "sionna1",
                    "HW_SKIP_ENV_CHECK": "1",
                }
            )
            process = subprocess.Popen(
                command,
                cwd=PATHS.root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((job, process, handle))
            print(
                f"started E196 config={config.name} gpu={gpu} pid={process.pid}",
                flush=True,
            )

        results = []
        failed = False
        for job, process, handle in processes:
            code = process.wait()
            handle.close()
            payload = {
                "experiment": VERSION,
                "config": job.config,
                "gpu": job.gpu,
                "pid": process.pid,
                "exit_code": int(code),
                "completed_at_unix": time.time(),
                "log": job.log,
            }
            atomic_json(PATHS.root / job.exit_code_path, payload)
            results.append(payload)
            failed |= code != 0
            print(
                f"finished E196 config={job.config} gpu={job.gpu} exit={code}",
                flush=True,
            )
        report = {
            "experiment": VERSION,
            "status": "failed" if failed else "complete",
            "arguments": vars(args),
            "jobs": [asdict(job) for job in jobs],
            "results": results,
            "wall_seconds": float(time.time() - started),
        }
        atomic_json(ANALYSIS_DIR / f"parallel_{args.selection_tag}.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if failed:
            raise SystemExit(1)
    except BaseException:
        error = traceback.format_exc()
        for _, process, handle in processes:
            if process.poll() is None:
                process.terminate()
            try:
                handle.close()
            except Exception:
                pass
        atomic_json(
            ANALYSIS_DIR / f"parallel_{args.selection_tag}_failure.json",
            {
                "experiment": VERSION,
                "status": "runner_failed",
                "arguments": vars(args),
                "jobs": [asdict(job) for job in jobs],
                "traceback": error,
                "wall_seconds": float(time.time() - started),
            },
        )
        raise


if __name__ == "__main__":
    main()
