#!/usr/bin/env python3
"""Run E200 finalist generation, preparation, strict folds, and aggregation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

from e200_config_search import portable_payload
from project_paths import PATHS


NAMESPACE = os.environ.get("HW_E200_NAMESPACE", "e200").strip().lower()
if NAMESPACE not in {"e200", "e201"}:
    raise ValueError(f"unsupported E200-compatible namespace: {NAMESPACE}")
VERSION = f"Round2-{NAMESPACE.upper()}-finalist-pipeline-runner"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / f"round2_{NAMESPACE}_finalist_pipeline"
)
LOG_DIR = ANALYSIS_DIR / "logs"
REPORT = ANALYSIS_DIR / "run.json"
GENERATOR = PATHS.round2_code / "generate_e200_finalist_paths.py"
PREPARE = PATHS.round2_code / "prepare_e200_candidate_cache.py"
TRAIN = PATHS.round2_code / "train_e200_finalist_calibrator.py"
RUN_ENV = PATHS.root / "code" / "tools" / "run_in_env.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", "generate", "prepare", "folds", "aggregate"), default="all"
    )
    parser.add_argument("--generation-gpus", default="4,5,0,1")
    parser.add_argument("--fold-gpus", default="0,1,2,3,4")
    parser.add_argument("--aggregate-gpu", default="5")
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(portable_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def environment(gpu: str, sionna: bool = False) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu,
            "E200_PHYSICAL_GPU": gpu,
            "HW_E200_NAMESPACE": NAMESPACE,
            "HW_SKIP_ENV_CHECK": "1",
        }
    )
    if sionna:
        value["HW_CONDA_ENV"] = "sionna1"
    return value


def launch_group(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    running = []
    for job in jobs:
        job["log"].parent.mkdir(parents=True, exist_ok=True)
        handle = job["log"].open("wb")
        process = subprocess.Popen(
            job["command"],
            cwd=PATHS.root,
            env=job["environment"],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        running.append({**job, "process": process, "handle": handle, "started": time.time()})
    results = []
    for job in running:
        returncode = job["process"].wait()
        job["handle"].close()
        result = {
            "name": job["name"],
            "gpu": job["gpu"],
            "returncode": int(returncode),
            "seconds": float(time.time() - job["started"]),
            "log": PATHS.relative(job["log"]),
        }
        exit_path = job["log"].with_suffix(".exit.json")
        if returncode != 0:
            result["traceback"] = "\n".join(
                job["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-300:]
            )
        atomic_json(exit_path, result)
        results.append(result)
    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        raise RuntimeError(f"E200 parallel jobs failed: {failed}")
    return results


def run_sync(
    name: str,
    command: list[str],
    gpu: str,
    sionna: bool = False,
) -> dict[str, Any]:
    log = LOG_DIR / f"{name}.log"
    started = time.time()
    with log.open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=PATHS.root,
            env=environment(gpu, sionna=sionna),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        "name": name,
        "gpu": gpu,
        "returncode": int(completed.returncode),
        "seconds": float(time.time() - started),
        "log": PATHS.relative(log),
    }
    if completed.returncode != 0:
        result["traceback"] = "\n".join(
            log.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]
        )
    atomic_json(log.with_suffix(".exit.json"), result)
    if completed.returncode != 0:
        raise RuntimeError(result["traceback"])
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "experiment": VERSION,
        "status": "running",
        "configuration": vars(args),
        "stages": {},
    }
    atomic_json(REPORT, report)
    try:
        if args.stage in ("all", "generate"):
            gpus = [value.strip() for value in args.generation_gpus.split(",") if value.strip()]
            if len(gpus) < 4:
                raise ValueError("E200 generation needs four GPU assignments")
            jobs = []
            for source in range(4):
                command = [
                    str(RUN_ENV),
                    "python",
                    "-u",
                    str(GENERATOR),
                    "--source",
                    str(source),
                    "--batch-size",
                    str(args.generation_batch_size),
                    "--cuda-device",
                    "0",
                ]
                if args.force:
                    command.append("--force")
                jobs.append(
                    {
                        "name": f"generate_source{source}",
                        "gpu": gpus[source],
                        "command": command,
                        "environment": environment(gpus[source], sionna=True),
                        "log": LOG_DIR / f"generate_source{source}.log",
                    }
                )
            report["stages"]["generate"] = launch_group(jobs)
            atomic_json(REPORT, report)
            if args.stage == "generate":
                report["status"] = "complete"
                report["seconds"] = float(time.time() - started)
                atomic_json(REPORT, report)
                return

        if args.stage in ("all", "prepare"):
            command = [str(RUN_ENV), "python", "-u", str(PREPARE)]
            if args.force:
                command.append("--force")
            report["stages"]["prepare_cache"] = run_sync(
                "prepare_cache", command, args.aggregate_gpu
            )
            label_command = [
                str(RUN_ENV),
                "python",
                "-u",
                str(TRAIN),
                "--stage",
                "prepare",
                "--device",
                "cuda:0",
            ]
            if args.force:
                label_command.append("--force")
            report["stages"]["prepare_labels"] = run_sync(
                "prepare_labels", label_command, args.aggregate_gpu
            )
            atomic_json(REPORT, report)
            if args.stage == "prepare":
                report["status"] = "complete"
                report["seconds"] = float(time.time() - started)
                atomic_json(REPORT, report)
                return

        if args.stage in ("all", "folds"):
            gpus = [value.strip() for value in args.fold_gpus.split(",") if value.strip()]
            if len(gpus) < 5:
                raise ValueError("E200 folds need five GPU assignments")
            jobs = []
            for outer in range(5):
                command = [
                    str(RUN_ENV),
                    "python",
                    "-u",
                    str(TRAIN),
                    "--stage",
                    "fold",
                    "--outer",
                    str(outer),
                    "--device",
                    "cuda:0",
                ]
                if args.force:
                    command.append("--force")
                jobs.append(
                    {
                        "name": f"fold{outer}",
                        "gpu": gpus[outer],
                        "command": command,
                        "environment": environment(gpus[outer]),
                        "log": LOG_DIR / f"fold{outer}.log",
                    }
                )
            report["stages"]["folds"] = launch_group(jobs)
            atomic_json(REPORT, report)
            if args.stage == "folds":
                report["status"] = "complete"
                report["seconds"] = float(time.time() - started)
                atomic_json(REPORT, report)
                return

        if args.stage in ("all", "aggregate"):
            command = [
                str(RUN_ENV),
                "python",
                "-u",
                str(TRAIN),
                "--stage",
                "aggregate",
                "--device",
                "cuda:0",
            ]
            report["stages"]["aggregate"] = run_sync(
                "aggregate", command, args.aggregate_gpu
            )
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = float(time.time() - started)
        atomic_json(REPORT, report)
        print(json.dumps(portable_payload(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
