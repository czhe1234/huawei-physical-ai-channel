#!/usr/bin/env python3
"""Run the five resumable E211 folds concurrently and aggregate fixed500."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from project_paths import PATHS


VERSION = "Round2-E211-parallel-runner"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e211_full_path_state_residual"
)
LOG_DIR = ANALYSIS_DIR / "logs" / "training"
RUN_REPORT = ANALYSIS_DIR / "parallel_run.json"
SCRIPT = PATHS.round2_code / "train_e211_full_path_state_residual.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,3,2,4")
    parser.add_argument("--aggregate-device", default="5")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def environment() -> dict[str, str]:
    return {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
    }


def command(*arguments: str) -> list[str]:
    return [str(Path(sys.executable)), "-u", str(SCRIPT), *arguments]


def gpu_state() -> list[tuple[int, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    output = []
    for line in completed.stdout.splitlines():
        memory, utilization = line.split(",")
        output.append((int(memory.strip()), int(utilization.strip())))
    return output


def failure_tail(path: Path, lines: int = 160) -> str:
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) < 5:
        raise ValueError("E211 requires five fold devices")
    devices = devices[:5]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload: dict[str, Any] = {
        "experiment": VERSION,
        "status": "running",
        "devices": [f"cuda:{device}" for device in devices],
        "aggregate_device": f"cuda:{args.aggregate_device}",
        "started_unix": started,
        "folds": [],
    }
    peak_memory = {device: 0 for device in devices}
    peak_utilization = {device: 0 for device in devices}
    atomic_json(RUN_REPORT, payload)

    processes: dict[int, tuple[str, subprocess.Popen[bytes], Any, Path, float]] = {}
    for outer, device in enumerate(devices):
        fold_command = command(
            "--stage", "fold", "--outer", str(outer), "--device", f"cuda:{device}"
        )
        if args.force:
            fold_command.append("--force")
        log_path = LOG_DIR / f"outer{outer}.log"
        handle = log_path.open("wb")
        process = subprocess.Popen(
            fold_command,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment(),
        )
        processes[outer] = (device, process, handle, log_path, time.time())
        print(
            f"started E211 outer={outer} device=cuda:{device} pid={process.pid}",
            flush=True,
        )

    failures: list[dict[str, Any]] = []
    while processes:
        state = gpu_state()
        for device in devices:
            index = int(device)
            if index < len(state):
                peak_memory[device] = max(peak_memory[device], state[index][0])
                peak_utilization[device] = max(
                    peak_utilization[device], state[index][1]
                )
        completed_outers = []
        for outer, (device, process, handle, log_path, fold_started) in processes.items():
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            record = {
                "outer": outer,
                "device": f"cuda:{device}",
                "pid": process.pid,
                "returncode": int(returncode),
                "log": PATHS.relative(log_path),
                "seconds": float(time.time() - fold_started),
            }
            atomic_json(log_path.with_suffix(".exit.json"), record)
            payload["folds"].append(record)
            payload["folds"].sort(key=lambda row: int(row["outer"]))
            if returncode != 0:
                failures.append(record)
            completed_outers.append(outer)
            print(f"finished E211 outer={outer} returncode={returncode}", flush=True)
        for outer in completed_outers:
            del processes[outer]
        payload["peak_memory_mib"] = peak_memory
        payload["peak_utilization_percent"] = peak_utilization
        payload["wall_seconds_before_aggregate"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        if processes:
            time.sleep(2.0)

    if failures:
        payload["status"] = "fold_failure"
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        excerpts = []
        for record in failures:
            path = PATHS.root / str(record["log"])
            excerpts.append(f"outer{record['outer']}:\n{failure_tail(path)}")
        raise RuntimeError("\n\n".join(excerpts))

    aggregate_log = LOG_DIR / "aggregate.log"
    aggregate_command = command(
        "--stage", "aggregate", "--device", f"cuda:{args.aggregate_device}"
    )
    aggregate_started = time.time()
    with aggregate_log.open("wb") as handle:
        completed = subprocess.run(
            aggregate_command,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment(),
            check=False,
        )
    aggregate = {
        "returncode": int(completed.returncode),
        "log": PATHS.relative(aggregate_log),
        "seconds": float(time.time() - aggregate_started),
    }
    atomic_json(aggregate_log.with_suffix(".exit.json"), aggregate)
    payload["aggregate"] = aggregate
    payload["status"] = "complete" if completed.returncode == 0 else "aggregate_failure"
    payload["wall_seconds"] = float(time.time() - started)
    atomic_json(RUN_REPORT, payload)
    if completed.returncode != 0:
        raise RuntimeError(f"E211 aggregate failed:\n{failure_tail(aggregate_log)}")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
