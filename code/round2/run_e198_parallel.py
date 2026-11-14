#!/usr/bin/env python3
"""Profile, resume, run, and aggregate the five strict E198 folds."""

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


VERSION = "Round2-E198-parallel-runner"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e198_cross_round_complex_field"
LOG_DIR = ANALYSIS_DIR / "logs"
RUN_REPORT = ANALYSIS_DIR / "parallel_run.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4")
    parser.add_argument("--prepare-device", default="4")
    parser.add_argument("--aggregate-device", default="5")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-fold", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def command(script: Path, *arguments: str) -> list[str]:
    return [str(Path(sys.executable)), "-u", str(script), *arguments]


def run_stage(command_value: list[str], log_path: Path, stage: str) -> dict[str, Any]:
    started = time.time()
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command_value,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    record = {
        "stage": stage,
        "returncode": completed.returncode,
        "log": PATHS.relative(log_path),
        "seconds": float(time.time() - started),
    }
    if completed.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError(f"E198 {stage} failed:\n" + "\n".join(lines[-200:]))
    return record


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) < 5:
        raise ValueError("E198 requires five fold devices")
    devices = devices[:5]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    prepare_script = PATHS.round2_code / "prepare_e198_cross_round_complex_field.py"
    train_script = PATHS.round2_code / "train_e198_cross_round_complex_field.py"
    started = time.time()
    payload: dict[str, Any] = {
        "experiment": VERSION,
        "status": "running",
        "devices": [f"cuda:{device}" for device in devices],
        "prepare_device": f"cuda:{args.prepare_device}",
        "aggregate_device": f"cuda:{args.aggregate_device}",
        "started_unix": started,
        "folds": [],
    }
    atomic_json(RUN_REPORT, payload)

    prepare_command = command(prepare_script)
    if args.force_prepare:
        prepare_command.append("--force")
    payload["prepare"] = run_stage(
        prepare_command, LOG_DIR / "prepare.log", "prepare"
    )
    atomic_json(RUN_REPORT, payload)

    if not args.skip_profile:
        payload["profile"] = run_stage(
            command(
                train_script,
                "--stage",
                "profile",
                "--outer",
                "0",
                "--device",
                f"cuda:{args.prepare_device}",
            ),
            LOG_DIR / "profile.log",
            "profile",
        )
        atomic_json(RUN_REPORT, payload)

    processes: dict[int, tuple[str, subprocess.Popen[bytes], Any, Path, float]] = {}
    for outer, device in enumerate(devices):
        fold_command = command(
            train_script,
            "--stage",
            "fold",
            "--outer",
            str(outer),
            "--device",
            f"cuda:{device}",
        )
        if args.force_fold:
            fold_command.append("--force-fold")
        log_path = LOG_DIR / f"outer{outer}.log"
        handle = log_path.open("wb")
        process = subprocess.Popen(
            fold_command,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        processes[outer] = (device, process, handle, log_path, time.time())
        print(
            f"started E198 outer={outer} device=cuda:{device} pid={process.pid}",
            flush=True,
        )

    failures: list[dict[str, Any]] = []
    while processes:
        complete = []
        for outer, (device, process, handle, log_path, fold_started) in processes.items():
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            record = {
                "outer": outer,
                "device": f"cuda:{device}",
                "pid": process.pid,
                "returncode": returncode,
                "log": PATHS.relative(log_path),
                "seconds": float(time.time() - fold_started),
            }
            payload["folds"].append(record)
            payload["folds"].sort(key=lambda row: int(row["outer"]))
            if returncode != 0:
                failures.append(record)
            complete.append(outer)
            print(f"finished E198 outer={outer} returncode={returncode}", flush=True)
        for outer in complete:
            del processes[outer]
        payload["wall_seconds_before_aggregate"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        if processes:
            time.sleep(1.0)

    if failures:
        payload["status"] = "fold_failure"
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        excerpts = []
        for record in failures:
            log_path = PATHS.root / str(record["log"])
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            excerpts.append(f"outer{record['outer']}:\n" + "\n".join(lines[-200:]))
        raise RuntimeError("\n\n".join(excerpts))

    payload["aggregate"] = run_stage(
        command(
            train_script,
            "--stage",
            "aggregate",
            "--device",
            f"cuda:{args.aggregate_device}",
        ),
        LOG_DIR / "aggregate.log",
        "aggregate",
    )
    payload["status"] = "complete"
    payload["wall_seconds"] = float(time.time() - started)
    atomic_json(RUN_REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
