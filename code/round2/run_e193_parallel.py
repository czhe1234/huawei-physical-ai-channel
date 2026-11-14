#!/usr/bin/env python3
"""Run the five independent E193 folds concurrently and aggregate them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from project_paths import PATHS


VERSION = "Round2-E193-parallel-runner"
LOG_DIR = PATHS.analysis / "results" / "round2_e193_propagation_memory" / "logs"
RUN_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e193_propagation_memory"
    / "parallel_run.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) < 5:
        raise ValueError("E193 requires five device entries for five concurrent folds")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    script = PATHS.round2_code / "train_e193_propagation_memory.py"
    started = time.time()
    processes: list[tuple[int, str, subprocess.Popen[bytes], object, Path]] = []
    for outer in range(5):
        device = devices[outer]
        log_path = LOG_DIR / f"outer{outer}.log"
        handle = log_path.open("wb")
        command = [
            str(python),
            "-u",
            str(script),
            "--stage",
            "fold",
            "--outer",
            str(outer),
            "--device",
            f"cuda:{device}",
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
        ]
        if args.force:
            command.append("--force")
        process = subprocess.Popen(
            command,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        processes.append((outer, device, process, handle, log_path))
        print(
            f"started E193 outer={outer} device=cuda:{device} pid={process.pid}",
            flush=True,
        )

    results = []
    failed = []
    for outer, device, process, handle, log_path in processes:
        code = process.wait()
        handle.close()
        record = {
            "outer": outer,
            "device": f"cuda:{device}",
            "pid": process.pid,
            "returncode": code,
            "log": PATHS.relative(log_path),
        }
        results.append(record)
        if code != 0:
            failed.append(record)
        print(
            f"finished E193 outer={outer} returncode={code} log={PATHS.relative(log_path)}",
            flush=True,
        )

    payload = {
        "experiment": VERSION,
        "status": "fold_failure" if failed else "folds_complete",
        "folds": results,
        "wall_seconds_before_aggregate": float(time.time() - started),
    }
    RUN_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failed:
        excerpts = []
        for record in failed:
            log_path = PATHS.root / str(record["log"])
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            excerpts.append(
                f"outer{record['outer']} tail:\n" + "\n".join(lines[-80:])
            )
        raise RuntimeError("\n\n".join(excerpts))

    aggregate_log = LOG_DIR / "aggregate.log"
    command = [
        str(python),
        "-u",
        str(script),
        "--stage",
        "aggregate",
        "--device",
        f"cuda:{devices[0]}",
        "--batch-size",
        str(args.batch_size),
    ]
    with aggregate_log.open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    payload["aggregate"] = {
        "returncode": completed.returncode,
        "log": PATHS.relative(aggregate_log),
    }
    payload["status"] = (
        "complete" if completed.returncode == 0 else "aggregate_failure"
    )
    payload["wall_seconds"] = float(time.time() - started)
    RUN_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        lines = aggregate_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        raise RuntimeError("aggregate tail:\n" + "\n".join(lines[-120:]))
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
