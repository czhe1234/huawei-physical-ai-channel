#!/usr/bin/env python3
"""Prepare, profile, run, resume, and aggregate five E194 folds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from project_paths import PATHS


VERSION = "Round2-E194-parallel-runner"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e194_shared_surface_complex_renderer"
)
LOG_DIR = ANALYSIS_DIR / "logs"
RUN_REPORT = ANALYSIS_DIR / "parallel_run.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def run_stage(script: Path, stage: str, device: str, log_path: Path, force: bool) -> None:
    command = [
        str(Path(sys.executable)),
        "-u",
        str(script),
        "--stage",
        stage,
        "--device",
        f"cuda:{device}",
    ]
    if force:
        command.append("--force")
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    if completed.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError(f"E194 {stage} failed:\n" + "\n".join(lines[-160:]))


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) < 5:
        raise ValueError("E194 requires five device entries")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    script = PATHS.round2_code / "train_e194_shared_surface_complex_renderer.py"
    started = time.time()
    payload: dict[str, object] = {
        "experiment": VERSION,
        "devices": devices[:5],
        "started_unix": started,
    }
    if not args.skip_prepare:
        run_stage(script, "prepare", devices[0], LOG_DIR / "prepare.log", args.force)
        payload["prepare"] = {"status": "complete", "log": PATHS.relative(LOG_DIR / "prepare.log")}
    if not args.skip_profile:
        run_stage(script, "profile", devices[0], LOG_DIR / "profile.log", False)
        payload["profile"] = {"status": "complete", "log": PATHS.relative(LOG_DIR / "profile.log")}

    processes: list[tuple[int, str, subprocess.Popen[bytes], object, Path]] = []
    for outer in range(5):
        device = devices[outer]
        log_path = LOG_DIR / f"outer{outer}.log"
        handle = log_path.open("wb")
        command = [
            str(Path(sys.executable)),
            "-u",
            str(script),
            "--stage",
            "fold",
            "--outer",
            str(outer),
            "--device",
            f"cuda:{device}",
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
        print(f"started E194 outer={outer} device=cuda:{device} pid={process.pid}", flush=True)

    fold_results = []
    failed = []
    for outer, device, process, handle, log_path in processes:
        returncode = process.wait()
        handle.close()
        record = {
            "outer": outer,
            "device": f"cuda:{device}",
            "pid": process.pid,
            "returncode": returncode,
            "log": PATHS.relative(log_path),
        }
        fold_results.append(record)
        if returncode != 0:
            failed.append(record)
        print(f"finished E194 outer={outer} returncode={returncode}", flush=True)
    payload["folds"] = fold_results
    payload["wall_seconds_before_aggregate"] = float(time.time() - started)
    if failed:
        payload["status"] = "fold_failure"
        RUN_REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        excerpts = []
        for record in failed:
            path = PATHS.root / str(record["log"])
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            excerpts.append(f"outer{record['outer']}:\n" + "\n".join(lines[-160:]))
        raise RuntimeError("\n\n".join(excerpts))

    aggregate_log = LOG_DIR / "aggregate.log"
    run_stage(script, "aggregate", devices[0], aggregate_log, False)
    payload["aggregate"] = {"status": "complete", "log": PATHS.relative(aggregate_log)}
    payload["status"] = "complete"
    payload["wall_seconds"] = float(time.time() - started)
    RUN_REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
