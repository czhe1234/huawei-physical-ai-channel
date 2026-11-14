#!/usr/bin/env python3
"""Profile, run, resume, and aggregate the five strict E195 folds."""

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


VERSION = "Round2-E195-parallel-runner"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e195_directional_probe_calibrator"
)
LOG_DIR = ANALYSIS_DIR / "logs"
RUN_REPORT = ANALYSIS_DIR / "parallel_run.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--force-field", action="store_true")
    parser.add_argument("--force-fold", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def command_for(
    script: Path,
    stage: str,
    device: str,
    *,
    outer: int | None = None,
    force_field: bool = False,
    force_fold: bool = False,
) -> list[str]:
    command = [
        str(Path(sys.executable)),
        "-u",
        str(script),
        "--stage",
        stage,
        "--device",
        f"cuda:{device}",
    ]
    if outer is not None:
        command.extend(("--outer", str(outer)))
    if force_field:
        command.append("--force-field")
    if force_fold:
        command.append("--force-fold")
    return command


def run_stage(
    command: list[str], log_path: Path, stage: str
) -> dict[str, Any]:
    started = time.time()
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command,
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
        raise RuntimeError(f"E195 {stage} failed:\n" + "\n".join(lines[-160:]))
    return record


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) < 5:
        raise ValueError("E195 requires five device entries")
    devices = devices[:5]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    script = PATHS.round2_code / "train_e195_directional_probe_calibrator.py"
    started = time.time()
    payload: dict[str, Any] = {
        "experiment": VERSION,
        "status": "running",
        "devices": [f"cuda:{device}" for device in devices],
        "started_unix": started,
        "folds": [],
    }
    atomic_json(RUN_REPORT, payload)

    if not args.skip_profile:
        profile_log = LOG_DIR / "canonical_profile.log"
        try:
            payload["profile"] = run_stage(
                command_for(
                    script,
                    "profile",
                    devices[0],
                    outer=0,
                    force_field=args.force_field,
                ),
                profile_log,
                "profile",
            )
        except Exception:
            payload["status"] = "profile_failure"
            payload["wall_seconds"] = float(time.time() - started)
            atomic_json(RUN_REPORT, payload)
            raise
        atomic_json(RUN_REPORT, payload)

    processes: dict[int, tuple[str, subprocess.Popen[bytes], Any, Path, float]] = {}
    for outer, device in enumerate(devices):
        log_path = LOG_DIR / f"outer{outer}_canonical.log"
        handle = log_path.open("wb")
        process = subprocess.Popen(
            command_for(
                script,
                "fold",
                device,
                outer=outer,
                force_field=args.force_field,
                force_fold=args.force_fold,
            ),
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        processes[outer] = (device, process, handle, log_path, time.time())
        print(
            f"started E195 outer={outer} device=cuda:{device} pid={process.pid}",
            flush=True,
        )

    failures: list[dict[str, Any]] = []
    while processes:
        completed_outer = []
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
            completed_outer.append(outer)
            print(
                f"finished E195 outer={outer} returncode={returncode}", flush=True
            )
        for outer in completed_outer:
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
            lines = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            excerpts.append(
                f"outer{record['outer']}:\n" + "\n".join(lines[-160:])
            )
        raise RuntimeError("\n\n".join(excerpts))

    aggregate_log = LOG_DIR / "canonical_aggregate.log"
    try:
        payload["aggregate"] = run_stage(
            command_for(script, "aggregate", devices[0]),
            aggregate_log,
            "aggregate",
        )
    except Exception:
        payload["status"] = "aggregate_failure"
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        raise
    payload["status"] = "complete"
    payload["wall_seconds"] = float(time.time() - started)
    atomic_json(RUN_REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
