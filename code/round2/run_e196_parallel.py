#!/usr/bin/env python3
"""Profile, run, resume, synthesize, and aggregate E196 across idle GPUs."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from project_paths import PATHS


VERSION = "Round2-E196-parallel-runner"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e196_mechanism_calibrator"
LOG_DIR = ANALYSIS_DIR / "logs"
RUN_REPORT = ANALYSIS_DIR / "parallel_run.json"
SCRIPT = PATHS.round2_code / "train_e196_mechanism_calibrator.py"
DOMAINS = ("pas", "pdp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--profile-device", default="3")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def command(
    stage: str,
    device: str,
    *,
    outer: int | None = None,
    domain: str | None = None,
    force: bool = False,
) -> list[str]:
    value = [
        str(Path(sys.executable)),
        "-u",
        str(SCRIPT),
        "--stage",
        stage,
        "--device",
        f"cuda:{device}",
    ]
    if outer is not None:
        value.extend(("--outer", str(outer)))
    if domain is not None:
        value.extend(("--domain", domain))
    if force:
        value.append("--force")
    return value


def run_sync(
    value: list[str], log_path: Path, name: str
) -> dict[str, Any]:
    started = time.time()
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            value,
            cwd=PATHS.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    result = {
        "name": name,
        "returncode": int(completed.returncode),
        "log": PATHS.relative(log_path),
        "seconds": float(time.time() - started),
    }
    if completed.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError(f"E196 {name} failed:\n" + "\n".join(lines[-200:]))
    return result


def run_domain_queue(
    devices: list[str], force: bool, payload: dict[str, Any]
) -> None:
    pending = deque((outer, domain) for outer in range(5) for domain in DOMAINS)
    active: dict[str, dict[str, Any]] = {}
    while pending or active:
        available = [device for device in devices if device not in active]
        while pending and available:
            device = available.pop(0)
            outer, domain = pending.popleft()
            log_path = LOG_DIR / f"outer{outer}_{domain}.log"
            handle = log_path.open("wb")
            process = subprocess.Popen(
                command(
                    "domain",
                    device,
                    outer=outer,
                    domain=domain,
                    force=force,
                ),
                cwd=PATHS.root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            active[device] = {
                "outer": outer,
                "domain": domain,
                "device": f"cuda:{device}",
                "pid": process.pid,
                "process": process,
                "handle": handle,
                "log_path": log_path,
                "started": time.time(),
            }
            print(
                f"started E196 outer={outer} domain={domain} "
                f"device=cuda:{device} pid={process.pid}",
                flush=True,
            )

        finished = []
        for device, job in active.items():
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            record = {
                "outer": job["outer"],
                "domain": job["domain"],
                "device": job["device"],
                "pid": job["pid"],
                "returncode": int(returncode),
                "log": PATHS.relative(job["log_path"]),
                "seconds": float(time.time() - job["started"]),
            }
            exit_path = LOG_DIR / f"outer{job['outer']}_{job['domain']}.exit.json"
            atomic_json(exit_path, {"experiment": VERSION, **record})
            payload["domains"].append(record)
            payload["domains"].sort(
                key=lambda row: (int(row["outer"]), str(row["domain"]))
            )
            atomic_json(RUN_REPORT, payload)
            print(
                f"finished E196 outer={job['outer']} domain={job['domain']} "
                f"returncode={returncode}",
                flush=True,
            )
            if returncode != 0:
                for other_device, other_job in active.items():
                    if other_device == device:
                        continue
                    other_process = other_job["process"]
                    if other_process.poll() is None:
                        other_process.terminate()
                for other_device, other_job in active.items():
                    if other_device == device:
                        continue
                    other_process = other_job["process"]
                    try:
                        other_process.wait(timeout=30.0)
                    except subprocess.TimeoutExpired:
                        other_process.kill()
                        other_process.wait()
                    other_job["handle"].close()
                lines = job["log_path"].read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                raise RuntimeError(
                    f"E196 outer{job['outer']} {job['domain']} failed:\n"
                    + "\n".join(lines[-200:])
                )
            finished.append(device)
        for device in finished:
            del active[device]
        if pending or active:
            time.sleep(1.0)


def main() -> None:
    args = parse_args()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload: dict[str, Any] = {
        "experiment": VERSION,
        "status": "running",
        "devices": [f"cuda:{device}" for device in devices],
        "profile": None,
        "domains": [],
        "folds": [],
        "aggregate": None,
        "started_unix": started,
    }
    atomic_json(RUN_REPORT, payload)
    try:
        if not args.skip_profile:
            payload["profile"] = run_sync(
                command("profile", args.profile_device),
                LOG_DIR / "profile.log",
                "profile",
            )
            atomic_json(RUN_REPORT, payload)

        run_domain_queue(devices, args.force, payload)

        fold_processes = []
        for outer in range(5):
            device = devices[outer % len(devices)]
            log_path = LOG_DIR / f"outer{outer}_synthesis.log"
            handle = log_path.open("wb")
            process = subprocess.Popen(
                command("fold", device, outer=outer, force=args.force),
                cwd=PATHS.root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            fold_processes.append(
                (outer, device, process, handle, log_path, time.time())
            )
            print(
                f"started E196 synthesis outer={outer} device=cuda:{device} "
                f"pid={process.pid}",
                flush=True,
            )

        failures = []
        for outer, device, process, handle, log_path, fold_started in fold_processes:
            returncode = process.wait()
            handle.close()
            record = {
                "outer": outer,
                "device": f"cuda:{device}",
                "pid": process.pid,
                "returncode": int(returncode),
                "log": PATHS.relative(log_path),
                "seconds": float(time.time() - fold_started),
            }
            atomic_json(
                LOG_DIR / f"outer{outer}_synthesis.exit.json",
                {"experiment": VERSION, **record},
            )
            payload["folds"].append(record)
            payload["folds"].sort(key=lambda row: int(row["outer"]))
            atomic_json(RUN_REPORT, payload)
            failures.extend([record] if returncode != 0 else [])
            print(
                f"finished E196 synthesis outer={outer} returncode={returncode}",
                flush=True,
            )
        if failures:
            excerpts = []
            for record in failures:
                lines = (PATHS.root / str(record["log"])).read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                excerpts.append(
                    f"outer{record['outer']}:\n" + "\n".join(lines[-200:])
                )
            raise RuntimeError("\n\n".join(excerpts))

        payload["aggregate"] = run_sync(
            command("aggregate", devices[0]),
            LOG_DIR / "aggregate.log",
            "aggregate",
        )
        payload["status"] = "complete"
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        payload["status"] = "failed"
        payload["traceback"] = traceback.format_exc()
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        raise


if __name__ == "__main__":
    main()
