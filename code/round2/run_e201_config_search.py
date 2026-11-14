#!/usr/bin/env python3
"""Profile and run an adaptive E201 full-Etoile configuration batch."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import numpy as np

from e201_config_search import (
    SearchConfig,
    batch1_configs,
    batch2_configs,
    batch3_configs,
    batch4_configs,
    batch5_configs,
    portable_payload,
)
from project_paths import PATHS


VERSION = "Round2-E201-adaptive-Etoile-search"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e201_config_search"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e201_config_search"
MANIFEST_DIR = ARTIFACT_DIR / "manifests"
LOG_DIR = ANALYSIS_DIR / "logs"
SCREEN_IDS = ARTIFACT_DIR / "screen_ids_excluding_fixed500.npy"
WORKER = PATHS.round2_code / "screen_e200_sionna_config.py"
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="4,5,0,1,2,3")
    parser.add_argument("--batch", type=int, choices=(1, 2, 3, 4, 5), default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--profile-per-site", type=int, default=16)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(portable_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_ids() -> np.ndarray:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    ids = np.setdiff1d(np.arange(4000, dtype=np.int64), query_ids, assume_unique=True)
    SCREEN_IDS.parent.mkdir(parents=True, exist_ok=True)
    if SCREEN_IDS.is_file():
        cached = np.load(SCREEN_IDS)
        if not np.array_equal(cached, ids):
            raise RuntimeError("E201 screen ID cache changed")
    else:
        np.save(SCREEN_IDS, ids)
    return ids


def config_path(config: SearchConfig) -> Path:
    return MANIFEST_DIR / "configs" / f"{safe_name(config.name)}.json"


def report_path(config: SearchConfig) -> Path:
    return ANALYSIS_DIR / "configs" / f"{safe_name(config.name)}.json"


def write_manifest(configs: list[SearchConfig], batch: int) -> None:
    atomic_json(
        MANIFEST_DIR / f"batch{batch}.json", [config.to_dict() for config in configs]
    )
    for config in configs:
        atomic_json(config_path(config), config.to_dict())


def command(config: SearchConfig, args: argparse.Namespace, limit: int = 0) -> list[str]:
    value = [
        str(PATHS.root / "code" / "tools" / "run_in_env.sh"),
        "python",
        "-u",
        str(WORKER),
        "--config-json",
        str(config_path(config)),
        "--ids-file",
        str(SCREEN_IDS),
        "--batch-size",
        str(args.batch_size),
        "--cuda-device",
        "0",
    ]
    if limit:
        value.extend(("--limit-per-site", str(limit)))
    if args.force:
        value.append("--force")
    return value


def environment(gpu: str) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu,
            "E201_PHYSICAL_GPU": gpu,
            "HW_CONDA_ENV": "sionna1",
            "HW_SKIP_ENV_CHECK": "1",
            "HW_E200_NAMESPACE": "e201",
        }
    )
    return value


def run_profile(config: SearchConfig, gpu: str, args: argparse.Namespace) -> dict[str, Any]:
    profile = replace(config, name=f"profile_n{args.profile_per_site}_{config.name}")
    atomic_json(config_path(profile), profile.to_dict())
    log = LOG_DIR / f"profile_batch{args.batch}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("wb") as handle:
        completed = subprocess.run(
            command(profile, args, args.profile_per_site),
            cwd=PATHS.root,
            env=environment(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        "config": profile.name,
        "gpu": gpu,
        "returncode": int(completed.returncode),
        "seconds": float(time.time() - started),
        "log": PATHS.relative(log),
    }
    atomic_json(LOG_DIR / f"profile_batch{args.batch}.exit.json", result)
    if completed.returncode != 0:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError("E201 profile failed:\n" + "\n".join(lines[-200:]))
    return result


def run_queue(
    configs: list[SearchConfig],
    gpus: list[str],
    args: argparse.Namespace,
    payload: dict[str, Any],
    run_report: Path,
) -> None:
    pending = deque(
        config
        for config in configs
        if args.force
        or not report_path(config).is_file()
        or json.loads(report_path(config).read_text(encoding="utf-8")).get("status")
        != "complete"
    )
    for config in configs:
        if config not in pending:
            payload["jobs"].append(
                {"config": config.name, "status": "reused", "returncode": 0}
            )
    active: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    while pending or active:
        for gpu in [value for value in gpus if value not in active]:
            if not pending:
                break
            config = pending.popleft()
            log = LOG_DIR / f"batch{args.batch}" / f"{safe_name(config.name)}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("wb")
            process = subprocess.Popen(
                command(config, args),
                cwd=PATHS.root,
                env=environment(gpu),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = {
                "config": config,
                "process": process,
                "handle": handle,
                "log": log,
                "started": time.time(),
            }
            print(f"started E201 config={config.name} gpu={gpu} pid={process.pid}", flush=True)
        finished = []
        for gpu, job in active.items():
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            record = {
                "config": job["config"].name,
                "gpu": gpu,
                "pid": int(job["process"].pid),
                "returncode": int(returncode),
                "seconds": float(time.time() - job["started"]),
                "log": PATHS.relative(job["log"]),
                "status": "complete" if returncode == 0 else "failed",
            }
            atomic_json(job["log"].with_suffix(".exit.json"), record)
            payload["jobs"].append(record)
            payload["jobs"].sort(key=lambda row: str(row["config"]))
            atomic_json(run_report, payload)
            if returncode != 0:
                failures.append(record)
            print(
                f"finished E201 config={job['config'].name} gpu={gpu} "
                f"returncode={returncode}",
                flush=True,
            )
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if pending or active:
            time.sleep(1.0)
    if failures:
        excerpts = []
        for record in failures:
            path = PATHS.root / record["log"]
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            excerpts.append(f"{record['config']}:\n" + "\n".join(lines[-160:]))
        raise RuntimeError("\n\n".join(excerpts))


def analyze(configs: list[SearchConfig], batch: int) -> dict[str, Any]:
    rows = []
    for config in configs:
        report = json.loads(report_path(config).read_text(encoding="utf-8"))
        rows.append(
            {
                "name": config.name,
                "configuration": config.to_dict(),
                "metrics": report["metrics"],
                "by_site": report["by_site"],
                "angle_alignment": report["angle_alignment"],
                "paths": report["paths"],
                "performance": report["performance"],
            }
        )
    rows.sort(key=lambda row: float(row["metrics"]["selection"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "experiment": VERSION,
        "status": f"batch{batch}_complete",
        "batch": batch,
        "decision_metric": (
            "raw PAS/angle-delay plus PDP, with no post-hoc angle shift; "
            "site/fold trends and residual shifts decide refinement"
        ),
        "configuration_count": len(configs),
        "ranking": rows,
    }


def main() -> None:
    args = parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("E201 requires at least one GPU")
    started = time.time()
    configs_by_batch = {
        1: batch1_configs,
        2: batch2_configs,
        3: batch3_configs,
        4: batch4_configs,
        5: batch5_configs,
    }
    configs = configs_by_batch[args.batch]()
    run_report = ANALYSIS_DIR / f"parallel_run_batch{args.batch}.json"
    batch_report = ANALYSIS_DIR / f"batch{args.batch}_analysis.json"
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ids = prepare_ids()
    write_manifest(configs, args.batch)
    payload: dict[str, Any] = {
        "experiment": VERSION,
        "status": "running",
        "gpus": gpus,
        "screen_ids": PATHS.relative(SCREEN_IDS),
        "screen_count": int(len(ids)),
        "canonical_fixed500_excluded": True,
        "seed_count": 1,
        "profile": None,
        "jobs": [],
        "started_unix": started,
    }
    atomic_json(run_report, payload)
    try:
        if not args.skip_profile:
            payload["profile"] = run_profile(configs[0], gpus[0], args)
            atomic_json(run_report, payload)
        run_queue(configs, gpus, args, payload, run_report)
        analysis = analyze(configs, args.batch)
        atomic_json(batch_report, analysis)
        payload["status"] = f"batch{args.batch}_complete"
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(run_report, payload)
        print(json.dumps(portable_payload(analysis), ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        payload["status"] = "failed"
        payload["traceback"] = traceback.format_exc()
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(run_report, payload)
        raise


if __name__ == "__main__":
    main()
