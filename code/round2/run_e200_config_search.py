#!/usr/bin/env python3
"""Profile and run the resumable multi-GPU E200 single-seed search."""

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

from e200_config_search import (
    SearchConfig,
    config_signature,
    geometry_configs,
    physics_configs,
    portable_payload,
)
from project_paths import PATHS


VERSION = "Round2-E200-config-search-runner"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e200_config_search"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e200_config_search"
MANIFEST_DIR = ARTIFACT_DIR / "manifests"
LOG_DIR = ANALYSIS_DIR / "logs"
SCREEN_IDS = ARTIFACT_DIR / "screen_ids_excluding_fixed500.npy"
RUN_REPORT = ANALYSIS_DIR / "parallel_run.json"
RANKING_REPORT = ANALYSIS_DIR / "ranking.json"
FINALISTS = MANIFEST_DIR / "finalists.json"
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--profile-per-site", type=int, default=32)
    parser.add_argument("--top-geometry", type=int, default=2)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--exclude-configs", default="")
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
            raise RuntimeError("E200 screen ID cache changed")
    else:
        np.save(SCREEN_IDS, ids)
    return ids


def write_manifest(name: str, configs: list[SearchConfig]) -> Path:
    path = MANIFEST_DIR / f"{name}.json"
    atomic_json(path, [config.to_dict() for config in configs])
    for config in configs:
        atomic_json(MANIFEST_DIR / "configs" / f"{safe_name(config.name)}.json", config.to_dict())
    return path


def report_path(config: SearchConfig) -> Path:
    return ANALYSIS_DIR / "configs" / f"{safe_name(config.name)}.json"


def command(config: SearchConfig, args: argparse.Namespace, limit_per_site: int = 0) -> list[str]:
    config_path = MANIFEST_DIR / "configs" / f"{safe_name(config.name)}.json"
    value = [
        str(PATHS.root / "code" / "tools" / "run_in_env.sh"),
        "python",
        "-u",
        str(WORKER),
        "--config-json",
        str(config_path),
        "--ids-file",
        str(SCREEN_IDS),
        "--batch-size",
        str(args.batch_size),
        "--cuda-device",
        "0",
    ]
    if limit_per_site:
        value.extend(("--limit-per-site", str(limit_per_site)))
    if args.force:
        value.append("--force")
    return value


def worker_environment(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu,
            "E200_PHYSICAL_GPU": gpu,
            "HW_CONDA_ENV": "sionna1",
            "HW_SKIP_ENV_CHECK": "1",
        }
    )
    return environment


def run_profile(config: SearchConfig, gpu: str, args: argparse.Namespace) -> dict[str, Any]:
    profile_config = replace(
        config,
        name=f"profile_n{args.profile_per_site}_{config.name}",
        family="profile",
    )
    atomic_json(
        MANIFEST_DIR / "configs" / f"{safe_name(profile_config.name)}.json",
        profile_config.to_dict(),
    )
    log_path = LOG_DIR / "profile.log"
    started = time.time()
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command(profile_config, args, args.profile_per_site),
            cwd=PATHS.root,
            env=worker_environment(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        "config": profile_config.name,
        "full_config": config.name,
        "gpu": gpu,
        "returncode": int(completed.returncode),
        "seconds": float(time.time() - started),
        "log": PATHS.relative(log_path),
    }
    if completed.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError("E200 profile failed:\n" + "\n".join(lines[-200:]))
    return result


def run_queue(
    stage: str,
    configs: list[SearchConfig],
    gpus: list[str],
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> None:
    pending = deque(
        config
        for config in configs
        if args.force
        or not report_path(config).is_file()
        or json.loads(report_path(config).read_text(encoding="utf-8")).get("status") != "complete"
    )
    for config in configs:
        if config not in pending:
            payload["jobs"].append(
                {"stage": stage, "config": config.name, "status": "reused", "returncode": 0}
            )
    active: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    while pending or active:
        available = [gpu for gpu in gpus if gpu not in active]
        while pending and available:
            gpu = available.pop(0)
            config = pending.popleft()
            log_path = LOG_DIR / stage / f"{safe_name(config.name)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("wb")
            process = subprocess.Popen(
                command(config, args),
                cwd=PATHS.root,
                env=worker_environment(gpu),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = {
                "stage": stage,
                "config": config,
                "process": process,
                "handle": handle,
                "log": log_path,
                "started": time.time(),
            }
            print(
                f"started E200 stage={stage} config={config.name} gpu={gpu} pid={process.pid}",
                flush=True,
            )
        finished = []
        for gpu, job in active.items():
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            record = {
                "stage": stage,
                "config": job["config"].name,
                "gpu": gpu,
                "pid": int(job["process"].pid),
                "returncode": int(returncode),
                "seconds": float(time.time() - job["started"]),
                "log": PATHS.relative(job["log"]),
                "status": "complete" if returncode == 0 else "failed",
            }
            atomic_json(
                LOG_DIR / stage / f"{safe_name(job['config'].name)}.exit.json",
                {"experiment": VERSION, **record},
            )
            payload["jobs"].append(record)
            payload["jobs"].sort(key=lambda row: (str(row["stage"]), str(row["config"])))
            atomic_json(RUN_REPORT, payload)
            if returncode != 0:
                failures.append(record)
            print(
                f"finished E200 stage={stage} config={job['config'].name} "
                f"gpu={gpu} returncode={returncode}",
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
            path = PATHS.root / str(record["log"])
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            excerpts.append(f"{record['config']}:\n" + "\n".join(lines[-120:]))
        raise RuntimeError("\n\n".join(excerpts))


def load_ranking(configs: list[SearchConfig]) -> list[dict[str, Any]]:
    rows = []
    by_name = {config.name: config for config in configs}
    for name, config in by_name.items():
        path = report_path(config)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(f"incomplete E200 config report: {path}")
        rows.append(
            {
                "name": name,
                "configuration": config.to_dict(),
                "metrics": payload["metrics"],
                "paths": payload["paths"],
                "angle_alignment": payload["angle_alignment"],
                "performance": payload["performance"],
            }
        )
    rows.sort(key=lambda row: float(row["metrics"]["selection"]), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def select_geometry(ranking: list[dict[str, Any]], count: int) -> list[SearchConfig]:
    selected: list[SearchConfig] = []
    signatures = set()
    for row in ranking:
        config = SearchConfig.from_dict(row["configuration"])
        signature = config_signature(config)
        if signature in signatures:
            continue
        selected.append(config)
        signatures.add(signature)
        if len(selected) >= count:
            break
    return selected


def select_finalists(ranking: list[dict[str, Any]], count: int) -> list[SearchConfig]:
    priorities = ("selection", "pas", "pdp", "ad", "difficult")
    selected: list[SearchConfig] = []
    names = set()
    for metric in priorities:
        for row in sorted(ranking, key=lambda value: float(value["metrics"][metric]), reverse=True):
            if row["name"] in names:
                continue
            selected.append(SearchConfig.from_dict(row["configuration"]))
            names.add(row["name"])
            break
        if len(selected) >= count:
            return selected[:count]
    for row in ranking:
        if row["name"] in names:
            continue
        selected.append(SearchConfig.from_dict(row["configuration"]))
        names.add(row["name"])
        if len(selected) >= count:
            break
    return selected


def main() -> None:
    args = parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    excluded = {value.strip() for value in args.exclude_configs.split(",") if value.strip()}
    if not gpus:
        raise ValueError("E200 requires at least one GPU")
    started = time.time()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ids = prepare_ids()
    all_geometry = geometry_configs()
    unknown_exclusions = excluded - {config.name for config in all_geometry}
    if unknown_exclusions:
        raise ValueError(f"unknown E200 exclusions: {sorted(unknown_exclusions)}")
    geometry = [config for config in all_geometry if config.name not in excluded]
    write_manifest("geometry", geometry)
    payload: dict[str, Any] = {
        "experiment": VERSION,
        "status": "running",
        "gpus": gpus,
        "screen_ids": PATHS.relative(SCREEN_IDS),
        "screen_count": int(len(ids)),
        "canonical_fixed500_excluded": True,
        "seed_count": 1,
        "excluded_configs": sorted(excluded),
        "profile": None,
        "jobs": [],
        "started_unix": started,
    }
    atomic_json(RUN_REPORT, payload)
    try:
        if not args.skip_profile:
            payload["profile"] = run_profile(geometry[0], gpus[0], args)
            atomic_json(RUN_REPORT, payload)
        run_queue("geometry", geometry, gpus, args, payload)
        geometry_ranking = load_ranking(geometry)
        parents = select_geometry(geometry_ranking, args.top_geometry)
        physics = physics_configs(parents)
        write_manifest("physics", physics)
        payload["selected_geometry_parents"] = [config.name for config in parents]
        atomic_json(RUN_REPORT, payload)
        run_queue("physics", physics, gpus, args, payload)
        all_configs = geometry + physics
        ranking = load_ranking(all_configs)
        finalists = select_finalists(ranking, args.finalists)
        write_manifest("finalists", finalists)
        atomic_json(
            RANKING_REPORT,
            {
                "experiment": VERSION,
                "status": "complete",
                "selection_rule": (
                    "0.70*(five-spatial-fold mean - 0.25*std) + 0.30*medium/heavy mean; "
                    "point score=0.45*PAS+0.40*PDP+0.15*angle-delay"
                ),
                "screen_count": int(len(ids)),
                "seed_count": 1,
                "ranking": ranking,
                "geometry_parents": [config.to_dict() for config in parents],
                "finalists": [config.to_dict() for config in finalists],
            },
        )
        payload["status"] = "complete"
        payload["finalists"] = [config.name for config in finalists]
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        print(json.dumps(portable_payload(payload), ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        payload["status"] = "failed"
        payload["traceback"] = traceback.format_exc()
        payload["wall_seconds"] = float(time.time() - started)
        atomic_json(RUN_REPORT, payload)
        raise


if __name__ == "__main__":
    main()
