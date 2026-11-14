"""Resumable multi-GPU propagation support generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import ctypes.util
from dataclasses import asdict, dataclass
import multiprocessing as mp
import json
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np

from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .io_utils import atomic_json, portable_path


@dataclass(frozen=True)
class PropagationTask:
    split: str
    site: int
    configuration: str
    seed: int
    cuda_device: int
    data_dir: str
    scene_dir: str
    output_dir: str
    batch_size: int
    point_ids: tuple[int, ...]
    positions: tuple[tuple[float, float, float], ...]
    bs_position: tuple[float, float, float]
    force: bool = False


def shard_path(output_dir: Path, batch_index: int) -> Path:
    return output_dir / f"batch_{batch_index:04d}.npz"


def task_report_path(output_dir: Path) -> Path:
    return output_dir / "task_report.json"


def _task_report(task: PropagationTask, position_count: int) -> dict[str, Any]:
    payload = asdict(task)
    for key in ("data_dir", "scene_dir", "output_dir"):
        payload[key] = portable_path(payload[key])
    payload["positions"] = f"{position_count} rows"
    return payload


def _valid_shard(path: Path, expected_ids: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return np.array_equal(loaded["point_ids"], expected_ids)
    except (OSError, ValueError, KeyError):
        return False


def _ensure_optix_available() -> None:
    configured = os.environ.get("DRJIT_LIBOPTIX_PATH")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return
        raise RuntimeError(f"DRJIT_LIBOPTIX_PATH does not point to a file: {path}")

    candidates = ("libnvoptix.so.1", ctypes.util.find_library("nvoptix"))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            ctypes.CDLL(candidate)
            return
        except OSError:
            pass
    raise RuntimeError(
        "OptiX library libnvoptix.so.1 was not found. Make the NVIDIA driver "
        "library directory visible through LD_LIBRARY_PATH, or set "
        "DRJIT_LIBOPTIX_PATH to the full libnvoptix library path."
    )


def _run_task(task: PropagationTask) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "failure.json"
    failure_path.unlink(missing_ok=True)
    point_ids = np.asarray(task.point_ids, dtype=np.int64)
    positions = np.asarray(task.positions, dtype=np.float32)
    existing_report = task_report_path(output_dir)
    if not task.force:
        previous = None
        if existing_report.is_file():
            previous = json.loads(existing_report.read_text(encoding="utf-8"))
        complete = True
        cached_path_count = 0
        batch_count = 0
        for batch_index, start in enumerate(range(0, len(point_ids), task.batch_size)):
            end = min(start + task.batch_size, len(point_ids))
            batch_count += 1
            shard = shard_path(output_dir, batch_index)
            if not _valid_shard(shard, point_ids[start:end]):
                complete = False
                break
            if previous is None or previous.get("status") != "complete":
                with np.load(shard, allow_pickle=False) as loaded:
                    cached_path_count += int(len(loaded["tau"]))
        if complete:
            if previous is not None and previous.get("status") == "complete":
                previous["task"] = _task_report(task, len(positions))
                previous["task_cache_hit"] = True
                atomic_json(existing_report, previous)
                return previous
            payload = {
                "status": "complete",
                "task": _task_report(task, len(positions)),
                "completed_batches": 0,
                "reused_batches": batch_count,
                "path_count": cached_path_count,
                "timing_seconds": {
                    "solver": 0.0,
                    "support_extraction": 0.0,
                    "wall": float(time.time() - started),
                },
                "peak_rss_gib": float(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
                ),
                "pid": os.getpid(),
                "task_cache_hit": True,
            }
            atomic_json(existing_report, payload)
            return payload
    completed = 0
    reused = 0
    path_count = 0
    solver_seconds = 0.0
    extraction_seconds = 0.0
    try:
        _ensure_optix_available()
        from round2.physical_prior.source_port import PathSupportSession
        from round2.physical_prior.source_port_common import save_support

        session = PathSupportSession(
            site=task.site,
            bs_position=np.asarray(task.bs_position, dtype=np.float32),
            config=task.configuration,
            seed=task.seed,
            cuda_device=task.cuda_device,
            data_dir=Path(task.data_dir),
            scene_dir=Path(task.scene_dir),
        )
        try:
            for batch_index, start in enumerate(
                range(0, len(point_ids), task.batch_size)
            ):
                end = min(start + task.batch_size, len(point_ids))
                local_ids = point_ids[start:end]
                output = shard_path(output_dir, batch_index)
                if not task.force and _valid_shard(output, local_ids):
                    reused += 1
                    with np.load(output, allow_pickle=False) as loaded:
                        path_count += int(len(loaded["tau"]))
                    continue
                support, report = session.solve(
                    positions=positions[start:end],
                    point_ids=local_ids,
                    split=task.split,
                )
                save_support(output, support, report)
                path_count += int(len(support["tau"]))
                solver_seconds += float(report["timings"]["path_solver_seconds"])
                extraction_seconds += float(
                    report["timings"]["support_extraction_seconds"]
                )
                completed += 1
        finally:
            session.close()
        report = {
            "status": "complete",
            "task": _task_report(task, len(positions)),
            "completed_batches": completed,
            "reused_batches": reused,
            "path_count": path_count,
            "timing_seconds": {
                "solver": solver_seconds,
                "support_extraction": extraction_seconds,
                "wall": float(time.time() - started),
            },
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "pid": os.getpid(),
        }
        atomic_json(task_report_path(output_dir), report)
        return report
    except BaseException as error:
        report = {
            "status": "failed",
            "task": _task_report(task, len(positions)),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "completed_batches": completed,
            "reused_batches": reused,
            "wall_seconds": float(time.time() - started),
            "pid": os.getpid(),
        }
        atomic_json(failure_path, report)
        return report


def _run_gpu_queue(tasks: tuple[PropagationTask, ...]) -> list[dict[str, Any]]:
    return [_run_task(task) for task in tasks]


def build_tasks(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    gpu_ids: tuple[int, ...],
    force: bool = False,
) -> list[PropagationTask]:
    if split not in ("train", "test"):
        raise ValueError(f"unsupported split: {split}")
    if not gpu_ids:
        raise ValueError("at least one CUDA device is required")
    data = load_official_data(paths.data_dir)
    positions = data.train_positions if split == "train" else data.test_positions
    sites = data.train_sites if split == "train" else data.test_sites
    tasks: list[PropagationTask] = []
    task_index = 0
    for configuration in ("base", "scatter"):
        for seed in config.seeds:
            for site in range(len(data.setup.bs_positions)):
                point_ids = np.flatnonzero(sites == site).astype(np.int64)
                output_dir = (
                    paths.propagation_dir
                    / split
                    / configuration
                    / f"seed_{seed:04d}"
                    / f"site_{site}"
                )
                tasks.append(
                    PropagationTask(
                        split=split,
                        site=site,
                        configuration=configuration,
                        seed=int(seed),
                        cuda_device=int(gpu_ids[task_index % len(gpu_ids)]),
                        data_dir=str(paths.data_dir),
                        scene_dir=str(
                            paths.scene_dir
                            / f"gpu_{gpu_ids[task_index % len(gpu_ids)]}"
                        ),
                        output_dir=str(output_dir),
                        batch_size=config.receiver_batch_size,
                        point_ids=tuple(map(int, point_ids)),
                        positions=tuple(
                            tuple(map(float, row)) for row in positions[point_ids]
                        ),
                        bs_position=tuple(map(float, data.setup.bs_positions[site])),
                        force=force,
                    )
                )
                task_index += 1
    return tasks


def run_propagation(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    gpu_ids: tuple[int, ...],
    force: bool = False,
    max_tasks: int | None = None,
    workers_per_gpu: int = 2,
) -> dict[str, Any]:
    started = time.time()
    tasks = build_tasks(paths, config, split, gpu_ids, force=force)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    queues: list[tuple[PropagationTask, ...]] = []
    for gpu in gpu_ids:
        local = tuple(task for task in tasks if task.cuda_device == gpu)
        lanes = min(workers_per_gpu, len(local))
        queues.extend(tuple(local[lane::lanes]) for lane in range(lanes))
    queues = [queue for queue in queues if queue]
    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(queues), mp_context=context) as pool:
        futures = [pool.submit(_run_gpu_queue, queue) for queue in queues]
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [result for result in results if result["status"] != "complete"]
    report = {
        "stage": "propagation",
        "status": "failed" if failures else "complete",
        "split": split,
        "task_count": len(tasks),
        "completed_task_count": len(results) - len(failures),
        "failed_task_count": len(failures),
        "gpu_ids": list(gpu_ids),
        "seed_count": config.seed_count,
        "receiver_batch_size": config.receiver_batch_size,
        "workers_per_gpu": workers_per_gpu,
        "path_count": int(sum(result.get("path_count", 0) for result in results)),
        "wall_seconds": float(time.time() - started),
        "measured_bottleneck": "physical path solving",
        "failures": [
            {
                "task": result["task"],
                "error": result.get("error"),
                "traceback": result.get("traceback"),
            }
            for result in failures
        ],
    }
    atomic_json(paths.report_dir / f"propagation_{split}.json", report)
    if failures:
        raise RuntimeError(
            f"{len(failures)} propagation tasks failed; see {paths.report_dir}"
        )
    return report
