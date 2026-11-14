#!/usr/bin/env python3
"""Run the resumable E216 ablation DAG across available GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from project_paths import PATHS
import run_e216_v16_rt_information_ablation as experiment


SCRIPT = PATHS.root / "code" / "round2" / "run_e216_v16_rt_information_ablation.py"
ROOT = experiment.ROOT
LOG_ROOT = ROOT / "logs"
REPORT = experiment.REPORT_ROOT / "parallel_run.json"
VARIANTS = tuple(experiment.VARIANTS)
REBUILD_V9 = tuple(
    name for name, value in experiment.VARIANTS.items() if value.rebuild_v9
)
REBUILD_E40 = tuple(
    name for name, value in experiment.VARIANTS.items() if value.rebuild_e40
)
PATH_VARIANTS = tuple(
    name for name, value in experiment.VARIANTS.items() if value.path_models_changed
)


@dataclass(frozen=True)
class Task:
    name: str
    arguments: tuple[str, ...]
    gpu: int = 0


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def exit_path(task: Task) -> Path:
    return LOG_ROOT / f"{task.name}_exit.json"


def log_path(task: Task) -> Path:
    return LOG_ROOT / f"{task.name}.log"


def completed(task: Task) -> bool:
    path = exit_path(task)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("exit_code", -1)) == 0


def run_task(task: Task, force: bool) -> dict[str, object]:
    if completed(task) and not force:
        return {
            "name": task.name,
            "status": "reused",
            "exit_code": 0,
            "gpu": task.gpu,
        }
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT),
        *task.arguments,
        "--device",
        f"cuda:{task.gpu}",
    ]
    started = time.time()
    error = None
    return_code = -1
    try:
        with log_path(task).open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command,
                cwd=PATHS.root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )
        return_code = int(process.returncode)
    except BaseException:
        error = traceback.format_exc()
    payload = {
        "name": task.name,
        "status": "complete" if return_code == 0 else "failed",
        "exit_code": return_code,
        "gpu": task.gpu,
        "arguments": list(task.arguments),
        "log": PATHS.relative(log_path(task)),
        "seconds": float(time.time() - started),
        "runner_traceback": error,
    }
    atomic_json(exit_path(task), payload)
    return payload


def execute_phase(
    name: str,
    tasks: list[Task],
    workers: int,
    force: bool,
    history: list[dict[str, object]],
) -> None:
    started = time.time()
    results: list[dict[str, object]] = []
    if workers == 1:
        for current in tasks:
            result = run_task(current, force)
            results.append(result)
            print(
                f"E216 phase={name} task={result['name']} "
                f"status={result['status']} seconds={result.get('seconds', 0.0):.1f}",
                flush=True,
            )
            if int(result["exit_code"]) != 0:
                break
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            queued = list(tasks)
            active: dict[object, Task] = {}
            busy_gpus: set[int] = set()
            failed = False

            def fill_workers() -> None:
                while not failed and len(active) < workers:
                    index = next(
                        (
                            position
                            for position, candidate in enumerate(queued)
                            if candidate.gpu not in busy_gpus
                        ),
                        None,
                    )
                    if index is None:
                        break
                    current = queued.pop(index)
                    busy_gpus.add(current.gpu)
                    active[pool.submit(run_task, current, force)] = current

            fill_workers()
            while active:
                done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in done:
                    current = active.pop(future)
                    busy_gpus.remove(current.gpu)
                    result = future.result()
                    results.append(result)
                    print(
                        f"E216 phase={name} task={result['name']} "
                        f"status={result['status']} seconds={result.get('seconds', 0.0):.1f}",
                        flush=True,
                    )
                    failed = failed or int(result["exit_code"]) != 0
                fill_workers()
    row = {
        "phase": name,
        "seconds": float(time.time() - started),
        "workers": workers,
        "tasks": sorted(results, key=lambda value: str(value["name"])),
    }
    history.append(row)
    atomic_json(
        REPORT,
        {
            "experiment": experiment.VERSION,
            "status": "running",
            "phases": history,
        },
    )
    failures = [value for value in results if int(value["exit_code"]) != 0]
    if failures:
        names = ", ".join(str(value["name"]) for value in failures)
        raise RuntimeError(f"E216 phase {name} failed: {names}")


def task(
    name: str,
    variant: str | None,
    stage: str,
    gpu: int,
    *extra: str,
) -> Task:
    arguments = ["--stage", stage]
    if variant is not None:
        arguments.extend(["--variant", variant])
    arguments.extend(extra)
    return Task(name=name, arguments=tuple(arguments), gpu=gpu)


def phases(gpus: tuple[int, ...]) -> list[tuple[str, list[Task], int]]:
    index = 0

    def next_gpu() -> int:
        nonlocal index
        value = gpus[index % len(gpus)]
        index += 1
        return value

    output: list[tuple[str, list[Task], int]] = []
    output.append(
        (
            "identity",
            [task("identity_single_seed", None, "identity", next_gpu())],
            1,
        )
    )
    output.append(
        (
            "seed42_spectra",
            [
                task(
                    f"spectra_seed42_{short}",
                    experiment.PRIMARY_VARIANT,
                    "spectra",
                    next_gpu(),
                    "--config",
                    config,
                )
                for short, config in (
                    ("base", experiment.BASE_NAME),
                    ("scatter", experiment.SCATTER_NAME),
                )
            ],
            2,
        )
    )
    output.append(
        (
            "path_tokens",
            [
                task(f"tokens_{name}", name, "tokens", next_gpu())
                for name in PATH_VARIANTS
            ],
            1,
        )
    )
    output.append(
        (
            "path_features",
            [
                task(f"features_{name}", name, "features", next_gpu())
                for name in PATH_VARIANTS
            ],
            min(3, len(gpus)),
        )
    )
    output.append(
        (
            "audit",
            [task(f"audit_{name}", name, "audit", next_gpu()) for name in VARIANTS],
            min(len(VARIANTS), len(gpus)),
        )
    )
    output.append(
        (
            "initialize_oof",
            [
                task(f"init_oof_{name}", name, "init-oof", next_gpu())
                for name in REBUILD_V9
            ],
            min(4, len(gpus)),
        )
    )
    output.append(
        (
            "e40",
            [
                task(
                    f"e40_{name}_{domain}_fold{fold}",
                    name,
                    "e40",
                    next_gpu(),
                    "--domain",
                    domain,
                    "--fold",
                    str(fold),
                )
                for name in REBUILD_E40
                for domain in ("pas", "pdp")
                for fold in range(5)
            ],
            len(gpus),
        )
    )
    output.append(
        (
            "v9_reliability",
            [
                task(
                    f"v9_reliability_{name}_{domain}",
                    name,
                    "v9-reliability",
                    next_gpu(),
                    "--domain",
                    domain,
                )
                for name in REBUILD_V9
                for domain in ("pas", "pdp")
            ],
            2,
        )
    )
    output.append(
        (
            "v9",
            [
                task(
                    f"v9_{name}_{domain}_fold{fold}",
                    name,
                    "v9",
                    next_gpu(),
                    "--domain",
                    domain,
                    "--fold",
                    str(fold),
                )
                for name in REBUILD_V9
                for domain in ("pas", "pdp")
                for fold in range(5)
            ],
            len(gpus),
        )
    )
    for stage in ("link-build", "link-train", "canonical"):
        output.append(
            (
                stage,
                [task(f"{stage}_{name}", name, stage, next_gpu()) for name in VARIANTS],
                min(len(VARIANTS), len(gpus)),
            )
        )
    output.append(
        (
            "e85",
            [task(f"e85_{name}", name, "e85", next_gpu()) for name in PATH_VARIANTS],
            min(len(PATH_VARIANTS), len(gpus)),
        )
    )
    output.append(
        (
            "e86",
            [task(f"e86_{name}", name, "e86", next_gpu()) for name in VARIANTS],
            min(len(VARIANTS), len(gpus)),
        )
    )
    output.append(
        (
            "e99",
            [task(f"e99_{name}", name, "e99", next_gpu()) for name in PATH_VARIANTS],
            min(len(PATH_VARIANTS), len(gpus)),
        )
    )
    output.append(
        (
            "e190",
            [
                task(
                    f"e190_{name}_outer{outer}",
                    name,
                    "e190",
                    next_gpu(),
                    "--outer",
                    str(outer),
                )
                for name in VARIANTS
                for outer in range(5)
            ],
            len(gpus),
        )
    )
    output.append(
        (
            "aggregate",
            [task(f"aggregate_{name}", name, "aggregate", next_gpu()) for name in VARIANTS],
            min(len(VARIANTS), len(gpus)),
        )
    )
    output.append(
        (
            "finalize",
            [task(f"finalize_{name}", name, "finalize", next_gpu()) for name in VARIANTS],
            min(len(VARIANTS), len(gpus)),
        )
    )
    output.append(
        (
            "summary",
            [task("summary_single_seed", None, "summary", next_gpu())],
            1,
        )
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--from-phase")
    parser.add_argument("--through-phase")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("at least one GPU is required")
    selected = phases(gpus)
    names = [name for name, _, _ in selected]
    start = names.index(args.from_phase) if args.from_phase else 0
    stop = names.index(args.through_phase) + 1 if args.through_phase else len(selected)
    history: list[dict[str, object]] = []
    started = time.time()
    try:
        for name, tasks, workers in selected[start:stop]:
            execute_phase(name, tasks, workers, args.force, history)
    except BaseException:
        atomic_json(
            REPORT,
            {
                "experiment": experiment.VERSION,
                "status": "failed",
                "seconds": float(time.time() - started),
                "phases": history,
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_json(
        REPORT,
        {
            "experiment": experiment.VERSION,
            "status": "complete" if stop == len(selected) else "partial_complete",
            "seconds": float(time.time() - started),
            "phases": history,
        },
    )


if __name__ == "__main__":
    main()
