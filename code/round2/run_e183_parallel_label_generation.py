#!/usr/bin/env python3
"""Run E183 family pipelines with scratch staging and resource-aware scheduling."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from project_paths import PATHS


VERSION = "Round2-E183-resource-aware-strict-label-generation"
FAMILIES = (
    "S0H0",
    "S0H1",
    "S0H2",
    "S0H3",
    "S0H4",
    "S1H1",
    "S1H2",
    "S1H3",
    "S1H4",
)
STEP_NAMES = (
    "foundation",
    "pre_sionna",
    "rbf",
    "v3_folds",
    "v3_targets",
    "v3_projection",
    "v4_calibration",
    "v3_test",
    "v4",
    "v6_prepare",
    "v6_evaluate",
    "v6_test",
    "carrier_bank",
    "carrier_stack",
    "v7_projection",
    "v7_lifecycle",
    "e142",
    "e120_prepare",
    "e120_train",
    "e120_endpoint",
    "family_actions_11",
    "e160_clean_carriers",
    "e160_clean_core",
    "e161_e162_actions",
)
# Only serialize steps that stream tens of GiB. The lighter preprocessing stages
# are CPU-heavy and should overlap across families instead of idling GPUs.
IO_HEAVY_STEPS = frozenset({12, 14, 16, 20, 21, 23})
ESTIMATED_UNION = {
    "S0H0": 906,
    "S0H1": 395,
    "S0H2": 786,
    "S0H3": 493,
    "S0H4": 1124,
    "S1H1": 670,
    "S1H2": 393,
    "S1H3": 1087,
    "S1H4": 1052,
}
CANONICAL_WORKSPACE_DIR = (
    PATHS.artifacts / "round2" / "e183_strict_action_labels" / "workspaces"
)
RESULT_DIR = PATHS.analysis / "results" / "round2_e183_strict_action_labels"
REPORT = RESULT_DIR / "parallel_label_generation_runner.json"
LOG_DIR = RESULT_DIR / "parallel_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--max-memory-mib", type=int, default=12_000)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stop-step", type=int, default=len(STEP_NAMES) - 1)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "hw_e183_strict_labels",
    )
    parser.add_argument("--no-scratch", action="store_true")
    parser.add_argument("--ram-scratch-workers", type=int, default=1)
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--force-completed", action="store_true")
    return parser.parse_args()


def normalize_pci_bus_id(value: str) -> str:
    domain, bus, slot = value.strip().split(":")
    return f"{int(domain, 16):04x}:{bus.lower()}:{slot.lower()}"


def discover_idle_gpus(max_memory_mib: int, max_utilization: int) -> list[dict[str, object]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    records = []
    for line in output.splitlines():
        index, uuid, name, pci_bus, memory, utilization = [
            value.strip() for value in line.split(",")
        ]
        pci = normalize_pci_bus_id(pci_bus)
        numa_path = Path("/sys/bus/pci/devices") / pci / "numa_node"
        numa = int(numa_path.read_text(encoding="ascii").strip())
        record = {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "pci_bus_id": pci,
            "numa_node": numa,
            "memory_used_mib": int(memory),
            "utilization": int(utilization),
        }
        if record["memory_used_mib"] <= max_memory_mib and record["utilization"] <= max_utilization:
            records.append(record)
    records.sort(
        key=lambda row: (
            0 if "H100" in str(row["name"]) else 1,
            int(row["memory_used_mib"]),
        )
    )
    return records


def parse_cpu_list(specification: str) -> list[int]:
    values: list[int] = []
    for item in specification.strip().split(","):
        if "-" in item:
            start, end = (int(value) for value in item.split("-", maxsplit=1))
            values.extend(range(start, end + 1))
        elif item:
            values.append(int(item))
    return values


def physical_core_groups(numa_node: int) -> list[list[int]]:
    cpus = parse_cpu_list(
        (Path("/sys/devices/system/node") / f"node{numa_node}" / "cpulist")
        .read_text(encoding="ascii")
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for cpu in cpus:
        topology = Path("/sys/devices/system/cpu") / f"cpu{cpu}" / "topology"
        package = int((topology / "physical_package_id").read_text(encoding="ascii"))
        core = int((topology / "core_id").read_text(encoding="ascii"))
        groups.setdefault((package, core), []).append(cpu)
    return [sorted(groups[key]) for key in sorted(groups, key=lambda key: min(groups[key]))]


def assign_cpu_sets(gpus: list[dict[str, object]]) -> dict[str, list[int]]:
    by_node: dict[int, list[dict[str, object]]] = {}
    for gpu in gpus:
        by_node.setdefault(int(gpu["numa_node"]), []).append(gpu)
    assignments: dict[str, list[int]] = {}
    for node, node_gpus in by_node.items():
        cores = physical_core_groups(node)
        if len(cores) < len(node_gpus):
            raise RuntimeError(f"NUMA node {node} has too few physical cores")
        for index, gpu in enumerate(node_gpus):
            start = index * len(cores) // len(node_gpus)
            end = (index + 1) * len(cores) // len(node_gpus)
            assignments[str(gpu["uuid"])] = [
                cpu for group in cores[start:end] for cpu in group
            ]
    return assignments


def cpu_spec(cpus: list[int]) -> str:
    values = sorted(cpus)
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = value
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


def write_report(state: dict[str, object], lock: threading.Lock) -> None:
    with lock:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        temporary = REPORT.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(REPORT)


def run_command(
    command: list[str],
    env: dict[str, str],
    log,
    cpus: list[int] | None = None,
) -> float:
    started = time.time()
    full_command = command if cpus is None else ["taskset", "-c", cpu_spec(cpus), *command]
    subprocess.run(
        full_command,
        cwd=PATHS.root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return float(time.time() - started)


def completed_family(family: str) -> bool:
    path = RESULT_DIR / f"{family}_strict_action_labels.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "family_scored"


def completed_steps(workspace: Path) -> set[int]:
    path = workspace / "e183_pipeline_report.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["index"])
        for row in payload.get("steps", [])
        if int(row.get("return_code", 1)) == 0
    }


def base_environment(gpu: dict[str, object], cpus: list[int], workspace_base: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu["uuid"])
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["HW_SKIP_ENV_CHECK"] = "1"
    env["E183_WORKSPACE_DIR"] = str(workspace_base)
    env["HW_MEMMAP_FLUSH_ROWS"] = "32"
    physical_cores = max(1, len(cpus) // 2)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(physical_cores)
    return env


def update_family_state(
    state: dict[str, object],
    family: str,
    lock: threading.Lock,
    **updates: object,
) -> None:
    with lock:
        record = dict(state["families"].get(family, {}))
        record.update(updates)
        state["families"][family] = record
    write_report(state, lock)


def stage_family(
    family: str,
    workspace_base: Path,
    env: dict[str, str],
    cpus: list[int],
    io_semaphore: threading.Semaphore,
    stage_lock: threading.Lock,
    log,
) -> float:
    destination = workspace_base / family
    if (destination / "e183_workspace_state.json").is_file():
        return 0.0
    started = time.time()
    with stage_lock, io_semaphore:
        destination.mkdir(parents=True, exist_ok=True)
        source = CANONICAL_WORKSPACE_DIR / family
        if (source / "e183_workspace_state.json").is_file():
            run_command(
                ["rsync", "-a", "--update", f"{source}/", f"{destination}/"],
                env,
                log,
                cpus,
            )
        if not (destination / "e183_workspace_state.json").is_file():
            generator = PATHS.round2_code / "generate_e183_clean_action_labels.py"
            run_command(
                [sys.executable, "-u", str(generator), "--phase", "prepare", "--family", family],
                env,
                log,
                cpus,
            )
    return float(time.time() - started)


def sync_family(
    family: str,
    workspace_base: Path,
    env: dict[str, str],
    io_semaphore: threading.Semaphore,
    state: dict[str, object],
    lock: threading.Lock,
    keep_scratch: bool,
) -> dict[str, object]:
    source = workspace_base / family
    destination = CANONICAL_WORKSPACE_DIR / family
    log_path = LOG_DIR / f"{family}_sync.log"
    started = time.time()
    update_family_state(state, family, lock, sync_status="running")
    try:
        with io_semaphore, log_path.open("a", encoding="utf-8") as log:
            destination.mkdir(parents=True, exist_ok=True)
            run_command(
                ["rsync", "-a", "--update", f"{source}/", f"{destination}/"],
                env,
                log,
            )
        if not keep_scratch:
            shutil.rmtree(source)
        result = {"status": "complete", "seconds": float(time.time() - started)}
    except Exception as error:
        result = {
            "status": "failed",
            "seconds": float(time.time() - started),
            "error": str(error),
        }
    update_family_state(
        state,
        family,
        lock,
        sync_status=result["status"],
        sync_seconds=result["seconds"],
        sync_error=result.get("error"),
    )
    return result


def run_family(
    family: str,
    gpu: dict[str, object],
    cpus: list[int],
    workspace_base: Path,
    args: argparse.Namespace,
    io_semaphore: threading.Semaphore,
    stage_lock: threading.Lock,
    state: dict[str, object],
    lock: threading.Lock,
) -> dict[str, object]:
    started = time.time()
    env = base_environment(gpu, cpus, workspace_base)
    generator = PATHS.round2_code / "generate_e183_clean_action_labels.py"
    scorer = PATHS.round2_code / "score_e183_clean_action_labels.py"
    log_path = LOG_DIR / f"{family}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    update_family_state(
        state,
        family,
        lock,
        status="staging",
        scratch_tier=(
            "ram"
            if workspace_base.is_relative_to(Path("/dev/shm"))
            else ("canonical" if workspace_base == CANONICAL_WORKSPACE_DIR else "ssd")
        ),
        gpu=gpu,
        cpu_spec=cpu_spec(cpus),
        physical_core_count=max(1, len(cpus) // 2),
        started_at=started,
    )
    try:
        with log_path.open("a", encoding="utf-8") as log:
            stage_seconds = stage_family(
                family, workspace_base, env, cpus, io_semaphore, stage_lock, log
            )
            workspace = workspace_base / family
            done = completed_steps(workspace)
            update_family_state(
                state,
                family,
                lock,
                status="pipeline",
                stage_seconds=stage_seconds,
                completed_steps=sorted(done),
            )
            step_seconds: dict[str, float] = {}
            for step in range(args.start_step, args.stop_step + 1):
                if step in done:
                    continue
                name = STEP_NAMES[step]
                update_family_state(
                    state,
                    family,
                    lock,
                    current_step=step,
                    current_step_name=name,
                    current_step_started_at=time.time(),
                )
                command = [
                    sys.executable,
                    "-u",
                    str(generator),
                    "--phase",
                    "pipeline",
                    "--family",
                    family,
                    "--device",
                    "cuda:0",
                    "--physical-gpu",
                    str(gpu["uuid"]),
                    "--start-step",
                    str(step),
                    "--stop-step",
                    str(step),
                ]
                if step in IO_HEAVY_STEPS:
                    with io_semaphore:
                        elapsed = run_command(command, env, log, cpus)
                else:
                    elapsed = run_command(command, env, log, cpus)
                step_seconds[f"{step:02d}_{name}"] = elapsed
                done.add(step)
                update_family_state(
                    state,
                    family,
                    lock,
                    completed_steps=sorted(done),
                    step_seconds=dict(step_seconds),
                )

            update_family_state(state, family, lock, status="scoring", current_step=None)
            score_command = [
                sys.executable,
                "-u",
                str(scorer),
                "--family",
                family,
                "--device",
                "cuda:0",
                "--batch-size",
                str(args.score_batch_size),
            ]
            with io_semaphore:
                score_seconds = run_command(score_command, env, log, cpus)
        result = {
            "status": "complete",
            "seconds": float(time.time() - started),
            "score_seconds": score_seconds,
            "step_seconds": step_seconds,
        }
    except Exception as error:
        result = {
            "status": "failed",
            "seconds": float(time.time() - started),
            "error": str(error),
        }
    update_family_state(
        state,
        family,
        lock,
        status=result["status"],
        seconds=result["seconds"],
        score_seconds=result.get("score_seconds"),
        error=result.get("error"),
        finished_at=time.time(),
    )
    return result


def main() -> None:
    args = parse_args()
    requested = [value.strip() for value in args.families.split(",") if value.strip()]
    unknown = sorted(set(requested) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown families: {unknown}")
    families = [
        family
        for family in requested
        if args.force_completed or not completed_family(family)
    ]
    if not families:
        print("All requested E183 families are already scored.", flush=True)
        return

    gpus = discover_idle_gpus(args.max_memory_mib, args.max_utilization)
    worker_count = min(args.max_workers, len(gpus), len(families))
    if worker_count < 1:
        raise RuntimeError("no idle GPU available")
    gpus = gpus[:worker_count]
    cpu_sets = assign_cpu_sets(gpus)
    workspace_base = (
        CANONICAL_WORKSPACE_DIR
        if args.no_scratch
        else args.scratch_root.resolve() / "workspaces"
    )
    workspace_base.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(workspace_base)
    if not args.no_scratch and usage.free < 500 * 1024**3:
        raise RuntimeError(
            f"scratch filesystem has only {usage.free / 1024**3:.1f} GiB free"
        )
    ram_workspace_base: Path | None = None
    ram_workers = 0
    if not args.no_scratch and args.ram_scratch_workers > 0:
        candidate = Path("/dev/shm") / "hw_e183_strict_labels" / "workspaces"
        candidate.mkdir(parents=True, exist_ok=True)
        ram_usage = shutil.disk_usage(candidate)
        if ram_usage.free >= 180 * 1024**3:
            ram_workspace_base = candidate
            ram_workers = min(args.ram_scratch_workers, worker_count)

    jobs: queue.Queue[str] = queue.Queue()
    for family in sorted(families, key=lambda value: ESTIMATED_UNION[value], reverse=True):
        jobs.put(family)
    state: dict[str, object] = {
        "experiment": VERSION,
        "status": "running",
        "started_at": time.time(),
        "scratch_enabled": not args.no_scratch,
        "scratch_free_gib_at_start": float(usage.free / 1024**3),
        "ram_scratch_workers": ram_workers,
        "io_workers": args.io_workers,
        "gpus": gpus,
        "families": {
            family: {"family": family, "status": "queued"} for family in families
        },
    }
    lock = threading.Lock()
    io_semaphore = threading.Semaphore(max(1, args.io_workers))
    stage_lock = threading.Lock()
    write_report(state, lock)

    sync_results: dict[str, dict[str, object]] = {}
    sync_futures: dict[Future[dict[str, object]], str] = {}
    sync_future_lock = threading.Lock()
    sync_executor = ThreadPoolExecutor(max_workers=1) if not args.no_scratch else None

    def worker(
        worker_index: int, gpu: dict[str, object]
    ) -> list[tuple[str, dict[str, object]]]:
        output = []
        cpus = cpu_sets[str(gpu["uuid"])]
        first_job = True
        while True:
            try:
                family = jobs.get_nowait()
            except queue.Empty:
                return output
            active_workspace_base = (
                ram_workspace_base
                if first_job and worker_index < ram_workers and ram_workspace_base is not None
                else workspace_base
            )
            first_job = False
            env = base_environment(gpu, cpus, active_workspace_base)
            result = run_family(
                family,
                gpu,
                cpus,
                active_workspace_base,
                args,
                io_semaphore,
                stage_lock,
                state,
                lock,
            )
            output.append((family, result))
            if result["status"] == "complete" and sync_executor is not None:
                future = sync_executor.submit(
                    sync_family,
                    family,
                    active_workspace_base,
                    env,
                    io_semaphore,
                    state,
                    lock,
                    args.keep_scratch,
                )
                with sync_future_lock:
                    sync_futures[future] = family
            jobs.task_done()

    family_results: list[tuple[str, dict[str, object]]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(worker, index, gpu) for index, gpu in enumerate(gpus)
        ]
        for future in as_completed(futures):
            family_results.extend(future.result())

    if sync_executor is not None:
        sync_executor.shutdown(wait=False)
        for future in as_completed(list(sync_futures)):
            sync_results[sync_futures[future]] = future.result()
        sync_executor.shutdown(wait=True)

    success = all(result["status"] == "complete" for _, result in family_results)
    success = success and all(
        result["status"] == "complete" for result in sync_results.values()
    )
    state["status"] = "complete" if success else "partial_failed"
    state["seconds"] = float(time.time() - float(state["started_at"]))
    state["finished_at"] = time.time()
    write_report(state, lock)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
