"""Run the reserved NeRF2 fold after the assigned H100 becomes available."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C

ROOT = C.REPORT / "formal_20261111"
REPORT = ROOT / "nerf2/parallel_dispatch"
ARTIFACTS = C.ART / "formal_20261111"
TARGET_GPU = "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
PORTABLE = PATHS.code_root / "round2/baseline_formal_20261111/nerf2_portable.py"
NEW_RF_MODULE = "round2.baseline_formal_20261111.newrf"
BASE_MODULE = "round2.baseline_formal_20261111.nerf2"


def active_module_commands(module):
    commands = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            arguments = (entry / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            if module in arguments:
                commands.append({"PID": int(entry.name), "arguments": arguments})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return commands


def verify_partition():
    for process in active_module_commands(BASE_MODULE):
        arguments = process["arguments"]
        if "--mode" not in arguments:
            continue
        mode = arguments[arguments.index("--mode") + 1]
        if mode not in ("run", "prediction-only"):
            continue
        if "--folds" not in arguments:
            raise RuntimeError("Canonical worker has an unspecified fold queue")
        folds = {int(value) for value in arguments[arguments.index("--folds") + 1].split(",")}
        if 4 in folds:
            raise RuntimeError("The canonical queue still includes reserved fold4")


def wait_for_device(deadline):
    previous = None
    while time.monotonic() < deadline:
        verify_partition()
        manifests = [
            ARTIFACTS / "newrf" / f"outer{k}/prediction/manifest.json"
            for k in range(5)
        ]
        ready = all(path.is_file() for path in manifests)
        workers = active_module_commands(NEW_RF_MODULE) if ready else []
        free_mib = None
        if ready and not workers:
            value = subprocess.check_output(
                ["nvidia-smi", "-i", TARGET_GPU, "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], text=True)
            free_mib = int(value.strip())
        state = {
            "completed_newrf_folds": sum(path.is_file() for path in manifests),
            "newrf_worker_PIDs": [item["PID"] for item in workers],
            "free_GPU_MiB": free_mib,
            "reserved_nerf2_fold": 4,
        }
        if state != previous:
            C.atomic_json(REPORT / "fold4_waiting.json", state)
            print(json.dumps(state), flush=True)
            previous = state
        if ready and not workers and free_mib >= 52 * 1024:
            return state
        time.sleep(30)
    raise TimeoutError("The assigned H100 did not become available within the dispatch window")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--portable-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--wait-hours", type=float, default=24)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("The dispatcher must use a CPU-only CUDA mask")
    if C.sha256(PORTABLE) != args.portable_sha256:
        raise RuntimeError("Portable source hash differs from the accepted version")
    if C.sha256(args.runtime_config) != args.runtime_sha256:
        raise RuntimeError("Runtime configuration differs from the accepted version")
    runtime = json.loads(args.runtime_config.read_text())
    if runtime["folds"] != [4] or runtime["gpu_uuid"] != TARGET_GPU:
        raise RuntimeError("Dispatch is restricted to fold4 on the assigned H100")
    with (REPORT / "fold4_waiter.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        availability = wait_for_device(time.monotonic() + args.wait_hours * 3600)
        verify_partition()
        if C.sha256(PORTABLE) != args.portable_sha256 or C.sha256(args.runtime_config) != args.runtime_sha256:
            raise RuntimeError("Accepted source or runtime changed while waiting")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=TARGET_GPU, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="4", HW_SKIP_ENV_CHECK="1")
        command = [
            str(PATHS.code_root / "tools/run_in_env.sh"), "python", "-u", "-m",
            "round2.baseline_formal_20261111.nerf2_portable",
            "--runtime-config", str(args.runtime_config), "--mode", "run",
        ]
        log_path = ROOT / "nerf2/portable_4/formal_run.log"
        with log_path.open("ab", buffering=0) as output:
            process = subprocess.Popen(
                command, cwd=PATHS.root, env=env, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        C.atomic_json(REPORT / "portable_launch.json", {
            "PID": process.pid, "argv": command, "CUDA_VISIBLE_DEVICES": TARGET_GPU,
            "portable_source_sha256": args.portable_sha256,
            "runtime_sha256": args.runtime_sha256, "availability": availability,
            "log": PATHS.relative(log_path),
        })
        result = process.wait()
        C.atomic_json(REPORT / "portable_exit.json", {"PID": process.pid, "exit_code": result})
        if result:
            raise RuntimeError(f"Reserved fold4 worker exited with code {result}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        REPORT.mkdir(parents=True, exist_ok=True)
        with (REPORT / "dispatch_failure.log").open("a") as output:
            output.write(traceback.format_exc() + "\n")
        raise

