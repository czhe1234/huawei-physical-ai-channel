"""Dispatch only the registered NeRF2 fold4 after NeWRF releases H1001."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C

R = C.REPORT / "freqminibatch_20261112"
TARGET = "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
MODULE = "round2.baseline_freqminibatch_20261112.nerf2"
NEW_RF_MODULE = "round2.baseline_freqminibatch_20261112.newrf"


def workers(module):
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            argv = (p / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            if module in argv:
                found.append({"pid": int(p.name), "argv": argv})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def partition():
    for item in workers(MODULE):
        argv = item["argv"]
        if "--mode" not in argv or argv[argv.index("--mode") + 1] != "run":
            continue
        if "--folds" not in argv:
            raise RuntimeError("Unregistered NeRF2 fold queue")
        values = set(argv[argv.index("--folds") + 1].split(","))
        if "4" in values:
            raise RuntimeError("Another NeRF2 queue contains reserved fold4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Waiter must have a CPU-only mask")
    spec_hash = C.sha256(args.spec)
    spec = json.loads(args.spec.read_text())
    if spec["gpu_uuid"] != TARGET or spec["module"] != MODULE or spec["folds"] != [4]:
        raise RuntimeError("Dispatch differs from registered assignment")
    folder = R / "dispatch"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "waiter.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + spec.get("wait_hours", 48) * 3600
        previous = None
        while True:
            partition()
            paths = [PATHS.root / value for value in spec["required_newrf_manifests"]]
            count = sum(p.is_file() for p in paths)
            active = workers(NEW_RF_MODULE)
            free = None
            if count == 5 and not active:
                free = int(subprocess.check_output(
                    ["nvidia-smi", "-i", TARGET, "--query-gpu=memory.free",
                     "--format=csv,noheader,nounits"], text=True).strip())
            state = {"ready_newrf_folds": count, "newrf_pids": [x["pid"] for x in active],
                     "free_mib": free, "reserved_nerf2_fold": 4}
            if state != previous:
                C.atomic_json(folder / "waiting.json", state)
                print(json.dumps(state), flush=True)
                previous = state
            if count == 5 and not active and free >= spec["minimum_free_mib"]:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("H1001 did not become available before dispatch deadline")
            time.sleep(30)
        if C.sha256(args.spec) != spec_hash:
            raise RuntimeError("Dispatch specification changed while waiting")
        for name, digest in spec["accepted_hashes"].items():
            if C.sha256(PATHS.root / name) != digest:
                raise RuntimeError(("Accepted source/config changed", name))
        partition()
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=TARGET, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONUNBUFFERED="1", HW_SKIP_ENV_CHECK="1",
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4")
        command = [str(PATHS.code_root / "tools/run_in_env.sh"), "python", "-u", "-m",
                   MODULE, *spec["arguments"]]
        path = folder / "fold4_run.log"
        with path.open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=PATHS.root, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        C.atomic_json(folder / "launch.json", {"pid": process.pid, "argv": command,
                     "gpu_uuid": TARGET, "spec_sha256": spec_hash, "availability": state})
        code = process.wait()
        C.atomic_json(folder / "exit.json", {"pid": process.pid, "exit_code": code})
        if code:
            raise RuntimeError(("Fold4 process failed", code))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        folder = R / "dispatch"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "failure.log").open("a") as log:
            log.write(traceback.format_exc() + "\n")
        raise
