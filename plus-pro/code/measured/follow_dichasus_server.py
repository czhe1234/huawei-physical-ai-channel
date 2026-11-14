"""Stream server progress over SSH for the existing local HTML dashboard.

The client receives only JSON metadata; the dataset worker is independent.
The --stream endpoint is read-only and opens no listening network port.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from dichasus_remote_transfer import atomic_json, confined
from publish_dichasus_progress import alive


def recent_events(root):
    path = root / "events.jsonl"
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 262144))
        lines = stream.read().splitlines()
    result = []
    for line in lines:
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if value.get("event") == "download_progress" and value.get("time", 0) > time.time() - 180:
            result.append(value)
    return result


def read_json_retry(path, attempts=5, delay=0.1):
    """Tolerate brief Windows sharing locks without hiding an unreadable file."""
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def staged_received_bytes(data_root, report_root, key, size):
    """Read verified ranges plus uncommitted receiving bytes without mutation."""
    journal = report_root / "download_ranges" / (key + ".json")
    try:
        ranges = read_json_retry(journal).get("ranges", {})
    except FileNotFoundError:
        ranges = {}
    received = sum(r["size"] for r in ranges.values())
    for path in confined(data_root, ".download/" + key).glob("range-*.partial"):
        index = path.name[len("range-"):-len(".partial")]
        if index not in ranges:
            try:
                received += path.stat().st_size
            except FileNotFoundError:
                pass
    return min(size, received)


def stream_server(root, data_root=None):
    while True:
        state = json.loads((root / "progress.json").read_text(encoding="utf-8"))
        pid = int(state.get("pid", 0))
        running = pid > 0 and alive(pid)
        if running:
            command = Path("/proc") / str(pid) / "cmdline"
            try:
                running = b"acquire_dichasus_server.py" in command.read_bytes()
            except OSError:
                running = False
        state["worker_alive"] = running
        if data_root is not None:
            for key, info in state.get("files", {}).items():
                if info.get("event") not in {"downloading", "download_progress"}:
                    continue
                info["received_bytes"] = staged_received_bytes(data_root, root, key, info["size"])
        print(json.dumps({"progress": state, "events": recent_events(root)}, ensure_ascii=False), flush=True)
        if state.get("status") in {"complete", "incomplete", "failed"} or not running:
            return
        time.sleep(1)


def follow(args):
    root = args.report_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(p) for p in ["python3", "-u", args.remote_script, "--stream",
                                               "--report-dir", args.remote_report_dir])
    if args.remote_data_root:
        command += " --data-root " + shlex.quote(args.remote_data_root)
    atomic_json(root / "server_observer_process.json", {"pid": os.getpid(), "started_at": time.time(),
                                                       "mode": "server_progress_only", "raw_downloads": False})
    samples = deque()
    while True:
        proc = None
        try:
            proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                     "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
                                     args.host, command], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    encoding="utf-8", errors="strict")
            last_log = 0
            for line in proc.stdout:
                payload = json.loads(line)
                state = payload["progress"]
                if state.get("execution_mode") != "server_direct":
                    raise ValueError("Server worker is not in direct-download mode")
                state["polled_at"] = time.time()
                received = sum(info.get("size", 0) if info.get("event") in {"complete", "server_download_verified"}
                               else info.get("received_bytes", info.get("verified_bytes", 0))
                               for info in state.get("files", {}).values())
                samples.append((state["polled_at"], received))
                while len(samples) > 2 and samples[1][0] < state["polled_at"] - 60:
                    samples.popleft()
                seconds = samples[-1][0] - samples[0][0]
                delta = samples[-1][1] - samples[0][1]
                state["observed_download_speed"] = delta / seconds if seconds >= 3 and delta >= 0 else None
                atomic_json(root / "server_recent_events.json", payload["events"])
                atomic_json(root / "progress.json", state)
                if time.time() - last_log > 30:
                    print(json.dumps({"status": "server_progress_received", "time": time.time(),
                                      "complete_bytes": state.get("complete_bytes")}), flush=True)
                    last_log = time.time()
                if state.get("status") in {"complete", "incomplete", "failed"}:
                    return
                if not state.get("worker_alive"):
                    break
            raise ConnectionError("SSH progress stream ended")
        except Exception as exc:
            print(json.dumps({"status": "server_progress_reconnecting", "time": time.time(),
                              "error_type": type(exc).__name__}), flush=True)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--remote-report-dir")
    parser.add_argument("--remote-script")
    parser.add_argument("--remote-data-root")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    if args.stream:
        stream_server(args.report_dir.resolve(), args.data_root.resolve() if args.data_root else None)
    else:
        if not all([args.host, args.remote_report_dir, args.remote_script]):
            parser.error("client mode requires --host, --remote-report-dir and --remote-script")
        follow(args)


if __name__ == "__main__":
    main()
