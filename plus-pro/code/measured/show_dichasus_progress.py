"""Read acquisition progress and render a self-refreshing local HTML file.

No listener, network connection, raw-data mutation or worker control is used.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

from publish_dichasus_progress import alive
from follow_dichasus_server import read_json_retry, staged_received_bytes


PHASES = {
    "preparing_server_file": "准备服务器文件", "server_download_verified": "服务器下载校验通过",
    "server_download_failed": "服务器下载或检查失败，正在重试",
    "preparing_local_file": "准备本地文件", "downloading": "下载中",
    "download_progress": "下载中", "local_verified": "本地校验通过，等待上传",
    "uploading": "上传中", "upload_progress": "上传中",
    "server_verifying": "服务器正在校验", "server_verified": "服务器校验通过",
    "local_deleted": "本地原件已删除", "complete": "完成",
    "download_or_inspect_failed": "下载或检查失败，正在重试",
    "upload_or_cleanup_failed": "上传或清理失败，正在重试",
}
FULLY_DOWNLOADED = {"server_download_verified", "local_verified", "uploading", "upload_progress", "server_verifying",
                    "server_verified", "local_deleted", "complete", "upload_or_cleanup_failed"}


def read_json(path):
    return read_json_retry(path)


def recent_rates(root, now):
    remote_events = root / "server_recent_events.json"
    current = read_json(root / "progress.json")
    if current.get("execution_mode") == "server_direct":
        if current.get("observed_download_speed") is not None:
            return {"download": current["observed_download_speed"], "upload": None}
        lines = [json.dumps(e) for e in read_json(remote_events)] if remote_events.exists() else []
        return rates_from_lines(lines, now)
    events = root / "events.jsonl"
    if not events.exists():
        return {"download": None, "upload": None}
    with events.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 262144))
        lines = stream.read().splitlines()
    return rates_from_lines(lines, now)


def rates_from_lines(lines, now):
    samples = {}
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if event.get("event") not in {"download_progress", "upload_progress"}:
            continue
        if event.get("time", 0) < now - 180:
            continue
        key = (event["event"].split("_")[0], event["key"])
        samples.setdefault(key, []).append(event)
    result = {"download": None, "upload": None}
    for (kind, _), values in samples.items():
        if len(values) < 2 or now - values[-1]["time"] > 90:
            continue
        newest = values[-1]
        older = next((v for v in reversed(values[:-1]) if newest["time"] - v["time"] >= 50), values[0])
        seconds = newest["time"] - older["time"]
        delta = newest["verified_bytes"] - older["verified_bytes"]
        if seconds > 0 and delta >= 0:
            result[kind] = (result[kind] or 0) + delta / seconds
    return result


def snapshot(root, manifest, data_root=None):
    state = read_json(root / "progress.json")
    now = time.time()
    server_direct = state.get("execution_mode") == "server_direct"
    worker_alive = bool(state.get("worker_alive")) if server_direct else alive(int(state.get("pid", 0)))
    progress = state.get("files", {})
    completed = 0
    local_bytes = 0
    deleted_bytes = 0
    active = []
    groups = {g["group"]: {"name": g["group"], "total_bytes": 0, "complete_bytes": 0,
                            "complete_files": 0, "total_files": 0, "phase": "等待处理"}
              for g in manifest["groups"]}
    for entry in manifest["files"]:
        info = progress.get(entry["key"], {})
        phase = info.get("event", "pending")
        size = entry["size"]
        staged = info.get("received_bytes", info.get("verified_bytes", 0)) if server_direct else info.get("verified_bytes", 0)
        if not server_direct and data_root is not None and phase in {"preparing_local_file", "downloading", "download_progress"}:
            staged = staged_received_bytes(data_root, root, entry["key"], size)
        received = size if phase in FULLY_DOWNLOADED else min(size, staged)
        local_bytes += received
        if phase == "complete":
            completed += size
        if info.get("local_deleted"):
            deleted_bytes += size
        if info and phase not in {"complete", "pending"}:
            transfer_bytes = min(size, staged)
            if phase in {"local_verified", "server_verifying", "server_verified", "local_deleted"}:
                transfer_bytes = size
            active.append({"name": entry["filename"], "path": entry["relative_path"], "phase": PHASES.get(phase, phase),
                           "phase_key": phase, "bytes": transfer_bytes, "size": size, "updated_at": info.get("time"),
                           "error": (info.get("traceback") or "").strip().splitlines()[-1:]})
        for name in entry["groups"]:
            group = groups[name]
            group["total_bytes"] += size
            group["total_files"] += 1
            if phase == "complete":
                group["complete_bytes"] += size
                group["complete_files"] += 1
            elif phase != "pending":
                group["phase"] = PHASES.get(phase, phase)
    for group in groups.values():
        if group["complete_files"] == group["total_files"]:
            group["phase"] = "完成"
        elif group["complete_files"] and group["phase"] == "等待处理":
            group["phase"] = "部分完成"
    cleanup = root / "legacy_local_cache_cleanup.json"
    if cleanup.exists():
        deleted_bytes += read_json(cleanup).get("deleted_bytes_this_run", 0)
    status = state.get("status", "unknown")
    if status == "running":
        if server_direct and now - state.get("polled_at", 0) > 15:
            status = "disconnected"
        elif not worker_alive:
            status = "stopped"
    return {"server_time": now, "updated_at": state.get("updated_at"), "status": status,
            "worker_alive": worker_alive, "complete_bytes": completed, "downloaded_bytes": local_bytes,
            "total_bytes": manifest["total_bytes"], "complete_files": state.get("complete_files", 0),
            "total_files": manifest["total_files"], "deleted_local_bytes": deleted_bytes,
            "active": sorted(active, key=lambda e: e.get("updated_at") or 0, reverse=True),
            "groups": list(groups.values()), "rates": recent_rates(root, now), "pid": state.get("pid"),
            "execution_mode": state.get("execution_mode", "local_then_upload"),
            "worker_host": state.get("worker_host"), "current_check": state.get("current_check")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = args.report_dir.resolve()
    manifest = read_json(root / "catalog_manifest.json")

    template = (Path(__file__).parent / "dichasus_progress_template.html").read_text(encoding="utf-8")
    samples = deque()
    last_data = None
    while True:
        read_failed = False
        try:
            data = snapshot(root, manifest, args.data_root.resolve() if args.data_root else None)
        except (OSError, ValueError) as exc:
            if not args.watch:
                raise
            samples.clear()
            read_failed = True
            print(json.dumps({"status": "progress_read_failed", "time": time.time(),
                              "error_type": type(exc).__name__, "retrying": True}), flush=True)
            if last_data is None:
                time.sleep(1)
                continue
            data = dict(last_data, server_time=time.time(), status="进度读取失败，正在重试",
                        rates={"download": None, "upload": None})
        if not read_failed and data["execution_mode"] != "server_direct" and args.data_root:
            samples.append((data["server_time"], data["downloaded_bytes"]))
            while len(samples) > 2 and samples[1][0] < data["server_time"] - 60:
                samples.popleft()
            seconds = samples[-1][0] - samples[0][0]
            delta = samples[-1][1] - samples[0][1]
            downloading = any(e["phase_key"] in {"preparing_local_file", "downloading", "download_progress"} for e in data["active"])
            if seconds >= 3:
                data["rates"]["download"] = delta / seconds if downloading and delta >= 0 else None
        if not read_failed:
            last_data = data
        payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
        output = template.replace("__DICHASUS_PROGRESS_JSON__", payload)
        partial = root / "dashboard.html.partial"
        partial.write_text(output, encoding="utf-8")
        for attempt in range(40):
            try:
                partial.replace(root / "dashboard.html")
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.1)
        print(json.dumps({"status": data["status"], "complete_bytes": data["complete_bytes"],
                          "current": [{"path": e["path"], "bytes": e["bytes"], "size": e["size"]} for e in data["active"]]}, ensure_ascii=False), flush=True)
        watching_remote = data["execution_mode"] == "server_direct" and data["status"] == "stopped"
        if not args.watch or (not read_failed and data["status"] not in {"running", "disconnected"} and not watching_remote):
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
