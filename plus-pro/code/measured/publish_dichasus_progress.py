"""Publish the local acquisition progress atomically while its worker runs."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def alive(pid):
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-report-dir", required=True)
    args = parser.parse_args()
    root = args.report_dir.resolve()
    staged = root / "progress.publish.snapshot.json"
    remote_tmp = args.remote_report_dir.rstrip("/") + "/progress.incoming.json"
    remote_final = args.remote_report_dir.rstrip("/") + "/progress.json"
    while True:
        state = json.loads((root / "progress.json").read_text(encoding="utf-8"))
        staged.write_bytes((root / "progress.json").read_bytes())
        try:
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                            str(staged), args.host + ":" + remote_tmp], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            code = "import os,sys;os.replace(sys.argv[1],sys.argv[2])"
            command = "python3 -c " + shlex.quote(code) + " " + shlex.quote(remote_tmp) + " " + shlex.quote(remote_final)
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", args.host, command],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            print(json.dumps({"time": time.time(), "status": "progress_published",
                              "complete_files": state.get("complete_files"),
                              "complete_bytes": state.get("complete_bytes")}), flush=True)
        except Exception as exc:
            print(json.dumps({"time": time.time(), "status": "progress_publish_failed",
                              "error_type": type(exc).__name__}), flush=True)
        if state.get("status") in {"complete", "incomplete", "failed"} or not alive(int(state["pid"])):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
