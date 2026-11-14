"""Download locally, upload through SSH, verify, then remove local originals.

All roots are explicit CLI arguments. Published identities and frozen URLs come
from catalog_dichasus.py. HTTP ranges and SSH pieces resume independently.
No remote HTTP download occurs. Logs never store signed storage redirects.
HTTP transfers bypass proxy settings and curl configuration, using OS routing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
import traceback

from dichasus_remote_transfer import atomic_json, confined, hashes, validate

MIB = 1024 ** 2


class WindowsInterface:
    """Resolve one named interface without changing routes; share a short cache."""

    def __init__(self, alias):
        if not re.fullmatch(r"[\w. ()-]{1,256}", alias):
            raise ValueError("Invalid download interface alias")
        if os.name != "nt":
            raise ValueError("Download interface aliases require Windows")
        self.alias = alias
        self.lock = threading.Lock()
        self.address = None
        self.error = None
        self.expires = 0
        self.last_forced = float("-inf")

    def resolve(self, failed_address=None):
        with self.lock:
            now = time.monotonic()
            # Eight range workers can fail against the same old address at once.
            # Only the first forces a fresh query; the others reuse its result.
            force = (failed_address is not None and failed_address == self.address
                     and now - self.last_forced >= 2)
            if force:
                self.last_forced = now
            if force or now >= self.expires:
                script = ("$ErrorActionPreference='Stop'; "
                          "$items=@(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { "
                          "$_.InterfaceAlias -eq $env:DICHASUS_INTERFACE_ALIAS -and "
                          "$_.AddressState -eq 'Preferred' }); "
                          "ConvertTo-Json -Compress -InputObject @($items.IPAddress)")
                try:
                    result = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                        env={**os.environ, "DICHASUS_INTERFACE_ALIAS": self.alias},
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if result.returncode:
                        raise ValueError("interface lookup failed")
                    addresses = json.loads(result.stdout.decode("utf-8-sig"))
                    usable = []
                    for value in addresses:
                        address = ipaddress.ip_address(value)
                        if (address.version == 4 and not address.is_link_local
                                and not address.is_loopback and not address.is_unspecified
                                and not address.is_multicast):
                            usable.append(str(address))
                    if len(set(usable)) != 1:
                        raise ValueError("expected exactly one Preferred IPv4 address")
                    self.address, self.error = usable[0], None
                except Exception:
                    self.address = None
                    self.error = "No usable Preferred IPv4 for download interface alias: " + self.alias
                self.expires = time.monotonic() + 2
            if self.error:
                raise RuntimeError(self.error)
            return self.address


def allocate_partial(path, size):
    """Use sparse staging on NTFS; every range must still pass its checksum."""
    with path.open("wb") as stream:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            control = kernel.DeviceIoControl
            control.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
            control.restype = wintypes.BOOL
            returned = wintypes.DWORD()
            # FSCTL_SET_SPARSE only affects newly created temporary files.
            handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
            control(handle, 0x000900C4,
                    None, 0, None, 0, ctypes.byref(returned), None)
            pointer = kernel.SetFilePointerEx
            pointer.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
            pointer.restype = wintypes.BOOL
            end_file = kernel.SetEndOfFile
            end_file.argtypes = [wintypes.HANDLE]
            end_file.restype = wintypes.BOOL
            # Python's Windows truncate uses a CRT path that can write zeros
            # across the entire extension even for a sparse file.
            if not pointer(handle, size, None, 0) or not end_file(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            return
        stream.truncate(size)


class Acquisition:
    execution_mode = "local_then_upload"
    staging_event = "preparing_local_file"
    verification_event = "local_verified"
    verification_directory = "local_verification"
    curl_retry_options = ("--retry", "4", "--retry-all-errors")
    curl_extra_options = ()
    progress_interval = 30

    def __init__(self, args):
        self.args = args
        # Bind only this downloader; do not change system routes or DNS.
        interface = getattr(args, "download_interface", "")
        interface_alias = getattr(args, "download_interface_alias", "")
        if interface and interface_alias:
            raise ValueError("Specify a download interface IP or alias, not both")
        self.dynamic_interface = WindowsInterface(interface_alias) if interface_alias else None
        resolve = getattr(args, "curl_resolve", None) or []
        network_options = []
        if interface:
            ipaddress.ip_address(interface)
            network_options += ["--interface", interface]
        for specification in resolve:
            host, port, addresses = specification.split(":", 2)
            if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or not 1 <= int(port) <= 65535:
                raise ValueError("Invalid download DNS override")
            for address in addresses.split(","):
                ipaddress.ip_address(address)
            network_options += ["--resolve", specification]
        self.curl_extra_options = (*self.curl_extra_options, *network_options)
        self.data = args.data_root.resolve()
        self.report = args.report_dir.resolve()
        self.data.mkdir(parents=True, exist_ok=True)
        self.report.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        self.items = self.manifest["files"]
        if args.only_keys:
            keys = set(args.only_keys.split(","))
            self.items = [e for e in self.items if e["key"] in keys]
            if len(self.items) != len(keys):
                raise ValueError("Unknown requested file key")
        if args.limit:
            self.items = self.items[:args.limit]
        self.state_path = self.report / "progress.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {"files": {}}
        self.state.update({"status": "running", "started_at": time.time(), "pid": os.getpid(),
                           "scope_files": len(self.items), "scope_bytes": sum(e["size"] for e in self.items),
                           "last_error": None, "download_connection": "direct",
                           "proxy_bypass": "*", "curl_config": "disabled",
                           "execution_mode": self.execution_mode})
        if network_options or interface_alias:
            self.state.update({"download_interface": interface, "download_resolve": resolve,
                               "download_interface_alias": interface_alias,
                               "download_tls_verification": "enabled; original HTTPS hostnames preserved"})
        else:
            for field in ("download_interface", "download_interface_alias", "download_resolve", "download_tls_verification"):
                self.state.pop(field, None)
        if self.execution_mode == "local_then_upload":
            for field in ("polled_at", "observed_download_speed", "tls_ca_bundle_sha256"):
                self.state.pop(field, None)
            self.state.update({"worker_host": socket.gethostname(), "worker_alive": True,
                               "download_stage": "local", "local_dataset_download_enabled": True,
                               "current_check": None})
            # Resume missing data first; previously verified originals are
            # inspected again before the final full-manifest completion check.
            self.items = sorted(self.items, key=lambda e:
                                self.state["files"].get(e["key"], {}).get("event") == "complete")
        self.lock = threading.Lock()
        self.events = self.report / "events.jsonl"
        self.failures = []

    def event(self, event, entry=None, **extra):
        value = {"time": time.time(), "event": event, **extra}
        if entry:
            value.update({"key": entry["key"], "relative_path": entry["relative_path"], "size": entry["size"]})
        with self.lock:
            with self.events.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            if entry:
                self.state["files"].setdefault(entry["key"], {}).update(value)
            self.state["updated_at"] = time.time()
            complete = [self.state["files"].get(e["key"], {}) for e in self.items]
            self.state["complete_files"] = sum(e.get("event") == "complete" for e in complete)
            self.state["complete_bytes"] = sum(e.get("size", 0) for e in complete if e.get("event") == "complete")
            atomic_json(self.state_path, self.state)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def command(self, action, entry=None, index=None):
        parts = ["python3", self.args.remote_helper, action, "--root", self.args.remote_root]
        if entry:
            parts += ["--key", entry["key"]]
        if index is not None:
            parts += ["--index", str(index)]
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                self.args.host, " ".join(shlex.quote(x) for x in parts)]

    def rpc(self, action, entry, payload=None):
        last = None
        for attempt in range(4):
            try:
                result = subprocess.run(self.command(action, entry),
                                        input=json.dumps(payload if payload is not None else entry).encode(),
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=7200)
                if result.returncode:
                    raise RuntimeError("SSH receiver: " + result.stderr.decode("utf-8", "replace")[-3000:])
                return json.loads(result.stdout)
            except Exception as exc:
                last = exc
                if attempt < 3:
                    time.sleep(min(2 ** attempt * 3, 20))
        raise last

    def download(self, entry):
        path = confined(self.data, entry["relative_path"])
        if path.exists():
            identity = hashes(path)
            validate(entry, identity)
            return path, identity
        if shutil.disk_usage(self.data).free < entry["size"] + 10 * 1024 ** 3:
            raise RuntimeError("Local space reserve would be exceeded; retaining checkpoints")
        work = confined(self.data, ".download/" + entry["key"])
        work.mkdir(parents=True, exist_ok=True)
        partial = work / "original.partial"
        journal = self.report / "download_ranges" / (entry["key"] + ".json")
        state = json.loads(journal.read_text(encoding="utf-8")) if journal.exists() else {"ranges": {}}
        if state.get("source", entry["source"]) != entry["source"]:
            raise ValueError("Download source identity changed")
        state.update({"source": entry["source"], "size": entry["size"], "chunk_size": self.args.download_chunk_mib * MIB})
        chunk_size = state["chunk_size"]
        if not partial.exists():
            state["ranges"] = {}
            self.event(self.staging_event, entry)
            allocate_partial(partial, entry["size"])
        elif partial.stat().st_size != entry["size"]:
            if partial.stat().st_size < entry["size"] and not state["ranges"]:
                self.event(self.staging_event, entry)
                allocate_partial(partial, entry["size"])
            else:
                raise ValueError("Partial file length changed")
        count = (entry["size"] + chunk_size - 1) // chunk_size
        progress = {"verified_ranges": 0, "verified_bytes": 0}

        def worker(index):
            start = index * chunk_size
            end = min(entry["size"], start + chunk_size) - 1
            n = end - start + 1
            old = state["ranges"].get(str(index))
            if old and old.get("start") == start and old.get("size") == n:
                with partial.open("rb") as stream:
                    stream.seek(start)
                    if hashlib.sha256(stream.read(n)).hexdigest() == old["sha256"]:
                        return index, old
            temporary = work / ("range-" + str(index) + ".partial")
            # -q must be first to disable curlrc; '*' also overrides proxy env vars.
            retry_options = list(self.curl_retry_options)
            if self.dynamic_interface:
                # curl must return binding error45 immediately, so a changed
                # PPPoE address is resolved before retrying the same range.
                retry_options = [x for x in retry_options if x != "--retry-all-errors"]
            cmd = ["curl.exe" if os.name == "nt" else "curl", "-q", "--noproxy", "*", "--fail", "--location",
                   "--silent", "--show-error"] + retry_options + list(self.curl_extra_options) + [
                   "--connect-timeout", "25", "--max-time", "1200", "--speed-limit", "1024",
                   "--speed-time", "90", "--header", "Accept-Encoding: identity",
                   "--range", str(start) + "-" + str(end), "--max-filesize", str(n), "--dump-header", "-",
                   "--output", str(temporary), entry["source"]]
            address = self.dynamic_interface.resolve() if self.dynamic_interface else None
            bound_cmd = cmd + ["--interface", address] if address else cmd
            result = subprocess.run(bound_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=6400)
            if result.returncode == 45 and self.dynamic_interface:
                address = self.dynamic_interface.resolve(failed_address=address)
                result = subprocess.run(cmd + ["--interface", address], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=6400)
            # Redirect Location headers are deliberately never written to logs.
            if result.returncode:
                raise RuntimeError("HTTP range failed: " + entry["key"] + " range " + str(index) + " curl=" + str(result.returncode))
            statuses = re.findall(rb"(?im)^HTTP/\S+\s+(\d+)", result.stdout)
            ranges = re.findall(rb"(?im)^content-range:\s*([^\r\n]+)", result.stdout)
            exact = ("bytes %d-%d/%d" % (start, end, entry["size"])).encode()
            if not statuses or not ((statuses[-1] == b"206" and ranges and ranges[-1].strip() == exact)
                                     or (statuses[-1] == b"200" and start == 0 and n == entry["size"])):
                raise ValueError("Server ignored or changed the requested range")
            if temporary.stat().st_size != n:
                raise ValueError("HTTP range truncated")
            digest = hashlib.sha256()
            with temporary.open("rb") as source, partial.open("r+b") as dest:
                dest.seek(start)
                for block in iter(lambda: source.read(4 * MIB), b""):
                    digest.update(block)
                    dest.write(block)
                dest.flush()
                os.fsync(dest.fileno())
            temporary.unlink()
            return index, {"start": start, "size": n, "sha256": digest.hexdigest()}

        self.event("downloading", entry, ranges=count)
        started = time.time()
        last_print = 0
        errors = []
        with ThreadPoolExecutor(max_workers=self.args.download_workers) as pool:
            pending = [pool.submit(worker, i) for i in range(count)]
            for future in as_completed(pending):
                try:
                    index, info = future.result()
                except Exception as exc:
                    errors.append(exc)
                    continue
                state["ranges"][str(index)] = info
                atomic_json(journal, state)
                progress["verified_ranges"] += 1
                progress["verified_bytes"] += info["size"]
                if time.time() - last_print > self.progress_interval or progress["verified_ranges"] == count:
                    self.event("download_progress", entry, **progress, elapsed_s=time.time() - started)
                    last_print = time.time()
        if errors:
            raise errors[0]
        identity = hashes(partial)
        validate(entry, identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Original appeared during download; refusing overwrite")
        self.publish_download(partial, path)
        atomic_json(self.report / self.verification_directory / (entry["key"] + ".json"),
                    {"source": entry["source"], "verified_at": time.time(), **identity})
        self.event(self.verification_event, entry, **identity)
        return path, identity

    def publish_download(self, partial, path):
        os.replace(str(partial), str(path))

    def upload(self, entry, path, identity, existing):
        validate(entry, identity)
        source_stat = path.stat()
        bound = (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ino, source_stat.st_dev)
        spec = {**entry, **identity}
        if existing.get("status") == "server_verified":
            if existing["sha256"] != identity["sha256"] or existing["size"] != identity["size"]:
                raise ValueError("Existing remote original differs from the local original")
            receipt = existing
        else:
            pieces = []
            chunk = self.args.upload_chunk_mib * MIB
            with path.open("rb") as stream:
                for index, start in enumerate(range(0, identity["size"], chunk)):
                    block = stream.read(min(chunk, identity["size"] - start))
                    pieces.append({"index": index, "start": start, "size": len(block),
                                   "sha256": hashlib.sha256(block).hexdigest()})
            spec["pieces"] = pieces
            self.rpc("prepare", entry, spec)
            complete = set(self.rpc("chunks", entry)["complete"])
            self.event("uploading", entry, pieces=len(pieces), resumed_pieces=len(complete))
            upload_started = time.time()

            def worker(piece):
                if piece["index"] in complete:
                    return piece["size"]
                for attempt in range(4):
                    proc = subprocess.Popen(self.command("receive", entry, piece["index"]),
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    try:
                        remaining = piece["size"]
                        with path.open("rb") as stream:
                            stream.seek(piece["start"])
                            while remaining:
                                block = stream.read(min(4 * MIB, remaining))
                                if not block:
                                    raise ValueError("Local original truncated during upload")
                                proc.stdin.write(block)
                                remaining -= len(block)
                        proc.stdin.close()
                        proc.stdin = None
                        out, err = proc.communicate(timeout=1800)
                        if proc.returncode:
                            raise RuntimeError("SSH receive failed: " + err.decode("utf-8", "replace")[-2000:])
                        received = json.loads(out)
                        if received.get("status") != "piece_verified" or received["size"] != piece["size"]:
                            raise ValueError("Server did not verify SSH piece")
                        return piece["size"]
                    except Exception:
                        proc.kill()
                        proc.communicate()
                        if attempt == 3:
                            raise
                        time.sleep(3 * (attempt + 1))

            verified_bytes = 0
            with ThreadPoolExecutor(max_workers=self.args.upload_workers) as pool:
                pending = [pool.submit(worker, piece) for piece in pieces]
                for future in as_completed(pending):
                    verified_bytes += future.result()
                    self.event("upload_progress", entry, verified_bytes=verified_bytes,
                               elapsed_s=time.time() - upload_started)
            self.event("server_verifying", entry)
            receipt = self.rpc("commit", entry)
        if receipt.get("status") != "server_verified" or receipt.get("key") != entry["key"] or receipt.get("relative_path") != entry["relative_path"]:
            raise ValueError("Missing or misbound remote verification receipt")
        if receipt["sha256"] != identity["sha256"] or receipt["size"] != identity["size"]:
            raise ValueError("Remote verification failed; local original retained")
        atomic_json(self.report / "server_verification" / (entry["key"] + ".json"), receipt)
        self.event("server_verified", entry, sha256=identity["sha256"])
        if self.args.delete_local_after_verify:
            # No recursive deletion: only this exact, validated local original.
            checked = confined(self.data, entry["relative_path"])
            current = checked.stat()
            if (current.st_size, current.st_mtime_ns, current.st_ino, current.st_dev) != bound:
                raise ValueError("Local original changed; refusing deletion")
            checked.unlink()
            self.event("local_deleted", entry, sha256=identity["sha256"])
        self.event("complete", entry, sha256=identity["sha256"], local_deleted=self.args.delete_local_after_verify)

    def run(self):
        work_queue = queue.Queue(maxsize=1)

        def producer():
            try:
                for entry in self.items:
                    for attempt in range(4):
                        try:
                            self.event("checking_existing", current_check=entry["relative_path"])
                            remote = self.rpc("inspect", entry)
                            self.state["current_check"] = None
                            path = confined(self.data, entry["relative_path"])
                            if remote.get("status") == "server_verified" and not path.exists():
                                atomic_json(self.report / "server_verification" / (entry["key"] + ".json"), remote)
                                self.event("complete", entry, sha256=remote["sha256"],
                                           local_deleted=self.state["files"].get(entry["key"], {}).get("local_deleted", False),
                                           reused_remote=True)
                                break
                            path, identity = self.download(entry)
                            work_queue.put((entry, path, identity, remote))
                            break
                        except Exception:
                            self.event("download_or_inspect_failed", entry, attempt=attempt + 1,
                                       traceback=traceback.format_exc())
                            if attempt == 3:
                                self.failures.append(entry["key"])
                            else:
                                time.sleep(min(10 * 2 ** attempt, 60))
            finally:
                work_queue.put(None)

        self.event("pipeline_started", local_free_bytes=shutil.disk_usage(self.data).free)
        producer_thread = threading.Thread(target=producer, name="local-downloader", daemon=True)
        producer_thread.start()
        while True:
            item = work_queue.get()
            if item is None:
                break
            entry, path, identity, remote = item
            for attempt in range(4):
                try:
                    self.upload(entry, path, identity, remote)
                    break
                except Exception:
                    self.event("upload_or_cleanup_failed", entry, attempt=attempt + 1,
                               traceback=traceback.format_exc())
                    if attempt == 3:
                        self.failures.append(entry["key"])
                    else:
                        time.sleep(min(10 * 2 ** attempt, 60))
        producer_thread.join()
        completed_keys = {key for key, value in self.state["files"].items() if value.get("event") == "complete"}
        self.failures.extend(e["key"] for e in self.items if e["key"] not in completed_keys)
        self.state["status"] = "complete" if not self.failures else "incomplete"
        self.state["failed_keys"] = sorted(set(self.failures))
        self.event("pipeline_finished", failures=self.state["failed_keys"])
        return 0 if not self.failures else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-helper", required=True)
    parser.add_argument("--download-workers", type=int, default=8)
    interfaces = parser.add_mutually_exclusive_group()
    interfaces.add_argument("--download-interface", default="", help="Local IP address for this downloader only")
    interfaces.add_argument("--download-interface-alias", default="", help="Windows interface alias; refresh its Preferred IPv4 automatically")
    parser.add_argument("--curl-resolve", action="append", default=[], help="Per-process host:port:IP DNS override")
    parser.add_argument("--download-chunk-mib", type=int, default=64)
    parser.add_argument("--upload-workers", type=int, default=4)
    parser.add_argument("--upload-chunk-mib", type=int, default=256)
    parser.add_argument("--delete-local-after-verify", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-keys", default="")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # OS lock is released on exit/crash and prevents duplicate upload/deletion.
    with (args.report_dir / "acquisition.lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = Acquisition(args)
        try:
            result = job.run()
        except Exception:
            job.state["status"] = "failed"
            job.event("fatal_error", traceback=traceback.format_exc())
            raise
        raise SystemExit(result)


if __name__ == "__main__":
    main()
