"""Resume the frozen DICHASUS manifest directly on the storage server.

No SSH upload or client-side dataset download is performed. Existing originals
are hashed before reuse; publication is atomic and never overwrites a raw file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import time
import traceback

from acquire_dichasus import Acquisition
from dichasus_remote_transfer import atomic_json, confined, hashes, validate, verified


class ServerAcquisition(Acquisition):
    execution_mode = "server_direct"
    staging_event = "preparing_server_file"
    verification_event = "server_download_verified"
    verification_directory = "server_download_verification"
    # Ubuntu's curl 7.68 predates --retry-all-errors; whole-file retries below
    # retain successfully journalled ranges after other curl failures.
    curl_retry_options = ("--retry", "4")
    progress_interval = 5

    def __init__(self, args):
        super().__init__(args)
        bundle = getattr(args, "ca_bundle", None)
        if bundle is None:
            candidate = self.report / "tls/dichasus-ca-bundle.pem"
            bundle = candidate if candidate.exists() else None
        if bundle is not None:
            bundle = bundle.resolve(strict=True)
            self.curl_extra_options = ("--cacert", str(bundle))
            self.state["tls_ca_bundle_sha256"] = hashes(bundle)["sha256"]
        self.state.update({"worker_host": socket.gethostname(), "worker_alive": True,
                           "local_dataset_download_enabled": False,
                           "download_stage": "server", "current_check": None})
        # Local staging counters do not describe bytes held by this server.
        for key, value in list(self.state["files"].items()):
            if value.get("event") != "complete":
                self.state["files"][key] = {"event": "pending", "verified_bytes": 0}

    def publish_download(self, partial, path):
        os.link(str(partial), str(path))
        partial.unlink()
        if os.name != "nt":
            descriptor = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def reuse_existing(self, entry, path):
        self.state["current_check"] = entry["relative_path"]
        self.event("checking_existing", current_check=entry["relative_path"])
        identity = hashes(path)
        validate(entry, identity)
        receipt_path = confined(self.data, ".verification/" + entry["key"] + ".json")
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (old.get("key") != entry["key"] or old.get("relative_path") != entry["relative_path"]
                    or old.get("sha256") != identity["sha256"] or old.get("size") != identity["size"]):
                raise ValueError("Existing verification receipt differs from the original")
        return identity

    def run(self):
        self.event("pipeline_started", server_free_bytes=shutil.disk_usage(self.data).free,
                   execution_mode=self.execution_mode, local_dataset_download_enabled=False)
        for entry in self.items:
            for attempt in range(4):
                try:
                    path = confined(self.data, entry["relative_path"])
                    reused = path.exists()
                    if reused:
                        identity = self.reuse_existing(entry, path)
                    else:
                        _, identity = self.download(entry)
                    receipt = verified(self.data, entry, path, identity)
                    receipt["acquisition_mode"] = self.execution_mode
                    atomic_json(self.report / "server_verification" / (entry["key"] + ".json"), receipt)
                    self.state["current_check"] = None
                    self.event("complete", entry, **identity, reused_remote=reused,
                               local_deleted=self.state["files"].get(entry["key"], {}).get("local_deleted", False))
                    break
                except Exception:
                    self.event("server_download_failed", entry, attempt=attempt + 1,
                               traceback=traceback.format_exc())
                    if attempt == 3:
                        self.failures.append(entry["key"])
                    else:
                        time.sleep(min(10 * 2 ** attempt, 60))
        completed = {key for key, info in self.state["files"].items() if info.get("event") == "complete"}
        self.failures.extend(e["key"] for e in self.items if e["key"] not in completed)
        self.state["status"] = "complete" if not self.failures else "incomplete"
        self.state["failed_keys"] = sorted(set(self.failures))
        self.state["current_check"] = None
        self.event("pipeline_finished", failures=self.state["failed_keys"])
        return 0 if not self.failures else 1


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--download-chunk-mib", type=int, default=64)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--only-keys", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / "acquisition.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = ServerAcquisition(args)
        try:
            result = job.run()
        except Exception:
            job.state["status"] = "failed"
            job.event("fatal_error", traceback=traceback.format_exc())
            raise
        raise SystemExit(result)


if __name__ == "__main__":
    main()
