"""Checks for server publication, range recovery and remote-only observation."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from acquire_dichasus_server import ServerAcquisition
from dichasus_remote_transfer import atomic_json, hashes
from show_dichasus_progress import snapshot


class ServerSafety(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dichasus-server-test-")
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root / "datasets"
        self.report = self.root / "reports"
        self.payload = bytes(range(251)) * 8500
        self.entry = {"key": "fixture-1", "filename": "fixture.bin", "relative_path": "fixture.bin",
                      "groups": ["fixture"], "size": len(self.payload), "checksum_type": "MD5",
                      "checksum": hashlib.md5(self.payload).hexdigest(),
                      "source": "https://example.invalid/fixture"}
        self.manifest = {"files": [self.entry], "groups": [{"group": "fixture"}],
                         "total_files": 1, "total_bytes": len(self.payload)}
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.job = ServerAcquisition(Namespace(manifest=manifest, data_root=self.data,
                                              report_dir=self.report, only_keys="", limit=0,
                                              download_chunk_mib=1, download_workers=3))

    def tearDown(self):
        assert self.root == Path(self.temporary.name).resolve()
        self.temporary.cleanup()

    def transport(self, payload=None, failed=None, bad_header=False):
        payload = self.payload if payload is None else payload
        requested = []
        lock = threading.Lock()

        def run(command, **kwargs):
            start, end = map(int, command[command.index("--range") + 1].split("-"))
            with lock:
                requested.append(start // (1024 ** 2))
            if start // (1024 ** 2) in (failed or set()):
                return subprocess.CompletedProcess(command, 22, b"", b"fixture HTTP failure")
            dest = Path(command[command.index("--output") + 1])
            dest.write_bytes(payload[start:end + 1])
            header = ("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes %d-%d/%d\r\n\r\n"
                      % (start + int(bad_header), end, len(self.payload))).encode()
            return subprocess.CompletedProcess(command, 0, header, b"")

        return run, requested

    def test_server_pipeline_publishes_verified_original(self):
        transport, _ = self.transport()
        with patch("acquire_dichasus.subprocess.run", side_effect=transport):
            self.assertEqual(self.job.run(), 0)
        self.assertEqual((self.data / "fixture.bin").read_bytes(), self.payload)
        receipt = json.loads((self.data / ".verification/fixture-1.json").read_text())
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(self.job.state["execution_mode"], "server_direct")
        self.assertFalse(self.job.state["local_dataset_download_enabled"])
        self.assertEqual(self.job.state["complete_files"], 1)
        events = [json.loads(line) for line in self.job.events.read_text().splitlines()]
        self.assertFalse(any(e["event"] == "server_download_failed" for e in events))

    def test_atomic_publication_preserves_existing_raw(self):
        target = self.data / "fixture.bin"
        partial = self.data / "fixture.partial"
        target.write_bytes(b"existing raw")
        partial.write_bytes(b"new staging")
        with self.assertRaises(FileExistsError):
            self.job.publish_download(partial, target)
        self.assertEqual(target.read_bytes(), b"existing raw")
        self.assertEqual(partial.read_bytes(), b"new staging")

    def test_corrupted_checkpoint_only_refetches_corrupt_or_missing_ranges(self):
        work = self.data / ".download/fixture-1"
        work.mkdir(parents=True)
        (work / "original.partial").write_bytes(b"!" + self.payload[1:])
        chunk = 1024 ** 2
        ranges = {str(i): {"start": i * chunk, "size": chunk,
                           "sha256": hashlib.sha256(self.payload[i * chunk:(i + 1) * chunk]).hexdigest()}
                  for i in (0, 1)}
        atomic_json(self.report / "download_ranges/fixture-1.json", {"ranges": ranges})
        transport, requested = self.transport()
        with patch("acquire_dichasus.subprocess.run", side_effect=transport):
            path, _ = self.job.download(self.entry)
        self.assertEqual(set(requested), {0, 2})
        self.assertEqual(path.read_bytes(), self.payload)

    def test_failed_range_keeps_other_completed_ranges_for_retry(self):
        transport, _ = self.transport(failed={0})
        with patch("acquire_dichasus.subprocess.run", side_effect=transport):
            with self.assertRaises(RuntimeError):
                self.job.download(self.entry)
        journal = json.loads((self.report / "download_ranges/fixture-1.json").read_text())
        self.assertEqual(set(journal["ranges"]), {"1", "2"})
        transport, requested = self.transport()
        with patch("acquire_dichasus.subprocess.run", side_effect=transport):
            path, _ = self.job.download(self.entry)
        self.assertEqual(requested, [0])
        self.assertEqual(path.read_bytes(), self.payload)

    def test_interrupted_allocation_without_verified_ranges_can_resume(self):
        work = self.data / ".download/fixture-1"
        work.mkdir(parents=True)
        (work / "original.partial").write_bytes(b"\x00" * 17)
        transport, _ = self.transport()
        with patch("acquire_dichasus.subprocess.run", side_effect=transport):
            path, _ = self.job.download(self.entry)
        self.assertEqual(path.read_bytes(), self.payload)

    def test_length_change_with_verified_ranges_is_retained(self):
        work = self.data / ".download/fixture-1"
        work.mkdir(parents=True)
        partial = work / "original.partial"
        partial.write_bytes(b"preserve this checkpoint")
        atomic_json(self.report / "download_ranges/fixture-1.json", {"ranges": {"0": {"size": 1024}}})
        with self.assertRaises(ValueError):
            self.job.download(self.entry)
        self.assertEqual(partial.read_bytes(), b"preserve this checkpoint")

    def test_inexact_http_range_is_not_published(self):
        transport, _ = self.transport(bad_header=True)
        with patch("acquire_dichasus.subprocess.run", side_effect=transport):
            with self.assertRaises(ValueError):
                self.job.download(self.entry)
        self.assertFalse((self.data / "fixture.bin").exists())

    def test_checksum_failure_retains_staging_without_publishing(self):
        transport, _ = self.transport(payload=b"?" * len(self.payload))
        with patch("acquire_dichasus.subprocess.run", side_effect=transport):
            with self.assertRaises(ValueError):
                self.job.download(self.entry)
        self.assertFalse((self.data / "fixture.bin").exists())
        self.assertTrue((self.data / ".download/fixture-1/original.partial").exists())

    def test_reuse_rejects_a_changed_original_receipt(self):
        target = self.data / "fixture.bin"
        target.write_bytes(self.payload)
        receipt = {**self.entry, **hashes(target), "sha256": "0" * 64}
        atomic_json(self.data / ".verification/fixture-1.json", receipt)
        with self.assertRaises(ValueError):
            self.job.reuse_existing(self.entry, target)
        self.assertEqual(target.read_bytes(), self.payload)

    def test_dashboard_uses_remote_liveness_and_detects_a_stale_connection(self):
        self.job.state["polled_at"] = time.time()
        self.job.event("pipeline_started")
        with patch("show_dichasus_progress.alive", side_effect=AssertionError("remote PID checked locally")):
            current = snapshot(self.report, self.manifest)
            self.assertEqual(current["status"], "running")
            self.job.state["polled_at"] = time.time() - 60
            self.job.event("pipeline_started")
            self.assertEqual(snapshot(self.report, self.manifest)["status"], "disconnected")


if __name__ == "__main__":
    unittest.main()
