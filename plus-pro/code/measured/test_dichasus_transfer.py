"""Safety tests for the verified-upload/local-deletion boundary."""
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from acquire_dichasus import Acquisition, WindowsInterface, allocate_partial
from dichasus_remote_transfer import confined, hashes, validate


class TransferSafety(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dichasus-transfer-safety-")
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root / "datasets"
        self.data.mkdir()
        self.source = self.data / "fixture.bin"
        self.source.write_bytes(b"verified transfer boundary\x00\x01")
        self.identity = hashes(self.source)
        self.entry = {"key": "fixture-1", "relative_path": "fixture.bin", "size": self.identity["size"],
                      "checksum_type": "MD5", "checksum": self.identity["md5"],
                      "source": "https://example.invalid/fixture"}
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"files": [self.entry]}), encoding="utf-8")
        args = Namespace(data_root=self.data, report_dir=self.root / "reports", manifest=manifest,
                         only_keys="", limit=0, upload_chunk_mib=1, upload_workers=1,
                         delete_local_after_verify=True)
        self.job = Acquisition(args)
        self.receipt = {"status": "server_verified", **self.entry, **self.identity}

    def tearDown(self):
        # The resolved test target must remain the exact directory allocated by
        # TemporaryDirectory, never a caller-supplied workspace or raw dataset.
        assert self.root == Path(self.temporary.name).resolve()
        self.temporary.cleanup()

    def test_path_escape_rejected(self):
        for relative in ("../outside.bin", "/outside.bin", "a/../../outside.bin", "a\\outside.bin", ""):
            with self.assertRaises((ValueError, RuntimeError)):
                confined(self.data, relative)

    def test_publisher_checksum_failure_keeps_original(self):
        with self.assertRaises(ValueError):
            validate({**self.entry, "checksum": "0" * 32}, self.identity)
        self.assertTrue(self.source.exists())

    def test_partial_file_can_be_fully_populated_and_verified(self):
        path = self.data / "staging.partial"
        size = 4 * 1024 ** 2 + 123
        allocate_partial(path, size)
        self.assertEqual(path.stat().st_size, size)
        expected = hashlib.sha256()
        with path.open("r+b") as stream:
            for start in range(0, size, 65536):
                block = bytes([start // 65536 % 251]) * min(65536, size - start)
                expected.update(block)
                stream.seek(start)
                stream.write(block)
        self.assertEqual(hashes(path)["sha256"], expected.hexdigest())

    def test_wrong_server_sha_retains_original(self):
        with self.assertRaises(ValueError):
            self.job.upload(self.entry, self.source, self.identity, {**self.receipt, "sha256": "0" * 64})
        self.assertTrue(self.source.exists())

    def test_missing_server_receipt_retains_original(self):
        with patch.object(self.job, "rpc", side_effect=[{}, {"complete": [0]}, {"status": "failed"}]):
            with self.assertRaises(ValueError):
                self.job.upload(self.entry, self.source, self.identity, {"status": "missing"})
        self.assertTrue(self.source.exists())

    def test_misbound_receipt_retains_original(self):
        with self.assertRaises(ValueError):
            self.job.upload(self.entry, self.source, self.identity, {**self.receipt, "relative_path": "other.bin"})
        self.assertTrue(self.source.exists())

    def test_verified_receipt_deletes_only_expected_original(self):
        neighbor = self.data / "keep.bin"
        neighbor.write_bytes(b"preserved")
        self.job.upload(self.entry, self.source, self.identity, self.receipt)
        self.assertFalse(self.source.exists())
        self.assertEqual(neighbor.read_bytes(), b"preserved")
        self.assertTrue((self.root / "reports/server_verification/fixture-1.json").exists())

    def test_changed_local_file_retained(self):
        def altered_rpc(action, entry, payload=None):
            if action == "prepare":
                return {}
            if action == "chunks":
                return {"complete": [0]}
            self.source.write_bytes(b"changed during upload")
            return self.receipt
        with patch.object(self.job, "rpc", side_effect=altered_rpc):
            with self.assertRaises(ValueError):
                self.job.upload(self.entry, self.source, self.identity, {"status": "missing"})
        self.assertTrue(self.source.exists())

    def test_interface_ip_and_alias_are_mutually_exclusive(self):
        args = Namespace(**{**vars(self.job.args), "download_interface": "192.0.2.1",
                            "download_interface_alias": "Broadband Connection"})
        with self.assertRaisesRegex(ValueError, "not both"):
            Acquisition(args)

    def test_interface_input_validation_and_legacy_binding(self):
        for alias in ("", "a\n", "a'", "a\"", "a;exit", "x" * 257):
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                WindowsInterface(alias)
        args = Namespace(**{**vars(self.job.args), "download_interface": "192.0.2.1"})
        job = Acquisition(args)
        self.assertEqual(job.curl_extra_options[-2:], ("--interface", "192.0.2.1"))
        self.assertIsNone(job.dynamic_interface)
        with self.assertRaises(ValueError):
            Acquisition(Namespace(**{**vars(self.job.args), "download_interface": "bad-ip"}))

    def test_alias_refreshes_after_ip_change_and_shares_queries(self):
        with patch("acquire_dichasus.os.name", "nt"):
            interface = WindowsInterface("Broadband Connection")
        now = [100.0]
        result = lambda ip: subprocess.CompletedProcess([], 0, json.dumps([ip]).encode(), b"")
        with patch("acquire_dichasus.time.monotonic", side_effect=lambda: now[0]), \
                patch("acquire_dichasus.subprocess.run", side_effect=[result("192.0.2.1"),
                                                                     result("192.0.2.2"),
                                                                     result("192.0.2.3")]) as run:
            with ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(list(pool.map(lambda _: interface.resolve(), range(8))), ["192.0.2.1"] * 8)
            self.assertEqual(run.call_count, 1)
            now[0] += 0.1
            self.assertEqual(interface.resolve(failed_address="192.0.2.1"), "192.0.2.2")
            with ThreadPoolExecutor(max_workers=8) as pool:
                addresses = list(pool.map(lambda _: interface.resolve(failed_address="192.0.2.1"), range(8)))
            self.assertEqual(addresses, ["192.0.2.2"] * 8)
            self.assertEqual(run.call_count, 2)
            now[0] += 3
            self.assertEqual(interface.resolve(), "192.0.2.3")
            self.assertEqual(run.call_args.kwargs["env"]["DICHASUS_INTERFACE_ALIAS"], "Broadband Connection")

    def test_alias_without_usable_address_never_falls_back_to_default(self):
        with patch("acquire_dichasus.os.name", "nt"):
            interface = WindowsInterface("Broadband Connection")
        self.job.dynamic_interface = interface
        self.job.args.download_chunk_mib = 1
        self.job.args.download_workers = 1
        self.source.unlink()
        with patch("acquire_dichasus.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, b"[]", b"")) as run:
            with self.assertRaisesRegex(RuntimeError, "No usable Preferred IPv4"):
                self.job.download(self.entry)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "powershell.exe")
        self.assertFalse(self.source.exists())

    def test_curl45_refreshes_binding_and_retries_same_range(self):
        with patch("acquire_dichasus.os.name", "nt"):
            interface = WindowsInterface("Broadband Connection")
        self.job.dynamic_interface = interface
        self.job.args.download_chunk_mib = 1
        self.job.args.download_workers = 1
        payload = self.source.read_bytes()
        self.source.unlink()
        commands = []

        def run(command, **kwargs):
            if command[0] == "powershell.exe":
                ip = "192.0.2.1" if not commands else "192.0.2.2"
                return subprocess.CompletedProcess(command, 0, json.dumps([ip]).encode(), b"")
            commands.append(command)
            if len(commands) == 1:
                return subprocess.CompletedProcess(command, 45, b"", b"binding failed")
            Path(command[command.index("--output") + 1]).write_bytes(payload)
            header = ("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-%d/%d\r\n\r\n"
                      % (len(payload) - 1, len(payload))).encode()
            return subprocess.CompletedProcess(command, 0, header, b"")

        with patch("acquire_dichasus.subprocess.run", side_effect=run):
            path, identity = self.job.download(self.entry)
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(identity, self.identity)
        self.assertEqual([c[c.index("--interface") + 1] for c in commands], ["192.0.2.1", "192.0.2.2"])
        self.assertTrue(all("--retry-all-errors" not in c for c in commands))
        self.assertTrue(all(c[c.index("--retry") + 1] == "4" for c in commands))


if __name__ == "__main__":
    unittest.main()
