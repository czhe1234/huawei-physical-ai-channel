"""Remove only legacy dc01 local transfer binaries after server verification.

Every collection must reconstruct the server-verified SHA256. JSON provenance
and download logs remain. A persisted exact-file plan makes deletion resumable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from dichasus_remote_transfer import atomic_json, confined, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    root, report = args.data_root.resolve(), args.report_dir.resolve()
    receipt = json.loads((report / "server_verification/darus-269445.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "server_verified" or receipt["relative_path"] != "dc01/dichasus-dc01.tfrecords":
        raise ValueError("Need the canonical dc01 server verification receipt")
    plan_path = report / "legacy_local_cache_deletion_plan.json"
    resuming = plan_path.exists()
    if resuming:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["server_sha256"] != receipt["sha256"]:
            raise ValueError("Server identity changed")
    else:
        targets = []
        for folder in ("parts", "upload_parts"):
            directory = confined(root, "dc01/" + folder)
            if not directory.exists():
                continue
            files = sorted(directory.glob("*.bin"))
            if not files:
                continue
            whole = hashlib.sha256()
            total = 0
            for path in files:
                relative = path.relative_to(root).as_posix()
                checked = confined(root, relative)
                before = checked.stat()
                if not re.fullmatch(r"[0-9]{12}_[0-9]{12}\.bin" if folder == "parts" else r"part[0-9]{3}\.bin", path.name):
                    raise ValueError("Unexpected cache file name")
                digest = hashlib.sha256()
                with checked.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
                        whole.update(block)
                        digest.update(block)
                        total += len(block)
                st = checked.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev):
                    raise ValueError("Cache changed while validating")
                targets.append({"relative_path": relative, "sha256": digest.hexdigest(),
                                "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                "inode": st.st_ino, "device": st.st_dev})
            if total != receipt["size"] or whole.hexdigest() != receipt["sha256"]:
                raise ValueError("Legacy cache does not reconstruct the server-verified original")
            print(json.dumps({"stage": "cache_identity_verified", "folder": folder, "bytes": total}), flush=True)
        plan = {"server_sha256": receipt["sha256"], "server_size": receipt["size"], "targets": targets}
        atomic_json(plan_path, plan)
    removed_bytes = 0
    removed_files = 0
    for target in plan["targets"]:
        path = confined(root, target["relative_path"])
        if not path.exists():
            continue
        st = path.stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev) != (target["size"], target["mtime_ns"], target["inode"], target["device"]):
            raise ValueError("Cache changed; refusing deletion")
        # A resumed plan also validates the bytes, not only stored metadata.
        if resuming and hashes(path)["sha256"] != target["sha256"]:
            raise ValueError("Cache checksum changed; refusing deletion")
        path.unlink()
        removed_files += 1
        removed_bytes += target["size"]
    result = {"status": "complete", "deleted_files_this_run": removed_files,
              "deleted_bytes_this_run": removed_bytes, "server_sha256": receipt["sha256"]}
    atomic_json(report / "legacy_local_cache_cleanup.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
