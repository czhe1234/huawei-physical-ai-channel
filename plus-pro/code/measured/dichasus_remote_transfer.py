"""SSH-side, stdlib-only atomic receiver for locally staged DICHASUS files.

Only manifest-listed paths below --root are accepted. Raw originals are never
overwritten. SHA256 and publisher checksum gate commit and local cleanup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(40):
        try:
            os.replace(str(tmp), str(path))
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.1)


def confined(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative or relative in {"", "."}:
        raise ValueError("Unsafe relative dataset path")
    target = root.joinpath(*p.parts)
    for part in [target] + list(target.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError("Symbolic links are not allowed")
    target.resolve().relative_to(root.resolve())
    return target


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            md5.update(block)
            sha.update(block)
            size += len(block)
    return {"size": size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def validate(entry, identity):
    if identity["size"] != entry["size"]:
        raise ValueError("File length mismatch")
    algorithm = (entry.get("checksum_type") or "").upper().replace("-", "")
    if algorithm and identity["md5" if algorithm == "MD5" else "sha256"] != entry["checksum"]:
        raise ValueError("Publisher checksum mismatch")
    if entry.get("sha256") and entry["sha256"] != identity["sha256"]:
        raise ValueError("Local-to-server SHA256 mismatch")


def layout(root, key):
    if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in key):
        raise ValueError("Unsafe file key")
    work = confined(root, ".transfer/" + key)
    work.mkdir(parents=True, exist_ok=True)
    return work


def verified(root, entry, target, identity):
    receipt = {"status": "server_verified", "key": entry["key"],
               "relative_path": entry["relative_path"], "source": entry["source"],
               "publisher_checksum_type": entry.get("checksum_type"),
               "publisher_checksum": entry.get("checksum"), "verified_at": time.time(), **identity}
    atomic_json(confined(root, ".verification/" + entry["key"] + ".json"), receipt)
    return receipt


def execute(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "inspect":
        entry = json.load(sys.stdin)
        target = confined(root, entry["relative_path"])
        if not target.exists():
            return {"status": "missing"}
        identity = hashes(target)
        validate(entry, identity)
        return verified(root, entry, target, identity)
    work = layout(root, args.key)
    if args.action == "prepare":
        entry = json.load(sys.stdin)
        confined(root, entry["relative_path"])
        if entry["key"] != args.key:
            raise ValueError("Transfer key mismatch")
        prior = work / "manifest.json"
        if prior.exists():
            old = json.loads(prior.read_text(encoding="utf-8"))
            if (old["sha256"], old["size"], old["relative_path"]) != (entry["sha256"], entry["size"], entry["relative_path"]):
                raise ValueError("Transfer resume identity changed")
        atomic_json(prior, entry)
        return {"status": "prepared"}
    entry = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    if args.action == "chunks":
        result = []
        for piece in entry["pieces"]:
            p = work / (str(piece["index"]) + ".bin")
            if p.exists() and p.stat().st_size == piece["size"] and hashes(p)["sha256"] == piece["sha256"]:
                result.append(piece["index"])
        return {"complete": result}
    if args.action == "receive":
        piece = next(p for p in entry["pieces"] if p["index"] == args.index)
        dest = work / (str(args.index) + ".bin")
        tmp = work / (str(args.index) + ".partial")
        sha = hashlib.sha256()
        size = 0
        with tmp.open("wb") as stream:
            while size < piece["size"]:
                block = sys.stdin.buffer.read(min(4 * 1024 ** 2, piece["size"] - size))
                if not block:
                    raise ValueError("Truncated SSH transfer")
                stream.write(block)
                sha.update(block)
                size += len(block)
            if sys.stdin.buffer.read(1):
                raise ValueError("SSH transfer exceeded manifest length")
            stream.flush()
            os.fsync(stream.fileno())
        if sha.hexdigest() != piece["sha256"]:
            raise ValueError("SSH transfer SHA256 mismatch")
        os.replace(str(tmp), str(dest))
        return {"status": "piece_verified", "index": args.index, "size": size}
    if args.action == "commit":
        target = confined(root, entry["relative_path"])
        if target.exists():
            identity = hashes(target)
            validate(entry, identity)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = work / "assembled.partial"
            with tmp.open("wb") as stream:
                for piece in entry["pieces"]:
                    p = work / (str(piece["index"]) + ".bin")
                    actual = hashes(p)
                    if actual["size"] != piece["size"] or actual["sha256"] != piece["sha256"]:
                        raise ValueError("Server chunk failed pre-commit validation")
                    with p.open("rb") as source:
                        for block in iter(lambda: source.read(8 * 1024 ** 2), b""):
                            stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            identity = hashes(tmp)
            validate(entry, identity)
            # Atomic no-overwrite publication on the same filesystem.
            os.link(str(tmp), str(target))
            tmp.unlink()
            dfd = os.open(str(target.parent), os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        result = verified(root, entry, target, identity)
        # Only this run's generated transfer pieces, after verified publication.
        for piece in entry["pieces"]:
            p = work / (str(piece["index"]) + ".bin")
            if p.exists():
                p.unlink()
        return result
    raise ValueError("Unknown operation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "prepare", "chunks", "receive", "commit"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key", default="")
    parser.add_argument("--index", type=int, default=-1)
    args = parser.parse_args()
    print(json.dumps(execute(args)), flush=True)


if __name__ == "__main__":
    main()
