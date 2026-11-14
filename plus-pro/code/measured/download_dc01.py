"""Download original DICHASUS dc01 locally, with resumable bounded HTTP ranges.

Only public data endpoints are used. Signed storage redirects stay in memory.
Parts are immutable after size and SHA256 verification. The final original file
is accepted only after its published MD5 has been matched.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
import urllib.request

API = "https://darus.uni-stuttgart.de/api/access/datafile/269445"
SIZE = 16870531926
MD5 = "7934b83963491b493a37556484d6ca42"
LOCAL = threading.local()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    # Windows readers may temporarily omit FILE_SHARE_DELETE.
    for attempt in range(100):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.05)


def get_range(start, end, destination, timeout=30):
    """Reject servers ignoring Range, truncation, and inconsistent objects."""
    url = getattr(LOCAL, "url", API)
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{end}",
        "User-Agent": "hw-measured-evaluation/1.0",
        "Accept-Encoding": "identity",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            expected = f"bytes {start}-{end}/{SIZE}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Server did not return the exact requested byte range")
            LOCAL.url = response.url
            digest = hashlib.sha256()
            n = 0
            with destination.open("wb") as out:
                while True:
                    data = response.read(65536)
                    if not data:
                        break
                    out.write(data)
                    digest.update(data)
                    n += len(data)
            if n != end - start + 1:
                raise RuntimeError(f"Truncated range: received {n} bytes")
            return digest.hexdigest()
    except Exception:
        LOCAL.url = API
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--probe-chunks", type=int, default=0)
    args = parser.parse_args()
    root = args.directory.resolve()
    parts = root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_mib * 1024**2
    total = (SIZE + chunk - 1) // chunk
    indices = list(range(min(total, args.probe_chunks) if args.probe_chunks else total))
    started = time.time()
    complete = 0
    bytes_done = 0
    failures = []

    def worker(i):
        start, end = i * chunk, min(SIZE, (i + 1) * chunk) - 1
        dest = parts / f"{start:012d}_{end:012d}.bin"
        meta = dest.with_suffix(".json")
        if dest.exists() and meta.exists() and dest.stat().st_size == end - start + 1:
            info = json.loads(meta.read_text())
            # Verify resume contents; size alone is not enough.
            if hashlib.sha256(dest.read_bytes()).hexdigest() == info.get("sha256"):
                return dest.stat().st_size, "resumed"
        last_error = None
        for attempt in range(4):
            try:
                tmp = dest.with_suffix(".partial")
                t = time.time()
                digest = get_range(start, end, tmp)
                os.replace(tmp, dest)
                atomic_json(meta, {"start": start, "end": end, "sha256": digest,
                                   "seconds": time.time() - t})
                return end - start + 1, "downloaded"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Range {start}-{end} failed: {last_error}")

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(worker, i): i for i in indices}
        last_print = 0.0
        last_progress = 0.0
        for future in futures.as_completed(pending):
            i = pending[future]
            try:
                n, state = future.result()
                complete += 1
                bytes_done += n
            except Exception:
                failures.append({"chunk": i, "traceback": traceback.format_exc()})
                print(json.dumps({"failed_chunk": i, "error": failures[-1]["traceback"]}), flush=True)
            now = time.time()
            progress = dict(stage="download", completed=complete, requested=len(indices),
                            total_chunks=total, verified_bytes=bytes_done,
                            elapsed_s=now-started, mean_MiB_s=bytes_done / 1024**2 / max(now-started, 1e-6),
                            failures=failures, source=API, full_size=SIZE,
                            probe=bool(args.probe_chunks))
            if now-last_progress>1 or complete==len(indices):
                try:
                    atomic_json(root / "download_progress.json", progress)
                    last_progress=now
                except PermissionError:
                    # A live progress viewer must not abort otherwise verified data.
                    print('PROGRESS_FILE_BUSY: acquisition continues',flush=True)
            if now - last_print > 15 or complete == len(indices):
                print(json.dumps(progress), flush=True)
                last_print = now
    if failures:
        raise RuntimeError(f"{len(failures)} byte ranges failed; rerun to resume")
    if args.probe_chunks:
        return
    target = root / "dichasus-dc01.tfrecords"
    if target.exists():
        raise RuntimeError("Final file already exists; validate it explicitly instead of overwriting")
    output = target.with_suffix(".assembling")
    md5, sha = hashlib.md5(), hashlib.sha256()
    with output.open("wb") as out:
        for i in range(total):
            start, end = i * chunk, min(SIZE, (i + 1) * chunk) - 1
            p = parts / f"{start:012d}_{end:012d}.bin"
            with p.open("rb") as src:
                while True:
                    data = src.read(1024**2)
                    if not data:
                        break
                    out.write(data)
                    md5.update(data)
                    sha.update(data)
    if output.stat().st_size != SIZE or md5.hexdigest() != MD5:
        raise RuntimeError("Original data failed the published size/MD5 check")
    os.replace(output, target)
    atomic_json(root / "download_verified.json", dict(
        file=target.name, size=SIZE, md5=md5.hexdigest(), sha256=sha.hexdigest(),
        source=API, doi="10.18419/darus-3831", license="CC BY 4.0",
        elapsed_s=time.time()-started, workers=args.workers))
    print("DOWNLOAD_VERIFIED", target.name, sha.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
