"""Acquire immutable author releases for the dc01 paper-baseline study.

Run on the local workstation first, then transfer the verified source package.
This command acquires sources only; it never runs an author's code or trains.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import time
from urllib.request import Request, urlopen
import zipfile


SOURCES = (
    ("newrf", "LuHaofan/NeWRF", "78feb38471f766f17764a28e8518a05697944107"),
    ("nerf2", "XPengZhao/NeRF2", "130b55ea3856aa2f5a42f9329cfa2e845c78e243"),
    ("neural_materials", "NVlabs/diff-rt-calibration", "cc70d813a5ac8cd976c0b43ca811712dd5a1629a"),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def acquire(workspace, specification):
    name, repository, commit = specification
    url = f"https://codeload.github.com/{repository}/zip/{commit}"
    archive_path = workspace / "artifacts/source_archives/paper_baselines" / f"{name}-{commit}.zip"
    if archive_path.is_file():
        blob = archive_path.read_bytes()
    else:
        request = Request(url, headers={"User-Agent": "dc01-paper-baseline-source-audit"})
        with urlopen(request, timeout=60) as response:
            blob = response.read(64 * 1024 * 1024 + 1)
        if len(blob) > 64 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large author archive: {name}")
        atomic_write(archive_path, blob)
    destination = workspace / "code/third_party/paper_baselines" / name
    files = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        prefixes = {PurePosixPath(item.filename).parts[0] for item in archive.infolist()}
        if len(prefixes) != 1:
            raise ValueError("Author archive must have a single enclosing directory")
        for item in archive.infolist():
            relative = PurePosixPath(*PurePosixPath(item.filename).parts[1:])
            if item.is_dir() or not relative.parts:
                continue
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("Unsafe archive member")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symbolic links are not accepted in source snapshots")
            output = destination.joinpath(*relative.parts).resolve()
            output.relative_to(destination.resolve())
            content = archive.read(item)
            if output.exists():
                if not output.is_file() or output.read_bytes() != content:
                    raise ValueError(f"Existing author snapshot differs: {name}/{relative}")
            else:
                atomic_write(output, content)
            files.append({"path": output.relative_to(workspace).as_posix(),
                          "bytes": len(content), "sha256": sha256(content)})
    license_files = [row["path"] for row in files if Path(row["path"]).name.lower().startswith("license")]
    if not license_files:
        raise ValueError(f"No license found for {name}")
    return {"name": name, "repository": repository, "commit": commit,
            "source_url": url, "archive": archive_path.relative_to(workspace).as_posix(),
            "archive_bytes": len(blob), "archive_sha256": sha256(blob),
            "license_files": license_files, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if not (workspace / "code/measured/README.md").is_file():
        raise ValueError("The selected workspace is not the measured-channel project")
    started = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        sources = list(executor.map(lambda spec: acquire(workspace, spec), SOURCES))
    report = {"status": "sources_verified_not_executed", "sources": sources,
              "elapsed_seconds": time.time() - started}
    report_path = workspace / "analysis/results/dc01_paper_baselines_single_frequency/source_manifest.json"
    atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
    for source in sources:
        print(json.dumps({"name": source["name"], "commit": source["commit"],
                          "files": len(source["files"]), "archive_bytes": source["archive_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
