#!/usr/bin/env python3
"""Reject machine-specific and retired project paths in maintained text."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKIP = {Path(__file__).resolve()}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".yml", ".yaml", ".toml", ".txt"}
PATTERNS = (
    ("home absolute path", re.compile(r"/home/[A-Za-z0-9_.-]+/")),
    ("root absolute path", re.compile(r"/root/")),
    ("installation-prefix path", re.compile(r"/opt/(?:anaconda|miniconda)[^\s]*")),
    ("Windows absolute path", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")),
    ("retired solution path", re.compile(r"(?<![A-Za-z0-9_])round1_solution(?:/|[\"'])")),
    ("retired data path", re.compile(r"(?<![A-Za-z0-9_])round1_data(?:/|[\"'])")),
    ("retired submission path", re.compile(r"(?<![A-Za-z0-9_])submissions/")),
    ("retired probe path", re.compile(r"(?<![A-Za-z0-9_])probe_work(?:/|[\"'])")),
    ("retired per-version directory", re.compile(r"versions/v\d+/")),
    ("external Codex session dependency", re.compile(r"CODEX_HOME/(?:sessions|backups)")),
    ("manual root concatenation", re.compile(r"(?:BASE_DIR|PROJECT_ROOT).*?[\"']/")),
)
LEGACY_ROOT_ENTRIES = (
    "round1_solution",
    "round1_data",
    "visualizations",
    "probe_work",
    "submissions",
    "project_paths.py",
    "Round1_Map.zip",
    "Round1_Test_Channel.npy",
    "赛题任务书.pdf",
)


def main() -> int:
    failures: list[str] = []
    for entry in LEGACY_ROOT_ENTRIES:
        path = PROJECT_ROOT / entry
        if path.exists() or path.is_symlink():
            failures.append(f"legacy root entry still exists: {entry}")

    record_docs = sorted((PROJECT_ROOT / "experiment_records").glob("*.md"))
    expected_record = PROJECT_ROOT / "experiment_records" / "EXPERIMENT_RECORD.md"
    if record_docs != [expected_record]:
        names = ", ".join(path.name for path in record_docs) or "none"
        failures.append(f"experiment_records must contain only EXPERIMENT_RECORD.md; found: {names}")

    allowed_version_metadata = {
        "checksums.sha256",
        "checksums.md5",
    }
    for path in sorted((PROJECT_ROOT / "versions").iterdir()):
        if path.is_dir():
            if path.name not in {"round1", "round2"}:
                failures.append(f"versions contains an unsupported directory: {path.name}")
                continue
            for archive in sorted(path.iterdir()):
                if not archive.is_file() or archive.suffix != ".zip":
                    failures.append(
                        f"versions/{path.name} may contain only zip files: {archive.name}"
                    )
        elif path.suffix != ".zip" and path.name not in allowed_version_metadata:
            failures.append(f"versions may contain only zip/checksum files: {path.name}")

    for path in sorted(PROJECT_ROOT.rglob("*")):
        if path.is_symlink():
            failures.append(f"symbolic links are not allowed: {path.relative_to(PROJECT_ROOT)}")
            continue
        if path in SKIP or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    relative = path.relative_to(PROJECT_ROOT)
                    failures.append(f"{relative}:{line_number}: {label}: {line.strip()}")
                    break
    if failures:
        print("Forbidden paths or layout entries found:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("No machine-specific paths, retired paths, or compatibility entries found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
