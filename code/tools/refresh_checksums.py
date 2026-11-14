#!/usr/bin/env python3
"""Refresh checksum manifests for submission archives.

Default mode recursively regenerates both manifests from Round 1 and Round 2
ZIPs below versions/. For online feedback renames, use --archive with --replace
to update only one zip entry.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from project_paths import PATHS


CHUNK_SIZE = 8 * 1024 * 1024
VERSION_RE = re.compile(r"^[vV](\d+)(?:_|\.)")
MANIFESTS = {
    "sha256": "checksums.sha256",
    "md5": "checksums.md5",
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            value.update(chunk)
    return value.hexdigest()


def archive_key(path: Path) -> tuple[int, int, str]:
    round_rank = {"round1": 0, "round2": 1}.get(path.parent.name, 2)
    match = VERSION_RE.match(path.name)
    return (round_rank, int(match.group(1)) if match else 10**9, path.name)


def normalize_archive_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PATHS.root / path
    return path.resolve()


def relative_archive_path(path: Path) -> str:
    return str(path.relative_to(PATHS.root))


def read_manifest(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        try:
            checksum, archive = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"bad checksum line in {path}: {line!r}") from exc
        entries.append((checksum, archive))
    return entries


def write_manifest(path: Path, entries: list[tuple[str, str]]) -> None:
    entries = sorted(entries, key=lambda item: archive_key(Path(item[1])))
    path.write_text(
        "".join(f"{checksum}  {archive}\n" for checksum, archive in entries),
        encoding="ascii",
    )


def refresh_all() -> int:
    archives = sorted(PATHS.versions.rglob("*.zip"), key=archive_key)
    sha_lines = []
    md5_lines = []
    for archive in archives:
        relative = archive.relative_to(PATHS.root)
        sha_lines.append(f"{digest(archive, 'sha256')}  {relative}")
        md5_lines.append(f"{digest(archive, 'md5')}  {relative}")
        print(relative, flush=True)
    (PATHS.versions / MANIFESTS["sha256"]).write_text(
        "\n".join(sha_lines) + "\n", encoding="ascii"
    )
    (PATHS.versions / MANIFESTS["md5"]).write_text(
        "\n".join(md5_lines) + "\n", encoding="ascii"
    )
    print(f"archives={len(archives)}")
    return 0


def refresh_selected(archives: list[str], replacements: list[str]) -> int:
    archive_paths = [normalize_archive_path(value) for value in archives]
    replacement_rel = {
        relative_archive_path(normalize_archive_path(value)) for value in replacements
    }

    for archive in archive_paths:
        if not archive.exists():
            raise FileNotFoundError(archive)
        if archive.suffix != ".zip":
            raise ValueError(f"not a zip archive: {archive}")
        if PATHS.versions not in archive.parents and archive.parent != PATHS.versions:
            raise ValueError(f"archive must be under versions/: {archive}")

    for algorithm, manifest_name in MANIFESTS.items():
        manifest = PATHS.versions / manifest_name
        entries = [
            item for item in read_manifest(manifest) if item[1] not in replacement_rel
        ]
        by_archive = {archive: checksum for checksum, archive in entries}
        for archive in archive_paths:
            relative = relative_archive_path(archive)
            by_archive[relative] = digest(archive, algorithm)
        write_manifest(manifest, [(checksum, archive) for archive, checksum in by_archive.items()])

    for archive in archive_paths:
        print(relative_archive_path(archive), flush=True)
    print(f"updated={len(archive_paths)} replaced={len(replacement_rel)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="Zip archive to hash and update in both manifests. May be repeated.",
    )
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        help="Old manifest archive path to remove, useful after pending->online rename.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly regenerate manifests from every zip in versions/.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.archive:
        return refresh_selected(args.archive, args.replace)
    if args.replace:
        raise SystemExit("--replace requires at least one --archive")
    return refresh_all()


if __name__ == "__main__":
    raise SystemExit(main())
