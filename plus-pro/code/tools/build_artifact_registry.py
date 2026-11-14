#!/usr/bin/env python3
"""Build the logical artifact-name to canonical-path registry."""

from __future__ import annotations

import json
from pathlib import Path

from project_paths import PATHS


ROOTS = (
    PATHS.analysis / "results",
    PATHS.experiment_records,
    PATHS.versions,
    PATHS.code_root,
    PATHS.shared_features,
    PATHS.checkpoints,
    PATHS.artifacts / "training_runs",
    PATHS.version_cache,
)


def add(registry: dict[str, str], key: str, path: Path) -> None:
    relative = str(path.relative_to(PATHS.root))
    previous = registry.get(key)
    if previous is None or previous == relative:
        registry[key] = relative


def main() -> int:
    registry: dict[str, str] = {}
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_file():
                add(registry, path.name, path)

    for cache in sorted(PATHS.version_cache.glob("v*")):
        if cache.is_dir():
            number = int(cache.name[1:])
            add(registry, f"cache_v{number}", cache)

    for run_dir in sorted((PATHS.artifacts / "training_runs").glob("*")):
        if run_dir.is_dir():
            add(registry, run_dir.name, run_dir)

    output = PATHS.artifacts / "artifact_registry.json"
    output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"entries={len(registry)} output={output.relative_to(PATHS.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
