"""Atomic cache and report helpers used by every long-running stage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from project_paths import PATHS


def portable_path(path: str | Path) -> str:
    """Serialize a workspace path relative to the relocatable project root."""

    relative = PATHS.relative(path)
    if Path(relative).is_absolute():
        return f"<external>/{Path(path).name}"
    return relative


def _portable_report_value(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, dict):
        return {key: _portable_report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_report_value(item) for item in value]
    if isinstance(value, str):
        root_prefix = f"{PATHS.root}{os.sep}"
        return value.replace(root_prefix, "")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_portable_report_value(payload), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_channel(path: Path, count: int) -> dict[str, Any]:
    value = np.load(path, mmap_mode="r")
    expected = (count, 256, 4, 192)
    if value.shape != expected or value.dtype != np.complex64:
        raise ValueError(
            f"invalid output {path}: {value.shape}/{value.dtype}, expected "
            f"{expected}/complex64"
        )
    for start in range(0, count, 8):
        block = np.asarray(value[start : start + 8])
        if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
            raise FloatingPointError(f"non-finite output block at row {start}")
    return {
        "path": portable_path(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": True,
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }
