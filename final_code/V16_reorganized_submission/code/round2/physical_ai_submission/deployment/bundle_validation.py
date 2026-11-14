from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RuntimePaths
from ..data.official_data import load_official_data
from ..data.artifact_io import atomic_json, file_sha256, portable_path


MANIFEST_NAME = "deployment_manifest.json"


def _check_file(path: Path, specification: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = specification.get("bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError(
            f"invalid file size for {path }: {path .stat ().st_size }, "
            f"expected {expected_bytes }"
        )
    expected_sha256 = specification.get("sha256")
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path }")


def _check_npy(path: Path, specification: dict[str, Any]) -> None:
    _check_file(path, specification)
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(item) for item in specification["shape"])
    expected_dtype = np.dtype(specification["dtype"])
    if value.shape != expected_shape or value.dtype != expected_dtype:
        raise ValueError(
            f"invalid array {path }: {value .shape }/{value .dtype }, expected "
            f"{expected_shape }/{expected_dtype }"
        )


def _check_npz(path: Path, specification: dict[str, Any]) -> None:
    _check_file(path, specification)
    expected = specification["arrays"]
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(expected):
            raise ValueError(
                f"invalid NPZ fields for {path }: {sorted (loaded .files )}, "
                f"expected {sorted (expected )}"
            )
        for name, array_specification in expected.items():
            value = loaded[name]
            shape = tuple(int(item) for item in array_specification["shape"])
            dtype = np.dtype(array_specification["dtype"])
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"invalid {path }:{name }: {value .shape }/{value .dtype }, "
                    f"expected {shape }/{dtype }"
                )


def _asset_path(paths: RuntimePaths, specification: dict[str, Any]) -> Path:
    roots = {
        "work": paths.work_dir,
        "model": paths.model_dir,
    }
    root_name = specification["root"]
    if root_name not in roots:
        raise ValueError(f"unsupported manifest root: {root_name }")
    return roots[root_name] / specification["path"]


def validate_deployment_bundle(paths: RuntimePaths) -> dict[str, Any]:
    pass

    manifest_path = paths.work_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported deployment manifest: {manifest_path }")

    data = load_official_data(paths.data_dir)
    if data.setup.test_count != 500:
        raise ValueError(f"unexpected test count: {data .setup .test_count }")
    for name, specification in manifest["official_inputs"].items():
        _check_file(paths.data_dir / name, specification)

    validated_assets: list[dict[str, Any]] = []
    for specification in manifest["assets"]:
        path = _asset_path(paths, specification)
        kind = specification["kind"]
        if kind == "npy":
            _check_npy(path, specification)
        elif kind == "npz":
            _check_npz(path, specification)
        elif kind == "file":
            _check_file(path, specification)
        else:
            raise ValueError(f"unsupported manifest asset kind: {kind }")
        validated_assets.append(
            {
                "path": portable_path(path),
                "kind": kind,
                "bytes": int(path.stat().st_size),
                "sha256": specification.get("sha256"),
            }
        )

    report = {
        "stage": "deployment_validation",
        "status": "complete",
        "manifest": portable_path(manifest_path),
        "baseline": manifest["baseline"],
        "official_input_count": len(manifest["official_inputs"]),
        "assets": validated_assets,
    }
    atomic_json(paths.report_dir / "deployment_validation.json", report)
    return report
