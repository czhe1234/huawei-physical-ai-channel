#!/usr/bin/env python3
"""Validate the portable Sionna inverse-calibration task package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from project_paths import PATHS


INPUT_DIR = PATHS.artifacts / "task_inputs"
SCENE_DIR = PATHS.artifacts / "external" / "restored_sionna_official_etoile"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = (
        PATHS.root / ".hw-project",
        PATHS.root / "AGENTS.md",
        INPUT_DIR / "task_config.json",
        INPUT_DIR / "official_channel_subset.npy",
        INPUT_DIR / "official_subset_metadata.npz",
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
        PATHS.round1 / "Round1_Setup.json",
        PATHS.round1 / "Round1_Map.ply",
        SCENE_DIR / "etoile.xml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    channels = np.load(INPUT_DIR / "official_channel_subset.npy", mmap_mode="r")
    metadata = np.load(INPUT_DIR / "official_subset_metadata.npz")
    if channels.shape != (288, 256, 4, 192) or channels.dtype != np.complex64:
        raise ValueError(f"unexpected official subset: {channels.shape} {channels.dtype}")
    if not np.isfinite(channels.real).all() or not np.isfinite(channels.imag).all():
        raise FloatingPointError("official subset contains non-finite values")
    for code in (0, 1, 2):
        if int(np.sum(metadata["split_code"] == code)) != 96:
            raise ValueError(f"split {code} does not contain 96 points")
    tile_sets = [set(metadata["tile_id"][metadata["split_code"] == code].tolist()) for code in (0, 1, 2)]
    if any(tile_sets[left] & tile_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("spatial tile sets overlap")
    if np.any(metadata["total_power"] <= 0.0):
        raise ValueError("exact-zero points must not be included in the calibration subset")

    scene_xml = SCENE_DIR / "etoile.xml"
    tree = ET.parse(scene_xml)
    referenced = []
    for item in tree.findall(".//string[@name='filename']"):
        referenced.append(SCENE_DIR / item.attrib["value"])
    missing_scene = [str(path) for path in referenced if not path.is_file()]
    if missing_scene:
        raise FileNotFoundError(f"scene references missing meshes: {missing_scene[:10]}")

    manifest_path = PATHS.root / "package_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            path = PATHS.root / row["path"]
            if not path.is_file() or path.stat().st_size != row["size"]:
                raise ValueError(f"manifest mismatch: {row['path']}")
            if sha256(path) != row["sha256"]:
                raise ValueError(f"sha256 mismatch: {row['path']}")

    summary = {
        "status": "pass",
        "channel_shape": list(channels.shape),
        "channel_dtype": str(channels.dtype),
        "split_counts": {
            str(code): int(np.sum(metadata["split_code"] == code)) for code in (0, 1, 2)
        },
        "split_tile_counts": {str(code): len(tile_sets[code]) for code in (0, 1, 2)},
        "scene_mesh_count": len(referenced),
        "manifest_checked": manifest_path.is_file(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

