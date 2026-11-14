#!/usr/bin/env python3
"""Build a self-contained task archive for inverse Sionna calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from powerfield_common import regime_of
from project_paths import PATHS


TASK_NAME = "sionna_official_inverse_calibration_task"
SOURCE_DIR = PATHS.code / "task_packages" / "sionna_official_inverse_calibration"
OUTPUT_PARENT = PATHS.artifacts / "portable_tasks"
STAGING_DIR = OUTPUT_PARENT / TASK_NAME
ARCHIVE_PATH = OUTPUT_PARENT / f"{TASK_NAME}_20260718.zip"
CHANNEL_ROWS_PER_SPLIT = 96
TILE_SIZE_M = 55.0
REGIME_NAMES = ("clear", "light", "medium", "heavy")
REGIME_ALLOCATION = np.asarray([20, 7, 14, 55], dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_directory(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"already exists; pass --force to rebuild: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def assign_spatial_tiles(
    tile_id: np.ndarray, regime_code: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    tiles = np.unique(tile_id[valid])
    tile_counts = np.zeros((len(tiles), 4), dtype=np.int32)
    tile_totals = np.zeros(len(tiles), dtype=np.int32)
    for index, tile in enumerate(tiles):
        mask = valid & (tile_id == tile)
        tile_totals[index] = int(np.sum(mask))
        for regime in range(4):
            tile_counts[index, regime] = int(np.sum(mask & (regime_code == regime)))

    total_counts = np.sum(tile_counts, axis=0)
    target_counts = total_counts / 3.0
    target_total = float(np.sum(tile_totals)) / 3.0
    rng = np.random.default_rng(206347)
    best: tuple[float, np.ndarray] | None = None
    for _ in range(250_000):
        assignment = rng.integers(0, 3, size=len(tiles), dtype=np.int8)
        if len(np.unique(assignment)) != 3:
            continue
        grouped = np.stack(
            [np.sum(tile_counts[assignment == group], axis=0) for group in range(3)]
        )
        if np.any(grouped < REGIME_ALLOCATION[None, :]):
            continue
        grouped_total = np.asarray(
            [np.sum(tile_totals[assignment == group]) for group in range(3)], dtype=np.float64
        )
        objective = float(
            np.sum(((grouped - target_counts[None, :]) / (target_counts[None, :] + 5.0)) ** 2)
            + 0.3 * np.sum(((grouped_total - target_total) / target_total) ** 2)
        )
        if best is None or objective < best[0]:
            best = (objective, assignment.copy())
    if best is None:
        raise RuntimeError("could not create three regime-balanced disjoint tile groups")
    assignment = best[1]
    output = np.full(len(tile_id), -1, dtype=np.int8)
    for tile, group in zip(tiles, assignment):
        output[tile_id == tile] = group
    return output


def select_channel_rows(
    tile_group: np.ndarray, regime_code: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(347206)
    selected: list[int] = []
    split_codes: list[int] = []
    for group in range(3):
        group_rows: list[int] = []
        for regime, count in enumerate(REGIME_ALLOCATION):
            candidates = np.flatnonzero(
                valid & (tile_group == group) & (regime_code == regime)
            )
            if len(candidates) < count:
                raise RuntimeError(
                    f"group {group} regime {regime} has {len(candidates)} rows, needs {count}"
                )
            group_rows.extend(rng.choice(candidates, size=int(count), replace=False).tolist())
        group_array = np.sort(np.asarray(group_rows, dtype=np.int32))
        if len(group_array) != CHANNEL_ROWS_PER_SPLIT:
            raise AssertionError(len(group_array))
        selected.extend(group_array.tolist())
        split_codes.extend([group] * len(group_array))
    return np.asarray(selected, dtype=np.int32), np.asarray(split_codes, dtype=np.int8)


def copy_project_inputs(root: Path) -> None:
    (root / "code").mkdir(parents=True)
    (root / "datasets" / "round1").mkdir(parents=True)
    (root / "datasets" / "task_documents").mkdir(parents=True)
    (root / "artifacts" / "task_inputs").mkdir(parents=True)
    (root / "artifacts" / "prior_evidence").mkdir(parents=True)
    (root / "analysis" / "task_results").mkdir(parents=True)

    shutil.copy2(SOURCE_DIR / "AGENTS.md", root / "AGENTS.md")
    shutil.copy2(SOURCE_DIR / "environment.sionna.yml", root / "environment.sionna.yml")
    shutil.copy2(SOURCE_DIR / "task_config.json", root / "artifacts" / "task_inputs" / "task_config.json")
    shutil.copy2(PATHS.root / ".hw-project", root / ".hw-project")
    for name in (
        "project_paths.py",
        "sionna_calibration_scoring.py",
        "validate_sionna_calibration_task.py",
        "run_sionna_calibration_smoke.py",
    ):
        shutil.copy2(PATHS.code / name, root / "code" / name)

    reference_dir = root / "code" / "reference_history"
    reference_dir.mkdir()
    for name in (
        "analyze_v206_sionna_config_search.py",
        "analyze_v206_official_system_identification.py",
        "analyze_v206_noise_mechanism_search.py",
        "analyze_v325_path_phase_transport_identifiability.py",
    ):
        shutil.copy2(PATHS.code / name, reference_dir / name)

    for name in (
        "Round1_Train_Pos.npy",
        "Round1_Test_Pos.npy",
        "Round1_Setup.json",
        "Round1_Map.ply",
    ):
        shutil.copy2(PATHS.round1 / name, root / "datasets" / "round1" / name)
    for name in ("task_book.txt", "赛题任务书.pdf"):
        shutil.copy2(
            PATHS.datasets / "task_documents" / name,
            root / "datasets" / "task_documents" / name,
        )

    scene_source = PATHS.artifacts / "external" / "restored_sionna_official_etoile"
    scene_target = root / "artifacts" / "external" / "restored_sionna_official_etoile"
    shutil.copytree(scene_source, scene_target)
    shutil.copy2(
        PATHS.shared_features / "height_geometry_cache.npz",
        root / "artifacts" / "task_inputs" / "height_geometry_cache.npz",
    )
    shutil.copy2(
        PATHS.shared_features / "train_total_power.npy",
        root / "artifacts" / "task_inputs" / "train_total_power.npy",
    )

    evidence = {
        PATHS.version_report_dir(206) / "v206_official_system_identification.json": "v206_official_system_identification.json",
        PATHS.version_report_dir(206) / "v206_inverse_sionna_calibration_summary.json": "v206_inverse_sionna_calibration_summary.json",
        PATHS.version_report_dir(206) / "v206_noise_mechanism_search.json": "v206_noise_mechanism_search.json",
        PATHS.version_report_dir(325) / "v325_path_phase_transport_identifiability.json": "v325_path_phase_transport_identifiability.json",
        PATHS.cache_dir(305) / "v305_train_component_patch_matches.npz": "v305_train_component_patch_matches.npz",
        PATHS.cache_dir(325) / "v325_component_center_gains.npz": "v325_component_center_gains.npz",
    }
    for source, name in evidence.items():
        shutil.copy2(source, root / "artifacts" / "prior_evidence" / name)


def build_official_subset(root: Path) -> dict[str, object]:
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    labels = regime_of(blocked).astype(str)
    regime_code = np.asarray([REGIME_NAMES.index(value) for value in labels], dtype=np.int8)
    power = np.load(PATHS.shared_features / "train_total_power.npy").astype(np.float64)
    valid = power > 0.0
    xy = positions[:, :2]
    tile = np.floor((xy - np.min(xy, axis=0, keepdims=True)) / TILE_SIZE_M).astype(np.int32)
    y_count = int(np.max(tile[:, 1])) + 1
    tile_id = tile[:, 0] * y_count + tile[:, 1]
    tile_group = assign_spatial_tiles(tile_id, regime_code, valid)
    selected, split_code = select_channel_rows(tile_group, regime_code, valid)

    source = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    target_path = root / "artifacts" / "task_inputs" / "official_channel_subset.npy"
    target = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(selected), 256, 4, 192),
    )
    for start in range(0, len(selected), 8):
        end = min(start + 8, len(selected))
        target[start:end] = np.asarray(source[selected[start:end]], dtype=np.complex64)
        target.flush()

    metadata_path = root / "artifacts" / "task_inputs" / "official_subset_metadata.npz"
    np.savez_compressed(
        metadata_path,
        official_indices=selected,
        split_code=split_code,
        regime_code=regime_code[selected],
        positions=positions[selected],
        blocked=blocked[selected],
        total_power=power[selected],
        tile_id=tile_id[selected],
        calibration_tile_ids=np.unique(tile_id[tile_group == 0]),
        validation_tile_ids=np.unique(tile_id[tile_group == 1]),
        audit_tile_ids=np.unique(tile_id[tile_group == 2]),
        exact_zero_indices=np.flatnonzero(~valid).astype(np.int32),
        tile_size_m=np.asarray(TILE_SIZE_M, dtype=np.float32),
    )
    return {
        "official_train_count": int(len(positions)),
        "subset_count": int(len(selected)),
        "split_counts": {
            name: int(np.sum(split_code == code))
            for code, name in enumerate(("calibration", "validation", "audit"))
        },
        "regime_counts_by_split": {
            name: {
                regime: int(np.sum((split_code == code) & (regime_code[selected] == regime_index)))
                for regime_index, regime in enumerate(REGIME_NAMES)
            }
            for code, name in enumerate(("calibration", "validation", "audit"))
        },
        "tile_ids_by_split": {
            name: np.unique(tile_id[selected][split_code == code]).astype(int).tolist()
            for code, name in enumerate(("calibration", "validation", "audit"))
        },
        "exact_zero_count_excluded": int(np.sum(~valid)),
    }


def write_manifest(root: Path, subset_summary: dict[str, object]) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        files.append(
            {"path": relative.as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest = {
        "task_name": TASK_NAME,
        "created_date": "2026-07-18",
        "purpose": "portable analysis-only inverse calibration of official-equivalent Sionna RT settings",
        "subset": subset_summary,
        "files": files,
    }
    manifest_path = root / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    checksum_rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        checksum_rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")


def build_archive(root: Path, archive: Path, force: bool) -> None:
    if archive.exists():
        if not force:
            raise FileExistsError(f"already exists; pass --force to rebuild: {archive}")
        archive.unlink()
    with zipfile.ZipFile(
        archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as handle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            handle.write(path, (Path(TASK_NAME) / path.relative_to(root)).as_posix())
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad is not None:
            raise RuntimeError(f"archive CRC failed: {bad}")
    archive_hash = sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    prepare_directory(STAGING_DIR, args.force)
    copy_project_inputs(STAGING_DIR)
    subset_summary = build_official_subset(STAGING_DIR)
    write_manifest(STAGING_DIR, subset_summary)
    build_archive(STAGING_DIR, ARCHIVE_PATH, args.force)
    report = {
        "task_directory": PATHS.relative(STAGING_DIR),
        "archive": PATHS.relative(ARCHIVE_PATH),
        "archive_size": ARCHIVE_PATH.stat().st_size,
        "archive_sha256": sha256(ARCHIVE_PATH),
        "subset": subset_summary,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

