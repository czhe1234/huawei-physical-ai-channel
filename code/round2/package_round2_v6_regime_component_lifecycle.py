#!/usr/bin/env python3
"""Package the validated non-Sionna regime/component lifecycle model as V6."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from data import load_positions
from package_round2_v4_calibrated_exact_projection import (
    MODEL_FILES as V4_MODEL_FILES,
    SOURCE_FILES as V4_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V6"
METHOD = "regime_component_lifecycle"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "regime_lifecycle_support"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_regime_lifecycle_support"
SOURCE_OUTPUT = ARTIFACT_DIR / "Round2_V6_Regime_Component_Lifecycle_Test_Channel.npy"
V4_OUTPUT = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "Round2_V4_Calibrated_Exact_Projection_Test_Channel.npy"
)
SOURCE_SELECTION = ARTIFACT_DIR / "v6_selection.json"
SOURCE_REGIMES = ARTIFACT_DIR / "regime_assignments.npz"
SOURCE_REPORT = ANALYSIS_DIR / "regime_component_lifecycle.json"
SUPPORT_REPORT = ANALYSIS_DIR / "support_stage_ablation.json"
GEODESIC_REPORT = ANALYSIS_DIR / "geodesic_corridor_features.json"
ROUTE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_route_topology"
    / "route_topology_features.json"
)
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)

V6_SOURCE_FILES = (
    (
        PATHS.round2_code / "package_round2_v4_calibrated_exact_projection.py",
        "code/round2/package_round2_v4_calibrated_exact_projection.py",
    ),
    (
        PATHS.round2_code / "build_route_topology_features.py",
        "code/round2/build_route_topology_features.py",
    ),
    (
        PATHS.round2_code / "build_geodesic_corridor_features.py",
        "code/round2/build_geodesic_corridor_features.py",
    ),
    (
        PATHS.round2_code / "run_v6_regime_component_lifecycle.py",
        "code/round2/run_v6_regime_component_lifecycle.py",
    ),
    (
        PATHS.round2_code / "package_round2_v6_regime_component_lifecycle.py",
        "code/round2/package_round2_v6_regime_component_lifecycle.py",
    ),
)

REPORT_FILES = (
    (SOURCE_SELECTION, "model/v6_selection.json"),
    (SOURCE_REPORT, "model/reports/regime_component_lifecycle.json"),
    (SUPPORT_REPORT, "model/reports/support_stage_ablation.json"),
    (GEODESIC_REPORT, "model/reports/geodesic_corridor_features.json"),
    (ROUTE_REPORT, "model/reports/route_topology_features.json"),
)


def _unique_files(files: tuple[tuple[object, str], ...]) -> tuple[tuple[object, str], ...]:
    output = []
    destinations: set[str] = set()
    for source, destination in files:
        if destination in destinations:
            continue
        destinations.add(destination)
        output.append((source, destination))
    return tuple(output)


SOURCE_FILES = _unique_files((*V4_SOURCE_FILES, *V6_SOURCE_FILES))
MODEL_FILES = _unique_files((*V4_MODEL_FILES, *REPORT_FILES))


def validate_output() -> dict[str, object]:
    setup = json.loads((PATHS.round2 / "Round2_Setup.json").read_text(encoding="utf-8"))
    expected = (
        int(setup["P_Test"]),
        int(setup["M"]),
        int(setup["N"]),
        int(setup["S"]),
    )
    output = np.load(SOURCE_OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 8):
        block = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    if output.shape != expected:
        raise ValueError(f"output shape {output.shape} != {expected}")
    if output.dtype != np.complex64:
        raise ValueError(f"output dtype {output.dtype} != complex64")
    if not finite:
        raise ValueError("output contains non-finite values")
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(SOURCE_OUTPUT.stat().st_size),
    }


def compare_to_v4() -> dict[str, object]:
    base = np.load(V4_OUTPUT, mmap_mode="r")
    output = np.load(SOURCE_OUTPUT, mmap_mode="r")
    if base.shape != output.shape:
        raise ValueError(f"V4/V6 shape mismatch: {base.shape} != {output.shape}")
    relative_change = np.zeros(len(base), dtype=np.float64)
    power_ratio = np.ones(len(base), dtype=np.float64)
    changed = np.zeros(len(base), dtype=bool)
    for start in range(0, len(base), 8):
        end = min(start + 8, len(base))
        old = np.asarray(base[start:end], dtype=np.complex64)
        new = np.asarray(output[start:end], dtype=np.complex64)
        old_power = np.sum(np.abs(old) ** 2, axis=(1, 2, 3), dtype=np.float64)
        new_power = np.sum(np.abs(new) ** 2, axis=(1, 2, 3), dtype=np.float64)
        delta_power = np.sum(np.abs(new - old) ** 2, axis=(1, 2, 3), dtype=np.float64)
        relative_change[start:end] = delta_power / np.maximum(old_power, 1e-30)
        power_ratio[start:end] = new_power / np.maximum(old_power, 1e-30)
        changed[start:end] = delta_power > np.maximum(old_power * 1e-12, 1e-30)

    regimes = np.load(SOURCE_REGIMES, allow_pickle=False)
    test_regime = regimes["test_regime"].astype(np.int8)
    _, _, _, _, test_site = load_positions()
    regime_names = ("clear", "light", "medium", "heavy")
    groups = []
    for site in (0, 1):
        for regime, name in enumerate(regime_names):
            rows = (test_site == site) & (test_regime == regime)
            groups.append(
                {
                    "site": site,
                    "regime": name,
                    "count": int(np.sum(rows)),
                    "changed_count": int(np.sum(changed[rows])),
                    "relative_change_mean": float(np.mean(relative_change[rows])),
                }
            )
    return {
        "changed_points": int(np.sum(changed)),
        "unchanged_points": int(np.sum(~changed)),
        "relative_change_mean": float(np.mean(relative_change)),
        "relative_change_median": float(np.median(relative_change)),
        "relative_change_p90": float(np.quantile(relative_change, 0.90)),
        "relative_change_max": float(np.max(relative_change)),
        "point_power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
        "groups": groups,
    }


def main() -> None:
    required = (SOURCE_OUTPUT, V4_OUTPUT, SOURCE_REGIMES)
    for source in (
        *required,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    ):
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    experiment = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    winner = selection["winner"]
    baseline = winner["baseline"]
    nested = winner["nested"]
    final_policy = winner["final_policy_oof"]
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiment": "Round2-E41",
        "sionna_used": False,
        "official_inputs_only": True,
        "primary_hypothesis": experiment["primary_hypothesis"],
        "validation": experiment["validation"],
        "local_validation": {
            "v4_replay": baseline,
            "nested_policy": nested,
            "nested_gain_vs_v4": float(nested["score"] - baseline["score"]),
            "nested_positive_fold_count": int(winner["positive_folds"]),
            "final_policy_oof": final_policy,
            "final_policy_gain_vs_v4": float(final_policy["score"] - baseline["score"]),
        },
        "deployment_policy": winner["final_policy"],
        "mechanism_chain": experiment["mechanism_chain"],
        "output_integrity": output_integrity,
        "output_delta_vs_v4": compare_to_v4(),
        "reproduction": (
            "Run the V4 reproduction pipeline; run build_route_topology_features.py and "
            "build_geodesic_corridor_features.py; then run "
            "run_v6_regime_component_lifecycle.py --phase all."
        ),
    }

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {ARCHIVE}")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(PATHS.root / ".hw-project", ".hw-project")
        archive.write(SOURCE_OUTPUT, "Round2_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.writestr("model/reports/", b"")
        for source, destination in MODEL_FILES:
            archive.write(source, destination)
        archive.writestr(
            "model/V6_VERSION.json",
            json.dumps(package_metadata, indent=2, ensure_ascii=False).encode("utf-8")
            + b"\n",
        )
        archive.writestr(
            "model/INFERENCE.txt",
            (package_metadata["reproduction"] + "\n").encode("ascii"),
        )
        for source, destination in SOURCE_FILES:
            archive.write(source, destination)

    expected_members = {
        "Round2_Test_Channel.npy",
        "model/full_epoch6.pt",
        "model/test_corrected_weights.npz",
        "model/full_group_calibration_model.npz",
        "model/calibrated_exact_projection_selection.json",
        "model/v6_selection.json",
        "model/V6_VERSION.json",
        "code/round2/build_route_topology_features.py",
        "code/round2/build_geodesic_corridor_features.py",
        "code/round2/run_v6_regime_component_lifecycle.py",
    }
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
        compression_types = {member.compress_type for member in archive.infolist()}
        missing = sorted(expected_members.difference(members))
        with archive.open("Round2_Test_Channel.npy") as handle:
            version = np.lib.format.read_magic(handle)
            packaged = np.lib.format._read_array_header(handle, version)
    if bad_member is not None:
        raise RuntimeError(f"ZIP CRC failure: {bad_member}")
    if missing:
        raise RuntimeError(f"ZIP is missing required members: {missing}")
    if members.count("Round2_Test_Channel.npy") != 1:
        raise RuntimeError("ZIP must contain one Round2_Test_Channel.npy at root")
    if compression_types != {zipfile.ZIP_STORED}:
        raise RuntimeError(f"unexpected ZIP compression types: {compression_types}")
    packaged_shape, _, packaged_dtype = packaged
    if tuple(packaged_shape) != tuple(output_integrity["shape"]):
        raise RuntimeError(f"packaged NPY shape mismatch: {packaged_shape}")
    if np.dtype(packaged_dtype) != np.dtype(np.complex64):
        raise RuntimeError(f"packaged NPY dtype mismatch: {packaged_dtype}")

    package_metadata["zip_integrity"] = {
        "crc": True,
        "compression": "ZIP_STORED",
        "bytes": int(ARCHIVE.stat().st_size),
        "member_count": len(members),
        "required_members_present": True,
        "members": members,
    }
    VERSION_REPORT.write_text(
        json.dumps(package_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(package_metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
