#!/usr/bin/env python3
"""Package the official-data-only complex carrier and lifecycle model as V7."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from package_round2_v6_regime_component_lifecycle import (
    MODEL_FILES as V6_MODEL_FILES,
    SOURCE_FILES as V6_SOURCE_FILES,
)
from project_paths import PATHS


VERSION = "V7"
METHOD = "complex_carrier_lifecycle"
ARCHIVE = PATHS.round2_versions / f"{VERSION}_{METHOD}_candidate_pending.zip"
STACK_DIR = PATHS.artifacts / "round2" / "complex_carrier_stacking"
OVERLAY_DIR = PATHS.artifacts / "round2" / "complex_carrier_lifecycle_overlay"
SOURCE_OUTPUT = OVERLAY_DIR / "Round2_V7_Stacked_Lifecycle_Test_Channel.npy"
SOURCE_WEIGHTS = STACK_DIR / "full_complex_stack_weights.npz"
STACK_SELECTION = STACK_DIR / "complex_stack_selection.json"
STACK_DEPLOYMENT = STACK_DIR / "complex_carrier_deployment.json"
OVERLAY_SELECTION = OVERLAY_DIR / "stacked_lifecycle_selection.json"
STACK_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_complex_carrier_stacking"
    / "complex_carrier_stacking.json"
)
PROJECTION_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_complex_carrier_stacking"
    / "complex_carrier_exact_projection.json"
)
OVERLAY_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_complex_carrier_lifecycle_overlay"
    / "stacked_lifecycle_overlay.json"
)
CARRIER_METADATA = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_bank"
    / "complex_carrier_bank_metadata.json"
)
V4_OUTPUT = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "Round2_V4_Calibrated_Exact_Projection_Test_Channel.npy"
)
V6_OUTPUT = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "Round2_V6_Regime_Component_Lifecycle_Test_Channel.npy"
)
VERSION_REPORT = (
    PATHS.analysis
    / "results"
    / "version_reports"
    / VERSION
    / f"{VERSION}_{METHOD}.json"
)


def _unique_files(files):
    output = []
    destinations: set[str] = set()
    for source, destination in files:
        if destination in destinations:
            continue
        destinations.add(destination)
        output.append((source, destination))
    return tuple(output)


V7_SOURCE_FILES = (
    (
        PATHS.round2_code / "build_v7_complex_carrier_bank.py",
        "code/round2/build_v7_complex_carrier_bank.py",
    ),
    (
        PATHS.round2_code / "analyze_v7_complex_carrier_stacking.py",
        "code/round2/analyze_v7_complex_carrier_stacking.py",
    ),
    (
        PATHS.round2_code / "run_v7_complex_carrier_exact_projection.py",
        "code/round2/run_v7_complex_carrier_exact_projection.py",
    ),
    (
        PATHS.round2_code / "analyze_v7_stacked_lifecycle_overlay.py",
        "code/round2/analyze_v7_stacked_lifecycle_overlay.py",
    ),
    (
        PATHS.round2_code / "package_round2_v7_complex_carrier_lifecycle.py",
        "code/round2/package_round2_v7_complex_carrier_lifecycle.py",
    ),
)

REPORT_FILES = (
    (SOURCE_WEIGHTS, "model/full_complex_stack_weights.npz"),
    (STACK_SELECTION, "model/complex_stack_selection.json"),
    (STACK_DEPLOYMENT, "model/complex_carrier_deployment.json"),
    (OVERLAY_SELECTION, "model/stacked_lifecycle_selection.json"),
    (CARRIER_METADATA, "model/reports/complex_carrier_bank_metadata.json"),
    (STACK_REPORT, "model/reports/complex_carrier_stacking.json"),
    (PROJECTION_REPORT, "model/reports/complex_carrier_exact_projection.json"),
    (OVERLAY_REPORT, "model/reports/stacked_lifecycle_overlay.json"),
)

SOURCE_FILES = _unique_files((*V6_SOURCE_FILES, *V7_SOURCE_FILES))
MODEL_FILES = _unique_files((*V6_MODEL_FILES, *REPORT_FILES))


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


def compare_output(reference_path) -> dict[str, float | int]:
    reference = np.load(reference_path, mmap_mode="r")
    output = np.load(SOURCE_OUTPUT, mmap_mode="r")
    relative_change = np.empty(len(output), dtype=np.float64)
    power_ratio = np.empty(len(output), dtype=np.float64)
    changed = np.zeros(len(output), dtype=bool)
    for start in range(0, len(output), 8):
        end = min(start + 8, len(output))
        old = np.asarray(reference[start:end], dtype=np.complex64)
        new = np.asarray(output[start:end], dtype=np.complex64)
        old_power = np.sum(np.abs(old) ** 2, axis=(1, 2, 3), dtype=np.float64)
        new_power = np.sum(np.abs(new) ** 2, axis=(1, 2, 3), dtype=np.float64)
        delta = np.sum(np.abs(new - old) ** 2, axis=(1, 2, 3), dtype=np.float64)
        relative_change[start:end] = delta / np.maximum(old_power, 1e-30)
        power_ratio[start:end] = new_power / np.maximum(old_power, 1e-30)
        changed[start:end] = delta > np.maximum(old_power * 1e-12, 1e-30)
    return {
        "changed_points": int(np.sum(changed)),
        "relative_change_mean": float(np.mean(relative_change)),
        "relative_change_median": float(np.median(relative_change)),
        "relative_change_p90": float(np.quantile(relative_change, 0.90)),
        "relative_change_max": float(np.max(relative_change)),
        "power_ratio_mean": float(np.mean(power_ratio)),
        "power_ratio_median": float(np.median(power_ratio)),
    }


def main() -> None:
    for source in (
        SOURCE_OUTPUT,
        V4_OUTPUT,
        V6_OUTPUT,
        *[path for path, _ in MODEL_FILES],
        *[path for path, _ in SOURCE_FILES],
    ):
        if not source.is_file():
            raise FileNotFoundError(source)

    output_integrity = validate_output()
    projection = json.loads(PROJECTION_REPORT.read_text(encoding="utf-8"))
    overlay = json.loads(OVERLAY_REPORT.read_text(encoding="utf-8"))
    winner = overlay["selection"]["winner"]
    if winner is None:
        raise RuntimeError("lifecycle overlay did not pass deployment criteria")
    baseline = winner["baseline"]
    nested = winner["nested"]
    final_policy = winner["final_policy_oof"]
    v4_nested = projection["v4_reference"]["nested"]
    package_metadata = {
        "version": VERSION,
        "method": METHOD,
        "status": "candidate_pending",
        "archive": PATHS.relative(ARCHIVE),
        "online_score": None,
        "source_experiments": ["Round2-E44", "Round2-E45", "Round2-E46", "Round2-E55"],
        "sionna_used": False,
        "official_inputs_only": True,
        "primary_hypothesis": (
            "Complementary official-data complex carriers improve channel coherence before "
            "the validated V4 spectral projection, while the V6 geodesic lifecycle action "
            "adds bounded path-support information afterward."
        ),
        "local_validation": {
            "v4_reference": v4_nested,
            "stacked_projection": baseline,
            "stacked_projection_gain_vs_v4": float(
                baseline["score"] - v4_nested["score"]
            ),
            "nested_lifecycle": nested,
            "nested_lifecycle_gain_vs_stack": float(
                nested["score"] - baseline["score"]
            ),
            "nested_lifecycle_gain_vs_v4": float(
                nested["score"] - v4_nested["score"]
            ),
            "positive_fold_count": int(winner["positive_folds"]),
            "final_policy_oof": final_policy,
        },
        "stack_config": projection["stack_config"],
        "deployment_policy": winner["final_policy"],
        "output_integrity": output_integrity,
        "output_delta_vs_v4": compare_output(V4_OUTPUT),
        "output_delta_vs_v6": compare_output(V6_OUTPUT),
        "reproduction": (
            "Run the V4 and V6 official-data pipelines; build the 17-carrier OOF/test bank; "
            "fit site/port8 complex ridge weights; replay the V4 exact projection; then run "
            "the stacked lifecycle overlay. No Sionna input is read."
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
            "model/V7_VERSION.json",
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
        "model/full_group_calibration_model.npz",
        "model/full_complex_stack_weights.npz",
        "model/stacked_lifecycle_selection.json",
        "model/V7_VERSION.json",
        "code/round2/build_v7_complex_carrier_bank.py",
        "code/round2/run_v7_complex_carrier_exact_projection.py",
        "code/round2/analyze_v7_stacked_lifecycle_overlay.py",
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
