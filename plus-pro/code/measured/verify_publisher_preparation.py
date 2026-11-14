"""Independently check publisher-calibrated dc01/dc02 preparations on CPU.

This module imports no training or evaluation code. It shares only the existing
TF Example decoder/CRC implementation; calibration, panel mapping, selection,
coordinates and spatial splits are checked here without importing the preparer.
Raw files are sampled, not re-hashed: their full-file identity is recorded by
the preparation command. All prepared channels are checked for finite values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from project_paths import PATHS
from dc01_data import example, masked_crc

SCHEMA = "dichasus-publisher-preparation-v1"
LEGACY_KEYS = (
    "source_id", "position", "original_position", "time", "group",
    "panel_position", "panel_direction", "frequency_indices", "origin", "rotation",
)
SAMPLE_QUANTILES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_input(relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts,
            "Manifest input path must be project-relative")
    resolved = (PATHS.root / path).resolve()
    require(resolved.is_relative_to(PATHS.root.resolve()),
            "Manifest input escapes the project root")
    require(resolved.is_file(), "A manifest input file is missing")
    return resolved


def exact_equal(left, right, name):
    left, right = np.asarray(left), np.asarray(right)
    require(left.dtype == right.dtype and left.shape == right.shape
            and np.array_equal(left, right), name + " is not exactly preserved")


def reference_coordinates(csv_path):
    """Construct the horizontal frame using orthogonal unit basis vectors."""
    with Path(csv_path).open() as stream:
        records = {row["Name"]: row for row in csv.DictReader(stream)}
    def xyz(name):
        return np.array([float(records[name][k])
                         for k in ("West", "South", "Height")], dtype=np.float64)
    origin = xyz("AU")
    north = xyz("NWU") - origin
    length = float(np.linalg.norm(north[:2]))
    require(length > 0, "Degenerate coordinate reference")
    east_basis = np.array([north[1] / length, -north[0] / length, 0.0])
    north_basis = np.array([north[0] / length, north[1] / length, 0.0])
    return origin, np.stack([east_basis, north_basis, [0.0, 0.0, 1.0]])


def reference_selection(positions, voxel_m, maximum_points, seed):
    """Choose the nearest raw position to each cell center, breaking ties by ID."""
    positions = np.asarray(positions, dtype=np.float64)
    require(positions.ndim == 2 and positions.shape[1] == 3
            and np.isfinite(positions).all(), "Invalid all-frame positions")
    cells = np.floor(positions[:, :2] / voxel_m).astype(np.int64)
    best = {}
    for source_id, (cell, position) in enumerate(zip(cells, positions)):
        key = tuple(int(v) for v in cell)
        center = (cell + 0.5) * voxel_m
        distance = float(np.sum((position[:2] - center) ** 2))
        candidate = (distance, source_id)
        if key not in best or candidate < best[key]:
            best[key] = candidate
    keys = sorted(best)
    group_lookup = {key: i for i, key in enumerate(keys)}
    selected = np.array([best[key][1] for key in keys], dtype=np.int64)
    if len(selected) > maximum_points:
        selected = np.random.default_rng(seed).choice(
            selected, maximum_points, replace=False)
    selected.sort()
    groups = np.array([group_lookup[tuple(cells[i])] for i in selected],
                      dtype=np.int64)
    return selected, groups, len(keys)


def minimum_distances(query, reference):
    query, reference = np.asarray(query), np.asarray(reference)
    require(len(reference) > 0, "Empty reference set for spatial guard")
    answer = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 128):
        delta = query[start:start + 128, None, :2] - reference[None, :, :2]
        answer[start:start + 128] = np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1)
    return answer


def verify_split(position, protocol, block_m, guard_m):
    """Rebuild fold membership and the protection band without the old splitter."""
    position = np.asarray(position, dtype=np.float64)
    n = len(position)
    tiles = np.floor(position[:, :2] / block_m).astype(np.int64)
    fold = (tiles[:, 0] + 2 * tiles[:, 1]) % 5
    require(protocol["block_m"] == block_m and protocol["guard_m"] == guard_m,
            "Split parameters differ from the preparation contract")
    require(np.array_equal(protocol["fold"], fold), "Spatial fold IDs differ")
    require(len(protocol["splits"]) == 5, "Expected five outer folds")
    digest = hashlib.sha256(json.dumps(
        protocol["splits"], sort_keys=True).encode()).hexdigest()
    require(protocol["sha256"] == digest, "Split internal digest is invalid")
    coverage = np.zeros(n, dtype=np.int64)
    summary = []
    for outer, actual in enumerate(protocol["splits"]):
        require(actual["outer"] == outer, "Unexpected outer fold order")
        test = np.flatnonzero(fold == outer)
        val = np.flatnonzero(fold == (outer + 1) % 5)
        possible = np.flatnonzero((fold != outer) & (fold != (outer + 1) % 5))
        held = np.concatenate([test, val])
        distance = minimum_distances(position[possible], position[held])
        fit = possible[distance >= guard_m]
        for name, expected in (("fit", fit), ("val", val), ("test", test)):
            require(np.array_equal(np.asarray(actual[name], dtype=np.int64), expected),
                    "Incorrect " + name + " membership in fold " + str(outer))
            require(len(np.unique(expected)) == len(expected), "Duplicate split entries")
        require(min(len(fit), len(val), len(test)) >= 16,
                "Insufficient points for the existing five-fold contract")
        require(not (set(fit) & set(held)), "Fit/held overlap")
        guard = minimum_distances(position[fit], position[held])
        require(bool(np.all(guard >= guard_m)), "Spatial protection band violated")
        nearest = minimum_distances(position[test], position[fit])
        require(np.allclose(actual["min_test_fit_distance"], nearest.min(),
                            rtol=1e-12, atol=1e-12), "Incorrect nearest-fit minimum")
        require(np.allclose(actual["test_fit_distance_quantiles"],
                            np.quantile(nearest, [0, .25, .5, .75, 1]),
                            rtol=1e-12, atol=1e-12), "Incorrect nearest-fit quantiles")
        coverage[test] += 1
        summary.append(dict(outer=outer, fit=len(fit), val=len(val), test=len(test),
                            min_fit_held_distance_m=float(guard.min())))
    require(bool(np.all(coverage == 1)), "Test coverage is not exactly once")
    return dict(test_once=True, physical_panels_bound_to_position=True,
                guard_m=guard_m, folds=summary)


def reference_calibration(csi, offsets, assignments, frequency_index):
    """Compute each raw-port/frequency value explicitly in complex128."""
    csi = np.asarray(csi)
    require(csi.shape == (64, 1024, 2), "Unexpected raw CSI shape")
    assignments = np.asarray(assignments)
    require(assignments.shape == (2, 4, 8), "Unexpected assignment shape")
    require(np.array_equal(np.sort(assignments.ravel()), np.arange(64)),
            "Antenna assignments do not cover each raw port exactly once")
    frequency_index = np.asarray(frequency_index, dtype=np.int64)
    cpo, sto = np.asarray(offsets["cpo"]), np.asarray(offsets["sto"])
    require(cpo.shape == (64,) and sto.shape == (64,)
            and np.isfinite(cpo).all() and np.isfinite(sto).all(),
            "Invalid publisher offsets")
    require(np.isfinite(csi).all(), "Non-finite sampled raw CSI")
    raw_index = (frequency_index + 512) % 1024
    calibrated = np.empty((64, len(frequency_index)), dtype=np.complex128)
    for port in range(64):
        raw_values = (csi[port, raw_index, 0].astype(np.float64)
                      + 1j * csi[port, raw_index, 1].astype(np.float64))
        angle = float(cpo[port]) + float(sto[port]) * (
            2.0 * np.pi * frequency_index.astype(np.float64) / 1024.0)
        calibrated[port] = raw_values * (np.cos(angle) + 1j * np.sin(angle))
    return calibrated[assignments].transpose(0, 2, 1, 3)


def compare_calibration(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, "Calibration shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "Non-finite calibration comparison")
    absolute = np.abs(actual.astype(np.complex128) - expected)
    # Preparation uses complex64 arithmetic; near-zero values need an absolute term.
    atol, rtol = max(1e-12, 4e-7 * float(np.abs(expected).max())), 4e-6
    tolerance = atol + rtol * np.abs(expected)
    ratio = absolute / tolerance
    require(bool(np.all(absolute <= tolerance)),
            "Sampled publisher calibration or antenna/frequency mapping differs")
    return dict(max_abs=float(absolute.max()),
                max_relative_tolerance=float(ratio.max()), atol=atol, rtol=rtol)


def sampled_record(raw, source_id, record_bytes):
    start = int(source_id) * record_bytes
    header = bytes(raw[start:start + 12])
    require(len(header) == 12, "Truncated raw record header")
    length, header_crc = struct.unpack("<QI", header)
    require(length + 16 == record_bytes, "Non-uniform raw TFRecord length")
    payload = bytes(raw[start + 12:start + 12 + length])
    require(len(payload) == length, "Truncated raw TFRecord")
    require(masked_crc(header[:8]) == header_crc, "Sampled raw header CRC differs")
    expected_crc = struct.unpack(
        "<I", bytes(raw[start + 12 + length:start + 16 + length]))[0]
    require(masked_crc(payload) == expected_crc, "Sampled raw payload CRC differs")
    return example(payload)


def verify_prepared(prepared, dataset, legacy=None):
    prepared = Path(prepared)
    manifest = json.loads((prepared / "preparation_manifest.json").read_text())
    require(manifest["schema"] == SCHEMA and manifest["status"] == "complete",
            "Preparation is not complete under the publisher schema")
    require(manifest["dataset"] == dataset, "Dataset identity differs")
    contract = {key: manifest[key] for key in
                ("schema", "dataset", "version", "inputs", "config", "code")}
    contract_hash = hashlib.sha256(json.dumps(
        contract, sort_keys=True).encode()).hexdigest()
    require(manifest["contract_hash"] == contract_hash,
            "Preparation contract digest is invalid")
    for key, name in (("prepare_sha256", "prepare_publisher_data.py"),
                      ("reader_sha256", "dc01_data.py")):
        require(file_sha256(Path(__file__).with_name(name)) == manifest["code"][key],
                "Preparation source code differs from its recorded identity")
    config = manifest["config"]
    require(config["frequency_stride"] == 4 and config["voxel_m"] == .20
            and config["maximum_points"] == 6000
            and config["selection_seed"] == 20261015
            and config["block_m"] == 4.0 and config["guard_m"] == .30,
            "Preparation differs from the frozen data contract")
    inputs = {}
    for name in ("raw", "spec", "offsets", "coordinates"):
        item = manifest["inputs"][name]
        path = project_input(item["path"])
        require(path.stat().st_size == item["bytes"], name + " input size differs")
        if name != "raw":
            require(file_sha256(path) == item["sha256"],
                    name + " input hash differs")
        inputs[name] = path
    expected_offsets = PATHS.datasets / "dichasus/dichasus-dcxx/context" / (
        "reftx-offsets-dichasus-" + dataset + ".json")
    require(inputs["offsets"] == expected_offsets.resolve(),
            "Offsets are not the publisher's per-file calibration")
    require(inputs["raw"].name == "dichasus-" + dataset + ".tfrecords",
            "Raw TFRecord belongs to another acquisition")
    raw_info = manifest["inputs"]["raw"]
    record_bytes, frames = int(raw_info["record_bytes"]), int(raw_info["frames"])
    require(record_bytes > 16 and frames > 0
            and record_bytes * frames == inputs["raw"].stat().st_size,
            "Raw frame count/layout differs")
    require(len(raw_info["sha256"]) == 64 and len(raw_info["md5"]) == 32,
            "Missing full-source hash receipt")
    outputs = manifest["outputs"]
    for name, item in outputs.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Output name escapes the prepared directory")
        path = prepared / relative
        require(path.is_file() and path.stat().st_size == item["bytes"],
                "Prepared output size differs: " + name)
        require(file_sha256(path) == item["sha256"],
                "Prepared output hash differs: " + name)
    for name in ("channels.npy", "selection.npz", "all_metadata.npz",
                 "split_manifest.json", "source_scan.json", "data_audit.json"):
        require(name in outputs, "Required output absent from manifest: " + name)
    require(file_sha256(prepared / "selection.npz") == manifest["selection_sha256"],
            "Selection file digest differs")
    require(file_sha256(prepared / "split_manifest.json") == manifest["split_sha256"],
            "Split file digest differs")
    scan = json.loads((prepared / "source_scan.json").read_text())
    require(scan["contract_hash"] == contract_hash
            and scan["sha256"] == raw_info["sha256"]
            and scan["md5"] == raw_info["md5"]
            and scan["frames"] == frames and scan["bytes"] == raw_info["bytes"]
            and scan["metadata_sha256"] == file_sha256(prepared / "all_metadata.npz")
            and scan["all_source_csi_finite"] is True,
            "Full-source scan receipt is inconsistent with the manifest")
    stat = inputs["raw"].stat()
    require(scan["raw_stat"] == [stat.st_size, stat.st_mtime_ns],
            "Raw source changed after the recorded full scan")
    spec = json.loads(inputs["spec"].read_text())
    offsets = json.loads(inputs["offsets"].read_text())
    with np.load(prepared / "selection.npz", allow_pickle=False) as loaded:
        selection = {key: loaded[key] for key in loaded.files}
    for key in LEGACY_KEYS + ("raw_position", "directions", "frequency_index",
                             "frequency_hz", "antenna_assignments"):
        require(key in selection, "Missing selection key: " + key)
    for key, values in selection.items():
        require(np.issubdtype(values.dtype, np.number) and np.isfinite(values).all(),
                "Non-numeric or non-finite selection entry: " + key)
    ids = selection["source_id"]
    require(ids.dtype == np.int64 and ids.ndim == 1 and len(ids) > 0
            and bool(np.all(np.diff(ids) > 0))
            and int(ids[0]) >= 0 and int(ids[-1]) < frames,
            "Invalid selected raw source IDs")
    count = len(ids)
    for name, shape in (("position", (count, 3)), ("original_position", (count, 3)),
                        ("raw_position", (count, 3)), ("time", (count,)),
                        ("group", (count,)), ("panel_position", (2, 3)),
                        ("panel_direction", (2, 3)), ("directions", (2, 3)),
                        ("origin", (3,)), ("rotation", (3, 3)),
                        ("frequency_hz", (256,)),
                        ("antenna_assignments", (2, 4, 8))):
        require(selection[name].shape == shape, "Invalid metadata shape: " + name)
    exact_equal(selection["raw_position"], selection["original_position"], "Raw-position alias")
    exact_equal(selection["frequency_index"], selection["frequency_indices"], "Frequency alias")
    exact_equal(selection["directions"], selection["panel_direction"], "Direction alias")
    assignments = np.asarray([panel["assignments"] for panel in spec["antennas"]])
    exact_equal(selection["antenna_assignments"], assignments, "Publisher antenna assignments")
    expected_fi = np.arange(0, 1024, 4, dtype=np.int64)
    exact_equal(selection["frequency_indices"], expected_fi, "Frequency subset")
    expected_fhz = float(spec["frequency"]) + (
        expected_fi.astype(np.float64) - 512) * float(spec["bandwidth"]) / 1024
    exact_equal(selection["frequency_hz"], expected_fhz, "Physical frequency grid")
    origin, rotation = reference_coordinates(inputs["coordinates"])
    require(np.allclose(selection["origin"], origin, rtol=0, atol=1e-12)
            and np.allclose(selection["rotation"], rotation, rtol=0, atol=1e-12),
            "Coordinate frame differs from registered scene")
    for name, expected in (
        ("position", (selection["raw_position"] - origin) @ rotation.T),
        ("panel_position", (np.array([p["location"] for p in spec["antennas"]])
                            - origin) @ rotation.T),
        ("panel_direction", np.array([p["direction"] for p in spec["antennas"]])
                            @ rotation.T),
    ):
        require(np.allclose(selection[name], expected, rtol=0, atol=2e-12),
                "Incorrect coordinate conversion: " + name)
    with np.load(prepared / "all_metadata.npz", allow_pickle=False) as loaded:
        metadata = {key: loaded[key] for key in loaded.files}
    for key, values in metadata.items():
        require(np.issubdtype(values.dtype, np.number) and np.isfinite(values).all(),
                "Invalid all-frame metadata: " + key)
    require(metadata["position"].shape == (frames, 3)
            and metadata["time"].shape == (frames,),
            "All-frame metadata shape differs")
    selected, groups, unique = reference_selection(
        metadata["position"], config["voxel_m"], config["maximum_points"],
        config["selection_seed"])
    exact_equal(ids, selected, "Independent representative selection")
    exact_equal(selection["group"], groups, "Spatial cell groups")
    exact_equal(selection["raw_position"], metadata["position"][ids], "Selected raw positions")
    exact_equal(selection["time"], metadata["time"][ids], "Selected timestamps")
    channel = np.load(prepared / "channels.npy", mmap_mode="r", allow_pickle=False)
    require(channel.shape == (count, 2, 8, 4, 256)
            and channel.dtype == np.dtype("complex64"),
            "Prepared channel shape/dtype differs")
    for first in range(0, count, 32):
        require(np.isfinite(channel[first:first + 32]).all(),
                "Non-finite prepared channel value")
    protocol = json.loads((prepared / "split_manifest.json").read_text())
    require(manifest["split_protocol_sha256"] == protocol["sha256"],
            "Protocol digest differs from the preparation manifest")
    split_check = verify_split(selection["position"], protocol,
                               config["block_m"], config["guard_m"])
    audit = json.loads((prepared / "data_audit.json").read_text())
    require(audit["status"] == "complete"
            and audit["channel_shape"] == list(channel.shape)
            and audit["selected_points"] == count
            and audit["source_records"] == frames,
            "Audit metadata differs from actual prepared inputs")
    legacy_check = dict(applicable=dataset == "dc01")
    if dataset == "dc01":
        legacy = Path(legacy or PATHS.artifacts / "measured/dc01_v1")
        with np.load(legacy / "selection.npz", allow_pickle=False) as old:
            for key in old.files:
                require(key in selection, "Missing old selection field: " + key)
                exact_equal(selection[key], old[key], "Legacy dc01 " + key)
        old_split = legacy / "split_manifest.json"
        # The authoritative old split lived with the original baseline assets.
        if not old_split.exists():
            old_split = PATHS.artifacts / "measured/dc01_baselines_v1/split_manifest.json"
        require(old_split.exists(), "Legacy dc01 split reference is missing")
        require(file_sha256(old_split) == file_sha256(prepared / "split_manifest.json"),
                "Legacy dc01 split bytes were not preserved")
        legacy_check.update(selection_all_old_keys_exact=True,
                            split_file_bytes_exact=True,
                            old_split_path=str(old_split.relative_to(PATHS.root)))
    raw = np.memmap(inputs["raw"], mode="r", dtype=np.uint8)
    sample_rows = np.unique(np.rint(
        np.asarray(SAMPLE_QUANTILES) * (count - 1)).astype(np.int64))
    sample_checks = []
    for row in sample_rows:
        record = sampled_record(raw, int(ids[row]), record_bytes)
        exact_equal(np.asarray(record["pos-tachy"]), selection["raw_position"][row],
                    "Sampled raw position")
        require(float(record["time"]) == float(selection["time"][row]),
                "Sampled raw timestamp differs")
        expected = reference_calibration(record["csi"], offsets, assignments, expected_fi)
        difference = compare_calibration(channel[row], expected)
        sample_checks.append(dict(prepared_row=int(row), source_id=int(ids[row]),
                                  **difference))
    return dict(status="passed", dataset=dataset, schema=SCHEMA,
                checked_prepared_positions=count, channel_shape=list(channel.shape),
                channel_dtype=str(channel.dtype), all_prepared_entries_finite=True,
                unique_source_voxels=unique, representative_selection_independent=True,
                calibration=dict(reference_precision="complex128",
                                 every_raw_port_checked=True, ports_per_sample=64,
                                 frequencies_per_port=256, samples=sample_checks,
                                 raw_sample_crc_verified=True),
                metadata=dict(physical_panel_mapping_exact=True,
                              frequency_subset_exact=True, physical_frequency_hz_exact=True,
                              coordinate_frame_checked=True, aliases_exact=True),
                split=split_check, legacy=legacy_check,
                source_identity=dict(full_raw_rehash_performed=False,
                                     source_sha256=raw_info["sha256"],
                                     source_md5=raw_info["md5"],
                                     note="Full raw hashes are the preparation receipt; sampled raw records rechecked here."),
                limitations=["Shared existing TF Example decoder; calibration, mapping, representative selection and spatial checks are independent.",
                             "Calibration verification does not establish mobile-TX absolute phase/time coherence.",
                             "No training, inference, channel-quality scoring or comparison of scientific outcomes performed."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dc01", "dc02"), required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--legacy-dir", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    base = (PATHS.artifacts / "measured").resolve()
    prepared = (args.prepared_dir or base / (args.dataset + "_publisher_v1")).resolve()
    require(prepared.parent == base and prepared.name.startswith(args.dataset + "_publisher_"),
            "Verifier writes only inside a new publisher preparation directory")
    require(prepared.is_dir(), "Prepared publisher version is missing")
    try:
        report = verify_prepared(prepared, args.dataset, args.legacy_dir)
    except Exception as error:
        report = dict(status="failed", dataset=args.dataset,
                      error_type=type(error).__name__, error=str(error))
        if not args.no_write:
            temporary = prepared / "verification.partial.json"
            temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            os.replace(temporary, prepared / "verification.json")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        raise
    if not args.no_write:
        temporary = prepared / "verification.partial.json"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        os.replace(temporary, prepared / "verification.json")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
