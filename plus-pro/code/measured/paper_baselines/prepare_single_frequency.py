"""Freeze a dc01 frequency slice and audit author-input compatibility.

This is data preparation and numerical verification, not model training or a
claim that three published algorithms have been reproduced successfully.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from prepare_sources import atomic_write
from single_frequency_metrics import evaluate


REPORT = "analysis/results/dc01_paper_baselines_single_frequency"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_numpy(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--working-frequency-index", type=int, default=128)
    args = parser.parse_args()
    started = time.time()
    root = args.workspace.resolve()
    prepared = root / "artifacts/measured/dc01_v1"
    out = root / "artifacts/measured/dc01_single_frequency_v1"
    report_dir = root / REPORT
    source_manifest_path = report_dir / "source_manifest.json"
    author_sources = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    checked_sources = []
    for source in author_sources["sources"]:
        for item in source["files"]:
            path = root / item["path"]
            path.resolve().relative_to(root)
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Author source identity mismatch: {item['path']}")
        checked_sources.append({"name": source["name"], "commit": source["commit"],
                                "files_verified": len(source["files"])})
    original_manifest = json.loads((prepared / "prepared_manifest.json").read_text())
    for name in ["channels.npy", "selection.npz"]:
        if digest(prepared / name) != original_manifest["files"][name]["sha256"]:
            raise ValueError(f"Prepared data identity mismatch: {name}")
    channels = np.load(prepared / "channels.npy", mmap_mode="r", allow_pickle=False)
    meta = np.load(prepared / "selection.npz", allow_pickle=False)
    audit = json.loads((prepared / "data_audit.json").read_text())
    index = args.working_frequency_index
    if not 0 <= index < channels.shape[-1]:
        raise ValueError("Frequency index outside the prepared measured view")
    raw_index = int(meta["frequency_indices"][index])
    frequency_hz = float(audit["frequency_hz"] + (raw_index - 512) * audit["occupied_bandwidth_hz"] / 1024)
    h = np.array(channels[..., index:index+1])
    if h.dtype != np.complex64 or not np.isfinite(h).all():
        raise ValueError("Invalid single-frequency data")
    split_path = root / "artifacts/measured/dc01_baselines_v1/split_manifest.json"
    split = json.loads(split_path.read_text())
    for fold in split["splits"]:
        fit, val, test = (np.asarray(fold[key]) for key in ["fit", "val", "test"])
        if set(fit) & set(val) or set(fit) & set(test) or set(val) & set(test):
            raise ValueError("Overlapping position split")
        if np.any(np.sum(np.abs(h[fit].astype(np.complex128)) ** 2, axis=(2, 3, 4)) <= 0):
            raise ValueError("The preset frequency has a zero-energy training position/panel")
    identity = {"parent_preparation_sha256": audit["preparation_sha256"],
                "parent_channel_sha256": original_manifest["files"]["channels.npy"]["sha256"],
                "source_sha256": audit["source_sha256"], "split_sha256": digest(split_path),
                "frequency_hz": frequency_hz, "raw_fftshifted_frequency_index": raw_index,
                "parent_working_frequency_index": index}
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing prepared slice has a different identity")
    save_numpy(out / "channels.npy", h)
    atomic_write(out / "split_manifest.json", split_path.read_bytes())
    temporary = out / "selection.npz.partial"
    with temporary.open("wb") as stream:
        data = {key: meta[key] for key in meta.files}
        data["frequency_indices"] = np.asarray([raw_index])
        np.savez(stream, **data)
    os.replace(temporary, out / "selection.npz")
    atomic_write(identity_path, json.dumps(identity, indent=2).encode())
    reread = np.load(out / "channels.npy", allow_pickle=False)
    if not np.array_equal(reread, channels[..., index:index+1]):
        raise ValueError("Single-frequency extraction is not exactly the original calibrated slice")
    control, _ = evaluate(reread, reread)
    phase, _ = evaluate(1j * reread, reread)
    if control["raw_global_energy_ratio_nmse"] != 0 or abs(phase["raw_global_energy_ratio_nmse"] - 2) > 1e-12:
        raise ValueError("Independent single-frequency evaluation invariant failed")
    compatibility = {
        "neural_materials": {
            "author_sources": "code/third_party/paper_baselines/neural_materials",
            "native_training": "wideband power plus RMS-delay-spread loss",
            "single_frequency_cfr_export": True,
            "matched_single_frequency_training": "requires explicit removal of the delay-spread term; label as an adaptation",
            "model_training_executed": False,
        },
        "newrf": {
            "author_sources": "code/third_party/paper_baselines/newrf",
            "native_training": "single-frequency complex channel plus DoA-guided ray sampling",
            "native_inference": "learned ray search, without measured test-location DoA",
            "nearby_subcarrier_support": "original paper Appendix B and synthesizers.py frequency argument",
            "current_data_gap": "dc01 has a mobile single-antenna TX and two fixed RX arrays; reciprocal moving-RX DoA is not provided",
            "forbidden_shortcut": "do not call fixed-array AoA the reciprocal moving-end DoA or use measured test CSI to infer query rays",
            "model_training_executed": False,
        },
        "nerf2": {
            "author_sources": "code/third_party/paper_baselines/nerf2",
            "native_measured_mimo_input": "26 complex observed subcarriers (52 real values)",
            "native_location_branch": "3D transmitter position, used by RSSI/spectrum tasks",
            "matched_single_frequency_position_task": "requires a position-conditioned complex-output adapter, not the unchanged MIMO configuration",
            "model_training_executed": False,
        },
    }
    result = {"status": "source_and_data_preflight_complete_training_pending_input_contract",
              "scene": "dichasus-dc01", "identity": identity,
              "shape": list(h.shape), "dtype": str(h.dtype),
              "source_versions_verified": checked_sources,
              "saved_slice_matches_parent_bitwise": True,
              "existing_spatial_split_reused": True,
              "training_executed": False, "performance_results": None,
              "original_composite_score_defined": False,
              "metric_invariants": {"identity_raw_nmse": control["raw_global_energy_ratio_nmse"],
                                    "quarter_turn_raw_nmse": phase["raw_global_energy_ratio_nmse"],
                                    "identity_array_pas_cosine": control["array_pas_cosine"]},
              "author_compatibility": compatibility,
              "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                        for path in sorted(out.iterdir()) if path.is_file()},
              "elapsed_seconds": time.time() - started}
    atomic_write(report_dir / "preflight.json", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
    print(json.dumps({key: result[key] for key in ["status", "shape", "dtype", "identity", "elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
