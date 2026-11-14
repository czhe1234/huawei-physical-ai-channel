#!/usr/bin/env python3
"""Run a small real-Sionna baseline on the portable official-H subset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from project_paths import PATHS
from sionna_calibration_scoring import METADATA_PATH, TRUTH_PATH, score_prediction


SCENE_XML = (
    PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
)
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frequency", type=float, default=6.693e9)
    parser.add_argument("--subcarrier-spacing", type=float, default=30_000.0)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--samples-per-src", type=int, default=200_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tx-azimuth-deg", type=float, default=-30.0)
    parser.add_argument("--tx-tilt-deg", type=float, default=-9.53)
    parser.add_argument("--rx-orientation", choices=("track_bs", "fixed"), default="track_bs")
    parser.add_argument("--synthetic-array", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--los", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refraction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffuse-reflection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-diffraction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def tx_target(azimuth_deg: float, tilt_deg: float, mi):
    radius = 140.0
    azimuth = math.radians(azimuth_deg)
    tilt = math.radians(tilt_deg)
    return mi.Point3f(
        float(BS_POSITION[0] + radius * math.cos(azimuth)),
        float(BS_POSITION[1] + radius * math.sin(azimuth)),
        float(BS_POSITION[2] + radius * math.tan(tilt)),
    )


def convert_cfr(raw: np.ndarray) -> np.ndarray:
    expected = (4, 1, 256, 1, 192)
    if raw.ndim != 6 or tuple(raw.shape[1:]) != expected:
        raise ValueError(f"unexpected CFR shape: {raw.shape}")
    return np.transpose(raw[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def main() -> None:
    args = parse_args()
    metadata = np.load(METADATA_PATH)
    count = min(args.count, len(metadata["positions"]))
    positions = metadata["positions"][:count].astype(np.float32)
    truth = np.load(TRUTH_PATH, mmap_mode="r")

    import drjit as dr
    import mitsuba as mi
    import sionna

    mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene(str(SCENE_XML), merge_shapes=True)
    scene.frequency = args.frequency
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="VH",
    )
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*[float(value) for value in BS_POSITION]),
        look_at=tx_target(args.tx_azimuth_deg, args.tx_tilt_deg, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    frequencies = (
        subcarrier_frequencies(192, args.subcarrier_spacing)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    solver = PathSolver()
    output_path = args.output or (
        PATHS.analysis / "task_results" / "sionna_smoke_candidate.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=(count, 256, 4, 192)
    )
    receiver_names: list[str] = []
    started = time.time()
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for local, position in enumerate(positions[start:end]):
            name = f"rx_{start + local:05d}"
            kwargs = {
                "name": name,
                "position": mi.Point3f(*[float(value) for value in position]),
            }
            if args.rx_orientation == "track_bs":
                kwargs["look_at"] = tx
            else:
                kwargs["orientation"] = mi.Point3f(0.0, 0.0, 0.0)
            scene.add(Receiver(**kwargs))
            receiver_names.append(name)
        paths = solver(
            scene,
            max_depth=args.max_depth,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=args.synthetic_array,
            los=args.los,
            specular_reflection=True,
            diffuse_reflection=args.diffuse_reflection,
            refraction=args.refraction,
            diffraction=args.diffraction,
            edge_diffraction=args.edge_diffraction,
            diffraction_lit_region=True,
            seed=args.seed + start,
        )
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        output[start:end] = convert_cfr(raw)
        output.flush()
        dr.sync_thread()
        print(f"generated {end}/{count}", flush=True)
    for name in receiver_names:
        scene.remove(name)

    score, _ = score_prediction(
        output,
        truth[:count],
        metadata["split_code"][:count],
        metadata["regime_code"][:count],
        batch_size=min(args.batch_size, 8),
    )
    report = {
        "status": "pass",
        "sionna_version": getattr(sionna, "__version__", "unknown"),
        "mitsuba_variant": mi.variant(),
        "count": count,
        "seconds": time.time() - started,
        "config": vars(args) | {"output": str(output_path)},
        "score": score,
    }
    report_path = PATHS.analysis / "task_results" / "sionna_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

