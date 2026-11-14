#!/usr/bin/env python3
"""v252: cache scalar Sionna path fields on the official-point-cloud scene.

The cache contains one complex scalar field coefficient per traced path.  It
does not contain a frequency response, a MIMO channel, or a submission target.
The coefficients are intended as physics features for an official-H-supervised
neural path calibrator.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import build_v251_pointcloud_sionna_paths as v251
from project_paths import PATHS


VERSION_NUMBER = 252
VERSION = "v252_pointcloud_sionna_path_fields"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = OUT_DIR / f"{VERSION}.npz"
REPORT = REPORT_DIR / f"{VERSION}_build.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--samples-per-src", type=int, default=500_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=25201)
    parser.add_argument("--frequency", type=float, default=6.0e9)
    parser.add_argument("--subcarrier-spacing", type=float, default=30e3)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    return parser.parse_args()


def allocate(count: int, max_paths: int) -> dict[str, np.ndarray]:
    return {
        "valid": np.zeros((count, max_paths), dtype=bool),
        "tau": np.full((count, max_paths), np.nan, dtype=np.float32),
        "path_length": np.full((count, max_paths), np.nan, dtype=np.float32),
        "delay_bin_physical": np.full((count, max_paths), np.nan, dtype=np.float32),
        "theta_t": np.full((count, max_paths), np.nan, dtype=np.float32),
        "phi_t": np.full((count, max_paths), np.nan, dtype=np.float32),
        "theta_r": np.full((count, max_paths), np.nan, dtype=np.float32),
        "phi_r": np.full((count, max_paths), np.nan, dtype=np.float32),
        "interaction_count": np.zeros((count, max_paths), dtype=np.int8),
        "interaction_mask": np.zeros((count, max_paths), dtype=np.int16),
        "score": np.full((count, max_paths), -np.inf, dtype=np.float32),
        "a_real": np.zeros((count, max_paths), dtype=np.float32),
        "a_imag": np.zeros((count, max_paths), dtype=np.float32),
        "path_power": np.zeros((count, max_paths), dtype=np.float32),
    }


def remove_receivers(scene, names: list[str]) -> None:
    for name in names:
        if scene.get(name) is not None:
            scene.remove(name)


def trace_split(
    prefix: str,
    positions: np.ndarray,
    args: argparse.Namespace,
    scene,
    tx,
    mi,
    Receiver,
    PathSolver,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays = allocate(len(positions), args.max_paths)
    names: list[str] = []
    solver = PathSolver()
    batch_reports = []
    started = time.time()
    for start in range(0, len(positions), args.batch_size):
        end = min(start + args.batch_size, len(positions))
        remove_receivers(scene, names)
        names = []
        for local, pos in enumerate(positions[start:end]):
            name = f"rx_{prefix}_{start + local:05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, pos)),
                    look_at=tx,
                )
            )
            names.append(name)
        t0 = time.time()
        paths = solver(
            scene,
            max_depth=args.max_depth,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=True,
            edge_diffraction=getattr(args, "edge_diffraction", True),
            diffraction_lit_region=True,
            seed=args.seed + (1_000_000 if prefix == "test" else 0) + start,
        )
        selected = v251.select_paths(
            paths,
            args.max_paths,
            args.max_depth,
            args.subcarrier_spacing,
        )
        counts = []
        for local, item in enumerate(selected):
            count = len(item["ids"])
            counts.append(count)
            if count == 0:
                continue
            row = start + local
            arrays["valid"][row, :count] = True
            for key in arrays:
                if key in {"valid", "path_power"}:
                    continue
                arrays[key][row, :count] = item[key]
            arrays["path_power"][row, :count] = (
                item["a_real"] ** 2 + item["a_imag"] ** 2
            ).astype(np.float32)
        seconds = time.time() - t0
        batch_reports.append(
            {
                "start": int(start),
                "end": int(end),
                "seconds": float(seconds),
                "path_count_mean": float(np.mean(counts)),
                "zero_path_count": int(np.sum(np.asarray(counts) == 0)),
            }
        )
        print(
            f"v252 {prefix} {end}/{len(positions)} seconds={seconds:.3f} "
            f"paths_mean={np.mean(counts):.2f}",
            flush=True,
        )
    remove_receivers(scene, names)
    count = arrays["valid"].sum(axis=1)
    total_power = arrays["path_power"].sum(axis=1)
    return arrays, {
        "count": int(len(positions)),
        "seconds": float(time.time() - started),
        "path_count_mean": float(np.mean(count)),
        "path_count_median": float(np.median(count)),
        "zero_path_count": int(np.sum(count == 0)),
        "nonzero_field_count": int(np.sum(total_power > 0)),
        "log10_path_power_quantiles": np.quantile(
            np.log10(np.maximum(total_power, 1e-40)), [0.0, 0.1, 0.5, 0.9, 1.0]
        ).astype(float).tolist(),
        "batches": batch_reports,
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scene_xml = v251.prepare_scene_files()

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    scene = load_scene(str(scene_xml), merge_shapes=True)
    scene.frequency = args.frequency
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*map(float, v251.BS_POS)),
        look_at=mi.Point3f(0.0, 0.0, 1.5),
    )
    scene.add(tx)
    train, train_report = trace_split(
        "train", train_pos, args, scene, tx, mi, Receiver, PathSolver
    )
    test, test_report = trace_split(
        "test", test_pos, args, scene, tx, mi, Receiver, PathSolver
    )
    payload: dict[str, np.ndarray] = {
        "train_pos": train_pos,
        "test_pos": test_pos,
        "bs_pos": v251.BS_POS,
    }
    for prefix, arrays in (("train", train), ("test", test)):
        for key, value in arrays.items():
            payload[f"{prefix}_{key}"] = value
    np.savez_compressed(OUTPUT, **payload)
    report = {
        "method": VERSION,
        "status": "physics_feature_cache_only",
        "primary_hypothesis": (
            "Scalar Sionna path fields on the official-point-cloud scene provide "
            "path-energy evidence missing from geometry-only path tokens."
        ),
        "compliance_boundary": {
            "uses": [
                "official point-cloud reconstructed scene",
                "official train/test coordinates",
                "Sionna per-path scalar field as a physics feature",
            ],
            "does_not_use": [
                "Paths.cfr or MIMO RT H",
                "synthetic channel labels",
                "restored Etoile mesh/object/material labels",
                "v212/v217 teacher outputs",
                "local replay truth",
                "hidden online labels",
            ],
            "final_model_requirement": (
                "path fields must be calibrated by a neural model trained on official train H"
            ),
        },
        "config": vars(args),
        "train": train_report,
        "test": test_report,
        "outputs": {
            "features": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
