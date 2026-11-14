#!/usr/bin/env python3
"""Generate the clean v206-calibrated Sionna train side for strict replay."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from project_paths import PATHS


VERSION = "v208_strict_sionna_dataset"
OUT_DIR = PATHS.cache_dir(208)
DATASET_DIR = OUT_DIR / "strict_dataset"
CLEAN_TRAIN = OUT_DIR / "v208_sionna_train_clean_H.npy"
TEST_REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
REPORT_DIR = PATHS.version_report_dir(208)
REPORT = REPORT_DIR / "v208_strict_sionna_dataset_generate.json"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
EXPECTED_SHAPE = (2000, 256, 4, 192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def convert_cfr(raw: np.ndarray) -> np.ndarray:
    expected = (4, 1, 256, 1, 192)
    if raw.ndim != 6 or tuple(raw.shape[1:]) != expected:
        raise ValueError(f"Unexpected CFR shape {raw.shape}")
    return np.transpose(raw[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def validate_channel(path, expected_shape: tuple[int, ...], batch: int = 16) -> dict[str, object]:
    value = np.load(path, mmap_mode="r")
    if value.shape != expected_shape or value.dtype != np.complex64:
        raise ValueError(f"Invalid channel {path}: {value.shape} {value.dtype}")
    finite = True
    power = np.empty(len(value), dtype=np.float64)
    for start in range(0, len(value), batch):
        block = np.asarray(value[start : start + batch])
        finite = finite and bool(np.isfinite(block).all())
        power[start : start + len(block)] = np.sum(
            np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
    if not finite:
        raise ValueError(f"Non-finite values in {path}")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
        "zero_power_count": int(np.sum(power == 0.0)),
        "log_power_quantiles": np.quantile(
            np.log(np.maximum(power, 1e-30)), [0.0, 0.1, 0.5, 0.9, 1.0]
        ).tolist(),
    }


def generate_clean_train(args: argparse.Namespace) -> dict[str, float | bool]:
    if CLEAN_TRAIN.is_file() and not args.force:
        validate_channel(CLEAN_TRAIN, EXPECTED_SHAPE)
        return {"reused": True, "seconds": 0.0, "solve_seconds": 0.0, "cfr_seconds": 0.0}

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    scene = load_scene(str(SCENE_XML), merge_shapes=True)
    scene.frequency = 6.0e9
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
    radius = 140.0
    azimuth = math.radians(-30.0)
    tilt = math.radians(-9.53)
    target = mi.Point3f(
        float(BS_POSITION[0] + radius * math.cos(azimuth)),
        float(BS_POSITION[1] + radius * math.sin(azimuth)),
        float(BS_POSITION[2] + radius * math.tan(tilt)),
    )
    tx = Transmitter(
        name="tx_v208_strict",
        position=mi.Point3f(*[float(value) for value in BS_POSITION]),
        look_at=target,
        power_dbm=44.0,
    )
    scene.add(tx)
    frequencies = subcarrier_frequencies(192, 30_000.0).cpu().numpy().astype(np.float32)
    solver = PathSolver()
    output = np.lib.format.open_memmap(
        CLEAN_TRAIN,
        mode="w+",
        dtype=np.complex64,
        shape=EXPECTED_SHAPE,
    )
    receiver_names: list[str] = []
    solve_seconds = 0.0
    cfr_seconds = 0.0
    started = time.time()
    for start in range(0, len(positions), args.batch_size):
        end = min(start + args.batch_size, len(positions))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for index, position in enumerate(positions[start:end], start=start):
            name = f"rx_train_{index:04d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(float(position[0]), float(position[1]), float(position[2])),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        t0 = time.time()
        paths = solver(
            scene,
            max_depth=16,
            max_num_paths_per_src=1_000_000,
            samples_per_src=1_000_000,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=42 + start,
        )
        solve_seconds += time.time() - t0
        t1 = time.time()
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        cfr_seconds += time.time() - t1
        output[start:end] = convert_cfr(raw)
        output.flush()
        print(f"generated clean train {end}/{len(positions)}", flush=True)
    for name in receiver_names:
        scene.remove(name)
    return {
        "reused": False,
        "seconds": float(time.time() - started),
        "solve_seconds": float(solve_seconds),
        "cfr_seconds": float(cfr_seconds),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_REFERENCE.is_file():
        raise FileNotFoundError(f"Missing v207 test reference: {TEST_REFERENCE}")

    timing = generate_clean_train(args)
    train_validation = validate_channel(CLEAN_TRAIN, EXPECTED_SHAPE)
    test_validation = validate_channel(TEST_REFERENCE, (500, 256, 4, 192))
    np.save(DATASET_DIR / "Round1_Train_Pos.npy", np.load(PATHS.round1 / "Round1_Train_Pos.npy"))
    np.save(DATASET_DIR / "Round1_Test_Pos.npy", np.load(PATHS.round1 / "Round1_Test_Pos.npy"))

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A strict synthetic benchmark must regenerate clean train truth under exactly the same calibrated "
            "configuration as the test reference, then retrain historical algorithms from a noisy released train view."
        ),
        "configuration": {
            "frequency_hz": 6.0e9,
            "tx_azimuth_deg": -30.0,
            "tx_tilt_deg": -9.53,
            "rx_orientation": "track_bs",
            "subcarrier_spacing_hz": 30_000.0,
            "max_depth": 16,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": True,
            "diffraction": False,
            "edge_diffraction": False,
            "samples_per_src": 1_000_000,
            "max_num_paths_per_src": 1_000_000,
            "seed_policy": "42 + batch_start",
            "materials": "canonical Sionna Etoile ITU defaults",
        },
        "train": train_validation,
        "test": test_validation,
        "timing": timing,
        "outputs": {
            "clean_train": PATHS.relative(CLEAN_TRAIN),
            "clean_test": PATHS.relative(TEST_REFERENCE),
            "dataset_directory": PATHS.relative(DATASET_DIR),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
