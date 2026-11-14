#!/usr/bin/env python3
"""Generate high-budget depth-5 all-interaction Sionna spectra for train/test."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import generate_v207_sionna_test_reference as v207
from project_paths import PATHS


VERSION_NUMBER = 461
VERSION = "v461_highbudget_d5_all_sionna_spectra"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
TINY = 1e-30
MP = 2
MH = 16
MV = 8
N_ANT = 4
S_SC = 192


def linkwise_pdp(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized 192-bin PDP and total power for each antenna link."""
    cir = np.fft.ifft(channel, axis=-1)
    power = np.abs(cir).astype(np.float32) ** 2
    total = power.sum(axis=-1, dtype=np.float64).astype(np.float32)
    shape = power / np.maximum(total[..., None], TINY)
    return shape.astype(np.float32), total


def joint_sh(channel: np.ndarray) -> np.ndarray:
    """Return the normalized scorer-traceback-like joint S-H power support."""
    array = channel.reshape(len(channel), MP, MH, MV, N_ANT, S_SC)
    array = array.transpose(0, 1, 5, 2, 4, 3)
    spectrum = np.fft.fft2(array, axes=(2, 3))
    power = np.abs(spectrum).astype(np.float32) ** 2
    sh = power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    sh /= np.maximum(sh.sum(axis=(1, 2), keepdims=True), TINY)
    return sh.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--samples-per-src", type=int, default=500_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=500_000)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def output_paths(prefix: str) -> dict[str, object]:
    return {
        "linkwise_pdp": OUT_DIR / f"{prefix}_linkwise_pdp.npy",
        "joint_sh": OUT_DIR / f"{prefix}_joint_sh.npy",
        "point_power": OUT_DIR / f"{prefix}_point_power.npy",
    }


def generate_split(
    *,
    prefix: str,
    positions: np.ndarray,
    scene,
    tx,
    frequencies,
    solver,
    mi,
    Receiver,
    args: argparse.Namespace,
) -> dict[str, object]:
    outputs = output_paths(prefix)
    expected = len(positions)
    if (
        not args.force
        and all(path.is_file() for path in outputs.values())
        and np.load(outputs["linkwise_pdp"], mmap_mode="r").shape[0] == expected
    ):
        return {"reused": True, "count": expected, "seconds": 0.0}
    linkwise = np.lib.format.open_memmap(
        outputs["linkwise_pdp"],
        mode="w+",
        dtype=np.float32,
        shape=(expected, 256 * 4, 192),
    )
    joint = np.lib.format.open_memmap(
        outputs["joint_sh"], mode="w+", dtype=np.float32, shape=(expected, 192, 16)
    )
    point_power = np.lib.format.open_memmap(
        outputs["point_power"], mode="w+", dtype=np.float64, shape=(expected,)
    )
    receiver_names: list[str] = []
    solve_seconds = 0.0
    cfr_seconds = 0.0
    started = time.time()
    for start in range(0, expected, args.batch_size):
        end = min(start + args.batch_size, expected)
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for index, position in enumerate(positions[start:end], start=start):
            name = f"rx_{prefix}_{index:05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, position)),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        t0 = time.time()
        paths = solver(
            scene,
            max_depth=5,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=True,
            edge_diffraction=True,
            diffraction_lit_region=True,
            seed=46_100 + (1_000_000 if prefix == "test" else 0) + start,
        )
        solve_seconds += time.time() - t0
        t1 = time.time()
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        h = v207.convert_cfr(raw)
        cfr_seconds += time.time() - t1
        link_shape, _ = linkwise_pdp(h)
        sh = joint_sh(h)
        linkwise[start:end] = link_shape.reshape(end - start, 256 * 4, 192)
        joint[start:end] = sh.astype(np.float32)
        point_power[start:end] = np.sum(
            np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        if end % 100 == 0 or end == expected:
            linkwise.flush()
            joint.flush()
            point_power.flush()
            print(
                f"v461 {prefix} {end}/{expected} elapsed={time.time() - started:.1f}",
                flush=True,
            )
    for name in receiver_names:
        scene.remove(name)
    return {
        "reused": False,
        "count": expected,
        "seconds": float(time.time() - started),
        "solve_seconds": float(solve_seconds),
        "cfr_seconds": float(cfr_seconds),
        "zero_power_count": int(np.sum(np.asarray(point_power) == 0.0)),
        "outputs": {name: PATHS.relative(path) for name, path in outputs.items()},
    }


def main() -> None:
    args = parse_args()
    for path in (SCENE_XML, PATHS.round1 / "Round1_Train_Pos.npy", PATHS.round1 / "Round1_Test_Pos.npy"):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

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
        name="tx_v461",
        position=mi.Point3f(*map(float, BS_POSITION)),
        look_at=target,
        power_dbm=44.0,
    )
    scene.add(tx)
    frequencies = subcarrier_frequencies(192, 30_000.0).cpu().numpy().astype(np.float32)
    solver = PathSolver()
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_count = min(max(args.train_count, 0), len(train_pos))
    test_count = min(max(args.test_count, 0), len(test_pos))
    train = generate_split(
        prefix="train",
        positions=train_pos[:train_count],
        scene=scene,
        tx=tx,
        frequencies=frequencies,
        solver=solver,
        mi=mi,
        Receiver=Receiver,
        args=args,
    )
    test = generate_split(
        prefix="test",
        positions=test_pos[:test_count],
        scene=scene,
        tx=tx,
        frequencies=frequencies,
        solver=solver,
        mi=mi,
        Receiver=Receiver,
        args=args,
    )
    report = {
        "method": VERSION,
        "status": "highbudget_spectra_ready",
        "primary_hypothesis": (
            "v205 selected depth-5 all-interaction 6 GHz as the best official-train "
            "configuration, but only a low-budget realization exists. A high-budget "
            "realization may recover missing diffuse/diffracted path support."
        ),
        "configuration": {
            "frequency_hz": 6.0e9,
            "tx_azimuth_deg": -30.0,
            "tx_tilt_deg": -9.53,
            "max_depth": 5,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "edge_diffraction": True,
            "samples_per_src": args.samples_per_src,
            "max_num_paths_per_src": args.max_num_paths_per_src,
            "materials": "canonical Sionna Etoile ITU defaults",
        },
        "environment": {
            "mitsuba_variant": mi.variant(),
            "single_visible_gpu_required": True,
        },
        "train": train,
        "test": test,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
