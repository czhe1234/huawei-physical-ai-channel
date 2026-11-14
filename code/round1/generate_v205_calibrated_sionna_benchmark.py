#!/usr/bin/env python3
"""Generate full real-Sionna benchmark domains selected by v205 calibration.

Each domain uses all 2,000 released train positions and all 500 hidden-test
positions.  Both splits are generated clean; train-only noise is injected by
the evaluator so several plausible release-noise mechanisms can share these
large clean caches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path

import numpy as np

import analyze_v205_sionna_config_calibration as calibration
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v205_calibrated_sionna_benchmark"
OUT_DIR = PATHS.cache_dir(205)
REPORT_DIR = PATHS.version_report_dir(205)
REPORT = REPORT_DIR / "v205_calibrated_sionna_benchmark_generate.json"
DEFAULT_PROFILES = (
    "sector_m30_track_d5_f6p0",
    "sector_m30_track_d5_f2p6",
    "sector_m30_track_d5_f6p7",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", default=None, help="Calibrated profile to generate; repeatable")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subcarrier-spacing", type=float, default=30_000.0)
    parser.add_argument("--num-subcarriers", type=int, default=192)
    parser.add_argument("--tx-power-dbm", type=float, default=44.0)
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    return parser.parse_args()


def remove_receivers(scene, names: list[str]) -> None:
    for name in names:
        if scene.get(name) is not None:
            scene.remove(name)


def generate_split(
    *,
    split: str,
    positions: np.ndarray,
    output: Path,
    profile: calibration.PathProfile,
    args: argparse.Namespace,
    scene,
    tx,
    mi,
    Receiver,
    PathSolver,
    frequencies: np.ndarray,
) -> dict[str, object]:
    channel = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(positions), 256, 4, args.num_subcarriers),
    )
    solver = PathSolver()
    receiver_names: list[str] = []
    power = np.empty(len(positions), dtype=np.float64)
    solve_seconds = 0.0
    cfr_seconds = 0.0
    started = time.time()

    for start in range(0, len(positions), args.batch_size):
        end = min(start + args.batch_size, len(positions))
        remove_receivers(scene, receiver_names)
        receiver_names = []
        for local, pos in enumerate(positions[start:end]):
            name = f"rx_{split}_{start + local:05d}"
            kwargs = {
                "name": name,
                "position": mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2])),
            }
            if profile.rx_orientation == "track_bs":
                kwargs["look_at"] = tx
            else:
                kwargs["orientation"] = mi.Point3f(0.0, 0.0, 0.0)
            scene.add(Receiver(**kwargs))
            receiver_names.append(name)

        t0 = time.time()
        paths = solver(
            scene,
            max_depth=profile.max_depth,
            max_num_paths_per_src=profile.max_num_paths_per_src,
            samples_per_src=profile.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=profile.diffuse_reflection,
            refraction=profile.refraction,
            diffraction=profile.diffraction,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=args.seed + (100_000 if split == "test" else 0) + start,
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
        block = calibration.convert_cfr(raw)
        if not (np.isfinite(block.real).all() and np.isfinite(block.imag).all()):
            raise FloatingPointError(f"Non-finite CFR in {profile.name}/{split} {start}:{end}")
        channel[start:end] = block
        channel.flush()
        power[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        print(
            f"{profile.name} {split} {end}/{len(positions)} "
            f"solve={solve_seconds:.1f}s cfr={cfr_seconds:.1f}s",
            flush=True,
        )

    remove_receivers(scene, receiver_names)
    return {
        "split": split,
        "count": int(len(positions)),
        "output": PATHS.relative(output),
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "seconds": float(time.time() - started),
        "solve_seconds": float(solve_seconds),
        "cfr_seconds": float(cfr_seconds),
        "zero_power_count": int(np.sum(power == 0.0)),
        "log_power_quantiles": np.quantile(np.log(np.maximum(power, 1e-30)), [0.0, 0.1, 0.5, 0.9, 1.0]).tolist(),
    }


def generate_domain(
    profile: calibration.PathProfile,
    args: argparse.Namespace,
    mi,
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
    subcarrier_frequencies,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
) -> dict[str, object]:
    domain_dir = OUT_DIR / f"benchmark_{profile.name}"
    domain_dir.mkdir(parents=True, exist_ok=True)
    scene = load_scene(str(calibration.SCENE_XML), merge_shapes=True)
    frequency = float(profile.frequency_hz or 3.5e9)
    scene.frequency = frequency
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
        name=f"tx_{profile.name}",
        position=mi.Point3f(*[float(value) for value in calibration.BS_POS]),
        look_at=calibration.tx_target(profile, mi),
        power_dbm=args.tx_power_dbm,
    )
    scene.add(tx)
    frequencies = subcarrier_frequencies(args.num_subcarriers, args.subcarrier_spacing).cpu().numpy().astype(np.float32)

    train_output = domain_dir / "train_clean_H.npy"
    test_output = domain_dir / "test_clean_H.npy"
    train_report = generate_split(
        split="train",
        positions=train_pos,
        output=train_output,
        profile=profile,
        args=args,
        scene=scene,
        tx=tx,
        mi=mi,
        Receiver=Receiver,
        PathSolver=PathSolver,
        frequencies=frequencies,
    )
    test_report = generate_split(
        split="test",
        positions=test_pos,
        output=test_output,
        profile=profile,
        args=args,
        scene=scene,
        tx=tx,
        mi=mi,
        Receiver=Receiver,
        PathSolver=PathSolver,
        frequencies=frequencies,
    )
    metadata = {
        "profile": asdict(profile),
        "frequency_hz": frequency,
        "subcarrier_spacing_hz": float(args.subcarrier_spacing),
        "num_subcarriers": int(args.num_subcarriers),
        "tx_power_dbm": float(args.tx_power_dbm),
        "array_order": "TX [polarization,horizontal,vertical], RX [polarization,vertical]",
        "train": train_report,
        "test": test_report,
    }
    metadata_path = domain_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata["metadata"] = PATHS.relative(metadata_path)
    return metadata


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    by_name = {profile.name: profile for profile in calibration.PROFILES}
    requested = tuple(args.profile or DEFAULT_PROFILES)
    missing = set(requested) - set(by_name)
    if missing:
        raise ValueError(f"Unknown calibrated profiles: {sorted(missing)}")
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_regimes = regime_of(geom["train_blocked"].astype(np.float32))
    test_regimes = regime_of(geom["test_blocked"].astype(np.float32))
    started = time.time()
    domains = []
    for name in requested:
        domains.append(
            generate_domain(
                by_name[name],
                args,
                mi,
                load_scene,
                Transmitter,
                Receiver,
                PlanarArray,
                PathSolver,
                subcarrier_frequencies,
                train_pos,
                test_pos,
            )
        )

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A current-method ablation is transferable only if its direction is stable across the "
            "three best calibrated Sionna carrier-frequency domains."
        ),
        "scene": PATHS.relative(calibration.SCENE_XML),
        "mitsuba_variant": mi.variant(),
        "train_shape": [len(train_pos), 256, 4, args.num_subcarriers],
        "test_shape": [len(test_pos), 256, 4, args.num_subcarriers],
        "train_regime_counts": {name: int(np.sum(train_regimes == name)) for name in ("clear", "light", "medium", "heavy")},
        "test_regime_counts": {name: int(np.sum(test_regimes == name)) for name in ("clear", "light", "medium", "heavy")},
        "domains": domains,
        "total_seconds": float(time.time() - started),
        "outputs": {"report": PATHS.relative(REPORT), "cache_root": PATHS.relative(OUT_DIR)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": PATHS.relative(REPORT), "domains": list(requested), "seconds": report["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
