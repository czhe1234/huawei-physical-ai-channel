#!/usr/bin/env python3
"""Generate a real Sionna RT dataset on the restored Etoile scene.

This is an analysis-only dataset generator.  It uses Sionna RT/Mitsuba to
trace paths and compute the full frequency-domain MIMO channel.  It does not
use the v203 proxy formulas and it does not create a leaderboard submission.

The script is intended to run in the local ``sionna1`` conda environment:

    PYTHONPATH=<project>/code HW_PROJECT_ROOT=<project> \
      conda run --no-capture-output -n sionna1 \
      python code/generate_v204_real_sionna_rt_dataset.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v204_real_sionna_rt_dataset"
OUT_DIR = PATHS.cache_dir(204)
REPORT_DIR = PATHS.version_report_dir(204)
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--test-count", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=204)
    parser.add_argument("--frequency", type=float, default=3.5e9)
    parser.add_argument("--subcarrier-spacing", type=float, default=30e3)
    parser.add_argument("--num-subcarriers", type=int, default=192)
    parser.add_argument("--tx-power-dbm", type=float, default=44.0)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--samples-per-src", type=int, default=40000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=40000)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    parser.add_argument("--merge-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffuse-reflection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-diffraction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refraction", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def stratified_indices(regimes: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if count >= len(regimes):
        return np.arange(len(regimes), dtype=np.int32)
    labels = np.asarray(regimes).astype(str)
    out: list[int] = []
    unique = ["clear", "light", "medium", "heavy"]
    totals = {r: int(np.sum(labels == r)) for r in unique}
    raw = {r: count * totals[r] / len(labels) for r in unique}
    alloc = {r: int(np.floor(raw[r])) for r in unique}
    remaining = count - sum(alloc.values())
    for r in sorted(unique, key=lambda x: raw[x] - alloc[x], reverse=True):
        if remaining <= 0:
            break
        alloc[r] += 1
        remaining -= 1
    for r in unique:
        candidates = np.flatnonzero(labels == r)
        if len(candidates) == 0 or alloc[r] <= 0:
            continue
        take = min(alloc[r], len(candidates))
        out.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(out) < count:
        rest = np.setdiff1d(np.arange(len(labels)), np.asarray(out, dtype=np.int32), assume_unique=False)
        out.extend(rng.choice(rest, size=count - len(out), replace=False).tolist())
    out_arr = np.asarray(out[:count], dtype=np.int32)
    out_arr.sort()
    return out_arr


def remove_receivers(scene, receiver_names: list[str]) -> None:
    for name in receiver_names:
        if scene.get(name) is not None:
            scene.remove(name)


def convert_cfr_to_competition_layout(h: np.ndarray) -> np.ndarray:
    # Sionna: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_freq]
    # Here num_tx=1 and num_time_steps=1.  Competition layout:
    # [point, tx_ant=256, rx_ant=4, subcarrier=192].
    if h.ndim != 6 or h.shape[1] != 4 or h.shape[2] != 1 or h.shape[3] != 256 or h.shape[4] != 1:
        raise ValueError(f"Unexpected CFR shape: {h.shape}")
    return np.transpose(h[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def run_split(
    *,
    split_name: str,
    positions: np.ndarray,
    output_path: Path,
    args: argparse.Namespace,
    scene,
    tx,
    mi,
    Transmitter,
    Receiver,
    PathSolver,
    subcarrier_frequencies,
) -> dict[str, object]:
    del Transmitter
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(positions), 256, 4, args.num_subcarriers),
    )
    freqs = subcarrier_frequencies(args.num_subcarriers, args.subcarrier_spacing).cpu().numpy().astype(np.float32)
    solver = PathSolver()
    receiver_names: list[str] = []
    batch_reports: list[dict[str, object]] = []
    start_all = time.time()

    for start in range(0, len(positions), args.batch_size):
        end = min(start + args.batch_size, len(positions))
        remove_receivers(scene, receiver_names)
        receiver_names = []
        for local, pos in enumerate(positions[start:end]):
            name = f"rx_{split_name}_{start + local:05d}"
            rx = Receiver(
                name=name,
                position=mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2])),
                look_at=tx,
            )
            scene.add(rx)
            receiver_names.append(name)

        t0 = time.time()
        paths = solver(
            scene,
            max_depth=args.max_depth,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=args.diffuse_reflection,
            refraction=args.refraction,
            diffraction=args.diffraction,
            edge_diffraction=args.edge_diffraction,
            diffraction_lit_region=True,
            seed=args.seed + 1000 * (split_name == "test") + start,
        )
        solve_seconds = time.time() - t0
        t1 = time.time()
        h = paths.cfr(
            mi.TensorXf(freqs),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        cfr_seconds = time.time() - t1
        block = convert_cfr_to_competition_layout(h)
        if not (np.isfinite(block.real).all() and np.isfinite(block.imag).all()):
            raise FloatingPointError(f"Non-finite Sionna CFR block {split_name} {start}:{end}")
        output[start:end] = block
        output.flush()
        power = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        batch_report = {
            "start": int(start),
            "end": int(end),
            "solve_seconds": float(solve_seconds),
            "cfr_seconds": float(cfr_seconds),
            "power_mean": float(np.mean(power)),
            "power_min": float(np.min(power)),
            "power_max": float(np.max(power)),
        }
        batch_reports.append(batch_report)
        print(
            f"{split_name} {end}/{len(positions)} "
            f"solve={solve_seconds:.3f}s cfr={cfr_seconds:.3f}s "
            f"power_mean={batch_report['power_mean']:.6e}",
            flush=True,
        )

    remove_receivers(scene, receiver_names)
    return {
        "split": split_name,
        "count": int(len(positions)),
        "output": PATHS.relative(output_path),
        "seconds": float(time.time() - start_all),
        "batches": batch_reports,
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
    from sionna.phy.channel import subcarrier_frequencies

    train_pos_all = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos_all = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_regimes_all = regime_of(geom["train_blocked"].astype(np.float32))
    test_regimes_all = regime_of(geom["test_blocked"].astype(np.float32))
    train_idx = stratified_indices(train_regimes_all, args.train_count, args.seed)
    test_idx = stratified_indices(test_regimes_all, args.test_count, args.seed + 1)
    train_pos = train_pos_all[train_idx]
    test_pos = test_pos_all[test_idx]

    print(f"mitsuba_variant={mi.variant()}", flush=True)
    print(f"loading_scene={SCENE_XML}", flush=True)
    scene = load_scene(str(SCENE_XML), merge_shapes=args.merge_shapes)
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
        position=mi.Point3f(float(BS_POS[0]), float(BS_POS[1]), float(BS_POS[2])),
        look_at=mi.Point3f(0.0, 0.0, 1.5),
        power_dbm=args.tx_power_dbm,
    )
    scene.add(tx)

    train_output = OUT_DIR / "v204_real_sionna_train_H.npy"
    test_output = OUT_DIR / "v204_real_sionna_test_H.npy"
    t0 = time.time()
    train_report = run_split(
        split_name="train",
        positions=train_pos,
        output_path=train_output,
        args=args,
        scene=scene,
        tx=tx,
        mi=mi,
        Transmitter=Transmitter,
        Receiver=Receiver,
        PathSolver=PathSolver,
        subcarrier_frequencies=subcarrier_frequencies,
    )
    test_report = run_split(
        split_name="test",
        positions=test_pos,
        output_path=test_output,
        args=args,
        scene=scene,
        tx=tx,
        mi=mi,
        Transmitter=Transmitter,
        Receiver=Receiver,
        PathSolver=PathSolver,
        subcarrier_frequencies=subcarrier_frequencies,
    )

    metadata = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "scene": PATHS.relative(SCENE_XML),
        "mitsuba_variant": mi.variant(),
        "sionna_rt": "real PathSolver + Paths.cfr, no proxy path-loss formulas",
        "array_config": {
            "tx": "PlanarArray rows=8 cols=16 pattern=tr38901 polarization=VH -> 256 antennas",
            "rx": "PlanarArray rows=2 cols=1 pattern=dipole polarization=cross -> 4 antennas",
            "output_layout": "[point, 256 tx antennas, 4 rx antennas, 192 subcarriers]",
        },
        "rt_config": vars(args),
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_regime_counts": {r: int(np.sum(train_regimes_all[train_idx].astype(str) == r)) for r in ["clear", "light", "medium", "heavy"]},
        "test_regime_counts": {r: int(np.sum(test_regimes_all[test_idx].astype(str) == r)) for r in ["clear", "light", "medium", "heavy"]},
        "train": train_report,
        "test": test_report,
        "total_seconds": float(time.time() - t0),
    }
    meta_path = OUT_DIR / "v204_real_sionna_rt_dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = REPORT_DIR / "v204_real_sionna_rt_dataset_generate.json"
    report_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metadata": PATHS.relative(meta_path), "report": PATHS.relative(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
