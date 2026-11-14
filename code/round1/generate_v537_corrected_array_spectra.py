#!/usr/bin/env python3
"""Generate compact train/test spectra for the v534/v535-corrected array."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

import generate_v530_heavy_specialist_sionna_spectra as v530
from project_paths import PATHS


VERSION_NUMBER = 537
VERSION = "v537_corrected_array_sbi"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER) / "physical_sources"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / "v537_corrected_array_spectra.json"
SEEDS = (42, 206, 420)

BASE_CONFIG = v530.SpectrumConfig(
    name="corrected_array_d10_sc015_seed42",
    frequency_hz=6.693e9,
    tx_azimuth_deg=-29.70,
    tx_tilt_deg=-7.65,
    max_depth=10,
    synthetic_array=True,
    diffraction=False,
    edge_diffraction=False,
    diffuse_reflection=True,
    scattering_coefficient=0.15,
    samples_per_src=1_000_000,
    max_num_paths_per_src=1_000_000,
    batch_size=8,
    seed=42,
    scattering_scope="all",
    tx_vertical_spacing=4.0,
)


def config_for_seed(seed: int) -> v530.SpectrumConfig:
    return replace(
        BASE_CONFIG,
        name=f"corrected_array_d10_sc015_seed{seed}",
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=("all", *(str(seed) for seed in SEEDS)), default="all")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        v530.SCENE_XML,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v530.OUT_DIR = OUT_DIR

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    v530.select_drjit_cuda_device(dr, args.cuda_device)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    v530.select_drjit_cuda_device(dr, args.cuda_device)
    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v537 requires Sionna 2.0.1, got {sionna.__version__}")

    def frequencies(_frequency_hz: float):
        return np.asarray(
            subcarrier_frequencies(v530.NUM_SUBCARRIERS, v530.SUBCARRIER_SPACING)
        ).astype(np.float32)

    rt_api = (load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies)
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    split_values = ("train", "test") if args.split == "all" else (args.split,)
    seeds = SEEDS if args.seed == "all" else (int(args.seed),)
    run_args = argparse.Namespace(
        selection="all",
        force=args.force,
        batch_size=args.batch_size,
        budget_scale=1.0,
        cuda_device=args.cuda_device,
    )
    results: dict[str, object] = {}
    for seed in seeds:
        config = config_for_seed(seed)
        seed_result = {}
        for split in split_values:
            positions = train_positions if split == "train" else test_positions
            point_ids = np.arange(len(positions), dtype=np.int64)
            seed_result[split] = v530.generate_split(
                config=config,
                split=split,
                positions=positions,
                point_ids=point_ids,
                args=run_args,
                mi=mi,
                dr=dr,
                rt_api=rt_api,
            )
        results[str(seed)] = {
            "configuration": asdict(config),
            "splits": seed_result,
        }
    report = {
        "method": VERSION,
        "status": "physical_source_generation",
        "primary_hypothesis": (
            "The v534/v535-inferred 4-wavelength TX vertical spacing and corrected orientation "
            "remove a shared array-response error from the existing v531 physical source library."
        ),
        "environment": {
            "sionna": sionna.__version__,
            "mitsuba_variant": mi.variant(),
            "single_visible_gpu": True,
        },
        "results": results,
        "report": PATHS.relative(REPORT),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
