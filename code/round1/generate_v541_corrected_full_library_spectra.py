#!/usr/bin/env python3
"""Generate the missing corrected-array spectra for a full physical library."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import numpy as np

import generate_v530_heavy_specialist_sionna_spectra as v530
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 541
VERSION = "v541_corrected_full_library_spectra"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER) / "physical_sources"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"

AZIMUTH_DEG = -29.70
TILT_DEG = -7.65
VERTICAL_SPACING = 4.0


def config(
    *,
    name: str,
    frequency_hz: float,
    max_depth: int,
    synthetic_array: bool = True,
    diffraction: bool = False,
    edge_diffraction: bool = False,
    diffuse_reflection: bool = False,
    scattering_coefficient: float = 0.0,
    samples_per_src: int,
    max_num_paths_per_src: int,
    batch_size: int,
    seed: int,
) -> v530.SpectrumConfig:
    return v530.SpectrumConfig(
        name=name,
        frequency_hz=frequency_hz,
        tx_azimuth_deg=AZIMUTH_DEG,
        tx_tilt_deg=TILT_DEG,
        max_depth=max_depth,
        synthetic_array=synthetic_array,
        diffraction=diffraction,
        edge_diffraction=edge_diffraction,
        diffuse_reflection=diffuse_reflection,
        scattering_coefficient=scattering_coefficient,
        samples_per_src=samples_per_src,
        max_num_paths_per_src=max_num_paths_per_src,
        batch_size=batch_size,
        seed=seed,
        scattering_scope="all",
        tx_vertical_spacing=VERTICAL_SPACING,
    )


CONFIGS = (
    config(
        name="corrected_depth16_f6p0",
        frequency_hz=6.0e9,
        max_depth=16,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=42,
    ),
    config(
        name="corrected_depth5_f2p6",
        frequency_hz=2.6e9,
        max_depth=5,
        diffraction=True,
        diffuse_reflection=True,
        samples_per_src=60_000,
        max_num_paths_per_src=60_000,
        batch_size=16,
        seed=2050,
    ),
    config(
        name="corrected_depth5_f6p0",
        frequency_hz=6.0e9,
        max_depth=5,
        diffraction=True,
        diffuse_reflection=True,
        samples_per_src=60_000,
        max_num_paths_per_src=60_000,
        batch_size=16,
        seed=2050,
    ),
    config(
        name="corrected_depth5_f6p7",
        frequency_hz=6.7e9,
        max_depth=5,
        diffraction=True,
        diffuse_reflection=True,
        samples_per_src=60_000,
        max_num_paths_per_src=60_000,
        batch_size=16,
        seed=2050,
    ),
    config(
        name="corrected_depth5_all_f6p0",
        frequency_hz=6.0e9,
        max_depth=5,
        diffraction=True,
        edge_diffraction=True,
        diffuse_reflection=True,
        samples_per_src=500_000,
        max_num_paths_per_src=500_000,
        batch_size=8,
        seed=46_100,
    ),
    config(
        name="corrected_depth20_scatter015_seed42",
        frequency_hz=6.693e9,
        max_depth=20,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=42,
    ),
    config(
        name="corrected_depth20_scatter015_seed206",
        frequency_hz=6.693e9,
        max_depth=20,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        batch_size=8,
        seed=206,
    ),
    config(
        name="corrected_depth16_nonsynthetic_edge",
        frequency_hz=6.693e9,
        max_depth=16,
        synthetic_array=False,
        diffraction=True,
        edge_diffraction=True,
        samples_per_src=3_000_000,
        max_num_paths_per_src=100_000,
        batch_size=2,
        seed=42,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=("all", *(item.name for item in CONFIGS)),
        default="all",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_existing_report() -> dict[str, object]:
    if not REPORT.is_file():
        return {}
    return json.loads(REPORT.read_text(encoding="utf-8")).get("results", {})


def main() -> None:
    args = parse_args()
    required = (
        v530.SCENE_XML,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
        PATHS.shared_features / "height_geometry_cache.npz",
    )
    for path in required:
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
        raise RuntimeError(f"v541 requires Sionna 2.0.1, got {sionna.__version__}")

    def frequencies(_frequency_hz: float) -> np.ndarray:
        return np.asarray(
            subcarrier_frequencies(v530.NUM_SUBCARRIERS, v530.SUBCARRIER_SPACING)
        ).astype(np.float32)

    rt_api = (load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies)
    positions = {
        "train": np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32),
        "test": np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32),
    }
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    heavy_ids = {
        "train": np.flatnonzero(regime_of(geometry["train_blocked"]) == "heavy").astype(np.int64),
        "test": np.flatnonzero(regime_of(geometry["test_blocked"]) == "heavy").astype(np.int64),
    }
    selected_configs = [
        item for item in CONFIGS if args.config in ("all", item.name)
    ]
    selected_splits = ("train", "test") if args.split == "all" else (args.split,)
    results = load_existing_report()
    for item in selected_configs:
        selection = "heavy" if item.name.endswith("nonsynthetic_edge") else "all"
        run_args = argparse.Namespace(
            selection=selection,
            force=args.force,
            batch_size=args.batch_size,
            budget_scale=1.0,
            cuda_device=args.cuda_device,
        )
        split_results = dict(results.get(item.name, {}).get("splits", {}))
        for split in selected_splits:
            point_ids = (
                heavy_ids[split]
                if selection == "heavy"
                else np.arange(len(positions[split]), dtype=np.int64)
            )
            split_results[split] = v530.generate_split(
                config=item,
                split=split,
                positions=positions[split][point_ids],
                point_ids=point_ids,
                args=run_args,
                mi=mi,
                dr=dr,
                rt_api=rt_api,
            )
            results[item.name] = {
                "configuration": asdict(item),
                "selection": selection,
                "splits": split_results,
            }
            report = {
                "method": VERSION,
                "status": "physical_source_generation",
                "primary_hypothesis": (
                    "Rebuilding every independent v531 physical source with the independently "
                    "inferred 4-wavelength TX vertical spacing and corrected BS orientation "
                    "removes a shared departure-angle response error rather than tuning v540."
                ),
                "fixed_array": {
                    "tx_horizontal_spacing_wavelengths": 0.5,
                    "tx_vertical_spacing_wavelengths": VERTICAL_SPACING,
                    "tx_azimuth_deg": AZIMUTH_DEG,
                    "tx_tilt_deg": TILT_DEG,
                    "rx_orientation": "look_at_tx",
                },
                "environment": {
                    "sionna": sionna.__version__,
                    "mitsuba_variant": mi.variant(),
                    "single_visible_gpu": True,
                    "requested_cuda_device": args.cuda_device,
                },
                "reused_corrected_d10_source": (
                    "artifacts/version_cache/v537/physical_sources/"
                    "corrected_array_d10_sc015_seed{42,206,420}_{split}_*.npy"
                ),
                "results": results,
                "report": PATHS.relative(REPORT),
            }
            REPORT.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps({"report": PATHS.relative(REPORT), "results": results}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
