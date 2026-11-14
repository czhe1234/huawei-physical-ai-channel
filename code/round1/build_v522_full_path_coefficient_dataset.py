#!/usr/bin/env python3
"""Build compact source-CFR path coefficients for all official train/test coordinates."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v504_scene_topology_sionna_inverse as v504
import analyze_v507_complex_path_frequency_oracle as v507
import analyze_v510_joint_delay_h_complex_oracle as v510
from project_paths import PATHS


VERSION_NUMBER = 522
VERSION = "v522_full_path_coefficient_dataset"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
SHARD_DIR = CACHE_DIR / "coefficient_shards"
OUTPUT = CACHE_DIR / "v522_source_path_coefficients_2500.npz"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
V475_DIR = PATHS.cache_dir(475)

FREQUENCY_HZ = 6.693e9
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
MAX_DEPTH = 16
SAMPLES_PER_SRC = 3_000_000
MAX_PATHS_PER_SRC = 100_000
PATH_LIMIT = 32
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    return parser.parse_args()


def load_geometry() -> dict[str, np.ndarray]:
    output = {
        "valid": np.zeros((2500, 128), dtype=bool),
        "tau": np.zeros((2500, 128), dtype=np.float32),
        "phi_t": np.zeros((2500, 128), dtype=np.float32),
        "interaction_count": np.zeros((2500, 128), dtype=np.int16),
    }
    seen = np.zeros(2500, dtype=bool)
    for path in sorted((V475_DIR / "shards").glob("shard_*/path_targets.npz")):
        metadata = np.load(path.parent / "metadata.npz")
        point_ids = metadata["position_id"].astype(np.int64)
        with np.load(path) as data:
            for name in output:
                output[name][point_ids] = data[name]
        seen[point_ids] = True
    if not np.all(seen):
        raise RuntimeError(f"missing v475 rows: {np.flatnonzero(~seen)[:10]}")
    return output


def select_and_fit(
    tau: np.ndarray,
    phi_t: np.ndarray,
    depth: np.ndarray,
    source_channel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau, h_center, depth = v510.path_candidates(tau, phi_t, depth)
    if not len(tau):
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 64), dtype=np.complex64),
        )
    matrix = v510.joint_basis(tau, h_center)
    target = v510.target_matrix(source_channel)
    if len(tau) > PATH_LIMIT:
        coefficients = v507.ridge_coefficients(matrix, target)
        energy = np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)
        score = np.log(np.maximum(energy, v507.TINY)) - 1e-6 * depth
        selected = np.argsort(score)[-PATH_LIMIT:]
        tau = tau[selected]
        h_center = h_center[selected]
        depth = depth[selected]
        matrix = matrix[:, selected]
    coefficients = v507.ridge_coefficients(matrix, target)
    return tau, h_center, depth, coefficients.astype(np.complex64)


def shard_path(shard_id: int):
    return SHARD_DIR / f"shard_{shard_id:05d}.npz"


def combine_shards() -> None:
    paths = [shard_path(shard_id) for shard_id in range(25)]
    if not all(path.is_file() for path in paths):
        return
    parts = []
    for path in paths:
        with np.load(path) as data:
            parts.append({name: np.asarray(data[name]) for name in data.files})
    point_ids = np.concatenate([part["position_ids"] for part in parts])
    order = np.argsort(point_ids)
    if not np.array_equal(point_ids[order], np.arange(2500)):
        raise ValueError("combined coefficient ids are not 0..2499")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        position_ids=point_ids[order],
        positions=np.concatenate([part["positions"] for part in parts])[order],
        valid=np.concatenate([part["valid"] for part in parts])[order],
        tau=np.concatenate([part["tau"] for part in parts])[order],
        h_center=np.concatenate([part["h_center"] for part in parts])[order],
        depth=np.concatenate([part["depth"] for part in parts])[order],
        source_coefficients=np.concatenate(
            [part["source_coefficients"] for part in parts]
        )[order],
        source_fit_coherence=np.concatenate(
            [part["source_fit_coherence"] for part in parts]
        )[order],
        source_path_count=np.concatenate(
            [part["source_path_count"] for part in parts]
        )[order],
    )


def main() -> None:
    args = parse_args()
    for path in (SCENE_XML, V475_DIR / "positions.npz"):
        if not path.is_file():
            raise FileNotFoundError(path)
    geometry = load_geometry()
    position_data = np.load(V475_DIR / "positions.npz")
    positions = position_data["positions"].astype(np.float32)

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v522 requires Sionna 2.0.1, got {sionna.__version__}")
    frequencies = np.asarray(
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
    ).astype(np.float32)
    scene = load_scene(
        str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
    )
    scene.frequency = FREQUENCY_HZ
    materials = v504.material_audit(scene)
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
    search = v501.SearchConfig(
        name="v522_source",
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-9.53,
        rx_orientation="track_bs",
        synthetic_array=True,
        frequency_hz=FREQUENCY_HZ,
        max_depth=MAX_DEPTH,
        samples_per_src=SAMPLES_PER_SRC,
        max_num_paths_per_src=MAX_PATHS_PER_SRC,
        seed=SEED,
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*[float(value) for value in v501.BS_POSITION]),
        look_at=v501.tx_target(search, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    shard_stop = min(25, args.start_shard + (args.limit_shards or 25))
    receiver_names: list[str] = []
    generated = 0

    for shard_id in range(args.start_shard, shard_stop):
        output_path = shard_path(shard_id)
        if output_path.is_file() and not args.force:
            continue
        point_ids = np.arange(shard_id * 100, (shard_id + 1) * 100, dtype=np.int64)
        local_valid = np.zeros((100, PATH_LIMIT), dtype=bool)
        local_tau = np.zeros((100, PATH_LIMIT), dtype=np.float64)
        local_h = np.zeros_like(local_tau)
        local_depth = np.zeros((100, PATH_LIMIT), dtype=np.float32)
        local_coefficients = np.zeros((100, PATH_LIMIT, 64), dtype=np.complex64)
        local_coherence = np.zeros(100, dtype=np.float32)
        local_path_count = np.zeros(100, dtype=np.int32)

        for batch_start in range(0, 100, args.batch_size):
            batch_end = min(batch_start + args.batch_size, 100)
            for name in receiver_names:
                scene.remove(name)
            receiver_names = []
            batch_ids = point_ids[batch_start:batch_end]
            for point_id in batch_ids:
                name = f"rx_{int(point_id):05d}"
                scene.add(
                    Receiver(
                        name=name,
                        position=mi.Point3f(
                            *[float(value) for value in positions[int(point_id)]]
                        ),
                        look_at=tx,
                    )
                )
                receiver_names.append(name)
            global_batch = int(batch_ids[0]) // args.batch_size
            paths = solver(
                scene,
                max_depth=MAX_DEPTH,
                max_num_paths_per_src=MAX_PATHS_PER_SRC,
                samples_per_src=SAMPLES_PER_SRC,
                synthetic_array=True,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=True,
                diffraction=False,
                edge_diffraction=False,
                diffraction_lit_region=True,
                seed=SEED + global_batch,
            )
            raw = paths.cfr(
                mi.TensorXf(frequencies),
                normalize_delays=False,
                normalize=False,
                out_type="numpy",
            )
            channels = v501.convert_cfr(np.asarray(raw), len(batch_ids))
            current_valid = np.asarray(paths.valid.numpy(), dtype=bool)[:, 0]
            current_tau = np.asarray(paths.tau.numpy(), dtype=np.float32)[:, 0]
            current_phi = np.asarray(paths.phi_t.numpy(), dtype=np.float32)[:, 0]
            current_interactions = (
                np.asarray(paths.interactions.numpy())[:, :, 0]
                .transpose(1, 2, 0)
            )
            current_depth = np.sum(current_interactions != 0, axis=2).astype(np.int16)

            for local_batch, point_id in enumerate(batch_ids):
                row = batch_start + local_batch
                candidate_valid = geometry["valid"][point_id]
                if np.any(candidate_valid):
                    tau_source = geometry["tau"][point_id, candidate_valid]
                    phi_source = geometry["phi_t"][point_id, candidate_valid]
                    depth_source = geometry["interaction_count"][point_id, candidate_valid]
                else:
                    candidate_valid = current_valid[local_batch]
                    tau_source = current_tau[local_batch, candidate_valid]
                    phi_source = current_phi[local_batch, candidate_valid]
                    depth_source = current_depth[local_batch, candidate_valid]
                tau, h_center, depth, coefficients = select_and_fit(
                    tau_source,
                    phi_source,
                    depth_source,
                    channels[local_batch],
                )
                count = len(tau)
                local_valid[row, :count] = True
                local_tau[row, :count] = tau
                local_h[row, :count] = h_center
                local_depth[row, :count] = depth
                local_coefficients[row, :count] = coefficients
                local_path_count[row] = count
                if count:
                    reconstructed = v510.channel_from_matrix(
                        v510.joint_basis(tau, h_center) @ coefficients
                    )
                    local_coherence[row] = v507.complex_metrics(
                        reconstructed, channels[local_batch]
                    )["complex_coherence"]
            del paths, raw, channels
            dr.sync_thread()

        np.savez_compressed(
            output_path,
            position_ids=point_ids,
            positions=positions[point_ids],
            valid=local_valid,
            tau=local_tau,
            h_center=local_h,
            depth=local_depth,
            source_coefficients=local_coefficients,
            source_fit_coherence=local_coherence,
            source_path_count=local_path_count,
        )
        generated += 1
        print(
            f"v522 shard {shard_id + 1}/25 elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    for name in receiver_names:
        scene.remove(name)
    combine_shards()
    complete = OUTPUT.is_file()
    report = {
        "method": VERSION,
        "status": "analysis_complete" if complete else "partial_cache",
        "primary_hypothesis": (
            "The v520 single-configuration path interface can be compactly reproduced at "
            "all 2000 train and 500 test coordinates without storing full Sionna H."
        ),
        "configuration": {
            "frequency_hz": FREQUENCY_HZ,
            "depth": MAX_DEPTH,
            "samples_per_src": SAMPLES_PER_SRC,
            "max_paths_per_src": MAX_PATHS_PER_SRC,
            "path_limit": PATH_LIMIT,
            "synthetic_array": True,
        },
        "materials": materials,
        "generated_shards_this_run": generated,
        "completed_shards": len(list(SHARD_DIR.glob("shard_*.npz"))),
        "outputs": {
            "coefficient_dataset": PATHS.relative(OUTPUT) if complete else None,
            "shards": PATHS.relative(SHARD_DIR),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    if complete:
        with np.load(OUTPUT) as data:
            report["summary"] = {
                "point_count": int(len(data["position_ids"])),
                "mean_paths": float(np.mean(data["source_path_count"])),
                "zero_path_points": int(np.sum(data["source_path_count"] == 0)),
                "mean_source_fit_coherence": float(
                    np.mean(data["source_fit_coherence"])
                ),
            }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
