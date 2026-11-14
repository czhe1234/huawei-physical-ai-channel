#!/usr/bin/env python3
"""Attach default-concrete CIR coefficients to the v291 geometry paths."""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import build_v251_pointcloud_sionna_paths as v251
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 316
VERSION = "v316_pointcloud_path_coefficients"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUT_DIR / "v316_pointcloud_path_coefficients.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
REFERENCE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
SCENE_XML = PATHS.cache_dir(291) / "scene" / "scene.xml"

BATCH_SIZE = 32
MAX_PATHS = 128
MAX_DEPTH = 16
SAMPLES_PER_SRC = 1_000_000
MAX_NUM_PATHS_PER_SRC = 1_000_000
SEED = 29101
FREQUENCY = 6.0e9
SUBCARRIER_SPACING = 30_000.0


def trace_split(
    prefix: str,
    positions: np.ndarray,
    reference: np.lib.npyio.NpzFile,
    scene,
    tx,
    mi,
    Receiver,
    PathSolver,
) -> tuple[np.ndarray, dict[str, object]]:
    coefficient = np.zeros((len(positions), MAX_PATHS), dtype=np.complex64)
    names: list[str] = []
    solver = PathSolver()
    maximum_length_error = 0.0
    maximum_angle_error = 0.0
    matched_path_count = 0
    reference_path_count = int(np.sum(reference[f"{prefix}_valid"]))
    started = time.time()
    for start in range(0, len(positions), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(positions))
        v251.remove_receivers(scene, names)
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
        paths = solver(
            scene,
            max_depth=MAX_DEPTH,
            max_num_paths_per_src=MAX_NUM_PATHS_PER_SRC,
            samples_per_src=SAMPLES_PER_SRC,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=SEED + (1_000_000 if prefix == "test" else 0) + start,
        )
        selected = v251.select_paths(
            paths,
            MAX_PATHS,
            MAX_DEPTH,
            SUBCARRIER_SPACING,
            depth_balanced_ranking=True,
        )
        cir, _tau = paths.cir(normalize_delays=False, out_type="numpy")
        raw = cir[:, 0, 0, 0, :, 0].astype(np.complex64)
        for local, item in enumerate(selected):
            row = start + local
            count = len(item["ids"])
            reference_ids = np.flatnonzero(reference[f"{prefix}_valid"][row])
            if count == 0 or len(reference_ids) == 0:
                continue
            length_error = np.abs(
                item["path_length"][:, None]
                - reference[f"{prefix}_path_length"][row, reference_ids][None, :]
            )
            phi_error = np.abs(
                np.angle(
                    np.exp(
                        1j
                        * (
                            item["phi_t"][:, None]
                            - reference[f"{prefix}_phi_t"][row, reference_ids][None, :]
                        )
                    )
                )
            )
            theta_error = np.abs(
                item["theta_t"][:, None]
                - reference[f"{prefix}_theta_t"][row, reference_ids][None, :]
            )
            depth_mismatch = (
                item["interaction_count"][:, None]
                != reference[f"{prefix}_interaction_count"][row, reference_ids][None, :]
            )
            cost = length_error + 20.0 * (phi_error + theta_error) + 1000.0 * depth_mismatch
            current_index, reference_index = linear_sum_assignment(cost)
            current_length_error = length_error[current_index, reference_index]
            current_angle_error = (
                phi_error[current_index, reference_index]
                + theta_error[current_index, reference_index]
            )
            keep = (
                ~depth_mismatch[current_index, reference_index]
                & (current_length_error <= 0.05)
                & (current_angle_error <= 1e-3)
            )
            current_index = current_index[keep]
            reference_index = reference_index[keep]
            if len(current_index):
                destination = reference_ids[reference_index]
                coefficient[row, destination] = raw[local, item["ids"][current_index]]
                matched_path_count += len(current_index)
                maximum_length_error = max(
                    maximum_length_error, float(np.max(current_length_error[keep]))
                )
                maximum_angle_error = max(
                    maximum_angle_error, float(np.max(current_angle_error[keep]))
                )
        print(f"v316 {prefix} {end}/{len(positions)}", flush=True)
    v251.remove_receivers(scene, names)
    valid = reference[f"{prefix}_valid"].astype(bool)
    power = np.abs(coefficient[valid]) ** 2
    return coefficient, {
        "count": int(len(positions)),
        "seconds": float(time.time() - started),
        "valid_path_count": int(np.sum(valid)),
        "coefficient_matched_path_count": int(matched_path_count),
        "coefficient_match_fraction": float(matched_path_count / max(reference_path_count, 1)),
        "nonzero_coefficient_count": int(np.sum(power > 0.0)),
        "log10_power": summarize(np.log10(np.maximum(power, 1e-30))),
        "maximum_path_length_replay_error_m": maximum_length_error,
        "maximum_departure_angle_replay_error_rad": maximum_angle_error,
    }


def main() -> None:
    for path in (REFERENCE, SCENE_XML):
        if not path.is_file():
            raise FileNotFoundError(path)
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    reference = np.load(REFERENCE)
    scene = load_scene(str(SCENE_XML), merge_shapes=True)
    scene.frequency = FREQUENCY
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
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_coefficient, train_report = trace_split(
        "train", train_pos, reference, scene, tx, mi, Receiver, PathSolver
    )
    test_coefficient, test_report = trace_split(
        "test", test_pos, reference, scene, tx, mi, Receiver, PathSolver
    )
    if (
        train_report["coefficient_match_fraction"] < 0.95
        or test_report["coefficient_match_fraction"] < 0.95
    ):
        raise RuntimeError("v316 coefficient replay covers less than 95% of v291 geometry paths")
    np.savez_compressed(
        OUTPUT,
        train_coefficient=train_coefficient,
        test_coefficient=test_coefficient,
    )
    report = {
        "method": VERSION,
        "status": "physical_feature_cache_only",
        "primary_hypothesis": (
            "Default-concrete Fresnel/path-loss coefficients contain path-energy information "
            "beyond geometry-only length/depth features."
        ),
        "compliance_boundary": {
            "uses": [
                "official point-cloud-derived mesh",
                "official train/test coordinates",
                "Sionna default concrete material CIR coefficients as candidate features",
            ],
            "does_not_use": [
                "original Etoile mesh/material/object labels",
                "synthetic channel targets or teacher outputs",
                "test truth or local replay truth",
                "direct RT channel submission construction",
            ],
        },
        "config": {
            "frequency": FREQUENCY,
            "max_depth": MAX_DEPTH,
            "samples_per_src": SAMPLES_PER_SRC,
            "max_num_paths_per_src": MAX_NUM_PATHS_PER_SRC,
            "seed": SEED,
            "material": "Sionna ITU concrete default",
            "array": "single isotropic V element; feature extraction only",
        },
        "train": train_report,
        "test": test_report,
        "outputs": {
            "coefficients": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
