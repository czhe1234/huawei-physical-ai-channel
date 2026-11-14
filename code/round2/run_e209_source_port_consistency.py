#!/usr/bin/env python3
"""Validate the source-aligned physical runtime against E185 Sionna RT 2.0.1."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment

from project_paths import PATHS
from round2.data import load_positions
from round2.physical_prior.source_port_common import (
    CONFIGS,
    E185_SEEDS,
    configure_materials,
    extract_path_support,
    metadata,
    prepare_scene_files,
    save_support,
    select_drjit_cuda_device,
    solver_seed,
    tx_target,
)


VERSION = "Round2-E209-source-port-consistency"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e209_source_port"
RUN_DIR = ARTIFACT_DIR / "runs"
LOG_DIR = ARTIFACT_DIR / "logs"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e209_source_port_consistency"
REPORT_PATH = ANALYSIS_DIR / "report.json"
SAMPLE_IDS = {0: np.asarray([2, 2003]), 1: np.asarray([0, 2002])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("reference", "port"))
    parser.add_argument("--config", choices=tuple(CONFIGS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--site", type=int, choices=(0, 1))
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_path(backend: str, config_name: str, seed: int, site: int) -> Path:
    return RUN_DIR / f"{backend}_{config_name}_seed{seed}_site{site}.npz"


def _runtime(backend: str):
    if backend == "reference":
        import sionna
        from sionna.rt import (
            PathSolver,
            PlanarArray,
            Receiver,
            Transmitter,
            load_scene,
        )

        if sionna.__version__ != "2.0.1":
            raise RuntimeError(f"expected Sionna 2.0.1, got {sionna.__version__}")
    else:
        from round2.physical_prior.physical_rt import (
            PathSolver,
            PlanarArray,
            Receiver,
            Transmitter,
            load_scene,
        )

    return PathSolver, PlanarArray, Receiver, Transmitter, load_scene


def run_worker(args: argparse.Namespace) -> None:
    if args.backend is None or args.config is None or args.seed is None or args.site is None:
        raise ValueError("worker requires backend, config, seed, and site")
    output = run_path(args.backend, args.config, args.seed, args.site)
    if output.is_file() and not args.force:
        print(json.dumps({"status": "cached", "output": str(output)}), flush=True)
        return

    started = time.time()
    select_drjit_cuda_device(args.cuda_device)
    PathSolver, PlanarArray, Receiver, Transmitter, load_scene = _runtime(args.backend)
    import mitsuba as mi

    data_started = time.time()
    config = CONFIGS[args.config]
    setup, train_positions, _, train_site, _ = load_positions()
    point_ids = SAMPLE_IDS[args.site]
    if not np.all(train_site[point_ids] == args.site):
        raise ValueError(f"sample IDs do not belong to site {args.site}")
    positions = np.asarray(train_positions[point_ids], dtype=np.float32)
    bs_position = np.asarray(setup.bs_positions[args.site], dtype=np.float32)
    data_finished = time.time()

    scene_started = time.time()
    scene = load_scene(
        str(prepare_scene_files()),
        merge_shapes=True,
        remove_duplicate_vertices=True,
    )
    scene.frequency = config.frequency_hz
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=config.tx_vertical_spacing,
        horizontal_spacing=config.tx_horizontal_spacing,
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
    configure_materials(scene, config.scattering_coefficient)
    transmitter = Transmitter(
        name=f"tx_site{args.site}_seed{args.seed}",
        position=mi.Point3f(*map(float, bs_position)),
        look_at=tx_target(
            bs_position,
            config.site_yaw_deg[args.site],
            config.site_tilt_deg[args.site],
        ),
        power_dbm=44.0,
    )
    scene.add(transmitter)
    for point_id, position in zip(point_ids, positions):
        scene.add(
            Receiver(
                name=f"rx_train_{int(point_id):05d}",
                position=mi.Point3f(*map(float, position)),
                look_at=transmitter,
            )
        )

    solver_started = time.time()
    paths = PathSolver()(
        scene,
        max_depth=config.max_depth,
        max_num_paths_per_src=config.max_num_paths_per_src,
        samples_per_src=config.samples_per_src,
        synthetic_array=config.synthetic_array,
        los=True,
        specular_reflection=True,
        diffuse_reflection=config.diffuse_reflection,
        refraction=True,
        diffraction=config.diffraction,
        edge_diffraction=config.edge_diffraction,
        diffraction_lit_region=True,
        seed=solver_seed(args.seed, args.site, "train", int(point_ids[0])),
    )
    solver_finished = time.time()
    support = extract_path_support(
        paths,
        scene,
        point_ids,
        positions,
        bs_position,
        config,
        args.seed,
    )
    finished = time.time()
    report = metadata(
        args.backend,
        config,
        args.seed,
        args.site,
        "train",
        point_ids,
        finished - started,
    )
    report["cuda_device"] = args.cuda_device
    report["path_counts"] = np.diff(support["offsets"]).tolist()
    report["timings"] = {
        "official_data_loading_seconds": data_finished - data_started,
        "scene_and_endpoint_setup_seconds": solver_started - scene_started,
        "path_solver_seconds": solver_finished - solver_started,
        "support_extraction_seconds": finished - solver_finished,
    }
    save_support(output, support, report)
    print(json.dumps({"status": "complete", "output": str(output), **report}), flush=True)


def _run_subprocess(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(
        result.stdout
        + ("\n[stderr]\n" + result.stderr if result.stderr else "")
        + f"\n[exit_code]={result.returncode}\n[wall_seconds]={time.time() - started:.6f}\n",
        encoding="utf-8",
    )
    return result.returncode


def run_all(args: argparse.Namespace) -> None:
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    units = [
        (config, seed, site)
        for config in CONFIGS
        for seed in E185_SEEDS
        for site in (0, 1)
    ]
    assignments = [units[index::len(gpus)] for index in range(len(gpus))]
    script = Path(__file__).resolve()

    def gpu_worker(gpu: int, local_units: list[tuple[str, int, int]]) -> list[dict[str, object]]:
        results = []
        for config, seed, site in local_units:
            for backend in ("reference", "port"):
                output = run_path(backend, config, seed, site)
                if output.is_file() and not args.force:
                    results.append({"output": str(output), "status": "cached"})
                    continue
                command = [
                    sys.executable,
                    str(script),
                    "--worker",
                    "--backend",
                    backend,
                    "--config",
                    config,
                    "--seed",
                    str(seed),
                    "--site",
                    str(site),
                    "--cuda-device",
                    str(gpu),
                ]
                if args.force:
                    command.append("--force")
                log_path = LOG_DIR / f"{backend}_{config}_seed{seed}_site{site}.log"
                code = _run_subprocess(command, log_path)
                results.append(
                    {
                        "output": str(output),
                        "log": str(log_path),
                        "exit_code": code,
                        "status": "complete" if code == 0 else "failed",
                    }
                )
                if code != 0:
                    raise RuntimeError(f"worker failed; see {log_path}")
        return results

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(gpu_worker, gpu, local_units)
            for gpu, local_units in zip(gpus, assignments)
            if local_units
        ]
        all_results = []
        for future in futures:
            all_results.extend(future.result())
    print(json.dumps({"status": "complete", "runs": all_results}, indent=2), flush=True)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {name: np.asarray(handle[name]) for name in handle.files}


def _sequence_key(values: dict[str, np.ndarray], index: int) -> tuple[object, ...]:
    depth = int(values["interaction_count"][index])
    return (
        depth,
        tuple(int(v) for v in values["interactions"][index, :depth]),
        tuple(int(v) for v in values["surface_plane"][index, :depth].reshape(-1)),
    )


def _angle_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.abs(left - right)
    difference[..., 1] = np.abs(np.angle(np.exp(1j * (left[..., 1] - right[..., 1]))))
    difference[..., 3] = np.abs(np.angle(np.exp(1j * (left[..., 3] - right[..., 3]))))
    return difference


def _point_metrics(reference: dict[str, np.ndarray], port: dict[str, np.ndarray], row: int):
    ref_slice = slice(reference["offsets"][row], reference["offsets"][row + 1])
    port_slice = slice(port["offsets"][row], port["offsets"][row + 1])
    ref_all = np.arange(ref_slice.start, ref_slice.stop)
    port_all = np.arange(port_slice.start, port_slice.stop)
    ref_indices = ref_all[np.argsort(reference["energy_rank"][ref_all])[:128]]
    port_indices = port_all[np.argsort(port["energy_rank"][port_all])[:128]]

    ref_groups: dict[tuple[object, ...], list[int]] = {}
    port_groups: dict[tuple[object, ...], list[int]] = {}
    for index in ref_indices:
        ref_groups.setdefault(_sequence_key(reference, int(index)), []).append(int(index))
    for index in port_indices:
        port_groups.setdefault(_sequence_key(port, int(index)), []).append(int(index))

    matched: list[tuple[int, int]] = []
    for key in ref_groups.keys() & port_groups.keys():
        left = np.asarray(ref_groups[key], dtype=np.int64)
        right = np.asarray(port_groups[key], dtype=np.int64)
        tau_cost = np.abs(reference["tau"][left, None] - port["tau"][None, right]) / 1e-8
        angle_cost = _angle_difference(
            reference["angles"][left, None, :], port["angles"][None, right, :]
        ).sum(axis=-1)
        rows, cols = linear_sum_assignment(tau_cost + angle_cost)
        matched.extend((int(left[i]), int(right[j])) for i, j in zip(rows, cols))

    ref_count = len(ref_indices)
    port_count = len(port_indices)
    matched_count = len(matched)
    ref_nonempty = ref_count > 0
    port_nonempty = port_count > 0
    if matched:
        left = np.asarray([pair[0] for pair in matched], dtype=np.int64)
        right = np.asarray([pair[1] for pair in matched], dtype=np.int64)
        delay_errors = np.abs(reference["tau"][left] - port["tau"][right])
        length_errors = np.abs(
            reference["path_length_m"][left] - port["path_length_m"][right]
        )
        angle_errors = _angle_difference(
            reference["angles"][left], port["angles"][right]
        ).reshape(-1)
        normal_errors = np.linalg.norm(
            reference["normals"][left] - port["normals"][right], axis=-1
        ).reshape(-1)
        incidence_errors = np.abs(
            reference["incidence_angle"][left] - port["incidence_angle"][right]
        ).reshape(-1)
        ref_energy = reference["relative_energy"][left].astype(np.float64)
        port_energy = port["relative_energy"][right].astype(np.float64)
        numerator = float(np.dot(ref_energy, port_energy))
        denominator = math.sqrt(
            float(np.dot(ref_energy, ref_energy)) * float(np.dot(port_energy, port_energy))
        )
        energy_cosine = numerator / max(denominator, 1e-30)
    else:
        delay_errors = np.empty(0)
        length_errors = np.empty(0)
        angle_errors = np.empty(0)
        normal_errors = np.empty(0)
        incidence_errors = np.empty(0)
        energy_cosine = float(ref_count == 0 and port_count == 0)
    return {
        "nonempty_equal": ref_nonempty == port_nonempty,
        "ref_count": ref_count,
        "port_count": port_count,
        "matched_count": matched_count,
        "precision": matched_count / max(port_count, 1),
        "recall": matched_count / max(ref_count, 1),
        "delay_errors": delay_errors,
        "length_errors": length_errors,
        "angle_errors": angle_errors,
        "normal_errors": normal_errors,
        "incidence_errors": incidence_errors,
        "energy_cosine": energy_cosine,
    }


def _quantile(parts: list[np.ndarray], value: float) -> float:
    present = [np.asarray(part).reshape(-1) for part in parts if np.asarray(part).size]
    return float(np.quantile(np.concatenate(present), value)) if present else 0.0


def compare() -> dict[str, object]:
    point_reports = []
    runtime = []
    for config in CONFIGS:
        for seed in E185_SEEDS:
            for site in (0, 1):
                ref_path = run_path("reference", config, seed, site)
                port_path = run_path("port", config, seed, site)
                if not ref_path.is_file() or not port_path.is_file():
                    raise FileNotFoundError(f"missing pair: {ref_path}, {port_path}")
                reference = _load(ref_path)
                port = _load(port_path)
                ref_meta = json.loads(str(reference["metadata"]))
                port_meta = json.loads(str(port["metadata"]))
                runtime.append(
                    {
                        "config": config,
                        "seed": seed,
                        "site": site,
                        "reference_seconds": ref_meta["elapsed_seconds"],
                        "port_seconds": port_meta["elapsed_seconds"],
                    }
                )
                if not np.array_equal(reference["point_ids"], port["point_ids"]):
                    raise ValueError("point IDs differ")
                for row, point_id in enumerate(reference["point_ids"]):
                    metrics = _point_metrics(reference, port, row)
                    point_reports.append(
                        {
                            "config": config,
                            "seed": seed,
                            "site": site,
                            "point_id": int(point_id),
                            **metrics,
                        }
                    )

    total_ref = sum(report["ref_count"] for report in point_reports)
    total_port = sum(report["port_count"] for report in point_reports)
    total_matched = sum(report["matched_count"] for report in point_reports)
    aggregate = {
        "point_count": len(point_reports),
        "nonempty_identity": float(
            np.mean([report["nonempty_equal"] for report in point_reports])
        ),
        "top128_precision": total_matched / max(total_port, 1),
        "top128_recall": total_matched / max(total_ref, 1),
        "length_error_p99_m": _quantile(
            [report["length_errors"] for report in point_reports], 0.99
        ),
        "angle_error_p99_rad": _quantile(
            [report["angle_errors"] for report in point_reports], 0.99
        ),
        "delay_error_p99_s": _quantile(
            [report["delay_errors"] for report in point_reports], 0.99
        ),
        "normal_error_p99": _quantile(
            [report["normal_errors"] for report in point_reports], 0.99
        ),
        "incidence_error_p99_rad": _quantile(
            [report["incidence_errors"] for report in point_reports], 0.99
        ),
        "relative_energy_cosine_min": float(
            min(report["energy_cosine"] for report in point_reports)
        ),
        "relative_energy_cosine_mean": float(
            np.mean([report["energy_cosine"] for report in point_reports])
        ),
    }
    thresholds = {
        "nonempty_identity": 1.0,
        "top128_precision": 0.99,
        "top128_recall": 0.99,
        "length_error_p99_m": 1e-3,
        "angle_error_p99_rad": 1e-5,
        "delay_error_p99_s": 5e-12,
        "relative_energy_cosine_min": 0.9999,
    }
    passed = (
        aggregate["nonempty_identity"] == thresholds["nonempty_identity"]
        and aggregate["top128_precision"] >= thresholds["top128_precision"]
        and aggregate["top128_recall"] >= thresholds["top128_recall"]
        and aggregate["length_error_p99_m"] <= thresholds["length_error_p99_m"]
        and aggregate["angle_error_p99_rad"] <= thresholds["angle_error_p99_rad"]
        and aggregate["delay_error_p99_s"] <= thresholds["delay_error_p99_s"]
        and aggregate["relative_energy_cosine_min"]
        >= thresholds["relative_energy_cosine_min"]
    )
    result = {
        "version": VERSION,
        "status": "passed" if passed else "failed",
        "sample_ids": {str(key): value.tolist() for key, value in SAMPLE_IDS.items()},
        "seeds": list(E185_SEEDS),
        "configs": list(CONFIGS),
        "thresholds": thresholds,
        "aggregate": aggregate,
        "runtime": runtime,
        "points": [
            {
                key: value
                for key, value in report.items()
                if not isinstance(value, np.ndarray)
            }
            for report in point_reports
        ],
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT_PATH)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    if args.run_all:
        run_all(args)
    if args.compare:
        result = compare()
        if result["status"] != "passed":
            raise SystemExit(2)
    if not (args.worker or args.run_all or args.compare):
        raise ValueError("select --worker, --run-all, or --compare")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
