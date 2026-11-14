#!/usr/bin/env python3
"""Test whether current-Sionna mechanism unions recover missing official paths."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v505_official_path_span_diagnostic as v505
from project_paths import PATHS


VERSION_NUMBER = 506
VERSION = "v506_path_mechanism_union_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
PATH_SET_DIR = CACHE_DIR / "path_sets"
POINT_METRIC_DIR = CACHE_DIR / "point_metrics"
REPORT = REPORT_DIR / f"{VERSION}.json"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"
V505_REPORT = PATHS.version_report_dir(505) / "v505_official_path_span_diagnostic.json"

FREQUENCY_HZ = 6.693e9
MAX_RETAINED_PATHS = 256
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TINY = 1e-30


@dataclass(frozen=True)
class MechanismConfig:
    name: str
    max_depth: int
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    samples_per_src: int = 1_000_000
    max_num_paths_per_src: int = 1_000_000
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--config-limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    return parser.parse_args()


def configs() -> list[MechanismConfig]:
    return [
        MechanismConfig("d3_base_200k_seed42", 3, samples_per_src=200_000, max_num_paths_per_src=200_000),
        MechanismConfig("d3_diff_200k_seed42", 3, diffraction=True, samples_per_src=200_000, max_num_paths_per_src=200_000),
        MechanismConfig("d3_edge_200k_seed42", 3, diffraction=True, edge_diffraction=True, samples_per_src=200_000, max_num_paths_per_src=200_000),
        MechanismConfig("d3_shadow_200k_seed42", 3, diffraction=True, diffraction_lit_region=False, samples_per_src=200_000, max_num_paths_per_src=200_000),
        MechanismConfig("d5_diff_1m_seed42", 5, diffraction=True),
        MechanismConfig("d8_diff_1m_seed42", 8, diffraction=True),
        MechanismConfig("d12_base_1m_seed42", 12),
        MechanismConfig("d16_base_1m_seed42", 16),
        MechanismConfig("d16_base_1m_seed206", 16, seed=206),
        MechanismConfig("d16_base_1m_seed420", 16, seed=420),
        MechanismConfig("d16_base_3m_seed42", 16, samples_per_src=3_000_000, max_num_paths_per_src=3_000_000),
        MechanismConfig("d16_diff_1m_seed42", 16, diffraction=True),
        MechanismConfig("d16_edge_1m_seed42", 16, diffraction=True, edge_diffraction=True),
        MechanismConfig("d16_shadow_1m_seed42", 16, diffraction=True, diffraction_lit_region=False),
        MechanismConfig("d20_base_1m_seed42", 20),
        MechanismConfig("d32_base_1m_seed42", 32),
    ]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def path_set_path(config: MechanismConfig, point_count: int) -> Path:
    return PATH_SET_DIR / f"{safe_name(config.name)}_points{point_count}.npz"


def metric_path(config_name: str, point_count: int) -> Path:
    return POINT_METRIC_DIR / f"{safe_name(config_name)}_points{point_count}.npz"


def deduplicate_paths(
    tau: np.ndarray, phi_t: np.ndarray, interaction_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(tau) == 0:
        return tau.astype(np.float32), phi_t.astype(np.float32), interaction_count.astype(np.int16)
    pdp_center = np.mod(
        v505.N_DELAY * v505.SUBCARRIER_SPACING * tau.astype(np.float64),
        v505.N_DELAY,
    )
    h_center = np.mod(
        8.0 * np.sin(phi_t.astype(np.float64) - v505.TX_ORIENTATION_RAD),
        v505.N_H,
    )
    key_delay = np.rint(pdp_center * 4.0).astype(np.int32)
    key_h = np.rint(h_center * 4.0).astype(np.int32)
    order = np.lexsort((tau, interaction_count))
    selected: list[int] = []
    seen: set[tuple[int, int]] = set()
    for index in order:
        key = (int(key_delay[index]), int(key_h[index]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(int(index))
        if len(selected) == MAX_RETAINED_PATHS:
            break
    keep = np.asarray(selected, dtype=np.int64)
    return (
        tau[keep].astype(np.float32),
        phi_t[keep].astype(np.float32),
        interaction_count[keep].astype(np.int16),
    )


def generate_path_set(
    config: MechanismConfig,
    indices: np.ndarray,
    positions: np.ndarray,
    batch_size: int,
    scene,
    tx,
    solver,
    mi,
    dr,
    Receiver,
) -> dict[str, np.ndarray]:
    output_path = path_set_path(config, len(indices))
    if output_path.is_file():
        data = np.load(output_path)
        return {key: np.asarray(data[key]) for key in data.files}

    valid_out = np.zeros((len(indices), MAX_RETAINED_PATHS), dtype=bool)
    tau_out = np.full((len(indices), MAX_RETAINED_PATHS), np.nan, dtype=np.float32)
    phi_out = np.full((len(indices), MAX_RETAINED_PATHS), np.nan, dtype=np.float32)
    depth_out = np.zeros((len(indices), MAX_RETAINED_PATHS), dtype=np.int16)
    receiver_names: list[str] = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(indices), batch_size)):
        end = min(start + batch_size, len(indices))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for point_id in indices[start:end]:
            name = f"rx_{int(point_id):05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*[float(value) for value in positions[point_id]]),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        paths = solver(
            scene,
            max_depth=config.max_depth,
            max_num_paths_per_src=config.max_num_paths_per_src,
            samples_per_src=config.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=config.diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=config.diffraction_lit_region,
            seed=config.seed + batch_index,
        )
        valid = np.asarray(paths.valid.numpy()[:, 0], dtype=bool)
        tau = np.asarray(paths.tau.numpy()[:, 0], dtype=np.float32)
        phi_t = np.asarray(paths.phi_t.numpy()[:, 0], dtype=np.float32)
        interactions = np.asarray(paths.interactions.numpy()[:, :, 0])
        interactions = interactions.transpose(1, 2, 0)
        interaction_count = np.sum(interactions != 0, axis=2).astype(np.int16)
        for local in range(end - start):
            keep = (
                valid[local]
                & np.isfinite(tau[local])
                & (tau[local] >= 0.0)
                & np.isfinite(phi_t[local])
            )
            local_tau, local_phi, local_depth = deduplicate_paths(
                tau[local, keep], phi_t[local, keep], interaction_count[local, keep]
            )
            count = len(local_tau)
            valid_out[start + local, :count] = True
            tau_out[start + local, :count] = local_tau
            phi_out[start + local, :count] = local_phi
            depth_out[start + local, :count] = local_depth
        del paths, valid, tau, phi_t, interactions, interaction_count
        dr.sync_thread()
        if end == len(indices) or (batch_index + 1) % 5 == 0:
            print(
                f"{config.name}: paths {end}/{len(indices)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    for name in receiver_names:
        scene.remove(name)
    dr.sync_thread()
    PATH_SET_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        indices=indices,
        valid=valid_out,
        tau=tau_out,
        phi_t=phi_out,
        interaction_count=depth_out,
    )
    return {
        "indices": indices,
        "valid": valid_out,
        "tau": tau_out,
        "phi_t": phi_out,
        "interaction_count": depth_out,
    }


def load_targets(indices: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    truth = np.load(TRAIN_H, mmap_mode="r")
    pdp = np.empty((len(indices), v505.N_DELAY), dtype=np.float32)
    joint_sh = np.empty((len(indices), v505.N_DELAY, v505.N_H), dtype=np.float32)
    nonzero = np.zeros(len(indices), dtype=bool)
    for row, point_id in enumerate(indices):
        target = v505.official_targets(truth[point_id])
        pdp[row] = target["pdp"]
        joint_sh[row] = target["joint_sh"]
        nonzero[row] = np.linalg.norm(target["pdp"]) > TINY
    return {"pdp": pdp, "joint_sh": joint_sh}, nonzero


def evaluate_path_set(
    name: str,
    path_set: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    regimes: np.ndarray,
    nonzero: np.ndarray,
) -> dict[str, object]:
    cache = metric_path(name, len(regimes))
    if cache.is_file():
        data = np.load(cache)
        metrics = {key: np.asarray(data[key]) for key in data.files if key not in {"indices", "regimes", "nonzero"}}
    else:
        metrics = {
            "oracle_pdp_cosine": np.zeros(len(regimes), dtype=np.float32),
            "oracle_joint_sh_cosine": np.zeros(len(regimes), dtype=np.float32),
            "pdp_support_coverage": np.zeros(len(regimes), dtype=np.float32),
            "joint_sh_support_coverage": np.zeros(len(regimes), dtype=np.float32),
            "path_count": np.zeros(len(regimes), dtype=np.float32),
            "mean_interaction_depth": np.zeros(len(regimes), dtype=np.float32),
        }
        for row in range(len(regimes)):
            valid = path_set["valid"][row].astype(bool)
            tau = path_set["tau"][row, valid].astype(np.float32)
            phi_t = path_set["phi_t"][row, valid].astype(np.float32)
            depth = path_set["interaction_count"][row, valid].astype(np.float32)
            metrics["path_count"][row] = len(tau)
            if not nonzero[row] or len(tau) == 0:
                continue
            pdp_atoms = v505.delay_kernel(tau, fft_sign=-1)
            sh_delay = v505.delay_kernel(tau, fft_sign=1)
            h_atoms, h_center = v505.horizontal_kernel(phi_t)
            sh_atoms = sh_delay[:, :, None] * h_atoms[:, None, :]
            pdp_reconstruction, pdp_weight = v505.fit_nonnegative_atoms(
                targets["pdp"][row], pdp_atoms
            )
            sh_reconstruction, sh_weight = v505.fit_nonnegative_atoms(
                targets["joint_sh"][row], sh_atoms
            )
            metrics["oracle_pdp_cosine"][row] = v505.row_cosine(
                targets["pdp"][row], pdp_reconstruction
            )
            metrics["oracle_joint_sh_cosine"][row] = v505.row_cosine(
                targets["joint_sh"][row], sh_reconstruction
            )
            pdp_center = np.argmax(pdp_atoms, axis=1).astype(np.float32)
            sh_center = np.argmax(sh_delay, axis=1).astype(np.float32)
            pdp_coverage, sh_coverage = v505.support_coverage(
                targets["pdp"][row],
                targets["joint_sh"][row],
                pdp_center,
                sh_center,
                h_center,
            )
            metrics["pdp_support_coverage"][row] = pdp_coverage
            metrics["joint_sh_support_coverage"][row] = sh_coverage
            combined = (pdp_weight + sh_weight) / 2.0
            metrics["mean_interaction_depth"][row] = float(np.sum(combined * depth))
        POINT_METRIC_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, regimes=regimes, nonzero=nonzero, **metrics)

    grouped = v505.grouped_summary(metrics, regimes, nonzero)
    heavy = grouped["heavy"]
    primary = 0.5 * heavy["oracle_pdp_cosine"]["mean"] + 0.5 * heavy["oracle_joint_sh_cosine"]["mean"]
    return {
        "name": name,
        "primary_heavy_oracle": float(primary),
        "metrics": grouped,
        "point_metrics": PATHS.relative(cache),
    }


def union_path_sets(path_sets: list[dict[str, np.ndarray]], indices: np.ndarray) -> dict[str, np.ndarray]:
    valid_out = np.zeros((len(indices), MAX_RETAINED_PATHS), dtype=bool)
    tau_out = np.full((len(indices), MAX_RETAINED_PATHS), np.nan, dtype=np.float32)
    phi_out = np.full((len(indices), MAX_RETAINED_PATHS), np.nan, dtype=np.float32)
    depth_out = np.zeros((len(indices), MAX_RETAINED_PATHS), dtype=np.int16)
    for row in range(len(indices)):
        tau_parts = []
        phi_parts = []
        depth_parts = []
        for path_set in path_sets:
            valid = path_set["valid"][row].astype(bool)
            tau_parts.append(path_set["tau"][row, valid])
            phi_parts.append(path_set["phi_t"][row, valid])
            depth_parts.append(path_set["interaction_count"][row, valid])
        tau, phi_t, depth = deduplicate_paths(
            np.concatenate(tau_parts), np.concatenate(phi_parts), np.concatenate(depth_parts)
        )
        count = len(tau)
        valid_out[row, :count] = True
        tau_out[row, :count] = tau
        phi_out[row, :count] = phi_t
        depth_out[row, :count] = depth
    return {
        "indices": indices,
        "valid": valid_out,
        "tau": tau_out,
        "phi_t": phi_out,
        "interaction_count": depth_out,
    }


def main() -> None:
    args = parse_args()
    for path in (SCENE_XML, TRAIN_H, V503_REPORT, V505_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    selected_configs = configs()
    if args.config_limit is not None:
        selected_configs = selected_configs[: args.config_limit]
    if args.force:
        for config in selected_configs:
            path_set_path(config, 160).unlink(missing_ok=True)
            metric_path(config.name, 160).unlink(missing_ok=True)
        metric_path(f"all_{len(selected_configs)}_config_union", 160).unlink(missing_ok=True)

    v503_report = json.loads(V503_REPORT.read_text(encoding="utf-8"))
    indices = np.asarray(v503_report["indices"], dtype=np.int64)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    regimes = v501.regime_codes()[indices]
    targets, nonzero = load_targets(indices)

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    import sionna
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v506 requires Sionna 2.0.1, got {sionna.__version__}")
    scene = load_scene(str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True)
    scene.frequency = FREQUENCY_HZ
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
        name="v506_fixed",
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-9.53,
        rx_orientation="track_bs",
        synthetic_array=True,
        frequency_hz=FREQUENCY_HZ,
        max_depth=16,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        seed=42,
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*[float(value) for value in v501.BS_POSITION]),
        look_at=v501.tx_target(search, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()

    path_sets = []
    results = []
    for config in selected_configs:
        path_set = generate_path_set(
            config,
            indices,
            positions,
            args.batch_size,
            scene,
            tx,
            solver,
            mi,
            dr,
            Receiver,
        )
        path_sets.append(path_set)
        result = evaluate_path_set(config.name, path_set, targets, regimes, nonzero)
        result["config"] = asdict(config)
        results.append(result)
        print(
            f"{config.name}: heavy_oracle={result['primary_heavy_oracle']:.6f} "
            f"pdp={result['metrics']['heavy']['oracle_pdp_cosine']['mean']:.6f} "
            f"sh={result['metrics']['heavy']['oracle_joint_sh_cosine']['mean']:.6f}",
            flush=True,
        )

    union = union_path_sets(path_sets, indices)
    union_name = f"all_{len(selected_configs)}_config_union"
    union_result = evaluate_path_set(
        union_name, union, targets, regimes, nonzero
    )
    ranking = sorted(results, key=lambda row: row["primary_heavy_oracle"], reverse=True)
    v505_report = json.loads(V505_REPORT.read_text(encoding="utf-8"))
    baseline = {
        "name": "v475_depth16_1m_path_tokens",
        "heavy_pdp_oracle": v505_report["metrics"]["heavy"]["oracle_pdp_cosine"]["mean"],
        "heavy_joint_sh_oracle": v505_report["metrics"]["heavy"]["oracle_joint_sh_cosine"]["mean"],
    }
    baseline["primary_heavy_oracle"] = 0.5 * (
        baseline["heavy_pdp_oracle"] + baseline["heavy_joint_sh_oracle"]
    )
    best = ranking[0]
    union_heavy = union_result["metrics"]["heavy"]
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "If a missing current-Sionna mechanism or stochastic path discovery setting is "
            "the official mismatch, at least one mechanism set or their union must materially "
            "raise the heavy free-weight delay and delay-angle oracle."
        ),
        "fixed_parameters": {
            "scene": PATHS.relative(SCENE_XML),
            "materials": "canonical Etoile ITU defaults, unchanged",
            "frequency_hz": FREQUENCY_HZ,
            "tx_array": "8x16 tr38901 VH",
            "rx_array": "2x1 dipole cross",
            "tx_azimuth_tilt_deg": [-30.0, -9.53],
            "rx_orientation": "track_bs",
            "merge_shapes": True,
            "remove_duplicate_vertices": True,
        },
        "point_count": int(len(indices)),
        "baseline": baseline,
        "ranking": ranking,
        "union": union_result,
        "decision": {
            "best_gain_vs_v475": float(
                best["primary_heavy_oracle"] - baseline["primary_heavy_oracle"]
            ),
            "union_gain_vs_v475": float(
                union_result["primary_heavy_oracle"] - baseline["primary_heavy_oracle"]
            ),
            "union_heavy_pdp_oracle": union_heavy["oracle_pdp_cosine"]["mean"],
            "union_heavy_joint_sh_oracle": union_heavy["oracle_joint_sh_cosine"]["mean"],
            "current_solver_switches_explain_missing_paths": bool(
                union_heavy["oracle_pdp_cosine"]["mean"] >= 0.95
                and union_heavy["oracle_joint_sh_cosine"]["mean"] >= 0.90
            ),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "path_set_dir": PATHS.relative(PATH_SET_DIR),
            "point_metric_dir": PATHS.relative(POINT_METRIC_DIR),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "best": best["name"],
                "best_heavy_oracle": best["primary_heavy_oracle"],
                "union_heavy_pdp": union_heavy["oracle_pdp_cosine"]["mean"],
                "union_heavy_joint_sh": union_heavy["oracle_joint_sh_cosine"]["mean"],
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
