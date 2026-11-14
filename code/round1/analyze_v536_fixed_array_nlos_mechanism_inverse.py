#!/usr/bin/env python3
"""Re-rank Sionna NLOS mechanisms after fixing the v534/v535 array configuration."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v502_shallow_diffraction_scattering_cross as v502
import analyze_v503_fixed_material_sionna_version_inverse as v503
import analyze_v504_scene_topology_sionna_inverse as v504
import analyze_v535_vertical_array_spacing_sionna_validation as v535
from project_paths import PATHS


VERSION_NUMBER = 536
VERSION = "v536_fixed_array_nlos_mechanism_inverse"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
POINT_DIR = CACHE_DIR / "point_metrics"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"

FREQUENCY_HZ = 6.693e9
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING_HZ = 30_000.0
SAMPLES_PER_SRC = 1_000_000
MAX_NUM_PATHS_PER_SRC = 200_000
SEED = 42
SPEED_OF_LIGHT = 299_792_458.0
TINY = 1e-30


@dataclass(frozen=True)
class MechanismConfig:
    name: str
    max_depth: int
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    scattering_coefficient: float = 0.0
    scattering_scope: str = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("balanced", "all"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--configs", nargs="*")
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configurations() -> list[MechanismConfig]:
    configs = [
        MechanismConfig(f"d{depth}_base", depth)
        for depth in (3, 5, 10, 16, 32)
    ]
    configs.extend(
        MechanismConfig(f"d{depth}_diff", depth, diffraction=True)
        for depth in (3, 5, 10, 16)
    )
    configs.extend(
        MechanismConfig(
            f"d{depth}_edge",
            depth,
            diffraction=True,
            edge_diffraction=True,
        )
        for depth in (3, 5, 10, 16)
    )
    configs.extend(
        [
            MechanismConfig("d10_sc005", 10, scattering_coefficient=0.05),
            MechanismConfig("d10_sc015", 10, scattering_coefficient=0.15),
            MechanismConfig("d10_sc030", 10, scattering_coefficient=0.30),
            MechanismConfig("d16_sc015", 16, scattering_coefficient=0.15),
            MechanismConfig(
                "d5_diff_sc015", 5, diffraction=True, scattering_coefficient=0.15
            ),
            MechanismConfig(
                "d10_diff_sc015", 10, diffraction=True, scattering_coefficient=0.15
            ),
            MechanismConfig(
                "d5_edge_sc015",
                5,
                diffraction=True,
                edge_diffraction=True,
                scattering_coefficient=0.15,
            ),
            MechanismConfig(
                "d10_edge_sc015",
                10,
                diffraction=True,
                edge_diffraction=True,
                scattering_coefficient=0.15,
            ),
        ]
    )
    return configs


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def report_path(stage: str):
    return REPORT_DIR / f"v536_{stage}.json"


def config_report_path(config: MechanismConfig, stage: str):
    return REPORT_DIR / "configs" / f"{safe_name(config.name)}_{stage}.json"


def metric_path(config: MechanismConfig, stage: str):
    return POINT_DIR / f"{safe_name(config.name)}_{stage}.npz"


def tx_target(mi):
    radius = 140.0
    azimuth = math.radians(-29.70)
    tilt = math.radians(-7.65)
    return mi.Point3f(
        float(v501.BS_POSITION[0] + radius * math.cos(azimuth)),
        float(v501.BS_POSITION[1] + radius * math.sin(azimuth)),
        float(v501.BS_POSITION[2] + radius * math.tan(tilt)),
    )


def remove_geometric_los(
    channel: np.ndarray,
    positions: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(
        positions.astype(np.float64) - np.asarray(v501.BS_POSITION, dtype=np.float64)[None, :],
        axis=1,
    )
    delays = distances / SPEED_OF_LIGHT
    atom = np.exp(
        -2j * np.pi * delays[:, None] * np.asarray(frequencies, dtype=np.float64)[None, :]
    ).astype(np.complex64)
    coefficients = np.mean(
        channel * np.conj(atom[:, None, None, :]), axis=-1, keepdims=True
    )
    return (channel - coefficients * atom[:, None, None, :]).astype(np.complex64)


def run_config(
    config: MechanismConfig,
    stage: str,
    official_indices: np.ndarray,
    positions: np.ndarray,
    truth_all: np.ndarray,
    truth_cache: dict[str, np.ndarray],
    batch_size: int,
    mi,
    dr,
    rt_api,
) -> dict[str, object]:
    output_report = config_report_path(config, stage)
    output_metrics = metric_path(config, stage)
    if output_report.is_file() and output_metrics.is_file():
        return json.loads(output_report.read_text(encoding="utf-8"))

    load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies = rt_api
    scene = load_scene(str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True)
    scene.frequency = FREQUENCY_HZ
    v502.set_scattering(scene, config)
    materials = v504.material_audit(scene)
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=4.0,
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
        name="tx_v536",
        position=mi.Point3f(*[float(value) for value in v501.BS_POSITION]),
        look_at=tx_target(mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()

    base_metric_names = (
        "pdp",
        "leaked_full",
        "leaked_joint",
        "conventional_pas",
        "horizontal_pas",
        "vertical_pas",
        "complex_coherence",
        "pred_log_power",
        "truth_log_power",
    )
    metrics = {
        name: np.empty(len(official_indices), dtype=np.float32)
        for name in base_metric_names
    }
    metrics.update(
        {
            f"residual_{name}": np.empty(len(official_indices), dtype=np.float32)
            for name in base_metric_names
        }
    )
    path_count = np.empty(len(official_indices), dtype=np.int32)
    diffuse_path_count = np.empty(len(official_indices), dtype=np.int32)
    diffraction_path_count = np.empty(len(official_indices), dtype=np.int32)
    receiver_names: list[str] = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(official_indices), batch_size)):
        end = min(start + batch_size, len(official_indices))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        batch_ids = official_indices[start:end]
        for point_id in batch_ids:
            name = f"rx_v536_{int(point_id):05d}"
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
            max_num_paths_per_src=MAX_NUM_PATHS_PER_SRC,
            samples_per_src=SAMPLES_PER_SRC,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=config.scattering_coefficient > 0.0,
            refraction=True,
            diffraction=config.diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=config.diffraction_lit_region,
            seed=SEED + batch_index,
        )
        interactions = np.asarray(paths.interactions.numpy(), dtype=np.int16)
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        contains_diffuse = np.any(interactions == 2, axis=0)
        contains_diffraction = np.any(interactions == 8, axis=0)
        path_count[start:end] = np.sum(valid, axis=(1, 2))
        diffuse_path_count[start:end] = np.sum(valid & contains_diffuse, axis=(1, 2))
        diffraction_path_count[start:end] = np.sum(valid & contains_diffraction, axis=(1, 2))
        try:
            raw = paths.cfr(
                mi.TensorXf(frequencies),
                normalize_delays=False,
                normalize=False,
                out_type="numpy",
            )
        except RuntimeError as error:
            if "exceeds the limit of 2^32" not in str(error):
                raise
            raw = v502.chunked_cfr(paths, frequencies, mi)
        prediction = v501.convert_cfr(np.asarray(raw), len(batch_ids))
        truth = np.asarray(truth_all[batch_ids], dtype=np.complex64)
        values = v535.batch_metrics(prediction, truth, truth_cache, batch_ids)
        residual_prediction = remove_geometric_los(
            prediction, positions[batch_ids], frequencies
        )
        residual_truth = remove_geometric_los(truth, positions[batch_ids], frequencies)
        # Residual truth is not in the shared cache, so compute both sides directly.
        residual_values = direct_metrics(residual_prediction, residual_truth)
        for name, value in values.items():
            metrics[name][start:end] = value
        for name, value in residual_values.items():
            metrics[f"residual_{name}"][start:end] = value
        del (
            paths,
            interactions,
            valid,
            raw,
            prediction,
            truth,
            values,
            residual_prediction,
            residual_truth,
            residual_values,
        )
        dr.sync_thread()
        if end == len(official_indices) or (batch_index + 1) % 25 == 0:
            print(
                f"{config.name}/{stage}: {end}/{len(official_indices)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    for name in receiver_names:
        scene.remove(name)
    dr.sync_thread()
    flush = getattr(dr, "flush_malloc_cache", None)
    if callable(flush):
        flush()

    POINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_metrics,
        official_indices=official_indices,
        path_count=path_count,
        diffuse_path_count=diffuse_path_count,
        diffraction_path_count=diffraction_path_count,
        **metrics,
    )
    result = {
        "method": VERSION,
        "stage": stage,
        "config": asdict(config),
        "point_count": int(len(official_indices)),
        "materials": materials,
        "metric_mean": {name: float(np.mean(value)) for name, value in metrics.items()},
        "path_count": {
            "mean": float(np.mean(path_count)),
            "diffuse_mean": float(np.mean(diffuse_path_count)),
            "diffraction_mean": float(np.mean(diffraction_path_count)),
            "zero_count": int(np.sum(path_count == 0)),
        },
        "seconds": float(time.time() - started),
        "point_metrics": PATHS.relative(output_metrics),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{config.name}: residual_pdp={result['metric_mean']['residual_pdp']:.6f} "
        f"residual_pas={result['metric_mean']['residual_conventional_pas']:.6f} "
        f"residual_coherence={result['metric_mean']['residual_complex_coherence']:.6f}",
        flush=True,
    )
    return result


def direct_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    count = len(prediction)
    pred_delay = np.abs(np.fft.ifft(prediction, axis=-1)).astype(np.float32) ** 2
    truth_delay = np.abs(np.fft.ifft(truth, axis=-1)).astype(np.float32) ** 2
    pred_delay = v535.unit_normalize(pred_delay, (3,))
    truth_delay = v535.unit_normalize(truth_delay, (3,))
    pdp = v535.cosine_from_unit(pred_delay, truth_delay, (3,)).mean(axis=(1, 2))

    pred6 = prediction.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    truth6 = truth.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    pred_leaked_power = np.abs(
        np.fft.fft2(pred6.transpose(0, 1, 5, 2, 4, 3), axes=(2, 3))
    ).astype(np.float32) ** 2
    truth_leaked_power = np.abs(
        np.fft.fft2(truth6.transpose(0, 1, 5, 2, 4, 3), axes=(2, 3))
    ).astype(np.float32) ** 2
    pred_leaked = v535.unit_normalize(pred_leaked_power, (2, 3))
    truth_leaked = v535.unit_normalize(truth_leaked_power, (2, 3))
    leaked_full = v535.cosine_from_unit(pred_leaked, truth_leaked, (2, 3)).mean(
        axis=(1, 2, 3)
    )
    pred_joint = v535.unit_normalize(
        np.sum(pred_leaked_power, axis=(1, 4, 5), dtype=np.float64), (1, 2)
    )
    truth_joint = v535.unit_normalize(
        np.sum(truth_leaked_power, axis=(1, 4, 5), dtype=np.float64), (1, 2)
    )
    leaked_joint = v535.cosine_from_unit(pred_joint, truth_joint, (1, 2))

    pred_conventional = np.abs(np.fft.fft2(pred6, axes=(2, 3))).astype(np.float32) ** 2
    truth_conventional = np.abs(np.fft.fft2(truth6, axes=(2, 3))).astype(np.float32) ** 2
    pred_conventional = np.sum(pred_conventional, axis=(1, 4, 5), dtype=np.float64)
    truth_conventional = np.sum(truth_conventional, axis=(1, 4, 5), dtype=np.float64)
    conventional_pas = v535.cosine_from_unit(
        v535.unit_normalize(pred_conventional, (1, 2)),
        v535.unit_normalize(truth_conventional, (1, 2)),
        (1, 2),
    )
    horizontal_pas = v535.cosine_from_unit(
        v535.unit_normalize(np.sum(pred_conventional, axis=2), (1,)),
        v535.unit_normalize(np.sum(truth_conventional, axis=2), (1,)),
        (1,),
    )
    vertical_pas = v535.cosine_from_unit(
        v535.unit_normalize(np.sum(pred_conventional, axis=1), (1,)),
        v535.unit_normalize(np.sum(truth_conventional, axis=1), (1,)),
        (1,),
    )
    pred_flat = prediction.reshape(count, -1)
    truth_flat = truth.reshape(count, -1)
    pred_power = np.sum(np.abs(pred_flat) ** 2, axis=1, dtype=np.float64)
    truth_power = np.sum(np.abs(truth_flat) ** 2, axis=1, dtype=np.float64)
    dot = np.sum(np.conj(pred_flat) * truth_flat, axis=1, dtype=np.complex128)
    coherence = np.abs(dot) / np.maximum(np.sqrt(pred_power * truth_power), TINY)
    return {
        "pdp": pdp.astype(np.float32),
        "leaked_full": leaked_full.astype(np.float32),
        "leaked_joint": leaked_joint.astype(np.float32),
        "conventional_pas": conventional_pas.astype(np.float32),
        "horizontal_pas": horizontal_pas.astype(np.float32),
        "vertical_pas": vertical_pas.astype(np.float32),
        "complex_coherence": coherence.astype(np.float32),
        "pred_log_power": np.log(np.maximum(pred_power, TINY)).astype(np.float32),
        "truth_log_power": np.log(np.maximum(truth_power, TINY)).astype(np.float32),
    }


def scope_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "positive_fraction": float(np.mean(selected > 0.0)),
    }


def analyze(
    results: list[dict[str, object]],
    regimes_all: np.ndarray,
    folds_all: np.ndarray,
    blocked_all: np.ndarray,
) -> dict[str, object]:
    point_data: dict[str, dict[str, np.ndarray]] = {}
    official_indices = None
    metric_names = (
        "pdp",
        "leaked_full",
        "conventional_pas",
        "complex_coherence",
        "residual_pdp",
        "residual_leaked_full",
        "residual_conventional_pas",
        "residual_complex_coherence",
    )
    for result in results:
        with np.load(PATHS.root / result["point_metrics"]) as data:
            current_indices = np.asarray(data["official_indices"], dtype=np.int64)
            if official_indices is None:
                official_indices = current_indices
            elif not np.array_equal(official_indices, current_indices):
                raise RuntimeError("v536 configurations used different point indices")
            point_data[result["config"]["name"]] = {
                name: np.asarray(data[name], dtype=np.float64) for name in metric_names
            }
    assert official_indices is not None
    regimes = regimes_all[official_indices]
    folds = folds_all[official_indices]
    blocked = blocked_all[official_indices]
    zero = np.asarray(
        np.load(PATHS.cache_dir(501) / "truth_metric_cache" / "truth_zero_mask.npy", mmap_mode="r")[
            official_indices
        ],
        dtype=bool,
    )
    nonzero = ~zero
    masks = {
        "nonzero": nonzero,
        "blocked": nonzero & (blocked > 0),
        "medium": nonzero & (regimes == 2),
        "heavy": nonzero & (regimes == 3),
        "medium_heavy": nonzero & (regimes >= 2),
    }
    baseline_name = "d16_base"
    baseline = point_data[baseline_name]
    ranking = []
    comparisons = {}
    for result in results:
        name = result["config"]["name"]
        values = point_data[name]
        absolute_score = (
            0.40 * values["residual_pdp"]
            + 0.35 * values["residual_conventional_pas"]
            + 0.25 * values["residual_complex_coherence"]
        )
        fold_scores = [
            float(np.mean(absolute_score[masks["medium_heavy"] & (folds == fold)]))
            for fold in range(5)
        ]
        ranking.append(
            {
                "name": name,
                "config": result["config"],
                "nlos_score_mean": float(np.mean(absolute_score[masks["medium_heavy"]])),
                "nlos_score_robust": float(np.mean(fold_scores) - 0.25 * np.std(fold_scores)),
                "fold_scores": fold_scores,
            }
        )
        deltas = {metric: values[metric] - baseline[metric] for metric in metric_names}
        combined = (
            0.40 * deltas["residual_pdp"]
            + 0.35 * deltas["residual_conventional_pas"]
            + 0.25 * deltas["residual_complex_coherence"]
        )
        fold_delta = [
            float(np.mean(combined[masks["medium_heavy"] & (folds == fold)]))
            for fold in range(5)
        ]
        comparisons[name] = {
            "baseline": baseline_name,
            "metrics": {
                metric: {scope: scope_summary(delta, mask) for scope, mask in masks.items()}
                for metric, delta in deltas.items()
            },
            "combined_delta": {
                scope: scope_summary(combined, mask) for scope, mask in masks.items()
            },
            "spatial_fold_delta": fold_delta,
            "predeclared_pass": bool(
                np.mean(deltas["residual_pdp"][masks["medium_heavy"]]) > 0.0
                and np.mean(deltas["residual_conventional_pas"][masks["medium_heavy"]]) > 0.0
                and np.mean(deltas["residual_complex_coherence"][masks["medium_heavy"]]) > 0.0
                and np.mean(combined[masks["medium_heavy"]]) >= 0.003
                and sum(value > 0.0 for value in fold_delta) >= 4
                and np.mean(combined[masks["medium"]]) > 0.0
                and np.mean(combined[masks["heavy"]]) > 0.0
            ),
        }
    ranking.sort(key=lambda item: item["nlos_score_robust"], reverse=True)
    passed = [item["name"] for item in ranking if comparisons[item["name"]]["predeclared_pass"]]
    return {
        "baseline": baseline_name,
        "score_definition": (
            "medium/heavy after geometric-LOS projection: "
            "0.40*PDP + 0.35*conventional_HV_PAS + 0.25*complex_coherence"
        ),
        "predeclared_gate": (
            "vs d16_base: residual PDP/PAS/coherence all positive, combined >=0.003, "
            "at least 4/5 spatial folds positive, and medium/heavy combined deltas both positive"
        ),
        "ranking": ranking,
        "passed_configs": passed,
        "comparisons": comparisons,
    }


def validate_metrics(results: list[dict[str, object]]) -> None:
    for result in results:
        with np.load(PATHS.root / result["point_metrics"]) as data:
            for name in data.files:
                value = np.asarray(data[name])
                if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                    raise FloatingPointError(f"non-finite {name}: {result['point_metrics']}")


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    if args.configs:
        requested = set(args.configs)
        configs = [config for config in configs if config.name in requested]
        missing = requested - {config.name for config in configs}
        if missing:
            raise ValueError(f"unknown configs: {sorted(missing)}")
    if not any(config.name == "d16_base" for config in configs):
        raise ValueError("d16_base must be included for paired analysis")

    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    truth = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    truth_cache = v501.prepare_truth_cache()
    regimes = v501.regime_codes()
    folds = v501.spatial_fold_codes(positions, regimes)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"]
    if args.stage == "all":
        official_indices = np.arange(len(positions), dtype=np.int64)
    else:
        official_indices = v503.balanced_indices(
            positions,
            regimes,
            np.asarray(truth_cache["log_power"], dtype=np.float32),
            160,
        )

    import drjit as dr
    import mitsuba as mi
    import sionna

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v536 requires Sionna 2.0.1, got {sionna.__version__}")
    frequencies = np.asarray(
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING_HZ)
    ).astype(np.float32)
    rt_api = (load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies)
    if args.force:
        for config in configs:
            config_report_path(config, args.stage).unlink(missing_ok=True)
            metric_path(config, args.stage).unlink(missing_ok=True)

    started = time.time()
    results = [
        run_config(
            config,
            args.stage,
            official_indices,
            positions,
            truth,
            truth_cache,
            args.batch_size,
            mi,
            dr,
            rt_api,
        )
        for config in configs
    ]
    validate_metrics(results)
    analysis = analyze(results, regimes, folds, blocked)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "stage": args.stage,
        "primary_hypothesis": (
            "After fixing the v534/v535 array and orientation, at least one NLOS mechanism should "
            "improve geometric-LOS-projected PDP, conventional PAS, and complex coherence together "
            "across medium/heavy points and spatial folds."
        ),
        "fixed_parameters": {
            "scene": PATHS.relative(SCENE_XML),
            "frequency_hz": FREQUENCY_HZ,
            "subcarrier_spacing_hz": SUBCARRIER_SPACING_HZ,
            "tx_array": "8x16 tr38901 VH, H=0.5 wavelength, V=4 wavelengths",
            "tx_orientation_deg": [-29.70, -7.65],
            "rx_array": "2x1 dipole cross, tracks BS",
            "samples_per_src": SAMPLES_PER_SRC,
            "max_num_paths_per_src": MAX_NUM_PATHS_PER_SRC,
            "seed": SEED,
            "synthetic_array": True,
        },
        "point_count": int(len(official_indices)),
        "official_indices": official_indices.tolist(),
        "configs": results,
        "analysis": analysis,
        "total_seconds": float(time.time() - started),
        "outputs": {
            "report": PATHS.relative(report_path(args.stage)),
            "submission_zip": None,
        },
    }
    report_path(args.stage).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": args.stage,
                "point_count": int(len(official_indices)),
                "passed_configs": analysis["passed_configs"],
                "top5": analysis["ranking"][:5],
                "report": PATHS.relative(report_path(args.stage)),
                "total_seconds": report["total_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
