#!/usr/bin/env python3
"""Validate the v534 LOS-inferred TX spacing in paired Sionna simulations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
from scipy.fft import fft2, ifft

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v503_fixed_material_sionna_version_inverse as v503
import analyze_v504_scene_topology_sionna_inverse as v504
from project_paths import PATHS


VERSION_NUMBER = 535
VERSION = "v535_vertical_array_spacing_sionna_validation"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
POINT_DIR = CACHE_DIR / "point_metrics"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"

FREQUENCY_HZ = 6.693e9
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING_HZ = 30_000.0
MAX_DEPTH = 16
SAMPLES_PER_SRC = 3_000_000
MAX_NUM_PATHS_PER_SRC = 100_000
SEED = 42
TINY = 1e-30


@dataclass(frozen=True)
class ArrayConfig:
    name: str
    vertical_spacing: float
    tx_azimuth_deg: float
    tx_tilt_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "balanced", "all"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configurations() -> list[ArrayConfig]:
    return [
        ArrayConfig("old_v0p5_old_orientation", 0.5, -30.0, -9.53),
        ArrayConfig("new_v4_old_orientation", 4.0, -30.0, -9.53),
        ArrayConfig("old_v0p5_new_orientation", 0.5, -29.70, -7.65),
        ArrayConfig("new_v4_new_orientation", 4.0, -29.70, -7.65),
    ]


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def report_path(stage: str):
    return REPORT_DIR / f"v535_{stage}.json"


def config_report_path(config: ArrayConfig, stage: str):
    return REPORT_DIR / "configs" / f"{safe_name(config.name)}_{stage}.json"


def metric_path(config: ArrayConfig, stage: str):
    return POINT_DIR / f"{safe_name(config.name)}_{stage}.npz"


def unit_normalize(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = np.sqrt(np.sum(value * value, axis=axes, keepdims=True, dtype=np.float64))
    return value / np.maximum(norm, TINY)


def cosine_from_unit(first: np.ndarray, second: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    return np.clip(np.sum(first * second, axis=axes, dtype=np.float64), 0.0, 1.0)


def batch_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    truth_cache: dict[str, np.ndarray],
    official_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(prediction)
    pred_delay = np.abs(ifft(prediction, axis=-1, workers=1)).astype(np.float32) ** 2
    pred_delay_unit = unit_normalize(pred_delay, (3,))
    truth_delay_unit = np.asarray(truth_cache["pdp_unit"][official_indices], dtype=np.float32)
    pdp = cosine_from_unit(pred_delay_unit, truth_delay_unit, (3,)).mean(axis=(1, 2))

    pred6 = prediction.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    truth6 = truth.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)

    pred_leaked = pred6.transpose(0, 1, 5, 2, 4, 3)
    pred_leaked = np.abs(fft2(pred_leaked, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    pred_leaked_unit = unit_normalize(pred_leaked, (2, 3))
    truth_leaked_unit = np.asarray(truth_cache["leaked_unit"][official_indices], dtype=np.float32)
    leaked_full = cosine_from_unit(pred_leaked_unit, truth_leaked_unit, (2, 3)).mean(axis=(1, 2, 3))

    pred_joint = np.sum(pred_leaked, axis=(1, 4, 5), dtype=np.float64)
    pred_joint_unit = unit_normalize(pred_joint, (1, 2))
    truth_joint_unit = np.asarray(
        truth_cache["leaked_collapsed_unit"][official_indices], dtype=np.float32
    )
    leaked_joint = cosine_from_unit(pred_joint_unit, truth_joint_unit, (1, 2))

    pred_conventional = np.abs(fft2(pred6, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    pred_conventional = np.sum(pred_conventional, axis=(1, 4, 5), dtype=np.float64)
    pred_conventional_unit = unit_normalize(pred_conventional, (1, 2))
    truth_conventional_unit = np.asarray(
        truth_cache["conventional_unit"][official_indices], dtype=np.float32
    )
    conventional_pas = cosine_from_unit(
        pred_conventional_unit, truth_conventional_unit, (1, 2)
    )

    truth_conventional = np.abs(fft2(truth6, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    truth_conventional = np.sum(truth_conventional, axis=(1, 4, 5), dtype=np.float64)
    pred_h = unit_normalize(np.sum(pred_conventional, axis=2), (1,))
    truth_h = unit_normalize(np.sum(truth_conventional, axis=2), (1,))
    pred_v = unit_normalize(np.sum(pred_conventional, axis=1), (1,))
    truth_v = unit_normalize(np.sum(truth_conventional, axis=1), (1,))
    horizontal_pas = cosine_from_unit(pred_h, truth_h, (1,))
    vertical_pas = cosine_from_unit(pred_v, truth_v, (1,))

    pred_flat = prediction.reshape(count, -1)
    truth_flat = truth.reshape(count, -1)
    pred_power = np.sum(np.abs(pred_flat) ** 2, axis=1, dtype=np.float64)
    truth_power = np.sum(np.abs(truth_flat) ** 2, axis=1, dtype=np.float64)
    dot = np.sum(np.conj(pred_flat) * truth_flat, axis=1, dtype=np.complex128)
    complex_coherence = np.abs(dot) / np.maximum(np.sqrt(pred_power * truth_power), TINY)

    return {
        "pdp": pdp.astype(np.float32),
        "leaked_full": leaked_full.astype(np.float32),
        "leaked_joint": leaked_joint.astype(np.float32),
        "conventional_pas": conventional_pas.astype(np.float32),
        "horizontal_pas": horizontal_pas.astype(np.float32),
        "vertical_pas": vertical_pas.astype(np.float32),
        "complex_coherence": complex_coherence.astype(np.float32),
        "pred_log_power": np.log(np.maximum(pred_power, TINY)).astype(np.float32),
        "truth_log_power": np.log(np.maximum(truth_power, TINY)).astype(np.float32),
    }


def tx_target(config: ArrayConfig, mi):
    radius = 140.0
    azimuth = math.radians(config.tx_azimuth_deg)
    tilt = math.radians(config.tx_tilt_deg)
    return mi.Point3f(
        float(v501.BS_POSITION[0] + radius * math.cos(azimuth)),
        float(v501.BS_POSITION[1] + radius * math.sin(azimuth)),
        float(v501.BS_POSITION[2] + radius * math.tan(tilt)),
    )


def run_config(
    config: ArrayConfig,
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
    materials = v504.material_audit(scene)
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=config.vertical_spacing,
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
        name="tx_v535",
        position=mi.Point3f(*[float(value) for value in v501.BS_POSITION]),
        look_at=tx_target(config, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()

    metric_names = (
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
    metrics = {name: np.empty(len(official_indices), dtype=np.float32) for name in metric_names}
    path_count = np.empty(len(official_indices), dtype=np.int32)
    receiver_names: list[str] = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(official_indices), batch_size)):
        end = min(start + batch_size, len(official_indices))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        batch_ids = official_indices[start:end]
        for point_id in batch_ids:
            name = f"rx_v535_{int(point_id):05d}"
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
            seed=SEED + batch_index,
        )
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        path_count[start:end] = valid.reshape(len(batch_ids), -1).sum(axis=1)
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        prediction = v501.convert_cfr(np.asarray(raw), len(batch_ids))
        truth = np.asarray(truth_all[batch_ids], dtype=np.complex64)
        values = batch_metrics(prediction, truth, truth_cache, batch_ids)
        for name, value in values.items():
            metrics[name][start:end] = value
        del paths, valid, raw, prediction, truth, values
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
        **metrics,
    )
    result = {
        "method": VERSION,
        "stage": stage,
        "config": asdict(config),
        "point_count": int(len(official_indices)),
        "materials": materials,
        "metric_mean": {name: float(np.mean(metrics[name])) for name in metric_names},
        "path_count": {
            "mean": float(np.mean(path_count)),
            "median": float(np.median(path_count)),
            "zero_count": int(np.sum(path_count == 0)),
        },
        "seconds": float(time.time() - started),
        "point_metrics": PATHS.relative(output_metrics),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{config.name}: conventional={result['metric_mean']['conventional_pas']:.6f} "
        f"V={result['metric_mean']['vertical_pas']:.6f} "
        f"leaked={result['metric_mean']['leaked_full']:.6f} "
        f"pdp={result['metric_mean']['pdp']:.6f}",
        flush=True,
    )
    return result


def summarize_scope(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values)[mask]
    return {
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "positive_fraction": float(np.mean(selected > 0.0)),
    }


def paired_analysis(
    results: list[dict[str, object]],
    regimes_all: np.ndarray,
    folds_all: np.ndarray,
) -> dict[str, object]:
    point_data: dict[str, dict[str, np.ndarray]] = {}
    official_indices = None
    for result in results:
        data = np.load(PATHS.root / result["point_metrics"])
        current_indices = np.asarray(data["official_indices"], dtype=np.int64)
        if official_indices is None:
            official_indices = current_indices
        elif not np.array_equal(official_indices, current_indices):
            raise RuntimeError("v535 configurations used different official point indices")
        point_data[result["config"]["name"]] = {
            name: np.asarray(data[name], dtype=np.float64)
            for name in (
                "pdp",
                "leaked_full",
                "leaked_joint",
                "conventional_pas",
                "horizontal_pas",
                "vertical_pas",
                "complex_coherence",
            )
        }
    assert official_indices is not None
    regimes = regimes_all[official_indices]
    folds = folds_all[official_indices]
    zero = np.load(PATHS.cache_dir(501) / "truth_metric_cache" / "truth_zero_mask.npy", mmap_mode="r")
    nonzero = ~np.asarray(zero[official_indices], dtype=bool)
    masks = {
        "all": np.ones(len(official_indices), dtype=bool),
        "nonzero": nonzero,
        "clear": (regimes == 0) & nonzero,
        "light": (regimes == 1) & nonzero,
        "medium": (regimes == 2) & nonzero,
        "heavy": (regimes == 3) & nonzero,
    }
    comparisons = {
        "spacing_only": ("new_v4_old_orientation", "old_v0p5_old_orientation"),
        "orientation_only_at_old_spacing": ("old_v0p5_new_orientation", "old_v0p5_old_orientation"),
        "orientation_after_spacing": ("new_v4_new_orientation", "new_v4_old_orientation"),
        "joint_new_vs_old": ("new_v4_new_orientation", "old_v0p5_old_orientation"),
    }
    analysis: dict[str, object] = {"comparisons": {}}
    for comparison, (new_name, old_name) in comparisons.items():
        comparison_result: dict[str, object] = {"new": new_name, "old": old_name, "metrics": {}}
        for metric in point_data[new_name]:
            delta = point_data[new_name][metric] - point_data[old_name][metric]
            comparison_result["metrics"][metric] = {
                scope: {"count": int(np.sum(mask)), **summarize_scope(delta, mask)}
                for scope, mask in masks.items()
                if np.any(mask)
            }
            comparison_result["metrics"][metric]["spatial_folds"] = [
                {
                    "fold": int(fold),
                    "count": int(np.sum((folds == fold) & nonzero)),
                    "mean": float(np.mean(delta[(folds == fold) & nonzero])),
                }
                for fold in np.unique(folds)
                if np.any((folds == fold) & nonzero)
            ]
        analysis["comparisons"][comparison] = comparison_result

    spacing = analysis["comparisons"]["spacing_only"]["metrics"]
    conventional_gain = spacing["conventional_pas"]["nonzero"]["mean"]
    fold_gain = [row["mean"] for row in spacing["conventional_pas"]["spatial_folds"]]
    vertical_gain = spacing["vertical_pas"]["nonzero"]["mean"]
    passed = conventional_gain >= 0.02 and vertical_gain >= 0.05 and min(fold_gain) > 0.0
    analysis["decision"] = {
        "pass": bool(passed),
        "predeclared_gate": {
            "nonzero_conventional_pas_gain_min": 0.02,
            "nonzero_vertical_pas_gain_min": 0.05,
            "conventional_pas_spatial_folds": "all positive",
        },
        "observed": {
            "nonzero_conventional_pas_gain": float(conventional_gain),
            "nonzero_vertical_pas_gain": float(vertical_gain),
            "conventional_pas_fold_gains": [float(value) for value in fold_gain],
        },
        "next_step": (
            "run all 2000 official train points with the same paired configurations"
            if passed
            else "reject spacing-only deployment; evaluate the separately declared joint spacing-orientation gate"
        ),
    }

    joint = analysis["comparisons"]["joint_new_vs_old"]["metrics"]
    joint_conventional_gain = joint["conventional_pas"]["nonzero"]["mean"]
    joint_vertical_gain = joint["vertical_pas"]["nonzero"]["mean"]
    joint_pdp_gain = joint["pdp"]["nonzero"]["mean"]
    joint_fold_gain = [row["mean"] for row in joint["conventional_pas"]["spatial_folds"]]
    joint_regime_gain = [
        joint["conventional_pas"][regime]["mean"]
        for regime in ("clear", "light", "medium", "heavy")
        if regime in joint["conventional_pas"]
    ]
    joint_passed = (
        joint_conventional_gain >= 0.10
        and joint_vertical_gain >= 0.15
        and joint_pdp_gain >= -0.01
        and min(joint_fold_gain) > 0.0
        and min(joint_regime_gain) > 0.0
    )
    analysis["joint_decision"] = {
        "pass": bool(joint_passed),
        "secondary_hypothesis": (
            "The v534-inferred 4-wavelength vertical spacing and new TX orientation are coupled; "
            "their joint configuration, rather than spacing alone, should improve conventional PAS "
            "across every spatial fold and propagation regime without materially degrading PDP."
        ),
        "predeclared_gate": {
            "nonzero_conventional_pas_gain_min": 0.10,
            "nonzero_vertical_pas_gain_min": 0.15,
            "nonzero_pdp_gain_min": -0.01,
            "conventional_pas_spatial_folds": "all positive",
            "conventional_pas_regimes": "clear/light/medium/heavy all positive",
        },
        "observed": {
            "nonzero_conventional_pas_gain": float(joint_conventional_gain),
            "nonzero_vertical_pas_gain": float(joint_vertical_gain),
            "nonzero_pdp_gain": float(joint_pdp_gain),
            "conventional_pas_fold_gains": [float(value) for value in joint_fold_gain],
            "conventional_pas_regime_gains": [float(value) for value in joint_regime_gain],
        },
        "next_step": (
            "freeze the joint array configuration and re-audit NLOS mechanisms with LOS parameters fixed"
            if joint_passed
            else "reject the joint array configuration as a global official-generator explanation"
        ),
    }
    return analysis


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

    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    truth = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    truth_cache = v501.prepare_truth_cache()
    regimes = v501.regime_codes()
    folds = v501.spatial_fold_codes(positions, regimes)
    truth_log_power = np.asarray(truth_cache["log_power"], dtype=np.float32)
    if args.stage == "all":
        official_indices = np.arange(len(positions), dtype=np.int64)
    else:
        count = 16 if args.stage == "smoke" else 160
        official_indices = v503.balanced_indices(positions, regimes, truth_log_power, count)

    import drjit as dr
    import mitsuba as mi
    import sionna

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v535 requires Sionna 2.0.1, got {sionna.__version__}")
    frequencies = np.asarray(
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING_HZ)
    ).astype(np.float32)
    rt_api = (load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies)

    configs = configurations()
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
    analysis = paired_analysis(results, regimes, folds)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "stage": args.stage,
        "primary_hypothesis": (
            "Changing only TX vertical spacing from the historical 0.5 wavelength assumption to the v534 "
            "LOS-inferred 4 wavelengths should improve official-train vertical and full conventional PAS in "
            "every spatial fold; orientation is crossed separately to prevent confounding."
        ),
        "secondary_hypothesis": (
            "The balanced-stage interaction indicates that the 4-wavelength spacing and v534-inferred "
            "orientation may be a coupled configuration. This secondary hypothesis is judged separately "
            "and does not overwrite a failure of the spacing-only primary gate."
        ),
        "environment": {
            "sionna": sionna.__version__,
            "mitsuba_variant": mi.variant(),
            "visible_gpu_count": 1,
        },
        "fixed_parameters": {
            "scene": PATHS.relative(SCENE_XML),
            "materials": "canonical Etoile ITU materials, scattering coefficient 0",
            "frequency_hz": FREQUENCY_HZ,
            "subcarrier_spacing_hz": SUBCARRIER_SPACING_HZ,
            "tx_array": "8x16 tr38901 VH, horizontal spacing 0.5 wavelength",
            "rx_array": "2x1 dipole cross, vertical spacing 0.5 wavelength, tracks BS",
            "max_depth": MAX_DEPTH,
            "samples_per_src": SAMPLES_PER_SRC,
            "max_num_paths_per_src": MAX_NUM_PATHS_PER_SRC,
            "synthetic_array": True,
            "seed": SEED,
            "mechanisms": "LOS + specular reflection + refraction; no diffuse/diffraction/edge",
        },
        "factorial_variables": {
            "tx_vertical_spacing_wavelengths": [0.5, 4.0],
            "tx_orientation_deg": [[-30.0, -9.53], [-29.70, -7.65]],
        },
        "point_count": int(len(official_indices)),
        "official_indices": official_indices.tolist(),
        "configs": results,
        "paired_analysis": analysis,
        "total_seconds": float(time.time() - started),
        "outputs": {
            "report": PATHS.relative(report_path(args.stage)),
            "point_metrics": [result["point_metrics"] for result in results],
            "submission_zip": None,
        },
    }
    report_path(args.stage).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "point_count": int(len(official_indices)),
                "spacing_only_decision": analysis["decision"],
                "joint_decision": analysis["joint_decision"],
                "report": PATHS.relative(report_path(args.stage)),
                "total_seconds": report["total_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
