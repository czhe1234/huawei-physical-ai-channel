#!/usr/bin/env python3
"""Test the fixed-material synthetic-array x edge-diffraction interaction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import time

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v503_fixed_material_sionna_version_inverse as v503
import analyze_v504_scene_topology_sionna_inverse as v504
from project_paths import PATHS


VERSION_NUMBER = 513
VERSION = "v513_fixed_material_array_edge_cross"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
POINT_DIR = CACHE_DIR / "point_metrics"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
FREQUENCY_HZ = 6.693e9
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
MAX_DEPTH = 16
DEFAULT_SAMPLES = 3_000_000
DEFAULT_MAX_PATHS = 100_000
DEFAULT_SEED = 42


@dataclass(frozen=True)
class CrossConfig:
    name: str
    synthetic_array: bool
    edge_diffraction: bool
    samples_per_src: int = DEFAULT_SAMPLES
    max_num_paths_per_src: int = DEFAULT_MAX_PATHS
    seed: int = DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--point-limit", type=int)
    parser.add_argument("--budget-cap", type=int)
    parser.add_argument("--max-paths", type=int)
    parser.add_argument("--config-limit", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configurations() -> list[CrossConfig]:
    return [
        CrossConfig("syn1_edge0_d16_3m_max100k_seed42", True, False),
        CrossConfig("syn0_edge0_d16_3m_max100k_seed42", False, False),
        CrossConfig("syn1_edge1_d16_3m_max100k_seed42", True, True),
        CrossConfig("syn0_edge1_d16_3m_max100k_seed42", False, True),
    ]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def config_report_path(config: CrossConfig, point_count: int):
    return REPORT_DIR / "configs" / f"{safe_name(config.name)}_points{point_count}.json"


def point_metric_path(config: CrossConfig, point_count: int):
    return POINT_DIR / f"{safe_name(config.name)}_points{point_count}.npz"


def stage_report_path(stage: str):
    return REPORT_DIR / f"v513_{stage}.json"


def run_config(
    config: CrossConfig,
    indices: np.ndarray,
    positions: np.ndarray,
    truth: np.ndarray,
    regime_codes: np.ndarray,
    batch_size: int,
    mi,
    dr,
    rt_api,
) -> dict[str, object]:
    report_path = config_report_path(config, len(indices))
    metric_path = point_metric_path(config, len(indices))
    if report_path.is_file() and metric_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies = rt_api
    scene = load_scene(
        str(SCENE_XML),
        merge_shapes=True,
        remove_duplicate_vertices=True,
    )
    scene.frequency = FREQUENCY_HZ
    materials = v504.material_audit(scene)
    if set(materials) != {"concrete", "marble", "metal", "wood"}:
        raise RuntimeError(f"unexpected canonical materials: {materials}")
    if any(
        value["scattering_coefficient"] != 0.0
        or value["xpd_coefficient"] != 0.0
        for value in materials.values()
    ):
        raise RuntimeError(f"fixed material audit failed: {materials}")

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
        name=config.name,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-9.53,
        rx_orientation="track_bs",
        synthetic_array=config.synthetic_array,
        frequency_hz=FREQUENCY_HZ,
        max_depth=MAX_DEPTH,
        diffraction=config.edge_diffraction,
        edge_diffraction=config.edge_diffraction,
        samples_per_src=config.samples_per_src,
        max_num_paths_per_src=config.max_num_paths_per_src,
        seed=config.seed,
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*[float(value) for value in v501.BS_POSITION]),
        look_at=v501.tx_target(search, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()

    metric_names = (
        "pdp",
        "leaked_full",
        "complex_coherence",
        "pred_log_power",
        "truth_log_power",
    )
    metric_parts = {name: [] for name in metric_names}
    path_count_parts: list[np.ndarray] = []
    receiver_names: list[str] = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(indices), batch_size)):
        end = min(start + batch_size, len(indices))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        batch_ids = indices[start:end]
        for point_id in batch_ids:
            name = f"rx_{int(point_id):05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(
                        *[float(value) for value in positions[point_id]]
                    ),
                    look_at=tx,
                )
            )
            receiver_names.append(name)

        paths = solver(
            scene,
            max_depth=MAX_DEPTH,
            max_num_paths_per_src=config.max_num_paths_per_src,
            samples_per_src=config.samples_per_src,
            synthetic_array=config.synthetic_array,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=config.edge_diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=True,
            seed=config.seed + batch_index,
        )
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        path_count_parts.append(
            valid.reshape(len(batch_ids), -1).sum(axis=1).astype(np.int32)
        )
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        prediction = v501.convert_cfr(np.asarray(raw), len(batch_ids))
        batch_truth = np.asarray(truth[batch_ids], dtype=np.complex64)
        metrics = v503.channel_metrics(prediction, batch_truth)
        for name in metric_names:
            metric_parts[name].append(metrics[name])

        del paths, valid, raw, prediction, batch_truth, metrics
        dr.sync_thread()
        if end == len(indices) or (batch_index + 1) % 10 == 0:
            print(
                f"{config.name}: {end}/{len(indices)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    for name in receiver_names:
        scene.remove(name)
    dr.sync_thread()
    flush = getattr(dr, "flush_malloc_cache", None)
    if callable(flush):
        flush()

    metrics = {name: np.concatenate(parts) for name, parts in metric_parts.items()}
    path_counts = np.concatenate(path_count_parts)
    POINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        metric_path,
        indices=indices,
        regimes=regime_codes[indices],
        valid_path_count=path_counts,
        **metrics,
    )
    summary = v503.summarize_metrics(metrics, regime_codes[indices])
    result = {
        "method": VERSION,
        "config": asdict(config),
        "point_count": int(len(indices)),
        "indices": indices.tolist(),
        "scene_xml": PATHS.relative(SCENE_XML),
        "materials": materials,
        "summary": summary,
        "selection": float(summary["overall"]["selection"]),
        "valid_path_count": {
            "mean": float(np.mean(path_counts)),
            "median": float(np.median(path_counts)),
            "p90": float(np.quantile(path_counts, 0.90)),
            "zero_count": int(np.sum(path_counts == 0)),
        },
        "seconds": float(time.time() - started),
        "point_metrics": PATHS.relative(metric_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{config.name}: selection={result['selection']:.6f} "
        f"pdp={summary['overall']['pdp_mean']:.6f} "
        f"pas={summary['overall']['leaked_full_mean']:.6f} "
        f"coh={summary['overall']['complex_coherence_mean']:.6f}",
        flush=True,
    )
    return result


def mean_for_mask(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def factorial_analysis(
    results: list[dict[str, object]],
    regime_codes: np.ndarray,
) -> dict[str, object]:
    by_key = {
        (
            bool(result["config"]["synthetic_array"]),
            bool(result["config"]["edge_diffraction"]),
        ): result
        for result in results
    }
    required = ((True, False), (False, False), (True, True), (False, True))
    if any(key not in by_key for key in required):
        return {"complete": False, "reason": "all four factorial cells are required"}

    point_data = {}
    indices = None
    regimes = None
    for key, result in by_key.items():
        data = np.load(PATHS.root / result["point_metrics"])
        current_indices = np.asarray(data["indices"], dtype=np.int64)
        if indices is None:
            indices = current_indices
            regimes = np.asarray(data["regimes"], dtype=np.int8)
        elif not np.array_equal(indices, current_indices):
            raise ValueError("factorial configurations used different point indices")
        point_data[key] = {
            name: np.asarray(data[name], dtype=np.float64)
            for name in ("pdp", "leaked_full", "complex_coherence")
        }

    assert indices is not None and regimes is not None
    t0 = point_data[(True, False)]
    f0 = point_data[(False, False)]
    t1 = point_data[(True, True)]
    f1 = point_data[(False, True)]
    masks = {
        "overall": np.ones(len(indices), dtype=bool),
        "clear": regimes == 0,
        "light": regimes == 1,
        "medium": regimes == 2,
        "heavy": regimes == 3,
    }
    analysis: dict[str, object] = {"complete": True, "by_scope": {}}
    for scope, mask in masks.items():
        if not np.any(mask):
            continue
        scope_result = {"count": int(np.sum(mask)), "metrics": {}}
        for metric in ("pdp", "leaked_full", "complex_coherence"):
            cell_means = {
                "synthetic_true_edge_off": mean_for_mask(t0[metric], mask),
                "synthetic_false_edge_off": mean_for_mask(f0[metric], mask),
                "synthetic_true_edge_on": mean_for_mask(t1[metric], mask),
                "synthetic_false_edge_on": mean_for_mask(f1[metric], mask),
            }
            array_false_main = 0.5 * (
                mean_for_mask(f0[metric] - t0[metric], mask)
                + mean_for_mask(f1[metric] - t1[metric], mask)
            )
            edge_main = 0.5 * (
                mean_for_mask(t1[metric] - t0[metric], mask)
                + mean_for_mask(f1[metric] - f0[metric], mask)
            )
            interaction_values = (f1[metric] - f0[metric]) - (
                t1[metric] - t0[metric]
            )
            combined_values = f1[metric] - t0[metric]
            scope_result["metrics"][metric] = {
                "cell_means": cell_means,
                "synthetic_false_main_effect": float(array_false_main),
                "edge_main_effect": float(edge_main),
                "factorial_interaction": mean_for_mask(interaction_values, mask),
                "combined_vs_baseline": mean_for_mask(combined_values, mask),
                "combined_positive_fraction": float(np.mean(combined_values[mask] > 0.0)),
            }
        analysis["by_scope"][scope] = scope_result

    overall = analysis["by_scope"]["overall"]["metrics"]
    heavy = analysis["by_scope"]["heavy"]["metrics"]
    materially_improves_both = (
        overall["pdp"]["combined_vs_baseline"] >= 0.005
        and overall["leaked_full"]["combined_vs_baseline"] >= 0.005
        and heavy["pdp"]["combined_vs_baseline"] >= 0.010
        and heavy["leaked_full"]["combined_vs_baseline"] >= 0.010
    )
    analysis["decision"] = {
        "materially_improves_both_pdp_and_pas": bool(materially_improves_both),
        "predeclared_thresholds": {
            "overall_pdp": 0.005,
            "overall_pas": 0.005,
            "heavy_pdp": 0.010,
            "heavy_pas": 0.010,
        },
        "next_step": (
            "retain the interaction as an official-configuration candidate"
            if materially_improves_both
            else "stop simple global PathSolver switch search and audit complex path coefficients/postprocessing"
        ),
    }
    return analysis


def validate_finite(paths: list) -> None:
    for path in paths:
        with np.load(path) as data:
            for name in data.files:
                values = np.asarray(data[name])
                if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite values in {path}: {name}")


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant(args.mitsuba_variant)
    import sionna
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    if sionna.__version__ != "2.0.1":
        raise RuntimeError(f"v513 requires Sionna 2.0.1, got {sionna.__version__}")
    frequencies = np.asarray(
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
    ).astype(np.float32)
    rt_api = (load_scene, PlanarArray, Receiver, Transmitter, PathSolver, frequencies)

    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    truth = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    regime_codes = v501.regime_codes()
    truth_log_power = np.load(
        PATHS.cache_dir(501) / "truth_metric_cache" / "truth_log_power.float32.npy",
        mmap_mode="r",
    )
    default_limit = 8 if args.stage == "smoke" else 160
    point_limit = min(args.point_limit or default_limit, len(positions))
    indices = v503.balanced_indices(
        positions,
        regime_codes,
        truth_log_power,
        point_limit,
    )

    configs = configurations()
    if args.max_paths is not None:
        configs = [
            replace(
                config,
                name=f"{config.name}_paths{args.max_paths}",
                max_num_paths_per_src=args.max_paths,
            )
            for config in configs
        ]
    if args.budget_cap is not None:
        configs = [
            replace(
                config,
                name=f"{config.name}_cap{args.budget_cap}",
                samples_per_src=min(config.samples_per_src, args.budget_cap),
                max_num_paths_per_src=min(config.max_num_paths_per_src, args.budget_cap),
            )
            for config in configs
        ]
    if args.config_limit is not None:
        configs = configs[: args.config_limit]
    if args.force:
        for config in configs:
            config_report_path(config, len(indices)).unlink(missing_ok=True)
            point_metric_path(config, len(indices)).unlink(missing_ok=True)

    results = [
        run_config(
            config,
            indices,
            positions,
            truth,
            regime_codes,
            args.batch_size,
            mi,
            dr,
            rt_api,
        )
        for config in configs
    ]
    factorial = factorial_analysis(results, regime_codes)
    metric_paths = [
        point_metric_path(config, len(indices))
        for config in configs
        if point_metric_path(config, len(indices)).is_file()
    ]
    validate_finite(metric_paths)

    ranking = sorted(results, key=lambda row: row["selection"], reverse=True)
    report = {
        "method": VERSION,
        "status": "analysis_complete" if args.stage == "full" else "smoke_complete",
        "stage": args.stage,
        "primary_hypothesis": (
            "With canonical fixed materials, non-synthetic array evaluation and edge "
            "diffraction may jointly recover the candidate-centered angular spread that "
            "neither switch explains alone."
        ),
        "environment": {
            "sionna": sionna.__version__,
            "mitsuba_variant": mi.variant(),
            "visible_gpu_count": 1,
        },
        "fixed_parameters": {
            "scene": PATHS.relative(SCENE_XML),
            "materials": "canonical fixed Sionna Etoile ITU defaults",
            "frequency_hz": FREQUENCY_HZ,
            "subcarrier_spacing_hz": SUBCARRIER_SPACING,
            "tx_array": "8x16 tr38901 VH 0.5lambda",
            "rx_array": "2x1 dipole cross 0.5lambda",
            "tx_azimuth_deg": -30.0,
            "tx_tilt_deg": -9.53,
            "rx_orientation": "track_bs",
            "max_depth": MAX_DEPTH,
            "samples_per_src": configs[0].samples_per_src if configs else None,
            "max_num_paths_per_src": configs[0].max_num_paths_per_src if configs else None,
            "path_buffer_constraint": (
                "100k retained candidates because synthetic_array=False expands the 128 TX "
                "elements and a 3M buffer exceeds Dr.Jit's 2^32 tensor-entry limit; an 8-point "
                "3M-ray convergence check against a 200k buffer changed PDP/PAS by less than "
                "0.001, while a 50k buffer produced material truncation"
            ),
            "seed": DEFAULT_SEED,
            "los": True,
            "specular_reflection": True,
            "refraction": True,
            "diffuse_reflection": False,
            "scattering": False,
        },
        "factorial_variables": {
            "synthetic_array": [True, False],
            "edge_diffraction": [False, True],
        },
        "point_count": int(len(indices)),
        "indices": indices.tolist(),
        "ranking": ranking,
        "factorial": factorial,
        "outputs": {
            "report": PATHS.relative(stage_report_path(args.stage)),
            "point_metrics": [PATHS.relative(path) for path in metric_paths],
            "submission_zip": None,
        },
    }
    stage_report_path(args.stage).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": args.stage,
                "point_count": int(len(indices)),
                "best": ranking[0]["config"]["name"] if ranking else None,
                "best_selection": ranking[0]["selection"] if ranking else None,
                "decision": factorial.get("decision"),
                "report": PATHS.relative(stage_report_path(args.stage)),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
