#!/usr/bin/env python3
"""Cross shallow depth, diffraction, and real non-zero diffuse scattering."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import time

import numpy as np

from analyze_v501_full_official_sionna_inverse_calibration import (
    BS_POSITION,
    NUM_SUBCARRIERS,
    SCENE_XML,
    SUBCARRIER_SPACING,
    SearchConfig,
    prepare_truth_cache,
    prediction_metrics,
    regime_codes,
    spatial_fold_codes,
    summarize_config,
    tx_target,
    convert_cfr,
)
from project_paths import PATHS


VERSION = "v502_shallow_diffraction_scattering_cross"
REPORT_DIR = PATHS.version_report_dir(502)
CACHE_DIR = PATHS.cache_dir(502)
POINT_DIR = CACHE_DIR / "point_metrics"
V501_FINAL = PATHS.version_report_dir(501) / "v501_final_seed.json"


@dataclass(frozen=True)
class MechanismConfig:
    name: str
    max_depth: int
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    scattering_coefficient: float = 0.0
    scattering_scope: str = "all"
    samples_per_src: int = 1_000_000
    max_num_paths_per_src: int = 1_000_000
    seed: int = 42
    source_name: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("grid", "refine", "seeds"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--config-limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def config_result_path(config: MechanismConfig) -> Path:
    return REPORT_DIR / "configs" / f"{safe_name(config.name)}.json"


def point_metric_path(config: MechanismConfig) -> Path:
    return POINT_DIR / f"{safe_name(config.name)}.npz"


def stage_report_path(stage: str) -> Path:
    return REPORT_DIR / f"v502_{stage}.json"


def grid_configs() -> list[MechanismConfig]:
    modes = (
        ("base", False, False, True, 0.0),
        ("diff_lit", True, False, True, 0.0),
        ("diff_shadow", True, False, False, 0.0),
        ("edge_lit", True, True, True, 0.0),
        ("scatter010", False, False, True, 0.10),
        ("scatter030", False, False, True, 0.30),
        ("diff_lit_sc010", True, False, True, 0.10),
        ("diff_lit_sc030", True, False, True, 0.30),
        ("diff_shadow_sc030", True, False, False, 0.30),
        ("edge_lit_sc030", True, True, True, 0.30),
    )
    configs = []
    for depth in (3, 5, 8, 12, 16):
        active_modes = modes
        if depth == 12:
            active_modes = modes[:8]
        elif depth == 16:
            active_modes = modes[:6]
        for mode, diffraction, edge, lit_region, scattering in active_modes:
            configs.append(
                MechanismConfig(
                    name=f"grid_d{depth}_{mode}_seed42",
                    max_depth=depth,
                    diffraction=diffraction,
                    edge_diffraction=edge,
                    diffraction_lit_region=lit_region,
                    scattering_coefficient=scattering,
                )
            )
    configs.append(MechanismConfig(name="grid_d32_base_seed42", max_depth=32))
    return configs


def load_report(stage: str) -> dict[str, object]:
    path = stage_report_path(stage)
    if not path.is_file():
        raise FileNotFoundError(f"run {stage} first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def refine_configs() -> list[MechanismConfig]:
    configs = []
    for depth in (10, 12, 14, 16, 18, 20):
        for scattering in (0.05, 0.10, 0.15, 0.20):
            if depth in (12, 16) and scattering == 0.10:
                continue
            configs.append(
                MechanismConfig(
                    name=f"refine_d{depth}_scatter{int(scattering * 1000):03d}_all_seed42",
                    max_depth=depth,
                    scattering_coefficient=scattering,
                )
            )
    for depth in (14, 16, 18):
        for scattering in (0.075, 0.125):
            configs.append(
                MechanismConfig(
                    name=f"refine_d{depth}_scatter{int(scattering * 1000):03d}_all_seed42",
                    max_depth=depth,
                    scattering_coefficient=scattering,
                )
            )
    for depth in (14, 16, 18):
        for scattering in (0.10, 0.15):
            configs.append(
                MechanismConfig(
                    name=f"refine_d{depth}_scatter{int(scattering * 1000):03d}_nonmetal_seed42",
                    max_depth=depth,
                    scattering_coefficient=scattering,
                    scattering_scope="nonmetal",
                )
            )
    return configs


def seed_configs() -> list[MechanismConfig]:
    reports = [load_report("grid"), load_report("refine")]
    available: dict[str, dict[str, object]] = {}
    for report in reports:
        for item in report["ranking_overall"]:
            config = item["config"]
            available[str(config["name"])] = config
    selected_names = (
        "grid_d32_base_seed42",
        "refine_d10_scatter150_all_seed42",
        "refine_d16_scatter150_all_seed42",
        "refine_d18_scatter150_nonmetal_seed42",
        "refine_d20_scatter150_all_seed42",
    )
    configs = []
    for source_name in selected_names:
        raw = available[source_name]
        base = MechanismConfig(**raw)
        for seed in (206, 420, 1337):
            configs.append(
                replace(
                    base,
                    name=f"seed_{source_name}_seed{seed}",
                    seed=seed,
                    source_name=source_name,
                )
            )
    return configs


def seed_aggregates(results: list[dict[str, object]]) -> list[dict[str, object]]:
    source_items = {}
    for stage in ("grid", "refine"):
        for item in load_report(stage)["ranking_overall"]:
            source_items[item["config"]["name"]] = item
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in results:
        source_name = item["config"]["source_name"]
        grouped.setdefault(source_name, [source_items[source_name]]).append(item)
    aggregates = []
    for source_name, items in grouped.items():
        robust = np.asarray([item["robust_selection"] for item in items])
        heavy_pdp = np.asarray(
            [item["by_regime"]["heavy"]["pdp_mean"] for item in items]
        )
        heavy_pas = np.asarray(
            [item["by_regime"]["heavy"]["leaked_full_mean"] for item in items]
        )
        aggregates.append(
            {
                "source_name": source_name,
                "source_config": source_items[source_name]["config"],
                "seeds": [int(item["config"]["seed"]) for item in items],
                "robust_mean": float(np.mean(robust)),
                "robust_std": float(np.std(robust)),
                "robust_conservative": float(np.mean(robust) - np.std(robust)),
                "heavy_pdp_mean": float(np.mean(heavy_pdp)),
                "heavy_pdp_std": float(np.std(heavy_pdp)),
                "heavy_pas_mean": float(np.mean(heavy_pas)),
                "heavy_pas_std": float(np.std(heavy_pas)),
                "heavy_spectral_mean": float(np.mean(0.5 * (heavy_pdp + heavy_pas))),
                "individual": [
                    {
                        "seed": int(item["config"]["seed"]),
                        "robust": float(item["robust_selection"]),
                        "heavy_pdp": float(item["by_regime"]["heavy"]["pdp_mean"]),
                        "heavy_pas": float(
                            item["by_regime"]["heavy"]["leaked_full_mean"]
                        ),
                    }
                    for item in items
                ],
            }
        )
    baseline = next(
        item for item in aggregates if item["source_name"] == "grid_d32_base_seed42"
    )
    baseline_by_seed = {item["seed"]: item for item in baseline["individual"]}
    for aggregate in aggregates:
        if aggregate is baseline:
            continue
        paired = {}
        for metric in ("robust", "heavy_pdp", "heavy_pas"):
            delta = np.asarray(
                [
                    item[metric] - baseline_by_seed[item["seed"]][metric]
                    for item in aggregate["individual"]
                ],
                dtype=np.float64,
            )
            paired[metric] = {
                "mean": float(np.mean(delta)),
                "std": float(np.std(delta)),
                "positive_seeds": int(np.sum(delta > 0.0)),
                "count": int(len(delta)),
                "values": delta.tolist(),
            }
        aggregate["paired_vs_d32"] = paired
    return sorted(
        aggregates,
        key=lambda item: (item["robust_conservative"], item["robust_mean"]),
        reverse=True,
    )


def load_v501_baseline() -> dict[str, object]:
    report = json.loads(V501_FINAL.read_text(encoding="utf-8"))
    for item in report["ranking"]:
        if item["config"]["name"] == "final_seed_42":
            return item
    raise ValueError("v501 final_seed_42 not found")


def add_mechanism_scores(result: dict[str, object], baseline: dict[str, object]) -> None:
    heavy = result["by_regime"]["heavy"]
    baseline_heavy = baseline["by_regime"]["heavy"]
    pdp = float(heavy["pdp_mean"])
    pas = float(heavy["leaked_full_mean"])
    result["heavy_spectral_mean"] = 0.5 * (pdp + pas)
    result["heavy_spectral_hmean"] = 2.0 * pdp * pas / max(pdp + pas, 1e-30)
    result["delta_vs_v501_seed42"] = {
        "robust_selection": float(result["robust_selection"] - baseline["robust_selection"]),
        "heavy_pdp": float(pdp - baseline_heavy["pdp_mean"]),
        "heavy_pas": float(pas - baseline_heavy["leaked_full_mean"]),
        "heavy_spectral_mean": float(
            result["heavy_spectral_mean"]
            - 0.5
            * (
                baseline_heavy["pdp_mean"]
                + baseline_heavy["leaked_full_mean"]
            )
        ),
    }


def path_count_summary(
    values: dict[str, np.ndarray], regimes: np.ndarray, zero_mask: np.ndarray
) -> dict[str, object]:
    nonzero = ~zero_mask

    def one(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "count": int(np.sum(mask)),
            "valid_paths_mean": float(np.mean(values["valid_path_count"][mask])),
            "diffuse_paths_mean": float(np.mean(values["diffuse_path_count"][mask])),
            "diffraction_paths_mean": float(np.mean(values["diffraction_path_count"][mask])),
            "points_with_diffuse": int(np.sum(values["diffuse_path_count"][mask] > 0)),
            "points_with_diffraction": int(
                np.sum(values["diffraction_path_count"][mask] > 0)
            ),
        }

    report = {
        "physical_nonzero_1984": one(nonzero),
        "by_regime": {
            name: one(nonzero & (regimes == code))
            for code, name in enumerate(("clear", "light", "medium", "heavy"))
        },
    }
    return report


def make_v501_config(config: MechanismConfig) -> SearchConfig:
    return SearchConfig(
        name=config.name,
        tx_azimuth_deg=-30.0,
        tx_tilt_deg=-12.0,
        rx_orientation="track_bs",
        synthetic_array=True,
        frequency_hz=6.5e9,
        max_depth=config.max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=config.scattering_coefficient > 0.0,
        refraction=True,
        diffraction=config.diffraction,
        edge_diffraction=config.edge_diffraction,
        samples_per_src=config.samples_per_src,
        max_num_paths_per_src=config.max_num_paths_per_src,
        seed=config.seed,
    )


def set_scattering(scene, config: MechanismConfig) -> None:
    for name, material in scene.radio_materials.items():
        active = config.scattering_scope == "all" or name != "metal"
        material.scattering_coefficient = (
            config.scattering_coefficient if active else 0.0
        )


def chunked_cfr(paths, frequencies: np.ndarray, mi, frequency_chunk: int = 32) -> np.ndarray:
    try:
        parts = []
        for start in range(0, len(frequencies), frequency_chunk):
            end = min(start + frequency_chunk, len(frequencies))
            parts.append(
                paths.cfr(
                    mi.TensorXf(frequencies[start:end]),
                    normalize_delays=False,
                    normalize=False,
                    out_type="numpy",
                )
            )
        return np.concatenate(parts, axis=-1)
    except RuntimeError as error:
        if frequency_chunk <= 1 or "exceeds the limit of 2^32" not in str(error):
            raise
        return chunked_cfr(paths, frequencies, mi, frequency_chunk // 2)


def run_config(
    config: MechanismConfig,
    positions: np.ndarray,
    truth: np.ndarray,
    truth_cache: dict[str, np.ndarray],
    folds: np.ndarray,
    regimes: np.ndarray,
    baseline: dict[str, object],
    batch_size: int,
    mi,
    dr,
    rt_api,
) -> dict[str, object]:
    result_path = config_result_path(config)
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("path_counts") is None and point_metric_path(config).is_file():
            cached = np.load(point_metric_path(config))
            path_values = {
                name: np.asarray(cached[name])
                for name in (
                    "valid_path_count",
                    "diffuse_path_count",
                    "diffraction_path_count",
                )
            }
            zero_mask = np.asarray(truth_cache["zero_mask"], dtype=bool)
            result["path_counts"] = path_count_summary(
                path_values, regimes, zero_mask
            )
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    load_scene, Transmitter, Receiver, PlanarArray, PathSolver, frequencies = rt_api
    scene = load_scene(str(SCENE_XML), merge_shapes=True)
    scene.frequency = 6.5e9
    set_scattering(scene, config)
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
    search_config = make_v501_config(config)
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*[float(value) for value in BS_POSITION]),
        look_at=tx_target(search_config, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()
    metric_names = (
        "pdp",
        "leaked_full",
        "leaked_collapsed",
        "conventional_pas",
        "complex_coherence",
        "pred_log_power",
    )
    metrics = {name: np.empty(len(positions), dtype=np.float32) for name in metric_names}
    path_values = {
        "valid_path_count": np.empty(len(positions), dtype=np.int32),
        "diffuse_path_count": np.empty(len(positions), dtype=np.int32),
        "diffraction_path_count": np.empty(len(positions), dtype=np.int32),
    }
    receiver_names: list[str] = []
    started = time.time()
    effective_batch_size = batch_size
    for batch_index, start in enumerate(
        range(0, len(positions), effective_batch_size)
    ):
        end = min(start + effective_batch_size, len(positions))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for local, position in enumerate(positions[start:end]):
            name = f"rx_{start + local:05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*[float(value) for value in position]),
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
            diffuse_reflection=config.scattering_coefficient > 0.0,
            refraction=True,
            diffraction=config.diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=config.diffraction_lit_region,
            seed=config.seed + batch_index,
        )
        interactions = np.asarray(paths.interactions.numpy(), dtype=np.int16)
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        contains_diffuse = np.any(interactions == 2, axis=0)
        contains_diffraction = np.any(interactions == 8, axis=0)
        path_values["valid_path_count"][start:end] = np.sum(valid, axis=(1, 2))
        path_values["diffuse_path_count"][start:end] = np.sum(
            valid & contains_diffuse, axis=(1, 2)
        )
        path_values["diffraction_path_count"][start:end] = np.sum(
            valid & contains_diffraction, axis=(1, 2)
        )
        if config.scattering_coefficient >= 0.3:
            raw = chunked_cfr(paths, frequencies, mi)
        else:
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
                raw = chunked_cfr(paths, frequencies, mi)
        prediction = convert_cfr(raw, end - start)
        batch_truth = np.asarray(truth[start:end], dtype=np.complex64)
        batch_metrics = prediction_metrics(
            prediction, batch_truth, truth_cache, start, end
        )
        for name, value in batch_metrics.items():
            metrics[name][start:end] = value
        del paths, interactions, valid, raw, prediction, batch_truth, batch_metrics
        dr.sync_thread()
        if (batch_index + 1) % 25 == 0 or end == len(positions):
            print(
                f"{config.name}: {end}/{len(positions)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    for name in receiver_names:
        scene.remove(name)
    dr.sync_thread()
    flush_cache = getattr(dr, "flush_malloc_cache", None)
    if callable(flush_cache):
        flush_cache()

    POINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(point_metric_path(config), **metrics, **path_values)
    zero_mask = np.asarray(truth_cache["zero_mask"], dtype=bool)
    result = summarize_config(
        search_config,
        metrics,
        np.asarray(truth_cache["log_power"], dtype=np.float32),
        zero_mask,
        folds,
        regimes,
        time.time() - started,
    )
    result["method"] = VERSION
    result["config"] = asdict(config)
    result["point_metrics"] = PATHS.relative(point_metric_path(config))
    result["path_counts"] = path_count_summary(path_values, regimes, zero_mask)
    add_mechanism_scores(result, baseline)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{config.name}: robust={result['robust_selection']:.6f} "
        f"heavy_pdp={result['by_regime']['heavy']['pdp_mean']:.6f} "
        f"heavy_pas={result['by_regime']['heavy']['leaked_full_mean']:.6f}",
        flush=True,
    )
    return result


def pareto_front(results: list[dict[str, object]]) -> list[dict[str, object]]:
    front = []
    for candidate in results:
        values = np.asarray(
            [
                candidate["robust_selection"],
                candidate["by_regime"]["heavy"]["pdp_mean"],
                candidate["by_regime"]["heavy"]["leaked_full_mean"],
            ],
            dtype=np.float64,
        )
        dominated = False
        for other in results:
            if other is candidate:
                continue
            other_values = np.asarray(
                [
                    other["robust_selection"],
                    other["by_regime"]["heavy"]["pdp_mean"],
                    other["by_regime"]["heavy"]["leaked_full_mean"],
                ],
                dtype=np.float64,
            )
            if np.all(other_values >= values) and np.any(other_values > values + 1e-9):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: item["robust_selection"], reverse=True)


def build_stage_report(
    stage: str,
    results: list[dict[str, object]],
    baseline: dict[str, object],
    total_seconds: float,
    sionna_version: str,
    mitsuba_variant: str,
) -> dict[str, object]:
    ranking_overall = sorted(
        results, key=lambda item: item["robust_selection"], reverse=True
    )
    safe_floor = float(baseline["robust_selection"] - 0.005)
    safe = [item for item in results if item["robust_selection"] >= safe_floor]
    ranking_heavy_safe = sorted(
        safe,
        key=lambda item: (item["heavy_spectral_hmean"], item["robust_selection"]),
        reverse=True,
    )
    ranking_heavy_pas = sorted(
        results,
        key=lambda item: item["by_regime"]["heavy"]["leaked_full_mean"],
        reverse=True,
    )
    ranking_heavy_pdp = sorted(
        results,
        key=lambda item: item["by_regime"]["heavy"]["pdp_mean"],
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": "analysis_in_progress" if stage == "grid" else "analysis_complete",
        "stage": stage,
        "point_count": 2000,
        "config_count": len(results),
        "sionna_version": sionna_version,
        "mitsuba_variant": mitsuba_variant,
        "scene": PATHS.relative(SCENE_XML),
        "v501_seed42_baseline": baseline,
        "selection_rule": (
            "Do not collapse the experiment to one score. Rank overall robust selection, "
            "heavy PDP, heavy leaked-PAS, and their Pareto front separately. Heavy-safe "
            "requires overall robust within 0.005 of v501 final_seed_42."
        ),
        "storage_policy": (
            "No full H or Paths is persisted. Each rejected configuration keeps only compact "
            "point metrics, path-type counts, and JSON evidence."
        ),
        "pruning_rule": (
            "After depths 3/5/8 consistently showed that diffraction+scattering underperformed "
            "scattering alone, depth12 skipped shadow/edge+scatter and depth16 skipped all "
            "diffraction+scatter repeats. Existing compact negative evidence was retained."
        ),
        "ranking_overall": ranking_overall,
        "ranking_heavy_safe": ranking_heavy_safe,
        "ranking_heavy_pas": ranking_heavy_pas,
        "ranking_heavy_pdp": ranking_heavy_pdp,
        "pareto_front": pareto_front(results),
        "total_seconds": total_seconds,
    }
    if stage == "seeds":
        report["seed_aggregates"] = seed_aggregates(results)
    return report


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    truth = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    regimes = regime_codes()
    folds = spatial_fold_codes(positions, regimes)
    truth_cache = prepare_truth_cache()
    baseline = load_v501_baseline()

    import drjit as dr
    import mitsuba as mi
    import sionna

    mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    frequencies = (
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    rt_api = (load_scene, Transmitter, Receiver, PlanarArray, PathSolver, frequencies)
    configs = {
        "grid": grid_configs,
        "refine": refine_configs,
        "seeds": seed_configs,
    }[args.stage]()
    if args.config_limit is not None:
        configs = configs[: args.config_limit]
    if args.force:
        for config in configs:
            config_result_path(config).unlink(missing_ok=True)
            point_metric_path(config).unlink(missing_ok=True)

    started = time.time()
    results = [
        run_config(
            config,
            positions,
            truth,
            truth_cache,
            folds,
            regimes,
            baseline,
            args.batch_size,
            mi,
            dr,
            rt_api,
        )
        for config in configs
    ]
    report = build_stage_report(
        args.stage,
        results,
        baseline,
        time.time() - started,
        getattr(sionna, "__version__", "unknown"),
        mi.variant(),
    )
    report_path = stage_report_path(args.stage)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": PATHS.relative(report_path),
                "best_overall": report["ranking_overall"][0] if results else None,
                "best_heavy_safe": report["ranking_heavy_safe"][0]
                if report["ranking_heavy_safe"]
                else None,
                "pareto_count": len(report["pareto_front"]),
                "total_seconds": report["total_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
