#!/usr/bin/env python3
"""Audit scene merging and diffraction topology with fixed official materials."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v503_fixed_material_sionna_version_inverse as v503
from project_paths import PATHS


VERSION_NUMBER = 504
VERSION = "v504_scene_topology_sionna_inverse"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
POINT_DIR = CACHE_DIR / "point_metrics"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
FREQUENCY_HZ = 6.693e9
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0


@dataclass(frozen=True)
class TopologyConfig:
    name: str
    merge_shapes: bool
    remove_duplicate_vertices: bool
    max_depth: int
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    samples_per_src: int = 200_000
    max_num_paths_per_src: int = 200_000
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "screen", "refine", "full"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--point-limit", type=int)
    parser.add_argument("--config-limit", type=int)
    parser.add_argument("--budget-cap", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def config_report_path(config: TopologyConfig, point_count: int) -> Path:
    return REPORT_DIR / "configs" / f"{safe_name(config.name)}_points{point_count}.json"


def point_metric_path(config: TopologyConfig, point_count: int) -> Path:
    return POINT_DIR / f"{safe_name(config.name)}_points{point_count}.npz"


def stage_report_path(stage: str) -> Path:
    return REPORT_DIR / f"v504_{stage}.json"


def topology_name(merge: bool, dedup: bool) -> str:
    return f"merge{int(merge)}_dedup{int(dedup)}"


def screen_configs() -> list[TopologyConfig]:
    configs: list[TopologyConfig] = []
    # Depth is the primary axis. The depth-3 audit showed that shadow-only and
    # edge diffraction are both materially worse, so only carry ordinary lit
    # diffraction through depth 8. Its depth-5 and depth-8 results already fall
    # below depth 3, so repeating it at depth 12/16 adds cost without resolving
    # the topology hypothesis.
    for depth in (3, 5, 8, 12, 16):
        modes = [("base", False)]
        if depth <= 8:
            modes.append(("diff_lit", True))
        for mode, diffraction in modes:
            configs.append(
                TopologyConfig(
                    name=f"merge0_dedup0_d{depth}_{mode}_200k_seed42",
                    merge_shapes=False,
                    remove_duplicate_vertices=False,
                    max_depth=depth,
                    diffraction=diffraction,
                )
            )

    # Audit vertex deduplication and shape merging at shallow, middle and deep
    # settings. One shallow diffraction run per topology is enough to
    # expose whether its wedge graph differs before the expensive refinement.
    for merge, dedup in ((False, True), (True, True), (True, False)):
        for depth, mode, diffraction in (
            (3, "base", False),
            (8, "base", False),
            (16, "base", False),
            (3, "diff_lit", True),
        ):
            configs.append(
                TopologyConfig(
                    name=(
                        f"{topology_name(merge, dedup)}_d{depth}_{mode}_200k_seed42"
                    ),
                    merge_shapes=merge,
                    remove_duplicate_vertices=dedup,
                    max_depth=depth,
                    diffraction=diffraction,
                )
            )
    return configs


def load_stage(stage: str) -> dict[str, object]:
    path = stage_report_path(stage)
    if not path.is_file():
        raise FileNotFoundError(f"run {stage} first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_configs(stage: str) -> list[TopologyConfig]:
    if stage == "smoke":
        return [
            TopologyConfig("smoke_merge1", True, False, 3, samples_per_src=20_000, max_num_paths_per_src=20_000),
            TopologyConfig("smoke_merge0", False, False, 3, samples_per_src=20_000, max_num_paths_per_src=20_000),
        ]
    if stage == "screen":
        return screen_configs()
    ranking = load_stage("screen")["ranking"]
    unique: list[TopologyConfig] = []
    seen: set[tuple[bool, bool, int, bool, bool, bool]] = set()
    for row in ranking:
        config = TopologyConfig(**row["config"])
        key = (
            config.merge_shapes,
            config.remove_duplicate_vertices,
            config.max_depth,
            config.diffraction,
            config.edge_diffraction,
            config.diffraction_lit_region,
        )
        if key not in seen:
            seen.add(key)
            unique.append(config)
        if len(unique) == 4:
            break
    if stage == "refine":
        return [
            replace(
                config,
                name=(
                    f"refine_{topology_name(config.merge_shapes, config.remove_duplicate_vertices)}_"
                    f"d{config.max_depth}_diff{int(config.diffraction)}_"
                    f"edge{int(config.edge_diffraction)}_lit{int(config.diffraction_lit_region)}_"
                    f"1m_seed{seed}"
                ),
                samples_per_src=1_000_000,
                max_num_paths_per_src=1_000_000,
                seed=seed,
            )
            for config in unique
            for seed in (42, 206, 420)
        ]
    refine = load_stage("refine")["seed_aggregates"]
    prototypes = [TopologyConfig(**row["prototype"]) for row in refine[:2]]
    return [
        replace(
            config,
            name=(
                f"full_{topology_name(config.merge_shapes, config.remove_duplicate_vertices)}_"
                f"d{config.max_depth}_diff{int(config.diffraction)}_"
                f"edge{int(config.edge_diffraction)}_lit{int(config.diffraction_lit_region)}_"
                f"1m_seed{seed}"
            ),
            samples_per_src=1_000_000,
            max_num_paths_per_src=1_000_000,
            seed=seed,
        )
        for config in prototypes
        for seed in (42, 206, 420)
    ]


def material_audit(scene) -> dict[str, dict[str, float]]:
    def scalar(value) -> float:
        return float(np.asarray(value).reshape(-1)[0])

    return {
        name: {
            "scattering_coefficient": scalar(material.scattering_coefficient),
            "xpd_coefficient": scalar(material.xpd_coefficient),
        }
        for name, material in scene.radio_materials.items()
        if name in {"marble", "metal", "wood", "concrete"}
    }


def run_config(
    config: TopologyConfig,
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
        merge_shapes=config.merge_shapes,
        remove_duplicate_vertices=config.remove_duplicate_vertices,
    )
    scene.frequency = FREQUENCY_HZ
    materials = material_audit(scene)
    if any(
        value["scattering_coefficient"] != 0.0 or value["xpd_coefficient"] != 0.0
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
        synthetic_array=True,
        frequency_hz=FREQUENCY_HZ,
        max_depth=config.max_depth,
        diffraction=config.diffraction,
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
    metric_parts = {name: [] for name in (
        "pdp", "leaked_full", "complex_coherence", "pred_log_power", "truth_log_power"
    )}
    valid_path_count: list[np.ndarray] = []
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
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        valid_path_count.append(np.sum(valid, axis=(1, 2)).astype(np.int32))
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        prediction = v501.convert_cfr(np.asarray(raw), end - start)
        batch_truth = np.asarray(truth[batch_ids], dtype=np.complex64)
        metrics = v503.channel_metrics(prediction, batch_truth)
        for name, value in metrics.items():
            metric_parts[name].append(value)
        del paths, valid, raw, prediction, batch_truth, metrics
        dr.sync_thread()
        if end == len(indices) or (batch_index + 1) % 10 == 0:
            print(
                f"{config.name}: {end}/{len(indices)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    for name in receiver_names:
        scene.remove(name)
    dr.sync_thread()
    flush = getattr(dr, "flush_malloc_cache", None)
    if callable(flush):
        flush()

    metrics = {name: np.concatenate(parts) for name, parts in metric_parts.items()}
    path_count = np.concatenate(valid_path_count)
    POINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        metric_path,
        indices=indices,
        regimes=regime_codes[indices],
        valid_path_count=path_count,
        **metrics,
    )
    summary = v503.summarize_metrics(metrics, regime_codes[indices])
    result = {
        "method": VERSION,
        "config": asdict(config),
        "point_count": int(len(indices)),
        "scene_xml": PATHS.relative(SCENE_XML),
        "materials": materials,
        "summary": summary,
        "selection": float(summary["overall"]["selection"]),
        "valid_path_count": {
            "mean": float(np.mean(path_count)),
            "median": float(np.median(path_count)),
            "p90": float(np.quantile(path_count, 0.90)),
            "zero_count": int(np.sum(path_count == 0)),
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
        f"paths={result['valid_path_count']['mean']:.1f}",
        flush=True,
    )
    return result


def seed_aggregates(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[bool, bool, int, bool, bool, bool], list[dict[str, object]]] = {}
    for result in results:
        config = result["config"]
        key = (
            bool(config["merge_shapes"]),
            bool(config["remove_duplicate_vertices"]),
            int(config["max_depth"]),
            bool(config["diffraction"]),
            bool(config["edge_diffraction"]),
            bool(config["diffraction_lit_region"]),
        )
        grouped.setdefault(key, []).append(result)
    rows = []
    for key, items in grouped.items():
        selection = np.asarray([item["selection"] for item in items], dtype=np.float64)
        pdp = np.asarray([item["summary"]["overall"]["pdp_mean"] for item in items])
        pas = np.asarray([item["summary"]["overall"]["leaked_full_mean"] for item in items])
        rows.append(
            {
                "key": {
                    "merge_shapes": key[0],
                    "remove_duplicate_vertices": key[1],
                    "max_depth": key[2],
                    "diffraction": key[3],
                    "edge_diffraction": key[4],
                    "diffraction_lit_region": key[5],
                },
                "prototype": dict(items[0]["config"]),
                "seed_count": int(len(items)),
                "selection_mean": float(np.mean(selection)),
                "selection_std": float(np.std(selection)),
                "selection_conservative": float(np.mean(selection) - np.std(selection)),
                "pdp_mean": float(np.mean(pdp)),
                "pas_mean": float(np.mean(pas)),
                "runs": [item["config"]["name"] for item in items],
            }
        )
    return sorted(rows, key=lambda row: row["selection_conservative"], reverse=True)


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
        raise RuntimeError(f"v504 requires Sionna 2.0.1, got {sionna.__version__}")
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
    default_limit = {"smoke": 8, "screen": 80, "refine": 160, "full": len(positions)}[
        args.stage
    ]
    point_limit = min(args.point_limit or default_limit, len(positions))
    if point_limit == len(positions):
        indices = np.arange(len(positions), dtype=np.int64)
    else:
        indices = v503.balanced_indices(
            positions, regime_codes, truth_log_power, point_limit
        )

    configs = stage_configs(args.stage)
    if args.budget_cap is not None:
        configs = [
            replace(
                config,
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
    ranking = sorted(results, key=lambda row: row["selection"], reverse=True)
    controls = v503.current_solver_controls(indices, regime_codes)
    report = {
        "method": VERSION,
        "status": "analysis_complete" if args.stage != "smoke" else "smoke_complete",
        "stage": args.stage,
        "primary_hypothesis": (
            "The remaining medium/heavy mismatch may come from merging 565 Etoile shapes by "
            "material, which changes object boundaries and diffraction topology even when all "
            "material electromagnetic parameters are fixed."
        ),
        "environment": {
            "sionna": sionna.__version__,
            "mitsuba_variant": mi.variant(),
            "visible_gpu_count": 1,
        },
        "fixed_parameters": {
            "frequency_hz": FREQUENCY_HZ,
            "subcarrier_spacing_hz": SUBCARRIER_SPACING,
            "tx_array": "8x16 tr38901 VH 0.5lambda",
            "rx_array": "2x1 dipole cross 0.5lambda",
            "tx_azimuth_deg": -30.0,
            "tx_tilt_deg": -9.53,
            "rx_orientation": "track_bs",
            "synthetic_array": True,
            "materials": "canonical fixed Sionna Etoile ITU defaults",
            "los": True,
            "specular_reflection": True,
            "refraction": True,
            "diffuse_reflection": False,
        },
        "point_count": int(len(indices)),
        "indices": indices.tolist(),
        "ranking": ranking,
        "current_solver_controls": controls,
        "seed_aggregates": seed_aggregates(results) if args.stage in {"refine", "full"} else [],
        "outputs": {
            "report": PATHS.relative(stage_report_path(args.stage)),
            "point_metrics_dir": PATHS.relative(POINT_DIR),
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
                "controls": [
                    {"name": row["name"], "selection": row["selection"]}
                    for row in controls
                ],
                "report": PATHS.relative(stage_report_path(args.stage)),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
