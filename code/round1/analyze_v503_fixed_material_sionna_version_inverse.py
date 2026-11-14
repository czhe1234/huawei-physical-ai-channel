#!/usr/bin/env python3
"""Compare fixed-material Sionna RT generations across solver/scene versions."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.fft import fft2, ifft
from scipy.stats import rankdata

from project_paths import PATHS


VERSION_NUMBER = 503
VERSION = "v503_fixed_material_sionna_version_inverse"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
POINT_DIR = CACHE_DIR / "point_metrics"
COMPAT_SCENE_DIR = CACHE_DIR / "sionna019_current_mesh_scene"
CURRENT_SCENE_DIR = (
    PATHS.artifacts / "external" / "restored_sionna_official_etoile"
)
CURRENT_SCENE_XML = CURRENT_SCENE_DIR / "etoile.xml"
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
FREQUENCY_HZ = 6.693e9
TINY = 1e-30
NONZERO_LOG_THRESHOLD = math.log(TINY) + 1e-3


@dataclass(frozen=True)
class OldSolverConfig:
    name: str
    scene_source: str
    max_depth: int
    diffraction: bool = False
    edge_diffraction: bool = False
    method: str = "fibonacci"
    num_samples: int = 200_000
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("smoke", "screen", "refine", "full"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--point-limit", type=int)
    parser.add_argument("--config-limit", type=int)
    parser.add_argument("--budget-cap", type=int)
    parser.add_argument("--mitsuba-variant", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def config_report_path(config: OldSolverConfig, point_count: int) -> Path:
    return REPORT_DIR / "configs" / f"{safe_name(config.name)}_points{point_count}.json"


def point_metric_path(config: OldSolverConfig, point_count: int) -> Path:
    return POINT_DIR / f"{safe_name(config.name)}_points{point_count}.npz"


def stage_report_path(stage: str) -> Path:
    return REPORT_DIR / f"v503_{stage}.json"


def unit_normalize(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    value64 = np.asarray(value, dtype=np.float64)
    norm = np.sqrt(np.sum(value64 * value64, axis=axes, keepdims=True))
    return (value / np.maximum(norm, TINY)).astype(np.float32)


def channel_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    count = len(prediction)
    pred_delay = np.abs(ifft(prediction, axis=-1, workers=1)).astype(np.float32) ** 2
    truth_delay = np.abs(ifft(truth, axis=-1, workers=1)).astype(np.float32) ** 2
    pred_delay = unit_normalize(pred_delay, (3,))
    truth_delay = unit_normalize(truth_delay, (3,))
    pdp = np.clip(
        np.sum(pred_delay * truth_delay, axis=3, dtype=np.float64).mean(axis=(1, 2)),
        0.0,
        1.0,
    )

    pred6 = prediction.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    truth6 = truth.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    pred_leaked = pred6.transpose(0, 1, 5, 2, 4, 3)
    truth_leaked = truth6.transpose(0, 1, 5, 2, 4, 3)
    pred_leaked = np.abs(fft2(pred_leaked, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    truth_leaked = np.abs(fft2(truth_leaked, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    pred_leaked = unit_normalize(pred_leaked, (2, 3))
    truth_leaked = unit_normalize(truth_leaked, (2, 3))
    leaked_full = np.clip(
        np.sum(pred_leaked * truth_leaked, axis=(2, 3), dtype=np.float64).mean(
            axis=(1, 2, 3)
        ),
        0.0,
        1.0,
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
        "complex_coherence": coherence.astype(np.float32),
        "pred_log_power": np.log(np.maximum(pred_power, TINY)).astype(np.float32),
        "truth_log_power": np.log(np.maximum(truth_power, TINY)).astype(np.float32),
    }


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if np.std(left_rank) <= 0.0 or np.std(right_rank) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def calibrated_power_mae(predicted: np.ndarray, truth: np.ndarray) -> float:
    offset = float(np.median(truth - predicted))
    return float(np.mean(np.abs(predicted + offset - truth)))


def summarize_mask(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    if not np.any(mask):
        return {"count": 0}
    predicted = metrics["pred_log_power"][mask]
    truth = metrics["truth_log_power"][mask]
    power_spearman = rank_correlation(predicted, truth)
    pdp = float(np.mean(metrics["pdp"][mask]))
    leaked = float(np.mean(metrics["leaked_full"][mask]))
    power_term = 0.5 + 0.5 * power_spearman
    selection = 0.45 * leaked + 0.45 * pdp + 0.10 * power_term
    return {
        "count": int(np.sum(mask)),
        "pdp_mean": pdp,
        "leaked_full_mean": leaked,
        "complex_coherence_mean": float(np.mean(metrics["complex_coherence"][mask])),
        "power_spearman": power_spearman,
        "calibrated_log_power_mae": calibrated_power_mae(predicted, truth),
        "selection": float(selection),
    }


def summarize_metrics(metrics: dict[str, np.ndarray], regimes: np.ndarray) -> dict[str, object]:
    nonzero = metrics["truth_log_power"] > NONZERO_LOG_THRESHOLD
    return {
        "overall": summarize_mask(metrics, nonzero),
        "by_regime": {
            name: summarize_mask(metrics, nonzero & (regimes == code))
            for code, name in enumerate(("clear", "light", "medium", "heavy"))
        },
    }


def farthest_sample(points: np.ndarray, count: int, start: int) -> np.ndarray:
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = start
    distance = np.sum((points - points[start]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(distance))
        candidate = np.sum((points - points[selected[index]]) ** 2, axis=1)
        distance = np.minimum(distance, candidate)
    return selected


def balanced_indices(
    positions: np.ndarray,
    regimes: np.ndarray,
    truth_log_power: np.ndarray,
    point_limit: int,
) -> np.ndarray:
    per_regime = max(point_limit // 4, 1)
    selected: list[np.ndarray] = []
    for regime in range(4):
        candidates = np.flatnonzero(
            (regimes == regime)
            & np.isfinite(truth_log_power)
            & (truth_log_power > NONZERO_LOG_THRESHOLD)
        )
        if len(candidates) == 0:
            continue
        threshold = float(np.quantile(truth_log_power[candidates], 0.20))
        candidates = candidates[truth_log_power[candidates] >= threshold]
        start = int(np.argmax(truth_log_power[candidates]))
        local = farthest_sample(positions[candidates, :2], min(per_regime, len(candidates)), start)
        selected.append(candidates[local])
    merged = np.unique(np.concatenate(selected))
    if len(merged) < point_limit:
        remaining = np.setdiff1d(np.arange(len(positions)), merged)
        valid = remaining[truth_log_power[remaining] > NONZERO_LOG_THRESHOLD]
        take = min(point_limit - len(merged), len(valid))
        if take:
            local = farthest_sample(positions[valid, :2], take, int(np.argmax(truth_log_power[valid])))
            merged = np.concatenate((merged, valid[local]))
    return np.sort(merged[:point_limit]).astype(np.int64)


def build_compat_scene(old_scene_xml: Path) -> Path:
    output_xml = COMPAT_SCENE_DIR / "etoile.xml"
    current_meshes = CURRENT_SCENE_DIR / "meshes"
    if output_xml.is_file() and len(list((COMPAT_SCENE_DIR / "meshes").glob("*.ply"))) == 565:
        return output_xml

    current_root = ET.parse(CURRENT_SCENE_XML).getroot()
    root = ET.Element("scene", {"version": current_root.attrib.get("version", "2.1.0")})
    visual_materials = {
        "concrete": ("0.539479", "0.539479", "0.539480"),
        "marble": ("0.701101", "0.644479", "0.485150"),
        "metal": ("0.219526", "0.219526", "0.254152"),
        "wood": ("0.509804", "0.167376", "0.059954"),
    }
    for material, color in visual_materials.items():
        outer = ET.SubElement(root, "bsdf", {"type": "twosided", "id": f"mat-itu_{material}"})
        inner = ET.SubElement(outer, "bsdf", {"type": "diffuse"})
        ET.SubElement(
            inner,
            "rgb",
            {"value": " ".join(color), "name": "reflectance"},
        )
    for child in current_root:
        if child.tag != "shape":
            continue
        shape = deepcopy(child)
        for ref in shape.findall("ref"):
            material = ref.attrib.get("id", "")
            if material in {"concrete", "marble", "metal", "wood"}:
                ref.set("id", f"mat-itu_{material}")
        root.append(shape)

    mesh_output = COMPAT_SCENE_DIR / "meshes"
    mesh_output.mkdir(parents=True, exist_ok=True)
    for source in current_meshes.glob("*.ply"):
        destination = mesh_output / source.name
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            shutil.copy2(source, destination)
    COMPAT_SCENE_DIR.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_xml, encoding="utf-8", xml_declaration=False)
    return output_xml


def base_grid() -> list[OldSolverConfig]:
    configs: list[OldSolverConfig] = []
    for scene_source in ("sionna019_bundle", "current_mesh"):
        for depth in (3, 5, 8, 12, 16):
            configs.append(
                OldSolverConfig(
                    name=f"{scene_source}_d{depth}_reflection_200k_seed42",
                    scene_source=scene_source,
                    max_depth=depth,
                )
            )
            if depth <= 8:
                configs.append(
                    OldSolverConfig(
                        name=f"{scene_source}_d{depth}_diffraction_200k_seed42",
                        scene_source=scene_source,
                        max_depth=depth,
                        diffraction=True,
                    )
                )
                configs.append(
                    OldSolverConfig(
                        name=f"{scene_source}_d{depth}_edge_200k_seed42",
                        scene_source=scene_source,
                        max_depth=depth,
                        diffraction=True,
                        edge_diffraction=True,
                    )
                )
    return configs


def screen_ranking() -> list[dict[str, object]]:
    path = stage_report_path("screen")
    if not path.is_file():
        raise FileNotFoundError(f"run screen first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["ranking"]


def stage_configs(stage: str) -> list[OldSolverConfig]:
    if stage == "smoke":
        return [
            OldSolverConfig("smoke_old_d3", "sionna019_bundle", 3, num_samples=20_000),
            OldSolverConfig("smoke_current_d3", "current_mesh", 3, num_samples=20_000),
        ]
    if stage == "screen":
        return base_grid()
    ranking = screen_ranking()
    unique: list[OldSolverConfig] = []
    seen: set[tuple[str, int, bool, bool]] = set()
    for row in ranking:
        config = OldSolverConfig(**row["config"])
        key = (
            config.scene_source,
            config.max_depth,
            config.diffraction,
            config.edge_diffraction,
        )
        if key not in seen:
            seen.add(key)
            unique.append(config)
        if len(unique) == 3:
            break
    if stage == "refine":
        return [
            replace(
                config,
                name=(
                    f"refine_{config.scene_source}_d{config.max_depth}_"
                    f"diff{int(config.diffraction)}_edge{int(config.edge_diffraction)}_"
                    f"1m_seed{seed}"
                ),
                num_samples=1_000_000,
                seed=seed,
            )
            for config in unique
            for seed in (42, 206, 420)
        ]
    refine_path = stage_report_path("refine")
    if not refine_path.is_file():
        raise FileNotFoundError(f"run refine first: {refine_path}")
    refine = json.loads(refine_path.read_text(encoding="utf-8"))["seed_aggregates"]
    best = refine[0]["prototype"]
    config = OldSolverConfig(**best)
    return [
        replace(
            config,
            name=(
                f"full_{config.scene_source}_d{config.max_depth}_"
                f"diff{int(config.diffraction)}_edge{int(config.edge_diffraction)}_"
                f"1m_seed{seed}"
            ),
            num_samples=1_000_000,
            seed=seed,
        )
        for seed in (42, 206, 420)
    ]


def tx_target() -> np.ndarray:
    radius = 140.0
    azimuth = math.radians(-30.0)
    tilt = math.radians(-9.53)
    return np.asarray(
        [
            BS_POSITION[0] + radius * math.cos(azimuth),
            BS_POSITION[1] + radius * math.sin(azimuth),
            BS_POSITION[2] + radius * math.tan(tilt),
        ],
        dtype=np.float32,
    )


def convert_old_cfr(raw: np.ndarray, count: int) -> np.ndarray:
    expected = (1, count, 4, 1, 256, 1, NUM_SUBCARRIERS)
    if tuple(raw.shape) != expected:
        raise ValueError(f"unexpected Sionna 0.19 CFR shape {raw.shape}, expected {expected}")
    return np.transpose(raw[0, :, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def run_config(
    config: OldSolverConfig,
    indices: np.ndarray,
    positions: np.ndarray,
    truth: np.ndarray,
    regimes: np.ndarray,
    batch_size: int,
    old_scene_xml: Path,
    rt_api,
) -> dict[str, object]:
    report_path = config_report_path(config, len(indices))
    metric_path = point_metric_path(config, len(indices))
    if report_path.is_file() and metric_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    (
        load_scene,
        PlanarArray,
        Receiver,
        Transmitter,
        subcarrier_frequencies,
        cir_to_ofdm_channel,
        sionna_config,
    ) = rt_api
    scene_xml = old_scene_xml
    if config.scene_source == "current_mesh":
        scene_xml = build_compat_scene(old_scene_xml)
    scene = load_scene(str(scene_xml))
    scene.frequency = FREQUENCY_HZ
    scene.synthetic_array = True
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
    tx = Transmitter(name="tx", position=BS_POSITION.tolist(), power_dbm=44.0)
    scene.add(tx)
    tx.look_at(tx_target().tolist())
    frequencies = subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)

    metric_parts = {name: [] for name in channel_metrics(
        np.zeros((1, 256, 4, NUM_SUBCARRIERS), dtype=np.complex64),
        np.zeros((1, 256, 4, NUM_SUBCARRIERS), dtype=np.complex64),
    )}
    started = time.time()
    receiver_names: list[str] = []
    for batch_index, start in enumerate(range(0, len(indices), batch_size)):
        end = min(start + batch_size, len(indices))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        batch_ids = indices[start:end]
        for local, point_id in enumerate(batch_ids):
            name = f"rx_{int(point_id):05d}"
            receiver = Receiver(name=name, position=positions[point_id].tolist())
            scene.add(receiver)
            receiver.look_at(tx)
            receiver_names.append(name)
        sionna_config.seed = config.seed + batch_index
        paths = scene.compute_paths(
            max_depth=config.max_depth,
            method=config.method,
            num_samples=config.num_samples,
            los=True,
            reflection=True,
            diffraction=config.diffraction,
            scattering=False,
            ris=False,
            edge_diffraction=config.edge_diffraction,
            check_scene=batch_index == 0,
        )
        paths.normalize_delays = False
        coefficients, delays = paths.cir(
            los=True,
            reflection=True,
            diffraction=config.diffraction,
            scattering=False,
            ris=False,
        )
        raw = cir_to_ofdm_channel(
            frequencies, coefficients, delays, normalize=False
        ).numpy()
        prediction = convert_old_cfr(raw, end - start)
        batch_truth = np.asarray(truth[batch_ids], dtype=np.complex64)
        batch_metrics = channel_metrics(prediction, batch_truth)
        for name, value in batch_metrics.items():
            metric_parts[name].append(value)
        del paths, coefficients, delays, raw, prediction, batch_truth, batch_metrics
        if end == len(indices) or (batch_index + 1) % 10 == 0:
            print(
                f"{config.name}: {end}/{len(indices)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    for name in receiver_names:
        scene.remove(name)

    metrics = {name: np.concatenate(parts) for name, parts in metric_parts.items()}
    POINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metric_path, indices=indices, regimes=regimes[indices], **metrics)
    summary = summarize_metrics(metrics, regimes[indices])
    result = {
        "method": VERSION,
        "config": asdict(config),
        "point_count": int(len(indices)),
        "indices": indices.tolist(),
        "scene_xml": PATHS.relative(scene_xml),
        "frequency_hz": FREQUENCY_HZ,
        "subcarrier_spacing_hz": SUBCARRIER_SPACING,
        "materials": "canonical fixed Sionna Etoile ITU materials; not fitted",
        "summary": summary,
        "selection": float(summary["overall"]["selection"]),
        "seconds": float(time.time() - started),
        "point_metrics": PATHS.relative(metric_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{config.name}: selection={result['selection']:.6f} "
        f"pdp={summary['overall']['pdp_mean']:.6f} "
        f"pas={summary['overall']['leaked_full_mean']:.6f}",
        flush=True,
    )
    return result


def current_solver_controls(indices: np.ndarray, regimes: np.ndarray) -> list[dict[str, object]]:
    controls = [
        (
            "sionna201_d16_6p693_1m",
            PATHS.cache_dir(501)
            / "point_metrics"
            / "physical_d16_1m_spec_refract_6p693.npz",
        ),
        (
            "sionna201_d32_6p5_1m_seed42",
            PATHS.cache_dir(501) / "point_metrics" / "final_seed_42.npz",
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, path in controls:
        if not path.is_file():
            continue
        data = np.load(path)
        metrics = {
            "pdp": np.asarray(data["pdp"])[indices],
            "leaked_full": np.asarray(data["leaked_full"])[indices],
            "complex_coherence": np.asarray(data["complex_coherence"])[indices],
            "pred_log_power": np.asarray(data["pred_log_power"])[indices],
            "truth_log_power": np.load(
                PATHS.cache_dir(501)
                / "truth_metric_cache"
                / "truth_log_power.float32.npy",
                mmap_mode="r",
            )[indices],
        }
        summary = summarize_metrics(metrics, regimes[indices])
        rows.append(
            {
                "name": name,
                "point_metrics": PATHS.relative(path),
                "selection": float(summary["overall"]["selection"]),
                "summary": summary,
            }
        )
    return rows


def seed_aggregates(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, bool, bool], list[dict[str, object]]] = {}
    for result in results:
        config = result["config"]
        key = (
            config["scene_source"],
            int(config["max_depth"]),
            bool(config["diffraction"]),
            bool(config["edge_diffraction"]),
        )
        grouped.setdefault(key, []).append(result)
    rows = []
    for key, items in grouped.items():
        selection = np.asarray([item["selection"] for item in items], dtype=np.float64)
        pdp = np.asarray(
            [item["summary"]["overall"]["pdp_mean"] for item in items], dtype=np.float64
        )
        pas = np.asarray(
            [item["summary"]["overall"]["leaked_full_mean"] for item in items],
            dtype=np.float64,
        )
        prototype = dict(items[0]["config"])
        rows.append(
            {
                "key": {
                    "scene_source": key[0],
                    "max_depth": key[1],
                    "diffraction": key[2],
                    "edge_diffraction": key[3],
                },
                "prototype": prototype,
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

    import sionna
    import mitsuba as mi
    from sionna import config as sionna_config
    from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
    from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene
    from sionna.rt import scene as sionna_scene

    if sionna.__version__ != "0.19.2":
        raise RuntimeError(f"v503 old-solver stage requires Sionna 0.19.2, got {sionna.__version__}")
    if args.mitsuba_variant != "auto" and mi.variant() != args.mitsuba_variant:
        raise RuntimeError(
            f"Sionna selected Mitsuba variant {mi.variant()}, expected {args.mitsuba_variant}. "
            "The old RT stack must choose its variant before any RT classes are imported."
        )
    old_scene_xml = Path(str(sionna_scene.etoile))
    if not old_scene_xml.is_file():
        raise FileNotFoundError(old_scene_xml)

    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    truth = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    regimes = np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"]
    regime_codes = np.full(len(regimes), 2, dtype=np.int8)
    regime_codes[regimes <= 0] = 0
    regime_codes[(regimes >= 1) & (regimes <= 25)] = 1
    regime_codes[(regimes > 25) & (regimes <= 100)] = 2
    regime_codes[regimes > 100] = 3
    truth_log_power = np.load(
        PATHS.cache_dir(501) / "truth_metric_cache" / "truth_log_power.float32.npy",
        mmap_mode="r",
    )

    default_limit = {"smoke": 8, "screen": 160, "refine": 400, "full": len(positions)}[
        args.stage
    ]
    point_limit = min(args.point_limit or default_limit, len(positions))
    if point_limit == len(positions):
        indices = np.arange(len(positions), dtype=np.int64)
    else:
        indices = balanced_indices(positions, regime_codes, truth_log_power, point_limit)
    configs = stage_configs(args.stage)
    if args.budget_cap is not None:
        configs = [replace(config, num_samples=min(config.num_samples, args.budget_cap)) for config in configs]
    if args.config_limit is not None:
        configs = configs[: args.config_limit]
    if args.force:
        for config in configs:
            config_report_path(config, len(indices)).unlink(missing_ok=True)
            point_metric_path(config, len(indices)).unlink(missing_ok=True)

    rt_api = (
        load_scene,
        PlanarArray,
        Receiver,
        Transmitter,
        subcarrier_frequencies,
        cir_to_ofdm_channel,
        sionna_config,
    )
    results = [
        run_config(
            config,
            indices,
            positions,
            truth,
            regime_codes,
            args.batch_size,
            old_scene_xml,
            rt_api,
        )
        for config in configs
    ]
    ranking = sorted(results, key=lambda row: row["selection"], reverse=True)
    controls = current_solver_controls(indices, regime_codes)
    report = {
        "method": VERSION,
        "status": "analysis_complete" if args.stage in {"screen", "refine", "full"} else "smoke_complete",
        "stage": args.stage,
        "primary_hypothesis": (
            "With canonical fixed materials, an older TensorFlow Sionna RT solver may reproduce "
            "official medium/heavy paths better than the Sionna 2.0.1 PathSolver used by v501."
        ),
        "environment": {
            "sionna": sionna.__version__,
            "mitsuba_variant": mi.variant(),
            "single_visible_gpu_required_by_launch": True,
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
            "scattering": False,
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
    print(json.dumps({
        "stage": args.stage,
        "point_count": len(indices),
        "best": ranking[0]["config"]["name"] if ranking else None,
        "best_selection": ranking[0]["selection"] if ranking else None,
        "controls": [{"name": row["name"], "selection": row["selection"]} for row in controls],
        "report": PATHS.relative(stage_report_path(args.stage)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
