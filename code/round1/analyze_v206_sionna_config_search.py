#!/usr/bin/env python3
"""Run staged inverse calibration of Sionna configuration on official train H."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import time

import numpy as np
from scipy.stats import rankdata

from powerfield_common import regime_of
from project_paths import PATHS


VERSION = "v206_sionna_config_search"
REPORT_DIR = PATHS.version_report_dir(206)
OUT_DIR = PATHS.cache_dir(206)
SPLIT_CACHE = OUT_DIR / "v206_spatial_calibration_split.npz"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0


@dataclass(frozen=True)
class SearchConfig:
    name: str
    tx_azimuth_deg: float = -29.0
    tx_tilt_deg: float = 0.0
    rx_orientation: str = "track_bs"
    frequency_hz: float = 6.0e9
    max_depth: int = 3
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    samples_per_src: int = 200_000
    max_num_paths_per_src: int = 200_000
    seed: int = 206


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "orientation",
            "solver",
            "depth_extension",
            "depth_tail",
            "frequency",
            "frequency_refine",
            "budget",
            "frequency_seed",
            "final_audit",
            "final_controls",
        ),
        required=True,
    )
    parser.add_argument("--count-per-split", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N configs for smoke testing")
    return parser.parse_args()


def report_path(stage: str):
    return REPORT_DIR / f"v206_sionna_{stage}_search.json"


def load_previous_best(stage: str) -> SearchConfig:
    previous = {
        "solver": "orientation",
        "depth_extension": "solver",
        "depth_tail": "depth_extension",
        "frequency": "depth_tail",
        "frequency_refine": "frequency",
        "budget": "frequency_refine",
        "frequency_seed": "budget",
        "final_audit": "frequency_seed",
        "final_controls": "frequency_seed",
    }[stage]
    path = report_path(previous)
    if not path.is_file():
        raise FileNotFoundError(f"Run --stage {previous} first: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    candidates = report["calibration_ranking"]
    transferable = max(
        candidates,
        key=lambda item: 0.5
        * (item["calibration"]["selection_score"] + item["validation"]["selection_score"]),
    )
    return SearchConfig(**transferable["config"])


def orientation_configs() -> list[SearchConfig]:
    configs = []
    for azimuth in (-33.0, -30.0, -28.85, -27.0, -24.0):
        for tilt in (0.0, -3.0, -6.0, -9.53, -12.0):
            name = f"az{azimuth:+05.2f}_tilt{tilt:+05.2f}_default200k"
            configs.append(SearchConfig(name=name, tx_azimuth_deg=azimuth, tx_tilt_deg=tilt))
    configs.extend(
        [
            SearchConfig(
                name="v205_baseline_all_d5_60k",
                tx_azimuth_deg=-30.0,
                tx_tilt_deg=-9.53,
                max_depth=5,
                diffuse_reflection=True,
                refraction=True,
                diffraction=True,
                samples_per_src=60_000,
                max_num_paths_per_src=60_000,
            ),
            SearchConfig(
                name="sionna_default_exact_azm30_tilt0_1m",
                tx_azimuth_deg=-30.0,
                tx_tilt_deg=0.0,
                samples_per_src=1_000_000,
                max_num_paths_per_src=1_000_000,
            ),
            SearchConfig(
                name="sionna_default_exact_azm30_tiltm9p53_1m",
                tx_azimuth_deg=-30.0,
                tx_tilt_deg=-9.53,
                samples_per_src=1_000_000,
                max_num_paths_per_src=1_000_000,
            ),
            SearchConfig(
                name="sionna_default_fixed_rx_control",
                tx_azimuth_deg=-30.0,
                tx_tilt_deg=0.0,
                rx_orientation="fixed",
            ),
        ]
    )
    return configs


def solver_configs(base: SearchConfig) -> list[SearchConfig]:
    modes = {
        "default": (False, True, False, False),
        "specular_only": (False, False, False, False),
        "default_plus_diffraction": (False, True, True, False),
        "default_plus_diffuse": (True, True, False, False),
        "all_no_edge": (True, True, True, False),
        "no_diffuse": (False, True, True, False),
        "no_refraction": (True, False, True, False),
        "all_with_edge": (True, True, True, True),
    }
    # no_diffuse and default_plus_diffraction intentionally share flags; retaining
    # only one avoids duplicate expensive solves.
    modes.pop("no_diffuse")
    configs = []
    for depth in (2, 3, 4, 5, 6):
        for mode, (diffuse, refract, diffract, edge) in modes.items():
            configs.append(
                replace(
                    base,
                    name=f"d{depth}_{mode}_200k",
                    max_depth=depth,
                    diffuse_reflection=diffuse,
                    refraction=refract,
                    diffraction=diffract,
                    edge_diffraction=edge,
                    samples_per_src=200_000,
                    max_num_paths_per_src=200_000,
                )
            )
    configs.append(
        replace(
            base,
            name="sionna_default_exact_selected_orientation_1m",
            max_depth=3,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            edge_diffraction=False,
            samples_per_src=1_000_000,
            max_num_paths_per_src=1_000_000,
        )
    )
    return configs


def frequency_configs(base: SearchConfig) -> list[SearchConfig]:
    frequencies = (2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.9, 5.5, 6.0, 6.5, 6.7, 7.0)
    return [
        replace(base, name=f"frequency_{frequency:.1f}ghz", frequency_hz=frequency * 1e9)
        for frequency in frequencies
    ]


def frequency_refine_configs(base: SearchConfig) -> list[SearchConfig]:
    frequencies = np.arange(5.4, 6.21, 0.1)
    return [
        replace(base, name=f"frequency_refine_{frequency:.1f}ghz", frequency_hz=float(frequency * 1e9))
        for frequency in frequencies
    ]


def depth_extension_configs(base: SearchConfig) -> list[SearchConfig]:
    modes = {
        "default": (False, True, False, False),
        "with_diffraction": (False, True, True, False),
        "all_no_edge": (True, True, True, False),
        "all_with_edge": (True, True, True, True),
    }
    configs = []
    for depth in (6, 7, 8, 10):
        for mode, (diffuse, refract, diffract, edge) in modes.items():
            configs.append(
                replace(
                    base,
                    name=f"d{depth}_{mode}_200k_extension",
                    max_depth=depth,
                    diffuse_reflection=diffuse,
                    refraction=refract,
                    diffraction=diffract,
                    edge_diffraction=edge,
                    samples_per_src=200_000,
                    max_num_paths_per_src=200_000,
                )
            )
    return configs


def depth_tail_configs(base: SearchConfig) -> list[SearchConfig]:
    configs = []
    for depth in (10, 12, 14, 16):
        for mode, diffract in (("default", False), ("with_diffraction", True)):
            configs.append(
                replace(
                    base,
                    name=f"d{depth}_{mode}_200k_tail",
                    max_depth=depth,
                    diffuse_reflection=False,
                    refraction=True,
                    diffraction=diffract,
                    edge_diffraction=False,
                    samples_per_src=200_000,
                    max_num_paths_per_src=200_000,
                )
            )
    return configs


def budget_configs(base: SearchConfig) -> list[SearchConfig]:
    configs = []
    for budget in (60_000, 200_000, 1_000_000):
        for seed in (206, 420):
            configs.append(
                replace(
                    base,
                    name=f"budget_{budget}_seed_{seed}",
                    samples_per_src=budget,
                    max_num_paths_per_src=budget,
                    seed=seed,
                )
            )
    return configs


def frequency_seed_configs(base: SearchConfig) -> list[SearchConfig]:
    configs = []
    for frequency in (2.6, 4.9, 5.5, 6.0, 6.5):
        for seed in (42, 420):
            configs.append(
                replace(
                    base,
                    name=f"frequency_{frequency:.1f}ghz_seed_{seed}_1m",
                    frequency_hz=frequency * 1e9,
                    samples_per_src=1_000_000,
                    max_num_paths_per_src=1_000_000,
                    seed=seed,
                )
            )
    return configs


def final_audit_configs() -> list[SearchConfig]:
    report = json.loads(report_path("frequency_seed").read_text(encoding="utf-8"))
    ranked = sorted(
        report["calibration_ranking"],
        key=lambda item: 0.5
        * (item["calibration"]["selection_score"] + item["validation"]["selection_score"]),
        reverse=True,
    )
    return [SearchConfig(**item["config"]) for item in ranked[:3]]


def final_control_configs() -> list[SearchConfig]:
    return [
        SearchConfig(
            name="final_control_v205_all_d5_60k",
            tx_azimuth_deg=-30.0,
            tx_tilt_deg=-9.53,
            frequency_hz=6.0e9,
            max_depth=5,
            diffuse_reflection=True,
            refraction=True,
            diffraction=True,
            samples_per_src=60_000,
            max_num_paths_per_src=60_000,
            seed=205,
        ),
        SearchConfig(
            name="final_control_sionna_default_d3_1m",
            tx_azimuth_deg=-30.0,
            tx_tilt_deg=-9.53,
            frequency_hz=6.0e9,
            max_depth=3,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            samples_per_src=1_000_000,
            max_num_paths_per_src=1_000_000,
            seed=42,
        ),
    ]


def select_subset(indices: np.ndarray, labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(indices) <= count:
        return np.sort(indices.astype(np.int32))
    rng = np.random.default_rng(seed)
    selected = []
    names = ("clear", "light", "medium", "heavy")
    raw = {name: count * np.mean(labels[indices] == name) for name in names}
    allocation = {name: int(math.floor(raw[name])) for name in names}
    while sum(allocation.values()) < count:
        name = max(names, key=lambda item: raw[item] - allocation[item])
        allocation[name] += 1
    for name in names:
        candidates = indices[labels[indices] == name]
        selected.extend(rng.choice(candidates, allocation[name], replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=np.int32))


def convert_cfr(raw: np.ndarray) -> np.ndarray:
    expected = (4, 1, 256, 1, NUM_SUBCARRIERS)
    if raw.ndim != 6 or tuple(raw.shape[1:]) != expected:
        raise ValueError(f"Unexpected CFR shape {raw.shape}")
    return np.transpose(raw[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def cosine(a: np.ndarray, b: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    numerator = np.sum(a * b, axis=axes, dtype=np.float64)
    aa = np.sum(a * a, axis=axes, dtype=np.float64)
    bb = np.sum(b * b, axis=axes, dtype=np.float64)
    return (numerator / np.maximum(np.sqrt(aa * bb), 1e-30)).astype(np.float32)


def metric_values(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    n = len(prediction)
    pred6 = prediction.reshape(n, 2, 16, 8, 4, 192)
    truth6 = truth.reshape(n, 2, 16, 8, 4, 192)

    pred_leaked = pred6.transpose(0, 1, 5, 2, 4, 3)
    truth_leaked = truth6.transpose(0, 1, 5, 2, 4, 3)
    pred_leaked = np.abs(np.fft.fft2(pred_leaked, axes=(2, 3))).astype(np.float32) ** 2
    truth_leaked = np.abs(np.fft.fft2(truth_leaked, axes=(2, 3))).astype(np.float32) ** 2
    leaked_full = cosine(pred_leaked, truth_leaked, (2, 3)).mean(axis=(1, 2, 3))

    pred_leaked_sum = np.sum(pred_leaked, axis=(1, 4, 5), dtype=np.float64)
    truth_leaked_sum = np.sum(truth_leaked, axis=(1, 4, 5), dtype=np.float64)
    leaked_collapsed = cosine(pred_leaked_sum, truth_leaked_sum, (1, 2))

    pred_delay = np.abs(np.fft.ifft(prediction, axis=-1)).astype(np.float32) ** 2
    truth_delay = np.abs(np.fft.ifft(truth, axis=-1)).astype(np.float32) ** 2
    pdp = cosine(pred_delay, truth_delay, (3,)).mean(axis=(1, 2))

    pred_flat = prediction.reshape(n, -1)
    truth_flat = truth.reshape(n, -1)
    pred_power = np.sum(np.abs(pred_flat) ** 2, axis=1, dtype=np.float64)
    truth_power = np.sum(np.abs(truth_flat) ** 2, axis=1, dtype=np.float64)
    complex_dot = np.sum(np.conj(pred_flat) * truth_flat, axis=1, dtype=np.complex128)
    coherence = np.abs(complex_dot) / np.maximum(np.sqrt(pred_power * truth_power), 1e-30)
    return {
        "leaked_full": leaked_full.astype(np.float32),
        "leaked_collapsed": leaked_collapsed.astype(np.float32),
        "pdp": pdp.astype(np.float32),
        "complex_coherence": coherence.astype(np.float32),
        "pred_log_power": np.log(np.maximum(pred_power, 1e-30)).astype(np.float32),
        "truth_log_power": np.log(np.maximum(truth_power, 1e-30)).astype(np.float32),
    }


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = rankdata(np.asarray(a, dtype=np.float64))
    bb = rankdata(np.asarray(b, dtype=np.float64))
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def summarize(values: dict[str, list[np.ndarray]], regimes: np.ndarray) -> dict[str, object]:
    merged = {name: np.concatenate(parts) for name, parts in values.items()}
    power_spearman = rank_correlation(merged["pred_log_power"], merged["truth_log_power"])
    summary: dict[str, object] = {
        "count": int(len(merged["pdp"])),
        "leaked_full_mean": float(np.mean(merged["leaked_full"])),
        "leaked_collapsed_mean": float(np.mean(merged["leaked_collapsed"])),
        "pdp_mean": float(np.mean(merged["pdp"])),
        "complex_coherence_mean": float(np.mean(merged["complex_coherence"])),
        "power_spearman": power_spearman,
        "power_offset_median": float(np.median(merged["truth_log_power"] - merged["pred_log_power"])),
        "simulated_zero_count": int(np.sum(merged["pred_log_power"] < -60.0)),
    }
    summary["selection_score"] = float(
        0.45 * summary["leaked_full_mean"]
        + 0.45 * summary["pdp_mean"]
        + 0.10 * (power_spearman + 1.0) / 2.0
    )
    summary["by_regime"] = {
        name: {
            "count": int(np.sum(regimes == name)),
            "leaked_full_mean": float(np.mean(merged["leaked_full"][regimes == name])),
            "pdp_mean": float(np.mean(merged["pdp"][regimes == name])),
        }
        for name in ("clear", "light", "medium", "heavy")
        if np.any(regimes == name)
    }
    return summary


def tx_target(config: SearchConfig, mi):
    radius = 140.0
    azimuth = math.radians(config.tx_azimuth_deg)
    tilt = math.radians(config.tx_tilt_deg)
    return mi.Point3f(
        float(BS_POSITION[0] + radius * math.cos(azimuth)),
        float(BS_POSITION[1] + radius * math.sin(azimuth)),
        float(BS_POSITION[2] + radius * math.tan(tilt)),
    )


def run_config(
    config: SearchConfig,
    positions: np.ndarray,
    truth: np.ndarray,
    split_code: np.ndarray,
    regimes: np.ndarray,
    batch_size: int,
    mi,
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
    frequencies,
) -> dict[str, object]:
    scene = load_scene(str(SCENE_XML), merge_shapes=True)
    scene.frequency = config.frequency_hz
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
    tx = Transmitter(
        name=f"tx_{config.name}",
        position=mi.Point3f(*[float(value) for value in BS_POSITION]),
        look_at=tx_target(config, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()
    values = {
        split: {name: [] for name in (
            "leaked_full", "leaked_collapsed", "pdp", "complex_coherence", "pred_log_power", "truth_log_power"
        )}
        for split in ("calibration", "validation")
    }
    regime_values = {"calibration": [], "validation": []}
    receiver_names: list[str] = []
    solve_seconds = 0.0
    cfr_seconds = 0.0
    started = time.time()
    for start in range(0, len(positions), batch_size):
        end = min(start + batch_size, len(positions))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for local, position in enumerate(positions[start:end]):
            name = f"rx_{start + local:05d}"
            kwargs = {
                "name": name,
                "position": mi.Point3f(float(position[0]), float(position[1]), float(position[2])),
            }
            if config.rx_orientation == "track_bs":
                kwargs["look_at"] = tx
            else:
                kwargs["orientation"] = mi.Point3f(0.0, 0.0, 0.0)
            scene.add(Receiver(**kwargs))
            receiver_names.append(name)
        t0 = time.time()
        paths = solver(
            scene,
            max_depth=config.max_depth,
            max_num_paths_per_src=config.max_num_paths_per_src,
            samples_per_src=config.samples_per_src,
            synthetic_array=True,
            los=config.los,
            specular_reflection=config.specular_reflection,
            diffuse_reflection=config.diffuse_reflection,
            refraction=config.refraction,
            diffraction=config.diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=True,
            seed=config.seed + start,
        )
        solve_seconds += time.time() - t0
        t1 = time.time()
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        cfr_seconds += time.time() - t1
        metrics = metric_values(convert_cfr(raw), truth[start:end])
        for code, split in ((0, "calibration"), (1, "validation")):
            mask = split_code[start:end] == code
            if not np.any(mask):
                continue
            for name, item in metrics.items():
                values[split][name].append(item[mask])
            regime_values[split].append(regimes[start:end][mask])
    for name in receiver_names:
        scene.remove(name)
    result = {
        "config": asdict(config),
        "calibration": summarize(values["calibration"], np.concatenate(regime_values["calibration"])),
        "validation": summarize(values["validation"], np.concatenate(regime_values["validation"])),
        "seconds": float(time.time() - started),
        "solve_seconds": float(solve_seconds),
        "cfr_seconds": float(cfr_seconds),
    }
    print(
        f"{config.name}: cal={result['calibration']['selection_score']:.6f} "
        f"val={result['validation']['selection_score']:.6f} seconds={result['seconds']:.1f}",
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not SPLIT_CACHE.is_file():
        raise FileNotFoundError("Run analyze_v206_official_system_identification.py first")

    if args.stage == "orientation":
        configs = orientation_configs()
        parent = None
    elif args.stage == "final_audit":
        configs = final_audit_configs()
        parent = {"source": "top-3 transferable configs from frequency_seed"}
    elif args.stage == "final_controls":
        configs = final_control_configs()
        parent = {"source": "v205 and Sionna-default controls"}
    else:
        parent_config = load_previous_best(args.stage)
        parent = asdict(parent_config)
        if args.stage == "solver":
            configs = solver_configs(parent_config)
        elif args.stage == "depth_extension":
            configs = depth_extension_configs(parent_config)
        elif args.stage == "depth_tail":
            configs = depth_tail_configs(parent_config)
        elif args.stage == "frequency":
            configs = frequency_configs(parent_config)
        elif args.stage == "frequency_refine":
            configs = frequency_refine_configs(parent_config)
        elif args.stage == "budget":
            configs = budget_configs(parent_config)
        else:
            configs = frequency_seed_configs(parent_config)
    if args.limit is not None:
        configs = configs[: args.limit]

    split = np.load(SPLIT_CACHE)
    positions_all = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"]
    labels = regime_of(blocked.astype(np.float32)).astype(str)
    if args.stage in ("final_audit", "final_controls"):
        development_calibration = select_subset(split["calibration_indices"], labels, 128, 2063)
        development_validation = select_subset(split["validation_indices"], labels, 128, 2064)
        calibration = np.setdiff1d(split["calibration_indices"], development_calibration).astype(np.int32)
        validation = np.setdiff1d(split["validation_indices"], development_validation).astype(np.int32)
    else:
        calibration = select_subset(split["calibration_indices"], labels, args.count_per_split, 2063)
        validation = select_subset(split["validation_indices"], labels, args.count_per_split, 2064)
    selected = np.concatenate([calibration, validation]).astype(np.int32)
    split_code = np.concatenate(
        [np.zeros(len(calibration), dtype=np.int8), np.ones(len(validation), dtype=np.int8)]
    )
    positions = positions_all[selected]
    regimes = labels[selected]
    official = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    truth = np.asarray(official[selected], dtype=np.complex64)

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    frequencies = subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING).cpu().numpy().astype(np.float32)
    started = time.time()
    results = []
    for config in configs:
        results.append(
            run_config(
                config,
                positions,
                truth,
                split_code,
                regimes,
                args.batch_size,
                mi,
                load_scene,
                Transmitter,
                Receiver,
                PlanarArray,
                PathSolver,
                frequencies,
            )
        )
    calibration_ranking = sorted(results, key=lambda item: item["calibration"]["selection_score"], reverse=True)
    validation_ranking = sorted(results, key=lambda item: item["validation"]["selection_score"], reverse=True)
    calibration_scores = np.asarray([item["calibration"]["selection_score"] for item in results])
    validation_scores = np.asarray([item["validation"]["selection_score"] for item in results])
    winner = calibration_ranking[0]
    transferable_ranking = sorted(
        results,
        key=lambda item: 0.5
        * (item["calibration"]["selection_score"] + item["validation"]["selection_score"]),
        reverse=True,
    )
    winner_validation_rank = next(
        index + 1 for index, item in enumerate(validation_ranking) if item["config"]["name"] == winner["config"]["name"]
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "stage": args.stage,
        "primary_hypothesis": (
            "With canonical Etoile materials fixed, official same-coordinate leaked-PAS/PDP can identify an "
            "equivalent antenna and path-solver configuration that generalizes to disjoint spatial tiles."
        ),
        "scene": PATHS.relative(SCENE_XML),
        "materials": "canonical Sionna Etoile ITU defaults; not optimized",
        "sionna_defaults_reference": {
            "max_depth": 3,
            "samples_per_src": 1_000_000,
            "max_num_paths_per_src": 1_000_000,
            "diffuse_reflection": False,
            "refraction": True,
            "diffraction": False,
        },
        "selection_formula": "0.45 leaked-full-slice PAS + 0.45 per-link PDP + 0.10 rescaled power Spearman",
        "parent_config": parent,
        "count_per_split": int(len(calibration)),
        "calibration_indices": calibration.tolist(),
        "validation_indices": validation.tolist(),
        "config_count": int(len(results)),
        "calibration_validation_score_pearson": float(np.corrcoef(calibration_scores, validation_scores)[0, 1])
        if len(results) > 1
        else None,
        "best_by_calibration": winner,
        "calibration_winner_validation_rank": int(winner_validation_rank),
        "best_by_validation": validation_ranking[0],
        "best_transferable": transferable_ranking[0],
        "calibration_ranking": calibration_ranking,
        "total_seconds": float(time.time() - started),
        "outputs": {"report": PATHS.relative(report_path(args.stage))},
    }
    report_path(args.stage).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": PATHS.relative(report_path(args.stage)),
                "best_by_calibration": winner,
                "calibration_winner_validation_rank": winner_validation_rank,
                "best_by_validation": validation_ranking[0],
                "total_seconds": report["total_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
