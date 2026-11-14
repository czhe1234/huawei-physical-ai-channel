#!/usr/bin/env python3
"""Calibrate Sionna RT configurations against all 2000 official train points."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.fft import fft2, ifft
from scipy.optimize import least_squares
from scipy.stats import rankdata

from project_paths import PATHS


VERSION = "v501_full_official_sionna_inverse_calibration"
REPORT_DIR = PATHS.version_report_dir(501)
CACHE_DIR = PATHS.cache_dir(501)
TRUTH_CACHE_DIR = CACHE_DIR / "truth_metric_cache"
POINT_DIR = CACHE_DIR / "point_metrics"
SCENE_XML = PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
TINY = 1e-30


@dataclass(frozen=True)
class SearchConfig:
    name: str
    tx_azimuth_deg: float = -30.0
    tx_tilt_deg: float = -9.53
    rx_orientation: str = "track_bs"
    synthetic_array: bool = True
    frequency_hz: float = 6.693e9
    max_depth: int = 16
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    samples_per_src: int = 200_000
    max_num_paths_per_src: int = 200_000
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "baselines",
            "depth_modes",
            "depth_high",
            "edge",
            "orientation",
            "array",
            "frequency",
            "budget_seed",
            "final_seed",
            "noise",
        ),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--config-limit", type=int)
    parser.add_argument("--point-limit", type=int)
    parser.add_argument("--budget-cap", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def config_result_path(config: SearchConfig) -> Path:
    return REPORT_DIR / "configs" / f"{safe_name(config.name)}.json"


def point_metric_path(config: SearchConfig) -> Path:
    return POINT_DIR / f"{safe_name(config.name)}.npz"


def stage_report_path(stage: str) -> Path:
    return REPORT_DIR / f"v501_{stage}.json"


def baseline_configs() -> list[SearchConfig]:
    return [
        SearchConfig(
            name="control_sionna_default_d3_1m_6p0",
            frequency_hz=6.0e9,
            max_depth=3,
            samples_per_src=1_000_000,
            max_num_paths_per_src=1_000_000,
        ),
        SearchConfig(
            name="control_v205_d5_60k_all_no_edge_6p0",
            frequency_hz=6.0e9,
            max_depth=5,
            diffuse_reflection=True,
            diffraction=True,
            samples_per_src=60_000,
            max_num_paths_per_src=60_000,
            seed=206,
        ),
        SearchConfig(
            name="control_v206_d16_1m_spec_refract_6p0",
            frequency_hz=6.0e9,
            max_depth=16,
            samples_per_src=1_000_000,
            max_num_paths_per_src=1_000_000,
        ),
        SearchConfig(
            name="physical_d16_1m_spec_refract_6p693",
            frequency_hz=6.693e9,
            max_depth=16,
            samples_per_src=1_000_000,
            max_num_paths_per_src=1_000_000,
        ),
    ]


def depth_mode_configs() -> list[SearchConfig]:
    modes = {
        "specular_only": (False, False, False),
        "specular_refraction": (False, True, False),
        "plus_diffraction": (False, True, True),
        "plus_diffuse": (True, True, False),
        "all_no_edge": (True, True, True),
    }
    configs: list[SearchConfig] = []
    for depth in (3, 6, 8, 10, 12, 14, 16, 20):
        for mode, (diffuse, refract, diffract) in modes.items():
            configs.append(
                SearchConfig(
                    name=f"depth{depth}_{mode}_200k_seed42",
                    max_depth=depth,
                    diffuse_reflection=diffuse,
                    refraction=refract,
                    diffraction=diffract,
                )
            )
    return configs


def high_depth_configs() -> list[SearchConfig]:
    configs = []
    for depth in (24, 28, 32):
        for mode, diffuse in (("specular_refraction", False), ("plus_diffuse", True)):
            configs.append(
                SearchConfig(
                    name=f"depth{depth}_{mode}_200k_seed42",
                    max_depth=depth,
                    diffuse_reflection=diffuse,
                    refraction=True,
                    diffraction=False,
                )
            )
    return configs


def load_stage_results(stage: str) -> list[dict[str, object]]:
    path = stage_report_path(stage)
    if not path.is_file():
        raise FileNotFoundError(f"run stage {stage} first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["ranking"]


def top_configs(stage: str, count: int) -> list[SearchConfig]:
    return [SearchConfig(**item["config"]) for item in load_stage_results(stage)[:count]]


def edge_configs() -> list[SearchConfig]:
    configs = []
    for index, base in enumerate(top_configs("depth_modes", 3)):
        configs.append(
            replace(
                base,
                name=f"edge_top{index + 1}_{base.name}",
                diffraction=True,
                edge_diffraction=True,
            )
        )
    return configs


def best_structural_config() -> SearchConfig:
    candidates = (
        load_stage_results("depth_modes")
        + load_stage_results("depth_high")
        + load_stage_results("edge")
    )
    best = max(candidates, key=lambda item: item["robust_selection"])
    return SearchConfig(**best["config"])


def orientation_configs() -> list[SearchConfig]:
    base = best_structural_config()
    return [
        replace(
            base,
            name=f"orientation_az{azimuth:+05.1f}_tilt{tilt:+05.2f}",
            tx_azimuth_deg=azimuth,
            tx_tilt_deg=tilt,
        )
        for azimuth in (-31.0, -30.5, -30.0, -29.5, -29.0, -28.5)
        for tilt in (-12.0, -9.53, -7.5, -5.0, -2.5, 0.0)
    ]


def array_configs() -> list[SearchConfig]:
    configs = []
    base = top_configs("orientation", 1)[0]
    for rx_orientation in ("track_bs", "fixed"):
        for synthetic_array in (True, False):
            if rx_orientation == "fixed" and not synthetic_array:
                continue
            configs.append(
                replace(
                    base,
                    name=f"array_top1_{rx_orientation}_synthetic{int(synthetic_array)}",
                    rx_orientation=rx_orientation,
                    synthetic_array=synthetic_array,
                )
            )
    return configs


def frequency_configs() -> list[SearchConfig]:
    base = top_configs("array", 1)[0]
    return [
        replace(base, name=f"frequency_{frequency:.3f}ghz", frequency_hz=frequency * 1e9)
        for frequency in (6.0, 6.5, 6.690, 6.692, 6.693, 6.694, 6.696)
    ]


def budget_seed_configs() -> list[SearchConfig]:
    configs = []
    for source, base in enumerate(top_configs("frequency", 3)):
        for budget in (200_000, 1_000_000, 3_000_000):
            for seed in (42, 206, 420):
                configs.append(
                    replace(
                        base,
                        name=f"budget_source{source + 1}_{budget}_seed{seed}",
                        samples_per_src=budget,
                        max_num_paths_per_src=budget,
                        seed=seed,
                    )
                )
    return configs


def final_seed_configs() -> list[SearchConfig]:
    base = top_configs("budget_seed", 1)[0]
    return [
        replace(base, name=f"final_seed_{seed}", seed=seed)
        for seed in (42, 206, 420, 1337, 2026)
    ]


def stage_configs(stage: str) -> list[SearchConfig]:
    return {
        "baselines": baseline_configs,
        "depth_modes": depth_mode_configs,
        "depth_high": high_depth_configs,
        "edge": edge_configs,
        "orientation": orientation_configs,
        "array": array_configs,
        "frequency": frequency_configs,
        "budget_seed": budget_seed_configs,
        "final_seed": final_seed_configs,
    }[stage]()


def unit_normalize(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    value64 = np.asarray(value, dtype=np.float64)
    norm = np.sqrt(np.sum(value64 * value64, axis=axes, keepdims=True, dtype=np.float64))
    return (value / np.maximum(norm, TINY)).astype(np.float32)


def prepare_truth_cache(batch_size: int = 8) -> dict[str, np.ndarray]:
    TRUTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdp_unit": TRUTH_CACHE_DIR / "truth_pdp_unit.float16.npy",
        "leaked_unit": TRUTH_CACHE_DIR / "truth_leaked_unit.float16.npy",
        "leaked_collapsed_unit": TRUTH_CACHE_DIR / "truth_leaked_collapsed_unit.float16.npy",
        "conventional_unit": TRUTH_CACHE_DIR / "truth_conventional_unit.float16.npy",
        "log_power": TRUTH_CACHE_DIR / "truth_log_power.float32.npy",
        "zero_mask": TRUTH_CACHE_DIR / "truth_zero_mask.npy",
    }
    if all(path.is_file() for path in paths.values()):
        return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}

    truth = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    count = len(truth)
    pdp_unit = np.lib.format.open_memmap(
        paths["pdp_unit"], mode="w+", dtype=np.float16, shape=(count, 256, 4, 192)
    )
    leaked_unit = np.lib.format.open_memmap(
        paths["leaked_unit"],
        mode="w+",
        dtype=np.float16,
        shape=(count, 2, 192, 16, 4, 8),
    )
    leaked_collapsed_unit = np.lib.format.open_memmap(
        paths["leaked_collapsed_unit"], mode="w+", dtype=np.float16, shape=(count, 192, 16)
    )
    conventional_unit = np.lib.format.open_memmap(
        paths["conventional_unit"], mode="w+", dtype=np.float16, shape=(count, 16, 8)
    )
    log_power = np.lib.format.open_memmap(
        paths["log_power"], mode="w+", dtype=np.float32, shape=(count,)
    )
    zero_mask = np.lib.format.open_memmap(
        paths["zero_mask"], mode="w+", dtype=np.bool_, shape=(count,)
    )
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        channel = np.asarray(truth[start:end], dtype=np.complex64)
        delay = np.abs(ifft(channel, axis=-1, workers=1)).astype(np.float32) ** 2
        pdp_unit[start:end] = unit_normalize(delay, (3,)).astype(np.float16)

        channel6 = channel.reshape(end - start, 2, 16, 8, 4, 192)
        leaked = channel6.transpose(0, 1, 5, 2, 4, 3)
        leaked = np.abs(fft2(leaked, axes=(2, 3), workers=1)).astype(np.float32) ** 2
        leaked_unit[start:end] = unit_normalize(leaked, (2, 3)).astype(np.float16)
        leaked_collapsed = np.sum(leaked, axis=(1, 4, 5), dtype=np.float32)
        leaked_collapsed_unit[start:end] = unit_normalize(
            leaked_collapsed, (1, 2)
        ).astype(np.float16)

        conventional = np.abs(fft2(channel6, axes=(2, 3), workers=1)).astype(np.float32) ** 2
        conventional = np.sum(conventional, axis=(1, 4, 5), dtype=np.float32)
        conventional_unit[start:end] = unit_normalize(conventional, (1, 2)).astype(np.float16)

        power = np.sum(np.abs(channel.reshape(end - start, -1)) ** 2, axis=1, dtype=np.float64)
        log_power[start:end] = np.log(np.maximum(power, TINY)).astype(np.float32)
        zero_mask[start:end] = power <= 1e-30
        print(f"truth metric cache {end}/{count}", flush=True)
    for value in (
        pdp_unit,
        leaked_unit,
        leaked_collapsed_unit,
        conventional_unit,
        log_power,
        zero_mask,
    ):
        value.flush()
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


def spatial_fold_codes(positions: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    cache = CACHE_DIR / "v501_spatial_five_folds.npy"
    if cache.is_file():
        folds = np.load(cache)
        if len(folds) == len(positions):
            return folds.astype(np.int8)
    xy = positions[:, :2]
    origin = np.min(xy, axis=0)
    tile_xy = np.floor((xy - origin[None, :]) / 55.0).astype(np.int32)
    _, group_inverse = np.unique(tile_xy, axis=0, return_inverse=True)
    group_count = int(group_inverse.max()) + 1
    group_sizes = np.bincount(group_inverse, minlength=group_count)
    group_regimes = np.zeros((group_count, 4), dtype=np.int32)
    for group in range(group_count):
        for regime in range(4):
            group_regimes[group, regime] = int(
                np.sum((group_inverse == group) & (regimes == regime))
            )
    fold_sizes = np.zeros(5, dtype=np.int32)
    fold_regimes = np.zeros((5, 4), dtype=np.int32)
    group_fold = np.full(group_count, -1, dtype=np.int8)
    target_size = len(positions) / 5.0
    target_regime = np.bincount(regimes, minlength=4) / 5.0
    for group in np.argsort(-group_sizes):
        costs = []
        for fold in range(5):
            size_cost = ((fold_sizes[fold] + group_sizes[group]) / target_size) ** 2
            regime_cost = np.mean(
                ((fold_regimes[fold] + group_regimes[group]) / np.maximum(target_regime, 1.0)) ** 2
            )
            costs.append(size_cost + 0.35 * regime_cost)
        selected = int(np.argmin(costs))
        group_fold[group] = selected
        fold_sizes[selected] += group_sizes[group]
        fold_regimes[selected] += group_regimes[group]
    folds = group_fold[group_inverse]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, folds)
    return folds


def regime_codes() -> np.ndarray:
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"]
    codes = np.full(len(blocked), 2, dtype=np.int8)
    codes[blocked <= 0] = 0
    codes[(blocked >= 1) & (blocked <= 25)] = 1
    codes[(blocked > 25) & (blocked <= 100)] = 2
    codes[blocked > 100] = 3
    return codes


def prediction_metrics(
    prediction: np.ndarray,
    truth_channel: np.ndarray,
    truth_cache: dict[str, np.ndarray],
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    count = end - start
    pred_delay = np.abs(ifft(prediction, axis=-1, workers=1)).astype(np.float32) ** 2
    pred_delay = unit_normalize(pred_delay, (3,))
    pdp = np.clip(np.sum(
        pred_delay * np.asarray(truth_cache["pdp_unit"][start:end], dtype=np.float32),
        axis=3,
        dtype=np.float64,
    ).mean(axis=(1, 2)), 0.0, 1.0)

    pred6 = prediction.reshape(count, 2, 16, 8, 4, 192)
    pred_leaked = pred6.transpose(0, 1, 5, 2, 4, 3)
    pred_leaked = np.abs(fft2(pred_leaked, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    pred_leaked_unit = unit_normalize(pred_leaked, (2, 3))
    leaked_full = np.clip(np.sum(
        pred_leaked_unit
        * np.asarray(truth_cache["leaked_unit"][start:end], dtype=np.float32),
        axis=(2, 3),
        dtype=np.float64,
    ).mean(axis=(1, 2, 3)), 0.0, 1.0)
    pred_collapsed = np.sum(pred_leaked, axis=(1, 4, 5), dtype=np.float32)
    pred_collapsed = unit_normalize(pred_collapsed, (1, 2))
    leaked_collapsed = np.clip(np.sum(
        pred_collapsed
        * np.asarray(truth_cache["leaked_collapsed_unit"][start:end], dtype=np.float32),
        axis=(1, 2),
        dtype=np.float64,
    ), 0.0, 1.0)

    conventional = np.abs(fft2(pred6, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    conventional = np.sum(conventional, axis=(1, 4, 5), dtype=np.float32)
    conventional = unit_normalize(conventional, (1, 2))
    conventional_pas = np.clip(np.sum(
        conventional
        * np.asarray(truth_cache["conventional_unit"][start:end], dtype=np.float32),
        axis=(1, 2),
        dtype=np.float64,
    ), 0.0, 1.0)

    pred_flat = prediction.reshape(count, -1)
    truth_flat = truth_channel.reshape(count, -1)
    pred_power = np.sum(np.abs(pred_flat) ** 2, axis=1, dtype=np.float64)
    truth_power = np.sum(np.abs(truth_flat) ** 2, axis=1, dtype=np.float64)
    complex_dot = np.sum(np.conj(pred_flat) * truth_flat, axis=1, dtype=np.complex128)
    coherence = np.abs(complex_dot) / np.maximum(np.sqrt(pred_power * truth_power), TINY)
    return {
        "pdp": pdp.astype(np.float32),
        "leaked_full": leaked_full.astype(np.float32),
        "leaked_collapsed": leaked_collapsed.astype(np.float32),
        "conventional_pas": conventional_pas.astype(np.float32),
        "complex_coherence": coherence.astype(np.float32),
        "pred_log_power": np.log(np.maximum(pred_power, TINY)).astype(np.float32),
    }


def estimate_noise_proxy(
    truth: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    count = len(truth)
    noise_variance = np.empty(count, dtype=np.float64)
    observed_mean_power = np.empty(count, dtype=np.float64)
    exponential_q25 = -math.log(0.75)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        channel = np.asarray(truth[start:end], dtype=np.complex64)
        delay_power = np.abs(ifft(channel, axis=-1, workers=1)).astype(np.float64) ** 2
        link_floor = np.quantile(delay_power, 0.25, axis=-1) / exponential_q25
        noise_variance[start:end] = NUM_SUBCARRIERS * np.median(
            link_floor.reshape(end - start, -1), axis=1
        )
        observed_mean_power[start:end] = np.mean(
            np.abs(channel.reshape(end - start, -1)) ** 2,
            axis=1,
            dtype=np.float64,
        )
    return noise_variance, observed_mean_power


def fit_noise_relations(
    noise_variance: np.ndarray,
    observed_mean_power: np.ndarray,
    zero_mask: np.ndarray,
) -> dict[str, object]:
    mask = (
        (~zero_mask)
        & np.isfinite(noise_variance)
        & np.isfinite(observed_mean_power)
        & (noise_variance > TINY)
        & (observed_mean_power > TINY)
    )
    power = observed_mean_power[mask]
    variance = noise_variance[mask]
    log_power = np.log(power)
    log_variance = np.log(variance)

    linear_initial = np.polyfit(log_power, log_variance, 1)
    linear_fit = least_squares(
        lambda value: value[0] + value[1] * log_power - log_variance,
        x0=np.asarray([linear_initial[1], linear_initial[0]], dtype=np.float64),
        bounds=([-100.0, 0.0], [100.0, 2.0]),
        loss="soft_l1",
    )
    power_log_a, power_gamma = [float(value) for value in linear_fit.x]

    floor_initial = max(float(np.quantile(variance, 0.05)), TINY)
    affine_fit = least_squares(
        lambda value: (
            np.logaddexp(value[0] + value[1] * log_power, value[2]) - log_variance
        ),
        x0=np.asarray(
            [power_log_a, power_gamma, math.log(floor_initial)], dtype=np.float64
        ),
        bounds=([-100.0, 0.0, -100.0], [100.0, 2.0, 100.0]),
        loss="soft_l1",
        max_nfev=5000,
    )
    affine_log_a, affine_gamma, affine_log_b = [
        float(value) for value in affine_fit.x
    ]

    fixed = float(np.median(variance))
    proportional_ratio = float(np.median(variance / power))
    models = {
        "fixed": {"variance": fixed},
        "proportional": {"ratio": proportional_ratio},
        "power_law": {"a": math.exp(power_log_a), "gamma": power_gamma},
        "affine_power_law": {
            "a": math.exp(affine_log_a),
            "gamma": affine_gamma,
            "b": math.exp(affine_log_b),
        },
    }
    predictions = {
        "fixed": np.full_like(variance, fixed),
        "proportional": proportional_ratio * power,
        "power_law": math.exp(power_log_a) * np.power(power, power_gamma),
        "affine_power_law": (
            math.exp(affine_log_a) * np.power(power, affine_gamma)
            + math.exp(affine_log_b)
        ),
    }
    diagnostics = {}
    for name, prediction in predictions.items():
        residual = np.log(np.maximum(prediction, TINY)) - log_variance
        diagnostics[name] = {
            "log_rmse": float(np.sqrt(np.mean(residual * residual))),
            "log_mae": float(np.mean(np.abs(residual))),
        }
    return {
        "estimator": (
            "Per official point, estimate the delay-domain white floor from the 25th "
            "power quantile of every link, correct the exponential quantile bias, then "
            "take the median across all 1024 links and map it back to CFR variance."
        ),
        "fit_count": int(np.sum(mask)),
        "proxy_power_log_pearson": float(np.corrcoef(log_power, log_variance)[0, 1]),
        "models": models,
        "fit_diagnostics": diagnostics,
    }


def noise_model_specs(fit: dict[str, object]) -> list[dict[str, object]]:
    models = fit["models"]
    specs: list[dict[str, object]] = [
        {"name": "clean_no_noise", "family": "clean", "factor": 0.0}
    ]
    for family in ("fixed", "proportional", "power_law", "affine_power_law"):
        for factor in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0):
            specs.append(
                {
                    "name": f"{family}_factor_{factor:g}",
                    "family": family,
                    "factor": factor,
                    "parameters": models[family],
                }
            )
    return specs


def model_noise_variance(
    spec: dict[str, object], clean_mean_power: np.ndarray
) -> np.ndarray:
    family = spec["family"]
    factor = float(spec["factor"])
    if family == "clean":
        return np.zeros_like(clean_mean_power, dtype=np.float64)
    parameters = spec["parameters"]
    if family == "fixed":
        value = np.full_like(clean_mean_power, float(parameters["variance"]))
    elif family == "proportional":
        value = float(parameters["ratio"]) * clean_mean_power
    elif family == "power_law":
        value = float(parameters["a"]) * np.power(
            np.maximum(clean_mean_power, TINY), float(parameters["gamma"])
        )
    elif family == "affine_power_law":
        value = (
            float(parameters["a"])
            * np.power(
                np.maximum(clean_mean_power, TINY), float(parameters["gamma"])
            )
            + float(parameters["b"])
        )
    else:
        raise ValueError(f"unknown noise family: {family}")
    return factor * value


def noise_adjusted_metrics(
    prediction: np.ndarray,
    truth_channel: np.ndarray,
    truth_cache: dict[str, np.ndarray],
    start: int,
    end: int,
    noise_variance: np.ndarray,
) -> dict[str, np.ndarray]:
    count = end - start
    sigma2 = np.asarray(noise_variance, dtype=np.float32)
    pred_delay = np.abs(ifft(prediction, axis=-1, workers=1)).astype(np.float32) ** 2
    pred_delay += sigma2[:, None, None, None] / NUM_SUBCARRIERS
    pred_delay = unit_normalize(pred_delay, (3,))
    pdp = np.clip(
        np.sum(
            pred_delay
            * np.asarray(truth_cache["pdp_unit"][start:end], dtype=np.float32),
            axis=3,
            dtype=np.float64,
        ).mean(axis=(1, 2)),
        0.0,
        1.0,
    )

    pred6 = prediction.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    pred_leaked = pred6.transpose(0, 1, 5, 2, 4, 3)
    pred_leaked = np.abs(fft2(pred_leaked, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    pred_leaked += sigma2[:, None, None, None, None, None] * (NUM_SUBCARRIERS * 16)
    pred_leaked_unit = unit_normalize(pred_leaked, (2, 3))
    leaked_full = np.clip(
        np.sum(
            pred_leaked_unit
            * np.asarray(truth_cache["leaked_unit"][start:end], dtype=np.float32),
            axis=(2, 3),
            dtype=np.float64,
        ).mean(axis=(1, 2, 3)),
        0.0,
        1.0,
    )
    pred_collapsed = np.sum(pred_leaked, axis=(1, 4, 5), dtype=np.float32)
    pred_collapsed = unit_normalize(pred_collapsed, (1, 2))
    leaked_collapsed = np.clip(
        np.sum(
            pred_collapsed
            * np.asarray(truth_cache["leaked_collapsed_unit"][start:end], dtype=np.float32),
            axis=(1, 2),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    conventional = np.abs(fft2(pred6, axes=(2, 3), workers=1)).astype(np.float32) ** 2
    conventional += sigma2[:, None, None, None, None, None] * (16 * 8)
    conventional = np.sum(conventional, axis=(1, 4, 5), dtype=np.float32)
    conventional = unit_normalize(conventional, (1, 2))
    conventional_pas = np.clip(
        np.sum(
            conventional
            * np.asarray(truth_cache["conventional_unit"][start:end], dtype=np.float32),
            axis=(1, 2),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    pred_flat = prediction.reshape(count, -1)
    truth_flat = truth_channel.reshape(count, -1)
    pred_power = np.sum(np.abs(pred_flat) ** 2, axis=1, dtype=np.float64)
    truth_power = np.sum(np.abs(truth_flat) ** 2, axis=1, dtype=np.float64)
    complex_dot = np.sum(np.conj(pred_flat) * truth_flat, axis=1, dtype=np.complex128)
    coherence = np.abs(complex_dot) / np.maximum(np.sqrt(pred_power * truth_power), TINY)
    expected_power = pred_power + pred_flat.shape[1] * noise_variance
    return {
        "pdp": pdp.astype(np.float32),
        "leaked_full": leaked_full.astype(np.float32),
        "leaked_collapsed": leaked_collapsed.astype(np.float32),
        "conventional_pas": conventional_pas.astype(np.float32),
        "complex_coherence": coherence.astype(np.float32),
        "pred_log_power": np.log(np.maximum(expected_power, TINY)).astype(np.float32),
    }


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = rankdata(np.asarray(a, dtype=np.float64))
    bb = rankdata(np.asarray(b, dtype=np.float64))
    if np.std(aa) <= TINY or np.std(bb) <= TINY:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def summarize_mask(
    metrics: dict[str, np.ndarray], truth_log_power: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}
    power_spearman = rank_correlation(metrics["pred_log_power"][mask], truth_log_power[mask])
    power_rank = (power_spearman + 1.0) / 2.0
    offset = float(np.median(truth_log_power[mask] - metrics["pred_log_power"][mask]))
    power_mae = np.abs(metrics["pred_log_power"][mask] + offset - truth_log_power[mask])
    leaked = float(np.mean(metrics["leaked_full"][mask]))
    pdp = float(np.mean(metrics["pdp"][mask]))
    coverage = float(np.mean(metrics["pred_log_power"][mask] > -60.0))
    legacy = 0.45 * leaked + 0.45 * pdp + 0.10 * power_rank
    online = 0.60 * pdp + 0.25 * leaked + 0.10 * power_rank + 0.05 * coverage
    return {
        "count": count,
        "pdp_mean": pdp,
        "leaked_full_mean": leaked,
        "leaked_collapsed_mean": float(np.mean(metrics["leaked_collapsed"][mask])),
        "conventional_pas_mean": float(np.mean(metrics["conventional_pas"][mask])),
        "complex_coherence_mean": float(np.mean(metrics["complex_coherence"][mask])),
        "power_spearman": power_spearman,
        "power_offset_median": offset,
        "calibrated_log_power_mae": float(np.mean(power_mae)),
        "coverage": coverage,
        "simulated_zero_count": int(np.sum(metrics["pred_log_power"][mask] <= -60.0)),
        "legacy_selection": legacy,
        "online_informed_selection": online,
        "selection_mean": 0.5 * (legacy + online),
    }


def summarize_config(
    config: SearchConfig,
    metrics: dict[str, np.ndarray],
    truth_log_power: np.ndarray,
    zero_mask: np.ndarray,
    folds: np.ndarray,
    regimes: np.ndarray,
    seconds: float,
) -> dict[str, object]:
    nonzero = ~zero_mask
    fold_reports = []
    for fold in range(5):
        mask = nonzero & (folds == fold)
        fold_reports.append({"fold": fold, **summarize_mask(metrics, truth_log_power, mask)})
    fold_scores = np.asarray(
        [item["selection_mean"] for item in fold_reports if item["count"] > 0],
        dtype=np.float64,
    )
    if len(fold_scores) == 0:
        fold_scores = np.asarray([0.0], dtype=np.float64)
    by_regime = {
        name: summarize_mask(metrics, truth_log_power, nonzero & (regimes == code))
        for code, name in enumerate(("clear", "light", "medium", "heavy"))
    }
    result = {
        "method": VERSION,
        "config": asdict(config),
        "all_2000": summarize_mask(metrics, truth_log_power, np.ones(len(zero_mask), dtype=bool)),
        "physical_nonzero_1984": summarize_mask(metrics, truth_log_power, nonzero),
        "official_exact_zero_16": summarize_mask(metrics, truth_log_power, zero_mask),
        "spatial_folds": fold_reports,
        "by_regime": by_regime,
        "fold_selection_mean": float(np.mean(fold_scores)),
        "fold_selection_std": float(np.std(fold_scores)),
        "fold_selection_min": float(np.min(fold_scores)),
        "robust_selection": float(np.mean(fold_scores) - 0.25 * np.std(fold_scores)),
        "seconds": float(seconds),
        "point_metrics": PATHS.relative(point_metric_path(config)),
    }
    return result


def tx_target(config: SearchConfig, mi):
    radius = 140.0
    azimuth = math.radians(config.tx_azimuth_deg)
    tilt = math.radians(config.tx_tilt_deg)
    return mi.Point3f(
        float(BS_POSITION[0] + radius * math.cos(azimuth)),
        float(BS_POSITION[1] + radius * math.sin(azimuth)),
        float(BS_POSITION[2] + radius * math.tan(tilt)),
    )


def convert_cfr(raw: np.ndarray, count: int) -> np.ndarray:
    expected = (4, 1, 256, 1, NUM_SUBCARRIERS)
    if raw.ndim != 6 or raw.shape[0] != count or tuple(raw.shape[1:]) != expected:
        raise ValueError(f"unexpected CFR shape: {raw.shape}")
    return np.transpose(raw[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def run_config(
    config: SearchConfig,
    positions: np.ndarray,
    truth: np.ndarray,
    truth_cache: dict[str, np.ndarray],
    folds: np.ndarray,
    regimes: np.ndarray,
    batch_size: int,
    mi,
    dr,
    rt_api,
) -> dict[str, object]:
    result_path = config_result_path(config)
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))

    load_scene, Transmitter, Receiver, PlanarArray, PathSolver, frequencies = rt_api
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
        name="tx",
        position=mi.Point3f(*[float(value) for value in BS_POSITION]),
        look_at=tx_target(config, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()
    metrics = {
        name: np.empty(len(positions), dtype=np.float32)
        for name in (
            "pdp",
            "leaked_full",
            "leaked_collapsed",
            "conventional_pas",
            "complex_coherence",
            "pred_log_power",
        )
    }
    receiver_names: list[str] = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(positions), batch_size)):
        end = min(start + batch_size, len(positions))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for local, position in enumerate(positions[start:end]):
            name = f"rx_{start + local:05d}"
            kwargs = {"name": name, "position": mi.Point3f(*[float(v) for v in position])}
            if config.rx_orientation == "track_bs":
                kwargs["look_at"] = tx
            else:
                kwargs["orientation"] = mi.Point3f(0.0, 0.0, 0.0)
            scene.add(Receiver(**kwargs))
            receiver_names.append(name)
        paths = solver(
            scene,
            max_depth=config.max_depth,
            max_num_paths_per_src=config.max_num_paths_per_src,
            samples_per_src=config.samples_per_src,
            synthetic_array=config.synthetic_array,
            los=config.los,
            specular_reflection=config.specular_reflection,
            diffuse_reflection=config.diffuse_reflection,
            refraction=config.refraction,
            diffraction=config.diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=True,
            seed=config.seed + batch_index,
        )
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        prediction = convert_cfr(raw, end - start)
        batch_truth = np.asarray(truth[start:end], dtype=np.complex64)
        batch_metrics = prediction_metrics(
            prediction, batch_truth, truth_cache, start, end
        )
        for name, value in batch_metrics.items():
            metrics[name][start:end] = value
        del paths, raw, prediction, batch_truth, batch_metrics
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
    np.savez_compressed(point_metric_path(config), **metrics)
    result = summarize_config(
        config,
        metrics,
        np.asarray(truth_cache["log_power"][: len(positions)], dtype=np.float32),
        np.asarray(truth_cache["zero_mask"][: len(positions)], dtype=bool),
        folds,
        regimes,
        time.time() - started,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (REPORT_DIR / "all_runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(
        f"{config.name}: robust={result['robust_selection']:.6f} "
        f"mean={result['fold_selection_mean']:.6f} min={result['fold_selection_min']:.6f}",
        flush=True,
    )
    return result


def run_noise_stage(
    positions: np.ndarray,
    truth: np.ndarray,
    truth_cache: dict[str, np.ndarray],
    folds: np.ndarray,
    regimes: np.ndarray,
    batch_size: int,
    mi,
    dr,
    rt_api,
) -> dict[str, object]:
    config = top_configs("final_seed", 1)[0]
    proxy_path = CACHE_DIR / "v501_truth_noise_proxy.npz"
    if proxy_path.is_file():
        proxy = np.load(proxy_path)
        noise_proxy = np.asarray(proxy["noise_variance"], dtype=np.float64)
        observed_mean_power = np.asarray(proxy["observed_mean_power"], dtype=np.float64)
    else:
        noise_proxy, observed_mean_power = estimate_noise_proxy(truth, batch_size)
        np.savez_compressed(
            proxy_path,
            noise_variance=noise_proxy,
            observed_mean_power=observed_mean_power,
        )
    zero_mask = np.asarray(truth_cache["zero_mask"], dtype=bool)
    fit = fit_noise_relations(noise_proxy, observed_mean_power, zero_mask)
    specs = noise_model_specs(fit)

    clean_metrics_path = point_metric_path(config)
    if not clean_metrics_path.is_file():
        raise FileNotFoundError(f"run final_seed first: {clean_metrics_path}")
    clean_metrics = np.load(clean_metrics_path)
    truth_log_power = np.asarray(truth_cache["log_power"], dtype=np.float32)
    nonzero = ~zero_mask
    log_power_offset = float(
        np.median(truth_log_power[nonzero] - clean_metrics["pred_log_power"][nonzero])
    )
    clean_power_gain = math.exp(log_power_offset)

    metric_names = (
        "pdp",
        "leaked_full",
        "leaked_collapsed",
        "conventional_pas",
        "complex_coherence",
        "pred_log_power",
    )
    candidate_metrics = {
        str(spec["name"]): {
            metric: np.empty(len(positions), dtype=np.float32) for metric in metric_names
        }
        for spec in specs
    }

    load_scene, Transmitter, Receiver, PlanarArray, PathSolver, frequencies = rt_api
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
        name="tx",
        position=mi.Point3f(*[float(value) for value in BS_POSITION]),
        look_at=tx_target(config, mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    solver = PathSolver()
    receiver_names: list[str] = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(positions), batch_size)):
        end = min(start + batch_size, len(positions))
        for name in receiver_names:
            scene.remove(name)
        receiver_names = []
        for local, position in enumerate(positions[start:end]):
            name = f"rx_{start + local:05d}"
            kwargs = {"name": name, "position": mi.Point3f(*[float(v) for v in position])}
            if config.rx_orientation == "track_bs":
                kwargs["look_at"] = tx
            else:
                kwargs["orientation"] = mi.Point3f(0.0, 0.0, 0.0)
            scene.add(Receiver(**kwargs))
            receiver_names.append(name)
        paths = solver(
            scene,
            max_depth=config.max_depth,
            max_num_paths_per_src=config.max_num_paths_per_src,
            samples_per_src=config.samples_per_src,
            synthetic_array=config.synthetic_array,
            los=config.los,
            specular_reflection=config.specular_reflection,
            diffuse_reflection=config.diffuse_reflection,
            refraction=config.refraction,
            diffraction=config.diffraction,
            edge_diffraction=config.edge_diffraction,
            diffraction_lit_region=True,
            seed=config.seed + batch_index,
        )
        raw = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        prediction = convert_cfr(raw, end - start)
        prediction *= np.float32(math.sqrt(clean_power_gain))
        batch_truth = np.asarray(truth[start:end], dtype=np.complex64)
        clean_mean_power = np.mean(
            np.abs(prediction.reshape(end - start, -1)) ** 2,
            axis=1,
            dtype=np.float64,
        )
        for spec in specs:
            name = str(spec["name"])
            sigma2 = model_noise_variance(spec, clean_mean_power)
            batch_metrics = noise_adjusted_metrics(
                prediction,
                batch_truth,
                truth_cache,
                start,
                end,
                sigma2,
            )
            for metric, value in batch_metrics.items():
                candidate_metrics[name][metric][start:end] = value
        del paths, raw, prediction, batch_truth
        dr.sync_thread()
        if (batch_index + 1) % 25 == 0 or end == len(positions):
            print(
                f"noise_models: {end}/{len(positions)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    for name in receiver_names:
        scene.remove(name)
    dr.sync_thread()
    flush_cache = getattr(dr, "flush_malloc_cache", None)
    if callable(flush_cache):
        flush_cache()

    point_archive = POINT_DIR / "noise_models_all_point_metrics.npz"
    POINT_DIR.mkdir(parents=True, exist_ok=True)
    archive_values = {
        "noise_proxy": noise_proxy.astype(np.float32),
        "observed_mean_power": observed_mean_power.astype(np.float32),
    }
    for name, metrics in candidate_metrics.items():
        for metric, value in metrics.items():
            archive_values[f"{safe_name(name)}__{metric}"] = value
    np.savez_compressed(point_archive, **archive_values)

    ranking = []
    for spec in specs:
        name = str(spec["name"])
        result = summarize_config(
            config,
            candidate_metrics[name],
            truth_log_power,
            zero_mask,
            folds,
            regimes,
            time.time() - started,
        )
        result["noise_model"] = spec
        result["point_metrics"] = PATHS.relative(point_archive)
        ranking.append(result)
    ranking.sort(key=lambda item: item["robust_selection"], reverse=True)
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "stage": "noise",
        "point_count": int(len(positions)),
        "clean_config": asdict(config),
        "clean_global_power_gain": clean_power_gain,
        "clean_global_power_gain_db": 10.0 * math.log10(clean_power_gain),
        "noise_proxy_fit": fit,
        "interpretation_limit": (
            "The delay-tail estimator is a noise proxy because unresolved clean multipath and "
            "finite-bandwidth leakage also contribute to the tail. Ranking therefore tests "
            "whether an expected white spectral floor explains official spectra; it does not "
            "claim direct recovery of the organizer's random noise realization."
        ),
        "storage_policy": (
            "The clean CFR and Paths objects are released after every batch. Only one compact "
            "point-metric archive and this JSON report are retained."
        ),
        "ranking": ranking,
        "total_seconds": float(time.time() - started),
    }
    report_path = stage_report_path("noise")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": PATHS.relative(report_path),
                "best": ranking[0] if ranking else None,
                "total_seconds": report["total_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
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

    configs = [] if args.stage == "noise" else stage_configs(args.stage)
    if args.budget_cap is not None:
        configs = [
            replace(
                config,
                samples_per_src=min(config.samples_per_src, args.budget_cap),
                max_num_paths_per_src=min(config.max_num_paths_per_src, args.budget_cap),
                name=f"{config.name}_cap{args.budget_cap}",
            )
            for config in configs
        ]
    if args.config_limit is not None:
        configs = configs[: args.config_limit]
    if args.point_limit is not None:
        configs = [
            replace(config, name=f"{config.name}_points{args.point_limit}")
            for config in configs
        ]
        positions = positions[: args.point_limit]
        truth = truth[: args.point_limit]
        regimes = regimes[: args.point_limit]
        folds = folds[: args.point_limit]
        truth_cache = {name: value[: args.point_limit] for name, value in truth_cache.items()}

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
    if args.stage == "noise":
        if args.point_limit is not None or args.budget_cap is not None:
            raise ValueError("noise stage requires all 2000 points and the selected full budget")
        run_noise_stage(
            positions,
            truth,
            truth_cache,
            folds,
            regimes,
            args.batch_size,
            mi,
            dr,
            rt_api,
        )
        return
    started = time.time()
    results = []
    for config in configs:
        if args.force:
            config_result_path(config).unlink(missing_ok=True)
            point_metric_path(config).unlink(missing_ok=True)
        results.append(
            run_config(
                config,
                positions,
                truth,
                truth_cache,
                folds,
                regimes,
                args.batch_size,
                mi,
                dr,
                rt_api,
            )
        )
    ranking = sorted(results, key=lambda item: item["robust_selection"], reverse=True)
    report = {
        "method": VERSION,
        "status": "analysis_in_progress",
        "stage": args.stage,
        "point_count": int(len(positions)),
        "config_count": int(len(configs)),
        "sionna_version": getattr(sionna, "__version__", "unknown"),
        "mitsuba_variant": mi.variant(),
        "scene": PATHS.relative(SCENE_XML),
        "materials": "canonical Etoile ITU materials; fixed",
        "selection": (
            "mean of legacy and online-informed PAS/PDP/power scores across all 1984 "
            "nonzero official points, minus 0.25 times five-spatial-fold std"
        ),
        "storage_policy": (
            "No full H or Paths object is persisted per configuration. Only compact point metrics "
            "and JSON evidence are retained; batch tensors are released immediately."
        ),
        "ranking": ranking,
        "total_seconds": float(time.time() - started),
    }
    report_path = stage_report_path(args.stage)
    if args.point_limit is not None or args.budget_cap is not None:
        report_path = REPORT_DIR / (
            f"v501_smoke_{args.stage}_points{len(positions)}_"
            f"cap{args.budget_cap or 'none'}.json"
        )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": PATHS.relative(report_path),
                "best": ranking[0] if ranking else None,
                "total_seconds": report["total_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
