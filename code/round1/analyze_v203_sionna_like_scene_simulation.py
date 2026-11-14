#!/usr/bin/env python3
"""v203 analysis-only: official-like controllable Sionna proxy simulation.

This script does not generate a leaderboard submission.  It builds a small
controlled benchmark from the official train/test positions and the restored
Sionna Etoile path-candidate cache.  The goal is to test why total-power
prediction can remain hard even when the high-level scene geometry is known.

The generated target is summary-level, not a full 3 GB channel tensor:

* clean total power for train/test;
* noisy train observations, mimicking "clean channel then noisy released train";
* simplified delay spectra from path delays.

Several predictors are trained on the synthetic train observations and tested
against the clean synthetic test truth:

* position/regime kNN;
* official visible point-cloud and geometry models;
* path-candidate models without hidden material/mesh identity;
* path-candidate models with material/mesh identity;
* a v200-style selective two-sided correction gate;
* oracles that reveal clean train labels or latent path/fading summaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


VERSION = "v203_sionna_like_scene_simulation"
RANDOM_STATE = 203
OUT_DIR = PATHS.cache_dir(203)
REPORT_DIR = PATHS.version_report_dir(203)
VIS_DIR = PATHS.analysis / "visualizations" / "v203_sionna_like_scene_simulation"
REPORT = REPORT_DIR / "v203_sionna_like_scene_simulation.json"
SYNTH_OUT = OUT_DIR / "v203_sionna_like_scene_simulation_outputs.npz"

N_DELAY = 192
K_POWER = 72
K_TEST = 32


@dataclass(frozen=True)
class InputData:
    train_pos: np.ndarray
    test_pos: np.ndarray
    train_blocked: np.ndarray
    test_blocked: np.ndarray
    train_regimes: np.ndarray
    test_regimes: np.ndarray
    train_basic_features: np.ndarray
    test_basic_features: np.ndarray
    train_path_features_public: np.ndarray
    test_path_features_public: np.ndarray
    train_path_features_material: np.ndarray
    test_path_features_material: np.ndarray
    train_candidates: dict[str, np.ndarray]
    test_candidates: dict[str, np.ndarray]
    material_names: np.ndarray


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def safe_corr(y: np.ndarray, p: np.ndarray, kind: str) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    if int(valid.sum()) < 3 or float(np.std(y[valid])) < 1e-12 or float(np.std(p[valid])) < 1e-12:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(y[valid], p[valid]).statistic)
    if kind == "spearman":
        return float(spearmanr(y[valid], p[valid]).statistic)
    raise ValueError(kind)


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    yv = y[valid]
    pv = p[valid]
    if len(yv) == 0:
        return {"count": 0}
    err = pv - yv
    abs_err = np.abs(err)
    return {
        "count": int(len(yv)),
        "mae_log": float(mean_absolute_error(yv, pv)),
        "rmse_log": float(np.sqrt(np.mean(err * err))),
        "bias_log": float(np.mean(err)),
        "p50_abs_log": float(np.quantile(abs_err, 0.50)),
        "p90_abs_log": float(np.quantile(abs_err, 0.90)),
        "r2": float(r2_score(yv, pv)) if len(yv) >= 2 else float("nan"),
        "pearson": safe_corr(yv, pv, "pearson"),
        "spearman": safe_corr(yv, pv, "spearman"),
    }


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


def metrics_by_regime(y: np.ndarray, pred: np.ndarray, regimes: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {"overall": regression_metrics(y, pred)}
    for regime in ["clear", "light", "medium", "heavy"]:
        mask = regimes.astype(str) == regime
        out[regime] = regression_metrics(y[mask], pred[mask]) if np.any(mask) else {"count": 0}
    return out


def one_hot_regime(regimes: np.ndarray) -> np.ndarray:
    return np.stack(
        [regimes == "clear", regimes == "light", regimes == "medium", regimes == "heavy"],
        axis=1,
    ).astype(np.float32)


def basic_position_features(pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    rel = pos.astype(np.float64) - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    x = (pos[:, 0] - 150.0) / 120.0
    y = (pos[:, 1] + 75.0) / 120.0
    ff = [x, y]
    for freq in (1.0, 2.0, 4.0, 8.0):
        ff.extend([np.sin(np.pi * freq * x), np.cos(np.pi * freq * x)])
        ff.extend([np.sin(np.pi * freq * y), np.cos(np.pi * freq * y)])
    regimes = regime_of(blocked)
    return np.column_stack(
        [
            pos / np.array([240.0, 240.0, 40.0]),
            rel / 240.0,
            dist / 320.0,
            np.log1p(dist),
            np.sin(az),
            np.cos(az),
            blocked / 220.0,
            np.log1p(np.maximum(blocked, 0.0)),
            dist * blocked / 60000.0,
            np.stack(ff, axis=1),
            one_hot_regime(regimes),
        ]
    ).astype(np.float32)


def load_feature_block(path: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    train = z["train_features"].astype(np.float32)
    test = z["test_features"].astype(np.float32)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"Bad feature block {path}")
    if "feature_names" in z.files:
        _ = [f"{prefix}:{str(name)}" for name in z["feature_names"]]
    return train, test


def candidate_dict(z: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
    keys = [
        "valid",
        "path_type",
        "mesh_id",
        "material_id",
        "path_length",
        "delay_bin",
        "score",
        "excess_length",
        "edge_margin",
        "incidence_cos",
        "los_hit_count",
        "los_clear",
        "los_first_material_id",
        "los_first_mesh_id",
    ]
    return {k: z[f"{prefix}_{k}"] for k in keys}


def weighted_stats(values: np.ndarray, weights: np.ndarray, default: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ws = np.maximum(weights.sum(axis=1), TINY)
    mean = (values * weights).sum(axis=1) / ws
    var = ((values - mean[:, None]) ** 2 * weights).sum(axis=1) / ws
    mean = np.where(ws > TINY * 10, mean, default)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32), ws.astype(np.float32)


def build_path_features(c: dict[str, np.ndarray], include_material: bool) -> np.ndarray:
    valid = c["valid"].astype(bool)
    score = np.nan_to_num(np.where(valid, np.maximum(c["score"].astype(np.float32), 0.0), 0.0), nan=0.0)
    score_log = np.log1p(120.0 * score)
    path_length = np.nan_to_num(np.where(valid, c["path_length"].astype(np.float32), 0.0), nan=0.0)
    delay_bin = np.nan_to_num(np.where(valid, c["delay_bin"].astype(np.float32), 0.0), nan=0.0)
    excess = np.nan_to_num(np.where(valid, c["excess_length"].astype(np.float32), 0.0), nan=0.0)
    incidence = np.nan_to_num(
        np.where(valid, np.maximum(c["incidence_cos"].astype(np.float32), 0.0), 0.0),
        nan=0.0,
    )
    edge = np.nan_to_num(np.where(valid, c["edge_margin"].astype(np.float32), 0.0), nan=0.0)
    path_type = c["path_type"].astype(np.int16)

    count_valid = valid.sum(axis=1).astype(np.float32)
    score_sum = score.sum(axis=1).astype(np.float32)
    score_max = score.max(axis=1).astype(np.float32)
    score_top3 = np.sort(score, axis=1)[:, -3:].sum(axis=1).astype(np.float32)
    length_mean, length_std, _ = weighted_stats(path_length, score + 1e-6)
    delay_mean, delay_std, _ = weighted_stats(delay_bin, score + 1e-6)
    incidence_mean, incidence_std, _ = weighted_stats(incidence, score + 1e-6)
    excess_mean, excess_std, _ = weighted_stats(excess, score + 1e-6)
    edge_mean, edge_std, _ = weighted_stats(edge, score + 1e-6)

    type_feats = []
    for t in range(4):
        mask = valid & (path_type == t)
        type_feats.append(mask.sum(axis=1).astype(np.float32))
        type_feats.append(np.where(mask, score, 0.0).sum(axis=1).astype(np.float32))
        type_feats.append(np.where(mask, score_log, 0.0).sum(axis=1).astype(np.float32))

    feats = [
        count_valid,
        score_sum,
        np.log1p(score_sum),
        score_max,
        score_top3,
        length_mean / 260.0,
        length_std / 160.0,
        delay_mean / 192.0,
        delay_std / 96.0,
        incidence_mean,
        incidence_std,
        excess_mean / 160.0,
        excess_std / 120.0,
        edge_mean / 40.0,
        edge_std / 40.0,
        c["los_hit_count"].astype(np.float32) / 120.0,
        c["los_clear"].astype(np.float32),
        *type_feats,
    ]

    if include_material:
        material = c["material_id"].astype(np.int16)
        mesh = c["mesh_id"].astype(np.int32)
        for m in range(5):
            mask = valid & (material == m)
            feats.append(mask.sum(axis=1).astype(np.float32))
            feats.append(np.where(mask, score, 0.0).sum(axis=1).astype(np.float32))
        first_mat = np.clip(c["los_first_material_id"].astype(np.int16), -1, 4)
        for m in range(-1, 5):
            feats.append((first_mat == m).astype(np.float32))
        # Mesh identity is too high-cardinality for direct one-hot.  These
        # hashed buckets approximate object/material labels without exposing a
        # full oracle.
        for mod in (17, 31, 53):
            bucket = np.where(valid, np.maximum(mesh, 0) % mod, -1)
            for b in range(min(mod, 17)):
                feats.append(np.where(bucket == b, score, 0.0).sum(axis=1).astype(np.float32))

    return np.column_stack(feats).astype(np.float32)


def load_inputs() -> InputData:
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)
    test_regimes = regime_of(test_blocked)

    basic_train = basic_position_features(train_pos, train_blocked)
    basic_test = basic_position_features(test_pos, test_blocked)
    for path, prefix in [
        (PATHS.shared_features / "map_features.npz", "map"),
        (PATHS.shared_features / "official_pointcloud_scene_cache.npz", "pointcloud"),
        (PATHS.shared_features / "sionna_etoile_features.npz", "sionna"),
    ]:
        tr, te = load_feature_block(path, prefix)
        basic_train = np.concatenate([basic_train, tr], axis=1)
        basic_test = np.concatenate([basic_test, te], axis=1)

    cand = np.load(PATHS.shared_features / "sionna_path_candidates.npz", allow_pickle=True)
    train_candidates = candidate_dict(cand, "train")
    test_candidates = candidate_dict(cand, "test")
    train_path_public = build_path_features(train_candidates, include_material=False)
    test_path_public = build_path_features(test_candidates, include_material=False)
    train_path_mat = build_path_features(train_candidates, include_material=True)
    test_path_mat = build_path_features(test_candidates, include_material=True)

    return InputData(
        train_pos=train_pos,
        test_pos=test_pos,
        train_blocked=train_blocked,
        test_blocked=test_blocked,
        train_regimes=train_regimes,
        test_regimes=test_regimes,
        train_basic_features=basic_train,
        test_basic_features=basic_test,
        train_path_features_public=train_path_public,
        test_path_features_public=test_path_public,
        train_path_features_material=train_path_mat,
        test_path_features_material=test_path_mat,
        train_candidates=train_candidates,
        test_candidates=test_candidates,
        material_names=cand["material_names"],
    )


def stack_candidates(data: InputData) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in data.train_candidates:
        out[key] = np.concatenate([data.train_candidates[key], data.test_candidates[key]], axis=0)
    return out


def gaussian_delay_spectrum(path_power: np.ndarray, delay_bin: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = len(path_power)
    grid = np.arange(N_DELAY, dtype=np.float32)
    spec = np.zeros((n, N_DELAY), dtype=np.float32)
    for start in range(0, n, 256):
        end = min(start + 256, n)
        pp = np.where(valid[start:end], path_power[start:end], 0.0).astype(np.float32)
        db = np.clip(delay_bin[start:end].astype(np.float32), 0.0, N_DELAY - 1.0)
        sigma = 1.30 + 0.35 * np.clip(db / 80.0, 0.0, 2.0)
        diff = grid[None, None, :] - db[:, :, None]
        kernel = np.exp(-0.5 * (diff / sigma[:, :, None]) ** 2).astype(np.float32)
        kernel /= np.maximum(kernel.sum(axis=2, keepdims=True), TINY)
        spec[start:end] = np.sum(pp[:, :, None] * kernel, axis=1)
    spec += 1e-10
    spec /= np.maximum(spec.sum(axis=1, keepdims=True), TINY)
    return spec.astype(np.float32)


def simulate_clean_targets(data: InputData, scenario: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = stack_candidates(data)
    n = len(c["valid"])
    valid = c["valid"].astype(bool)
    score = np.nan_to_num(np.where(valid, np.maximum(c["score"].astype(np.float64), 0.0), 0.0), nan=0.0)
    length = np.nan_to_num(np.where(valid, np.maximum(c["path_length"].astype(np.float64), 1.0), 1.0), nan=1.0)
    delay = np.nan_to_num(np.where(valid, c["delay_bin"].astype(np.float64), 0.0), nan=0.0)
    excess = np.nan_to_num(np.where(valid, np.maximum(c["excess_length"].astype(np.float64), 0.0), 0.0), nan=0.0)
    incidence = np.nan_to_num(
        np.where(valid, np.maximum(c["incidence_cos"].astype(np.float64), 0.0), 0.0),
        nan=0.0,
    )
    path_type = c["path_type"].astype(np.int16)
    material = c["material_id"].astype(np.int16)
    mesh = c["mesh_id"].astype(np.int32)
    los_hit = c["los_hit_count"].astype(np.float64)

    max_mesh = int(np.max(np.maximum(mesh, 0))) + 1
    max_mat = max(5, int(np.max(np.maximum(material, 0))) + 1)
    material_gain = rng.normal(0.0, 0.0 if scenario == "visible_paths" else 0.38, size=max_mat)
    material_phase = rng.uniform(-np.pi, np.pi, size=max_mat)
    mesh_gain = rng.normal(0.0, 0.0 if scenario in {"visible_paths", "hidden_material"} else 0.24, size=max_mesh)
    mesh_phase = rng.uniform(-np.pi, np.pi, size=max_mesh)
    mesh_gain[0] = mesh_gain[0] if max_mesh else 0.0

    type_loss = np.choose(np.clip(path_type, 0, 3), [0.0, -1.05, -1.55, -1.85]).astype(np.float64)
    log_path_power = (
        1.65
        + 1.18 * np.log1p(90.0 * score)
        - 2.12 * np.log(np.maximum(length, 1.0) / 40.0)
        - 0.0125 * excess
        + 0.68 * np.log1p(2.2 * incidence)
        + type_loss
    )
    log_path_power -= 0.014 * los_hit[:, None]
    if scenario in {"hidden_material", "coherent_multipath", "official_like_noisy_train"}:
        mat_idx = np.clip(material, 0, max_mat - 1)
        log_path_power += material_gain[mat_idx]
    if scenario in {"coherent_multipath", "official_like_noisy_train"}:
        mesh_idx = np.clip(mesh, 0, max_mesh - 1)
        log_path_power += mesh_gain[mesh_idx]
        corridor = np.sin((np.concatenate([data.train_pos, data.test_pos], axis=0)[:, 0] - 82.0) / 22.0)
        corridor += 0.7 * np.cos((np.concatenate([data.train_pos, data.test_pos], axis=0)[:, 1] + 45.0) / 18.0)
        log_path_power += 0.18 * corridor[:, None]
    log_path_power = np.where(valid, log_path_power, -80.0)
    path_power = np.exp(log_path_power).astype(np.float64)

    incoherent_power = np.maximum(path_power.sum(axis=1), 1e-20)
    clean_power = incoherent_power.copy()
    coherent_factor = np.ones(n, dtype=np.float64)
    if scenario in {"coherent_multipath", "official_like_noisy_train"}:
        sub = np.linspace(-0.5, 0.5, 192, endpoint=False, dtype=np.float64)
        phase = 2.0 * np.pi * (delay[:, :, None] * sub[None, None, :])
        mat_idx = np.clip(material, 0, max_mat - 1)
        mesh_idx = np.clip(mesh, 0, max_mesh - 1)
        static_phase = material_phase[mat_idx] + mesh_phase[mesh_idx] + 0.017 * length
        complex_amp = np.sqrt(path_power)[:, :, None] * np.exp(1j * (phase + static_phase[:, :, None]))
        h_freq = np.sum(np.where(valid[:, :, None], complex_amp, 0.0), axis=1)
        coherent_power = np.mean(np.abs(h_freq) ** 2, axis=1)
        coherent_factor = np.clip(coherent_power / np.maximum(incoherent_power, 1e-20), 0.15, 3.50)
        clean_power = incoherent_power * coherent_factor

    # A weak diffuse/scatter floor makes blocked heavy regions less purely
    # specular, closer to an urban ray-tracing channel summary.
    regimes_all = np.concatenate([data.train_regimes, data.test_regimes]).astype(str)
    heavy_bonus = np.select(
        [regimes_all == "clear", regimes_all == "light", regimes_all == "medium", regimes_all == "heavy"],
        [0.02, 0.04, 0.075, 0.11],
        default=0.06,
    )
    diffuse = np.exp(np.log(np.maximum(incoherent_power, 1e-20)) - 2.6) * heavy_bonus
    clean_power = np.maximum(clean_power + diffuse, 1e-20)

    delay_spectrum = gaussian_delay_spectrum(path_power + diffuse[:, None] / max(path_power.shape[1], 1), delay, valid)
    return {
        "clean_power": clean_power.astype(np.float32),
        "clean_log_power": np.log(clean_power).astype(np.float32),
        "path_power": path_power.astype(np.float32),
        "path_log_power": log_path_power.astype(np.float32),
        "incoherent_log_power": np.log(np.maximum(incoherent_power, 1e-20)).astype(np.float32),
        "coherent_factor": coherent_factor.astype(np.float32),
        "delay_spectrum": delay_spectrum,
        "material_gain": material_gain.astype(np.float32),
        "mesh_gain": mesh_gain.astype(np.float32),
    }


def add_train_noise(clean_log: np.ndarray, scenario: str, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed + 1000)
    x = clean_log.astype(np.float64)
    centered = (x - np.mean(x)) / max(float(np.std(x)), 1e-6)
    if scenario == "official_like_noisy_train":
        sigma = 0.080 + 0.040 * np.clip(centered, -1.5, 2.5) ** 2 + 0.035 * np.clip(centered, 0.0, 2.5)
        bias = 0.025 * np.clip(centered, -1.0, 2.0)
    elif scenario == "coherent_multipath":
        sigma = 0.055 + 0.025 * np.abs(centered)
        bias = 0.0
    elif scenario == "hidden_material":
        sigma = np.full_like(x, 0.045)
        bias = 0.0
    else:
        sigma = np.full_like(x, 0.025)
        bias = 0.0
    observed = x + bias + rng.normal(0.0, sigma, size=len(x))
    noise = observed - x
    report = {
        "sigma": summarize(sigma),
        "noise": summarize(noise),
        "noise_abs_vs_clean_spearman": safe_corr(np.abs(noise), x, "spearman"),
        "noise_abs_vs_clean_pearson": safe_corr(np.abs(noise), x, "pearson"),
        "noise_signed_vs_clean_spearman": safe_corr(noise, x, "spearman"),
    }
    return observed.astype(np.float32), report


def effective_knn_predict(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
    y_train: np.ndarray,
    k: int = K_POWER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dxy = np.linalg.norm(test_pos[:, None, :2] - train_pos[None, :, :2], axis=2)
    db = np.abs(test_blocked[:, None].astype(np.float32) - train_blocked[None, :].astype(np.float32))
    mismatch = (test_regimes[:, None] != train_regimes[None, :]).astype(np.float32)
    effective = dxy + 0.105 * db + 6.0 * mismatch
    idx = np.argpartition(effective, k, axis=1)[:, :k]
    rows = np.arange(len(test_pos))[:, None]
    idx = idx[rows, np.argsort(effective[rows, idx], axis=1)]
    selected = effective[rows, idx]
    same = (test_regimes[:, None] == train_regimes[idx]).astype(np.float32)
    weights = np.exp(-selected / 18.0).astype(np.float32) * (0.48 + 0.52 * same)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    pred = np.sum(weights * y_train[idx], axis=1).astype(np.float32)
    confidence = (np.sum(weights * same, axis=1) * np.clip(1.0 - selected[:, 0] / 42.0, 0.0, 1.0)).astype(np.float32)
    return pred, idx.astype(np.int32), confidence


def build_model(kind: str, seed: int) -> object:
    if kind == "extra":
        return make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True),
            ExtraTreesRegressor(
                n_estimators=260,
                min_samples_leaf=4,
                max_features=0.82,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if kind == "hgb":
        return make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True),
            HistGradientBoostingRegressor(
                max_iter=260,
                learning_rate=0.045,
                max_leaf_nodes=31,
                min_samples_leaf=14,
                l2_regularization=0.035,
                random_state=seed,
            ),
        )
    if kind == "rf":
        return make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True),
            RandomForestRegressor(
                n_estimators=180,
                min_samples_leaf=5,
                max_features=0.65,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(kind)


def model_predictors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for offset, kind in enumerate(["extra", "hgb", "rf"]):
        model = build_model(kind, seed + offset)
        model.fit(x_train, y_train)
        preds[kind] = np.asarray(model.predict(x_test), dtype=np.float32)
    stack = np.stack([preds["extra"], preds["hgb"], preds["rf"]], axis=0)
    preds["mean"] = np.mean(stack, axis=0).astype(np.float32)
    preds["spread"] = np.std(stack, axis=0).astype(np.float32)
    return preds


def test_consensus(base: np.ndarray, pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray, k: int = K_TEST) -> tuple[np.ndarray, np.ndarray]:
    feats = np.column_stack([pos[:, :2] / 80.0, blocked[:, None] / 80.0, one_hot_regime(regimes)]).astype(np.float32)
    nbr = NearestNeighbors(n_neighbors=min(k + 1, len(pos)), metric="euclidean")
    nbr.fit(feats)
    dist, idx = nbr.kneighbors(feats)
    idx = idx[:, 1:]
    dist = dist[:, 1:]
    same = (regimes[:, None] == regimes[idx]).astype(np.float32)
    w = np.exp(-dist / 0.45).astype(np.float32) * (0.55 + 0.45 * same)
    w /= np.maximum(w.sum(axis=1, keepdims=True), TINY)
    pred = np.sum(w * base[idx], axis=1).astype(np.float32)
    agreement = np.sum(w * same, axis=1).astype(np.float32)
    return pred, agreement


def v200_style_gate(
    base: np.ndarray,
    geom_pred: np.ndarray,
    path_pred: np.ndarray,
    model_spread: np.ndarray,
    consensus: np.ndarray,
    train_conf: np.ndarray,
    test_agreement: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    d_geom = geom_pred - base
    d_path = path_pred - base
    d_cons = consensus - base
    agree = np.exp(-np.abs(d_geom - d_path) / 0.55)
    spread_quality = np.clip(1.0 - model_spread / 0.50, 0.0, 1.0)
    confidence = np.clip(0.32 * train_conf + 0.22 * test_agreement + 0.34 * agree + 0.12 * spread_quality, 0.0, 1.0)
    same_up = (d_geom > 0.08) & (d_path > 0.08)
    same_down = (d_geom < -0.08) & (d_path < -0.08)
    consensus_not_opposite_up = d_cons > -0.18
    consensus_not_opposite_down = d_cons < 0.18
    active_up = same_up & consensus_not_opposite_up & (confidence >= 0.62)
    active_down = same_down & consensus_not_opposite_down & (confidence >= 0.62)
    raw = np.median(np.stack([d_geom, d_path, d_cons], axis=1), axis=1)
    final = np.zeros(len(base), dtype=np.float32)
    caps_up = np.asarray([0.10 if r == "clear" else 0.07 if r == "light" else 0.14 if r == "medium" else 0.12 for r in regimes], dtype=np.float32)
    caps_down = np.asarray([-0.07 if r == "clear" else -0.05 if r == "light" else -0.13 if r == "medium" else -0.16 for r in regimes], dtype=np.float32)
    gain = np.clip(0.40 + 0.45 * confidence, 0.35, 0.88).astype(np.float32)
    final[active_up] = np.minimum(raw[active_up] * gain[active_up], caps_up[active_up])
    final[active_down] = np.maximum(raw[active_down] * gain[active_down], caps_down[active_down])
    label = np.full(len(base), "no_action", dtype="U16")
    label[active_up] = "power_up"
    label[active_down] = "power_down"
    corrected = (base + final).astype(np.float32)
    return {
        "corrected": corrected,
        "final_delta": final.astype(np.float32),
        "active": np.abs(final) > 1e-7,
        "label": label,
        "confidence": confidence.astype(np.float32),
        "delta_geom": d_geom.astype(np.float32),
        "delta_path": d_path.astype(np.float32),
        "delta_consensus": d_cons.astype(np.float32),
    }


def run_one_scenario(data: InputData, scenario: str, seed: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    synth = simulate_clean_targets(data, scenario, seed)
    n_train = len(data.train_pos)
    clean_train = synth["clean_log_power"][:n_train]
    clean_test = synth["clean_log_power"][n_train:]
    observed_train, noise_report = add_train_noise(clean_train, scenario, seed)

    knn_pred, knn_idx, knn_conf = effective_knn_predict(
        data.train_pos,
        data.test_pos,
        data.train_blocked,
        data.test_blocked,
        data.train_regimes,
        data.test_regimes,
        observed_train,
        k=K_POWER,
    )
    clean_knn_pred, _, _ = effective_knn_predict(
        data.train_pos,
        data.test_pos,
        data.train_blocked,
        data.test_blocked,
        data.train_regimes,
        data.test_regimes,
        clean_train,
        k=K_POWER,
    )

    x_basic_train = data.train_basic_features
    x_basic_test = data.test_basic_features
    x_path_public_train = np.concatenate([data.train_basic_features, data.train_path_features_public], axis=1)
    x_path_public_test = np.concatenate([data.test_basic_features, data.test_path_features_public], axis=1)
    x_path_mat_train = np.concatenate([data.train_basic_features, data.train_path_features_material], axis=1)
    x_path_mat_test = np.concatenate([data.test_basic_features, data.test_path_features_material], axis=1)

    basic_preds = model_predictors(x_basic_train, observed_train, x_basic_test, seed + 10)
    path_public_preds = model_predictors(x_path_public_train, observed_train, x_path_public_test, seed + 20)
    path_mat_preds = model_predictors(x_path_mat_train, observed_train, x_path_mat_test, seed + 30)
    clean_path_public = model_predictors(x_path_public_train, clean_train, x_path_public_test, seed + 40)
    clean_path_mat = model_predictors(x_path_mat_train, clean_train, x_path_mat_test, seed + 50)

    consensus, test_agreement = test_consensus(knn_pred, data.test_pos, data.test_blocked, data.test_regimes)
    gate = v200_style_gate(
        base=knn_pred,
        geom_pred=basic_preds["mean"],
        path_pred=path_public_preds["mean"],
        model_spread=path_public_preds["spread"],
        consensus=consensus,
        train_conf=knn_conf,
        test_agreement=test_agreement,
        regimes=data.test_regimes,
    )

    true_error = clean_test - knn_pred
    active = gate["active"].astype(bool)
    sign_correct = np.zeros_like(active, dtype=bool)
    sign_correct[active] = np.sign(gate["final_delta"][active]) == np.sign(true_error[active])

    latent_incoherent = synth["incoherent_log_power"][n_train:]
    latent_coherent = np.log(np.maximum(synth["coherent_factor"][n_train:], 1e-12)).astype(np.float32)
    latent_oracle = latent_incoherent + latent_coherent
    predictors = {
        "knn_observed_train": knn_pred,
        "knn_clean_train_oracle": clean_knn_pred,
        "basic_geometry_model": basic_preds["mean"],
        "path_public_model": path_public_preds["mean"],
        "path_material_model": path_mat_preds["mean"],
        "path_public_clean_train_oracle": clean_path_public["mean"],
        "path_material_clean_train_oracle": clean_path_mat["mean"],
        "latent_path_power_oracle": latent_oracle,
        "v200_style_gate": gate["corrected"],
    }

    metrics = {name: metrics_by_regime(clean_test, pred, data.test_regimes) for name, pred in predictors.items()}
    rmse_rank = sorted((metrics[name]["overall"]["rmse_log"], name) for name in metrics)
    report: dict[str, object] = {
        "scenario": scenario,
        "seed": int(seed),
        "noise_report": noise_report,
        "clean_power_summary": {
            "train": summarize(clean_train),
            "test": summarize(clean_test),
            "train_test_shift": {
                "mean_delta_test_minus_train": float(np.mean(clean_test) - np.mean(clean_train)),
                "std_ratio_test_over_train": float(np.std(clean_test) / max(float(np.std(clean_train)), 1e-9)),
            },
        },
        "predictor_metrics": metrics,
        "rmse_rank": [{"name": name, "rmse_log": float(rmse)} for rmse, name in rmse_rank],
        "gate_summary": {
            "active_count": int(active.sum()),
            "active_fraction": float(np.mean(active)),
            "up_count": int(np.sum(gate["label"] == "power_up")),
            "down_count": int(np.sum(gate["label"] == "power_down")),
            "sign_correct_active_fraction": float(np.mean(sign_correct[active])) if np.any(active) else float("nan"),
            "base_rmse": metrics["knn_observed_train"]["overall"]["rmse_log"],
            "gate_rmse": metrics["v200_style_gate"]["overall"]["rmse_log"],
            "gate_delta_rmse": float(
                metrics["v200_style_gate"]["overall"]["rmse_log"] - metrics["knn_observed_train"]["overall"]["rmse_log"]
            ),
            "final_delta": summarize(gate["final_delta"]),
            "confidence": summarize(gate["confidence"]),
        },
        "latent_diagnostics": {
            "coherent_factor": summarize(synth["coherent_factor"][n_train:]),
            "coherent_factor_vs_base_abs_error_spearman": safe_corr(
                np.abs(np.log(np.maximum(synth["coherent_factor"][n_train:], 1e-12))),
                np.abs(clean_test - knn_pred),
                "spearman",
            ),
            "hidden_material_gain_std": float(np.std(synth["material_gain"])),
            "hidden_mesh_gain_std": float(np.std(synth["mesh_gain"])),
        },
    }
    arrays = {
        f"{scenario}_clean_train": clean_train.astype(np.float32),
        f"{scenario}_clean_test": clean_test.astype(np.float32),
        f"{scenario}_observed_train": observed_train.astype(np.float32),
        f"{scenario}_knn_pred": knn_pred.astype(np.float32),
        f"{scenario}_basic_pred": basic_preds["mean"].astype(np.float32),
        f"{scenario}_path_public_pred": path_public_preds["mean"].astype(np.float32),
        f"{scenario}_path_material_pred": path_mat_preds["mean"].astype(np.float32),
        f"{scenario}_gate_pred": gate["corrected"].astype(np.float32),
        f"{scenario}_gate_delta": gate["final_delta"].astype(np.float32),
        f"{scenario}_gate_active": active.astype(bool),
        f"{scenario}_gate_sign_correct": sign_correct.astype(bool),
        f"{scenario}_coherent_factor_test": synth["coherent_factor"][n_train:].astype(np.float32),
    }
    return report, arrays


def make_plots(report: dict[str, object], arrays: dict[str, np.ndarray], data: InputData) -> list[str]:
    paths: list[str] = []
    scenarios = report["scenarios"]
    names = [
        "knn_observed_train",
        "basic_geometry_model",
        "path_public_model",
        "path_material_model",
        "v200_style_gate",
        "path_material_clean_train_oracle",
        "latent_path_power_oracle",
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.11
    x = np.arange(len(scenarios))
    for j, name in enumerate(names):
        values = [scenarios[s]["predictor_metrics"][name]["overall"]["rmse_log"] for s in scenarios]
        ax.bar(x + (j - (len(names) - 1) / 2) * width, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(list(scenarios.keys()), rotation=15, ha="right")
    ax.set_ylabel("Clean test log-power RMSE")
    ax.set_title("v203 official-like synthetic benchmark: predictor RMSE")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = VIS_DIR / "v203_synthetic_power_rmse_by_scenario.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(PATHS.relative(path))

    scenario = "official_like_noisy_train"
    y = arrays[f"{scenario}_clean_test"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, key, color in [
        ("kNN", f"{scenario}_knn_pred", "tab:gray"),
        ("path public", f"{scenario}_path_public_pred", "tab:blue"),
        ("path material", f"{scenario}_path_material_pred", "tab:orange"),
        ("v200 gate", f"{scenario}_gate_pred", "tab:green"),
    ]:
        ax.scatter(y, arrays[key], s=12, alpha=0.45, label=name, color=color)
    lo = float(np.quantile(y, 0.01))
    hi = float(np.quantile(y, 0.99))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Synthetic clean test log power")
    ax.set_ylabel("Prediction")
    ax.set_title("Official-like scenario: clean truth vs predictors")
    ax.legend()
    fig.tight_layout()
    path = VIS_DIR / "v203_official_like_prediction_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(PATHS.relative(path))

    fig, ax = plt.subplots(figsize=(7, 5))
    clean_train = arrays[f"{scenario}_clean_train"]
    observed_train = arrays[f"{scenario}_observed_train"]
    noise = observed_train - clean_train
    ax.scatter(clean_train, np.abs(noise), s=10, alpha=0.45, color="tab:red")
    ax.set_xlabel("Synthetic clean train log power")
    ax.set_ylabel("|observed - clean| log noise")
    ax.set_title("Official-like scenario: train noise vs power")
    fig.tight_layout()
    path = VIS_DIR / "v203_train_noise_vs_power.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(PATHS.relative(path))

    fig, ax = plt.subplots(figsize=(7, 5))
    active = arrays[f"{scenario}_gate_active"].astype(bool)
    sign_correct = arrays[f"{scenario}_gate_sign_correct"].astype(bool)
    colors = np.where(~active, "lightgray", np.where(sign_correct, "tab:green", "tab:red"))
    ax.scatter(data.test_pos[:, 0], data.test_pos[:, 1], s=18, c=colors, alpha=0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Official-like scenario: v200-style gate direction")
    fig.tight_layout()
    path = VIS_DIR / "v203_gate_direction_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(PATHS.relative(path))
    return paths


def main() -> None:
    ensure_dirs()
    data = load_inputs()
    scenarios = [
        "visible_paths",
        "hidden_material",
        "coherent_multipath",
        "official_like_noisy_train",
    ]
    reports: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    for i, scenario in enumerate(scenarios):
        print(f"running v203 scenario {scenario}", flush=True)
        rep, arr = run_one_scenario(data, scenario, RANDOM_STATE + 97 * i)
        reports[scenario] = rep
        arrays.update(arr)

    report: dict[str, object] = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "no_submission_generated": True,
        "question": "Can an official-like controllable Sionna proxy explain why total-power prediction saturates?",
        "hypothesis": (
            "Visible geometry should predict power well in a simple path-loss scene, but hidden material gains, "
            "coherent multipath and noisy train labels should create a residual ceiling.  If so, continuing to "
            "expand v200-style labels without new hidden-state information is expected to saturate."
        ),
        "inputs": {
            "train_pos": PATHS.relative(PATHS.round1 / "Round1_Train_Pos.npy"),
            "test_pos": PATHS.relative(PATHS.round1 / "Round1_Test_Pos.npy"),
            "height_geometry": PATHS.relative(PATHS.shared_features / "height_geometry_cache.npz"),
            "map_features": PATHS.relative(PATHS.shared_features / "map_features.npz"),
            "official_pointcloud_features": PATHS.relative(PATHS.shared_features / "official_pointcloud_scene_cache.npz"),
            "sionna_etoile_features": PATHS.relative(PATHS.shared_features / "sionna_etoile_features.npz"),
            "sionna_path_candidates": PATHS.relative(PATHS.shared_features / "sionna_path_candidates.npz"),
        },
        "parameters": {
            "random_state": RANDOM_STATE,
            "n_train": int(len(data.train_pos)),
            "n_test": int(len(data.test_pos)),
            "n_path_candidates": int(data.train_candidates["valid"].shape[1]),
            "n_delay": N_DELAY,
            "knn_k_power": K_POWER,
            "test_consensus_k": K_TEST,
            "full_h_generated": False,
            "scenarios": scenarios,
        },
        "feature_dimensions": {
            "basic": int(data.train_basic_features.shape[1]),
            "path_public": int(data.train_path_features_public.shape[1]),
            "path_material": int(data.train_path_features_material.shape[1]),
        },
        "scenarios": reports,
    }
    plot_paths = make_plots(report, arrays, data)
    report["plots"] = plot_paths
    report["outputs"] = {
        "npz": PATHS.relative(SYNTH_OUT),
        "json": PATHS.relative(REPORT),
        "visualization_dir": PATHS.relative(VIS_DIR),
    }

    np.savez_compressed(
        SYNTH_OUT,
        train_pos=data.train_pos,
        test_pos=data.test_pos,
        train_blocked=data.train_blocked,
        test_blocked=data.test_blocked,
        train_regimes=data.train_regimes.astype("U8"),
        test_regimes=data.test_regimes.astype("U8"),
        **arrays,
    )
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": PATHS.relative(REPORT), "plots": plot_paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
