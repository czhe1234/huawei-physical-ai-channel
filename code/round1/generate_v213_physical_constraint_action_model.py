#!/usr/bin/env python3
"""Generate v213: physical-constraint action model on the v209 framework.

This candidate keeps the v209/v200 style action framework as the generator and
uses Sionna only as a physical feature/constraint source.  The important change
from v210-v212 is that the Sionna PDP is not trusted with a fixed hand-written
weight.  A supervised trust model is trained on official train H:

* truth target: released train normalized D / U-D / H-D delay marginals;
* physical inputs: high-budget Sionna train marginals, low-budget multi-carrier
  Sionna stability, geometry/regime and disagreement with a train-OOF anchor;
* test action: predicted trust controls how much v209 may move toward the
  Sionna-constrained PDP target.

Raw Sionna phase, raw Sionna total power and raw complex H are never copied.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import generate_v178_test_track_complex_consensus as track
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_v87_easy_field_projection import MH, MP, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v213_physical_constraint_action_model"

OUT_DIR = PATHS.cache_dir(213)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(213)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v213_Round1_Test_Channel.npy"
TRAIN_SIONNA_MARGINALS = OUT_DIR / "v213_train_sionna_constraint_marginals.npz"
TEST_MARGINALS = OUT_DIR / "v213_test_physical_constraint_marginals.npz"
ACTION_OUT = OUT_DIR / "v213_physical_constraint_actions.npz"
REPORT = REPORT_DIR / "v213_physical_constraint_action_model_generate.json"
ZIP_OUT = PATHS.round1_versions / "v213_physical_constraint_action_model_candidate_pending.zip"

BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
V200_BASE = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"
V212_BASE = PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy"
V196_TARGET = PATHS.cache_dir(196) / "v196_train_guided_delay_target.npy"
V196_TRAIN = PATHS.cache_dir(196) / "v196_train_delay_marginals.npz"
V212_TEST_MARGINALS = PATHS.cache_dir(212) / "v212_multidomain_sionna_pdp_marginals.npz"

HIGH_TRAIN = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
HIGH_TEST = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
LOW_TRAIN = [
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f2p6" / "train_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p0" / "train_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p7" / "train_clean_H.npy",
]
LOW_TEST = [
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f2p6" / "test_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p0" / "test_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p7" / "test_clean_H.npy",
]

FLOOR = 1e-12
BATCH = 4
TEST_BATCH = 6
TRAIN_K = 72
IPF_ITERS = 7
BIAS_EXPONENT_D = 0.55
BIAS_EXPONENT_DU = 0.38
BIAS_EXPONENT_DH = 0.34

D_ALPHA_CAP = {"clear": 0.0, "light": 0.055, "medium": 0.56, "heavy": 0.76}
DU_ALPHA_CAP = {"clear": 0.0, "light": 0.020, "medium": 0.19, "heavy": 0.31}
DH_ALPHA_CAP = {"clear": 0.0, "light": 0.017, "medium": 0.16, "heavy": 0.27}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)), axis=1).astype(np.float32)


def delay_support_from_h(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = v196.delay_power_from_h(h)
    total = power.sum(axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
    support = power / np.maximum(total[:, None, None, None, None, None], TINY)
    return support.astype(np.float32), total


def support_marginals(support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du = support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh = support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    return normalize(d, axis=1), normalize(du, axis=(1, 2)), normalize(dh, axis=(1, 2))


def blend_valid(values: list[np.ndarray], valid: list[np.ndarray], axes: tuple[int, ...] | int) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(values, axis=0).astype(np.float32)
    valid_stack = np.stack(valid, axis=0).astype(np.float32)
    shape = (len(values), len(values[0])) + (1,) * (stack.ndim - 2)
    weights = valid_stack.reshape(shape)
    count = valid_stack.sum(axis=0).astype(np.float32)
    blended = stack.sum(axis=0) * 0.0
    blended = np.sum(stack * weights, axis=0) / np.maximum(
        count.reshape((len(count),) + (1,) * (stack.ndim - 2)),
        1.0,
    )
    return normalize(blended, axis=axes), count


def train_oof_anchor(truth: dict[str, np.ndarray], regimes: np.ndarray, blocked: np.ndarray) -> dict[str, np.ndarray]:
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    dxy = np.linalg.norm(train_pos[:, None, :2] - train_pos[None, :, :2], axis=2).astype(np.float32)
    db = np.abs(blocked[:, None].astype(np.float32) - blocked[None, :].astype(np.float32))
    mismatch = (regimes[:, None] != regimes[None, :]).astype(np.float32)
    effective = dxy + 0.105 * db + 6.0 * mismatch
    np.fill_diagonal(effective, np.inf)
    idx = np.argpartition(effective, TRAIN_K, axis=1)[:, :TRAIN_K]
    rows = np.arange(len(train_pos))[:, None]
    idx = idx[rows, np.argsort(effective[rows, idx], axis=1)]
    selected = effective[rows, idx]
    same = (regimes[:, None] == regimes[idx]).astype(np.float32)
    valid = truth["valid"][idx].astype(np.float32)
    weights = np.exp(-selected / 18.0).astype(np.float32) * (0.48 + 0.52 * same) * valid
    empty = weights.sum(axis=1) <= TINY
    if np.any(empty):
        weights[empty] = np.exp(-selected[empty] / 24.0).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return {
        "d": normalize(np.einsum("ik,ikd->id", weights, truth["d"][idx], optimize=True), axis=1),
        "du": normalize(np.einsum("ik,ikud->iud", weights, truth["du"][idx], optimize=True), axis=(1, 2)),
        "dh": normalize(np.einsum("ik,ikhd->ihd", weights, truth["dh"][idx], optimize=True), axis=(1, 2)),
    }


def build_train_sionna_marginals() -> dict[str, np.ndarray]:
    if TRAIN_SIONNA_MARGINALS.is_file():
        data = np.load(TRAIN_SIONNA_MARGINALS)
        return {key: data[key] for key in data.files}

    truth = np.load(V196_TRAIN)
    high = np.load(HIGH_TRAIN, mmap_mode="r")
    lows = [np.load(path, mmap_mode="r") for path in LOW_TRAIN]
    n = len(high)
    arrays = {
        "truth_d": truth["d"].astype(np.float32),
        "truth_du": truth["du"].astype(np.float32),
        "truth_dh": truth["dh"].astype(np.float32),
        "truth_valid": truth["valid"].astype(np.float32),
        "high_d": np.empty((n, S_SC), dtype=np.float32),
        "high_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "high_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "multi_d": np.empty((n, S_SC), dtype=np.float32),
        "multi_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "multi_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "high_total_power": np.empty(n, dtype=np.float32),
        "multi_valid_count": np.empty(n, dtype=np.float32),
        "multi_pair_l1": np.empty(n, dtype=np.float32),
    }
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        high_support, high_total = delay_support_from_h(np.asarray(high[start:end], dtype=np.complex64))
        high_zero = high_total <= 1e-20
        if np.any(high_zero):
            # A zero physical reference has no trustworthy shape.  Use a flat
            # fallback only so downstream feature construction remains finite.
            high_support[high_zero] = 1.0 / np.prod(high_support.shape[1:])
        arrays["high_d"][start:end], arrays["high_du"][start:end], arrays["high_dh"][start:end] = support_marginals(high_support)
        arrays["high_total_power"][start:end] = high_total

        low_d: list[np.ndarray] = []
        low_du: list[np.ndarray] = []
        low_dh: list[np.ndarray] = []
        low_valid: list[np.ndarray] = []
        for ref in lows:
            support, total = delay_support_from_h(np.asarray(ref[start:end], dtype=np.complex64))
            valid = total > 1e-20
            if np.any(~valid):
                support[~valid] = high_support[~valid]
            d, du, dh = support_marginals(support)
            low_d.append(d)
            low_du.append(du)
            low_dh.append(dh)
            low_valid.append(valid.astype(np.float32))
        arrays["multi_d"][start:end], valid_count = blend_valid(low_d, low_valid, axes=1)
        arrays["multi_du"][start:end], _ = blend_valid(low_du, low_valid, axes=(1, 2))
        arrays["multi_dh"][start:end], _ = blend_valid(low_dh, low_valid, axes=(1, 2))
        arrays["multi_valid_count"][start:end] = valid_count
        arrays["multi_pair_l1"][start:end] = (
            l1_rows(low_d[0], low_d[1]) + l1_rows(low_d[0], low_d[2]) + l1_rows(low_d[1], low_d[2])
        ) / 3.0
        print(f"v213 train Sionna marginals {end}/{n}", flush=True)

    np.savez_compressed(TRAIN_SIONNA_MARGINALS, **arrays)
    data = np.load(TRAIN_SIONNA_MARGINALS)
    return {key: data[key] for key in data.files}


def build_test_marginals() -> dict[str, np.ndarray]:
    if TEST_MARGINALS.is_file():
        data = np.load(TEST_MARGINALS)
        return {key: data[key] for key in data.files}

    v212_m = np.load(V212_TEST_MARGINALS)
    base = np.load(BASE, mmap_mode="r")
    n = len(base)
    arrays = {
        "current_d": np.empty((n, S_SC), dtype=np.float32),
        "current_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "current_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "high_d": v212_m["v207_d"].astype(np.float32),
        "high_du": v212_m["v207_du"].astype(np.float32),
        "high_dh": v212_m["v207_dh"].astype(np.float32),
        "multi_d": v212_m["multi_d"].astype(np.float32),
        "multi_du": v212_m["multi_du"].astype(np.float32),
        "multi_dh": v212_m["multi_dh"].astype(np.float32),
        "anchor_d": v212_m["v196_d"].astype(np.float32),
        "anchor_du": v212_m["v196_du"].astype(np.float32),
        "anchor_dh": v212_m["v196_dh"].astype(np.float32),
        "high_total_power": v212_m["v207_total_power"].astype(np.float32),
        "multi_valid_count": v212_m["multi_valid_count"].astype(np.float32),
        "multi_pair_l1": v212_m["multi_pair_l1"].astype(np.float32),
    }
    for start in range(0, n, TEST_BATCH):
        end = min(start + TEST_BATCH, n)
        support, _ = delay_support_from_h(np.asarray(base[start:end], dtype=np.complex64))
        arrays["current_d"][start:end], arrays["current_du"][start:end], arrays["current_dh"][start:end] = support_marginals(support)
        print(f"v213 current test marginals {end}/{n}", flush=True)
    np.savez_compressed(TEST_MARGINALS, **arrays)
    data = np.load(TEST_MARGINALS)
    return {key: data[key] for key in data.files}


def ratio_by_regime(truth: np.ndarray, ref: np.ndarray, regimes: np.ndarray, exponent: float) -> dict[str, np.ndarray]:
    ratios: dict[str, np.ndarray] = {}
    for regime in ["clear", "light", "medium", "heavy"]:
        mask = regimes.astype(str) == regime
        if not np.any(mask):
            ratios[regime] = np.ones(truth.shape[1:], dtype=np.float32)
            continue
        mean_truth = np.maximum(truth[mask].mean(axis=0), FLOOR)
        mean_ref = np.maximum(ref[mask].mean(axis=0), FLOOR)
        ratio = np.clip(mean_truth / mean_ref, 0.20, 5.0).astype(np.float32)
        ratios[regime] = np.power(ratio, exponent).astype(np.float32)
    return ratios


def apply_ratio_by_regime(x: np.ndarray, ratios: dict[str, np.ndarray], regimes: np.ndarray, axes: tuple[int, ...] | int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for regime in ["clear", "light", "medium", "heavy"]:
        mask = regimes.astype(str) == regime
        if np.any(mask):
            out[mask] = x[mask] * ratios[regime]
    return normalize(out, axis=axes)


def d_stats(d: np.ndarray) -> np.ndarray:
    idx = np.arange(d.shape[1], dtype=np.float32)
    peak = np.argmax(d, axis=1).astype(np.float32) / float(d.shape[1] - 1)
    mean = np.sum(d * idx[None, :], axis=1) / float(d.shape[1] - 1)
    var = np.sum(d * (idx[None, :] - mean[:, None] * float(d.shape[1] - 1)) ** 2, axis=1)
    spread = np.sqrt(np.maximum(var, 0.0)) / float(d.shape[1] - 1)
    entropy = -np.sum(d * np.log(np.maximum(d, FLOOR)), axis=1) / np.log(float(d.shape[1]))
    top1 = np.max(d, axis=1)
    early = d[:, :24].sum(axis=1)
    late = d[:, 72:].sum(axis=1)
    return np.stack([peak, mean, spread, entropy, top1, early, late], axis=1).astype(np.float32)


def make_features(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    current: dict[str, np.ndarray],
    high: dict[str, np.ndarray],
    multi: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    high_total: np.ndarray,
    valid_count: np.ndarray,
    pair_l1: np.ndarray,
) -> np.ndarray:
    regime_names = ["clear", "light", "medium", "heavy"]
    one_hot = np.stack([(regimes.astype(str) == name).astype(np.float32) for name in regime_names], axis=1)
    base = [
        pos.astype(np.float32),
        blocked[:, None].astype(np.float32) / 160.0,
        one_hot,
        np.log1p(np.maximum(high_total, 0.0))[:, None].astype(np.float32),
        (valid_count[:, None].astype(np.float32) / 3.0),
        pair_l1[:, None].astype(np.float32),
    ]
    distances = [
        l1_rows(high["d"], multi["d"]),
        l1_rows(high["du"], multi["du"]),
        l1_rows(high["dh"], multi["dh"]),
        l1_rows(high["d"], anchor["d"]),
        l1_rows(high["du"], anchor["du"]),
        l1_rows(high["dh"], anchor["dh"]),
        l1_rows(multi["d"], anchor["d"]),
        l1_rows(multi["du"], anchor["du"]),
        l1_rows(multi["dh"], anchor["dh"]),
        l1_rows(high["d"], current["d"]),
        l1_rows(multi["d"], current["d"]),
        l1_rows(anchor["d"], current["d"]),
    ]
    stats = [d_stats(current["d"]), d_stats(high["d"]), d_stats(multi["d"]), d_stats(anchor["d"])]
    return np.concatenate(base + [np.stack(distances, axis=1).astype(np.float32)] + stats, axis=1)


def train_trust_models(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    valid: np.ndarray,
) -> tuple[dict[str, ExtraTreesRegressor], dict[str, object]]:
    models: dict[str, ExtraTreesRegressor] = {}
    diagnostics: dict[str, object] = {}
    mask = valid.astype(bool)
    x = features[mask]
    for key, y_all in targets.items():
        y = y_all[mask].astype(np.float32)
        oof = np.zeros_like(y, dtype=np.float32)
        folds = KFold(n_splits=5, shuffle=True, random_state=213)
        for train_idx, val_idx in folds.split(x):
            model = ExtraTreesRegressor(
                n_estimators=120,
                min_samples_leaf=8,
                max_features=0.78,
                random_state=213,
                n_jobs=-1,
            )
            model.fit(x[train_idx], y[train_idx])
            oof[val_idx] = model.predict(x[val_idx]).astype(np.float32)
        final = ExtraTreesRegressor(
            n_estimators=320,
            min_samples_leaf=6,
            max_features=0.82,
            random_state=213,
            n_jobs=-1,
        )
        final.fit(x, y)
        models[key] = final
        corr = float(np.corrcoef(y, oof)[0, 1]) if np.std(oof) > 1e-8 and np.std(y) > 1e-8 else 0.0
        diagnostics[key] = {
            "target": summarize(y),
            "oof_pred": summarize(oof),
            "oof_r2": float(r2_score(y, oof)),
            "oof_corr": corr,
            "oof_mae": float(np.mean(np.abs(y - oof))),
        }
    diagnostics["valid_count"] = int(np.sum(mask))
    return models, diagnostics


def build_physical_action_model(regimes_train: np.ndarray, regimes_test: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_m = build_train_sionna_marginals()
    test_m = build_test_marginals()
    truth = {
        "d": train_m["truth_d"].astype(np.float32),
        "du": train_m["truth_du"].astype(np.float32),
        "dh": train_m["truth_dh"].astype(np.float32),
        "valid": train_m["truth_valid"].astype(np.float32),
    }
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    anchor_train = train_oof_anchor(truth, regimes_train, geom["train_blocked"].astype(np.float32))

    high_ratios = {
        "d": ratio_by_regime(truth["d"], train_m["high_d"], regimes_train, BIAS_EXPONENT_D),
        "du": ratio_by_regime(truth["du"], train_m["high_du"], regimes_train, BIAS_EXPONENT_DU),
        "dh": ratio_by_regime(truth["dh"], train_m["high_dh"], regimes_train, BIAS_EXPONENT_DH),
    }
    multi_ratios = {
        "d": ratio_by_regime(truth["d"], train_m["multi_d"], regimes_train, BIAS_EXPONENT_D),
        "du": ratio_by_regime(truth["du"], train_m["multi_du"], regimes_train, BIAS_EXPONENT_DU),
        "dh": ratio_by_regime(truth["dh"], train_m["multi_dh"], regimes_train, BIAS_EXPONENT_DH),
    }
    high_train = {
        "d": apply_ratio_by_regime(train_m["high_d"], high_ratios["d"], regimes_train, axes=1),
        "du": apply_ratio_by_regime(train_m["high_du"], high_ratios["du"], regimes_train, axes=(1, 2)),
        "dh": apply_ratio_by_regime(train_m["high_dh"], high_ratios["dh"], regimes_train, axes=(1, 2)),
    }
    multi_train = {
        "d": apply_ratio_by_regime(train_m["multi_d"], multi_ratios["d"], regimes_train, axes=1),
        "du": apply_ratio_by_regime(train_m["multi_du"], multi_ratios["du"], regimes_train, axes=(1, 2)),
        "dh": apply_ratio_by_regime(train_m["multi_dh"], multi_ratios["dh"], regimes_train, axes=(1, 2)),
    }
    high_test = {
        "d": apply_ratio_by_regime(test_m["high_d"], high_ratios["d"], regimes_test, axes=1),
        "du": apply_ratio_by_regime(test_m["high_du"], high_ratios["du"], regimes_test, axes=(1, 2)),
        "dh": apply_ratio_by_regime(test_m["high_dh"], high_ratios["dh"], regimes_test, axes=(1, 2)),
    }
    multi_test = {
        "d": apply_ratio_by_regime(test_m["multi_d"], multi_ratios["d"], regimes_test, axes=1),
        "du": apply_ratio_by_regime(test_m["multi_du"], multi_ratios["du"], regimes_test, axes=(1, 2)),
        "dh": apply_ratio_by_regime(test_m["multi_dh"], multi_ratios["dh"], regimes_test, axes=(1, 2)),
    }
    current_train = anchor_train
    current_test = {
        "d": test_m["current_d"].astype(np.float32),
        "du": test_m["current_du"].astype(np.float32),
        "dh": test_m["current_dh"].astype(np.float32),
    }
    anchor_test = {
        "d": test_m["anchor_d"].astype(np.float32),
        "du": test_m["anchor_du"].astype(np.float32),
        "dh": test_m["anchor_dh"].astype(np.float32),
    }

    sionna_train = {
        key: normalize(0.82 * high_train[key] + 0.18 * multi_train[key], axis=(1, 2) if key != "d" else 1)
        for key in ["d", "du", "dh"]
    }
    sionna_test = {
        key: normalize(0.82 * high_test[key] + 0.18 * multi_test[key], axis=(1, 2) if key != "d" else 1)
        for key in ["d", "du", "dh"]
    }

    err_anchor = {key: l1_rows(anchor_train[key], truth[key]) for key in ["d", "du", "dh"]}
    err_sionna = {key: l1_rows(sionna_train[key], truth[key]) for key in ["d", "du", "dh"]}
    trust_targets = {}
    for key in ["d", "du", "dh"]:
        relative = 0.5 + (err_anchor[key] - err_sionna[key]) / 0.58
        absolute = np.clip((1.42 - err_sionna[key]) / 0.95, 0.0, 1.0)
        trust_targets[key] = np.clip(relative, 0.0, 1.0) * absolute

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_features = make_features(
        train_pos,
        geom["train_blocked"].astype(np.float32),
        regimes_train,
        current_train,
        high_train,
        multi_train,
        anchor_train,
        train_m["high_total_power"].astype(np.float32),
        train_m["multi_valid_count"].astype(np.float32),
        train_m["multi_pair_l1"].astype(np.float32),
    )
    test_features = make_features(
        test_pos,
        geom["test_blocked"].astype(np.float32),
        regimes_test,
        current_test,
        high_test,
        multi_test,
        anchor_test,
        test_m["high_total_power"].astype(np.float32),
        test_m["multi_valid_count"].astype(np.float32),
        test_m["multi_pair_l1"].astype(np.float32),
    )
    models, model_diag = train_trust_models(train_features, trust_targets, truth["valid"])
    trust_pred = {key: np.clip(models[key].predict(test_features), 0.0, 1.0).astype(np.float32) for key in ["d", "du", "dh"]}

    target = {}
    for key in ["d", "du", "dh"]:
        axes = (1, 2) if key != "d" else 1
        shape = (len(regimes_test),) + (1,) * (sionna_test[key].ndim - 1)
        target[key] = normalize(
            trust_pred[key].reshape(shape) * sionna_test[key]
            + (1.0 - trust_pred[key]).reshape(shape) * anchor_test[key],
            axis=axes,
        )

    high_total_zero = test_m["high_total_power"].astype(np.float32) <= 1e-20
    pair_stability = np.clip(1.10 - 0.70 * test_m["multi_pair_l1"].astype(np.float32), 0.0, 1.0)
    valid_coverage = np.clip(test_m["multi_valid_count"].astype(np.float32) / 3.0, 0.0, 1.0)
    high_multi_agree = np.clip(1.05 - 0.50 * l1_rows(high_test["d"], multi_test["d"]), 0.0, 1.0)
    constraint_quality = np.clip(
        0.18 + 0.34 * pair_stability + 0.22 * valid_coverage + 0.26 * high_multi_agree,
        0.0,
        1.20,
    ).astype(np.float32)

    alpha_d = np.zeros(len(regimes_test), dtype=np.float32)
    alpha_du = np.zeros(len(regimes_test), dtype=np.float32)
    alpha_dh = np.zeros(len(regimes_test), dtype=np.float32)
    for regime in ["clear", "light", "medium", "heavy"]:
        mask = regimes_test.astype(str) == regime
        if not np.any(mask):
            continue
        alpha_d[mask] = D_ALPHA_CAP[regime] * trust_pred["d"][mask] * constraint_quality[mask]
        alpha_du[mask] = DU_ALPHA_CAP[regime] * trust_pred["du"][mask] * constraint_quality[mask]
        alpha_dh[mask] = DH_ALPHA_CAP[regime] * trust_pred["dh"][mask] * constraint_quality[mask]
    blocked = (regimes_test.astype(str) == "clear") | high_total_zero | (constraint_quality < 0.18)
    alpha_d[blocked] = 0.0
    alpha_du[blocked] = 0.0
    alpha_dh[blocked] = 0.0

    desired = {
        "d": normalize((1.0 - alpha_d[:, None]) * current_test["d"] + alpha_d[:, None] * target["d"], axis=1),
        "du": normalize((1.0 - alpha_du[:, None, None]) * current_test["du"] + alpha_du[:, None, None] * target["du"], axis=(1, 2)),
        "dh": normalize((1.0 - alpha_dh[:, None, None]) * current_test["dh"] + alpha_dh[:, None, None] * target["dh"], axis=(1, 2)),
    }
    diagnostics = {
        "model": model_diag,
        "train_error": {f"err_anchor_{key}": summarize(err_anchor[key]) for key in ["d", "du", "dh"]},
        "train_sionna_error": {f"err_sionna_{key}": summarize(err_sionna[key]) for key in ["d", "du", "dh"]},
    }
    actions = {
        "regimes": regimes_test.astype("U8"),
        "alpha_d": alpha_d.astype(np.float32),
        "alpha_du": alpha_du.astype(np.float32),
        "alpha_dh": alpha_dh.astype(np.float32),
        "trust_d": trust_pred["d"],
        "trust_du": trust_pred["du"],
        "trust_dh": trust_pred["dh"],
        "constraint_quality": constraint_quality,
        "pair_stability": pair_stability.astype(np.float32),
        "valid_coverage": valid_coverage.astype(np.float32),
        "high_multi_agree": high_multi_agree.astype(np.float32),
        "high_total_zero": high_total_zero.astype(bool),
        "desired_d": desired["d"].astype(np.float32),
        "desired_du": desired["du"].astype(np.float32),
        "desired_dh": desired["dh"].astype(np.float32),
        "target_d": target["d"].astype(np.float32),
        "target_du": target["du"].astype(np.float32),
        "target_dh": target["dh"].astype(np.float32),
    }
    return actions, diagnostics


def apply_projection(actions: dict[str, np.ndarray], regimes: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    desired_d = actions["desired_d"].astype(np.float32)
    desired_du = actions["desired_du"].astype(np.float32)
    desired_dh = actions["desired_dh"].astype(np.float32)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "d_l1_actual": [],
        "du_l1_actual": [],
        "dh_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }
    for start in range(0, len(source), TEST_BATCH):
        end = min(start + TEST_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, MP, MH, 8, N_ANT, S_SC)
        power0 = (np.abs(arr0).astype(np.float32) ** 2).astype(np.float32)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        work = support0.copy()
        for _ in range(IPF_ITERS):
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_d[start:end], work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)),
                "d",
                0.98,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_du[start:end], work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)),
                "du",
                0.38,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_dh[start:end], work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)),
                "dh",
                0.34,
            )
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support0.shape)

        gain = np.sqrt((work * total0[:, None, None, None, None, None]) / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.18, 5.80).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.44, 2.45)
        gain[strong1024] = np.clip(gain[strong1024], 0.60, 1.70)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.30)
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.22 * (gain[local] - 1.0)

        h1 = np.fft.fft((arr0 * gain).reshape(end - start, 256, N_ANT, S_SC), axis=-1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        power1 = v196.delay_power_from_h(h1.astype(np.complex64))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        diag["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["d_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                ),
                axis=1,
            ).astype(np.float32)
        )
        diag["du_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        diag["dh_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 3, 4), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v213 physical-constraint projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v213_physical_constraint_action_model.py",
            PATHS.code / "generate_v209_adaptive_metric_projection_controller.py",
            PATHS.code / "generate_v212_multidomain_sionna_pdp_ensemble.py",
            PATHS.code / "generate_v207_sionna_test_reference.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 20):
        block = np.asarray(output[start : start + 20], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    for path in [BASE, V200_BASE, V212_BASE, V196_TARGET, V196_TRAIN, V212_TEST_MARGINALS, HIGH_TRAIN, HIGH_TEST, *LOW_TRAIN, *LOW_TEST]:
        require_file(path)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes_train = regime_of(geom["train_blocked"].astype(np.float32))
    regimes_test = regime_of(geom["test_blocked"].astype(np.float32))
    actions, model_diagnostics = build_physical_action_model(regimes_train, regimes_test)
    apply_diag = apply_projection(actions, regimes_test)
    action_diag = {**actions, **apply_diag}
    np.savez_compressed(ACTION_OUT, **action_diag)

    comparison_to_v209 = track.compare_output_to(OUTPUT, BASE, regimes_test)
    comparison_to_v200 = track.compare_output_to(OUTPUT, V200_BASE, regimes_test)
    comparison_to_v212 = track.compare_output_to(OUTPUT, V212_BASE, regimes_test)
    if comparison_to_v209["shape"] != [500, 256, 4, 192] or comparison_to_v209["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v209['shape']} {comparison_to_v209['dtype']}")
    if not comparison_to_v209["finite"]:
        raise ValueError("v213 output contains non-finite values")

    active = actions["alpha_d"] > 1e-7
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "Sionna PDP/delay should be used as a physical constraint instead "
            "of a fixed direct answer.  Official train H can supervise a trust "
            "model that predicts when the Sionna normalized PDP is better than "
            "the v209-style train/test anchor, then only that learned trust "
            "controls the v209 action projection."
        ),
        "new_information_used": {
            "official_train_supervision": "released train H trains a trust model for high-budget/multidomain Sionna PDP reliability",
            "sionna_physical_features": "normalized D/U-D/H-D, multi-carrier stability, high-vs-multi agreement, delay spread and regime/blocked features",
            "framework_boundary": "base is v209; Sionna phase, raw power and raw complex H are not copied",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "high_train_reference": PATHS.relative(HIGH_TRAIN),
            "high_test_reference": PATHS.relative(HIGH_TEST),
            "low_train_references": [PATHS.relative(path) for path in LOW_TRAIN],
            "low_test_references": [PATHS.relative(path) for path in LOW_TEST],
            "ipf_iters": IPF_ITERS,
            "train_k": TRAIN_K,
            "bias_exponents": {"d": BIAS_EXPONENT_D, "du": BIAS_EXPONENT_DU, "dh": BIAS_EXPONENT_DH},
            "alpha_caps": {"d": D_ALPHA_CAP, "du": DU_ALPHA_CAP, "dh": DH_ALPHA_CAP},
            "phase_policy": "preserve v209 delay-domain phase; no Sionna phase transport",
            "power_policy": "preserve v209 point total power exactly",
        },
        "action_counts": {
            "support_active": int(np.sum(active)),
            "blocked_count": int(np.sum(~active)),
            "active_by_regime": {regime: int(np.sum(active & (regimes_test.astype(str) == regime))) for regime in ["clear", "light", "medium", "heavy"]},
        },
        "model_diagnostics": model_diagnostics,
        "diagnostics": {
            "overall": {
                key: summarize(value)
                for key, value in action_diag.items()
                if key
                not in {
                    "regimes",
                    "desired_d",
                    "desired_du",
                    "desired_dh",
                    "target_d",
                    "target_du",
                    "target_dh",
                }
            },
            "by_regime": by_regime_summary(
                regimes_test,
                {
                    "alpha_d": actions["alpha_d"],
                    "alpha_du": actions["alpha_du"],
                    "alpha_dh": actions["alpha_dh"],
                    "trust_d": actions["trust_d"],
                    "trust_du": actions["trust_du"],
                    "trust_dh": actions["trust_dh"],
                    "constraint_quality": actions["constraint_quality"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "d_l1_actual": apply_diag["d_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v209": comparison_to_v209,
        "comparison_to_v200": comparison_to_v200,
        "comparison_to_v212_reference": comparison_to_v212,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "train_sionna_marginals": PATHS.relative(TRAIN_SIONNA_MARGINALS),
            "test_marginals": PATHS.relative(TEST_MARGINALS),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
