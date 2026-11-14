#!/usr/bin/env python3
"""v225 analysis: official graph-residual spectrum field.

This is an analysis-only strict official-data probe.  v223/v224 showed that
released train H full-tensor support and the combined train/test graph contain
positive signal, but direct propagation is too weak.  v225 keeps the same data
boundary and adds a learned residual field:

* targets are D, U-D and H-D delay/PDP marginals extracted from official train H;
* anchors are leave-one-out official train-H graph anchors and v224 test graph
  propagated support;
* features are official coordinates, height/point-cloud/geodesic summaries,
  multi-bounce geometric path tokens, and anchor spectrum summaries;
* no Sionna generated channels, v212/v217 teacher outputs, hidden labels or
  synthetic replay truth are used for generation.

The v208 local scorer is used only after generation as the hard diagnostic gate.
If this does not pass the v212 gate, no submission ZIP is produced.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

import analyze_v208_strict_historical_replay as v208
import analyze_v223_official_fulltensor_support_signal as v223
import generate_v196_train_guided_delay_pdp_mouth as v196
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import MH, MP, N_ANT, S_SC, summarize
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 225
VERSION = "v225_official_graph_residual_field"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v225_official_graph_residual_field.json"
ACTION_OUT = OUT_DIR / "v225_official_graph_residual_actions.npz"

BASE_V209 = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
BASE_V224 = PATHS.cache_dir(224) / "v224_mix0.58_it8_moderate_Round1_Test_Channel.npy"
V224_DELAY_TARGET = PATHS.cache_dir(224) / "v224_delay_mix0.58_it8.npy"

GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
GEODESIC_CACHE = PATHS.shared_features / "geodesic_support_features.npz"
PATH_CACHE = PATHS.shared_features / "multibounce_path_candidates.npz"

FLOOR = 1e-12
K_ANCHOR = 84
PROJ_BATCH = 5
PCA_DIMS = {"d": 56, "du": 64, "dh": 72}


@dataclass
class Prepared:
    x: np.ndarray
    anchors: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    coeff: dict[str, np.ndarray]
    regimes: np.ndarray


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)),
        axis=1,
    ).astype(np.float32)


def make_coeff(anchor: np.ndarray, target: np.ndarray, basis: v215.Basis) -> np.ndarray:
    residual = np.clip(
        np.log(np.maximum(target.reshape(len(target), -1), FLOOR))
        - np.log(np.maximum(anchor.reshape(len(anchor), -1), FLOOR)),
        -8.0,
        8.0,
    ).astype(np.float32)
    raw = (residual - basis.mean[None, :]) @ basis.components.T
    return (raw / basis.coeff_scale[None, :]).astype(np.float32)


def reconstruct(anchor: np.ndarray, coeff_norm: np.ndarray, basis: v215.Basis) -> np.ndarray:
    delta = (coeff_norm.astype(np.float32) * basis.coeff_scale[None, :]) @ basis.components
    logits = np.log(np.maximum(anchor.reshape(len(anchor), -1), FLOOR)) + basis.mean[None, :] + delta
    logits -= logits.max(axis=1, keepdims=True)
    flat = np.exp(logits).astype(np.float32)
    flat /= np.maximum(flat.sum(axis=1, keepdims=True), TINY)
    return flat.reshape((len(anchor),) + basis.shape).astype(np.float32)


def delay_marginals_from_support(support: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "d": normalize(support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32), axis=1),
        "du": normalize(support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32), axis=(1, 2)),
        "dh": normalize(support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32), axis=(1, 2)),
    }


def delay_stats(d: np.ndarray) -> np.ndarray:
    arr = normalize(d, axis=d.ndim - 1)
    idx = np.arange(arr.shape[-1], dtype=np.float32)
    shape = (1,) * (arr.ndim - 1) + (arr.shape[-1],)
    mean = np.sum(arr * idx.reshape(shape), axis=-1)
    var = np.sum(arr * (idx.reshape(shape) - mean[..., None]) ** 2, axis=-1)
    entropy = -np.sum(arr * np.log(np.maximum(arr, FLOOR)), axis=-1) / math.log(arr.shape[-1])
    early = arr[..., :16].sum(axis=-1)
    mid = arr[..., 16:48].sum(axis=-1)
    late = arr[..., 48:].sum(axis=-1)
    top = np.max(arr, axis=-1)
    arg = np.argmax(arr, axis=-1).astype(np.float32) / max(arr.shape[-1] - 1, 1)
    return np.concatenate(
        [
            (mean / max(arr.shape[-1] - 1, 1)).reshape(len(arr), -1),
            (np.sqrt(np.maximum(var, 0.0)) / max(arr.shape[-1] - 1, 1)).reshape(len(arr), -1),
            entropy.reshape(len(arr), -1),
            early.reshape(len(arr), -1),
            mid.reshape(len(arr), -1),
            late.reshape(len(arr), -1),
            top.reshape(len(arr), -1),
            arg.reshape(len(arr), -1),
        ],
        axis=1,
    ).astype(np.float32)


def position_features(pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
    rel = pos.astype(np.float32) - bs[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arcsin(np.clip(rel[:, 2] / np.maximum(dist, 1e-6), -1.0, 1.0))
    xy = pos[:, :2].astype(np.float32)
    fourier = []
    for scale in (1.0, 2.0, 4.0, 8.0, 12.0):
        fourier.extend(
            [
                np.sin(np.pi * scale * xy[:, 0] / 260.0),
                np.cos(np.pi * scale * xy[:, 0] / 260.0),
                np.sin(np.pi * scale * xy[:, 1] / 260.0),
                np.cos(np.pi * scale * xy[:, 1] / 260.0),
            ]
        )
    onehot = np.stack(
        [(regimes.astype(str) == name).astype(np.float32) for name in ("clear", "light", "medium", "heavy")],
        axis=1,
    )
    return np.concatenate(
        [
            pos.astype(np.float32) / np.asarray([260.0, 260.0, 60.0], dtype=np.float32),
            rel / 260.0,
            (dist / 260.0)[:, None],
            np.stack([np.sin(az), np.cos(az), np.sin(el), np.cos(el)], axis=1),
            blocked[:, None].astype(np.float32) / 220.0,
            onehot,
            np.stack(fourier, axis=1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def path_features(prefix: str) -> np.ndarray:
    cache = np.load(PATH_CACHE)
    valid = cache[f"{prefix}_valid"].astype(np.float32)
    path_type = cache[f"{prefix}_path_type"].astype(np.float32)
    delay_bin = cache[f"{prefix}_delay_bin"].astype(np.float32)
    score = cache[f"{prefix}_score"].astype(np.float32)
    excess = cache[f"{prefix}_excess_length"].astype(np.float32)
    inc = cache[f"{prefix}_incidence_cos"].astype(np.float32)
    edge = np.nan_to_num(cache[f"{prefix}_edge_margin"].astype(np.float32), posinf=250.0)
    tx = cache[f"{prefix}_tx_dir"].astype(np.float32)
    rx = cache[f"{prefix}_rx_dir"].astype(np.float32)
    plane = cache[f"{prefix}_plane_id"].astype(np.float32)
    plane2 = cache[f"{prefix}_plane_id_2"].astype(np.float32)
    numeric = np.concatenate(
        [
            valid[..., None],
            path_type[..., None] / 3.0,
            delay_bin[..., None] / 16.0,
            score[..., None] / 4.0,
            np.log1p(np.maximum(excess, 0.0))[..., None] / 6.0,
            inc[..., None],
            np.log1p(np.maximum(edge, 0.0))[..., None] / 6.0,
            tx,
            rx,
            np.sin(delay_bin[..., None]),
            np.cos(delay_bin[..., None]),
            np.where(plane[..., None] >= 0.0, plane[..., None] / 80.0, 0.0),
            np.where(plane2[..., None] >= 0.0, plane2[..., None] / 80.0, 0.0),
        ],
        axis=2,
    )
    numeric[valid <= 0.0] = 0.0
    flat = numeric.reshape(len(valid), -1)
    # Add order-invariant summaries so tree models can use path populations,
    # not only fixed token positions.
    summaries = []
    for arr in (delay_bin, score, excess, inc, edge):
        masked = np.where(valid > 0, arr, np.nan)
        summaries.extend(
            [
                np.nanmin(masked, axis=1),
                np.nanmean(masked, axis=1),
                np.nanmax(masked, axis=1),
                np.nanstd(masked, axis=1),
            ]
        )
    return np.nan_to_num(
        np.concatenate([flat, np.stack(summaries, axis=1).astype(np.float32)], axis=1),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


def scene_features(prefix: str) -> np.ndarray:
    scene = np.load(SCENE_CACHE)
    geo = np.load(GEODESIC_CACHE)
    return np.nan_to_num(
        np.concatenate(
            [
                scene[f"{prefix}_features"].astype(np.float32),
                geo[f"{prefix}_features"].astype(np.float32),
            ],
            axis=1,
        ),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


def make_features(
    prefix: str,
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    anchors: dict[str, np.ndarray],
    current: dict[str, np.ndarray] | None = None,
    row_ids: np.ndarray | None = None,
) -> np.ndarray:
    scene = scene_features(prefix)
    paths = path_features(prefix)
    if row_ids is not None:
        scene = scene[row_ids]
        paths = paths[row_ids]
    blocks = [
        position_features(pos, blocked, regimes),
        scene,
        paths,
    ]
    for key in ("d", "du", "dh"):
        blocks.append(delay_stats(anchors[key]))
    if current is not None:
        for key in ("d", "du", "dh"):
            blocks.append(delay_stats(current[key]))
            blocks.append(l1_rows(anchors[key], current[key])[:, None])
    return np.nan_to_num(
        np.concatenate(blocks, axis=1),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0), 1e-5).astype(np.float32)
    return (
        np.clip((train_x - mean) / std, -8.0, 8.0).astype(np.float32),
        np.clip((test_x - mean) / std, -8.0, 8.0).astype(np.float32),
        {"mean": mean, "std": std},
    )


def weighted_average(target: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    selected = target[idx]
    shape = (len(idx), idx.shape[1]) + (1,) * (target.ndim - 1)
    return normalize(
        np.sum(selected * weights.reshape(shape), axis=1).astype(np.float32),
        axis=tuple(range(1, target.ndim)),
    )


def build_train_anchors(
    targets: dict[str, np.ndarray],
    valid_ids: np.ndarray,
    train_pos: np.ndarray,
    train_blocked: np.ndarray,
) -> dict[str, np.ndarray]:
    idx, weights = v218.neighbor_index_weights(
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_pos[valid_ids],
        train_blocked[valid_ids],
        query_ids=valid_ids,
        bank_ids=valid_ids,
        k=K_ANCHOR,
    )
    return {key: weighted_average(targets[key][valid_ids], idx, weights) for key in ("d", "du", "dh")}


def current_delay_marginals(path: Path) -> dict[str, np.ndarray]:
    source = np.load(path, mmap_mode="r")
    parts = []
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        power = v196.delay_power_from_h(np.asarray(source[start:end], dtype=np.complex64))
        support = power / np.maximum(power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        parts.append(delay_marginals_from_support(support))
    return {
        key: normalize(
            np.concatenate([part[key] for part in parts], axis=0),
            axis=tuple(range(1, parts[0][key].ndim)),
        )
        for key in ("d", "du", "dh")
    }


def load_targets_and_features() -> tuple[Prepared, Prepared, dict[str, v215.Basis], dict[str, object]]:
    clean, _raw, _log_power, valid = v215.load_targets()
    targets_all = {key: clean[key].astype(np.float32) for key in ("d", "du", "dh")}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)
    test_regimes = regime_of(test_blocked)
    valid_ids = np.flatnonzero(valid.astype(bool))

    train_anchors = build_train_anchors(targets_all, valid_ids, train_pos, train_blocked)
    test_support = np.load(V224_DELAY_TARGET, mmap_mode="r")
    test_anchors = delay_marginals_from_support(test_support)
    current_test = current_delay_marginals(BASE_V224)

    bases: dict[str, v215.Basis] = {}
    train_coeff: dict[str, np.ndarray] = {}
    for key in ("d", "du", "dh"):
        basis, coeff = v215.fit_basis(train_anchors[key], targets_all[key][valid_ids], PCA_DIMS[key])
        bases[key] = basis
        train_coeff[key] = coeff

    x_train_raw = make_features(
        "train",
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_regimes[valid_ids],
        train_anchors,
        current=train_anchors,
        row_ids=valid_ids,
    )
    x_test_raw = make_features("test", test_pos, test_blocked, test_regimes, test_anchors, current=current_test)
    x_train, x_test, feature_stats = standardize(x_train_raw, x_test_raw)
    train_prepared = Prepared(
        x=x_train,
        anchors=train_anchors,
        targets={key: targets_all[key][valid_ids] for key in ("d", "du", "dh")},
        coeff=train_coeff,
        regimes=train_regimes[valid_ids],
    )
    test_prepared = Prepared(
        x=x_test,
        anchors=test_anchors,
        targets=test_anchors,
        coeff={key: np.zeros((len(test_pos), PCA_DIMS[key]), dtype=np.float32) for key in ("d", "du", "dh")},
        regimes=test_regimes,
    )
    diagnostics = {
        "valid_train_count": int(len(valid_ids)),
        "feature_dim": int(x_train.shape[1]),
        "train_anchor_scores": {
            key: summarize(row_cos(train_anchors[key], targets_all[key][valid_ids])) for key in ("d", "du", "dh")
        },
        "test_current_anchor_l1": {key: summarize(l1_rows(test_anchors[key], current_test[key])) for key in ("d", "du", "dh")},
        "feature_stats": feature_stats,
        "valid_ids": valid_ids,
        "test_current": current_test,
        "test_pos": test_pos,
        "test_blocked": test_blocked,
        "test_regimes": test_regimes,
    }
    return train_prepared, test_prepared, bases, diagnostics


def fit_predict_models(
    train: Prepared,
    test: Prepared,
    bases: dict[str, v215.Basis],
) -> dict[str, dict[str, np.ndarray]]:
    predictions: dict[str, dict[str, np.ndarray]] = {}
    specs = {
        "extratrees_leaf2": ExtraTreesRegressor(
            n_estimators=900,
            min_samples_leaf=2,
            max_features=0.55,
            bootstrap=True,
            random_state=22502,
            n_jobs=-1,
        ),
        "extratrees_leaf5": ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=5,
            max_features=0.72,
            bootstrap=True,
            random_state=22505,
            n_jobs=-1,
        ),
        "histgb_l2": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=170,
                learning_rate=0.045,
                l2_regularization=0.08,
                max_leaf_nodes=24,
                min_samples_leaf=24,
                random_state=22517,
            ),
            n_jobs=1,
        ),
    }
    for model_name, model in specs.items():
        predictions[model_name] = {}
        for key in ("d", "du", "dh"):
            model.fit(train.x, train.coeff[key])
            coeff_pred = model.predict(test.x).astype(np.float32)
            predictions[model_name][key] = reconstruct(test.anchors[key], coeff_pred, bases[key])
            print(f"v225 fitted {model_name} {key}", flush=True)
    return predictions


def blend_targets(
    anchor: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    regimes: np.ndarray,
    alpha_by_regime: dict[str, float],
    anchor_mix: float,
) -> dict[str, np.ndarray]:
    alpha = np.asarray([alpha_by_regime[str(r)] for r in regimes], dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for key in ("d", "du", "dh"):
        axes = 1 if key == "d" else (1, 2)
        shape = (len(regimes),) + (1,) * (anchor[key].ndim - 1)
        learned = normalize(anchor_mix * anchor[key] + (1.0 - anchor_mix) * pred[key], axis=axes)
        out[key] = normalize(
            (1.0 - alpha.reshape(shape)) * current[key]
            + alpha.reshape(shape) * learned,
            axis=axes,
        )
    return out


def apply_delay_targets(
    source_path: Path,
    output_path: Path,
    targets: dict[str, np.ndarray],
    regimes: np.ndarray,
    gain_clip_scale: float,
) -> dict[str, np.ndarray]:
    source = np.load(source_path, mmap_mode="r")
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "d_l1_actual": [],
        "du_l1_actual": [],
        "dh_l1_actual": [],
        "power_factor": [],
        "gain_min": [],
        "gain_p90": [],
        "gain_max": [],
    }
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        p0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        cir = np.fft.ifft(h0, axis=-1)
        arr = cir.reshape(end - start, MP, MH, v196.MV, N_ANT, S_SC)
        power0 = (np.abs(arr).astype(np.float32) ** 2).astype(np.float32)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support = power0 / total0[:, None, None, None, None, None]
        base_m = delay_marginals_from_support(support)
        work = support.copy()
        for _ in range(7):
            v196.apply_delay_ratio(
                work,
                np.maximum(targets["d"][start:end], FLOOR)
                / np.maximum(work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32), FLOOR),
                "d",
                0.96,
            )
            v196.apply_delay_ratio(
                work,
                np.maximum(targets["du"][start:end], FLOOR)
                / np.maximum(work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32), FLOOR),
                "du",
                0.42,
            )
            v196.apply_delay_ratio(
                work,
                np.maximum(targets["dh"][start:end], FLOOR)
                / np.maximum(work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32), FLOOR),
                "dh",
                0.38,
            )
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support.shape)
        gain = np.sqrt((work * total0[:, None, None, None, None, None]) / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.18, 5.80 * gain_clip_scale)
        flat = support.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support.shape)
        strong1024 = top_mask(flat, 1024).reshape(support.shape)
        strong8192 = top_mask(flat, 8192).reshape(support.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.46, 2.45 * gain_clip_scale)
        gain[strong1024] = np.clip(gain[strong1024], 0.62, 1.76 * gain_clip_scale)
        gain[strong128] = np.clip(gain[strong128], 0.80, 1.34 * gain_clip_scale)
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.35 * (gain[local] - 1.0)
        h1 = np.fft.fft((arr * gain).reshape(end - start, 256, N_ANT, S_SC), axis=-1).astype(np.complex64)
        p1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        h1 *= np.sqrt(p0 / p1)[:, None, None, None].astype(np.float32)
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        final_power = v196.delay_power_from_h(h1.astype(np.complex64))
        final_support = final_power / np.maximum(final_power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        final_m = delay_marginals_from_support(final_support)
        diag["d_l1_actual"].append(l1_rows(final_m["d"], base_m["d"]))
        diag["du_l1_actual"].append(l1_rows(final_m["du"], base_m["du"]))
        diag["dh_l1_actual"].append(l1_rows(final_m["dh"], base_m["dh"]))
        diag["power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / p0).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        if end == len(source) or end % 100 == 0:
            print(f"v225 projection {output_path.name} {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def run_variants(
    predictions: dict[str, dict[str, np.ndarray]],
    test: Prepared,
    diagnostics: dict[str, object],
) -> list[dict[str, object]]:
    current: dict[str, np.ndarray] = diagnostics["test_current"]  # type: ignore[assignment]
    regimes: np.ndarray = diagnostics["test_regimes"]  # type: ignore[assignment]
    variants = []
    configs = [
        ("moderate", BASE_V224, {"clear": 0.0, "light": 0.04, "medium": 0.34, "heavy": 0.48}, 0.54, 1.0),
        ("strong", BASE_V224, {"clear": 0.0, "light": 0.06, "medium": 0.52, "heavy": 0.70}, 0.38, 1.08),
        ("delayonly_over", BASE_V224, {"clear": 0.0, "light": 0.08, "medium": 0.66, "heavy": 0.86}, 0.24, 1.16),
        ("v209_strong", BASE_V209, {"clear": 0.0, "light": 0.08, "medium": 0.62, "heavy": 0.84}, 0.32, 1.14),
    ]
    for model_name, pred in predictions.items():
        for cfg_name, source, alpha, anchor_mix, gain_clip_scale in configs:
            target = blend_targets(test.anchors, pred, current, regimes, alpha, anchor_mix)
            out = OUT_DIR / f"v225_{model_name}_{cfg_name}_Round1_Test_Channel.npy"
            diag = apply_delay_targets(source, out, target, regimes, gain_clip_scale)
            score = v208.score_prediction(out)
            variants.append(
                {
                    "name": f"{model_name}_{cfg_name}",
                    "source": PATHS.relative(source),
                    "output": PATHS.relative(out),
                    "alpha": alpha,
                    "anchor_mix": anchor_mix,
                    "gain_clip_scale": gain_clip_scale,
                    "score": score,
                    "diagnostics": {key: summarize(value) for key, value in diag.items()},
                    "target_l1_to_anchor": {key: summarize(l1_rows(target[key], test.anchors[key])) for key in ("d", "du", "dh")},
                    "target_l1_to_current": {key: summarize(l1_rows(target[key], current[key])) for key in ("d", "du", "dh")},
                }
            )
            print(json.dumps(variants[-1]["score"], ensure_ascii=False), flush=True)
    return variants


def main() -> None:
    for path in [BASE_V209, BASE_V224, V224_DELAY_TARGET, GEOM_CACHE, SCENE_CACHE, GEODESIC_CACHE, PATH_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    train, test, bases, diagnostics = load_targets_and_features()
    predictions = fit_predict_models(train, test, bases)
    variants = run_variants(predictions, test, diagnostics)
    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        regimes=diagnostics["test_regimes"].astype("U8"),  # type: ignore[index]
        **{f"pred_{model}_{key}": pred[key] for model, pred in predictions.items() for key in ("d", "du", "dh")},
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "v224's graph propagated support is useful but too deterministic; "
            "a learned residual field from official train-H labels and official "
            "geometry/path features may identify the propagation modes that are "
            "actually externally predictable."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official Round1_Train_Channel.npy targets",
                "official train/test positions",
                "official point-cloud/geodesic/multibounce geometry caches",
                "v209 and v224 official-data backbones/anchors",
            ],
            "generation_does_not_use": [
                "Sionna generated train/test labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "synthetic replay truth",
            ],
            "note": "Analysis-only tree residual field; no ZIP unless the local v212 gate is exceeded.",
        },
        "local_gate_required": 0.645604,
        "diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if key not in {"valid_ids", "test_current", "test_pos", "test_blocked", "test_regimes", "feature_stats"}
        },
        "variants": variants,
        "best_variant": best["name"],
        "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
        "gate_passed": bool(best["score"]["score_traceback_leaked"] >= 0.645604),
        "outputs": {
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
