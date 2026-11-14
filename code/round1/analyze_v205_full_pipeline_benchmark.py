#!/usr/bin/env python3
"""Run current-method and new-information ablations on the v205 benchmark.

The benchmark has known clean test truth.  Train observations are generated
from clean Sionna train H under several plausible release-noise mechanisms.
This script evaluates the actual method families that produced the online
progress: physical phase-aware interpolation, leaked spectral consensus,
delay/PDP projection, and selective total-power correction.  It also evaluates
a point-cloud/path-feature propagation-state gate suggested by v204.
"""

from __future__ import annotations

from collections import defaultdict
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

import analyze_v203_sionna_like_scene_simulation as v203
import analyze_v205_sionna_config_calibration as calibration
import generate_v187_leaked_pas_lifecycle_projection as leaked
import generate_v189_leaked_multimarginal_overdrive as v189
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_geometric_steering_blend import unit_vectors
from generate_physical_local_linear_v11 import predict_v11
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v205_full_pipeline_benchmark"
OUT_DIR = PATHS.cache_dir(205)
REPORT_DIR = PATHS.version_report_dir(205)
REPORT = REPORT_DIR / "v205_full_pipeline_benchmark.json"
POINT_METRICS = OUT_DIR / "v205_primary_hybrid_point_metrics.npz"
DOMAIN_PROFILES = (
    "sector_m30_track_d5_f6p0",
    "sector_m30_track_d5_f2p6",
    "sector_m30_track_d5_f6p7",
)
NOISE_SCENARIOS = {
    "clean": {"relative_snr_db": None, "floor_db_below_median": None},
    "relative_25db": {"relative_snr_db": 25.0, "floor_db_below_median": None},
    "hybrid_rel30_floor45": {"relative_snr_db": 30.0, "floor_db_below_median": 45.0},
}
PRIMARY_PROFILE = DOMAIN_PROFILES[0]
PRIMARY_SCENARIO = "hybrid_rel30_floor45"
PILOT_CLEAN_SCORES = {
    "physical_base": 0.6767102397552536,
    "learned_state_gate": 0.7528023675506790,
    "learned_state_plus_power": 0.7533224995064399,
    "leaked_multimarginal": 0.7486424629541420,
    "delay_pdp": 0.7519968464251086,
    "full_pipeline_power": 0.7524777582229593,
    "oracle_state_gate": 0.8207668564705444,
    "oracle_total_power": 0.8175191556374180,
    "oracle_pas_pdp": 0.7958262748584279,
    "oracle_pas_pdp_and_power": 0.9368206357360205,
}
BS_POS = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
SPEED_OF_LIGHT = 299_792_458.0
REGIMES = ("clear", "light", "medium", "heavy")
BATCH = 8


def total_power(h: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return {"count": 0}
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
        "min": float(np.min(value)),
        "p10": float(np.quantile(value, 0.10)),
        "p50": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def make_observed_train(
    clean_h,
    scenario: dict[str, float | None],
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    clean_power = np.empty(len(clean_h), dtype=np.float64)
    for start in range(0, len(clean_h), 32):
        end = min(start + 32, len(clean_h))
        clean_power[start:end] = total_power(np.asarray(clean_h[start:end], dtype=np.complex64))
    element_count = float(np.prod(clean_h.shape[1:]))
    clean_rms = np.sqrt(clean_power / element_count)
    positive = clean_rms > 0.0
    reference_rms = float(np.median(clean_rms[positive]))
    relative_snr = scenario["relative_snr_db"]
    floor_db = scenario["floor_db_below_median"]
    if relative_snr is None and floor_db is None:
        return np.asarray(clean_h), {
            "relative_snr_db": None,
            "floor_db_below_median": None,
            "reference_rms": reference_rms,
            "observed_is_clean_view": True,
        }

    relative_sigma = (
        clean_rms * 10.0 ** (-float(relative_snr) / 20.0)
        if relative_snr is not None
        else np.zeros_like(clean_rms)
    )
    floor_sigma = reference_rms * 10.0 ** (-float(floor_db) / 20.0) if floor_db is not None else 0.0
    sigma = np.sqrt(relative_sigma * relative_sigma + floor_sigma * floor_sigma).astype(np.float32)
    observed = np.empty(clean_h.shape, dtype=np.complex64)
    rng = np.random.default_rng(seed)
    for start in range(0, len(clean_h), 16):
        end = min(start + 16, len(clean_h))
        clean = np.asarray(clean_h[start:end], dtype=np.complex64)
        real = rng.standard_normal(clean.shape, dtype=np.float32)
        imag = rng.standard_normal(clean.shape, dtype=np.float32)
        noise = (real + 1j * imag) * (sigma[start:end, None, None, None] / np.sqrt(2.0))
        observed[start:end] = clean + noise.astype(np.complex64)
    observed_power = total_power(observed)
    actual_noise = observed - np.asarray(clean_h)
    noise_power = total_power(actual_noise)
    return observed, {
        "relative_snr_db": relative_snr,
        "floor_db_below_median": floor_db,
        "reference_rms": reference_rms,
        "sigma": summarize(sigma),
        "observed_log_power": summarize(np.log(np.maximum(observed_power, 1e-30))),
        "actual_point_snr_db": summarize(10.0 * np.log10(np.maximum(clean_power, 1e-30) / np.maximum(noise_power, 1e-30))),
        "observed_is_clean_view": False,
    }


def infer_low_state_threshold(log_power: np.ndarray) -> tuple[float, dict[str, object]]:
    value = np.sort(np.asarray(log_power, dtype=np.float64))
    lo = max(1, int(0.005 * len(value)))
    hi = max(lo + 1, int(0.60 * len(value)))
    gap = np.diff(value)
    index = lo + int(np.argmax(gap[lo:hi]))
    threshold = float(0.5 * (value[index] + value[index + 1]))
    return threshold, {
        "threshold": threshold,
        "gap": float(gap[index]),
        "lower_count": int(index + 1),
        "lower_fraction": float((index + 1) / len(value)),
        "lower_edge": float(value[index]),
        "upper_edge": float(value[index + 1]),
    }


def physical_predict(
    observed_train: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    frequency_hz: float,
    subcarrier_spacing_hz: float,
) -> tuple[np.ndarray, dict[str, object]]:
    max_k = 24
    tree = cKDTree(train_pos[:, :2])
    distances, indices = tree.query(test_pos[:, :2], k=max_k, workers=-1)
    all_pos = np.concatenate([train_pos, test_pos], axis=0)
    dist_bs = np.linalg.norm(all_pos - BS_POS[None, :], axis=1)
    directions = unit_vectors(all_pos)
    offset = len(train_pos)
    beta = 2.0 * math.pi * frequency_hz / SPEED_OF_LIGHT
    bandwidth = 192.0 * subcarrier_spacing_hz
    gamma = 2.0 * math.pi * bandwidth / SPEED_OF_LIGHT
    args = SimpleNamespace(
        k=24,
        power=2.0,
        beta=beta,
        gamma=gamma,
        alpha=1.0,
        ridge=0.3,
        steering_scale=4.71238898038,
        fallback_k=4,
        idw_k=16,
        ratio_small_thresh=0.50,
        ratio_idw_thresh=0.60,
        neg_frac_thresh=0.15,
        cos_thresh=0.99,
        gain_cap=1.25,
        severe_ratio_small_thresh=0.30,
        severe_neg_frac_thresh=0.30,
        mode="gain_only",
        blend_affine=0.25,
    )
    output = np.empty((len(test_pos), 256, 4, 192), dtype=np.complex64)
    branches: defaultdict[str, int] = defaultdict(int)
    gains = np.empty(len(test_pos), dtype=np.float32)
    for i in range(len(test_pos)):
        idx = indices[i]
        neighbor_h = np.asarray(observed_train[idx], dtype=np.complex64)
        query_index = offset + i
        prediction, diag = predict_v11(
            neighbor_h,
            distances[i],
            train_pos[idx, :2] - test_pos[i, :2],
            dist_bs[idx] - dist_bs[query_index],
            directions,
            idx,
            query_index,
            args,
        )
        output[i] = prediction
        branches[str(diag["branch"])] += 1
        gains[i] = float(diag["gain"])
        if (i + 1) % 50 == 0:
            print(f"physical predictor {i + 1}/{len(test_pos)}", flush=True)
    return output, {
        "k": 24,
        "beta_rad_per_m": beta,
        "gamma_rad_per_m": gamma,
        "frequency_hz": frequency_hz,
        "subcarrier_spacing_hz": subcarrier_spacing_hz,
        "branches": dict(branches),
        "gain": summarize(gains),
    }


def fit_state_and_power_models(
    observed_train: np.ndarray,
    clean_test,
    features,
    seed: int,
) -> dict[str, object | np.ndarray]:
    train_log = np.log(np.maximum(total_power(observed_train), 1e-30)).astype(np.float32)
    test_clean_power = total_power(np.asarray(clean_test))
    test_nonzero = test_clean_power > 1e-25
    threshold, threshold_report = infer_low_state_threshold(train_log)
    train_nonzero = train_log > threshold
    x_train = np.concatenate([features.train_basic_features, features.train_path_features_public], axis=1)
    x_test = np.concatenate([features.test_basic_features, features.test_path_features_public], axis=1)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)

    classifier = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.80,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    classifier.fit(x_train, train_nonzero.astype(np.int32))
    nonzero_probability = classifier.predict_proba(x_test)[:, 1].astype(np.float32)

    extra = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.82,
        random_state=seed + 1,
        n_jobs=-1,
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=0.04,
        random_state=seed + 2,
    )
    extra.fit(x_train[train_nonzero], train_log[train_nonzero])
    hgb.fit(x_train[train_nonzero], train_log[train_nonzero])
    pred_extra = extra.predict(x_test).astype(np.float32)
    pred_hgb = hgb.predict(x_test).astype(np.float32)
    predicted_log_power = (0.5 * pred_extra + 0.5 * pred_hgb).astype(np.float32)
    power_spread = np.abs(pred_extra - pred_hgb).astype(np.float32)
    auc = float(roc_auc_score(test_nonzero.astype(np.int32), nonzero_probability))
    return {
        "train_log_power": train_log,
        "train_nonzero": train_nonzero,
        "test_nonzero": test_nonzero,
        "nonzero_probability": nonzero_probability,
        "predicted_log_power": predicted_log_power,
        "power_spread": power_spread,
        "report": {
            "threshold": threshold_report,
            "train_nonzero_count": int(np.sum(train_nonzero)),
            "test_true_nonzero_count": int(np.sum(test_nonzero)),
            "test_state_auc": auc,
            "probability": summarize(nonzero_probability),
            "predicted_log_power": summarize(predicted_log_power),
            "power_spread": summarize(power_spread),
        },
    }


def apply_state_gate(h: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state_power_scale = np.clip((probability - 0.22) / 0.56, 0.0, 1.0).astype(np.float32)
    state_power_scale[probability < 0.18] = 0.0
    return (h * np.sqrt(state_power_scale)[:, None, None, None]).astype(np.complex64), state_power_scale


def test_graph_matrix(
    test_pos: np.ndarray,
    test_blocked: np.ndarray,
    test_regimes: np.ndarray,
) -> tuple[sparse.csr_matrix, dict[str, np.ndarray]]:
    return v189.build_weight_matrix(test_pos, test_blocked, test_regimes)


def leaked_multimarginal_projection(
    source: np.ndarray,
    regimes: np.ndarray,
    matrix: sparse.csr_matrix,
    graph_diag: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    shape = (len(source), 2, 192, 16, 4, 8)
    support = np.empty(shape, dtype=np.float32)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        spec = leaked.leaked_forward(source[start:end])
        power = np.abs(spec).astype(np.float32) ** 2
        total = np.sum(power, axis=(1, 2, 3, 4, 5), keepdims=True, dtype=np.float64)
        support[start:end] = power / np.maximum(total, TINY)
    flat = support.reshape(len(source), -1)
    consensus_flat = np.asarray(matrix @ flat, dtype=np.float32)
    consensus_flat = v189.normalize_rows(consensus_flat)
    consensus = consensus_flat.reshape(shape)
    agreement = graph_diag["neighbor_agreement"].astype(np.float32)
    alpha_full = np.asarray(
        [v189.ALPHA_SCALE[str(regime)] for regime in regimes], dtype=np.float32
    ) * (0.78 + 0.22 * agreement)
    alpha_margin = np.asarray(
        [v189.MARGINAL_SCALE[str(regime)] for regime in regimes], dtype=np.float32
    ) * (0.76 + 0.24 * agreement)
    work = v189.normalize_rows(
        ((1.0 - alpha_full[:, None]) * flat + alpha_full[:, None] * consensus_flat)
    ).reshape(shape)

    sh0 = support.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    psh0 = support.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    shue0 = support.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shv0 = support.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    shc = consensus.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    pshc = consensus.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    shuec = consensus.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shvc = consensus.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    desired_sh = v189.blended_marginal(sh0, shc, alpha_margin[:, None, None])
    desired_psh = v189.blended_marginal(psh0, pshc, alpha_margin[:, None, None, None])
    desired_shue = v189.blended_marginal(shue0, shuec, alpha_margin[:, None, None, None])
    desired_shv = v189.blended_marginal(shv0, shvc, alpha_margin[:, None, None, None])
    for _ in range(v189.IPF_ITERS):
        v189.apply_ratio(work, v189.safe_ratio(desired_sh, work.sum(axis=(1, 4, 5))), "sh", 0.70)
        v189.apply_ratio(work, v189.safe_ratio(desired_psh, work.sum(axis=(4, 5))), "psh", 0.42)
        v189.apply_ratio(work, v189.safe_ratio(desired_shue, work.sum(axis=(1, 5))), "shue", 0.46)
        v189.apply_ratio(work, v189.safe_ratio(desired_shv, work.sum(axis=(1, 4))), "shv", 0.42)
        work = v189.normalize_rows(work.reshape(len(source), -1)).reshape(shape)

    output = np.empty_like(source)
    changed_l1 = np.sum(np.abs(work.reshape(len(source), -1) - flat), axis=1).astype(np.float32)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = source[start:end]
        spec0 = leaked.leaked_forward(h0)
        power0 = np.abs(spec0).astype(np.float32) ** 2
        total0 = np.sum(power0, axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
        h_power0 = total_power(h0).astype(np.float32)
        target_power = work[start:end] * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.16, 6.80)
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.72 * (gain[local] - 1.0)
        h1 = leaked.leaked_inverse(spec0 * gain)
        power1 = total_power(h1)
        scale = np.sqrt(h_power0 / np.maximum(power1, 1e-30)).astype(np.float32)
        scale[h_power0 <= 1e-30] = 0.0
        output[start:end] = h1 * scale[:, None, None, None]
    return output, {
        "alpha_full": summarize(alpha_full),
        "alpha_margin": summarize(alpha_margin),
        "target_l1": summarize(changed_l1),
    }


def train_delay_marginals(observed_train: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    n = len(observed_train)
    d = np.empty((n, 192), dtype=np.float32)
    du = np.empty((n, 4, 192), dtype=np.float32)
    dh = np.empty((n, 16, 192), dtype=np.float32)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        power = v196.delay_power_from_h(observed_train[start:end])
        total = np.maximum(power.sum(axis=(1, 2, 3, 4, 5)), TINY)
        d[start:end] = power.sum(axis=(1, 2, 3, 4)) / total[:, None]
        du[start:end] = power.sum(axis=(1, 2, 3)) / total[:, None, None]
        dh[start:end] = power.sum(axis=(1, 3, 4)) / total[:, None, None]
    return {"d": d, "du": du, "dh": dh, "valid": valid.astype(np.float32)}


def delay_pdp_projection(
    source: np.ndarray,
    observed_train: np.ndarray,
    train_valid: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
    matrix: sparse.csr_matrix,
    graph_diag: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    train_marginals = train_delay_marginals(observed_train, train_valid)
    train_idx, train_weights, train_conf = v196.geometry_train_weights(
        train_pos,
        test_pos,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
        train_marginals["valid"],
    )
    train_pred = v196.predict_train_marginals(train_marginals, train_idx, train_weights)
    support = np.empty((len(source), 2, 16, 8, 4, 192), dtype=np.float32)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        power = v196.delay_power_from_h(source[start:end])
        support[start:end] = power / np.maximum(
            power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY
        )
    d0 = support.sum(axis=(1, 2, 3, 4))
    du0 = support.sum(axis=(1, 2, 3))
    dh0 = support.sum(axis=(1, 3, 4))
    dc = np.asarray(matrix @ d0.reshape(len(source), -1), dtype=np.float32).reshape(d0.shape)
    duc = np.asarray(matrix @ du0.reshape(len(source), -1), dtype=np.float32).reshape(du0.shape)
    dhc = np.asarray(matrix @ dh0.reshape(len(source), -1), dtype=np.float32).reshape(dh0.shape)
    agreement = graph_diag["neighbor_agreement"]
    alpha_d = v196.alpha_by_regime(test_regimes, v196.D_ALPHA, train_conf, agreement)
    alpha_du = v196.alpha_by_regime(test_regimes, v196.DU_ALPHA, train_conf, agreement)
    alpha_dh = v196.alpha_by_regime(test_regimes, v196.DH_ALPHA, train_conf, agreement)
    desired_d = v196.mix_target(d0, train_pred["d"], dc, test_regimes, alpha_d)
    desired_du = v196.mix_target(du0, train_pred["du"], duc, test_regimes, alpha_du)
    desired_dh = v196.mix_target(dh0, train_pred["dh"], dhc, test_regimes, alpha_dh)
    work = support.copy()
    for _ in range(v196.IPF_ITERS):
        v196.apply_delay_ratio(work, v196.safe_ratio(desired_d, work.sum(axis=(1, 2, 3, 4))), "d", 0.92)
        v196.apply_delay_ratio(work, v196.safe_ratio(desired_du, work.sum(axis=(1, 2, 3))), "du", 0.58)
        v196.apply_delay_ratio(work, v196.safe_ratio(desired_dh, work.sum(axis=(1, 3, 4))), "dh", 0.44)
        work = v196.normalize_rows(work.reshape(len(source), -1), axis=1).reshape(work.shape)

    output = np.empty_like(source)
    target_l1 = np.sum(np.abs(work.reshape(len(source), -1) - support.reshape(len(source), -1)), axis=1)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = source[start:end]
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, 2, 16, 8, 4, 192)
        power0 = np.abs(arr0).astype(np.float32) ** 2
        total0 = power0.sum(axis=(1, 2, 3, 4, 5))
        gain = np.sqrt(
            work[start:end] * total0[:, None, None, None, None, None]
            / np.maximum(power0, 1e-30)
        ).astype(np.float32)
        gain = np.clip(gain, 0.16, 6.40)
        for local, regime in enumerate(test_regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.42 * (gain[local] - 1.0)
        h1 = np.fft.fft((arr0 * gain).reshape(end - start, 256, 4, 192), axis=-1)
        p0 = total_power(h0)
        p1 = total_power(h1)
        scale = np.sqrt(p0 / np.maximum(p1, 1e-30)).astype(np.float32)
        scale[p0 <= 1e-30] = 0.0
        output[start:end] = h1 * scale[:, None, None, None]
    return output, {
        "alpha_d": summarize(alpha_d),
        "alpha_du": summarize(alpha_du),
        "alpha_dh": summarize(alpha_dh),
        "target_l1": summarize(target_l1),
        "train_confidence": summarize(train_conf),
    }


def apply_selective_power(
    source: np.ndarray,
    predicted_log_power: np.ndarray,
    nonzero_probability: np.ndarray,
    spread: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    current_power = total_power(source)
    current_log = np.log(np.maximum(current_power, 1e-30))
    delta = predicted_log_power - current_log
    spread_limit = float(np.quantile(spread, 0.70))
    active = (
        (current_power > 1e-25)
        & (nonzero_probability >= 0.72)
        & (spread <= spread_limit)
        & (np.abs(delta) >= 0.10)
    )
    applied = np.zeros(len(source), dtype=np.float32)
    applied[active] = np.clip(0.42 * delta[active], -0.42, 0.42).astype(np.float32)
    output = source * np.exp(0.5 * applied)[:, None, None, None].astype(np.float32)
    return output.astype(np.complex64), {
        "active_count": int(np.sum(active)),
        "up_count": int(np.sum(active & (applied > 0))),
        "down_count": int(np.sum(active & (applied < 0))),
        "spread_limit": spread_limit,
        "applied_log_power_delta": summarize(applied[active]),
    }, applied


def scale_to_true_power(source: np.ndarray, truth) -> np.ndarray:
    p0 = total_power(source)
    pt = total_power(np.asarray(truth))
    scale = np.sqrt(pt / np.maximum(p0, 1e-30)).astype(np.float32)
    scale[pt <= 1e-30] = 0.0
    return (source * scale[:, None, None, None]).astype(np.complex64)


def oracle_metric_projection(source: np.ndarray, truth, preserve_true_power: bool) -> np.ndarray:
    output = np.empty_like(source)
    for start in range(0, len(source), 4):
        end = min(start + 4, len(source))
        value = source[start:end].reshape(end - start, 2, 16, 8, 4, 192).astype(np.complex64)
        target = np.asarray(truth[start:end], dtype=np.complex64).reshape(value.shape)
        original_power = np.sum(np.abs(value) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
        true_power = np.sum(np.abs(target) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
        for _ in range(3):
            angle = np.fft.fft2(value, axes=(2, 3))
            target_angle = np.fft.fft2(target, axes=(2, 3))
            current_pas = np.abs(angle).astype(np.float32) ** 2
            desired_pas = np.abs(target_angle).astype(np.float32) ** 2
            current_pas /= np.maximum(current_pas.sum(axis=(1, 2, 3), keepdims=True), 1e-30)
            desired_pas /= np.maximum(desired_pas.sum(axis=(1, 2, 3), keepdims=True), 1e-30)
            angle *= np.clip(np.sqrt((desired_pas + 1e-12) / (current_pas + 1e-12)), 0.08, 12.0)
            value = np.fft.ifft2(angle, axes=(2, 3))

            delay = np.fft.ifft(value, axis=-1)
            target_delay = np.fft.ifft(target, axis=-1)
            current_pdp = np.abs(delay).astype(np.float32) ** 2
            desired_pdp = np.abs(target_delay).astype(np.float32) ** 2
            current_pdp /= np.maximum(current_pdp.sum(axis=-1, keepdims=True), 1e-30)
            desired_pdp /= np.maximum(desired_pdp.sum(axis=-1, keepdims=True), 1e-30)
            delay *= np.clip(np.sqrt((desired_pdp + 1e-12) / (current_pdp + 1e-12)), 0.08, 12.0)
            value = np.fft.fft(delay, axis=-1)
        final_power = np.sum(np.abs(value) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
        desired_total = true_power if preserve_true_power else original_power
        scale = np.sqrt(desired_total / np.maximum(final_power, 1e-30)).astype(np.float32)
        scale[desired_total <= 1e-30] = 0.0
        output[start:end] = (value * scale[:, None, None, None, None, None]).reshape(end - start, 256, 4, 192)
    return output


def evaluate_methods(
    methods: dict[str, np.ndarray],
    truth,
    regimes: np.ndarray,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    accum: dict[str, defaultdict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in methods
    }
    errors = {name: [0.0, 0.0] for name in methods}
    for start in range(0, len(truth), BATCH):
        end = min(start + BATCH, len(truth))
        truth_block = np.asarray(truth[start:end], dtype=np.complex64)
        truth_views = calibration.spectral_views(truth_block)
        truth_energy = truth_views["energy"]
        for name, value in methods.items():
            pred = value[start:end]
            scores = calibration.spectral_scores(pred, truth_views)
            pred_energy = total_power(pred)
            true_zero = truth_energy <= 1e-25
            pred_zero = pred_energy <= 1e-25
            for key, item in scores.items():
                accum[name][key].append(item)
            for key in ("pas", "pdp", "leaked_sh", "ad_support"):
                zero_aware = scores[key].copy()
                zero_aware[true_zero & pred_zero] = 1.0
                zero_aware[true_zero & ~pred_zero] = 0.0
                accum[name][f"{key}_zero_aware"].append(zero_aware)
            diff = pred - truth_block
            errors[name][0] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
            errors[name][1] += float(np.sum(np.abs(truth_block) ** 2, dtype=np.float64))

    reports: dict[str, object] = {}
    merged_metrics: dict[str, dict[str, np.ndarray]] = {}
    truth_power_all = total_power(np.asarray(truth))
    nonzero = truth_power_all > 1e-25
    for name in methods:
        merged = {key: np.concatenate(parts) for key, parts in accum[name].items()}
        merged_metrics[name] = merged
        nmse = errors[name][0] / max(errors[name][1], 1e-30)
        pas = float(np.mean(merged["pas_zero_aware"]))
        pdp = float(np.mean(merged["pdp_zero_aware"]))
        score = 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse)
        nonzero_nmse_num = float(np.sum(np.abs(methods[name][nonzero] - np.asarray(truth)[nonzero]) ** 2, dtype=np.float64))
        nonzero_nmse_den = float(np.sum(np.abs(np.asarray(truth)[nonzero]) ** 2, dtype=np.float64))
        nonzero_nmse = nonzero_nmse_num / max(nonzero_nmse_den, 1e-30)
        nonzero_pas = float(np.mean(merged["pas"][nonzero]))
        nonzero_pdp = float(np.mean(merged["pdp"][nonzero]))
        report: dict[str, object] = {
            "score_zero_aware": score,
            "pas_zero_aware": pas,
            "pdp_zero_aware": pdp,
            "global_nmse": nmse,
            "nonzero_score": 0.4 * nonzero_pas + 0.4 * nonzero_pdp + 0.2 / (1.0 + nonzero_nmse),
            "nonzero_pas": nonzero_pas,
            "nonzero_pdp": nonzero_pdp,
            "nonzero_nmse": nonzero_nmse,
            "leaked_sh_nonzero": float(np.mean(merged["leaked_sh"][nonzero])),
            "ad_support_nonzero": float(np.mean(merged["ad_support"][nonzero])),
            "predicted_zero_count": int(np.sum(total_power(methods[name]) <= 1e-25)),
        }
        for regime in REGIMES:
            mask = regimes.astype(str) == regime
            report[f"score_{regime}"] = float(
                0.4 * np.mean(merged["pas_zero_aware"][mask])
                + 0.4 * np.mean(merged["pdp_zero_aware"][mask])
            )
        reports[name] = report
    return reports, merged_metrics


def evaluate_run(
    profile_name: str,
    scenario_name: str,
    features,
    graph_matrix: sparse.csr_matrix,
    graph_diag: dict[str, np.ndarray],
    positions: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    seed: int,
    full_spectral: bool,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]] | None]:
    domain_dir = OUT_DIR / f"benchmark_{profile_name}"
    metadata = json.loads((domain_dir / "metadata.json").read_text(encoding="utf-8"))
    clean_train = np.load(domain_dir / "train_clean_H.npy", mmap_mode="r")
    clean_test = np.load(domain_dir / "test_clean_H.npy", mmap_mode="r")
    scenario = NOISE_SCENARIOS[scenario_name]
    observed_train, noise_report = make_observed_train(clean_train, scenario, seed)
    base, physical_report = physical_predict(
        observed_train,
        positions["train"],
        positions["test"],
        float(metadata["frequency_hz"]),
        float(metadata["subcarrier_spacing_hz"]),
    )
    model = fit_state_and_power_models(observed_train, clean_test, features, seed + 100)
    state, state_scale = apply_state_gate(base, model["nonzero_probability"])
    state_power, state_power_report, state_power_delta = apply_selective_power(
        state,
        model["predicted_log_power"],
        model["nonzero_probability"],
        model["power_spread"],
    )
    test_nonzero = np.asarray(model["test_nonzero"], dtype=bool)
    oracle_state = base.copy()
    oracle_state[~test_nonzero] = 0.0
    oracle_power = scale_to_true_power(base, clean_test)
    methods = {
        "physical_base": base,
        "learned_state_gate": state,
        "learned_state_plus_power": state_power,
        "oracle_state_gate": oracle_state,
        "oracle_total_power": oracle_power,
    }
    leaked_report: dict[str, object] | None = None
    delay_report: dict[str, object] | None = None
    final_power_report: dict[str, object] | None = None
    final_power_delta = np.zeros(len(base), dtype=np.float32)
    if full_spectral:
        leaked_output, leaked_report = leaked_multimarginal_projection(
            state, geometry["test_regimes"], graph_matrix, graph_diag
        )
        delay_output, delay_report = delay_pdp_projection(
            leaked_output,
            observed_train,
            model["train_nonzero"],
            positions["train"],
            positions["test"],
            geometry["train_blocked"],
            geometry["test_blocked"],
            geometry["train_regimes"],
            geometry["test_regimes"],
            graph_matrix,
            graph_diag,
        )
        final_output, final_power_report, final_power_delta = apply_selective_power(
            delay_output,
            model["predicted_log_power"],
            model["nonzero_probability"],
            model["power_spread"],
        )
        oracle_metric = oracle_metric_projection(base, clean_test, preserve_true_power=False)
        oracle_metric_power = scale_to_true_power(oracle_metric, clean_test)
        methods.update(
            {
                "leaked_multimarginal": leaked_output,
                "delay_pdp": delay_output,
                "full_pipeline_power": final_output,
                "oracle_pas_pdp": oracle_metric,
                "oracle_pas_pdp_and_power": oracle_metric_power,
            }
        )
    metrics, point_metrics = evaluate_methods(methods, clean_test, geometry["test_regimes"])
    base_score = float(metrics["physical_base"]["score_zero_aware"])
    for value in metrics.values():
        value["delta_score_vs_physical_base"] = float(value["score_zero_aware"] - base_score)
    run_report = {
        "profile": profile_name,
        "scenario": scenario_name,
        "full_spectral": full_spectral,
        "frequency_hz": metadata["frequency_hz"],
        "noise": noise_report,
        "physical_predictor": physical_report,
        "state_power_models": model["report"],
        "state_gate": {
            "power_scale": summarize(state_scale),
            "zero_scale_count": int(np.sum(state_scale == 0.0)),
        },
        "state_plus_power": state_power_report,
        "leaked_projection": leaked_report,
        "delay_projection": delay_report,
        "final_power": final_power_report,
        "metrics": metrics,
    }
    saved = None
    if profile_name == PRIMARY_PROFILE and scenario_name == PRIMARY_SCENARIO:
        np.savez_compressed(
            POINT_METRICS,
            test_regimes=geometry["test_regimes"].astype("U8"),
            test_nonzero=test_nonzero,
            nonzero_probability=model["nonzero_probability"],
            predicted_log_power=model["predicted_log_power"],
            power_spread=model["power_spread"],
            state_power_scale=state_scale,
            state_power_delta=state_power_delta,
            final_power_delta=final_power_delta,
            **{
                f"{method}_{metric}": values[metric]
                for method, values in point_metrics.items()
                for metric in ("pas_zero_aware", "pdp_zero_aware", "leaked_sh", "ad_support")
            },
        )
        saved = point_metrics
    del methods, observed_train, base, state, state_power
    gc.collect()
    return run_report, saved


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features = v203.load_inputs()
    positions = {
        "train": np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64),
        "test": np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64),
    }
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    geometry = {
        "train_blocked": geom["train_blocked"].astype(np.float32),
        "test_blocked": geom["test_blocked"].astype(np.float32),
    }
    geometry["train_regimes"] = regime_of(geometry["train_blocked"])
    geometry["test_regimes"] = regime_of(geometry["test_blocked"])
    matrix, graph_diag = test_graph_matrix(
        positions["test"], geometry["test_blocked"], geometry["test_regimes"]
    )
    run_specs = [
        (PRIMARY_PROFILE, "relative_25db", False),
        (PRIMARY_PROFILE, PRIMARY_SCENARIO, True),
        (DOMAIN_PROFILES[1], PRIMARY_SCENARIO, False),
        (DOMAIN_PROFILES[2], PRIMARY_SCENARIO, False),
    ]
    runs = []
    for run_index, (profile, scenario, full_spectral) in enumerate(run_specs):
        print(f"starting benchmark {profile}/{scenario}", flush=True)
        run, _ = evaluate_run(
            profile,
            scenario,
            features,
            matrix,
            graph_diag,
            positions,
            geometry,
            seed=205_000 + run_index * 1000,
            full_spectral=full_spectral,
        )
        runs.append(run)
        REPORT.write_text(
            json.dumps(
                {
                    "method": VERSION,
                    "status": "analysis_in_progress",
                    "pilot_clean_scores": PILOT_CLEAN_SCORES,
                    "completed_runs": runs,
                    "remaining_runs": len(run_specs) - len(runs),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "profile": profile,
                    "scenario": scenario,
                    "scores": {name: value["score_zero_aware"] for name, value in run["metrics"].items()},
                },
                indent=2,
            ),
            flush=True,
        )

    method_names = sorted(set.intersection(*(set(run["metrics"]) for run in runs)))
    robustness = {}
    for method in method_names:
        deltas = np.asarray(
            [run["metrics"][method]["delta_score_vs_physical_base"] for run in runs],
            dtype=np.float64,
        )
        if method in PILOT_CLEAN_SCORES:
            pilot_delta = PILOT_CLEAN_SCORES[method] - PILOT_CLEAN_SCORES["physical_base"]
            deltas = np.concatenate([np.asarray([pilot_delta], dtype=np.float64), deltas])
        robustness[method] = {
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "positive_runs": int(np.sum(deltas > 0.0)),
            "run_count": int(len(deltas)),
            "deltas": deltas.tolist(),
        }
    transferable = sorted(
        (
            (value["min_delta"], value["mean_delta"], name)
            for name, value in robustness.items()
            if not name.startswith("oracle") and name != "physical_base"
        ),
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A method family is suitable for the next online candidate only if it improves known-truth score "
            "across calibrated carrier domains and plausible train-noise mechanisms, rather than in one toy split."
        ),
        "benchmark_generation": PATHS.relative(
            REPORT_DIR / "v205_calibrated_sionna_benchmark_generate.json"
        ),
        "pilot_clean": {
            "profile": PRIMARY_PROFILE,
            "scenario": "clean",
            "status": "completed_before_runtime-pruning; exact aggregate scores retained",
            "scores": PILOT_CLEAN_SCORES,
        },
        "run_specs": [
            {"profile": profile, "scenario": scenario, "full_spectral": full_spectral}
            for profile, scenario, full_spectral in run_specs
        ],
        "graph": {
            "nnz": int(matrix.nnz),
            "neighbor_agreement": summarize(graph_diag["neighbor_agreement"]),
        },
        "runs": runs,
        "robustness": robustness,
        "transferable_nonoracle_ranking": [
            {"method": name, "min_delta": min_delta, "mean_delta": mean_delta}
            for min_delta, mean_delta, name in transferable
        ],
        "outputs": {
            "report": PATHS.relative(REPORT),
            "primary_hybrid_point_metrics": PATHS.relative(POINT_METRICS),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": PATHS.relative(REPORT), "ranking": report["transferable_nonoracle_ranking"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
