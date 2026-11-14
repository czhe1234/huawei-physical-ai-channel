#!/usr/bin/env python3
"""Test latent path occupancy, repeated detection, and censored path energy.

This is an official-only diagnostic.  Geometry candidates are observed through
several complementary channel marginals.  A path can exist without being detected in
every marginal, so unmatched candidates remain latent instead of becoming hard
negatives.  The spatial predictor is deliberately kept as the same patch-wise
KNN used by the hard-label baseline; this isolates the observation model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit
from scipy.spatial import cKDTree
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import average_precision_score

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v328_patch_existence_support_field as v328
import analyze_v402_visible_path_residual_factorization as v402
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 403
VERSION = "v403_latent_occupancy_detection_paths"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
FIGURE_DIR = REPORT_DIR / "版本四零三潜在路径检测图"
REPORT = REPORT_DIR / f"{VERSION}.json"
EVIDENCE_CACHE = OUT_DIR / "v403_candidate_multiview_evidence.npz"
OOF_CACHE = OUT_DIR / "v403_latent_occupancy_oof.npz"
FIGURE = FIGURE_DIR / "潜在路径检测与条件能量结果.png"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

VIEW_FAMILIES = ((0, 2), (2, 6), (6, 14))
VIEW_NAMES = (
    "极化一",
    "极化二",
    "终端端口一",
    "终端端口二",
    "终端端口三",
    "终端端口四",
    "垂直阵元一",
    "垂直阵元二",
    "垂直阵元三",
    "垂直阵元四",
    "垂直阵元五",
    "垂直阵元六",
    "垂直阵元七",
    "垂直阵元八",
)

K_SUPPORT = 16
K_ENERGY = 8
DISTANCE_FLOOR = 0.25
DETECTION_POSITIVE_QUANTILE = 0.20
EM_ITERATIONS = 6
EM_DAMPING = 0.50
PRIOR_FLOOR_SCALE = 0.50
MAX_PATHS = 8
MAX_PATH_MASS = 0.95
MIN_BREAKTHROUGH_DELTA = 0.01
MIN_FOLD_WINS = 4
MIN_AP_LIFT = 0.02
MIN_ENERGY_RMSE_REDUCTION = 0.10


@dataclass(frozen=True)
class DetectionCalibration:
    threshold: np.ndarray
    positive_rate: np.ndarray
    false_rate: np.ndarray
    prevalence: float


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
            font_manager.fontManager.addfont(path)
            selected = font_manager.FontProperties(fname=path).get_name()
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )


def calibration_parameters() -> tuple[float, float]:
    report = json.loads(v304.REPORT.read_text(encoding="utf-8"))
    calibration = report["los_calibration"]
    return float(calibration["slope_bins_per_meter"]), float(
        calibration["intercept_bins"]
    )


def candidate_centers(
    candidate: dict[str, np.ndarray], slope: float, intercept: float
) -> tuple[np.ndarray, np.ndarray]:
    delay = intercept + slope * candidate["path_length"].astype(np.float32)
    s_center = np.remainder(-delay, 192.0).astype(np.float32)
    h_center = v305.expected_h_bin(candidate["reflection_point"]).astype(np.float32)
    return s_center, h_center


def _sample_local_map(
    maps: np.ndarray, s_center: np.ndarray, h_center: np.ndarray
) -> np.ndarray:
    """Sample a compact periodic 3x3 support window for each candidate."""
    count = len(s_center)
    output = np.zeros((count, len(maps)), dtype=np.float32)
    s0 = np.floor(s_center).astype(np.int64)
    h0 = np.floor(h_center).astype(np.int64)
    for ds in (-1, 0, 1):
        s_index = np.remainder(s0 + ds, 192)
        s_distance = np.abs((s0 + ds).astype(np.float32) - s_center)
        s_weight = np.exp(-0.5 * np.square(s_distance / 0.85))
        for dh in (-1, 0, 1):
            h_index = np.remainder(h0 + dh, 16)
            h_distance = np.abs((h0 + dh).astype(np.float32) - h_center)
            h_weight = np.exp(-0.5 * np.square(h_distance / 0.85))
            output += (
                maps[:, s_index, h_index].T
                * (s_weight * h_weight)[:, None]
            ).astype(np.float32)
    return output


def build_multiview_evidence(
    targets: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    slope: float,
    intercept: float,
) -> dict[str, np.ndarray]:
    if EVIDENCE_CACHE.is_file():
        cached = np.load(EVIDENCE_CACHE)
        return {key: cached[key] for key in cached.files}

    valid = candidate["score"] > 0.0
    s_center, h_center = candidate_centers(candidate, slope, intercept)
    evidence = np.zeros((*valid.shape, len(VIEW_NAMES)), dtype=np.float32)
    local_mass = np.zeros(valid.shape, dtype=np.float32)
    for point_id in range(len(valid)):
        active = np.flatnonzero(valid[point_id])
        if len(active) == 0:
            continue
        maps = np.concatenate(
            [
                targets["psh"][point_id],
                np.moveaxis(targets["shue"][point_id], -1, 0),
                np.moveaxis(targets["shv"][point_id], -1, 0),
            ],
            axis=0,
        ).astype(np.float32)
        view_total = maps.sum(axis=(1, 2), keepdims=True, dtype=np.float64)
        maps /= np.maximum(view_total, 1e-30).astype(np.float32)
        evidence[point_id, active] = _sample_local_map(
            maps, s_center[point_id, active], h_center[point_id, active]
        )
        local_mass[point_id, active] = _sample_local_map(
            targets["sh"][point_id][None],
            s_center[point_id, active],
            h_center[point_id, active],
        )[:, 0]
        if (point_id + 1) % 250 == 0:
            print(f"v403 candidate evidence {point_id + 1}/{len(valid)}", flush=True)

    log_evidence = np.log(np.maximum(evidence, 1e-10)).astype(np.float32)
    np.savez_compressed(
        EVIDENCE_CACHE,
        log_evidence=log_evidence,
        local_mass=local_mass,
        view_names=np.asarray(VIEW_NAMES),
    )
    return {
        "log_evidence": log_evidence,
        "local_mass": local_mass,
        "view_names": np.asarray(VIEW_NAMES),
    }


def patch_exposures(
    candidate: dict[str, np.ndarray], valid: np.ndarray, point_ids: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return v328.patch_rows(candidate, valid, np.asarray(point_ids, dtype=np.int64))


def patch_predict(
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    source: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    xy: np.ndarray,
    *,
    log_average: bool = False,
    source_weight: np.ndarray | None = None,
) -> np.ndarray:
    output = np.zeros((len(query_ids), valid.shape[1]), dtype=np.float32)
    query_lookup = {int(point): row for row, point in enumerate(query_ids)}
    fit_patch = patch_exposures(candidate, valid, fit_ids)
    query_patch = patch_exposures(candidate, valid, query_ids)
    for patch, (query_point, query_candidate) in query_patch.items():
        if patch not in fit_patch:
            continue
        fit_point, fit_candidate = fit_patch[patch]
        if source_weight is not None:
            usable = source_weight[fit_point, fit_candidate] > 1e-6
            fit_point = fit_point[usable]
            fit_candidate = fit_candidate[usable]
            if len(fit_point) == 0:
                continue
        tree = cKDTree(xy[fit_point])
        count = min(K_ENERGY if log_average else K_SUPPORT, len(fit_point))
        distance, local = tree.query(xy[query_point], k=count)
        if count == 1:
            distance = distance[:, None]
            local = local[:, None]
        weight = 1.0 / np.maximum(distance, DISTANCE_FLOOR) ** 2
        value = source[fit_point[local], fit_candidate[local]].astype(np.float64)
        if source_weight is not None:
            confidence = source_weight[
                fit_point[local], fit_candidate[local]
            ].astype(np.float64)
            weight *= confidence
        denominator = np.sum(weight, axis=1)
        if log_average:
            prediction = np.zeros(len(query_point), dtype=np.float64)
            usable = denominator > 1e-30
            prediction[usable] = np.exp(
                np.sum(
                    weight[usable] * np.log(np.maximum(value[usable], 1e-7)),
                    axis=1,
                )
                / denominator[usable]
            )
        else:
            prediction = np.sum(weight * value, axis=1) / np.maximum(
                denominator, 1e-30
            )
        rows = np.asarray([query_lookup[int(point)] for point in query_point])
        output[rows, query_candidate] = prediction.astype(np.float32)
    return output


def crossfit_patch_predict(
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    source: np.ndarray,
    fit_ids: np.ndarray,
    folds: list[np.ndarray],
    xy: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(fit_ids), valid.shape[1]), dtype=np.float32)
    lookup = {int(point): row for row, point in enumerate(fit_ids)}
    covered: list[int] = []
    for fold_ids in folds:
        query = np.intersect1d(fit_ids, fold_ids)
        if len(query) == 0:
            continue
        bank = np.setdiff1d(fit_ids, query)
        prediction = patch_predict(
            candidate, valid, source, bank, query, xy, log_average=False
        )
        rows = np.asarray([lookup[int(point)] for point in query], dtype=np.int64)
        output[rows] = prediction
        covered.extend(query.tolist())
    if len(np.unique(covered)) != len(fit_ids):
        raise AssertionError("cross-fit patch prediction did not cover fit IDs")
    return output


def calibrate_detections(
    fit_ids: np.ndarray,
    valid: np.ndarray,
    hard_mass: np.ndarray,
    log_evidence: np.ndarray,
    prior: np.ndarray,
) -> DetectionCalibration:
    active = valid[fit_ids]
    mass = hard_mass[fit_ids][active]
    evidence = log_evidence[fit_ids][active]
    prior_value = prior[active]
    positive = mass > 0.0
    if np.sum(positive) < 50:
        raise RuntimeError("not enough confirmed paths for detection calibration")
    threshold = np.quantile(
        evidence[positive], DETECTION_POSITIVE_QUANTILE, axis=0
    ).astype(np.float32)
    detection = evidence >= threshold[None]
    conservative_negative = (~positive) & (prior_value <= 0.01)
    if np.sum(conservative_negative) < 100:
        conservative_negative = ~positive
    positive_rate = np.mean(detection[positive], axis=0)
    false_rate = np.mean(detection[conservative_negative], axis=0)
    positive_rate = np.clip(positive_rate, 0.55, 0.995).astype(np.float32)
    false_rate = np.minimum(false_rate, positive_rate - 0.02)
    false_rate = np.clip(false_rate, 0.005, 0.95).astype(np.float32)
    prevalence = float(np.mean(positive))
    return DetectionCalibration(threshold, positive_rate, false_rate, prevalence)


def detection_log_likelihood_ratio(
    evidence: np.ndarray, calibration: DetectionCalibration
) -> np.ndarray:
    detected = evidence >= calibration.threshold[None]
    p = calibration.positive_rate[None]
    f = calibration.false_rate[None]
    raw = np.where(
        detected,
        np.log(p) - np.log(f),
        np.log1p(-p) - np.log1p(-f),
    )
    family = [np.mean(raw[:, start:end], axis=1) for start, end in VIEW_FAMILIES]
    return np.mean(np.stack(family, axis=1), axis=1).astype(np.float32)


def latent_fit_occupancy(
    fit_ids: np.ndarray,
    folds: list[np.ndarray],
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    hard_mass: np.ndarray,
    log_evidence: np.ndarray,
    xy: np.ndarray,
) -> tuple[np.ndarray, DetectionCalibration, dict[str, object]]:
    hard_binary = (hard_mass > 0.0).astype(np.float32)
    hard_prior = crossfit_patch_predict(
        candidate, valid, hard_binary, fit_ids, folds, xy
    )
    fit_valid = valid[fit_ids]
    fit_hard = hard_binary[fit_ids]
    q = np.zeros_like(fit_hard, dtype=np.float32)
    prevalence = float(np.mean(fit_hard[fit_valid]))
    q[fit_valid] = np.clip(
        hard_prior[fit_valid] + PRIOR_FLOOR_SCALE * prevalence, 0.002, 0.995
    )
    q[fit_hard > 0.0] = 1.0
    calibration = calibrate_detections(
        fit_ids, valid, hard_mass, log_evidence, hard_prior
    )

    for _ in range(EM_ITERATIONS):
        source = np.zeros_like(valid, dtype=np.float32)
        source[fit_ids] = q
        spatial = crossfit_patch_predict(
            candidate, valid, source, fit_ids, folds, xy
        )
        prior = np.clip(
            spatial[fit_valid] + PRIOR_FLOOR_SCALE * calibration.prevalence,
            0.002,
            0.995,
        )
        evidence = log_evidence[fit_ids][fit_valid]
        likelihood = detection_log_likelihood_ratio(evidence, calibration)
        updated = expit(logit(prior) + likelihood).astype(np.float32)
        current = q[fit_valid]
        q[fit_valid] = EM_DAMPING * current + (1.0 - EM_DAMPING) * updated
        q[fit_hard > 0.0] = 1.0

    source = np.zeros_like(valid, dtype=np.float32)
    source[fit_ids] = q
    latent_unmatched = q[fit_valid & (fit_hard == 0.0)]
    return source, calibration, {
        "confirmed_positive_count": int(np.sum(fit_hard > 0.0)),
        "unmatched_posterior_mean": float(np.mean(latent_unmatched)),
        "unmatched_posterior_p90": float(np.quantile(latent_unmatched, 0.90)),
        "unmatched_above_0_5": int(np.sum(latent_unmatched >= 0.5)),
        "positive_rate": calibration.positive_rate.tolist(),
        "false_rate": calibration.false_rate.tolist(),
    }


def calibrate_pseudo_energy(
    fit_ids: np.ndarray,
    hard_mass: np.ndarray,
    local_mass: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    positive = hard_mass[fit_ids] > 0.0
    x = np.log(np.maximum(local_mass[fit_ids][positive], 1e-7))[:, None]
    y = np.log(np.maximum(hard_mass[fit_ids][positive], 1e-7))
    model = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=300).fit(x, y)
    prediction = np.exp(
        model.predict(np.log(np.maximum(local_mass, 1e-7)).reshape(-1, 1))
    ).reshape(local_mass.shape)
    prediction = np.clip(prediction, 1e-7, 0.75).astype(np.float32)
    prediction[hard_mass > 0.0] = hard_mass[hard_mass > 0.0]
    fitted = model.predict(x)
    return prediction, {
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "confirmed_log_rmse": float(np.sqrt(np.mean(np.square(fitted - y)))),
    }


def precision_recall(score: np.ndarray, mass: np.ndarray) -> dict[str, float]:
    binary = mass > 0.0
    ap = float(average_precision_score(binary, score))
    selected = v328.precision_recall_at_target(score, mass, v328.PRECISION_TARGET)
    return {"average_precision": ap, **selected}


def energy_metrics(
    prediction: np.ndarray, truth: np.ndarray, active: np.ndarray
) -> dict[str, float]:
    pred = np.maximum(prediction[active].astype(np.float64), 1e-7)
    target = np.maximum(truth[active].astype(np.float64), 1e-7)
    log_error = np.log(pred) - np.log(target)
    return {
        "count": int(len(pred)),
        "log_rmse": float(np.sqrt(np.mean(np.square(log_error)))),
        "log_mae": float(np.mean(np.abs(log_error))),
        "mass_mae": float(np.mean(np.abs(pred - target))),
    }


def render_candidate_field(
    point_ids: np.ndarray,
    candidate: dict[str, np.ndarray],
    expected_mass: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    output = np.zeros((len(point_ids), 192, 16), dtype=np.float32)
    _s_center, h_center = candidate_centers(candidate, slope, intercept)
    delay = intercept + slope * candidate["path_length"]
    for row, point_id in enumerate(point_ids):
        active = np.flatnonzero(
            (candidate["score"][point_id] > 0.0)
            & (expected_mass[row] > 1e-6)
        )
        if len(active) == 0:
            continue
        active = active[np.argsort(expected_mass[row, active])[::-1][:MAX_PATHS]]
        mass = expected_mass[row, active].astype(np.float64)
        total = float(np.sum(mass))
        if total > MAX_PATH_MASS:
            mass *= MAX_PATH_MASS / total
        for candidate_id, value in zip(active, mass):
            output[row] += v402.render_path(
                float(delay[point_id, candidate_id]),
                v402.exact_h_atom(float(h_center[point_id, candidate_id])),
                float(value),
            )
    return output


def factorized_prediction(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    target: np.ndarray,
    candidate: dict[str, np.ndarray],
    fit_expected: np.ndarray,
    query_expected: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_weight: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, dict[str, float]]:
    fit_path = render_candidate_field(
        fit_ids, candidate, fit_expected, slope, intercept
    )
    fit_visible = np.minimum(fit_path, target[fit_ids]).astype(np.float32)
    fit_residual = np.maximum(target[fit_ids] - fit_visible, 0.0).astype(np.float32)
    local_lookup = np.full(len(target), -1, dtype=np.int64)
    local_lookup[fit_ids] = np.arange(len(fit_ids))
    residual_anchor = np.sum(
        neighbor_weight[:, :, None, None]
        * fit_residual[local_lookup[neighbor_ids]],
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    query_path = render_candidate_field(
        query_ids, candidate, query_expected, slope, intercept
    )
    prediction = v402.normalize(residual_anchor + query_path)
    return prediction, {
        "fit_visible_mass_mean": float(np.mean(fit_visible.sum(axis=(1, 2)))),
        "query_path_mass_mean": float(np.mean(query_path.sum(axis=(1, 2)))),
        "query_active_fraction": float(
            np.mean(np.sum(query_expected > 1e-6, axis=1) > 0)
        ),
    }


def fold_experiment(
    fold: int,
    query_ids: np.ndarray,
    valid_ids: np.ndarray,
    folds: list[np.ndarray],
    positions: np.ndarray,
    blocked: np.ndarray,
    target: np.ndarray,
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    hard_mass: np.ndarray,
    evidence: dict[str, np.ndarray],
    slope: float,
    intercept: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    xy = positions[:, :2]
    hard_binary = (hard_mass > 0.0).astype(np.float32)
    hard_query = patch_predict(
        candidate, valid, hard_binary, fit_ids, query_ids, xy
    )
    latent_source, calibration, latent_diagnostic = latent_fit_occupancy(
        fit_ids,
        folds,
        candidate,
        valid,
        hard_mass,
        evidence["log_evidence"],
        xy,
    )
    latent_query = patch_predict(
        candidate, valid, latent_source, fit_ids, query_ids, xy
    )

    query_valid = valid[query_ids]
    truth_mass = hard_mass[query_ids]
    hard_existence = precision_recall(
        hard_query[query_valid], truth_mass[query_valid]
    )
    latent_existence = precision_recall(
        latent_query[query_valid], truth_mass[query_valid]
    )

    pseudo_energy, energy_calibration = calibrate_pseudo_energy(
        fit_ids, hard_mass, evidence["local_mass"]
    )
    hard_energy = patch_predict(
        candidate,
        valid,
        hard_mass,
        fit_ids,
        query_ids,
        xy,
        log_average=True,
        source_weight=hard_binary,
    )
    latent_energy = patch_predict(
        candidate,
        valid,
        pseudo_energy,
        fit_ids,
        query_ids,
        xy,
        log_average=True,
        source_weight=latent_source,
    )
    positive_query = truth_mass > 0.0
    hard_energy_metric = energy_metrics(hard_energy, truth_mass, positive_query)
    latent_energy_metric = energy_metrics(latent_energy, truth_mass, positive_query)

    neighbor_ids, neighbor_weight = v402.neighbors(query_ids, fit_ids, xy)
    baseline = v402.knn_average(target, neighbor_ids, neighbor_weight)
    hard_fit_expected = hard_mass[fit_ids]
    hard_query_expected = hard_query * hard_energy
    latent_fit_expected = (
        latent_source[fit_ids] * pseudo_energy[fit_ids]
    ).astype(np.float32)
    latent_query_expected = (latent_query * latent_energy).astype(np.float32)
    hard_prediction, hard_render = factorized_prediction(
        fit_ids,
        query_ids,
        target,
        candidate,
        hard_fit_expected,
        hard_query_expected,
        neighbor_ids,
        neighbor_weight,
        slope,
        intercept,
    )
    latent_prediction, latent_render = factorized_prediction(
        fit_ids,
        query_ids,
        target,
        candidate,
        latent_fit_expected,
        latent_query_expected,
        neighbor_ids,
        neighbor_weight,
        slope,
        intercept,
    )
    hybrid_query_expected = (hard_query * latent_energy).astype(np.float32)
    hybrid_prediction, hybrid_render = factorized_prediction(
        fit_ids,
        query_ids,
        target,
        candidate,
        hard_fit_expected,
        hybrid_query_expected,
        neighbor_ids,
        neighbor_weight,
        slope,
        intercept,
    )
    truth = target[query_ids]
    base_values = v402.point_metrics(baseline, truth)
    hard_values = v402.point_metrics(hard_prediction, truth)
    hybrid_values = v402.point_metrics(hybrid_prediction, truth)
    latent_values = v402.point_metrics(latent_prediction, truth)
    regimes = regime_of(blocked[query_ids]).astype(str)
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "detection_calibration": {
            "threshold": calibration.threshold.tolist(),
            "positive_rate": calibration.positive_rate.tolist(),
            "false_rate": calibration.false_rate.tolist(),
            "prevalence": calibration.prevalence,
        },
        "latent_diagnostic": latent_diagnostic,
        "existence": {
            "hard": hard_existence,
            "latent": latent_existence,
            "ap_lift": latent_existence["average_precision"]
            - hard_existence["average_precision"],
            "recall_lift_at_70pct_precision": latent_existence["count_recall"]
            - hard_existence["count_recall"],
        },
        "energy": {
            "calibration": energy_calibration,
            "hard": hard_energy_metric,
            "latent": latent_energy_metric,
            "relative_log_rmse_reduction": (
                hard_energy_metric["log_rmse"] - latent_energy_metric["log_rmse"]
            )
            / max(hard_energy_metric["log_rmse"], 1e-30),
        },
        "spectrum": {
            "baseline": v402.metric_summary(baseline, truth),
            "hard_factorization": v402.metric_summary(hard_prediction, truth),
            "hard_existence_censored_energy": v402.metric_summary(
                hybrid_prediction, truth
            ),
            "latent_factorization": v402.metric_summary(latent_prediction, truth),
            "hard_weighted_delta": float(
                np.mean(hard_values["weighted"] - base_values["weighted"])
            ),
            "latent_weighted_delta": float(
                np.mean(latent_values["weighted"] - base_values["weighted"])
            ),
            "censored_energy_weighted_delta": float(
                np.mean(hybrid_values["weighted"] - base_values["weighted"])
            ),
            "censored_energy_minus_hard": float(
                np.mean(hybrid_values["weighted"] - hard_values["weighted"])
            ),
            "latent_minus_hard": float(
                np.mean(latent_values["weighted"] - hard_values["weighted"])
            ),
            "latent_by_regime": {
                regime: float(
                    np.mean(
                        latent_values["weighted"][regimes == regime]
                        - base_values["weighted"][regimes == regime]
                    )
                )
                for regime in ("clear", "light", "medium", "heavy")
                if np.any(regimes == regime)
            },
        },
        "render": {
            "hard": hard_render,
            "hard_existence_censored_energy": hybrid_render,
            "latent": latent_render,
        },
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "query_h_used_for_prediction": False,
        },
    }
    arrays = {
        "ids": query_ids,
        "baseline": baseline,
        "hard": hard_prediction,
        "hybrid": hybrid_prediction,
        "latent": latent_prediction,
        "truth": truth,
        "hard_probability": hard_query,
        "latent_probability": latent_query,
    }
    return row, arrays


def aggregate_report(rows: list[dict[str, object]]) -> dict[str, object]:
    ap_lift = np.asarray([row["existence"]["ap_lift"] for row in rows])
    recall_lift = np.asarray(
        [row["existence"]["recall_lift_at_70pct_precision"] for row in rows]
    )
    energy_reduction = np.asarray(
        [row["energy"]["relative_log_rmse_reduction"] for row in rows]
    )
    hard_delta = np.asarray(
        [row["spectrum"]["hard_weighted_delta"] for row in rows]
    )
    latent_delta = np.asarray(
        [row["spectrum"]["latent_weighted_delta"] for row in rows]
    )
    censored_energy_delta = np.asarray(
        [row["spectrum"]["censored_energy_weighted_delta"] for row in rows]
    )
    censored_energy_minus_hard = np.asarray(
        [row["spectrum"]["censored_energy_minus_hard"] for row in rows]
    )
    latent_minus_hard = np.asarray(
        [row["spectrum"]["latent_minus_hard"] for row in rows]
    )
    passed = bool(
        np.mean(ap_lift) >= MIN_AP_LIFT
        and np.mean(energy_reduction) >= MIN_ENERGY_RMSE_REDUCTION
        and np.mean(latent_delta) >= MIN_BREAKTHROUGH_DELTA
        and np.sum(latent_delta > 0.0) >= MIN_FOLD_WINS
    )
    return {
        "existence_ap_lift": summarize(ap_lift),
        "existence_recall_lift_at_70pct_precision": summarize(recall_lift),
        "conditional_energy_log_rmse_reduction": summarize(energy_reduction),
        "hard_factorization_weighted_delta": summarize(hard_delta),
        "hard_existence_censored_energy_weighted_delta": summarize(
            censored_energy_delta
        ),
        "censored_energy_minus_hard_weighted": summarize(
            censored_energy_minus_hard
        ),
        "latent_factorization_weighted_delta": summarize(latent_delta),
        "latent_minus_hard_weighted": summarize(latent_minus_hard),
        "fold_wins_vs_baseline": int(np.sum(latent_delta > 0.0)),
        "passed": passed,
    }


def plot_report(report: dict[str, object]) -> None:
    configure_chinese_font()
    rows = report["fold_rows"]
    fold = np.arange(len(rows))
    hard_ap = [row["existence"]["hard"]["average_precision"] for row in rows]
    latent_ap = [row["existence"]["latent"]["average_precision"] for row in rows]
    hard_rmse = [row["energy"]["hard"]["log_rmse"] for row in rows]
    latent_rmse = [row["energy"]["latent"]["log_rmse"] for row in rows]
    hard_delta = [row["spectrum"]["hard_weighted_delta"] for row in rows]
    hybrid_delta = [
        row["spectrum"]["censored_energy_weighted_delta"] for row in rows
    ]
    latent_delta = [row["spectrum"]["latent_weighted_delta"] for row in rows]

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    width = 0.36
    axes[0].bar(fold - width / 2, hard_ap, width, label="硬标签")
    axes[0].bar(fold + width / 2, latent_ap, width, label="潜在存在")
    axes[0].set_title("路径存在平均精确率")
    axes[0].set_xlabel("空间折")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(fold - width / 2, hard_rmse, width, label="仅正样本")
    axes[1].bar(fold + width / 2, latent_rmse, width, label="截断能量")
    axes[1].set_title("条件能量对数误差，越低越好")
    axes[1].set_xlabel("空间折")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].plot(fold, hard_delta, "o-", label="硬存在+正样本能量")
    axes[2].plot(fold, hybrid_delta, "s-", label="硬存在+截断能量")
    axes[2].plot(fold, latent_delta, "^-", label="潜在存在+截断能量")
    axes[2].axhline(0.0, color="#333333", linewidth=0.8)
    axes[2].axhline(
        MIN_BREAKTHROUGH_DELTA, color="#2d7f4f", linestyle="--", linewidth=1.0
    )
    axes[2].set_title("相对普通全谱近邻的联合谱增益")
    axes[2].set_xlabel("空间折")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("v403 路径真实存在与观测检测分离实验", fontsize=15)
    figure.savefig(FIGURE, dpi=190, facecolor="white")
    plt.close(figure)


def main() -> None:
    required = (
        TARGET_CACHE,
        GEOMETRY_CACHE,
        v297.COMPONENT_CACHE,
        v304.REPORT,
        v305.MATCH_CACHE,
        v306.TRAIN_CANDIDATES,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    target_data = np.load(TARGET_CACHE)
    targets = {key: target_data[key] for key in target_data.files}
    target = v402.normalize(targets["sh"].astype(np.float32))
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    blocked = np.load(GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    candidate_data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: candidate_data[key] for key in candidate_data.files}
    valid, hard_mass = v328.build_labels(candidate)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, positions[:, :2])
    slope, intercept = calibration_parameters()
    evidence = build_multiview_evidence(targets, candidate, slope, intercept)

    fold_rows: list[dict[str, object]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for fold, query_ids in enumerate(folds):
        row, part = fold_experiment(
            fold,
            query_ids,
            valid_ids,
            folds,
            positions,
            blocked,
            target,
            candidate,
            valid,
            hard_mass,
            evidence,
            slope,
            intercept,
        )
        fold_rows.append(row)
        arrays.append(part)
        print(
            f"v403 fold={fold} AP_lift={row['existence']['ap_lift']:+.5f} "
            f"energy={row['energy']['relative_log_rmse_reduction']:+.3%} "
            f"joint={row['spectrum']['latent_weighted_delta']:+.6f}",
            flush=True,
        )

    aggregate = aggregate_report(fold_rows)
    ids = np.concatenate([part["ids"] for part in arrays])
    if len(np.unique(ids)) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover of valid official train points")
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [np.full(len(part["ids"]), fold, dtype=np.int16) for fold, part in enumerate(arrays)]
        ),
        baseline=np.concatenate([part["baseline"] for part in arrays]),
        hard=np.concatenate([part["hard"] for part in arrays]),
        hybrid=np.concatenate([part["hybrid"] for part in arrays]),
        latent=np.concatenate([part["latent"] for part in arrays]),
        truth=np.concatenate([part["truth"] for part in arrays]),
        hard_probability=np.concatenate([part["hard_probability"] for part in arrays]),
        latent_probability=np.concatenate(
            [part["latent_probability"] for part in arrays]
        ),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Separating latent path occupancy from repeated-view detection, and treating "
            "unmatched path energy as censored rather than zero, improves source-specific "
            "existence, conditional energy, and visible-path factorization."
        ),
        "compliance": {
            "uses": [
                "official train H-derived joint spectra and complementary marginals",
                "official train coordinates",
                "official Round1 point-cloud reflection candidates",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic target or path coefficient",
                "online scores as labels",
            ],
        },
        "isolation": (
            "Hard and latent methods use the same patch-wise KNN spatial predictor; "
            "only the observation/label model and censored energy bank differ."
        ),
        "parameters": {
            "view_names": list(VIEW_NAMES),
            "view_families": [list(value) for value in VIEW_FAMILIES],
            "positive_detection_quantile": DETECTION_POSITIVE_QUANTILE,
            "em_iterations": EM_ITERATIONS,
            "em_damping": EM_DAMPING,
            "support_k": K_SUPPORT,
            "energy_k": K_ENERGY,
            "maximum_rendered_paths": MAX_PATHS,
        },
        "label_summary": {
            "valid_candidate_pairs": int(np.sum(valid)),
            "confirmed_positive_pairs": int(np.sum(hard_mass > 0.0)),
            "confirmed_positive_mass": float(np.sum(hard_mass, dtype=np.float64)),
        },
        "fold_rows": fold_rows,
        "aggregate": aggregate,
        "decision_rule": {
            "existence_ap_lift": f">={MIN_AP_LIFT}",
            "conditional_energy_log_rmse_reduction": f">={MIN_ENERGY_RMSE_REDUCTION}",
            "joint_weighted_delta": f">={MIN_BREAKTHROUGH_DELTA}",
            "joint_fold_wins": f">={MIN_FOLD_WINS}/5",
            "passed": bool(aggregate["passed"]),
        },
        "outputs": {
            "evidence": PATHS.relative(EVIDENCE_CACHE),
            "oof": PATHS.relative(OOF_CACHE),
            "figure": PATHS.relative(FIGURE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_report(report)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
