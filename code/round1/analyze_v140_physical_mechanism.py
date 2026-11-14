#!/usr/bin/env python3
"""Audit the physical mechanism behind v140's H-angle support correction."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from generate_v140_balanced_hminus2_overdrive import build_risk, support_features
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(140)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v140_physical_mechanism_analysis.json"
POINT_CSV = REPORT_DIR / "v140_physical_mechanism_points.csv"

MH = 16
RANDOM_STATE = 14091
REGIMES = ("clear", "light", "medium", "heavy")


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = mask & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or x[valid].std() == 0 or y[valid].std() == 0:
        return {"count": int(valid.sum()), "pearson": 0.0, "spearman": 0.0}
    return {
        "count": int(valid.sum()),
        "pearson": float(pearsonr(x[valid], y[valid]).statistic),
        "spearman": float(spearmanr(x[valid], y[valid]).statistic),
    }


def normalize_support(values: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    x = np.maximum(x, floor)
    return (x / np.maximum(x.sum(axis=(1, 2, 3), keepdims=True), TINY)).astype(np.float32)


def normalize_truth(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return (x / np.maximum(x.sum(axis=(1, 2, 3), keepdims=True), TINY)).astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(len(a), -1).astype(np.float64)
    bb = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(aa * bb, axis=1)
        / np.maximum(np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1)), 1e-30)
    ).astype(np.float32)


def marginal_h(values: np.ndarray) -> np.ndarray:
    return values.sum(axis=(2, 3)).astype(np.float32)


def distribution_entropy(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
    return (-(x * np.log(np.maximum(x, 1e-30))).sum(axis=1) / math.log(x.shape[1])).astype(np.float32)


def circular_center(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
    angle = 2.0 * np.pi * np.arange(x.shape[1], dtype=np.float64) / x.shape[1]
    moment = x @ np.exp(1j * angle)
    center = (np.angle(moment) % (2.0 * np.pi)) * x.shape[1] / (2.0 * np.pi)
    return center.astype(np.float32), np.abs(moment).astype(np.float32)


def signed_circular_delta(a: np.ndarray, b: np.ndarray, period: float = MH) -> np.ndarray:
    return ((np.asarray(a) - np.asarray(b) + period / 2.0) % period - period / 2.0).astype(np.float32)


def circular_abs_delta(a: np.ndarray, b: np.ndarray, period: float = MH) -> np.ndarray:
    return np.abs(signed_circular_delta(a, b, period)).astype(np.float32)


def transform_batch(
    source: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    mode: str,
    temperature_c: float = 0.0,
    risk: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "identity":
        return np.array(source, copy=True)
    if mode == "temperature":
        if risk is None:
            raise ValueError("temperature mode requires risk")
        exponent = (1.0 + temperature_c * risk)[:, None, None, None]
        out = np.power(np.maximum(source, 1e-12), exponent).astype(np.float32)
    elif mode == "linear":
        out = (1.0 - w1[:, None, None, None]) * source + w1[:, None, None, None] * np.roll(source, -1, axis=1)
        out = (1.0 - w2[:, None, None, None]) * out + w2[:, None, None, None] * np.roll(source, -2, axis=1)
    elif mode == "log":
        log_source = np.log(np.maximum(source, 1e-12))
        out = np.exp(
            (1.0 - w1[:, None, None, None]) * log_source
            + w1[:, None, None, None] * np.log(np.maximum(np.roll(source, -1, axis=1), 1e-12))
        ).astype(np.float32)
        out = np.exp(
            (1.0 - w2[:, None, None, None]) * np.log(np.maximum(out, 1e-12))
            + w2[:, None, None, None] * np.log(np.maximum(np.roll(source, -2, axis=1), 1e-12))
        ).astype(np.float32)
    else:
        raise ValueError(mode)
    out /= np.maximum(out.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return out.astype(np.float32)


def evaluate_operator(
    source: np.ndarray,
    truth: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    mode: str,
    temperature_c: float = 0.0,
    risk: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    score = np.empty(len(source), dtype=np.float32)
    h = np.empty((len(source), MH), dtype=np.float32)
    for start in range(0, len(source), 64):
        end = min(start + 64, len(source))
        block = transform_batch(
            source[start:end],
            w1[start:end],
            w2[start:end],
            mode,
            temperature_c,
            None if risk is None else risk[start:end],
        )
        score[start:end] = row_cos(block, truth[start:end])
        h[start:end] = marginal_h(block)
    return {"score": score, "h": h}


def gain_report(
    gain: np.ndarray,
    valid: np.ndarray,
    regimes: np.ndarray,
    risk: np.ndarray,
) -> dict[str, object]:
    report: dict[str, object] = {
        "all": summarize(gain[valid]),
        "positive_fraction": float(np.mean(gain[valid] > 0)),
        "by_regime": {},
        "by_risk": {},
    }
    for name in REGIMES:
        mask = valid & (regimes == name)
        report["by_regime"][name] = {
            "gain": summarize(gain[mask]),
            "positive_fraction": float(np.mean(gain[mask] > 0)) if np.any(mask) else 0.0,
        }
    for low, high in ((0.0, 0.2), (0.2, 0.64), (0.64, 0.9), (0.9, 1.000001)):
        mask = valid & (risk >= low) & (risk < high)
        report["by_risk"][f"{low:.2f}-{high:.2f}"] = {
            "gain": summarize(gain[mask]),
            "positive_fraction": float(np.mean(gain[mask] > 0)) if np.any(mask) else 0.0,
        }
    return report


def marginal_oracles(source_h: np.ndarray, truth_h: np.ndarray) -> dict[str, np.ndarray | dict[str, object]]:
    base = row_cos(source_h, truth_h)
    shifts = np.arange(-4, 5, dtype=np.int32)
    temperatures = np.asarray([0.70, 0.85, 1.00, 1.15, 1.30, 1.60, 2.00], dtype=np.float32)

    shift_scores = np.stack([row_cos(np.roll(source_h, int(s), axis=1), truth_h) for s in shifts], axis=1)
    shift_index = np.argmax(shift_scores, axis=1)
    best_shift = shifts[shift_index]
    best_shift_score = shift_scores[np.arange(len(source_h)), shift_index]

    temp_values = []
    for temperature in temperatures:
        value = np.power(np.maximum(source_h, 1e-12), float(temperature)).astype(np.float32)
        value /= np.maximum(value.sum(axis=1, keepdims=True), TINY)
        temp_values.append(value)
    temp_scores = np.stack([row_cos(value, truth_h) for value in temp_values], axis=1)
    temp_index = np.argmax(temp_scores, axis=1)
    best_temp = temperatures[temp_index]
    best_temp_score = temp_scores[np.arange(len(source_h)), temp_index]

    joint_scores = np.empty((len(source_h), len(shifts), len(temperatures)), dtype=np.float32)
    for shift_index_value, shift in enumerate(shifts):
        shifted = np.roll(source_h, int(shift), axis=1)
        for temp_index_value, temperature in enumerate(temperatures):
            value = np.power(np.maximum(shifted, 1e-12), float(temperature)).astype(np.float32)
            value /= np.maximum(value.sum(axis=1, keepdims=True), TINY)
            joint_scores[:, shift_index_value, temp_index_value] = row_cos(value, truth_h)
    flat_index = np.argmax(joint_scores.reshape(len(source_h), -1), axis=1)
    best_joint_shift = shifts[flat_index // len(temperatures)]
    best_joint_temp = temperatures[flat_index % len(temperatures)]
    best_joint_score = joint_scores.reshape(len(source_h), -1)[np.arange(len(source_h)), flat_index]
    return {
        "base_score": base,
        "best_shift": best_shift.astype(np.int32),
        "best_shift_score": best_shift_score,
        "best_temperature": best_temp,
        "best_temperature_score": best_temp_score,
        "best_joint_shift": best_joint_shift.astype(np.int32),
        "best_joint_temperature": best_joint_temp,
        "best_joint_score": best_joint_score,
        "grid": {"shifts": shifts.tolist(), "temperatures": temperatures.tolist()},
    }


def shift_counts(values: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {str(shift): int(np.sum(mask & (values == shift))) for shift in range(-4, 5)}


def spatial_oof(
    features: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    circular: bool = False,
) -> dict[str, object]:
    selected = np.flatnonzero(valid)
    groups = KMeans(n_clusters=25, random_state=RANDOM_STATE, n_init=20).fit_predict(positions[:, :2])
    splits = GroupKFold(n_splits=5).split(selected, groups=groups[selected])
    if circular:
        angle = 2.0 * np.pi * target / MH
        labels = np.column_stack([np.sin(angle), np.cos(angle)]).astype(np.float32)
        prediction = np.empty((len(selected), 2), dtype=np.float32)
    else:
        labels = target.astype(np.float32)
        prediction = np.empty(len(selected), dtype=np.float32)
    x = np.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    for train_local, validation_local in splits:
        train = selected[train_local]
        validation = selected[validation_local]
        model = ExtraTreesRegressor(
            n_estimators=350,
            max_features=0.65,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train], labels[train])
        prediction[validation_local] = model.predict(x[validation]).astype(np.float32)
    truth_selected = target[selected]
    if circular:
        predicted_angle = np.arctan2(prediction[:, 0], prediction[:, 1])
        predicted = (predicted_angle * MH / (2.0 * np.pi)).astype(np.float32)
        error = circular_abs_delta(predicted, truth_selected)
        baseline = circular_abs_delta(np.zeros_like(truth_selected), truth_selected)
        return {
            "count": int(len(selected)),
            "circular_mae_bins": float(error.mean()),
            "zero_baseline_mae_bins": float(baseline.mean()),
            "within_0p5_bin": float(np.mean(error <= 0.5)),
            "within_1_bin": float(np.mean(error <= 1.0)),
        }
    return {
        **safe_corr(prediction, truth_selected, np.ones(len(selected), dtype=bool)),
        "prediction": summarize(prediction),
        "target": summarize(truth_selected),
        "top20_actual_mean": float(np.mean(truth_selected[np.argsort(prediction)[-max(1, len(prediction) // 5) :]])),
    }


def calibration_evidence() -> dict[str, object]:
    sweep_path = PATHS.analysis / "results/json/inverse_rt_calibration_sweep_coarse.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    best = sweep["best"]
    matched_zero = None
    for row in sweep["top_results"]:
        same = all(
            row[key] == best[key]
            for key in ("steering_scale", "h_sign", "v_sign", "delay_offset", "v_offset")
        )
        if same and row["h_offset"] == 0.0:
            matched_zero = row
            break
    return {
        "best": best,
        "same_calibration_h_offset_zero": matched_zero,
        "h_minus1_assigned_power_delta": (
            float(best["assigned_power_frac"] - matched_zero["assigned_power_frac"])
            if matched_zero is not None
            else None
        ),
        "interpretation_limit": (
            "The inverse-RT sweep independently prefers h_offset=-1, but its power advantage over the "
            "same h_offset=0 configuration is small and does not by itself explain v140 online gain."
        ),
    }


def main() -> None:
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    geodesic_values = geodesic["train_features"].astype(np.float32)
    geodesic_index = {name: index for index, name in enumerate(geodesic_names)}
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")
    matched_probability = np.load(PATHS.shared_features / "train_v129_crossfit_residual_probs.npy").astype(np.float32)

    source = normalize_support(np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"))
    raw_truth = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(np.float32)
    truth_power = raw_truth.sum(axis=(1, 2, 3))
    valid = truth_power > 0
    truth = normalize_truth(raw_truth)
    risk = build_risk(regimes, matched_probability, geodesic_values, geodesic_names)
    active = risk > 0.64
    extreme = risk > 0.90
    w1 = np.where(active, 0.165 * risk, 0.0).astype(np.float32)
    w2 = np.where(extreme, 0.060 * risk, 0.0).astype(np.float32)
    zero = np.zeros(len(source), dtype=np.float32)

    source_score = row_cos(source, truth)
    source_h = marginal_h(source)
    truth_h = marginal_h(truth)
    operator_values: dict[str, dict[str, np.ndarray]] = {
        "log_h1_only": evaluate_operator(source, truth, w1, zero, "log"),
        "log_h2_only": evaluate_operator(source, truth, zero, w2, "log"),
        "log_v140": evaluate_operator(source, truth, w1, w2, "log"),
        "linear_v140": evaluate_operator(source, truth, w1, w2, "linear"),
    }
    temperature_search: dict[str, object] = {}
    best_temperature_key = ""
    best_temperature_mean = -np.inf
    for coefficient in (0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
        value = evaluate_operator(source, truth, zero, zero, "temperature", coefficient, risk)
        gain = value["score"] - source_score
        key = f"risk_temperature_{coefficient:.2f}"
        temperature_search[key] = gain_report(gain, valid, regimes, risk)
        mean = float(np.mean(gain[valid]))
        if mean > best_temperature_mean:
            best_temperature_mean = mean
            best_temperature_key = key

    operator_report = {}
    for name, value in operator_values.items():
        operator_report[name] = gain_report(value["score"] - source_score, valid, regimes, risk)

    v140_h = operator_values["log_v140"]["h"]
    source_center, source_concentration = circular_center(source_h)
    truth_center, truth_concentration = circular_center(truth_h)
    v140_center, v140_concentration = circular_center(v140_h)
    center_valid = valid & (source_concentration > 0.15) & (truth_concentration > 0.15)
    truth_source_delta = signed_circular_delta(truth_center, source_center)
    v140_applied_delta = signed_circular_delta(v140_center, source_center)
    source_entropy = distribution_entropy(source_h)
    truth_entropy = distribution_entropy(truth_h)
    v140_entropy = distribution_entropy(v140_h)
    entropy_excess = source_entropy - truth_entropy

    oracle = marginal_oracles(source_h, truth_h)
    base_h_score = oracle["base_score"]
    best_shift = oracle["best_shift"]
    best_temperature = oracle["best_temperature"]
    best_joint_shift = oracle["best_joint_shift"]
    best_joint_temperature = oracle["best_joint_temperature"]
    oracle_report = {
        "grid": oracle["grid"],
        "base_h_score": summarize(base_h_score[valid]),
        "v140_h_gain": summarize((row_cos(v140_h, truth_h) - base_h_score)[valid]),
        "shift_only_gain": summarize((oracle["best_shift_score"] - base_h_score)[valid]),
        "temperature_only_gain": summarize((oracle["best_temperature_score"] - base_h_score)[valid]),
        "joint_shift_temperature_gain": summarize((oracle["best_joint_score"] - base_h_score)[valid]),
        "best_shift_counts": {
            "all": shift_counts(best_shift, valid),
            "active": shift_counts(best_shift, valid & active),
            "extreme": shift_counts(best_shift, valid & extreme),
            "by_regime": {name: shift_counts(best_shift, valid & (regimes == name)) for name in REGIMES},
        },
        "best_temperature": summarize(best_temperature[valid]),
        "best_joint_shift_counts": shift_counts(best_joint_shift, valid),
        "best_joint_temperature": summarize(best_joint_temperature[valid]),
        "v140_direction_agreement": {
            "active_best_shift_negative": float(np.mean(best_shift[valid & active] < 0)),
            "active_best_shift_zero": float(np.mean(best_shift[valid & active] == 0)),
            "active_best_shift_positive": float(np.mean(best_shift[valid & active] > 0)),
            "extreme_best_shift_negative": float(np.mean(best_shift[valid & extreme] < 0)),
        },
    }

    direct_vector = positions[:, :2] - BS_POS[None, :2]
    direct_angle = np.arctan2(direct_vector[:, 1], direct_vector[:, 0])
    first_angle = np.arctan2(
        geodesic_values[:, geodesic_index["first_dir_sin"]],
        geodesic_values[:, geodesic_index["first_dir_cos"]],
    )
    route_turn_angle = np.angle(np.exp(1j * (first_angle - direct_angle))).astype(np.float32)
    distance_3d = np.linalg.norm(positions - BS_POS[None, :], axis=1)
    direct_uy = (positions[:, 1] - BS_POS[1]) / np.maximum(distance_3d, 1e-6)
    first_sin = geodesic_values[:, geodesic_index["first_dir_sin"]]
    direct_bin = (-12.0 * direct_uy - 1.0) % MH
    first_bin = (-12.0 * first_sin - 1.0) % MH
    geometry_delta = signed_circular_delta(first_bin, direct_bin)
    v140_gain = operator_values["log_v140"]["score"] - source_score
    geometry_report = {
        "center_valid_fraction": float(np.mean(center_valid[valid])),
        "truth_minus_source_center_bins": summarize(truth_source_delta[center_valid]),
        "v140_applied_center_shift_bins": summarize(v140_applied_delta[valid & active]),
        "source_entropy_minus_truth": summarize(entropy_excess[valid]),
        "v140_entropy_change": summarize((v140_entropy - source_entropy)[valid & active]),
        "direct_path_bin_error": {
            "source": summarize(circular_abs_delta(source_center, direct_bin)[center_valid]),
            "truth": summarize(circular_abs_delta(truth_center, direct_bin)[center_valid]),
        },
        "geodesic_first_path_bin_error": {
            "source": summarize(circular_abs_delta(source_center, first_bin)[center_valid]),
            "truth": summarize(circular_abs_delta(truth_center, first_bin)[center_valid]),
        },
        "correlations": {
            "truth_source_delta_vs_geodesic_direct_bin_delta": safe_corr(truth_source_delta, geometry_delta, center_valid),
            "truth_source_delta_vs_route_turn_angle": safe_corr(truth_source_delta, route_turn_angle, center_valid),
            "v140_gain_vs_risk": safe_corr(v140_gain, risk, valid),
            "v140_gain_vs_entropy_excess": safe_corr(v140_gain, entropy_excess, valid),
            "v140_gain_vs_abs_center_error": safe_corr(v140_gain, np.abs(truth_source_delta), center_valid),
            "v140_gain_vs_signed_center_error": safe_corr(v140_gain, truth_source_delta, center_valid),
        },
    }

    basic = np.column_stack(
        [
            positions[:, 0],
            positions[:, 1],
            np.linalg.norm(positions - BS_POS[None, :], axis=1),
            np.sin(direct_angle),
            np.cos(direct_angle),
            blocked,
        ]
    ).astype(np.float32)
    physical_features = np.concatenate(
        [basic, geodesic_values, pointcloud["train_features"].astype(np.float32)], axis=1
    )
    full_features = np.concatenate(
        [physical_features, matched_probability, support_features(source)], axis=1
    )
    oof_report = {}
    for name, features in (("physical_94", physical_features), ("full_525", full_features)):
        oof_report[name] = {
            "v140_gain": spatial_oof(features, v140_gain, positions, valid),
            "abs_center_error": spatial_oof(features, np.abs(truth_source_delta), positions, center_valid),
            "signed_center_error": spatial_oof(features, truth_source_delta, positions, center_valid, circular=True),
        }

    report = {
        "method": "v140_physical_mechanism_analysis",
        "status": "analysis_only_no_candidate",
        "hypothesis": (
            "v140 works because NLOS/geodesic regions contain a systematic H-angle support error, "
            "while geometric log overlap also sharpens an over-diffuse support; separate displacement "
            "from sharpening and test whether either effect follows observable propagation geometry"
        ),
        "train_analogue_limit": (
            "The audit uses the established matched-risk train analogue: v125-equivalent cross-fit "
            "support plus matched v129 residual probabilities. It is semantically aligned for risk, "
            "but it is not an exact cross-fit reconstruction of the recursively blended v129 target."
        ),
        "data": {
            "count": int(len(source)),
            "valid_nonzero": int(valid.sum()),
            "source_shape": list(source.shape),
            "active_fraction": float(np.mean(active)),
            "extreme_fraction": float(np.mean(extreme)),
            "source_score": summarize(source_score[valid]),
        },
        "operator_ablation": operator_report,
        "temperature_search": {
            "best_mean_key": best_temperature_key,
            "variants": temperature_search,
        },
        "h_marginal_oracle": oracle_report,
        "geometry_relationship": geometry_report,
        "inverse_rt_calibration_evidence": calibration_evidence(),
        "spatial_oof_learnability": oof_report,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "point_csv": PATHS.relative(POINT_CSV),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = [
        "index", "x", "y", "regime", "valid", "risk", "active", "extreme",
        "source_score", "v140_gain", "source_h_entropy", "truth_h_entropy",
        "source_h_center", "truth_h_center", "center_concentration_source",
        "center_concentration_truth", "truth_minus_source_center", "v140_applied_center_shift",
        "best_shift", "best_temperature", "best_joint_shift", "best_joint_temperature",
        "direct_bin", "geodesic_first_bin", "geodesic_minus_direct_bin", "route_turn_angle",
    ]
    with POINT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(source)):
            writer.writerow(
                {
                    "index": index,
                    "x": float(positions[index, 0]),
                    "y": float(positions[index, 1]),
                    "regime": str(regimes[index]),
                    "valid": int(valid[index]),
                    "risk": float(risk[index]),
                    "active": int(active[index]),
                    "extreme": int(extreme[index]),
                    "source_score": float(source_score[index]),
                    "v140_gain": float(v140_gain[index]),
                    "source_h_entropy": float(source_entropy[index]),
                    "truth_h_entropy": float(truth_entropy[index]),
                    "source_h_center": float(source_center[index]),
                    "truth_h_center": float(truth_center[index]),
                    "center_concentration_source": float(source_concentration[index]),
                    "center_concentration_truth": float(truth_concentration[index]),
                    "truth_minus_source_center": float(truth_source_delta[index]),
                    "v140_applied_center_shift": float(v140_applied_delta[index]),
                    "best_shift": int(best_shift[index]),
                    "best_temperature": float(best_temperature[index]),
                    "best_joint_shift": int(best_joint_shift[index]),
                    "best_joint_temperature": float(best_joint_temperature[index]),
                    "direct_bin": float(direct_bin[index]),
                    "geodesic_first_bin": float(first_bin[index]),
                    "geodesic_minus_direct_bin": float(geometry_delta[index]),
                    "route_turn_angle": float(route_turn_angle[index]),
                }
            )
    print(f"wrote {REPORT}")
    print(f"wrote {POINT_CSV}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
