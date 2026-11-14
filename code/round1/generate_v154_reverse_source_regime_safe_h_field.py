#!/usr/bin/env python3
"""Generate v154 with a regime-safe reverse-source local H residual field.

The v152 reverse-source local-H field had strong spatial OOF gain, but its
per-regime audit showed a clear failure mode: light-regime points were harmed
while clear/medium/heavy points benefited.  This version keeps the same
source-matching idea, but makes strength and allowed regimes part of a nested
spatial selection instead of blindly applying one fixed action everywhere.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesClassifier

from analyze_post_v143_hidden_information import normalize_without_floor, support_score
from analyze_v145_gated_local_h_field import local_h_predictions
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos, support_features
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    digest,
    output_comparison,
)
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v154_reverse_source_regime_safe_h_field"
OUT_DIR = PATHS.cache_dir(154)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(154)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v154_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v154_reverse_source_regime_safe_h_field_target.npy"
FIELD_OUT = OUT_DIR / "v154_reverse_source_regime_safe_h_field.npz"
OOF_OUT = OUT_DIR / "v154_reverse_source_regime_safe_h_field_oof.npz"
REPORT = REPORT_DIR / "v154_reverse_source_regime_safe_h_field_generate.json"
ZIP_OUT = PATHS.round1_versions / "v154_reverse_source_regime_safe_h_field_candidate_pending.zip"

RANDOM_STATE = 15401
REVERSE_K = 8
LOCAL_K = 8
DISTANCE_FLOOR = 0.25
MAX_DISTANCE = 20.0
STRENGTH_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
REGIMES = ("clear", "light", "medium", "heavy")
REGIME_CONFIGS = (
    ("all", REGIMES),
    ("no_light", ("clear", "medium", "heavy")),
    ("medium_heavy", ("medium", "heavy")),
    ("clear_heavy", ("clear", "heavy")),
    ("heavy_only", ("heavy",)),
)
MIN_MEAN_FOR_ZIP = 0.0010


def summarize_gain(values: np.ndarray) -> dict[str, object]:
    return {**summarize(values), "positive_fraction": float(np.mean(values > 0.0))}


def normalize_h(values: np.ndarray) -> np.ndarray:
    return (values / np.maximum(values.sum(axis=1, keepdims=True), TINY)).astype(
        np.float32
    )


def config_name(strength: float, regime_key: str) -> str:
    return f"s{strength:.2f}_{regime_key}"


def parse_config(name: str) -> tuple[float, str]:
    strength_text, regime_key = name.split("_", 1)
    return float(strength_text[1:]), regime_key


def allowed_regimes(regime_key: str) -> tuple[str, ...]:
    lookup = dict(REGIME_CONFIGS)
    return tuple(lookup[regime_key])


def regime_mask(regimes: np.ndarray, regime_key: str) -> np.ndarray:
    allowed = allowed_regimes(regime_key)
    return np.isin(regimes, allowed)


def interpolate_h_marginal(
    positions_source: np.ndarray,
    h_source: np.ndarray,
    positions_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=REVERSE_K, n_jobs=-1)
    model.fit(positions_source[:, :2])
    distance, local = model.kneighbors(positions_query[:, :2])
    inverse = 1.0 / np.maximum(distance, DISTANCE_FLOOR)
    weight = inverse / np.maximum(inverse.sum(axis=1, keepdims=True), 1e-12)
    prediction = np.sum(weight[..., None] * h_source[local], axis=1)
    return (
        normalize_h(prediction),
        local.astype(np.int32),
        distance.astype(np.float32),
        weight.astype(np.float32),
    )


def replace_h_marginal(current: np.ndarray, h_target: np.ndarray) -> np.ndarray:
    current_h = current.sum(axis=(2, 3))
    ratio = h_target / np.maximum(current_h, 1e-10)
    output = (current * ratio[:, :, None, None]).astype(np.float32)
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output


def domain_auc(train_support: np.ndarray, test_support: np.ndarray) -> dict[str, float]:
    x = np.concatenate(
        [support_features(train_support), support_features(test_support)], axis=0
    )
    y = np.concatenate(
        [
            np.zeros(len(train_support), dtype=np.int32),
            np.ones(len(test_support), dtype=np.int32),
        ],
        axis=0,
    )
    prediction = np.empty(len(y), dtype=np.float32)
    for train_index, validation_index in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    ).split(x, y):
        model = ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train_index], y[train_index])
        prediction[validation_index] = model.predict_proba(
            x[validation_index]
        )[:, 1]
    return {"auc": float(roc_auc_score(y, prediction))}


def adjust_support(
    current: np.ndarray,
    prediction: np.ndarray,
    gate: np.ndarray,
    strength: float,
) -> np.ndarray:
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0) * (strength * gate.astype(np.float32))[:, None]
    ).astype(np.float32)
    output = (current * ratio[:, :, None, None]).astype(np.float32)
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output


def compute_strength_gains(
    current: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    local_gate: np.ndarray,
) -> dict[float, np.ndarray]:
    base = support_score(current, truth)
    gains: dict[float, np.ndarray] = {}
    for strength in STRENGTH_GRID:
        adjusted = adjust_support(current, prediction, local_gate, strength)
        gains[strength] = (support_score(adjusted, truth) - base).astype(np.float32)
    return gains


def score_config(
    gains_by_strength: dict[float, np.ndarray],
    regimes: np.ndarray,
    strength: float,
    regime_key: str,
    fold_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    active_regime = regime_mask(regimes, regime_key)
    gain = np.where(active_regime, gains_by_strength[strength], 0.0).astype(np.float32)
    active = active_regime & (np.abs(gains_by_strength[strength]) > 0.0)
    row: dict[str, object] = {
        "config": config_name(strength, regime_key),
        "strength": strength,
        "regime_key": regime_key,
        "allowed_regimes": list(allowed_regimes(regime_key)),
        "full_gain_mean": float(np.mean(gain)),
        "full_gain_std": float(np.std(gain)),
        "active_fraction": float(np.mean(active)),
        "active_gain_mean": float(np.mean(gain[active])) if np.any(active) else 0.0,
        "positive_fraction": float(np.mean(gain > 0.0)),
        "positive_fraction_active": (
            float(np.mean(gain[active] > 0.0)) if np.any(active) else 0.0
        ),
    }
    if fold_ids is not None:
        fold_means = [
            float(np.mean(gain[fold_ids == fold]))
            for fold in sorted(np.unique(fold_ids).tolist())
        ]
        row["fold_means"] = fold_means
        row["all_folds_positive"] = bool(np.all(np.asarray(fold_means) > 0.0))
    return gain, row


def select_config(
    gains_by_strength: dict[float, np.ndarray],
    regimes: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[tuple[float, str], dict[str, object]]:
    rows = []
    best_config = (STRENGTH_GRID[0], REGIME_CONFIGS[0][0])
    best_key = (-np.inf, -np.inf, -np.inf)
    for strength in STRENGTH_GRID:
        for regime_key, _ in REGIME_CONFIGS:
            _, row = score_config(gains_by_strength, regimes, strength, regime_key, fold_ids)
            rows.append(row)
            active = float(row["active_fraction"])
            full_gain = float(row["full_gain_mean"])
            active_gain = float(row["active_gain_mean"])
            fold_means = np.asarray(row["fold_means"], dtype=np.float32)
            eligible = (
                active >= 0.08
                and active <= 0.75
                and bool(row["all_folds_positive"])
                and float(row["positive_fraction_active"]) >= 0.50
            )
            if not eligible:
                continue
            lcb = full_gain - 0.25 * float(np.std(fold_means))
            key = (lcb, full_gain, active_gain)
            if key > best_key:
                best_key = key
                best_config = (strength, regime_key)
    if not np.isfinite(best_key[0]):
        for row in rows:
            active = float(row["active_fraction"])
            if active > 0.80:
                continue
            full_gain = float(row["full_gain_mean"])
            active_gain = float(row["active_gain_mean"])
            fold_means = np.asarray(row["fold_means"], dtype=np.float32)
            lcb = full_gain - 0.50 * float(np.std(fold_means))
            key = (lcb, full_gain, active_gain)
            if key > best_key:
                best_key = key
                best_config = (float(row["strength"]), str(row["regime_key"]))
    rows.sort(key=lambda item: item["full_gain_mean"], reverse=True)
    return best_config, {
        "searched_configs": len(rows),
        "selected_config": config_name(*best_config),
        "top_configs": rows[:12],
    }


def nested_oof_audit(
    positions_train: np.ndarray,
    reverse_current: np.ndarray,
    truth: np.ndarray,
    h_target: np.ndarray,
    valid: np.ndarray,
    regimes_train: np.ndarray,
) -> dict[str, object]:
    groups = KMeans(n_clusters=25, random_state=RANDOM_STATE, n_init=20).fit_predict(
        positions_train[:, :2]
    )
    selected = np.flatnonzero(valid)
    outer = GroupKFold(n_splits=5)
    oof_gain = np.zeros(len(positions_train), dtype=np.float32)
    oof_selected = np.full(len(positions_train), "none", dtype=object)
    oof_active = np.zeros(len(positions_train), dtype=bool)
    fold_reports = []
    selected_configs = []

    for fold, (outer_train_local, outer_validation_local) in enumerate(
        outer.split(selected, groups=groups[selected])
    ):
        outer_train = selected[outer_train_local]
        outer_validation = selected[outer_validation_local]

        inner = GroupKFold(n_splits=4)
        inner_prediction = np.zeros((len(positions_train), h_target.shape[1]), dtype=np.float32)
        inner_distance = np.zeros(len(positions_train), dtype=np.float32)
        inner_fold_ids = np.full(len(outer_train), -1, dtype=np.int16)
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
            inner.split(outer_train, groups=groups[outer_train])
        ):
            inner_train = outer_train[inner_train_local]
            inner_validation = outer_train[inner_validation_local]
            prediction, _, distance = local_h_predictions(
                positions_train, h_target, inner_train, inner_validation
            )
            inner_prediction[inner_validation] = prediction[LOCAL_K]
            inner_distance[inner_validation] = distance[LOCAL_K]
            inner_fold_ids[inner_validation_local] = inner_fold

        inner_local_gate = inner_distance[outer_train] <= MAX_DISTANCE
        inner_gains = compute_strength_gains(
            reverse_current[outer_train],
            truth[outer_train],
            inner_prediction[outer_train],
            inner_local_gate,
        )
        selected_config, config_report = select_config(
            inner_gains,
            regimes_train[outer_train],
            inner_fold_ids,
        )
        selected_configs.append(config_name(*selected_config))

        validation_prediction, _, validation_distance = local_h_predictions(
            positions_train, h_target, outer_train, outer_validation
        )
        validation_local_gate = validation_distance[LOCAL_K] <= MAX_DISTANCE
        validation_gains = compute_strength_gains(
            reverse_current[outer_validation],
            truth[outer_validation],
            validation_prediction[LOCAL_K],
            validation_local_gate,
        )
        validation_gain, validation_row = score_config(
            validation_gains,
            regimes_train[outer_validation],
            selected_config[0],
            selected_config[1],
        )
        validation_active = (
            validation_local_gate & regime_mask(regimes_train[outer_validation], selected_config[1])
        )
        oof_gain[outer_validation] = validation_gain
        oof_selected[outer_validation] = config_name(*selected_config)
        oof_active[outer_validation] = validation_active

        fold_reports.append(
            {
                "fold": fold,
                "selected_config": config_name(*selected_config),
                "train_count": int(len(outer_train)),
                "validation_count": int(len(outer_validation)),
                "validation_gain": summarize_gain(validation_gain),
                "active_fraction": float(np.mean(validation_active)),
                "active_gain": (
                    summarize_gain(validation_gain[validation_active])
                    if np.any(validation_active)
                    else {}
                ),
                "local_gate_fraction": float(np.mean(validation_local_gate)),
                "validation_config_row": validation_row,
                "inner_selection": config_report,
            }
        )

    valid_gain = oof_gain[valid]
    valid_active = oof_active[valid]
    by_regime = {}
    for regime in REGIMES:
        mask = valid & (regimes_train == regime)
        by_regime[regime] = {
            "count": int(mask.sum()),
            "gain": summarize_gain(oof_gain[mask]),
            "active_fraction": float(np.mean(oof_active[mask])),
        }
    config_counts = Counter(selected_configs)
    modal_config_name = config_counts.most_common(1)[0][0]
    return {
        "outer_folds": 5,
        "inner_folds": 4,
        "local_k": LOCAL_K,
        "maximum_distance": MAX_DISTANCE,
        "strength_grid": list(STRENGTH_GRID),
        "regime_configs": {
            key: list(values) for key, values in REGIME_CONFIGS
        },
        "fold_reports": fold_reports,
        "oof_gain": oof_gain,
        "oof_selected": oof_selected,
        "oof_active": oof_active,
        "summary": {
            "full_gain": summarize_gain(valid_gain),
            "active_fraction": float(np.mean(valid_active)),
            "active_gain": (
                summarize_gain(valid_gain[valid_active]) if np.any(valid_active) else {}
            ),
            "fold_means": [
                row["validation_gain"]["mean"] for row in fold_reports
            ],
            "all_folds_positive": bool(
                np.all([row["validation_gain"]["mean"] > 0.0 for row in fold_reports])
            ),
            "selected_config_counts": {
                key: int(value) for key, value in sorted(config_counts.items())
            },
            "modal_config": modal_config_name,
            "by_regime": by_regime,
        },
    }


def build_final_target(
    positions_train: np.ndarray,
    positions_test: np.ndarray,
    h_target: np.ndarray,
    valid: np.ndarray,
    v140_target: np.ndarray,
    regimes_test: np.ndarray,
    modal_config_name: str,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    strength, regime_key = parse_config(modal_config_name)
    train_index = np.flatnonzero(valid)
    model = NearestNeighbors(n_neighbors=LOCAL_K, n_jobs=-1)
    model.fit(positions_train[train_index, :2])
    distance, local = model.kneighbors(positions_test[:, :2])
    neighbors = train_index[local]
    inverse = 1.0 / np.maximum(distance, DISTANCE_FLOOR)
    weight = inverse / np.maximum(inverse.sum(axis=1, keepdims=True), 1e-12)
    values = h_target[neighbors]
    prediction = np.sum(weight[..., None] * values, axis=1).astype(np.float32)
    variance = np.sum(weight[..., None] * (values - prediction[:, None]) ** 2, axis=1)
    dispersion = np.sqrt(np.mean(variance, axis=1)).astype(np.float32)
    local_gate = distance[:, LOCAL_K - 1] <= MAX_DISTANCE
    active = local_gate & regime_mask(regimes_test, regime_key)
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0) * (strength * active.astype(np.float32))[:, None]
    ).astype(np.float32)
    final_target = (v140_target * ratio[:, :, None, None]).astype(np.float32)
    final_target /= np.maximum(final_target.sum(axis=(1, 2, 3), keepdims=True), TINY)
    diagnostics = {
        "modal_config": modal_config_name,
        "strength": strength,
        "regime_key": regime_key,
        "allowed_regimes": list(allowed_regimes(regime_key)),
        "active_fraction": float(np.mean(active)),
        "local_gate_fraction": float(np.mean(local_gate)),
        "active_by_regime": {
            regime: float(np.mean(active[regimes_test == regime]))
            for regime in REGIMES
        },
        "prediction": summarize(prediction),
        "dispersion": summarize(dispersion),
        "eighth_neighbor_distance": summarize(distance[:, LOCAL_K - 1]),
    }
    arrays = {
        "prediction": prediction,
        "dispersion": dispersion,
        "gate": active.astype(np.float32),
        "local_gate": local_gate.astype(np.float32),
        "neighbor_indices": neighbors.astype(np.int32),
        "neighbor_distances": distance.astype(np.float32),
        "neighbor_weights": weight.astype(np.float32),
        "selected_strength": (strength * active.astype(np.float32)).astype(np.float32),
    }
    return final_target, diagnostics, arrays


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(
        np.float32
    )
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(
        np.float32
    )
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    probability_train = np.load(
        PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
    ).astype(np.float32)

    source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    train_risk = build_risk(
        regimes_train,
        probability_train,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    train_current, _ = apply_risk_shift(source, train_risk)
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    valid = truth_raw.sum(axis=(1, 2, 3)) > 0
    truth = normalize_without_floor(truth_raw)
    truth_h = truth.sum(axis=(2, 3))

    v140_target = normalize_support(
        np.load(PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"),
        1e-12,
    )
    test_h = v140_target.sum(axis=(2, 3))
    reverse_h, reverse_neighbors, reverse_distances, reverse_weights = (
        interpolate_h_marginal(positions_test, test_h, positions_train)
    )
    reverse_current = replace_h_marginal(train_current, reverse_h)
    h_target = np.clip(
        np.log(np.maximum(truth_h, 1e-10))
        - np.log(np.maximum(reverse_h, 1e-10)),
        -3.0,
        3.0,
    ).astype(np.float32)

    audit = nested_oof_audit(
        positions_train,
        reverse_current,
        truth,
        h_target,
        valid,
        regimes_train,
    )
    summary = audit["summary"]
    accepted_local = bool(
        summary["full_gain"]["mean"] >= MIN_MEAN_FOR_ZIP
        and summary["all_folds_positive"]
    )

    final_target, test_decision, field_arrays = build_final_target(
        positions_train,
        positions_test,
        h_target,
        valid,
        v140_target,
        regimes_test,
        str(summary["modal_config"]),
    )
    np.save(TARGET_OUT, final_target)
    np.savez_compressed(
        FIELD_OUT,
        reverse_h=reverse_h,
        reverse_neighbors=reverse_neighbors,
        reverse_distances=reverse_distances,
        reverse_weights=reverse_weights,
        train_h_residual=h_target,
        **field_arrays,
    )
    np.savez_compressed(
        OOF_OUT,
        oof_gain=audit["oof_gain"].astype(np.float32),
        oof_active=audit["oof_active"].astype(bool),
        oof_selected=np.asarray(audit["oof_selected"], dtype=object),
        valid=valid.astype(bool),
    )

    weights = np.load(
        PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_weights.npy"
    ).astype(np.float32)
    args = argparse.Namespace(
        base=str(PATHS.cache_dir(129) / "v129_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.028,medium:0.470,heavy:0.780",
        min_gate=0.12,
        max_gate=2.95,
        conf_center=0.22,
        conf_width=0.38,
        entropy_low=0.02,
        entropy_high=0.90,
        gain_clip="0.34,3.20",
        top_gain_clip="0.48,2.35",
        top_components=5632,
        target_floor=1e-12,
        project_batch=20,
    )
    projection_report = projection(
        final_target, weights, regimes_test, positions_test.astype(np.float64), args
    )

    output = np.load(OUTPUT, mmap_mode="r")
    baseline = np.load(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", mmap_mode="r"
    )
    comparison = output_comparison(output, baseline, regimes_test)
    if output.shape != baseline.shape or output.dtype != np.complex64:
        raise ValueError(
            f"invalid output layout: shape={output.shape} dtype={output.dtype}"
        )
    if not comparison["finite"]:
        raise ValueError("v154 output contains non-finite values")

    v152_target = normalize_support(
        np.load(PATHS.cache_dir(152) / "v152_reverse_source_local_h_field_target.npy"),
        1e-12,
    )
    final_h = final_target.sum(axis=(2, 3))
    report_args = dict(vars(args))
    report_args["base"] = PATHS.relative(args.base)
    report_args["output"] = PATHS.relative(args.output)
    report = {
        "method": VERSION,
        "status": "candidate_pending" if accepted_local else "rejected_local",
        "base": "v140 online 0.501352; v152 reverse-source local H signal",
        "primary_hypothesis": (
            "v152's reverse-source local H residual has real signal, but the "
            "light-regime part is locally harmful.  A nested spatial selector "
            "over strength and allowed physical regimes should keep the strong "
            "clear/medium/heavy gains while avoiding the v152 light failure mode."
        ),
        "only_primary_change": (
            "keep the v152 reverse-source residual label, but choose local-H "
            "strength and allowed blocked-regimes by nested spatial OOF; the "
            "selected modal policy is then applied to the v140 target before "
            "the unchanged v140 projection."
        ),
        "model": {
            "reverse_k": REVERSE_K,
            "local_k": LOCAL_K,
            "maximum_distance": MAX_DISTANCE,
            "distance_floor": DISTANCE_FLOOR,
            "strength_grid": list(STRENGTH_GRID),
            "regime_configs": {
                key: list(values) for key, values in REGIME_CONFIGS
            },
            "min_mean_for_zip": MIN_MEAN_FOR_ZIP,
        },
        "reverse_source": {
            "train_current_vs_truth": summarize(
                support_score(train_current[valid], truth[valid])
            ),
            "reverse_current_vs_truth": summarize(
                support_score(reverse_current[valid], truth[valid])
            ),
            "train_current_to_test_v140_domain": domain_auc(train_current, v140_target),
            "reverse_current_to_test_v140_domain": domain_auc(
                reverse_current, v140_target
            ),
            "nearest_test_distance": summarize(reverse_distances[:, 0]),
            "kth_test_distance": summarize(reverse_distances[:, REVERSE_K - 1]),
        },
        "nested_spatial_oof": {
            "outer_folds": audit["outer_folds"],
            "inner_folds": audit["inner_folds"],
            "local_k": audit["local_k"],
            "maximum_distance": audit["maximum_distance"],
            "strength_grid": audit["strength_grid"],
            "regime_configs": audit["regime_configs"],
            "fold_reports": audit["fold_reports"],
            **summary,
        },
        "accepted_local": accepted_local,
        "test_decision": test_decision,
        "target_diagnostics": {
            "target_to_v140_target": summarize(row_cos(final_target, v140_target)),
            "target_to_v152_target": summarize(row_cos(final_target, v152_target)),
            "h_marginal_l1_change": summarize(np.sum(np.abs(final_h - test_h), axis=1)),
            "h_marginal_max_ratio": summarize(
                np.max(final_h / np.maximum(test_h, 1e-12), axis=1)
            ),
        },
        "projection": projection_report,
        "comparison_to_v140": comparison,
        "preserved_from_v140": {
            "geodesic_risk_and_h_shift": True,
            "projection_weights": PATHS.relative(
                PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_weights.npy"
            ),
            "projection_args": report_args,
            "complex_phase": True,
            "total_power": True,
            "pol_ue_allocation": True,
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "field": PATHS.relative(FIELD_OUT),
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
        "integrity": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": comparison["finite"],
            "output_bytes": int(output.nbytes),
            "zip_created": accepted_local,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if accepted_local:
        with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(OUTPUT, "Round1_Test_Channel.npy")
            for source_path, name in (
                (__file__, "generate_v154_reverse_source_regime_safe_h_field.py"),
                (
                    PATHS.code / "generate_v152_reverse_source_local_h_field.py",
                    "generate_v152_reverse_source_local_h_field.py",
                ),
                (
                    PATHS.code / "generate_v145_local_h_residual_field.py",
                    "generate_v145_local_h_residual_field.py",
                ),
                (
                    PATHS.code / "analyze_v145_gated_local_h_field.py",
                    "analyze_v145_gated_local_h_field.py",
                ),
                (
                    PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                    "generate_v140_balanced_hminus2_overdrive.py",
                ),
                (
                    PATHS.code / "generate_v143_transductive_geodesic_risk.py",
                    "generate_v143_transductive_geodesic_risk.py",
                ),
                (
                    PATHS.code / "generate_v112_kernel_moe_path_support.py",
                    "generate_v112_kernel_moe_path_support.py",
                ),
                (
                    PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                    "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                ),
                (PATHS.code / "powerfield_common.py", "powerfield_common.py"),
                (PATHS.code_root / "project_paths.py", "project_paths.py"),
            ):
                archive.write(source_path, name)
        with zipfile.ZipFile(ZIP_OUT) as archive:
            bad_member = archive.testzip()
            members = archive.namelist()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failure: {bad_member}")
        report["integrity"].update(
            {
                "zip_crc": True,
                "zip_members": members,
                "sha256": digest(ZIP_OUT, "sha256"),
                "md5": digest(ZIP_OUT, "md5"),
            }
        )
    else:
        if ZIP_OUT.exists():
            ZIP_OUT.unlink()
        report["integrity"].update({"zip_crc": None, "zip_members": []})

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT}")
    if accepted_local:
        print(f"wrote {ZIP_OUT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
