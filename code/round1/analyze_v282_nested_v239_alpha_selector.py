#!/usr/bin/env python3
"""Nested official-only pointwise alpha selector for the v239 PDP decoder."""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v233_official_joint_ad_manifold as v233
import analyze_v236_official_path_pdp_decoder as v236
import analyze_v238_raw_path_conditional_angle as v238
import analyze_v239_nnls_path_energy_decoder as v239
import analyze_v274_planar_barrier_attention as v274
import analyze_v281_v239_crossfit_pdp_current as v281
import generate_v215_pointcloud_physics_neural_field as v215
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 282
VERSION = "v282_nested_v239_alpha_selector"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
ACTION_OUT = OUT_DIR / "v282_v239_alpha_actions.npz"
OUTPUT = OUT_DIR / "v282_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
V281_CACHE = PATHS.cache_dir(281) / "v281_v239_crossfit_pdp_current.npz"

LOCAL_GATE = 0.645604
OUTER_SEEDS = v281.OUTER_SEEDS
INNER_FOLDS = 3
ALPHA_GRID = np.linspace(0.0, 0.80, 33, dtype=np.float32)
MIN_OFFICIAL_DELTA = 5e-4
SELECTOR_TREES = 700


def fit_stage1(
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    seed: int,
    support: np.ndarray,
    log_power: np.ndarray,
    target_d: np.ndarray,
    nnls_weights: np.ndarray,
    train_pos: np.ndarray,
    train_blocked: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    fit_anchor, fit_power, fit_weights = v281.make_anchor(
        fit_ids,
        fit_ids,
        support,
        log_power,
        train_pos,
        train_blocked,
        True,
    )
    val_anchor, val_power, val_weights = v281.make_anchor(
        validation_ids,
        fit_ids,
        support,
        log_power,
        train_pos,
        train_blocked,
        False,
    )
    fit_anchor_d = v239.normalize_d(fit_anchor.sum(axis=(1, 2)))
    val_anchor_d = v239.normalize_d(val_anchor.sum(axis=(1, 2)))
    fit_raw = v281.features_for(
        fit_ids,
        fit_anchor,
        fit_power,
        fit_weights,
        train_pos,
        train_blocked,
        fit_anchor_d,
    )
    val_raw = v281.features_for(
        validation_ids,
        val_anchor,
        val_power,
        val_weights,
        train_pos,
        train_blocked,
        val_anchor_d,
    )
    fit_features, [val_features], _ = v233.standardize_feature_sets(
        fit_raw, [val_raw]
    )
    model, training = v239.train_model(
        fit_features,
        v236.raw_path_delay("train", fit_ids),
        target_d[fit_ids],
        nnls_weights[fit_ids],
        seed,
        v281.FROZEN_EPOCHS,
        device,
        None,
    )
    model_d = v239.predict_model(
        model,
        val_features,
        v236.raw_path_delay("train", validation_ids),
        device,
    ).d
    del model, fit_anchor, val_anchor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_anchor_d, model_d, training


def spectrum_descriptors(value: np.ndarray) -> np.ndarray:
    x = v239.normalize_d(value)
    bins = np.arange(x.shape[1], dtype=np.float32)[None, :]
    mean = np.sum(x * bins, axis=1, keepdims=True) / x.shape[1]
    variance = np.sum(x * (bins - mean * x.shape[1]) ** 2, axis=1, keepdims=True)
    std = np.sqrt(np.maximum(variance, 0.0)) / x.shape[1]
    entropy = -np.sum(x * np.log(np.maximum(x, 1e-12)), axis=1, keepdims=True)
    entropy /= np.log(x.shape[1])
    boundaries = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192)
    bands = np.stack(
        [x[:, lo:hi].sum(axis=1) for lo, hi in zip(boundaries[:-1], boundaries[1:])],
        axis=1,
    )
    top_index = np.argpartition(x, -6, axis=1)[:, -6:]
    top_mass = np.take_along_axis(x, top_index, axis=1)
    order = np.argsort(top_mass, axis=1)[:, ::-1]
    top_index = np.take_along_axis(top_index, order, axis=1)
    top_mass = np.take_along_axis(top_mass, order, axis=1)
    angle = 2.0 * np.pi * top_index.astype(np.float32) / x.shape[1]
    return np.concatenate(
        [
            mean.astype(np.float32),
            std.astype(np.float32),
            entropy.astype(np.float32),
            bands.astype(np.float32),
            top_mass.astype(np.float32),
            np.sin(angle).astype(np.float32),
            np.cos(angle).astype(np.float32),
        ],
        axis=1,
    )


def selector_features(
    ids: np.ndarray,
    anchor_d: np.ndarray,
    model_d: np.ndarray,
    node_raw: np.ndarray,
) -> np.ndarray:
    anchor = v239.normalize_d(anchor_d)
    model = v239.normalize_d(model_d)
    cosine = v239.row_cos(anchor, model)[:, None]
    l1 = np.sum(np.abs(model - anchor), axis=1, keepdims=True)
    cdf = np.mean(
        np.abs(np.cumsum(model, axis=1) - np.cumsum(anchor, axis=1)),
        axis=1,
        keepdims=True,
    )
    feature = np.concatenate(
        [
            np.asarray(node_raw[ids], dtype=np.float32),
            np.sqrt(np.maximum(anchor, 0.0)),
            np.sqrt(np.maximum(model, 0.0)),
            (model - anchor).astype(np.float32),
            spectrum_descriptors(anchor),
            spectrum_descriptors(model),
            cosine.astype(np.float32),
            l1.astype(np.float32),
            cdf.astype(np.float32),
        ],
        axis=1,
    )
    return np.nan_to_num(feature, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def optimal_alpha(
    anchor_d: np.ndarray,
    model_d: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    for alpha in ALPHA_GRID:
        prediction = v239.normalize_d(
            (1.0 - float(alpha)) * anchor_d + float(alpha) * model_d
        )
        scores.append(v239.row_cos(prediction, truth_d))
    score = np.stack(scores, axis=1)
    best = np.argmax(score, axis=1)
    return ALPHA_GRID[best].astype(np.float32), np.take_along_axis(
        score, best[:, None], axis=1
    )[:, 0].astype(np.float32)


def make_selector(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=SELECTOR_TREES,
        max_depth=18,
        min_samples_leaf=10,
        max_features=0.62,
        n_jobs=-1,
        random_state=seed,
    )


def current_from_alpha(
    anchor_d: np.ndarray,
    model_d: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    return v239.normalize_d(
        (1.0 - alpha[:, None]) * anchor_d + alpha[:, None] * model_d
    )


def main() -> None:
    for path in (BASE, V281_CACHE, v239.TARGET_OUT):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_targets, _raw, _power, valid_delay = v215.load_targets()
    target_d = v239.normalize_d(clean_targets["d"])
    support, valid_support, log_power = v238.build_raw_support()
    valid_ids = np.flatnonzero(valid_delay & valid_support).astype(np.int64)
    nnls_weights, _reconstruction, _nnls_report = v239.build_nnls_labels(
        target_d, valid_ids
    )
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v238.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    (
        all_pos,
        signatures,
        blocker_count,
        scene_features,
        _footprint,
        _z_max,
        _x_min,
        _y_min,
        _resolution,
    ) = v274.load_geometry()
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    v281_data = np.load(V281_CACHE)

    outer_rows: list[dict[str, object]] = []
    nested_ids_parts: list[np.ndarray] = []
    nested_anchor_parts: list[np.ndarray] = []
    nested_model_parts: list[np.ndarray] = []
    nested_truth_parts: list[np.ndarray] = []
    for outer_fold, seed in enumerate(OUTER_SEEDS):
        outer_mask = v281_data["fold"] == outer_fold
        outer_ids = v281_data["ids"][outer_mask].astype(np.int64)
        outer_anchor = v281_data["anchor_d"][outer_mask].astype(np.float32)
        outer_model = v281_data["model_d"][outer_mask].astype(np.float32)
        outer_current = v281_data["current_d"][outer_mask].astype(np.float32)
        outer_truth = v281_data["truth_d"][outer_mask].astype(np.float32)
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        stratify = regime_of(train_blocked[outer_fit]).astype(str)
        splitter = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=seed + 82_000
        )
        train_ids_parts: list[np.ndarray] = []
        train_anchor_parts: list[np.ndarray] = []
        train_model_parts: list[np.ndarray] = []
        inner_training: list[dict[str, object]] = []
        for inner_fold, (fit_local, val_local) in enumerate(
            splitter.split(outer_fit, stratify)
        ):
            fit_ids = outer_fit[fit_local]
            val_ids = outer_fit[val_local]
            anchor_d, model_d, training = fit_stage1(
                fit_ids,
                val_ids,
                seed + 100_000 + inner_fold * 1_000,
                support,
                log_power,
                target_d,
                nnls_weights,
                train_pos,
                train_blocked,
                device,
            )
            train_ids_parts.append(val_ids)
            train_anchor_parts.append(anchor_d)
            train_model_parts.append(model_d)
            inner_training.append(
                {
                    "fold": inner_fold,
                    "fit_count": int(len(fit_ids)),
                    "validation_count": int(len(val_ids)),
                    "training": training,
                }
            )
        selector_ids = np.concatenate(train_ids_parts)
        selector_anchor = np.concatenate(train_anchor_parts)
        selector_model = np.concatenate(train_model_parts)
        selector_truth = target_d[selector_ids]
        y_alpha, oracle_score = optimal_alpha(
            selector_anchor, selector_model, selector_truth
        )
        x_train = selector_features(
            selector_ids, selector_anchor, selector_model, node_raw
        )
        selector = make_selector(seed + 282_000)
        selector.fit(x_train, y_alpha)
        predicted_alpha = np.clip(
            selector.predict(
                selector_features(outer_ids, outer_anchor, outer_model, node_raw)
            ),
            0.0,
            float(ALPHA_GRID[-1]),
        ).astype(np.float32)
        adaptive = current_from_alpha(
            outer_anchor, outer_model, predicted_alpha
        )
        baseline_cos = v239.row_cos(outer_current, outer_truth)
        adaptive_cos = v239.row_cos(adaptive, outer_truth)
        delta = adaptive_cos - baseline_cos
        outer_rows.append(
            {
                "fold": outer_fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "inner_training": inner_training,
                "selector_train": {
                    "count": int(len(selector_ids)),
                    "optimal_alpha": summarize(y_alpha),
                    "oracle_cos": summarize(oracle_score),
                },
                "predicted_alpha": summarize(predicted_alpha),
                "baseline_cos": summarize(baseline_cos),
                "adaptive_cos": summarize(adaptive_cos),
                "delta": summarize(delta),
                "positive_fraction": float(np.mean(delta > 0.0)),
            }
        )
        nested_ids_parts.append(selector_ids)
        nested_anchor_parts.append(selector_anchor)
        nested_model_parts.append(selector_model)
        nested_truth_parts.append(selector_truth)
        print(
            f"v282 outer {outer_fold + 1}/{len(OUTER_SEEDS)} "
            f"baseline={baseline_cos.mean():.6f} adaptive={adaptive_cos.mean():.6f} "
            f"delta={delta.mean():+.6f} alpha={predicted_alpha.mean():.3f}",
            flush=True,
        )

    fold_delta = np.asarray(
        [row["delta"]["mean"] for row in outer_rows], dtype=np.float64
    )
    official_selected = bool(
        np.mean(fold_delta) >= MIN_OFFICIAL_DELTA
        and np.sum(fold_delta > 0.0) >= 3
    )
    local_score = None
    projection_diagnostics = None
    test_alpha = None
    if official_selected:
        train_ids = v281_data["ids"].astype(np.int64)
        train_anchor = v281_data["anchor_d"].astype(np.float32)
        train_model = v281_data["model_d"].astype(np.float32)
        train_truth = v281_data["truth_d"].astype(np.float32)
        train_alpha, _ = optimal_alpha(train_anchor, train_model, train_truth)
        final_selector = make_selector(282_999)
        final_selector.fit(
            selector_features(train_ids, train_anchor, train_model, node_raw),
            train_alpha,
        )
        test_anchor_support = np.load(
            v238.OUT_DIR / "v238_test_anchor.npy", mmap_mode="r"
        )
        test_anchor = v239.normalize_d(
            np.asarray(test_anchor_support, dtype=np.float32).sum(axis=(1, 2))
        )
        test_model = v239.normalize_d(np.load(v239.TARGET_OUT).astype(np.float32))
        test_ids = np.arange(len(target_d), len(all_pos), dtype=np.int64)
        test_alpha = np.clip(
            final_selector.predict(
                selector_features(test_ids, test_anchor, test_model, node_raw)
            ),
            0.0,
            float(ALPHA_GRID[-1]),
        ).astype(np.float32)
        target_test_d = current_from_alpha(test_anchor, test_model, test_alpha)
        current = v225.current_delay_marginals(BASE)
        targets = {
            "d": target_test_d,
            "du": current["du"],
            "dh": current["dh"],
        }
        projection_diagnostics = v225.apply_delay_targets(
            BASE,
            OUTPUT,
            targets,
            regime_of(test_blocked),
            1.02,
        )
        local_score = v208.score_prediction(OUTPUT)
        np.savez_compressed(
            ACTION_OUT,
            alpha=test_alpha,
            anchor_d=test_anchor,
            model_d=test_model,
            target_d=target_test_d,
            current_d=current["d"],
        )

    report = {
        "method": VERSION,
        "status": (
            "local_gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "The frozen v239 path correction has a large stable mean PDP gain but harms "
            "roughly forty percent of points; a nested official-only selector can predict "
            "the pointwise path blend alpha from geometry and anchor/model spectrum evidence."
        ),
        "compliance_boundary": {
            "uses": [
                "nested cross-fit v239-like official PDP currents",
                "official train/test coordinates and point-cloud node features",
                "official anchor/model PDP disagreement",
                "strict v239 official-only carrier after official selection",
            ],
            "does_not_use": [
                "outer validation H in stage-1 or selector training",
                "synthetic/Sionna labels",
                "v212/v217 teacher",
                "hidden labels or RT H/path coefficients",
                "local replay truth for training or selector selection",
            ],
            "local_truth_usage": (
                "one final gate score after nested official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "inner_stage1_folds": INNER_FOLDS,
            "stage1_epochs": v281.FROZEN_EPOCHS,
            "alpha_grid": ALPHA_GRID.tolist(),
            "selector": {
                "type": "ExtraTreesRegressor",
                "trees": SELECTOR_TREES,
                "max_depth": 18,
                "min_samples_leaf": 10,
                "max_features": 0.62,
            },
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "fold_delta": fold_delta.tolist(),
            "mean_delta": float(np.mean(fold_delta)),
            "wins": int(np.sum(fold_delta > 0.0)),
            "selected": official_selected,
        },
        "test_alpha": summarize(test_alpha) if test_alpha is not None else None,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in projection_diagnostics.items()}
            if projection_diagnostics is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "actions": PATHS.relative(ACTION_OUT) if ACTION_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
