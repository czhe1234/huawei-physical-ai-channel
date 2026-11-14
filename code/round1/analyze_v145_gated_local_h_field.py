#!/usr/bin/env python3
"""Nested spatial-OOF audit for a gated local H-marginal residual field."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from analyze_post_v143_hidden_information import normalize_without_floor, support_score
from analyze_v143_matched_residual_probability import basic_features
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos, support_features
from generate_v143_transductive_geodesic_risk import apply_risk_shift
from powerfield_common import TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(145)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v145_gated_local_h_field_audit.json"
OOF_OUT = REPORT_DIR / "v145_gated_local_h_field_nested_oof.npz"

RANDOM_STATE = 14501
K_VALUES = (2, 4, 8)
STRENGTHS = (0.15, 0.25, 0.40, 0.60)
BADNESS_QUANTILES = (0.0, 0.40, 0.60, 0.75)
DISPERSION_QUANTILES = (0.0, 0.50, 0.75)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
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


def stable_features(
    prefix: str,
    positions: np.ndarray,
    blocked: np.ndarray,
    current: np.ndarray,
    geodesic: np.lib.npyio.NpzFile,
    pointcloud: np.lib.npyio.NpzFile,
) -> np.ndarray:
    basic, _ = basic_features(positions, blocked)
    values = np.concatenate(
        [
            basic,
            geodesic[f"{prefix}_features"].astype(np.float32),
            pointcloud[f"{prefix}_features"].astype(np.float32),
            support_features(current),
        ],
        axis=1,
    )
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def robust_scale(
    train: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(train, axis=0, keepdims=True)
    scale = np.quantile(train, 0.75, axis=0, keepdims=True) - np.quantile(
        train, 0.25, axis=0, keepdims=True
    )
    scale = np.maximum(scale, 1e-3)
    return (
        np.clip((train - median) / scale, -8.0, 8.0).astype(np.float32),
        np.clip((query - median) / scale, -8.0, 8.0).astype(np.float32),
    )


def fit_badness(
    features: np.ndarray,
    badness: np.ndarray,
    train_index: np.ndarray,
    query_index: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features[train_index], badness[train_index])
    return model.predict(features[query_index]).astype(np.float32)


def local_h_predictions(
    positions: np.ndarray,
    h_target: np.ndarray,
    train_index: np.ndarray,
    query_index: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    model = NearestNeighbors(n_neighbors=max(K_VALUES), n_jobs=-1)
    model.fit(positions[train_index, :2])
    distance, local = model.kneighbors(positions[query_index, :2])
    neighbors = train_index[local]
    inverse = 1.0 / np.maximum(distance, 0.25)
    prediction = {}
    dispersion = {}
    maximum_distance = {}
    for k in K_VALUES:
        weight = inverse[:, :k]
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
        values = h_target[neighbors[:, :k]]
        mean = np.sum(weight[..., None] * values, axis=1)
        variance = np.sum(weight[..., None] * (values - mean[:, None]) ** 2, axis=1)
        prediction[k] = mean.astype(np.float32)
        dispersion[k] = np.sqrt(np.mean(variance, axis=1)).astype(np.float32)
        maximum_distance[k] = distance[:, k - 1].astype(np.float32)
    return prediction, dispersion, maximum_distance


def h_gain(
    current_h: np.ndarray,
    truth_h: np.ndarray,
    prediction: np.ndarray,
    gate: np.ndarray,
    strength: float | np.ndarray,
) -> np.ndarray:
    scale = np.asarray(strength, dtype=np.float32)
    if scale.ndim == 0:
        scale = np.full(len(gate), float(scale), dtype=np.float32)
    adjusted = current_h * np.exp(
        np.clip(prediction, -3.0, 3.0)
        * (scale * gate)[:, None]
    )
    adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), TINY)
    return row_cos(adjusted, truth_h) - row_cos(current_h, truth_h)


def config_key(config: dict[str, float | int]) -> str:
    return (
        f"k={config['k']};strength={config['strength']:.2f};"
        f"bad_q={config['badness_quantile']:.2f};"
        f"disp_q={config['dispersion_quantile']:.2f}"
    )


def choose_config(
    current_h: np.ndarray,
    truth_h: np.ndarray,
    predictions: dict[int, np.ndarray],
    dispersions: dict[int, np.ndarray],
    maximum_distances: dict[int, np.ndarray],
    predicted_badness: np.ndarray,
    selected: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, object]]:
    rows = []
    for k in K_VALUES:
        bad_values = predicted_badness[selected]
        dispersion_values = dispersions[k][selected]
        for strength in STRENGTHS:
            for bad_q in BADNESS_QUANTILES:
                bad_threshold = (
                    -np.inf
                    if bad_q == 0.0
                    else float(np.quantile(bad_values, bad_q))
                )
                for dispersion_q in DISPERSION_QUANTILES:
                    dispersion_threshold = (
                        np.inf
                        if dispersion_q == 0.0
                        else float(np.quantile(dispersion_values, dispersion_q))
                    )
                    gate = (
                        (predicted_badness >= bad_threshold)
                        & (dispersions[k] <= dispersion_threshold)
                        & (maximum_distances[k] <= 20.0)
                    ).astype(np.float32)
                    gain = h_gain(
                        current_h[selected],
                        truth_h[selected],
                        predictions[k][selected],
                        gate[selected],
                        strength,
                    )
                    active = float(np.mean(gate[selected] > 0))
                    active_mask = gate[selected] > 0
                    positive_active = float(
                        np.mean(gain[active_mask] > 0)
                    ) if np.any(active_mask) else 0.0
                    mean = float(gain.mean())
                    standard_error = float(gain.std() / np.sqrt(len(gain)))
                    rows.append(
                        {
                            "k": k,
                            "strength": strength,
                            "badness_quantile": bad_q,
                            "dispersion_quantile": dispersion_q,
                            "badness_threshold": bad_threshold,
                            "dispersion_threshold": dispersion_threshold,
                            "active_fraction": active,
                            "gain_mean": mean,
                            "gain_std": float(gain.std()),
                            "gain_lcb": mean - 0.50 * standard_error,
                            "positive_fraction": float(np.mean(gain > 0)),
                            "positive_fraction_active": positive_active,
                        }
                    )
    eligible = [
        row
        for row in rows
        if row["active_fraction"] >= 0.10
        and row["positive_fraction_active"] >= 0.505
    ]
    ranked = sorted(eligible or rows, key=lambda row: row["gain_lcb"], reverse=True)
    best = ranked[0]
    return best, {"best": best, "top10": ranked[:10]}


def build_gate(
    predicted_badness: np.ndarray,
    dispersion: np.ndarray,
    maximum_distance: np.ndarray,
    badness_threshold: float,
    dispersion_threshold: float,
) -> np.ndarray:
    return (
        (predicted_badness >= badness_threshold)
        & (dispersion <= dispersion_threshold)
        & (maximum_distance <= 20.0)
    ).astype(np.float32)


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
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(
        PATHS.shared_features / "official_pointcloud_scene_cache.npz"
    )
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    residual_probability = np.load(
        PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
    ).astype(np.float32)
    source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    risk = build_risk(
        regimes_train,
        residual_probability,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    current, _ = apply_risk_shift(source, risk)
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    valid = truth_raw.sum(axis=(1, 2, 3)) > 0
    truth = normalize_without_floor(truth_raw)
    current_h = current.sum(axis=(2, 3))
    truth_h = truth.sum(axis=(2, 3))
    h_target = np.clip(
        np.log(np.maximum(truth_h, 1e-10))
        - np.log(np.maximum(current_h, 1e-10)),
        -3.0,
        3.0,
    ).astype(np.float32)
    badness = 1.0 - support_score(current, truth)
    features_train = stable_features(
        "train",
        positions_train,
        blocked_train,
        current,
        geodesic,
        pointcloud,
    )

    selected = np.flatnonzero(valid)
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions_train[:, :2])
    outer = GroupKFold(n_splits=5)
    nested_prediction = np.zeros_like(h_target)
    nested_gate = np.zeros(len(current), dtype=np.float32)
    nested_badness = np.zeros(len(current), dtype=np.float32)
    nested_dispersion = np.zeros(len(current), dtype=np.float32)
    nested_max_distance = np.zeros(len(current), dtype=np.float32)
    nested_strength = np.zeros(len(current), dtype=np.float32)
    fold_reports = []

    for outer_fold, (outer_train_local, outer_val_local) in enumerate(
        outer.split(selected, groups=groups[selected])
    ):
        outer_train = selected[outer_train_local]
        outer_val = selected[outer_val_local]
        inner_badness = np.zeros(len(current), dtype=np.float32)
        inner_predictions = {
            k: np.zeros_like(h_target, dtype=np.float32) for k in K_VALUES
        }
        inner_dispersions = {
            k: np.zeros(len(current), dtype=np.float32) for k in K_VALUES
        }
        inner_distances = {
            k: np.zeros(len(current), dtype=np.float32) for k in K_VALUES
        }
        inner = GroupKFold(n_splits=4)
        for inner_fold, (train_local, val_local) in enumerate(
            inner.split(outer_train, groups=groups[outer_train])
        ):
            train_index = outer_train[train_local]
            val_index = outer_train[val_local]
            inner_badness[val_index] = fit_badness(
                features_train,
                badness,
                train_index,
                val_index,
                RANDOM_STATE + 100 * outer_fold + inner_fold,
            )
            prediction, dispersion, max_distance = local_h_predictions(
                positions_train, h_target, train_index, val_index
            )
            for k in K_VALUES:
                inner_predictions[k][val_index] = prediction[k]
                inner_dispersions[k][val_index] = dispersion[k]
                inner_distances[k][val_index] = max_distance[k]

        best, search_report = choose_config(
            current_h,
            truth_h,
            inner_predictions,
            inner_dispersions,
            inner_distances,
            inner_badness,
            outer_train,
        )
        outer_badness = fit_badness(
            features_train,
            badness,
            outer_train,
            outer_val,
            RANDOM_STATE + 1000 + outer_fold,
        )
        prediction, dispersion, max_distance = local_h_predictions(
            positions_train, h_target, outer_train, outer_val
        )
        k = int(best["k"])
        gate = build_gate(
            outer_badness,
            dispersion[k],
            max_distance[k],
            float(best["badness_threshold"]),
            float(best["dispersion_threshold"]),
        )
        nested_prediction[outer_val] = prediction[k]
        nested_gate[outer_val] = gate
        nested_badness[outer_val] = outer_badness
        nested_dispersion[outer_val] = dispersion[k]
        nested_max_distance[outer_val] = max_distance[k]
        nested_strength[outer_val] = float(best["strength"])
        fold_gain = h_gain(
            current_h[outer_val],
            truth_h[outer_val],
            prediction[k],
            gate,
            float(best["strength"]),
        )
        fold_reports.append(
            {
                "fold": outer_fold,
                "train_count": int(len(outer_train)),
                "validation_count": int(len(outer_val)),
                "selected_config": best,
                "selected_key": config_key(best),
                "validation_h_gain": summarize(fold_gain),
                "validation_positive_fraction": float(np.mean(fold_gain > 0)),
                "validation_active_fraction": float(np.mean(gate > 0)),
                "inner_search": search_report,
            }
        )

    ratio = np.exp(
        np.clip(nested_prediction, -3.0, 3.0)
        * (nested_strength * nested_gate)[:, None]
    )
    nested_adjusted = (current * ratio[:, :, None, None]).astype(np.float32)
    nested_adjusted /= np.maximum(
        nested_adjusted.sum(axis=(1, 2, 3), keepdims=True), TINY
    )
    nested_gain = support_score(nested_adjusted, truth) - support_score(current, truth)
    active_valid = valid & (nested_gate > 0)
    h_nested_gain = h_gain(
        current_h[selected],
        truth_h[selected],
        nested_prediction[selected],
        nested_gate[selected],
        nested_strength[selected],
    )

    config_counts = Counter(row["selected_key"] for row in fold_reports)
    modal_key, modal_count = config_counts.most_common(1)[0]
    modal = next(
        row["selected_config"]
        for row in fold_reports
        if row["selected_key"] == modal_key
    )

    test_current = normalize_support(
        np.load(
            PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
        ),
        1e-12,
    )
    features_test = stable_features(
        "test",
        positions_test,
        blocked_test,
        test_current,
        geodesic,
        pointcloud,
    )
    test_badness = fit_badness(
        np.concatenate([features_train, features_test], axis=0),
        np.concatenate([badness, np.zeros(len(features_test), dtype=np.float32)]),
        selected,
        np.arange(len(features_train), len(features_train) + len(features_test)),
        RANDOM_STATE + 2000,
    )
    test_prediction, test_dispersion, test_distance = local_h_predictions(
        np.concatenate([positions_train, positions_test], axis=0),
        np.concatenate(
            [h_target, np.zeros((len(positions_test), h_target.shape[1]), dtype=np.float32)],
            axis=0,
        ),
        selected,
        np.arange(len(positions_train), len(positions_train) + len(positions_test)),
    )
    k = int(modal["k"])
    bad_threshold = float(
        np.quantile(nested_badness[selected], float(modal["badness_quantile"]))
        if float(modal["badness_quantile"]) > 0
        else -np.inf
    )
    dispersion_threshold = float(
        np.quantile(
            nested_dispersion[selected], float(modal["dispersion_quantile"])
        )
        if float(modal["dispersion_quantile"]) > 0
        else np.inf
    )
    test_gate = build_gate(
        test_badness,
        test_dispersion[k],
        test_distance[k],
        bad_threshold,
        dispersion_threshold,
    )

    np.savez_compressed(
        OOF_OUT,
        prediction=nested_prediction.astype(np.float32),
        gate=nested_gate.astype(np.float32),
        predicted_badness=nested_badness.astype(np.float32),
        dispersion=nested_dispersion.astype(np.float32),
        maximum_distance=nested_max_distance.astype(np.float32),
        strength=nested_strength.astype(np.float32),
        test_prediction=test_prediction[k].astype(np.float32),
        test_gate=test_gate.astype(np.float32),
        test_predicted_badness=test_badness.astype(np.float32),
        test_dispersion=test_dispersion[k].astype(np.float32),
        test_maximum_distance=test_distance[k].astype(np.float32),
    )

    by_regime = {}
    for regime in ("clear", "light", "medium", "heavy"):
        mask = valid & (regimes_train == regime)
        by_regime[regime] = {
            "count": int(mask.sum()),
            "gain": summarize(nested_gain[mask]),
            "active_fraction": float(np.mean(nested_gate[mask] > 0)),
        }
    report = {
        "method": "v145_gated_local_h_field_nested_spatial_oof_audit",
        "status": "analysis_only",
        "base": "v140 online 0.501352",
        "primary_hypothesis": (
            "H residuals form a local spatial field, but correction should be "
            "limited to points with predicted high error and agreeing neighbors"
        ),
        "excluded": {
            "v144_v_correction": True,
            "residual_probability_features": True,
            "delay_correction": True,
            "complex_phase_correction": True,
        },
        "stable_feature_dim": int(features_train.shape[1]),
        "nested_oof": {
            "outer_folds": 5,
            "inner_folds": 4,
            "h_marginal_gain": summarize(h_nested_gain),
            "full_support_gain": summarize(nested_gain[valid]),
            "positive_fraction": float(np.mean(nested_gain[valid] > 0)),
            "active_fraction": float(np.mean(nested_gate[valid] > 0)),
            "active_gain": summarize(nested_gain[active_valid]),
            "active_positive_fraction": float(
                np.mean(nested_gain[active_valid] > 0)
            ),
            "by_regime": by_regime,
            "folds": fold_reports,
        },
        "config_stability": {
            "counts": dict(config_counts),
            "modal_config": modal,
            "modal_key": modal_key,
            "modal_fold_count": int(modal_count),
        },
        "test_audit": {
            "modal_k": k,
            "modal_strength": float(modal["strength"]),
            "badness_threshold": bad_threshold,
            "dispersion_threshold": dispersion_threshold,
            "active_fraction": float(np.mean(test_gate > 0)),
            "prediction": summarize(test_prediction[k]),
            "predicted_badness": summarize(test_badness),
            "dispersion": summarize(test_dispersion[k]),
            "maximum_neighbor_distance": summarize(test_distance[k]),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "oof_arrays": PATHS.relative(OOF_OUT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {REPORT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
