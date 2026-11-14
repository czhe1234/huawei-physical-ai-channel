#!/usr/bin/env python3
"""Audit capacity and learnability of independent log-support H drift/diffusion."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from analyze_v140_physical_mechanism import (
    normalize_support,
    normalize_truth,
    row_cos,
    summarize,
)
from generate_v140_balanced_hminus2_overdrive import build_risk, support_features
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(140)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v140_drift_diffusion_capacity_analysis.json"
ARRAYS = REPORT_DIR / "v140_drift_diffusion_capacity_arrays.npz"

RANDOM_STATE = 14099
U_GRID = np.asarray([-0.50, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50], dtype=np.float32)
D_GRID = np.asarray([0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50], dtype=np.float32)
REGIMES = ("clear", "light", "medium", "heavy")


def safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = mask & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or x[valid].std() == 0 or y[valid].std() == 0:
        return {"count": int(valid.sum()), "pearson": 0.0, "spearman": 0.0}
    return {
        "count": int(valid.sum()),
        "pearson": float(pearsonr(x[valid], y[valid]).statistic),
        "spearman": float(spearmanr(x[valid], y[valid]).statistic),
    }


def pool_coarse(values: np.ndarray) -> np.ndarray:
    # Preserve H exactly; pool V by 2 and delay by 8: (16, 8, 192) -> (16, 4, 24).
    pooled = values.reshape(len(values), 16, 4, 2, 24, 8).sum(axis=(3, 5))
    return (
        pooled / np.maximum(pooled.sum(axis=(1, 2, 3), keepdims=True), TINY)
    ).astype(np.float32)


def log_derivatives(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.log(np.maximum(values, 1e-12)).astype(np.float32)
    gradient = 0.5 * (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1))
    laplacian = np.roll(z, -1, axis=1) + np.roll(z, 1, axis=1) - 2.0 * z
    return z, gradient.astype(np.float32), laplacian.astype(np.float32)


def transform(values: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
    z, gradient, laplacian = log_derivatives(values)
    shape = (len(values),) + (1,) * (values.ndim - 1)
    output = np.exp(
        z + u.reshape(shape) * gradient + d.reshape(shape) * laplacian
    ).astype(np.float32)
    output /= np.maximum(
        output.sum(axis=tuple(range(1, output.ndim)), keepdims=True), TINY
    )
    return output


def grid_oracle(source: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    z, gradient, laplacian = log_derivatives(source)
    base = row_cos(source, truth)
    scores = np.empty((len(source), len(U_GRID), len(D_GRID)), dtype=np.float32)
    for u_index, u_value in enumerate(U_GRID):
        for d_index, d_value in enumerate(D_GRID):
            output = np.exp(z + u_value * gradient + d_value * laplacian).astype(np.float32)
            output /= np.maximum(
                output.sum(axis=tuple(range(1, output.ndim)), keepdims=True), TINY
            )
            scores[:, u_index, d_index] = row_cos(output, truth)
    flat = scores.reshape(len(source), -1)
    best_index = np.argmax(flat, axis=1)
    best_u = U_GRID[best_index // len(D_GRID)]
    best_d = D_GRID[best_index % len(D_GRID)]
    zero_u = int(np.flatnonzero(U_GRID == 0.0)[0])
    return {
        "base_score": base,
        "best_score": flat[np.arange(len(source)), best_index],
        "best_u": best_u.astype(np.float32),
        "best_d": best_d.astype(np.float32),
        "drift_score": scores[:, :, 0].max(axis=1),
        "diffusion_score": scores[:, zero_u, :].max(axis=1),
    }


def full_scores(
    source: np.ndarray,
    truth: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    scores = np.empty(len(source), dtype=np.float32)
    for start in range(0, len(source), 32):
        end = min(start + 32, len(source))
        output = transform(source[start:end], u[start:end], d[start:end])
        scores[start:end] = row_cos(output, truth[start:end])
    return scores


def coefficient_counts(values: np.ndarray, grid: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {
        f"{float(value):.2f}": int(np.sum(mask & np.isclose(values, value)))
        for value in grid
    }


def make_features(
    positions: np.ndarray,
    blocked: np.ndarray,
    geodesic: np.ndarray,
    pointcloud: np.ndarray,
    probability: np.ndarray,
    source: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    vector = positions - BS_POS[None, :]
    distance = np.linalg.norm(vector, axis=1)
    azimuth = np.arctan2(vector[:, 1], vector[:, 0])
    basic = np.column_stack(
        [
            positions[:, 0],
            positions[:, 1],
            distance,
            np.sin(azimuth),
            np.cos(azimuth),
            blocked,
        ]
    ).astype(np.float32)
    physical = np.concatenate([basic, geodesic, pointcloud], axis=1).astype(np.float32)
    full = np.concatenate(
        [physical, probability, support_features(source)], axis=1
    ).astype(np.float32)
    features = {
        "physical_94": np.nan_to_num(physical, nan=0.0, posinf=8.0, neginf=-8.0),
        "full_519": np.nan_to_num(full, nan=0.0, posinf=8.0, neginf=-8.0),
    }
    return features, {name: int(value.shape[1]) for name, value in features.items()}


def model_oof(
    features: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    prediction = np.zeros_like(labels, dtype=np.float32)
    selected = np.flatnonzero(valid)
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions[:, :2])
    for train_local, validation_local in GroupKFold(n_splits=5).split(
        selected, groups=groups[selected]
    ):
        train = selected[train_local]
        validation = selected[validation_local]
        estimator = ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.65,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        estimator.fit(features[train], labels[train])
        prediction[validation] = estimator.predict(features[validation]).astype(np.float32)
    return prediction


def local_oof(
    labels: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.zeros_like(labels, dtype=np.float32)
    active = np.zeros(len(labels), dtype=bool)
    selected = np.flatnonzero(valid)
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions[:, :2])
    for train_local, validation_local in GroupKFold(n_splits=5).split(
        selected, groups=groups[selected]
    ):
        train = selected[train_local]
        validation = selected[validation_local]
        tree = cKDTree(positions[train, :2])
        distance, index = tree.query(positions[validation, :2], k=8, workers=-1)
        weight = 1.0 / np.maximum(distance, 0.25) ** 2.0
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), TINY)
        prediction[validation] = np.sum(
            weight[:, :, None] * labels[train[index]], axis=1
        ).astype(np.float32)
        active[validation] = distance[:, -1] <= 20.0
    prediction[~active] = 0.0
    return prediction, active


def gain_summary(gain: np.ndarray, valid: np.ndarray, regimes: np.ndarray) -> dict[str, object]:
    return {
        "all": summarize(gain[valid]),
        "positive_fraction": float(np.mean(gain[valid] > 0)),
        "by_regime": {
            name: summarize(gain[valid & (regimes == name)]) for name in REGIMES
        },
    }


def prediction_report(
    prediction: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    return {
        "u": safe_corr(prediction[:, 0], labels[:, 0], valid),
        "d": safe_corr(prediction[:, 1], labels[:, 1], valid),
        "full_gain": safe_corr(prediction[:, 2], labels[:, 2], valid),
        "u_sign_accuracy_nonzero": float(
            np.mean(
                np.sign(prediction[valid & (np.abs(labels[:, 0]) >= 0.10), 0])
                == np.sign(labels[valid & (np.abs(labels[:, 0]) >= 0.10), 0])
            )
        ),
    }


def evaluate_prediction(
    source: np.ndarray,
    truth: np.ndarray,
    base_score: np.ndarray,
    coarse_source: np.ndarray,
    coarse_truth: np.ndarray,
    coarse_base_score: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    regimes: np.ndarray,
    active: np.ndarray | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {}
    candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    if active is None:
        active = np.ones(len(source), dtype=bool)
    valid_prediction = valid & active
    for strength in (0.25, 0.50, 1.00):
        u = np.clip(strength * prediction[:, 0], U_GRID.min(), U_GRID.max())
        d = np.clip(strength * prediction[:, 1], D_GRID.min(), D_GRID.max())
        for gate_name, gate in (
            ("all", valid_prediction),
            ("predicted_positive", valid_prediction & (prediction[:, 2] > 0)),
            (
                "predicted_top50",
                valid_prediction
                & (prediction[:, 2] >= np.quantile(prediction[valid_prediction, 2], 0.50)),
            ),
            (
                "predicted_top25",
                valid_prediction
                & (prediction[:, 2] >= np.quantile(prediction[valid_prediction, 2], 0.75)),
            ),
            ):
            gated_u = np.where(gate, u, 0.0).astype(np.float32)
            gated_d = np.where(gate, d, 0.0).astype(np.float32)
            coarse_output = transform(coarse_source, gated_u, gated_d)
            coarse_score = row_cos(coarse_output, coarse_truth)
            key = f"strength_{strength:.2f}_{gate_name}"
            coarse_gain = coarse_score - coarse_base_score
            report[key] = {
                "active_fraction": float(np.mean(gate[valid])),
                "coarse_gain": gain_summary(coarse_gain, valid, regimes),
            }
            candidates.append(
                (float(np.mean(coarse_gain[valid])), key, gated_u, gated_d)
            )
    for _, key, gated_u, gated_d in sorted(candidates, reverse=True)[:3]:
        score = full_scores(source, truth, gated_u, gated_d)
        report[key]["full_support_gain"] = gain_summary(
            score - base_score, valid, regimes
        )
    return report


def main() -> None:
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = height["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    geodesic_npz = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    geodesic_names = [str(value) for value in geodesic_npz["feature_names"]]
    geodesic = geodesic_npz["train_features"].astype(np.float32)
    pointcloud = np.load(PATHS.shared_features / "official_pointcloud_scene_cache.npz")[
        "train_features"
    ].astype(np.float32)
    probability = np.load(
        PATHS.shared_features / "train_v129_testlike_residual_probs.npy"
    ).astype(np.float32)
    source = normalize_support(
        np.load(PATHS.shared_features / "train_v129_testlike_target.npy")
    )
    raw_truth = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    valid = raw_truth.sum(axis=(1, 2, 3)) > 0
    truth = normalize_truth(raw_truth)
    risk = build_risk(regimes, probability, geodesic, geodesic_names)

    coarse_source = pool_coarse(source)
    coarse_truth = pool_coarse(truth)
    oracle = grid_oracle(coarse_source, coarse_truth)
    base_full_score = row_cos(source, truth)
    oracle_full_score = full_scores(
        source, truth, oracle["best_u"], oracle["best_d"]
    )
    oracle_full_gain = oracle_full_score - base_full_score
    labels = np.column_stack(
        [oracle["best_u"], oracle["best_d"], oracle_full_gain]
    ).astype(np.float32)

    features, feature_dimensions = make_features(
        positions, blocked, geodesic, pointcloud, probability, source
    )
    predictions: dict[str, np.ndarray] = {}
    model_reports: dict[str, object] = {}
    for name, values in features.items():
        prediction = model_oof(values, labels, positions, valid)
        predictions[name] = prediction
        model_reports[name] = {
            "predictability": prediction_report(prediction, labels, valid),
            "application": evaluate_prediction(
                source,
                truth,
                base_full_score,
                coarse_source,
                coarse_truth,
                oracle["base_score"],
                prediction,
                valid,
                regimes,
            ),
        }

    local_prediction, local_active = local_oof(labels, positions, valid)
    predictions["xy_k8_20m"] = local_prediction
    model_reports["xy_k8_20m"] = {
        "active_fraction": float(np.mean(local_active[valid])),
        "predictability": prediction_report(
            local_prediction, labels, valid & local_active
        ),
        "application": evaluate_prediction(
            source,
            truth,
            base_full_score,
            coarse_source,
            coarse_truth,
            oracle["base_score"],
            local_prediction,
            valid,
            regimes,
            local_active,
        ),
    }

    w1 = np.where(risk > 0.64, 0.165 * risk, 0.0).astype(np.float32)
    w2 = np.where(risk > 0.90, 0.060 * risk, 0.0).astype(np.float32)
    v140_u = w1 + 2.0 * w2
    v140_d = 0.5 * w1 + 2.0 * w2

    np.savez_compressed(
        ARRAYS,
        valid=valid,
        risk=risk,
        best_u=oracle["best_u"],
        best_d=oracle["best_d"],
        coarse_gain=oracle["best_score"] - oracle["base_score"],
        full_gain=oracle_full_gain,
        local_active=local_active,
        **{f"prediction_{name}": value for name, value in predictions.items()},
    )

    report = {
        "method": "v140_log_support_drift_diffusion_capacity",
        "status": "analysis_only_no_candidate",
        "base": "test-like train v129 target reconstructed in section 140",
        "operator": (
            "z*=z+u*0.5*(roll_-1(z)-roll_+1(z))"
            "+d*(roll_-1(z)+roll_+1(z)-2*z); d>=0; normalize(exp(z*))"
        ),
        "grid": {"u": U_GRID.tolist(), "d": D_GRID.tolist()},
        "limitations": (
            "Oracle selection uses coarse joint support (H=16,V=4,D=24) and is then applied once "
            "to full support. The reconstructed train source still has the documented source-domain "
            "gap, so online transfer is not claimed."
        ),
        "data": {
            "count": int(len(source)),
            "valid_nonzero": int(valid.sum()),
            "source_shape": list(source.shape),
            "coarse_shape": list(coarse_source.shape),
            "feature_dimensions": feature_dimensions,
        },
        "oracle_capacity": {
            "coarse_base_score": summarize(oracle["base_score"][valid]),
            "drift_only_gain": summarize(
                (oracle["drift_score"] - oracle["base_score"])[valid]
            ),
            "diffusion_only_gain": summarize(
                (oracle["diffusion_score"] - oracle["base_score"])[valid]
            ),
            "joint_coarse_gain": summarize(
                (oracle["best_score"] - oracle["base_score"])[valid]
            ),
            "coarse_selected_full_support_gain": gain_summary(
                oracle_full_gain, valid, regimes
            ),
            "u_counts": coefficient_counts(oracle["best_u"], U_GRID, valid),
            "d_counts": coefficient_counts(oracle["best_d"], D_GRID, valid),
            "by_regime": {
                name: {
                    "u": summarize(oracle["best_u"][valid & (regimes == name)]),
                    "d": summarize(oracle["best_d"][valid & (regimes == name)]),
                    "full_gain": summarize(
                        oracle_full_gain[valid & (regimes == name)]
                    ),
                }
                for name in REGIMES
            },
        },
        "v140_effective_taylor_coefficients": {
            "definition": "u~=w1+2*w2; d~=0.5*w1+2*w2",
            "u": summarize(v140_u),
            "d": summarize(v140_d),
            "correlations": {
                "risk_vs_oracle_abs_u": safe_corr(
                    risk, np.abs(oracle["best_u"]), valid
                ),
                "risk_vs_oracle_d": safe_corr(risk, oracle["best_d"], valid),
                "risk_vs_oracle_full_gain": safe_corr(
                    risk, oracle_full_gain, valid
                ),
                "v140_u_vs_oracle_u": safe_corr(
                    v140_u, oracle["best_u"], valid
                ),
                "v140_d_vs_oracle_d": safe_corr(
                    v140_d, oracle["best_d"], valid
                ),
            },
        },
        "spatial_oof": model_reports,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "arrays": PATHS.relative(ARRAYS),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"wrote {ARRAYS}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
