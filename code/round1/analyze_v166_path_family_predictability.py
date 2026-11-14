#!/usr/bin/env python3
"""Analysis-only audit for path-family predictability.

This does not create a submission.  It tests whether path-family labels
inferred from training truth AD components are predictable from official/Sionna
geometry features.  The goal is to decide whether a future v166 should use
path-family information as a selector, instead of directly forcing path
templates into the current best output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from generate_v87_easy_field_projection import summarize
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


OUT_DIR = PATHS.analysis / "results" / "version_reports" / "v166"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "v166_path_family_predictability_analysis.json"
PRED_OUT = PATHS.shared_features / "v166_path_family_predictability_predictions.npz"


def finite_block(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def path_summary(npz: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    valid = npz[f"{prefix}_valid"].astype(bool)
    ptype = npz[f"{prefix}_path_type"]
    delay = finite_block(npz[f"{prefix}_delay_bin"])
    score = finite_block(npz[f"{prefix}_score"])
    excess = finite_block(npz[f"{prefix}_excess_length"])
    incidence = finite_block(npz[f"{prefix}_incidence_cos"])
    edge = finite_block(npz[f"{prefix}_edge_margin"])
    tx = finite_block(npz[f"{prefix}_tx_dir"])
    rx = finite_block(npz[f"{prefix}_rx_dir"])

    n = valid.shape[0]
    rows = np.arange(n)
    out: list[np.ndarray] = []
    out.append(valid.mean(axis=1))
    for family in (0, 1, 2):
        mask = valid & (ptype == family)
        count = mask.sum(axis=1).astype(np.float32)
        safe_score = np.where(mask, score, -1e9)
        best = np.argmax(safe_score, axis=1)
        has = count > 0
        out.extend(
            [
                count / max(1, valid.shape[1]),
                np.where(has, safe_score[rows, best], -8.0) / 4.0,
                np.where(has, delay[rows, best], 0.0) / 16.0,
                np.log1p(np.where(has, np.maximum(excess[rows, best], 0.0), 0.0)) / 7.0,
                np.where(has, incidence[rows, best], 0.0),
                np.where(has, edge[rows, best], 0.0) / 0.35,
            ]
        )
        out.append(np.where(has[:, None], tx[rows, best], 0.0))
        out.append(np.where(has[:, None], rx[rows, best], 0.0))

    # Coarse ordered delays/scores preserve path-multiplicity shape.
    d = np.where(valid, delay, 0.0)
    s = np.where(valid, np.clip(score, -8.0, 4.0), -8.0)
    out.append(d[:, : min(6, d.shape[1])] / 16.0)
    out.append(s[:, : min(6, s.shape[1])] / 4.0)
    return finite_block(np.concatenate([a if a.ndim == 2 else a[:, None] for a in out], axis=1))


def base_features(prefix: str) -> np.ndarray:
    pos = np.load(PATHS.round1 / f"Round1_{'Train' if prefix == 'train' else 'Test'}_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = height[f"{prefix}_blocked"].astype(np.float32)
    d_bs = np.linalg.norm(pos - BS_POS[None, :], axis=1).astype(np.float32)
    az = np.arctan2(pos[:, 1] - BS_POS[1], pos[:, 0] - BS_POS[0]).astype(np.float32)
    scalars = np.stack(
        [
            (pos[:, 0] - 150.0) / 100.0,
            (pos[:, 1] + 55.0) / 120.0,
            d_bs / 220.0,
            np.sin(az),
            np.cos(az),
            np.log1p(blocked) / np.log(220.0),
            (blocked <= 0).astype(np.float32),
            ((blocked > 0) & (blocked <= 25)).astype(np.float32),
            ((blocked > 25) & (blocked <= 100)).astype(np.float32),
            (blocked > 100).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return scalars


def build_feature_matrix(prefix: str) -> tuple[np.ndarray, list[str]]:
    blocks = [base_features(prefix)]
    names = [f"base:{i}" for i in range(blocks[-1].shape[1])]

    for npz_path, key, label in [
        (PATHS.shared_features / "geodesic_support_features.npz", f"{prefix}_features", "geodesic"),
        (PATHS.shared_features / "official_pointcloud_scene_cache.npz", f"{prefix}_features", "pointcloud"),
        (PATHS.shared_features / "sionna_etoile_features.npz", f"{prefix}_features", "sionna_scene"),
    ]:
        z = np.load(npz_path, allow_pickle=True)
        block = finite_block(z[key])
        blocks.append(block)
        names.extend([f"{label}:{i}" for i in range(block.shape[1])])

    blocks.append(path_summary(np.load(PATHS.shared_features / "sionna_path_candidates.npz"), prefix))
    names.extend([f"sionna_path:{i}" for i in range(blocks[-1].shape[1])])
    blocks.append(path_summary(np.load(PATHS.shared_features / "multibounce_path_candidates.npz"), prefix))
    names.extend([f"multibounce_path:{i}" for i in range(blocks[-1].shape[1])])

    return finite_block(np.concatenate(blocks, axis=1)), names


def path_family_labels() -> dict[str, np.ndarray]:
    z = np.load(PATHS.shared_features / "multibounce_component_path_assignments.npz")
    power = z["component_power"].astype(np.float32)
    total = z["total_power"].astype(np.float32)
    path_type = z["path_type"]
    best = z["best_path"]
    rows = np.arange(best.shape[0])[:, None]
    best_type = path_type[rows, best]
    assigned = (
        (z["delay_err"].astype(np.float32) <= 2.0)
        & (z["h_err"].astype(np.float32) <= 3.0)
        & (z["v_err"].astype(np.float32) <= 2.0)
        & (total[:, None] > 0.0)
    )
    denom = np.maximum(total, TINY)
    labels: dict[str, np.ndarray] = {}
    labels["assigned_power_frac"] = (power * assigned).sum(axis=1) / denom
    for family, name in [(0, "los"), (1, "one"), (2, "two")]:
        mask = assigned & (best_type == family)
        labels[f"{name}_power_frac"] = (power * mask).sum(axis=1) / denom
        labels[f"{name}_component_frac"] = mask.mean(axis=1).astype(np.float32)
    top0_assigned = assigned[:, 0]
    labels["top1_two"] = (top0_assigned & (best_type[:, 0] == 2)).astype(np.float32)
    labels["top1_one"] = (top0_assigned & (best_type[:, 0] == 1)).astype(np.float32)
    labels["top1_los"] = (top0_assigned & (best_type[:, 0] == 0)).astype(np.float32)
    labels["valid_nonzero"] = (total > 0.0).astype(np.float32)
    return {k: finite_block(v) for k, v in labels.items()}


def grouped_folds(pos: np.ndarray) -> np.ndarray:
    km = KMeans(n_clusters=25, random_state=166, n_init=10)
    return km.fit_predict(pos[:, :2])


def regression_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    pred = np.zeros_like(y, dtype=np.float32)
    valid = np.isfinite(y)
    for train_idx, valid_idx in GroupKFold(n_splits=5).split(x, y, groups):
        train_idx = train_idx[valid[train_idx]]
        model = HistGradientBoostingRegressor(
            max_iter=240,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=166,
        )
        model.fit(x[train_idx], y[train_idx])
        pred[valid_idx] = model.predict(x[valid_idx]).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)
    rho = spearmanr(y[valid], pred[valid]).correlation
    return pred, {
        "mean": float(np.mean(y[valid])),
        "pred_mean": float(np.mean(pred[valid])),
        "r2": float(r2_score(y[valid], pred[valid])),
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "mae": float(np.mean(np.abs(y[valid] - pred[valid]))),
        "top_decile_label_mean": float(np.mean(y[pred >= np.quantile(pred, 0.90)])),
        "bottom_decile_label_mean": float(np.mean(y[pred <= np.quantile(pred, 0.10)])),
    }


def classification_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    pred = np.zeros_like(y, dtype=np.float32)
    for train_idx, valid_idx in GroupKFold(n_splits=5).split(x, y, groups):
        model = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=166,
        )
        model.fit(x[train_idx], y[train_idx].astype(np.int32))
        pred[valid_idx] = model.predict_proba(x[valid_idx])[:, 1].astype(np.float32)
    if len(np.unique(y)) < 2:
        auc = ap = 0.0
    else:
        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
    return pred, {
        "positive_rate": float(np.mean(y)),
        "pred_mean": float(np.mean(pred)),
        "auc": float(auc),
        "average_precision": float(ap),
        "top_decile_positive_rate": float(np.mean(y[pred >= np.quantile(pred, 0.90)])),
        "bottom_decile_positive_rate": float(np.mean(y[pred <= np.quantile(pred, 0.10)])),
    }


def main() -> None:
    x_train, names = build_feature_matrix("train")
    x_test, _ = build_feature_matrix("test")
    scaler = StandardScaler()
    x_train = finite_block(scaler.fit_transform(x_train))
    x_test = finite_block(scaler.transform(x_test))

    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    groups = grouped_folds(pos_train)
    labels = path_family_labels()
    nonzero = labels["valid_nonzero"] > 0
    # Keep exact-zero rows in features but neutralize labels for model fitting.
    x_fit = x_train[nonzero]
    groups_fit = groups[nonzero]

    reg_targets = [
        "assigned_power_frac",
        "los_power_frac",
        "one_power_frac",
        "two_power_frac",
        "two_component_frac",
    ]
    cls_targets = ["top1_los", "top1_one", "top1_two"]

    oof_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for target in reg_targets:
        y = labels[target][nonzero].astype(np.float32)
        pred, metric = regression_oof(x_fit, y, groups_fit)
        full = np.zeros(len(x_train), dtype=np.float32)
        full[nonzero] = pred
        oof_predictions[target] = full
        metrics[target] = metric
        model = HistGradientBoostingRegressor(
            max_iter=280,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=166,
        )
        model.fit(x_fit, y)
        test_predictions[target] = np.clip(model.predict(x_test), 0.0, 1.0).astype(np.float32)

    for target in cls_targets:
        y = labels[target][nonzero].astype(np.float32)
        pred, metric = classification_oof(x_fit, y, groups_fit)
        full = np.zeros(len(x_train), dtype=np.float32)
        full[nonzero] = pred
        oof_predictions[target] = full
        metrics[target] = metric
        model = HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=166,
        )
        model.fit(x_fit, y.astype(np.int32))
        test_predictions[target] = model.predict_proba(x_test)[:, 1].astype(np.float32)

    # Candidate selector diagnostics for a future version.
    two_pred = oof_predictions["two_power_frac"][nonzero]
    two_label = labels["two_power_frac"][nonzero]
    top_q = {
        f"q{int(q * 100)}": {
            "threshold": float(np.quantile(two_pred, q)),
            "count": int(np.sum(two_pred >= np.quantile(two_pred, q))),
            "label_mean": float(np.mean(two_label[two_pred >= np.quantile(two_pred, q)])),
            "label_p90": float(np.quantile(two_label[two_pred >= np.quantile(two_pred, q)], 0.90)),
        }
        for q in (0.80, 0.85, 0.90, 0.95)
    }

    np.savez_compressed(
        PRED_OUT,
        train_nonzero=nonzero,
        train_groups=groups,
        feature_names=np.asarray(names, dtype="U48"),
        **{f"oof_{k}": v for k, v in oof_predictions.items()},
        **{f"test_{k}": v for k, v in test_predictions.items()},
        **{f"label_{k}": v for k, v in labels.items()},
    )

    report = {
        "method": "v166_path_family_predictability_analysis",
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Path-family mixtures inferred from train truth AD components are "
            "useful only if official/Sionna geometry can predict them in spatial OOF."
        ),
        "inputs": {
            "assignments": PATHS.relative(PATHS.shared_features / "multibounce_component_path_assignments.npz"),
            "sionna_paths": PATHS.relative(PATHS.shared_features / "sionna_path_candidates.npz"),
            "multibounce_paths": PATHS.relative(PATHS.shared_features / "multibounce_path_candidates.npz"),
            "official_pointcloud": PATHS.relative(PATHS.shared_features / "official_pointcloud_scene_cache.npz"),
        },
        "feature_count": int(x_train.shape[1]),
        "train_nonzero_count": int(np.sum(nonzero)),
        "fold_group_count": int(len(np.unique(groups))),
        "label_summary": {key: summarize(value[nonzero]) for key, value in labels.items() if key != "valid_nonzero"},
        "metrics": metrics,
        "two_power_top_quantiles": top_q,
        "test_prediction_summary": {key: summarize(value) for key, value in test_predictions.items()},
        "outputs": {
            "predictions": PATHS.relative(PRED_OUT),
            "report": PATHS.relative(REPORT),
        },
        "interpretation": {
            "two_bounce_selector": (
                "If two_power_frac Spearman/AUC-like enrichment is weak, use multibounce only "
                "as a diagnostic or conservative gate, not as a direct target."
            ),
            "next_candidate_rule": (
                "A v166 submission is justified only if high-confidence predicted two-bounce "
                "regions are small and enriched, so they can gate support diffusion without "
                "overriding v163/v165 globally."
            ),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
