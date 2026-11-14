#!/usr/bin/env python3
"""v199 analysis-only: audit whether v196 total power is trustworthy.

This script does not generate a submission.  It answers the methodological
question raised after v198:

    Is v196/base per-point total power reliable enough to keep fixed?

The audit has two parts:

1. Train OOF: can true total power be predicted from only official visible
   information such as UE position, BS distance, blocked count and regime?
2. Test diagnostics: how does v196 total power compare with train-kNN power,
   geometry model power and test-test local power consensus?

The output classifies test points as possible power-up, power-down, conflict
or no-action.  These labels are diagnostic only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import generate_v178_test_track_complex_consensus as track
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_v87_easy_field_projection import summarize
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


VERSION = "v199_power_field_reliability_audit"
OUT_DIR = PATHS.cache_dir(199)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(199)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OOF_OUT = OUT_DIR / "v199_train_power_oof.npz"
TEST_DIAG_OUT = OUT_DIR / "v199_test_power_diagnostics.npz"
SUSPECTS_CSV = REPORT_DIR / "v199_power_suspect_points.csv"
REPORT = REPORT_DIR / "v199_power_field_reliability_audit.json"

BASE_V196 = PATHS.cache_dir(196) / "v196_Round1_Test_Channel.npy"
BASE_V189 = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
BASE_V198 = PATHS.cache_dir(198) / "v198_Round1_Test_Channel.npy"

K_TRAIN = 72
K_TEST = 96
RANDOM_STATE = 199


@dataclass
class NeighborResult:
    idx: np.ndarray
    weights: np.ndarray
    confidence: np.ndarray
    first_effective_distance: np.ndarray


def safe_corr(y: np.ndarray, p: np.ndarray, kind: str) -> float:
    if len(y) < 3 or float(np.std(y)) < 1e-12 or float(np.std(p)) < 1e-12:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(y, p).statistic)
    if kind == "spearman":
        return float(spearmanr(y, p).statistic)
    raise ValueError(kind)


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    err = p - y
    abs_err = np.abs(err)
    return {
        "count": int(len(y)),
        "mae_log": float(mean_absolute_error(y, p)),
        "rmse_log": float(np.sqrt(np.mean(err * err))),
        "bias_log": float(np.mean(err)),
        "p50_abs_log": float(np.quantile(abs_err, 0.50)),
        "p90_abs_log": float(np.quantile(abs_err, 0.90)),
        "r2": float(r2_score(y, p)) if len(y) >= 2 else float("nan"),
        "pearson": safe_corr(y, p, "pearson"),
        "spearman": safe_corr(y, p, "spearman"),
    }


def metrics_by_regime(
    y: np.ndarray,
    pred: dict[str, np.ndarray],
    regimes: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, p in pred.items():
        item: dict[str, object] = {"overall": regression_metrics(y[valid], p[valid])}
        for regime in ["clear", "light", "medium", "heavy"]:
            mask = valid & (regimes.astype(str) == regime)
            item[regime] = regression_metrics(y[mask], p[mask]) if np.any(mask) else {"count": 0}
        out[name] = item
    return out


def feature_matrix(pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    b = np.asarray(blocked, dtype=np.float64)
    rel = p - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    regimes = regime_of(blocked)
    one_hot = np.stack(
        [regimes == "clear", regimes == "light", regimes == "medium", regimes == "heavy"],
        axis=1,
    ).astype(np.float64)
    return np.column_stack(
        [
            p / 220.0,
            rel / 220.0,
            dist / 320.0,
            np.log1p(dist),
            np.sin(az),
            np.cos(az),
            b / 220.0,
            np.log1p(b),
            (dist * b) / 60000.0,
            one_hot,
        ]
    ).astype(np.float32)


def effective_neighbor_weights(
    query_pos: np.ndarray,
    ref_pos: np.ndarray,
    query_blocked: np.ndarray,
    ref_blocked: np.ndarray,
    query_regimes: np.ndarray,
    ref_regimes: np.ndarray,
    valid_ref: np.ndarray,
    k: int,
    exclude_self: bool = False,
) -> NeighborResult:
    dxy = np.linalg.norm(query_pos[:, None, :2] - ref_pos[None, :, :2], axis=2).astype(np.float32)
    db = np.abs(query_blocked[:, None].astype(np.float32) - ref_blocked[None, :].astype(np.float32))
    mismatch = (query_regimes[:, None] != ref_regimes[None, :]).astype(np.float32)
    effective = dxy + 0.105 * db + 6.0 * mismatch
    effective[:, ~valid_ref.astype(bool)] = np.inf
    if exclude_self and len(query_pos) == len(ref_pos):
        np.fill_diagonal(effective, np.inf)
    idx = np.argpartition(effective, k, axis=1)[:, :k]
    rows = np.arange(len(query_pos))[:, None]
    idx = idx[rows, np.argsort(effective[rows, idx], axis=1)]
    selected = effective[rows, idx]
    same = (query_regimes[:, None] == ref_regimes[idx]).astype(np.float32)
    weights = np.exp(-selected / 18.0).astype(np.float32) * (0.48 + 0.52 * same)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    confidence = (np.sum(weights * same, axis=1) * np.clip(1.0 - selected[:, 0] / 42.0, 0.0, 1.0)).astype(
        np.float32
    )
    return NeighborResult(
        idx=idx.astype(np.int32),
        weights=weights.astype(np.float32),
        confidence=confidence,
        first_effective_distance=selected[:, 0].astype(np.float32),
    )


def build_models() -> dict[str, object]:
    return {
        "extra": ExtraTreesRegressor(
            n_estimators=260,
            min_samples_leaf=5,
            max_features=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingRegressor(
            max_iter=280,
            learning_rate=0.045,
            max_leaf_nodes=23,
            min_samples_leaf=16,
            l2_regularization=0.04,
            random_state=RANDOM_STATE,
        ),
    }


def cross_fit_models(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    preds = {name: np.full(len(y), np.nan, dtype=np.float32) for name in build_models()}
    fold_id = np.full(len(y), -1, dtype=np.int32)
    valid_idx = np.flatnonzero(valid)
    splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_local, val_local) in enumerate(splitter.split(valid_idx)):
        train_idx = valid_idx[train_local]
        val_idx = valid_idx[val_local]
        fold_id[val_idx] = fold
        for name, model in build_models().items():
            model.fit(x[train_idx], y[train_idx])
            preds[name][val_idx] = model.predict(x[val_idx]).astype(np.float32)
    preds["model_mean"] = np.nanmean(np.stack([preds["extra"], preds["hgb"]]), axis=0).astype(np.float32)
    return preds, {"fold_id": fold_id}


def compute_total_power(path) -> np.ndarray:
    h = np.load(path, mmap_mode="r")
    total = np.empty(len(h), dtype=np.float32)
    for start in range(0, len(h), 8):
        end = min(start + 8, len(h))
        block = np.asarray(h[start:end], dtype=np.complex64)
        power = v196.delay_power_from_h(block)
        total[start:end] = np.maximum(power.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
    return total


def test_power_consensus(log_power: np.ndarray, pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    idx, raw_w = track.neighbor_graph(pos, blocked, regimes, k=K_TEST)
    out = np.empty(len(pos), dtype=np.float32)
    agreement = np.empty(len(pos), dtype=np.float32)
    effective = np.empty(len(pos), dtype=np.float32)
    local_std = np.empty(len(pos), dtype=np.float32)
    local_z = np.empty(len(pos), dtype=np.float32)
    for i in range(len(pos)):
        nb = idx[i].astype(np.int64)
        same = (regimes[nb] == regimes[i]).astype(np.float32)
        w = raw_w[i].astype(np.float32) * (0.50 + 0.50 * same)
        w /= max(float(w.sum()), 1e-12)
        values = log_power[nb].astype(np.float32)
        mean = float(np.sum(w * values))
        var = float(np.sum(w * (values - mean) ** 2))
        out[i] = np.float32(mean)
        local_std[i] = np.float32(np.sqrt(max(var, 1e-12)))
        local_z[i] = np.float32((log_power[i] - mean) / max(float(local_std[i]), 1e-6))
        agreement[i] = np.float32(np.sum(raw_w[i] * same))
        effective[i] = np.float32(1.0 / max(float(np.sum(w * w)), 1e-12))
    return out, {
        "test_neighbor_idx": idx.astype(np.int32),
        "test_neighbor_weights": raw_w.astype(np.float32),
        "test_neighbor_agreement": agreement,
        "test_effective_neighbor_count": effective,
        "test_local_log_std": local_std,
        "test_local_log_z": local_z,
    }


def classify_test_points(
    regimes: np.ndarray,
    base_log: np.ndarray,
    knn_log: np.ndarray,
    model_log: np.ndarray,
    consensus_log: np.ndarray,
    train_conf: np.ndarray,
    test_agreement: np.ndarray,
    model_spread: np.ndarray,
) -> dict[str, np.ndarray]:
    d_knn = knn_log - base_log
    d_model = model_log - base_log
    d_consensus = consensus_log - base_log
    model_agreement = np.exp(-np.abs(knn_log - model_log) / 1.15).astype(np.float32)
    spread_quality = np.clip(1.0 - model_spread / 1.40, 0.0, 1.0).astype(np.float32)
    confidence = np.clip(0.40 * train_conf + 0.25 * test_agreement + 0.25 * model_agreement + 0.10 * spread_quality, 0.0, 1.0)

    signal = np.stack([d_knn, d_model, d_consensus], axis=1).astype(np.float32)
    sign_votes = np.sign(signal)
    positive_votes = np.sum(signal > 0.08, axis=1)
    negative_votes = np.sum(signal < -0.08, axis=1)
    conflict = ((d_knn > 0.10) & (d_model < -0.10)) | ((d_knn < -0.10) & (d_model > 0.10))
    conflict |= ((d_consensus > 0.16) & ((d_knn < -0.12) | (d_model < -0.12)))
    conflict |= ((d_consensus < -0.16) & ((d_knn > 0.12) | (d_model > 0.12)))

    label = np.full(len(regimes), "no_action", dtype=object)
    power_up = (positive_votes >= 2) & (d_knn > 0.06) & (d_model > 0.06) & (confidence >= 0.62) & ~conflict
    power_down = (negative_votes >= 2) & (d_knn < -0.06) & (d_model < -0.06) & (confidence >= 0.62) & ~conflict
    label[conflict] = "conflict"
    label[power_up] = "power_up"
    label[power_down] = "power_down"

    # Conservative suggested deltas for future design, not for immediate submission.
    raw = np.median(signal, axis=1).astype(np.float32)
    cap = np.asarray(
        [
            0.05 if str(r) == "clear" else 0.06 if str(r) == "light" else 0.09 if str(r) == "medium" else 0.10
            for r in regimes
        ],
        dtype=np.float32,
    )
    suggested = np.clip(raw * (0.30 + 0.40 * confidence), -cap, cap).astype(np.float32)
    suggested[label == "conflict"] = 0.0
    suggested[label == "no_action"] = 0.0
    return {
        "delta_knn": d_knn.astype(np.float32),
        "delta_model": d_model.astype(np.float32),
        "delta_consensus": d_consensus.astype(np.float32),
        "evidence_signal": raw.astype(np.float32),
        "power_confidence": confidence.astype(np.float32),
        "power_label": label.astype("U16"),
        "suggested_log_delta": suggested,
        "positive_votes": positive_votes.astype(np.int16),
        "negative_votes": negative_votes.astype(np.int16),
        "conflict": conflict.astype(bool),
    }


def label_counts(labels: np.ndarray, regimes: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {}
    for label in ["power_up", "power_down", "conflict", "no_action"]:
        mask = labels.astype(str) == label
        item = {"count": int(np.sum(mask)), "fraction": float(np.mean(mask))}
        for regime in ["clear", "light", "medium", "heavy"]:
            rmask = regimes.astype(str) == regime
            item[regime] = int(np.sum(mask & rmask))
        out[label] = item
    return out


def write_suspect_csv(
    labels: np.ndarray,
    regimes: np.ndarray,
    blocked: np.ndarray,
    pos: np.ndarray,
    base_log: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> None:
    order = np.argsort(-np.abs(diagnostics["suggested_log_delta"]))
    fields = [
        "test_index",
        "label",
        "regime",
        "blocked",
        "x",
        "y",
        "z",
        "base_log_power",
        "delta_knn",
        "delta_model",
        "delta_consensus",
        "confidence",
        "suggested_log_delta",
    ]
    with SUSPECTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in order:
            if labels[i] == "no_action" and abs(float(diagnostics["suggested_log_delta"][i])) < 1e-8:
                continue
            writer.writerow(
                {
                    "test_index": int(i),
                    "label": str(labels[i]),
                    "regime": str(regimes[i]),
                    "blocked": int(blocked[i]),
                    "x": float(pos[i, 0]),
                    "y": float(pos[i, 1]),
                    "z": float(pos[i, 2]),
                    "base_log_power": float(base_log[i]),
                    "delta_knn": float(diagnostics["delta_knn"][i]),
                    "delta_model": float(diagnostics["delta_model"][i]),
                    "delta_consensus": float(diagnostics["delta_consensus"][i]),
                    "confidence": float(diagnostics["power_confidence"][i]),
                    "suggested_log_delta": float(diagnostics["suggested_log_delta"][i]),
                }
            )


def make_plots(report_data: dict[str, object], train_true: np.ndarray, train_pred: np.ndarray, regimes: np.ndarray, test_diag: dict[str, np.ndarray]) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is diagnostic only.
        report_data["plot_error"] = repr(exc)
        return paths

    fig, ax = plt.subplots(figsize=(7, 6))
    for regime, color in [("clear", "tab:green"), ("light", "tab:blue"), ("medium", "tab:orange"), ("heavy", "tab:red")]:
        mask = regimes.astype(str) == regime
        ax.scatter(train_true[mask], train_pred[mask], s=8, alpha=0.45, label=regime, color=color)
    lo = float(np.nanquantile(train_true, 0.01))
    hi = float(np.nanquantile(train_true, 0.99))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Train true log total power")
    ax.set_ylabel("OOF predicted log total power")
    ax.legend(markerscale=2)
    ax.set_title("v199 train OOF power predictability")
    path = REPORT_DIR / "v199_train_oof_power_scatter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(PATHS.relative(path))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = test_diag["power_label"].astype(str)
    values = [test_diag["evidence_signal"][labels == label] for label in ["power_up", "power_down", "conflict", "no_action"]]
    ax.boxplot(values, labels=["up", "down", "conflict", "no_action"], showfliers=False)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Median evidence log delta vs v196")
    ax.set_title("v199 test power evidence by diagnostic label")
    path = REPORT_DIR / "v199_test_power_evidence_boxplot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(PATHS.relative(path))
    return paths


def main() -> None:
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)
    test_regimes = regime_of(test_blocked)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)

    train_marginals = v196.build_train_delay_marginals()
    train_total = train_marginals["total_power"].astype(np.float32)
    valid_train = (train_marginals["valid"].astype(np.float32) > 0.5) & np.isfinite(train_total) & (train_total > 1e-20)
    train_log = np.log(np.maximum(train_total, TINY)).astype(np.float32)

    train_neighbors = effective_neighbor_weights(
        train_pos,
        train_pos,
        train_blocked,
        train_blocked,
        train_regimes,
        train_regimes,
        valid_train,
        K_TRAIN,
        exclude_self=True,
    )
    oof_knn = np.sum(train_neighbors.weights * train_log[train_neighbors.idx], axis=1).astype(np.float32)
    x_train = feature_matrix(train_pos, train_blocked)
    model_oof, fold_diag = cross_fit_models(x_train, train_log, valid_train)
    oof_preds = {"knn": oof_knn, **model_oof}

    np.savez_compressed(
        TRAIN_OOF_OUT,
        train_log=train_log,
        valid_train=valid_train,
        regimes=train_regimes.astype("U8"),
        train_blocked=train_blocked,
        oof_knn=oof_knn,
        train_neighbor_idx=train_neighbors.idx,
        train_neighbor_weights=train_neighbors.weights,
        train_neighbor_confidence=train_neighbors.confidence,
        **model_oof,
        **fold_diag,
    )

    # Fit final models and compute test evidence.
    models = build_models()
    x_test = feature_matrix(test_pos, test_blocked)
    test_model_preds: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train[valid_train], train_log[valid_train])
        test_model_preds[name] = model.predict(x_test).astype(np.float32)
    test_model_mean = np.mean(np.stack([test_model_preds["extra"], test_model_preds["hgb"]]), axis=0).astype(np.float32)
    test_model_spread = np.abs(test_model_preds["extra"] - test_model_preds["hgb"]).astype(np.float32)

    test_neighbors = effective_neighbor_weights(
        test_pos,
        train_pos,
        test_blocked,
        train_blocked,
        test_regimes,
        train_regimes,
        valid_train,
        K_TRAIN,
        exclude_self=False,
    )
    test_knn_log = np.sum(test_neighbors.weights * train_log[test_neighbors.idx], axis=1).astype(np.float32)

    total_v196 = compute_total_power(BASE_V196)
    total_v189 = compute_total_power(BASE_V189)
    total_v198 = compute_total_power(BASE_V198) if BASE_V198.is_file() else np.full_like(total_v196, np.nan)
    log_v196 = np.log(np.maximum(total_v196, TINY)).astype(np.float32)
    log_v189 = np.log(np.maximum(total_v189, TINY)).astype(np.float32)
    log_v198 = np.log(np.maximum(total_v198, TINY)).astype(np.float32)

    consensus_log, consensus_diag = test_power_consensus(log_v196, test_pos, test_blocked, test_regimes)
    point_diag = classify_test_points(
        test_regimes,
        log_v196,
        test_knn_log,
        test_model_mean,
        consensus_log,
        test_neighbors.confidence,
        consensus_diag["test_neighbor_agreement"],
        test_model_spread,
    )
    labels = point_diag["power_label"].astype(str)

    np.savez_compressed(
        TEST_DIAG_OUT,
        regimes=test_regimes.astype("U8"),
        test_blocked=test_blocked,
        test_pos=test_pos.astype(np.float32),
        total_v196=total_v196,
        total_v189=total_v189,
        total_v198=total_v198,
        log_v196=log_v196,
        log_v189=log_v189,
        log_v198=log_v198,
        test_knn_log=test_knn_log,
        test_model_extra_log=test_model_preds["extra"],
        test_model_hgb_log=test_model_preds["hgb"],
        test_model_mean_log=test_model_mean,
        test_model_spread=test_model_spread,
        test_consensus_log=consensus_log,
        test_train_neighbor_idx=test_neighbors.idx,
        test_train_neighbor_weights=test_neighbors.weights,
        test_train_confidence=test_neighbors.confidence,
        **consensus_diag,
        **point_diag,
    )
    write_suspect_csv(labels, test_regimes, test_blocked, test_pos, log_v196, point_diag)

    train_metrics = metrics_by_regime(train_log, oof_preds, train_regimes, valid_train)
    model_vs_base = {
        "delta_knn": summarize(point_diag["delta_knn"]),
        "delta_model": summarize(point_diag["delta_model"]),
        "delta_consensus": summarize(point_diag["delta_consensus"]),
        "evidence_signal": summarize(point_diag["evidence_signal"]),
        "power_confidence": summarize(point_diag["power_confidence"]),
        "suggested_log_delta": summarize(point_diag["suggested_log_delta"]),
        "v198_log_delta_vs_v196": summarize(log_v198 - log_v196),
    }
    label_summary = label_counts(labels, test_regimes)
    by_regime = by_regime_summary(
        test_regimes,
        {
            "delta_knn": point_diag["delta_knn"],
            "delta_model": point_diag["delta_model"],
            "delta_consensus": point_diag["delta_consensus"],
            "evidence_signal": point_diag["evidence_signal"],
            "power_confidence": point_diag["power_confidence"],
            "suggested_log_delta": point_diag["suggested_log_delta"],
            "v198_log_delta_vs_v196": log_v198 - log_v196,
        },
    )

    report: dict[str, object] = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "question": "Is v196/base total power reliable enough to keep fixed?",
        "design": {
            "no_submission_generated": True,
            "train_oof": "predict train true log total power from official-visible geometry/regime features",
            "test_diagnostics": "compare v196 log power with train-kNN, geometry model and test-test consensus",
            "labels_are_diagnostic_only": ["power_up", "power_down", "conflict", "no_action"],
        },
        "parameters": {
            "train_k": K_TRAIN,
            "test_k": K_TEST,
            "base": PATHS.relative(BASE_V196),
            "compare_v189": PATHS.relative(BASE_V189),
            "compare_v198": PATHS.relative(BASE_V198),
            "features": [
                "ue_xyz",
                "relative_to_bs_xyz",
                "distance_to_bs",
                "log1p_distance",
                "azimuth_sin_cos",
                "blocked",
                "log1p_blocked",
                "distance_times_blocked",
                "regime_one_hot",
            ],
        },
        "train_oof_metrics": train_metrics,
        "test_summary": {
            "overall": model_vs_base,
            "labels": label_summary,
            "by_regime": by_regime,
            "v196_vs_v189_log_delta": summarize(log_v196 - log_v189),
            "v198_vs_v196_log_delta": summarize(log_v198 - log_v196),
        },
        "interpretation": {
            "what_this_can_answer": (
                "whether visible geometry contains enough signal to question fixed v196 power, "
                "and which test points have consistent up/down evidence"
            ),
            "what_this_cannot_answer": (
                "hidden test true power; online feedback is still required before using any suggested deltas"
            ),
        },
        "outputs": {
            "train_oof": PATHS.relative(TRAIN_OOF_OUT),
            "test_diagnostics": PATHS.relative(TEST_DIAG_OUT),
            "suspects_csv": PATHS.relative(SUSPECTS_CSV),
            "report": PATHS.relative(REPORT),
        },
    }
    plots = make_plots(report, train_log[valid_train], oof_preds["model_mean"][valid_train], train_regimes[valid_train], point_diag)
    report["outputs"]["plots"] = plots  # type: ignore[index]
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
